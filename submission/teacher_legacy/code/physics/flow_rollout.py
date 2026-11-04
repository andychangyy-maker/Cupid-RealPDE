"""Low-resolution incompressible advection--diffusion flow rollout.

This is deliberately a classical, non-learned baseline.  It advances the
observed two-component velocity with semi-Lagrangian self-advection, optional
viscous diffusion and an optional Helmholtz projection.  A closure increment
can be estimated only from transitions already present in the input history;
this compensates for pressure, three-dimensional motion and forcing that a
2-D crop cannot resolve.

The solver is not intended to replace the original 3-D CFD.  Its purpose is to
test whether cheap PDE-generated off-trajectory states contain useful training
signal before producing a large augmented dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class FlowRolloutConfig:
    """Numerical choices for one released-frame-to-frame update."""

    advection_scale: float = 1.0
    viscosity: float = 1.0e-6
    projection_strength: float = 0.0
    closure_weight: float = 0.0
    closure_decay: float = 0.95
    closure_steps: int = 5
    closure_clip: float = 3.0
    advect_closure: bool = True
    maccormack: bool = False
    boundary_width: int = 0

    def validate(self) -> None:
        finite = (
            self.advection_scale, self.viscosity,
            self.projection_strength, self.closure_weight,
            self.closure_decay, self.closure_clip,
        )
        if any(not math.isfinite(value) for value in finite):
            raise ValueError("rollout values must be finite")
        if self.advection_scale < 0.0 or self.viscosity < 0.0:
            raise ValueError("advection scale and viscosity must be non-negative")
        if not 0.0 <= self.projection_strength <= 1.0:
            raise ValueError("projection strength must lie in [0,1]")
        if self.closure_weight < 0.0 or not 0.0 < self.closure_decay <= 1.0:
            raise ValueError("closure weight/decay are invalid")
        if self.closure_steps <= 0 or self.closure_clip < 0.0:
            raise ValueError("closure steps must be positive and clip non-negative")
        if self.boundary_width < 0:
            raise ValueError("boundary width must be non-negative")


def _fluid_mask(history: torch.Tensor) -> torch.Tensor:
    """Return [B,1,H,W] mask; airfoil/PIV holes are zero for all input frames."""
    moving = history.abs().amax(dim=(1, 4)) > 1.0e-12
    return moving.unsqueeze(1).to(history.dtype)


def _base_grid(batch: int, height: int, width: int, reference: torch.Tensor,
               ) -> torch.Tensor:
    row = torch.linspace(-1.0, 1.0, height, device=reference.device,
                         dtype=reference.dtype)
    column = torch.linspace(-1.0, 1.0, width, device=reference.device,
                            dtype=reference.dtype)
    yy, xx = torch.meshgrid(row, column, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(
        batch, -1, -1, -1)


def _sample_masked(field: torch.Tensor, grid: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    numerator = F.grid_sample(
        field * mask, grid, mode="bilinear", padding_mode="border",
        align_corners=True)
    denominator = F.grid_sample(
        mask, grid, mode="bilinear", padding_mode="border",
        align_corners=True)
    sampled = numerator / denominator.clamp_min(1.0e-4)
    return torch.where(denominator > 0.05, sampled, torch.zeros_like(sampled))


def _advect(field: torch.Tensor, velocity: torch.Tensor, mask: torch.Tensor,
            dt: float, length_x: float, length_y: float,
            advection_scale: float, velocity_scale: torch.Tensor,
            reverse: bool = False) -> torch.Tensor:
    batch, _channels, height, width = field.shape
    grid = _base_grid(batch, height, width, field).clone()
    direction = -1.0 if reverse else 1.0
    physical = velocity * velocity_scale
    # grid_sample row coordinates grow downwards whereas physical y grows up.
    grid[..., 0] -= (
        direction * 2.0 * dt * advection_scale
        * physical[:, 0] / length_x)
    grid[..., 1] += (
        direction * 2.0 * dt * advection_scale
        * physical[:, 1] / length_y)
    return _sample_masked(field, grid, mask)


def _local_clamp(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    maximum = F.max_pool2d(reference, 3, stride=1, padding=1)
    minimum = -F.max_pool2d(-reference, 3, stride=1, padding=1)
    return torch.maximum(minimum, torch.minimum(maximum, value))


def _diffuse(field: torch.Tensor, viscosity: float, dt: float,
             dx: float, dy: float) -> torch.Tensor:
    if viscosity == 0.0:
        return field
    padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
    center = padded[..., 1:-1, 1:-1]
    laplacian = (
        (padded[..., 1:-1, 2:] - 2.0 * center
         + padded[..., 1:-1, :-2]) / (dx * dx)
        + (padded[..., 2:, 1:-1] - 2.0 * center
           + padded[..., :-2, 1:-1]) / (dy * dy))
    return field + viscosity * dt * laplacian


def _project_periodic(field: torch.Tensor, strength: float,
                      dx: float, dy: float) -> torch.Tensor:
    """Blend with the periodic Helmholtz projection on the rectangular crop."""
    if strength == 0.0:
        return field
    _batch, _channels, height, width = field.shape
    # Work in array coordinates: the row-direction velocity is physical -v.
    array_velocity = torch.stack((field[:, 0], -field[:, 1]), dim=1)
    transformed = torch.fft.rfft2(array_velocity, dim=(-2, -1))
    kx = 2.0 * math.pi * torch.fft.rfftfreq(
        width, d=dx, device=field.device).to(field.dtype)
    ky = 2.0 * math.pi * torch.fft.fftfreq(
        height, d=dy, device=field.device).to(field.dtype)
    ky, kx = torch.meshgrid(ky, kx, indexing="ij")
    denominator = (kx.square() + ky.square()).clamp_min(1.0e-12)
    dot = kx * transformed[:, 0] + ky * transformed[:, 1]
    projected_x = transformed[:, 0] - kx * dot / denominator
    projected_y = transformed[:, 1] - ky * dot / denominator
    projected_x[..., 0, 0] = transformed[:, 0, 0, 0]
    projected_y[..., 0, 0] = transformed[:, 1, 0, 0]
    projected = torch.fft.irfft2(
        torch.stack((projected_x, projected_y), dim=1),
        s=(height, width), dim=(-2, -1))
    projected = torch.stack((projected[:, 0], -projected[:, 1]), dim=1)
    return field + strength * (projected - field)


def _restore_boundaries(field: torch.Tensor, reference: torch.Tensor,
                        width: int) -> torch.Tensor:
    if width == 0:
        return field
    result = field.clone()
    result[..., :width, :] = reference[..., :width, :]
    result[..., -width:, :] = reference[..., -width:, :]
    result[..., :, :width] = reference[..., :, :width]
    result[..., :, -width:] = reference[..., :, -width:]
    return result


def _advance(field: torch.Tensor, mask: torch.Tensor, config: FlowRolloutConfig,
             dt: float, dx: float, dy: float, velocity_scale: torch.Tensor,
             boundary_reference: torch.Tensor) -> torch.Tensor:
    _batch, _channels, height, width = field.shape
    advected = _advect(
        field, field, mask, dt, dx * (width - 1), dy * (height - 1),
        config.advection_scale, velocity_scale)
    if config.maccormack:
        backward = _advect(
            advected, advected, mask, dt, dx * (width - 1),
            dy * (height - 1), config.advection_scale, velocity_scale,
            reverse=True)
        advected = _local_clamp(
            advected + 0.5 * (field - backward), field)
    advanced = _diffuse(advected, config.viscosity, dt, dx, dy)
    advanced = _project_periodic(
        advanced, config.projection_strength, dx, dy)
    advanced = _restore_boundaries(
        advanced, boundary_reference, config.boundary_width)
    return advanced * mask


def _estimate_closure(history: torch.Tensor, mask: torch.Tensor,
                      config: FlowRolloutConfig, dt: float, dx: float,
                      dy: float, velocity_scale: torch.Tensor) -> torch.Tensor:
    if config.closure_weight == 0.0:
        return history.new_zeros((history.shape[0], 2, *history.shape[2:4]))
    start = max(1, history.shape[1] - config.closure_steps)
    residuals = []
    for index in range(start, history.shape[1]):
        previous = history[:, index - 1].permute(0, 3, 1, 2)
        observed = history[:, index].permute(0, 3, 1, 2)
        numerical = _advance(
            previous, mask, config, dt, dx, dy, velocity_scale, observed)
        residuals.append(observed - numerical)
    closure = torch.stack(residuals, dim=0).mean(dim=0)
    if config.closure_clip:
        differences = history[:, 1:, ..., :2] - history[:, :-1, ..., :2]
        rms = differences.square().mean(dim=(1, 2, 3)).sqrt()
        limit = config.closure_clip * rms[:, :, None, None]
        closure = torch.maximum(-limit, torch.minimum(limit, closure))
    return closure * mask


@torch.no_grad()
def rollout_velocity(history: torch.Tensor, steps: int,
                     config: FlowRolloutConfig, *, dt: float, dx: float,
                     dy: float, velocity_scale: torch.Tensor | float = 1.0,
                     ) -> torch.Tensor:
    """Roll out [B,T,H,W,2] history into [B,steps,H,W,2]."""
    config.validate()
    if history.ndim != 5 or history.shape[-1] < 2:
        raise ValueError("history must have shape [B,T,H,W,C>=2]")
    if steps <= 0 or dt <= 0.0 or dx <= 0.0 or dy <= 0.0:
        raise ValueError("steps and physical spacings must be positive")
    history = history[..., :2]
    mask = _fluid_mask(history)
    scale = torch.as_tensor(
        velocity_scale, device=history.device, dtype=history.dtype)
    if scale.ndim == 0:
        scale = scale.repeat(history.shape[0])
    if scale.shape != (history.shape[0],):
        raise ValueError("velocity_scale must be scalar or [batch]")
    scale = scale[:, None, None, None]
    state = history[:, -1].permute(0, 3, 1, 2)
    boundary = state.clone()
    closure = _estimate_closure(
        history, mask, config, dt, dx, dy, scale)
    outputs = []
    for horizon in range(steps):
        state = _advance(
            state, mask, config, dt, dx, dy, scale, boundary)
        if config.closure_weight:
            if config.advect_closure:
                closure = _advect(
                    closure, state, mask, dt,
                    dx * (state.shape[-1] - 1),
                    dy * (state.shape[-2] - 1),
                    config.advection_scale, scale)
            state = state + (
                config.closure_weight
                * config.closure_decay ** horizon * closure)
            state = _restore_boundaries(
                state, boundary, config.boundary_width) * mask
        outputs.append(state.permute(0, 2, 3, 1))
    return torch.stack(outputs, dim=1)

