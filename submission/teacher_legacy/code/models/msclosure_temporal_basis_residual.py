"""Zero-mean DCT innovation head for the multiple-shooting closure.

Unlike a local energy multiplier, this module can change the temporal waveform
and velocity direction while preserving every pixel's 20-frame mean.  A fixed
non-DC DCT basis removes the learned shape/amplitude scale ambiguity seen in a
free Conv3D decoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .msclosure_local_energy import MSLocalEnergyController
from .multiple_shooting_closure import MultipleShootingClosure


@dataclass(frozen=True)
class MSTemporalBasisResidualConfig:
    modes: int = 4
    hidden_channels: int = 64
    blocks: int = 2
    groups: int = 8
    spatial_kernel: int = 5
    # Fit-only DCT-K4 residual coefficients have normalized vector p99 below
    # 1.81 (u/v maximum); three leaves headroom without an effectively
    # unbounded early-training waveform.
    coefficient_limit_normalized: float = 3.0
    energy_reference: float = 0.01

    def validate(self, context_channels: int, out_steps: int) -> None:
        if not 1 <= self.modes < out_steps:
            raise ValueError("DCT modes must be in [1, out_steps)")
        if min(context_channels, self.hidden_channels, self.blocks,
               self.groups) <= 0:
            raise ValueError("temporal-basis dimensions must be positive")
        if self.hidden_channels % self.groups:
            raise ValueError("groups must divide hidden channels")
        if self.spatial_kernel <= 0 or self.spatial_kernel % 2 != 1:
            raise ValueError("spatial kernel must be positive and odd")
        if (self.coefficient_limit_normalized <= 0
                or self.energy_reference <= 0):
            raise ValueError("temporal-basis scales must be positive")


class _SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int, kernel: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv2d(
            channels, channels, kernel, padding=kernel // 2,
            groups=channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(torch.nn.functional.silu(self.norm1(value)))
        hidden = self.pointwise(torch.nn.functional.silu(self.norm2(hidden)))
        return value + hidden


class MSTemporalBasisResidualController(nn.Module):
    """Predict spatial coefficients for fixed zero-mean temporal modes."""

    def __init__(self, context_channels: int, out_steps: int,
                 config: MSTemporalBasisResidualConfig =
                 MSTemporalBasisResidualConfig()) -> None:
        super().__init__()
        config.validate(context_channels, out_steps)
        self.context_channels = int(context_channels)
        self.out_steps = int(out_steps)
        self.config = config
        # context + history/base coefficients + history/base means + two TKE
        # maps + coordinates + free mask
        input_channels = (
            context_channels + 4 * config.modes + 4 + 2 + 2 + 1)
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, config.hidden_channels, 1, bias=False),
            nn.GroupNorm(config.groups, config.hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            _SpatialResidualBlock(
                config.hidden_channels, config.groups,
                config.spatial_kernel)
            for _ in range(config.blocks))
        self.output = nn.Conv2d(
            config.hidden_channels, 2 * config.modes, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        time = torch.arange(out_steps, dtype=torch.float32) + 0.5
        modes = torch.arange(1, config.modes + 1,
                             dtype=torch.float32).unsqueeze(1)
        # Unit temporal RMS: mean_t basis[k,t]^2 == 1.
        basis = math.sqrt(2.0) * torch.cos(
            math.pi * modes * time.unsqueeze(0) / out_steps)
        basis = basis - basis.mean(dim=1, keepdim=True)
        basis = basis / basis.square().mean(
            dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
        self.register_buffer("basis", basis)

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": "msclosure_temporal_basis_residual_controller",
            "context_channels": self.context_channels,
            "out_steps": self.out_steps,
            "input_channels": self.input_channels,
            **asdict(self.config),
        }

    def temporal_coefficients(self, value: torch.Tensor) -> torch.Tensor:
        """Project normalized [B,T,H,W,2] velocity onto the fixed basis."""
        if (value.ndim != 5 or value.shape[1] != self.out_steps
                or value.shape[-1] != 2):
            raise ValueError("velocity must have shape [B,T,H,W,2]")
        centered = value - value.mean(dim=1, keepdim=True)
        return torch.einsum(
            "bthwc,kt->bckhw", centered,
            self.basis.to(value)) / self.out_steps

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        expected = (2, self.config.modes)
        if (coefficients.ndim != 5
                or coefficients.shape[1:3] != expected):
            raise ValueError("coefficients must have shape [B,2,K,H,W]")
        value = torch.einsum(
            "bckhw,kt->bthwc", coefficients,
            self.basis.to(coefficients))
        return value - value.mean(dim=1, keepdim=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (features.ndim != 4
                or features.shape[1] != self.input_channels):
            raise ValueError("temporal-basis feature shape changed")
        hidden = self.stem(features)
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.output(hidden)
        limit = self.config.coefficient_limit_normalized
        bounded = limit * torch.tanh(raw / limit)
        batch, _channels, height, width = bounded.shape
        return bounded.reshape(
            batch, 2, self.config.modes, height, width)


class FrozenMSClosureTemporalBasisResidual(nn.Module):
    """Attach a trainable zero-mean innovation head to a frozen closure."""

    def __init__(self, base: MultipleShootingClosure,
                 controller: MSTemporalBasisResidualController | None = None,
                 config: MSTemporalBasisResidualConfig =
                 MSTemporalBasisResidualConfig()) -> None:
        super().__init__()
        self.base = base
        self.controller = controller or MSTemporalBasisResidualController(
            base.config.context_channels, base.config.out_steps, config)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": "frozen_msclosure_temporal_basis_residual",
            "base": self.base.architecture_config(),
            "controller": self.controller.architecture_config(),
            "preserves_temporal_mean": True,
            "pressure_bypass": True,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.controller.train(mode)
        return self

    def _history_in_target_coordinates(
            self, history: torch.Tensor) -> torch.Tensor:
        physical = (
            history[..., :2] * self.base.std_input[:2].to(history)
            + self.base.mean_input[:2].to(history))
        return (
            physical - self.base.mean_target[:2].to(history)
        ) / self.base.std_target[:2].to(history)

    def forward_with_aux(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            components = self.base.forward_components(history)
        base_prediction = components["prediction"].detach()
        base_uv = base_prediction[..., :2]
        history_uv = self._history_in_target_coordinates(history)
        base_coefficients = self.controller.temporal_coefficients(base_uv)
        history_coefficients = self.controller.temporal_coefficients(history_uv)
        base_mean = base_uv.mean(dim=1).permute(0, 3, 1, 2)
        history_mean = history_uv.mean(dim=1).permute(0, 3, 1, 2)

        physical_base = components["physical_uv"].detach()
        physical_history = (
            history[..., :2] * self.base.std_input[:2].to(history)
            + self.base.mean_input[:2].to(history))
        scale = self.controller.config.energy_reference
        tke_features = torch.stack((
            torch.log1p(MSLocalEnergyController.physical_tke(
                physical_base).clamp_min(0.0) / scale),
            torch.log1p(MSLocalEnergyController.physical_tke(
                physical_history).clamp_min(0.0) / scale),
        ), dim=1)
        coordinates = self.base._coordinates(base_uv[:, 0])
        free = components["free_mask"].permute(0, 3, 1, 2).to(base_uv)
        features = torch.cat((
            components["context"].detach(),
            base_coefficients.detach().flatten(1, 2),
            history_coefficients.detach().flatten(1, 2),
            base_mean.detach(), history_mean.detach(), tke_features.detach(),
            coordinates, free,
        ), dim=1)
        coefficients = self.controller(features)
        normalized_delta = self.controller.reconstruct(coefficients)
        normalized_delta = normalized_delta * free.permute(
            0, 2, 3, 1).unsqueeze(1)
        corrected_uv = base_uv + normalized_delta
        prediction = torch.cat((
            corrected_uv, base_prediction[..., 2:]), dim=-1)
        return {
            "prediction": prediction,
            "base_prediction": base_prediction,
            "normalized_delta": normalized_delta,
            "coefficients": coefficients,
            "base_coefficients": base_coefficients,
            "history_coefficients": history_coefficients,
            "free_mask": components["free_mask"],
        }

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(history)["prediction"]
