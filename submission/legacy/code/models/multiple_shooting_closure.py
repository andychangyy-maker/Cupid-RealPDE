"""Multiple-shooting conservative UV closure for Track 1 discovery.

The model has no direct future-field decoder.  An explicit causal-time
encoder builds one history context, then a shared physical-state cell advances
four five-frame shooting segments.  The rollout modes differ only in their
forward segment start: ``carry_detached`` starts from the previous predicted
segment end, while ``reset_anchor`` starts from the terminal-history anchor.
The default truncated gradient graph preserves the original five-step
multiple-shooting training.  An optional connected graph keeps the carried
physical state differentiable across segments without changing forward values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .pyramid_causal3d_operator import (
    FramewiseSpatialResidual,
    PyramidCausal3DOperator,
    SeparableSpaceTimeBlock,
    _sinusoidal_embedding,
)


MULTIPLE_SHOOTING_MODES = ("reset_anchor", "carry_detached")
SEGMENT_GRADIENT_MODES = ("truncated", "connected")


@dataclass(frozen=True)
class MultipleShootingClosureConfig:
    in_steps: int = 20
    out_steps: int = 20
    in_channels: int = 3
    out_channels: int = 3
    segment_length: int = 5
    e0_channels: int = 32
    e1_channels: int = 64
    e2_channels: int = 128
    e0_blocks: int = 2
    e1_blocks: int = 2
    e2_blocks: int = 2
    groups: int = 8
    axis_kernel: int = 5
    channel_mlp_ratio: float = 4.0
    context_channels: int = 32
    cell_channels: int = 48
    cell_blocks: int = 2
    step_embedding_channels: int = 8
    max_displacement: float = 1.0
    normalized_flux_limit: float = 0.025
    normalized_source_limit: float = 0.10

    @property
    def segments(self) -> int:
        return self.out_steps // self.segment_length

    def validate(self) -> None:
        positive = (
            self.in_steps, self.out_steps, self.in_channels,
            self.out_channels, self.segment_length,
            self.e0_channels, self.e1_channels, self.e2_channels,
            self.e0_blocks, self.e1_blocks, self.e2_blocks,
            self.groups, self.axis_kernel, self.context_channels,
            self.cell_channels, self.cell_blocks,
            self.step_embedding_channels,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("multiple-shooting dimensions must be positive")
        if self.in_steps != 20 or self.out_steps != 20:
            raise ValueError("multiple shooting is frozen to 20 -> 20")
        if self.in_channels < 2 or self.out_channels != 3:
            raise ValueError("external API must expose at least UV input and UVP output")
        if self.out_steps % self.segment_length:
            raise ValueError("segment length must divide the horizon")
        widths = (self.e0_channels, self.e1_channels, self.e2_channels,
                  self.context_channels, self.cell_channels)
        if any(width % self.groups for width in widths):
            raise ValueError("all hidden widths must divide groups")
        bounds = (self.max_displacement, self.normalized_flux_limit,
                  self.normalized_source_limit)
        if any(value <= 0 for value in bounds):
            raise ValueError("all physical closure bounds must be positive")


class CellResidual(nn.Module):
    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return value + hidden


class PhysicalClosureCell(nn.Module):
    """One shared bounded advection/flux/source step."""

    def __init__(self, config: MultipleShootingClosureConfig):
        super().__init__()
        inputs = (2 + config.context_channels + 2
                  + config.step_embedding_channels)
        self.stem = nn.Conv2d(inputs, config.cell_channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            CellResidual(config.cell_channels, config.groups)
            for _ in range(config.cell_blocks))
        self.displacement_head = nn.Conv2d(config.cell_channels, 2, 1)
        self.flux_head = nn.Conv2d(config.cell_channels, 4, 1)
        self.source_head = nn.Conv2d(config.cell_channels, 2, 1)
        for head in (self.displacement_head, self.flux_head,
                     self.source_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def telescoping_flux_divergence(
            flux: torch.Tensor, free_mask: torch.Tensor) -> torch.Tensor:
        """Masked internal-face flux with an exact telescoping divergence."""
        if flux.ndim != 4 or flux.shape[1] != 4:
            raise ValueError("flux must be [B,4,H,W]")
        if (free_mask.ndim != 4 or free_mask.shape[1] != 1
                or free_mask.shape[-2:] != flux.shape[-2:]):
            raise ValueError("free mask must be [B,1,H,W]")
        flux_x, flux_y = flux[:, :2], flux[:, 2:]
        face_x = 0.5 * (flux_x[..., :-1] + flux_x[..., 1:])
        face_x = face_x * (free_mask[..., :-1] * free_mask[..., 1:])
        face_x = F.pad(face_x, (1, 1, 0, 0))
        div_x = face_x[..., 1:] - face_x[..., :-1]
        face_y = 0.5 * (flux_y[..., :-1, :] + flux_y[..., 1:, :])
        face_y = face_y * (free_mask[..., :-1, :] * free_mask[..., 1:, :])
        face_y = F.pad(face_y, (0, 0, 1, 1))
        div_y = face_y[..., 1:, :] - face_y[..., :-1, :]
        return div_x + div_y

    def forward(
            self, state_physical: torch.Tensor, context: torch.Tensor,
            coordinates: torch.Tensor, step_embedding: torch.Tensor,
            free_mask: torch.Tensor,
            target_mean: torch.Tensor, target_std: torch.Tensor,
            config: MultipleShootingClosureConfig,
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state_normalized = (
            (state_physical - target_mean) / target_std).permute(0, 3, 1, 2)
        step = step_embedding[:, :, None, None].expand(
            -1, -1, state_physical.shape[1], state_physical.shape[2])
        hidden = F.silu(self.stem(torch.cat(
            (state_normalized, context, coordinates, step), dim=1)))
        for block in self.blocks:
            hidden = block(hidden)

        displacement = config.max_displacement * torch.tanh(
            self.displacement_head(hidden)).permute(0, 2, 3, 1)
        # ``grid_sample`` at the identity grid is only numerically, not
        # bitwise, equal to its source.  Subtracting the same zero-displacement
        # sample makes update-zero persistence exact while preserving the
        # first-order displacement gradient.
        warped = PyramidCausal3DOperator.warp(
            state_physical, displacement[:, None])[:, 0]
        identity_sample = PyramidCausal3DOperator.warp(
            state_physical, torch.zeros_like(displacement)[:, None])[:, 0]
        advected = state_physical + (warped - identity_sample)
        raw_flux = config.normalized_flux_limit * torch.tanh(
            self.flux_head(hidden))
        flux_delta_normalized = -self.telescoping_flux_divergence(
            raw_flux, free_mask)
        source_normalized = config.normalized_source_limit * torch.tanh(
            self.source_head(hidden)).permute(0, 2, 3, 1)
        flux_delta_physical = (
            flux_delta_normalized.permute(0, 2, 3, 1) * target_std)
        source_physical = source_normalized * target_std
        next_state = advected + flux_delta_physical + source_physical
        return next_state, {
            "displacement": displacement,
            "raw_flux_normalized": raw_flux.permute(0, 2, 3, 1),
            "flux_delta_normalized": flux_delta_normalized.permute(
                0, 2, 3, 1),
            "source_normalized": source_normalized,
        }


class MultipleShootingClosure(nn.Module):
    """Causal history encoder plus four five-step physical shooting solves."""

    def __init__(self, config: MultipleShootingClosureConfig,
                 rollout_mode: str = "carry_detached",
                 segment_gradient_mode: str = "truncated"):
        super().__init__()
        config.validate()
        if rollout_mode not in MULTIPLE_SHOOTING_MODES:
            raise ValueError(f"unknown rollout mode {rollout_mode!r}")
        if segment_gradient_mode not in SEGMENT_GRADIENT_MODES:
            raise ValueError(
                f"unknown segment gradient mode {segment_gradient_mode!r}")
        self.config = config
        self.rollout_mode = str(rollout_mode)
        self.segment_gradient_mode = str(segment_gradient_mode)
        self._reset_override = False

        # UV is sliced before this stem; Sim pressure has no graph edge.
        self.stem = nn.Conv2d(4, config.e0_channels, 3, stride=2, padding=1)
        self.stage0 = nn.ModuleList(
            SeparableSpaceTimeBlock(
                config.e0_channels, config, "causal_temporal_candidate")
            for _ in range(config.e0_blocks))
        self.down1 = nn.Conv2d(
            config.e0_channels, config.e1_channels, 3, stride=2, padding=1)
        self.stage1 = nn.ModuleList(
            SeparableSpaceTimeBlock(
                config.e1_channels, config, "causal_temporal_candidate")
            for _ in range(config.e1_blocks))
        self.down2 = nn.Conv2d(
            config.e1_channels, config.e2_channels, 3, stride=2, padding=1)
        self.stage2 = nn.ModuleList(
            SeparableSpaceTimeBlock(
                config.e2_channels, config, "causal_temporal_candidate")
            for _ in range(config.e2_blocks))
        self.observed_time0 = nn.Parameter(_sinusoidal_embedding(
            config.in_steps, config.e0_channels))
        self.observed_time1 = nn.Parameter(_sinusoidal_embedding(
            config.in_steps, config.e1_channels))
        self.observed_time2 = nn.Parameter(_sinusoidal_embedding(
            config.in_steps, config.e2_channels))

        self.context2 = nn.Conv2d(
            config.e2_channels, config.context_channels, 1)
        self.context1 = nn.Conv2d(
            config.e1_channels, config.context_channels, 1)
        self.context0 = nn.Conv2d(
            config.e0_channels, config.context_channels, 1)
        self.context_refine = FramewiseSpatialResidual(
            config.context_channels, config.context_channels, config.groups)
        self.step_embedding = nn.Parameter(_sinusoidal_embedding(
            config.out_steps, config.step_embedding_channels))
        self.cell = PhysicalClosureCell(config)

        self.register_buffer("mean_input", torch.zeros(3))
        self.register_buffer("std_input", torch.ones(3))
        self.register_buffer("mean_target", torch.zeros(3))
        self.register_buffer("std_target", torch.ones(3))

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": "multiple_shooting_closure",
            "rollout_mode": self.rollout_mode,
            "segment_gradient_mode": self.segment_gradient_mode,
            "single_causal_variable": "cross_segment_state_carry",
            "direct_future_decoder": False,
            "pressure_input_graph_edge": False,
            **asdict(self.config),
        }

    @torch.no_grad()
    def set_normalizer_statistics(
            self, mean_input: torch.Tensor, mean_target: torch.Tensor,
            std_input: torch.Tensor, std_target: torch.Tensor) -> None:
        for name, value in (
                ("mean_input", mean_input), ("mean_target", mean_target),
                ("std_input", std_input), ("std_target", std_target)):
            target = getattr(self, name)
            source = torch.as_tensor(value, device=target.device,
                                     dtype=target.dtype)
            if source.shape != (3,):
                raise ValueError("normalizer buffers must have shape [3]")
            target.copy_(source.clamp_min(1.0e-8)
                         if name.startswith("std") else source)

    @staticmethod
    def _frame_apply(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = value.shape
        output = module(value.reshape(batch * steps, channels, height, width))
        return output.reshape(batch, steps, output.shape[1],
                              output.shape[2], output.shape[3])

    @staticmethod
    def _coordinates(reference: torch.Tensor) -> torch.Tensor:
        batch, height, width = reference.shape[:3]
        y = torch.linspace(-1, 1, height, device=reference.device,
                           dtype=reference.dtype)
        x = torch.linspace(-1, 1, width, device=reference.device,
                           dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx)).unsqueeze(0).expand(batch, -1, -1, -1)

    @staticmethod
    def _add_time(value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return value + embedding.to(value).view(1, value.shape[1], -1, 1, 1)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        if (history.ndim != 5 or history.shape[1] != self.config.in_steps
                or history.shape[-1] < 2):
            raise ValueError("history must be [B,20,H,W,C>=2]")
        uv = history[..., :2]
        batch, steps, height, width, _channels = uv.shape
        coordinates = self._coordinates(uv[:, 0]).unsqueeze(1).expand(
            batch, steps, -1, -1, -1)
        frames = uv.permute(0, 1, 4, 2, 3)
        stage0 = self._frame_apply(
            self.stem, torch.cat((frames, coordinates), dim=2))
        stage0 = self._add_time(stage0, self.observed_time0)
        for block in self.stage0:
            stage0 = block(stage0)
        stage1 = self._add_time(
            self._frame_apply(self.down1, stage0), self.observed_time1)
        for block in self.stage1:
            stage1 = block(stage1)
        stage2 = self._add_time(
            self._frame_apply(self.down2, stage1), self.observed_time2)
        for block in self.stage2:
            stage2 = block(stage2)

        context = self.context2(stage2[:, -1])
        context = F.interpolate(
            context, size=stage1.shape[-2:], mode="bilinear",
            align_corners=False) + self.context1(stage1[:, -1])
        context = F.interpolate(
            context, size=stage0.shape[-2:], mode="bilinear",
            align_corners=False) + self.context0(stage0[:, -1])
        context = F.interpolate(
            context, size=(height, width), mode="bilinear",
            align_corners=False)
        return self.context_refine(context)

    def set_reset_override(self, enabled: bool) -> None:
        self._reset_override = bool(enabled)

    @staticmethod
    def _outer_boundary_mask(reference: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(reference[..., :1], dtype=torch.bool)
        mask[:, 0, :, :] = True
        mask[:, -1, :, :] = True
        mask[:, :, 0, :] = True
        mask[:, :, -1, :] = True
        return mask

    def forward_components(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encode_history(history)
        uv_normalized = history[..., :2]
        input_mean = self.mean_input[:2].to(history)
        input_std = self.std_input[:2].to(history)
        target_mean = self.mean_target[:2].to(history)
        target_std = self.std_target[:2].to(history)
        history_physical = uv_normalized * input_std + input_mean
        anchor = history_physical[:, -1]
        # Real obstacle cells are zero in physical units.  A tolerance keeps
        # the mask stable through normalize/de-normalize round-off.
        solid = history_physical.abs().amax(dim=(1, 4)).unsqueeze(-1) <= 1.0e-6
        outer = self._outer_boundary_mask(anchor)
        geometry_anchor = solid | outer
        coordinates = self._coordinates(anchor)

        state = anchor.detach()
        predictions, starts, ends = [], [], []
        displacements, raw_fluxes, fluxes, sources = [], [], [], []
        for segment in range(self.config.segments):
            # Segment zero always starts from the detached observed anchor.
            # Later connected segments preserve exactly the same forward value
            # while allowing their loss to credit the preceding physical
            # rollout.  Truncated remains the backward-compatible default.
            carry = (state if (segment > 0 and
                               self.segment_gradient_mode == "connected")
                     else state.detach())
            use_reset = self.rollout_mode == "reset_anchor" or self._reset_override
            segment_base = anchor.detach() if use_reset else carry
            state = segment_base
            state = torch.where(geometry_anchor, anchor, state)
            starts.append(state)
            for local_step in range(self.config.segment_length):
                horizon = segment * self.config.segment_length + local_step
                embedding = self.step_embedding[horizon].to(history).view(
                    1, -1).expand(history.shape[0], -1)
                state, components = self.cell(
                    state, context, coordinates, embedding,
                    (~geometry_anchor).permute(0, 3, 1, 2).to(history.dtype),
                    target_mean, target_std, self.config)
                state = torch.where(geometry_anchor, anchor, state)
                predictions.append(state)
                displacements.append(components["displacement"])
                raw_fluxes.append(components["raw_flux_normalized"])
                fluxes.append(components["flux_delta_normalized"])
                sources.append(components["source_normalized"])
            ends.append(state)

        physical_uv = torch.stack(predictions, dim=1)
        normalized_uv = (physical_uv - target_mean) / target_std
        physical_zero_p = physical_uv.new_zeros(())
        normalized_zero_p = ((physical_zero_p - self.mean_target[2].to(history))
                             / self.std_target[2].to(history))
        pressure = normalized_uv.new_full(
            (*normalized_uv.shape[:-1], 1), normalized_zero_p)
        prediction = torch.cat((normalized_uv, pressure), dim=-1)
        flux_stack = torch.stack(fluxes, dim=1)
        flux_global_sum = flux_stack.sum(dim=(2, 3))
        if (not torch.jit.is_tracing()
                and float(flux_global_sum.detach().abs().max()) > 1.0e-6):
            raise RuntimeError("masked finite-volume flux failed to telescope")
        return {
            "prediction": prediction,
            "physical_uv": physical_uv,
            # Expose the already-computed multiscale history context to small
            # deployed sidecars without running the expensive encoder twice.
            "context": context,
            "segment_starts": torch.stack(starts, dim=1),
            "segment_ends": torch.stack(ends, dim=1),
            "displacement": torch.stack(displacements, dim=1),
            "raw_flux_normalized": torch.stack(raw_fluxes, dim=1),
            "flux_delta_normalized": flux_stack,
            "flux_delta_global_sum": flux_global_sum,
            "source_normalized": torch.stack(sources, dim=1),
            "solid_mask": solid,
            "outer_boundary_mask": outer,
            "free_mask": ~geometry_anchor,
        }

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_components(history)["prediction"]


def build_multiple_shooting_closure(
        rollout_mode: str = "carry_detached",
        segment_gradient_mode: str = "truncated", **overrides: object,
        ) -> MultipleShootingClosure:
    return MultipleShootingClosure(
        MultipleShootingClosureConfig(**overrides), rollout_mode,
        segment_gradient_mode)
