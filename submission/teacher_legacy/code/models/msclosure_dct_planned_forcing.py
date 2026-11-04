"""History-only DCT plan injected into every multiple-shooting step.

The plan head predicts four non-DC temporal coefficients for each velocity
component before future rollout begins.  Their fixed DCT reconstruction is
converted to a per-step increment.  Every closure call sees both the planned
value and its increment, and the increment is also applied as a physical
forcing on free cells.  Future truth is neither accepted nor needed by this
module; it can supervise the exposed coefficients in a training objective.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .multiple_shooting_closure import (
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
    PhysicalClosureCell,
)
from .pyramid_causal3d_operator import PyramidCausal3DOperator


DCT_PLAN_MODES = 4
DCT_PLAN_FEEDBACK_CHANNELS = 4
DCT_PLAN_HIDDEN_CHANNELS = 64
DCT_PLAN_BLOCKS = 2
DCT_PLAN_SPATIAL_KERNEL = 5
DCT_PLAN_GROUPS = 8
# context + history coefficients + history mean + anchor velocity + history
# TKE + coordinates + free mask.
DCT_PLAN_NON_CONTEXT_FEATURES = 8 + 2 + 2 + 1 + 2 + 1


@dataclass(frozen=True)
class DCTPlannedForcingMultipleShootingConfig(
        MultipleShootingClosureConfig):
    """Configuration alias retaining the complete base-model contract."""


class _PlanResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(DCT_PLAN_GROUPS, channels)
        self.depthwise = nn.Conv2d(
            channels, channels, DCT_PLAN_SPATIAL_KERNEL,
            padding=DCT_PLAN_SPATIAL_KERNEL // 2, groups=channels)
        self.norm2 = nn.GroupNorm(DCT_PLAN_GROUPS, channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(F.silu(self.norm1(value)))
        hidden = self.pointwise(F.silu(self.norm2(hidden)))
        return value + hidden


class CausalDCTPlanHead(nn.Module):
    """Predict signed unit-RMS DCT coefficients from observed history only."""

    def __init__(self, context_channels: int, out_steps: int) -> None:
        super().__init__()
        if context_channels <= 0:
            raise ValueError("context channels must be positive")
        if out_steps != 20:
            raise ValueError("the DCT plan is frozen to a 20-frame horizon")
        self.context_channels = int(context_channels)
        self.out_steps = int(out_steps)
        self.input_channels = (
            self.context_channels + DCT_PLAN_NON_CONTEXT_FEATURES)
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels, DCT_PLAN_HIDDEN_CHANNELS, 1,
                bias=False),
            nn.GroupNorm(DCT_PLAN_GROUPS, DCT_PLAN_HIDDEN_CHANNELS),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            _PlanResidualBlock(DCT_PLAN_HIDDEN_CHANNELS)
            for _ in range(DCT_PLAN_BLOCKS))
        self.output = nn.Conv2d(
            DCT_PLAN_HIDDEN_CHANNELS, 2 * DCT_PLAN_MODES, 1)
        # Zero plan is exactly the base model at initialization.  The output
        # stays linear: direct coefficient supervision supplies its scale.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        time = torch.arange(out_steps, dtype=torch.float32) + 0.5
        modes = torch.arange(
            1, DCT_PLAN_MODES + 1, dtype=torch.float32).unsqueeze(1)
        basis = math.sqrt(2.0) * torch.cos(
            math.pi * modes * time.unsqueeze(0) / out_steps)
        basis = basis - basis.mean(dim=1, keepdim=True)
        basis = basis / basis.square().mean(
            dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
        self.register_buffer("basis", basis)

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": "causal_dct_plan_head",
            "context_channels": self.context_channels,
            "input_channels": self.input_channels,
            "out_steps": self.out_steps,
            "modes": DCT_PLAN_MODES,
            "hidden_channels": DCT_PLAN_HIDDEN_CHANNELS,
            "blocks": DCT_PLAN_BLOCKS,
            "spatial_kernel": DCT_PLAN_SPATIAL_KERNEL,
            "groups": DCT_PLAN_GROUPS,
            "output_parameterization": "zero_initialized_unbounded_linear",
        }

    def temporal_coefficients(self, value: torch.Tensor) -> torch.Tensor:
        """Project normalized ``[B,20,H,W,2]`` velocity onto the plan basis."""
        if (value.ndim != 5 or value.shape[1] != self.out_steps
                or value.shape[-1] != 2):
            raise ValueError("velocity must have shape [B,20,H,W,2]")
        centered = value - value.mean(dim=1, keepdim=True)
        return torch.einsum(
            "bthwc,kt->bckhw", centered,
            self.basis.to(value)) / self.out_steps

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        if (coefficients.ndim != 5
                or coefficients.shape[1:3] != (2, DCT_PLAN_MODES)):
            raise ValueError("coefficients must have shape [B,2,4,H,W]")
        plan = torch.einsum(
            "bckhw,kt->bthwc", coefficients,
            self.basis.to(coefficients))
        return plan - plan.mean(dim=1, keepdim=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (features.ndim != 4
                or features.shape[1] != self.input_channels):
            raise ValueError("DCT plan feature shape changed")
        hidden = self.stem(features)
        for block in self.blocks:
            hidden = block(hidden)
        output = self.output(hidden)
        batch, _channels, height, width = output.shape
        return output.reshape(
            batch, 2, DCT_PLAN_MODES, height, width)


class DCTPlanConditionedPhysicalClosureCell(PhysicalClosureCell):
    """The existing physical cell plus a separate four-channel plan stem.

    A separate projection is algebraically equivalent to concatenating the
    four channels into the existing 3x3 stem.  Keeping the original stem
    unchanged makes a zero plan bitwise identical to the base cell.
    """

    def __init__(self, config: MultipleShootingClosureConfig) -> None:
        super().__init__(config)
        self.plan_stem = nn.Conv2d(
            DCT_PLAN_FEEDBACK_CHANNELS, config.cell_channels, 3,
            padding=1, bias=False)
        nn.init.zeros_(self.plan_stem.weight)

    def forward(
            self, state_physical: torch.Tensor, context: torch.Tensor,
            plan_context: torch.Tensor, coordinates: torch.Tensor,
            step_embedding: torch.Tensor, free_mask: torch.Tensor,
            target_mean: torch.Tensor, target_std: torch.Tensor,
            config: MultipleShootingClosureConfig,
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (plan_context.ndim != 4
                or plan_context.shape[1] != DCT_PLAN_FEEDBACK_CHANNELS
                or plan_context.shape[-2:] != state_physical.shape[1:3]):
            raise ValueError("plan context must have shape [B,4,H,W]")
        state_normalized = (
            (state_physical - target_mean) / target_std).permute(0, 3, 1, 2)
        step = step_embedding[:, :, None, None].expand(
            -1, -1, state_physical.shape[1], state_physical.shape[2])
        base_hidden = self.stem(torch.cat(
            (state_normalized, context, coordinates, step), dim=1))
        hidden = F.silu(base_hidden + self.plan_stem(plan_context))
        for block in self.blocks:
            hidden = block(hidden)

        displacement = config.max_displacement * torch.tanh(
            self.displacement_head(hidden)).permute(0, 2, 3, 1)
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


class DCTPlannedForcingMultipleShootingClosure(MultipleShootingClosure):
    """Multiple shooting driven by one causal, history-only DCT plan."""

    def __init__(
            self, config: MultipleShootingClosureConfig,
            rollout_mode: str = "carry_detached") -> None:
        if rollout_mode != "carry_detached":
            raise ValueError(
                "DCT planned forcing only supports carry_detached rollout")
        super().__init__(config, rollout_mode=rollout_mode)
        original_cell = self.cell
        conditioned_cell = DCTPlanConditionedPhysicalClosureCell(config)
        incompatible = conditioned_cell.load_state_dict(
            original_cell.state_dict(), strict=False)
        if (incompatible.unexpected_keys
                or incompatible.missing_keys != ["plan_stem.weight"]):
            raise RuntimeError("physical-cell state transfer changed")
        self.cell = conditioned_cell
        self.plan_head = CausalDCTPlanHead(
            config.context_channels, config.out_steps)

    def set_reset_override(self, enabled: bool) -> None:
        if enabled:
            raise ValueError(
                "DCT planned forcing cannot reset segment carry")
        super().set_reset_override(False)

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": (
                "dct_planned_forcing_multiple_shooting_closure"),
            "rollout_mode": self.rollout_mode,
            "direct_future_decoder": False,
            "pressure_input_graph_edge": False,
            "history_only_plan": True,
            "truth_enters_rollout": False,
            "plan_feedback_channels": DCT_PLAN_FEEDBACK_CHANNELS,
            "forcing": (
                "free_mask*target_std*(plan[t]-plan[t-1])"),
            "plan_head": self.plan_head.architecture_config(),
            **asdict(self.config),
        }

    def _plan_features(
            self, history_target: torch.Tensor, context: torch.Tensor,
            coordinates: torch.Tensor, free_mask: torch.Tensor,
            ) -> torch.Tensor:
        centered = history_target - history_target.mean(
            dim=1, keepdim=True)
        coefficients = self.plan_head.temporal_coefficients(history_target)
        history_mean = history_target.mean(dim=1).permute(0, 3, 1, 2)
        anchor = history_target[:, -1].permute(0, 3, 1, 2)
        history_tke = 0.5 * centered.square().mean(dim=1).sum(dim=-1)
        history_tke = torch.log1p(history_tke).unsqueeze(1)
        return torch.cat((
            context,
            coefficients.flatten(1, 2),
            history_mean,
            anchor,
            history_tke,
            coordinates,
            free_mask.permute(0, 3, 1, 2).to(history_target),
        ), dim=1)

    def _segment_aligned_plan_forcing(
            self, applied_plan: torch.Tensor) -> torch.Tensor:
        """Differentiate plan increments consistently with detached carry.

        The forward value is always ``plan[t] - plan[t-1]``.  At the first
        step of later shooting segments, however, ``state`` already contains
        the previous plan value through a detached carry.  Detaching only the
        matching prior-plan operand gives the forcing the same backward
        boundary without changing a single forward value.
        """
        if (applied_plan.ndim != 5
                or applied_plan.shape[1] != self.config.out_steps
                or applied_plan.shape[-1] != 2):
            raise ValueError(
                "applied plan must have shape [B,20,H,W,2]")
        increments = []
        for horizon in range(self.config.out_steps):
            if horizon == 0:
                prior = torch.zeros_like(applied_plan[:, horizon])
            else:
                prior = applied_plan[:, horizon - 1]
                if horizon % self.config.segment_length == 0:
                    prior = prior.detach()
            increments.append(applied_plan[:, horizon] - prior)
        return torch.stack(increments, dim=1)

    def forward_components(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encode_history(history)
        uv_normalized = history[..., :2]
        input_mean = self.mean_input[:2].to(history)
        input_std = self.std_input[:2].to(history)
        target_mean = self.mean_target[:2].to(history)
        target_std = self.std_target[:2].to(history)
        history_physical = uv_normalized * input_std + input_mean
        history_target = (
            history_physical - target_mean) / target_std
        anchor = history_physical[:, -1]
        solid = history_physical.abs().amax(
            dim=(1, 4)).unsqueeze(-1) <= 1.0e-6
        outer = self._outer_boundary_mask(anchor)
        geometry_anchor = solid | outer
        free_mask = ~geometry_anchor
        coordinates = self._coordinates(anchor)

        plan_features = self._plan_features(
            history_target, context, coordinates, free_mask)
        predicted_coefficients = self.plan_head(plan_features)
        predicted_plan = self.plan_head.reconstruct(predicted_coefficients)
        applied_plan = predicted_plan * free_mask.to(predicted_plan).unsqueeze(1)
        planned_forcing_normalized = self._segment_aligned_plan_forcing(
            applied_plan)
        planned_forcing_physical = (
            planned_forcing_normalized * target_std)
        cell_plan_context = torch.cat((
            applied_plan, planned_forcing_normalized), dim=-1)

        state = anchor.detach()
        predictions, starts, ends = [], [], []
        displacements, raw_fluxes, fluxes, sources = [], [], [], []
        for segment in range(self.config.segments):
            carry = state.detach()
            state = torch.where(geometry_anchor, anchor, carry)
            starts.append(state)
            for local_step in range(self.config.segment_length):
                horizon = segment * self.config.segment_length + local_step
                embedding = self.step_embedding[horizon].to(history).view(
                    1, -1).expand(history.shape[0], -1)
                plan_context = cell_plan_context[:, horizon].permute(
                    0, 3, 1, 2)
                state, components = self.cell(
                    state, context, plan_context, coordinates, embedding,
                    free_mask.permute(0, 3, 1, 2).to(history.dtype),
                    target_mean, target_std, self.config)
                state = state + planned_forcing_physical[:, horizon]
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
        normalized_zero_p = (
            (physical_zero_p - self.mean_target[2].to(history))
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
            "free_mask": free_mask,
            "predicted_coefficients": predicted_coefficients,
            "predicted_plan_normalized": predicted_plan,
            "applied_plan_normalized": applied_plan,
            "planned_forcing_normalized": planned_forcing_normalized,
            "planned_forcing_physical": planned_forcing_physical,
            "cell_plan_context_normalized": cell_plan_context,
        }


def build_dct_planned_forcing_multiple_shooting(
        rollout_mode: str = "carry_detached", **overrides: object,
        ) -> DCTPlannedForcingMultipleShootingClosure:
    return DCTPlannedForcingMultipleShootingClosure(
        DCTPlannedForcingMultipleShootingConfig(**overrides), rollout_mode)
