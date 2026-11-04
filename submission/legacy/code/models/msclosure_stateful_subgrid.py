"""Stateful subgrid dynamics for the multiple-shooting physical closure.

The latent state is deliberately not a decoder or an additive future-field
residual.  A bounded spatial state is initialized from the causal history
context, updated from the model's current velocity at every rollout step, and
can affect the prediction only by conditioning the existing bounded physical
closure cell.  Both the physical state and the latent state are detached at
the same five-frame shooting boundaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .multiple_shooting_closure import (
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
    PhysicalClosureCell,
)


SUBGRID_STATE_OVERRIDES = ("recurrent", "static", "reset")


@dataclass(frozen=True)
class StatefulSubgridMultipleShootingConfig(MultipleShootingClosureConfig):
    """Base closure recipe plus one small bounded latent-state width."""

    subgrid_channels: int = 12

    def validate(self) -> None:
        super().validate()
        if self.subgrid_channels <= 0:
            raise ValueError("subgrid channels must be positive")


class BoundedSubgridState(nn.Module):
    """Initialize and recurrently update an elementwise bounded field."""

    def __init__(self, config: StatefulSubgridMultipleShootingConfig) -> None:
        super().__init__()
        self.context_channels = int(config.context_channels)
        self.state_channels = int(config.subgrid_channels)
        self.step_channels = int(config.step_embedding_channels)
        self.history_projection = nn.Conv2d(
            self.context_channels, self.state_channels, 1)
        update_inputs = (
            2 + self.context_channels + self.state_channels
            + self.step_channels)
        self.update_projection = nn.Conv2d(
            update_inputs, self.state_channels, 3, padding=1)
        # A zero context has no arbitrary constant latent state.  The weights
        # remain live so h is nonzero and the zero-initialized conditioner can
        # receive a gradient as soon as the physical heads become active.
        nn.init.zeros_(self.history_projection.bias)
        nn.init.zeros_(self.update_projection.bias)

    @staticmethod
    def _mask(state: torch.Tensor, free_mask: torch.Tensor) -> torch.Tensor:
        if (free_mask.ndim != 4 or free_mask.shape[1] != 1
                or free_mask.shape[0] != state.shape[0]
                or free_mask.shape[-2:] != state.shape[-2:]):
            raise ValueError("free mask must have shape [B,1,H,W]")
        return state * free_mask.to(dtype=state.dtype)

    def initialize(
            self, context: torch.Tensor,
            free_mask: torch.Tensor) -> torch.Tensor:
        if (context.ndim != 4
                or context.shape[1] != self.context_channels):
            raise ValueError("history context must have shape [B,C,H,W]")
        return self._mask(
            torch.tanh(self.history_projection(context)), free_mask)

    def forward(
            self, current_uv_normalized: torch.Tensor,
            context: torch.Tensor, state: torch.Tensor,
            step_embedding: torch.Tensor,
            free_mask: torch.Tensor) -> torch.Tensor:
        if (current_uv_normalized.ndim != 4
                or current_uv_normalized.shape[1] != 2):
            raise ValueError("current normalized UV must have shape [B,2,H,W]")
        if (context.ndim != 4
                or context.shape[1] != self.context_channels):
            raise ValueError("history context must have shape [B,C,H,W]")
        if (state.ndim != 4 or state.shape[1] != self.state_channels):
            raise ValueError("subgrid state must have shape [B,Hc,H,W]")
        if (step_embedding.ndim != 2
                or step_embedding.shape[0] != state.shape[0]
                or step_embedding.shape[1] != self.step_channels):
            raise ValueError("step embedding must have shape [B,E]")
        spatial_shape = state.shape[-2:]
        if (current_uv_normalized.shape[-2:] != spatial_shape
                or context.shape[-2:] != spatial_shape):
            raise ValueError("subgrid update spatial shapes must match")
        step = step_embedding[:, :, None, None].expand(
            -1, -1, *spatial_shape)
        proposal = torch.tanh(self.update_projection(torch.cat((
            current_uv_normalized, context, state, step), dim=1)))
        return self._mask(proposal, free_mask)


class StatefulConditionedPhysicalClosureCell(PhysicalClosureCell):
    """Base physical cell with a zero-initialized latent context adapter."""

    def __init__(
            self, config: StatefulSubgridMultipleShootingConfig) -> None:
        super().__init__(config)
        self.subgrid_channels = int(config.subgrid_channels)
        self.subgrid_context_projection = nn.Conv2d(
            self.subgrid_channels, config.context_channels, 1, bias=False)
        # This is the only new path into the scored prediction.  Zero init
        # therefore preserves the base closure exactly at initialization.
        nn.init.zeros_(self.subgrid_context_projection.weight)

    def forward(
            self, state_physical: torch.Tensor, context: torch.Tensor,
            subgrid_state: torch.Tensor, coordinates: torch.Tensor,
            step_embedding: torch.Tensor, free_mask: torch.Tensor,
            target_mean: torch.Tensor, target_std: torch.Tensor,
            config: StatefulSubgridMultipleShootingConfig,
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (subgrid_state.ndim != 4
                or subgrid_state.shape[1] != self.subgrid_channels
                or subgrid_state.shape[0] != state_physical.shape[0]
                or subgrid_state.shape[-2:] != state_physical.shape[1:3]):
            raise ValueError("subgrid state must have shape [B,Hc,H,W]")
        masked_state = subgrid_state * free_mask.to(subgrid_state)
        condition = self.subgrid_context_projection(masked_state)
        next_state, components = super().forward(
            state_physical, context + condition, coordinates,
            step_embedding, free_mask, target_mean, target_std, config)
        return next_state, {
            **components,
            "subgrid_context_condition": condition,
        }


class StatefulSubgridMultipleShootingClosure(MultipleShootingClosure):
    """Carry one learned subgrid state alongside the physical UV state."""

    def __init__(
            self, config: StatefulSubgridMultipleShootingConfig,
            rollout_mode: str = "carry_detached") -> None:
        if rollout_mode != "carry_detached":
            raise ValueError(
                "stateful subgrid closure only supports carry_detached")
        super().__init__(config, rollout_mode=rollout_mode)
        original_cell = self.cell
        conditioned_cell = StatefulConditionedPhysicalClosureCell(config)
        incompatible = conditioned_cell.load_state_dict(
            original_cell.state_dict(), strict=False)
        if (incompatible.unexpected_keys or incompatible.missing_keys
                != ["subgrid_context_projection.weight"]):
            raise RuntimeError("physical-cell state transfer changed")
        self.cell = conditioned_cell
        self.subgrid_dynamics = BoundedSubgridState(config)
        self._subgrid_override = "recurrent"

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": (
                "stateful_subgrid_multiple_shooting_closure"),
            "rollout_mode": self.rollout_mode,
            "direct_future_decoder": False,
            "direct_subgrid_output": False,
            "pressure_input_graph_edge": False,
            "subgrid_channels": self.config.subgrid_channels,
            "subgrid_bound": "elementwise tanh in [-1,1]",
            "subgrid_update_inputs": (
                "current_target_normalized_uv+history_context+h+step"),
            "subgrid_output_path": (
                "zero_initialized_context_adapter_to_physical_cell_only"),
            "subgrid_segment_boundary": (
                "carry_detached_with_physical_state"),
            "diagnostic_overrides": list(SUBGRID_STATE_OVERRIDES[1:]),
            **asdict(self.config),
        }

    def set_reset_override(self, enabled: bool) -> None:
        if enabled:
            raise ValueError(
                "stateful subgrid closure cannot reset physical carry")
        super().set_reset_override(False)

    def set_subgrid_override(self, mode: str | None) -> None:
        """Select a same-checkpoint mechanism diagnostic.

        ``static`` holds the history-initialized state for all steps.
        ``reset`` recurs within each shooting segment but restores the
        history-initialized value at the next segment.  Neither changes the
        physical-state carry or any checkpoint tensor.
        """
        resolved = "recurrent" if mode is None else str(mode)
        if resolved not in SUBGRID_STATE_OVERRIDES:
            raise ValueError(f"unknown subgrid override {resolved!r}")
        self._subgrid_override = resolved

    def forward_components(
            self, history: torch.Tensor, *,
            collect_subgrid_diagnostics: bool = True,
            ) -> dict[str, torch.Tensor]:
        context = self.encode_history(history)
        uv_normalized = history[..., :2]
        input_mean = self.mean_input[:2].to(history)
        input_std = self.std_input[:2].to(history)
        target_mean = self.mean_target[:2].to(history)
        target_std = self.std_target[:2].to(history)
        history_physical = uv_normalized * input_std + input_mean
        anchor = history_physical[:, -1]
        solid = history_physical.abs().amax(
            dim=(1, 4)).unsqueeze(-1) <= 1.0e-6
        outer = self._outer_boundary_mask(anchor)
        geometry_anchor = solid | outer
        free_mask = ~geometry_anchor
        free_channels_first = free_mask.permute(0, 3, 1, 2).to(history.dtype)
        coordinates = self._coordinates(anchor)

        initial_subgrid_state = self.subgrid_dynamics.initialize(
            context, free_channels_first)
        subgrid_state = initial_subgrid_state
        state = anchor.detach()
        predictions, starts, ends = [], [], []
        displacements, raw_fluxes, fluxes, sources = [], [], [], []
        h_after_steps = []
        condition_abs_max, condition_rms = [], []
        condition_geometry_abs_max = []
        h_segment_starts, h_segment_ends = [], []

        for segment in range(self.config.segments):
            state = state.detach()
            if segment:
                if self._subgrid_override == "reset":
                    subgrid_state = initial_subgrid_state.detach()
                else:
                    subgrid_state = subgrid_state.detach()
            state = torch.where(geometry_anchor, anchor, state)
            starts.append(state)
            if collect_subgrid_diagnostics:
                h_segment_starts.append(subgrid_state)

            for local_step in range(self.config.segment_length):
                horizon = segment * self.config.segment_length + local_step
                embedding = self.step_embedding[horizon].to(history).view(
                    1, -1).expand(history.shape[0], -1)
                current_uv_normalized = (
                    (state - target_mean) / target_std).permute(0, 3, 1, 2)
                if self._subgrid_override == "static":
                    next_subgrid_state = subgrid_state
                else:
                    next_subgrid_state = self.subgrid_dynamics(
                        current_uv_normalized, context, subgrid_state,
                        embedding, free_channels_first)

                state, components = self.cell(
                    state, context, next_subgrid_state, coordinates,
                    embedding, free_channels_first, target_mean, target_std,
                    self.config)
                state = torch.where(geometry_anchor, anchor, state)
                subgrid_state = next_subgrid_state

                predictions.append(state)
                displacements.append(components["displacement"])
                raw_fluxes.append(components["raw_flux_normalized"])
                fluxes.append(components["flux_delta_normalized"])
                sources.append(components["source_normalized"])
                if collect_subgrid_diagnostics:
                    h_after_steps.append(subgrid_state)
                    condition = components[
                        "subgrid_context_condition"].detach().float()
                    condition_abs_max.append(
                        condition.abs().amax(dim=(1, 2, 3)))
                    condition_rms.append(
                        condition.square().mean(dim=(1, 2, 3)).sqrt())
                    condition_geometry_abs_max.append((
                        condition * geometry_anchor.permute(0, 3, 1, 2)
                    ).abs().amax(dim=(1, 2, 3)))
            ends.append(state)
            if collect_subgrid_diagnostics:
                h_segment_ends.append(subgrid_state)

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

        result = {
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
        }
        if not collect_subgrid_diagnostics:
            return result

        h_after_stack = torch.stack(h_after_steps, dim=1)
        geometry_channels_first = geometry_anchor.permute(0, 3, 1, 2)
        geometry_time = geometry_channels_first.unsqueeze(1)
        result.update({
            "subgrid_initial_state": initial_subgrid_state,
            "subgrid_state_after_update": h_after_stack,
            "subgrid_segment_starts": torch.stack(
                h_segment_starts, dim=1),
            "subgrid_segment_ends": torch.stack(h_segment_ends, dim=1),
            "subgrid_final_state": subgrid_state,
            "subgrid_condition_abs_max_by_step": torch.stack(
                condition_abs_max, dim=1),
            "subgrid_condition_rms_by_step": torch.stack(
                condition_rms, dim=1),
            "subgrid_condition_geometry_abs_max_by_step": torch.stack(
                condition_geometry_abs_max, dim=1),
            "subgrid_state_abs_max": h_after_stack.abs().amax(),
            "subgrid_geometry_abs_max": (
                h_after_stack * geometry_time).abs().amax(),
            "subgrid_all_finite": torch.isfinite(h_after_stack).all(),
        })
        return result

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_components(
            history, collect_subgrid_diagnostics=False)["prediction"]


def build_stateful_subgrid_multiple_shooting(
        rollout_mode: str = "carry_detached", **overrides: object,
        ) -> StatefulSubgridMultipleShootingClosure:
    return StatefulSubgridMultipleShootingClosure(
        StatefulSubgridMultipleShootingConfig(**overrides), rollout_mode)
