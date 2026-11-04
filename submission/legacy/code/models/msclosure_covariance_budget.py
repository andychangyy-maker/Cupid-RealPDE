"""Covariance-budget feedback for the multiple-shooting UV closure.

The base closure already carries the physical velocity across five-frame
shooting segments.  This variant augments that state with the population
statistics of the future frames generated so far.  A causal history context
predicts one local, positive-semidefinite 2x2 covariance budget, and every
closure call sees both the consumed and remaining budget.

All running statistics are computed from the model's own predictions.  The
terminal history frame is the rollout anchor, but is deliberately not counted
as one of the twenty future covariance samples.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import torch
from torch import nn
import torch.nn.functional as F

from .multiple_shooting_closure import (
    MULTIPLE_SHOOTING_MODES,
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
    PhysicalClosureCell,
)


COVARIANCE_FEEDBACK_CHANNELS = 8


@dataclass(frozen=True)
class CovarianceBudgetMultipleShootingConfig(MultipleShootingClosureConfig):
    """Configuration alias retaining the complete base-model contract."""


class LocalPSDCovarianceBudget(nn.Module):
    """Predict a local Cholesky factor in fit-reference-whitened units.

    The zero-initialized projection produces the identity covariance.  If
    ``L_ref`` is the Cholesky factor of the fit covariance reference, the
    corresponding covariance in target-normalized velocity coordinates is
    ``L_ref @ (A @ A.T) @ L_ref.T``.
    """

    _UNIT_SOFTPLUS_SHIFT = math.log(math.expm1(1.0))

    def __init__(self, context_channels: int) -> None:
        super().__init__()
        if context_channels <= 0:
            raise ValueError("context channels must be positive")
        self.context_channels = int(context_channels)
        self.projection = nn.Conv2d(context_channels, 3, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return packed ``[uu, uv, vv]`` covariance in whitened units."""
        if context.ndim != 4 or context.shape[1] != self.context_channels:
            raise ValueError("budget context must have shape [B,C,H,W]")
        # Accumulation and covariance feedback remain FP32 even when a later
        # deployment stores and executes the convolution weights in FP16.
        raw = self.projection(context).permute(0, 2, 3, 1).float()
        shift = self._UNIT_SOFTPLUS_SHIFT
        diagonal_u = F.softplus(raw[..., 0] + shift)
        cross = raw[..., 1]
        diagonal_v = F.softplus(raw[..., 2] + shift)
        return torch.stack((
            diagonal_u.square(),
            diagonal_u * cross,
            cross.square() + diagonal_v.square(),
        ), dim=-1)


class CovarianceBudgetMultipleShootingClosure(MultipleShootingClosure):
    """Multiple shooting with an explicit local covariance-budget state."""

    def __init__(
            self, config: MultipleShootingClosureConfig,
            rollout_mode: str = "carry_detached") -> None:
        if rollout_mode not in MULTIPLE_SHOOTING_MODES:
            raise ValueError(f"unknown rollout mode {rollout_mode!r}")
        super().__init__(config, rollout_mode=rollout_mode)
        # Reuse the exact physical closure implementation.  Only its context
        # width changes: the original context is concatenated with two running
        # means, three consumed-M2 entries, and three budget-residual entries.
        self._covariance_cell_config = replace(
            config,
            context_channels=(
                config.context_channels + COVARIANCE_FEEDBACK_CHANNELS),
        )
        self.cell = PhysicalClosureCell(self._covariance_cell_config)
        self.covariance_budget = LocalPSDCovarianceBudget(
            config.context_channels)
        # This is a coordinate-system statistic, not a learned calibration.
        # Training computes it from fit targets once and stores it in the
        # checkpoint, just as it stores the channel normalizer statistics.
        self.register_buffer(
            "covariance_reference_cholesky", torch.eye(2, dtype=torch.float32))

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": "covariance_budget_multiple_shooting_closure",
            "rollout_mode": self.rollout_mode,
            "direct_future_decoder": False,
            "pressure_input_graph_edge": False,
            "covariance_budget": True,
            "covariance_budget_space": (
                "fit_reference_whitened_target_normalized_uv"),
            "covariance_feedback_channels": COVARIANCE_FEEDBACK_CHANNELS,
            "covariance_population_denominator": self.config.out_steps,
            "anchor_counted_as_future_sample": False,
            "welford_accumulator_dtype": "float32",
            **asdict(self.config),
        }

    @torch.no_grad()
    def set_covariance_reference(self, covariance: torch.Tensor) -> None:
        """Set the fit-only 2x2 target-normalized covariance reference.

        A genuinely indefinite matrix is rejected.  A singular PSD reference
        receives only a floating-point-resolution eigenvalue floor so that it
        can define an invertible whitening transform.
        """
        target = self.covariance_reference_cholesky
        value = torch.as_tensor(
            covariance, device=target.device, dtype=torch.float64)
        if value.shape != (2, 2):
            raise ValueError("covariance reference must have shape [2,2]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("covariance reference must be finite")
        symmetric = 0.5 * (value + value.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        scale = eigenvalues.abs().max().clamp_min(
            torch.finfo(torch.float64).tiny)
        negative_tolerance = (
            16.0 * torch.finfo(torch.float64).eps * scale)
        if bool(eigenvalues.min() < -negative_tolerance):
            raise ValueError("covariance reference must be positive semidefinite")
        eigenvalue_floor = torch.finfo(torch.float32).eps * scale
        regularized = (
            eigenvectors
            @ torch.diag(eigenvalues.clamp_min(eigenvalue_floor))
            @ eigenvectors.T)
        cholesky, info = torch.linalg.cholesky_ex(regularized)
        if bool(info):
            raise ValueError("covariance reference could not be factorized")
        target.copy_(cholesky.to(dtype=target.dtype))

    def covariance_reference(self) -> torch.Tensor:
        """Return the stored reference covariance as an FP32 matrix."""
        factor = self.covariance_reference_cholesky.float()
        return factor @ factor.T

    @staticmethod
    def _packed_covariance_matrix(packed: torch.Tensor) -> torch.Tensor:
        if packed.shape[-1] != 3:
            raise ValueError("packed covariance must end in [uu,uv,vv]")
        row_u = torch.stack((packed[..., 0], packed[..., 1]), dim=-1)
        row_v = torch.stack((packed[..., 1], packed[..., 2]), dim=-1)
        return torch.stack((row_u, row_v), dim=-2)

    def _white_to_target_normalized(
            self, packed: torch.Tensor) -> torch.Tensor:
        matrix = self._packed_covariance_matrix(packed.float())
        factor = self.covariance_reference_cholesky.float()
        return torch.matmul(torch.matmul(factor, matrix), factor.T)

    def _whiten_anchor_relative_state(
            self, state_physical: torch.Tensor, anchor_physical: torch.Tensor,
            target_std: torch.Tensor) -> torch.Tensor:
        normalized_delta = (
            (state_physical.float() - anchor_physical.float())
            / target_std.float())
        factor = self.covariance_reference_cholesky.float()
        whitened_u = normalized_delta[..., 0] / factor[0, 0]
        whitened_v = (
            normalized_delta[..., 1] - factor[1, 0] * whitened_u
        ) / factor[1, 1]
        return torch.stack((whitened_u, whitened_v), dim=-1)

    @staticmethod
    def welford_update(
            running_mean: torch.Tensor, running_m2: torch.Tensor,
            sample: torch.Tensor, count: int,
            free_mask: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add one vector sample to packed FP32 Welford statistics."""
        if count < 0:
            raise ValueError("Welford count must be non-negative")
        if sample.shape[-1] != 2 or running_mean.shape != sample.shape:
            raise ValueError("Welford vectors must end in two components")
        if running_m2.shape != (*sample.shape[:-1], 3):
            raise ValueError("Welford M2 must be packed as [uu,uv,vv]")
        mean = running_mean.float()
        m2 = running_m2.float()
        value = sample.float()
        if count == 0:
            next_mean = value
            next_m2 = torch.zeros_like(m2)
        else:
            delta = value - mean
            next_mean = mean + delta / float(count + 1)
            multiplier = float(count) / float(count + 1)
            increment = multiplier * torch.stack((
                delta[..., 0].square(),
                delta[..., 0] * delta[..., 1],
                delta[..., 1].square(),
            ), dim=-1)
            next_m2 = m2 + increment
        if free_mask is not None:
            if free_mask.shape != (*sample.shape[:-1], 1):
                raise ValueError("free mask must have shape [B,H,W,1]")
            free = free_mask.to(dtype=torch.float32)
            next_mean = next_mean * free
            next_m2 = next_m2 * free
        return next_mean, next_m2

    @staticmethod
    def _feedback_channels(
            running_mean: torch.Tensor, running_m2: torch.Tensor,
            predicted_budget: torch.Tensor, out_steps: int,
            free_mask: torch.Tensor) -> torch.Tensor:
        achieved = running_m2 / float(out_steps)
        residual = predicted_budget - achieved
        feedback = torch.cat((running_mean, achieved, residual), dim=-1)
        feedback = feedback * free_mask.to(dtype=feedback.dtype)
        return feedback.permute(0, 3, 1, 2)

    def forward_components(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encode_history(history)
        uv_normalized = history[..., :2]
        input_mean = self.mean_input[:2].to(history)
        input_std = self.std_input[:2].to(history)
        target_mean = self.mean_target[:2].to(history)
        target_std = self.std_target[:2].to(history)
        history_physical = uv_normalized * input_std + input_mean
        anchor = history_physical[:, -1]
        solid = history_physical.abs().amax(dim=(1, 4)).unsqueeze(-1) <= 1.0e-6
        outer = self._outer_boundary_mask(anchor)
        geometry_anchor = solid | outer
        free_mask = ~geometry_anchor
        coordinates = self._coordinates(anchor)

        predicted_budget_white = self.covariance_budget(context)
        predicted_budget_white = (
            predicted_budget_white * free_mask.to(torch.float32))
        running_mean = torch.zeros_like(anchor, dtype=torch.float32)
        running_m2 = anchor.new_zeros(
            (*anchor.shape[:-1], 3), dtype=torch.float32)
        count = 0

        state = anchor.detach()
        predictions, starts, ends = [], [], []
        displacements, raw_fluxes, fluxes, sources = [], [], [], []
        mean_checkpoints, covariance_checkpoints = [], []
        residual_checkpoints, statistic_segment_starts = [], []
        for segment in range(self.config.segments):
            carry = state.detach()
            if segment:
                running_mean = running_mean.detach()
                running_m2 = running_m2.detach()
            statistic_segment_starts.append(torch.cat((
                running_mean, running_m2 / float(self.config.out_steps)),
                dim=-1))
            use_reset = (
                self.rollout_mode == "reset_anchor" or self._reset_override)
            segment_base = anchor.detach() if use_reset else carry
            state = torch.where(geometry_anchor, anchor, segment_base)
            starts.append(state)
            for local_step in range(self.config.segment_length):
                feedback = self._feedback_channels(
                    running_mean, running_m2, predicted_budget_white,
                    self.config.out_steps, free_mask)
                cell_context = torch.cat(
                    (context, feedback.to(dtype=context.dtype)), dim=1)
                horizon = segment * self.config.segment_length + local_step
                embedding = self.step_embedding[horizon].to(history).view(
                    1, -1).expand(history.shape[0], -1)
                state, components = self.cell(
                    state, cell_context, coordinates, embedding,
                    free_mask.permute(0, 3, 1, 2).to(history.dtype),
                    target_mean, target_std, self._covariance_cell_config)
                state = torch.where(geometry_anchor, anchor, state)
                whitened_state = self._whiten_anchor_relative_state(
                    state, anchor, target_std)
                running_mean, running_m2 = self.welford_update(
                    running_mean, running_m2, whitened_state, count,
                    free_mask=free_mask)
                count += 1

                predictions.append(state)
                displacements.append(components["displacement"])
                raw_fluxes.append(components["raw_flux_normalized"])
                fluxes.append(components["flux_delta_normalized"])
                sources.append(components["source_normalized"])
            ends.append(state)
            realized = running_m2 / float(self.config.out_steps)
            mean_checkpoints.append(running_mean)
            covariance_checkpoints.append(realized)
            residual_checkpoints.append(predicted_budget_white - realized)

        if count != self.config.out_steps:
            raise RuntimeError("Welford count does not match rollout horizon")
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

        realized_white = running_m2 / float(self.config.out_steps)
        residual_white = predicted_budget_white - realized_white
        covariance_checkpoints_white = torch.stack(
            covariance_checkpoints, dim=1)
        residual_checkpoints_white = torch.stack(
            residual_checkpoints, dim=1)
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
            # Public diagnostics use target-normalized u/v coordinates.
            "predicted_budget": self._white_to_target_normalized(
                predicted_budget_white),
            "realized_covariance": self._white_to_target_normalized(
                realized_white),
            "budget_residual": self._white_to_target_normalized(
                residual_white),
            "realized_covariance_checkpoints": (
                self._white_to_target_normalized(
                    covariance_checkpoints_white)),
            "budget_residual_checkpoints": self._white_to_target_normalized(
                residual_checkpoints_white),
            "covariance_checkpoint_steps": torch.arange(
                self.config.segment_length, self.config.out_steps + 1,
                self.config.segment_length, device=history.device),
            # Whitened packed values make accumulator audits inexpensive.
            "predicted_budget_white_packed": predicted_budget_white,
            "realized_covariance_white_packed": realized_white,
            "budget_residual_white_packed": residual_white,
            "welford_mean_white": running_mean,
            "welford_mean_checkpoints_white": torch.stack(
                mean_checkpoints, dim=1),
            "welford_statistic_segment_starts_white": torch.stack(
                statistic_segment_starts, dim=1),
        }


def build_covariance_budget_multiple_shooting(
        rollout_mode: str = "carry_detached", **overrides: object,
        ) -> CovarianceBudgetMultipleShootingClosure:
    return CovarianceBudgetMultipleShootingClosure(
        CovarianceBudgetMultipleShootingConfig(**overrides), rollout_mode)
