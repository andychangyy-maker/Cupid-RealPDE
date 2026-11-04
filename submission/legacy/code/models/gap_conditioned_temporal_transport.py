"""Gap-conditioned variant of the small temporal-transport forecaster.

This module deliberately changes only the decoder query.  The spatial
encoder, causal mixer, transport decomposition, and output heads are reused
unchanged from :mod:`models.temporal_transport`.  It is therefore suitable
for a small, paired discovery experiment asking a narrow question: does
telling a forecaster how far after the observed history its target begins
help it learn a more useful short-horizon representation?

``gap_steps=0`` is an *exact* no-op.  This makes the ordinary competition
20->20 task the deployment path, and lets fixed-lead and lead-blind control
arms use precisely the same tensors and forward operation at lead zero.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .temporal_transport import (
    HorizonQueryDecoder,
    TemporalTransportConfig,
    TemporalTransportOperator,
)


class GapConditionedHorizonQueryDecoder(nn.Module):
    """Horizon decoder with a zero-preserving continuous gap condition."""

    def __init__(self, config: TemporalTransportConfig, max_gap_steps: int):
        super().__init__()
        if max_gap_steps <= 0:
            raise ValueError("max_gap_steps must be positive")
        # Initialise the shared path bit-for-bit as the legacy decoder.  This
        # is convenient for comparisons, while every arm in the new screen
        # still instantiates the same augmented parameter tensors.
        base = HorizonQueryDecoder(config)
        self.out_steps = int(base.out_steps)
        self.scale = float(base.scale)
        self.query = base.query
        self.key = base.key
        self.value = base.value
        self.query_projection = base.query_projection
        self.decoder = base.decoder
        self.max_gap_steps = int(max_gap_steps)
        # Features are all exactly zero at gap=0; bias=False preserves that
        # invariant, so normal inference needs no special calibration.
        self.gap_projection = nn.Linear(3, config.hidden_channels, bias=False)

    def _condition(
            self, gap_steps: torch.Tensor | None, batch: int,
            reference: torch.Tensor) -> torch.Tensor:
        if gap_steps is None:
            return reference.new_zeros(batch, reference.shape[2])
        gap = torch.as_tensor(gap_steps, device=reference.device,
                              dtype=reference.dtype)
        if gap.ndim == 0:
            gap = gap.expand(batch)
        if gap.shape != (batch,):
            raise ValueError("gap_steps must be scalar or shape [B]")
        if torch.any(gap < 0) or torch.any(gap > self.max_gap_steps):
            raise ValueError("gap_steps lies outside the configured range")
        phase = gap / float(self.max_gap_steps)
        features = torch.stack((
            phase,
            torch.sin(torch.pi * phase),
            1.0 - torch.cos(torch.pi * phase),
        ), dim=1)
        return self.gap_projection(features)

    @staticmethod
    def _project_frames(
            projection: nn.Module, trajectory: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = trajectory.shape
        projected = projection(trajectory.reshape(
            batch * steps, channels, height, width))
        return projected.reshape(batch, steps, channels, height, width)

    def forward(
            self, trajectory: torch.Tensor,
            gap_steps: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        if trajectory.ndim != 5:
            raise ValueError("query decoder requires [B,T,C,H,W]")
        batch, _steps, channels, height, width = trajectory.shape
        keys = self._project_frames(self.key, trajectory)
        values = self._project_frames(self.value, trajectory)
        condition = self._condition(gap_steps, batch, trajectory)
        queries = self.query_projection(self.query).to(trajectory)
        queries = queries.unsqueeze(0) + condition.unsqueeze(1)
        scores = torch.einsum("btdhw,bod->bothw", keys, queries) / self.scale
        attention = scores.float().softmax(dim=2).to(values)
        context = torch.einsum("bothw,btdhw->bodhw", attention, values)
        horizon = queries.view(batch, self.out_steps, channels, 1, 1)
        decoded = context + trajectory[:, -1:].expand(
            -1, self.out_steps, -1, -1, -1) + horizon
        decoded = self.decoder(decoded.reshape(
            batch * self.out_steps, channels, height, width))
        return decoded.reshape(
            batch, self.out_steps, channels, height, width), attention


class GapConditionedTemporalTransportOperator(TemporalTransportOperator):
    """Temporal transport operator whose decoder may see a target gap."""

    def __init__(
            self, config: TemporalTransportConfig,
            *, max_gap_steps: int = 20):
        super().__init__(config)
        self.max_gap_steps = int(max_gap_steps)
        self.horizon_decoder = GapConditionedHorizonQueryDecoder(
            config, self.max_gap_steps)

    def architecture_config(self) -> dict:
        result = super().architecture_config()
        result["architecture"] = "gap_conditioned_temporal_transport"
        result["max_gap_steps"] = self.max_gap_steps
        return result

    def forward_components(
            self, history: torch.Tensor,
            gap_steps: torch.Tensor | None = None,
            ):
        # This is the parent decomposition verbatim except for the query
        # decoder call.  Keeping it local avoids modifying the production
        # point model merely for a discovery-only screen.
        self._validate_history(history)
        target_size = tuple(history.shape[2:4])
        trajectory = self.encode_history(history)
        decoded, attention = self.horizon_decoder(trajectory, gap_steps)
        batch, steps, channels, latent_height, latent_width = decoded.shape

        flat = decoded.reshape(batch * steps, channels, latent_height, latent_width)
        raw_displacement = self.transport_head(flat).reshape(
            batch, steps, 2, latent_height, latent_width)
        displacement = self._resize_horizon_field(raw_displacement, target_size)
        displacement = self.config.max_displacement * torch.tanh(displacement)
        transported = self.warp(self._persistence(history), displacement)

        raw_fluctuation = self.fluctuation_head(flat).reshape(
            batch, steps, self.config.out_channels, latent_height, latent_width)
        fluctuation = self._resize_horizon_field(raw_fluctuation, target_size)
        fluctuation = self.config.residual_limit * torch.tanh(
            fluctuation / self.config.residual_limit)
        fluctuation = fluctuation - fluctuation.mean(dim=1, keepdim=True)

        # A shifted target block can have a different time-constant mean as
        # well as different within-block fluctuations.  Reuse the same tiny
        # condition here; otherwise the exactly zero-time-mean fluctuation
        # path would make that part of the gap semantics unrepresentable.
        gap_condition = self.horizon_decoder._condition(
            gap_steps, batch, trajectory).view(batch, channels, 1, 1)
        mean_feature = self.mean_decoder(
            trajectory[:, -1] + gap_condition)
        mean_delta = self.mean_head(mean_feature)
        if mean_delta.shape[-2:] != target_size:
            mean_delta = F.interpolate(
                mean_delta, size=target_size, mode="bilinear",
                align_corners=False)
        mean_delta = mean_delta.permute(0, 2, 3, 1)
        mean_delta = self.config.residual_limit * torch.tanh(
            mean_delta / self.config.residual_limit)

        # Import lazily to keep the public import surface aligned with the
        # original module and avoid copying its dataclass definition.
        from .temporal_transport import TemporalTransportOutput
        prediction = transported + mean_delta.unsqueeze(1) + fluctuation
        return TemporalTransportOutput(
            prediction=prediction, transported=transported,
            mean_delta=mean_delta, fluctuation=fluctuation,
            displacement=displacement, attention=attention)

    def forward(
            self, history: torch.Tensor,
            gap_steps: torch.Tensor | None = None,
            ) -> torch.Tensor:
        return self.forward_components(history, gap_steps).prediction
