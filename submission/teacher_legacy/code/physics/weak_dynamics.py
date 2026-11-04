"""Signed block-integral dynamics matching for noisy velocity forecasts.

The competition exposes two velocity components but not the complete pressure,
forcing, geometry, or out-of-plane state needed for a closed Navier--Stokes
residual.  This module therefore never forces a guessed PDE to zero.  It
matches observed temporal increments, divergence, and vorticity only after
integrating them over fixed spacetime blocks.  The top-hat block functions are
a simple discrete weak form: derivative noise cancels while coherent transport
survives.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


def signed_dynamics_fields(velocity: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return signed dt/divergence/vorticity fields from physical u/v.

    Args:
        velocity: ``[B,T,H,W,2]`` physical velocity.

    Spatial fields use the common upper-left cell grid ``[H-1,W-1]`` so every
    sum is a finite-volume boundary difference rather than an absolute-gradient
    statistic.
    """
    if velocity.ndim != 5 or velocity.shape[-1] != 2:
        raise ValueError("velocity must have shape (B,T,H,W,2)")
    if min(velocity.shape[1:4]) < 2:
        raise ValueError("weak dynamics requires T,H,W >= 2")
    temporal = velocity[:, 1:] - velocity[:, :-1]
    du_dx = velocity[:, :, :, 1:, 0] - velocity[:, :, :, :-1, 0]
    dv_dx = velocity[:, :, :, 1:, 1] - velocity[:, :, :, :-1, 1]
    du_dy = velocity[:, :, 1:, :, 0] - velocity[:, :, :-1, :, 0]
    dv_dy = velocity[:, :, 1:, :, 1] - velocity[:, :, :-1, :, 1]
    divergence = (
        du_dx[:, :, :-1, :] + dv_dy[:, :, :, :-1]).unsqueeze(-1)
    vorticity = (
        dv_dx[:, :, :-1, :] - du_dy[:, :, :, :-1]).unsqueeze(-1)
    return {
        "temporal": temporal,
        "divergence": divergence,
        "vorticity": vorticity,
    }


def block_integral_moments(
        field: torch.Tensor, temporal_block: int, spatial_block: int,
        ) -> torch.Tensor:
    """Average a signed ``[B,T,H,W,C]`` field over non-overlapping blocks."""
    if field.ndim != 5:
        raise ValueError("weak field must have shape (B,T,H,W,C)")
    if temporal_block <= 0 or spatial_block <= 0:
        raise ValueError("weak block dimensions must be positive")
    if (field.shape[1] < temporal_block
            or field.shape[2] < spatial_block
            or field.shape[3] < spatial_block):
        raise ValueError("weak block is larger than its field")
    channels_first = field.permute(0, 4, 1, 2, 3)
    return F.avg_pool3d(
        channels_first,
        kernel_size=(temporal_block, spatial_block, spatial_block),
        stride=(temporal_block, spatial_block, spatial_block))


def weak_dynamics_loss_physical(
        prediction_uv: torch.Tensor, target_uv: torch.Tensor, *,
        temporal_block: int = 4,
        spatial_blocks: Sequence[int] = (4, 8),
        scale_floor: float = 1.0e-4,
        return_components: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match normalized signed weak moments of physical velocity dynamics.

    Target RMS is computed only from each detached training batch and is used
    as a unit conversion, not a target-dependent inference input.  A floor
    prevents nearly cancelling divergence moments from receiving unbounded
    weight.  Absolute velocity is intentionally absent: the ordinary point
    objective retains that responsibility.
    """
    if prediction_uv.shape != target_uv.shape:
        raise ValueError("weak prediction and target shapes must match")
    if not 0.0 < scale_floor < float("inf"):
        raise ValueError("weak dynamics scale floor must be positive")
    blocks = tuple(int(value) for value in spatial_blocks)
    if not blocks or any(value <= 0 for value in blocks):
        raise ValueError("weak dynamics spatial blocks must be positive")
    predicted_fields = signed_dynamics_fields(prediction_uv.float())
    target_fields = signed_dynamics_fields(target_uv.float())
    components: dict[str, torch.Tensor] = {}
    for name in predicted_fields:
        for spatial_block in blocks:
            predicted = block_integral_moments(
                predicted_fields[name], temporal_block, spatial_block)
            target = block_integral_moments(
                target_fields[name], temporal_block, spatial_block)
            scale = target.detach().square().mean(
                dim=(0, 2, 3, 4), keepdim=True).sqrt().clamp_min(scale_floor)
            components[f"{name}_b{spatial_block}"] = F.smooth_l1_loss(
                (predicted - target) / scale, torch.zeros_like(predicted))
    total = torch.stack(tuple(components.values())).mean()
    if return_components:
        return total, components
    return total
