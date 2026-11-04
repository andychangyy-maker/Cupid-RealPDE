"""Strictly proper finite-ensemble objectives for functional predictions."""

from __future__ import annotations

import torch


def conditional_distribution_targets(
        base_target: torch.Tensor, future_target: torch.Tensor, *,
        members_only: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Route synthetic futures without silently changing UQ calibration.

    Members always receive the future branch.  In the isolated v2 contract,
    quantile/SPS heads retain the released base truth by object identity.
    """
    if base_target.shape != future_target.shape:
        raise ValueError("base and future targets must have identical shapes")
    return future_target, (base_target if members_only else future_target)


def functional_energy_score(
        members: torch.Tensor, target: torch.Tensor, *,
        channels: int = 2, epsilon: float = 1.0e-12,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return energy score, target attraction, and ensemble repulsion.

    Each forecast is treated as one function over time and space rather than
    as independent pixels.  The unbiased finite-ensemble form is

    ``mean_m ||x_m-y|| - mean_{m<m'} ||x_m-x_m'|| / 2``.

    Norms are root-mean-square Hilbert norms, which keeps the loss scale
    independent of grid resolution.  Only u/v are used by default because
    Track 1 uncertainty scoring excludes pressure.
    """
    if members.ndim != 6 or target.ndim != 5:
        raise ValueError("members/target must be [B,M,T,H,W,C]/[B,T,H,W,C]")
    if members.shape[0] != target.shape[0] or members.shape[2:] != target.shape[1:]:
        raise ValueError("members and target shapes are not aligned")
    if members.shape[1] < 2:
        raise ValueError("energy score needs at least two ensemble members")
    if channels <= 0 or channels > members.shape[-1]:
        raise ValueError("invalid energy-score channel count")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    forecasts = members[..., :channels].float()
    truth = target[..., :channels].float().unsqueeze(1)
    functional_dims = tuple(range(2, forecasts.ndim))
    target_distance = torch.sqrt(
        (forecasts - truth).square().mean(dim=functional_dims) + epsilon)
    attraction = target_distance.mean()

    pairs = []
    for first in range(forecasts.shape[1]):
        for second in range(first + 1, forecasts.shape[1]):
            pairs.append(torch.sqrt(
                (forecasts[:, first] - forecasts[:, second]).square().mean(
                    dim=tuple(range(1, forecasts.ndim - 1))) + epsilon))
    repulsion = torch.stack(pairs, dim=1).mean()
    return attraction - 0.5 * repulsion, attraction, repulsion


def pointwise_ensemble_crps(
        members: torch.Tensor, target: torch.Tensor, *, channels: int = 2,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return empirical CRPS averaged over forecast elements.

    Unlike :func:`functional_energy_score`, every time/pixel/channel element
    is treated as its own scalar predictive distribution.  For ``M`` members
    this is ``mean_m |x_m-y| - mean_{m,m'} |x_m-x_m'| / 2``.  The diagonal
    pair terms are exactly zero and are included in the empirical-distribution
    normalization.
    """
    if members.ndim != 6 or target.ndim != 5:
        raise ValueError("members/target must be [B,M,T,H,W,C]/[B,T,H,W,C]")
    if members.shape[0] != target.shape[0] or members.shape[2:] != target.shape[1:]:
        raise ValueError("members and target shapes are not aligned")
    if members.shape[1] < 2:
        raise ValueError("pointwise CRPS needs at least two ensemble members")
    if channels <= 0 or channels > members.shape[-1]:
        raise ValueError("invalid CRPS channel count")

    forecasts = members[..., :channels].float()
    truth = target[..., :channels].float().unsqueeze(1)
    attraction = (forecasts - truth).abs().mean()
    pair_sum = forecasts.new_zeros(())
    for first in range(forecasts.shape[1]):
        for second in range(first + 1, forecasts.shape[1]):
            pair_sum = pair_sum + (
                forecasts[:, first] - forecasts[:, second]).abs().mean()
    members_count = forecasts.shape[1]
    repulsion = 2.0 * pair_sum / (members_count * members_count)
    return attraction - 0.5 * repulsion, attraction, repulsion
