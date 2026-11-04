"""Local temporal covariance descriptors for 20-step velocity forecasts.

The descriptors deliberately operate only on ``u/v``.  They provide a small,
fully differentiable objective for asking whether a forecast has the local
lagged covariance of its target, while a batch-rolled history provides a
deterministic null anchor.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn.functional as F


Anchor = Literal["candidate", "null"]
POOL_SIZE = (4, 8)
HISTORY_CROSS_STEPS = 8
CROSS_LAGS = (0, 1, 2, 3)
AUTOCOV_LAGS = (1, 2, 3, 4)


def _validate_inputs(history: torch.Tensor, future: torch.Tensor) -> None:
    """Check the narrow, intentional input contract used by this objective."""
    if history.ndim != 5 or future.ndim != 5:
        raise ValueError("history and future must have shape [B,20,H,W,3]")
    if history.shape != future.shape:
        raise ValueError("history and future must have identical shapes")
    if history.shape[1] != 20 or history.shape[-1] != 3:
        raise ValueError("history and future must have shape [B,20,H,W,3]")
    if history.shape[0] < 1:
        raise ValueError("batch dimension must be non-empty")
    if not (history.is_floating_point() and future.is_floating_point()):
        raise TypeError("history and future must be floating point tensors")


def _pooled_uv(value: torch.Tensor) -> torch.Tensor:
    """Average-pool velocity fields to ``[B,T,4,8,2]`` without detaching."""
    batch, steps, height, width, _channels = value.shape
    uv = value[..., :2].permute(0, 1, 4, 2, 3).reshape(
        batch * steps, 2, height, width)
    pooled = F.adaptive_avg_pool2d(uv, POOL_SIZE)
    return pooled.reshape(batch, steps, 2, *POOL_SIZE).permute(0, 1, 3, 4, 2)


def _circular_covariance(
        left: torch.Tensor, right: torch.Tensor,
        lags: tuple[int, ...]) -> torch.Tensor:
    """Per-location circular covariance for already time-centred inputs."""
    values = [
        (left * torch.roll(right, shifts=-lag, dims=1)).mean(dim=1)
        for lag in lags
    ]
    return torch.stack(values, dim=1)


def _autocovariance(
        value: torch.Tensor, lags: tuple[int, ...]) -> torch.Tensor:
    """Conventional non-wrapping temporal autocovariance."""
    return torch.stack([
        (value[:, :-lag] * value[:, lag:]).mean(dim=1)
        for lag in lags
    ], dim=1)


def local_lag_covariance_descriptor(
        history: torch.Tensor, future: torch.Tensor,
        *, anchor: Anchor = "candidate",
        donor_positions: Sequence[int] | torch.Tensor | None = None,
        ) -> torch.Tensor:
    """Return ``[B,8,4,8,2]`` lag-covariance statistics.

    The first four descriptor planes are circular cross-covariances at lags
    0--3 between the last eight history frames and first eight future frames.
    Both eight-frame series are independently time-centred.  The last four
    planes are full-future, zero-time-mean autocovariances at lags 1--4.

    ``anchor='candidate'`` compares each future to its own history.  The null
    anchor uses the explicit batch permutation in ``donor_positions``; it
    therefore changes only the cross-covariance portion.  The permutation must
    be complete and have no fixed positions.  Raw-window identity is not
    available at this tensor boundary and must additionally be checked by the
    schedule builder.
    """
    _validate_inputs(history, future)
    if anchor not in ("candidate", "null"):
        raise ValueError("anchor must be 'candidate' or 'null'")
    if anchor == "candidate" and donor_positions is not None:
        raise ValueError("candidate anchor must not receive donor positions")
    if anchor == "null" and donor_positions is None:
        raise ValueError("null anchor requires explicit donor positions")

    pooled_history = _pooled_uv(history)
    pooled_future = _pooled_uv(future)
    if anchor == "null":
        positions = torch.as_tensor(
            donor_positions, dtype=torch.long, device=history.device)
        expected = torch.arange(history.shape[0], device=history.device)
        if positions.ndim != 1 or positions.shape[0] != history.shape[0]:
            raise ValueError("donor positions must have shape [batch]")
        if not torch.equal(torch.sort(positions).values, expected):
            raise ValueError("donor positions must be a batch permutation")
        if bool(torch.any(positions == expected)):
            raise ValueError("donor positions must not contain fixed points")
        pooled_history = pooled_history.index_select(0, positions)

    history_tail = pooled_history[:, -HISTORY_CROSS_STEPS:]
    future_head = pooled_future[:, :HISTORY_CROSS_STEPS]
    history_tail = history_tail - history_tail.mean(dim=1, keepdim=True)
    future_head = future_head - future_head.mean(dim=1, keepdim=True)
    cross = _circular_covariance(history_tail, future_head, CROSS_LAGS)

    future_zero_mean = pooled_future - pooled_future.mean(dim=1, keepdim=True)
    autocovariance = _autocovariance(future_zero_mean, AUTOCOV_LAGS)
    return torch.cat((cross, autocovariance), dim=1)


def local_lag_covariance_loss(
        history: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor,
        *, anchor: Anchor = "candidate",
        donor_positions: Sequence[int] | torch.Tensor | None = None,
        ) -> torch.Tensor:
    """Return the raw MSE between forecast covariance descriptors.

    Inputs are already normalized by the fit-only field normalizer.  Keeping
    this descriptor loss unnormalized avoids dividing by near-zero target
    covariance in quiet regions; its single training weight is calibrated
    from prediction-space gradients after a shared point-only warmup.
    """
    _validate_inputs(history, prediction)
    _validate_inputs(history, target)
    prediction_descriptor = local_lag_covariance_descriptor(
        history, prediction, anchor=anchor,
        donor_positions=donor_positions)
    target_descriptor = local_lag_covariance_descriptor(
        history, target, anchor=anchor,
        donor_positions=donor_positions)
    return (prediction_descriptor - target_descriptor).square().mean()
