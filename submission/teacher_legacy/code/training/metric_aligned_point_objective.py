"""Differentiable Track-1 point objective in physical u/v coordinates.

This follows the v9 scorer exactly for Rel-L2, TKE and MVPE, including the
per-sample relative errors, four probe profiles, mean-error score mapping and
published 46/10/9 point weights.  The only omitted channels are unmeasured
pressure channels; callers must convert normalized forecasts to physical units
before invoking the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class MetricAlignedPointWeights:
    rel_l2: float = 46.0
    tke: float = 10.0
    mvpe: float = 9.0

    def validate(self) -> None:
        values = (self.rel_l2, self.tke, self.mvpe)
        if (any(not math.isfinite(value) or value < 0.0 for value in values)
                or sum(values) <= 0.0):
            raise ValueError("point weights must be finite, nonnegative and nonzero")

    def tensor(self, reference: torch.Tensor) -> torch.Tensor:
        self.validate()
        return reference.new_tensor((self.rel_l2, self.tke, self.mvpe))


@dataclass
class MetricAlignedPointErrors:
    rel_l2: torch.Tensor
    tke: torch.Tensor
    mvpe: torch.Tensor


def _validate_fields(
        prediction: torch.Tensor,
        target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (prediction.shape != target.shape or prediction.ndim != 5
            or prediction.shape[-1] < 2):
        raise ValueError(
            "point objective requires aligned [B,T,H,W,C>=2] fields")
    if not (torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise FloatingPointError("point objective received non-finite fields")
    return prediction[..., :2].float(), target[..., :2].float()


def _relative_l2(
        prediction: torch.Tensor, target: torch.Tensor,
        epsilon: float) -> torch.Tensor:
    batch = prediction.shape[0]
    numerator = torch.linalg.vector_norm(
        (prediction - target).reshape(batch, -1), dim=1)
    denominator = torch.linalg.vector_norm(
        target.reshape(batch, -1), dim=1).clamp_min(epsilon)
    return numerator / denominator


def kinetic_energy(velocity: torch.Tensor) -> torch.Tensor:
    """Official population temporal variance, summed over u/v and halved."""
    fluctuation = velocity - velocity.mean(dim=1, keepdim=True)
    return 0.5 * fluctuation.square().mean(dim=1).sum(dim=-1)


def _mvpe_probe_coordinates(
        height: int, width: int, sub_s_real: int = 2,
        ) -> tuple[list[int], list[int]]:
    if sub_s_real <= 0:
        raise ValueError("MVPE subsampling factor must be positive")
    distance, center_x, center_y, probes = 16, 10, 32, 9
    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(height / (probes + 1)))
    probe_y = [
        probe_center_y + interval_y * offset
        for offset in range(-(probes - 1) // 2,
                            probes - (probes - 1) // 2)
    ]
    probe_y = [value for value in probe_y if 0 <= value < height]
    probe_x = []
    for index in range(4):
        if int((2 * distance + center_x) / sub_s_real) < width:
            value = int(((index + 1) * distance + center_x) / sub_s_real)
        else:
            value = int(
                (0.5 * (index + 2) * distance + center_x) / sub_s_real)
        if 0 <= value < width:
            probe_x.append(value)
    return probe_y, probe_x


def mvpe_relative_l2_per_sample(
        prediction: torch.Tensor, target: torch.Tensor, *,
        epsilon: float = 1.0e-8, sub_s_real: int = 2) -> torch.Tensor:
    """Exact differentiable v9 four-profile MVPE relative error."""
    batch, _steps, height, width, _channels = prediction.shape
    probe_y, probe_x = _mvpe_probe_coordinates(
        height, width, sub_s_real)
    if not probe_y or not probe_x:
        # Match the official zero-error fallback while retaining a valid graph.
        return prediction.reshape(batch, -1).sum(dim=1) * 0.0
    errors = []
    for x_value in probe_x:
        predicted = prediction[:, :, probe_y, x_value, :].mean(
            dim=1).reshape(batch, -1)
        truth = target[:, :, probe_y, x_value, :].mean(
            dim=1).reshape(batch, -1)
        errors.append(
            torch.linalg.vector_norm(predicted - truth, dim=1)
            / torch.linalg.vector_norm(truth, dim=1).clamp_min(epsilon))
    return torch.stack(errors, dim=0).mean(dim=0)


def metric_aligned_point_errors(
        prediction_physical: torch.Tensor,
        target_physical: torch.Tensor, *, epsilon: float = 1.0e-8,
        ) -> MetricAlignedPointErrors:
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("point objective epsilon must be finite and positive")
    prediction, target = _validate_fields(
        prediction_physical, target_physical)
    rel_l2 = _relative_l2(prediction, target, epsilon)
    tke = _relative_l2(
        kinetic_energy(prediction), kinetic_energy(target), epsilon)
    mvpe = mvpe_relative_l2_per_sample(
        prediction, target, epsilon=epsilon)
    return MetricAlignedPointErrors(rel_l2=rel_l2, tke=tke, mvpe=mvpe)


def metric_aligned_point_objective(
        prediction_physical: torch.Tensor,
        target_physical: torch.Tensor, *,
        weights: MetricAlignedPointWeights | None = None,
        epsilon: float = 1.0e-8,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Maximize the official 46/10/9 score combination on one batch.

    The official evaluator first averages each per-sample error and then maps
    that mean to a score.  This function preserves that order.  Scores in the
    diagnostic dictionary use the official 0--100 scale; the optimized loss
    is one minus the same weighted score normalized to 0--1.
    """
    resolved = MetricAlignedPointWeights() if weights is None else weights
    errors = metric_aligned_point_errors(
        prediction_physical, target_physical, epsilon=epsilon)
    mean_errors = torch.stack((
        errors.rel_l2.mean(), errors.tke.mean(), errors.mvpe.mean()))
    normalized_scores = 1.0 / (1.0 + 0.5 * mean_errors.clamp_min(0.0))
    weight = resolved.tensor(normalized_scores)
    weighted_score = (normalized_scores * weight).sum() / weight.sum()
    loss = 1.0 - weighted_score
    if not torch.isfinite(loss):
        raise FloatingPointError("metric-aligned point objective is non-finite")
    official_scores = 100.0 * normalized_scores
    contribution = (official_scores * (weight / 100.0)).sum()
    return loss, {
        "rel_l2_error": mean_errors[0].detach(),
        "tke_rel_l2_error": mean_errors[1].detach(),
        "mvpe_rel_l2_error": mean_errors[2].detach(),
        "rel_l2_score": official_scores[0].detach(),
        "tke_score": official_scores[1].detach(),
        "mvpe_score": official_scores[2].detach(),
        "published_point_contribution": contribution.detach(),
        "weighted_point_objective": loss.detach(),
    }

