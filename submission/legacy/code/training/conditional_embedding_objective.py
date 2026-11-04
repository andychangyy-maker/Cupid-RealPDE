"""Conditional output-distribution alignment for spatiotemporal flow fields.

This is a compact PyTorch adaptation of the conditional embedding operator
discrepancy (CEOD) used by TL-DeepONet.  Rather than applying an RBF kernel to
millions of raw pixels, each flow is represented by differentiable physical
summaries that retain temporal means, fluctuation amplitudes, increments,
spatial gradients, vorticity and cross-stream profiles.  Conditioning uses the
same target-free summaries of the observed history.
"""

from __future__ import annotations

import math

import torch


def flow_summary(value: torch.Tensor) -> torch.Tensor:
    """Return sample-level physical summaries from ``[B,T,H,W,C>=2]``."""
    if value.ndim != 5 or value.shape[1] < 2 or value.shape[-1] < 2:
        raise ValueError("flow summary needs [B,T>=2,H,W,C>=2]")
    velocity = value[..., :2].float()
    spatial_mean = velocity.mean(dim=(2, 3)).flatten(1)
    spatial_std = velocity.std(dim=(2, 3), unbiased=False).flatten(1)
    increments = (velocity[:, 1:] - velocity[:, :-1]).square().mean(
        dim=(2, 3)).clamp_min(1.0e-12).sqrt().flatten(1)
    dy = velocity[:, :, 1:] - velocity[:, :, :-1]
    dx = velocity[:, :, :, 1:] - velocity[:, :, :, :-1]
    gradient_rms = torch.cat((
        dy.square().mean(dim=(2, 3)).clamp_min(1.0e-12).sqrt(),
        dx.square().mean(dim=(2, 3)).clamp_min(1.0e-12).sqrt(),
    ), dim=-1).flatten(1)
    # Curl on the common interior, using the image axes as fixed spatial axes.
    du_dy = velocity[:, :, 1:, :-1, 0] - velocity[:, :, :-1, :-1, 0]
    dv_dx = velocity[:, :, :-1, 1:, 1] - velocity[:, :, :-1, :-1, 1]
    vorticity_rms = (dv_dx - du_dy).square().mean(
        dim=(2, 3)).clamp_min(1.0e-12).sqrt()
    cross_stream_profile = velocity.mean(dim=(1, 3)).flatten(1)
    return torch.cat((
        spatial_mean, spatial_std, increments, gradient_rms,
        vorticity_rms, cross_stream_profile,
    ), dim=1)


def standardize_detached(
        value: torch.Tensor, reference: torch.Tensor | None = None,
        ) -> torch.Tensor:
    """Standardize features without allowing scale-collapse gradients."""
    source = value if reference is None else reference
    if source.ndim != 2 or value.ndim != 2 or source.shape[1] != value.shape[1]:
        raise ValueError("conditional embedding features must be rank two")
    center = source.mean(dim=0, keepdim=True).detach()
    scale = source.std(dim=0, unbiased=False, keepdim=True).detach().clamp_min(
        1.0e-4)
    return (value - center) / scale


def pairwise_squared_distance(
        left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("RBF inputs must be compatible rank-two tensors")
    left_norm = left.square().sum(dim=1, keepdim=True)
    right_norm = right.square().sum(dim=1, keepdim=True).transpose(0, 1)
    return (left_norm + right_norm - 2.0 * left @ right.transpose(0, 1)
            ).clamp_min(0.0) / max(left.shape[1], 1)


def median_rbf_kernel(
        left: torch.Tensor, right: torch.Tensor,
        *, bandwidth_source: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    distance = pairwise_squared_distance(left, right)
    source = distance if bandwidth_source is None else bandwidth_source
    positive = source.detach()[source.detach() > 1.0e-8]
    bandwidth = (
        positive.median() if positive.numel() else source.new_tensor(1.0)
    ).clamp_min(1.0e-4)
    return torch.exp(-distance / bandwidth), bandwidth


def conditional_embedding_discrepancy(
        history: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor,
        *, regularization: float = 0.10,
        ) -> tuple[torch.Tensor, dict[str, float]]:
    """Compare ``P(target|history)`` with ``P(prediction|history)``.

    The two empirical conditional operators share the same history samples.
    The trace expression follows the finite-sample CEOD estimator, with a
    detached normalization denominator so its scale is stable across batches.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target fields must have equal shape")
    batch = int(prediction.shape[0])
    if batch < 3:
        raise ValueError("conditional embedding needs at least three samples")
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("CEOD regularization must be positive and finite")

    condition = standardize_detached(flow_summary(history))
    target_summary = flow_summary(target)
    prediction_summary = flow_summary(prediction)
    joint = torch.cat((target_summary, prediction_summary), dim=0)
    target_features = standardize_detached(target_summary, joint)
    prediction_features = standardize_detached(prediction_summary, joint)

    condition_distance = pairwise_squared_distance(condition, condition)
    kxx, condition_bandwidth = median_rbf_kernel(
        condition, condition, bandwidth_source=condition_distance)
    output_joint = torch.cat((target_features, prediction_features), dim=0)
    output_distance = pairwise_squared_distance(output_joint, output_joint)
    target_kernel, output_bandwidth = median_rbf_kernel(
        target_features, target_features, bandwidth_source=output_distance)
    prediction_kernel, _ = median_rbf_kernel(
        prediction_features, prediction_features,
        bandwidth_source=output_distance)
    cross_kernel, _ = median_rbf_kernel(
        target_features, prediction_features,
        bandwidth_source=output_distance)

    identity = torch.eye(batch, device=kxx.device, dtype=kxx.dtype)
    inverse = torch.linalg.solve(
        kxx + batch * float(regularization) * identity, identity)
    left = inverse @ kxx @ inverse
    target_term = torch.trace(target_kernel @ left)
    prediction_term = torch.trace(prediction_kernel @ left)
    cross_term = torch.trace(cross_kernel @ left)
    raw = target_term + prediction_term - 2.0 * cross_term
    denominator = (target_term.detach() + prediction_term.detach()).clamp_min(
        1.0e-6)
    loss = (raw / denominator).clamp_min(0.0)
    if not torch.isfinite(loss):
        raise FloatingPointError("conditional embedding loss is non-finite")
    metrics = {
        "raw": float(raw.detach()),
        "normalized": float(loss.detach()),
        "target_term": float(target_term.detach()),
        "prediction_term": float(prediction_term.detach()),
        "cross_term": float(cross_term.detach()),
        "condition_bandwidth": float(condition_bandwidth.detach()),
        "output_bandwidth": float(output_bandwidth.detach()),
        "summary_dimensions": float(target_summary.shape[1]),
    }
    return loss, metrics
