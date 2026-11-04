"""Reusable C208 RN1 role-native objectives and formal prefix gates.

This module contains only the CPU/GPU-agnostic mathematics shared by the
formal RN1 preflight and a future full Sim run.  It deliberately has no data,
checkpoint, runner, or CUDA dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


PASS_STATUS = "GO_RN1_TO_SIM10000"
STOP_STATUS = "STOP_RN1_PREFIX200"

AUDIT_UPDATES = (50, 100, 150, 200)
ROLE_SCALE = 0.020
ROLE_MEMBER_GROUPS = {"a": (0, 2), "b": (2, 4)}
EXACT_REPLACEMENT_TOLERANCE = 1e-7

POINT_COMPONENT_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")
SEEN_ROLE_GAIN_MIN = 0.05
UNSEEN_ROLE_GAIN_MIN = 0.025
UNSEEN_GAIN_FRACTION_OF_SEEN_MIN = 0.5
P1_RATIO_MAX = 1.01
POINT_DELTA_FLOOR = -0.03
COMPONENT_DELTA_FLOOR = -0.05
SOFT_SPS_DELTA_FLOOR = -0.005

ROLE_GRAD_BALANCE_MIN = 0.10
ROLE_TO_AGGREGATE_GRAD_RATIO_RANGE = (0.25, 4.0)
ROLE_GRAD_DIFFERENCE_MIN = 0.05

_ANCHOR_KEYS = (
    *POINT_COMPONENT_KEYS,
    "point_score",
    "native_soft_sps_proxy",
    "aggregate_p1_distill_loss",
    "a_role_loss",
    "b_role_loss",
    "role_loss",
    "swapped_role_loss",
    "specialization_margin",
)
_GATE_EPSILON = 1e-12


def _normalizer_uv(normalizer) -> tuple[torch.Tensor, torch.Tensor]:
    mean = normalizer.mean_target[:2]
    std = normalizer.std_target[:2]
    if mean.numel() != 2 or std.numel() != 2:
        raise ValueError("normalizer must provide two target velocity channels")
    if not bool(torch.isfinite(mean).all()) \
            or not bool(torch.isfinite(std).all()) \
            or bool((std == 0).any()):
        raise ValueError("normalizer target velocity statistics are invalid")
    return mean, std


def exact_zero_mean_fluctuation_loss(
        prediction: torch.Tensor, aggregate_teacher: torch.Tensor, normalizer,
        ) -> torch.Tensor:
    """Recompute the frozen component zero-mean fluctuation loss exactly."""
    if (prediction.ndim != 5
            or aggregate_teacher.ndim != prediction.ndim
            or aggregate_teacher.shape[:-1] != prediction.shape[:-1]
            or aggregate_teacher.shape[-1] != 2
            or prediction.shape[-1] < 2):
        raise ValueError(
            "aggregate teacher must align with prediction as [B,T,H,W,2]")
    mean, std = _normalizer_uv(normalizer)
    if not bool(torch.isfinite(prediction[..., :2]).all()) \
            or not bool(torch.isfinite(aggregate_teacher).all()):
        raise FloatingPointError("non-finite aggregate fluctuation input")

    # Keep this operation order identical to
    # train_ensemble_unet2d.component_point_distillation_loss.  Algebraically
    # cancelling the physical mean earlier can change float32 rounding.
    teacher_normalized = (aggregate_teacher.float() - mean) / std
    student_uv = prediction[..., :2].float()
    student_mean = student_uv.mean(dim=1, keepdim=True)
    teacher_mean = teacher_normalized.mean(dim=1, keepdim=True)
    student_fluctuation = student_uv - student_mean
    teacher_fluctuation = teacher_normalized - teacher_mean
    loss = F.smooth_l1_loss(
        student_fluctuation / ROLE_SCALE,
        teacher_fluctuation / ROLE_SCALE)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite aggregate fluctuation loss")
    return loss


def validate_exact_component_replacement(
        aggregate_fluctuation: torch.Tensor,
        detached_component: torch.Tensor,
        tolerance: float = EXACT_REPLACEMENT_TOLERANCE,
        ) -> float:
    """Return the replacement error, raising unless it is at most ``1e-7``."""
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("replacement tolerance must be finite and non-negative")
    if aggregate_fluctuation.ndim != 0 or detached_component.ndim != 0:
        raise ValueError("replacement losses must be scalar tensors")
    if detached_component.requires_grad:
        raise ValueError("reference component must be detached")
    error = abs(
        float(aggregate_fluctuation.detach()) - float(detached_component))
    if not math.isfinite(error) or error > tolerance:
        raise RuntimeError(
            "aggregate fluctuation does not exactly replace the frozen "
            f"component: error={error!r}, tolerance={tolerance!r}")
    return error


def role_distillation_losses(
        members: torch.Tensor, role_batch: Mapping[str, torch.Tensor],
        normalizer,
        ) -> dict[str, torch.Tensor]:
    """Compute assigned, swapped, and margin losses for four C208 members."""
    if members.ndim != 6 or members.shape[1] != 4 \
            or members.shape[-1] < 2:
        raise ValueError("members must have shape [B,4,T,H,W,C>=2]")
    _mean, std = _normalizer_uv(normalizer)
    expected_shape = (
        members.shape[0], *members.shape[2:5], 2)
    try:
        teacher_a = role_batch["a_fluctuation_uv_physical"]
        teacher_b = role_batch["b_fluctuation_uv_physical"]
    except KeyError as error:
        raise KeyError(f"missing RN1 role tensor: {error.args[0]}") from error
    for name, teacher in (("A", teacher_a), ("B", teacher_b)):
        if teacher.shape != expected_shape:
            raise ValueError(
                f"teacher {name} fluctuation must have shape {expected_shape}")
        if not bool(torch.isfinite(teacher).all()):
            raise FloatingPointError(
                f"teacher {name} fluctuation contains non-finite values")
    if not bool(torch.isfinite(members[..., :2]).all()):
        raise FloatingPointError("member predictions contain non-finite values")

    member_uv = members[..., :2].float()
    member_fluctuation = (
        member_uv - member_uv.mean(dim=2, keepdim=True))

    # A/B tensors are physical *increments*: divide by std only.  Subtracting
    # mean_target here would silently turn an increment into an absolute field.
    teacher_normalized = {
        "a": teacher_a.float() / std,
        "b": teacher_b.float() / std,
    }

    def assigned_loss(group_name: str, teacher_name: str) -> torch.Tensor:
        start, stop = ROLE_MEMBER_GROUPS[group_name]
        student = member_fluctuation[:, start:stop]
        target = teacher_normalized[teacher_name][:, None].expand_as(student)
        return F.smooth_l1_loss(
            student / ROLE_SCALE, target / ROLE_SCALE)

    loss_a = assigned_loss("a", "a")
    loss_b = assigned_loss("b", "b")
    loss_swapped = 0.5 * (
        assigned_loss("a", "b") + assigned_loss("b", "a"))
    loss_role = 0.5 * (loss_a + loss_b)
    return {
        "a": loss_a,
        "b": loss_b,
        "role": loss_role,
        "swapped": loss_swapped,
        "margin": loss_swapped - loss_role,
    }


def compose_role_native_distillation(
        control_distill: torch.Tensor,
        aggregate_fluctuation: torch.Tensor,
        role_losses: Mapping[str, torch.Tensor],
        ) -> torch.Tensor:
    """Replace only the aggregate fluctuation term: D_control-L_agg+L_role."""
    if control_distill.ndim != 0 or aggregate_fluctuation.ndim != 0:
        raise ValueError("distillation inputs must be scalar tensors")
    try:
        role = role_losses["role"]
    except KeyError as error:
        raise KeyError("role losses do not contain 'role'") from error
    if role.ndim != 0:
        raise ValueError("role loss must be a scalar tensor")
    result = control_distill - aggregate_fluctuation + role
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("non-finite role-native distillation loss")
    return result


def _materialize_gradient(
        gradient: torch.Tensor | None, reference: torch.Tensor,
        ) -> torch.Tensor:
    return torch.zeros_like(reference) if gradient is None else gradient


def _tensor_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).detach())


def _grads_unchanged(
        before: Sequence[torch.Tensor | None],
        parameters: Sequence[torch.nn.Parameter],
        ) -> bool:
    for saved, parameter in zip(before, parameters, strict=True):
        if saved is None:
            if parameter.grad is not None:
                return False
        elif parameter.grad is None or not torch.equal(saved, parameter.grad):
            return False
    return True


def gradient_audit(
        iteration: int, members: torch.Tensor,
        features: Sequence[torch.Tensor],
        aggregate_fluctuation: torch.Tensor,
        role_losses: Mapping[str, torch.Tensor],
        parameters: Iterable[torch.nn.Parameter],
        distill_weight: float,
        ) -> dict:
    """Audit output/shared-feature gradients without writing any ``.grad``."""
    if not features:
        raise ValueError("gradient audit requires the model feature sequence")
    feature = features[-1]
    if not isinstance(feature, torch.Tensor) or not feature.requires_grad:
        raise ValueError("last shared feature must require gradients")
    if not members.requires_grad:
        raise ValueError("member outputs must require gradients")
    if not math.isfinite(float(distill_weight)) or distill_weight < 0.0:
        raise ValueError("distillation weight must be finite and non-negative")
    parameters = list(parameters)
    saved_grads = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters]
    grads_none_before = all(saved is None for saved in saved_grads)

    weighted_a = 0.5 * distill_weight * role_losses["a"]
    weighted_b = 0.5 * distill_weight * role_losses["b"]
    weighted_aggregate = distill_weight * aggregate_fluctuation
    grad_a, feature_a = torch.autograd.grad(
        weighted_a, (members, feature), retain_graph=True,
        allow_unused=True)
    grad_b, feature_b = torch.autograd.grad(
        weighted_b, (members, feature), retain_graph=True,
        allow_unused=True)
    grad_aggregate = torch.autograd.grad(
        weighted_aggregate, members, retain_graph=True,
        allow_unused=True)[0]
    grad_a = _materialize_gradient(grad_a, members)
    grad_b = _materialize_gradient(grad_b, members)
    grad_aggregate = _materialize_gradient(grad_aggregate, members)
    feature_a = _materialize_gradient(feature_a, feature)
    feature_b = _materialize_gradient(feature_b, feature)

    grads_none_after = all(parameter.grad is None for parameter in parameters)
    grads_unchanged = _grads_unchanged(saved_grads, parameters)
    grad_role = grad_a + grad_b
    norm_a = _tensor_norm(grad_a)
    norm_b = _tensor_norm(grad_b)
    norm_role = _tensor_norm(grad_role)
    norm_aggregate = _tensor_norm(grad_aggregate)
    norm_difference = _tensor_norm(grad_role - grad_aggregate)
    feature_a_norm = _tensor_norm(feature_a)
    feature_b_norm = _tensor_norm(feature_b)
    cross_a_to_b = float(grad_a[:, 2:].abs().max())
    cross_b_to_a = float(grad_b[:, :2].abs().max())
    self_a = _tensor_norm(grad_a[:, :2])
    self_b = _tensor_norm(grad_b[:, 2:])
    balance = min(norm_a, norm_b) / max(norm_a, norm_b, 1e-30)
    role_to_aggregate = norm_role / max(norm_aggregate, 1e-30)
    relative_difference = norm_difference / max(norm_aggregate, 1e-30)
    cosine = float(
        torch.dot(grad_role.float().flatten(),
                  grad_aggregate.float().flatten())
        / max(norm_role * norm_aggregate, 1e-30))
    audited_losses = (
        role_losses["a"], role_losses["b"], role_losses["role"],
        aggregate_fluctuation)
    finite = (
        all(bool(torch.isfinite(value)) for value in audited_losses)
        and all(math.isfinite(value) for value in (
            norm_a, norm_b, norm_role, norm_aggregate, norm_difference,
            feature_a_norm, feature_b_norm, cross_a_to_b, cross_b_to_a,
            self_a, self_b, balance, role_to_aggregate,
            relative_difference, cosine))
        and all(bool(torch.isfinite(value).all()) for value in (
            grad_a, grad_b, grad_aggregate, feature_a, feature_b)))
    checks = {
        "finite": finite,
        "parameter_grads_none_before": grads_none_before,
        "parameter_grads_none_after": grads_none_after,
        "parameter_grads_unchanged": grads_unchanged,
        "a_own_group_nonzero": self_a > 0.0,
        "b_own_group_nonzero": self_b > 0.0,
        "a_cross_group_exact_zero": cross_a_to_b == 0.0,
        "b_cross_group_exact_zero": cross_b_to_a == 0.0,
        "a_feature_path_nonzero": feature_a_norm > 0.0,
        "b_feature_path_nonzero": feature_b_norm > 0.0,
        "role_group_gradient_balance": balance >= ROLE_GRAD_BALANCE_MIN,
        "role_to_aggregate_gradient_ratio": (
            ROLE_TO_AGGREGATE_GRAD_RATIO_RANGE[0]
            <= role_to_aggregate
            <= ROLE_TO_AGGREGATE_GRAD_RATIO_RANGE[1]),
        "role_gradient_materially_distinct": (
            relative_difference >= ROLE_GRAD_DIFFERENCE_MIN),
    }
    return {
        "iteration": int(iteration),
        "pass": all(checks.values()),
        "checks": checks,
        "norm_g_a": norm_a,
        "norm_g_b": norm_b,
        "norm_g_role": norm_role,
        "norm_g_aggregate": norm_aggregate,
        "norm_g_role_minus_g_aggregate": norm_difference,
        "norm_feature_a": feature_a_norm,
        "norm_feature_b": feature_b_norm,
        "cross_a_to_b_max_abs": cross_a_to_b,
        "cross_b_to_a_max_abs": cross_b_to_a,
        "role_group_gradient_balance": balance,
        "role_to_aggregate_gradient_ratio": role_to_aggregate,
        "relative_role_gradient_difference": relative_difference,
        "role_aggregate_cosine": cosine,
        "cosine_is_diagnostic_only": True,
    }


def screen_gradient_audits(audits: Sequence[Mapping]) -> dict:
    """Require one passing audit at each frozen RN1 audit update."""
    rows = list(audits)
    updates = [row.get("iteration") for row in rows]
    checks = {
        "all_audits_present": updates == list(AUDIT_UPDATES),
        "all_audits_pass": (
            len(rows) == len(AUDIT_UPDATES)
            and all(row.get("pass") is True for row in rows)),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "updates": updates,
        "audits": rows,
    }


def _finite_anchor(anchor: Mapping[str, float]) -> dict[str, float]:
    values = {key: float(anchor[key]) for key in _ANCHOR_KEYS}
    return values


def _role_gain(control: Mapping[str, float],
               candidate: Mapping[str, float]) -> float | None:
    reference = control["role_loss"]
    if not math.isfinite(reference) or reference <= 0.0:
        return None
    gain = 1.0 - candidate["role_loss"] / reference
    return gain if math.isfinite(gain) else None


def _relative_at_most(value: float, reference: float,
                      multiplier: float) -> bool:
    if reference <= _GATE_EPSILON:
        return value <= reference + _GATE_EPSILON
    return value <= multiplier * reference + _GATE_EPSILON


def _anchor_gate(
        control_input: Mapping[str, float],
        candidate_input: Mapping[str, float],
        role_gain_min: float,
        seen_role_gain: float | None = None,
        ) -> dict:
    control = _finite_anchor(control_input)
    candidate = _finite_anchor(candidate_input)
    finite = all(math.isfinite(value) for value in (
        *control.values(), *candidate.values()))
    role_gain = _role_gain(control, candidate) if finite else None
    point_delta = candidate["point_score"] - control["point_score"]
    component_delta = {
        key: candidate[key] - control[key] for key in POINT_COMPONENT_KEYS}
    soft_sps_delta = (
        candidate["native_soft_sps_proxy"]
        - control["native_soft_sps_proxy"])
    p1_reference = control["aggregate_p1_distill_loss"]
    p1_ratio = (
        candidate["aggregate_p1_distill_loss"] / p1_reference
        if p1_reference > _GATE_EPSILON else None)
    checks = {
        "finite": finite,
        "role_gain": (
            role_gain is not None
            and role_gain >= role_gain_min - _GATE_EPSILON),
        "a_role_non_regression": (
            candidate["a_role_loss"]
            <= control["a_role_loss"] + _GATE_EPSILON),
        "b_role_non_regression": (
            candidate["b_role_loss"]
            <= control["b_role_loss"] + _GATE_EPSILON),
        "aggregate_p1_non_regression": _relative_at_most(
            candidate["aggregate_p1_distill_loss"],
            control["aggregate_p1_distill_loss"], P1_RATIO_MAX),
        "point_non_regression": (
            point_delta >= POINT_DELTA_FLOOR - _GATE_EPSILON),
        "components_non_regression": all(
            delta >= COMPONENT_DELTA_FLOOR - _GATE_EPSILON
            for delta in component_delta.values()),
        "soft_sps_non_regression": (
            soft_sps_delta >= SOFT_SPS_DELTA_FLOOR - _GATE_EPSILON),
        "positive_specialization_margin": (
            candidate["specialization_margin"] > 0.0),
    }
    if seen_role_gain is not None:
        checks["role_gain_tracks_seen"] = (
            role_gain is not None
            and role_gain >= (
                UNSEEN_GAIN_FRACTION_OF_SEEN_MIN * seen_role_gain
                - _GATE_EPSILON))
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "role_gain": role_gain,
        "aggregate_p1_ratio": p1_ratio,
        "deltas": {
            "point_score": point_delta,
            **component_delta,
            "native_soft_sps_proxy": soft_sps_delta,
        },
    }


def prefix_gate(
        audits: Sequence[Mapping],
        seen_control: Mapping[str, float],
        seen_candidate: Mapping[str, float],
        unseen_control: Mapping[str, float],
        unseen_candidate: Mapping[str, float],
        *, batch_identity: bool, prefix_identity: bool,
        ) -> dict:
    """Apply the frozen Seen/Unseen RN1 prefix-200 GO/STOP contract."""
    gradient_screen = screen_gradient_audits(audits)
    seen = _anchor_gate(
        seen_control, seen_candidate, SEEN_ROLE_GAIN_MIN)
    unseen = _anchor_gate(
        unseen_control, unseen_candidate, UNSEEN_ROLE_GAIN_MIN,
        seen_role_gain=seen["role_gain"])
    checks = {
        "batch_identity": batch_identity is True,
        "prefix_identity": prefix_identity is True,
        "gradient_screen": gradient_screen["pass"],
        "seen": seen["pass"],
        "unseen": unseen["pass"],
    }
    passed = all(checks.values())
    return {
        "status": PASS_STATUS if passed else STOP_STATUS,
        "pass": passed,
        "checks": checks,
        "seen": seen,
        "unseen": unseen,
        "gradient_screen": gradient_screen,
    }


__all__ = [
    "AUDIT_UPDATES",
    "COMPONENT_DELTA_FLOOR",
    "EXACT_REPLACEMENT_TOLERANCE",
    "PASS_STATUS",
    "P1_RATIO_MAX",
    "POINT_COMPONENT_KEYS",
    "POINT_DELTA_FLOOR",
    "ROLE_GRAD_BALANCE_MIN",
    "ROLE_GRAD_DIFFERENCE_MIN",
    "ROLE_MEMBER_GROUPS",
    "ROLE_SCALE",
    "ROLE_TO_AGGREGATE_GRAD_RATIO_RANGE",
    "SEEN_ROLE_GAIN_MIN",
    "SOFT_SPS_DELTA_FLOOR",
    "STOP_STATUS",
    "UNSEEN_GAIN_FRACTION_OF_SEEN_MIN",
    "UNSEEN_ROLE_GAIN_MIN",
    "compose_role_native_distillation",
    "exact_zero_mean_fluctuation_loss",
    "gradient_audit",
    "prefix_gate",
    "role_distillation_losses",
    "screen_gradient_audits",
    "validate_exact_component_replacement",
]
