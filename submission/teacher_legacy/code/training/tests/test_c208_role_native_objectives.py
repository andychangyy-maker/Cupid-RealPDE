from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import c208_role_native_objectives as subject  # noqa: E402
import train_ensemble_unet2d as frozen_objectives  # noqa: E402


def normalizer():
    return SimpleNamespace(
        mean_target=torch.tensor([10.0, -20.0, 0.0]),
        std_target=torch.tensor([2.0, 4.0, 1.0]),
    )


def role_targets(norm):
    temporal = torch.tensor([-1.0, 1.0]).view(1, 2, 1, 1, 1)
    a_normalized = temporal * torch.tensor([0.020, 0.015])
    b_normalized = temporal * torch.tensor([0.010, 0.005])
    role_batch = {
        "a_fluctuation_uv_physical": (
            a_normalized * norm.std_target[:2]),
        "b_fluctuation_uv_physical": (
            b_normalized * norm.std_target[:2]),
    }
    return a_normalized, b_normalized, role_batch


def add_third_channel(member_uv):
    return torch.cat(
        (member_uv, torch.zeros_like(member_uv[..., :1])), dim=-1)


def test_role_assignment_uses_std_only_and_swapped_assignment_is_worse():
    norm = normalizer()
    a_normalized, b_normalized, role_batch = role_targets(norm)
    member_uv = torch.cat((
        a_normalized[:, None].expand(-1, 2, -1, -1, -1, -1),
        b_normalized[:, None].expand(-1, 2, -1, -1, -1, -1),
    ), dim=1)
    losses = subject.role_distillation_losses(
        add_third_channel(member_uv), role_batch, norm)

    assert losses["a"] == pytest.approx(0.0, abs=1e-8)
    assert losses["b"] == pytest.approx(0.0, abs=1e-8)
    assert losses["role"] == pytest.approx(0.0, abs=1e-8)
    assert float(losses["swapped"]) > 0.0
    assert torch.equal(losses["margin"], losses["swapped"])

    swapped = torch.cat((member_uv[:, 2:], member_uv[:, :2]), dim=1)
    swapped_losses = subject.role_distillation_losses(
        add_third_channel(swapped), role_batch, norm)
    assert float(swapped_losses["role"]) > float(losses["role"])


def test_aggregate_component_is_exact_and_role_native_is_one_term_replacement():
    torch.manual_seed(7)
    norm = SimpleNamespace(
        mean_target=torch.tensor([0.15, -0.01, 0.0]),
        std_target=torch.tensor([0.10, 0.02, 1.0]),
    )
    student = torch.randn(2, 4, 20, 40, 3, requires_grad=True)
    teacher = (
        student.detach()[..., :2] * norm.std_target[:2]
        + norm.mean_target[:2])
    teacher = teacher + torch.linspace(-0.004, 0.006, 4).view(
        1, 4, 1, 1, 1)
    control, components = frozen_objectives.component_point_distillation_loss(
        student, teacher, norm)
    aggregate = subject.exact_zero_mean_fluctuation_loss(
        student, teacher, norm)
    error = subject.validate_exact_component_replacement(
        aggregate, components["distill_zero_mean_fluctuation"])
    assert error <= 1e-7

    replacement = {"role": 1.25 * aggregate}
    role_native = subject.compose_role_native_distillation(
        control, aggregate, replacement)
    expected = (
        2.0 * components["distill_temporal_mean"]
        + replacement["role"].detach()
        + 2.0 * components["distill_mean_profile"]
        + 0.25 * components["distill_physical_tke_map"])
    assert torch.allclose(
        role_native.detach(), expected, rtol=0.0, atol=1e-6)

    with pytest.raises(RuntimeError, match="does not exactly replace"):
        subject.validate_exact_component_replacement(
            aggregate + 1e-4,
            components["distill_zero_mean_fluctuation"])


def test_gradient_audit_masks_groups_reaches_shared_feature_and_never_writes_grad():
    norm = SimpleNamespace(
        mean_target=torch.zeros(3), std_target=torch.ones(3))
    a_normalized, b_normalized, role_batch = role_targets(norm)
    shared_parameter = torch.nn.Parameter(torch.tensor(0.004))
    temporal = torch.tensor([-1.0, 1.0]).view(1, 2, 1, 1, 1)
    feature = shared_parameter * temporal.expand(-1, -1, -1, -1, 2)
    member_uv = torch.cat((
        (a_normalized + feature)[:, None].expand(-1, 2, -1, -1, -1, -1),
        (b_normalized + 2.0 * feature)[:, None].expand(
            -1, 2, -1, -1, -1, -1),
    ), dim=1)
    members = add_third_channel(member_uv)
    prediction = members.mean(dim=1)
    aggregate_teacher = 0.5 * sum(role_batch.values())
    roles = subject.role_distillation_losses(members, role_batch, norm)
    aggregate = subject.exact_zero_mean_fluctuation_loss(
        prediction, aggregate_teacher, norm)

    audit = subject.gradient_audit(
        50, members, [feature], aggregate, roles,
        [shared_parameter], distill_weight=0.005)
    assert audit["pass"] is True
    assert audit["cross_a_to_b_max_abs"] == 0.0
    assert audit["cross_b_to_a_max_abs"] == 0.0
    assert audit["norm_feature_a"] > 0.0
    assert audit["norm_feature_b"] > 0.0
    assert audit["cosine_is_diagnostic_only"] is True
    assert shared_parameter.grad is None

    subject.compose_role_native_distillation(
        aggregate, aggregate, roles).backward()
    assert shared_parameter.grad is not None
    assert torch.isfinite(shared_parameter.grad)


def passing_audits():
    return [{"iteration": update, "pass": True, "checks": {}}
            for update in subject.AUDIT_UPDATES]


def anchor(role_loss: float, *, role_gain_margin: float = 0.1):
    return {
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 85.0,
        "point_score": 86.0,
        "native_soft_sps_proxy": 0.5,
        "aggregate_p1_distill_loss": 10.0,
        "a_role_loss": 2.0,
        "b_role_loss": 3.0,
        "role_loss": role_loss,
        "swapped_role_loss": role_loss + role_gain_margin,
        "specialization_margin": role_gain_margin,
    }


def boundary_candidate(control, role_loss):
    candidate = dict(control)
    candidate.update({
        "rel_l2_score": control["rel_l2_score"] - 0.05,
        "tke_rel_l2_score": control["tke_rel_l2_score"] - 0.05,
        "mvpe_rel_l2_score": control["mvpe_rel_l2_score"] - 0.05,
        "point_score": control["point_score"] - 0.03,
        "native_soft_sps_proxy": control["native_soft_sps_proxy"] - 0.005,
        "aggregate_p1_distill_loss": (
            1.01 * control["aggregate_p1_distill_loss"]),
        "role_loss": role_loss,
        "swapped_role_loss": role_loss + 1e-8,
        "specialization_margin": 1e-8,
    })
    return candidate


def test_formal_gate_includes_inclusive_seen_unseen_and_identity_boundaries():
    seen_control = anchor(1.0)
    unseen_control = anchor(1.0)
    seen_candidate = boundary_candidate(seen_control, 0.95)
    unseen_candidate = boundary_candidate(unseen_control, 0.975)
    result = subject.prefix_gate(
        passing_audits(), seen_control, seen_candidate,
        unseen_control, unseen_candidate,
        batch_identity=True, prefix_identity=True)

    assert set(result) == {
        "status", "pass", "checks", "seen", "unseen", "gradient_screen"}
    assert result["pass"] is True
    assert result["status"] == subject.PASS_STATUS
    assert result["seen"]["role_gain"] == pytest.approx(0.05)
    assert result["unseen"]["role_gain"] == pytest.approx(0.025)
    assert all(result["checks"].values())

    wrong_identity = subject.prefix_gate(
        passing_audits(), seen_control, seen_candidate,
        unseen_control, unseen_candidate,
        batch_identity=False, prefix_identity=True)
    assert wrong_identity["pass"] is False
    assert wrong_identity["status"] == subject.STOP_STATUS
    assert wrong_identity["checks"]["batch_identity"] is False


def test_four_audit_screen_and_formal_gate_stop_when_one_audit_is_missing():
    audits = passing_audits()
    assert subject.screen_gradient_audits(audits)["pass"] is True
    missing = audits[:-1]
    screen = subject.screen_gradient_audits(missing)
    assert screen["pass"] is False
    assert screen["checks"]["all_audits_present"] is False

    seen_control = anchor(1.0)
    unseen_control = anchor(1.0)
    result = subject.prefix_gate(
        missing,
        seen_control, boundary_candidate(seen_control, 0.95),
        unseen_control, boundary_candidate(unseen_control, 0.975),
        batch_identity=True, prefix_identity=True)
    assert result["pass"] is False
    assert result["status"] == subject.STOP_STATUS
    assert result["checks"]["gradient_screen"] is False

    below_seen_gain = boundary_candidate(seen_control, 0.951)
    result = subject.prefix_gate(
        audits, seen_control, below_seen_gain,
        unseen_control, boundary_candidate(unseen_control, 0.975),
        batch_identity=True, prefix_identity=True)
    assert result["pass"] is False
    assert result["seen"]["checks"]["role_gain"] is False
