from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluate_c208_teacher_role_headroom.py")
SPEC = importlib.util.spec_from_file_location("c208_trh1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_compose_role_members_is_zero_sum_and_preserves_non_uv() -> None:
    prediction = torch.tensor(
        [[[[[0.25, -0.50, 7.0]]], [[[0.50, -0.25, 8.0]]]]],
        dtype=torch.float32)
    member_offsets = torch.tensor(
        [-0.3, -0.1, 0.1, 0.3], dtype=torch.float32
    ).view(1, 4, 1, 1, 1, 1)
    members = prediction.unsqueeze(1).repeat(1, 4, 1, 1, 1, 1)
    members[..., :2] += member_offsets
    original_pressure = members[..., 2:].clone()
    teacher_a = torch.tensor(
        [[[[[1.0, 2.0]]], [[[3.0, 6.0]]]]], dtype=torch.float32)
    teacher_b = torch.tensor(
        [[[[[4.0, 1.0]]], [[[2.0, 5.0]]]]], dtype=torch.float32)
    std = torch.tensor([2.0, 4.0], dtype=torch.float32)
    prediction_before = prediction.clone()
    members_before = members.clone()

    candidate, audit = subject.compose_role_members(
        prediction, members, teacher_a, teacher_b, std)

    assert torch.equal(prediction, prediction_before)
    assert torch.equal(members, members_before)
    assert torch.allclose(
        candidate[..., :2].mean(dim=1), prediction[..., :2], atol=1e-6)
    assert torch.equal(candidate[..., 2:], original_pressure)
    assert audit["non_uv_members_reused_by_identity"]
    assert audit["member_mean_change_max_abs_physical"] <= 1e-6
    assert audit["teacher_role_time_mean_max_abs_physical"] <= 1e-6
    expected_role = 0.5 * (
        (teacher_a - teacher_a.mean(dim=1, keepdim=True))
        - (teacher_b - teacher_b.mean(dim=1, keepdim=True))) / std.view(
            1, 1, 1, 1, 2)
    baseline_centered = members[..., :2] - members[..., :2].mean(
        dim=1, keepdim=True)
    assert torch.allclose(
        candidate[:, :2, ..., :2]
        - prediction[..., :2].unsqueeze(1)
        - baseline_centered[:, :2],
        expected_role.unsqueeze(1).expand(-1, 2, -1, -1, -1, -1),
        atol=1e-6)
    assert torch.allclose(
        candidate[:, 2:, ..., :2]
        - prediction[..., :2].unsqueeze(1)
        - baseline_centered[:, 2:],
        -expected_role.unsqueeze(1).expand(-1, 2, -1, -1, -1, -1),
        atol=1e-6)


def test_pad_zero_pressure_preserves_uv_bitwise() -> None:
    uv = torch.randn((2, 3, 4, 5, 2), dtype=torch.float32)
    full = subject.pad_zero_pressure(uv)
    assert torch.equal(full[..., :2], uv)
    assert int(torch.count_nonzero(full[..., 2])) == 0


def test_fp16_member_mean_audit_is_post_cast_physical_error() -> None:
    torch.manual_seed(123)
    prediction = (torch.randn((2, 10, 2, 2, 3)) * 2.0).half()
    members = prediction.unsqueeze(1).repeat(1, 4, 1, 1, 1, 1)
    members += (torch.randn_like(members) * 0.5).half()
    teacher_a = (torch.randn((2, 10, 2, 2, 2)) * 0.2).half()
    teacher_b = (torch.randn((2, 10, 2, 2, 2)) * 0.2).half()
    std = torch.tensor([0.0968, 0.0160])

    candidate, audit = subject.compose_role_members(
        prediction, members, teacher_a, teacher_b, std)
    measured = float((
        (candidate[..., :2].float().mean(dim=1)
         - members[..., :2].float().mean(dim=1)).abs()
        * std.view(1, 1, 1, 1, 2)).max())

    assert measured > 0.0
    assert audit["member_mean_change_max_abs_physical"] == pytest.approx(
        measured)
    assert measured <= subject.MEMBER_MEAN_TOLERANCE_PHYSICAL


def test_zero_teacher_role_is_member_bitwise_identity() -> None:
    torch.manual_seed(9)
    prediction = torch.randn((1, 3, 2, 2, 3)).half()
    members = torch.randn((1, 4, 3, 2, 2, 3)).half()
    teacher = torch.randn((1, 3, 2, 2, 2))
    candidate, _audit = subject.compose_role_members(
        prediction, members, teacher, teacher.clone(),
        torch.tensor([0.0968, 0.0160]))
    assert torch.equal(candidate, members)


@pytest.mark.parametrize("bad", ["shape", "finite", "std"])
def test_compose_role_members_rejects_invalid_inputs(bad: str) -> None:
    prediction = torch.zeros((1, 2, 1, 1, 3))
    members = torch.zeros((1, 4, 2, 1, 1, 3))
    teacher_a = torch.zeros((1, 2, 1, 1, 2))
    teacher_b = torch.zeros_like(teacher_a)
    std = torch.ones(2)
    if bad == "shape":
        teacher_a = torch.zeros((1, 1, 1, 1, 2))
    elif bad == "finite":
        members[0, 0, 0, 0, 0, 0] = torch.nan
    else:
        std[0] = 0.0
    with pytest.raises((ValueError, FloatingPointError)):
        subject.compose_role_members(
            prediction, members, teacher_a, teacher_b, std)


def _passing_report() -> dict:
    return {
        "hard_guards": {"contract": True},
        "pooled": {"sps_delta": subject.SPS_DELTA_MIN},
        "aoa_rows": [
            {"aoa": aoa, "sps_delta": delta}
            for aoa, delta in zip(
                subject.AOAS,
                (0.1, 0.1, 0.1, 0.1,
                 subject.WORST_AOA_SPS_DELTA_MIN))],
        "rank_delta": {
            "element_log_spearman": subject.ELEMENT_SPEARMAN_DELTA_MIN,
            "sample_spearman": subject.SAMPLE_SPEARMAN_DELTA_MIN,
            "top_error_recall_at_top_spread": (
                subject.TOP_ERROR_RECALL_DELTA_MIN),
        },
    }


def test_gate_requires_exact_five_aoas_and_every_boundary() -> None:
    report = _passing_report()
    gate = subject.gate_decision(report)
    assert gate["passed"]
    assert gate["decision"] == "GO_TRH1_TO_ZERO_SUM_ROLE_BRANCH"

    report = _passing_report()
    report["aoa_rows"] = report["aoa_rows"][:4]
    assert not subject.gate_decision(report)["passed"]

    report = _passing_report()
    report["aoa_rows"][-1]["aoa"] = 15
    assert not subject.gate_decision(report)["passed"]

    report = _passing_report()
    report["hard_guards"]["contract"] = False
    gate = subject.gate_decision(report)
    assert not gate["passed"]
    assert gate["decision"] == "STOP_TRH1_NO_ROLE_HEADROOM"

    report = _passing_report()
    report["aoa_rows"][0]["sps_delta"] = 0.0
    gate = subject.gate_decision(report)
    assert not gate["checks"]["positive_aoas"]
    assert not gate["passed"]


@pytest.mark.parametrize(
    ("mutate", "key"),
    [
        (lambda report: report["pooled"].update(
            sps_delta=subject.SPS_DELTA_MIN - 1e-9), "pooled_sps_delta"),
        (lambda report: report["aoa_rows"][0].update(
            sps_delta=subject.WORST_AOA_SPS_DELTA_MIN - 1e-9), "worst_aoa"),
        (lambda report: report["rank_delta"].update(
            element_log_spearman=subject.ELEMENT_SPEARMAN_DELTA_MIN - 1e-9),
         "element_spearman_delta"),
        (lambda report: report["rank_delta"].update(
            sample_spearman=subject.SAMPLE_SPEARMAN_DELTA_MIN - 1e-9),
         "sample_spearman_delta"),
        (lambda report: report["rank_delta"].update(
            top_error_recall_at_top_spread=(
                subject.TOP_ERROR_RECALL_DELTA_MIN - 1e-9)),
         "top_error_recall_delta"),
        (lambda report: report["aoa_rows"][-1].update(sps_delta=np.nan),
         "aoa_rows_finite"),
    ])
def test_gate_fails_closed_below_each_boundary(mutate, key: str) -> None:
    report = _passing_report()
    mutate(report)
    gate = subject.gate_decision(report)
    assert not gate["checks"][key]
    assert not gate["passed"]


def test_rank_metrics_detects_aligned_spread() -> None:
    error = np.arange(1.0, 101.0)
    aligned = error.copy()
    reversed_spread = aligned[::-1].copy()
    sample_error = np.arange(1.0, 21.0)
    sample_aligned = sample_error.copy()

    good = subject.rank_metrics(
        error, aligned, sample_error, sample_aligned)
    bad = subject.rank_metrics(
        error, reversed_spread, sample_error, sample_aligned[::-1].copy())

    assert good["element_log_spearman"] == pytest.approx(1.0)
    assert good["sample_spearman"] == pytest.approx(1.0)
    assert good["top_error_recall_at_top_spread"] == pytest.approx(1.0)
    assert bad["element_log_spearman"] == pytest.approx(-1.0)
    assert bad["sample_spearman"] == pytest.approx(-1.0)
    assert bad["top_error_recall_at_top_spread"] == pytest.approx(0.0)
    assert good["top_fraction_actual"] == pytest.approx(0.1)


def test_safe_spearman_degenerate_inputs_return_finite_zero() -> None:
    assert subject.safe_spearman(np.ones(10), np.arange(10.0)) == 0.0
    assert subject.safe_spearman(np.ones(1), np.ones(1)) == 0.0
    values = np.arange(10.0)
    values[-1] = np.nan
    assert subject.safe_spearman(values, np.arange(10.0)) == 0.0


def test_safe_spearman_uses_average_ranks_for_ties() -> None:
    left = np.array([1.0, 1.0, 2.0, 3.0])
    right = np.array([1.0, 2.0, 2.0, 3.0])
    assert subject.safe_spearman(left, right) == pytest.approx(5.0 / 6.0)
