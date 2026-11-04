from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_p1_direct_dense_exact_interval_head as student  # noqa: E402


class _Normalizer:
    mean_input = torch.zeros(1, 1, 1, 1, 3)
    std_input = torch.ones(1, 1, 1, 1, 3)
    mean_target = torch.tensor([1.0, -2.0, 0.0])
    std_target = torch.tensor([2.0, 4.0, 1.0])


def _formal_args(**overrides) -> argparse.Namespace:
    values = {
        "updates": 4000,
        "batch_size": 8,
        "eval_batch_size": 16,
        "reconstruction_batch_size": 16,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "time_budget_seconds": 1200.0,
        "seed": 0,
        "num_workers": 0,
        "device": "cuda:0",
        "held_fold": 4,
        "expected_point_sha256": student.FORMAL_POINT_SHA256,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _passing_representation() -> dict:
    return {
        "passed": True,
        "unrepresentable_union": 0,
        "unrepresentable_fraction": 0.0,
        "worst_case_sps_loss_bound": 0.0,
        "min_width_log_margin": 0.5,
        "min_center_fraction_margin": 0.5,
    }


def _passing_evaluation() -> dict:
    teacher_width = student.FOLD04_MAX_WIDTH / student.MAX_WIDTH_RATIO
    return {
        "finite_ordered_absolute_bounds": True,
        "truth_passes": 1,
        "held_rows": student.EXPECTED_HELD_TAIL[4],
        "exact_p0_ceiling_with_direct_point": {
            "sps_score": 53.30,
            "coverage": student.FOLD04_MIN_COVERAGE + 0.005,
            "mean_width": teacher_width,
        },
        "student": {
            "sps_score": student.FOLD04_MIN_SPS,
            "coverage": student.FOLD04_MIN_COVERAGE,
            "mean_width": student.FOLD04_MAX_WIDTH,
        },
    }


def _centered_endpoints(point: torch.Tensor) -> torch.Tensor:
    reference = point.new_tensor((0.02150, 0.01100)).view(
        *((1,) * (point.ndim - 1)), 2)
    return torch.stack((point - 0.5 * reference,
                        point + 0.5 * reference), dim=-1)


def test_contract_locks_direct_head_and_formal_constants() -> None:
    head = student.build_head(torch.device("cpu"), 0)
    assert student.ARCHITECTURE == "distilled_interval_head"
    assert head.architecture_config()["architecture"] == student.ARCHITECTURE
    assert sum(value.numel() for value in head.parameters()) == 45_568
    assert student.FORMAL_POINT_SHA256 == (
        "758b44b2ce124d9c1bf72657d1a39247481c4794a6f2837bac8ffc072836ce07")
    assert student.TRAINING_TIME_BUDGET_SECONDS == 1200.0
    assert student.MAX_UNREPRESENTABLE_FRACTION == 0.0


def test_validate_args_rejects_any_frozen_run_change() -> None:
    student.validate_args(_formal_args())
    with pytest.raises(ValueError, match="SHA758b"):
        student.validate_args(_formal_args(expected_point_sha256="0" * 64))
    with pytest.raises(ValueError, match="hyperparameters are frozen"):
        student.validate_args(_formal_args(updates=3999))
    with pytest.raises(ValueError, match="hyperparameters are frozen"):
        student.validate_args(_formal_args(time_budget_seconds=1201.0))
    with pytest.raises(ValueError, match="held-fold"):
        student.validate_args(_formal_args(held_fold=5))


def test_representation_requires_every_scalar_strictly_inside_limits() -> None:
    point = torch.zeros(1, 1, 1, 1, 2, dtype=torch.float64)
    endpoints = _centered_endpoints(point)
    inside = student.representability_from_physical(point, endpoints)
    assert inside["passed"]
    assert inside["unrepresentable_union"] == 0
    assert inside["min_width_log_margin"] > 0.0
    assert inside["min_center_fraction_margin"] > 0.0

    reference = torch.tensor((0.02150, 0.01100), dtype=torch.float64)
    boundary = endpoints.clone()
    width = reference[0]
    center = 1.5 * width
    boundary[..., 0, 0] = center - 0.5 * width
    boundary[..., 0, 1] = center + 0.5 * width
    outside = student.representability_from_physical(point, boundary)
    assert not outside["passed"]
    assert outside["center_beyond_limit"] == 1
    assert outside["unrepresentable_union"] == 1


def test_representation_rejects_nonpositive_width_without_exception() -> None:
    point = torch.zeros(1, 1, 1, 1, 2)
    endpoints = _centered_endpoints(point)
    endpoints[..., 1, 1] = endpoints[..., 1, 0]
    result = student.representability_from_physical(point, endpoints)
    assert not result["passed"]
    assert result["structurally_invalid"] == 1
    assert math.isfinite(result["max_abs_log_width_ratio"])


def test_streaming_preflight_records_global_tail_statistics(monkeypatch) -> None:
    rows = 3
    histories = torch.zeros(rows, 20, 1, 1, 3)
    point = torch.tensor([1.0, -2.0]).view(1, 1, 1, 1, 2).expand(
        rows, 20, 1, 1, 2)
    anchors = _centered_endpoints(point)

    class _Dataset:
        def __init__(self):
            self.anchors = anchors

        def __len__(self):
            return rows

        def __getitem__(self, index):
            return histories[index], self.anchors[index]

    class _Model:
        def eval(self):
            return self

        def _decode(self, _history):
            return []

    def _fused(_model, x, _features, _heads):
        return torch.zeros((*x.shape[:-1], 3), dtype=x.dtype)

    monkeypatch.setattr(student, "fused_mean_prediction", _fused)
    result = student.representation_preflight(
        _Dataset(), _Model(), None, _Normalizer(),
        torch.device("cpu"), batch_size=2)

    assert result["passed"]
    assert result["rows"] == rows
    assert result["total_scalar_endpoints"] == rows * 20 * 2
    assert result["unrepresentable_union"] == 0
    assert result["p99_9_abs_log_width_ratio_conservative"] >= 0.0
    assert result["p99_9_center_fraction_conservative"] == pytest.approx(0.0)


def test_physical_endpoint_objective_matches_only_absolute_teacher() -> None:
    prediction = torch.zeros(1, 1, 1, 1, 3)
    lower_extent = torch.tensor([[[[[0.5, 0.25]]]]])
    upper_extent = torch.tensor([[[[[1.0, 0.5]]]]])
    _point, lower, upper = student.absolute_endpoints_from_ftq(
        prediction, lower_extent, upper_extent, _Normalizer())
    teacher = torch.stack((lower, upper), dim=-1)
    exact, _ = student.exact_endpoint_objective(
        prediction, lower_extent, upper_extent, teacher, _Normalizer())
    shifted, _ = student.exact_endpoint_objective(
        prediction, lower_extent, upper_extent,
        teacher + 0.01, _Normalizer())
    assert exact.item() == 0.0
    assert shifted.item() > 0.0


def test_absolute_bounds_use_signed_physical_accumulator_extents() -> None:
    shape = (1, 20, 32, 64, 2)
    point = torch.tensor([10.0, -3.0]).view(
        1, 1, 1, 1, 2).expand(shape)
    lower = torch.tensor([11.0, -5.0]).view(
        1, 1, 1, 1, 2).expand(shape)
    upper = torch.tensor([13.0, -2.0]).view(
        1, 1, 1, 1, 2).expand(shape)
    truth = torch.tensor([12.0, -4.0]).view(
        1, 1, 1, 1, 2).expand(shape)
    lower_extent, upper_extent = student.physical_extents_from_absolute(
        point, lower, upper)
    assert lower_extent[0, 0, 0, 0, 0].item() == -1.0
    accumulator = student.quantile_teacher.new_accumulator()
    student.quantile_teacher.update_accumulator(
        accumulator, point, lower_extent, upper_extent, truth)
    metrics = student.quantile_teacher.finalize_accumulator(accumulator)
    assert metrics["coverage"] == 1.0
    assert metrics["mean_width"] == pytest.approx(2.5)


def test_fresh_head_is_seed_exact_and_deploys_as_fp16() -> None:
    first = student.build_head(torch.device("cpu"), 0)
    _ = torch.randn(100)
    second = student.build_head(torch.device("cpu"), 0)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name
    deployed = student.prepare_deploy_head(first)
    assert not deployed.training
    assert {value.dtype for value in deployed.state_dict().values()} == {
        torch.float16}


def test_fold04_gate_accepts_boundaries_and_rejects_any_failed_gate() -> None:
    evaluation = _passing_evaluation()
    gate = student.student_gate(
        _formal_args(), _passing_representation(), evaluation,
        True, True, 4000, 120.0, 240.0, "float16")
    assert gate["passed"]

    evaluation["student"]["sps_score"] -= 1e-6
    assert not student.student_gate(
        _formal_args(), _passing_representation(), evaluation,
        True, True, 4000, 120.0, 240.0, "float16")["passed"]
    assert not student.student_gate(
        _formal_args(), {**_passing_representation(), "passed": False},
        _passing_evaluation(), True, True, 4000,
        120.0, 240.0, "float16")["passed"]
    assert not student.student_gate(
        _formal_args(), _passing_representation(), _passing_evaluation(),
        True, True, 4000, 120.0, 1200.001, "float16")["passed"]
    assert not student.student_gate(
        _formal_args(), _passing_representation(), _passing_evaluation(),
        False, True, 4000, 120.0, 240.0, "float16")["passed"]
