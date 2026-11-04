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

import train_p1_direct_unbounded_exact_interval_head as student  # noqa: E402


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
        "structurally_invalid": 0,
        "inverse_nonfinite": 0,
        "fp16_roundtrip_nonfinite": 0,
        "fp16_roundtrip_nonpositive_width": 0,
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


def test_contract_locks_new_geometry_and_permanent_stop() -> None:
    head = student.build_head(torch.device("cpu"), 0)
    assert student.ARCHITECTURE == "direct_unbounded_absolute_interval_head"
    assert sum(value.numel() for value in head.parameters()) == 45_568
    assert student.completion_status(False, 4) == (
        "STOP_DIRECT_UNBOUNDED_EXACT_INTERVAL_FOLD04_PERMANENT")
    assert student.completion_status(True, 4) == (
        "PASS_DIRECT_UNBOUNDED_EXACT_INTERVAL_FOLD04")
    assert not hasattr(head.config, "max_center_fraction")
    assert not hasattr(head.config, "max_log_width_ratio")


def test_validate_args_rejects_any_frozen_change() -> None:
    student.validate_args(_formal_args())
    with pytest.raises(ValueError, match="SHA758b"):
        student.validate_args(
            _formal_args(expected_point_sha256="0" * 64))
    with pytest.raises(ValueError, match="hyperparameters are frozen"):
        student.validate_args(_formal_args(updates=3999))


def test_inverse_is_unique_for_displaced_wide_absolute_interval() -> None:
    point = torch.tensor([3.0, -1.0], dtype=torch.float64).view(
        1, 1, 1, 1, 2)
    reference = torch.tensor([0.0215, 0.011], dtype=torch.float64).view(
        1, 1, 1, 1, 2)
    expected_log = torch.tensor([2.3, -2.2], dtype=torch.float64).view(
        1, 1, 1, 1, 2)
    expected_center = torch.tensor([5.0, -6.0], dtype=torch.float64).view(
        1, 1, 1, 1, 2)
    width = reference * torch.exp(expected_log)
    midpoint = point + reference * expected_center
    endpoints = torch.stack(
        (midpoint - 0.5 * width, midpoint + 0.5 * width), dim=-1)
    raw_log, raw_center, valid = student.inverse_raw_from_physical(
        point, endpoints)
    assert valid.all()
    torch.testing.assert_close(raw_log, expected_log)
    torch.testing.assert_close(raw_center, expected_center)


def test_inverse_marks_nonpositive_width_structurally_invalid() -> None:
    point = torch.zeros(1, 1, 1, 1, 2)
    endpoints = torch.zeros(1, 1, 1, 1, 2, 2)
    _raw_log, _raw_center, valid = student.inverse_raw_from_physical(
        point, endpoints)
    assert not valid.any()


def test_train_only_preflight_accepts_values_beyond_bounded_head(
        monkeypatch) -> None:
    rows = 2
    histories = torch.zeros(rows, 20, 1, 1, 3)
    point = torch.tensor([1.0, -2.0]).view(1, 1, 1, 1, 2).expand(
        rows, 20, 1, 1, 2)
    reference = torch.tensor([0.0215, 0.011]).view(1, 1, 1, 1, 2)
    raw_log = torch.zeros_like(point)
    raw_center = torch.zeros_like(point)
    raw_log[1, ..., 0] = 2.2
    raw_center[1, ..., 0] = 4.0
    width = reference * torch.exp(raw_log)
    midpoint = point + reference * raw_center
    anchors = torch.stack(
        (midpoint - 0.5 * width, midpoint + 0.5 * width), dim=-1)

    class _Dataset:
        def __init__(self):
            self.anchors = anchors

        def __len__(self):
            return rows

        def __getitem__(self, index):
            return histories[index], self.anchors[index]

    class _Model:
        target_std = _Normalizer.std_target

        def eval(self):
            return self

        def _decode(self, _history):
            return []

    def _fused(_model, x, _features, _heads):
        return torch.zeros((*x.shape[:-1], 3), dtype=x.dtype)

    monkeypatch.setattr(student, "fused_mean_prediction", _fused)
    head = student.build_head(torch.device("cpu"), 0)
    result = student.representation_preflight(
        _Dataset(), _Model(), None, head, _Normalizer(),
        torch.device("cpu"), batch_size=1)
    assert result["passed"]
    assert result["rows"] == rows
    assert result["total_scalar_intervals"] == rows * 20 * 2
    assert result["max_abs_inverse_log_width"] > 2.0
    assert result["max_abs_inverse_center_reference_units"] > 1.5
    assert result["targets_read"] is False
    assert math.isfinite(result["fp16_roundtrip_max_abs_endpoint_error"])


def test_deploy_is_fp16_and_gate_keeps_fold04_thresholds() -> None:
    deployed = student.prepare_deploy_head(
        student.build_head(torch.device("cpu"), 0))
    assert not deployed.training
    assert {value.dtype for value in deployed.state_dict().values()} == {
        torch.float16}
    gate = student.student_gate(
        _formal_args(), _passing_representation(), _passing_evaluation(),
        True, True, 4000, 120.0, 240.0, "float16")
    assert gate["passed"]
    failed = student.student_gate(
        _formal_args(), {
            **_passing_representation(),
            "fp16_roundtrip_nonpositive_width": 1,
            "passed": False,
        }, _passing_evaluation(), True, True, 4000,
        120.0, 240.0, "float16")
    assert not failed["passed"]
