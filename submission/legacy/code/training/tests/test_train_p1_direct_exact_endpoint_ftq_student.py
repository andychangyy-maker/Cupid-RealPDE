from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_p1_direct_exact_endpoint_ftq_student as student  # noqa: E402


class _Normalizer:
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
        "time_budget_seconds": 1800.0,
        "seed": 0,
        "num_workers": 0,
        "device": "cuda:0",
        "held_fold": 4,
        "expected_point_sha256": student.FORMAL_POINT_SHA256,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _fold_labels(counts: dict[int, int]) -> list[int]:
    return [fold for fold in range(5) for _ in range(counts[fold])]


def test_contract_locks_exact_s2c_graph_and_formal_constants() -> None:
    assert student.ARCHITECTURE == (
        "fast_temporal_quantile_head_fused_mean_member_free")
    assert student.FastTemporalQuantileHeadConfig() == (
        student.FastTemporalQuantileHeadConfig())
    assert student.FORMAL_POINT_SHA256 == (
        "758b44b2ce124d9c1bf72657d1a39247481c4794a6f2837bac8ffc072836ce07")
    assert (student.WEIGHT_B, student.WEIGHT_C) == (0.625, 0.25)
    assert student.POOLED_PROMOTION_SPS == 53.462071


def test_validate_args_rejects_any_frozen_run_change() -> None:
    student.validate_args(_formal_args())
    with pytest.raises(ValueError, match="SHA758b"):
        student.validate_args(_formal_args(
            expected_point_sha256="0" * 64))
    with pytest.raises(ValueError, match="hyperparameters are frozen"):
        student.validate_args(_formal_args(updates=3999))
    with pytest.raises(ValueError, match="held-fold"):
        student.validate_args(_formal_args(held_fold=5))


def test_all_five_fold_counts_are_dynamic_and_exact() -> None:
    point_folds = _fold_labels(student.EXPECTED_HELD_POINT)
    tail_folds = _fold_labels(student.EXPECTED_HELD_TAIL)
    for held_fold in range(5):
        partitions = student.formal_partitions(
            point_folds, tail_folds, held_fold)
        assert len(partitions["point_held"]) == (
            student.EXPECTED_HELD_POINT[held_fold])
        assert len(partitions["tail_held"]) == (
            student.EXPECTED_HELD_TAIL[held_fold])
        assert len(partitions["point_train"]) == (
            student.EXPECTED_POINT_WINDOWS
            - student.EXPECTED_HELD_POINT[held_fold])
        assert len(partitions["tail_train_unused"]) == (
            student.EXPECTED_TAIL_WINDOWS
            - student.EXPECTED_HELD_TAIL[held_fold])
        assert {point_folds[index]
                for index in partitions["point_train"]} == (
                    set(range(5)) - {held_fold})
        assert {tail_folds[index]
                for index in partitions["tail_held"]} == {held_fold}


def test_anchor_training_dataset_has_no_truth_interface() -> None:
    class Histories:
        def __getitem__(self, row):
            return torch.full((1,), float(row)), torch.empty(0)

    anchors = torch.zeros(2, 1, 1, 1, 2, 2)
    dataset = student.AnchorTrainingDataset(Histories(), [3, 7], anchors)

    history, endpoint = dataset[1]

    torch.testing.assert_close(history, torch.tensor([7.0]))
    assert endpoint.shape[-2:] == (2, 2)


def test_exact_endpoint_loss_is_zero_only_for_matching_selected_endpoints() -> None:
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


def test_nonzero_point_absolute_bounds_convert_to_equivalent_signed_extents(
        ) -> None:
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
    residual = truth - point

    assert lower_extent[0, 0, 0, 0, 0].item() == -1.0
    torch.testing.assert_close(
        (residual >= -lower_extent) & (residual <= upper_extent),
        (truth >= lower) & (truth <= upper))
    torch.testing.assert_close(
        lower_extent + upper_extent, upper - lower)
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
    assert first.architecture_config() == second.architecture_config()
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name
    deployed = student.prepare_deploy_head(first)
    assert not deployed.training
    assert {value.dtype for value in deployed.state_dict().values()} == {
        torch.float16}


def test_fold04_gate_accepts_exact_boundaries_and_rejects_below_sps() -> None:
    teacher_width = student.FOLD04_MAX_WIDTH / student.MAX_WIDTH_RATIO
    evaluation = {
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
    gate = student.student_gate(
        _formal_args(), evaluation, True, True, 4000, 1800.0, "float16")
    assert gate["passed"]
    evaluation["student"]["sps_score"] -= 1e-6
    assert not student.student_gate(
        _formal_args(), evaluation, True, True,
        4000, 1800.0, "float16")["passed"]


def test_non_fold04_gate_uses_relative_ceiling_not_53_20() -> None:
    evaluation = {
        "exact_p0_ceiling_with_direct_point": {
            "sps_score": 52.929,
            "coverage": 0.90,
            "mean_width": 0.016,
        },
        "student": {
            "sps_score": 52.83,
            "coverage": 0.8951,
            "mean_width": 0.01679,
        },
    }
    gate = student.student_gate(
        _formal_args(held_fold=1), evaluation,
        True, True, 4000, 1799.0, "float16")
    assert gate["passed"]
    assert "fold04_sps_ge_53_20" not in gate["checks"]


def test_gate_requires_point_invariance_complete_run_and_fp16() -> None:
    evaluation = {
        "exact_p0_ceiling_with_direct_point": {
            "sps_score": 53.25,
            "coverage": 0.90,
            "mean_width": 0.016,
        },
        "student": {
            "sps_score": 53.20,
            "coverage": 0.90,
            "mean_width": 0.016,
        },
    }
    assert not student.student_gate(
        _formal_args(), evaluation, False, True,
        4000, 1000.0, "float16")["passed"]
    assert not student.student_gate(
        _formal_args(), evaluation, True, False,
        4000, 1000.0, "float16")["passed"]
    assert not student.student_gate(
        _formal_args(), evaluation, True, True,
        3999, 1000.0, "float16")["passed"]
    assert not student.student_gate(
        _formal_args(), evaluation, True, True,
        4000, 1000.0, "float32")["passed"]


def test_fold04_anchor_bank_stays_below_one_gibibyte() -> None:
    rows = student.EXPECTED_POINT_WINDOWS - student.EXPECTED_HELD_POINT[4]
    stored_bytes = rows * 20 * 32 * 64 * 2 * 2 * 2
    assert stored_bytes == 665_845_760
    assert stored_bytes < 1024 ** 3
