from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_p1_direct_ftq_quantile_student as student  # noqa: E402
import train_p1_direct_native_quantile_teacher as teacher  # noqa: E402


def test_p0_full_7q_formula_is_monotone_and_applies_one_c_shift() -> None:
    shape = (2, 3, 1, 1, 2)
    levels = torch.arange(7, dtype=torch.float32).view(1, 1, 1, 1, 1, 7)
    qa = levels.expand(*shape, 7).clone()
    qb = qa + 4.0
    pa = torch.zeros(shape)
    pb = torch.full(shape, 2.0)
    pc = torch.full(shape, 10.0)

    actual = teacher.compose_p0_absolute_quantiles(qa, qb, pa, pb, pc)

    base_point = pa.lerp(pb, teacher.WEIGHT_B)
    correction = teacher.WEIGHT_C * (
        pc.mean(dim=1, keepdim=True)
        - base_point.mean(dim=1, keepdim=True))
    expected = qa.lerp(qb, teacher.WEIGHT_B) + correction.unsqueeze(-1)
    torch.testing.assert_close(actual, expected)
    assert torch.all(actual[..., 1:] > actual[..., :-1])


def test_fixed_fold_expansion_yields_2032_544_and_572_153() -> None:
    point_counts = (508, 508, 508, 508, 544)
    tail_counts = (143, 143, 143, 143, 153)
    manifest = []
    assignments = []
    for fold, (point_count, tail_count) in enumerate(zip(
            point_counts, tail_counts)):
        item = {
            "file": f"condition_{fold}.h5",
            "re": fold,
            "aoa": fold * 5,
            "train_starts": list(range(point_count)),
            "val_starts": list(range(tail_count)),
        }
        manifest.append(item)
        assignments.append({
            "file": item["file"], "re": item["re"],
            "aoa": item["aoa"], "fold": fold,
        })
    fold_manifest = {"assignments": assignments}

    point_folds = teacher.partition_fold_labels(
        manifest, fold_manifest, "train_starts")
    tail_folds = teacher.partition_fold_labels(
        manifest, fold_manifest, "val_starts")
    point_train, point_held = teacher.fold_partition_indices(point_folds)
    tail_train, tail_held = teacher.fold_partition_indices(tail_folds)

    assert (len(point_train), len(point_held)) == (2032, 544)
    assert (len(tail_train), len(tail_held)) == (572, 153)
    assert all(point_folds[index] != 4 for index in point_train)
    assert all(tail_folds[index] != 4 for index in tail_train)
    assert set(point_train).isdisjoint(point_held)
    assert set(tail_train).isdisjoint(tail_held)


def test_two_training_phases_do_not_relabel_tail_truth_as_pretraining() -> None:
    assert teacher.training_stage(1, 1000, 4000) == (
        "point2032_p0_7q_pretrain")
    assert teacher.training_stage(1000, 1000, 4000) == (
        "point2032_p0_7q_pretrain")
    assert teacher.training_stage(1001, 1000, 4000) == (
        "tail572_truth_refine")
    assert teacher.training_stage(4000, 1000, 4000) == (
        "tail572_truth_refine")
    with pytest.raises(ValueError):
        teacher.training_stage(0, 1000, 4000)


def test_absolute_outcome_quantiles_use_frozen_point_center() -> None:
    prediction = torch.tensor([[[[[2.0, -1.0, 99.0]]]]])
    quantiles = torch.arange(14, dtype=torch.float32).reshape(
        1, 1, 1, 1, 2, 7)
    normalizer = SimpleNamespace(
        mean_target=torch.tensor([10.0, 20.0, 0.0]),
        std_target=torch.tensor([2.0, 4.0, 1.0]),
    )

    actual = teacher.absolute_outcome_quantiles(
        prediction, quantiles, normalizer)

    point = torch.tensor([[[[[14.0, 16.0]]]]]).unsqueeze(-1)
    scale = torch.tensor([2.0, 4.0]).view(1, 1, 1, 1, 2, 1)
    torch.testing.assert_close(actual, point + quantiles * scale)


def test_ftq_distillation_is_zero_only_for_matching_full_7q() -> None:
    target_std = torch.tensor([0.1, 0.02, 1.0])
    teacher_quantiles = torch.linspace(-1.0, 1.0, 14).reshape(
        1, 1, 1, 1, 2, 7)
    matching = student.quantile_distillation_objective(
        teacher_quantiles.clone(), teacher_quantiles, target_std)
    shifted = student.quantile_distillation_objective(
        teacher_quantiles + 0.1, teacher_quantiles, target_std)

    assert float(matching) == 0.0
    assert float(shifted) > 0.0


def test_teacher_endpoint_recenter_preserves_absolute_bounds_and_width() -> None:
    native_prediction = torch.tensor([[[[[2.0, -1.0, 7.0]]]]])
    direct_prediction = torch.tensor([[[[[2.2, -1.3, 8.0]]]]])
    lower_native = torch.tensor([[[[[0.5, 0.4]]]]])
    upper_native = torch.tensor([[[[[0.7, 0.8]]]]])

    lower_direct, upper_direct = student.recenter_interval_extents(
        lower_native, upper_native, native_prediction, direct_prediction)

    torch.testing.assert_close(
        direct_prediction[..., :2] - lower_direct,
        native_prediction[..., :2] - lower_native)
    torch.testing.assert_close(
        direct_prediction[..., :2] + upper_direct,
        native_prediction[..., :2] + upper_native)
    torch.testing.assert_close(
        lower_direct + upper_direct, lower_native + upper_native)


def test_teacher_endpoint_recenter_allows_signed_extent_when_bounds_ordered() -> None:
    native_prediction = torch.tensor([[[[[0.0, 0.0, 0.0]]]]])
    direct_prediction = torch.tensor([[[[[2.0, 0.0, 0.0]]]]])
    lower_native = torch.tensor([[[[[0.5, 0.5]]]]])
    upper_native = torch.tensor([[[[[0.7, 0.7]]]]])

    lower_direct, upper_direct = student.recenter_interval_extents(
        lower_native, upper_native,
        native_prediction, direct_prediction)
    assert torch.any(upper_direct < 0.0)
    torch.testing.assert_close(
        direct_prediction[..., :2] - lower_direct,
        native_prediction[..., :2] - lower_native)
    torch.testing.assert_close(
        direct_prediction[..., :2] + upper_direct,
        native_prediction[..., :2] + upper_native)
    assert torch.all(lower_direct + upper_direct >= 0.0)


def test_formal_student_head_is_deployed_fp16() -> None:
    head = student.FastTemporalQuantileHead(
        student.FastTemporalQuantileHeadConfig())
    deployed = student.prepare_deploy_head(head)
    assert not deployed.training
    assert {value.dtype for value in deployed.state_dict().values()} == {
        torch.float16}


def test_fresh_ftq_constructor_resets_exact_seed() -> None:
    first, first_metadata = student.build_head(None, torch.device("cpu"), 0)
    _ = torch.randn(1000)
    second, second_metadata = student.build_head(None, torch.device("cpu"), 0)
    assert first_metadata["constructor_seed_reset"] == 0
    assert second_metadata["constructor_seed_reset"] == 0
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name


def test_single_student_cache_stays_under_three_gib() -> None:
    tensor_bytes = (
        student.EXPECTED_DISTILL_ROWS
        * int(np.prod(student.LABEL_SHAPE, dtype=np.int64)) * 2)
    assert tensor_bytes < 3 * 1024**3
    assert tensor_bytes == 2_986_475_520


def test_latest_teacher_and_student_gates_are_exact() -> None:
    reference = teacher.FOLD04_REFERENCE
    teacher_args = SimpleNamespace(
        updates=4000, anchor_only_updates=1000, batch_size=8,
        eval_batch_size=16, reconstruction_batch_size=16,
        lr=1e-4, weight_decay=1e-5, grad_clip=1.0,
        truth_pinball_weight=0.25, soft_sps_weight=1.0,
        coverage_weight=0.25, coverage_target=reference["coverage"],
        soft_tau_fraction=0.10, seed=0, num_workers=0,
        device="cuda:0")
    teacher_metrics = {
        "sps_score": teacher.TEACHER_MIN_SPS,
        "coverage": reference["coverage"] - teacher.MAX_COVERAGE_DROP,
        "mean_width": reference["mean_width"] * teacher.MAX_WIDTH_RATIO,
    }
    assert teacher.teacher_gate(
        teacher_args, teacher_metrics, True, 4000)["passed"]
    teacher_metrics["sps_score"] -= 1e-6
    assert not teacher.teacher_gate(
        teacher_args, teacher_metrics, True, 4000)["passed"]

    evaluation = {
        "teacher_recentered_to_direct": {"sps_score": 53.30},
        "student": {
            "sps_score": 53.20,
            "coverage": reference["coverage"] - student.MAX_COVERAGE_DROP,
            "mean_width": reference["mean_width"] * student.MAX_WIDTH_RATIO,
        },
    }
    student_args = SimpleNamespace(
        updates=4000, batch_size=8, eval_batch_size=16,
        materialize_batch_size=16, lr=1e-4, weight_decay=1e-5,
        grad_clip=1.0, seed=0, num_workers=0, device="cuda:0",
        start_head=None)
    assert student.student_gate(
        student_args, evaluation, True, True, 4000, "float16")["passed"]
    assert not student.student_gate(
        student_args, evaluation, True, False, 4000, "float16")["passed"]
    evaluation["student"]["sps_score"] = 53.199
    assert not student.student_gate(
        student_args, evaluation, True, True, 4000, "float16")["passed"]


def test_incomplete_or_fp32_formal_run_cannot_pass() -> None:
    reference = teacher.FOLD04_REFERENCE
    teacher_args = SimpleNamespace(
        updates=4000, anchor_only_updates=1000, batch_size=8,
        eval_batch_size=16, reconstruction_batch_size=16,
        lr=1e-4, weight_decay=1e-5, grad_clip=1.0,
        truth_pinball_weight=0.25, soft_sps_weight=1.0,
        coverage_weight=0.25, coverage_target=reference["coverage"],
        soft_tau_fraction=0.10, seed=0, num_workers=0,
        device="cuda:0")
    metrics = {
        "sps_score": 99.0,
        "coverage": reference["coverage"],
        "mean_width": reference["mean_width"],
    }
    assert not teacher.teacher_gate(
        teacher_args, metrics, True, 3999)["passed"]

    student_args = SimpleNamespace(
        updates=4000, batch_size=8, eval_batch_size=16,
        materialize_batch_size=16, lr=1e-4, weight_decay=1e-5,
        grad_clip=1.0, seed=0, num_workers=0, device="cuda:0",
        start_head=None)
    evaluation = {
        "teacher_recentered_to_direct": {"sps_score": 53.30},
        "student": {
            "sps_score": 53.30,
            "coverage": reference["coverage"],
            "mean_width": reference["mean_width"],
        },
    }
    assert not student.student_gate(
        student_args, evaluation, True, True, 4000, "float32")["passed"]
