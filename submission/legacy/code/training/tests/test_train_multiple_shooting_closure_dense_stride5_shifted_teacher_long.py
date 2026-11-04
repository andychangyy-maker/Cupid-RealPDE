from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_multiple_shooting_closure_dense_stride5_shifted_teacher_long as subject  # noqa: E402


def test_recipe_reuses_exact_dense_control_contract():
    assert subject.TRAIN_STRIDE == subject.dense.TRAIN_STRIDE == 5
    assert subject.HELD_STRIDE == subject.dense.HELD_STRIDE == 20
    assert subject.control.FIT_AOAS == (0, 5, 15, 20)
    assert subject.control.HELD_AOA == 10
    assert subject.control.UPDATES == 12_000
    assert subject.control.BATCH_SIZE == 8
    assert subject.control.LOSS_WEIGHTS == (1.0, 4.0, 1.0)
    assert subject.control.MSE_WEIGHT == 0.25
    assert subject.SHIFT_FRAMES == 5
    assert subject.OVERLAP_FRAMES == 15
    assert subject.DENSE_CONTROL_REFERENCE["sample_stream_sha256"] == (
        "118d296c3421ff75ec71ffd867aea35bf"
        "218622bf258477aadba9db820658cde")


def test_shifted_history_uses_same_dense_keys_and_is_valid_at_tail():
    trajectory = torch.arange(1000 * 3, dtype=torch.float32).reshape(1000, 1, 1, 3)
    windows = subject.dense.dense_windows([trajectory])
    dataset = subject.base.SimWindowDataset([trajectory], windows)
    support = subject.validate_shifted_teacher_support([trajectory], windows)
    assert support["maximum_main_start"] == 955
    assert support["all_shifted_histories_in_released_trajectories"]
    shifted = subject.shifted_history_batch(dataset, [0, len(windows) - 1])
    assert torch.equal(shifted[0], trajectory[5:25])
    assert torch.equal(shifted[1], trajectory[960:980])


def test_overlap_teacher_is_detached_and_uses_exact_frame_alignment():
    normalizer = subject.base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3),
        torch.device("cpu"))
    prediction = torch.randn(2, 20, 16, 16, 3, requires_grad=True)
    teacher = torch.randn(2, 20, 16, 16, 3, requires_grad=True)
    loss, details = subject.shifted_teacher_accuracy(prediction, teacher, normalizer)
    loss.backward()
    assert prediction.grad is not None
    assert teacher.grad is None
    assert set(details) == {"rel_l2", "tke_rel_l2", "mvpe_rel_l2", "accuracy_objective"}
    with pytest.raises(ValueError, match="20-frame"):
        subject.shifted_teacher_accuracy(prediction[:, :19], teacher, normalizer)


def test_dense_identity_and_full_evaluation_comparison_are_exact():
    manifest = {
        "windows": 15_360, "stride": 5,
        "trajectory_start_sha256": (
            "f84a435bc9d49b39eb350eb99f590fd9"
            "9c119843efd7f199d478d577f3744be1"),
        "first": [0, 0], "last": [79, 955],
        "fit_inventory_sha256": (
            "c8dbbc06b61701d0b259a32c0d6a8451"
            "d4b5a0ae7c12966af27e097ccba4aab8"),
    }
    normalizer = subject.base.ChannelNormalizer(
        torch.tensor((0.8572806119918823, -0.006138242781162262,
                      1.4708149433135986)),
        torch.tensor((0.8571790456771851, -0.006042769644409418,
                      1.4449578523635864)),
        torch.tensor((0.44620218873023987, 0.21658000349998474,
                      14.079401969909668)),
        torch.tensor((0.4462870657444, 0.2166111022233963,
                      13.80954360961914)), torch.device("cpu"))
    assert subject.dense_control_identity(manifest, normalizer, 15_360)["all_matched"]
    schedule = subject.control.make_schedule(15_360, 12_000, 8)
    assert hashlib.sha256(schedule.tobytes()).hexdigest() == (
        subject.DENSE_CONTROL_REFERENCE["sample_stream_sha256"])
    row = {"rel_l2_score": 1., "tke_rel_l2_score": 2., "mvpe_rel_l2_score": 3.,
           "point_contribution_46_10_9": 4.}
    compared = subject._evaluation_comparison(
        {"aggregate": row, "per_trajectory": {"a": row}, "per_prefix_horizon": {"1": row}},
        {"aggregate": row, "per_trajectory": {"a": row}, "per_prefix_horizon": {"1": row}})
    assert compared["aggregate"]["point_contribution_46_10_9"] == 0.
    assert compared["per_trajectory"]["a"]["rel_l2_score"] == 0.
    assert compared["per_prefix_horizon"]["1"]["tke_rel_l2_score"] == 0.


def test_formal_requires_acknowledgement_before_cuda_or_reference_access():
    args = subject.argparse.Namespace(
        execute=False, acknowledge="", sim_dir=None, output_dir=None,
        dense_control_result=None)
    with pytest.raises(PermissionError, match="formal run requires"):
        subject.formal(args, torch.device("cpu"))
