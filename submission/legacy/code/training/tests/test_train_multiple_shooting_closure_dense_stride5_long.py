from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_multiple_shooting_closure_dense_stride5_long as subject  # noqa: E402


def test_dense_windows_change_only_start_stride_and_keep_endpoint_convention():
    trajectories = [torch.empty(1000, 1, 1, 3) for _ in range(2)]
    windows = subject.dense_windows(trajectories)
    assert len(windows) == 384
    assert windows[0] == subject.base.SimWindow(0, 0)
    assert windows[1] == subject.base.SimWindow(0, 5)
    assert windows[191] == subject.base.SimWindow(0, 955)
    assert windows[192] == subject.base.SimWindow(1, 0)
    assert windows[-1] == subject.base.SimWindow(1, 955)


def test_recipe_reuses_sparse_scaled_control_contract():
    assert subject.control.FIT_AOAS == (0, 5, 15, 20)
    assert subject.control.HELD_AOA == 10
    assert subject.control.UPDATES == 12_000
    assert subject.control.BATCH_SIZE == 8
    assert subject.control.LR == 2.0e-4
    assert subject.control.WEIGHT_DECAY == 1.0e-5
    assert subject.control.EMA_DECAY == 0.999
    assert subject.control.LOSS_WEIGHTS == (1.0, 4.0, 1.0)
    assert subject.control.MSE_WEIGHT == 0.25
    assert subject.TRAIN_STRIDE == 5
    assert subject.HELD_STRIDE == 20


def test_formal_dense_manifest_and_stream_are_stable():
    trajectories = [torch.empty(1000, 1, 1, 3) for _ in range(80)]
    windows = subject.dense_windows(trajectories)
    manifest = subject.window_manifest(windows)
    assert manifest["windows"] == 15_360
    assert manifest["trajectory_start_sha256"] == (
        "f84a435bc9d49b39eb350eb99f590fd9"
        "9c119843efd7f199d478d577f3744be1")
    schedule = subject.control.make_schedule(
        len(windows), subject.control.UPDATES, subject.control.BATCH_SIZE)
    assert hashlib.sha256(schedule.tobytes()).hexdigest() == (
        "118d296c3421ff75ec71ffd867aea35bf"
        "218622bf258477aadba9db820658cde")
