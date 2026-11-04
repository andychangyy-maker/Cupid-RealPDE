from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_standard24k_real_rollout_exposure.py"
SPEC = importlib.util.spec_from_file_location("standard24_rollout", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_rollout_windows_never_cross_prefix_boundary():
    trajectories = [torch.zeros(100, 4, 8, 2), torch.zeros(61, 4, 8, 2)]
    windows = MODULE.build_rollout_windows(trajectories)
    assert [(row.trajectory, row.start) for row in windows] == [
        (0, 0), (0, 20), (0, 40), (1, 0)]
    assert all(row.start + 60 <= len(trajectories[row.trajectory])
               for row in windows)


def test_rollout_dataset_returns_three_real_uvp_segments():
    values = torch.arange(80 * 2, dtype=torch.float32).reshape(80, 1, 1, 2)
    dataset = MODULE.RolloutDataset(
        [values], [MODULE.RolloutWindow(0, 5)])
    history, target1, target2, trajectory, start = dataset[0]
    assert history.shape == target1.shape == target2.shape == (20, 1, 1, 3)
    assert torch.equal(history[..., :2], values[5:25])
    assert torch.equal(target1[..., :2], values[25:45])
    assert torch.equal(target2[..., :2], values[45:65])
    assert torch.count_nonzero(history[..., 2]) == 0
    assert (trajectory, start) == (0, 5)


def test_rollout_schedule_is_deterministic_and_full_batch():
    first = MODULE.rollout_batch_schedule(11, 17)
    second = MODULE.rollout_batch_schedule(11, 17)
    assert first == second
    assert len(first) == 17
    assert all(len(batch) == MODULE.ROLLOUT_BATCH_SIZE for batch in first)
    assert all(0 <= index < 11 for batch in first for index in batch)


def test_generated_history_keeps_zero_pressure():
    class Normalizer:
        mean_input = torch.tensor([1.0, 2.0, 0.0])
        std_input = torch.tensor([2.0, 4.0, 1.0])

    value = torch.zeros(2, 20, 4, 8, 3)
    value[..., 0] = 3.0
    value[..., 1] = 6.0
    normalized = MODULE.normalize_generated_history(value, Normalizer())
    assert torch.all(normalized[..., 0] == 1.0)
    assert torch.all(normalized[..., 1] == 1.0)
    assert torch.count_nonzero(normalized[..., 2]) == 0
    value[..., 2] = 1.0
    with pytest.raises(ValueError, match="pressure"):
        MODULE.normalize_generated_history(value, Normalizer())


def test_formal_rollout_contract_is_fixed_and_deploy_neutral():
    assert MODULE.ROLLOUT_STRIDE == 20
    assert MODULE.ROLLOUT_BATCH_SIZE == 4
    assert MODULE.GENERATED_FRACTION == pytest.approx(0.5)
    assert MODULE.ROLLOUT_OUTER_WEIGHT == pytest.approx(0.1)
    source = PATH.read_text(encoding="utf-8")
    assert "with torch.random.fork_rng" in source
    assert '"inference_graph_changed": False' in source
    assert source.count("weighted_loss.backward()") == 1
