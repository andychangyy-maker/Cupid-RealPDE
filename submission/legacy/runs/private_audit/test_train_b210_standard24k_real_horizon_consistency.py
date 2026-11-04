from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PRIVATE = Path(__file__).resolve().parent
if str(PRIVATE) not in sys.path:
    sys.path.insert(0, str(PRIVATE))

import train_b210_standard24k_real_horizon_consistency as horizon  # noqa: E402


def test_cross_horizon_dataset_physical_alignment():
    values = torch.arange(50 * 2, dtype=torch.float32).reshape(50, 1, 1, 2)
    dataset = horizon.CrossHorizonDataset(
        [values], [horizon.CrossHorizonWindow(0, 3)])
    history_long, target_long, history_short, trajectory, start = dataset[0]
    assert torch.equal(history_long[..., :2], values[3:23])
    assert torch.equal(target_long[..., :2], values[23:43])
    assert torch.equal(history_short[..., :2], values[8:28])
    assert torch.equal(
        target_long[horizon.SHIFT_STEPS:
                    horizon.SHIFT_STEPS + horizon.CONSISTENCY_STEPS, ..., :2],
        values[28:37])
    assert (trajectory, start) == (0, 3)
    assert torch.count_nonzero(history_short[..., 2]) == 0


def test_budget_matched_loss_keeps_value_and_reallocates_gradient():
    direct = torch.tensor(2.0, requires_grad=True)
    consistency = torch.tensor(8.0, requires_grad=True)
    mixed, scale = horizon.budget_matched_loss(direct, consistency, 0.2)
    assert mixed.detach() == pytest.approx(2.0)
    assert scale.detach() == pytest.approx(0.25)
    mixed.backward()
    assert direct.grad == pytest.approx(0.8)
    assert consistency.grad == pytest.approx(0.05)


def test_extra_schedule_is_deterministic_and_full_batch():
    first = horizon.extra_batch_schedule(17, 31)
    second = horizon.extra_batch_schedule(17, 31)
    assert first == second
    assert len(first) == 31
    assert all(len(batch) == horizon.EXTRA_BATCH_SIZE for batch in first)


def test_frozen_cross_horizon_contract():
    assert horizon.SHIFT_STEPS == 5
    assert horizon.CONSISTENCY_STEPS == 9
    assert horizon.CONSISTENCY_BUDGET_FRACTION == pytest.approx(0.20)
    assert horizon.EXTRA_OUTER_WEIGHT == pytest.approx(0.10)
    source = Path(horizon.__file__).read_text(encoding="utf-8")
    assert "teacher_overlap.detach()" in source
    assert '"inference_graph_changed": False' in source
