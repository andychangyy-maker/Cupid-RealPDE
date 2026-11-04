from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from train_p1_direct_mean_slices import (  # noqa: E402
    MeanSliceGuard,
    decide_gate,
    mean_slice_loss,
)


class TinyPointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            mean_fluctuation=True, out_channels=3)
        self.register_buffer("scalar_audit", torch.tensor(1.0))
        self.member_point_heads = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(4, 63, 1)])
            for _ in range(4)])


def _metrics(point: float, rel: float = 10.0, tke: float = 20.0,
             mvpe: float = 30.0):
    return {
        "rel_l2_score": rel,
        "tke_rel_l2_score": tke,
        "mvpe_rel_l2_score": mvpe,
        "point_score": point,
        "by_aoa": {
            str(value): {
                "point_score": point,
                "rel_l2_score": rel,
                "tke_rel_l2_score": tke,
                "mvpe_rel_l2_score": mvpe,
            }
            for value in range(5)},
    }


def test_guard_restores_weight_decay_and_updates_only_first_two_rows() -> None:
    torch.manual_seed(0)
    model = TinyPointModel()
    guard = MeanSliceGuard(model)
    immutable_before = guard.immutable_state_sha256(model)
    mean_before = guard.capture_mean_rows()
    optimizer = torch.optim.AdamW(
        guard.parameters, lr=1e-2, weight_decay=0.1)

    loss = sum(parameter.sum() for parameter in guard.parameters)
    loss.backward()
    optimizer.step()
    guard.restore_frozen_rows()

    assert guard.frozen_rows_bitwise_equal()
    assert guard.frozen_rows_max_abs_drift() == 0.0
    assert guard.immutable_state_sha256(model) == immutable_before
    assert any(not torch.equal(
        parameter[:2], mean_before[name])
        for name, parameter in guard.parameter_names.items())


def test_mean_slice_loss_ignores_teacher_fluctuation_with_same_mean() -> None:
    torch.manual_seed(1)
    student = torch.randn(2, 3, 20, 32, 3)
    teacher = torch.randn(2, 3, 20, 32, 2)
    truth = torch.randn_like(student)
    zero_mean_change = torch.randn_like(teacher)
    zero_mean_change -= zero_mean_change.mean(dim=1, keepdim=True)

    first, _ = mean_slice_loss(student, teacher, truth)
    second, _ = mean_slice_loss(student, teacher + zero_mean_change, truth)

    assert torch.allclose(first, second, rtol=0.0, atol=1e-5)


def test_gate_uses_actual_baseline_and_requires_every_condition() -> None:
    initial = _metrics(91.40)
    final = _metrics(91.45, rel=10.01, tke=20.0000005, mvpe=30.01)
    passing = decide_gate(
        initial, final, oracle_point_score=91.47,
        fluctuation_drift=1e-8,
        immutable_state_equal=True, frozen_rows_equal=True)
    assert passing["passed"]
    assert passing["oracle_gap_recovery"] == pytest.approx(5.0 / 7.0)

    assert not decide_gate(
        initial, final, oracle_point_score=91.47,
        fluctuation_drift=1.1e-7,
        immutable_state_equal=True, frozen_rows_equal=True)["passed"]
    assert not decide_gate(
        initial, final, oracle_point_score=91.47,
        fluctuation_drift=1e-8,
        immutable_state_equal=False, frozen_rows_equal=True)["passed"]
