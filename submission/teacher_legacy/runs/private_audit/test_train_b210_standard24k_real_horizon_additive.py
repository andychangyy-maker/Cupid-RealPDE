from __future__ import annotations

from argparse import Namespace
import sys
from pathlib import Path

import pytest
import torch


PRIVATE = Path(__file__).resolve().parent
if str(PRIVATE) not in sys.path:
    sys.path.insert(0, str(PRIVATE))

import train_b210_standard24k_real_horizon_additive as additive  # noqa: E402


def install_fake_accuracy(monkeypatch, warmup=1500):
    real = additive.standard24.self_describing.real

    def accuracy(_prediction, _target, _normalizer, weights):
        assert weights == (1.0, 0.0, 0.0)
        return torch.tensor(3.0, requires_grad=True), {}

    monkeypatch.setattr(
        real.ft12.npt1.ensemble_train.common,
        "accuracy_objective", accuracy)
    monkeypatch.setattr(
        real.ft12.npt1.joint_real, "real_loss_args",
        lambda: Namespace(mse_warmup_updates=warmup))


def test_additive_mixer_is_inactive_during_warmup(monkeypatch):
    install_fake_accuracy(monkeypatch)
    for fraction in (0.0, 0.2):
        mixed, _scale, metrics = additive.additive_rel_vs_structure_mixer(
            torch.tensor(5.0), torch.tensor(2.0), torch.zeros(1),
            torch.zeros(1), None, 1500, fraction)
        assert float(mixed.detach()) == pytest.approx(5.0)
        assert float(metrics["additive_coefficient"]) == pytest.approx(0.0)


def test_additive_arms_have_equal_value_and_different_aux_gradient(monkeypatch):
    install_fake_accuracy(monkeypatch)
    rows = []
    for fraction in (0.0, 0.2):
        direct = torch.tensor(5.0, requires_grad=True)
        consistency = torch.tensor(2.0, requires_grad=True)
        mixed, scale, metrics = additive.additive_rel_vs_structure_mixer(
            direct, consistency, torch.zeros(1), torch.zeros(1), None,
            1501, fraction)
        mixed.backward()
        rows.append((float(mixed.detach()), float(scale.detach()),
                     float(direct.grad), consistency.grad,
                     float(metrics["structure_arm"])))
    assert rows[0][0] == pytest.approx(rows[1][0]) == pytest.approx(5.6)
    assert rows[0][1] == pytest.approx(rows[1][1]) == pytest.approx(1.5)
    assert rows[0][2] == pytest.approx(rows[1][2]) == pytest.approx(1.0)
    assert rows[0][3] is None
    assert float(rows[1][3]) == pytest.approx(0.3)
    assert (rows[0][4], rows[1][4]) == (0.0, 1.0)


def test_additive_contract_is_frozen():
    assert additive.ADDITIVE_COEFFICIENT == pytest.approx(0.20)
    assert additive.BUDGET_MIXER_KIND == (
        "post_warmup_additive_rel_vs_structure_020")
