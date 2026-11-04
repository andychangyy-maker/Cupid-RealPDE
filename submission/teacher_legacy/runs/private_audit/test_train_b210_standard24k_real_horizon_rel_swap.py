from __future__ import annotations

from argparse import Namespace
import sys
from pathlib import Path

import pytest
import torch


PRIVATE = Path(__file__).resolve().parent
if str(PRIVATE) not in sys.path:
    sys.path.insert(0, str(PRIVATE))

import train_b210_standard24k_real_horizon_rel_swap as swap  # noqa: E402


def install_fake_accuracy(monkeypatch, warmup=1500):
    real = swap.standard24.self_describing.real

    def accuracy(_prediction, _target, _normalizer, weights):
        assert weights == (1.0, 0.0, 0.0)
        return torch.tensor(3.0, requires_grad=True), {}

    monkeypatch.setattr(
        real.ft12.npt1.ensemble_train.common,
        "accuracy_objective", accuracy)
    monkeypatch.setattr(
        real.ft12.npt1.joint_real, "real_loss_args",
        lambda: Namespace(mse_warmup_updates=warmup))


def test_rel_swap_is_inactive_for_control_and_warmup(monkeypatch):
    install_fake_accuracy(monkeypatch)
    for iteration, fraction in ((2000, 0.0), (1500, 0.2)):
        direct = torch.tensor(5.0)
        consistency = torch.tensor(2.0)
        mixed, scale, metrics = swap.rel_component_budget_mixer(
            direct, consistency, torch.zeros(1), torch.zeros(1), None,
            iteration, fraction)
        assert float(mixed.detach()) == pytest.approx(5.0)
        assert float(scale.detach()) == pytest.approx(1.5)
        assert float(metrics["rel_swap_coefficient"]) == pytest.approx(0.0)


def test_rel_swap_preserves_scalar_and_replaces_only_rel_gradient(monkeypatch):
    install_fake_accuracy(monkeypatch)
    direct = torch.tensor(5.0, requires_grad=True)
    consistency = torch.tensor(2.0, requires_grad=True)
    mixed, scale, metrics = swap.rel_component_budget_mixer(
        direct, consistency, torch.zeros(1), torch.zeros(1), None,
        1501, 0.2)
    assert mixed.detach() == pytest.approx(5.0)
    assert scale.detach() == pytest.approx(1.5)
    assert float(metrics["rel_swap_coefficient"]) == pytest.approx(0.2)
    mixed.backward()
    assert direct.grad == pytest.approx(1.0)
    assert consistency.grad == pytest.approx(0.3)


def test_rel_swap_contract_is_frozen():
    assert swap.REL_SWAP_COEFFICIENT == pytest.approx(0.20)
    assert swap.BUDGET_MIXER_KIND == "post_warmup_rel_component_swap_020"
