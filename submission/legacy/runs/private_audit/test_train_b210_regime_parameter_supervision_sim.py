from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_regime_parameter_supervision_sim.py"
SPEC = importlib.util.spec_from_file_location("regime_supervision", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_regime_supervisor_uses_every_scale_and_backpropagates():
    MODULE._REGIME_TARGETS = torch.linspace(-1, 1, 200).reshape(100, 2)
    supervisor = MODULE.RegimeParameterSupervisor((12, 8, 4))
    supervisor.set_trajectory_ids(torch.tensor([0, 17, 99]))
    features = [
        torch.randn(3, 12, 2, 4, requires_grad=True),
        torch.randn(3, 8, 4, 8, requires_grad=True),
        torch.randn(3, 4, 8, 16, requires_grad=True),
    ]
    loss, levels = supervisor(features, None)
    assert torch.isfinite(loss)
    assert len(levels) == 3
    loss.backward()
    assert all(feature.grad is not None for feature in features)
    assert all(bool(torch.isfinite(feature.grad).all()) for feature in features)


def test_regime_supervisor_rejects_invalid_ids():
    MODULE._REGIME_TARGETS = torch.zeros(100, 2)
    supervisor = MODULE.RegimeParameterSupervisor((12, 8, 4))
    try:
        supervisor.set_trajectory_ids(torch.tensor([100]))
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("invalid trajectory id was accepted")


def test_sim_update_sets_ids_from_positional_supervisor_slot():
    MODULE._REGIME_TARGETS = torch.zeros(100, 2)
    supervisor = MODULE.RegimeParameterSupervisor((12, 8, 4))
    seen = {}
    original = MODULE._ORIGINAL_SIM_UPDATE

    def fake_update(*args, **kwargs):
        seen["ids"] = supervisor._trajectory_ids.clone()
        return "ok"

    MODULE._ORIGINAL_SIM_UPDATE = fake_update
    try:
        batch = (None, None, torch.tensor([4, 9]), None)
        positional = [None, batch] + [None] * 12 + [supervisor]
        assert MODULE.regime_sim_update(*positional) == "ok"
    finally:
        MODULE._ORIGINAL_SIM_UPDATE = original
    assert torch.equal(seen["ids"], torch.tensor([4, 9]))
