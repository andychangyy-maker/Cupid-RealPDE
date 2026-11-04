from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import b210_controller_functional_rehearsal as subject


class IdentityNormalizer:
    def preprocess(self, history, target):
        return history, target


class ToyController(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, features):
        return features[-1] * self.scale


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(1, 1, 1, bias=False)
        nn.init.ones_(self.backbone.weight)
        self.energy_controller = ToyController()
        self.config = SimpleNamespace(physical_energy_log_limit=1.0)

    def _decode(self, history):
        value = history[:, 0, ..., :1].permute(0, 3, 1, 2)
        return [self.backbone(value)]

    def _physical_energy_delta(self, features, _history):
        return self.energy_controller(features)


def batch():
    history = torch.ones(2, 2, 2, 2, 1)
    target = torch.zeros_like(history)
    trajectory = torch.tensor([0, 1])
    indices = torch.tensor([0, 1])
    return history, target, trajectory, indices


def test_bounded_delta_detaches_backbone_but_trains_controller():
    model = ToyModel()
    value = subject.bounded_energy_delta(
        model, batch()[0], detach_backbone=True)
    value.sum().backward()
    assert model.backbone.weight.grad is None
    assert model.energy_controller.scale.grad is not None


def test_delayed_calibration_changes_only_controller_gradient(monkeypatch):
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common,
        "set_model_stats", lambda *_args: None)
    sim = {
        "loader": [batch()],
        "normalizer": IdentityNormalizer(),
        "cache": torch.zeros(2, 1, 2, 2, dtype=torch.float16),
        "cache_report": {"sha256": "a" * 64},
        "manifest_sha256": "b" * 64,
        "normalizer_sha256": "c" * 64,
        "windows": 2,
        "trajectories": 2,
    }
    model = ToyModel()
    callback = subject.ControllerFunctionalBackward(
        sim, IdentityNormalizer(), torch.device("cpu"),
        calibration_update=2, target_gradient_ratio=0.15)
    # The formal path checks update-0 identity. Give this toy a deliberately
    # different target only after that contract has been exercised elsewhere.
    callback.initial_identity_mae = 0.0
    model.backbone.weight.grad = torch.ones_like(model.backbone.weight)
    model.energy_controller.scale.grad = torch.ones_like(
        model.energy_controller.scale)
    with pytest.raises(RuntimeError, match="identity"):
        callback(1, model)
    callback.rows.clear()
    callback.initial_identity_mae = None
    monkeypatch.setattr(subject, "INITIAL_IDENTITY_MAE_MAX", 2.0)
    inactive = callback(1, model)
    assert inactive["controller_functional_active"] == 0.0
    assert all(torch.isfinite(torch.as_tensor(value))
               for value in inactive.values())
    backbone_before = model.backbone.weight.grad.clone()
    controller_before = model.energy_controller.scale.grad.clone()
    values = callback(2, model)
    assert values["controller_functional_active"] == 1.0
    assert values["controller_functional_calibrated_ratio"] == pytest.approx(
        0.15)
    assert torch.equal(model.backbone.weight.grad, backbone_before)
    assert not torch.equal(
        model.energy_controller.scale.grad, controller_before)
    assert callback.contract()["inference_graph_changed"] is False
    assert model.training is True
