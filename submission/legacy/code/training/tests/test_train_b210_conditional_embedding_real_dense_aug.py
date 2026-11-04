from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_conditional_embedding_real_dense_aug as subject


class IdentityNormalizer:
    mean_target = torch.zeros(3)
    std_target = torch.ones(3)

    def postprocess_prediction(self, value):
        return value * self.std_target + self.mean_target


def test_conditional_embedding_auxiliary_has_finite_point_gradient():
    torch.manual_seed(29)
    history = torch.randn(8, 20, 8, 12, 3)
    target = torch.randn(8, 20, 8, 12, 3)
    prediction = (target + 0.2 * torch.randn_like(target)).requires_grad_()
    loss, metrics = subject.conditional_embedding_auxiliary(
        history, prediction, target, target, IdentityNormalizer())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics["ceod_normalized"] >= 0.0
    assert metrics["ceod_weighted_prediction_gradient_norm"] > 0.0
    assert metrics["ceod_uses_target_free_history_condition"] == 1.0


def test_candidate_injects_ceod_and_control_does_not(monkeypatch):
    calls = []

    def fake_train(*args, **kwargs):
        calls.append(kwargs)
        return {}, {"sealed": True}

    monkeypatch.setattr(subject, "ORIGINAL_TRAIN", fake_train)
    control_training, _ = subject.train_arm(
        access=SimpleNamespace(arm="control"))
    candidate_training, _ = subject.train_arm(
        access=SimpleNamespace(arm="candidate"))
    assert calls[0].get("region_auxiliary") is None
    assert calls[0].get("region_auxiliary_weight", 0.0) == 0.0
    assert calls[1]["region_auxiliary"] is subject.conditional_embedding_auxiliary
    assert calls[1]["region_auxiliary_weight"] == subject.CEOD_WEIGHT
    assert control_training["conditional_embedding"]["enabled"] is False
    assert candidate_training["conditional_embedding"]["enabled"] is True


def test_dense_data_and_augmentation_are_identical_between_arms():
    trajectories = [torch.zeros(80, 4, 4, 2), torch.zeros(85, 4, 4, 2)]
    windows = subject.identical_dense_windows(trajectories, [])
    assert [(row.trajectory, row.start) for row in windows["control"]] == [
        (row.trajectory, row.start) for row in windows["candidate"]]
    augmentation = subject.identical_dense_augmentation()
    assert augmentation["control"] == augmentation["candidate"]
    assert augmentation["control"] is not augmentation["candidate"]
