from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_encoder_preserving_real_dense_aug as subject


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Linear(2, 3)
        self.encoder = torch.nn.Linear(3, 4)
        self.bottleneck = torch.nn.Linear(4, 4)
        self.decoder = torch.nn.Linear(4, 2)
        self.energy_controller = torch.nn.Linear(2, 1)


def test_encoder_partition_freezes_only_stem_and_encoder(monkeypatch):
    model = TinyModel()
    total = sum(value.numel() for value in model.parameters())
    frozen = sum(
        value.numel() for name, value in model.named_parameters()
        if name.startswith(subject.FROZEN_PREFIXES))
    monkeypatch.setattr(subject, "EXPECTED_FROZEN_PARAMETERS", frozen)
    monkeypatch.setattr(
        subject, "EXPECTED_TRAINABLE_PARAMETERS", total - frozen)
    trainable = subject.configure_encoder_preserving_trainable(model, total)
    assert sum(value.numel() for value in trainable) == total - frozen
    for name, value in model.named_parameters():
        assert value.requires_grad is (not name.startswith(
            subject.FROZEN_PREFIXES))
    assert all(
        value.requires_grad for value in model.energy_controller.parameters())


def test_train_arm_temporarily_installs_candidate_partition(monkeypatch):
    sentinel = object()
    calls = []

    def original(*args, **kwargs):
        calls.append(subject.ft12.configure_trainable)
        return {"sentinel": sentinel}, {"seal": True}

    default = subject.ft12.configure_trainable
    monkeypatch.setattr(subject, "ORIGINAL_TRAIN_CANDIDATE", original)
    model = TinyModel()
    control, _ = subject.train_arm(
        model, None, None, None, None, None, None, None,
        SimpleNamespace(arm="control"))
    candidate, _ = subject.train_arm(
        model, None, None, None, None, None, None, None,
        SimpleNamespace(arm="candidate"))
    assert calls == [default, subject.configure_encoder_preserving_trainable]
    assert subject.ft12.configure_trainable is default
    assert control["real_transfer_strategy"] == "full_finetune"
    assert candidate["real_transfer_strategy"] == "freeze_stem_encoder"
    assert candidate["frozen_parameter_state_unchanged"] is True


def test_identical_dense_augmentation_has_independent_copies():
    value = subject.identical_dense_augmentation()
    assert value["control"] == value["candidate"] == subject.paired.AUGMENTATION
    value["control"]["input_noise_std"] = 9.0
    assert value["candidate"]["input_noise_std"] == 0.01


def test_identical_dense_windows_match_exactly():
    trajectories = [torch.zeros(100, 2, 2, 2), torch.zeros(83, 2, 2, 2)]
    value = subject.identical_dense_windows(trajectories, [])
    control = [(row.trajectory, row.start) for row in value["control"]]
    candidate = [(row.trajectory, row.start) for row in value["candidate"]]
    assert control == candidate
    assert len(control) == 22


def test_partition_rejects_wrong_contract(monkeypatch):
    model = TinyModel()
    total = sum(value.numel() for value in model.parameters())
    monkeypatch.setattr(subject, "EXPECTED_FROZEN_PARAMETERS", 1)
    monkeypatch.setattr(subject, "EXPECTED_TRAINABLE_PARAMETERS", total - 1)
    with pytest.raises(RuntimeError, match="partition changed"):
        subject.configure_encoder_preserving_trainable(model, total)
