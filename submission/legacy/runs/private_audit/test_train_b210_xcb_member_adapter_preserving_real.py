from __future__ import annotations

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (REPO_ROOT, PRIVATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_xcb_member_adapter_preserving_real as subject


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.member_feature_adapters = torch.nn.ModuleList(
            (torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)))
        self.cross_conditioner = torch.nn.Linear(4, 4)
        self.energy_controller = torch.nn.Linear(4, 2)


def test_candidate_partition_freezes_only_member_adapters():
    model = TinyModel()
    subject._FREEZE_ADAPTERS = True
    try:
        trainable = subject.configure_trainable(
            model, sum(value.numel() for value in model.parameters()))
    finally:
        subject._FREEZE_ADAPTERS = False
    trainable_ids = {id(value) for value in trainable}
    adapters = tuple(model.member_feature_adapters.parameters())

    assert adapters
    assert all(not value.requires_grad for value in adapters)
    assert not trainable_ids.intersection(id(value) for value in adapters)
    assert all(value.requires_grad for value in model.backbone.parameters())
    assert all(value.requires_grad for value in model.cross_conditioner.parameters())
    assert all(value.requires_grad for value in model.energy_controller.parameters())


def test_selected_state_contains_only_adapter_namespace():
    model = TinyModel()
    state = subject.selected_state(model)

    assert state
    assert all(name.startswith(subject.ADAPTER_PREFIX) for name in state)


def test_paired_inputs_are_identical():
    augmentation = subject.identical_augmentation()

    assert augmentation["control"] == augmentation["candidate"]
    assert augmentation["control"] is not augmentation["candidate"]
