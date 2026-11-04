from __future__ import annotations

from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_layerwise_lr_real_dense_aug as subject


class TinyTransferModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Conv2d(2, 4, 3, padding=1)
        self.encoder = torch.nn.ModuleList((
            torch.nn.Conv2d(4, 8, 3, stride=2, padding=1),))
        self.decoder = torch.nn.Conv2d(8, 4, 3, padding=1)
        self.head = torch.nn.Conv2d(4, 2, 1)


def test_layerwise_partition_is_disjoint_complete_and_order_preserving():
    model = TinyTransferModel()
    supplied = tuple(model.parameters())
    slow, fast = subject.partition_optimizer_parameters(model, supplied)
    slow_ids = {id(value) for value in slow}
    fast_ids = {id(value) for value in fast}

    assert slow and fast
    assert not slow_ids.intersection(fast_ids)
    assert slow_ids.union(fast_ids) == {id(value) for value in supplied}
    assert tuple(value for value in supplied if id(value) in slow_ids) == slow


def test_layerwise_adamw_uses_fixed_ratio_and_all_parameters():
    model = TinyTransferModel()
    subject._CURRENT_MODEL = model
    subject._CURRENT_ARM = "candidate"
    try:
        optimizer = subject.layerwise_adamw(
            tuple(model.parameters()), lr=2.0e-4, weight_decay=1.0e-4)
    finally:
        subject._CURRENT_MODEL = None
        subject._CURRENT_ARM = None

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["layerwise_role"] == "stem_encoder"
    assert optimizer.param_groups[0]["lr"] == 2.0e-5
    assert optimizer.param_groups[1]["lr"] == 2.0e-4
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        tuple(model.parameters()))


def test_layerwise_factory_rejects_partial_encoder():
    model = TinyTransferModel()
    supplied = tuple(
        value for name, value in model.named_parameters()
        if name != "stem.weight")
    try:
        subject.partition_optimizer_parameters(model, supplied)
    except RuntimeError as error:
        assert "full encoder" in str(error)
    else:
        raise AssertionError("partial encoder was accepted")
