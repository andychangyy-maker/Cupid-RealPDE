from __future__ import annotations

from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
import train_b210_nlt_real_dense_aug as subject


def tiny_model() -> EnsembleUNet2d:
    return EnsembleUNet2d(EnsembleUNet2dConfig(
        base=8,
        depth=3,
        groups=4,
        difference_steps=2,
        ensemble_members=2,
        uq_hidden_channels=8,
        physical_energy_controller=True,
        physical_energy_hidden_channels=8,
    )).float()


def test_backbone_nlt_is_identity_and_excludes_heads_controller():
    torch.manual_seed(17)
    model = tiny_model().eval()
    value = torch.randn(1, 20, 16, 24, 3)
    before = model.forward_distribution(value)
    rows = subject.install_backbone_nlt(model)
    after = model.forward_distribution(value)

    assert all(torch.equal(first, second)
               for first, second in zip(before, after))
    assert rows
    assert all(subject.select_backbone_convolution(row.name, row.module)
               for row in rows)
    assert not any(row.name.startswith("energy_controller") for row in rows)
    assert not any("head" in row.name for row in rows)


def test_arm_trainability_is_the_only_training_difference():
    control = tiny_model()
    subject.install_backbone_nlt(control)
    total = sum(value.numel() for value in control.parameters())
    control_trainable = subject.configure_arm_trainable(
        control, "control", total)
    control_rows = subject.installed_transforms(control)
    assert all(row.source_weight.requires_grad for row in control_rows)
    assert all(not row.transform.factor.requires_grad for row in control_rows)

    candidate = tiny_model()
    subject.install_backbone_nlt(candidate)
    candidate_trainable = subject.configure_arm_trainable(
        candidate, "candidate", total)
    candidate_rows = subject.installed_transforms(candidate)
    assert all(not row.source_weight.requires_grad for row in candidate_rows)
    assert all(row.transform.factor.requires_grad for row in candidate_rows)
    assert sum(value.numel() for value in control_trainable) != sum(
        value.numel() for value in candidate_trainable)
    assert all(
        value.requires_grad
        for value in candidate.energy_controller.parameters())


def test_ema_style_nlt_state_merges_to_strict_legacy_model():
    torch.manual_seed(19)
    model = tiny_model().eval()
    rows = subject.install_backbone_nlt(model)
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            row.transform.factor.add_(0.0005 * index)
            row.transform.offset.add_(0.0001 * index)
    value = torch.randn(1, 20, 16, 24, 3)
    adapted = model.forward_distribution(value)
    merged_state = subject.merged_standard_state(model, model.state_dict())
    legacy = EnsembleUNet2d(model.config).float().eval()
    legacy.load_state_dict(merged_state, strict=True)
    merged = legacy.forward_distribution(value)

    assert all(torch.equal(first, second)
               for first, second in zip(adapted, merged))
    assert not any("parametrizations.weight" in key for key in merged_state)
