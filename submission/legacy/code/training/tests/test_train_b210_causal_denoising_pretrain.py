from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_causal_denoising_pretrain as subject


def test_energy_scaled_noise_is_bounded_and_real_pressure_stays_zero():
    history = torch.ones(2, 20, 4, 8, 3)
    history[..., 2] = 0.0
    generator = torch.Generator().manual_seed(7)
    corrupted, sigma = subject.corrupt_history(
        history, generator, zero_pressure=True)
    assert sigma.shape == (2, 3)
    assert torch.all(sigma >= 0.0)
    assert torch.all(sigma <= subject.NOISE_SIGMA_MAX)
    assert torch.count_nonzero(sigma[:, 2]) == 0
    assert torch.count_nonzero(corrupted[..., 2]) == 0


def test_noise_sigma_obeys_energy_and_cap():
    low = torch.ones(1, 20, 4, 8, 3)
    high = 1000.0 * low
    assert torch.all(subject.history_noise_sigma(high) == subject.NOISE_SIGMA_MAX)
    assert torch.all(subject.history_noise_sigma(low) < subject.NOISE_SIGMA_MAX)


def test_balanced_relative_l2_ignores_masked_real_pressure():
    target = torch.ones(2, 20, 4, 8, 3)
    prediction = target.clone()
    prediction[1, ..., 2] = 1000.0
    mask = torch.ones(2, 3, dtype=torch.bool)
    mask[1, 2] = False
    assert subject.balanced_relative_l2(prediction, target, mask) < 1.0e-5


def test_balanced_relative_l2_weights_sample_channels_equally():
    target = torch.ones(2, 2, 1, 1, 2)
    prediction = target.clone()
    prediction[0, ..., 0] = 2.0
    mask = torch.ones(2, 2, dtype=torch.bool)
    # One of four sample/channel cells has relative L2 exactly one.
    value = subject.balanced_relative_l2(prediction, target, mask)
    assert abs(float(value) - 0.25) < 1.0e-5


def test_multihorizon_tke_score_is_zero_at_truth_and_phase_sensitive():
    target = torch.zeros(2, 20, 2, 3, 2)
    target[:, :, :, :, 0] = torch.linspace(-1.0, 1.0, 20)[None, :, None, None]
    target[:, :, :, :, 1] = torch.linspace(1.0, -0.5, 20)[None, :, None, None]
    exact, exact_by_horizon = subject.multi_horizon_physical_tke_score_loss(
        target, target)
    assert abs(float(exact)) < 1.0e-7
    assert tuple(exact_by_horizon) == subject.TKE_HORIZONS
    collapsed = target.mean(dim=1, keepdim=True).expand_as(target)
    wrong, wrong_by_horizon = subject.multi_horizon_physical_tke_score_loss(
        collapsed, target)
    assert float(wrong) > 0.1
    assert all(float(value) > 0.0 for value in wrong_by_horizon.values())


def test_xcb_shared_selection_excludes_deployment_heads():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Linear(2, 2)
            self.encoder = nn.Linear(2, 2)
            self.bottleneck = nn.Linear(2, 2)
            self.up = nn.Linear(2, 2)
            self.decoder = nn.Linear(2, 2)
            self.cross_conditioner = nn.Linear(2, 2)
            self.point_heads = nn.Linear(2, 2)
            self.member_feature_adapters = nn.Linear(2, 2)

    model = Tiny()
    names = subject.xcb_shared_parameter_names(model)
    assert any(name.startswith("cross_conditioner.") for name in names)
    assert not any(name.startswith("point_heads.") for name in names)
    assert not any(
        name.startswith("member_feature_adapters.") for name in names)
    trainable = subject.configure_named_parameters(model, names)
    assert sum(parameter.numel() for parameter in trainable) == sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if name in names)
    assert all(
        parameter.requires_grad == (name in names)
        for name, parameter in model.named_parameters())
