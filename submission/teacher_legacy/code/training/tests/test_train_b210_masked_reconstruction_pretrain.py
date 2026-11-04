from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_masked_reconstruction_pretrain as subject


def test_real_velocity_is_promoted_once_to_exact_zero_pressure_contract():
    u = np.arange(48, dtype=np.float32).reshape(4, 3, 4)
    v = -u
    raw = subject.real_velocity_channels(u, v)
    assert raw.shape == (4, 3, 4, 2)
    assert raw.dtype == np.float32
    assert raw.flags.c_contiguous
    dataset = subject.b210.rehearsal.ensemble_train.common.repro.WindowDataset(
        [torch.from_numpy(raw)], [subject.b210.rehearsal.ensemble_train.common.repro.Window(0, 0)])
    history, target = dataset[0]
    assert history.shape == (4, 3, 4, 3)
    assert target.shape == (0, 3, 4, 3)
    np.testing.assert_array_equal(history[..., 0].numpy(), u)
    np.testing.assert_array_equal(history[..., 1].numpy(), v)
    np.testing.assert_array_equal(history[..., 2].numpy(), np.zeros_like(u))


def test_spatial_mask_is_shared_across_time_and_channels_by_contract():
    generator = torch.Generator().manual_seed(3)
    mask = subject.spatial_mask(
        4, 32, 64, generator, torch.device("cpu"), ratio=0.70)
    assert mask.shape == (4, 32, 64)
    fraction = float(mask.float().mean())
    assert 0.55 < fraction < 0.85
    expanded = mask[:, None, :, :, None].expand(4, 20, 32, 64, 3)
    assert torch.equal(expanded[:, 0, :, :, 0], expanded[:, 19, :, :, 2])


def test_masked_loss_does_not_read_unmasked_targets():
    prediction = torch.zeros(2, 20, 4, 8, 3)
    target = torch.ones_like(prediction)
    mask = torch.zeros(2, 4, 8, dtype=torch.bool)
    mask[:, :2, :4] = True
    first = subject.masked_reconstruction_loss(prediction, target, mask)
    target[:, :, 2:, 4:] = 1000.0
    second = subject.masked_reconstruction_loss(prediction, target, mask)
    assert torch.equal(first, second)


def test_reconstruction_readout_shape():
    readout = subject.ReconstructionReadout([16, 8], 20, 3)
    features = [torch.randn(2, 16, 4, 8), torch.randn(2, 8, 8, 16)]
    value = readout(features, (32, 64))
    assert value.shape == (2, 20, 32, 64, 3)


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Linear(2, 2)
        self.encoder = torch.nn.Linear(2, 2)
        self.member_point_heads = torch.nn.Linear(2, 2)
        self.energy_controller = torch.nn.Linear(2, 2)


def test_only_shared_backbone_is_trainable():
    model = TinyBackbone()
    selected = subject.configure_shared_backbone(model)
    assert sum(parameter.numel() for parameter in selected) == 12
    assert model.stem.weight.requires_grad
    assert model.encoder.weight.requires_grad
    assert not model.member_point_heads.weight.requires_grad
    assert not model.energy_controller.weight.requires_grad
