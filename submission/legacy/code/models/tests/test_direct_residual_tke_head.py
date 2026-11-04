from __future__ import annotations

import sys
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for value in (CODE_ROOT, TRAINING_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from models.direct_residual_tke_head import (  # noqa: E402
    DirectResidualTKEHead,
    FrozenB210DirectTKE,
)
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


def _base() -> EnsembleUNet2d:
    model = EnsembleUNet2d(EnsembleUNet2dConfig(
        base=8, depth=3, groups=4, difference_steps=2,
        ensemble_members=3, uq_hidden_channels=8,
        mean_fluctuation=True, shared_member_uv_mean=True,
        physical_energy_controller=True,
        physical_energy_member_sync=True,
    ))
    model.set_normalizer_statistics(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3))
    return model


def test_b210_head_contract_and_zero_identity():
    head = DirectResidualTKEHead()
    assert sum(value.numel() for value in head.parameters()) == 7_201
    assert torch.count_nonzero(head.body[-1].weight) == 0
    base = _base().eval()
    wrapper = FrozenB210DirectTKE(
        base, DirectResidualTKEHead(8, 8, 4)).eval()
    x = torch.randn(2, 20, 8, 16, 3)
    with torch.no_grad():
        expected = base.forward_uq_training(x)
        actual = wrapper.forward_uq_training(x)
    for index in range(5):
        assert torch.equal(actual[index], expected[index])


def test_head_preserves_temporal_mean_and_pressure():
    wrapper = FrozenB210DirectTKE(
        _base(), DirectResidualTKEHead(8, 8, 4))
    with torch.no_grad():
        wrapper.head.body[-1].weight.fill_(0.1)
        wrapper.head.body[-1].bias.fill_(0.05)
    x = torch.randn(2, 20, 8, 16, 3)
    with torch.no_grad():
        baseline = wrapper.base.forward_uq_training(x)
        candidate = wrapper.forward_uq_training(x)
    torch.testing.assert_close(
        candidate[0][..., :2].mean(dim=1),
        baseline[0][..., :2].mean(dim=1), atol=1e-6, rtol=1e-6)
    assert torch.equal(candidate[0][..., 2:], baseline[0][..., 2:])


def test_differentiable_uq_fusion_reaches_members():
    model = _base()
    x = torch.randn(1, 20, 8, 16, 3)
    features = model._decode(x)
    members = torch.randn(
        1, model.config.ensemble_members, model.config.out_steps,
        8, 16, model.config.out_channels, requires_grad=True)
    fused = model._fused_uq_features(
        features, members, x, detach_members=False)
    fused.square().mean().backward()
    assert members.grad is not None
    assert float(members.grad.abs().sum()) > 0.0
