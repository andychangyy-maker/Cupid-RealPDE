from __future__ import annotations

import pytest
import torch
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
    HistoryConditionedSpatialTransport,
)


def module() -> HistoryConditionedSpatialTransport:
    return HistoryConditionedSpatialTransport(
        feature_channels=8, hidden_channels=8, out_steps=4,
        groups=4, max_displacement=1.5)


def test_zero_initialized_spatial_transport_is_exact_identity_with_gradient():
    layer = module()
    prediction = torch.randn(2, 4, 5, 7, 3, requires_grad=True)
    feature = torch.randn(2, 8, 5, 7)
    history = torch.randn(2, 6, 5, 7, 3)
    output = layer(prediction, feature, history)
    assert torch.equal(output, prediction)
    output.square().mean().backward()
    assert layer.output.weight.grad is not None
    assert torch.count_nonzero(layer.output.weight.grad)


def test_learned_spatial_transport_preserves_mean_and_pressure():
    layer = module()
    with torch.no_grad():
        layer.output.bias.fill_(0.2)
    prediction = torch.randn(2, 4, 5, 7, 3)
    output = layer(
        prediction, torch.randn(2, 8, 5, 7),
        torch.randn(2, 6, 5, 7, 3))
    assert torch.allclose(
        output[..., :2].float().mean((2, 3)),
        prediction[..., :2].float().mean((2, 3)), atol=2e-6, rtol=0.0)
    assert torch.equal(output[..., 2:], prediction[..., 2:])
    assert not torch.equal(output[..., :2], prediction[..., :2])


def test_integer_spatial_warp_moves_values_in_sampling_direction():
    uv = torch.zeros(1, 1, 3, 4, 2)
    uv[0, 0, :, :, 0] = torch.arange(4).view(1, 4)
    displacement = torch.zeros_like(uv)
    displacement[..., 0] = 1.0
    warped = HistoryConditionedSpatialTransport._warp(uv, displacement)
    assert torch.equal(warped[0, 0, :, :-1, 0], uv[0, 0, :, 1:, 0])
    assert torch.equal(warped[0, 0, :, -1, 0], uv[0, 0, :, -1, 0])


def test_spatial_transport_is_exclusive_with_other_transports():
    with pytest.raises(ValueError, match="separate arms"):
        EnsembleUNet2dConfig(
            base=8, depth=2, groups=4,
            spatial_transport_warp=True,
            temporal_band_mixer=True).validate()


def test_enabling_spatial_transport_preserves_legacy_initialization():
    common = dict(
        in_steps=6, out_steps=4, in_channels=3, out_channels=3,
        base=8, depth=2, groups=4, mean_fluctuation=True,
        ensemble_members=2, uq_hidden_channels=8,
        uncertainty_base_widths=(0.1, 0.1), temporal_band_count=2,
    )
    torch.manual_seed(19)
    legacy = EnsembleUNet2d(EnsembleUNet2dConfig(**common))
    torch.manual_seed(19)
    candidate = EnsembleUNet2d(EnsembleUNet2dConfig(
        **common, spatial_transport_warp=True,
        spatial_transport_channels=8))
    candidate_legacy = {
        key: value for key, value in candidate.state_dict().items()
        if not key.startswith("temporal_transport.")
    }
    assert legacy.state_dict().keys() == candidate_legacy.keys()
    assert all(
        torch.equal(value, candidate_legacy[key])
        for key, value in legacy.state_dict().items())
    history = torch.randn(1, 6, 8, 8, 3)
    legacy.eval()
    candidate.eval()
    with torch.no_grad():
        assert torch.equal(candidate(history), legacy(history))
