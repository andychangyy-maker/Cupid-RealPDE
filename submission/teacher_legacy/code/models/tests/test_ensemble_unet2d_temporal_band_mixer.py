from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
    HistoryConditionedTemporalBandMixer,
)


def mixer() -> HistoryConditionedTemporalBandMixer:
    return HistoryConditionedTemporalBandMixer(
        feature_channels=8, hidden_channels=8, out_steps=20, bands=4,
        groups=4, log_gain_limit=0.5, phase_limit=0.5)


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(71)
    prediction = torch.randn(2, 20, 8, 16, 3)
    feature = torch.randn(2, 8, 8, 16)
    history = torch.randn(2, 20, 8, 16, 3)
    return prediction, feature, history


def test_zero_projection_is_bitwise_identity_and_has_gradient():
    model = mixer()
    prediction, feature, history = inputs()
    output = model(prediction, feature, history)
    assert torch.equal(output, prediction)

    target = torch.randn_like(output)
    (output - target).square().mean().backward()
    assert model.output.weight.grad is not None
    assert float(model.output.weight.grad.abs().sum()) > 0.0


def test_trained_mixer_preserves_temporal_mean_and_pressure():
    model = mixer()
    prediction, feature, history = inputs()
    with torch.no_grad():
        model.output.weight.normal_(std=0.02)
        model.output.bias.normal_(std=0.02)
    output = model(prediction, feature, history)
    assert not torch.equal(output[..., :2], prediction[..., :2])
    torch.testing.assert_close(
        output[..., :2].mean(dim=1),
        prediction[..., :2].mean(dim=1), rtol=0.0, atol=3.0e-7)
    assert torch.equal(output[..., 2:], prediction[..., 2:])


def test_frequency_bank_excludes_dc_and_covers_each_non_dc_frequency_once():
    assignment = mixer().frequency_assignment
    assert torch.equal(assignment[0], torch.zeros_like(assignment[0]))
    assert torch.equal(
        assignment[1:].sum(dim=1),
        torch.ones_like(assignment[1:, 0]))
    assert torch.all(assignment.sum(dim=0) > 0)


def test_temporal_band_and_isoenergetic_arms_are_mutually_exclusive():
    config = EnsembleUNet2dConfig(
        isoenergetic_temporal_transport=True, temporal_band_mixer=True)
    with pytest.raises(ValueError, match="separate arms"):
        config.validate()


def test_enabling_band_mixer_preserves_every_legacy_state_and_prediction():
    common = dict(
        base=8, depth=3, groups=4, difference_steps=2,
        ensemble_members=3, uq_hidden_channels=8)
    torch.manual_seed(83)
    baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common)).eval()
    torch.manual_seed(83)
    candidate = EnsembleUNet2d(EnsembleUNet2dConfig(
        **common, temporal_band_mixer=True,
        temporal_band_channels=8, temporal_band_count=4)).eval()
    candidate_state = candidate.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(
            candidate_state[name], value, rtol=0.0, atol=0.0)
    for model in (baseline, candidate):
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
    history = torch.randn(2, 20, 8, 16, 3)
    assert torch.equal(candidate(history), baseline(history))
