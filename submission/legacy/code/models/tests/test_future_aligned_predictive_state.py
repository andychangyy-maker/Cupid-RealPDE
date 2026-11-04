from __future__ import annotations

import sys
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.future_aligned_predictive_state import FAPSConfig, FutureAlignedPredictiveState


def tiny_model() -> FutureAlignedPredictiveState:
    torch.manual_seed(9)
    return FutureAlignedPredictiveState(FAPSConfig(hidden_channels=16, latent_channels=8, groups=8))


def test_shared_decoder_reads_both_states_without_history_or_future_skip():
    model = tiny_model()
    history = torch.randn(2, 20, 8, 12, 3)
    target = torch.randn(2, 20, 8, 12, 3)
    forecast = model.forecast_components(history)
    analysis = model.analysis_components(target, forecast["base"])
    assert forecast["prediction"].shape == target.shape
    assert analysis["reconstructed_residual"].shape == target.shape
    assert len(list(model.forecaster.decoder.modules())) > 1
    # The only decoder call sites take (latent, fixed persistence).
    assert model.forecaster.decoder.latent_channels == 8


def test_future_encoder_is_stripped_and_forecast_is_bitwise_identical():
    model = tiny_model().eval()
    history = torch.randn(2, 20, 8, 12, 3)
    with torch.inference_mode():
        before = model(history)
        deployed = model.deployment().eval()
        after = deployed(history)
    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    assert not hasattr(deployed, "future_encoder")
    assert all("future_encoder" not in key for key in deployed.state_dict())


def test_alignment_stop_gradient_reaches_history_not_future_encoder():
    model = tiny_model()
    history = torch.randn(2, 20, 8, 12, 3)
    target = torch.randn(2, 20, 8, 12, 3)
    forecast = model.forecast_components(history)
    analysis = model.analysis_components(target, forecast["base"])
    loss = (forecast["z_history"] - analysis["z_future"].detach()).square().mean()
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.forecaster.history_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.future_encoder.parameters())


def test_fixed_latent_permutation_changes_decoder_output():
    model = tiny_model().eval()
    persistence = torch.randn(2, 8, 12, 3)
    latent = torch.randn(2, 8, 2, 3)
    with torch.inference_mode():
        ordinary = model.forecaster.decoder(latent, persistence)
        permuted = model.forecaster.decoder(latent[:, torch.tensor([1, 0, 2, 3, 4, 5, 6, 7])], persistence)
    assert float((ordinary - permuted).abs().mean()) > 1.0e-7
