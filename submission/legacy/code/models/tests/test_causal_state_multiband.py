from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.causal_state_multiband import (  # noqa: E402
    CausalStateMultibandConfig,
    CausalStateMultibandForecaster,
)


def config() -> CausalStateMultibandConfig:
    return CausalStateMultibandConfig(state_channels=32, decoder_channels=24,
                                      context_channels=32, groups=8)


def make(mode: str) -> CausalStateMultibandForecaster:
    torch.manual_seed(19)
    return CausalStateMultibandForecaster(config(), mode)


def test_arms_have_identical_parameters_and_output_contract():
    cumulative, independent = make("cumulative").eval(), make("independent").eval()
    left, right = cumulative.state_dict(), independent.state_dict()
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)
    x = torch.randn(2, 20, 32, 64, 3)
    with torch.inference_mode():
        a, b = cumulative(x), independent(x)
    assert a.shape == b.shape == (2, 20, 32, 64, 3)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_cumulative_and_independent_states_are_distinct_after_visible_transition():
    model = make("cumulative").eval()
    with torch.no_grad():
        for parameter in model.transition.parameters():
            parameter.normal_(std=.03)
    x = torch.randn(2, 20, 32, 64, 3)
    with torch.inference_mode():
        terminal = model.encode_history(x)
        cumulative = model.future_states(terminal, "cumulative")
        independent = model.future_states(terminal, "independent")
        frozen = model.future_states(terminal, "frozen")
        native_prediction = model.forward_components(x, state_mode="cumulative")["prediction"]
        independent_prediction = model.forward_components(x, state_mode="independent")["prediction"]
    assert float((cumulative[:, 1:] - independent[:, 1:]).abs().mean()) > 1e-6
    assert float((native_prediction - independent_prediction).abs().mean()) > 1e-8
    torch.testing.assert_close(frozen, terminal.unsqueeze(1).expand_as(frozen), rtol=0., atol=0.)


def test_bands_have_exact_pool_separation_and_nonzero_rms_when_heads_visible():
    model = make("cumulative").eval()
    with torch.no_grad():
        for layer in (model.decoder.low_out, model.decoder.mid_out, model.decoder.high_out):
            layer.weight.normal_(std=.1)
    x = torch.randn(2, 20, 32, 64, 3)
    with torch.inference_mode():
        parts = model.forward_components(x)
    assert float(parts["low"].square().mean().sqrt()) > 0.
    assert float(parts["mid"].square().mean().sqrt()) > 0.
    assert float(parts["high"].square().mean().sqrt()) > 0.
    mid = parts["mid"].permute(0, 1, 4, 2, 3).reshape(-1, 3, 32, 64)
    high = parts["high"].permute(0, 1, 4, 2, 3).reshape(-1, 3, 32, 64)
    assert float(torch.nn.functional.avg_pool2d(mid, 4).abs().max()) < 2e-6
    assert float(torch.nn.functional.avg_pool2d(high, 2).abs().max()) < 2e-6
    low = parts["low"].permute(0, 1, 4, 2, 3).reshape(-1, 3, 32, 64)
    assert float((low * mid).mean().abs()) < 2e-6
    assert float((low * high).mean().abs()) < 2e-6
    assert float((mid * high).mean().abs()) < 2e-6


def test_decoder_has_no_direct_history_feature_input():
    model = make("cumulative")
    # The only decoder forward arguments are a horizon state and terminal
    # state; the latter reaches it exclusively through FilmResidual.film.
    assert not any("history" in name for name, _ in model.decoder.named_parameters())
    assert all(module.in_channels != 3 for module in model.decoder.modules()
               if isinstance(module, torch.nn.Conv2d))


def test_transition_and_all_three_bands_receive_gradients():
    model = make("cumulative")
    x, target = torch.randn(2, 20, 32, 64, 3), torch.randn(2, 20, 32, 64, 3)
    (model(x) - target).square().mean().backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
               for parameter in model.transition.parameters())
    for head in (model.decoder.low_out, model.decoder.mid_out, model.decoder.high_out):
        assert head.weight.grad is not None and torch.count_nonzero(head.weight.grad)


def test_invalid_mode_and_shape_are_rejected():
    with pytest.raises(ValueError, match="rollout_mode"):
        CausalStateMultibandForecaster(config(), "bad")
    with pytest.raises(ValueError, match="expected history"):
        make("cumulative")(torch.randn(1, 19, 32, 64, 3))
