from __future__ import annotations

import sys
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.local_stable_oscillator import (  # noqa: E402
    LocalStableOscillatorConfig, LocalStableOscillatorForecaster)


def make(family: str = "complex") -> LocalStableOscillatorForecaster:
    torch.manual_seed(210)
    return LocalStableOscillatorForecaster(LocalStableOscillatorConfig(), family)


def test_shape_finite_and_first_backward_reaches_history_and_modal_paths():
    model = make(); x, target = torch.randn(1, 20, 32, 64, 3), torch.randn(1, 20, 32, 64, 3)
    output = model(x)
    assert output.shape == (1, 20, 32, 64, 3) and torch.isfinite(output).all()
    (output - target).square().mean().backward()
    for parameter in (model.frame_stem[0].weight, model.history_cell.gates.weight,
                      model.modal_head[-1].weight, model.rate_head[-1].bias):
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad)


def test_matched_complex_and_real_arms_have_identical_state_and_parameters():
    complex_arm, real_arm = make("complex"), make("real")
    assert complex_arm.state_dict().keys() == real_arm.state_dict().keys()
    assert all(torch.equal(complex_arm.state_dict()[key], real_arm.state_dict()[key])
               for key in complex_arm.state_dict())
    assert sum(p.numel() for p in complex_arm.parameters()) == sum(p.numel() for p in real_arm.parameters())


def test_zero_raw_temporal_coordinate_makes_complex_and_real_outputs_bitwise_same():
    complex_arm, real_arm = make("complex").eval(), make("real").eval()
    x = torch.randn(1, 20, 32, 64, 3)
    with torch.inference_mode():
        left, right = complex_arm(x), real_arm(x)
    assert torch.equal(left, right)


def test_real_poles_are_non_growing():
    model = make("real")
    with torch.no_grad():
        # Exercise nonzero temporal2 as well as the initialized zero case.
        model.rate_head[-1].bias[model.config.modes:].fill_(1.5)
        rates = model.rate_maps(torch.randn(2, 96, 8, 16))
    assert float(rates["real_log_rates"].max()) <= -model.config.gamma_floor + 1e-7
    torch.testing.assert_close(rates["real_log_rates"].amax(dim=2), -rates["gamma"])


def test_second_rate_is_trainable_from_common_zero_initialization_in_both_families():
    x, target = torch.randn(1, 20, 32, 64, 3), torch.randn(1, 20, 32, 64, 3)
    for family in ("complex", "real"):
        model = make(family)
        (model(x) - target).square().mean().backward()
        gradient = model.rate_head[-1].bias.grad[model.config.modes:]
        assert torch.count_nonzero(gradient)


def test_residual_time_mean_is_exactly_the_slow_state_residual():
    model = make().eval()
    with torch.inference_mode():
        parts = model.forward_components(torch.randn(2, 20, 32, 64, 3))
    torch.testing.assert_close(parts["residual"].mean(1), parts["slow_mean_residual"], rtol=0., atol=2e-7)


def test_zero_frequency_override_changes_a_perturbed_complex_temporal_path():
    model = make().eval()
    with torch.no_grad():
        model.rate_head[-1].bias[model.config.modes:].fill_(1.0)
    x = torch.randn(1, 20, 32, 64, 3)
    with torch.inference_mode():
        complex_prediction = model.forward_components(x, temporal_family="complex")["prediction"]
        zero_prediction = model.forward_components(x, temporal_family="zero_frequency")["prediction"]
    assert float((complex_prediction - zero_prediction).abs().max()) > 1e-8
