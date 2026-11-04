from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.history_conditioned_spectral_operator import (
    ARM_MODES,
    HistoryConditionedSpectralConfig,
    HistoryConditionedSpectralOperator,
)


def config() -> HistoryConditionedSpectralConfig:
    return HistoryConditionedSpectralConfig(width=16, layers=2, modes_y=4,
                                            modes_x=6, rank=4,
                                            condition_hidden=24, groups=8)


def make(mode: str) -> HistoryConditionedSpectralOperator:
    torch.manual_seed(7)
    return HistoryConditionedSpectralOperator(config(), mode)


def test_all_arms_have_identical_topology_state_and_initial_prediction():
    models = [make(mode).eval() for mode in ARM_MODES]
    states = [model.state_dict() for model in models]
    assert states[0].keys() == states[1].keys() == states[2].keys()
    assert all(torch.equal(states[0][key], state[key])
               for state in states[1:] for key in states[0])
    x = torch.randn(2, 20, 16, 32, 3)
    with torch.inference_mode():
        values = [model(x) for model in models]
    torch.testing.assert_close(values[0], values[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(values[1], values[2], rtol=0.0, atol=0.0)


def test_conditioned_gate_varies_per_history_while_static_rank_does_not():
    conditional = make("conditional_rank").eval()
    static = make("static_rank").eval()
    # Make the final map visible without training a full model.
    with torch.no_grad():
        conditional.conditioner.last.weight.normal_()
        static.load_state_dict(conditional.state_dict())
    x = torch.randn(3, 20, 16, 32, 3)
    x[1] += .7
    with torch.inference_mode():
        c_gate = conditional.forward_components(x)["gate"]
        s_gate = static.forward_components(x)["gate"]
    assert float(c_gate.std(dim=0).mean()) > 1.0e-5
    torch.testing.assert_close(s_gate[0], s_gate[1], rtol=0.0, atol=0.0)


def test_static_base_rank_correction_is_forward_zero_but_fully_finite():
    model = make("static_base").eval()
    x = torch.randn(2, 20, 16, 32, 3)
    with torch.inference_mode():
        parts = model.forward_components(x)
    assert torch.isfinite(parts["prediction"]).all()
    assert torch.equal(parts["gate"], torch.zeros_like(parts["gate"]))


def test_conditioner_has_no_ghost_gradient_in_static_control():
    x = torch.randn(2, 20, 16, 32, 3)
    static = make("static_rank")
    conditional = make("conditional_rank")
    with torch.no_grad():
        for model in (static, conditional):
            model.static_gate.fill_(.2)
            model.conditioner.last.weight.normal_(std=.05)
    static(x).square().mean().backward()
    conditional(x).square().mean().backward()
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
               for parameter in static.conditioner.parameters())
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
               for parameter in conditional.conditioner.parameters())


def test_gate_override_changes_conditional_operator_output():
    model = make("conditional_rank").eval()
    with torch.no_grad():
        model.static_gate.fill_(.1)
        model.conditioner.last.weight.normal_(std=.1)
    x = torch.randn(3, 20, 16, 32, 3)
    with torch.inference_mode():
        native = model.forward_components(x)
        disabled = model.forward_components(x, gate_override=native["static_gate"])
        shuffled = model.forward_components(
            x, gate_override=torch.roll(native["gate"], 1, dims=0))
    assert float((native["prediction"] - disabled["prediction"]).abs().mean()) > 0.0
    assert float((native["prediction"] - shuffled["prediction"]).abs().mean()) > 0.0


def test_invalid_modes_and_grids_are_rejected():
    with pytest.raises(ValueError, match="arm_mode"):
        HistoryConditionedSpectralOperator(config(), "bad")
    model = make("conditional_rank")
    with pytest.raises(ValueError, match="spectral modes"):
        model(torch.randn(1, 20, 8, 8, 3))
