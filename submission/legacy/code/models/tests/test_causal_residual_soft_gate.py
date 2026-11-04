from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from models.causal_residual_soft_gate import (
    CausalResidualSoftGate,
    CausalResidualSoftGateConfig,
)
from models.causal_phase_direction import CausalPhaseDirectionConfig


def _config(mode: str = "causal_temporal") -> CausalResidualSoftGateConfig:
    return CausalResidualSoftGateConfig(mixer_mode=mode)


def _inputs(batch: int = 2):
    torch.manual_seed(19)
    history = torch.randn(batch, 20, 8, 8, 3)
    prediction = torch.randn(batch, 20, 8, 8, 3)
    lower = torch.rand(batch, 20, 8, 8, 2) + 0.1
    upper = torch.rand(batch, 20, 8, 8, 2) + 0.1
    return history, prediction, lower, upper


def test_matched_control_has_identical_parameter_shapes_and_values():
    torch.manual_seed(31)
    control = CausalResidualSoftGate(_config("framewise_spatial_control"))
    torch.manual_seed(31)
    candidate = CausalResidualSoftGate(_config("causal_temporal"))
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    assert control_state.keys() == candidate_state.keys()
    assert all(torch.equal(control_state[key], candidate_state[key])
               for key in control_state)
    assert sum(value.numel() for value in control.parameters()) == 41_671
    assert sum(value.numel() for value in candidate.parameters()) == 41_671


def test_update_zero_is_bitwise_native_distribution_identity():
    model = CausalResidualSoftGate(_config()).eval()
    history, prediction, lower, upper = _inputs()
    with torch.no_grad():
        values = model.forward_components(history, prediction, lower, upper)
    assert torch.count_nonzero(values.specialist_residual) > 0
    assert torch.equal(values.prediction, prediction)
    assert torch.equal(values.lower_extent, lower)
    assert torch.equal(values.upper_extent, upper)


def test_both_role_updates_are_zero_time_mean_and_preserve_width_pressure():
    model = CausalResidualSoftGate(_config()).eval()
    with torch.no_grad():
        model.point_gate_logits.copy_(torch.tensor([[[[[0.3, -0.2]]]]]))
        model.uq_gate_logits.copy_(torch.tensor([[[[[-0.4, 0.1]]]]]))
    history, prediction, lower, upper = _inputs()
    with torch.no_grad():
        values = model.forward_components(history, prediction, lower, upper)
    torch.testing.assert_close(
        values.specialist_residual.mean(dim=1),
        torch.zeros_like(values.specialist_residual[:, 0]),
        atol=1.0e-7, rtol=0.0)
    torch.testing.assert_close(
        values.point_delta.mean(dim=1),
        torch.zeros_like(values.point_delta[:, 0]),
        atol=1.0e-7, rtol=0.0)
    torch.testing.assert_close(
        values.uq_center_delta.mean(dim=1),
        torch.zeros_like(values.uq_center_delta[:, 0]),
        atol=1.0e-7, rtol=0.0)
    assert torch.equal(values.prediction[..., 2:], prediction[..., 2:])

    base_width = lower + upper
    candidate_width = values.lower_extent + values.upper_extent
    torch.testing.assert_close(candidate_width, base_width, atol=3.0e-7, rtol=0.0)
    base_center = prediction[..., :2] + 0.5 * (upper - lower)
    candidate_center = values.prediction[..., :2] + 0.5 * (
        values.upper_extent - values.lower_extent)
    torch.testing.assert_close(
        candidate_center - base_center, values.uq_center_delta,
        atol=3.0e-7, rtol=0.0)


def test_point_and_uq_gates_have_isolated_gradients():
    model = CausalResidualSoftGate(_config()).train()
    history, prediction, lower, upper = _inputs()
    values = model.forward_components(history, prediction, lower, upper)
    values.prediction[..., :2].square().mean().backward()
    assert model.point_gate_logits.grad is not None
    assert torch.count_nonzero(model.point_gate_logits.grad)
    assert model.uq_gate_logits.grad is None

    model.zero_grad(set_to_none=True)
    values = model.forward_components(history, prediction, lower, upper)
    lower_absolute = values.prediction[..., :2].detach() - values.lower_extent
    upper_absolute = values.prediction[..., :2].detach() + values.upper_extent
    (lower_absolute.square().mean() + upper_absolute.square().mean()).backward()
    assert model.uq_gate_logits.grad is not None
    assert torch.count_nonzero(model.uq_gate_logits.grad)
    assert model.point_gate_logits.grad is None


def test_zero_gate_bootstrap_reaches_specialist_on_second_update():
    model = CausalResidualSoftGate(_config()).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history, prediction, lower, upper = _inputs(batch=1)
    target = torch.randn_like(prediction[..., :2])

    optimizer.zero_grad(set_to_none=True)
    values = model.forward_components(history, prediction, lower, upper)
    (values.prediction[..., :2] - target).square().mean().backward()
    assert torch.count_nonzero(model.point_gate_logits.grad)
    first_specialist_gradient = sum(
        0 if parameter.grad is None else int(torch.count_nonzero(parameter.grad))
        for parameter in model.specialist.parameters())
    assert first_specialist_gradient == 0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    values = model.forward_components(history, prediction, lower, upper)
    (values.prediction[..., :2] - target).square().mean().backward()
    second_specialist_gradient = sum(
        0 if parameter.grad is None else int(torch.count_nonzero(parameter.grad))
        for parameter in model.specialist.parameters())
    assert second_specialist_gradient > 0


def test_architecture_config_roundtrip_and_validation():
    config = _config()
    model = CausalResidualSoftGate(config)
    stored = model.architecture_config()
    assert stored.pop("architecture") == "causal_residual_soft_gate"
    for key in (
            "anchor_responsibility", "specialist_responsibility",
            "point_fusion", "uq_fusion", "hard_replacement_available",
            "pressure_passthrough", "native_interval_width_preserved"):
        stored.pop(key)
    rebuilt = CausalResidualSoftGate(CausalResidualSoftGateConfig(**stored))
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert asdict(rebuilt.config) == asdict(config)

    with pytest.raises(ValueError, match="limits"):
        CausalResidualSoftGateConfig(point_gate_limit=0.0).validate()
    with pytest.raises(ValueError, match="std"):
        CausalResidualSoftGateConfig(specialist_head_std=0.0).validate()
    with pytest.raises(ValueError, match="mixer_mode"):
        CausalResidualSoftGateConfig(mixer_mode="not_a_mode").validate()


def test_default_specialist_topology_remains_the_audited_phase_topology():
    model = CausalResidualSoftGate(_config())
    expected = CausalPhaseDirectionConfig(mixer_mode="causal_temporal")
    assert asdict(model.specialist.config) == asdict(expected)
