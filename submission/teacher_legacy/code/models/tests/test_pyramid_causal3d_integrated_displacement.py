from __future__ import annotations

from unittest import mock

import pytest
import torch

from models.pyramid_causal3d_integrated_displacement import (
    PyramidCausal3DIntegratedDisplacementOperator,
)
from models.pyramid_causal3d_operator import (
    PyramidCausal3DOperator,
    PyramidCausal3DOperatorConfig,
)


def tiny_config() -> PyramidCausal3DOperatorConfig:
    return PyramidCausal3DOperatorConfig(
        e0_channels=16, e1_channels=24, e2_channels=32,
        e0_blocks=1, e1_blocks=1, e2_blocks=1,
        groups=4, channel_mlp_ratio=2.0)


def paired_models():
    torch.manual_seed(501)
    absolute = PyramidCausal3DIntegratedDisplacementOperator(
        tiny_config(), "absolute")
    torch.manual_seed(501)
    integrated = PyramidCausal3DIntegratedDisplacementOperator(
        tiny_config(), "horizon_integrated")
    return absolute, integrated


def test_semantics_validate_and_add_no_state_or_parameters():
    absolute, integrated = paired_models()
    base = PyramidCausal3DOperator(tiny_config())
    assert absolute.state_dict().keys() == integrated.state_dict().keys()
    assert absolute.state_dict().keys() == base.state_dict().keys()
    assert all(torch.equal(value, integrated.state_dict()[key])
               for key, value in absolute.state_dict().items())
    assert sum(p.numel() for p in absolute.parameters()) == sum(
        p.numel() for p in integrated.parameters())
    assert not list(absolute.named_buffers()) == []
    with pytest.raises(ValueError, match="displacement_semantics"):
        PyramidCausal3DIntegratedDisplacementOperator(tiny_config(), "bad")


def test_absolute_mode_is_bitwise_identical_to_parent_for_arbitrary_state():
    torch.manual_seed(502)
    parent = PyramidCausal3DOperator(tiny_config()).eval()
    absolute = PyramidCausal3DIntegratedDisplacementOperator(
        tiny_config(), "absolute").eval()
    absolute.load_state_dict(parent.state_dict(), strict=True)
    with torch.no_grad():
        parent.transport_head.weight.normal_(mean=0.0, std=0.2)
        parent.transport_head.bias.uniform_(-0.3, 0.3)
        absolute.load_state_dict(parent.state_dict(), strict=True)
    history = torch.randn(1, 20, 16, 16, 3)
    with torch.inference_mode():
        parent_components = parent.forward_components(history)
        absolute_components = absolute.forward_components(history)
    assert parent_components.keys() == absolute_components.keys()
    for key in parent_components:
        torch.testing.assert_close(
            parent_components[key], absolute_components[key],
            rtol=0.0, atol=0.0)


def test_absolute_mode_remains_bitwise_parent_equivalent_after_optimizer_steps():
    torch.manual_seed(503)
    parent = PyramidCausal3DOperator(tiny_config()).train()
    torch.manual_seed(503)
    absolute = PyramidCausal3DIntegratedDisplacementOperator(
        tiny_config(), "absolute").train()
    parent_optimizer = torch.optim.AdamW(
        parent.parameters(), lr=3.0e-4, weight_decay=1.0e-5)
    absolute_optimizer = torch.optim.AdamW(
        absolute.parameters(), lr=3.0e-4, weight_decay=1.0e-5)
    history = torch.randn(1, 20, 16, 16, 3)
    target = torch.randn_like(history)
    for _step in range(3):
        for model, optimizer in (
                (parent, parent_optimizer), (absolute, absolute_optimizer)):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(history) - target).square().mean()
            loss.backward()
            optimizer.step()
        assert all(torch.equal(value, absolute.state_dict()[key])
                   for key, value in parent.state_dict().items())


def test_update_zero_complete_outputs_are_bitwise_identical():
    absolute, integrated = paired_models()
    history = torch.randn(1, 20, 16, 16, 3)
    with torch.inference_mode():
        first = absolute.forward_components(history)
        second = integrated.forward_components(history)
    assert torch.count_nonzero(first["displacement"]) == 0
    assert torch.count_nonzero(second["displacement"]) == 0
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)


def test_integrated_basis_matches_constant_and_impulse_references():
    _absolute, integrated = paired_models()
    raw = torch.full((1, 20, 8, 2, 3), 0.125)
    actual = integrated.apply_displacement_basis(raw)
    horizon = torch.arange(1, 21, dtype=raw.dtype).view(1, 20, 1, 1, 1)
    torch.testing.assert_close(actual, horizon * raw, rtol=0.0, atol=0.0)

    impulse = torch.zeros_like(raw)
    impulse[:, 7] = 2.0
    response = integrated.apply_displacement_basis(impulse)
    assert torch.count_nonzero(response[:, :7]) == 0
    assert torch.count_nonzero(response[:, 7:]) == response[:, 7:].numel()
    torch.testing.assert_close(response[:, 7:],
                               torch.full_like(response[:, 7:], 2.0),
                               rtol=0.0, atol=0.0)


def test_cumsum_commutes_bitwise_with_per_horizon_pixel_shuffle():
    _absolute, integrated = paired_models()
    raw = torch.arange(2 * 20 * 8 * 3 * 5, dtype=torch.float32).reshape(
        2, 20, 8, 3, 5) / 1024.0
    before_shuffle = integrated._pixel_shuffle(
        torch.cumsum(raw, dim=1), 2, (6, 10))
    after_shuffle = torch.cumsum(
        integrated._pixel_shuffle(raw, 2, (6, 10)), dim=1)
    torch.testing.assert_close(before_shuffle, after_shuffle, rtol=0.0, atol=0.0)


def test_horizon_one_is_exact_and_displacements_remain_bounded():
    absolute, integrated = paired_models()
    with torch.no_grad():
        absolute.transport_head.weight.normal_(mean=0.0, std=0.15)
        absolute.transport_head.bias.uniform_(-0.2, 0.2)
        integrated.load_state_dict(absolute.state_dict(), strict=True)
    history = torch.randn(1, 20, 16, 16, 3)
    first = absolute.forward_components(history)
    second = integrated.forward_components(history)
    torch.testing.assert_close(first["displacement"][:, 0],
                               second["displacement"][:, 0],
                               rtol=0.0, atol=0.0)
    assert float(first["displacement"].abs().max()) <= 4.0
    assert float(second["displacement"].abs().max()) <= 4.0


def test_each_raw_horizon_has_finite_nonzero_future_gradient():
    _absolute, integrated = paired_models()
    raw = (0.04 * torch.randn(1, 20, 8, 2, 2)).requires_grad_()
    bounded = 4.0 * torch.tanh(integrated.apply_displacement_basis(raw))
    weights = torch.arange(1, 21, dtype=raw.dtype).view(1, 20, 1, 1, 1)
    (bounded * weights).sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert torch.count_nonzero(raw.grad.reshape(20, -1), dim=1).eq(
        raw[0].numel() // 20).all()


def test_both_semantics_call_the_same_source_warp_once():
    absolute, integrated = paired_models()
    history = torch.randn(1, 20, 16, 16, 3)
    for model in (absolute, integrated):
        original = model.warp
        with mock.patch.object(model, "warp", wraps=original) as wrapped:
            output = model(history)
        assert output.shape == (1, 20, 16, 16, 3)
        assert wrapped.call_count == 1


def test_architecture_roundtrip_records_only_the_basis_difference():
    absolute, integrated = paired_models()
    left = absolute.architecture_config()
    right = integrated.architecture_config()
    differing = {key for key in left if left[key] != right[key]}
    assert differing == {"displacement_semantics"}
    stored = dict(right)
    assert stored.pop("architecture") == (
        "pyramid_causal3d_integrated_displacement_screen")
    semantics = stored.pop("displacement_semantics")
    interaction = stored.pop("interaction_mode")
    assert interaction == "causal_temporal_candidate"
    rebuilt = PyramidCausal3DIntegratedDisplacementOperator(
        PyramidCausal3DOperatorConfig(**stored), semantics)
    rebuilt.load_state_dict(integrated.state_dict(), strict=True)
