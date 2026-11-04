from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.temporal_transport import (  # noqa: E402
    TemporalTransportConfig,
    TemporalTransportOperator,
)


def config(**overrides) -> TemporalTransportConfig:
    values = dict(
        in_steps=6, out_steps=5, in_channels=3, out_channels=3,
        hidden_channels=8, spatial_blocks=1, spatial_downsample=2,
        temporal_blocks=2, temporal_kernel=3, groups=4,
        max_displacement=2.0,
    )
    values.update(overrides)
    return TemporalTransportConfig(**values)


def test_forecast_and_component_shapes_and_exact_zero_time_mean():
    model = TemporalTransportOperator(config())
    history = torch.randn(2, 6, 7, 11, 3)
    result = model.forward_components(history)

    assert result.prediction.shape == (2, 5, 7, 11, 3)
    assert result.transported.shape == (2, 5, 7, 11, 3)
    assert result.mean_delta.shape == (2, 7, 11, 3)
    assert result.fluctuation.shape == (2, 5, 7, 11, 3)
    assert result.displacement.shape == (2, 5, 7, 11, 2)
    assert result.attention.shape == (2, 5, 6, 4, 6)
    torch.testing.assert_close(
        result.fluctuation.mean(dim=1),
        torch.zeros_like(result.fluctuation[:, 0]),
        rtol=0.0, atol=2.0e-7)
    torch.testing.assert_close(
        result.prediction,
        result.transported + result.mean_delta.unsqueeze(1)
        + result.fluctuation)


def test_temporal_mixer_is_strictly_causal():
    torch.manual_seed(17)
    model = TemporalTransportOperator(config()).eval()
    first = torch.randn(1, 6, 8, 10, 3)
    second = first.clone()
    second[:, 4:] = 100.0 * torch.randn_like(second[:, 4:])

    first_states = model.encode_history(first)
    second_states = model.encode_history(second)
    torch.testing.assert_close(
        first_states[:, :4], second_states[:, :4], rtol=0.0, atol=0.0)
    assert not torch.equal(first_states[:, 4:], second_states[:, 4:])


def test_candidate_and_control_have_exactly_matched_parameter_shapes():
    candidate = TemporalTransportOperator(config(
        mixer_mode="causal_temporal"))
    control = TemporalTransportOperator(config(
        mixer_mode="framewise_spatial_control"))

    candidate_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in candidate.named_parameters()
    }
    control_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in control.named_parameters()
    }
    assert candidate_shapes == control_shapes
    assert sum(parameter.numel() for parameter in candidate.parameters()) == (
        sum(parameter.numel() for parameter in control.parameters()))
    for candidate_block, control_block in zip(
            candidate.temporal_mixer.blocks,
            control.temporal_mixer.blocks, strict=True):
        assert candidate_block.depthwise.weight.shape == (
            control_block.depthwise.weight.shape)
    history = torch.randn(1, 6, 7, 11, 3)
    assert candidate(history).shape == control(history).shape == (
        1, 5, 7, 11, 3)


def test_matched_control_shape_and_frame_independence():
    torch.manual_seed(19)
    model = TemporalTransportOperator(config(
        mixer_mode="framewise_spatial_control")).eval()
    trajectory = torch.randn(2, 6, 8, 4, 6)
    changed = trajectory.clone()
    # Preserve one arbitrary frame and replace every other latent frame.
    changed[:, :3] = 50.0 * torch.randn_like(changed[:, :3])
    changed[:, 4:] = 50.0 * torch.randn_like(changed[:, 4:])

    original_states = model.temporal_mixer(trajectory)
    changed_states = model.temporal_mixer(changed)
    assert original_states.shape == trajectory.shape
    torch.testing.assert_close(
        original_states[:, 3], changed_states[:, 3], rtol=0.0, atol=0.0)
    assert not torch.equal(original_states[:, :3], changed_states[:, :3])
    assert not torch.equal(original_states[:, 4:], changed_states[:, 4:])


def test_candidate_state_uses_earlier_causal_history():
    torch.manual_seed(21)
    model = TemporalTransportOperator(config(
        mixer_mode="causal_temporal")).eval()
    trajectory = torch.randn(2, 6, 8, 4, 6)
    changed = trajectory.clone()
    changed[:, 1] = 50.0 * torch.randn_like(changed[:, 1])

    original_states = model.temporal_mixer(trajectory)
    changed_states = model.temporal_mixer(changed)
    # A changed earlier frame affects a later state through the causal kernel,
    # while the state preceding that edit remains bit-identical.
    torch.testing.assert_close(
        original_states[:, 0], changed_states[:, 0], rtol=0.0, atol=0.0)
    assert not torch.equal(original_states[:, 2], changed_states[:, 2])


def test_all_output_heads_and_temporal_core_receive_gradient():
    torch.manual_seed(23)
    model = TemporalTransportOperator(config())
    history = torch.randn(2, 6, 8, 10, 3, requires_grad=True)
    target = torch.randn(2, 5, 8, 10, 3)
    loss = (model(history) - target).square().mean()
    loss.backward()

    parameters = (
        model.transport_head.weight,
        model.mean_head.weight,
        model.fluctuation_head.weight,
        model.temporal_mixer.blocks[0].depthwise.weight,
        model.horizon_decoder.query,
        model.frame_encoder.stem.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0
    assert history.grad is not None
    assert torch.isfinite(history.grad).all()


def test_pixel_warp_direction_and_displacement_gradient():
    source = torch.zeros(1, 3, 5, 1)
    source[0, :, :, 0] = torch.arange(5).view(1, 5)
    displacement = torch.zeros(1, 2, 3, 5, 2, requires_grad=True)
    with torch.no_grad():
        displacement[..., 0].fill_(1.0)
    warped = TemporalTransportOperator.warp(source, displacement)

    assert torch.equal(warped[0, 0, :, :-1, 0], source[0, :, 1:, 0])
    assert torch.equal(warped[0, 0, :, -1, 0], source[0, :, -1, 0])
    warped[..., 0].square().mean().backward()
    assert displacement.grad is not None
    assert float(displacement.grad[..., 0].abs().sum()) > 0.0


def test_horizon_attention_is_a_probability_distribution():
    model = TemporalTransportOperator(config())
    result = model.forward_components(torch.randn(2, 6, 8, 10, 3))
    torch.testing.assert_close(
        result.attention.sum(dim=2),
        torch.ones_like(result.attention[:, :, 0]),
        rtol=1.0e-6, atol=1.0e-6)
    assert torch.all(result.attention >= 0.0)


def test_config_roundtrip_and_input_validation():
    model = TemporalTransportOperator(config(
        mixer_mode="framewise_spatial_control"))
    stored = model.architecture_config()
    assert stored.pop("architecture") == "temporal_transport"
    assert stored["mixer_mode"] == "framewise_spatial_control"
    rebuilt = TemporalTransportOperator(TemporalTransportConfig(**stored))
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert rebuilt.config == model.config

    with pytest.raises(ValueError, match="expected history"):
        model(torch.randn(1, 5, 8, 10, 3))
    with pytest.raises(ValueError, match="divisible"):
        TemporalTransportConfig(hidden_channels=10, groups=4).validate()
    with pytest.raises(ValueError, match="mixer_mode"):
        TemporalTransportConfig(mixer_mode="flatten_time").validate()
