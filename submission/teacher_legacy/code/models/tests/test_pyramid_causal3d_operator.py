import pytest
import torch
from torch.nn import functional as F

from models.pyramid_causal3d_operator import (
    MatchedCausalAxisMixer,
    PyramidCausal3DOperator,
    PyramidCausal3DOperatorConfig,
)


def tiny_config(**overrides: object) -> PyramidCausal3DOperatorConfig:
    values = dict(
        e0_channels=16,
        e1_channels=24,
        e2_channels=32,
        e0_blocks=1,
        e1_blocks=1,
        e2_blocks=1,
        groups=4,
        channel_mlp_ratio=2.0,
    )
    values.update(overrides)
    return PyramidCausal3DOperatorConfig(**values)


@pytest.mark.parametrize("height,width", ((16, 16), (32, 64), (64, 128)))
def test_dynamic_forecast_shapes_and_finite(height: int, width: int):
    model = PyramidCausal3DOperator(tiny_config()).eval()
    history = torch.randn(1, 20, height, width, 3)
    with torch.inference_mode():
        prediction = model(history)
    assert prediction.shape == (1, 20, height, width, 3)
    assert torch.isfinite(prediction).all()
    assert model.stem.in_channels == 5


def test_default_parameter_count_is_frozen_in_moderate_range():
    model = PyramidCausal3DOperator(PyramidCausal3DOperatorConfig())
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 8_929_552
    assert 5_000_000 <= count <= 10_000_000


def test_paired_modes_have_bitwise_identical_initial_state():
    torch.manual_seed(301)
    candidate = PyramidCausal3DOperator(
        tiny_config(), "causal_temporal_candidate")
    torch.manual_seed(301)
    control = PyramidCausal3DOperator(
        tiny_config(), "framewise_spatial_control")
    assert candidate.state_dict().keys() == control.state_dict().keys()
    assert all(torch.equal(value, control.state_dict()[key])
               for key, value in candidate.state_dict().items())
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in control.parameters())


def test_candidate_is_causal_and_control_has_no_cross_time_path():
    torch.manual_seed(302)
    candidate = MatchedCausalAxisMixer(
        8, 3, 4, "causal_temporal_candidate").eval()
    torch.manual_seed(302)
    control = MatchedCausalAxisMixer(
        8, 3, 4, "framewise_spatial_control").eval()
    control.load_state_dict(candidate.state_dict(), strict=True)
    trajectory = torch.randn(1, 20, 8, 4, 6)
    changed = trajectory.clone()
    changed[:, 1] += 10.0 * torch.randn_like(changed[:, 1])
    candidate_first = candidate(trajectory)
    candidate_second = candidate(changed)
    control_first = control(trajectory)
    control_second = control(changed)
    torch.testing.assert_close(
        candidate_first[:, 0], candidate_second[:, 0], rtol=0.0, atol=0.0)
    assert not torch.equal(candidate_first[:, 2], candidate_second[:, 2])
    torch.testing.assert_close(
        control_first[:, 2], control_second[:, 2], rtol=0.0, atol=0.0)


def test_axis_contract_has_equal_elements_and_macs():
    source = torch.randn(2, 20, 8, 3, 7)
    candidate = MatchedCausalAxisMixer(
        8, 5, 4, "causal_temporal_candidate")
    control = MatchedCausalAxisMixer(
        8, 5, 4, "framewise_spatial_control")
    control.load_state_dict(candidate.state_dict(), strict=True)
    assert candidate(source).shape == control(source).shape == source.shape
    candidate_contract = candidate.last_axis_contract
    control_contract = control.last_axis_contract
    assert candidate_contract is not None and control_contract is not None
    assert candidate_contract["canonical_shape"] == control_contract["canonical_shape"]
    assert candidate_contract["output_elements"] == control_contract["output_elements"]
    assert candidate_contract["macs"] == control_contract["macs"]
    assert candidate.kernel_size == control.kernel_size == 5
    assert candidate.left_pad == control.left_pad == 4
    assert candidate_contract["cross_time"] is True
    assert candidate_contract["control_time_folded"] is False
    assert control_contract["cross_time"] is False
    assert control_contract["control_time_folded"] is True


def _fixed_reference_mixer(mode: str) -> MatchedCausalAxisMixer:
    """Build a tiny mixer whose every arithmetic coefficient is explicit."""
    mixer = MatchedCausalAxisMixer(2, 3, 1, mode).eval()
    with torch.no_grad():
        mixer.norm.weight.copy_(torch.tensor([1.20, 0.80]))
        mixer.norm.bias.copy_(torch.tensor([-0.10, 0.20]))
        mixer.depthwise.weight.copy_(torch.tensor(
            [[[[[0.20]], [[-0.30]], [[0.50]]]],
             [[[[-0.40]], [[0.10]], [[0.25]]]]]))
        mixer.projection.weight.copy_(torch.tensor(
            [[[[[0.60]]], [[[-0.20]]]],
             [[[[0.15]]], [[[0.35]]]],
             [[[[-0.45]]], [[[0.10]]]],
             [[[[0.25]]], [[[0.55]]]]]))
        mixer.projection.bias.copy_(torch.tensor([0.05, -0.10, 0.20, -0.15]))
    return mixer


def _reference_project(mixer: MatchedCausalAxisMixer,
                       sequence: torch.Tensor) -> torch.Tensor:
    """Independent [1,C,L] Conv1d reference for one axial sequence."""
    kernel = mixer.depthwise.weight.detach().squeeze(-1).squeeze(-1)
    filtered = F.conv1d(F.pad(sequence, (2, 0)), kernel, groups=2)
    projection = mixer.projection.weight.detach().squeeze(-1).squeeze(-1)
    projected = F.conv1d(filtered, projection,
                         bias=mixer.projection.bias.detach())
    value, gate = projected.chunk(2, dim=1)
    return value * F.silu(gate)


def test_candidate_matches_independent_left_causal_conv1d_reference():
    mixer = _fixed_reference_mixer("causal_temporal_candidate")
    source = torch.arange(1 * 4 * 2 * 2 * 5, dtype=torch.float32).reshape(
        1, 4, 2, 2, 5) / 17.0
    actual = mixer(source)

    # Normalize each frame exactly as the declared common preprocessing, then
    # independently gather each fixed (y,x) time sequence.  No model packing
    # or restoration helper participates in this reference.
    normalized = F.group_norm(
        source.reshape(4, 2, 2, 5), 1,
        mixer.norm.weight.detach(), mixer.norm.bias.detach(), mixer.norm.eps
    ).reshape(1, 4, 2, 2, 5)
    expected_update = torch.empty_like(source)
    for y in range(2):
        for x in range(5):
            sequence = normalized[0, :, :, y, x].transpose(0, 1).unsqueeze(0)
            update = _reference_project(mixer, sequence)
            expected_update[0, :, :, y, x] = update[0].transpose(0, 1)
    torch.testing.assert_close(actual, source + expected_update,
                               rtol=2.0e-6, atol=2.0e-6)
    assert mixer.last_axis_contract is not None
    assert mixer.last_axis_contract["packed_shape"] == (1, 2, 4, 2, 5)


def test_control_matches_independent_left_causal_x_conv1d_reference():
    mixer = _fixed_reference_mixer("framewise_spatial_control")
    source = torch.arange(1 * 4 * 2 * 2 * 5, dtype=torch.float32).reshape(
        1, 4, 2, 2, 5) / 17.0
    actual = mixer(source)

    normalized = F.group_norm(
        source.reshape(4, 2, 2, 5), 1,
        mixer.norm.weight.detach(), mixer.norm.bias.detach(), mixer.norm.eps
    ).reshape(1, 4, 2, 2, 5)
    expected_update = torch.empty_like(source)
    for step in range(4):
        for y in range(2):
            sequence = normalized[0, step, :, y, :].unsqueeze(0)
            update = _reference_project(mixer, sequence)
            expected_update[0, step, :, y, :] = update[0]
    torch.testing.assert_close(actual, source + expected_update,
                               rtol=2.0e-6, atol=2.0e-6)
    assert mixer.last_axis_contract is not None
    assert mixer.last_axis_contract["packed_shape"] == (4, 2, 5, 2, 1)


def test_component_roles_zero_initial_transport_and_normalizer():
    model = PyramidCausal3DOperator(tiny_config())
    model.set_normalizer_statistics(
        mean_input=torch.tensor([2.0, 3.0, 4.0]),
        mean_target=torch.tensor([1.0, 1.0, 1.0]),
        std_input=torch.tensor([4.0, 6.0, 8.0]),
        std_target=torch.tensor([2.0, 3.0, 4.0]),
    )
    history = torch.randn(1, 20, 16, 16, 3)
    components = model.forward_components(history)
    assert torch.count_nonzero(components["displacement"]) == 0
    torch.testing.assert_close(
        components["transported"],
        components["persistence"].unsqueeze(1).expand_as(
            components["transported"]), rtol=1.0e-5, atol=2.0e-5)
    torch.testing.assert_close(
        components["fluctuation"].mean(dim=1),
        torch.zeros_like(components["fluctuation"][:, 0]),
        rtol=0.0, atol=2.0e-7)
    torch.testing.assert_close(
        components["prediction"],
        components["transported"] + components["mean_delta"].unsqueeze(1)
        + components["fluctuation"])

    with torch.no_grad():
        model.transport_head.bias.fill_(1.0)
    moved = model.forward_components(history)
    assert 0.0 < float(moved["displacement"].abs().max()) < 4.0
    model.set_zero_displacement_override(True)
    zeroed = model.forward_components(history)
    assert torch.count_nonzero(zeroed["displacement"]) == 0
    torch.testing.assert_close(
        zeroed["transported"], zeroed["persistence"].unsqueeze(1).expand_as(
            zeroed["transported"]), rtol=0.0, atol=0.0)


def test_model_axis_bypass_is_exact_identity_and_safely_reset():
    torch.manual_seed(304)
    model = PyramidCausal3DOperator(tiny_config()).eval()
    history = torch.randn(1, 20, 16, 16, 3)
    with torch.inference_mode():
        native = model.encode_history(history)
        try:
            model.set_axis_bypass_override(True)
            bypass = model.encode_history(history)
            assert all(contract["bypassed"] is True
                       for contract in model.axis_execution_contract())
        finally:
            model.set_axis_bypass_override(False)
        restored = model.encode_history(history)
    assert any(not torch.equal(first, second)
               for first, second in zip(native, bypass, strict=True))
    for first, second in zip(native, restored, strict=True):
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_finite_backward_and_all_key_paths_are_live():
    torch.manual_seed(303)
    model = PyramidCausal3DOperator(tiny_config())
    history = torch.randn(1, 20, 16, 16, 3, requires_grad=True)
    target = torch.randn(1, 20, 16, 16, 3)
    loss = (model(history) - target).square().mean()
    loss.backward()
    parameters = (
        model.stem.weight,
        model.stage0[0].axis.depthwise.weight,
        model.stage1[0].axis.depthwise.weight,
        model.stage2[0].axis.depthwise.weight,
        model.observed_to_future_logits,
        model.observed_time0,
        model.horizon2,
        model.mean_head.weight,
        model.fluctuation_head.weight,
        model.transport_head.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert history.grad is not None and torch.isfinite(history.grad).all()


def test_architecture_roundtrip_and_validation():
    model = PyramidCausal3DOperator(
        tiny_config(), "framewise_spatial_control")
    stored = model.architecture_config()
    assert stored.pop("architecture") == "pyramid_causal3d_operator"
    mode = stored.pop("interaction_mode")
    rebuilt = PyramidCausal3DOperator(
        PyramidCausal3DOperatorConfig(**stored), mode)
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    with pytest.raises(ValueError, match="H/W divisible"):
        model(torch.randn(1, 20, 18, 16, 3))
