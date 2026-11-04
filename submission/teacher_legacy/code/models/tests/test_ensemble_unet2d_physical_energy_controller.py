from __future__ import annotations

import inspect
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
    PhysicalEnergyController,
)


def _model(enabled: bool, *, member_sync: bool = False) -> EnsembleUNet2d:
    model = EnsembleUNet2d(EnsembleUNet2dConfig(
        base=8, depth=3, groups=4, difference_steps=2,
        ensemble_members=3, uq_hidden_channels=8,
        uq_flow_context=True, mean_fluctuation=True,
        physical_energy_controller=enabled,
        physical_energy_member_sync=member_sync,
        physical_energy_hidden_channels=4,
        physical_energy_log_limit=0.8))
    model.set_normalizer_statistics(
        torch.zeros(3), torch.zeros(3),
        torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
    return model


def _copy_legacy_state(
        legacy: EnsembleUNet2d,
        candidate: EnsembleUNet2d) -> None:
    incompatible = candidate.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(key.startswith("energy_controller.")
               for key in incompatible.missing_keys)


def test_default_off_has_no_new_keys_and_strictly_loads_legacy_state():
    legacy = _model(enabled=False)
    legacy_state = legacy.state_dict()
    assert not any(key.startswith("energy_controller.")
                   for key in legacy_state)

    old_config = legacy.architecture_config()
    assert old_config.pop("architecture") == "ensemble_unet"
    old_config.pop("physical_energy_controller")
    old_config.pop("physical_energy_member_sync")
    old_config.pop("physical_energy_hidden_channels")
    old_config.pop("physical_energy_log_limit")
    rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**old_config))
    rebuilt.load_state_dict(legacy_state, strict=True)
    assert rebuilt.energy_controller is None
    assert rebuilt.state_dict().keys() == legacy_state.keys()


@pytest.mark.parametrize("member_sync", [False, True])
def test_enabled_update_zero_is_bitwise_legacy_for_point_and_intervals(
        member_sync):
    torch.manual_seed(4)
    legacy = _model(enabled=False).eval()
    with torch.no_grad():
        for parameter in legacy.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    candidate = _model(enabled=True, member_sync=member_sync).eval()
    _copy_legacy_state(legacy, candidate)
    assert torch.count_nonzero(candidate.energy_controller.body[-1].weight) == 0
    assert torch.count_nonzero(candidate.energy_controller.body[-1].bias) == 0

    x = torch.randn(2, 20, 16, 32, 3)
    with torch.no_grad():
        expected = legacy.forward_distribution(x)
        actual = candidate.forward_distribution(x)
        expected_point = legacy(x)
        actual_point = candidate(x)

    assert torch.equal(actual_point, expected_point)
    for expected_value, actual_value in zip(expected, actual):
        assert torch.equal(actual_value, expected_value)


def test_member_sync_controls_members_point_and_uq_from_one_state():
    torch.manual_seed(7)
    model = _model(enabled=True, member_sync=True).eval()
    with torch.no_grad():
        model.energy_controller.body[-1].weight.fill_(0.2)
        model.energy_controller.body[-1].bias.fill_(0.1)
        for heads in model.member_point_heads:
            for head in heads:
                head.weight.normal_(std=0.02)
                head.bias.normal_(std=0.01)
    x = torch.randn(2, 20, 16, 32, 3)
    with torch.no_grad():
        features = model._decode(x)
        legacy_members = model._members_from_features(x, features)
        expected_members = model._energy_controlled_members(
            legacy_members, features)
        prediction, members, _quantiles, _lower, _upper, _features, _ = (
            model.forward_uq_training(x))

    assert torch.equal(members, expected_members)
    assert torch.equal(prediction, expected_members.mean(dim=1))
    assert not torch.equal(members, legacy_members)
    legacy_mean = legacy_members[..., :2].mean(dim=2)
    corrected_mean = members[..., :2].mean(dim=2)
    torch.testing.assert_close(corrected_mean, legacy_mean)
    assert torch.equal(members[..., 2:], legacy_members[..., 2:])


def test_member_sync_zero_controller_preserves_ieee_payloads_and_gradient():
    model = _model(enabled=True, member_sync=True)
    finite = torch.tensor([
        0.0, -0.0, torch.finfo(torch.float32).smallest_normal,
        -torch.finfo(torch.float32).smallest_normal,
    ], dtype=torch.float32)
    members = finite.reshape(1, 1, 2, 1, 1, 2).repeat(1, 3, 1, 2, 2, 1)
    members = members.clone().requires_grad_(True)
    features = [torch.randn(1, model.config.base, 2, 2)]
    controlled = model._energy_controlled_members(members, features)
    assert torch.equal(
        controlled.detach().view(torch.int32),
        members.detach().view(torch.int32))

    gradient_members = torch.randn(
        1, 3, 5, 2, 2, 3, requires_grad=True)
    gradient_controlled = model._energy_controlled_members(
        gradient_members, features)
    gradient_controlled[..., :2].square().mean().backward()
    final = model.energy_controller.body[-1]
    assert final.weight.grad is not None
    assert float(final.weight.grad.abs().sum()) > 0.0


def test_controller_preserves_mvpe_mean_and_pressure_and_scales_energy():
    torch.manual_seed(11)
    model = _model(enabled=True)
    output = model.energy_controller.body[-1]
    with torch.no_grad():
        output.weight.fill_(0.3)
        output.bias.fill_(0.1)

    batch, steps, height, width = 2, 7, 6, 10
    prediction = torch.randn(batch, steps, height, width, 3)
    finest = torch.randn(batch, model.config.base, height, width)
    controlled = model._energy_controlled_prediction(prediction, [finest])

    raw_delta = model.energy_controller(finest)
    delta_logk = (
        model.config.physical_energy_log_limit * torch.tanh(raw_delta)
    ).permute(0, 2, 3, 1)
    amplitude = torch.exp(0.5 * delta_logk).unsqueeze(1)
    input_mean = prediction[..., :2].mean(dim=1, keepdim=True)
    input_fluctuation = prediction[..., :2] - input_mean
    expected_uv = input_mean + input_fluctuation * amplitude

    torch.testing.assert_close(
        controlled[..., :2], expected_uv, rtol=2e-6, atol=3e-7)
    torch.testing.assert_close(
        controlled[..., :2].mean(dim=1), input_mean.squeeze(1),
        rtol=2e-6, atol=3e-7)
    assert torch.equal(controlled[..., 2:], prediction[..., 2:])

    physical_scale = model.target_std[:2].view(1, 1, 1, 1, 2)
    legacy_physical_mean = input_mean * physical_scale
    controlled_physical_mean = (
        controlled[..., :2] * physical_scale).mean(dim=1, keepdim=True)
    target_mean = torch.randn(batch, 1, height, width, 2)
    legacy_mvpe = (
        legacy_physical_mean - target_mean).square().sum(dim=-1)
    controlled_mvpe = (
        controlled_physical_mean - target_mean
    ).square().sum(dim=-1)
    torch.testing.assert_close(
        controlled_mvpe, legacy_mvpe, rtol=3e-6, atol=5e-7)

    controlled_fluctuation = controlled[..., :2] - input_mean
    physical_input_fluctuation = input_fluctuation * physical_scale
    physical_controlled_fluctuation = (
        controlled_fluctuation * physical_scale)
    legacy_energy = (
        0.5 * physical_input_fluctuation.square().sum(dim=-1).mean(dim=1))
    controlled_energy = (
        0.5 * physical_controlled_fluctuation.square().sum(dim=-1).mean(dim=1))
    torch.testing.assert_close(
        controlled_energy,
        legacy_energy * torch.exp(delta_logk.squeeze(-1)),
        rtol=4e-6, atol=5e-7)
    assert float(delta_logk.detach().std()) > 0.01


def test_default_controller_remains_feature_only_and_sample_conditional():
    assert tuple(inspect.signature(PhysicalEnergyController.forward).parameters) == (
        "self", "feature")
    assert tuple(
        inspect.signature(
            EnsembleUNet2d._energy_controlled_prediction).parameters
    ) == ("self", "prediction", "features", "history")

    torch.manual_seed(17)
    model = _model(enabled=True)
    with torch.no_grad():
        model.energy_controller.body[-1].weight.fill_(0.2)
    prediction = torch.randn(2, 5, 4, 8, 3)
    features = [torch.randn(2, model.config.base, 4, 8)]
    captured: list[torch.Tensor] = []

    def capture(_module, inputs):
        captured.append(inputs[0])

    handle = model.energy_controller.register_forward_pre_hook(capture)
    model._energy_controlled_prediction(prediction, features)
    handle.remove()

    assert len(captured) == 1
    assert captured[0].data_ptr() == features[-1].data_ptr()
    raw_delta = model.energy_controller(features[-1])
    assert raw_delta.shape == (2, 1, 4, 8)
    assert float(raw_delta.detach().flatten(2).std(dim=-1).min()) > 0.0
    assert not torch.equal(raw_delta[0], raw_delta[1])


def test_controller_only_training_gets_a_point_gradient_at_update_zero():
    torch.manual_seed(23)
    model = _model(enabled=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("energy_controller."))
    for heads in model.member_point_heads:
        for head in heads:
            with torch.no_grad():
                head.weight.normal_(std=0.02)
                head.bias.normal_(std=0.01)

    prediction = model(torch.randn(1, 20, 16, 32, 3))
    prediction[..., :2].square().mean().backward()

    final = model.energy_controller.body[-1]
    assert final.weight.grad is not None
    assert float(final.weight.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if not name.startswith("energy_controller."))


@pytest.mark.parametrize("overrides, message", [
    ({"physical_energy_controller": True, "mean_fluctuation": False},
     "requires mean_fluctuation"),
    ({"physical_energy_hidden_channels": 0}, "hidden channels"),
    ({"physical_energy_log_limit": 0.0}, "log limit"),
    ({"physical_energy_member_sync": True}, "requires the physical energy"),
])
def test_controller_config_rejects_invalid_contract(overrides, message):
    config = dict(
        base=8, depth=3, groups=4, difference_steps=2,
        ensemble_members=3, uq_hidden_channels=8)
    config.update(overrides)
    with pytest.raises(ValueError, match=message):
        EnsembleUNet2d(EnsembleUNet2dConfig(**config))
