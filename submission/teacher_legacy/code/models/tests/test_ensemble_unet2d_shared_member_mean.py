from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig


def _model(shared: bool) -> EnsembleUNet2d:
    model = EnsembleUNet2d(EnsembleUNet2dConfig(
        base=8, depth=3, groups=4, difference_steps=2,
        ensemble_members=4, uq_hidden_channels=8,
        uq_flow_context=True, mean_fluctuation=True,
        temporal_uq_refinement=True, temporal_uq_channels=8,
        temporal_uq_blocks=1, temporal_uq_kernel=3,
        shared_member_uv_mean=shared,
        disagreement_scale_fraction=0.01))
    model.set_normalizer_statistics(
        torch.zeros(3), torch.zeros(3),
        torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
    return model


def test_shared_mean_preserves_roles_pressure_and_aggregate():
    torch.manual_seed(3)
    raw = torch.randn(2, 4, 7, 3, 5, 3)
    role = EnsembleUNet2d._members_with_shared_uv_mean(raw)

    raw_uv = raw[..., :2]
    role_uv = role[..., :2]
    raw_fluctuation = raw_uv - raw_uv.mean(dim=2, keepdim=True)
    role_fluctuation = role_uv - role_uv.mean(dim=2, keepdim=True)
    role_means = role_uv.mean(dim=2)

    torch.testing.assert_close(
        role_means, role_means[:, :1].expand_as(role_means),
        rtol=0.0, atol=3e-7)
    torch.testing.assert_close(
        role_fluctuation, raw_fluctuation, rtol=0.0, atol=3e-7)
    torch.testing.assert_close(
        role_uv.mean(dim=1), raw_uv.mean(dim=1), rtol=0.0, atol=3e-7)
    assert torch.equal(role[..., 2:], raw[..., 2:])


def test_candidate_point_path_is_bitwise_identity_and_schema_is_unchanged():
    torch.manual_seed(5)
    control = _model(shared=False)
    candidate = _model(shared=True)
    for heads in control.member_point_heads:
        for head in heads:
            torch.nn.init.normal_(head.weight, std=0.02)
            torch.nn.init.normal_(head.bias, std=0.01)
    candidate.load_state_dict(control.state_dict(), strict=True)
    x = torch.randn(1, 20, 16, 32, 3)

    control_values = control.forward_uq_training(x)
    candidate_values = candidate.forward_uq_training(x)

    assert torch.equal(candidate_values[0], control_values[0])
    assert torch.equal(candidate.forward(x), control.forward(x))
    assert torch.equal(candidate_values[1][..., 2:], control_values[1][..., 2:])
    candidate_means = candidate_values[1][..., :2].mean(dim=2)
    torch.testing.assert_close(
        candidate_means, candidate_means[:, :1].expand_as(candidate_means),
        rtol=0.0, atol=1e-5)
    torch.testing.assert_close(
        candidate_values[1][..., :2]
        - candidate_values[1][..., :2].mean(dim=2, keepdim=True),
        control_values[1][..., :2]
        - control_values[1][..., :2].mean(dim=2, keepdim=True),
        rtol=0.0, atol=1e-5)
    assert candidate.state_dict().keys() == control.state_dict().keys()
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in control.parameters())


def test_candidate_uq_fusion_and_temporal_refiner_receive_effective_roles():
    torch.manual_seed(6)
    model = _model(shared=True)
    for member_index, heads in enumerate(model.member_point_heads):
        for head in heads:
            torch.nn.init.normal_(head.weight, std=0.01)
            with torch.no_grad():
                head.bias[:3].add_(0.1 * member_index)
    captured = {}

    def fusion_hook(_module, inputs):
        captured["fusion"] = inputs[0].detach()

    def temporal_hook(_module, inputs):
        captured["temporal"] = inputs[0].detach()

    fusion_handle = model.uq_fusion.register_forward_pre_hook(fusion_hook)
    temporal_handle = model.temporal_uq_refiner.member_projection.register_forward_pre_hook(
        temporal_hook)
    x = torch.randn(1, 20, 16, 32, 3)
    prediction, members, *_ = model.forward_uq_training(x)
    fusion_handle.remove()
    temporal_handle.remove()

    detached = members[..., :2].detach()
    member_std = detached.std(dim=1, unbiased=False)
    member_range = detached.amax(dim=1) - detached.amin(dim=1)
    target_std = model.target_std[:2].view(1, 1, 1, 1, 2)
    reference = detached.new_tensor(
        model.config.uncertainty_base_widths).view(1, 1, 1, 1, 2)
    reference = reference * model.config.disagreement_scale_fraction
    member_std = torch.log1p(member_std * target_std / reference)
    member_range = torch.log1p(member_range * target_std / reference)
    disagreement = torch.cat((member_std, member_range), dim=-1)
    expected_disagreement = disagreement.permute(0, 1, 4, 2, 3).reshape(
        1, -1, 16, 32)
    projected_channels = (model.config.depth - 1) * model.config.uq_hidden_channels
    actual_disagreement = captured["fusion"][
        :, projected_channels:projected_channels + 4 * model.config.out_steps]
    torch.testing.assert_close(actual_disagreement, expected_disagreement)
    torch.testing.assert_close(
        captured["fusion"][:, -70:], model._flow_context(x, prediction))

    expected_temporal = torch.cat((
        detached.mean(dim=1),
        detached.std(dim=1, unbiased=False),
        detached.amax(dim=1) - detached.amin(dim=1)), dim=-1
    ).permute(0, 4, 1, 2, 3)
    torch.testing.assert_close(captured["temporal"], expected_temporal)


def test_legacy_missing_flag_defaults_false_and_true_config_roundtrips():
    legacy = _model(shared=False)
    legacy_config = legacy.architecture_config()
    assert legacy_config.pop("architecture") == "ensemble_unet"
    legacy_config.pop("shared_member_uv_mean")
    rebuilt_legacy = EnsembleUNet2d(EnsembleUNet2dConfig(**legacy_config))
    rebuilt_legacy.load_state_dict(legacy.state_dict(), strict=True)
    assert rebuilt_legacy.config.shared_member_uv_mean is False

    candidate = _model(shared=True)
    candidate_config = candidate.architecture_config()
    assert candidate_config.pop("architecture") == "ensemble_unet"
    assert candidate_config["shared_member_uv_mean"] is True
    rebuilt_candidate = EnsembleUNet2d(EnsembleUNet2dConfig(**candidate_config))
    rebuilt_candidate.load_state_dict(candidate.state_dict(), strict=True)
    assert rebuilt_candidate.config.shared_member_uv_mean is True


def test_shared_mean_gradient_couples_means_but_not_centered_roles():
    torch.manual_seed(7)
    raw = torch.randn(1, 4, 6, 2, 3, 3, requires_grad=True)
    role = EnsembleUNet2d._members_with_shared_uv_mean(raw)
    role[:, 0, ..., :2].mean(dim=1).square().mean().backward()
    mean_grad = raw.grad[..., :2]
    assert all(float(mean_grad[:, index].abs().sum()) > 0.0
               for index in range(4))
    torch.testing.assert_close(
        mean_grad, mean_grad[:, :, :1].expand_as(mean_grad),
        rtol=0.0, atol=2e-8)

    centered_raw = torch.randn(1, 4, 6, 2, 3, 3, requires_grad=True)
    centered_role = EnsembleUNet2d._members_with_shared_uv_mean(centered_raw)
    fluctuation = (
        centered_role[:, 0, ..., :2]
        - centered_role[:, 0, ..., :2].mean(dim=1, keepdim=True))
    fluctuation.square().mean().backward()
    assert float(centered_raw.grad[:, 0, ..., :2].abs().sum()) > 0.0
    torch.testing.assert_close(
        centered_raw.grad[:, 1:, ..., :2],
        torch.zeros_like(centered_raw.grad[:, 1:, ..., :2]),
        rtol=0.0, atol=2e-8)


def test_mean_only_member_disagreement_is_removed():
    time = torch.linspace(-1.0, 1.0, 5).view(1, 1, 5, 1, 1, 1)
    member_mean = torch.tensor([-1.5, -0.5, 0.5, 1.5]).view(
        1, 4, 1, 1, 1, 1)
    uv = (time + member_mean).expand(1, 4, 5, 2, 3, 2).clone()
    pressure = torch.randn(1, 4, 5, 2, 3, 1)
    raw = torch.cat((uv, pressure), dim=-1)

    role = EnsembleUNet2d._members_with_shared_uv_mean(raw)

    assert float(raw[..., :2].std(dim=1, unbiased=False).mean()) > 0.5
    torch.testing.assert_close(
        role[..., :2].std(dim=1, unbiased=False),
        torch.zeros_like(role[..., :2].std(dim=1, unbiased=False)),
        rtol=0.0, atol=2e-7)


@pytest.mark.parametrize("bad", [
    torch.zeros(2, 3, 4, 5, 2),
    torch.zeros(2, 3, 4, 5, 6, 1),
])
def test_shared_mean_rejects_invalid_member_contract(bad: torch.Tensor):
    with pytest.raises(ValueError, match="members must have shape"):
        EnsembleUNet2d._members_with_shared_uv_mean(bad)
