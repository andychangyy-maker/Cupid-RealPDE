from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.ensemble_unet2d import (
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
    ScalarPotentialResidual,
)


def _model(mode: str) -> EnsembleUNet2d:
    options = dict(
        base=8, depth=3, groups=4, difference_steps=2,
        ensemble_members=3, uq_hidden_channels=8)
    if mode != "none":
        options.update(
            scalar_potential_residual_mode=mode,
            scalar_potential_residual_channels=8,
            scalar_potential_residual_blocks=1)
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**options))
    model.set_normalizer_statistics(
        torch.zeros(3), torch.zeros(3), torch.ones(3),
        torch.tensor([0.2, 0.1, 0.3]))
    return model


def _divergence(velocity: torch.Tensor) -> torch.Tensor:
    du_dx, _ = ScalarPotentialResidual.central_derivatives(velocity[..., 0])
    _, dv_dy = ScalarPotentialResidual.central_derivatives(velocity[..., 1])
    return du_dx + dv_dy


def test_legacy_and_active_zero_are_bitwise_and_modes_share_parameters():
    torch.manual_seed(29)
    legacy = _model("none")
    torch.manual_seed(31)
    control = _model("gradient")
    torch.manual_seed(31)
    candidate = _model("curl")
    assert control.state_dict().keys() == candidate.state_dict().keys()
    for name, value in control.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name

    torch.manual_seed(29)
    active_zero = _model("curl")
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, active_zero.state_dict()[name]), name
    x = torch.randn(1, 20, 16, 32, 3)
    legacy_values = legacy.forward_uq_training(x)
    active_values = active_zero.forward_uq_training(x)
    # Point, members, quantiles, and both selected UQ extents are all exact
    # when the candidate's final scalar projection is at its zero start.
    for legacy_value, active_value in zip(legacy_values[:5], active_values[:5]):
        assert torch.equal(legacy_value, active_value)
    assert torch.equal(legacy(x), active_zero(x))

    control_values = control.forward_uq_training(x)
    candidate_values = candidate.forward_uq_training(x)
    for control_value, candidate_value in zip(control_values[:5],
                                               candidate_values[:5]):
        assert torch.equal(control_value, candidate_value)
    audit = candidate.scalar_potential_residual_audit(x)
    assert set(audit) == {"psi", "physical_correction", "normalized_correction"}
    assert torch.equal(audit["psi"], torch.zeros_like(audit["psi"]))
    assert torch.equal(audit["physical_correction"],
                       torch.zeros_like(audit["physical_correction"]))


def test_curl_is_divergence_free_has_fixed_support_and_preserves_pressure():
    torch.manual_seed(32)
    model = _model("curl")
    assert model.scalar_potential_residual is not None
    with torch.no_grad():
        # A spatially varying potential exercises the scalar head, tanh, and
        # the tensor-product operator rather than only a constant bias.
        model.scalar_potential_residual.output.weight.normal_(std=0.04)
        model.scalar_potential_residual.output.bias.fill_(0.2)
    x = torch.randn(1, 20, 16, 32, 3)
    audit = model.scalar_potential_residual_audit(x)
    psi = audit["psi"]
    physical = audit["physical_correction"]
    normalized = audit["normalized_correction"]
    assert torch.equal(psi[..., :2, :], torch.zeros_like(psi[..., :2, :]))
    assert torch.equal(psi[..., -2:, :], torch.zeros_like(psi[..., -2:, :]))
    assert torch.equal(psi[..., :, :2], torch.zeros_like(psi[..., :, :2]))
    assert torch.equal(psi[..., :, -2:], torch.zeros_like(psi[..., :, -2:]))
    assert torch.equal(physical[..., 0, :, :],
                       torch.zeros_like(physical[..., 0, :, :]))
    assert torch.equal(physical[..., -1, :, :],
                       torch.zeros_like(physical[..., -1, :, :]))
    assert torch.equal(physical[..., :, 0, :],
                       torch.zeros_like(physical[..., :, 0, :]))
    assert torch.equal(physical[..., :, -1, :],
                       torch.zeros_like(physical[..., :, -1, :]))
    assert float(_divergence(physical).abs().max()) < 2.0e-7
    assert float(physical.abs().max()) <= 0.05 + 1.0e-7
    torch.testing.assert_close(
        normalized * model.target_std[:2].view(1, 1, 1, 1, 2), physical,
        rtol=0.0, atol=0.0)

    base = _model("curl")
    base.load_state_dict(model.state_dict(), strict=True)
    # Disable only the application, retaining the same ordinary point/UQ
    # tensors so the observed delta is precisely the shared correction.
    base.scalar_potential_residual = None
    candidate_prediction, candidate_members, *_ = model.forward_uq_training(x)
    base_prediction, base_members, *_ = base.forward_uq_training(x)
    torch.testing.assert_close(
        candidate_prediction[..., :2] - base_prediction[..., :2], normalized)
    torch.testing.assert_close(
        candidate_members[..., :2] - base_members[..., :2],
        normalized[:, None].expand_as(candidate_members[..., :2]))
    assert torch.equal(candidate_prediction[..., 2:], base_prediction[..., 2:])
    assert torch.equal(candidate_members[..., 2:], base_members[..., 2:])


def test_scalar_potential_configuration_contract():
    with pytest.raises(ValueError, match="mode and channels"):
        EnsembleUNet2dConfig(scalar_potential_residual_channels=8).validate()
    with pytest.raises(ValueError, match="none/gradient/curl"):
        EnsembleUNet2dConfig(
            scalar_potential_residual_mode="bad",
            scalar_potential_residual_channels=8).validate()
    with pytest.raises(ValueError, match="limit must be positive"):
        EnsembleUNet2dConfig(
            scalar_potential_residual_mode="curl",
            scalar_potential_residual_channels=8,
            scalar_potential_residual_limit=0.0).validate()


def test_nonzero_curl_audit_replays_and_is_batch_permutation_equivariant():
    torch.manual_seed(33)
    model = _model("curl").eval()
    assert model.scalar_potential_residual is not None
    with torch.no_grad():
        model.scalar_potential_residual.output.weight.normal_(std=0.03)
        model.scalar_potential_residual.output.bias.fill_(0.1)
    x = torch.randn(3, 20, 16, 32, 3)
    first = model.scalar_potential_residual_audit(x)
    second = model.scalar_potential_residual_audit(x)
    for key in first:
        assert torch.equal(first[key], second[key])
    permutation = torch.tensor([2, 0, 1])
    permuted = model.scalar_potential_residual_audit(x[permutation])
    inverse = torch.argsort(permutation)
    for key in first:
        assert torch.equal(first[key], permuted[key][inverse])
