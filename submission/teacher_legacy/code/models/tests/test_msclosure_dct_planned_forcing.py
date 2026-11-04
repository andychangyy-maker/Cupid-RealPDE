import pytest
import torch

from models.msclosure_dct_planned_forcing import (
    DCT_PLAN_FEEDBACK_CHANNELS,
    DCTPlannedForcingMultipleShootingClosure,
    DCTPlannedForcingMultipleShootingConfig,
)
from models.multiple_shooting_closure import (
    MultipleShootingClosure,
)


def _tiny_config() -> DCTPlannedForcingMultipleShootingConfig:
    return DCTPlannedForcingMultipleShootingConfig(
        e0_channels=8, e1_channels=16, e2_channels=24,
        e0_blocks=1, e1_blocks=1, e2_blocks=1,
        context_channels=8, cell_channels=16, cell_blocks=1,
    )


def _matched_models():
    config = _tiny_config()
    torch.manual_seed(41)
    base = MultipleShootingClosure(config)
    with torch.no_grad():
        base.cell.displacement_head.weight.normal_(std=0.01)
        base.cell.flux_head.weight.normal_(std=0.01)
        base.cell.source_head.weight.normal_(std=0.01)
    torch.manual_seed(99)
    candidate = DCTPlannedForcingMultipleShootingClosure(config)
    incompatible = candidate.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        "cell.plan_stem.weight",
        "plan_head.basis",
        "plan_head.stem.0.weight",
        "plan_head.stem.1.weight",
        "plan_head.stem.1.bias",
        "plan_head.blocks.0.norm1.weight",
        "plan_head.blocks.0.norm1.bias",
        "plan_head.blocks.0.depthwise.weight",
        "plan_head.blocks.0.depthwise.bias",
        "plan_head.blocks.0.norm2.weight",
        "plan_head.blocks.0.norm2.bias",
        "plan_head.blocks.0.pointwise.weight",
        "plan_head.blocks.0.pointwise.bias",
        "plan_head.blocks.1.norm1.weight",
        "plan_head.blocks.1.norm1.bias",
        "plan_head.blocks.1.depthwise.weight",
        "plan_head.blocks.1.depthwise.bias",
        "plan_head.blocks.1.norm2.weight",
        "plan_head.blocks.1.norm2.bias",
        "plan_head.blocks.1.pointwise.weight",
        "plan_head.blocks.1.pointwise.bias",
        "plan_head.output.weight",
        "plan_head.output.bias",
    }
    return base.eval(), candidate.eval()


def test_dct_plan_basis_is_zero_mean_orthogonal_and_unit_rms():
    model = DCTPlannedForcingMultipleShootingClosure(_tiny_config())
    basis = model.plan_head.basis
    gram = basis @ basis.T / model.config.out_steps
    torch.testing.assert_close(
        basis.mean(dim=1), torch.zeros(4), atol=1e-7, rtol=0)
    torch.testing.assert_close(
        gram, torch.eye(4), atol=1e-6, rtol=0)


def test_dct_coefficient_reconstruction_projection_roundtrip():
    model = DCTPlannedForcingMultipleShootingClosure(_tiny_config())
    coefficients = torch.randn(2, 2, 4, 7, 9)
    reconstructed = model.plan_head.reconstruct(coefficients)
    recovered = model.plan_head.temporal_coefficients(reconstructed)
    torch.testing.assert_close(
        recovered, coefficients, atol=1e-6, rtol=1e-6)


def test_planned_forcing_rejects_reset_rollout_contract():
    with pytest.raises(ValueError, match="only supports carry_detached"):
        DCTPlannedForcingMultipleShootingClosure(
            _tiny_config(), rollout_mode="reset_anchor")

    model = DCTPlannedForcingMultipleShootingClosure(_tiny_config())
    with pytest.raises(ValueError, match="cannot reset segment carry"):
        model.set_reset_override(True)
    model.set_reset_override(False)
    assert model._reset_override is False


def test_forcing_only_segmented_jacobian_matches_dct_plan_exactly():
    """Detached segment carry must not delete each new segment's plan path."""
    model = DCTPlannedForcingMultipleShootingClosure(_tiny_config())
    coefficients = torch.randn(
        1, 2, 4, 1, 1, dtype=torch.float64, requires_grad=True)
    plan = model.plan_head.reconstruct(coefficients)
    forcing = model._segment_aligned_plan_forcing(plan)

    ordinary_prior = torch.cat((torch.zeros_like(plan[:, :1]), plan[:, :-1]),
                               dim=1)
    assert torch.equal(forcing, plan - ordinary_prior)

    state = torch.zeros_like(plan[:, 0])
    outputs = []
    for segment in range(model.config.segments):
        state = state.detach()
        start = segment * model.config.segment_length
        stop = start + model.config.segment_length
        for horizon in range(start, stop):
            state = state + forcing[:, horizon]
            outputs.append(state)
    output = torch.stack(outputs, dim=1)
    effective_basis = model.plan_head.basis.to(coefficients)
    effective_basis = effective_basis - effective_basis.mean(
        dim=1, keepdim=True)

    for horizon in range(model.config.out_steps):
        gradient, = torch.autograd.grad(
            output[0, horizon, 0, 0, 0], coefficients,
            retain_graph=True)
        expected = torch.zeros_like(coefficients)
        expected[0, 0, :, 0, 0] = effective_basis[:, horizon]
        torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=0)


def test_zero_plan_is_bitwise_identical_to_original_ms_and_pressure_blind():
    base, candidate = _matched_models()
    history = torch.randn(2, 20, 16, 16, 3)
    with torch.no_grad():
        expected = base(history)
        actual = candidate(history)
    assert torch.equal(actual, expected)

    changed_pressure = history.clone()
    changed_pressure[..., 2] = torch.randn_like(changed_pressure[..., 2])
    with torch.no_grad():
        changed = candidate(changed_pressure)
    assert torch.equal(actual, changed)


def test_plan_is_exposed_applied_as_free_forcing_and_carried_by_segment():
    torch.manual_seed(42)
    model = DCTPlannedForcingMultipleShootingClosure(_tiny_config())
    with torch.no_grad():
        model.plan_head.output.bias.copy_(torch.tensor(
            (0.08, -0.06, 0.04, -0.02,
             -0.07, 0.05, -0.03, 0.01)))
    history = torch.randn(2, 20, 16, 16, 3)
    history[:, :, 5:8, 6:10, :2] = 0.0
    values = model.forward_components(history)

    assert values["predicted_coefficients"].shape == (2, 2, 4, 16, 16)
    assert values["predicted_plan_normalized"].shape == (
        2, 20, 16, 16, 2)
    assert values["planned_forcing_normalized"].shape == (
        2, 20, 16, 16, 2)
    assert values["cell_plan_context_normalized"].shape == (
        2, 20, 16, 16, DCT_PLAN_FEEDBACK_CHANNELS)
    torch.testing.assert_close(
        values["predicted_plan_normalized"].mean(dim=1),
        torch.zeros_like(values["predicted_plan_normalized"][:, 0]),
        atol=1e-7, rtol=0)
    torch.testing.assert_close(
        values["planned_forcing_normalized"].cumsum(dim=1),
        values["applied_plan_normalized"], atol=1e-7, rtol=0)

    free = values["free_mask"].unsqueeze(1).expand_as(
        values["planned_forcing_normalized"][..., :1])
    assert torch.count_nonzero(torch.where(
        free, torch.zeros_like(values["planned_forcing_normalized"]),
        values["planned_forcing_normalized"])) == 0
    assert torch.count_nonzero(
        values["planned_forcing_normalized"][free.expand_as(
            values["planned_forcing_normalized"])]) > 0
    torch.testing.assert_close(
        values["planned_forcing_physical"],
        values["planned_forcing_normalized"]
        * model.std_target[:2], atol=0, rtol=0)

    # The base closure heads are zero, so its update is exact persistence.
    # The telescoping plan forcing must therefore be the full free-cell path.
    anchor = history[:, -1, ..., :2]
    expected_physical = (
        anchor.unsqueeze(1)
        + values["applied_plan_normalized"] * model.std_target[:2])
    torch.testing.assert_close(
        values["physical_uv"], expected_physical, atol=1e-6, rtol=0)
    assert torch.equal(
        values["segment_starts"][:, 1:],
        values["segment_ends"][:, :-1].detach())
    assert not values["segment_starts"].requires_grad
    assert torch.count_nonzero(values["prediction"][..., 2]) == 0


def test_parameter_increment_is_only_plan_head_and_four_channel_cell_stem():
    config = _tiny_config()
    base = MultipleShootingClosure(config)
    candidate = DCTPlannedForcingMultipleShootingClosure(config)
    base_parameters = sum(value.numel() for value in base.parameters())
    candidate_parameters = sum(
        value.numel() for value in candidate.parameters())
    plan_parameters = sum(
        value.numel() for value in candidate.plan_head.parameters())
    feedback_parameters = (
        DCT_PLAN_FEEDBACK_CHANNELS * config.cell_channels * 3 * 3)
    assert candidate_parameters - base_parameters == (
        plan_parameters + feedback_parameters)
    assert candidate.architecture_config()["history_only_plan"] is True
    assert candidate.architecture_config()["truth_enters_rollout"] is False


def test_output_and_direct_coefficient_losses_reach_joint_plan_paths():
    torch.manual_seed(43)
    model = DCTPlannedForcingMultipleShootingClosure(_tiny_config())
    with torch.no_grad():
        model.plan_head.output.weight.normal_(std=0.005)
        model.plan_head.output.bias.fill_(0.02)
        model.cell.source_head.weight.normal_(std=0.02)
    values = model.forward_components(torch.randn(2, 20, 16, 16, 3))
    output_loss = values["prediction"][:, -1, ..., :2].square().mean()
    coefficient_loss = values["predicted_coefficients"].square().mean()
    (output_loss + coefficient_loss).backward()

    gradients = (
        model.plan_head.output.weight.grad,
        model.plan_head.stem[0].weight.grad,
        model.cell.plan_stem.weight.grad,
    )
    for gradient in gradients:
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert torch.count_nonzero(gradient) > 0
