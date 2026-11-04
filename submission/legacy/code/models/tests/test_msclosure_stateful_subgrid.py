from __future__ import annotations

import pytest
import torch

from models.msclosure_stateful_subgrid import (
    StatefulSubgridMultipleShootingClosure,
    StatefulSubgridMultipleShootingConfig,
)
from models.multiple_shooting_closure import MultipleShootingClosure


def _tiny_config(**overrides) -> StatefulSubgridMultipleShootingConfig:
    values = {
        "e0_channels": 8,
        "e1_channels": 16,
        "e2_channels": 24,
        "e0_blocks": 1,
        "e1_blocks": 1,
        "e2_blocks": 1,
        "context_channels": 8,
        "cell_channels": 16,
        "cell_blocks": 1,
        "step_embedding_channels": 8,
        "subgrid_channels": 12,
    }
    values.update(overrides)
    return StatefulSubgridMultipleShootingConfig(**values)


def _matched_models():
    config = _tiny_config()
    torch.manual_seed(73)
    base = MultipleShootingClosure(config)
    with torch.no_grad():
        base.cell.displacement_head.weight.normal_(std=0.01)
        base.cell.flux_head.weight.normal_(std=0.01)
        base.cell.source_head.weight.normal_(std=0.01)
    torch.manual_seed(191)
    candidate = StatefulSubgridMultipleShootingClosure(config)
    incompatible = candidate.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        "cell.subgrid_context_projection.weight",
        "subgrid_dynamics.history_projection.weight",
        "subgrid_dynamics.history_projection.bias",
        "subgrid_dynamics.update_projection.weight",
        "subgrid_dynamics.update_projection.bias",
    }
    return base.eval(), candidate.eval()


def _activate_physical_heads(model):
    with torch.no_grad():
        model.cell.source_head.weight.normal_(std=0.04)
        model.cell.flux_head.weight.normal_(std=0.02)


def _activate_subgrid_path(model):
    _activate_physical_heads(model)
    with torch.no_grad():
        model.cell.subgrid_context_projection.weight.normal_(std=0.08)


def test_zero_initialized_conditioner_is_exact_base_compatibility():
    base, candidate = _matched_models()
    history = torch.randn(2, 20, 16, 16, 3)
    history[:, :, 5:8, 6:10, :2] = 0.0
    with torch.no_grad():
        expected = base(history)
        values = candidate.forward_components(history)
    assert torch.equal(values["prediction"], expected)
    assert torch.count_nonzero(
        values["subgrid_condition_abs_max_by_step"]) == 0

    changed_pressure = history.clone()
    changed_pressure[..., 2] = torch.randn_like(changed_pressure[..., 2])
    with torch.no_grad():
        assert torch.equal(candidate(changed_pressure), expected)


def test_subgrid_state_is_bounded_finite_masked_and_carried():
    model = StatefulSubgridMultipleShootingClosure(_tiny_config()).eval()
    _activate_subgrid_path(model)
    history = torch.randn(2, 20, 16, 16, 3)
    history[:, :, 4:8, 5:10, :2] = 0.0
    with torch.no_grad():
        values = model.forward_components(history)

    after = values["subgrid_state_after_update"]
    assert after.shape == (2, 20, 12, 16, 16)
    assert values["subgrid_segment_starts"].shape == (2, 4, 12, 16, 16)
    assert values["subgrid_segment_ends"].shape == (2, 4, 12, 16, 16)
    assert bool(values["subgrid_all_finite"])
    assert float(values["subgrid_state_abs_max"]) <= 1.0
    assert float(values["subgrid_geometry_abs_max"]) == 0.0

    geometry = values["free_mask"].logical_not().permute(
        0, 3, 1, 2).unsqueeze(1)
    assert torch.count_nonzero(after * geometry) == 0
    assert torch.count_nonzero(
        values["subgrid_condition_geometry_abs_max_by_step"]) == 0
    anchor = history[:, -1, ..., :2]
    physical_geometry = geometry.permute(0, 1, 3, 4, 2).expand(
        -1, model.config.out_steps, -1, -1, 2)
    expected_anchor = anchor.unsqueeze(1).expand_as(values["physical_uv"])
    assert torch.equal(
        values["physical_uv"][physical_geometry],
        expected_anchor[physical_geometry])
    assert torch.equal(
        values["subgrid_segment_starts"][:, 1:],
        values["subgrid_segment_ends"][:, :-1])
    assert torch.equal(
        values["segment_starts"][:, 1:], values["segment_ends"][:, :-1])
    assert values["flux_delta_global_sum"].abs().max() <= 1e-6


def test_same_checkpoint_static_and_reset_diagnostics_change_only_state_flow():
    torch.manual_seed(79)
    model = StatefulSubgridMultipleShootingClosure(_tiny_config()).eval()
    _activate_subgrid_path(model)
    checkpoint = {
        name: value.detach().clone() for name, value in model.state_dict().items()}
    history = torch.randn(1, 20, 16, 16, 3)

    model.set_subgrid_override("recurrent")
    recurrent = model.forward_components(history)
    model.set_subgrid_override("static")
    static = model.forward_components(history)
    model.set_subgrid_override("reset")
    reset = model.forward_components(history)

    assert torch.equal(
        recurrent["prediction"][:, :5], reset["prediction"][:, :5])
    assert not torch.equal(
        recurrent["prediction"][:, 5:], reset["prediction"][:, 5:])
    assert not torch.equal(recurrent["prediction"], static["prediction"])
    assert torch.equal(
        recurrent["subgrid_segment_starts"][:, 1:],
        recurrent["subgrid_segment_ends"][:, :-1])
    initial = reset["subgrid_initial_state"].unsqueeze(1).expand(
        -1, model.config.segments - 1, -1, -1, -1)
    assert torch.equal(reset["subgrid_segment_starts"][:, 1:], initial)
    static_initial = static["subgrid_initial_state"].unsqueeze(1).expand_as(
        static["subgrid_state_after_update"])
    assert torch.equal(static["subgrid_state_after_update"], static_initial)

    saved_update = model.subgrid_dynamics.update_projection.weight.detach().clone()
    with torch.no_grad():
        model.subgrid_dynamics.update_projection.weight.normal_(std=2.0)
    model.set_subgrid_override("static")
    assert torch.equal(model(history), static["prediction"])
    with torch.no_grad():
        model.subgrid_dynamics.update_projection.weight.copy_(saved_update)
    for values in (recurrent, static, reset):
        assert torch.equal(
            values["segment_starts"][:, 1:],
            values["segment_ends"][:, :-1])
    assert all(torch.equal(model.state_dict()[name], value)
               for name, value in checkpoint.items())

    model.set_subgrid_override(None)
    assert model._subgrid_override == "recurrent"
    assert torch.equal(model(history), recurrent["prediction"])
    with pytest.raises(ValueError, match="unknown subgrid override"):
        model.set_subgrid_override("unknown")
    with pytest.raises(ValueError, match="cannot reset physical carry"):
        model.set_reset_override(True)


def test_latent_and_physical_graphs_detach_at_the_same_boundaries():
    torch.manual_seed(83)
    model = StatefulSubgridMultipleShootingClosure(_tiny_config())
    _activate_subgrid_path(model)
    prior_h = []
    physical_inputs = []

    state_hook = model.subgrid_dynamics.register_forward_pre_hook(
        lambda _module, inputs: prior_h.append(inputs[2]))
    cell_hook = model.cell.register_forward_pre_hook(
        lambda _module, inputs: physical_inputs.append(inputs[0]))
    history = torch.randn(1, 20, 16, 16, 3)
    prediction = model(history)
    state_hook.remove()
    cell_hook.remove()

    assert len(prior_h) == len(physical_inputs) == 20
    assert prior_h[0].requires_grad
    for boundary in (5, 10, 15):
        assert not prior_h[boundary].requires_grad
        assert not physical_inputs[boundary].requires_grad
    assert prior_h[14].requires_grad and prior_h[16].requires_grad
    assert physical_inputs[14].requires_grad
    assert physical_inputs[16].requires_grad

    loss = prediction[:, -1, ..., :2].square().mean()
    early_h, late_h, early_uv, late_uv = torch.autograd.grad(
        loss,
        (prior_h[14], prior_h[16], physical_inputs[14], physical_inputs[16]),
        allow_unused=True)
    # Stacking all twenty outputs leaves a formal zero-gradient path from the
    # selected last frame to earlier frames; detach is therefore witnessed by
    # an exactly zero gradient rather than necessarily by ``None``.
    for gradient in (early_h, early_uv):
        assert gradient is None or torch.count_nonzero(gradient) == 0
    for gradient in (late_h, late_uv):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_state_update_uses_current_target_normalized_velocity():
    model = StatefulSubgridMultipleShootingClosure(_tiny_config()).eval()
    mean_input = torch.tensor((2.0, -1.0, 0.7))
    std_input = torch.tensor((0.5, 2.0, 1.3))
    mean_target = torch.tensor((-0.4, 0.6, -0.2))
    std_target = torch.tensor((1.7, 0.8, 1.1))
    model.set_normalizer_statistics(
        mean_input, mean_target, std_input, std_target)
    update_inputs = []
    update_outputs = []
    cell_subgrid_inputs = []
    input_hook = model.subgrid_dynamics.update_projection.register_forward_pre_hook(
        lambda _module, inputs: update_inputs.append(inputs[0].detach()))
    output_hook = model.subgrid_dynamics.register_forward_hook(
        lambda _module, _inputs, output: update_outputs.append(output.detach()))
    cell_hook = model.cell.register_forward_pre_hook(
        lambda _module, inputs: cell_subgrid_inputs.append(inputs[2].detach()))
    history = torch.randn(1, 20, 16, 16, 3)
    with torch.no_grad():
        values = model.forward_components(history)
    input_hook.remove()
    output_hook.remove()
    cell_hook.remove()

    anchor_physical = (
        history[:, -1, ..., :2] * std_input[:2] + mean_input[:2])
    expected = ((anchor_physical - mean_target[:2])
                / std_target[:2]).permute(0, 3, 1, 2)
    torch.testing.assert_close(update_inputs[0][:, :2], expected)
    first_prediction_normalized = (
        (values["physical_uv"][:, 0] - mean_target[:2])
        / std_target[:2]).permute(0, 3, 1, 2)
    torch.testing.assert_close(
        update_inputs[1][:, :2], first_prediction_normalized)
    assert torch.equal(update_outputs[0], cell_subgrid_inputs[0])


def test_zero_conditioner_has_staged_but_live_subgrid_gradients():
    torch.manual_seed(89)
    model = StatefulSubgridMultipleShootingClosure(_tiny_config())
    _activate_physical_heads(model)
    history = torch.randn(1, 20, 16, 16, 3)

    model(history)[..., :2].square().mean().backward()
    conditioner_gradient = model.cell.subgrid_context_projection.weight.grad
    assert conditioner_gradient is not None
    assert torch.isfinite(conditioner_gradient).all()
    assert torch.count_nonzero(conditioner_gradient) > 0
    for parameter in (
            model.subgrid_dynamics.history_projection.weight,
            model.subgrid_dynamics.update_projection.weight):
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.cell.subgrid_context_projection.weight.normal_(std=0.08)
    model(history)[..., :2].square().mean().backward()
    for parameter in (
            model.subgrid_dynamics.history_projection.weight,
            model.subgrid_dynamics.update_projection.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_checkpoint_roundtrip_is_strict_and_does_not_persist_diagnostic_mode(
        tmp_path):
    torch.manual_seed(97)
    config = _tiny_config()
    model = StatefulSubgridMultipleShootingClosure(config).eval()
    _activate_subgrid_path(model)
    history = torch.randn(1, 20, 16, 16, 3)
    model.set_subgrid_override("recurrent")
    expected = model(history)
    model.set_subgrid_override("reset")
    checkpoint = tmp_path / "stateful_subgrid.pt"
    torch.save(model.state_dict(), checkpoint)

    rebuilt = StatefulSubgridMultipleShootingClosure(config).eval()
    rebuilt.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True),
        strict=True)
    assert rebuilt._subgrid_override == "recurrent"
    assert torch.equal(rebuilt(history), expected)


def test_stateful_subgrid_default_is_small_without_an_arbitrary_width_gate():
    assert _tiny_config().subgrid_channels == 12
    for channels in (8, 12, 16):
        _tiny_config(subgrid_channels=channels).validate()
    _tiny_config(subgrid_channels=24).validate()
    with pytest.raises(ValueError, match="must be positive"):
        _tiny_config(subgrid_channels=0).validate()
