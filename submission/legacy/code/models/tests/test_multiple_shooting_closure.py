import torch

from models.multiple_shooting_closure import (
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
    PhysicalClosureCell,
)


def tiny_config() -> MultipleShootingClosureConfig:
    return MultipleShootingClosureConfig(
        e0_channels=8, e1_channels=16, e2_channels=24,
        e0_blocks=1, e1_blocks=1, e2_blocks=1,
        context_channels=8, cell_channels=16, cell_blocks=1,
        step_embedding_channels=8)


def paired_models():
    torch.manual_seed(210)
    control = MultipleShootingClosure(tiny_config(), "reset_anchor")
    candidate = MultipleShootingClosure(tiny_config(), "carry_detached")
    candidate.load_state_dict(control.state_dict())
    return control, candidate


def paired_gradient_models():
    torch.manual_seed(210)
    truncated = MultipleShootingClosure(
        tiny_config(), "carry_detached", "truncated")
    connected = MultipleShootingClosure(
        tiny_config(), "carry_detached", "connected")
    connected.load_state_dict(truncated.state_dict(), strict=True)
    return truncated, connected


def test_update_zero_is_identical_persistence_and_pressure_invariant():
    control, candidate = paired_models()
    history = torch.randn(1, 20, 16, 16, 3)
    expected = history[:, -1:, ..., :2].expand(-1, 20, -1, -1, -1)
    control_prediction = control(history)
    candidate_prediction = candidate(history)
    assert torch.equal(control_prediction, candidate_prediction)
    assert torch.equal(candidate_prediction[..., :2], expected)
    assert torch.count_nonzero(candidate_prediction[..., 2]) == 0
    changed_pressure = history.clone()
    changed_pressure[..., 2] = torch.randn_like(changed_pressure[..., 2])
    assert torch.equal(candidate(history), candidate(changed_pressure))


def test_masked_flux_divergence_telescopes_exactly():
    flux = 0.025 * torch.tanh(torch.randn(2, 4, 12, 20))
    free = torch.ones(2, 1, 12, 20)
    free[:, :, 3:7, 8:11] = 0
    divergence = PhysicalClosureCell.telescoping_flux_divergence(flux, free)
    assert divergence.sum(dim=(-2, -1)).abs().max() <= 1e-6
    assert torch.count_nonzero(divergence[:, :, 3:7, 8:11]) == 0


def test_only_cross_segment_carry_changes_rollout_and_override_matches():
    control, candidate = paired_models()
    with torch.no_grad():
        control.cell.source_head.bias.fill_(0.05)
        candidate.cell.source_head.bias.fill_(0.05)
    history = torch.randn(1, 20, 16, 16, 3)
    control_prediction = control(history)
    candidate_prediction = candidate(history)
    assert torch.equal(control_prediction[:, :5], candidate_prediction[:, :5])
    assert not torch.equal(control_prediction[:, 5:], candidate_prediction[:, 5:])
    candidate.set_reset_override(True)
    assert torch.equal(control_prediction, candidate(history))


def test_update_zero_closure_heads_receive_finite_gradient():
    _control, candidate = paired_models()
    history = torch.randn(1, 20, 16, 16, 3)
    candidate(history)[..., :2].square().mean().backward()
    names = (
        "cell.displacement_head.weight",
        "cell.flux_head.weight",
        "cell.source_head.weight",
    )
    parameters = dict(candidate.named_parameters())
    for name in names:
        gradient = parameters[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_solid_mask_is_stable_after_normalizer_round_trip():
    _control, candidate = paired_models()
    mean_input = torch.tensor((2.0, -1.0, 0.0))
    std_input = torch.tensor((0.5, 2.0, 1.0))
    candidate.set_normalizer_statistics(
        mean_input, torch.zeros(3), std_input, torch.ones(3))
    history = torch.randn(1, 20, 16, 16, 3)
    normalized_physical_zero = -mean_input[:2] / std_input[:2]
    history[:, :, 4:8, 5:9, :2] = normalized_physical_zero
    components = candidate.forward_components(history)
    assert components["solid_mask"][:, 4:8, 5:9].all()
    assert components["flux_delta_global_sum"].abs().max() <= 1e-6


def test_connected_credit_preserves_state_dict_and_forward_bitwise():
    truncated, connected = paired_gradient_models()
    assert MultipleShootingClosure(tiny_config()).segment_gradient_mode == (
        "truncated")
    truncated_state = truncated.state_dict()
    connected_state = connected.state_dict()
    assert truncated_state.keys() == connected_state.keys()
    assert all(torch.equal(truncated_state[name], connected_state[name])
               for name in truncated_state)

    history = torch.randn(1, 20, 16, 16, 3)
    truncated_components = truncated.forward_components(history)
    connected_components = connected.forward_components(history)
    assert truncated_components.keys() == connected_components.keys()
    for name in truncated_components:
        assert torch.equal(
            truncated_components[name], connected_components[name]), name


def _last_five_gradients_at_segment_ends(
        model: MultipleShootingClosure, history: torch.Tensor,
        ) -> list[torch.Tensor | None]:
    step_outputs: list[torch.Tensor] = []

    def capture_step_output(_module, _inputs, output) -> None:
        step_outputs.append(output[0])

    handle = model.cell.register_forward_hook(capture_step_output)
    try:
        prediction = model(history)
    finally:
        handle.remove()
    assert len(step_outputs) == 20
    last_five_loss = prediction[:, -5:, ..., :2].square().mean()
    return list(torch.autograd.grad(
        last_five_loss,
        (step_outputs[4], step_outputs[9], step_outputs[14]),
        allow_unused=True))


def test_connected_credit_is_the_only_cross_segment_gradient_change():
    truncated, connected = paired_gradient_models()
    history = torch.randn(1, 20, 16, 16, 3)
    truncated_gradients = _last_five_gradients_at_segment_ends(
        truncated, history)
    connected_gradients = _last_five_gradients_at_segment_ends(
        connected, history)

    assert all(gradient is None or torch.count_nonzero(gradient) == 0
               for gradient in truncated_gradients)
    for gradient in connected_gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
