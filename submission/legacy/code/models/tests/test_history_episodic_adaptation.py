from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.history_episodic_adaptation import (  # noqa: E402
    HistoryEpisodicAdaptationConfig,
    HistoryEpisodicForecaster,
    normalized_fixed_radius_update,
    one_step_fast_adaptation,
    per_sample_fast_gradients,
)
from models.temporal_transport import TemporalTransportOperator  # noqa: E402


def config(**overrides) -> HistoryEpisodicAdaptationConfig:
    values = dict(
        max_context_steps=20,
        max_output_steps=20,
        allowed_context_steps=(12, 16, 20),
        allowed_output_steps=(4, 20),
        in_channels=3,
        out_channels=3,
        hidden_channels=4,
        spatial_blocks=1,
        spatial_downsample=2,
        temporal_blocks=1,
        temporal_kernel=3,
        groups=2,
        max_displacement=2.0,
    )
    values.update(overrides)
    return HistoryEpisodicAdaptationConfig(**values)


def test_variable_context_and_horizon_shapes():
    torch.manual_seed(11)
    model = HistoryEpisodicForecaster(config()).eval()

    for context_steps in (12, 16, 20):
        history = torch.randn(2, context_steps, 5, 7, 3)
        for output_steps in (4, 20):
            result = model.forward_components(
                history, output_steps=output_steps)
            expected = (2, output_steps, 5, 7, 3)
            assert result.prediction.shape == expected
            assert result.transported.shape == expected
            assert result.fluctuation.shape == expected
            assert result.mean_delta.shape == (2, 5, 7, 3)
            assert result.displacement.shape == (
                2, output_steps, 5, 7, 2)
            assert result.attention.shape == (
                2, output_steps, context_steps, 3, 4)
            torch.testing.assert_close(
                result.attention.sum(dim=2),
                torch.ones_like(result.attention[:, :, 0]),
                rtol=1.0e-6,
                atol=1.0e-6,
            )
            torch.testing.assert_close(
                result.fluctuation.mean(dim=1),
                torch.zeros_like(result.fluctuation[:, 0]),
                rtol=0.0,
                atol=2.0e-7,
            )


def test_deterministic_warp_matches_grid_sample_and_has_displacement_gradient():
    torch.manual_seed(12)
    source = torch.randn(2, 5, 7, 3)
    displacement = 3.0 * torch.randn(2, 4, 5, 7, 2)
    expected = TemporalTransportOperator.warp(source, displacement)
    actual = HistoryEpisodicForecaster._warp(source, displacement)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)

    differentiable_displacement = displacement.clone().requires_grad_(True)
    output = HistoryEpisodicForecaster._warp(
        source, differentiable_displacement)
    gradient, = torch.autograd.grad(output.square().mean(),
                                    differentiable_displacement)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_fresh_fast_state_is_functional_and_forward_is_deterministic():
    torch.manual_seed(13)
    model = HistoryEpisodicForecaster(config()).eval()
    history = torch.randn(2, 20, 5, 7, 3)

    first_state = model.new_fast_vector(2)
    second_state = model.new_fast_vector(2)
    assert first_state.data_ptr() != second_state.data_ptr()
    assert torch.count_nonzero(first_state) == 0
    assert torch.count_nonzero(second_state) == 0

    first = model(history, output_steps=4)
    second = model(history, output_steps=4)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    supplied = torch.randn(2, model.config.hidden_channels)
    original = supplied.clone()
    adapted_output = model(history, supplied, 4)
    torch.testing.assert_close(supplied, original, rtol=0.0, atol=0.0)
    assert not torch.equal(first, adapted_output)


def test_normalized_fixed_radius_update_and_three_arm_semantics():
    fast_state = torch.zeros(4, 2)
    gradient = torch.tensor([
        [3.0, 4.0],
        [0.0, 0.0],
        [-12.0, 5.0],
        [8.0, -15.0],
    ])
    untouched = fast_state.clone()
    updated = normalized_fixed_radius_update(
        fast_state, gradient, radius=0.10, epsilon=1.0e-8)
    torch.testing.assert_close(fast_state, untouched, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(updated[[0, 2, 3]], dim=1),
        torch.full((3,), 0.10),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    torch.testing.assert_close(updated[1], fast_state[1])

    static = one_step_fast_adaptation(
        fast_state, gradient, mode="static", radius=0.10)
    assert static is fast_state
    self_adapted = one_step_fast_adaptation(
        fast_state, gradient, mode="self", radius=0.10)
    torch.testing.assert_close(self_adapted, updated)

    donor_indices = torch.tensor([2, 3, 0, 1], dtype=torch.long)
    donor_adapted = one_step_fast_adaptation(
        fast_state,
        gradient,
        mode="donor",
        radius=0.10,
        donor_indices=donor_indices,
    )
    torch.testing.assert_close(
        donor_adapted,
        normalized_fixed_radius_update(
            fast_state,
            gradient.index_select(0, donor_indices),
            radius=0.10,
        ),
    )
    explicit_donor = one_step_fast_adaptation(
        fast_state,
        gradient,
        mode="donor",
        radius=0.10,
        donor_gradient=gradient.index_select(0, donor_indices),
    )
    torch.testing.assert_close(explicit_donor, donor_adapted)

    with pytest.raises(ValueError, match="exactly one"):
        one_step_fast_adaptation(
            fast_state, gradient, mode="donor", radius=0.10)
    with pytest.raises(ValueError, match="finite"):
        normalized_fixed_radius_update(
            fast_state,
            torch.full_like(gradient, float("nan")),
            radius=0.10,
        )


def test_support_loss_uses_frozen_slices_and_only_uv_channels():
    model = HistoryEpisodicForecaster(config()).eval()
    model.set_normalizer_statistics(
        mean_input=torch.tensor([1.0, 2.0, 3.0]),
        mean_target=torch.tensor([0.5, 1.0, 1.5]),
        std_input=torch.tensor([2.0, 3.0, 4.0]),
        std_target=torch.tensor([1.0, 1.5, 2.0]),
    )
    history = torch.randn(2, 20, 4, 6, 3)
    history[..., 2] = 1.0e4
    fast_state = model.new_fast_vector(2)
    observed_calls: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def zero_forward(
            this: HistoryEpisodicForecaster,
            observed: torch.Tensor,
            state: torch.Tensor | None = None,
            steps: int | None = None,
            *,
            fast_state: torch.Tensor | None = None,
            output_steps: int | None = None,
            ) -> torch.Tensor:
        del this
        if state is not None and fast_state is not None:
            raise AssertionError("fast state was passed twice")
        if steps is not None and output_steps is not None:
            raise AssertionError("output steps were passed twice")
        actual_state = fast_state if fast_state is not None else state
        actual_steps = output_steps if output_steps is not None else steps
        assert actual_state is not None
        assert actual_steps is not None
        observed_calls.append((observed.clone(), actual_state, actual_steps))
        return torch.zeros(
            observed.shape[0],
            actual_steps,
            observed.shape[2],
            observed.shape[3],
            3,
            dtype=observed.dtype,
            device=observed.device,
        )

    model.forward = MethodType(zero_forward, model)
    losses = model.support_loss(history, fast_state)
    target = model._history_in_target_coordinates(history)
    expected = 0.5 * (
        target[:, 12:16, ..., :2].square().mean(dim=(1, 2, 3, 4))
        + target[:, 16:20, ..., :2].square().mean(dim=(1, 2, 3, 4))
    )
    torch.testing.assert_close(losses, expected)

    assert len(observed_calls) == 2
    torch.testing.assert_close(observed_calls[0][0], history[:, 0:12])
    torch.testing.assert_close(observed_calls[1][0], history[:, 4:16])
    assert observed_calls[0][1] is fast_state
    assert observed_calls[1][1] is fast_state
    assert observed_calls[0][2] == observed_calls[1][2] == 4


def test_support_gradient_and_adapted_api_are_per_sample_and_finite():
    torch.manual_seed(17)
    model = HistoryEpisodicForecaster(config()).eval()
    history = torch.randn(2, 20, 4, 6, 3)
    fast_state = model.new_fast_vector(2, requires_grad=True)

    losses = model.support_loss(history, fast_state)
    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()
    gradient = per_sample_fast_gradients(losses, fast_state)
    assert gradient.shape == fast_state.shape
    assert torch.isfinite(gradient).all()
    assert torch.all(torch.linalg.vector_norm(gradient, dim=1) > 0.0)

    self_state, self_diagnostics = model.adapted_fast_state(history)
    static_state, static_diagnostics = model.adapted_fast_state(
        history, mode="static")
    donor_state, donor_diagnostics = model.adapted_fast_state(
        history,
        mode="donor",
        donor_indices=torch.tensor([1, 0]),
    )
    assert self_state.shape == static_state.shape == donor_state.shape == (
        2, model.config.hidden_channels)
    assert self_diagnostics["mode"] == "self"
    assert static_diagnostics["mode"] == "static"
    assert donor_diagnostics["mode"] == "donor"
    torch.testing.assert_close(
        self_diagnostics["step_norm"],
        torch.full((2,), model.config.fast_radius),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        static_diagnostics["step_norm"], torch.zeros(2))
    torch.testing.assert_close(static_state, torch.zeros_like(static_state))
    torch.testing.assert_close(
        donor_state, self_state.index_select(0, torch.tensor([1, 0])),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_batch_permutation_equivariance_and_no_persistent_episode_state():
    torch.manual_seed(19)
    model = HistoryEpisodicForecaster(config()).eval()
    history = torch.randn(3, 20, 4, 6, 3)
    fast_state = torch.randn(3, model.config.hidden_channels)
    permutation = torch.tensor([2, 0, 1])
    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    prediction = model(history, fast_state, 4)
    permuted_prediction = model(
        history.index_select(0, permutation),
        fast_state.index_select(0, permutation),
        4,
    )
    torch.testing.assert_close(
        permuted_prediction,
        prediction.index_select(0, permutation),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    adapted, _ = model.adapted_fast_state(history)
    permuted_adapted, _ = model.adapted_fast_state(
        history.index_select(0, permutation))
    torch.testing.assert_close(
        permuted_adapted,
        adapted.index_select(0, permutation),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    repeated, _ = model.adapted_fast_state(history)
    torch.testing.assert_close(repeated, adapted, rtol=0.0, atol=0.0)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            value, state_before[name], rtol=0.0, atol=0.0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_architecture_roundtrip_defaults_and_input_validation():
    model = HistoryEpisodicForecaster(config())
    stored = model.architecture_config()
    assert stored.pop("architecture") == "history_episodic_adaptation"
    assert stored["fast_epsilon"] == 1.0e-8
    # Exercise the JSON representation of tuple-valued configuration fields.
    stored["allowed_context_steps"] = list(stored["allowed_context_steps"])
    stored["allowed_output_steps"] = list(stored["allowed_output_steps"])
    rebuilt = HistoryEpisodicForecaster(
        HistoryEpisodicAdaptationConfig(**stored))
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert rebuilt.config == model.config

    with pytest.raises(ValueError, match="expected history"):
        model(torch.randn(1, 13, 4, 6, 3))
    with pytest.raises(ValueError, match="output_steps"):
        model(torch.randn(1, 20, 4, 6, 3), output_steps=5)
    with pytest.raises(ValueError, match="fast_state must have shape"):
        model(
            torch.randn(2, 20, 4, 6, 3),
            torch.zeros(1, model.config.hidden_channels),
            4,
        )
    with pytest.raises(ValueError, match="complete max-context"):
        model.support_loss(
            torch.randn(1, 16, 4, 6, 3),
            model.new_fast_vector(1),
        )
