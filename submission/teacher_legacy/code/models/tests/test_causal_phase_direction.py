from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
    FrozenB210CausalPhase,
    POOLED_HISTORY_MODE,
)


def tiny_config(**overrides: object) -> CausalPhaseDirectionConfig:
    values: dict[str, object] = {
        "in_steps": 5,
        "out_steps": 5,
        "in_channels": 3,
        "out_channels": 3,
        "hidden_channels": 8,
        "spatial_blocks": 1,
        "spatial_downsample": 2,
        "temporal_blocks": 2,
        "temporal_kernel": 3,
        "groups": 4,
    }
    values.update(overrides)
    return CausalPhaseDirectionConfig(**values)


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(311)
    history = torch.randn(2, 5, 8, 10, 3)
    base = torch.randn(2, 5, 8, 10, 3)
    # Exercise the zero-fluctuation fallback at one pixel/channel too.
    base[:, :, 0, 0, 0] = 2.25
    return history, base


def activate_head(model: CausalPhaseDirectionAdapter) -> None:
    with torch.no_grad():
        torch.manual_seed(719)
        model.direction_head.weight.normal_(std=0.08)
        model.direction_head.bias.normal_(std=0.02)


def temporal_geometry(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.float()
    mean = value.mean(dim=1, keepdim=True)
    rms = (value - mean).square().mean(dim=1, keepdim=True).sqrt()
    return mean, rms


def test_update_zero_is_bitwise_identity_with_live_head_gradient() -> None:
    history, base = inputs()
    model = CausalPhaseDirectionAdapter(tiny_config()).train()
    output = model(history, base)
    assert torch.equal(output, base)

    loss = (output[..., :2] - torch.randn_like(output[..., :2])).square().mean()
    loss.backward()
    gradient = model.direction_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_projection_preserves_temporal_mean_rms_and_pressure() -> None:
    history, base = inputs()
    model = CausalPhaseDirectionAdapter(tiny_config()).eval()
    activate_head(model)
    output = model(history, base)
    assert not torch.equal(output[..., :2], base[..., :2])
    assert torch.equal(output[..., 2:], base[..., 2:])
    components = model.forward_components(history, base)
    assert components.direction_gate.shape == base[..., :1].shape
    assert bool((components.direction_gate > 0).all())
    assert bool((components.direction_gate < 1).all())

    base_mean, base_rms = temporal_geometry(base[..., :2])
    output_mean, output_rms = temporal_geometry(output[..., :2])
    torch.testing.assert_close(output_mean, base_mean, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(output_rms, base_rms, atol=2.0e-6, rtol=2.0e-6)
    # A truly constant source trajectory has no phase degree of freedom.
    assert torch.equal(output[:, :, 0, 0, 0], base[:, :, 0, 0, 0])


def test_gradient_reaches_both_history_and_forecast_paths_after_head_opens() -> None:
    history, base = inputs()
    history.requires_grad_(True)
    model = CausalPhaseDirectionAdapter(tiny_config()).train()
    activate_head(model)
    target = torch.randn_like(base)
    loss = (model(history, base)[..., :2] - target[..., :2]).square().mean()
    loss.backward()

    parameters = (
        model.frame_encoder.stem.weight,
        model.temporal_mixer.blocks[0].depthwise.weight,
        model.forecast_encoder.stem.weight,
        model.fusion[0].weight,
        model.direction_head.weight,
        model.gate_head.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert history.grad is not None
    by_time = history.grad.abs().sum(dim=(0, 2, 3, 4))
    assert bool((by_time > 0).all())
    early_to_late = by_time[:len(by_time) // 2].sum() / by_time[
        len(by_time) // 2:].sum()
    # This is a structural anti-degeneracy check, not a score gate: early
    # frames must retain a material direct path instead of only 1e-3 leakage.
    assert early_to_late > 0.1


def test_output_is_sensitive_to_early_observed_history() -> None:
    history, base = inputs()
    model = CausalPhaseDirectionAdapter(tiny_config()).eval()
    activate_head(model)
    changed = history.clone()
    changed[:, 0] += 0.75
    original_output = model(history, base)
    changed_output = model(changed, base)
    assert (changed_output - original_output).abs().max() > 1.0e-7


def test_temporal_mixer_is_strictly_causal() -> None:
    history, _base = inputs()
    model = CausalPhaseDirectionAdapter(tiny_config()).eval()
    changed = history.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 7.0
    original_states = model.encode_history(history)
    changed_states = model.encode_history(changed)
    # States 0..2 cannot depend on observations 3..4.
    assert torch.equal(original_states[:, :3], changed_states[:, :3])
    assert not torch.equal(original_states[:, 3:], changed_states[:, 3:])


def test_matched_spatial_control_has_identical_state_and_parameter_contract() -> None:
    torch.manual_seed(991)
    candidate = CausalPhaseDirectionAdapter(tiny_config())
    torch.manual_seed(991)
    control = CausalPhaseDirectionAdapter(tiny_config(
        mixer_mode="framewise_spatial_control"))
    candidate_state = candidate.state_dict()
    control_state = control.state_dict()
    assert candidate_state.keys() == control_state.keys()
    for name in candidate_state:
        assert candidate_state[name].shape == control_state[name].shape
        assert torch.equal(candidate_state[name], control_state[name]), name
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in control.parameters())


def test_pooled_history_removes_mixer_parameters_and_keeps_frame_path() -> None:
    full = CausalPhaseDirectionAdapter(tiny_config())
    pooled = CausalPhaseDirectionAdapter(tiny_config(
        mixer_mode=POOLED_HISTORY_MODE))
    assert isinstance(pooled.temporal_mixer, nn.Identity)
    assert not any(name.startswith("temporal_mixer.")
                   for name in pooled.state_dict())
    removed = sum(parameter.numel()
                  for parameter in full.temporal_mixer.parameters())
    assert removed > 0
    assert sum(parameter.numel() for parameter in pooled.parameters()) == (
        sum(parameter.numel() for parameter in full.parameters()) - removed)
    assert sum(parameter.numel() for parameter in pooled.frame_encoder.parameters()) == (
        sum(parameter.numel() for parameter in full.frame_encoder.parameters()))


def test_pooled_history_is_time_permutation_invariant_but_uses_history() -> None:
    history, base = inputs()
    model = CausalPhaseDirectionAdapter(tiny_config(
        mixer_mode=POOLED_HISTORY_MODE)).eval()
    assert torch.equal(model.encode_history(history), model.encode_frames(history))
    activate_head(model)
    reference = model(history, base)
    permutation = torch.tensor((3, 0, 4, 1, 2))
    permuted = model(history[:, permutation], base)
    torch.testing.assert_close(permuted, reference, atol=2.0e-6, rtol=2.0e-6)

    changed = history.clone()
    changed[:, :, 2:5, 3:7] += 0.75
    assert not torch.equal(model(changed, base)[..., :2], reference[..., :2])


class ToyB210(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return history * self.scale

    def forward_distribution(
            self, history: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point = self(history)
        return point, torch.full_like(point[..., :2], 0.4), torch.full_like(
            point[..., :2], 0.7)

    def architecture_config(self) -> dict[str, object]:
        return {"architecture": "toy_b210"}


def test_frozen_wrapper_keeps_base_eval_and_preserves_absolute_bounds() -> None:
    history, _base = inputs()
    base = ToyB210()
    adapter = CausalPhaseDirectionAdapter(tiny_config())
    wrapper = FrozenB210CausalPhase(base, adapter).train()
    assert wrapper.training
    assert wrapper.adapter.training
    assert not wrapper.base_model.training
    assert not any(parameter.requires_grad for parameter in base.parameters())

    prediction, lower, upper = wrapper.forward_distribution(history)
    with torch.no_grad():
        base_prediction, expected_lower, expected_upper = (
            base.forward_distribution(history))
    assert torch.equal(prediction, base_prediction)
    assert torch.equal(lower, expected_lower)
    assert torch.equal(upper, expected_upper)

    activate_head(adapter)
    prediction, lower, upper = wrapper.forward_distribution(history)
    base_prediction, base_lower, base_upper = base.forward_distribution(history)
    torch.testing.assert_close(
        prediction[..., :2] - lower,
        base_prediction[..., :2] - base_lower,
        atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(
        prediction[..., :2] + upper,
        base_prediction[..., :2] + base_upper,
        atol=2.0e-6, rtol=2.0e-6)
    assert torch.equal(prediction[..., 2:], base_prediction[..., 2:])


def test_wrapper_can_augment_adapter_history_without_moving_base_geometry() -> None:
    history, _base = inputs()
    base = ToyB210()
    adapter = CausalPhaseDirectionAdapter(tiny_config()).eval()
    activate_head(adapter)
    wrapper = FrozenB210CausalPhase(base, adapter).eval()
    augmented = history.clone()
    augmented[:, 0] += 0.6
    clean_output = wrapper(history, adapter_history=history)
    augmented_output = wrapper(history, adapter_history=augmented)
    assert not torch.equal(clean_output[..., :2], augmented_output[..., :2])
    base_prediction = base(history)
    base_mean, base_rms = temporal_geometry(base_prediction[..., :2])
    for output in (clean_output, augmented_output):
        mean, rms = temporal_geometry(output[..., :2])
        torch.testing.assert_close(mean, base_mean, atol=2.0e-6, rtol=2.0e-6)
        torch.testing.assert_close(rms, base_rms, atol=2.0e-6, rtol=2.0e-6)
