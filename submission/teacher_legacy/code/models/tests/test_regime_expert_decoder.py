from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.ensemble_unet2d import (
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
    REGIME_EXPERT_DECODER_MODES,
)


def tiny_config(**overrides) -> EnsembleUNet2dConfig:
    values = {
        "base": 8,
        "depth": 3,
        "groups": 4,
        "difference_steps": 2,
        "ensemble_members": 3,
        "uq_hidden_channels": 8,
        "mean_profile_channels": 8,
        "multi_scale_regime_film": True,
        "residual_quantile_intervals": True,
    }
    values.update(overrides)
    return EnsembleUNet2dConfig(**values)


def upgrade(base: EnsembleUNet2d, mode: str) -> EnsembleUNet2d:
    candidate = EnsembleUNet2d(replace(
        base.config, dual_decoder_mode=mode))
    incompatible = candidate.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(name.startswith((
        "fluctuation_up.",
        "fluctuation_decoder.",
        "fluctuation_decoder_regime_film.",
        "regime_expert_router.",
        "regime_expert_router_mean",
        "regime_expert_router_std",
        "regime_expert_mean_profile_refiner.",
    )) for name in incompatible.missing_keys)
    candidate.copy_primary_into_regime_expert_()
    return candidate


def test_regime_experts_reject_deformable_memory():
    with pytest.raises(ValueError, match="deformable"):
        tiny_config(
            depth=4,
            deformable_history_memory_mode="aligned",
            dual_decoder_mode=REGIME_EXPERT_DECODER_MODES[0],
        ).validate()


def test_fixed_and_router_modes_have_identical_state_shapes_and_parameters():
    models = [EnsembleUNet2d(tiny_config(dual_decoder_mode=mode))
              for mode in REGIME_EXPERT_DECODER_MODES]
    names_and_shapes = [
        [(name, tuple(value.shape)) for name, value in model.state_dict().items()]
        for model in models
    ]
    assert names_and_shapes[0] == names_and_shapes[1]
    assert sum(p.numel() for p in models[0].parameters()) == sum(
        p.numel() for p in models[1].parameters())
    assert sum(p.numel() for p in models[0].regime_expert_router.parameters()) == (
        80 * 16 + 16 + 16 + 1)


def test_single_checkpoint_upgrade_is_bitwise_identity_for_point_and_uq():
    torch.manual_seed(12)
    base = EnsembleUNet2d(tiny_config()).eval()
    fixed = upgrade(base, "regime_expert_fixed_average").eval()
    routed = upgrade(base, "regime_expert_history_router").eval()
    # A non-constant frozen router must still preserve the common function
    # while the two decoder/refiner copies are bitwise equal.
    torch.manual_seed(13)
    router_state = {
        name: torch.randn_like(value)
        for name, value in fixed.regime_expert_router.state_dict().items()
    }
    for model in (fixed, routed):
        model.regime_expert_router.load_state_dict(router_state, strict=True)
        model.set_regime_expert_router_standardization(
            torch.randn(80), torch.rand(80) + 0.25)

    x = torch.randn(2, 20, 8, 16, 3)
    with torch.inference_mode():
        expected = base.forward_uq_training(x)[:5]
        fixed_values = fixed.forward_uq_training(x)[:5]
        routed_values = routed.forward_uq_training(x)[:5]
    for reference, fixed_value, routed_value in zip(
            expected, fixed_values, routed_values):
        assert torch.equal(reference, fixed_value)
        assert torch.equal(reference, routed_value)


def test_router_reads_only_detached_history_and_fixed_arm_ignores_its_value():
    torch.manual_seed(21)
    base = EnsembleUNet2d(tiny_config())
    fixed = upgrade(base, "regime_expert_fixed_average").eval()
    routed = upgrade(base, "regime_expert_history_router").eval()
    routed.load_state_dict(fixed.state_dict(), strict=True)
    # Break expert symmetry so point routing has an observable effect.
    with torch.no_grad():
        fixed.regime_expert_mean_profile_refiner[-1].bias.add_(0.25)
        routed.regime_expert_mean_profile_refiner[-1].bias.add_(0.25)
        fixed.regime_expert_router[-1].bias.fill_(4.0)
        routed.regime_expert_router[-1].bias.fill_(4.0)
    x = torch.randn(2, 20, 8, 16, 3, requires_grad=True)
    fixed_prediction = fixed(x)
    routed_prediction = routed(x)
    assert not torch.equal(fixed_prediction, routed_prediction)
    fixed_prediction.square().mean().backward()
    fixed_router_grad = fixed.regime_expert_router[-1].bias.grad
    assert fixed_router_grad is not None
    assert torch.count_nonzero(fixed_router_grad) == 0
    assert x.grad is not None
    # The descriptor is detached: a router-only loss cannot update history.
    x.grad.zero_()
    fixed.zero_grad(set_to_none=True)
    fixed.regime_expert_router_logits(x).square().mean().backward()
    assert torch.count_nonzero(x.grad) == 0
    assert fixed.regime_expert_router[-1].bias.grad is not None


def test_both_modes_execute_two_decoders_two_refiners_and_router():
    torch.manual_seed(31)
    base = EnsembleUNet2d(tiny_config()).eval()
    counts = []
    for mode in REGIME_EXPERT_DECODER_MODES:
        model = upgrade(base, mode).eval()
        observed = {"primary": 0, "expert": 0, "mean0": 0,
                    "mean1": 0, "router": 0}
        hooks = [
            model.decoder[-1].register_forward_hook(
                lambda *_args: observed.__setitem__(
                    "primary", observed["primary"] + 1)),
            model.fluctuation_decoder[-1].register_forward_hook(
                lambda *_args: observed.__setitem__(
                    "expert", observed["expert"] + 1)),
            model.mean_profile_refiner.register_forward_hook(
                lambda *_args: observed.__setitem__(
                    "mean0", observed["mean0"] + 1)),
            model.regime_expert_mean_profile_refiner.register_forward_hook(
                lambda *_args: observed.__setitem__(
                    "mean1", observed["mean1"] + 1)),
            model.regime_expert_router.register_forward_hook(
                lambda *_args: observed.__setitem__(
                    "router", observed["router"] + 1)),
        ]
        with torch.inference_mode():
            model(torch.randn(1, 20, 8, 16, 3))
        for hook in hooks:
            hook.remove()
        counts.append(observed)
    assert counts[0] == counts[1] == {
        "primary": 1, "expert": 1, "mean0": 1, "mean1": 1,
        # Once for decoder blend and once for routed mean correction.
        "router": 1,
    }
