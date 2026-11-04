from __future__ import annotations

import sys
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
    HistoryConditionedOperatorBasis,
)


def _common() -> dict[str, object]:
    return {
        "base": 8,
        "depth": 4,
        "groups": 4,
        "difference_steps": 2,
        "ensemble_members": 3,
        "uq_hidden_channels": 8,
        "mean_fluctuation": True,
        "cross_conditioned_backbone": True,
        "cross_conditioned_channels": 4,
    }


def _normalizer(model: EnsembleUNet2d) -> None:
    model.set_normalizer_statistics(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3))


def _activate_hcobf(
        model: EnsembleUNet2d, *, activate_projection: bool = True) -> None:
    with torch.no_grad():
        for heads in model.member_point_heads:
            for head in heads:
                channels = model.config.out_channels
                head.weight[channels:].normal_(std=0.03)
                head.bias[channels:].normal_(std=0.03)
        if activate_projection:
            for basis in model.history_operator_basis:
                basis.output_projection.weight.normal_(std=0.02)


def test_hcobf_update_zero_is_legacy_bitwise_and_strict_loads_roundtrip():
    torch.manual_seed(210)
    legacy = EnsembleUNet2d(EnsembleUNet2dConfig(**_common())).eval()
    torch.manual_seed(210)
    candidate = EnsembleUNet2d(EnsembleUNet2dConfig(
        **_common(), history_conditioned_operator_basis=True)).eval()
    _normalizer(legacy)
    _normalizer(candidate)
    # All legacy state stays byte-identical; an old checkpoint still has its
    # original strict model/load contract.
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
    restored_legacy = EnsembleUNet2d(EnsembleUNet2dConfig(**_common()))
    restored_legacy.load_state_dict(legacy.state_dict(), strict=True)
    restored_candidate = EnsembleUNet2d(EnsembleUNet2dConfig(
        **_common(), history_conditioned_operator_basis=True))
    restored_candidate.load_state_dict(candidate.state_dict(), strict=True)
    x = torch.randn(2, 20, 16, 32, 3)
    for left, right in zip(
            legacy.forward_uq_training(x)[:5],
            candidate.forward_uq_training(x)[:5]):
        assert torch.equal(left, right)


def test_hcobf_is_mean_isolated_pressure_bypassed_and_condition_observable():
    torch.manual_seed(211)
    model = EnsembleUNet2d(EnsembleUNet2dConfig(
        **_common(), history_conditioned_operator_basis=True)).eval()
    _normalizer(model)
    _activate_hcobf(model)
    x = torch.randn(3, 20, 16, 32, 3)
    native, members, quantiles, lower, upper = model.forward_uq_training(x)[:5]
    assert lower.shape[-1] == upper.shape[-1] == 2
    condition = model.history_operator_condition(x)
    model.set_history_operator_condition_override(condition.flip(0))
    (permuted, permuted_members, permuted_quantiles, permuted_lower,
     permuted_upper) = model.forward_uq_training(x)[:5]
    model.set_history_operator_condition_override(None)
    # The only changed readout is fluctuation and its final temporal projection
    # keeps the forecast mean intact, while swapping donor conditions matters.
    torch.testing.assert_close(
        members.mean(dim=2), permuted_members.mean(dim=2),
        rtol=0.0, atol=2.0e-6)
    torch.testing.assert_close(
        native[..., 2:], permuted[..., 2:], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        members[..., 2:], permuted_members[..., 2:], rtol=0.0, atol=0.0)
    assert quantiles.shape[-2] == lower.shape[-1] == upper.shape[-1] == 2
    assert permuted_quantiles.shape == quantiles.shape
    assert permuted_lower.shape == lower.shape
    assert permuted_upper.shape == upper.shape
    assert not torch.allclose(native, permuted, rtol=0.0, atol=1.0e-7)


def test_hcobf_zero_projection_staggers_basis_gradients_until_update_two():
    torch.manual_seed(214)
    model = EnsembleUNet2d(EnsembleUNet2dConfig(
        **_common(), history_conditioned_operator_basis=True))
    _normalizer(model)
    _activate_hcobf(model, activate_projection=False)
    x = torch.randn(2, 20, 16, 32, 3)

    def backward() -> None:
        model.zero_grad(set_to_none=True)
        model.forward_distribution(x)[0].square().mean().backward()

    backward()
    assert all(
        basis.output_projection.weight.grad is not None
        and basis.output_projection.weight.grad.abs().max() > 0
        for basis in model.history_operator_basis)
    assert all(
        parameter.grad is None or parameter.grad.abs().max() == 0
        for name, parameter in model.named_parameters()
        if name.startswith(("history_operator_conditioner.",
                            "history_operator_basis."))
        and "output_projection" not in name)
    with torch.no_grad():
        for basis in model.history_operator_basis:
            basis.output_projection.weight.add_(
                basis.output_projection.weight.grad, alpha=-0.1)
    backward()
    assert model.history_operator_conditioner[0].weight.grad.abs().max() > 0
    assert all(basis.spectral_weight.grad.abs().max() > 0
               and basis.operator_logits.weight.grad.abs().max() > 0
               for basis in model.history_operator_basis)


def test_hcobf_global_path_preserves_feature_shape_under_cpu_amp():
    basis = HistoryConditionedOperatorBasis(8).eval()
    feature = torch.randn(2, 8, 7, 13)
    condition = torch.randn(2, 16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        value = basis._global(basis.input_projection(feature))
    assert value.shape == (2, basis.compact_channels, 7, 13)
    assert value.dtype == torch.bfloat16


def test_hcobf_paired_arms_have_identical_parameter_and_operator_counts():
    common = _common()
    torch.manual_seed(212)
    aligned = EnsembleUNet2d(EnsembleUNet2dConfig(
        **common, history_conditioned_operator_basis=True)).eval()
    torch.manual_seed(212)
    permuted = EnsembleUNet2d(EnsembleUNet2dConfig(
        **common, history_conditioned_operator_basis=True)).eval()
    assert sum(value.numel() for value in aligned.parameters()) == sum(
        value.numel() for value in permuted.parameters())
    assert aligned.state_dict().keys() == permuted.state_dict().keys()
    calls = [0, 0]
    handles = []
    for index, model in enumerate((aligned, permuted)):
        for basis in model.history_operator_basis:
            handles.append(basis.register_forward_hook(
                lambda _module, _args, _value, index=index: calls.__setitem__(
                    index, calls[index] + 1)))
    x = torch.randn(2, 20, 16, 32, 3)
    aligned.forward_distribution(x)
    condition = permuted.history_operator_condition(x)
    permuted.set_history_operator_condition_override(condition.flip(0))
    permuted.forward_distribution(x)
    permuted.set_history_operator_condition_override(None)
    for handle in handles:
        handle.remove()
    assert calls == [3, 3]
