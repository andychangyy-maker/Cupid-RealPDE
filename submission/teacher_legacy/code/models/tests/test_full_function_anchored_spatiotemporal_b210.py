from __future__ import annotations

import pytest
import torch

from models.full_function_anchored_spatiotemporal_b210 import (
    FullFunctionAnchoredSTB210,
)
from models.spatiotemporal_b210 import (
    SpatiotemporalB210,
    SpatiotemporalB210Config,
)


def _config() -> SpatiotemporalB210Config:
    return SpatiotemporalB210Config(
        e0_channels=4, e1_channels=8, e2_channels=12, e3_channels=16,
        e0_blocks=1, e1_blocks=1, e2_blocks=1, e3_blocks=1,
        groups=2, channel_mlp_ratio=1.0, decoder_blocks=1)


def _stats(kind: str) -> dict[str, torch.Tensor]:
    if kind == "sim":
        return {
            "mean_input": torch.tensor([1.0, -2.0, .5]),
            "mean_target": torch.tensor([.25, -.75, 1.5]),
            "std_input": torch.tensor([2.0, 3.0, 4.0]),
            "std_target": torch.tensor([.5, 1.5, 2.5]),
        }
    return {
        "mean_input": torch.tensor([-.5, .75, 0.0]),
        "mean_target": torch.tensor([.1, -.2, 0.0]),
        "std_input": torch.tensor([.8, 1.25, 1.0]),
        "std_target": torch.tensor([1.1, .9, 1.0]),
    }


def _single() -> SpatiotemporalB210:
    torch.manual_seed(501)
    model = SpatiotemporalB210(_config(), "causal_temporal_candidate")
    sim = _stats("sim")
    model.set_normalizer_statistics(
        sim["mean_input"], sim["mean_target"],
        sim["std_input"], sim["std_target"])
    return model


def _model(mode: str) -> FullFunctionAnchoredSTB210:
    return FullFunctionAnchoredSTB210.from_single_model(
        _single(), mode, _stats("sim"), _stats("real"))


def _physical(batch: int = 1) -> torch.Tensor:
    torch.manual_seed(502)
    return torch.randn(batch, 20, 8, 8, 3)


def test_matched_states_parameters_calls_and_mutually_exclusive_roles() -> None:
    control = _model("trainable_source_control")
    candidate = _model("frozen_source_candidate")
    assert control.state_dict().keys() == candidate.state_dict().keys()
    for name, value in control.state_dict().items():
        torch.testing.assert_close(
            value, candidate.state_dict()[name], rtol=0.0, atol=0.0)
    assert sum(value.numel() for value in control.parameters()) == sum(
        value.numel() for value in candidate.parameters())
    assert all(value.requires_grad for value in control.source_model.parameters())
    assert all(
        not value.requires_grad
        for value in control.real_residual_model.parameters())
    assert not control.residual_gain.requires_grad
    assert all(
        not value.requires_grad
        for value in candidate.source_model.parameters())
    assert all(
        value.requires_grad
        for value in candidate.real_residual_model.parameters())
    assert candidate.residual_gain.requires_grad
    control.train()
    candidate.train()
    assert control.source_model.training
    assert not control.real_residual_model.training
    assert not candidate.source_model.training
    assert candidate.real_residual_model.training

    with torch.inference_mode():
        outputs = {
            name: model.eval().forward_physical(_physical())["prediction_physical"]
            for name, model in (("control", control), ("candidate", candidate))}
    torch.testing.assert_close(
        outputs["control"], outputs["candidate"], rtol=0.0, atol=0.0)
    assert len(control.execution_contract()["source"]) == 4
    assert len(control.execution_contract()["real_residual"]) == 4
    for branch in control.execution_contract().values():
        assert all(row["cross_time"] is True for row in branch)
        assert all(row["control_time_folded"] is False for row in branch)


def test_update_zero_is_complete_sim_physical_function_all_three_channels() -> None:
    single = _single().eval()
    candidate = FullFunctionAnchoredSTB210.from_single_model(
        single, "frozen_source_candidate", _stats("sim"), _stats("real"))
    physical = _physical()
    sim = _stats("sim")
    sim_history = (
        physical - sim["mean_input"].view(1, 1, 1, 1, 3)
    ) / sim["std_input"].view(1, 1, 1, 1, 3)
    with torch.inference_mode():
        expected = (
            single(sim_history)
            * sim["std_target"].view(1, 1, 1, 1, 3)
            + sim["mean_target"].view(1, 1, 1, 1, 3))
        components = candidate.eval().forward_physical(physical)
    torch.testing.assert_close(
        components["source_physical"], expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        components["prediction_physical"], expected, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(components["residual_physical"]) == 0
    # Channel three is intentionally retained inside the trainable model; the
    # official Real evaluator is the only layer that discards unmeasured p.
    assert torch.count_nonzero(expected[..., 2]) > 0


def test_source_uses_sim_stats_residual_uses_real_stats_and_common_install_fails():
    model = _model("frozen_source_candidate")
    sim, real = _stats("sim"), _stats("real")
    torch.testing.assert_close(
        model.source_model.persistence_scale,
        sim["std_input"] / sim["std_target"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        model.real_residual_model.persistence_scale,
        real["std_input"] / real["std_target"], rtol=0.0, atol=0.0)
    before = model.source_physical(_physical())
    with torch.no_grad():
        model._real_mean_input.add_(17.0)
        model.real_residual_model.set_normalizer_statistics(
            model._real_mean_input, model._real_mean_target,
            model._real_std_input, model._real_std_target)
    after = model.source_physical(_physical())
    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    with pytest.raises(RuntimeError, match="common normalizer"):
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3))


def test_nonzero_gain_matches_independent_physical_fusion_without_target_mean():
    model = _model("frozen_source_candidate").eval()
    physical = _physical()
    real = _stats("real")
    gain = torch.tensor([.2, -.3, .4])
    with torch.no_grad():
        model.residual_gain.copy_(gain)
        source = model.source_physical(physical)
        real_history = (
            physical - real["mean_input"].view(1, 1, 1, 1, 3)
        ) / real["std_input"].view(1, 1, 1, 1, 3)
        residual = model.real_residual_model(real_history)
        expected = source + (
            gain.view(1, 1, 1, 1, 3) * residual
            * real["std_target"].view(1, 1, 1, 1, 3))
        actual = model.forward_physical(physical)["prediction_physical"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_domain_consistency_audit_detects_each_wrapper_normalizer_mutation():
    for domain, name in (("sim", "mean_input"), ("real", "std_target")):
        model = _model("frozen_source_candidate")
        with torch.no_grad():
            getattr(model, f"_{domain}_{name}").add_(.125)
        with pytest.raises(RuntimeError, match=f"{domain} member"):
            model._assert_domain_normalizers_consistent()


def test_control_and_candidate_gradient_destinations_are_disjoint() -> None:
    real = _stats("real")
    physical = _physical()
    normalized = (
        physical - real["mean_input"].view(1, 1, 1, 1, 3)
    ) / real["std_input"].view(1, 1, 1, 1, 3)
    target = torch.randn_like(normalized) + 2.0
    for mode in ("trainable_source_control", "frozen_source_candidate"):
        model = _model(mode)
        source_before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name.startswith("source_model.") or name.startswith("_sim_")}
        residual_before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if (name.startswith("real_residual_model.")
                or name.startswith("_real_") or name == "residual_gain")}
        loss = (model(normalized) - target).square().mean()
        loss.backward()
        source_grads = [value.grad for value in model.source_model.parameters()]
        residual_grads = [
            value.grad for value in model.real_residual_model.parameters()]
        if mode == "trainable_source_control":
            assert all(value is not None for value in source_grads)
            assert all(value is None for value in residual_grads)
            assert model.residual_gain.grad is None
        else:
            assert all(value is None for value in source_grads)
            assert all(value is not None for value in residual_grads)
            assert all(torch.count_nonzero(value) == 0 for value in residual_grads)
            assert torch.count_nonzero(model.residual_gain.grad) > 0

        optimizer = torch.optim.SGD(
            (value for value in model.parameters() if value.requires_grad),
            lr=.1)
        optimizer.step()
        if mode == "frozen_source_candidate":
            optimizer.zero_grad(set_to_none=True)
            (model(normalized) - target).square().mean().backward()
            assert any(
                value.grad is not None and torch.count_nonzero(value.grad) > 0
                for value in model.real_residual_model.parameters())
        current = model.state_dict()
        source_equal = all(torch.equal(current[name], value)
                           for name, value in source_before.items())
        residual_equal = all(torch.equal(current[name], value)
                             for name, value in residual_before.items())
        if mode == "trainable_source_control":
            assert not source_equal and residual_equal
        else:
            assert source_equal and not residual_equal
