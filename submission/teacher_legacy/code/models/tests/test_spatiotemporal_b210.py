from __future__ import annotations

import torch

from models.spatiotemporal_b210 import (
    SpatiotemporalB210,
    SpatiotemporalB210Config,
)


def tiny_config(**overrides: object) -> SpatiotemporalB210Config:
    values: dict[str, object] = {
        "e0_channels": 8,
        "e1_channels": 16,
        "e2_channels": 24,
        "e3_channels": 32,
        "e0_blocks": 1,
        "e1_blocks": 1,
        "e2_blocks": 1,
        "e3_blocks": 1,
        "groups": 4,
        "channel_mlp_ratio": 2.0,
    }
    values.update(overrides)
    return SpatiotemporalB210Config(**values)


def history(batch: int = 1, height: int = 16, width: int = 24) -> torch.Tensor:
    return torch.randn(batch, 20, height, width, 3)


def test_shape_components_and_explicit_time_pyramids_are_finite():
    model = SpatiotemporalB210(tiny_config()).eval()
    source = history()
    with torch.inference_mode():
        observed = model.encode_history(source)
        future = model.project_future_pyramid(observed)
        components = model.forward_components(source)
    assert [tuple(value.shape[:3]) for value in observed] == [
        (1, 20, 8), (1, 20, 16), (1, 20, 24), (1, 20, 32)]
    assert [tuple(value.shape[:3]) for value in future] == [
        (1, 20, 8), (1, 20, 16), (1, 20, 24), (1, 20, 32)]
    assert components["decoded_future"].shape == (1, 20, 8, 16, 24)
    assert components["prediction"].shape == source.shape
    assert components["displacement"].shape == (1, 20, 16, 24, 2)
    assert torch.isfinite(components["prediction"]).all()
    assert model.stem.in_channels == 5
    assert model.architecture_config()["flattened_history_bypass"] is False


def test_paired_modes_have_bitwise_identical_state_and_only_axis_mode_differs():
    torch.manual_seed(901)
    candidate = SpatiotemporalB210(
        tiny_config(), "causal_temporal_candidate")
    torch.manual_seed(901)
    control = SpatiotemporalB210(
        tiny_config(), "framewise_spatial_control")
    assert candidate.state_dict().keys() == control.state_dict().keys()
    for name, value in candidate.state_dict().items():
        torch.testing.assert_close(
            value, control.state_dict()[name], rtol=0.0, atol=0.0)
    assert sum(value.numel() for value in candidate.parameters()) == sum(
        value.numel() for value in control.parameters())
    candidate_modules = dict(candidate.named_modules())
    control_modules = dict(control.named_modules())
    assert candidate_modules.keys() == control_modules.keys()
    differing_attributes = []
    for name in candidate_modules:
        left = candidate_modules[name]
        right = control_modules[name]
        if hasattr(left, "interaction_mode"):
            if left.interaction_mode != right.interaction_mode:
                differing_attributes.append(name)
    assert differing_attributes == [
        "", "stage0.0.axis", "stage1.0.axis",
        "stage2.0.axis", "stage3.0.axis"]


def test_all_scales_report_matched_axial_elements_macs_and_semantics():
    torch.manual_seed(902)
    candidate = SpatiotemporalB210(
        tiny_config(), "causal_temporal_candidate").eval()
    torch.manual_seed(902)
    control = SpatiotemporalB210(
        tiny_config(), "framewise_spatial_control").eval()
    source = history()
    with torch.inference_mode():
        candidate(source)
        control(source)
    candidate_contracts = candidate.axis_execution_contract()
    control_contracts = control.axis_execution_contract()
    assert len(candidate_contracts) == len(control_contracts) == 4
    for left, right in zip(candidate_contracts, control_contracts, strict=True):
        assert (left["stage"], left["block"]) == (
            right["stage"], right["block"])
        assert left["canonical_shape"] == right["canonical_shape"]
        assert left["output_elements"] == right["output_elements"]
        assert left["macs"] == right["macs"]
        assert left["kernel_size"] == right["kernel_size"] == 5
        assert left["left_pad"] == right["left_pad"] == 4
        assert left["right_pad"] == right["right_pad"] == 0
        assert left["cross_time"] is True
        assert left["control_time_folded"] is False
        assert right["cross_time"] is False
        assert right["control_time_folded"] is True


def test_candidate_is_left_causal_and_control_is_frame_independent_each_scale():
    torch.manual_seed(903)
    candidate = SpatiotemporalB210(
        tiny_config(), "causal_temporal_candidate").eval()
    torch.manual_seed(903)
    control = SpatiotemporalB210(
        tiny_config(), "framewise_spatial_control").eval()
    source = history()
    changed = source.clone()
    changed[:, 10:] += 5.0 * torch.randn_like(changed[:, 10:])
    one_frame_changed = source.clone()
    one_frame_changed[:, 10] += 5.0 * torch.randn_like(
        one_frame_changed[:, 10])
    with torch.inference_mode():
        candidate_first = candidate.encode_history(source)
        candidate_second = candidate.encode_history(changed)
        candidate_one_frame = candidate.encode_history(one_frame_changed)
        control_first = control.encode_history(source)
        control_second = control.encode_history(changed)
        control_one_frame = control.encode_history(one_frame_changed)
    for first, second in zip(candidate_first, candidate_second, strict=True):
        torch.testing.assert_close(
            first[:, :10], second[:, :10], rtol=0.0, atol=0.0)
    for first, second in zip(control_first, control_second, strict=True):
        torch.testing.assert_close(
            first[:, :10], second[:, :10], rtol=0.0, atol=0.0)
    unchanged = torch.tensor(
        [index for index in range(20) if index != 10], dtype=torch.long)
    for first, second in zip(
            control_first, control_one_frame, strict=True):
        # The changed frame itself is processed, but no other observed frame
        # can receive it through the control's spatial-x-only mixers.
        assert not torch.equal(first[:, 10], second[:, 10])
        torch.testing.assert_close(
            first.index_select(1, unchanged),
            second.index_select(1, unchanged), rtol=0.0, atol=0.0)
    # A single changed frame does reach later encoder coordinates in the
    # candidate, demonstrating that the causal path is active rather than an
    # accidental framewise implementation.
    assert all(not torch.equal(first[:, 11], second[:, 11])
               for first, second in zip(
                   candidate_first, candidate_one_frame, strict=True))


def test_mean_fluctuation_transport_decomposition_and_normalizer():
    model = SpatiotemporalB210(tiny_config())
    model.set_normalizer_statistics(
        mean_input=torch.tensor([2.0, 3.0, 4.0]),
        mean_target=torch.tensor([1.0, 1.0, 1.0]),
        std_input=torch.tensor([4.0, 6.0, 8.0]),
        std_target=torch.tensor([2.0, 3.0, 4.0]),
    )
    source = torch.zeros(1, 20, 16, 24, 3)
    components = model.forward_components(source)
    expected = torch.tensor([0.5, 2.0 / 3.0, 0.75]).view(1, 1, 1, 3)
    torch.testing.assert_close(components["persistence"], expected.expand(
        1, 16, 24, 3))
    assert torch.count_nonzero(components["displacement"]) == 0
    torch.testing.assert_close(
        components["fluctuation"].mean(dim=1),
        torch.zeros_like(components["mean_delta"]),
        rtol=0.0, atol=2.0e-7)
    torch.testing.assert_close(
        components["prediction"],
        components["transported"]
        + components["mean_delta"].unsqueeze(1)
        + components["fluctuation"])


def test_finite_backward_keeps_encoder_future_decoder_and_heads_live():
    torch.manual_seed(904)
    model = SpatiotemporalB210(tiny_config())
    source = history()
    target = torch.randn_like(source)
    loss = (model(source) - target).square().mean()
    loss.backward()
    parameters = (
        model.stem.weight,
        model.stage0[0].axis.depthwise.weight,
        model.stage1[0].axis.projection.weight,
        model.stage2[0].spatial.body[2].weight,
        model.stage3[0].channel_mlp[2].weight,
        model.observed_to_future_logits,
        model.horizon0,
        model.horizon3,
        model.coarse_spatial[0].body[2].weight,
        model.fuse2[0].body[5].weight,
        model.fuse0[0].skip.weight,
        model.output_spatial[0].body[2].weight,
        model.mean_head.weight,
        model.fluctuation_head.weight,
        model.transport_head.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_cpu_backward_at_singleton_coarsest_scale_is_layout_safe():
    """The 8x8 FFAR unit grid collapses stage three to one spatial cell."""
    torch.manual_seed(905)
    model = SpatiotemporalB210(tiny_config())
    source = history(height=8, width=8)
    target = torch.randn_like(source)
    (model(source) - target).square().mean().backward()
    for parameter in (
            model.stage3[0].axis.norm.weight,
            model.stage3[0].spatial.body[0].weight,
            model.stage3[0].spatial.body[3].weight,
            model.stage3[0].spatial.body[5].weight,
            ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_default_capacity_and_frozen_four_scale_topology():
    model = SpatiotemporalB210(SpatiotemporalB210Config())
    count = sum(parameter.numel() for parameter in model.parameters())
    assert model.config.widths == (64, 128, 256, 512)
    assert model.config.stage_blocks == (2, 2, 2, 2)
    assert sum(len(stage) for stage in (
        model.stage0, model.stage1, model.stage2, model.stage3)) == 8
    assert count == 29_187_608
    assert len(model.state_dict()) == 238
