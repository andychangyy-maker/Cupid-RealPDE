import torch

from models.unet2d import HistoryAxisAdapter, UNet2d, UNet2dConfig


def config(mode: str) -> UNet2dConfig:
    return UNet2dConfig(
        in_steps=6,
        out_steps=4,
        in_channels=3,
        out_channels=3,
        base=8,
        depth=3,
        groups=4,
        difference_steps=2,
        use_coordinates=True,
        use_regime_film=True,
        history_axis_adapter_mode=mode,
        history_axis_adapter_channels=8,
        history_axis_adapter_blocks=2,
        history_axis_adapter_kernel=3,
    )


def test_candidate_and_control_have_identical_state_and_parameter_count():
    torch.manual_seed(210)
    temporal = HistoryAxisAdapter(config("causal_temporal"))
    torch.manual_seed(210)
    spatial = HistoryAxisAdapter(config("framewise_spatial_control"))
    assert temporal.state_dict().keys() == spatial.state_dict().keys()
    assert all(torch.equal(value, spatial.state_dict()[key])
               for key, value in temporal.state_dict().items())
    assert sum(p.numel() for p in temporal.parameters()) == sum(
        p.numel() for p in spatial.parameters())


def test_adapter_is_exact_identity_at_initialization_and_trains():
    torch.manual_seed(7)
    adapter = HistoryAxisAdapter(config("causal_temporal"))
    history = torch.randn(2, 6, 8, 12, 3)
    assert torch.equal(adapter(history), history)
    loss = adapter(history).square().mean()
    loss.backward()
    assert adapter.output_projection.weight.grad is not None
    assert adapter.output_projection.weight.grad.abs().sum() > 0


def test_temporal_arm_is_causal_and_control_has_no_cross_frame_path():
    torch.manual_seed(11)
    temporal = HistoryAxisAdapter(config("causal_temporal"))
    torch.manual_seed(11)
    spatial = HistoryAxisAdapter(config("framewise_spatial_control"))
    with torch.no_grad():
        temporal.output_projection.weight.normal_(0.0, 0.1)
        spatial.output_projection.weight.copy_(
            temporal.output_projection.weight)
        temporal.output_projection.bias.zero_()
        spatial.output_projection.bias.zero_()
    history = torch.randn(1, 6, 8, 12, 3)
    changed_late = history.clone()
    changed_late[:, -1] += 2.0
    temporal_before = temporal(history)
    temporal_after = temporal(changed_late)
    assert torch.equal(temporal_before[:, :-1], temporal_after[:, :-1])
    spatial_before = spatial(history)
    spatial_after = spatial(changed_late)
    assert torch.equal(spatial_before[:, :-1], spatial_after[:, :-1])

    changed_early = history.clone()
    changed_early[:, 0] += 2.0
    temporal_after_early = temporal(changed_early)
    spatial_after_early = spatial(changed_early)
    assert not torch.equal(
        temporal_before[:, 1:], temporal_after_early[:, 1:])
    assert torch.equal(spatial_before[:, 1:], spatial_after_early[:, 1:])


def test_unet_adapter_changes_only_stem_history_path_at_update_zero():
    torch.manual_seed(31)
    baseline = UNet2d(config("none"))
    torch.manual_seed(31)
    temporal = UNet2d(config("causal_temporal"))
    legacy = baseline.state_dict()
    adapted = temporal.state_dict()
    assert all(torch.equal(value, adapted[key])
               for key, value in legacy.items())
    history = torch.randn(2, 6, 16, 24, 3)
    baseline.eval()
    temporal.eval()
    with torch.inference_mode():
        assert torch.equal(baseline(history), temporal(history))


def test_adapter_config_round_trip():
    model = UNet2d(config("causal_temporal"))
    stored = dict(model.architecture_config())
    assert stored.pop("architecture") == "unet"
    assert UNet2dConfig(**stored) == model.config
