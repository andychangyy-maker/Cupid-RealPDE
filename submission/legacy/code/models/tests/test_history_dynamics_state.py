import torch

from models.unet2d import (
    HistoryDynamicsState,
    UNet2d,
    UNet2dConfig,
    build_history_features,
    history_regime_summary,
)


def config(mode: str | None = None, **overrides: object) -> UNet2dConfig:
    values = dict(
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
        multi_scale_regime_film=True,
        cross_conditioned_backbone=True,
        cross_conditioned_steps=3,
        cross_conditioned_channels=8,
    )
    if mode is not None:
        values.update(
            history_dynamics_state_mode=mode,
            history_dynamics_state_channels=8,
            history_dynamics_state_blocks=2,
            history_dynamics_state_kernel=3,
        )
    values.update(overrides)
    return UNet2dConfig(**values)


def test_none_mode_is_bitwise_legacy_compatible():
    legacy_config = config()
    explicit_none = config("none")
    torch.manual_seed(91)
    legacy = UNet2d(legacy_config)
    torch.manual_seed(91)
    disabled = UNet2d(explicit_none)
    assert legacy.state_dict().keys() == disabled.state_dict().keys()
    assert all(torch.equal(value, disabled.state_dict()[key])
               for key, value in legacy.state_dict().items())
    history = torch.randn(2, 6, 16, 20, 3)
    legacy.eval()
    disabled.eval()
    with torch.inference_mode():
        assert torch.equal(legacy(history), disabled(history))


def test_candidate_and_control_match_parameter_shapes_and_seeded_state():
    torch.manual_seed(211)
    temporal = UNet2d(config("causal_temporal"))
    torch.manual_seed(211)
    control = UNet2d(config("framewise_spatial_control"))
    assert temporal.state_dict().keys() == control.state_dict().keys()
    assert all(torch.equal(value, control.state_dict()[key])
               for key, value in temporal.state_dict().items())
    assert sum(parameter.numel() for parameter in temporal.parameters()) == sum(
        parameter.numel() for parameter in control.parameters())


def test_temporal_state_is_causal_and_control_has_no_cross_frame_path():
    torch.manual_seed(17)
    temporal = HistoryDynamicsState(config("causal_temporal"))
    torch.manual_seed(17)
    control = HistoryDynamicsState(config("framewise_spatial_control"))
    history = torch.randn(1, 6, 9, 13, 3)
    changed_late = history.clone()
    changed_late[:, -1] += 2.0
    assert torch.equal(
        temporal.encode_frames(history)[:, :-1],
        temporal.encode_frames(changed_late)[:, :-1])
    assert torch.equal(
        control.encode_frames(history)[:, :-1],
        control.encode_frames(changed_late)[:, :-1])

    changed_early = history.clone()
    changed_early[:, 0] += 2.0
    assert not torch.equal(
        temporal.encode_frames(history)[:, 1:],
        temporal.encode_frames(changed_early)[:, 1:])
    assert torch.equal(
        control.encode_frames(history)[:, 1:],
        control.encode_frames(changed_early)[:, 1:])


def test_shared_state_reaches_stem_regime_and_local_dynamics_consumers():
    model = UNet2d(config("causal_temporal"))
    history = torch.randn(2, 6, 16, 20, 3)
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, inputs):
            captured[name] = inputs[0]
        return hook

    handles = [
        model.stem.register_forward_pre_hook(capture("stem")),
        model.regime_film[0].register_forward_pre_hook(capture("regime")),
        model.cross_conditioner.body[0].register_forward_pre_hook(
            capture("dynamics")),
    ]
    try:
        model.eval()
        with torch.inference_mode():
            state = model.history_dynamics_state(history)
            model(history)
    finally:
        for handle in handles:
            handle.remove()

    raw_features = build_history_features(
        history, model.config.difference_steps, model.config.use_coordinates,
        model.config.acceleration_steps)
    assert torch.equal(captured["stem"][:, :raw_features.shape[1]], raw_features)
    assert torch.equal(captured["stem"][:, raw_features.shape[1]:], state)

    expected_regime = torch.cat((
        history_regime_summary(history),
        state.mean(dim=(2, 3)),
        state.std(dim=(2, 3), unbiased=False),
    ), dim=-1)
    assert torch.equal(captured["regime"], expected_regime)
    assert torch.equal(captured["dynamics"][:, -state.shape[1]:], state)


def test_history_dynamics_state_config_round_trip():
    model = UNet2d(config("causal_temporal"))
    stored = dict(model.architecture_config())
    assert stored.pop("architecture") == "unet"
    assert UNet2dConfig(**stored) == model.config
