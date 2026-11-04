import pytest
import torch

from models.ensemble_unet2d import EnsembleUNet2dConfig
from models.unet2d import UNet2d, UNet2dConfig


def config(mode: str = "none", **overrides: object) -> UNet2dConfig:
    values = dict(
        in_steps=20,
        out_steps=4,
        in_channels=3,
        out_channels=3,
        base=8,
        depth=4,
        groups=4,
        difference_steps=2,
        use_coordinates=True,
        use_regime_film=True,
        multi_scale_regime_film=True,
        cross_conditioned_backbone=True,
        cross_conditioned_steps=3,
        cross_conditioned_channels=8,
        deformable_history_memory_mode=mode,
    )
    values.update(overrides)
    return UNet2dConfig(**values)


def random_history() -> torch.Tensor:
    return torch.randn(2, 20, 64, 128, 3)


def test_none_mode_is_bitwise_legacy_compatible():
    legacy_config = config()
    explicit_none = config("none")
    torch.manual_seed(617)
    legacy = UNet2d(legacy_config)
    torch.manual_seed(617)
    disabled = UNet2d(explicit_none)
    assert legacy.state_dict().keys() == disabled.state_dict().keys()
    assert all(torch.equal(value, disabled.state_dict()[key])
               for key, value in legacy.state_dict().items())
    history = random_history()
    legacy.eval()
    disabled.eval()
    with torch.inference_mode():
        assert torch.equal(legacy(history), disabled(history))


def test_aligned_and_permuted_have_identical_initial_state_and_identity_output():
    torch.manual_seed(618)
    aligned = UNet2d(config("aligned"))
    torch.manual_seed(618)
    permuted = UNet2d(config("spatially_permuted"))
    assert aligned.state_dict().keys() == permuted.state_dict().keys()
    assert all(torch.equal(value, permuted.state_dict()[key])
               for key, value in aligned.state_dict().items())
    assert sum(value.numel() for value in aligned.parameters()) == sum(
        value.numel() for value in permuted.parameters())

    torch.manual_seed(618)
    legacy = UNet2d(config("none"))
    history = random_history()
    aligned.eval()
    permuted.eval()
    legacy.eval()
    with torch.inference_mode():
        assert torch.equal(aligned(history), permuted(history))
        assert torch.equal(aligned(history), legacy(history))


def test_active_construction_preserves_legacy_tensors_and_rng_stream():
    torch.manual_seed(619)
    legacy = UNet2d(config("none"))
    legacy_rng = torch.get_rng_state().clone()
    torch.manual_seed(619)
    active = UNet2d(config("aligned"))
    active_rng = torch.get_rng_state().clone()

    assert torch.equal(legacy_rng, active_rng)
    active_state = active.state_dict()
    for key, value in legacy.state_dict().items():
        assert torch.equal(value, active_state[key])
    assert not any(
        key.startswith("deformable_history_memory.")
        for key in legacy.state_dict())
    reloaded = UNet2d(config("none"))
    reloaded.load_state_dict(legacy.state_dict(), strict=True)
    assert sum(value.numel() for value in active.parameters()) - sum(
        value.numel() for value in legacy.parameters()) == 5_240


def test_selected_offset_modes_and_runtime_zero_override_are_observable():
    torch.manual_seed(620)
    aligned = UNet2d(config("aligned"))
    torch.manual_seed(620)
    permuted = UNet2d(config("spatially_permuted"))
    for model in (aligned, permuted):
        memory = model.deformable_history_memory
        assert memory is not None
        for projection in memory.value_projections:
            torch.nn.init.normal_(projection.weight, mean=0.0, std=0.05)
            torch.nn.init.normal_(projection.bias, mean=0.0, std=0.05)
    # Restore equal nonzero memory value projections after independently
    # consuming the same RNG stream above.
    permuted.deformable_history_memory.load_state_dict(
        aligned.deformable_history_memory.state_dict())
    history = random_history()
    aligned.eval()
    permuted.eval()
    with torch.inference_mode():
        aligned_features = aligned._decode(history)
        permuted_features = permuted._decode(history)
        aligned.set_deformable_history_memory_zero_offset(True)
        zero_offset_features = aligned._decode(history)
        aligned.set_deformable_history_memory_zero_offset(False)
    assert not torch.equal(aligned_features[0], permuted_features[0])
    assert not torch.equal(aligned_features[0], zero_offset_features[0])


def test_active_memory_has_live_value_then_query_gradients_at_both_scales():
    torch.manual_seed(621)
    model = UNet2d(config("aligned"))
    memory = model.deformable_history_memory
    assert memory is not None
    history = random_history()

    # Zero residual readouts preserve the update-zero identity, but their own
    # gradients must already be live so the branch can open on the first step.
    loss = sum(feature.square().mean() for feature in model._decode(history))
    loss.backward()
    assert all(projection.weight.grad is not None
               and torch.count_nonzero(projection.weight.grad) > 0
               for projection in memory.value_projections)

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for projection in memory.value_projections:
            torch.nn.init.normal_(projection.weight, mean=0.0, std=0.05)
    loss = sum(feature.square().mean() for feature in model._decode(history))
    loss.backward()
    assert all(head.weight.grad is not None
               and torch.count_nonzero(head.weight.grad) > 0
               for head in memory.query_heads)
    assert memory.frame_encoder.weight.grad is not None
    assert torch.count_nonzero(memory.frame_encoder.weight.grad) > 0


def test_frozen_contract_and_dual_decoder_rejection():
    with pytest.raises(ValueError, match="anchors are frozen"):
        UNet2d(config("aligned", deformable_history_memory_anchors=(0, 5, 10, 19)))
    with pytest.raises(ValueError, match="depth-four"):
        UNet2d(config("aligned", depth=3))
    with pytest.raises(ValueError, match="single-decoder"):
        EnsembleUNet2dConfig(**{
            **config("aligned").__dict__,
            "dual_decoder_mode": "shared_fused",
        }).validate()


def test_runtime_zero_offset_requires_an_active_memory_branch():
    with pytest.raises(RuntimeError, match="disabled"):
        UNet2d(config("none")).set_deformable_history_memory_zero_offset(True)
