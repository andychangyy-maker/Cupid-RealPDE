import torch

from models.hierarchical_axial_temporal_operator import (
    AxialHistoryBlock,
    HATOConfig,
    HierarchicalAxialTemporalOperator,
)


def tiny_config(**overrides: object) -> HATOConfig:
    values = dict(
        e0_channels=16,
        e1_channels=24,
        e2_channels=32,
        heads_e1=4,
        heads_e2=4,
        groups=4,
    )
    values.update(overrides)
    return HATOConfig(**values)


def history(height: int = 16, width: int = 16) -> torch.Tensor:
    return torch.randn(1, 20, height, width, 3)


def test_tiny_dynamic_shape_is_finite_and_point_only():
    model = HierarchicalAxialTemporalOperator(tiny_config())
    result = model(history())
    assert result.shape == (1, 20, 16, 16, 3)
    assert torch.isfinite(result).all()
    assert not hasattr(model, "uncertainty_head")
    assert model.e0[0].in_channels == 5


def test_fixed_coordinates_break_simple_spatial_roll_equivariance():
    """Coordinates remain fixed when physical values are rolled."""
    torch.manual_seed(117)
    model = HierarchicalAxialTemporalOperator(tiny_config()).eval()
    source = history()
    shifted = torch.roll(source, shifts=(4, 4), dims=(2, 3))
    with torch.inference_mode():
        reference = model(source)
        shifted_prediction = model(shifted)
    # If coordinates were rolled along with the physical field, an otherwise
    # convolutional stem could be translation equivariant.  HATO constructs
    # coordinates anew at fixed locations, so undoing only the data roll does
    # not reproduce the original prediction.
    undone = torch.roll(shifted_prediction, shifts=(-4, -4), dims=(2, 3))
    assert not torch.allclose(reference, undone, atol=1e-6, rtol=1e-5)


def test_mean_fluctuation_decomposition_is_exact():
    model = HierarchicalAxialTemporalOperator(tiny_config())
    source = history()
    parts = model.forward_components(source)
    assert torch.allclose(parts["fluctuation"].mean(dim=1),
                          torch.zeros_like(parts["mean_delta"][:, 0]),
                          atol=2e-6, rtol=0)
    assert torch.allclose(
        parts["prediction"], parts["persistence"] + parts["mean_delta"]
        + parts["fluctuation"], atol=1e-6, rtol=0)
    assert torch.allclose(parts["raw_residual"], parts["mean_delta"]
                          + parts["fluctuation"], atol=1e-6, rtol=0)
    # The two heads begin close to persistence, rather than making a large
    # random residual that would conceal an optimizer/harness regression.
    assert parts["raw_residual"].square().mean().sqrt() < 0.1


def test_persistence_updates_with_normalizer_statistics():
    model = HierarchicalAxialTemporalOperator(tiny_config())
    model.set_normalizer_statistics(
        mean_input=torch.tensor([2.0, 3.0, 4.0]),
        mean_target=torch.tensor([1.0, 1.0, 1.0]),
        std_input=torch.tensor([4.0, 6.0, 8.0]),
        std_target=torch.tensor([2.0, 3.0, 4.0]),
    )
    source = torch.zeros(1, 20, 16, 16, 3)
    parts = model.forward_components(source)
    expected = torch.tensor([0.5, 2.0 / 3.0, 0.75]).view(1, 1, 1, 1, 3)
    assert torch.allclose(parts["persistence"], expected)
    assert torch.allclose(model._persistence(source), expected[:, 0])


def test_candidate_and_control_have_bitwise_identical_state_and_parameter_count():
    torch.manual_seed(123)
    candidate = HierarchicalAxialTemporalOperator(
        tiny_config(), "temporal_candidate")
    torch.manual_seed(123)
    control = HierarchicalAxialTemporalOperator(
        tiny_config(), "spatial_window_control")
    assert candidate.state_dict().keys() == control.state_dict().keys()
    assert all(torch.equal(value, control.state_dict()[key])
               for key, value in candidate.state_dict().items())
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in control.parameters())
    assert candidate.architecture_config()["interaction_mode"] == "temporal_candidate"
    assert control.architecture_config()["interaction_mode"] == "spatial_window_control"


def test_modes_change_output_and_have_live_gradients():
    torch.manual_seed(124)
    candidate = HierarchicalAxialTemporalOperator(
        tiny_config(), "temporal_candidate")
    torch.manual_seed(124)
    control = HierarchicalAxialTemporalOperator(
        tiny_config(), "spatial_window_control")
    source = history()
    candidate_output = candidate(source)
    control_output = control(source)
    assert not torch.equal(candidate_output, control_output)
    candidate_output.square().mean().backward()
    control_output.square().mean().backward()
    for model in (candidate, control):
        gradient = model.e1_blocks[0].attention.in_proj_weight.grad
        assert gradient is not None and torch.count_nonzero(gradient) > 0
        assert model.mean_head.weight.grad is not None
        assert torch.count_nonzero(model.mean_head.weight.grad) > 0
        assert model.fluctuation_head.weight.grad is not None
        assert torch.count_nonzero(model.fluctuation_head.weight.grad) > 0
        assert model.e1_observed_time.grad is not None
        assert torch.count_nonzero(model.e1_observed_time.grad) > 0


def test_control_padding_crop_and_matched_sequence_count():
    # E1-ish 8x8 and E2-ish 2x4 cover right padding and the tiny-screen
    # bottom padding.  Candidate/control both process exactly the same count
    # of length-20 attention sequences before cropping back to the source.
    for height, width in ((8, 8), (2, 4)):
        values = torch.randn(1, 20, 24, height, width)
        candidate = AxialHistoryBlock(
            24, 4, 2.0, 4, "temporal_candidate")
        control = AxialHistoryBlock(
            24, 4, 2.0, 4, "spatial_window_control")
        candidate_sequence, candidate_h, candidate_w = candidate._temporal_sequences(values)
        control_sequence, control_h, control_w = control._spatial_sequences(values)
        assert candidate_sequence.shape == control_sequence.shape
        assert (candidate_h, candidate_w) == (control_h, control_w)
        assert control(values).shape == values.shape
        assert candidate(values).shape == values.shape
