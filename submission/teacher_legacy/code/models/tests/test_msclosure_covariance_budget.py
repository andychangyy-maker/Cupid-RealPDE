import torch

from models.msclosure_covariance_budget import (
    COVARIANCE_FEEDBACK_CHANNELS,
    CovarianceBudgetMultipleShootingClosure,
    CovarianceBudgetMultipleShootingConfig,
    LocalPSDCovarianceBudget,
)
from models.multiple_shooting_closure import (
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
)


def _tiny_config() -> CovarianceBudgetMultipleShootingConfig:
    return CovarianceBudgetMultipleShootingConfig(
        e0_channels=8, e1_channels=16, e2_channels=24,
        e0_blocks=1, e1_blocks=1, e2_blocks=1,
        context_channels=8, cell_channels=16, cell_blocks=1,
    )


def test_config_compatibility_and_parameter_increment_is_minimal():
    config = _tiny_config()
    base = MultipleShootingClosure(config)
    candidate = CovarianceBudgetMultipleShootingClosure(config)
    base_parameters = sum(value.numel() for value in base.parameters())
    candidate_parameters = sum(
        value.numel() for value in candidate.parameters())
    expected_increment = (
        COVARIANCE_FEEDBACK_CHANNELS * config.cell_channels * 3 * 3
        + config.context_channels * 3 + 3)
    assert candidate_parameters - base_parameters == expected_increment

    # The model also accepts the original config class directly.
    original_config = MultipleShootingClosureConfig(**config.__dict__)
    rebuilt = CovarianceBudgetMultipleShootingClosure(original_config)
    assert rebuilt.config == original_config
    assert rebuilt.architecture_config()["covariance_feedback_channels"] == 8


def test_psd_budget_zero_initializes_to_reference_and_masks_geometry():
    torch.manual_seed(8)
    model = CovarianceBudgetMultipleShootingClosure(_tiny_config()).eval()
    reference = torch.tensor(((0.25, 0.05), (0.05, 0.50)))
    model.set_covariance_reference(reference)
    torch.testing.assert_close(
        model.covariance_reference(), reference, atol=1e-6, rtol=1e-6)

    history = torch.randn(2, 20, 16, 16, 3)
    history[:, :, 5:8, 6:10, :2] = 0.0
    with torch.no_grad():
        values = model.forward_components(history)
    budget = values["predicted_budget"]
    free = values["free_mask"][..., 0]
    torch.testing.assert_close(
        budget[free], reference.expand_as(budget)[free],
        atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(budget[~free]) == 0
    assert torch.linalg.eigvalsh(budget[free]).min() >= -1e-7

    # All closure heads start at zero, so twenty repeated anchors consume no
    # covariance budget.  Pressure is physical zero and cannot affect UV.
    assert torch.count_nonzero(values["realized_covariance"]) == 0
    torch.testing.assert_close(
        values["budget_residual"], budget, atol=0, rtol=0)
    assert torch.count_nonzero(values["prediction"][..., 2]) == 0
    changed_pressure = history.clone()
    changed_pressure[..., 2] = torch.randn_like(changed_pressure[..., 2])
    with torch.no_grad():
        changed = model(changed_pressure)
    assert torch.equal(values["prediction"], changed)


def test_welford_first_sample_is_zero_m2_and_matches_population_covariance():
    samples = torch.tensor((
        (1.0, 4.0),
        (2.0, 1.0),
        (-1.0, 3.0),
        (5.0, -2.0),
    ), dtype=torch.float16).view(1, 4, 1, 1, 2)
    mean = torch.zeros(1, 1, 1, 2, dtype=torch.float16)
    m2 = torch.zeros(1, 1, 1, 3, dtype=torch.float16)
    for count in range(samples.shape[1]):
        mean, m2 = CovarianceBudgetMultipleShootingClosure.welford_update(
            mean, m2, samples[:, count], count)
        assert mean.dtype == torch.float32
        assert m2.dtype == torch.float32
        if count == 0:
            assert torch.equal(mean, samples[:, 0].float())
            assert torch.count_nonzero(m2) == 0

    value = samples.float()
    centered = value - value.mean(dim=1, keepdim=True)
    direct = torch.einsum(
        "bthwc,bthwd->bhwcd", centered, centered) / value.shape[1]
    packed = torch.stack((
        direct[..., 0, 0], direct[..., 0, 1], direct[..., 1, 1]),
        dim=-1)
    torch.testing.assert_close(
        m2 / value.shape[1], packed, atol=1e-6, rtol=1e-6)


def test_rollout_welford_excludes_anchor_and_carries_detached_segments():
    torch.manual_seed(9)
    model = CovarianceBudgetMultipleShootingClosure(_tiny_config())
    with torch.no_grad():
        model.cell.source_head.bias.copy_(
            torch.tensor((0.06, -0.04)))
    history = torch.randn(2, 20, 16, 16, 3)
    values = model.forward_components(history)

    # There is no target/teacher-forced branch or train-only state update.
    with torch.no_grad():
        inference_prediction = model.eval()(history)
    assert torch.equal(values["prediction"].detach(), inference_prediction)

    normalized_uv = values["prediction"][..., :2].float()
    centered = normalized_uv - normalized_uv.mean(dim=1, keepdim=True)
    direct = torch.einsum(
        "bthwc,bthwd->bhwcd", centered, centered
    ) / model.config.out_steps
    torch.testing.assert_close(
        values["realized_covariance"], direct, atol=2e-6, rtol=2e-6)

    # The statistics presented at the second segment start are exactly the
    # first five generated frames.  The terminal-history anchor is absent.
    anchor = (
        history[:, -1, ..., :2] * model.std_input[:2]
        + model.mean_input[:2])
    first_segment = values["physical_uv"][:, :5]
    white = torch.stack([
        model._whiten_anchor_relative_state(
            first_segment[:, index], anchor, model.std_target[:2])
        for index in range(5)
    ], dim=1)
    white_mean = white.mean(dim=1)
    white_centered = white - white_mean.unsqueeze(1)
    white_matrix = torch.einsum(
        "bthwc,bthwd->bhwcd", white_centered, white_centered)
    white_packed_over_horizon = torch.stack((
        white_matrix[..., 0, 0], white_matrix[..., 0, 1],
        white_matrix[..., 1, 1]), dim=-1) / model.config.out_steps
    expected_start = torch.cat((white_mean, white_packed_over_horizon), dim=-1)
    free = values["free_mask"].to(expected_start)
    expected_start = expected_start * free
    actual_start = values[
        "welford_statistic_segment_starts_white"][:, 1]
    torch.testing.assert_close(
        actual_start, expected_start, atol=2e-6, rtol=2e-6)
    assert not values[
        "welford_statistic_segment_starts_white"].requires_grad


def test_budget_head_is_psd_without_an_artificial_upper_bound():
    head = LocalPSDCovarianceBudget(4)
    with torch.no_grad():
        head.projection.weight.normal_(std=0.4)
        head.projection.bias.copy_(torch.tensor((1.5, -2.0, -1.0)))
        packed = head(torch.randn(3, 4, 7, 9))
    matrix = CovarianceBudgetMultipleShootingClosure._packed_covariance_matrix(
        packed)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    assert bool(torch.isfinite(matrix).all())
    assert eigenvalues.min() >= -1e-5
    assert packed[..., 0].min() > 0
    assert packed[..., 2].min() > 0


def test_end_to_end_loss_reaches_covariance_budget_head():
    torch.manual_seed(10)
    model = CovarianceBudgetMultipleShootingClosure(_tiny_config())
    with torch.no_grad():
        model.cell.source_head.weight.normal_(std=0.02)
    prediction = model(torch.randn(2, 20, 16, 16, 3))
    prediction[..., :2].square().mean().backward()
    gradient = model.covariance_budget.projection.weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert torch.count_nonzero(gradient) > 0
