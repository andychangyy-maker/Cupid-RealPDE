from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from local_lag_covariance import (  # noqa: E402
    local_lag_covariance_descriptor,
    local_lag_covariance_loss,
)


def fields(batch: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(210)
    history = torch.randn(batch, 20, 11, 17, 3, generator=generator)
    target = torch.randn(batch, 20, 11, 17, 3, generator=generator)
    prediction = torch.randn(batch, 20, 11, 17, 3, generator=generator)
    return history, prediction, target


def test_identity_prediction_has_zero_loss():
    history, _prediction, target = fields()
    loss = local_lag_covariance_loss(history, target, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_loss_is_raw_descriptor_mse_without_quiet_region_division():
    history, prediction, target = fields()
    prediction_descriptor = local_lag_covariance_descriptor(
        history, prediction)
    target_descriptor = local_lag_covariance_descriptor(history, target)
    expected = (prediction_descriptor - target_descriptor).square().mean()
    torch.testing.assert_close(
        local_lag_covariance_loss(history, prediction, target), expected)


def test_candidate_and_null_descriptor_are_finite_with_expected_shape():
    history, prediction, _target = fields()
    candidate = local_lag_covariance_descriptor(
        history, prediction, anchor="candidate")
    null = local_lag_covariance_descriptor(
        history, prediction, anchor="null", donor_positions=[1, 2, 0])
    assert candidate.shape == (3, 8, 4, 8, 2)
    assert null.shape == candidate.shape
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(null).all()


def test_null_anchor_changes_only_cross_covariance():
    history, prediction, _target = fields()
    candidate = local_lag_covariance_descriptor(history, prediction)
    null = local_lag_covariance_descriptor(
        history, prediction, anchor="null", donor_positions=[1, 2, 0])
    assert not torch.allclose(candidate[:, :4], null[:, :4])
    torch.testing.assert_close(candidate[:, 4:], null[:, 4:])


def test_temporally_constant_future_is_finite():
    history, _prediction, _target = fields()
    constant = torch.randn(3, 1, 11, 17, 3).expand(-1, 20, -1, -1, -1).clone()
    descriptor = local_lag_covariance_descriptor(history, constant)
    loss = local_lag_covariance_loss(history, constant, constant)
    assert torch.isfinite(descriptor).all()
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_pressure_does_not_affect_descriptor_or_loss():
    history, prediction, target = fields()
    changed_history = history.clone()
    changed_prediction = prediction.clone()
    changed_target = target.clone()
    changed_history[..., 2] += 100.0
    changed_prediction[..., 2] -= 37.0
    changed_target[..., 2] += 5.0
    torch.testing.assert_close(
        local_lag_covariance_descriptor(history, prediction),
        local_lag_covariance_descriptor(changed_history, changed_prediction))
    torch.testing.assert_close(
        local_lag_covariance_loss(history, prediction, target),
        local_lag_covariance_loss(
            changed_history, changed_prediction, changed_target))


def test_prediction_gradient_is_finite_and_nonzero():
    history, prediction, target = fields()
    prediction.requires_grad_()
    loss = local_lag_covariance_loss(history, prediction, target)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() > 0.0


def test_null_anchor_requires_a_fixed_point_free_permutation():
    history, prediction, target = fields()
    with pytest.raises(ValueError, match="explicit donor positions"):
        local_lag_covariance_descriptor(history, prediction, anchor="null")
    with pytest.raises(ValueError, match="batch permutation"):
        local_lag_covariance_loss(
            history, prediction, target, anchor="null",
            donor_positions=[1, 1, 0])
    with pytest.raises(ValueError, match="fixed points"):
        local_lag_covariance_loss(
            history, prediction, target, anchor="null",
            donor_positions=[0, 2, 1])


def test_null_anchor_rejects_singleton_batch():
    history, prediction, target = fields(batch=1)
    with pytest.raises(ValueError, match="fixed points"):
        local_lag_covariance_descriptor(
            history, prediction, anchor="null", donor_positions=[0])
    with pytest.raises(ValueError, match="fixed points"):
        local_lag_covariance_loss(
            history, prediction, target, anchor="null", donor_positions=[0])
