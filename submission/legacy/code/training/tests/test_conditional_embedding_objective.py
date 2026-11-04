from __future__ import annotations

from pathlib import Path
import sys

import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import conditional_embedding_objective as subject


def sample_fields():
    torch.manual_seed(17)
    history = torch.randn(8, 20, 8, 12, 3)
    target = torch.randn(8, 20, 8, 12, 3)
    return history, target


def test_flow_summary_retains_physical_axes_and_gradients():
    history, _target = sample_fields()
    history.requires_grad_()
    summary = subject.flow_summary(history)
    assert summary.shape[0] == history.shape[0]
    assert summary.shape[1] > 150
    summary.square().mean().backward()
    assert history.grad is not None
    assert torch.isfinite(history.grad).all()
    assert float(history.grad.abs().sum()) > 0.0


def test_identical_conditional_distributions_have_zero_discrepancy():
    history, target = sample_fields()
    loss, metrics = subject.conditional_embedding_discrepancy(
        history, target, target)
    assert float(loss) < 1.0e-6
    assert metrics["summary_dimensions"] > 150
    assert metrics["condition_bandwidth"] > 0.0
    assert metrics["output_bandwidth"] > 0.0


def test_shifted_prediction_has_positive_finite_loss_and_gradient():
    history, target = sample_fields()
    prediction = (target.detach().clone() + torch.linspace(
        0.0, 1.0, target.shape[1]).view(1, -1, 1, 1, 1))
    prediction.requires_grad_()
    loss, metrics = subject.conditional_embedding_discrepancy(
        history, prediction, target)
    assert float(loss) > 0.0
    assert metrics["raw"] > 0.0
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad.abs().sum()) > 0.0


def test_bad_shapes_and_small_batches_are_rejected():
    history, target = sample_fields()
    try:
        subject.conditional_embedding_discrepancy(
            history[:2], target[:2], target[:2])
    except ValueError as error:
        assert "three samples" in str(error)
    else:
        raise AssertionError("batch two was accepted")
    try:
        subject.conditional_embedding_discrepancy(
            history, target[..., :2], target)
    except ValueError as error:
        assert "equal shape" in str(error)
    else:
        raise AssertionError("mismatched output shapes were accepted")
