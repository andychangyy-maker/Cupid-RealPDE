from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structured_interval_head import StructuredIntervalHead
from conditioned_residual_corrector import ConditionedResidualCorrector


def test_structured_head_shapes_initialization_and_backward():
    for architecture in ("pointwise", "spatial", "dilated", "deep"):
        for asymmetric in (False, True):
            for global_context in (False, True):
                head = StructuredIntervalHead(
                    hidden=4, architecture=architecture,
                    asymmetric=asymmetric, global_context=global_context)
                prediction = torch.randn(
                    2, 3, 4, 6, 2, requires_grad=True)
                observation = torch.randn(2, 5, 4, 6, 3)

                lower, upper = head(prediction, observation)

                expected = torch.tensor(
                    [0.01075, 0.00550]).view(1, 1, 1, 1, 2)
                assert lower.shape == prediction.shape
                assert upper.shape == prediction.shape
                assert torch.allclose(lower, expected.expand_as(lower))
                assert torch.allclose(upper, expected.expand_as(upper))
                assert torch.isfinite(lower).all() and torch.isfinite(upper).all()
                (lower.mean() + upper.mean()).backward()
                assert head.network[-1].weight.grad is not None


def test_structured_head_rejects_mismatched_spatial_shape():
    head = StructuredIntervalHead(hidden=4)
    prediction = torch.zeros(1, 3, 4, 6, 2)
    observation = torch.zeros(1, 5, 5, 6, 3)

    try:
        head(prediction, observation)
    except ValueError as error:
        assert "spatial shapes" in str(error)
    else:
        raise AssertionError("mismatched spatial shape was accepted")


def test_condition_context_requires_and_uses_two_values():
    head = StructuredIntervalHead(
        hidden=4, architecture="deep", asymmetric=True,
        condition_context=True)
    prediction = torch.randn(2, 3, 4, 6, 2)
    observation = torch.randn(2, 5, 4, 6, 3)
    condition = torch.randn(2, 2)
    lower, upper = head(prediction, observation, condition)
    assert lower.shape == prediction.shape
    assert upper.shape == prediction.shape
    try:
        head(prediction, observation)
    except ValueError as error:
        assert "condition" in str(error)
    else:
        raise AssertionError("missing condition was accepted")


def test_conditioned_corrector_starts_as_identity_and_backpropagates():
    corrector = ConditionedResidualCorrector(
        hidden=4, context_hidden=4, architecture="deep")
    prediction = torch.randn(2, 3, 4, 6, 2, requires_grad=True)
    observation = torch.randn(2, 5, 4, 6, 3)
    corrected, condition = corrector(prediction, observation)
    assert corrected.shape == prediction.shape
    assert condition.shape == (2, 2)
    assert torch.allclose(corrected, prediction)
    (corrected.square().mean() + condition.square().mean()).backward()
    assert corrector.correction_head.weight.grad is not None
