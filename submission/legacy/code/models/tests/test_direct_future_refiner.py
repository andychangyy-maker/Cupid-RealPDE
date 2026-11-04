from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.direct_future_refiner import (  # noqa: E402
    DirectFutureRefiner,
    DirectFutureRefinerConfig,
)


def make_model() -> DirectFutureRefiner:
    return DirectFutureRefiner(DirectFutureRefinerConfig(
        steps=5, channels=3, hidden_channels=8, blocks=2, groups=4))


def test_refiner_shape_zero_condition_equivalence_and_gradient():
    torch.manual_seed(7)
    model = make_model()
    future = torch.randn(2, 5, 7, 11, 3, requires_grad=True)
    ordinary = model(future)
    explicit_zero = model(future, torch.zeros(2))
    noisy = model(future, torch.tensor([.2, .8]))
    assert ordinary.shape == future.shape
    torch.testing.assert_close(ordinary, explicit_zero, rtol=0.0, atol=0.0)
    assert not torch.equal(ordinary, noisy)
    loss = noisy.square().mean()
    loss.backward()
    assert future.grad is not None and torch.isfinite(future.grad).all()
    assert model.blocks[0].step.weight.grad is not None
    assert float(model.blocks[0].step.weight.grad.abs().sum()) > 0.0


def test_refiner_is_bounded_and_validates_contract():
    model = make_model()
    future = torch.randn(2, 5, 7, 11, 3)
    correction = model.correction(future)
    assert float(correction.abs().max()) <= model.config.residual_limit + 1e-6
    with pytest.raises(ValueError, match="expected future"):
        model(torch.randn(2, 4, 7, 11, 3))
    with pytest.raises(ValueError, match="non-negative"):
        model(future, torch.tensor([-.1, .2]))
    with pytest.raises(ValueError, match="divisible"):
        DirectFutureRefinerConfig(hidden_channels=10, groups=4).validate()
