from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fno_interval_head import FNOIntervalHead


def test_zero_initialized_head_returns_base_widths_and_backpropagates():
    head = FNOIntervalHead(hidden=8)
    prediction = torch.randn(2, 5, 4, 6, 2, requires_grad=True)
    last_observed = torch.randn(2, 4, 6, 2)

    width = head(prediction, last_observed)

    expected = torch.tensor([0.01075, 0.00550]).view(1, 1, 1, 1, 2)
    assert width.shape == prediction.shape
    assert torch.allclose(width, expected.expand_as(width))
    assert torch.all(width > 0)
    width.mean().backward()
    assert head.network[-1].weight.grad is not None


def test_head_accepts_singleton_time_axis_for_last_observation():
    head = FNOIntervalHead(hidden=4)
    prediction = torch.zeros(1, 3, 2, 4, 3)
    last_observed = torch.zeros(1, 1, 2, 4, 3)

    width = head(prediction, last_observed)

    assert width.shape == (1, 3, 2, 4, 2)
    assert torch.isfinite(width).all()
