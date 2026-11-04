from __future__ import annotations

import sys
from pathlib import Path

import torch


PRIVATE = Path(__file__).resolve().parent
if str(PRIVATE) not in sys.path:
    sys.path.insert(0, str(PRIVATE))

import audit_b210_prefix_horizon_structure as audit  # noqa: E402


def test_fluctuation_removes_per_channel_spatial_mean():
    value = torch.randn(2, 3, 4, 5, 2) + torch.tensor([10.0, -4.0])
    result = audit.fluctuation(value)
    assert torch.allclose(
        result.mean(dim=(2, 3)), torch.zeros(2, 3, 2), atol=2e-6)


def test_fluctuation_cosine_is_scale_and_offset_invariant():
    truth = torch.randn(2, 3, 4, 5, 2)
    prediction = 3.0 * truth + torch.tensor([7.0, -2.0])
    cosine = audit.fluctuation_cosine(prediction, truth)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-12)


def test_structure_accumulator_detects_better_short_pattern():
    truth = torch.randn(2, 3, 4, 5, 2)
    long = -truth
    short = 2.0 * truth
    accumulator = audit.StructureAccumulator()
    accumulator.update(long, short, truth)
    report = accumulator.report()
    assert report["long_fluctuation_cosine"] < -0.999999
    assert report["short_fluctuation_cosine"] > 0.999999
    assert report["short_better_fluctuation_frame_fraction"] == 1.0
