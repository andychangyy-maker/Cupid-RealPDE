from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / (
    "evaluate_b210_point_oof_budget_allocation.py")
SPEC = importlib.util.spec_from_file_location("point_oof_allocator", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def test_budgeted_scale_preserves_each_spatial_width_budget():
    generator = torch.Generator().manual_seed(3)
    lower = torch.rand(2, 3, 5, 7, 2, generator=generator) + 0.01
    upper = torch.rand(2, 3, 5, 7, 2, generator=generator) + 0.01
    signal = torch.randn(2, 3, 5, 7, 2, generator=generator)
    scale = subject.budgeted_scale(lower, upper, signal, 0.5, 1.5)
    before = (lower + upper).mean(dim=(2, 3))
    after = ((lower + upper) * scale).mean(dim=(2, 3))
    torch.testing.assert_close(after, before, rtol=1e-6, atol=1e-7)


def test_zero_strength_is_exact_identity():
    lower = torch.rand(1, 2, 4, 4, 2) + 0.01
    upper = torch.rand(1, 2, 4, 4, 2) + 0.01
    signal = torch.randn_like(lower)
    scale = subject.budgeted_scale(lower, upper, signal, 0.0, 4.0)
    torch.testing.assert_close(scale, torch.ones_like(scale), rtol=0.0, atol=0.0)


def test_grid_parser_rejects_nonpositive_sharpness():
    with pytest.raises(ValueError, match="positive"):
        subject.parse_grid("0,1", positive=True)
