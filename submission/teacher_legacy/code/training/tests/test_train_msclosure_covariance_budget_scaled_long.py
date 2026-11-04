from __future__ import annotations

from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_msclosure_covariance_budget_scaled_long as subject  # noqa: E402


def test_exact_covariance_match_has_finite_zero_gradient():
    truth = torch.randn(2, 5, 7, 3)
    prediction = truth.detach().clone().requires_grad_()
    free = torch.ones(2, 5, 7, 1, dtype=torch.bool)
    error = subject.covariance_relative_error(prediction, truth, free)
    assert torch.equal(error, torch.zeros_like(error))
    error.mean().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) == 0
