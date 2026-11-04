from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / (
    "evaluate_ensemble_uq_fold_committee.py")
SPEC = importlib.util.spec_from_file_location("fold_committee", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def test_exact_sps_accumulator_and_merge():
    truth = torch.ones(1, 20, 32, 64, 2)
    prediction = truth.clone()
    lower = torch.tensor([0.1, 0.5]).view(1, 1, 1, 1, 2).expand_as(truth)
    upper = torch.tensor([0.1, 1.0]).view(1, 1, 1, 1, 2).expand_as(truth)
    row = subject.exact_sps_update(prediction, truth, lower, upper)
    assert row["count"] == truth.numel()
    assert row["covered"] == truth.numel()
    assert row["width_sum"] / row["count"] == pytest.approx(0.85)
    merged = subject.merge_sps([row, row])
    assert merged["scored_elements"] == 2 * truth.numel()
    assert merged["coverage"] == pytest.approx(1.0)
    assert merged["mean_width"] == pytest.approx(0.85)


def test_exact_sps_rejects_shape_mismatch():
    value = torch.zeros(1, 1, 1, 1, 2)
    with pytest.raises(ValueError, match="same shape"):
        subject.exact_sps_update(value, value[..., :1], value, value)


def test_merge_sps_rejects_empty_accumulator():
    with pytest.raises(ValueError, match="empty"):
        subject.merge_sps([])
