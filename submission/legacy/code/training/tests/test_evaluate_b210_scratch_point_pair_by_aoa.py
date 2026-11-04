from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "evaluation" / "evaluate_b210_scratch_point_pair_by_aoa.py")
SPEC = importlib.util.spec_from_file_location("scratch_pair_by_aoa", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_close_metrics_accepts_only_sub_milli_batch_drift():
    subject.close_metrics({"score": 90.0 + 5e-4}, {"score": 90.0})
    with pytest.raises(RuntimeError, match="pooled metric replay changed"):
        subject.close_metrics({"score": 90.0 + 2e-3}, {"score": 90.0})
