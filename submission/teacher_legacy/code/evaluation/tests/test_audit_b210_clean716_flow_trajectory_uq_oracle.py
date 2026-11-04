from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


PATH = Path(__file__).resolve().parents[1] / "audit_b210_clean716_flow_trajectory_uq_oracle.py"
SPEC = importlib.util.spec_from_file_location("flow_trajectory_uq_oracle", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_zero_flow_envelope_is_exact_point() -> None:
    torch.manual_seed(2)
    point = torch.randn(3, 5, 8, 12, 2)
    flow = torch.zeros(3, 8, 12, 2)
    lower, upper = MODULE.trajectory_envelope(point, flow)
    assert torch.equal(lower, point)
    assert torch.equal(upper, point)


def test_envelope_is_ordered_and_contains_point() -> None:
    torch.manual_seed(3)
    point = torch.randn(2, 5, 8, 12, 2)
    flow = 0.25 * torch.randn(2, 8, 12, 2)
    lower, upper = MODULE.trajectory_envelope(point, flow)
    assert lower.shape == point.shape == upper.shape
    assert torch.all(lower <= point)
    assert torch.all(point <= upper)
    assert torch.all(lower <= upper)


def test_delta_report_uses_preregistered_sps_gate_fields() -> None:
    native = {
        "overall": {"sps_score": 40.0, "coverage": 0.7, "scored_mean_width": 0.01},
        "by_aoa": {str(v): {"sps_score": 40.0} for v in (0, 5, 10, 15, 20)},
    }
    candidate = {
        "overall": {"sps_score": 40.6, "coverage": 0.71, "scored_mean_width": 0.011},
        "by_aoa": {str(v): {"sps_score": 40.1 + v / 100.0}
                   for v in (0, 5, 10, 15, 20)},
    }
    row = MODULE.delta_report(candidate, native)
    assert np.isclose(row["sps_score"], 0.6)
    assert row["positive_aoa"] == 5
    assert np.isclose(row["worst_aoa_sps"], 0.1)

