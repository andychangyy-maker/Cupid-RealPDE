from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "evaluate_b210_point_uq_factorial_clean716.py"
)
SPEC = importlib.util.spec_from_file_location("point_uq_factorial", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def row(rel, tke, mvpe, sps, point_sha, runtime="same"):
    metrics = {
        "rel_l2_score": rel,
        "tke_rel_l2_score": tke,
        "mvpe_rel_l2_score": mvpe,
        "native_sps_score": sps,
    }
    state = {
        "samples": 1,
        "rel_error_sum": 1.0,
        "tke_error_sum": 2.0,
        "mvpe_error_sum": 3.0,
    }
    return {
        "metrics": metrics,
        "metric_state": state,
        "by_aoa": {
            str(aoa): {"native_sps_score": sps} for aoa in (0, 5, 10, 15, 20)
        },
        "point_prediction_sha256": point_sha,
        "runtime_source_sha256": runtime,
    }


def test_factorial_gate_accepts_strong_uq_on_strong_point() -> None:
    results = {
        "AA": row(96.0, 81.0, 96.0, 54.2, "a"),
        "DD": row(96.1, 81.1, 96.1, 54.0, "d"),
        "DA": row(96.1, 81.1, 96.1, 54.3, "d"),
        "AD": row(96.0, 81.0, 96.0, 54.1, "a"),
    }
    report = subject.compare_factorial(results)
    assert report["gate"]["pass"] is True
    assert abs(report["DA_minus_DD"]["non_time_delta"] - 0.075) < 1e-12


def test_parse_arms_requires_complete_factorial() -> None:
    try:
        subject.parse_arms(["AA=/a.zip", "DD=/d.zip"])
    except ValueError as error:
        assert "requires exactly" in str(error)
    else:
        raise AssertionError("incomplete factorial was accepted")
