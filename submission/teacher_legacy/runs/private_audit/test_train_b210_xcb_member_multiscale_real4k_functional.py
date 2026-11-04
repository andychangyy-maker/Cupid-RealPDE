from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_xcb_member_multiscale_real4k_functional.py"
SPEC = importlib.util.spec_from_file_location("functional_real4k", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def comparison(rel=0.02, tke=0.02, mvpe=0.01, sps=0.04):
    delta = {
        "rel_l2_score": rel,
        "tke_rel_l2_score": tke,
        "mvpe_rel_l2_score": mvpe,
        "native_sps_score": sps,
        "point_score": 0.0,
        "native_sps_coverage": 0.0,
        "native_sps_mean_width": 0.0,
    }
    return {
        "delta": delta,
        "by_aoa": {
            str(aoa): {"delta": dict(delta)}
            for aoa in MODULE.base.dr1.AOA_VALUES},
    }


def test_official_weights_and_release_gate_are_explicit():
    point, non_time = MODULE.official_deltas(comparison()["delta"])
    assert point == 0.46 * 0.02 + 0.10 * 0.02 + 0.09 * 0.01
    assert non_time == point + 0.25 * 0.04
    gate = MODULE.release_gate(comparison(), {})
    assert gate["pass"] is True
    assert gate["positive_aoas"] == 5


def test_gate_rejects_sps_only_or_aoa_local_signal():
    only_sps = MODULE.release_gate(
        comparison(rel=0.0, tke=0.0, mvpe=0.0, sps=0.20), {})
    assert only_sps["checks"]["official_point"] is False
    local = comparison()
    local["by_aoa"]["0"]["delta"]["rel_l2_score"] = -1.0
    local["by_aoa"]["5"]["delta"]["rel_l2_score"] = -1.0
    gate = MODULE.release_gate(local, {})
    assert gate["checks"]["aoa_count"] is False
