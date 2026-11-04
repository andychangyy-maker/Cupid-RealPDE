from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/select_b210_dual_decoder_standard24k.py"
SPEC = importlib.util.spec_from_file_location("select_dual_24k", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def arm(rel=0.0, tke=0.0, mvpe=0.0, sps=0.0):
    return {"metrics": {
        "rel_l2_score": rel,
        "tke_rel_l2_score": tke,
        "mvpe_rel_l2_score": mvpe,
        "native_sps_score": sps,
        "native_sps_coverage": 0.85,
        "native_sps_mean_width": 0.07,
    }}


def gates(max_abs=0.2):
    row = {"raw_finite": True, "max_abs": max_abs}
    return {
        "fluctuation_role_gates": dict(row),
        "uq_fluctuation_role_gates": dict(row),
    }


def test_official_weights_and_pass_gate():
    control = arm()
    candidate = arm(rel=0.01, tke=0.02, mvpe=0.01, sps=0.20)
    result = MODULE.select(control, candidate, gates())
    expected_point = 0.46 * 0.01 + 0.10 * 0.02 + 0.09 * 0.01
    assert result["official_point_delta"] == expected_point
    assert result["official_non_time_delta"] == expected_point + 0.25 * 0.20
    assert result["pass"] is True


def test_sps_component_and_saturation_are_hard_gates():
    control = arm()
    assert MODULE.select(
        control, arm(rel=0.1, tke=0.1, mvpe=0.1, sps=0.09),
        gates())["checks"]["native_sps"] is False
    assert MODULE.select(
        control, arm(rel=-0.05, tke=0.5, mvpe=0.5, sps=0.5),
        gates())["checks"]["components"] is False
    assert MODULE.select(
        control, arm(rel=0.1, tke=0.1, mvpe=0.1, sps=0.5),
        gates(0.96))["checks"]["gates_not_saturated"] is False
