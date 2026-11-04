from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = (
    ROOT / "runs/private_audit"
    / "select_b210_dual_decoder_standard24k_real4k.py")
SPEC = importlib.util.spec_from_file_location("select_dual_24k_real", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def metrics(rel=0.0, tke=0.0, mvpe=0.0, sps=0.0):
    return {
        "rel_l2_score": rel,
        "tke_rel_l2_score": tke,
        "mvpe_rel_l2_score": mvpe,
        "native_sps_score": sps,
        "native_sps_coverage": 0.90,
        "native_sps_mean_width": 0.02,
    }


def arm(values, aoa_values=None):
    return {
        "metrics": values,
        "metrics_by_aoa": {
            str(aoa): dict(values if aoa_values is None else aoa_values[aoa])
            for aoa in (0, 5, 10, 15, 20)},
        "evaluation_manifest": {"sha256": "a", "tail_windows": 716},
        "normalizer_statistics_sha256": "b",
        "target_audit": {"sha256": "c"},
        "access_audit": {"tail_target_read_before_both_seals": False},
    }


def test_official_weights_and_pass_gate():
    control = arm(metrics())
    candidate = arm(metrics(rel=0.01, tke=0.02, mvpe=0.01, sps=0.20))
    result = MODULE.select(control, candidate)
    expected_point = 0.46 * 0.01 + 0.10 * 0.02 + 0.09 * 0.01
    assert result["official_point_delta"] == expected_point
    assert result["official_non_time_delta"] == expected_point + 0.25 * 0.20
    assert result["positive_aoas"] == 5
    assert result["pass"] is True


def test_identity_and_aoa_robustness_are_hard_gates():
    control = arm(metrics())
    candidate = arm(metrics(rel=0.1, tke=0.1, mvpe=0.1, sps=0.5))
    candidate["normalizer_statistics_sha256"] = "changed"
    result = MODULE.select(control, candidate)
    assert result["checks"]["identical_research_contract"] is False

    bad_aoa = {
        aoa: metrics(rel=(-0.20 if aoa in (0, 5, 10) else 0.1), sps=0.1)
        for aoa in (0, 5, 10, 15, 20)}
    candidate = arm(metrics(rel=0.1, tke=0.1, mvpe=0.1, sps=0.5), bad_aoa)
    result = MODULE.select(control, candidate)
    assert result["checks"]["positive_aoas"] is False
    assert result["checks"]["worst_aoa_point"] is False
