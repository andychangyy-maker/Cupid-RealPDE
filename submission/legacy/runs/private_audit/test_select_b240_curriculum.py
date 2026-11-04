from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/select_b240_curriculum.py"
SPEC = importlib.util.spec_from_file_location("select_curriculum", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def summary(rel: float, tke: float, mvpe: float) -> dict:
    return {"paired_candidate": {"terminal_anchor": {
        "rel_l2_score": rel,
        "tke_rel_l2_score": tke,
        "mvpe_rel_l2_score": mvpe,
    }}}


def test_selects_dense_only_above_margin_and_component_floor():
    result = MODULE.decide(summary(1.2, 1.2, 1.2), summary(1.0, 1.0, 1.0))
    assert result["selected_curriculum"] == "dense_stride10_then_standard"
    assert abs(result["phase_value_point65"] - 0.13) < 1e-12
    unsafe = MODULE.decide(summary(1.3, 0.9, 1.3), summary(1.0, 1.0, 1.0))
    assert unsafe["selected_curriculum"] == "standard_then_standard"


def test_neutral_defaults_to_simpler_standard_curriculum():
    result = MODULE.decide(summary(1.01, 1.01, 1.01), summary(1.0, 1.0, 1.0))
    assert result["selected_curriculum"] == "standard_then_standard"
