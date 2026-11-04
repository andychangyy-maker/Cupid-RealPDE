from __future__ import annotations

from pathlib import Path
import sys


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_multiscale_energy_real4k_rehearsal as subject  # noqa: E402


def _comparison(point=0.06, tke=0.16, rel=0.0, mvpe=0.0, sps=0.01):
    delta = {
        "point_score": point,
        "tke_rel_l2_score": tke,
        "rel_l2_score": rel,
        "mvpe_rel_l2_score": mvpe,
        "native_sps_score": sps,
        "native_sps_coverage": 0.0,
        "native_sps_mean_width": 0.0,
    }
    by_aoa = {
        str(aoa): {
            "delta": dict(delta),
            "candidate_non_time_proxy_delta": 0.01,
        }
        for aoa in subject.dr1.AOA_VALUES
    }
    return {"delta": delta, "by_aoa": by_aoa}


def test_release_gate_requires_reference_and_old_absolute_signal():
    passed = subject.release_gate(_comparison(), _comparison())
    assert passed["pass"] is True
    assert passed["decision"].startswith("PASS_")

    weak_old = _comparison(tke=0.149)
    stopped = subject.release_gate(_comparison(), weak_old)
    assert stopped["pass"] is False
    assert stopped["checks"]["old_tke"] is False


def test_release_gate_keeps_aoa_count_and_floor_hard():
    reference = _comparison()
    reference["by_aoa"]["0"]["candidate_non_time_proxy_delta"] = -0.01
    reference["by_aoa"]["5"]["candidate_non_time_proxy_delta"] = -0.01
    stopped = subject.release_gate(reference, _comparison())
    assert stopped["checks"]["aoa_count"] is False

    reference = _comparison()
    reference["by_aoa"]["20"]["delta"]["point_score"] = -0.061
    stopped = subject.release_gate(reference, _comparison())
    assert stopped["checks"]["aoa_floor"] is False


def test_contract_fixes_one_rehearsal_weight_and_no_inference_change():
    assert subject.REHEARSAL_WEIGHT == 0.05
    assert subject.SIM_STRIDE == 20
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert source.count("rehearsal.backward_sim_rehearsal(") == 1
    assert '"inference_graph_changed": False' in source
    assert "Real backward -> Sim backward -> clip -> optimizer -> EMA" in source
