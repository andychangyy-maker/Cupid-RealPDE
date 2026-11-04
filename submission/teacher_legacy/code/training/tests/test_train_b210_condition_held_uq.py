from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "code/training"
sys.path.insert(0, str(TRAINING))
PATH = TRAINING / "train_b210_condition_held_uq.py"
SPEC = importlib.util.spec_from_file_location("condition_held_uq", PATH)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def test_sps_comparison_preserves_point_and_uses_published_weight():
    control = {
        "rel_l2_score": 90.0, "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 70.0, "native_sps_score": 40.0,
    }
    candidate = dict(control, native_sps_score=41.0)
    result = SUBJECT.sps_comparison(candidate, control)
    assert result["native_sps_delta"] == 1.0
    assert result["non_time_final_contribution_delta"] == 0.25
    assert result["point_metrics_bitwise"] is True


def test_point_replay_gate_uses_point_weight_not_native_sps():
    frozen = {
        "rel_l2_score": 90.0, "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 70.0, "native_sps_score": 40.0,
    }
    current = dict(frozen, rel_l2_score=90.1, native_sps_score=20.0)
    result = SUBJECT.point_replay_check(current, frozen)
    assert abs(result["absolute_point_final_contribution"] - 0.046) < 1e-12
    assert result["pass"] is True
    assert result["native_sps_not_used_for_point_replay_identity"] is True


def test_full_uq_gate_reuses_existing_stability_thresholds(tmp_path, monkeypatch):
    base_metrics = {
        "rel_l2_score": 90.0, "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 70.0, "native_sps_score": 40.0,
    }
    candidate_metrics = dict(base_metrics, native_sps_score=40.2)
    base_state = {
        "arm": "baseline", "samples": 1, "rel_error_sum": 1.0,
        "tke_error_sum": 1.0, "mvpe_error_sum": 1.0,
    }
    candidate_state = dict(base_state, arm="candidate")
    pairs = []
    for held_aoa, delta in zip(SUBJECT.held.AOA_VALUES,
                               (0.2, 0.2, 0.2, 0.2, -0.02)):
        control_path = tmp_path / f"control{held_aoa}.json"
        control_row = {
            "schema": SUBJECT.held.SCHEMA,
            "experiment": SUBJECT.held.EXPERIMENT,
            "held_aoa": held_aoa,
            "manifest": {"semantic_sha256": "a" * 64},
            "normalizer": {"statistics_sha256": "b" * 64},
            "start": {"sha256": "c" * 64},
            "arms": {"dense4k": {
                "metrics": base_metrics,
                "metric_state": base_state,
                "target_audit": {"held": held_aoa},
            }},
        }
        control_path.write_text(json.dumps(control_row), encoding="utf-8")
        control_sha = SUBJECT.dr1.sha256(control_path)
        fold_path = tmp_path / f"uq{held_aoa}.json"
        fold_row = {
            "schema": SUBJECT.FOLD_SCHEMA,
            "experiment": SUBJECT.EXPERIMENT,
            "held_aoa": held_aoa,
            "control": {
                "path": str(control_path), "sha256": control_sha,
                "metrics": base_metrics, "metric_state": base_state,
                "target_audit": {"held": held_aoa},
            },
            "uq": {
                "baseline_metric_state": base_state,
                "candidate_metric_state": candidate_state,
            },
            "point_replay_identity": {
                "sample_stream_matches_frozen_control": True,
            },
            "point_replay_quality_check": {
                "pass": True,
                "absolute_point_final_contribution": 0.01,
            },
            "comparison": {"native_sps_delta": delta},
        }
        fold_path.write_text(json.dumps(fold_row), encoding="utf-8")
        pairs.append((str(fold_path), SUBJECT.dr1.sha256(fold_path)))

    monkeypatch.setattr(
        SUBJECT.smu1, "merge_metric_states", lambda states: states[0])
    monkeypatch.setattr(
        SUBJECT.smu1, "finalize_metric_state",
        lambda state: (base_metrics if state["arm"] == "baseline"
                       else candidate_metrics))
    result = SUBJECT.aggregate_results(pairs)
    assert result["gate"]["validation_gate_pass"] is True
    assert result["gate"]["positive_folds"] == 4
    assert result["gate"]["worst_fold"] == -0.02
    assert result["decision"] == "PASS_CONDITION_HELD_UQ_ONLY_GATE"
