from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "code/training"
sys.path.insert(0, str(TRAINING))
PATH = TRAINING / "train_b210_condition_held_real.py"
SPEC = importlib.util.spec_from_file_location("condition_held_real", PATH)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


class Window:
    def __init__(self, trajectory, start):
        self.trajectory = trajectory
        self.start = start


def test_hierarchical_schedule_balances_aoa_then_trajectory():
    metadata = [
        {"aoa": 0, "re": 3000},
        {"aoa": 0, "re": 4000},
        {"aoa": 5, "re": 3000},
        {"aoa": 5, "re": 4000},
    ]
    windows = [
        Window(trajectory, start)
        for trajectory in range(4) for start in (0, 5, 10)
    ]
    result = SUBJECT.hierarchical_batch_schedule(
        windows, metadata, updates=10, seed=3, batch_size=8)
    assert result["aoa_draw_counts"] == {"0": 40, "5": 40}
    assert len(result["batches"]) == 10
    assert all(len(batch) == 8 for batch in result["batches"])
    assert sum(result["trajectory_draw_counts"].values()) == 80


def test_endpoint_contract_restores_generic_trainer_globals():
    names = (
        "EXPERIMENT", "ENDPOINT_SCHEMA", "PRIMARY_ENDPOINT", "UPDATES",
        "SCHEDULER_T_MAX", "HELD_AOA")
    before = {name: getattr(SUBJECT.small4k, name) for name in names}
    with SUBJECT.endpoint_contract("dense12k", 20):
        assert SUBJECT.small4k.EXPERIMENT == SUBJECT.EXPERIMENT
        assert SUBJECT.small4k.UPDATES == 12_000
        assert SUBJECT.small4k.HELD_AOA == 20
    assert {name: getattr(SUBJECT.small4k, name) for name in names} == before


def test_metric_comparison_uses_published_final_weights():
    control = {
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 70.0,
        "native_sps_score": 40.0,
    }
    candidate = {key: value + 1.0 for key, value in control.items()}
    result = SUBJECT.metric_comparison(candidate, control)
    assert result["point_final_contribution_delta"] == 0.65
    assert result["non_time_final_contribution_delta"] == 0.90


def test_full_loao_gate_requires_three_negative_folds(tmp_path, monkeypatch):
    rows = []
    for held_aoa, delta in zip(SUBJECT.AOA_VALUES, (-1.0, -1.0, 1.0, 1.0, 1.0)):
        path = tmp_path / f"held{held_aoa}.json"
        row = {
            "schema": SUBJECT.SCHEMA,
            "experiment": SUBJECT.EXPERIMENT,
            "held_aoa": held_aoa,
            "arms": {
                "dense4k": {"metric_state": {"arm": "control"}},
                "dense12k": {"metric_state": {"arm": "candidate"}},
            },
            "comparisons": {
                "dense12k_minus_dense4k": {
                    "non_time_final_contribution_delta": delta,
                },
            },
        }
        path.write_text(__import__("json").dumps(row), encoding="utf-8")
        rows.append((str(path), SUBJECT.dr1.sha256(path)))

    control = {"rel_l2_score": 90.0, "tke_rel_l2_score": 80.0,
               "mvpe_rel_l2_score": 70.0, "native_sps_score": 40.0}
    candidate = {key: value - 1.0 for key, value in control.items()}
    monkeypatch.setattr(
        SUBJECT.smu1, "merge_metric_states", lambda states: states[0])
    monkeypatch.setattr(
        SUBJECT.smu1, "finalize_metric_state",
        lambda state: control if state["arm"] == "control" else candidate)
    result = SUBJECT.aggregate_results(rows)
    ordering = result["known_official_ordering_check"]
    assert ordering["direction_matches_official"]
    assert ordering["negative_folds"] == 2
    assert ordering["fold_direction_pass"] is False
    assert ordering["validation_gate_pass"] is False
    assert result["proxy_decision"] == "STOP_REAL_STAGE_CONDITION_HELD_DIRECTION"


def test_balanced_gate_uses_preregistered_pooled_fold_and_worst_limits(
        tmp_path, monkeypatch):
    json = __import__("json")
    metrics = {
        "rel_l2_score": 90.0, "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 70.0, "native_sps_score": 40.0,
    }
    candidate_metrics = {key: value + 1.0 for key, value in metrics.items()}
    pairs = []
    fold_deltas = (0.1, 0.1, 0.1, 0.1, -0.04)
    for held_aoa, delta in zip(SUBJECT.AOA_VALUES, fold_deltas):
        control_path = tmp_path / f"control{held_aoa}.json"
        control_row = {
            "schema": SUBJECT.SCHEMA,
            "experiment": SUBJECT.EXPERIMENT,
            "held_aoa": held_aoa,
            "manifest": {"semantic_sha256": "a" * 64},
            "normalizer": {"statistics_sha256": "b" * 64},
            "start": {"sha256": "c" * 64},
            "arms": {"dense4k": {
                "metrics": metrics,
                "metric_state": {"arm": "control"},
                "target_audit": {"held": held_aoa},
            }},
        }
        control_path.write_text(json.dumps(control_row), encoding="utf-8")
        control_sha = SUBJECT.dr1.sha256(control_path)
        fold_path = tmp_path / f"balanced{held_aoa}.json"
        fold_row = {
            "schema": SUBJECT.BALANCED_SCHEMA,
            "experiment": SUBJECT.EXPERIMENT,
            "held_aoa": held_aoa,
            "control": {
                "path": str(control_path),
                "sha256": control_sha,
                "metrics": metrics,
                "metric_state": {"arm": "control"},
                "target_audit": {"held": held_aoa},
            },
            "arm": {"metric_state": {"arm": "candidate"}},
            "comparison": {"non_time_final_contribution_delta": delta},
        }
        fold_path.write_text(json.dumps(fold_row), encoding="utf-8")
        pairs.append((str(fold_path), SUBJECT.dr1.sha256(fold_path)))

    monkeypatch.setattr(
        SUBJECT.smu1, "merge_metric_states", lambda states: states[0])
    monkeypatch.setattr(
        SUBJECT.smu1, "finalize_metric_state",
        lambda state: metrics if state["arm"] == "control" else candidate_metrics)
    result = SUBJECT.aggregate_balanced_results(pairs)
    assert result["gate"] == {
        "thresholds": SUBJECT.BALANCED_GATE,
        "exact_full_loao": True,
        "pooled_pass": True,
        "positive_folds": 4,
        "positive_folds_pass": True,
        "worst_fold": -0.04,
        "worst_fold_pass": True,
        "validation_gate_pass": True,
    }
    assert result["proxy_decision"] == "PASS_BALANCED4K_CONDITION_HELD_GATE"
