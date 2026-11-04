from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import pytest


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from train_s3a_point_pilot import (
    FORMAL_COUNTS,
    FORMAL_INPUTS,
    decide_gate,
    expected_smoke_errors,
    select_held_fold_windows,
    target_model_config,
)


@dataclass(frozen=True)
class Window:
    trajectory: int
    start: int


def test_select_held_fold_windows_excludes_held_training_trajectories() -> None:
    metadata = [{"file": "a.h5"}, {"file": "b.h5"}, {"file": "c.h5"}]
    assignments = {"a.h5": 4, "b.h5": 1, "c.h5": 4}
    train = [Window(0, 0), Window(1, 0), Window(2, 0)]
    val = [Window(0, 20), Window(1, 20), Window(2, 20)]

    selected_train, held_train, selected_val = select_held_fold_windows(
        metadata, train, val, assignments, held_fold=4)

    assert selected_train == [Window(1, 0)]
    assert held_train == [Window(0, 0), Window(2, 0)]
    assert selected_val == [Window(0, 20), Window(2, 20)]


def test_select_held_fold_windows_fails_on_assignment_drift() -> None:
    with pytest.raises(ValueError, match="do not match trajectories"):
        select_held_fold_windows(
            [{"file": "a.h5"}], [Window(0, 0)], [Window(0, 20)],
            {"different.h5": 4}, held_fold=4)


def test_target_model_config_changes_only_mean_profile_branch() -> None:
    source = {
        "architecture": "ensemble_unet",
        "base": 128,
        "mean_profile_channels": 0,
        "multi_scale_regime_film": False,
        "use_regime_film": True,
    }

    target = target_model_config(source)

    assert target["mean_profile_channels"] == 64
    assert target["multi_scale_regime_film"] is False
    assert target["use_regime_film"] is True
    assert {key for key in target if target[key] != source.get(key)} == {
        "mean_profile_channels"}


def test_gate_requires_all_three_registered_conditions() -> None:
    passing = {
        "point_score": 0.010,
        "tke_rel_l2_score": -0.050,
        "rel_l2_score": 0.010,
        "mvpe_rel_l2_score": -0.001,
    }
    assert decide_gate(passing)["passed"]
    assert not decide_gate({**passing, "point_score": 0.0099})["passed"]
    assert not decide_gate({**passing, "tke_rel_l2_score": -0.0501})["passed"]
    assert not decide_gate({
        **passing, "rel_l2_score": 0.0099, "mvpe_rel_l2_score": 0.0099,
    })["passed"]


def test_expected_smoke_must_match_full_formal_contract(tmp_path: Path) -> None:
    initial = {
        "rel_l2_score": 1.0,
        "tke_rel_l2_score": 2.0,
        "mvpe_rel_l2_score": 3.0,
        "point_score": 4.0,
    }
    data = dict(FORMAL_COUNTS)
    cache_sha = "a" * 64
    revision = "b" * 40
    scorer_sha = "c" * 64
    payload = {
        "status": "SMOKE_COMPLETE_NO_SCIENTIFIC_GATE",
        "completed_iteration": 20,
        "training_config": {
            "smoke": True,
            "formal": False,
            "updates": 20,
            "eval_interval": 20,
            "time_budget_seconds": 300.0,
            "held_fold": 4,
        },
        "counts": data,
        "input_hashes": FORMAL_INPUTS,
        "structure_change": {
            "initialization_identity": {
                "bitwise_equal": True,
                "max_abs_diff": 0.0,
            },
        },
        "teacher_cache_sha256": cache_sha,
        "code_revision": revision,
        "official_scorer_sha256": scorer_sha,
        "initial_metrics": initial,
    }
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert expected_smoke_errors(
        initial, path, data, FORMAL_INPUTS,
        cache_sha, revision, scorer_sha) == []

    payload["status"] = "STOP_TIME_BUDGET"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "not a successful smoke" in " ".join(expected_smoke_errors(
        initial, path, data, FORMAL_INPUTS,
        cache_sha, revision, scorer_sha))
