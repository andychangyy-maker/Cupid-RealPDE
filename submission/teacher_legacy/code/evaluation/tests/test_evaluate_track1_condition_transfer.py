from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "code/evaluation/evaluate_track1_condition_transfer.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("condition_transfer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_rows():
    rows = []
    for index in range(10):
        rows.append({
            "file": f"trajectory_{index}.h5",
            "frames": 100,
            "re": 3000 + 1000 * index,
            "aoa": 5 * (index % 5),
            "train_starts": [0, 20],
            "val_starts": [40],
            "val_boundary": 40,
        })
    return rows


def test_condition_partitions_hold_whole_trajectories_exactly_once():
    rows = synthetic_rows()
    labels, condition_manifest = MODULE.condition_labels(rows)
    assert len(labels) == 10
    partitions = MODULE.build_partitions(
        rows, condition_manifest["trajectory_assignments"])
    for scheme in partitions["schemes"].values():
        assert scheme["every_trajectory_held_exactly_once"] is True
        for fold in scheme["folds"]:
            assert not set(fold["fit_files"]) & set(fold["held_files"])
            assert fold["trajectory_overlap"] == []
            assert fold["held_prefix_windows"] > 0


def test_sufficient_statistics_exactly_reproduce_official_group_scores():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(4, 20, 3, 4, 3)).astype(np.float32)
    target[..., 2] = 0.0
    target[:, :, 0, 0, :2] = 0.0
    prediction = target + rng.normal(scale=0.1, size=target.shape).astype(np.float32)
    prediction[..., 2] = 0.0
    width = np.full_like(prediction, 0.2)
    lower = prediction - width
    upper = prediction + width

    stats = MODULE.sufficient_statistics(prediction, lower, upper, target)
    selected = np.asarray([0, 2, 3], dtype=np.int64)
    scored = MODULE.score_statistics(stats, selected)
    p = prediction[selected]
    t = target[selected]
    lo = lower[selected]
    hi = upper[selected]
    channels = MODULE.OFFICIAL.measured_channels(t)
    sps, coverage = MODULE.OFFICIAL.aggregate_sps(
        p, t, channels, lower=lo, upper=hi)
    assert np.isclose(scored["rel_l2_score"], MODULE.OFFICIAL.score_error(
        np.mean(MODULE.OFFICIAL.rel_l2_per_sample(p, t, channels))))
    assert np.isclose(scored["tke_score"], MODULE.OFFICIAL.score_error(
        np.mean(MODULE.OFFICIAL.tke_rel_l2_per_sample(p, t, channels))))
    assert np.isclose(scored["mvpe_score"], MODULE.OFFICIAL.score_error(
        np.mean(MODULE.OFFICIAL.mvpe_rel_l2_per_sample(p, t))))
    assert np.isclose(scored["sps_score"], MODULE.OFFICIAL.score_sps(sps))
    assert np.isclose(scored["coverage"], coverage)


def test_ranking_check_rejects_mismatched_local_proxy_without_retuning():
    models = {
        "a": {
            "pooled": {"non_time_final_contribution": 70.0},
            "condition_transfer_proxy": 72.0,
        },
        "b": {
            "pooled": {"non_time_final_contribution": 71.0},
            "condition_transfer_proxy": 69.0,
        },
    }
    official = {
        "a": {"non_time_final_contribution": 73.0},
        "b": {"non_time_final_contribution": 72.0},
    }
    report = MODULE.compare_rankings(models, official)
    assert report["official_non_time_rank"] == ["a", "b"]
    assert report["local_pooled_exact_rank_match"] is False
    assert report["condition_transfer_exact_rank_match"] is True
    assert "must not be repaired" in report["interpretation"]
