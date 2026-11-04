from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "code/training"
sys.path.insert(0, str(TRAINING))
SCRIPT = TRAINING / "train_b210_full_released_real_support.py"
SPEC = importlib.util.spec_from_file_location("b210_full_real_support", SCRIPT)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def _write_h5(path: Path, *, frames: int, re: int, aoa: int) -> None:
    values = np.zeros((frames, 8, 8), dtype=np.float32)
    values[45:] = 123.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=values)
        handle.create_dataset("v", data=-values)
        handle.create_dataset("re", data=re)
        handle.create_dataset("aoa", data=aoa)


def _row(filename: str, *, aoa: int, frames: int = 55) -> dict:
    return {
        "file": filename,
        "frames": frames,
        "re": 4_000,
        "aoa": aoa,
        "val_boundary": 45,
        "train_starts": [0],
    }


def _seal(path: Path) -> dict:
    path.write_bytes(b"sealed")
    return {
        "path": str(path),
        "file_sha256": SUBJECT.dr1.sha256(path),
        "model_state_sha256": "a" * 64,
    }


def test_support_windows_differ_only_after_prefix_boundary():
    trajectory = torch.zeros(55, 4, 4, 2)
    trajectory[45:] = 123.0
    metadata = [{
        "file": "fit.h5", "frames": 55, "prefix_frames": 45,
        "re": 4_000, "aoa": 0,
    }]
    prefix = SUBJECT.support_windows(
        [trajectory], metadata, support="prefix", stride=5)
    full = SUBJECT.support_windows(
        [trajectory], metadata, support="full", stride=5)
    assert [window.start for window in prefix] == [0, 5]
    assert [window.start for window in full] == [0, 5, 10, 15]
    assert all(window.start + 40 <= 45 for window in prefix)
    assert any(window.start + 40 > 45 for window in full)


def test_sparse_prefix_normalizer_is_insensitive_to_full_tail():
    first = torch.zeros(55, 4, 4, 2)
    second = first.clone()
    first[45:] = 10.0
    second[45:] = 10_000.0
    windows = SUBJECT.manifest_prefix_windows([_row("fit.h5", aoa=0)])
    stats = []
    for trajectory in (first, second):
        sufficient = SUBJECT.smu1.window_sufficient_statistics(
            [trajectory], windows)
        stats.append(SUBJECT.smu1.fold_statistics(sufficient, [0]))
    assert all(torch.equal(left, right)
               for left, right in zip(stats[0], stats[1]))


def test_full_loader_reads_tail_but_never_held_before_four_seals(tmp_path):
    _write_h5(tmp_path / "fit.h5", frames=55, re=4_000, aoa=0)
    _write_h5(tmp_path / "held.h5", frames=55, re=4_000, aoa=10)
    fit = _row("fit.h5", aoa=0)
    held = _row("held.h5", aoa=10)
    audit = SUBJECT.FullSupportAccessAudit(held_aoa=10)
    trajectories, metadata = SUBJECT.load_full_rows(
        tmp_path, [fit], "fit", audit)
    assert trajectories[0].shape[0] == 55
    assert torch.equal(
        trajectories[0][45:, ..., 0],
        torch.full_like(trajectories[0][45:, ..., 0], 123.0))
    assert metadata[0]["prefix_frames"] == 45
    with pytest.raises(RuntimeError, match="four endpoint seals"):
        SUBJECT.load_full_rows(tmp_path, [held], "held", audit)
    for arm in SUBJECT.ARM_ORDER:
        for endpoint in SUBJECT.ENDPOINT_ORDER:
            seal = _seal(tmp_path / f"{arm}_{endpoint}.pth")
            audit.seal(arm, endpoint, seal)
    held_trajectories, _ = SUBJECT.load_full_rows(
        tmp_path, [held], "held", audit)
    assert held_trajectories[0].shape[0] == 55


def test_held_open_rejects_endpoint_drift(tmp_path):
    audit = SUBJECT.FullSupportAccessAudit(held_aoa=20)
    seals = {}
    for arm in SUBJECT.ARM_ORDER:
        for endpoint in SUBJECT.ENDPOINT_ORDER:
            key = SUBJECT.endpoint_key(arm, endpoint)
            seals[key] = _seal(tmp_path / f"{arm}_{endpoint}.pth")
            audit.seal(arm, endpoint, seals[key])
    Path(seals[SUBJECT.endpoint_key("full", SUBJECT.PRIMARY_ENDPOINT)][
        "path"]).write_bytes(b"drifted")
    with pytest.raises(RuntimeError, match="drifted before held open"):
        audit.before_open(_row("held.h5", aoa=20), "held")


def test_excluded_duplicate_is_rejected_before_h5_open(tmp_path):
    audit = SUBJECT.FullSupportAccessAudit(held_aoa=10)
    row = _row(SUBJECT.dr1.EXCLUDED_BASENAME, aoa=0)
    with pytest.raises(RuntimeError, match="excluded duplicate"):
        SUBJECT.load_full_rows(tmp_path, [row], "fit", audit)


def test_endpoint_contract_restores_existing_small4k_runner():
    names = (
        "EXPERIMENT", "ENDPOINT_SCHEMA", "PRIMARY_ENDPOINT", "UPDATES",
        "SCHEDULER_T_MAX", "HELD_AOA")
    before = {name: getattr(SUBJECT.small4k, name) for name in names}
    with SUBJECT.endpoint_contract(20):
        assert SUBJECT.small4k.EXPERIMENT == SUBJECT.EXPERIMENT
        assert SUBJECT.small4k.ENDPOINT_SCHEMA == SUBJECT.ENDPOINT_SCHEMA
        assert SUBJECT.small4k.PRIMARY_ENDPOINT == SUBJECT.PRIMARY_ENDPOINT
        assert SUBJECT.small4k.UPDATES == 4_000
        assert SUBJECT.small4k.SCHEDULER_T_MAX == 12_000
        assert SUBJECT.small4k.HELD_AOA == 20
    assert {name: getattr(SUBJECT.small4k, name) for name in names} == before


def test_contract_locks_pure_sim_start_and_no_clean716_interface():
    assert SUBJECT.CONTRACT["matched"][
        "start_checkpoint_sha256"] == SUBJECT.FORMAL_START_SHA256
    assert SUBJECT.CONTRACT["held_data"]["clean716_manifest_accepted"] is False
    assert "--clean716" not in SCRIPT.read_text(encoding="utf-8")


def test_metric_comparison_uses_published_46_10_9_25_weights():
    control = {
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 70.0,
        "native_sps_score": 40.0,
    }
    full = {key: value + 1.0 for key, value in control.items()}
    comparison = SUBJECT.metric_comparison(full, control)
    assert comparison["point_final_contribution_delta"] == pytest.approx(0.65)
    assert comparison["non_time_final_contribution_delta"] == pytest.approx(0.90)
    assert comparison["time_held_constant_not_estimated"] is True


def test_aggregate_merges_states_and_has_no_score_gate(tmp_path, monkeypatch):
    pairs = []
    for held_aoa in SUBJECT.HELD_AOAS:
        path = tmp_path / f"held{held_aoa}.json"
        endpoints = {}
        for arm in SUBJECT.ARM_ORDER:
            endpoints[arm] = {"endpoints": {}}
            for endpoint in SUBJECT.ENDPOINT_ORDER:
                endpoints[arm]["endpoints"][endpoint] = {
                    "metric_state": {
                        "arm": arm, "endpoint": endpoint,
                        "held": held_aoa,
                    }}
        row = {
            "schema": SUBJECT.SCHEMA,
            "experiment": SUBJECT.EXPERIMENT,
            "contract_sha256": SUBJECT.CONTRACT_SHA256,
            "held_aoa": held_aoa,
            "start": {"checkpoint_sha256": "b" * 64},
            "manifest": {
                "file_sha256": SUBJECT.ft12.FORMAL_MANIFEST_FILE_SHA256},
            "counts": {
                "observed_held_full_windows":
                    SUBJECT.EXPECTED_FOLD_COUNTS[held_aoa][
                        "held_full_windows"]},
            "arms": endpoints,
        }
        path.write_text(json.dumps(row), encoding="utf-8")
        pairs.append((str(path), SUBJECT.dr1.sha256(path)))

    monkeypatch.setattr(
        SUBJECT.smu1, "merge_metric_states",
        lambda states: {"arm": states[0]["arm"],
                        "endpoint": states[0]["endpoint"]})
    base = {
        "rel_l2_score": 90.0, "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 70.0, "native_sps_score": 40.0,
    }
    monkeypatch.setattr(
        SUBJECT.smu1, "finalize_metric_state",
        lambda state: ({key: value + 1.0 for key, value in base.items()}
                       if state["arm"] == "full" else dict(base)))
    result = SUBJECT.aggregate_results(pairs)
    assert result["held_aoas"] == [10, 20]
    assert result["held_full_windows"] == 1_399
    assert result["primary_comparison"][
        "non_time_final_contribution_delta"] == pytest.approx(0.90)
    assert result["score_gate"] is None
    assert result["decision"].endswith("NO_SCORE_GATE")
