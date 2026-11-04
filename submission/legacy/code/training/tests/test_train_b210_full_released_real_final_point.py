from __future__ import annotations

from dataclasses import asdict
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
SCRIPT = TRAINING / "train_b210_full_released_real_final_point.py"
SPEC = importlib.util.spec_from_file_location("b210_final_full_point", SCRIPT)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def _partition(total: int, count: int) -> list[int]:
    base, remainder = divmod(total, count)
    return [base + (index < remainder) for index in range(count)]


def _exact_rows() -> list[dict]:
    rows = []
    for aoa, trajectories in SUBJECT.EXPECTED_TRAJECTORIES_BY_AOA.items():
        full_counts = _partition(
            SUBJECT.EXPECTED_FULL_WINDOWS_BY_AOA[aoa], trajectories)
        sparse_counts = _partition(
            SUBJECT.EXPECTED_NORMALIZER_WINDOWS_BY_AOA[aoa], trajectories)
        for index, (full_count, sparse_count) in enumerate(
                zip(full_counts, sparse_counts)):
            frames = 40 + 5 * (full_count - 1)
            starts = [20 * value for value in range(sparse_count)]
            assert starts[-1] + 40 <= frames
            rows.append({
                "file": f"aoa{aoa}_{index}.h5",
                "frames": frames,
                "re": 4_000 + index,
                "aoa": aoa,
                "val_boundary": starts[-1] + 40,
                "train_starts": starts,
            })
    return rows


def _write_h5(path: Path, row: dict) -> None:
    values = np.zeros((row["frames"], 8, 8), dtype=np.float32)
    values[row["val_boundary"]:] = 17.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=values)
        handle.create_dataset("v", data=-values)
        handle.create_dataset("re", data=row["re"])
        handle.create_dataset("aoa", data=row["aoa"])


def _model_config() -> dict:
    return {
        "architecture": "ensemble_unet",
        **asdict(SUBJECT.sim_tke.b210_config("candidate")),
    }


def _normalizer_report() -> dict:
    values = {
        "mean_input": [0.1, 0.2, 0.0],
        "mean_target": [0.1, 0.2, 0.0],
        "std_input": [1.0, 1.0, 1.0],
        "std_target": [1.0, 1.0, 1.0],
    }
    return {"values": values, "sha256": SUBJECT.dr1.canonical_sha256(values)}


def _data_seals() -> tuple[dict, dict]:
    manifest = SUBJECT.make_manifest_seal({
        "semantic_sha256": SUBJECT.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256,
        "windows": SUBJECT.EXPECTED_NORMALIZER_WINDOWS,
        "excluded": SUBJECT.dr1.EXCLUDED_BASENAME,
    }, SUBJECT.validate_released_inventory(_exact_rows()))
    access = SUBJECT._with_sha256({
        "schema": SUBJECT.ACCESS_SEAL_SCHEMA,
        "opened_file_allowlist": [row["file"] for row in _exact_rows()],
        "clean716_manifest_read": False,
    })
    return manifest, access


def test_contract_freezes_exact_final_recipe_and_has_no_clean716_cli():
    assert SUBJECT.CONTRACT["start"][
        "checkpoint_sha256"] == SUBJECT.FORMAL_START_SHA256
    assert SUBJECT.CONTRACT["data"]["trajectories"] == 81
    assert SUBJECT.CONTRACT["data"]["optimizer_windows"] == 13_202
    assert SUBJECT.CONTRACT["data"]["normalizer_windows"] == 2_544
    assert SUBJECT.CONTRACT["data"]["clean716_manifest_accepted"] is False
    assert SUBJECT.CONTRACT["training"] == {
        "updates": 4_000,
        "batch_size": 8,
        "seed": 4,
        "optimizer": "AdamW",
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": 12_000,
        "ema": "FP32 EMA999 from update 0 through update 4000",
        "ema_decay": 0.999,
        "loss": (
            "unchanged native Real joint point/member/quantile/soft-SPS "
            "plus controller-only early TKE auxiliary"),
        "augmentation": SUBJECT.AUGMENTATION,
        "endpoint_selection_allowed": False,
    }
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'add_argument("--clean716' not in source
    assert 'add_argument("--evaluation-manifest' not in source
    assert SUBJECT.validate_training_recipe() == {
        "updates": 4_000, "batch_size": 8, "seed": 4, "lr": 2e-4,
        "weight_decay": 1e-5, "grad_clip": 1.0,
        "scheduler_t_max": 12_000, "ema_decay": 0.999,
    }


def test_exact_inventory_is_81_unique_full13202_and_sparse2544():
    rows = _exact_rows()
    inventory = SUBJECT.validate_released_inventory(rows)
    assert inventory["trajectories"] == inventory["unique_files"] == 81
    assert inventory["full_optimizer_windows"] == 13_202
    assert inventory["normalizer_windows"] == 2_544
    assert inventory["full_optimizer_windows_by_aoa"] == {
        0: 2_604, 5: 2_249, 10: 2_988, 15: 2_822, 20: 2_539}
    assert inventory["excluded_present"] is False


def test_manifest_seal_survives_json_round_trip():
    manifest, _ = _data_seals()
    expected_keys = {"0", "5", "10", "15", "20"}
    for field in SUBJECT.PER_AOA_INVENTORY_FIELDS:
        assert set(manifest["inventory"][field]) == expected_keys
        assert all(isinstance(key, str)
                   for key in manifest["inventory"][field])
    round_tripped = json.loads(json.dumps(manifest))
    assert round_tripped == manifest
    SUBJECT._validate_self_seal(
        round_tripped, "round-tripped manifest seal")


@pytest.mark.parametrize("failure", ("duplicate", "excluded"))
def test_inventory_rejects_duplicate_or_excluded_file(failure):
    rows = _exact_rows()
    if failure == "duplicate":
        rows[-1]["file"] = rows[0]["file"]
    else:
        rows[-1]["file"] = SUBJECT.dr1.EXCLUDED_BASENAME
    with pytest.raises(RuntimeError, match="inventory changed"):
        SUBJECT.validate_released_inventory(rows)


def test_access_reads_complete_allowlist_once_then_closes(tmp_path):
    row = {
        "file": "fit.h5", "frames": 55, "re": 4_000, "aoa": 0,
        "val_boundary": 45, "train_starts": [0],
    }
    _write_h5(tmp_path / row["file"], row)
    audit = SUBJECT.FinalFullAccessAudit()
    trajectories, metadata = SUBJECT.support.load_full_rows(
        tmp_path, [row], "fit", audit)
    assert trajectories[0].shape[0] == 55
    assert metadata[0]["prefix_frames"] == 45
    assert torch.equal(
        trajectories[0][45:, ..., 0],
        torch.full_like(trajectories[0][45:, ..., 0], 17.0))
    seal = audit.close_data([row])
    SUBJECT._validate_self_seal(seal, "test access")
    assert seal["complete_trajectory_arrays_read"] is True
    assert seal["clean716_manifest_read"] is False
    with pytest.raises(RuntimeError, match="after data access closed"):
        SUBJECT.support.load_full_rows(tmp_path, [row], "fit", audit)


def test_access_rejects_excluded_duplicate_before_h5_open(tmp_path):
    row = {
        "file": SUBJECT.dr1.EXCLUDED_BASENAME,
        "frames": 55, "re": 4_000, "aoa": 0,
        "val_boundary": 45, "train_starts": [0],
    }
    audit = SUBJECT.FinalFullAccessAudit()
    with pytest.raises(RuntimeError, match="excluded duplicate"):
        SUBJECT.support.load_full_rows(tmp_path, [row], "fit", audit)


def test_endpoint_contract_restores_reused_trainer():
    names = (
        "EXPERIMENT", "ENDPOINT_SCHEMA", "PRIMARY_ENDPOINT", "UPDATES",
        "SCHEDULER_T_MAX", "HELD_AOA")
    before = {name: getattr(SUBJECT.small4k, name) for name in names}
    with SUBJECT.endpoint_contract():
        assert SUBJECT.small4k.EXPERIMENT == SUBJECT.EXPERIMENT
        assert SUBJECT.small4k.ENDPOINT_SCHEMA == SUBJECT.ENDPOINT_SCHEMA
        assert SUBJECT.small4k.HELD_AOA == SUBJECT.NO_HELD_AOA
        assert SUBJECT.small4k.UPDATES == 4_000
        assert SUBJECT.small4k.SCHEDULER_T_MAX == 12_000
    assert {name: getattr(SUBJECT.small4k, name) for name in names} == before


def test_support_aggregate_is_sha_bound_without_a_score_gate(tmp_path):
    path = tmp_path / "aggregate.json"
    row = {
        "schema": SUBJECT.support.AGGREGATE_SCHEMA,
        "experiment": SUBJECT.support.EXPERIMENT,
        "contract_sha256": SUBJECT.support.CONTRACT_SHA256,
        "held_aoas": list(SUBJECT.support.HELD_AOAS),
        "start_checkpoint_sha256": SUBJECT.FORMAL_START_SHA256,
        "held_full_windows": 1_399,
        "primary_endpoint": SUBJECT.PRIMARY_ENDPOINT,
        "raw_endpoint_role": "diagnostic_only_no_endpoint_selection",
        "primary_comparison": {
            "point_final_contribution_delta": 0.01,
            "non_time_final_contribution_delta": 0.02,
        },
        "score_gate": None,
        "decision": "COMPLETE_TWO_FOLD_MATCHED_EVIDENCE_NO_SCORE_GATE",
        "submission_authorized": False,
    }
    path.write_text(json.dumps(row), encoding="utf-8")
    report = SUBJECT.validate_support_aggregate(
        path, SUBJECT.dr1.sha256(path))
    assert report["primary_comparison"] == row["primary_comparison"]
    assert report["method_adoption_was_manual"] is True
    path.write_text(json.dumps({**row, "decision": "PASS"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity changed"):
        SUBJECT.validate_support_aggregate(path, SUBJECT.dr1.sha256(path))


def test_ema_rebind_preserves_state_and_adds_data_seals(tmp_path):
    state = {"weight": torch.tensor([1.25], dtype=torch.float32)}
    state_sha = SUBJECT.dr1.model_state_sha256(state)
    normalizer = _normalizer_report()
    manifest, access = _data_seals()
    start = {"checkpoint_sha256": SUBJECT.FORMAL_START_SHA256}
    stream_sha = "c" * 64
    path = tmp_path / f"{SUBJECT.PRIMARY_ENDPOINT}.pth"
    payload = {
        "schema": SUBJECT.ENDPOINT_SCHEMA,
        "experiment": SUBJECT.EXPERIMENT,
        "phase": "real",
        "iteration": SUBJECT.UPDATES,
        "endpoint": SUBJECT.PRIMARY_ENDPOINT,
        "primary": True,
        "held_aoa": SUBJECT.NO_HELD_AOA,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": _model_config(),
        "start": start,
        "normalizer": normalizer["values"],
        "normalizer_sha256": normalizer["sha256"],
        "sample_stream_sha256": stream_sha,
        "ema": {
            "schema": SUBJECT.small4k.ema_lib.EMA_SEAL_SCHEMA,
            "update_order": SUBJECT.small4k.ema_lib.UPDATE_ORDER_CONTRACT,
            "updates": SUBJECT.UPDATES,
            "decay": SUBJECT.EMA_DECAY,
            "shadow_dtype": "torch.float32",
            "materialized_model_state_sha256": state_sha,
        },
        "optimizer_state_saved": False,
    }
    torch.save(payload, path)
    precursor = {
        "path": str(path),
        "file_sha256": SUBJECT.dr1.sha256(path),
        "model_state_sha256": state_sha,
    }
    seal = SUBJECT.finalize_ema_endpoint(
        precursor, start_report=start, normalizer_report=normalizer,
        stream_sha256=stream_sha, manifest_seal=manifest,
        access_seal=access)
    rebound = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.equal(rebound["model_state_dict"]["weight"], state["weight"])
    assert rebound["training_scope"] == "all_clean81_full13202"
    assert rebound["manifest_seal"] == manifest
    assert rebound["access_seal"] == access
    report = SUBJECT.validate_endpoint(
        path, seal, endpoint=SUBJECT.PRIMARY_ENDPOINT, primary=True,
        manifest_seal=manifest, access_seal=access)
    assert report["strict_fp32_state"] is True
    assert report["strict_model_load"] is False


def test_seal_detects_mutation():
    seal = SUBJECT._with_sha256({"schema": "test", "value": 1})
    SUBJECT._validate_self_seal(seal, "test")
    seal["value"] = 2
    with pytest.raises(RuntimeError, match="canonical SHA changed"):
        SUBJECT._validate_self_seal(seal, "test")
