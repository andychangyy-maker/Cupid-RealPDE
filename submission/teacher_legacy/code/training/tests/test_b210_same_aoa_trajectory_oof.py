from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "code/training"
sys.path.insert(0, str(TRAINING))


def load(name: str):
    path = TRAINING / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PLAN = load("b210_same_aoa_trajectory_folds.py")
POINT = load("train_b210_same_aoa_point_oof_fold.py")
CACHE = load("build_b210_same_aoa_point_oof_residual_cache.py")
UQ = load("train_b210_same_aoa_trajectory_native_uq.py")


def grid_rows():
    return [
        {"file": f"{re}_{aoa}.h5", "frames": 40, "re": re, "aoa": aoa,
         "train_starts": [0], "val_boundary": 40}
        for re in (100, 200, 300, 400, 500)
        for aoa in PLAN.AOA_VALUES]


def test_established_formula_is_trajectory_disjoint_and_keeps_every_aoa(
        monkeypatch):
    rows = grid_rows()
    monkeypatch.setattr(PLAN, "validate_rows", lambda _rows: {})
    assignments = PLAN.assignment_rows(rows)
    assert len(assignments) == len(rows)
    assert len({row["file"] for row in assignments}) == len(rows)
    assert {row["fold"] for row in assignments} == set(range(5))
    for fold in range(5):
        held = [row for row in assignments if row["fold"] == fold]
        assert {row["aoa"] for row in held} == set(PLAN.AOA_VALUES)
    assert PLAN.FORMULA == "(rank(re)+2*rank(aoa)) % folds"


def test_frozen_counts_reconstruct_full_3341_inventory():
    assert sum(row["held_trajectories"]
               for row in PLAN.EXPECTED_FOLDS.values()) == 81
    assert sum(row["held_oof_windows"]
               for row in PLAN.EXPECTED_FOLDS.values()) == 3_341
    assert sum(PLAN.EXPECTED_TOTAL["normalizer_prefix_windows"]
               - row["normalizer_prefix_windows"]
               for row in PLAN.EXPECTED_FOLDS.values()) == 2_544
    assert sum(PLAN.EXPECTED_TOTAL["full_optimizer_windows"]
               - row["full_optimizer_windows"]
               for row in PLAN.EXPECTED_FOLDS.values()) == 13_202
    assert all(set(row["held_trajectories_by_aoa"])
               == set(PLAN.AOA_VALUES)
               for row in PLAN.EXPECTED_FOLDS.values())


def test_formal_manifest_assignment_digest_when_available():
    path = Path(
        "/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/"
        "runtime_train_manifest.json")
    if not path.is_file():
        pytest.skip("private formal clean81 manifest is not in this checkout")
    rows = json.loads(path.read_text(encoding="utf-8"))
    plan = PLAN.build_plan(rows)
    assert plan["assignment_sha256"] == PLAN.EXPECTED_ASSIGNMENT_SHA256
    assert all(row["counts"] == PLAN.EXPECTED_FOLDS[row["fold"]]
               for row in plan["fold_reports"])


def test_point_access_guard_rejects_held_and_post_seal(tmp_path):
    audit = POINT.FitOnlyAccessAudit(2, ["fit.h5"], ["held.h5"])
    audit.record_open({"file": "fit.h5", "aoa": 0}, "fit")
    with pytest.raises(RuntimeError, match="non-fit"):
        audit.before_open({"file": "held.h5", "aoa": 0}, "fit")
    endpoint = tmp_path / "endpoint.pth"
    endpoint.write_bytes(b"sealed")
    seal = {
        "endpoint": POINT.ENDPOINT, "primary": True,
        "iteration": POINT.UPDATES, "path": str(endpoint),
        "file_sha256": POINT.dr1.sha256(endpoint),
    }
    audit.seal(seal)
    with pytest.raises(RuntimeError, match="non-fit"):
        audit.before_open({"file": "fit.h5", "aoa": 0}, "fit")
    report = audit.report()
    assert report["allowlist_exact"] is True
    assert report["held_trajectories_opened"] is False


def test_mixed_aoa_endpoint_context_restores_reused_trainer_globals():
    names = ("EXPERIMENT", "ENDPOINT_SCHEMA", "PRIMARY_ENDPOINT",
             "UPDATES", "SCHEDULER_T_MAX", "HELD_AOA")
    before = {name: getattr(POINT.small4k, name) for name in names}
    with POINT.endpoint_contract():
        assert POINT.small4k.HELD_AOA == POINT.MIXED_AOA_SENTINEL
        assert POINT.small4k.EXPERIMENT == POINT.EXPERIMENT
    assert {name: getattr(POINT.small4k, name) for name in names} == before


def test_cache_loader_binds_new_fold_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(CACHE, "EXPECTED_SAMPLES", 2)
    identities = [
        {"file": "a.h5", "start": 0, "re": 100, "aoa": 0, "fold": 0},
        {"file": "b.h5", "start": 0, "re": 200, "aoa": 5, "fold": 1},
    ]
    residuals = torch.zeros((2, 20, 32, 64, 2), dtype=torch.float16)
    factors = torch.tensor([0.5, 0.75])
    payload = {
        "schema": CACHE.SCHEMA, "folds": 5,
        "fold_formula": PLAN.FORMULA,
        "fold_plan_assignment_sha256": PLAN.EXPECTED_ASSIGNMENT_SHA256,
        "window_stride": 20, "samples": 2,
        "manifest": {"semantic_sha256": CACHE.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256},
        "manifest_file_sha256": CACHE.ft12.FORMAL_MANIFEST_FILE_SHA256,
        "clean716_manifest_read": False, "evaluation_manifest_read": False,
        "sample_identities": identities,
        "sample_identity_sha256": CACHE.dr1.canonical_sha256(identities),
        "signed_residual_physical_uv": residuals,
        "signed_residual_physical_uv_sha256": CACHE._sha_tensor_bytes(residuals),
        "oof_point_factor": factors,
        "oof_point_factor_sha256": CACHE._sha_tensor_bytes(factors),
        "ordered_history_stream_sha256": "1" * 64,
        "ordered_target_stream_sha256": "2" * 64,
        "fold_results": [],
    }
    path = tmp_path / "cache.pth"
    torch.save(payload, path)
    observed, report = CACHE.load_cache(path, CACHE.dr1.sha256(path))
    assert observed["sample_identities"] == identities
    assert report["samples"] == 2
    assert report["fold_plan_assignment_sha256"] == PLAN.EXPECTED_ASSIGNMENT_SHA256


def test_uq_partition_uses_fold_not_physical_aoa(monkeypatch):
    identities = [
        {"file": f"sample_{fold}_{aoa}.h5", "start": 0,
         "re": 100 + fold, "aoa": aoa, "fold": fold}
        for fold in range(5) for aoa in PLAN.AOA_VALUES]
    monkeypatch.setattr(UQ, "EXPECTED_SAMPLES", len(identities))
    expected = dict(UQ.fold_plan.EXPECTED_FOLDS[2])
    expected["held_oof_windows"] = 5
    expected["fit_oof_windows"] = 20
    monkeypatch.setitem(UQ.fold_plan.EXPECTED_FOLDS, 2, expected)
    fit, held, report = UQ.partition_indices(identities, 2)
    assert len(fit) == 20 and len(held) == 5
    assert set(report["held_aoa_windows"]) == set(PLAN.AOA_VALUES)
    assert all(identities[index]["fold"] == 2 for index in held)
    assert all(identities[index]["fold"] != 2 for index in fit)


def test_new_clis_do_not_accept_clean716_or_evaluation_manifest():
    for name in (
            "train_b210_same_aoa_point_oof_fold.py",
            "build_b210_same_aoa_point_oof_residual_cache.py",
            "train_b210_same_aoa_trajectory_native_uq.py"):
        source = (TRAINING / name).read_text(encoding="utf-8")
        assert 'add_argument("--clean716' not in source
        assert 'add_argument("--evaluation-manifest' not in source
    assert (UQ.UPDATES_PER_FOLD, UQ.CYCLES, UQ.TOTAL_UPDATES) == (800, 4, 3200)
