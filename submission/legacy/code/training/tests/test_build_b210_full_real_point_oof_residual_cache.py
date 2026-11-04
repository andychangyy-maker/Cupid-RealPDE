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
SCRIPT = TRAINING / "build_b210_full_real_point_oof_residual_cache.py"
SPEC = importlib.util.spec_from_file_location("b210_full_point_oof_cache", SCRIPT)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def rows():
    return [
        {"file": f"sample_{aoa}.h5", "frames": 40, "re": 4_000 + aoa,
         "aoa": aoa, "train_starts": [0], "val_boundary": 40}
        for aoa in SUBJECT.point_fold.AOA_VALUES]


def test_full_manifest_is_derived_from_all_legal_stride20_windows(monkeypatch):
    monkeypatch.setattr(SUBJECT, "EXPECTED_SAMPLES", 5)
    manifest = SUBJECT.full_evaluation_rows(rows())
    assert [row["val_starts"] for row in manifest] == [[0]] * 5
    assignments = SUBJECT.fold_assignments(manifest)
    assert assignments == {
        f"sample_{aoa}.h5": fold
        for fold, aoa in enumerate(SUBJECT.point_fold.AOA_VALUES)}


def test_full_manifest_rejects_excluded_file(monkeypatch):
    changed = rows()
    changed[0]["file"] = SUBJECT.dr1.EXCLUDED_BASENAME
    monkeypatch.setattr(SUBJECT, "EXPECTED_SAMPLES", 5)
    with pytest.raises(ValueError, match="excluded"):
        SUBJECT.full_evaluation_rows(changed)


def _fold_results(tmp_path: Path, manifest: list[dict]):
    assignments = SUBJECT.fold_assignments(manifest)
    pairs = []
    for fold, aoa in enumerate(SUBJECT.point_fold.AOA_VALUES):
        checkpoint = tmp_path / f"held{aoa}.pth"
        state = {"weight": torch.tensor([float(aoa)])}
        state_sha = SUBJECT.dr1.model_state_sha256(state)
        torch.save({
            "schema": SUBJECT.point_fold.ENDPOINT_SCHEMA,
            "experiment": SUBJECT.point_fold.EXPERIMENT,
            "phase": "real",
            "iteration": SUBJECT.point_fold.UPDATES,
            "endpoint": SUBJECT.point_fold.ENDPOINT,
            "held_aoa": aoa,
            "start": {"checkpoint_sha256": SUBJECT.point_fold.START_SHA256},
            "model_state_dict": state,
        }, checkpoint)
        result = {
            "schema": SUBJECT.point_fold.SCHEMA,
            "held_aoa": aoa,
            "held_fold": fold,
            "folds": SUBJECT.point_fold.FOLDS,
            "aoa_values": list(SUBJECT.point_fold.AOA_VALUES),
            "decision": "COMPLETE_FULL_REAL_POINT_OOF_FOLD",
            "held_data_opened": False,
            "clean716_manifest_read": False,
            "start": {"checkpoint_sha256": SUBJECT.point_fold.START_SHA256},
            "manifest_file_sha256": SUBJECT.ft12.FORMAL_MANIFEST_FILE_SHA256,
            "manifest": {
                "semantic_sha256": SUBJECT.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256},
            "source_kind": "test",
            "fit": {"seal": {
                "path": str(checkpoint),
                "file_sha256": SUBJECT.dr1.sha256(checkpoint),
                "model_state_sha256": state_sha,
            }},
        }
        result_path = tmp_path / f"held{aoa}.json"
        result = SUBJECT.point_fold.seal_json_payload(result)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        pairs.append((str(result_path), SUBJECT.dr1.sha256(result_path)))
    return assignments, pairs


def test_fold_registry_requires_exact_whole_aoa_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(SUBJECT, "EXPECTED_SAMPLES", 5)
    manifest = SUBJECT.full_evaluation_rows(rows())
    assignments, pairs = _fold_results(tmp_path, manifest)
    checkpoints, reports = SUBJECT.validate_fold_results(
        manifest, assignments, pairs)
    assert set(checkpoints) == set(range(5))
    assert [row["held_aoa"] for row in reports] == [0, 5, 10, 15, 20]
    with pytest.raises(ValueError, match="one SHA-bound"):
        SUBJECT.validate_fold_results(manifest, assignments, pairs[:-1])


def test_fold_registry_rejects_clean716_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(SUBJECT, "EXPECTED_SAMPLES", 5)
    manifest = SUBJECT.full_evaluation_rows(rows())
    assignments, pairs = _fold_results(tmp_path, manifest)
    path = Path(pairs[0][0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop(SUBJECT.point_fold.JSON_SELF_SEAL_KEY)
    payload["clean716_manifest_read"] = True
    payload = SUBJECT.point_fold.seal_json_payload(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    pairs[0] = (str(path), SUBJECT.dr1.sha256(path))
    with pytest.raises(RuntimeError, match="identity"):
        SUBJECT.validate_fold_results(manifest, assignments, pairs)


def test_cli_has_no_external_evaluation_manifest():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'add_argument("--evaluation-manifest' not in source
    assert 'add_argument("--clean716' not in source
    assert SUBJECT.EXPECTED_SAMPLES == 3_341
