from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "code/training"
sys.path.insert(0, str(TRAINING))
SCRIPT = TRAINING / "seal_b210_full_real_point_oof_result.py"
SPEC = importlib.util.spec_from_file_location("seal_full_point_oof", SCRIPT)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def unsealed_payload():
    return {
        "schema": SUBJECT.point_fold.SCHEMA,
        "held_aoa": 0,
        "held_fold": 0,
        "folds": SUBJECT.point_fold.FOLDS,
        "aoa_values": list(SUBJECT.point_fold.AOA_VALUES),
        "decision": "COMPLETE_FULL_REAL_POINT_OOF_FOLD",
        "held_data_opened": False,
        "clean716_manifest_read": False,
        "start": {"checkpoint_sha256": SUBJECT.point_fold.START_SHA256},
        "manifest_file_sha256": SUBJECT.ft12.FORMAL_MANIFEST_FILE_SHA256,
        "manifest": {
            "semantic_sha256": SUBJECT.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256},
        "fit": {"seal": {"path": "/immutable/endpoint.pth"}},
    }


def test_derivation_never_overwrites_and_preserves_checkpoint_binding(
        tmp_path, monkeypatch):
    source = tmp_path / "fold_result.unsealed.json"
    source.write_text(json.dumps(unsealed_payload()), encoding="utf-8")
    source_sha = SUBJECT.dr1.sha256(source)
    monkeypatch.setattr(SUBJECT, "validate_unsealed_result", lambda _payload: {
        "file_sha256": "a" * 64,
        "model_state_sha256": "b" * 64,
    })
    output = tmp_path / "fold_result.sealed.json"
    report = SUBJECT.derive(source, source_sha, output)
    assert source.read_text(encoding="utf-8") == json.dumps(unsealed_payload())
    sealed = json.loads(output.read_text(encoding="utf-8"))
    SUBJECT.point_fold.verify_json_payload(sealed)
    assert sealed["json_seal_derivation"]["source_file_sha256"] == source_sha
    assert sealed["json_seal_derivation"]["checkpoint_unchanged"] is True
    assert report["checkpoint_file_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="must not overwrite"):
        SUBJECT.derive(source, source_sha, source)


def test_unsealed_contract_rejects_clean716_claim(monkeypatch):
    payload = unsealed_payload()
    payload["clean716_manifest_read"] = True
    with pytest.raises(RuntimeError, match="contract changed"):
        SUBJECT.validate_unsealed_result(payload)
