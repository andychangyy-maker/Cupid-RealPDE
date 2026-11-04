from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "code/training"
sys.path.insert(0, str(TRAINING))
SCRIPT = TRAINING / "train_b210_full_real_point_oof_fold.py"
SPEC = importlib.util.spec_from_file_location("b210_full_point_oof", SCRIPT)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def test_whole_aoa_fold_contract_covers_exact_full_inventory():
    assert SUBJECT.AOA_VALUES == (0, 5, 10, 15, 20)
    assert [SUBJECT.aoa_fold(value) for value in SUBJECT.AOA_VALUES] == list(range(5))
    assert sum(row["held_trajectories"] for row in SUBJECT.EXPECTED_FOLDS.values()) == 81
    assert sum(
        SUBJECT.EXPECTED_TOTAL["normalizer_prefix_windows"]
        - row["normalizer_prefix_windows"]
        for row in SUBJECT.EXPECTED_FOLDS.values()) == 2_544
    assert sum(
        SUBJECT.EXPECTED_TOTAL["full_optimizer_windows"]
        - row["full_optimizer_windows"]
        for row in SUBJECT.EXPECTED_FOLDS.values()) == 13_202
    assert sum(row["held_oof_windows"] for row in SUBJECT.EXPECTED_FOLDS.values()) == 3_341


def test_full_window_count_uses_every_legal_start():
    rows = [{"frames": 40}, {"frames": 59}, {"frames": 60}, {"frames": 81}]
    assert SUBJECT.full_window_count(rows, 20) == 1 + 1 + 2 + 3
    assert SUBJECT.full_window_count(rows, 5) == 1 + 4 + 5 + 9
    with pytest.raises(ValueError, match="positive"):
        SUBJECT.full_window_count(rows, 0)


def test_fit_only_access_rejects_held_excluded_and_post_seal(tmp_path):
    audit = SUBJECT.FitOnlyAccessAudit(10, ["fit.h5"])
    fit = {"file": "fit.h5", "aoa": 5}
    audit.record_open(fit, "fit")
    with pytest.raises(RuntimeError, match="rejected"):
        audit.before_open({"file": "held.h5", "aoa": 10}, "fit")
    with pytest.raises(RuntimeError, match="rejected"):
        audit.before_open({
            "file": SUBJECT.dr1.EXCLUDED_BASENAME, "aoa": 5}, "fit")
    endpoint = tmp_path / "ema.pth"
    endpoint.write_bytes(b"sealed")
    seal = {
        "endpoint": SUBJECT.ENDPOINT,
        "primary": True,
        "iteration": SUBJECT.UPDATES,
        "path": str(endpoint),
        "file_sha256": SUBJECT.dr1.sha256(endpoint),
    }
    audit.seal(seal)
    with pytest.raises(RuntimeError, match="rejected"):
        audit.before_open(fit, "fit")
    report = audit.report()
    assert report["allowlist_exact"] is True
    assert report["held_trajectories_opened"] is False
    assert report["clean716_manifest_read"] is False
    assert report["excluded_opened"] is False


def test_endpoint_contract_restores_reused_trainer_globals():
    names = (
        "EXPERIMENT", "ENDPOINT_SCHEMA", "PRIMARY_ENDPOINT",
        "UPDATES", "SCHEDULER_T_MAX", "HELD_AOA")
    before = {name: getattr(SUBJECT.small4k, name) for name in names}
    with SUBJECT.endpoint_contract(15):
        assert SUBJECT.small4k.EXPERIMENT == SUBJECT.EXPERIMENT
        assert SUBJECT.small4k.ENDPOINT_SCHEMA == SUBJECT.ENDPOINT_SCHEMA
        assert SUBJECT.small4k.HELD_AOA == 15
        assert SUBJECT.small4k.SCHEDULER_T_MAX == 12_000
    assert {name: getattr(SUBJECT.small4k, name) for name in names} == before


def test_cli_has_no_clean716_or_evaluation_manifest_input():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'add_argument("--clean716' not in source
    assert 'add_argument("--evaluation-manifest' not in source
    assert SUBJECT.START_SHA256 == (
        "8a6d9a1055269f95a308b42e73ee94b00b005edbf550430dfa096a610e2e0f29")


def test_json_self_seal_is_computed_after_roundtrip_and_detects_change():
    sealed = SUBJECT.seal_json_payload({
        "per_aoa": {str(aoa): {"windows": index + 1}
                    for index, aoa in enumerate(SUBJECT.AOA_VALUES)},
        "tuple_value": (1, 2),
    })
    reloaded = json.loads(json.dumps(sealed))
    assert SUBJECT.verify_json_payload(reloaded) == reloaded[
        SUBJECT.JSON_SELF_SEAL_KEY]["payload_without_self_seal_sha256"]
    assert set(reloaded["per_aoa"]) == {"0", "5", "10", "15", "20"}
    reloaded["per_aoa"]["10"]["windows"] += 1
    with pytest.raises(RuntimeError, match="changed"):
        SUBJECT.verify_json_payload(reloaded)
