from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "code/training"
sys.path.insert(0, str(TRAINING))
SCRIPT = TRAINING / "train_b210_xcb_full_released_real_final_point.py"
SPEC = importlib.util.spec_from_file_location("b210_xcb_final_full_point", SCRIPT)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def test_xcb_architecture_contract_is_exact_existing_topology():
    config, parameters, controller = SUBJECT.architecture_contract()
    assert config.base == 210
    assert config.depth == 4
    assert config.ensemble_members == 4
    assert config.cross_conditioned_backbone is True
    assert config.cross_conditioned_steps == 5
    assert config.cross_conditioned_channels == 64
    assert config.member_adapter_channels == 64
    assert config.mean_preserving_member_adapters is False
    assert config.detach_member_adapter_context is False
    assert parameters == 180_553_371
    assert controller == 7_169


def test_xcb_contract_changes_only_lineage_not_full_data_recipe():
    assert SUBJECT.CONTRACT["start"] == {
        "kind": (
            "exact XCB/member Standard24k Sim endpoint; never an old Real or "
            "UQ checkpoint"),
        "checkpoint_sha256": SUBJECT.FORMAL_START_SHA256,
        "phase": "sim",
        "local_iteration": 4_000,
        "cumulative_supervised_updates": 24_000,
        "required_status": SUBJECT.START_EXPECTED_STATUS,
        "architecture": (
            "B210 depth4/member4 + cross-conditioned backbone + multiscale "
            "member adapters"),
        "parameters": 180_553_371,
        "controller_parameters": 7_169,
        "provenance_caveat": (
            "The from-random Sim24k lifecycle used only released competition "
            "assets, including its existing Real-closure augmentation and P1 "
            "distillation; this stage nevertheless starts before any prior "
            "Real4k/UQ checkpoint."),
    }
    assert SUBJECT.CONTRACT["data"] == SUBJECT.base.CONTRACT["data"]
    assert SUBJECT.CONTRACT["training"] == SUBJECT.base.CONTRACT["training"]
    assert SUBJECT.CONTRACT["data"]["optimizer_windows"] == 13_202
    assert SUBJECT.CONTRACT["data"]["normalizer_windows"] == 2_544
    assert SUBJECT.CONTRACT["data"]["clean716_manifest_accepted"] is False
    assert SUBJECT.CONTRACT["method_evidence"][
        "xcb_matched_full_support_claimed"] is False


def test_configured_base_binds_and_restores_lineage_hooks():
    names = (
        "EXPERIMENT", "SCHEMA", "ENDPOINT_SCHEMA", "FORMAL_BRANCH",
        "ACKNOWLEDGEMENT", "FORMAL_START_SHA256", "MODEL_LINEAGE",
        "START_EXPECTED_ITERATION", "START_EXPECTED_STATUS",
        "START_MATCHES_SUPPORT_LINEAGE", "COMPLETION_EVENT", "CONTRACT",
        "CONTRACT_SHA256", "architecture_contract")
    before = {name: getattr(SUBJECT.base, name) for name in names}
    support_start = SUBJECT.base.SUPPORT_EVIDENCE_START_SHA256
    with SUBJECT.configured_base():
        assert SUBJECT.base.EXPERIMENT == SUBJECT.EXPERIMENT
        assert SUBJECT.base.SCHEMA == SUBJECT.SCHEMA
        assert SUBJECT.base.ENDPOINT_SCHEMA == SUBJECT.ENDPOINT_SCHEMA
        assert SUBJECT.base.FORMAL_START_SHA256 == SUBJECT.FORMAL_START_SHA256
        assert SUBJECT.base.START_EXPECTED_ITERATION == 4_000
        assert SUBJECT.base.START_EXPECTED_STATUS == SUBJECT.START_EXPECTED_STATUS
        assert SUBJECT.base.START_MATCHES_SUPPORT_LINEAGE is False
        assert SUBJECT.base.architecture_contract() == (
            SUBJECT.architecture_contract())
        # Method evidence remains honestly bound to the B210 matched experiment.
        assert SUBJECT.base.SUPPORT_EVIDENCE_START_SHA256 == support_start
    assert {name: getattr(SUBJECT.base, name) for name in names} == before


def test_transferred_support_evidence_remains_bound_to_b210_start(tmp_path):
    row = {
        "schema": SUBJECT.base.support.AGGREGATE_SCHEMA,
        "experiment": SUBJECT.base.support.EXPERIMENT,
        "contract_sha256": SUBJECT.base.support.CONTRACT_SHA256,
        "held_aoas": list(SUBJECT.base.support.HELD_AOAS),
        "start_checkpoint_sha256": (
            SUBJECT.base.SUPPORT_EVIDENCE_START_SHA256),
        "held_full_windows": 1_399,
        "primary_endpoint": SUBJECT.base.PRIMARY_ENDPOINT,
        "raw_endpoint_role": "diagnostic_only_no_endpoint_selection",
        "primary_comparison": {
            "point_final_contribution_delta": 0.01,
            "non_time_final_contribution_delta": 0.02,
        },
        "score_gate": None,
        "decision": "COMPLETE_TWO_FOLD_MATCHED_EVIDENCE_NO_SCORE_GATE",
        "submission_authorized": False,
    }
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(row), encoding="utf-8")
    with SUBJECT.configured_base():
        report = SUBJECT.base.validate_support_aggregate(
            path, SUBJECT.base.dr1.sha256(path))
    assert row["start_checkpoint_sha256"] != SUBJECT.FORMAL_START_SHA256
    assert report["method_adoption_was_manual"] is True


def test_main_is_only_a_configured_delegation(monkeypatch):
    observed = {}

    def fake_main():
        observed.update({
            "experiment": SUBJECT.base.EXPERIMENT,
            "schema": SUBJECT.base.SCHEMA,
            "start": SUBJECT.base.FORMAL_START_SHA256,
            "architecture": SUBJECT.base.architecture_contract(),
        })

    monkeypatch.setattr(SUBJECT.base, "main", fake_main)
    SUBJECT.main()
    assert observed["experiment"] == SUBJECT.EXPERIMENT
    assert observed["schema"] == SUBJECT.SCHEMA
    assert observed["start"] == SUBJECT.FORMAL_START_SHA256
    assert observed["architecture"][1:] == (180_553_371, 7_169)


def test_wrapper_adds_no_clean716_or_old_real_checkpoint_cli():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "add_argument" not in source
    assert "clean716" not in source.lower()
    assert "real4k_20260901" not in source
    assert "nativeuq" not in source.lower()
