from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_real_small4k_full_clean_point as subject  # noqa: E402


def _state(samples: int) -> dict:
    state = subject.smu1.new_metric_state()
    state.update({
        "samples": samples,
        "rel_error_sum": 0.1 * samples,
        "tke_error_sum": 0.2 * samples,
        "mvpe_error_sum": 0.3 * samples,
        "sps_numerator": 0.5 * 100 * samples,
        "sps_scored": 100 * samples,
        "sps_covered": 80 * samples,
        "sps_width_sum": 5 * samples,
    })
    return state


def _audit(samples: int) -> dict:
    return {
        "ordered_raw_target_sha256": "a" * 64,
        "ordered_scored_mask_sha256": "b" * 64,
        "samples": samples,
        "scored_elements": 100 * samples,
    }


def _passing_comparison() -> dict:
    return {
        "control": {},
        "candidate": {},
        "delta": {
            "rel_l2_score": 0.01,
            "tke_rel_l2_score": 0.21,
            "mvpe_rel_l2_score": 0.01,
            "point_score": 0.03,
            "native_sps_score": 0.31,
        },
        "direct_delta": 0.03,
        "non_time_delta": 0.11,
    }


def _confirmation_row() -> dict:
    samples = subject.dr1.EXPECTED_AOA_COUNTS[0]
    state = _state(samples)
    metrics = subject.smu1.finalize_metric_state(state)
    comparison = _passing_comparison()
    return {
        "schema": subject.small4k.SCHEMA,
        "experiment": subject.small4k.EXPERIMENT,
        "contract_sha256": subject.small4k.CONTRACT_SHA256,
        "held_aoa": 0,
        "samples": samples,
        "primary_endpoint": "ema_u4000",
        "endpoint_selection_allowed": False,
        "decision": "PASS_B210_SMALL4K_HELD0_INDEPENDENT_CONFIRMATION",
        "evidence_class": {
            "held10": "discovery_only_post_hoc_not_confirmation",
            "held0": "single_independent_confirmation",
            "pooled_official": False,
            "submission": False,
        },
        "discovery": {"sha256": subject.DISCOVERY_SHA256},
        "start": {"checkpoint_sha256": subject.SIM10K_SHA256},
        "manifest_file_sha256": subject.ft12.FORMAL_MANIFEST_FILE_SHA256,
        "manifest": {
            "semantic_sha256": subject.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256,
            "windows": subject.FIT_SAMPLES,
            "excluded": subject.dr1.EXCLUDED_BASENAME,
        },
        "prefix_access_audit": {
            "tail_arrays_read": 0,
            "excluded_opened": False,
            "held_first_open_after_endpoint_seal": True,
            "held_complete_passes": 1,
        },
        "comparison": comparison,
        "confirmation_gate": subject.small4k.confirmation_gate(comparison),
        "endpoint": {
            "seal": {
                "endpoint": "ema_u4000",
                "primary": True,
                "iteration": 4_000,
                "model_state_sha256": "c" * 64,
            },
            "metrics": metrics,
            "metric_state": state,
            "held_target_audit": _audit(samples),
        },
    }


def _latency_row() -> dict:
    return {
        "evidence_class": "s2_deployment_archive_latency_abba",
        "sequence": ["c208", "b210", "b210", "c208"],
        "decision": "PASS_B210_ABBA_LATENCY",
        "gates": {"one": True, "two": True},
        "ratios": {"pooled_b210_over_c208": 1.01},
        "archive_sha256": {"b210": "d" * 64},
        "b210_peak_reserved_bytes": 1_000,
    }


def test_contract_freezes_full_clean_point_stage_only():
    assert subject.FORMAL_BRANCH == "codex/b210-small4k-release"
    assert subject.STATUS == "POINT_STAGE_ONLY_NOT_SUBMITTABLE"
    assert subject.ENDPOINT_SCHEMA == (
        "b210_np_functional_real_small4k_full_refit_endpoint_v1")
    assert subject.FIT_SAMPLES == 2544
    assert subject.UPDATES == subject.small4k.UPDATES == 4_000
    assert subject.BATCH_SIZE == subject.small4k.BATCH_SIZE == 8
    assert subject.SEED == subject.small4k.SEED == 4
    assert subject.LR == subject.small4k.LR == pytest.approx(2e-4)
    assert subject.WEIGHT_DECAY == pytest.approx(1e-5)
    assert subject.SCHEDULER_T_MAX == 12_000
    assert subject.EMA_DECAY == pytest.approx(0.999)
    assert subject.WALL_BUDGET_SECONDS == 60 * 60
    output = subject.CONTRACT["output"]
    assert output["sole_model_endpoint"] == "ema_u4000"
    assert output["candidate_specific_oof_uq_refit_required"] is True
    assert output["release_checkpoint_written"] is False
    assert output["submission_authorized"] is False


def test_all_authorization_digests_are_full_and_frozen():
    assert subject.SIM10K_SHA256 == (
        "f86faac5c376d2ee52ceb9209e407e2a42c114d79753dfd5fe2a51072207d85a")
    for value in (
            subject.SIM10K_SHA256, subject.DISCOVERY_SHA256,
            subject.CONFIRMATION_SHA256, subject.LATENCY_SHA256):
        assert len(value) == 64
        int(value, 16)


def test_full_clean_access_is_prefix_only_and_excludes_bad_file(tmp_path):
    access = subject.FullCleanAccessAudit()
    row = {"file": "good.h5", "aoa": 0, "val_boundary": 100}
    access.before_open(row, "fit")
    access.record_open(row, "fit")
    with pytest.raises(RuntimeError, match="cannot open held"):
        access.before_open(row, "held")
    with pytest.raises(RuntimeError, match="excluded H5"):
        access.before_open({
            "file": subject.dr1.EXCLUDED_BASENAME,
            "aoa": 0,
            "val_boundary": 100,
        }, "fit")
    precursor = {
        "endpoint": "ema_u4000",
        "primary": True,
        "iteration": 4_000,
        "path": str(tmp_path / "point.pth"),
        "model_state_sha256": "1" * 64,
    }
    access.seal(precursor)
    final = {**precursor, "schema": subject.ENDPOINT_SCHEMA,
             "file_sha256": "2" * 64}
    access.bind_final(final)
    report = access.report(["good.h5"])
    assert report["allowlist_exact"] is True
    assert report["prefix_only"] is True
    assert report["held_or_tail_labels_read"] is False
    assert report["tail_arrays_read"] == 0
    assert report["excluded_opened"] is False


def test_confirmation_requires_independent_pass_and_same_sim():
    row = _confirmation_row()
    report = subject.validate_confirmation_row(row)
    assert report["confirmation_role"] == "independent_held0"
    assert report["start_checkpoint_sha256"] == subject.SIM10K_SHA256
    changed = _confirmation_row()
    changed["start"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="Sim binding"):
        subject.validate_confirmation_row(changed)
    changed = _confirmation_row()
    changed["confirmation_gate"]["pass"] = False
    with pytest.raises(RuntimeError, match="not PASS"):
        subject.validate_confirmation_row(changed)


def test_latency_authorization_requires_abba_pass():
    report = subject.validate_latency_summary(_latency_row())
    assert report["decision"] == "PASS_B210_ABBA_LATENCY"
    changed = _latency_row()
    changed["gates"]["one"] = False
    with pytest.raises(RuntimeError, match="not a strict ABBA PASS"):
        subject.validate_latency_summary(changed)
    changed = _latency_row()
    changed["decision"] = "STOP_B210_ABBA_LATENCY"
    with pytest.raises(RuntimeError, match="not a strict ABBA PASS"):
        subject.validate_latency_summary(changed)


def _precursor_checkpoint(tmp_path: Path) -> tuple[Path, dict]:
    state = OrderedDict(weight=torch.ones(2, dtype=torch.float32))
    state_sha = subject.dr1.model_state_sha256(state)
    normalizer = {
        "mean_input": [0.0, 0.0, 0.0],
        "mean_target": [0.0, 0.0, 0.0],
        "std_input": [1.0, 1.0, 1.0],
        "std_target": [1.0, 1.0, 1.0],
    }
    payload = {
        "schema": subject.small4k.ENDPOINT_SCHEMA,
        "experiment": subject.small4k.EXPERIMENT,
        "endpoint": "ema_u4000",
        "primary": True,
        "iteration": 4_000,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": {"architecture": "ensemble_unet", "tiny": True},
        "start": {"checkpoint_sha256": subject.SIM10K_SHA256},
        "normalizer": normalizer,
        "normalizer_sha256": subject.dr1.canonical_sha256(normalizer),
        "sample_stream_sha256": "4" * 64,
        "ema": {
            "schema": "fp32_named_parameter_ema_seal_v1",
            "update_order": "after_optimizer_step_before_scheduler_step",
            "updates": 4_000,
            "decay": 0.999,
            "shadow_dtype": "torch.float32",
            "materialized_model_state_sha256": state_sha,
        },
    }
    path = tmp_path / "ema999_u4000_primary.pth"
    torch.save(payload, path)
    seal = {
        "endpoint": "ema_u4000",
        "path": str(path),
        "file_sha256": subject.dr1.sha256(path),
        "model_state_sha256": state_sha,
        "iteration": 4_000,
        "primary": True,
    }
    return path, seal


def test_rebind_emits_exact_uq_consumer_interface(tmp_path, monkeypatch):
    _path, precursor = _precursor_checkpoint(tmp_path)
    marker = object()
    monkeypatch.setattr(
        subject.ft12, "validate_b210_model_config", lambda _config: marker)
    monkeypatch.setattr(subject.ft12, "expected_b210_config", lambda: marker)
    manifest = {
        "windows": 2544,
        "excluded": subject.dr1.EXCLUDED_BASENAME,
    }
    authorization = {"point_stage_authorized": True}
    seal = subject.rebind_full_clean_endpoint(
        precursor, fit_samples=2544, manifest_report=manifest,
        authorization=authorization)
    path = Path(seal["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema"] == subject.ENDPOINT_SCHEMA
    assert payload["phase"] == "real"
    assert payload["iteration"] == 4_000
    assert payload["endpoint"] == "ema_u4000"
    assert payload["primary"] is True
    assert payload["training_scope"] == "full_clean2544"
    assert payload["fit_samples"] == 2544
    assert payload["status"] == subject.STATUS
    assert payload["manifest_file_sha256"] == (
        subject.ft12.FORMAL_MANIFEST_FILE_SHA256)
    assert payload["manifest_semantic_sha256"] == (
        subject.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    assert set(payload["normalizer"]) == {
        "mean_input", "mean_target", "std_input", "std_target"}
    assert payload["evidence_bindings"] == {
        "sim10k_sha256": subject.SIM10K_SHA256,
        "held10_discovery_sha256": subject.DISCOVERY_SHA256,
        "held0_confirmation_sha256": subject.CONFIRMATION_SHA256,
        "latency_summary_sha256": subject.LATENCY_SHA256,
    }
    assert payload["ema"]["updates"] == 4_000
    assert payload["ema"]["decay"] == pytest.approx(0.999)
    assert payload["ema"]["shadow_dtype"] == "torch.float32"
    assert payload["ema"]["materialized_model_state_sha256"] == (
        payload["model_state_sha256"])
    assert payload["candidate_specific_oof_uq_refit_required"] is True
    assert payload["submission_authorized"] is False
    report = subject.validate_point_checkpoint(path, seal)
    assert report["strict_b210_config"] is True
    assert report["strict_state"] is True
    assert report["flat_normalizer"] is True


def test_rebind_rejects_non_fp32_state(tmp_path, monkeypatch):
    path, precursor = _precursor_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model_state_dict"]["weight"] = torch.ones(2, dtype=torch.float16)
    changed_sha = subject.dr1.model_state_sha256(payload["model_state_dict"])
    payload["model_state_sha256"] = changed_sha
    payload["ema"]["materialized_model_state_sha256"] = changed_sha
    torch.save(payload, path)
    precursor["file_sha256"] = subject.dr1.sha256(path)
    precursor["model_state_sha256"] = changed_sha
    monkeypatch.setattr(
        subject.ft12, "validate_b210_model_config", lambda _config: object())
    with pytest.raises(TypeError, match="remain FP32"):
        subject.rebind_full_clean_endpoint(
            precursor, fit_samples=2544,
            manifest_report={
                "windows": 2544,
                "excluded": subject.dr1.EXCLUDED_BASENAME,
            },
            authorization={"point_stage_authorized": True})


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        run_full_clean=True,
        contract_only=False,
        output_dir=Path("out"),
        real_dir=Path("real"),
        clean_manifest=Path("manifest.json"),
        start_sim=Path("sim.pth"),
        start_sim_sha256=subject.SIM10K_SHA256,
        discovery_fold10=Path("discovery.json"),
        confirmation_held0=Path("confirmation.json"),
        latency_summary=Path("latency.json"),
        device="cuda:0",
        hardware_class="rtx4080_super",
        execute_formal_gpu=True,
        acknowledge_protocol=subject.FORMAL_ACKNOWLEDGEMENT,
    )


def test_cli_requires_every_evidence_file_and_explicit_acknowledgement():
    subject.validate_args(_args())
    missing = _args()
    missing.latency_summary = None
    with pytest.raises(ValueError, match="missing required"):
        subject.validate_args(missing)
    wrong = _args()
    wrong.acknowledge_protocol = ""
    with pytest.raises(ValueError, match="requires acknowledgement"):
        subject.validate_args(wrong)


def test_source_has_no_forbidden_split_or_endpoint_selection():
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    assert "clean716" not in source
    assert "fixed725" not in source
    assert "aggregate-only" not in source
    assert "raw_u4000" not in source
