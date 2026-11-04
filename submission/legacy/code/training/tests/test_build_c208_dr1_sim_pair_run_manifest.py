from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "build_c208_dr1_sim_pair_run_manifest.py")
SPEC = importlib.util.spec_from_file_location("c208_dr1_pair_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def write_arm(tmp_path: Path, variant: str, started: str,
              gpu_uuid: str = "GPU-1") -> tuple[Path, Path]:
    checkpoint = tmp_path / f"{variant}.pth"
    checkpoint.write_bytes(f"{variant}-checkpoint".encode("ascii"))
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audit = {
        "schema": "c208_dr1_sim_audit_v1",
        "variant": variant,
        "run_ordinal": 1 if variant == "candidate" else 2,
        "started_at_utc": started,
        "endpoint_checkpoint_sha256": checkpoint_sha,
        "hardware_fingerprint": {
            "gpu_uuid": gpu_uuid,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "torch_version": "2.0",
            "cuda_runtime_version": "12.1",
            "cudnn_version": 8900,
        },
    }
    audit_path = tmp_path / f"{variant}_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return checkpoint, audit_path


def test_manifest_binds_sequential_same_gpu_pair(tmp_path):
    candidate, candidate_audit = write_arm(
        tmp_path, "candidate", "2026-08-26T00:00:00Z")
    control, control_audit = write_arm(
        tmp_path, "control", "2026-08-26T01:00:00Z")

    manifest = subject.build_manifest(
        candidate, candidate_audit, control, control_audit,
        subject.FORMAL_SOURCE_COMMIT, True)

    assert manifest["sequence"] == ["candidate", "control"]
    assert manifest["source_git_commit"] == subject.FORMAL_SOURCE_COMMIT
    assert manifest["candidate"]["checkpoint_sha256"] == subject.sha256(
        candidate)
    assert manifest["control"]["audit_sha256"] == subject.sha256(
        control_audit)
    assert manifest["candidate"]["mtime_is_scientific_gate"] is False


def test_manifest_rejects_wrong_source_hardware_or_order(tmp_path):
    candidate, candidate_audit = write_arm(
        tmp_path, "candidate", "2026-08-26T02:00:00Z")
    control, control_audit = write_arm(
        tmp_path, "control", "2026-08-26T01:00:00Z")

    with pytest.raises(ValueError, match="source commit"):
        subject.build_manifest(
            candidate, candidate_audit, control, control_audit,
            "0" * 40, True)
    with pytest.raises(RuntimeError, match="candidate did not start"):
        subject.build_manifest(
            candidate, candidate_audit, control, control_audit,
            subject.FORMAL_SOURCE_COMMIT, True)

    candidate, candidate_audit = write_arm(
        tmp_path, "candidate", "2026-08-26T00:00:00Z", "GPU-2")
    with pytest.raises(RuntimeError, match="different hardware"):
        subject.build_manifest(
            candidate, candidate_audit, control, control_audit,
            subject.FORMAL_SOURCE_COMMIT, True)
