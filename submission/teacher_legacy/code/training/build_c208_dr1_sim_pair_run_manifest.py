#!/usr/bin/env python3
"""Bind the sequential C208-DR1 Sim pair to immutable run artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


SCHEMA = "c208_dr1_sim_pair_run_manifest_v1"
FORMAL_SOURCE_COMMIT = "7e432298c868f23ac5104296a2dca1cf5af7e16f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audit(path: Path, variant: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{variant} Sim audit must be a mapping")
    if value.get("schema") != "c208_dr1_sim_audit_v1":
        raise ValueError(f"{variant} Sim audit schema changed")
    if value.get("variant") != variant:
        raise ValueError(f"{variant} Sim audit variant changed")
    if value.get("run_ordinal") != (1 if variant == "candidate" else 2):
        raise ValueError(f"{variant} Sim run ordinal changed")
    return value


def parse_utc(value: object, label: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid UTC timestamp") from exc


def artifact_row(checkpoint: Path, audit_path: Path, audit: dict) -> dict:
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != audit.get("endpoint_checkpoint_sha256"):
        raise RuntimeError(
            f"checkpoint SHA disagrees with {audit.get('variant')} Sim audit")
    return {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": sha256(audit_path),
        "audit_mtime_ns": audit_path.stat().st_mtime_ns,
        "started_at_utc": audit["started_at_utc"],
        "mtime_is_scientific_gate": False,
    }


def build_manifest(
        candidate_checkpoint: Path, candidate_audit_path: Path,
        control_checkpoint: Path, control_audit_path: Path,
        source_git_commit: str, same_shell_sequential: bool) -> dict:
    if source_git_commit != FORMAL_SOURCE_COMMIT:
        raise ValueError("formal Sim source commit changed")
    if not same_shell_sequential:
        raise ValueError(
            "formal pair requires an explicit same-shell sequential assertion")
    candidate_audit = load_audit(candidate_audit_path, "candidate")
    control_audit = load_audit(control_audit_path, "control")
    if (candidate_audit.get("hardware_fingerprint")
            != control_audit.get("hardware_fingerprint")):
        raise RuntimeError("Sim audits have different hardware fingerprints")
    candidate_started = parse_utc(
        candidate_audit.get("started_at_utc"), "candidate started_at_utc")
    control_started = parse_utc(
        control_audit.get("started_at_utc"), "control started_at_utc")
    if not candidate_started < control_started:
        raise RuntimeError("candidate did not start before control")
    return {
        "schema": SCHEMA,
        "source_git_commit": source_git_commit,
        "sequence": ["candidate", "control"],
        "same_shell_sequential": True,
        "candidate_completed_before_control_started": True,
        "candidate": artifact_row(
            candidate_checkpoint, candidate_audit_path, candidate_audit),
        "control": artifact_row(
            control_checkpoint, control_audit_path, control_audit),
        "shared_hardware_fingerprint": candidate_audit[
            "hardware_fingerprint"],
        "evidence_note": (
            "The source-7e audits predate completed_at_utc. Sequentiality is "
            "the runner's explicit same-shell assertion; mtimes are recorded "
            "only as operational evidence and are not a scientific gate."),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--control-audit", type=Path, required=True)
    parser.add_argument(
        "--source-git-commit", default=FORMAL_SOURCE_COMMIT)
    parser.add_argument(
        "--assert-same-shell-sequential", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest = build_manifest(
        args.candidate_checkpoint, args.candidate_audit,
        args.control_checkpoint, args.control_audit,
        args.source_git_commit, args.assert_same_shell_sequential)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "event": "c208_dr1_sim_pair_run_manifest_complete",
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "source_git_commit": args.source_git_commit,
    }), flush=True)


if __name__ == "__main__":
    main()
