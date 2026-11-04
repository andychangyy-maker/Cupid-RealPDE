#!/usr/bin/env python3
"""Prefix-only clean81 paired Real LOAO gate for C208-DR1.

The orchestrator validates the two completed Sim arms, freezes a pair
attestation, and starts one fresh worker process per held AoA.  A worker opens
only other-four-AoA training prefixes until both u4000 endpoints have been
written and sealed.  Only then may it open the held-AoA prefixes for scoring.

This program never constructs or reads legacy 2576/725 or clean716 windows.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_ensemble_unet2d as ensemble_train  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


EXPERIMENT = "c208_dr1_clean81_paired_real_loao"
SCHEMA = "c208_dr1_clean81_loao_v1"
SIM_AUDIT_SCHEMA = "c208_dr1_sim_audit_v1"
PAIR_ATTESTATION_SCHEMA = "c208_dr1_sim_pair_attestation_v1"
RUN_MANIFEST_SCHEMA = "c208_dr1_sim_pair_run_manifest_v1"
FORMAL_SIM_SOURCE_COMMIT = (
    "7e432298c868f23ac5104296a2dca1cf5af7e16f")
EXCLUDED_BASENAME = "7575_0.h5"
FORMAL_MANIFEST_SHA256 = (
    "d0f52e7ffdec0fad0cbd653ebb41a3d0d3c76dd2ee99a2b927cc9b0d14a76d09")
FORMAL_PREFIX_STATS_SHA256 = (
    "ecf89261376864ad90c3e8a4c2e577122010ed9e4bf60db171236c0d02180b5c")
EXPECTED_PARAMETERS = 175_387_412
AOA_VALUES = (0, 5, 10, 15, 20)
FOLD_ORDER = (10, 0, 20, 15, 5)
EXPECTED_AOA_COUNTS = {0: 502, 5: 433, 10: 576, 15: 544, 20: 489}
EXPECTED_FIT_COUNTS = {
    held: 2544 - count for held, count in EXPECTED_AOA_COUNTS.items()}
EXPECTED_TRAJECTORIES = 81
EXPECTED_WINDOWS = 2544
UPDATES = 4000
SEED = 4
TIME_SCORE = 90.344910
GENERALIZED_Q = 0.6854
GENERALIZED_WEIGHTS = {
    "rel_l2_score": 0.5,
    "tke_rel_l2_score": 0.1,
    "mvpe_rel_l2_score": 0.1,
    "time_score": 0.1,
    "native_sps_score": 0.2,
}
DR_FIELDS = (
    "temporal_jitter_max",
    "input_gain_std",
    "input_noise_std",
    "input_blur_probability",
    "input_blur_strength",
)
CONTROL_DR = {
    "temporal_jitter_max": 0,
    "input_gain_std": 0.0,
    "input_noise_std": 0.0,
    "input_blur_probability": 0.0,
    "input_blur_strength": 0.0,
}
CANDIDATE_DR = {
    "temporal_jitter_max": 9,
    "input_gain_std": 0.02,
    "input_noise_std": 0.02,
    "input_blur_probability": 0.25,
    "input_blur_strength": 0.25,
}


CONTRACT = {
    "evidence_class": (
        "paired Real-finetuning clean2544 LOAO; not full-lifecycle LOAO"),
    "fold_order": list(FOLD_ORDER),
    "fresh_process_per_fold": True,
    "real_training": {
        "updates": UPDATES,
        "batch_size": 8,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "scheduler_t_max": 5000,
        "seed": SEED,
        "accuracy_weights": [5.0, 3.0, 2.0],
        "sps_uq_only": True,
    },
    "data": {
        "trajectories": EXPECTED_TRAJECTORIES,
        "prefix_windows": EXPECTED_WINDOWS,
        "tail_arrays_read": 0,
        "excluded": EXCLUDED_BASENAME,
        "prefix_slice": "u/v[:val_boundary,::2,::2]",
        "manifest_sha256": FORMAL_MANIFEST_SHA256,
    },
    "gate_a": {
        "pooled_point_532_delta_min": 0.10,
        "pooled_exact_non_time_final_delta_min": 0.10,
        "component_floor": -0.02,
        "sps_floor": -0.10,
        "positive_folds_min": 4,
        "worst_fold_exact_non_time_final_delta_min": -0.05,
    },
    "quick_stop": (
        "STOP when both AoA10 and AoA0 are non-positive, or any fold exact "
        "non-Time final delta is below -0.05"),
    "submission_authorized": False,
}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Use the Sim trainer's single registered tensor-state digest."""
    return ensemble_train.model_state_sha256(state)


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def git_identity() -> dict[str, str]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", *arguments), cwd=REPO_ROOT, text=True).strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
    }


def validate_formal_git_identity(identity: dict[str, str]) -> dict[str, object]:
    """Reject code drift while allowing only recorded read-only data mounts."""
    lines = [line for line in identity["status_porcelain"].splitlines()
             if line]
    allowed_mounts = []
    rejected = []
    for line in lines:
        relative = line[3:] if line.startswith("?? ") else ""
        path = REPO_ROOT / relative
        if (relative in {"data", "kits/kits"} and path.is_symlink()):
            allowed_mounts.append(relative)
        else:
            rejected.append(line)
    if rejected:
        raise RuntimeError(
            f"formal DR1 LOAO worktree has code/unknown drift: {rejected}")
    return {
        **identity,
        "allowed_untracked_runtime_mounts": sorted(allowed_mounts),
        "scientific_worktree_clean": True,
    }


def normalized_sha256(value: str, label: str) -> str:
    result = value.strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef"
                                for character in result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def validate_runtime_manifest(
        manifest: object, expected_sha256: str = FORMAL_MANIFEST_SHA256,
        ) -> tuple[list[dict], dict]:
    if not isinstance(manifest, list):
        raise TypeError("clean runtime manifest must be a list")
    required = {
        "file", "frames", "re", "aoa", "train_starts", "val_boundary"}
    rows: list[dict] = []
    seen: set[str] = set()
    for raw in manifest:
        if not isinstance(raw, dict):
            raise TypeError("clean runtime manifest rows must be mappings")
        if "val_starts" in raw:
            raise ValueError("runtime manifest must not contain val_starts")
        missing = required - set(raw)
        if missing:
            raise ValueError(f"runtime manifest row is missing {sorted(missing)}")
        filename = raw["file"]
        if (not isinstance(filename, str) or not filename
                or Path(filename).name != filename):
            raise ValueError("runtime manifest file keys must be basenames")
        if filename == EXCLUDED_BASENAME:
            raise ValueError(f"excluded trajectory {filename} entered manifest")
        if filename in seen:
            raise ValueError(f"duplicate runtime manifest file {filename}")
        seen.add(filename)
        aoa = int(raw["aoa"])
        if aoa not in AOA_VALUES:
            raise ValueError(f"unexpected AoA {aoa} in runtime manifest")
        frames = int(raw["frames"])
        boundary = int(raw["val_boundary"])
        starts = list(raw["train_starts"])
        if frames <= 0 or not 0 < boundary <= frames:
            raise ValueError(f"invalid prefix boundary for {filename}")
        if (not starts or any(
                isinstance(start, bool) or not isinstance(start, int)
                or start < 0 or start + 40 > boundary for start in starts)
                or starts != sorted(set(starts))):
            raise ValueError(f"invalid training starts for {filename}")
        rows.append({
            "file": filename,
            "frames": frames,
            "re": int(raw["re"]),
            "aoa": aoa,
            "train_starts": starts,
            "val_boundary": boundary,
        })
    digest = canonical_sha256(rows)
    expected = normalized_sha256(expected_sha256, "manifest SHA-256")
    counts = {
        aoa: sum(len(row["train_starts"]) for row in rows
                 if row["aoa"] == aoa)
        for aoa in AOA_VALUES
    }
    checks = {
        "semantic_sha256": digest == expected,
        "trajectories": len(rows) == EXPECTED_TRAJECTORIES,
        "windows": sum(counts.values()) == EXPECTED_WINDOWS,
        "aoa_counts": counts == EXPECTED_AOA_COUNTS,
        "excluded_absent": EXCLUDED_BASENAME not in seen,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"clean81 runtime manifest contract failed: {failed}")
    return rows, {
        "semantic_sha256": digest,
        "trajectories": len(rows),
        "windows": sum(counts.values()),
        "aoa_windows": {str(key): value for key, value in counts.items()},
        "excluded": EXCLUDED_BASENAME,
        "val_starts_present": False,
    }


def load_manifest(path: Path, expected_sha256: str) -> tuple[list[dict], dict]:
    return validate_runtime_manifest(
        json.loads(path.read_text(encoding="utf-8")), expected_sha256)


class PrefixAccessAudit:
    """State machine guarding H5 access before and after dual sealing."""

    def __init__(self, held_aoa: int):
        if held_aoa not in AOA_VALUES:
            raise ValueError(f"unexpected held AoA {held_aoa}")
        self.held_aoa = held_aoa
        self.control_seal: dict | None = None
        self.candidate_seal: dict | None = None
        self.events: list[dict] = []
        self.opened: set[str] = set()
        self.max_frame_exclusive: dict[str, int] = {}

    @property
    def both_sealed(self) -> bool:
        return self.control_seal is not None and self.candidate_seal is not None

    def seal(self, arm: str, path: Path, file_sha256: str,
             state_sha256: str) -> None:
        if arm not in {"control", "candidate"}:
            raise ValueError(f"unknown arm {arm}")
        record = {
            "path": str(path.resolve()),
            "file_sha256": normalized_sha256(file_sha256, f"{arm} seal file"),
            "model_state_sha256": normalized_sha256(
                state_sha256, f"{arm} seal model state"),
            "sealed_monotonic_ns": time.monotonic_ns(),
        }
        if arm == "control":
            if self.control_seal is not None:
                raise RuntimeError("control endpoint was sealed twice")
            self.control_seal = record
        else:
            if self.candidate_seal is not None:
                raise RuntimeError("candidate endpoint was sealed twice")
            self.candidate_seal = record

    def before_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        aoa = int(row["aoa"])
        if filename == EXCLUDED_BASENAME:
            raise RuntimeError("excluded H5 reached the open boundary")
        if phase == "fit":
            if aoa == self.held_aoa:
                raise RuntimeError("held AoA reached fit-prefix H5 opener")
            if self.both_sealed:
                raise RuntimeError("fit prefix opened after endpoints were sealed")
        elif phase == "held":
            if aoa != self.held_aoa:
                raise RuntimeError("other-four AoA reached held-prefix H5 opener")
            if not self.both_sealed:
                raise RuntimeError("held prefix cannot open before both seals")
        else:
            raise ValueError(f"unknown prefix access phase {phase}")

    def record_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        if filename in self.opened:
            raise RuntimeError(f"H5 file {filename} was opened more than once")
        boundary = int(row["val_boundary"])
        self.opened.add(filename)
        self.max_frame_exclusive[filename] = boundary
        self.events.append({
            "file": filename,
            "aoa": int(row["aoa"]),
            "phase": phase,
            "slice": "u/v[:val_boundary,::2,::2]",
            "max_read_frame_exclusive": boundary,
            "max_read_frame_inclusive": boundary - 1,
            "both_arms_sealed_before_open": self.both_sealed,
            "opened_monotonic_ns": time.monotonic_ns(),
        })

    def report(self, expected_allowlist: Sequence[str]) -> dict:
        expected = sorted(expected_allowlist)
        opened = sorted(self.opened)
        held_events = [row for row in self.events if row["phase"] == "held"]
        return {
            "opened_file_allowlist": opened,
            "expected_file_allowlist": expected,
            "allowlist_exact": opened == expected,
            "max_read_frame_exclusive_by_file": dict(sorted(
                self.max_frame_exclusive.items())),
            "tail_arrays_read": 0,
            "excluded_opened": EXCLUDED_BASENAME in self.opened,
            "held_first_open_after_both_seals": bool(
                held_events and held_events[0]["both_arms_sealed_before_open"]),
            "control_seal": self.control_seal,
            "candidate_seal": self.candidate_seal,
            "events": self.events,
        }


def load_prefix_rows(
        real_dir: Path, rows: Sequence[dict], phase: str,
        audit: PrefixAccessAudit,
        ) -> tuple[list[torch.Tensor], list[dict], list]:
    trajectories: list[torch.Tensor] = []
    metadata: list[dict] = []
    windows = []
    root = real_dir.resolve()
    for row in rows:
        audit.before_open(row, phase)
        filename = str(row["file"])
        path = (root / filename).resolve()
        if path.parent != root or path.name != filename:
            raise ValueError(f"unsafe H5 path {filename}")
        if not path.is_file():
            raise FileNotFoundError(path)
        boundary = int(row["val_boundary"])
        with h5py.File(path, "r") as handle:
            u_dataset = handle["u"]
            v_dataset = handle["v"]
            if u_dataset.shape != v_dataset.shape:
                raise ValueError(f"u/v shapes differ in {filename}")
            if int(u_dataset.shape[0]) != int(row["frames"]):
                raise ValueError(f"frame count changed for {filename}")
            if int(np.asarray(handle["re"])) != int(row["re"]):
                raise ValueError(f"Re changed for {filename}")
            if int(np.asarray(handle["aoa"])) != int(row["aoa"]):
                raise ValueError(f"AoA changed for {filename}")
            u = np.asarray(
                u_dataset[:boundary, ::2, ::2], dtype=np.float32)
            v = np.asarray(
                v_dataset[:boundary, ::2, ::2], dtype=np.float32)
        if u.shape != v.shape or int(u.shape[0]) != boundary:
            raise ValueError(f"prefix shape mismatch in {filename}")
        audit.record_open(row, phase)
        trajectory = len(trajectories)
        values = np.stack((u, v), axis=-1)
        trajectories.append(torch.from_numpy(np.ascontiguousarray(values)))
        metadata.append({
            "file": filename,
            "frames": boundary,
            "source_frames": int(row["frames"]),
            "re": int(row["re"]),
            "aoa": int(row["aoa"]),
            "val_boundary": boundary,
        })
        windows.extend(
            ensemble_train.common.repro.Window(trajectory, int(start))
            for start in row["train_starts"])
    return trajectories, metadata, windows


def load_json_mapping(path: Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON mapping")
    return value


def _audit_training_config(audit: dict) -> dict:
    config = audit.get("training_config")
    if not isinstance(config, dict):
        raise ValueError("Sim audit has no complete training_config")
    expected = audit.get("training_config_sha256")
    if canonical_sha256(config) != expected:
        raise ValueError("Sim audit training_config SHA mismatch")
    return config


def validate_sim_run_manifest(
        run_manifest: object, *, control_audit_sha256: str,
        candidate_audit_sha256: str, control_checkpoint_sha256: str,
        candidate_checkpoint_sha256: str) -> dict:
    if not isinstance(run_manifest, dict):
        raise TypeError("Sim pair run manifest must be a mapping")
    expected = {
        "schema": RUN_MANIFEST_SCHEMA,
        "source_git_commit": FORMAL_SIM_SOURCE_COMMIT,
        "sequence": ["candidate", "control"],
        "same_shell_sequential": True,
        "candidate_completed_before_control_started": True,
    }
    for key, value in expected.items():
        if run_manifest.get(key) != value:
            raise ValueError(f"Sim pair run manifest {key} changed")
    for arm, audit_sha, checkpoint_sha in (
            ("control", control_audit_sha256, control_checkpoint_sha256),
            ("candidate", candidate_audit_sha256,
             candidate_checkpoint_sha256)):
        row = run_manifest.get(arm)
        if not isinstance(row, dict):
            raise ValueError(f"Sim pair run manifest has no {arm} row")
        if row.get("audit_sha256") != audit_sha:
            raise RuntimeError(f"{arm} audit is not bound to run manifest")
        if row.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError(
                f"{arm} checkpoint is not bound to run manifest")
    return dict(run_manifest)


def validate_sim_audit_pair(
        control: dict, candidate: dict,
        run_manifest: dict | None = None) -> dict:
    for label, audit, expected_dr in (
            ("control", control, CONTROL_DR),
            ("candidate", candidate, CANDIDATE_DR)):
        if audit.get("schema") != SIM_AUDIT_SCHEMA:
            raise ValueError(f"{label} Sim audit schema changed")
        if audit.get("variant") != label:
            raise ValueError(f"{label} Sim audit variant changed")
        for key in (
                "initial_model_state_sha256", "base_sample_stream_sha256",
                "jitter_shift_stream_sha256", "model_config_sha256",
                "endpoint_model_state_sha256", "endpoint_checkpoint_sha256"):
            normalized_sha256(str(audit.get(key, "")), f"{label} {key}")
        model_config = audit.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError(f"{label} Sim audit has no model_config")
        if canonical_sha256(model_config) != audit["model_config_sha256"]:
            raise ValueError(f"{label} Sim audit model_config SHA mismatch")
        training = _audit_training_config(audit)
        observed_dr = {key: training.get(key) for key in DR_FIELDS}
        if observed_dr != expected_dr:
            raise ValueError(f"{label} DR bundle changed: {observed_dr}")
        if training.get("teacher_cache") is not None:
            raise ValueError(f"{label} teacher_cache must be None")
        if float(training.get("distill_weight", math.nan)) != 0.0:
            raise ValueError(f"{label} distill_weight must be zero")
        hardware = audit.get("hardware_fingerprint")
        if not isinstance(hardware, dict):
            raise ValueError(f"{label} has no hardware fingerprint")
        required_hardware = {
            "gpu_uuid", "gpu_name", "torch_version",
            "cuda_runtime_version", "cudnn_version"}
        if required_hardware - set(hardware):
            raise ValueError(
                f"{label} hardware fingerprint is incomplete")
        expected_common = {
            "phase": "sim", "updates": 10000, "batch_size": 8,
            "seed": SEED, "num_workers": 0, "window_stride": 10,
        }
        for key, value in expected_common.items():
            if training.get(key) != value:
                raise ValueError(f"{label} Sim {key} changed")
    if (control["initial_model_state_sha256"]
            != candidate["initial_model_state_sha256"]):
        raise RuntimeError("Sim initial model-state SHAs are not paired")
    if (control["base_sample_stream_sha256"]
            != candidate["base_sample_stream_sha256"]):
        raise RuntimeError("Sim base sample streams are not paired")
    if control["model_config_sha256"] != candidate["model_config_sha256"]:
        raise RuntimeError("Sim model configs differ")
    if control["hardware_fingerprint"] != candidate["hardware_fingerprint"]:
        raise RuntimeError("Sim arms did not run on the same physical GPU stack")
    if (candidate.get("run_ordinal"), control.get("run_ordinal")) != (1, 2):
        raise RuntimeError("formal Sim order must be candidate then control")
    try:
        candidate_started = datetime.fromisoformat(
            str(candidate["started_at_utc"]).replace("Z", "+00:00"))
        control_started = datetime.fromisoformat(
            str(control["started_at_utc"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise ValueError("Sim audits lack valid UTC start evidence") from exc
    interval_verified = False
    if (candidate.get("completed_at_utc") is not None
            and control.get("completed_at_utc") is not None):
        try:
            candidate_completed = datetime.fromisoformat(
                str(candidate["completed_at_utc"]).replace("Z", "+00:00"))
            control_completed = datetime.fromisoformat(
                str(control["completed_at_utc"]).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Sim audits have invalid UTC completion evidence") \
                from exc
        if not (
                candidate_started <= candidate_completed
                <= control_started <= control_completed):
            raise RuntimeError(
                "Sim arms were not sequential candidate then control")
        interval_verified = True
    else:
        if run_manifest is None:
            raise ValueError(
                "legacy Sim audits require an external sequential run manifest")
        if not candidate_started < control_started:
            raise RuntimeError("candidate did not start before control")
    control_training = _audit_training_config(control)
    candidate_training = _audit_training_config(candidate)
    changed = {
        key for key in set(control_training) | set(candidate_training)
        if control_training.get(key) != candidate_training.get(key)}
    operational = {"output_dir", "sim_audit_variant"}
    if changed - (operational | set(DR_FIELDS)):
        raise RuntimeError(
            f"unregistered Sim config differences: {sorted(changed)}")
    if not set(DR_FIELDS).issubset(changed):
        raise RuntimeError(
            f"Sim config diff omitted DR fields: {sorted(set(DR_FIELDS)-changed)}")
    return {
        "initial_model_state_sha256": control[
            "initial_model_state_sha256"],
        "base_sample_stream_sha256": control["base_sample_stream_sha256"],
        "control_jitter_shift_stream_sha256": control[
            "jitter_shift_stream_sha256"],
        "candidate_jitter_shift_stream_sha256": candidate[
            "jitter_shift_stream_sha256"],
        "model_config_sha256": control["model_config_sha256"],
        "config_diff": {
            key: {"control": control_training[key],
                  "candidate": candidate_training[key]}
            for key in DR_FIELDS
        },
        "ignored_operational_diff": {
            key: {"control": control_training.get(key),
                  "candidate": candidate_training.get(key)}
            for key in sorted(changed & operational)
        },
        "hardware_fingerprint": control["hardware_fingerprint"],
        "sequential_order": {
            "first": "candidate",
            "second": "control",
            "candidate_started_at_utc": candidate["started_at_utc"],
            "candidate_completed_at_utc": candidate.get("completed_at_utc"),
            "control_started_at_utc": control["started_at_utc"],
            "control_completed_at_utc": control.get("completed_at_utc"),
            "audit_intervals_verified": interval_verified,
            "external_run_manifest_required": not interval_verified,
        },
        "teacher_cache": None,
        "distill_weight": 0.0,
    }


def inspect_sim_checkpoint(path: Path, expected_file_sha256: str,
                           audit: dict) -> dict:
    expected_file = normalized_sha256(
        expected_file_sha256, f"{path.name} expected checkpoint")
    actual_file = sha256(path)
    if actual_file != expected_file:
        raise RuntimeError(f"Sim checkpoint file SHA changed for {path}")
    if audit["endpoint_checkpoint_sha256"] != actual_file:
        raise RuntimeError(f"Sim audit endpoint file SHA mismatch for {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("phase") != "sim":
        raise ValueError(f"{path} is not a Sim checkpoint")
    if int(payload.get("iteration", -1)) != 10000:
        raise ValueError(f"{path} is not the terminal Sim10k checkpoint")
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, Mapping):
        raise ValueError(f"{path} has no model config/state")
    config_sha = canonical_sha256(config)
    state_sha = model_state_sha256(state)
    if config_sha != audit["model_config_sha256"]:
        raise RuntimeError(f"{path} config disagrees with Sim audit")
    if state_sha != audit["endpoint_model_state_sha256"]:
        raise RuntimeError(f"{path} endpoint state disagrees with Sim audit")
    stored = dict(config)
    if stored.pop("architecture", None) != "ensemble_unet":
        raise ValueError(f"{path} is not an ensemble U-Net")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
    parameters = sum(value.numel() for value in model.parameters())
    del model, payload, state
    gc.collect()
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"{path} parameter count {parameters} changed")
    return {
        "path": str(path.resolve()),
        "checkpoint_sha256": actual_file,
        "observed_file_mtime_ns": path.stat().st_mtime_ns,
        "model_state_sha256": state_sha,
        "model_config_sha256": config_sha,
        "parameters": parameters,
    }


def build_pair_attestation(args: argparse.Namespace) -> dict:
    control_audit = load_json_mapping(args.control_sim_audit, "control Sim audit")
    candidate_audit = load_json_mapping(
        args.candidate_sim_audit, "candidate Sim audit")
    control_audit_sha = sha256(args.control_sim_audit)
    candidate_audit_sha = sha256(args.candidate_sim_audit)
    run_manifest = validate_sim_run_manifest(
        load_json_mapping(args.sim_pair_run_manifest, "Sim pair run manifest"),
        control_audit_sha256=control_audit_sha,
        candidate_audit_sha256=candidate_audit_sha,
        control_checkpoint_sha256=args.control_sim_sha256,
        candidate_checkpoint_sha256=args.candidate_sim_sha256)
    paired = validate_sim_audit_pair(
        control_audit, candidate_audit, run_manifest)
    control = inspect_sim_checkpoint(
        args.control_sim, args.control_sim_sha256, control_audit)
    candidate = inspect_sim_checkpoint(
        args.candidate_sim, args.candidate_sim_sha256, candidate_audit)
    if control["model_config_sha256"] != candidate["model_config_sha256"]:
        raise RuntimeError("validated Sim checkpoint configs differ")
    return {
        "schema": PAIR_ATTESTATION_SCHEMA,
        "control": control,
        "candidate": candidate,
        "control_sim_audit": {
            "path": str(args.control_sim_audit.resolve()),
            "file_sha256": control_audit_sha,
            "observed_file_mtime_ns": args.control_sim_audit.stat().st_mtime_ns,
            "mtime_is_scientific_gate": False,
        },
        "candidate_sim_audit": {
            "path": str(args.candidate_sim_audit.resolve()),
            "file_sha256": candidate_audit_sha,
            "observed_file_mtime_ns": (
                args.candidate_sim_audit.stat().st_mtime_ns),
            "mtime_is_scientific_gate": False,
        },
        "sim_pair_run_manifest": {
            "path": str(args.sim_pair_run_manifest.resolve()),
            "file_sha256": sha256(args.sim_pair_run_manifest),
            "payload": run_manifest,
            "mtime_is_scientific_gate": False,
        },
        "paired": paired,
        "pass": True,
    }


def load_pair_attestation(path: Path) -> dict:
    value = load_json_mapping(path, "Sim pair attestation")
    if value.get("schema") != PAIR_ATTESTATION_SCHEMA or value.get("pass") is not True:
        raise ValueError("Sim pair attestation is not a passed DR1 attestation")
    for arm in ("control", "candidate"):
        row = value.get(arm)
        if not isinstance(row, dict):
            raise ValueError(f"pair attestation has no {arm} row")
        actual = sha256(Path(row["path"]))
        if actual != row["checkpoint_sha256"]:
            raise RuntimeError(f"{arm} Sim checkpoint drifted after attestation")
    return value


def load_model(path: Path, normalizer: smu1.FoldNormalizer,
               device: torch.device, expected_state_sha256: str,
               expected_config_sha256: str) -> EnsembleUNet2d:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    state = payload["model_state_dict"]
    if canonical_sha256(config) != expected_config_sha256:
        raise RuntimeError(f"model config drifted in {path}")
    if model_state_sha256(state) != expected_state_sha256:
        raise RuntimeError(f"model state drifted in {path}")
    stored = dict(config)
    if stored.pop("architecture", None) != "ensemble_unet":
        raise ValueError(f"{path} is not an ensemble U-Net")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
    model.load_state_dict(state, strict=True)
    if sum(value.numel() for value in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("C208 parameter count changed")
    ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def bootstrap_mask_by_trajectory(trajectory_count: int) -> dict[int, list[bool]]:
    result = {}
    for trajectory in range(trajectory_count):
        mask = ensemble_train.trajectory_member_mask(
            torch.tensor([trajectory]), 0.75, SEED, 4)[0]
        result[trajectory] = [bool(value) for value in mask.tolist()]
    return result


def save_sealed_endpoint(
        path: Path, model: EnsembleUNet2d, arm: str, held_aoa: int,
        start: dict, normalizer: smu1.FoldNormalizer,
        stream_sha256: str) -> dict:
    state = ensemble_train.common.checkpoint_state(model)
    state_sha = model_state_sha256(state)
    payload = {
        "schema": "c208_dr1_clean81_sealed_endpoint_v1",
        "phase": "real",
        "iteration": UPDATES,
        "arm": arm,
        "held_aoa": held_aoa,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "start_sim_checkpoint_sha256": start["checkpoint_sha256"],
        "start_sim_model_state_sha256": start["model_state_sha256"],
        "sample_stream_sha256": stream_sha256,
        "normalizer": ensemble_train.common.normalizer_dict(normalizer),
        "held_prefix_opened_before_seal": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256(path),
        "model_state_sha256": state_sha,
        "model_config_sha256": canonical_sha256(payload["model_config"]),
        "iteration": UPDATES,
    }


def load_sealed_model(path: Path, seal: dict,
                      normalizer: smu1.FoldNormalizer,
                      device: torch.device) -> EnsembleUNet2d:
    if sha256(path) != seal["file_sha256"]:
        raise RuntimeError(f"sealed endpoint file drifted: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "c208_dr1_clean81_sealed_endpoint_v1":
        raise ValueError(f"invalid sealed endpoint {path}")
    if payload.get("model_state_sha256") != seal["model_state_sha256"]:
        raise RuntimeError(f"sealed endpoint state metadata drifted: {path}")
    config = payload["model_config"]
    state = payload["model_state_dict"]
    if model_state_sha256(state) != seal["model_state_sha256"]:
        raise RuntimeError(f"sealed endpoint tensor state drifted: {path}")
    stored = dict(config)
    if stored.pop("architecture", None) != "ensemble_unet":
        raise ValueError(f"sealed endpoint architecture changed: {path}")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
    model.load_state_dict(state, strict=True)
    ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def generalized_final(metrics: Mapping[str, float]) -> float:
    values = {
        "rel_l2_score": float(metrics["rel_l2_score"]),
        "tke_rel_l2_score": float(metrics["tke_rel_l2_score"]),
        "mvpe_rel_l2_score": float(metrics["mvpe_rel_l2_score"]),
        "time_score": TIME_SCORE,
        "native_sps_score": float(metrics["native_sps_score"]),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("generalized-final components must be finite/non-negative")
    return float(sum(
        GENERALIZED_WEIGHTS[key] * value ** GENERALIZED_Q
        for key, value in values.items()) ** (1.0 / GENERALIZED_Q))


def compare_metrics(candidate: dict, control: dict) -> dict:
    delta = smu1.metric_delta(candidate, control)
    control_final = generalized_final(control)
    candidate_final = generalized_final(candidate)
    return {
        "control": control,
        "candidate": candidate,
        "delta": delta,
        "control_exact_fixed_time_final": control_final,
        "candidate_exact_fixed_time_final": candidate_final,
        "exact_non_time_final_delta": candidate_final - control_final,
    }


def fold_worker(args: argparse.Namespace) -> dict:
    identity = validate_formal_git_identity(git_identity())
    if args.held_aoa not in AOA_VALUES:
        raise ValueError("fold worker requires one formal held AoA")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse fold output {args.output_dir}")
    rows, manifest_report = load_manifest(
        args.clean_manifest, args.clean_manifest_sha256)
    pair = load_pair_attestation(args.pair_attestation)
    fit_rows = [row for row in rows if row["aoa"] != args.held_aoa]
    held_rows = [row for row in rows if row["aoa"] == args.held_aoa]
    if sum(len(row["train_starts"]) for row in fit_rows) != (
            EXPECTED_FIT_COUNTS[args.held_aoa]):
        raise RuntimeError("other-four fit count changed")
    if sum(len(row["train_starts"]) for row in held_rows) != (
            EXPECTED_AOA_COUNTS[args.held_aoa]):
        raise RuntimeError("held fold count changed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal DR1 LOAO fold requires CUDA")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.fold_wall_budget_seconds
    access = PrefixAccessAudit(args.held_aoa)

    fit_trajectories, fit_metadata, fit_windows = load_prefix_rows(
        args.real_dir, fit_rows, "fit", access)
    if len(fit_windows) != EXPECTED_FIT_COUNTS[args.held_aoa]:
        raise RuntimeError("loaded other-four fit count changed")
    fit_dataset = ensemble_train.common.repro.WindowDataset(
        fit_trajectories, fit_windows)
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, fit_windows)
    statistics = smu1.fold_statistics(sufficient, range(len(fit_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    schedule = smu1.paired_batch_schedule(
        range(len(fit_windows)), fit_windows, updates=UPDATES, seed=SEED)
    frozen_masks = bootstrap_mask_by_trajectory(len(fit_trajectories))

    arm_reports = {}
    for arm in ("control", "candidate"):
        start = pair[arm]
        ensemble_train.seed_everything(SEED)
        model = load_model(
            Path(start["path"]), normalizer, device,
            start["model_state_sha256"], start["model_config_sha256"])
        initial_state_sha = model_state_sha256(model.state_dict())
        training = smu1.train_arm(
            model, fit_dataset, schedule, normalizer, device, deadline,
            f"dr1_held{args.held_aoa}_{arm}",
            bootstrap_mask_by_trajectory=frozen_masks)
        if training["actual_stream_sha256"] != schedule["sha256"]:
            raise RuntimeError(f"{arm} Real sample stream changed")
        seal_path = args.output_dir / f"{arm}_sealed_u4000.pth"
        seal = save_sealed_endpoint(
            seal_path, model, arm, args.held_aoa, start, normalizer,
            schedule["sha256"])
        access.seal(
            arm, seal_path, seal["file_sha256"],
            seal["model_state_sha256"])
        arm_reports[arm] = {
            "start_model_state_sha256": initial_state_sha,
            "training": training,
            "seal": seal,
        }
        del model
        torch.cuda.empty_cache()
        gc.collect()
    if not access.both_sealed:
        raise RuntimeError("both endpoints were not sealed before held load")

    held_trajectories, held_metadata, held_windows = load_prefix_rows(
        args.real_dir, held_rows, "held", access)
    if len(held_windows) != EXPECTED_AOA_COUNTS[args.held_aoa]:
        raise RuntimeError("loaded held count changed")
    held_dataset = ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    evaluate_indices = list(range(len(held_windows)))
    for arm in ("control", "candidate"):
        seal = arm_reports[arm]["seal"]
        model = load_sealed_model(
            Path(seal["path"]), seal, normalizer, device)
        metrics, state, held_audit = smu1.evaluate_model(
            model, held_dataset, evaluate_indices, normalizer,
            device, deadline)
        arm_reports[arm].update({
            "metrics": metrics,
            "metric_state": state,
            "held_target_audit": held_audit,
        })
        del model
        torch.cuda.empty_cache()
        gc.collect()
    if (arm_reports["control"]["held_target_audit"]
            != arm_reports["candidate"]["held_target_audit"]):
        raise RuntimeError("paired held targets/masks differ")
    access_report = access.report([row["file"] for row in rows])
    if not (
            access_report["allowlist_exact"]
            and access_report["tail_arrays_read"] == 0
            and not access_report["excluded_opened"]
            and access_report["held_first_open_after_both_seals"]):
        raise RuntimeError("prefix access audit failed")
    comparison = compare_metrics(
        arm_reports["candidate"]["metrics"],
        arm_reports["control"]["metrics"])
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "mode": "fold_worker",
        "held_aoa": args.held_aoa,
        "fit_samples": len(fit_windows),
        "samples": len(held_windows),
        "manifest": manifest_report,
        "normalizer": {
            "kind": "other-four optimizer-prefix windows only",
            "held_target_in_normalizer": False,
            "statistics": [value.detach().cpu().tolist()
                           for value in statistics],
            "statistics_sha256": smu1.statistics_digest(statistics),
        },
        "sample_stream_sha256": schedule["sha256"],
        "fit_ordered_key_sha256": smu1.ordered_held_key_digest(
            range(len(fit_windows)), fit_windows, fit_metadata),
        "held_ordered_key_sha256": smu1.ordered_held_key_digest(
            range(len(held_windows)), held_windows, held_metadata),
        "control": arm_reports["control"],
        "candidate": arm_reports["candidate"],
        "comparison": comparison,
        "prefix_access_audit": access_report,
        "git": identity,
        "elapsed_seconds": time.perf_counter() - started,
        "release_checkpoint_written": False,
        "submission_authorized": False,
    }


def merge_fold_results(rows: Sequence[dict]) -> dict:
    if len(rows) != len(AOA_VALUES):
        raise ValueError("formal DR1 gate requires all five folds")
    if [int(row["held_aoa"]) for row in rows] != list(FOLD_ORDER):
        raise ValueError("DR1 fold order changed")
    if sum(int(row["samples"]) for row in rows) != EXPECTED_WINDOWS:
        raise RuntimeError("pooled fold samples do not equal clean2544")
    control_state = smu1.merge_metric_states([
        row["control"]["metric_state"] for row in rows])
    candidate_state = smu1.merge_metric_states([
        row["candidate"]["metric_state"] for row in rows])
    if (int(control_state["samples"]) != EXPECTED_WINDOWS
            or int(candidate_state["samples"]) != EXPECTED_WINDOWS):
        raise RuntimeError("pooled metric states do not cover clean2544")
    pooled = compare_metrics(
        smu1.finalize_metric_state(candidate_state),
        smu1.finalize_metric_state(control_state))
    delta = pooled["delta"]
    conditions = {
        "pooled_point_532_delta": delta["point_score"] >= 0.10,
        "pooled_exact_non_time_final_delta": (
            pooled["exact_non_time_final_delta"] >= 0.10),
        "rel_floor": delta["rel_l2_score"] >= -0.02,
        "tke_floor": delta["tke_rel_l2_score"] >= -0.02,
        "mvpe_floor": delta["mvpe_rel_l2_score"] >= -0.02,
        "sps_floor": delta["native_sps_score"] >= -0.10,
        "positive_folds": sum(
            row["comparison"]["exact_non_time_final_delta"] > 0.0
            for row in rows) >= 4,
        "worst_fold": min(
            row["comparison"]["exact_non_time_final_delta"]
            for row in rows) >= -0.05,
        "samples_exact": sum(int(row["samples"]) for row in rows) == 2544,
        "paired_streams": all(
            row["control"]["training"]["actual_stream_sha256"]
            == row["candidate"]["training"]["actual_stream_sha256"]
            == row["sample_stream_sha256"] for row in rows),
        "prefix_access": all(
            row["prefix_access_audit"]["allowlist_exact"]
            and row["prefix_access_audit"]["tail_arrays_read"] == 0
            and row["prefix_access_audit"][
                "held_first_open_after_both_seals"]
            for row in rows),
    }
    passed = all(conditions.values())
    return {
        "pooled_control_state": control_state,
        "pooled_candidate_state": candidate_state,
        "pooled": pooled,
        "gate": {
            "pass": passed,
            "decision": "GO_DR1_GATE_B" if passed else "STOP_DR1_GATE_A",
            "conditions": conditions,
        },
    }


def fold_command(args: argparse.Namespace, held_aoa: int,
                 output_dir: Path, pair_attestation: Path) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()),
        "--fold-worker", "--held-aoa", str(held_aoa),
        "--output-dir", str(output_dir),
        "--real-dir", str(args.real_dir),
        "--clean-manifest", str(args.clean_manifest),
        "--clean-manifest-sha256", args.clean_manifest_sha256,
        "--pair-attestation", str(pair_attestation),
        "--device", args.device,
        "--fold-wall-budget-seconds", str(args.fold_wall_budget_seconds),
    ]


def orchestrate(args: argparse.Namespace) -> dict:
    identity = validate_formal_git_identity(git_identity())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    rows, manifest_report = load_manifest(
        args.clean_manifest, args.clean_manifest_sha256)
    if sha256(args.clean_prefix_stats) != FORMAL_PREFIX_STATS_SHA256:
        raise RuntimeError("clean81 prefix stats artifact SHA changed")
    pair = build_pair_attestation(args)
    args.output_dir.mkdir(parents=True)
    runtime_manifest = args.output_dir / "runtime_train_manifest.json"
    write_json_atomic(runtime_manifest, rows)
    # Every fresh worker consumes the immutable copy under this run root, not
    # the caller-owned source path which could drift after PREPARE.
    args.clean_manifest = runtime_manifest
    pair_path = args.output_dir / "sim_pair_attestation.json"
    write_json_atomic(pair_path, pair)
    prepare = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract": CONTRACT,
        "git": identity,
        "manifest": manifest_report,
        "runtime_manifest": {
            "path": str(runtime_manifest.resolve()),
            "file_sha256": sha256(runtime_manifest),
            "semantic_sha256": manifest_report["semantic_sha256"],
        },
        "clean_prefix_stats": {
            "path": str(args.clean_prefix_stats.resolve()),
            "file_sha256": FORMAL_PREFIX_STATS_SHA256,
            "note": "Sim-only common SPS scale; never used as LOAO fold stats",
        },
        "sim_pair_attestation": {
            "path": str(pair_path.resolve()),
            "file_sha256": sha256(pair_path),
        },
        "decision": "DR1_GATE_A_PREPARED",
        "submission_authorized": False,
    }
    write_json_atomic(args.output_dir / "PREPARE.json", prepare)
    if args.prepare_only:
        return {
            **prepare,
            "decision": "DR1_GATE_A_PREPARED_NO_GPU",
            "release_checkpoint_written": False,
            "submission_authorized": False,
        }
    started = time.perf_counter()
    fold_rows = []
    for held in FOLD_ORDER:
        elapsed = time.perf_counter() - started
        remaining = args.total_wall_budget_seconds - elapsed
        if remaining <= 0:
            raise TimeoutError("DR1 total Real LOAO wall budget expired")
        fold_dir = args.output_dir / f"held_{held:02d}"
        command = fold_command(args, held, fold_dir, pair_path)
        completed = subprocess.run(
            command, cwd=REPO_ROOT, text=True, check=False,
            timeout=remaining)
        result_path = fold_dir / "result.json"
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(
                f"held AoA {held} fresh worker failed with "
                f"exit {completed.returncode}")
        row = load_json_mapping(result_path, f"held AoA {held} result")
        if row.get("decision") != "DR1_FOLD_COMPLETE":
            raise RuntimeError(f"held AoA {held} did not complete")
        fold_rows.append(row["result"])
        write_json_atomic(args.output_dir / "progress.json", {
            **prepare,
            "decision": "DR1_GATE_A_IN_PROGRESS",
            "completed_folds": [entry["held_aoa"] for entry in fold_rows],
            "folds": fold_rows,
            "elapsed_seconds": time.perf_counter() - started,
        })
        current_delta = fold_rows[-1]["comparison"][
            "exact_non_time_final_delta"]
        if current_delta < -0.05:
            return {
                **prepare,
                "decision": "STOP_DR1_FOLD_BELOW_FLOOR",
                "folds": fold_rows,
                "elapsed_seconds": time.perf_counter() - started,
            }
        if len(fold_rows) >= 2 and all(
                row["comparison"]["exact_non_time_final_delta"] <= 0.0
                for row in fold_rows[:2]):
            return {
                **prepare,
                "decision": "STOP_DR1_AOA10_AOA0_NONPOSITIVE",
                "folds": fold_rows,
                "elapsed_seconds": time.perf_counter() - started,
            }
    merged = merge_fold_results(fold_rows)
    return {
        **prepare,
        **merged,
        "decision": merged["gate"]["decision"],
        "folds": fold_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "release_checkpoint_written": False,
        "submission_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument(
        "--clean-manifest-sha256", default=FORMAL_MANIFEST_SHA256)
    parser.add_argument("--clean-prefix-stats", type=Path)
    parser.add_argument("--control-sim", type=Path)
    parser.add_argument("--candidate-sim", type=Path)
    parser.add_argument("--control-sim-sha256")
    parser.add_argument("--candidate-sim-sha256")
    parser.add_argument("--control-sim-audit", type=Path)
    parser.add_argument("--candidate-sim-audit", type=Path)
    parser.add_argument("--sim-pair-run-manifest", type=Path)
    parser.add_argument("--pair-attestation", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold-worker", action="store_true")
    parser.add_argument("--held-aoa", type=int)
    parser.add_argument(
        "--fold-wall-budget-seconds", type=float, default=60.0 * 60.0)
    parser.add_argument(
        "--total-wall-budget-seconds", type=float, default=5.0 * 60.0 * 60.0)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.clean_manifest_sha256 = normalized_sha256(
        args.clean_manifest_sha256, "clean manifest SHA-256")
    if args.fold_wall_budget_seconds <= 0 or args.total_wall_budget_seconds <= 0:
        raise ValueError("wall budgets must be positive")
    if args.contract_only:
        return
    common = (args.clean_manifest, args.real_dir)
    if any(value is None for value in common):
        raise ValueError("clean manifest and Real directory are required")
    if args.fold_worker:
        if args.prepare_only:
            raise ValueError("fold worker cannot also be prepare-only")
        if args.held_aoa not in AOA_VALUES or args.pair_attestation is None:
            raise ValueError("fold worker requires held AoA and pair attestation")
    else:
        required = (
            args.clean_prefix_stats, args.control_sim, args.candidate_sim,
            args.control_sim_sha256, args.candidate_sim_sha256,
            args.control_sim_audit, args.candidate_sim_audit,
            args.sim_pair_run_manifest)
        if any(value is None for value in required):
            raise ValueError("orchestrator is missing frozen Sim/clean assets")
        args.control_sim_sha256 = normalized_sha256(
            args.control_sim_sha256, "control Sim SHA-256")
        args.candidate_sim_sha256 = normalized_sha256(
            args.candidate_sim_sha256, "candidate Sim SHA-256")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        print(json.dumps({
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "contract": CONTRACT,
            "decision": "CONTRACT_ONLY_NO_DATA_NO_GPU",
        }, ensure_ascii=False, sort_keys=True), flush=True)
        return
    try:
        if args.fold_worker:
            result = fold_worker(args)
            summary = {
                "schema": SCHEMA,
                "decision": "DR1_FOLD_COMPLETE",
                "result": result,
            }
        else:
            summary = orchestrate(args)
        write_json_atomic(args.output_dir / "result.json", summary)
        print(json.dumps({
            "event": "c208_dr1_clean81_loao_complete",
            "decision": summary["decision"],
        }), flush=True)
    except Exception as exc:
        stop = {
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "decision": "STOP_DR1_INFRASTRUCTURE_OR_CONTRACT",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "release_checkpoint_written": False,
            "submission_authorized": False,
        }
        if args.output_dir.exists():
            write_json_atomic(args.output_dir / "result.json", stop)
        print(json.dumps(stop, ensure_ascii=False, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
