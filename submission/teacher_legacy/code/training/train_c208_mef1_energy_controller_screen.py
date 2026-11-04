#!/usr/bin/env python3
"""Necessary-condition screen for the C208 MEF1 energy controller.

Only the 7,105-parameter controller is optimized.  The three controller
endpoints are sealed before internal targets open; held-AoA5 prefixes remain
closed unless the frozen internal point-only gate passes.  This runner never
calls ``forward_distribution`` and never computes SPS, NonTime, or Final.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fp32_ema_shadow as ema_lib  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_phase_residual_probe as point_probe  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_ensemble_unet2d as ensemble_train  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


EXPERIMENT = "c208_mef1_physical_energy_controller_screen"
SCHEMA = "c208_mef1_energy_controller_screen_v1"
ENDPOINT_SCHEMA = "c208_mef1_controller_endpoints_v1"
PRETRAIN_SCHEMA = "c208_mef1_pre_training_contract_v1"
INTERNAL_EVIDENCE_SCHEMA = "c208_mef1_internal_evidence_v1"
FORMAL_BRANCH = "codex/c208-mef1-p0"
FORMAL_ACKNOWLEDGEMENT = "MEF1_P0_POINT_ONLY_HELD5"
FORMAL_START_SHA256 = (
    "6674fcc6a1d777e6407b99bff1a14ee0e106c706a4b1cb67a9108e850ae809e3")
FORMAL_MANIFEST_SHA256 = dr1.FORMAL_MANIFEST_SHA256
HELD_AOA = 5
INTERNAL_FOLD = 2
FOLDS = 5
UPDATES = 1000
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
SEED = 4
LR = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SCHEDULER_T_MAX = 5000
EMA_DECAY = 0.999
LEGACY_PARAMETERS = 175_387_412
CONTROLLER_PARAMETERS = 7_105
TOTAL_PARAMETERS = 175_394_517
WALL_BUDGET_SECONDS = 45.0 * 60.0
GPU_BUDGET_SECONDS = 0.5 * 60.0 * 60.0

ASSIGNMENT_DIGESTS = {
    "all81": "cc60c1ee246ae479b154252cd90b2b954d6ceaee32c4f5188ae85b9fca420a52",
    "fit67": "204b42afd687413932130c77ad01ba77f9864efad5637c1dccef6081a5884b0b",
    "train54": "f7c55b94ce747e856a3a2e2ad6384c68030fdf750b73918435b1c414ea11a1fb",
    "internal13": "6f3d1183ce7395b8f5a83c8f462b947f26a57d5e75c30d1aa1da47c84f5b3866",
}
WINDOW_DIGESTS = {
    "train1695": "762da2beb93fb9e855af24df1f861f8504a9f20603cbca678f3f7c5de2a66cda",
    "internal416": "8b6afb18ad3256ca9f47e672a0142a0312bc20eb207976bcec74c60b4d3b2dec",
    "held433": "2fa48c7ac38211a8e0c8e627c8e43c9a08c2dd6d5d6ec589668ddda10c5c52cd",
}
STREAM_SHA256 = (
    "d1a1fdb4a18b72e93c3a323e020b31f361c51bda3ee606f96f978b644c86471c")
ENDPOINT_ORDER = ("identity", "raw", "ema")
POINT_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")


CONTRACT = {
    "evidence_class": (
        "controller-label held5-unopened with a frozen P1 prior; not "
        "full-lifecycle LOAO"),
    "start_sim_sha256": FORMAL_START_SHA256,
    "controller": {
        "trainable_parameters": CONTROLLER_PARAMETERS,
        "legacy_parameters": LEGACY_PARAMETERS,
        "hidden_channels": 32,
        "delta_logk_limit": 1.0,
        "target_free": True,
    },
    "training": {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "updates": UPDATES,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": SCHEDULER_T_MAX,
        "scheduler_eta_min": 0.0,
        "ema_decay": EMA_DECAY,
        "update_order": ema_lib.UPDATE_ORDER_CONTRACT,
        "accuracy_weights": [5.0, 3.0, 2.0],
        "dtype": "float32",
        "amp": False,
        "compile": False,
        "losses": ["accuracy_objective_532"],
    },
    "data": {
        "manifest_sha256": FORMAL_MANIFEST_SHA256,
        "split_formula": "(rank_full81(re)+2*rank_full81(aoa)) % 5",
        "assignment_before_held_filter": True,
        "internal_fold": INTERNAL_FOLD,
        "optimizer": {"trajectories": 54, "windows": 1695},
        "internal": {"trajectories": 13, "windows": 416},
        "held5": {"trajectories": 14, "windows": 433},
        "normalizer": "optimizer54/1695 only",
        "excluded": dr1.EXCLUDED_BASENAME,
        "tail_arrays_read": 0,
    },
    "evaluation": {
        "point_only": True,
        "endpoint_order": list(ENDPOINT_ORDER),
        "internal_complete_passes": 3,
        "held_complete_passes_after_internal_go": 3,
        "sps": "not_computed/undefined",
        "non_time": "not_computed/undefined",
        "final": "not_computed/undefined",
    },
    "gate": {
        "tke_delta_min": 0.200,
        "rel_delta_min": -0.010,
        "point_532_delta_min": 0.050,
        "direct_delta_min": 0.010,
        "mvpe_abs_delta_max": 0.0005,
        "positive_internal_aoas_min": 3,
        "worst_internal_aoa_direct_min": -0.010,
    },
    "budgets": {
        "gpu_seconds": GPU_BUDGET_SECONDS,
        "wall_seconds": WALL_BUDGET_SECONDS,
        "automatic_retries": 0,
        "maximum_recorded_infra_repair_attempt": 1,
    },
    "release_checkpoint_written": False,
    "submission_authorized": False,
}


def assignment_records(rows: Sequence[dict]) -> list[dict]:
    """Assign folds from full-manifest metadata without reading targets."""
    re_values = sorted({int(row["re"]) for row in rows})
    aoa_values = sorted({int(row["aoa"]) for row in rows})
    re_rank = {value: index for index, value in enumerate(re_values)}
    aoa_rank = {value: index for index, value in enumerate(aoa_values)}
    return [
        {
            "file": str(row["file"]),
            "re": int(row["re"]),
            "aoa": int(row["aoa"]),
            "fold": (
                re_rank[int(row["re"])] + 2 * aoa_rank[int(row["aoa"])])
            % FOLDS,
            "windows": len(row["train_starts"]),
        }
        for row in rows
    ]


def ordered_window_records(rows: Sequence[dict]) -> list[dict]:
    return [
        {"file": str(row["file"]), "start": int(start)}
        for row in rows
        for start in row["train_starts"]
    ]


def freeze_partitions(
        rows: Sequence[dict], *, verify_frozen: bool = True) -> dict:
    """Freeze full-81 ranks first, then remove held AoA5."""
    assignments = assignment_records(rows)
    by_file = {str(row["file"]): row for row in rows}
    fold_by_file = {
        str(record["file"]): int(record["fold"])
        for record in assignments
    }
    fit_rows = [row for row in rows if int(row["aoa"]) != HELD_AOA]
    train_rows = [
        row for row in fit_rows
        if fold_by_file[str(row["file"])] != INTERNAL_FOLD]
    internal_rows = [
        row for row in fit_rows
        if fold_by_file[str(row["file"])] == INTERNAL_FOLD]
    held_rows = [row for row in rows if int(row["aoa"]) == HELD_AOA]
    assignment_subsets = {
        "all81": assignments,
        "fit67": [
            record for record in assignments if int(record["aoa"]) != HELD_AOA],
        "train54": [
            record for record in assignments
            if int(record["aoa"]) != HELD_AOA
            and int(record["fold"]) != INTERNAL_FOLD],
        "internal13": [
            record for record in assignments
            if int(record["aoa"]) != HELD_AOA
            and int(record["fold"]) == INTERNAL_FOLD],
    }
    windows = {
        "train1695": ordered_window_records(train_rows),
        "internal416": ordered_window_records(internal_rows),
        "held433": ordered_window_records(held_rows),
    }
    report = {
        "formula": "(rank_full81(re)+2*rank_full81(aoa)) % 5",
        "full_manifest_ranked_before_held_filter": True,
        "assignments": assignments,
        "assignment_sha256": {
            key: dr1.canonical_sha256(value)
            for key, value in assignment_subsets.items()
        },
        "window_records": windows,
        "window_sha256": {
            key: dr1.canonical_sha256(value) for key, value in windows.items()
        },
        "counts": {
            "train_trajectories": len(train_rows),
            "train_windows": len(windows["train1695"]),
            "internal_trajectories": len(internal_rows),
            "internal_windows": len(windows["internal416"]),
            "held_trajectories": len(held_rows),
            "held_windows": len(windows["held433"]),
        },
        "internal_by_aoa": {
            str(aoa): {
                "trajectories": sum(
                    int(row["aoa"]) == aoa for row in internal_rows),
                "windows": sum(
                    len(row["train_starts"]) for row in internal_rows
                    if int(row["aoa"]) == aoa),
            }
            for aoa in (0, 10, 15, 20)
        },
    }
    if set(by_file) != {str(record["file"]) for record in assignments}:
        raise RuntimeError("fold assignment did not cover each manifest row")
    if verify_frozen:
        expected_counts = {
            "train_trajectories": 54,
            "train_windows": 1695,
            "internal_trajectories": 13,
            "internal_windows": 416,
            "held_trajectories": 14,
            "held_windows": 433,
        }
        expected_by_aoa = {
            "0": {"trajectories": 4, "windows": 128},
            "10": {"trajectories": 3, "windows": 96},
            "15": {"trajectories": 4, "windows": 128},
            "20": {"trajectories": 2, "windows": 64},
        }
        checks = {
            "assignment_digests": (
                report["assignment_sha256"] == ASSIGNMENT_DIGESTS),
            "window_digests": report["window_sha256"] == WINDOW_DIGESTS,
            "counts": report["counts"] == expected_counts,
            "internal_by_aoa": report["internal_by_aoa"] == expected_by_aoa,
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"MEF1 frozen split contract failed: {failed}")
    return {
        "train_rows": train_rows,
        "internal_rows": internal_rows,
        "held_rows": held_rows,
        "report": report,
    }


def build_batch_schedule(
        all_rows: Sequence[dict], train_rows: Sequence[dict], *,
        updates: int = UPDATES, batch_size: int = BATCH_SIZE,
        seed: int = SEED, verify_frozen: bool = True) -> dict:
    """Build local batches and freeze their global clean2544 identities."""
    if updates <= 0 or batch_size <= 0:
        raise ValueError("schedule updates and batch size must be positive")
    global_trajectory = {
        str(row["file"]): index for index, row in enumerate(all_rows)}
    global_sample = {
        (str(record["file"]), int(record["start"])): index
        for index, record in enumerate(ordered_window_records(all_rows))
    }
    local_windows = ordered_window_records(train_rows)
    local_trajectory = []
    for local_row, row in enumerate(train_rows):
        local_trajectory.extend([local_row] * len(row["train_starts"]))
    generator = torch.Generator().manual_seed(seed)
    batches: list[list[int]] = []
    while len(batches) < updates:
        order = torch.randperm(
            len(local_windows), generator=generator).tolist()
        batches.extend([
            order[start:start + batch_size]
            for start in range(0, len(order), batch_size)
        ])
    batches = batches[:updates]
    rows = []
    for batch in batches:
        rows.append({
            "sample_indices": [
                global_sample[(
                    str(local_windows[index]["file"]),
                    int(local_windows[index]["start"]))]
                for index in batch
            ],
            "trajectory_ids": [
                global_trajectory[str(
                    train_rows[local_trajectory[index]]["file"])]
                for index in batch
            ],
        })
    digest = dr1.canonical_sha256(rows)
    if verify_frozen and digest != STREAM_SHA256:
        raise RuntimeError("MEF1 u1000 global-ID stream digest changed")
    return {
        "batches": batches,
        "rows": rows,
        "local_sample_to_global": [
            global_sample[(str(row["file"]), int(row["start"]))]
            for row in local_windows
        ],
        "local_trajectory_to_global": [
            global_trajectory[str(row["file"])] for row in train_rows
        ],
        "sha256": digest,
        "updates": updates,
        "batch_size": batch_size,
        "seed": seed,
    }


class MEF1AccessLedger:
    """Guard train/internal/held H5 access and exact endpoint-pass order."""

    def __init__(
            self, train_files: Sequence[str], internal_files: Sequence[str],
            held_files: Sequence[str], journal_path: Path | None = None):
        self.allowed = {
            "train": set(train_files),
            "internal": set(internal_files),
            "held": set(held_files),
        }
        if any(self.allowed[left] & self.allowed[right]
               for left, right in (("train", "internal"),
                                   ("train", "held"),
                                   ("internal", "held"))):
            raise ValueError("MEF1 prefix partitions must be file-disjoint")
        self.journal_path = journal_path
        self.opened: dict[str, str] = {}
        self.max_frame_exclusive: dict[str, int] = {}
        self.seals: OrderedDict[str, dict] = OrderedDict()
        self.evaluations = {"internal": [], "held": []}
        self.internal_gate_pass: bool | None = None
        self.internal_result_seal: dict | None = None
        self.events: list[dict] = []
        self._flush()

    @property
    def all_endpoints_sealed(self) -> bool:
        return tuple(self.seals) == ENDPOINT_ORDER

    @property
    def held_authorized(self) -> bool:
        if self.internal_gate_pass is not True or self.internal_result_seal is None:
            return False
        path = Path(str(self.internal_result_seal["path"]))
        try:
            return path.is_file() and dr1.sha256(path) == (
                self.internal_result_seal["file_sha256"])
        except OSError:
            return False

    def _flush(self) -> None:
        if self.journal_path is not None:
            dr1.write_json_atomic(self.journal_path, self.report())

    def before_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        if filename == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded 7575_0 reached the H5 opener")
        if phase not in self.allowed or filename not in self.allowed[phase]:
            raise RuntimeError(f"{filename} is not allowed in {phase} phase")
        if filename in self.opened:
            raise RuntimeError(f"H5 file {filename} was opened more than once")
        if phase == "train" and self.seals:
            raise RuntimeError("optimizer prefix opened after endpoint sealing")
        if phase == "internal":
            if not self.all_endpoints_sealed:
                raise RuntimeError("internal prefix opened before three seals")
            if self.evaluations["internal"]:
                raise RuntimeError("internal prefix opened after scoring began")
        if phase == "held":
            if not self.held_authorized:
                raise RuntimeError("held5 prefix opened before internal GO")
            if self.evaluations["internal"] != list(ENDPOINT_ORDER):
                raise RuntimeError("held5 opened before three internal passes")
            if self.evaluations["held"]:
                raise RuntimeError("held5 prefix opened after scoring began")

    def record_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        if phase == "held" and not self.held_authorized:
            raise RuntimeError("held5 evidence seal changed during H5 access")
        opened_ns = time.monotonic_ns()
        sealed_before_open = None
        if phase == "held":
            sealed_before_open = bool(
                self.internal_result_seal is not None
                and int(self.internal_result_seal["sealed_monotonic_ns"])
                <= opened_ns
                and int(self.internal_result_seal["event_index"])
                < len(self.events))
            if not sealed_before_open:
                raise RuntimeError("held5 opened before internal result seal")
        self.opened[filename] = phase
        self.max_frame_exclusive[filename] = int(row["val_boundary"])
        self.events.append({
            "event": "prefix_open",
            "phase": phase,
            "file": filename,
            "aoa": int(row["aoa"]),
            "slice": "u/v[:val_boundary,::2,::2]",
            "max_read_frame_exclusive": int(row["val_boundary"]),
            "internal_result_sealed_before_open": sealed_before_open,
            "monotonic_ns": opened_ns,
        })
        self._flush()

    def seal_endpoint(
            self, endpoint: str, bundle_path: Path, file_sha256: str,
            controller_state_sha256: str) -> None:
        expected = ENDPOINT_ORDER[len(self.seals)] if len(self.seals) < 3 else None
        if endpoint != expected:
            raise RuntimeError(
                f"expected {expected} endpoint seal, got {endpoint}")
        if set(self.opened) != self.allowed["train"]:
            raise RuntimeError("optimizer prefixes were not opened exactly")
        if self.evaluations["internal"] or self.evaluations["held"]:
            raise RuntimeError("endpoint sealed after evaluation began")
        self.seals[endpoint] = {
            "bundle_path": str(bundle_path.resolve()),
            "bundle_file_sha256": dr1.normalized_sha256(
                file_sha256, f"{endpoint} bundle file"),
            "controller_state_sha256": dr1.normalized_sha256(
                controller_state_sha256, f"{endpoint} controller state"),
            "monotonic_ns": time.monotonic_ns(),
        }
        self.events.append({"event": "endpoint_seal", "endpoint": endpoint})
        self._flush()

    def record_evaluation(self, phase: str, endpoint: str) -> None:
        if phase not in self.evaluations:
            raise ValueError(f"unknown evaluation phase {phase}")
        if len(self.evaluations[phase]) >= len(ENDPOINT_ORDER):
            raise RuntimeError(f"{phase} already has three complete point passes")
        expected = ENDPOINT_ORDER[len(self.evaluations[phase])]
        if endpoint != expected:
            raise RuntimeError(
                f"expected {phase} {expected} pass, got {endpoint}")
        if not self.all_endpoints_sealed:
            raise RuntimeError("evaluation began before three endpoint seals")
        if set(name for name, value in self.opened.items() if value == phase) != (
                self.allowed[phase]):
            raise RuntimeError(f"{phase} prefixes were not opened exactly")
        if phase == "held" and not self.held_authorized:
            raise RuntimeError("held5 evaluation began before internal GO")
        self.evaluations[phase].append(endpoint)
        self.events.append({
            "event": "point_evaluation",
            "phase": phase,
            "endpoint": endpoint,
            "complete_pass_index": len(self.evaluations[phase]),
            "monotonic_ns": time.monotonic_ns(),
        })
        self._flush()

    def record_internal_gate(self, passed: bool) -> None:
        if self.evaluations["internal"] != list(ENDPOINT_ORDER):
            raise RuntimeError("internal gate recorded before three passes")
        if self.internal_gate_pass is not None:
            raise RuntimeError("internal gate was recorded twice")
        self.internal_gate_pass = bool(passed)
        self.events.append({
            "event": "internal_gate",
            "pass": self.internal_gate_pass,
            "held_authorized": self.held_authorized,
            "monotonic_ns": time.monotonic_ns(),
        })
        self._flush()

    def seal_internal_result(
            self, path: Path, file_sha256: str, gate_pass: bool) -> None:
        if self.evaluations["internal"] != list(ENDPOINT_ORDER):
            raise RuntimeError("internal result sealed before three passes")
        if self.internal_result_seal is not None:
            raise RuntimeError("internal result was sealed twice")
        if self.evaluations["held"] or any(
                phase == "held" for phase in self.opened.values()):
            raise RuntimeError("internal result sealed after held5 access")
        gate_pass = bool(gate_pass)
        if (self.internal_gate_pass is not None
                and self.internal_gate_pass != gate_pass):
            raise RuntimeError("sealed internal gate differs from recorded gate")
        expected = dr1.normalized_sha256(
            file_sha256, "internal evidence file")
        resolved = path.resolve()
        if not resolved.is_file() or dr1.sha256(resolved) != expected:
            raise RuntimeError("internal evidence file failed SHA seal")
        self.internal_gate_pass = gate_pass
        sealed_ns = time.monotonic_ns()
        self.internal_result_seal = {
            "path": str(resolved),
            "file_sha256": expected,
            "gate_pass": gate_pass,
            "sealed_monotonic_ns": sealed_ns,
            "event_index": len(self.events),
        }
        self.events.append({
            "event": "internal_result_seal",
            "gate_pass": gate_pass,
            "file_sha256": expected,
            "sealed_monotonic_ns": sealed_ns,
        })
        self._flush()

    def report(self) -> dict:
        opened_by_phase = {
            phase: sorted(
                name for name, opened_phase in self.opened.items()
                if opened_phase == phase)
            for phase in self.allowed
        }
        held_events = [
            event for event in self.events
            if event.get("event") == "prefix_open"
            and event.get("phase") == "held"
        ]
        return {
            "schema": "c208_mef1_access_ledger_v1",
            "allowed_file_allowlists": {
                key: sorted(value) for key, value in self.allowed.items()},
            "opened_file_allowlists": opened_by_phase,
            "opened_exact": {
                phase: set(opened_by_phase[phase]) == self.allowed[phase]
                for phase in self.allowed
            },
            "max_read_frame_exclusive_by_file": dict(sorted(
                self.max_frame_exclusive.items())),
            "endpoint_seals": dict(self.seals),
            "all_endpoints_sealed": self.all_endpoints_sealed,
            "evaluation_order": self.evaluations,
            "internal_gate_pass": self.internal_gate_pass,
            "internal_result_seal": self.internal_result_seal,
            "held_authorized": self.held_authorized,
            "internal_result_sealed_before_held": all(
                event["internal_result_sealed_before_open"] is True
                for event in held_events),
            "held5_open_count": len(opened_by_phase["held"]),
            "tail_arrays_read": 0,
            "excluded_7575_absent": dr1.EXCLUDED_BASENAME not in self.opened,
            "events": self.events,
        }


def point_metrics_from_state(state: Mapping[str, float]) -> dict[str, float]:
    samples = int(state["samples"])
    if samples <= 0:
        raise ValueError("point error state has no samples")
    official = point_probe.full.OFFICIAL
    result = {
        "rel_l2_score": official.score_error(
            float(state["rel_error_sum"]) / samples),
        "tke_rel_l2_score": official.score_error(
            float(state["tke_error_sum"]) / samples),
        "mvpe_rel_l2_score": official.score_error(
            float(state["mvpe_error_sum"]) / samples),
    }
    result["point_532_score"] = (
        0.5 * result["rel_l2_score"]
        + 0.3 * result["tke_rel_l2_score"]
        + 0.2 * result["mvpe_rel_l2_score"])
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("point-only official scores are non-finite")
    return result


def compare_point_metrics(
        candidate: Mapping[str, float], identity: Mapping[str, float]) -> dict:
    delta = {
        key: float(candidate[key]) - float(identity[key])
        for key in (*POINT_KEYS, "point_532_score")
    }
    delta["direct_score"] = (
        0.47201126 * delta["rel_l2_score"]
        + 0.08586562 * delta["tke_rel_l2_score"]
        + 0.09410214 * delta["mvpe_rel_l2_score"])
    if not all(math.isfinite(value) for value in delta.values()):
        raise FloatingPointError("point-only score deltas are non-finite")
    return delta


def raw_diagnostic(
        raw: Mapping[str, float], identity: Mapping[str, float]) -> dict:
    delta = compare_point_metrics(raw, identity)
    conditions = {
        "finite": all(math.isfinite(float(value)) for value in raw.values()),
        "mvpe_invariant": abs(delta["mvpe_rel_l2_score"]) <= 0.0005,
    }
    return {"pass": all(conditions.values()), "conditions": conditions,
            "delta": delta, "selection_role": "diagnostic_only"}


def primary_gate(
        candidate: Mapping[str, float], identity: Mapping[str, float], *,
        per_aoa_candidate: Mapping[str, Mapping[str, float]] | None = None,
        per_aoa_identity: Mapping[str, Mapping[str, float]] | None = None,
        require_per_aoa: bool) -> dict:
    delta = compare_point_metrics(candidate, identity)
    conditions = {
        "tke": delta["tke_rel_l2_score"] >= 0.200,
        "rel": delta["rel_l2_score"] >= -0.010,
        "point_532": delta["point_532_score"] >= 0.050,
        "direct": delta["direct_score"] >= 0.010,
        "mvpe": abs(delta["mvpe_rel_l2_score"]) <= 0.0005,
    }
    per_aoa_delta: dict[str, dict] = {}
    if require_per_aoa:
        if per_aoa_candidate is None or per_aoa_identity is None:
            raise ValueError("internal gate requires per-AoA point metrics")
        if set(per_aoa_candidate) != {"0", "10", "15", "20"} or (
                set(per_aoa_identity) != set(per_aoa_candidate)):
            raise ValueError("internal per-AoA metrics are incomplete")
        per_aoa_delta = {
            key: compare_point_metrics(
                per_aoa_candidate[key], per_aoa_identity[key])
            for key in sorted(per_aoa_candidate, key=int)
        }
        direct_values = [row["direct_score"] for row in per_aoa_delta.values()]
        conditions["positive_aoas"] = sum(
            value > 0.0 for value in direct_values) >= 3
        conditions["worst_aoa"] = min(direct_values) >= -0.010
    passed = all(conditions.values())
    return {
        "pass": passed,
        "conditions": conditions,
        "delta": delta,
        "per_aoa_delta": per_aoa_delta,
    }


def _clone_tensor_state(
        state: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().cpu().clone()) for name, value in state.items())


def controller_state(model: EnsembleUNet2d) -> OrderedDict[str, torch.Tensor]:
    if model.energy_controller is None:
        raise RuntimeError("MEF1 model has no energy controller")
    return _clone_tensor_state(model.energy_controller.state_dict())


def controller_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return ema_lib.tensor_state_sha256(state)


def legacy_state_sha256(model: EnsembleUNet2d) -> str:
    state = OrderedDict(
        (name, value) for name, value in model.state_dict().items()
        if not name.startswith("energy_controller."))
    return ema_lib.tensor_state_sha256(state)


def normalizer_payload(normalizer: smu1.FoldNormalizer) -> dict:
    values = {
        key: getattr(normalizer, key).detach().cpu().tolist()
        for key in ("mean_input", "mean_target", "std_input", "std_target")
    }
    return {"values": values, "sha256": dr1.canonical_sha256(values)}


def inspect_start_checkpoint(
        path: Path, expected_sha256: str,
        normalizer: smu1.FoldNormalizer, device: torch.device,
        ) -> tuple[EnsembleUNet2d, dict]:
    expected = dr1.normalized_sha256(expected_sha256, "start Sim checkpoint")
    actual = dr1.sha256(path)
    if actual != expected or actual != FORMAL_START_SHA256:
        raise RuntimeError("MEF1 start checkpoint SHA-256 changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (not isinstance(payload, dict) or payload.get("phase") != "sim"
            or int(payload.get("iteration", -1)) != 10_000):
        raise ValueError("MEF1 start must be the frozen Sim10k checkpoint")
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, Mapping):
        raise ValueError("MEF1 start lacks model config/state")
    stored = dict(config)
    if stored.pop("architecture", None) != "ensemble_unet":
        raise ValueError("MEF1 start is not an ensemble U-Net")
    stored.update({
        "physical_energy_controller": True,
        "physical_energy_hidden_channels": 32,
        "physical_energy_log_limit": 1.0,
    })
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored)).float()
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {
        f"energy_controller.{name}"
        for name in model.energy_controller.state_dict()
    }
    if (set(incompatible.missing_keys) != expected_missing
            or incompatible.unexpected_keys):
        raise RuntimeError(
            "legacy start did not load with controller-only missing keys")
    parameter_count = sum(value.numel() for value in model.parameters())
    controller_count = sum(
        value.numel() for value in model.energy_controller.parameters())
    if (parameter_count != TOTAL_PARAMETERS
            or controller_count != CONTROLLER_PARAMETERS
            or parameter_count - controller_count != LEGACY_PARAMETERS):
        raise RuntimeError("MEF1 model/controller parameter count changed")
    if any(value.dtype != torch.float32 for value in model.parameters()):
        raise TypeError("formal MEF1 model parameters must be FP32")
    ensemble_train.common.set_model_stats(model, normalizer)
    identity = controller_state(model)
    if any(bool(torch.count_nonzero(value)) for name, value in identity.items()
           if name in {"body.5.weight", "body.5.bias"}):
        raise RuntimeError("MEF1 output projection is not zero at identity")
    loaded_legacy_sha = legacy_state_sha256(model)
    report = {
        "path": str(path.resolve()),
        "checkpoint_sha256": actual,
        "checkpoint_model_state_sha256": dr1.model_state_sha256(state),
        "checkpoint_model_config_sha256": dr1.canonical_sha256(config),
        "phase": "sim",
        "iteration": 10_000,
        "missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "legacy_parameters": LEGACY_PARAMETERS,
        "controller_parameters": CONTROLLER_PARAMETERS,
        "total_parameters": TOTAL_PARAMETERS,
        "loaded_legacy_state_sha256": loaded_legacy_sha,
        "identity_controller_state_sha256": controller_state_sha256(identity),
    }
    return model.to(device), report


def _check_deadlines(
        wall_deadline: float, gpu_deadline: float, label: str) -> None:
    now = time.perf_counter()
    if now > wall_deadline:
        raise TimeoutError(f"MEF1 wall budget expired during {label}")
    if now > gpu_deadline:
        raise TimeoutError(f"MEF1 GPU budget expired during {label}")


def freeze_controller_only(model: EnsembleUNet2d) -> tuple[torch.nn.Parameter, ...]:
    if model.energy_controller is None:
        raise RuntimeError("cannot freeze a model without MEF1 controller")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("energy_controller."))
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad)
    if sum(value.numel() for value in trainable) != CONTROLLER_PARAMETERS:
        raise RuntimeError("MEF1 trainable parameter set is not controller-only")
    return trainable


def train_controller(
        model: EnsembleUNet2d, dataset, schedule: Mapping[str, object],
        normalizer: smu1.FoldNormalizer, device: torch.device,
        wall_deadline: float, gpu_deadline: float, *,
        initial_legacy_state_sha256: str | None = None,
        ) -> tuple[dict, ema_lib.FP32EMAShadow]:
    trainable = freeze_controller_only(model)
    tagged = ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(
        tagged, batch_sampler=schedule["batches"], num_workers=0,
        pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(
        trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX, eta_min=0.0)
    ema = ema_lib.FP32EMAShadow(model.energy_controller, EMA_DECAY)
    initial_legacy_sha = (
        legacy_state_sha256(model)
        if initial_legacy_state_sha256 is None
        else dr1.normalized_sha256(
            initial_legacy_state_sha256, "initial legacy model state"))
    actual_rows = []
    history = []
    started = time.perf_counter()
    expected_updates = int(schedule["updates"])
    local_sample_to_global = list(schedule["local_sample_to_global"])
    local_trajectory_to_global = list(schedule["local_trajectory_to_global"])
    for iteration, batch in enumerate(loader, start=1):
        if iteration > expected_updates:
            break
        _check_deadlines(wall_deadline, gpu_deadline, "controller training")
        history_value, target, trajectory_ids, sample_indices = batch
        actual = {
            "sample_indices": [
                int(local_sample_to_global[index])
                for index in sample_indices.tolist()
            ],
            "trajectory_ids": [
                int(local_trajectory_to_global[index])
                for index in trajectory_ids.tolist()
            ],
        }
        expected = schedule["rows"][iteration - 1]
        if actual != expected:
            raise RuntimeError(
                f"MEF1 global-ID stream mismatch at update {iteration}")
        actual_rows.append(actual)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        history_value = history_value.to(
            device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(
            device=device, dtype=torch.float32, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history_value, target)
        prediction = model(history_norm)
        loss, details = ensemble_train.common.accuracy_objective(
            prediction, target_norm, normalizer, (5.0, 3.0, 2.0))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("MEF1 point objective is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, GRAD_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("MEF1 controller gradient is non-finite")
        optimizer.step()
        ema.update_after_optimizer_step(
            model.energy_controller, optimizer_step=iteration)
        scheduler.step()
        if iteration == 1 or iteration % 100 == 0 or (
                iteration == expected_updates):
            row = {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "rel_l2": float(details["rel_l2"]),
                "tke_rel_l2": float(details["tke_rel_l2"]),
                "mvpe_rel_l2": float(details["mvpe_rel_l2"]),
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps({"event": "mef1_progress", **row}), flush=True)
    if len(actual_rows) != expected_updates or ema.updates != expected_updates:
        raise RuntimeError("MEF1 controller did not complete the fixed updates")
    actual_stream_sha = dr1.canonical_sha256(actual_rows)
    if actual_stream_sha != schedule["sha256"]:
        raise RuntimeError("MEF1 observed global-ID stream digest changed")
    if any(parameter.grad is not None for name, parameter in model.named_parameters()
           if not name.startswith("energy_controller.")):
        raise RuntimeError("a frozen legacy parameter received a gradient")
    final_legacy_sha = legacy_state_sha256(model)
    if final_legacy_sha != initial_legacy_sha:
        raise RuntimeError("MEF1 training mutated frozen legacy state")
    return ({
        "updates": expected_updates,
        "batch_size": int(schedule["batch_size"]),
        "seed": int(schedule["seed"]),
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "scheduler_t_max": SCHEDULER_T_MAX,
        "scheduler_eta_min": 0.0,
        "ema_decay": EMA_DECAY,
        "ema_updates": ema.updates,
        "actual_stream_sha256": actual_stream_sha,
        "initial_legacy_state_sha256": initial_legacy_sha,
        "final_legacy_state_sha256": final_legacy_sha,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }, ema)


def save_endpoint_bundle(
        path: Path, *, identity_state: Mapping[str, torch.Tensor],
        raw_state: Mapping[str, torch.Tensor],
        ema_state: Mapping[str, torch.Tensor], ema_metadata: Mapping[str, object],
        start: Mapping[str, object], normalizer: Mapping[str, object],
        stream_sha256: str, legacy_state_sha256_value: str,
        ) -> dict:
    states = {
        "identity": _clone_tensor_state(identity_state),
        "raw": _clone_tensor_state(raw_state),
        "ema": _clone_tensor_state(ema_state),
    }
    keys = [list(state) for state in states.values()]
    if not keys[0] or not all(value == keys[0] for value in keys[1:]):
        raise RuntimeError("MEF1 endpoint controller schemas differ")
    if any(sum(tensor.numel() for tensor in state.values())
           != CONTROLLER_PARAMETERS for state in states.values()):
        raise RuntimeError("MEF1 endpoint contains non-controller tensors")
    state_sha = {
        endpoint: controller_state_sha256(state)
        for endpoint, state in states.items()
    }
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "controller_states": states,
        "controller_state_sha256": state_sha,
        "controller_state_keys": keys[0],
        "controller_parameters_per_endpoint": CONTROLLER_PARAMETERS,
        "legacy_model_state_saved": False,
        "legacy_state_sha256": legacy_state_sha256_value,
        "start": dict(start),
        "normalizer": dict(normalizer),
        "sample_stream_sha256": stream_sha256,
        "ema": dict(ema_metadata),
        "sps": "not_computed/undefined",
        "non_time": "not_computed/undefined",
        "final": "not_computed/undefined",
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    file_sha = dr1.sha256(path)
    loaded = load_endpoint_bundle(path, file_sha)
    if loaded["controller_state_sha256"] != state_sha:
        raise RuntimeError("MEF1 sealed endpoint hashes changed on reload")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha,
        "controller_state_sha256": state_sha,
        "controller_state_keys": keys[0],
        "controller_parameters_per_endpoint": CONTROLLER_PARAMETERS,
        "legacy_model_state_saved": False,
    }


def load_endpoint_bundle(path: Path, expected_file_sha256: str) -> dict:
    if dr1.sha256(path) != dr1.normalized_sha256(
            expected_file_sha256, "MEF1 endpoint bundle"):
        raise RuntimeError("MEF1 endpoint bundle file SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != ENDPOINT_SCHEMA:
        raise ValueError("invalid MEF1 endpoint bundle schema")
    if payload.get("legacy_model_state_saved") is not False:
        raise RuntimeError("MEF1 bundle unexpectedly contains legacy model state")
    states = payload.get("controller_states")
    expected_hashes = payload.get("controller_state_sha256")
    if (not isinstance(states, dict) or tuple(states) != ENDPOINT_ORDER
            or not isinstance(expected_hashes, dict)):
        raise ValueError("MEF1 endpoint bundle is incomplete or reordered")
    for endpoint in ENDPOINT_ORDER:
        state = states[endpoint]
        if (not isinstance(state, Mapping)
                or controller_state_sha256(state) != expected_hashes[endpoint]
                or sum(value.numel() for value in state.values())
                != CONTROLLER_PARAMETERS):
            raise RuntimeError(f"MEF1 {endpoint} controller state failed seal")
    return payload


def load_controller_endpoint(
        model: EnsembleUNet2d, state: Mapping[str, torch.Tensor],
        expected_sha256: str) -> None:
    if model.energy_controller is None:
        raise RuntimeError("MEF1 model has no controller at endpoint load")
    if controller_state_sha256(state) != expected_sha256:
        raise RuntimeError("MEF1 controller endpoint hash changed before load")
    model.energy_controller.load_state_dict(state, strict=True)
    if controller_state_sha256(controller_state(model)) != expected_sha256:
        raise RuntimeError("strict-loaded MEF1 controller differs from seal")


def _update_raw_tensor_hash(digest, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))


def _empty_point_state() -> dict[str, float]:
    return {
        "samples": 0,
        "rel_error_sum": 0.0,
        "tke_error_sum": 0.0,
        "mvpe_error_sum": 0.0,
    }


def _add_point_state(total: dict[str, float], row: Mapping[str, float]) -> None:
    for key in total:
        total[key] += float(row[key])


@torch.inference_mode()
def evaluate_point_endpoint(
        model, dataset, sample_aoas: Sequence[int],
        normalizer: smu1.FoldNormalizer, device: torch.device,
        wall_deadline: float, gpu_deadline: float,
        ledger: MEF1AccessLedger, *, phase: str, endpoint: str) -> dict:
    """Run one complete point-only pass and derive AoA rows from that pass."""
    if len(dataset) != len(sample_aoas) or not sample_aoas:
        raise ValueError("MEF1 point dataset/AoA labels are empty or misaligned")
    batches = [
        list(range(start, min(start + EVAL_BATCH_SIZE, len(dataset))))
        for start in range(0, len(dataset), EVAL_BATCH_SIZE)
    ]
    loader = DataLoader(
        dataset, batch_sampler=batches, num_workers=0,
        pin_memory=device.type == "cuda")
    pooled = _empty_point_state()
    by_aoa = {
        str(aoa): _empty_point_state()
        for aoa in sorted(set(int(value) for value in sample_aoas))
    }
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    model.eval()
    for indices, (history, target) in zip(batches, loader, strict=True):
        _check_deadlines(wall_deadline, gpu_deadline, f"{phase} {endpoint} pass")
        target_cpu = target.contiguous()
        mask_cpu = target_cpu[..., :2].ne(0.0).to(torch.uint8).contiguous()
        _update_raw_tensor_hash(target_hasher, target_cpu)
        _update_raw_tensor_hash(mask_hasher, mask_cpu)
        history = history.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.float32, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history, target)
        prediction_norm = model(history_norm)
        if not isinstance(prediction_norm, torch.Tensor):
            raise TypeError("MEF1 point forward did not return one tensor")
        prediction = normalizer.postprocess_prediction(prediction_norm)
        physical_target = normalizer.postprocess_target(target_norm)
        prediction_np = np.ascontiguousarray(
            prediction.detach().cpu().numpy(), dtype=np.float32)
        target_np = np.ascontiguousarray(
            physical_target.detach().cpu().numpy(), dtype=np.float32)
        if not np.all(np.isfinite(prediction_np)):
            raise FloatingPointError("MEF1 point prediction is non-finite")
        _add_point_state(
            pooled, point_probe.point_error_state(prediction_np, target_np))
        batch_aoas = np.asarray(
            [int(sample_aoas[index]) for index in indices], dtype=np.int64)
        for aoa in sorted(set(batch_aoas.tolist())):
            selected = batch_aoas == aoa
            _add_point_state(
                by_aoa[str(aoa)], point_probe.point_error_state(
                    prediction_np[selected], target_np[selected]))
    ledger.record_evaluation(phase, endpoint)
    return {
        "endpoint": endpoint,
        "phase": phase,
        "samples": len(dataset),
        "metrics": point_metrics_from_state(pooled),
        "point_error_state": pooled,
        "per_aoa_metrics": {
            key: point_metrics_from_state(value)
            for key, value in by_aoa.items()
        },
        "per_aoa_point_error_state": by_aoa,
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
        "complete_point_passes": 1,
        "forward_api": "model(x)",
        "sps": "not_computed/undefined",
        "non_time": "not_computed/undefined",
        "final": "not_computed/undefined",
    }


def sample_aoas(windows: Sequence, metadata: Sequence[dict]) -> list[int]:
    return [
        int(metadata[int(window.trajectory)]["aoa"])
        for window in windows
    ]


def evaluate_sealed_partition(
        model: EnsembleUNet2d, bundle: Mapping[str, object], dataset,
        aoas: Sequence[int], normalizer: smu1.FoldNormalizer,
        device: torch.device, wall_deadline: float, gpu_deadline: float,
        ledger: MEF1AccessLedger, *, phase: str) -> dict:
    reports = {}
    for endpoint in ENDPOINT_ORDER:
        load_controller_endpoint(
            model, bundle["controller_states"][endpoint],
            bundle["controller_state_sha256"][endpoint])
        reports[endpoint] = evaluate_point_endpoint(
            model, dataset, aoas, normalizer, device,
            wall_deadline, gpu_deadline, ledger,
            phase=phase, endpoint=endpoint)
    target_identities = {
        (
            report["ordered_raw_target_sha256"],
            report["ordered_scored_mask_sha256"],
        )
        for report in reports.values()
    }
    if len(target_identities) != 1:
        raise RuntimeError(f"MEF1 {phase} targets/masks changed across endpoints")
    return {
        "reports": reports,
        "target_identity_exact": True,
        "complete_point_passes": 3,
    }


def _access_contract_ok(
        report: Mapping[str, object], *, held_expected: bool) -> bool:
    exact = report["opened_exact"]
    internal_seal = report["internal_result_seal"]
    return bool(
        exact["train"]
        and exact["internal"]
        and (exact["held"] if held_expected else not report["held5_open_count"])
        and report["all_endpoints_sealed"]
        and report["evaluation_order"]["internal"] == list(ENDPOINT_ORDER)
        and internal_seal is not None
        and bool(internal_seal["gate_pass"]) is held_expected
        and bool(report["held_authorized"]) is held_expected
        and report["internal_result_sealed_before_held"]
        and (report["evaluation_order"]["held"] == list(ENDPOINT_ORDER)
             if held_expected else report["evaluation_order"]["held"] == [])
        and report["tail_arrays_read"] == 0
        and report["excluded_7575_absent"]
    )


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    identity = dr1.validate_formal_git_identity(dr1.git_identity())
    if identity["branch"] != FORMAL_BRANCH:
        raise RuntimeError(
            f"formal MEF1 requires branch {FORMAL_BRANCH}, got {identity['branch']}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, args.clean_manifest_sha256)
    partitions = freeze_partitions(rows)
    train_rows = partitions["train_rows"]
    internal_rows = partitions["internal_rows"]
    held_rows = partitions["held_rows"]
    schedule = build_batch_schedule(rows, train_rows)
    if schedule["sha256"] != STREAM_SHA256:
        raise RuntimeError("formal MEF1 stream does not match preregistration")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal MEF1 screen requires CUDA")
    args.output_dir.mkdir(parents=True)
    dr1.write_json_atomic(args.output_dir / "run_owner.json", {
        "schema": "c208_mef1_run_owner_v1",
        "experiment": EXPERIMENT,
        "git": identity,
        "infra_repair_attempt": args.infra_repair_attempt,
    })
    wall_deadline = started + WALL_BUDGET_SECONDS
    gpu_deadline = started + GPU_BUDGET_SECONDS
    ledger = MEF1AccessLedger(
        [str(row["file"]) for row in train_rows],
        [str(row["file"]) for row in internal_rows],
        [str(row["file"]) for row in held_rows],
        args.output_dir / "access_ledger.json")

    train_trajectories, train_metadata, train_windows = dr1.load_prefix_rows(
        args.real_dir, train_rows, "train", ledger)
    if len(train_windows) != 1695 or len(train_trajectories) != 54:
        raise RuntimeError("MEF1 optimizer partition changed after H5 load")
    train_dataset = ensemble_train.common.repro.WindowDataset(
        train_trajectories, train_windows)
    sufficient = smu1.window_sufficient_statistics(
        train_trajectories, train_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(train_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    normalizer_report = normalizer_payload(normalizer)
    normalizer_report["source_statistics_sha256"] = (
        smu1.statistics_digest(statistics))

    ensemble_train.seed_everything(SEED)
    model, start = inspect_start_checkpoint(
        args.start_sim, args.start_sim_sha256, normalizer, device)
    identity_controller = controller_state(model)
    pre_training = {
        "schema": PRETRAIN_SCHEMA,
        "experiment": EXPERIMENT,
        "contract": CONTRACT,
        "manifest": manifest_report,
        "split": partitions["report"],
        "schedule": {
            "rows": schedule["rows"],
            "sha256": schedule["sha256"],
            "updates": schedule["updates"],
            "batch_size": schedule["batch_size"],
            "seed": schedule["seed"],
            "global_sample_domain": "full clean81 ordered 2544 windows",
            "global_trajectory_domain": "full clean81 ordered 81 rows",
            "bootstrap": False,
        },
        "normalizer": {
            **normalizer_report,
            "source": "optimizer54/1695 only",
            "internal_windows_used": 0,
            "held_windows_used": 0,
        },
        "start": start,
        "access_before_optimizer": ledger.report(),
        "infra_repair_attempt": args.infra_repair_attempt,
        "written_before_first_optimizer_step": True,
    }
    dr1.write_json_atomic(
        args.output_dir / "pre_training_contract.json", pre_training)

    training, ema = train_controller(
        model, train_dataset, schedule, normalizer, device,
        wall_deadline, gpu_deadline,
        initial_legacy_state_sha256=start["loaded_legacy_state_sha256"])
    if training["actual_stream_sha256"] != STREAM_SHA256:
        raise RuntimeError("MEF1 training stream differs from frozen stream")
    raw_controller = controller_state(model)
    raw_sha_before_ema_seal = controller_state_sha256(raw_controller)
    ema_seal_payload = ema.seal(model.energy_controller)
    ema_controller = ema_seal_payload.pop("model_state")
    if controller_state_sha256(ema_controller) != (
            ema_seal_payload["materialized_model_state_sha256"]):
        raise RuntimeError("materialized controller EMA hash changed")
    raw_sha_after_ema_seal = controller_state_sha256(controller_state(model))
    if raw_sha_after_ema_seal != raw_sha_before_ema_seal:
        raise RuntimeError("materializing controller EMA mutated raw endpoint")
    if (int(ema_seal_payload["updates"]) != UPDATES
            or float(ema_seal_payload["decay"]) != EMA_DECAY):
        raise RuntimeError("MEF1 EMA counter/decay changed")

    bundle_path = args.output_dir / "controller_endpoints_u1000.pth"
    bundle_seal = save_endpoint_bundle(
        bundle_path,
        identity_state=identity_controller,
        raw_state=raw_controller,
        ema_state=ema_controller,
        ema_metadata=ema_seal_payload,
        start=start,
        normalizer=normalizer_report,
        stream_sha256=schedule["sha256"],
        legacy_state_sha256_value=training["final_legacy_state_sha256"])
    for endpoint in ENDPOINT_ORDER:
        ledger.seal_endpoint(
            endpoint, bundle_path, bundle_seal["file_sha256"],
            bundle_seal["controller_state_sha256"][endpoint])
    bundle = load_endpoint_bundle(bundle_path, bundle_seal["file_sha256"])
    del ema, ema_controller, raw_controller, train_dataset
    del train_trajectories, train_metadata, train_windows, sufficient
    gc.collect()
    torch.cuda.empty_cache()

    internal_trajectories, internal_metadata, internal_windows = (
        dr1.load_prefix_rows(
            args.real_dir, internal_rows, "internal", ledger))
    if len(internal_windows) != 416 or len(internal_trajectories) != 13:
        raise RuntimeError("MEF1 internal partition changed after H5 load")
    internal_dataset = ensemble_train.common.repro.WindowDataset(
        internal_trajectories, internal_windows)
    internal = evaluate_sealed_partition(
        model, bundle, internal_dataset,
        sample_aoas(internal_windows, internal_metadata), normalizer,
        device, wall_deadline, gpu_deadline, ledger, phase="internal")
    internal_legacy_sha = legacy_state_sha256(model)
    if internal_legacy_sha != training["final_legacy_state_sha256"]:
        raise RuntimeError("MEF1 internal evaluation mutated legacy state")
    internal["legacy_state_sha256_after"] = internal_legacy_sha
    internal_reports = internal["reports"]
    internal_raw = raw_diagnostic(
        internal_reports["raw"]["metrics"],
        internal_reports["identity"]["metrics"])
    internal_primary = primary_gate(
        internal_reports["ema"]["metrics"],
        internal_reports["identity"]["metrics"],
        per_aoa_candidate=internal_reports["ema"]["per_aoa_metrics"],
        per_aoa_identity=internal_reports["identity"]["per_aoa_metrics"],
        require_per_aoa=True)
    internal_pass = bool(internal_raw["pass"] and internal_primary["pass"])
    access_before_internal_seal = ledger.report()
    if access_before_internal_seal["held5_open_count"] != 0:
        raise RuntimeError("held5 opened before internal evidence seal")
    internal_evidence = {
        "schema": INTERNAL_EVIDENCE_SCHEMA,
        "experiment": EXPERIMENT,
        "manifest": manifest_report,
        "split": {
            "counts": partitions["report"]["counts"],
            "internal_by_aoa": partitions["report"]["internal_by_aoa"],
            "assignment_sha256": partitions["report"]["assignment_sha256"],
            "window_sha256": partitions["report"]["window_sha256"],
        },
        "sample_stream_sha256": schedule["sha256"],
        "normalizer": normalizer_report,
        "endpoint_bundle": bundle_seal,
        "internal_three_pass_evidence": internal,
        "target_and_mask_identity_exact": internal["target_identity_exact"],
        "raw_diagnostic": internal_raw,
        "primary_gate": internal_primary,
        "combined_gate_pass": internal_pass,
        "legacy_state_sha256": internal_legacy_sha,
        "undefined_metrics": {
            "sps": "not_computed/undefined",
            "non_time": "not_computed/undefined",
            "final": "not_computed/undefined",
        },
        "access_before_internal_result_seal": access_before_internal_seal,
        "held5_open_count": access_before_internal_seal["held5_open_count"],
        "git": identity,
    }
    internal_evidence_path = args.output_dir / "internal_evidence.json"
    dr1.write_json_atomic(internal_evidence_path, internal_evidence)
    internal_evidence_file_sha = dr1.sha256(internal_evidence_path)
    ledger.seal_internal_result(
        internal_evidence_path, internal_evidence_file_sha, internal_pass)
    internal_evidence_seal = dict(ledger.internal_result_seal)
    del internal_dataset, internal_trajectories, internal_metadata
    del internal_windows
    gc.collect()

    base_result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract": CONTRACT,
        "manifest": manifest_report,
        "split_digests": {
            "assignment": partitions["report"]["assignment_sha256"],
            "windows": partitions["report"]["window_sha256"],
        },
        "normalizer": normalizer_report,
        "sample_stream_sha256": schedule["sha256"],
        "start": start,
        "training": training,
        "endpoint_bundle": bundle_seal,
        "ema": ema_seal_payload,
        "raw_model_unchanged_by_ema_seal": (
            raw_sha_before_ema_seal == raw_sha_after_ema_seal),
        "internal": internal,
        "internal_raw_diagnostic": internal_raw,
        "internal_primary_gate": internal_primary,
        "internal_evidence_seal": internal_evidence_seal,
        "undefined_metrics": {
            "sps": "not_computed/undefined",
            "non_time": "not_computed/undefined",
            "final": "not_computed/undefined",
        },
        "git": identity,
        "infra_repair_attempt": args.infra_repair_attempt,
        "release_checkpoint_written": False,
        "submission_authorized": False,
    }
    if not internal_pass:
        access_report = ledger.report()
        if not _access_contract_ok(access_report, held_expected=False):
            raise RuntimeError("MEF1 STOP access ledger failed")
        torch.cuda.synchronize(device)
        _check_deadlines(wall_deadline, gpu_deadline, "internal STOP")
        return {
            **base_result,
            "decision": (
                "STOP_MEF1_CONTRACT" if not internal_raw["pass"]
                else "STOP_MEF1_NO_INTERNAL_SIGNAL"),
            "held5": "unopened",
            "prefix_access_audit": access_report,
            "elapsed_seconds": time.perf_counter() - started,
        }

    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", ledger)
    if len(held_windows) != 433 or len(held_trajectories) != 14:
        raise RuntimeError("MEF1 held5 partition changed after H5 load")
    held_dataset = ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    held = evaluate_sealed_partition(
        model, bundle, held_dataset,
        sample_aoas(held_windows, held_metadata), normalizer,
        device, wall_deadline, gpu_deadline, ledger, phase="held")
    held_legacy_sha = legacy_state_sha256(model)
    if held_legacy_sha != training["final_legacy_state_sha256"]:
        raise RuntimeError("MEF1 held5 evaluation mutated legacy state")
    held["legacy_state_sha256_after"] = held_legacy_sha
    held_reports = held["reports"]
    held_raw = raw_diagnostic(
        held_reports["raw"]["metrics"],
        held_reports["identity"]["metrics"])
    held_primary = primary_gate(
        held_reports["ema"]["metrics"],
        held_reports["identity"]["metrics"],
        require_per_aoa=False)
    access_report = ledger.report()
    if not _access_contract_ok(access_report, held_expected=True):
        raise RuntimeError("MEF1 held5 access ledger failed")
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    _check_deadlines(wall_deadline, gpu_deadline, "formal completion")
    held_pass = bool(held_raw["pass"] and held_primary["pass"])
    return {
        **base_result,
        "decision": (
            "GO_MEF1_CONTROLLER_NECESSARY_CONDITION"
            if held_pass else (
                "STOP_MEF1_CONTRACT" if not held_raw["pass"]
                else "STOP_MEF1_HELD5")),
        "held5": held,
        "held5_raw_diagnostic": held_raw,
        "held5_primary_gate": held_primary,
        "prefix_access_audit": access_report,
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument(
        "--clean-manifest-sha256", default=FORMAL_MANIFEST_SHA256)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256", default=FORMAL_START_SHA256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--infra-repair-attempt", type=int, choices=(0, 1),
                        default=0)
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def validate_formal_args(args: argparse.Namespace) -> None:
    if args.output_dir is None or args.clean_manifest is None or (
            args.start_sim is None):
        raise ValueError(
            "formal MEF1 requires output, clean manifest, and start Sim paths")
    args.clean_manifest_sha256 = dr1.normalized_sha256(
        args.clean_manifest_sha256, "clean manifest SHA-256")
    args.start_sim_sha256 = dr1.normalized_sha256(
        args.start_sim_sha256, "start Sim SHA-256")
    if args.clean_manifest_sha256 != FORMAL_MANIFEST_SHA256:
        raise ValueError("formal MEF1 requires the frozen clean81 manifest")
    if args.start_sim_sha256 != FORMAL_START_SHA256:
        raise ValueError("formal MEF1 requires the frozen Sim10k checkpoint")
    if not args.execute_formal_gpu:
        raise ValueError("formal MEF1 requires --execute-formal-gpu")
    if args.acknowledge_protocol != FORMAL_ACKNOWLEDGEMENT:
        raise ValueError("formal MEF1 protocol acknowledgement is missing")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")


def main() -> None:
    args = parse_args()
    if args.contract_only:
        print(json.dumps({
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "contract": CONTRACT,
            "decision": "CONTRACT_ONLY_NO_DATA_NO_GPU",
        }, sort_keys=True))
        return
    validate_formal_args(args)
    try:
        result = run(args)
        dr1.write_json_atomic(args.output_dir / "result.json", result)
        print(json.dumps({
            "event": "mef1_screen_complete",
            "decision": result["decision"],
        }), flush=True)
    except Exception as exc:
        owner_path = args.output_dir / "run_owner.json"
        if owner_path.is_file():
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            if owner.get("schema") == "c208_mef1_run_owner_v1":
                ledger_path = args.output_dir / "access_ledger.json"
                staged_ledger = (
                    json.loads(ledger_path.read_text(encoding="utf-8"))
                    if ledger_path.is_file() else None)
                dr1.write_json_atomic(args.output_dir / "result.json", {
                    "schema": SCHEMA,
                    "experiment": EXPERIMENT,
                    "decision": "STOP_MEF1_CONTRACT",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "infra_repair_attempt": args.infra_repair_attempt,
                    "access_ledger": staged_ledger,
                    "access_ledger_file_sha256": (
                        dr1.sha256(ledger_path) if ledger_path.is_file()
                        else None),
                    "automatic_retry_started": False,
                    "release_checkpoint_written": False,
                    "submission_authorized": False,
                })
        raise


if __name__ == "__main__":
    main()
