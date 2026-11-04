#!/usr/bin/env python3
"""Produce the frozen B210 small4k full-clean2544 point-stage checkpoint.

This stage reruns the already confirmed 4000-update recipe on every allowed
clean81 prefix window.  Its sole model artifact is an EMA-u4000 point-stage
checkpoint.  It is not submittable: candidate-specific OOF UQ/refit remains
mandatory and this program never constructs or reads held-tail data.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import json
import math
from pathlib import Path
import shutil
import time
from typing import Mapping, Sequence

import torch

import compare_b210_np_to_frozen_c208 as frozen_compare
import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k


dr1 = ft12.dr1
npt1 = ft12.npt1
smu1 = ft12.smu1

EXPERIMENT = "b210_np_functional_real_small4k_full_clean_point"
SCHEMA = "b210_np_functional_real_small4k_full_clean_point_v1"
ENDPOINT_SCHEMA = "b210_np_functional_real_small4k_full_refit_endpoint_v1"
FORMAL_BRANCH = "codex/b210-small4k-release"
FORMAL_ACKNOWLEDGEMENT = "B210_SMALL4K_FULL_CLEAN_POINT_STAGE"
STATUS = "POINT_STAGE_ONLY_NOT_SUBMITTABLE"

SIM10K_SHA256 = (
    "f86faac5c376d2ee52ceb9209e407e2a42c114d79753dfd5fe2a51072207d85a")
DISCOVERY_SHA256 = (
    "4a5d533f5869618dde72d456ccc4c175a179b8218a7b4c6e5b652077e818736a")
CONFIRMATION_SHA256 = (
    "f469d930e2f47fcdb9d29c7ee394084276ee91ec707ff953749aa4a3136c1f95")
LATENCY_SHA256 = (
    "14153465006b0ff2a70677a400f2ebfd07386978c75e7835848feda7b3a3cf0e")

UPDATES = small4k.UPDATES
BATCH_SIZE = small4k.BATCH_SIZE
SEED = small4k.SEED
LR = small4k.LR
WEIGHT_DECAY = small4k.WEIGHT_DECAY
GRAD_CLIP = small4k.GRAD_CLIP
SCHEDULER_T_MAX = small4k.SCHEDULER_T_MAX
EMA_DECAY = small4k.EMA_DECAY
PRIMARY_ENDPOINT = small4k.PRIMARY_ENDPOINT
FIT_SAMPLES = dr1.EXPECTED_WINDOWS
WALL_BUDGET_SECONDS = 60 * 60
MIN_FREE_BYTES = 3 * 1024 ** 3

CONTRACT = {
    "authorization": {
        "sim10k_sha256": SIM10K_SHA256,
        "held10_discovery_sha256": DISCOVERY_SHA256,
        "held10_role": "post_hoc_discovery_only",
        "held0_independent_confirmation_sha256": CONFIRMATION_SHA256,
        "held0_required_decision": (
            "PASS_B210_SMALL4K_HELD0_INDEPENDENT_CONFIRMATION"),
        "latency_summary_sha256": LATENCY_SHA256,
        "latency_required_decision": "PASS_B210_ABBA_LATENCY",
    },
    "training": {
        "scope": "full_clean2544",
        "fit_samples": FIT_SAMPLES,
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": SCHEDULER_T_MAX,
        "grad_clip": GRAD_CLIP,
        "backbone_residual_dropout": 0.20,
        "native_losses": "Real joint point/member/quantile/soft-SPS",
        "tke_aux": "controller-only .01 cosine u1-u1500",
        "ema": "FP32 EMA999 continuous u0-u4000",
        "endpoint_selection_allowed": False,
    },
    "data": {
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": ft12.FORMAL_MANIFEST_SEMANTIC_SHA256,
        "trajectories": dr1.EXPECTED_TRAJECTORIES,
        "prefix_windows": FIT_SAMPLES,
        "normalizer": "all 2544 allowed prefix windows",
        "excluded": dr1.EXCLUDED_BASENAME,
        "held_or_tail_labels_read": False,
        "tail_arrays_read": 0,
    },
    "output": {
        "checkpoint_schema": ENDPOINT_SCHEMA,
        "sole_model_endpoint": PRIMARY_ENDPOINT,
        "status": STATUS,
        "candidate_specific_oof_uq_refit_required": True,
        "release_checkpoint_written": False,
        "submission_authorized": False,
    },
    "hard_stop": [
        "git_or_hardware", "sim_or_evidence_sha", "evidence_semantics",
        "manifest_or_excluded_file", "oom_or_nonfinite", "ema_or_stream",
        "wall_budget_60min",
    ],
}
CONTRACT_SHA256 = dr1.canonical_sha256(CONTRACT)


class FullCleanAccessAudit:
    """Permit exactly the clean81 training-prefix allowlist and no held read."""

    def __init__(self) -> None:
        self.opened: set[str] = set()
        self.events: list[dict] = []
        self.precursor_seal: dict | None = None
        self.final_seal: dict | None = None

    def before_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        if phase != "fit":
            raise RuntimeError("full-clean point stage cannot open held data")
        if filename == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded H5 reached full-clean point opener")
        if self.precursor_seal is not None:
            raise RuntimeError("full-clean prefix opened after endpoint seal")

    def record_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        if phase != "fit" or filename in self.opened:
            raise RuntimeError("invalid or duplicate full-clean prefix open")
        self.opened.add(filename)
        self.events.append({
            "file": filename,
            "aoa": int(row["aoa"]),
            "phase": "fit",
            "slice": "u/v[:val_boundary,::2,::2]",
            "max_read_frame_exclusive": int(row["val_boundary"]),
        })

    def seal(self, value: Mapping[str, object]) -> None:
        """Receive the same-weight small4k precursor from the reused loop."""
        if self.precursor_seal is not None:
            raise RuntimeError("duplicate full-clean precursor seal")
        if (value.get("endpoint") != PRIMARY_ENDPOINT
                or value.get("primary") is not True
                or int(value.get("iteration", -1)) != UPDATES):
            raise RuntimeError("full-clean precursor endpoint changed")
        self.precursor_seal = dict(value)

    def bind_final(self, value: Mapping[str, object]) -> None:
        if self.precursor_seal is None or self.final_seal is not None:
            raise RuntimeError("invalid full-clean final endpoint binding")
        if (value.get("endpoint") != PRIMARY_ENDPOINT
                or value.get("primary") is not True
                or int(value.get("iteration", -1)) != UPDATES):
            raise RuntimeError("full-clean final endpoint changed")
        if (value.get("model_state_sha256")
                != self.precursor_seal.get("model_state_sha256")):
            raise RuntimeError("full-clean rebind changed EMA model state")
        self.final_seal = dict(value)

    def report(self, expected_allowlist: Sequence[str]) -> dict:
        return {
            "opened_file_allowlist": sorted(self.opened),
            "expected_file_allowlist": sorted(expected_allowlist),
            "allowlist_exact": sorted(self.opened) == sorted(expected_allowlist),
            "prefix_only": all(row["phase"] == "fit" for row in self.events),
            "held_or_tail_labels_read": False,
            "tail_arrays_read": 0,
            "excluded_opened": dr1.EXCLUDED_BASENAME in self.opened,
            "precursor_model_state_sha256": (
                None if self.precursor_seal is None
                else self.precursor_seal["model_state_sha256"]),
            "final_endpoint_seal": self.final_seal,
            "events": self.events,
        }


def _read_json_mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON evidence must be a mapping: {path}")
    return value


def _load_exact_json(path: Path, expected_sha: str, label: str) -> tuple[dict, dict]:
    expected = dr1.normalized_sha256(expected_sha, label)
    actual = dr1.sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} changed: {path}")
    return _read_json_mapping(path), {
        "path": str(path.resolve()), "sha256": actual,
    }


def validate_confirmation_row(row: Mapping[str, object]) -> dict:
    if (row.get("schema") != small4k.SCHEMA
            or row.get("experiment") != small4k.EXPERIMENT
            or row.get("contract_sha256") != small4k.CONTRACT_SHA256
            or int(row.get("held_aoa", -1)) != 0):
        raise RuntimeError("held0 confirmation identity changed")
    if (row.get("primary_endpoint") != PRIMARY_ENDPOINT
            or row.get("endpoint_selection_allowed") is not False
            or row.get("decision")
            != "PASS_B210_SMALL4K_HELD0_INDEPENDENT_CONFIRMATION"):
        raise RuntimeError("held0 confirmation PASS policy changed")
    evidence = row.get("evidence_class", {})
    if (evidence.get("held10")
            != "discovery_only_post_hoc_not_confirmation"
            or evidence.get("held0") != "single_independent_confirmation"
            or evidence.get("pooled_official") is not False
            or evidence.get("submission") is not False):
        raise RuntimeError("held0 confirmation evidence role changed")
    if (row.get("discovery", {}).get("sha256") != DISCOVERY_SHA256
            or row.get("start", {}).get("checkpoint_sha256") != SIM10K_SHA256):
        raise RuntimeError("held0 confirmation evidence/Sim binding changed")
    gate = small4k.confirmation_gate(row.get("comparison", {}))
    if (row.get("confirmation_gate") != gate or gate["pass"] is not True):
        raise RuntimeError("held0 independent confirmation gate is not PASS")
    if (row.get("manifest_file_sha256")
            != ft12.FORMAL_MANIFEST_FILE_SHA256
            or row.get("manifest", {}).get("semantic_sha256")
            != ft12.FORMAL_MANIFEST_SEMANTIC_SHA256
            or row.get("manifest", {}).get("windows") != FIT_SAMPLES
            or row.get("manifest", {}).get("excluded")
            != dr1.EXCLUDED_BASENAME):
        raise RuntimeError("held0 confirmation manifest changed")
    access = row.get("prefix_access_audit", {})
    if (access.get("tail_arrays_read") != 0
            or access.get("excluded_opened") is not False
            or access.get("held_first_open_after_endpoint_seal") is not True
            or access.get("held_complete_passes") != 1):
        raise RuntimeError("held0 confirmation access evidence changed")
    endpoint = row.get("endpoint", {})
    state = frozen_compare._validated_metric_state(
        endpoint.get("metric_state"), int(row["samples"]),
        "held0 confirmed ema_u4000")
    metrics = smu1.finalize_metric_state(state)
    seal = endpoint.get("seal", {})
    if (endpoint.get("metrics") != metrics
            or seal.get("endpoint") != PRIMARY_ENDPOINT
            or seal.get("primary") is not True
            or int(seal.get("iteration", -1)) != UPDATES):
        raise RuntimeError("held0 confirmed endpoint changed")
    frozen_compare._validated_audit(
        endpoint.get("held_target_audit"), int(row["samples"]),
        "held0 confirmed ema_u4000")
    return {
        "held_aoa": 0,
        "decision": row["decision"],
        "start_checkpoint_sha256": SIM10K_SHA256,
        "endpoint": PRIMARY_ENDPOINT,
        "endpoint_model_state_sha256": seal.get("model_state_sha256"),
        "metrics": metrics,
        "confirmation_role": "independent_held0",
    }


def validate_latency_summary(row: Mapping[str, object]) -> dict:
    gates = row.get("gates")
    ratios = row.get("ratios")
    if (row.get("evidence_class") != "s2_deployment_archive_latency_abba"
            or row.get("sequence") != ["c208", "b210", "b210", "c208"]
            or row.get("decision") != "PASS_B210_ABBA_LATENCY"
            or not isinstance(gates, Mapping) or not gates
            or any(value is not True for value in gates.values())
            or not isinstance(ratios, Mapping)
            or any(not math.isfinite(float(value)) for value in ratios.values())):
        raise RuntimeError("B210 latency summary is not a strict ABBA PASS")
    b210_archive_sha = dr1.normalized_sha256(
        str(row.get("archive_sha256", {}).get("b210")),
        "latency B210 archive SHA")
    if int(row.get("b210_peak_reserved_bytes", -1)) < 0:
        raise RuntimeError("B210 latency peak-memory evidence changed")
    return {
        "decision": row["decision"],
        "sequence": list(row["sequence"]),
        "gates": dict(gates),
        "ratios": dict(ratios),
        "b210_archive_sha256": b210_archive_sha,
        "b210_peak_reserved_bytes": int(row["b210_peak_reserved_bytes"]),
    }


def load_authorization(args: argparse.Namespace) -> dict:
    discovery_row, discovery_source = _load_exact_json(
        args.discovery_fold10, DISCOVERY_SHA256, "held10 discovery SHA")
    discovery = small4k.validate_discovery_row(discovery_row)
    if discovery["start_checkpoint_sha256"] != SIM10K_SHA256:
        raise RuntimeError("held10 discovery Sim10k SHA changed")
    confirmation_row, confirmation_source = _load_exact_json(
        args.confirmation_held0, CONFIRMATION_SHA256,
        "held0 confirmation SHA")
    confirmation = validate_confirmation_row(confirmation_row)
    latency_row, latency_source = _load_exact_json(
        args.latency_summary, LATENCY_SHA256, "latency summary SHA")
    latency = validate_latency_summary(latency_row)
    return {
        "sim10k_sha256": SIM10K_SHA256,
        "held10_discovery": {**discovery_source, **discovery},
        "held0_confirmation": {**confirmation_source, **confirmation},
        "latency": {**latency_source, **latency},
        "point_stage_authorized": True,
        "submission_authorized": False,
    }


def rebind_full_clean_endpoint(
        precursor: Mapping[str, object], *, fit_samples: int,
        manifest_report: Mapping[str, object],
        authorization: Mapping[str, object],
        ) -> dict:
    """Atomically replace precursor metadata without changing EMA weights."""
    path = Path(str(precursor["path"]))
    if dr1.sha256(path) != precursor.get("file_sha256"):
        raise RuntimeError("full-clean precursor checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != small4k.ENDPOINT_SCHEMA
            or payload.get("experiment") != small4k.EXPERIMENT
            or payload.get("endpoint") != PRIMARY_ENDPOINT
            or payload.get("primary") is not True
            or int(payload.get("iteration", -1)) != UPDATES
            or not isinstance(state, Mapping)):
        raise RuntimeError("full-clean precursor payload changed")
    ft12.validate_b210_model_config(payload["model_config"])
    state_sha = dr1.model_state_sha256(state)
    if (state_sha != precursor.get("model_state_sha256")
            or state_sha != payload.get("model_state_sha256")):
        raise RuntimeError("full-clean precursor state SHA changed")
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in state.values() if isinstance(value, torch.Tensor)):
        raise TypeError("full-clean point state must remain FP32")
    normalizer = payload.get("normalizer")
    if (not isinstance(normalizer, Mapping)
            or set(normalizer) != {
                "mean_input", "mean_target", "std_input", "std_target"}):
        raise RuntimeError("full-clean normalizer is not flat")
    ema = payload.get("ema", {})
    if (ema.get("schema") != small4k.ema_lib.EMA_SEAL_SCHEMA
            or ema.get("update_order")
            != small4k.ema_lib.UPDATE_ORDER_CONTRACT
            or int(ema.get("updates", -1)) != UPDATES
            or float(ema.get("decay", -1.0)) != EMA_DECAY
            or ema.get("shadow_dtype") != "torch.float32"
            or ema.get("materialized_model_state_sha256") != state_sha):
        raise RuntimeError("full-clean EMA metadata changed")

    evidence_bindings = {
        "sim10k_sha256": SIM10K_SHA256,
        "held10_discovery_sha256": DISCOVERY_SHA256,
        "held0_confirmation_sha256": CONFIRMATION_SHA256,
        "latency_summary_sha256": LATENCY_SHA256,
    }
    final_payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real",
        "iteration": UPDATES,
        "endpoint": PRIMARY_ENDPOINT,
        "primary": True,
        "training_scope": "full_clean2544",
        "fit_samples": fit_samples,
        "status": STATUS,
        "model_state_dict": OrderedDict(
            (name, value.detach().cpu()) for name, value in state.items()),
        "model_state_sha256": state_sha,
        "model_config": payload["model_config"],
        "start": payload["start"],
        "normalizer": dict(normalizer),
        "normalizer_sha256": payload["normalizer_sha256"],
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": ft12.FORMAL_MANIFEST_SEMANTIC_SHA256,
        "manifest_windows": int(manifest_report["windows"]),
        "manifest_excluded": manifest_report["excluded"],
        "sample_stream_sha256": payload["sample_stream_sha256"],
        "ema": ema,
        "optimizer_state_saved": False,
        "held_or_tail_labels_read": False,
        "tail_arrays_read": 0,
        "endpoint_selection_allowed": False,
        "candidate_specific_oof_uq_refit_required": True,
        "evidence_bindings": evidence_bindings,
        "authorization_sha256": dr1.canonical_sha256(authorization),
        "release_checkpoint_written": False,
        "submission_authorized": False,
    }
    if (final_payload["start"].get("checkpoint_sha256") != SIM10K_SHA256
            or fit_samples != FIT_SAMPLES
            or final_payload["manifest_windows"] != FIT_SAMPLES
            or final_payload["manifest_excluded"] != dr1.EXCLUDED_BASENAME):
        raise RuntimeError("full-clean endpoint provenance changed")
    temporary = path.with_name(path.name + ".full-clean.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(final_payload, temporary)
    temporary.replace(path)
    final_seal = {
        "schema": ENDPOINT_SCHEMA,
        "endpoint": PRIMARY_ENDPOINT,
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "model_config_sha256": dr1.canonical_sha256(
            final_payload["model_config"]),
        "normalizer_sha256": final_payload["normalizer_sha256"],
        "sample_stream_sha256": final_payload["sample_stream_sha256"],
        "iteration": UPDATES,
        "primary": True,
        "training_scope": "full_clean2544",
        "fit_samples": fit_samples,
        "status": STATUS,
    }
    del payload, state, final_payload
    return final_seal


def validate_point_checkpoint(path: Path, seal: Mapping[str, object]) -> dict:
    if dr1.sha256(path) != seal.get("file_sha256"):
        raise RuntimeError("full-clean point checkpoint changed after seal")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema": ENDPOINT_SCHEMA,
        "phase": "real",
        "iteration": UPDATES,
        "endpoint": PRIMARY_ENDPOINT,
        "primary": True,
        "training_scope": "full_clean2544",
        "fit_samples": FIT_SAMPLES,
        "status": STATUS,
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": ft12.FORMAL_MANIFEST_SEMANTIC_SHA256,
        "manifest_windows": FIT_SAMPLES,
        "manifest_excluded": dr1.EXCLUDED_BASENAME,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise RuntimeError("full-clean point consumer interface changed")
    config = ft12.validate_b210_model_config(payload["model_config"])
    state = payload.get("model_state_dict")
    if (not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError("full-clean point strict state changed")
    ema = payload.get("ema", {})
    if (ema.get("schema") != small4k.ema_lib.EMA_SEAL_SCHEMA
            or ema.get("update_order")
            != small4k.ema_lib.UPDATE_ORDER_CONTRACT
            or int(ema.get("updates", -1)) != UPDATES
            or float(ema.get("decay", -1.0)) != EMA_DECAY
            or ema.get("shadow_dtype") != "torch.float32"
            or ema.get("materialized_model_state_sha256")
            != seal["model_state_sha256"]):
        raise RuntimeError("full-clean point EMA consumer interface changed")
    expected_evidence = {
        "sim10k_sha256": SIM10K_SHA256,
        "held10_discovery_sha256": DISCOVERY_SHA256,
        "held0_confirmation_sha256": CONFIRMATION_SHA256,
        "latency_summary_sha256": LATENCY_SHA256,
    }
    if (payload.get("evidence_bindings") != expected_evidence
            or payload.get("start", {}).get("checkpoint_sha256")
            != SIM10K_SHA256
            or payload.get("candidate_specific_oof_uq_refit_required")
            is not True
            or payload.get("held_or_tail_labels_read") is not False
            or payload.get("tail_arrays_read") != 0
            or payload.get("endpoint_selection_allowed") is not False
            or payload.get("release_checkpoint_written") is not False
            or payload.get("submission_authorized") is not False):
        raise RuntimeError("full-clean point evidence/status interface changed")
    dr1.normalized_sha256(
        str(payload.get("sample_stream_sha256")), "point sample stream SHA")
    normalizer = payload.get("normalizer")
    if (not isinstance(normalizer, Mapping)
            or set(normalizer) != {
                "mean_input", "mean_target", "std_input", "std_target"}
            or dr1.canonical_sha256(normalizer)
            != payload.get("normalizer_sha256")):
        raise RuntimeError("full-clean point flat normalizer changed")
    return {
        "strict_b210_config": config == ft12.expected_b210_config(),
        "strict_state": True,
        "flat_normalizer": True,
        "consumer_interface": required,
    }


def _disk_preflight(output_dir: Path) -> dict:
    probe = output_dir.parent.resolve()
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError("no existing parent for full-clean output")
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"full-clean output disk has {free}; requires {MIN_FREE_BYTES}")
    return {"probe": str(probe), "free_bytes_before_run": free,
            "minimum_free_bytes": MIN_FREE_BYTES}


def run(args: argparse.Namespace) -> dict:
    identity = dr1.validate_formal_git_identity(dr1.git_identity())
    if identity["branch"] != FORMAL_BRANCH:
        raise RuntimeError("full-clean point stage is on the wrong branch")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    authorization = load_authorization(args)
    start_sha = dr1.normalized_sha256(args.start_sim_sha256, "start Sim SHA")
    if start_sha != SIM10K_SHA256:
        raise RuntimeError("full-clean point stage received wrong Sim10k SHA")
    device = torch.device(args.device)
    hardware = ft12.validate_hardware(device, args.hardware_class)
    disk = _disk_preflight(args.output_dir)
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("full-clean manifest file SHA changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    if (len(rows) != dr1.EXPECTED_TRAJECTORIES
            or sum(len(row["train_starts"]) for row in rows) != FIT_SAMPLES
            or any(str(row["file"]) == dr1.EXCLUDED_BASENAME for row in rows)):
        raise RuntimeError("full-clean clean81 manifest membership changed")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + WALL_BUDGET_SECONDS
    torch.set_float32_matmul_precision("high")
    access = FullCleanAccessAudit()
    trajectories, metadata, windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", access)
    if len(windows) != FIT_SAMPLES:
        raise RuntimeError("full-clean loaded prefix count changed")
    dataset = npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    sufficient = smu1.window_sufficient_statistics(trajectories, windows)
    statistics_values = smu1.fold_statistics(
        sufficient, range(FIT_SAMPLES))
    normalizer = smu1.FoldNormalizer(statistics_values, device)
    normalizer_report = ft12.normalizer_payload(normalizer)
    schedule = smu1.paired_batch_schedule(
        range(FIT_SAMPLES), windows, updates=UPDATES, seed=SEED)
    schedule["updates"] = UPDATES

    npt1.ensemble_train.seed_everything(SEED)
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, SIM10K_SHA256, normalizer, device)
    training, precursor = small4k.train_candidate(
        model, dataset, schedule, normalizer, device, deadline,
        args.output_dir, start_report, access)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    final_seal = rebind_full_clean_endpoint(
        precursor, fit_samples=len(windows), manifest_report=manifest_report,
        authorization=authorization)
    access.bind_final(final_seal)
    checkpoint_path = Path(str(final_seal["path"]))
    consumer_validation = validate_point_checkpoint(
        checkpoint_path, final_seal)
    access_report = access.report([row["file"] for row in rows])
    if not (access_report["allowlist_exact"]
            and access_report["prefix_only"]
            and access_report["held_or_tail_labels_read"] is False
            and access_report["tail_arrays_read"] == 0
            and access_report["excluded_opened"] is False
            and access_report["final_endpoint_seal"] == final_seal):
        raise RuntimeError("full-clean prefix access contract failed")
    if time.perf_counter() > deadline:
        raise TimeoutError("full-clean point stage exceeded 60-minute budget")

    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "status": STATUS,
        "training_scope": "full_clean2544",
        "fit_samples": FIT_SAMPLES,
        "authorization": authorization,
        "manifest": manifest_report,
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "normalizer": {
            **normalizer_report,
            "kind": "all 2544 allowed clean81 prefix windows",
            "held_or_tail_target_in_normalizer": False,
        },
        "sample_stream_sha256": schedule["sha256"],
        "fit_ordered_key_sha256": smu1.ordered_held_key_digest(
            range(FIT_SAMPLES), windows, metadata),
        "start": start_report,
        "training": training,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "endpoint_selection_allowed": False,
        "point_checkpoint": final_seal,
        "consumer_validation": consumer_validation,
        "prefix_access_audit": access_report,
        "git": identity,
        "hardware": hardware,
        "disk_preflight": disk,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "elapsed_seconds": time.perf_counter() - started,
        "candidate_specific_oof_uq_refit_required": True,
        "release_checkpoint_written": False,
        "submission_authorized": False,
        "decision": (
            "COMPLETE_B210_SMALL4K_FULL_CLEAN_"
            "POINT_STAGE_ONLY_NOT_SUBMITTABLE"),
    }
    dr1.write_json_atomic(args.output_dir / "point_stage_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-full-clean", action="store_true")
    mode.add_argument("--contract-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256")
    parser.add_argument("--discovery-fold10", type=Path)
    parser.add_argument("--confirmation-held0", type=Path)
    parser.add_argument("--latency-summary", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hardware-class", choices=("rtx4080_super",),
        default="rtx4080_super")
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    execution_values = (
        args.output_dir, args.clean_manifest, args.start_sim,
        args.start_sim_sha256, args.discovery_fold10,
        args.confirmation_held0, args.latency_summary)
    if args.run_full_clean:
        if any(value is None for value in execution_values):
            raise ValueError("full-clean point runner is missing required inputs")
        if (not args.execute_formal_gpu
                or args.acknowledge_protocol != FORMAL_ACKNOWLEDGEMENT):
            raise ValueError("full-clean GPU execution requires acknowledgement")
        args.start_sim_sha256 = dr1.normalized_sha256(
            args.start_sim_sha256, "start-sim-sha256")
    elif any(value is not None for value in execution_values) or (
            args.execute_formal_gpu):
        raise ValueError("contract-only mode received execution inputs")


def write_stop(output_dir: Path, error: BaseException) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stop.json"
    if path.exists():
        return
    if isinstance(error, torch.cuda.OutOfMemoryError):
        decision = "STOP_B210_FULL_CLEAN_POINT_OOM"
    elif isinstance(error, TimeoutError):
        decision = "STOP_B210_FULL_CLEAN_POINT_WALL_BUDGET"
    else:
        decision = "STOP_B210_FULL_CLEAN_POINT_RUNTIME"
    dr1.write_json_atomic(path, {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "status": STATUS,
        "decision": decision,
        "error_type": type(error).__name__,
        "error": str(error),
        "release_checkpoint_written": False,
        "submission_authorized": False,
    })


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        print(json.dumps({
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "contract": CONTRACT,
            "contract_sha256": CONTRACT_SHA256,
            "decision": "CONTRACT_ONLY_NO_DATA_NO_GPU",
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return
    try:
        result = run(args)
    except Exception as error:
        if args.output_dir is not None and not isinstance(error, FileExistsError):
            write_stop(args.output_dir, error)
        raise
    print(json.dumps({
        "event": "b210_small4k_full_clean_point_complete",
        "status": result["status"],
        "decision": result["decision"],
        "checkpoint": result["point_checkpoint"],
        "result": str(args.output_dir / "point_stage_result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
