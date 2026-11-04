#!/usr/bin/env python3
"""Matched B210 prefix-versus-full released-Real support experiment.

For held AoA 10 or 20, both arms restart from one immutable scratch Sim10k
checkpoint and use the same B210 architecture, native Real loss, augmentation,
normalizer, seed, optimizer, scheduler, and 4,000-update budget.  The sole
method difference is optimizer support:

* ``prefix``: other-AoA clean81 prefixes, stride 5;
* ``full``: the same other-AoA trajectories in full, stride 5.

The held AoA is absent from both training and normalization.  Its complete
released trajectories are opened only after raw and EMA endpoints for both
arms have been written and SHA-sealed.  This runner never accepts or reads a
clean716 manifest and applies no score gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import time
from typing import Iterator

import h5py
import numpy as np
import torch

import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_np_functional_sim as sim_tke
import train_b210_scratch_real_dense_aug as dense4k
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1


EXPERIMENT = "b210_full_released_real_support"
SCHEMA = "b210_full_released_real_support_fold_v1"
AGGREGATE_SCHEMA = "b210_full_released_real_support_aggregate_v1"
ENDPOINT_SCHEMA = "b210_full_released_real_support_endpoint_v1"
FORMAL_BRANCH = "codex/b210-full-released-real-support"
ACKNOWLEDGEMENT = "RUN_B210_FULL_RELEASED_REAL_SUPPORT_MATCHED_V1"
FORMAL_START_SHA256 = (
    "8a6d9a1055269f95a308b42e73ee94b00b005edbf550430dfa096a610e2e0f29")

HELD_AOAS = (10, 20)
ARM_ORDER = ("prefix", "full")
PRIMARY_ENDPOINT = "ema_u4000"
DIAGNOSTIC_ENDPOINT = "raw_u4000"
ENDPOINT_ORDER = (DIAGNOSTIC_ENDPOINT, PRIMARY_ENDPOINT)
UPDATES = 4_000
SEED = 4
OPTIMIZER_STRIDE = 5
NORMALIZER_STRIDE = 20
EVALUATION_STRIDE = 20
SCHEDULER_T_MAX = 12_000
EMA_DECAY = 0.999
DEFAULT_WALL_BUDGET_SECONDS = 3 * 60 * 60

AUGMENTATION = dict(dense4k.AUGMENTATION)
FINAL_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_rel_l2_score": 0.10,
    "mvpe_rel_l2_score": 0.09,
    "native_sps_score": 0.25,
}

# Counts are frozen against the SHA-bound clean81 runtime manifest.  The
# normalizer count uses its original sparse prefix starts; optimizer windows
# use stride 5; held evaluation uses every legal full-trajectory stride-20
# start (including the one guard start per trajectory omitted by clean716).
EXPECTED_FOLD_COUNTS = {
    10: {
        "fit_trajectories": 63,
        "held_trajectories": 18,
        "normalizer_prefix_windows": 1_968,
        "prefix_optimizer_windows": 7_683,
        "full_optimizer_windows": 10_214,
        "held_full_windows": 756,
    },
    20: {
        "fit_trajectories": 65,
        "held_trajectories": 16,
        "normalizer_prefix_windows": 2_055,
        "prefix_optimizer_windows": 8_025,
        "full_optimizer_windows": 10_663,
        "held_full_windows": 643,
    },
}

CONTRACT = {
    "question": (
        "Does replacing prefix-only Real optimizer support with all released "
        "frames improve one otherwise identical B210 Real4k fit?"),
    "folds": list(HELD_AOAS),
    "arms": {
        "prefix": "other-AoA prefix frames, stride 5",
        "full": "same other-AoA full trajectories, stride 5",
    },
    "matched": {
        "start": (
            "one identical B210 Sim10k checkpoint with no Real H5 or "
            "Real-closure supervision"),
        "start_caveat": (
            "Sim soft-SPS uses fixed Real statistics for scale; this is not "
            "a claim of zero Real-derived information"),
        "start_checkpoint_sha256": FORMAL_START_SHA256,
        "architecture": "strict B210 candidate",
        "loss": "native Real joint point/member/quantile/soft-SPS + early TKE aux",
        "augmentation": AUGMENTATION,
        "normalizer": "same other-AoA sparse-prefix statistics in both arms",
        "seed": SEED,
        "updates": UPDATES,
        "optimizer": "AdamW(lr=2e-4, weight_decay=1e-5)",
        "scheduler": f"CosineAnnealingLR(T_max={SCHEDULER_T_MAX})",
        "ema": f"FP32 EMA decay={EMA_DECAY}",
    },
    "held_data": {
        "AoAs": list(HELD_AOAS),
        "evaluation": "full trajectory, every legal stride-20 20->20 window",
        "opened": "only after four raw/EMA endpoint seals",
        "in_normalizer": False,
        "in_optimizer": False,
        "in_teacher": False,
        "in_uq_fit": False,
        "clean716_manifest_accepted": False,
    },
    "endpoints": {
        "primary": PRIMARY_ENDPOINT,
        "diagnostic_only": DIAGNOSTIC_ENDPOINT,
        "held_endpoint_selection": False,
    },
    "excluded": dr1.EXCLUDED_BASENAME,
    "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
    "manifest_semantic_sha256": ft12.FORMAL_MANIFEST_SEMANTIC_SHA256,
    "score_gate": None,
    "submission_authorized": False,
}
CONTRACT_SHA256 = dr1.canonical_sha256(CONTRACT)


def endpoint_key(arm: str, endpoint: str) -> str:
    if arm not in ARM_ORDER or endpoint not in ENDPOINT_ORDER:
        raise ValueError(f"invalid endpoint identity: {arm}/{endpoint}")
    return f"{arm}:{endpoint}"


class FullSupportAccessAudit:
    """Keep held-condition H5 files behind all four endpoint seals."""

    def __init__(self, held_aoa: int) -> None:
        if held_aoa not in HELD_AOAS:
            raise ValueError(f"unsupported held AoA {held_aoa}")
        self.held_aoa = held_aoa
        self.seals: dict[str, dict] = {}
        self.fit_opened: list[str] = []
        self.held_opened: list[str] = []
        self.events: list[dict] = []
        self.evaluations: list[str] = []
        self._seals_verified_before_held = False

    @property
    def all_sealed(self) -> bool:
        expected = {
            endpoint_key(arm, endpoint)
            for arm in ARM_ORDER for endpoint in ENDPOINT_ORDER
        }
        return set(self.seals) == expected

    def seal(self, arm: str, endpoint: str,
             value: Mapping[str, object]) -> None:
        key = endpoint_key(arm, endpoint)
        if key in self.seals:
            raise RuntimeError(f"endpoint sealed twice: {key}")
        path = Path(str(value["path"]))
        file_sha = dr1.normalized_sha256(
            str(value["file_sha256"]), f"{key} file SHA")
        state_sha = dr1.normalized_sha256(
            str(value["model_state_sha256"]), f"{key} state SHA")
        if not path.is_file() or dr1.sha256(path) != file_sha:
            raise RuntimeError(f"endpoint file was not sealed: {key}")
        self.seals[key] = {
            "arm": arm,
            "endpoint": endpoint,
            "path": str(path.resolve()),
            "file_sha256": file_sha,
            "model_state_sha256": state_sha,
            "sealed_monotonic_ns": time.monotonic_ns(),
        }

    def _verify_seals_once(self) -> None:
        if not self.all_sealed:
            raise RuntimeError("held AoA opened before all four endpoint seals")
        if self._seals_verified_before_held:
            return
        for key, seal in self.seals.items():
            path = Path(seal["path"])
            if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
                raise RuntimeError(f"sealed endpoint drifted before held open: {key}")
        self._seals_verified_before_held = True

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        aoa = int(row["aoa"])
        if filename == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded duplicate reached full-support opener")
        if phase == "fit":
            if aoa == self.held_aoa or self.seals:
                raise RuntimeError("fit H5 is held-condition or opened after sealing")
            if filename in self.fit_opened:
                raise RuntimeError(f"fit H5 opened twice: {filename}")
            return
        if phase != "held" or aoa != self.held_aoa:
            raise RuntimeError("held H5 is not the requested held condition")
        self._verify_seals_once()
        if filename in self.held_opened:
            raise RuntimeError(f"held H5 opened twice: {filename}")

    def record_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        target = self.fit_opened if phase == "fit" else self.held_opened
        if filename in target:
            raise RuntimeError(f"H5 open record repeated: {filename}")
        target.append(filename)
        self.events.append({
            "file": filename,
            "aoa": int(row["aoa"]),
            "phase": phase,
            "slice": "u/v[:,::2,::2]",
            "max_read_frame_exclusive": int(row["frames"]),
            "all_four_endpoints_sealed_before_open": self.all_sealed,
        })

    def record_evaluation(self, arm: str, endpoint: str) -> None:
        key = endpoint_key(arm, endpoint)
        if not self._seals_verified_before_held:
            raise RuntimeError("evaluation began before sealed held-data open")
        if key in self.evaluations:
            raise RuntimeError(f"endpoint evaluated twice: {key}")
        self.evaluations.append(key)

    def report(self, fit_rows: Sequence[Mapping[str, object]],
               held_rows: Sequence[Mapping[str, object]]) -> dict:
        expected_fit = sorted(str(row["file"]) for row in fit_rows)
        expected_held = sorted(str(row["file"]) for row in held_rows)
        expected_evaluations = sorted(
            endpoint_key(arm, endpoint)
            for arm in ARM_ORDER for endpoint in ENDPOINT_ORDER)
        return {
            "held_aoa": self.held_aoa,
            "fit_opened": sorted(self.fit_opened),
            "expected_fit": expected_fit,
            "fit_allowlist_exact": sorted(self.fit_opened) == expected_fit,
            "held_opened_after_all_seals": sorted(self.held_opened),
            "expected_held": expected_held,
            "held_allowlist_exact": sorted(self.held_opened) == expected_held,
            "four_endpoint_seals": dict(self.seals),
            "all_seals_verified_before_held": self._seals_verified_before_held,
            "held_target_opened_before_all_seals": False,
            "held_condition_in_normalizer_optimizer_teacher_or_uq": False,
            "clean716_manifest_read": False,
            "evaluations": list(self.evaluations),
            "evaluation_set_exact": sorted(self.evaluations)
            == expected_evaluations,
            "events": list(self.events),
        }


class EmaSealCapture:
    """Capture the generic small4k EMA seal without authorizing held reads."""

    def __init__(self) -> None:
        self.value: dict | None = None

    def seal(self, value: Mapping[str, object]) -> None:
        if self.value is not None:
            raise RuntimeError("EMA endpoint was sealed twice")
        self.value = dict(value)


@contextmanager
def endpoint_contract(held_aoa: int) -> Iterator[None]:
    """Bind the existing B210 Real4k trainer to this endpoint schema."""
    if held_aoa not in HELD_AOAS:
        raise ValueError(f"unsupported held AoA {held_aoa}")
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "PRIMARY_ENDPOINT": PRIMARY_ENDPOINT,
        "UPDATES": UPDATES,
        "SCHEDULER_T_MAX": SCHEDULER_T_MAX,
        "HELD_AOA": held_aoa,
    }
    original = {name: getattr(small4k, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(small4k, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(small4k, name, value)


def load_full_rows(
        real_dir: Path, rows: Sequence[Mapping[str, object]], phase: str,
        audit: FullSupportAccessAudit,
        *, raw_grid: bool = False,
        ) -> tuple[list[torch.Tensor], list[dict]]:
    """Load complete released u/v trajectories after access checks."""
    root = real_dir.resolve()
    trajectories: list[torch.Tensor] = []
    metadata: list[dict] = []
    for row in rows:
        audit.before_open(row, phase)
        filename = str(row["file"])
        path = (root / filename).resolve()
        if path.parent != root or path.name != filename:
            raise ValueError(f"unsafe released-Real path: {filename}")
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            u_dataset = handle["u"]
            v_dataset = handle["v"]
            if u_dataset.shape != v_dataset.shape:
                raise ValueError(f"u/v shapes differ in {filename}")
            frames = int(row["frames"])
            if int(u_dataset.shape[0]) != frames:
                raise ValueError(f"frame count changed in {filename}")
            if int(np.asarray(handle["re"])) != int(row["re"]):
                raise ValueError(f"Re metadata changed in {filename}")
            if int(np.asarray(handle["aoa"])) != int(row["aoa"]):
                raise ValueError(f"AoA metadata changed in {filename}")
            stride = 1 if raw_grid else 2
            u = np.asarray(u_dataset[:, ::stride, ::stride], dtype=np.float32)
            v = np.asarray(v_dataset[:, ::stride, ::stride], dtype=np.float32)
            if raw_grid and u.shape[1:] != (64, 128):
                raise ValueError(f'raw PIV grid changed in {filename}')
        if u.shape != v.shape or int(u.shape[0]) != int(row["frames"]):
            raise RuntimeError(f"full trajectory shape mismatch: {filename}")
        audit.record_open(row, phase)
        trajectories.append(torch.from_numpy(np.ascontiguousarray(
            np.stack((u, v), axis=-1))))
        metadata.append({
            "file": filename,
            "frames": int(row["frames"]),
            "prefix_frames": int(row["val_boundary"]),
            "re": int(row["re"]),
            "aoa": int(row["aoa"]),
        })
    return trajectories, metadata


def manifest_prefix_windows(rows: Sequence[Mapping[str, object]]) -> list:
    """Build the SHA-frozen sparse prefix windows used by the normalizer."""
    window_type = ft12.npt1.ensemble_train.common.repro.Window
    windows = []
    for trajectory, row in enumerate(rows):
        windows.extend(
            window_type(trajectory, int(start))
            for start in row["train_starts"])
    return windows


def support_windows(
        trajectories: Sequence[torch.Tensor],
        metadata: Sequence[Mapping[str, object]], *, support: str,
        stride: int = OPTIMIZER_STRIDE,
        ) -> list:
    """Build prefix or complete-trajectory windows at one fixed stride."""
    if support not in ARM_ORDER:
        raise ValueError(f"unknown optimizer support: {support}")
    if stride <= 0 or len(trajectories) != len(metadata):
        raise ValueError("invalid full-support window inputs")
    window_type = ft12.npt1.ensemble_train.common.repro.Window
    windows = []
    for trajectory, (values, row) in enumerate(zip(trajectories, metadata)):
        if int(values.shape[0]) != int(row["frames"]):
            raise RuntimeError("trajectory length differs from bound metadata")
        limit = int(row["prefix_frames"] if support == "prefix" else row["frames"])
        if limit < 40 or limit > int(values.shape[0]):
            raise ValueError("support limit cannot form a valid 20->20 window")
        windows.extend(
            window_type(trajectory, start)
            for start in range(0, limit - 40 + 1, stride))
    return windows


def full_evaluation_windows(
        trajectories: Sequence[torch.Tensor],
        metadata: Sequence[Mapping[str, object]],
        ) -> list:
    return support_windows(
        trajectories, metadata, support="full", stride=EVALUATION_STRIDE)


def save_raw_endpoint(
        path: Path, model, *, arm: str, held_aoa: int,
        start_report: Mapping[str, object],
        normalizer: smu1.FoldNormalizer, stream_sha256: str,
        ) -> dict:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(f"refusing to overwrite raw endpoint: {path}")
    state = ft12.npt1.ensemble_train.common.checkpoint_state(model)
    state_sha = dr1.model_state_sha256(state)
    normalizer_report = ft12.normalizer_payload(normalizer)
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real",
        "iteration": UPDATES,
        "arm": arm,
        "support": arm,
        "endpoint": DIAGNOSTIC_ENDPOINT,
        "primary": False,
        "held_aoa": held_aoa,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "start": dict(start_report),
        "normalizer": dict(normalizer_report["values"]),
        "normalizer_sha256": str(normalizer_report["sha256"]),
        "sample_stream_sha256": stream_sha256,
        "ema": None,
        "optimizer_state_saved": False,
        "held_condition_opened_before_seal": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    seal = {
        "arm": arm,
        "support": arm,
        "endpoint": DIAGNOSTIC_ENDPOINT,
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "model_config_sha256": dr1.canonical_sha256(payload["model_config"]),
        "sample_stream_sha256": stream_sha256,
        "iteration": UPDATES,
        "primary": False,
        "optimizer_state_saved": False,
    }
    del state
    return seal


def load_endpoint_model(
        seal: Mapping[str, object], normalizer: smu1.FoldNormalizer,
        device: torch.device, *, arm: str, endpoint: str, held_aoa: int,
        start_sha256: str):
    path = Path(str(seal["path"]))
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError(f"sealed endpoint file changed: {arm}/{endpoint}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != ENDPOINT_SCHEMA
            or payload.get("experiment") != EXPERIMENT
            or payload.get("phase") != "real"
            or payload.get("endpoint") != endpoint
            or int(payload.get("iteration", -1)) != UPDATES
            or int(payload.get("held_aoa", -1)) != held_aoa
            or not isinstance(state, Mapping)):
        raise RuntimeError(f"endpoint payload identity changed: {arm}/{endpoint}")
    if payload.get("arm") is not None and payload.get("arm") != arm:
        raise RuntimeError(f"raw endpoint arm changed: {arm}/{endpoint}")
    if payload.get("start", {}).get("checkpoint_sha256") != start_sha256:
        raise RuntimeError("endpoint no longer starts from the shared Sim10k")
    if (payload.get("model_state_sha256") != seal["model_state_sha256"]
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError(f"endpoint tensor state changed: {arm}/{endpoint}")
    if dr1.canonical_sha256(payload["model_config"]) != seal[
            "model_config_sha256"]:
        raise RuntimeError(f"endpoint architecture changed: {arm}/{endpoint}")
    expected_normalizer = ft12.normalizer_payload(normalizer)
    if (payload.get("normalizer") != expected_normalizer["values"]
            or payload.get("normalizer_sha256")
            != expected_normalizer["sha256"]):
        raise RuntimeError(f"endpoint normalizer changed: {arm}/{endpoint}")
    config = ft12.validate_b210_model_config(
        payload["model_config"], sim_tke.b210_config("candidate"))
    model = small4k.EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    if sum(value.numel() for value in model.parameters()) != ft12.EXPECTED_PARAMETERS:
        raise RuntimeError("B210 endpoint parameter count changed")
    ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def uniform_schedule(windows: Sequence) -> dict:
    schedule = smu1.paired_batch_schedule(
        range(len(windows)), windows, updates=UPDATES, seed=SEED)
    schedule["updates"] = UPDATES
    return schedule


def train_arm(
        arm: str, *, args: argparse.Namespace,
        audit: FullSupportAccessAudit, dataset, windows: Sequence,
        normalizer: smu1.FoldNormalizer, start_sha256: str,
        deadline: float,
        ) -> dict:
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown full-support arm: {arm}")
    device = torch.device(args.device)
    ft12.npt1.ensemble_train.seed_everything(SEED)
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha256, normalizer, device,
        expected_config=sim_tke.b210_config("candidate"),
        expected_parameters=ft12.EXPECTED_PARAMETERS,
        expected_controller_parameters=ft12.EXPECTED_CONTROLLER_PARAMETERS)
    initial_state_sha = dr1.model_state_sha256(model.state_dict())
    schedule = uniform_schedule(windows)
    arm_dir = args.output_dir / arm
    arm_dir.mkdir()
    capture = EmaSealCapture()
    with endpoint_contract(args.held_aoa):
        training, ema_seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            arm_dir, start_report, capture,
            input_augmentation=AUGMENTATION,
            endpoint_filename=f"{PRIMARY_ENDPOINT}.pth",
            expected_parameters=ft12.EXPECTED_PARAMETERS,
            scheduler_t_max=SCHEDULER_T_MAX)
    if capture.value != ema_seal:
        raise RuntimeError(f"{arm} EMA endpoint was not captured")
    ema_seal = {
        **ema_seal,
        "arm": arm,
        "support": arm,
    }
    raw_seal = save_raw_endpoint(
        arm_dir / f"{DIAGNOSTIC_ENDPOINT}.pth", model, arm=arm,
        held_aoa=args.held_aoa, start_report=start_report,
        normalizer=normalizer, stream_sha256=schedule["sha256"])
    audit.seal(arm, DIAGNOSTIC_ENDPOINT, raw_seal)
    audit.seal(arm, PRIMARY_ENDPOINT, ema_seal)
    report = {
        "support": arm,
        "fit_windows": len(windows),
        "optimizer_stride": OPTIMIZER_STRIDE,
        "updates": UPDATES,
        "sample_stream_sha256": schedule["sha256"],
        "input_augmentation": dict(AUGMENTATION),
        "start": start_report,
        "initial_model_state_sha256": initial_state_sha,
        "training": training,
        "endpoints": {
            DIAGNOSTIC_ENDPOINT: {"seal": raw_seal},
            PRIMARY_ENDPOINT: {"seal": ema_seal},
        },
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def matched_recipe_projection(report: Mapping[str, object]) -> dict:
    training = report["training"]
    keys = (
        "updates", "batch_size", "seed", "optimizer", "lr",
        "weight_decay", "grad_clip", "scheduler_t_max", "ema_decay",
        "input_augmentation", "point_accuracy_weights",
        "member_accuracy_weights", "legacy_tke_gradient_floor",
        "legacy_tke_gradient_floor_on_members",
    )
    return {key: training[key] for key in keys}


def metric_comparison(
        candidate: Mapping[str, float], control: Mapping[str, float]) -> dict:
    delta = {
        key: float(candidate[key]) - float(control[key])
        for key in FINAL_WEIGHTS
    }
    point = sum(
        FINAL_WEIGHTS[key] * delta[key]
        for key in ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score"))
    non_time = point + FINAL_WEIGHTS["native_sps_score"] * delta[
        "native_sps_score"]
    return {
        "control": dict(control),
        "candidate": dict(candidate),
        "delta": delta,
        "point_final_contribution_delta": float(point),
        "non_time_final_contribution_delta": float(non_time),
        "time_held_constant_not_estimated": True,
    }


def run_fold(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    if args.held_aoa not in HELD_AOAS:
        raise ValueError(f"held AoA must be one of {HELD_AOAS}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal full-support matched training requires CUDA")
    git_identity = dr1.git_identity()
    if git_identity["branch"] != FORMAL_BRANCH:
        raise RuntimeError(f"formal run requires branch {FORMAL_BRANCH}")
    if git_identity["status_porcelain"]:
        raise RuntimeError("formal run requires a clean scientific worktree")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 runtime manifest file identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    expected = EXPECTED_FOLD_COUNTS[args.held_aoa]
    if (len(fit_rows) != expected["fit_trajectories"]
            or len(held_rows) != expected["held_trajectories"]):
        raise RuntimeError("fixed fold trajectory counts changed")
    start_sha = dr1.sha256(args.start_sim)
    if (start_sha != FORMAL_START_SHA256
            or start_sha != dr1.normalized_sha256(
                args.start_sim_sha256, "start Sim10k SHA")):
        raise RuntimeError("scratch Sim10k checkpoint SHA changed")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    audit = FullSupportAccessAudit(args.held_aoa)

    fit_trajectories, fit_metadata = load_full_rows(
        args.real_dir, fit_rows, "fit", audit)
    normalizer_windows = manifest_prefix_windows(fit_rows)
    if len(normalizer_windows) != expected["normalizer_prefix_windows"]:
        raise RuntimeError("fixed fold normalizer-window count changed")
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, normalizer_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(normalizer_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "shared_fit_normalizer.pt")

    windows_by_arm = {
        arm: support_windows(
            fit_trajectories, fit_metadata, support=arm,
            stride=OPTIMIZER_STRIDE)
        for arm in ARM_ORDER
    }
    if len(windows_by_arm["prefix"]) != expected["prefix_optimizer_windows"]:
        raise RuntimeError("fixed fold prefix-window count changed")
    if len(windows_by_arm["full"]) != expected["full_optimizer_windows"]:
        raise RuntimeError("fixed fold full-window count changed")
    datasets = {
        arm: ft12.npt1.ensemble_train.common.repro.WindowDataset(
            fit_trajectories, windows)
        for arm, windows in windows_by_arm.items()
    }
    reports = {}
    for arm in ARM_ORDER:
        reports[arm] = train_arm(
            arm, args=args, audit=audit, dataset=datasets[arm],
            windows=windows_by_arm[arm], normalizer=normalizer,
            start_sha256=start_sha, deadline=deadline)

    starts = {
        report["start"]["checkpoint_model_state_sha256"]
        for report in reports.values()
    }
    initial_states = {
        report["initial_model_state_sha256"] for report in reports.values()
    }
    recipes = [matched_recipe_projection(reports[arm]) for arm in ARM_ORDER]
    if len(starts) != 1 or len(initial_states) != 1 or recipes[0] != recipes[1]:
        raise RuntimeError("paired arms do not share start/model/recipe")
    if not audit.all_sealed:
        raise RuntimeError("held data cannot open without four endpoint seals")

    del datasets, windows_by_arm, sufficient, fit_trajectories
    gc.collect()
    held_trajectories, held_metadata = load_full_rows(
        args.real_dir, held_rows, "held", audit)
    held_windows = full_evaluation_windows(
        held_trajectories, held_metadata)
    if len(held_windows) != expected["held_full_windows"]:
        raise RuntimeError("fixed full held-AoA window count changed")
    held_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    held_indices = list(range(len(held_windows)))
    target_audit = None
    evaluation_order = (
        ("prefix", PRIMARY_ENDPOINT), ("full", PRIMARY_ENDPOINT),
        ("prefix", DIAGNOSTIC_ENDPOINT), ("full", DIAGNOSTIC_ENDPOINT),
    )
    for arm, endpoint in evaluation_order:
        seal = reports[arm]["endpoints"][endpoint]["seal"]
        model = load_endpoint_model(
            seal, normalizer, device, arm=arm, endpoint=endpoint,
            held_aoa=args.held_aoa, start_sha256=start_sha)
        audit.record_evaluation(arm, endpoint)
        metrics, metric_state, current_target_audit = smu1.evaluate_model(
            model, held_dataset, held_indices, normalizer, device, deadline)
        if target_audit is None:
            target_audit = current_target_audit
        elif target_audit != current_target_audit:
            raise RuntimeError("held target audit changed between endpoints")
        reports[arm]["endpoints"][endpoint].update({
            "metrics": metrics,
            "metric_state": metric_state,
            "held_target_audit": current_target_audit,
        })
        del model
        gc.collect()
        torch.cuda.empty_cache()

    access_report = audit.report(fit_rows, held_rows)
    if not (
            access_report["fit_allowlist_exact"]
            and access_report["held_allowlist_exact"]
            and access_report["all_seals_verified_before_held"]
            and access_report["evaluation_set_exact"]
            and not access_report["held_target_opened_before_all_seals"]
            and not access_report[
                "held_condition_in_normalizer_optimizer_teacher_or_uq"]
            and not access_report["clean716_manifest_read"]):
        raise RuntimeError("full-support access contract failed")
    comparisons = {
        PRIMARY_ENDPOINT: metric_comparison(
            reports["full"]["endpoints"][PRIMARY_ENDPOINT]["metrics"],
            reports["prefix"]["endpoints"][PRIMARY_ENDPOINT]["metrics"]),
        DIAGNOSTIC_ENDPOINT: metric_comparison(
            reports["full"]["endpoints"][DIAGNOSTIC_ENDPOINT]["metrics"],
            reports["prefix"]["endpoints"][DIAGNOSTIC_ENDPOINT]["metrics"]),
    }
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "held_aoa": args.held_aoa,
        "manifest": {
            **manifest_report,
            "file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
            "path": str(args.clean_manifest.resolve()),
        },
        "counts": {
            **expected,
            "observed_fit_trajectories": len(fit_rows),
            "observed_held_trajectories": len(held_rows),
            "observed_normalizer_prefix_windows": len(normalizer_windows),
            "observed_prefix_optimizer_windows": reports["prefix"][
                "fit_windows"],
            "observed_full_optimizer_windows": reports["full"][
                "fit_windows"],
            "observed_held_full_windows": len(held_windows),
        },
        "normalizer": {
            "kind": "shared other-AoA sparse-prefix statistics",
            "statistics_sha256": smu1.statistics_digest(statistics),
            "held_condition_in_normalizer": False,
            "identical_between_arms": True,
        },
        "start": {
            "path": str(args.start_sim.resolve()),
            "checkpoint_sha256": start_sha,
            "same_checkpoint_both_arms": True,
            "same_loaded_initial_state_both_arms": True,
            "from_scratch_sim10k": True,
        },
        "matched_recipe": recipes[0],
        "sole_method_difference": "prefix versus full released-Real support",
        "arms": reports,
        "comparisons": comparisons,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "raw_endpoint_role": "diagnostic_only_no_endpoint_selection",
        "held_ordered_key_sha256": smu1.ordered_held_key_digest(
            held_indices, held_windows, held_metadata),
        "access_audit": access_report,
        "git": git_identity,
        "evidence_class": (
            "Real-stage whole-AoA-held matched support experiment; Sim10k "
            "start may contain all released Sim conditions"),
        "score_gate": None,
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
        "decision": "COMPLETE_MATCHED_EVIDENCE_NO_SCORE_GATE",
    }
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    return result


def aggregate_results(pairs: Sequence[Sequence[str]]) -> dict:
    if len(pairs) != len(HELD_AOAS):
        raise ValueError("aggregate requires exactly held-AoA10 and held-AoA20")
    rows = []
    sources = []
    for raw_path, raw_sha in pairs:
        path = Path(raw_path)
        expected_sha = dr1.normalized_sha256(raw_sha, "fold result SHA")
        actual_sha = dr1.sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"fold result SHA changed: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("schema") != SCHEMA
                or row.get("experiment") != EXPERIMENT
                or row.get("contract_sha256") != CONTRACT_SHA256):
            raise ValueError(f"fold result contract changed: {path}")
        rows.append(row)
        sources.append({"path": str(path.resolve()), "sha256": actual_sha})
    rows.sort(key=lambda row: HELD_AOAS.index(int(row["held_aoa"])))
    held = tuple(int(row["held_aoa"]) for row in rows)
    if held != HELD_AOAS:
        raise ValueError("aggregate fold membership changed")
    start_shas = {row["start"]["checkpoint_sha256"] for row in rows}
    manifest_shas = {row["manifest"]["file_sha256"] for row in rows}
    if len(start_shas) != 1 or manifest_shas != {
            ft12.FORMAL_MANIFEST_FILE_SHA256}:
        raise RuntimeError("folds do not share the same Sim start/manifest")

    endpoints = {}
    for endpoint in ENDPOINT_ORDER:
        arms = {}
        for arm in ARM_ORDER:
            state = smu1.merge_metric_states([
                row["arms"][arm]["endpoints"][endpoint]["metric_state"]
                for row in rows
            ])
            arms[arm] = {
                "metric_state": state,
                "metrics": smu1.finalize_metric_state(state),
            }
        endpoints[endpoint] = {
            "arms": arms,
            "comparison": metric_comparison(
                arms["full"]["metrics"], arms["prefix"]["metrics"]),
        }
    return {
        "schema": AGGREGATE_SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "sources": sources,
        "held_aoas": list(held),
        "start_checkpoint_sha256": next(iter(start_shas)),
        "held_full_windows": sum(
            int(row["counts"]["observed_held_full_windows"])
            for row in rows),
        "endpoints": endpoints,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_comparison": endpoints[PRIMARY_ENDPOINT]["comparison"],
        "raw_endpoint_role": "diagnostic_only_no_endpoint_selection",
        "score_gate": None,
        "decision": "COMPLETE_TWO_FOLD_MATCHED_EVIDENCE_NO_SCORE_GATE",
        "submission_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract-only", action="store_true")
    mode.add_argument("--fold-worker", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--held-aoa", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--wall-budget-seconds", type=float,
        default=DEFAULT_WALL_BUDGET_SECONDS)
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    parser.add_argument(
        "--fold-result", nargs=2, action="append", metavar=("PATH", "SHA256"))
    parser.add_argument("--aggregate-output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    fold_values = (
        args.held_aoa, args.output_dir, args.real_dir, args.clean_manifest,
        args.start_sim, args.start_sim_sha256)
    if args.contract_only:
        if (any(value is not None for value in fold_values)
                or args.fold_result or args.aggregate_output
                or args.execute_formal_gpu or args.acknowledge_protocol):
            raise ValueError("contract-only mode received execution arguments")
        return
    if args.fold_worker:
        if any(value is None for value in fold_values):
            raise ValueError("fold worker is missing required paths/identity")
        if args.held_aoa not in HELD_AOAS:
            raise ValueError(f"held AoA must be one of {HELD_AOAS}")
        if args.fold_result or args.aggregate_output:
            raise ValueError("fold worker received aggregate arguments")
        if (not args.execute_formal_gpu
                or args.acknowledge_protocol != ACKNOWLEDGEMENT):
            raise ValueError("fold worker requires the exact acknowledgement")
        if not args.real_dir.is_dir() or not args.clean_manifest.is_file():
            raise FileNotFoundError("released Real directory/manifest is missing")
        if not args.start_sim.is_file():
            raise FileNotFoundError(args.start_sim)
        args.start_sim_sha256 = dr1.normalized_sha256(
            args.start_sim_sha256, "start Sim10k SHA")
        if args.start_sim_sha256 != FORMAL_START_SHA256:
            raise ValueError("fold worker requires the frozen pure-Sim start")
        return
    if (any(value is not None for value in fold_values)
            or args.execute_formal_gpu or args.acknowledge_protocol):
        raise ValueError("aggregate-only mode received fold execution arguments")
    if not args.fold_result or args.aggregate_output is None:
        raise ValueError("aggregate requires fold results and output")
    if args.aggregate_output.exists():
        raise FileExistsError(args.aggregate_output)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        print(json.dumps({
            "experiment": EXPERIMENT,
            "contract_sha256": CONTRACT_SHA256,
            "contract": CONTRACT,
            "expected_fold_counts": EXPECTED_FOLD_COUNTS,
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.aggregate_only:
        result = aggregate_results(args.fold_result)
        dr1.write_json_atomic(args.aggregate_output, result)
        print(json.dumps({
            "event": "b210_full_released_real_support_aggregate_complete",
            "primary_comparison": result["primary_comparison"],
            "output": str(args.aggregate_output),
        }), flush=True)
        return
    result = run_fold(args)
    print(json.dumps({
        "event": "b210_full_released_real_support_fold_complete",
        "held_aoa": result["held_aoa"],
        "primary_comparison": result["comparisons"][PRIMARY_ENDPOINT],
        "output": str(args.output_dir / "fold_result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
