#!/usr/bin/env python3
"""Candidate-specific clean716 UQ OOF/refit stage for B210 small4k.

The input is the future full-clean2544 EMA-u4000 point artifact.  Point,
backbone, controller, and member-head state is frozen bitwise and kept in eval
mode.  Only the native UQ modules train on clean temporal-tail residuals.

Five whole-condition folds use the historical
``(rank(Re) + 2*rank(AoA)) % 5`` assignment and a fixed u800 endpoint.  Held
labels cannot select an endpoint.  Only a passing pooled sufficient-state gate
authorizes the fixed all716 u800 UQ-only refit.  This stage never builds a ZIP
and never authorizes submission.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import json
import math
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import fp32_ema_shadow as ema_lib
import train_b210_np_functional_real_ft12_loao as ft12
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_npt1_member_sync_tke_loao as npt1
import train_c208_smu1_paired_loao as smu1
import train_ensemble_unet2d as ensemble_train
import train_ensemble_uq_oof as uq_train
from models.ensemble_unet2d import EnsembleUNet2d
from train_uq_budget_reallocator import trajectory_folds


EXPERIMENT = "b210_small4k_candidate_clean716_uq_release"
SCHEMA = "b210_small4k_clean716_uq_run_v1"
FOLD_ENDPOINT_SCHEMA = "b210_small4k_clean716_uq_fold_endpoint_v1"
REFIT_ENDPOINT_SCHEMA = "b210_small4k_clean716_uq_refit_endpoint_v1"
INPUT_ENDPOINT_SCHEMA = "b210_np_functional_real_small4k_full_refit_endpoint_v1"
INPUT_EXPERIMENT = "b210_np_functional_real_small4k_full_clean_point"
INPUT_STATUS = "POINT_STAGE_ONLY_NOT_SUBMITTABLE"
REFIT_STAGE = "UQ_REFIT_RELEASE_STAGE"
FORMAL_BRANCH = "codex/b210-small4k-uq-release"
FORMAL_ACKNOWLEDGEMENT = "B210_SMALL4K_CLEAN716_UQ_RELEASE"

FOLDS = 5
FOLD_UPDATES = 800
REFIT_UPDATES = 800
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
SEED = 12
LR = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SCHEDULER_T_MAX = 800
LOG_INTERVAL = 100
WINDOW_STRIDE = 20
VAL_FRACTION = 0.2
EXPECTED_TAIL_WINDOWS = 716
EXPECTED_TRAJECTORIES = dr1.EXPECTED_TRAJECTORIES
EXPECTED_PARAMETERS = ft12.EXPECTED_PARAMETERS
EXPECTED_CONTROLLER_PARAMETERS = ft12.EXPECTED_CONTROLLER_PARAMETERS
FORMAL_MANIFEST_FILE_SHA256 = npt1.FORMAL_MANIFEST_FILE_SHA256
FORMAL_MANIFEST_SEMANTIC_SHA256 = dr1.FORMAL_MANIFEST_SHA256
EMA_DECAY = 0.999
WALL_BUDGET_SECONDS = 120 * 60
MIN_FREE_BYTES = 5 * 1024**3

# The exact historical sharp2 recipe used by the formal C208 oof800/refit800
# artifact.  The B210 candidate already contains temporal/refinement modules;
# this stage does not expand or replace its architecture.
PINBALL_WEIGHT = 0.05
SOFT_SPS_WEIGHT = 2.0
SOFT_TAU_FRACTION = 0.035
CVAR_WEIGHT = 0.35
CVAR_FRACTION = 0.25
INTERVAL_SCORE_WEIGHT = 0.0025
COVERAGE_TARGET = 0.85

# Reuse the established RUS1 promotion gate rather than inventing a new gate.
POOLED_GAIN_GATE = 0.10
GROUP_POSITIVE_GATE = 4
FOLD_WORST_GATE = -0.03
AOA_WORST_GATE = -0.05

EVIDENCE_BINDING_KEYS = (
    "sim10k_sha256",
    "held10_discovery_sha256",
    "held0_confirmation_sha256",
    "latency_summary_sha256",
)
EXPECTED_EVIDENCE_BINDINGS = {
    "sim10k_sha256": (
        "f86faac5c376d2ee52ceb9209e407e2a42c114d79753dfd5fe2a51072207d85a"
    ),
    "held10_discovery_sha256": (
        "4a5d533f5869618dde72d456ccc4c175a179b8218a7b4c6e5b652077e818736a"
    ),
    "held0_confirmation_sha256": (
        "f469d930e2f47fcdb9d29c7ee394084276ee91ec707ff953749aa4a3136c1f95"
    ),
    "latency_summary_sha256": (
        "14153465006b0ff2a70677a400f2ebfd07386978c75e7835848feda7b3a3cf0e"
    ),
}

UQ_PREFIXES = uq_train.UQ_PREFIXES

CONTRACT = {
    "hypothesis": (
        "Candidate-specific clean716 residual-only UQ adaptation can improve "
        "the strict B210 small4k native interval utility without changing any "
        "point, backbone, controller, or member-head state."
    ),
    "input": {
        "schema": INPUT_ENDPOINT_SCHEMA,
        "experiment": INPUT_EXPERIMENT,
        "phase": "real",
        "iteration": 4000,
        "endpoint": "ema_u4000",
        "primary": True,
        "training_scope": "full_clean2544",
        "fit_samples": dr1.EXPECTED_WINDOWS,
        "status": INPUT_STATUS,
        "evidence_bindings": EXPECTED_EVIDENCE_BINDINGS,
        "expected_sha256": "required and checked before deserialization",
        "strict_b210": {
            "parameters": EXPECTED_PARAMETERS,
            "controller_parameters": EXPECTED_CONTROLLER_PARAMETERS,
            "base": 210,
            "backbone_residual_dropout": 0.20,
            "member_sync_controller": True,
        },
    },
    "data": {
        "manifest_file_sha256": FORMAL_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": FORMAL_MANIFEST_SEMANTIC_SHA256,
        "trajectories": EXPECTED_TRAJECTORIES,
        "tail_windows": EXPECTED_TAIL_WINDOWS,
        "slice": "u/v[val_boundary:,::2,::2]",
        "prefix_frames_read": 0,
        "excluded": dr1.EXCLUDED_BASENAME,
    },
    "oof": {
        "folds": FOLDS,
        "fold_definition": (
            "whole Re/AoA condition isolation: (rank(Re)+2*rank(AoA)) % 5; not AoA-LOAO"
        ),
        "updates_per_fold": FOLD_UPDATES,
        "endpoint": "fixed_terminal_u800",
        "held_checkpoint_selection": False,
        "aggregation": "additive sufficient metric state; never fold mean",
        "baseline": "same point artifact native UQ at update0",
    },
    "training": {
        "trainable": list(UQ_PREFIXES),
        "frozen_eval": "point/backbone/controller/member heads",
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": SCHEDULER_T_MAX,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "pinball_weight": PINBALL_WEIGHT,
        "soft_sps_weight": SOFT_SPS_WEIGHT,
        "soft_tau_fraction": SOFT_TAU_FRACTION,
        "cvar_weight": CVAR_WEIGHT,
        "cvar_fraction": CVAR_FRACTION,
        "interval_score_weight": INTERVAL_SCORE_WEIGHT,
        "coverage_target": COVERAGE_TARGET,
        "teacher": False,
        "hgrad": False,
        "rehearsal": False,
        "point_gain_attribution": False,
    },
    "promotion_gate": {
        "pooled_sps_delta_min": POOLED_GAIN_GATE,
        "positive_folds_min": GROUP_POSITIVE_GATE,
        "worst_fold_delta_min": FOLD_WORST_GATE,
        "positive_aoas_min": GROUP_POSITIVE_GATE,
        "worst_aoa_delta_min": AOA_WORST_GATE,
    },
    "refit": {
        "authorized_only_after_oof_gate": True,
        "windows": EXPECTED_TAIL_WINDOWS,
        "updates": REFIT_UPDATES,
        "fresh_from_same_point_artifact": True,
        "stage": REFIT_STAGE,
        "exact_verifier_required": True,
    },
    "hard_stop": [
        "git_or_input_sha_or_schema",
        "manifest_or_bad_file",
        "cuda_or_hardware",
        "disk_or_oom",
        "nonfinite_or_zero_gradient",
        "frozen_state_drift",
        "held_endpoint_selection_or_metric_state",
        "wall_budget_120min",
    ],
    "zip_written": False,
    "submission_authorized": False,
}
CONTRACT_SHA256 = dr1.canonical_sha256(CONTRACT)


class FoldEndpointAudit:
    """Require a hashed fixed terminal before any held scoring."""

    def __init__(self) -> None:
        self.seals: dict[int, dict] = {}
        self.evaluations: list[dict] = []

    def seal(self, fold: int, seal: Mapping[str, object]) -> None:
        if fold in self.seals:
            raise RuntimeError(f"fold {fold} terminal was sealed twice")
        if (
            seal.get("endpoint") != "fixed_terminal_u800"
            or int(seal.get("iteration", -1)) != FOLD_UPDATES
        ):
            raise RuntimeError("fold seal is not the fixed u800 terminal")
        self.seals[fold] = dict(seal)

    def before_evaluate(self, fold: int, role: str) -> None:
        if fold not in self.seals:
            raise RuntimeError("held scoring cannot precede terminal seal")
        if role not in {"native_baseline", "fixed_terminal_candidate"}:
            raise ValueError(f"unknown held evaluation role {role}")
        self.evaluations.append(
            {
                "fold": fold,
                "role": role,
                "terminal_sealed_before_held_score": True,
            }
        )

    def report(self) -> dict:
        expected = [
            {"fold": fold, "role": role, "terminal_sealed_before_held_score": True}
            for fold in range(FOLDS)
            for role in ("native_baseline", "fixed_terminal_candidate")
        ]
        return {
            "terminal_seals": self.seals,
            "evaluations": self.evaluations,
            "fixed_terminal_only": self.evaluations == expected,
            "held_checkpoint_selection": False,
        }


class TailAccessAudit:
    """Record the clean tail-only H5 slice contract."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.opened: set[str] = set()

    def before_open(self, row: Mapping[str, object]) -> None:
        name = str(row["file"])
        if name == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded H5 reached clean716 opener")
        if name in self.opened:
            raise RuntimeError(f"tail H5 opened twice: {name}")

    def record_open(self, row: Mapping[str, object]) -> None:
        name = str(row["file"])
        boundary = int(row["val_boundary"])
        self.opened.add(name)
        self.events.append(
            {
                "file": name,
                "slice": "u/v[val_boundary:,::2,::2]",
                "first_frame_read": boundary,
                "prefix_frames_read": 0,
            }
        )

    def report(self, expected: Sequence[str]) -> dict:
        allowlist = sorted(expected)
        opened = sorted(self.opened)
        return {
            "opened_file_allowlist": opened,
            "expected_file_allowlist": allowlist,
            "allowlist_exact": opened == allowlist,
            "prefix_frames_read": 0,
            "tail_arrays_read": len(self.events),
            "excluded_opened": dr1.EXCLUDED_BASENAME in self.opened,
            "events": self.events,
        }


def sharp2_args() -> argparse.Namespace:
    """Return only the frozen fields consumed by ``uq_objective``."""
    return argparse.Namespace(
        pinball_weight=PINBALL_WEIGHT,
        soft_sps_weight=SOFT_SPS_WEIGHT,
        soft_tau_fraction=SOFT_TAU_FRACTION,
        cvar_weight=CVAR_WEIGHT,
        cvar_fraction=CVAR_FRACTION,
        interval_score_weight=INTERVAL_SCORE_WEIGHT,
        coverage_target=COVERAGE_TARGET,
        extent_distill_weight=0.0,
        coverage_bce_weight=0.0,
    )


def validate_flat_normalizer(
    payload: Mapping[str, object],
) -> tuple[tuple[torch.Tensor, ...], dict]:
    raw = payload.get("normalizer")
    keys = ("mean_input", "mean_target", "std_input", "std_target")
    if not isinstance(raw, Mapping) or set(raw) != set(keys):
        raise ValueError("point artifact normalizer is not the flat schema")
    values = {key: raw[key] for key in keys}
    expected = dr1.normalized_sha256(
        str(payload.get("normalizer_sha256")), "normalizer SHA"
    )
    actual = dr1.canonical_sha256(values)
    if actual != expected:
        raise RuntimeError("point artifact normalizer SHA changed")
    tensors = tuple(torch.as_tensor(values[key], dtype=torch.float32) for key in keys)
    if any(
        value.shape != (3,) or not bool(torch.isfinite(value).all())
        for value in tensors
    ):
        raise ValueError("point artifact normalizer values are invalid")
    if bool((tensors[2] <= 0.0).any()) or bool((tensors[3] <= 0.0).any()):
        raise ValueError("point artifact normalizer std must be positive")
    return tensors, {"values": values, "sha256": actual, "flat": True}


def _validate_evidence_bindings(payload: Mapping[str, object]) -> dict:
    raw = payload.get("evidence_bindings")
    if not isinstance(raw, Mapping) or set(raw) != set(EVIDENCE_BINDING_KEYS):
        raise ValueError("point artifact evidence bindings changed")
    result = {
        key: dr1.normalized_sha256(str(raw[key]), key) for key in EVIDENCE_BINDING_KEYS
    }
    if result != EXPECTED_EVIDENCE_BINDINGS:
        raise RuntimeError("point artifact evidence SHA bindings changed")
    start = payload.get("start")
    if (
        not isinstance(start, Mapping)
        or dr1.normalized_sha256(str(start.get("checkpoint_sha256")), "start Sim SHA")
        != result["sim10k_sha256"]
    ):
        raise RuntimeError("point artifact start is not bound to Sim10k")
    return result


def load_point_artifact(
    path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[dict, smu1.FoldNormalizer, dict]:
    """SHA-check, validate, and strict-load the future full-clean endpoint."""
    expected = dr1.normalized_sha256(expected_sha256, "point checkpoint SHA")
    actual = dr1.sha256(path)
    if actual != expected:
        raise RuntimeError("B210 small4k point checkpoint SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("B210 point artifact must be a mapping")
    fixed = {
        "schema": INPUT_ENDPOINT_SCHEMA,
        "experiment": INPUT_EXPERIMENT,
        "phase": "real",
        "iteration": 4000,
        "endpoint": "ema_u4000",
        "primary": True,
        "training_scope": "full_clean2544",
        "fit_samples": dr1.EXPECTED_WINDOWS,
        "status": INPUT_STATUS,
        "manifest_file_sha256": FORMAL_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": FORMAL_MANIFEST_SEMANTIC_SHA256,
        "manifest_windows": dr1.EXPECTED_WINDOWS,
        "manifest_excluded": dr1.EXCLUDED_BASENAME,
        "optimizer_state_saved": False,
        "held_or_tail_labels_read": False,
        "tail_arrays_read": 0,
        "endpoint_selection_allowed": False,
        "candidate_specific_oof_uq_refit_required": True,
        "release_checkpoint_written": False,
        "submission_authorized": False,
    }
    changed = [key for key, value in fixed.items() if payload.get(key) != value]
    if changed:
        raise ValueError(f"B210 point artifact contract changed: {changed}")
    stream_sha = dr1.normalized_sha256(
        str(payload.get("sample_stream_sha256")), "sample stream SHA"
    )
    bindings = _validate_evidence_bindings(payload)
    dr1.normalized_sha256(str(payload.get("authorization_sha256")), "authorization SHA")
    ema = payload.get("ema")
    if (
        not isinstance(ema, Mapping)
        or ema.get("schema") != ema_lib.EMA_SEAL_SCHEMA
        or ema.get("update_order") != ema_lib.UPDATE_ORDER_CONTRACT
        or int(ema.get("updates", -1)) != 4000
        or float(ema.get("decay", -1.0)) != EMA_DECAY
        or ema.get("shadow_dtype") != "torch.float32"
    ):
        raise ValueError("B210 point artifact EMA contract changed")

    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("B210 point artifact lacks model config/state")
    parsed = ft12.validate_b210_model_config(config)
    state_sha = dr1.model_state_sha256(state)
    frozen_state_sha = dr1.model_state_sha256(
        {key: value for key, value in state.items() if not key.startswith(UQ_PREFIXES)}
    )
    if state_sha != dr1.normalized_sha256(
        str(payload.get("model_state_sha256")), "model state SHA"
    ):
        raise RuntimeError("B210 point model-state SHA changed")
    if (
        dr1.normalized_sha256(
            str(ema.get("materialized_model_state_sha256")),
            "EMA materialized state SHA",
        )
        != state_sha
    ):
        raise RuntimeError("B210 point EMA does not bind the model state")
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ):
        raise TypeError("B210 point floating state must be FP32")

    statistics, normalizer_report = validate_flat_normalizer(payload)
    normalizer = smu1.FoldNormalizer(statistics, device)
    model = EnsembleUNet2d(parsed).float()
    model.load_state_dict(state, strict=True)
    total = sum(value.numel() for value in model.parameters())
    controller = sum(value.numel() for value in model.energy_controller.parameters())
    if total != EXPECTED_PARAMETERS or controller != EXPECTED_CONTROLLER_PARAMETERS:
        raise RuntimeError(
            f"B210 point parameter contract changed: {total}/{controller}"
        )
    if any(value.dtype != torch.float32 for value in model.parameters()):
        raise TypeError("B210 point model parameters must be FP32")
    ensemble_train.common.set_model_stats(model, normalizer)
    model.load_state_dict(model.state_dict(), strict=True)
    del model
    gc.collect()
    return (
        payload,
        normalizer,
        {
            "path": str(path.resolve()),
            "checkpoint_sha256": actual,
            "model_state_sha256": state_sha,
            "frozen_state_sha256": frozen_state_sha,
            "model_config_sha256": dr1.canonical_sha256(config),
            "normalizer": normalizer_report,
            "sample_stream_sha256": stream_sha,
            "evidence_bindings": bindings,
            "strict_load": True,
            "parameters": total,
            "controller_parameters": controller,
            **fixed,
        },
    )


def load_point_model(
    payload: Mapping[str, object], normalizer: smu1.FoldNormalizer, device: torch.device
) -> EnsembleUNet2d:
    parsed = ft12.validate_b210_model_config(payload["model_config"])
    model = EnsembleUNet2d(parsed).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def derive_clean716_manifest(
    rows: Sequence[Mapping[str, object]],
    expected_tail_count: int = EXPECTED_TAIL_WINDOWS,
) -> tuple[list[dict], dict]:
    if len(rows) != EXPECTED_TRAJECTORIES:
        raise RuntimeError("clean716 requires exactly 81 trajectories")
    result = []
    for row in rows:
        name = str(row["file"])
        if name == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded trajectory entered clean716")
        frames = int(row["frames"])
        starts = list(range(0, frames - 40 + 1, WINDOW_STRIDE))
        if len(starts) < 2:
            raise ValueError(f"tail trajectory is too short: {name}")
        count = max(1, math.ceil(len(starts) * VAL_FRACTION))
        tail_starts = starts[-count:]
        if tail_starts[0] != int(row["val_boundary"]):
            raise RuntimeError(f"clean716 boundary changed for {name}")
        result.append(
            {
                **dict(row),
                "val_starts": tail_starts,
            }
        )
    total = sum(len(row["val_starts"]) for row in result)
    if total != expected_tail_count:
        raise RuntimeError(
            f"clean716 tail count changed: {total} != {expected_tail_count}"
        )
    return result, {
        "trajectories": len(result),
        "tail_windows": total,
        "excluded": dr1.EXCLUDED_BASENAME,
        "semantic_sha256": dr1.canonical_sha256(result),
    }


def load_tail_rows(
    real_dir: Path,
    rows: Sequence[Mapping[str, object]],
    audit: TailAccessAudit,
) -> tuple[list[torch.Tensor], list[dict], list]:
    trajectories: list[torch.Tensor] = []
    metadata: list[dict] = []
    windows = []
    root = real_dir.resolve()
    for row in rows:
        audit.before_open(row)
        name = str(row["file"])
        path = (root / name).resolve()
        if path.parent != root or path.name != name:
            raise ValueError(f"unsafe clean716 H5 path {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        boundary = int(row["val_boundary"])
        frames = int(row["frames"])
        with h5py.File(path, "r") as handle:
            u_source, v_source = handle["u"], handle["v"]
            if u_source.shape != v_source.shape:
                raise ValueError(f"u/v shapes differ in {name}")
            if int(u_source.shape[0]) != frames:
                raise ValueError(f"frame count changed in {name}")
            if int(np.asarray(handle["re"])) != int(row["re"]):
                raise ValueError(f"Re changed in {name}")
            if int(np.asarray(handle["aoa"])) != int(row["aoa"]):
                raise ValueError(f"AoA changed in {name}")
            u = np.asarray(u_source[boundary:, ::2, ::2], dtype=np.float32)
            v = np.asarray(v_source[boundary:, ::2, ::2], dtype=np.float32)
        if u.shape != v.shape or int(u.shape[0]) != frames - boundary:
            raise ValueError(f"tail slice shape changed in {name}")
        audit.record_open(row)
        trajectory = len(trajectories)
        trajectories.append(
            torch.from_numpy(np.ascontiguousarray(np.stack((u, v), axis=-1)))
        )
        metadata.append(
            {
                "file": name,
                "frames": frames - boundary,
                "source_frames": frames,
                "re": int(row["re"]),
                "aoa": int(row["aoa"]),
                "val_boundary": boundary,
            }
        )
        for start in row["val_starts"]:
            relative = int(start) - boundary
            if relative < 0 or relative + 40 > frames - boundary:
                raise RuntimeError(f"invalid clean716 tail start in {name}")
            windows.append(ensemble_train.common.repro.Window(trajectory, relative))
    return trajectories, metadata, windows


def uq_modules(model: EnsembleUNet2d) -> tuple[torch.nn.Module, ...]:
    modules: list[torch.nn.Module] = [
        model.uq_projections,
        model.uq_fusion,
        model.quantile_head,
    ]
    for value in (
        model.utility_refinement_head,
        model.temporal_uq_refiner,
        model.coverage_calibration_head,
        getattr(model, "residual_mixture_head", None),
        getattr(model, "output_space_uq_allocator", None),
        getattr(model, "output_space_uq_budgeter", None),
    ):
        if value is not None:
            modules.append(value)
    return tuple(modules)


def configure_uq_only(model: EnsembleUNet2d) -> tuple[torch.nn.Parameter, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in uq_modules(model):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any(not name.startswith(UQ_PREFIXES) for name in names):
        raise RuntimeError("B210 UQ-only trainability contract changed")
    if model.energy_controller is None or any(
        value.requires_grad for value in model.energy_controller.parameters()
    ):
        raise RuntimeError("B210 controller was not frozen")
    return trainable


def set_uq_training_mode(model: EnsembleUNet2d) -> None:
    """Keep B210 backbone dropout disabled while UQ modules train."""
    model.eval()
    for module in uq_modules(model):
        module.train()
    if model.training:
        raise RuntimeError("B210 point model left eval mode during UQ training")
    if model.energy_controller is None or model.energy_controller.training:
        raise RuntimeError("B210 controller left eval mode during UQ training")


def partition_state(model: EnsembleUNet2d, *, uq: bool) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith(UQ_PREFIXES) == uq
    }


def train_uq_terminal(
    model: EnsembleUNet2d,
    dataset,
    indices: Sequence[int],
    normalizer: smu1.FoldNormalizer,
    device: torch.device,
    deadline: float,
    seed: int,
    label: str,
    objective_args: argparse.Namespace | None = None,
    extent_teachers: tuple[torch.nn.Module, torch.nn.Module] | None = None,
    extent_teacher_weight_b: float = 0.0,
) -> dict:
    if not indices:
        raise ValueError("UQ optimizer partition is empty")
    uq_train.seed_everything(seed)
    parameters = configure_uq_only(model)
    frozen_before = dr1.model_state_sha256(partition_state(model, uq=False))
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    iterator = ensemble_train.common.repro.cycling(loader)
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX
    )
    if objective_args is None:
        objective_args = sharp2_args()
    if not 0.0 <= extent_teacher_weight_b <= 1.0:
        raise ValueError("extent teacher weight must lie in [0, 1]")
    if (float(objective_args.extent_distill_weight) > 0.0) != (
        extent_teachers is not None
    ):
        raise ValueError(
            "positive extent distillation and teacher models must be supplied together"
        )
    history = []
    sums: dict[str, float] = {}
    count = 0
    started = time.perf_counter()
    gradient_preflight = None
    completed = 0
    for iteration in range(1, FOLD_UPDATES + 1):
        if time.perf_counter() > deadline:
            raise TimeoutError("B210 clean716 UQ wall budget expired")
        set_uq_training_mode(model)
        history_batch, target = next(iterator)
        history_batch = history_batch.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history_batch, target)
        optimizer.zero_grad(set_to_none=True)
        prediction, quantiles, extents, logits = uq_train.quantile_forward(
            model, history_norm
        )
        teacher_extents = None
        if extent_teachers is not None:
            teacher_extents = uq_train.blended_teacher_extents(
                history_norm, extent_teachers, extent_teacher_weight_b
            )
        loss, components = uq_train.uq_objective(
            objective_args,
            model,
            prediction,
            quantiles,
            target_norm,
            normalizer,
            teacher_extents=teacher_extents,
            interval_extents=extents,
            coverage_logits=logits,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite B210 UQ loss at {label} u{iteration}")
        loss.backward()
        gradients = [value.grad for value in parameters if value.grad is not None]
        if not gradients or any(
            not bool(torch.isfinite(value).all()) for value in gradients
        ):
            raise FloatingPointError(
                f"invalid B210 UQ gradient at {label} u{iteration}"
            )
        if iteration == 1:
            squared = sum(
                float(value.detach().float().square().sum()) for value in gradients
            )
            if not math.isfinite(squared) or squared <= 0.0:
                raise RuntimeError("B210 UQ gradient preflight is zero")
            gradient_preflight = {
                "iteration": 1,
                "nonzero": True,
                "finite": True,
                "l2_norm": math.sqrt(squared),
            }
        clipped = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
        if not bool(torch.isfinite(clipped)):
            raise FloatingPointError("B210 UQ clipped gradient is non-finite")
        optimizer.step()
        scheduler.step()
        completed = iteration
        for key, value in components.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration % LOG_INTERVAL == 0:
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count for key, value in sums.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(
                json.dumps(
                    {
                        "event": "b210_clean716_uq_progress",
                        "label": label,
                        **row,
                    }
                ),
                flush=True,
            )
            sums.clear()
            count = 0
    if completed != FOLD_UPDATES:
        raise RuntimeError("B210 UQ did not reach the fixed u800 terminal")
    frozen_after = dr1.model_state_sha256(partition_state(model, uq=False))
    if frozen_after != frozen_before:
        raise RuntimeError("B210 point/backbone/controller/member state drifted")
    model.eval()
    return {
        "updates": completed,
        "batch_size": BATCH_SIZE,
        "seed": seed,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler_t_max": SCHEDULER_T_MAX,
        "objective": {
            key: value
            for key, value in vars(objective_args).items()
            if isinstance(value, (bool, float, int, str)) or value is None
        },
        "extent_teacher_active": extent_teachers is not None,
        "extent_teacher_weight_b": extent_teacher_weight_b,
        "gradient_preflight": gradient_preflight,
        "trainable_parameters": sum(value.numel() for value in parameters),
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "frozen_state_bitwise": True,
        "history": history,
        "wall_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_grouped(
    model: EnsembleUNet2d,
    dataset,
    indices: Sequence[int],
    metadata: Sequence[Mapping[str, object]],
    normalizer: smu1.FoldNormalizer,
    device: torch.device,
    deadline: float,
) -> tuple[dict, dict[int, dict]]:
    if not indices:
        raise ValueError("held UQ partition is empty")
    tagged = ensemble_train.TrajectoryTaggedDataset(dataset)
    batches = [
        list(indices[start : start + EVAL_BATCH_SIZE])
        for start in range(0, len(indices), EVAL_BATCH_SIZE)
    ]
    loader = DataLoader(
        tagged, batch_sampler=batches, num_workers=0, pin_memory=device.type == "cuda"
    )
    pooled = smu1.new_metric_state()
    by_aoa = {aoa: smu1.new_metric_state() for aoa in dr1.AOA_VALUES}
    model.eval()
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for history, target, trajectory_ids, _sample_indices in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("B210 UQ held scoring exceeded wall budget")
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, _target_norm = normalizer.preprocess(history, target)
        prediction_norm, lower_norm, upper_norm = model.forward_distribution(
            history_norm
        )
        prediction = normalizer.postprocess_prediction(prediction_norm.float())
        lower = lower_norm.float() * scale
        upper = upper_norm.float() * scale
        truth = target.float()
        smu1.update_metric_state(pooled, prediction, truth, lower, upper)
        labels = [int(metadata[int(value)]["aoa"]) for value in trajectory_ids.tolist()]
        for aoa in sorted(set(labels)):
            mask = torch.as_tensor([value == aoa for value in labels], device=device)
            smu1.update_metric_state(
                by_aoa[aoa], prediction[mask], truth[mask], lower[mask], upper[mask]
            )
    if int(pooled["samples"]) != len(indices):
        raise RuntimeError("B210 held scorer did not consume its partition")
    nonempty = {
        key: value for key, value in by_aoa.items() if int(value["samples"]) > 0
    }
    return pooled, nonempty


def point_state_equal(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> bool:
    keys = ("samples", "rel_error_sum", "tke_error_sum", "mvpe_error_sum")
    return all(baseline[key] == candidate[key] for key in keys)


def save_fold_endpoint(
    path: Path,
    model: EnsembleUNet2d,
    fold: int,
    point_sha256: str,
    training: Mapping[str, object],
) -> dict:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    state = {
        key: value.detach().cpu().clone()
        for key, value in partition_state(model, uq=True).items()
    }
    state_sha = dr1.model_state_sha256(state)
    payload = {
        "schema": FOLD_ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "fold": fold,
        "iteration": FOLD_UPDATES,
        "endpoint": "fixed_terminal_u800",
        "held_checkpoint_selection": False,
        "point_checkpoint_sha256": point_sha256,
        "uq_state_dict": state,
        "uq_state_sha256": state_sha,
        "training": dict(training),
        "point_gain_attribution": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "uq_state_sha256": state_sha,
        "fold": fold,
        "iteration": FOLD_UPDATES,
        "endpoint": "fixed_terminal_u800",
    }


def _delta_row(
    label: int, baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict:
    before = smu1.finalize_metric_state(dict(baseline))
    after = smu1.finalize_metric_state(dict(candidate))
    return {
        "group": label,
        "baseline": before,
        "candidate": after,
        "sps_delta": float(after["native_sps_score"] - before["native_sps_score"]),
    }


def promotion_gate(
    baseline_folds: Sequence[Mapping[str, object]],
    candidate_folds: Sequence[Mapping[str, object]],
    baseline_aoas: Mapping[int, Mapping[str, object]],
    candidate_aoas: Mapping[int, Mapping[str, object]],
) -> dict:
    if len(baseline_folds) != FOLDS or len(candidate_folds) != FOLDS:
        raise ValueError("promotion requires all five OOF folds")
    if set(baseline_aoas) != set(dr1.AOA_VALUES) or set(candidate_aoas) != set(
        dr1.AOA_VALUES
    ):
        raise ValueError("promotion requires all five AoA groups")
    baseline_state = smu1.merge_metric_states([dict(value) for value in baseline_folds])
    candidate_state = smu1.merge_metric_states(
        [dict(value) for value in candidate_folds]
    )
    baseline_metrics = smu1.finalize_metric_state(baseline_state)
    candidate_metrics = smu1.finalize_metric_state(candidate_state)
    pooled_delta = float(
        candidate_metrics["native_sps_score"] - baseline_metrics["native_sps_score"]
    )
    fold_rows = [
        _delta_row(index, baseline_folds[index], candidate_folds[index])
        for index in range(FOLDS)
    ]
    aoa_rows = [
        _delta_row(aoa, baseline_aoas[aoa], candidate_aoas[aoa])
        for aoa in dr1.AOA_VALUES
    ]
    fold_deltas = [row["sps_delta"] for row in fold_rows]
    aoa_deltas = [row["sps_delta"] for row in aoa_rows]
    conditions = {
        "pooled_sps_gain_ge_0_10": pooled_delta >= POOLED_GAIN_GATE,
        "positive_folds_ge_4_of_5": (
            sum(value > 0.0 for value in fold_deltas) >= GROUP_POSITIVE_GATE
        ),
        "worst_fold_ge_minus_0_03": min(fold_deltas) >= FOLD_WORST_GATE,
        "positive_aoas_ge_4_of_5": (
            sum(value > 0.0 for value in aoa_deltas) >= GROUP_POSITIVE_GATE
        ),
        "worst_aoa_ge_minus_0_05": min(aoa_deltas) >= AOA_WORST_GATE,
    }
    return {
        "pass": all(conditions.values()),
        "conditions": conditions,
        "pooled_sps_delta": pooled_delta,
        "baseline_oof": {
            "metrics": baseline_metrics,
            "metric_state": baseline_state,
        },
        "candidate_oof": {
            "metrics": candidate_metrics,
            "metric_state": candidate_state,
        },
        "folds": fold_rows,
        "aoas": aoa_rows,
        "aggregation": "additive_sufficient_metric_state",
        "decision": (
            "PASS_B210_SMALL4K_CLEAN716_UQ_OOF_PROMOTION"
            if all(conditions.values())
            else "STOP_B210_SMALL4K_CLEAN716_UQ_OOF_GATE"
        ),
    }


def merge_aoa_state(target: dict[int, dict], source: Mapping[int, dict]) -> None:
    for aoa, state in source.items():
        if aoa not in target:
            target[aoa] = smu1.new_metric_state()
        for key in target[aoa]:
            target[aoa][key] += state[key]


def save_refit_endpoint(
    path: Path,
    model: EnsembleUNet2d,
    point_report: Mapping[str, object],
    normalizer_report: Mapping[str, object],
    training: Mapping[str, object],
    gate: Mapping[str, object],
) -> dict:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    state = ensemble_train.common.checkpoint_state(model)
    state_sha = dr1.model_state_sha256(state)
    payload = {
        "schema": REFIT_ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "stage": REFIT_STAGE,
        "phase": "real",
        "iteration": REFIT_UPDATES,
        "endpoint": "fixed_terminal_all716_u800",
        "primary": True,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "point_checkpoint_sha256": point_report["checkpoint_sha256"],
        "point_model_state_sha256": point_report["model_state_sha256"],
        "normalizer": dict(normalizer_report["values"]),
        "normalizer_sha256": normalizer_report["sha256"],
        "training": dict(training),
        "oof_promotion_gate": dict(gate),
        "optimizer_state_saved": False,
        "point_gain_attribution": False,
        "exact_verifier_required": True,
        "zip_written": False,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "schema": REFIT_ENDPOINT_SCHEMA,
        "stage": REFIT_STAGE,
        "iteration": REFIT_UPDATES,
        "exact_verifier_required": True,
        "zip_written": False,
        "submission_authorized": False,
    }


def validate_hardware(device: torch.device, hardware_class: str) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal B210 clean716 UQ requires CUDA")
    name = torch.cuda.get_device_name(device).lower()
    expected = "4090" if hardware_class == "rtx4090" else "4080"
    if expected not in name or (
        hardware_class == "rtx4080_super" and "super" not in name
    ):
        raise RuntimeError(
            f"B210 UQ hardware class {hardware_class} does not match {name}"
        )
    return {"hardware_class": hardware_class, "device_name": name}


def disk_preflight(output_dir: Path) -> dict:
    probe = output_dir.parent.resolve()
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError("no existing parent for B210 UQ output")
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"B210 UQ output has {free} free bytes; requires {MIN_FREE_BYTES}"
        )
    return {
        "probe": str(probe),
        "free_bytes_before_run": free,
        "minimum_free_bytes": MIN_FREE_BYTES,
    }


def run(args: argparse.Namespace) -> dict:
    identity = dr1.validate_formal_git_identity(dr1.git_identity())
    if identity["branch"] != FORMAL_BRANCH:
        raise RuntimeError("formal B210 clean716 UQ is on the wrong branch")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    device = torch.device(args.device)
    hardware = validate_hardware(device, args.hardware_class)
    disk = disk_preflight(args.output_dir)
    if dr1.sha256(args.clean_manifest) != FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("B210 clean manifest file SHA changed")
    clean_rows, clean_report = dr1.load_manifest(
        args.clean_manifest, FORMAL_MANIFEST_SEMANTIC_SHA256
    )
    tail_rows, tail_report = derive_clean716_manifest(clean_rows)
    payload, normalizer, point_report = load_point_artifact(
        args.point_checkpoint, args.point_checkpoint_sha256, device
    )

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + WALL_BUDGET_SECONDS
    torch.set_float32_matmul_precision("high")
    access = TailAccessAudit()
    trajectories, metadata, windows = load_tail_rows(args.real_dir, tail_rows, access)
    if len(windows) != EXPECTED_TAIL_WINDOWS:
        raise RuntimeError("loaded clean716 tail count changed")
    access_report = access.report([row["file"] for row in tail_rows])
    if (
        not access_report["allowlist_exact"]
        or access_report["prefix_frames_read"] != 0
        or access_report["excluded_opened"]
    ):
        raise RuntimeError("clean716 tail access contract failed")
    dataset = ensemble_train.common.repro.WindowDataset(trajectories, windows)
    sample_folds, fold_manifest = trajectory_folds(tail_rows, FOLDS)
    if len(sample_folds) != EXPECTED_TAIL_WINDOWS:
        raise RuntimeError("condition-fold labels do not cover clean716")
    if set(sample_folds) != set(range(FOLDS)):
        raise RuntimeError("condition-fold assignment is incomplete")
    all_indices = list(range(EXPECTED_TAIL_WINDOWS))
    endpoint_audit = FoldEndpointAudit()
    fold_rows = []
    baseline_folds: list[dict] = []
    candidate_folds: list[dict] = []
    baseline_aoas: dict[int, dict] = {}
    candidate_aoas: dict[int, dict] = {}

    for fold in range(FOLDS):
        train_indices = [index for index in all_indices if sample_folds[index] != fold]
        held_indices = [index for index in all_indices if sample_folds[index] == fold]
        if set(train_indices).intersection(held_indices):
            raise RuntimeError("held clean716 window entered optimizer indices")
        if sorted(train_indices + held_indices) != all_indices:
            raise RuntimeError("fold partition does not cover clean716")
        candidate = load_point_model(payload, normalizer, device)
        training = train_uq_terminal(
            candidate,
            dataset,
            train_indices,
            normalizer,
            device,
            deadline,
            SEED + fold + 1,
            f"fold{fold}",
        )
        if (
            training["frozen_state_sha256_before"]
            != point_report["frozen_state_sha256"]
        ):
            raise RuntimeError("fold frozen state is not bound to point artifact")
        seal = save_fold_endpoint(
            args.output_dir / f"fold_{fold:02d}_terminal_uq.pth",
            candidate,
            fold,
            point_report["checkpoint_sha256"],
            training,
        )
        endpoint_audit.seal(fold, seal)

        endpoint_audit.before_evaluate(fold, "native_baseline")
        baseline = load_point_model(payload, normalizer, device)
        baseline_state, baseline_groups = evaluate_grouped(
            baseline, dataset, held_indices, metadata, normalizer, device, deadline
        )
        del baseline
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        endpoint_audit.before_evaluate(fold, "fixed_terminal_candidate")
        candidate_state, candidate_groups = evaluate_grouped(
            candidate, dataset, held_indices, metadata, normalizer, device, deadline
        )
        if not point_state_equal(baseline_state, candidate_state):
            raise RuntimeError("UQ-only fold changed point metric state")
        baseline_folds.append(baseline_state)
        candidate_folds.append(candidate_state)
        merge_aoa_state(baseline_aoas, baseline_groups)
        merge_aoa_state(candidate_aoas, candidate_groups)
        row = {
            "fold": fold,
            "train_windows": len(train_indices),
            "held_windows": len(held_indices),
            "endpoint": seal,
            "training": training,
            "baseline": {
                "metrics": smu1.finalize_metric_state(baseline_state),
                "metric_state": baseline_state,
            },
            "candidate": {
                "metrics": smu1.finalize_metric_state(candidate_state),
                "metric_state": candidate_state,
            },
            "point_metric_state_bitwise": True,
            "held_checkpoint_selection": False,
            "point_gain_attribution": False,
        }
        row["sps_delta"] = float(
            row["candidate"]["metrics"]["native_sps_score"]
            - row["baseline"]["metrics"]["native_sps_score"]
        )
        fold_rows.append(row)
        dr1.write_json_atomic(args.output_dir / f"fold_{fold:02d}_result.json", row)
        print(
            json.dumps(
                {
                    "event": "b210_clean716_uq_fold_complete",
                    "fold": fold,
                    "sps_delta": row["sps_delta"],
                }
            ),
            flush=True,
        )
        del candidate
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gate = promotion_gate(
        baseline_folds, candidate_folds, baseline_aoas, candidate_aoas
    )
    endpoint_report = endpoint_audit.report()
    if not endpoint_report["fixed_terminal_only"]:
        raise RuntimeError("held evaluation violated fixed-terminal order")

    refit_seal = None
    refit_training = None
    if gate["pass"]:
        refit = load_point_model(payload, normalizer, device)
        refit_training = train_uq_terminal(
            refit,
            dataset,
            all_indices,
            normalizer,
            device,
            deadline,
            SEED + FOLDS + 1,
            "all716_refit",
        )
        if (
            refit_training["frozen_state_sha256_before"]
            != point_report["frozen_state_sha256"]
        ):
            raise RuntimeError("refit frozen state is not bound to point artifact")
        refit_seal = save_refit_endpoint(
            args.output_dir / "all716_uq_refit_u800.pth",
            refit,
            point_report,
            point_report["normalizer"],
            refit_training,
            gate,
        )
        del refit
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "evidence_class": (
            "clean716 whole-condition OOF UQ and gated all-tail refit; "
            "not official, fixed725, ZIP, or submission evidence"
        ),
        "point": point_report,
        "manifest": {
            "clean2544": clean_report,
            "clean716": tail_report,
            "manifest_file_sha256": FORMAL_MANIFEST_FILE_SHA256,
            "fold_manifest": fold_manifest,
        },
        "tail_access": access_report,
        "fold_endpoint_audit": endpoint_report,
        "folds": fold_rows,
        "promotion_gate": gate,
        "refit_training": refit_training,
        "refit_endpoint": refit_seal,
        "git": identity,
        "hardware": hardware,
        "disk_preflight": disk,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "elapsed_seconds": time.perf_counter() - started,
        "point_gain_attribution": False,
        "uq_stage_checkpoint_written": refit_seal is not None,
        "exact_verifier_required": refit_seal is not None,
        "zip_written": False,
        "submission_authorized": False,
        "decision": gate["decision"],
    }
    dr1.write_json_atomic(args.output_dir / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--contract-only", action="store_true")
    parser.add_argument("--point-checkpoint", type=Path)
    parser.add_argument("--point-checkpoint-sha256")
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hardware-class",
        choices=("rtx4080_super", "rtx4090"),
        default="rtx4080_super",
    )
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.run:
        required = {
            "point_checkpoint": args.point_checkpoint,
            "point_checkpoint_sha256": args.point_checkpoint_sha256,
            "clean_manifest": args.clean_manifest,
            "output_dir": args.output_dir,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise ValueError(f"B210 clean716 UQ run missing {missing}")
        args.point_checkpoint_sha256 = dr1.normalized_sha256(
            args.point_checkpoint_sha256, "point-checkpoint-sha256"
        )
        if (
            not args.execute_formal_gpu
            or args.acknowledge_protocol != FORMAL_ACKNOWLEDGEMENT
        ):
            raise ValueError("formal B210 clean716 UQ requires acknowledgement")
    elif any(
        (
            args.point_checkpoint,
            args.point_checkpoint_sha256,
            args.clean_manifest,
            args.output_dir,
            args.execute_formal_gpu,
        )
    ):
        raise ValueError("contract-only received execution arguments")


def write_stop(output_dir: Path, error: BaseException) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stop.json"
    if path.exists():
        return
    decision = (
        "STOP_B210_SMALL4K_CLEAN716_UQ_OOM"
        if isinstance(error, torch.cuda.OutOfMemoryError)
        else "STOP_B210_SMALL4K_CLEAN716_UQ_RUNTIME"
    )
    dr1.write_json_atomic(
        path,
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "contract_sha256": CONTRACT_SHA256,
            "decision": decision,
            "error_type": type(error).__name__,
            "error": str(error),
            "zip_written": False,
            "submission_authorized": False,
        },
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        print(
            json.dumps(
                {
                    "experiment": EXPERIMENT,
                    "schema": SCHEMA,
                    "contract_sha256": CONTRACT_SHA256,
                    "contract": CONTRACT,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    try:
        result = run(args)
    except BaseException as error:
        if args.output_dir is not None:
            write_stop(args.output_dir, error)
        raise
    print(
        json.dumps(
            {
                "event": "b210_small4k_clean716_uq_complete",
                "decision": result["decision"],
                "pooled_sps_delta": result["promotion_gate"]["pooled_sps_delta"],
                "refit_endpoint": result["refit_endpoint"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
