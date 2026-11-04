#!/usr/bin/env python3
"""Verify the final B210 INT8 archive on the exact clean716 replay.

This verifier deliberately does not import or invoke the legacy 82/725 archive
evaluator.  A controller process binds every source SHA, then launches two
fresh sequential workers: one replays the raw all716 UQ-refit checkpoint and
one safely extracts and replays the exact INT8 ZIP.  Both workers reconstruct
the same 81-trajectory/716-tail stream and emit additive sufficient metric
state.  The final gate measures packaging damage only; it is neither a
generalization estimate nor an official score, and it never submits anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for import_path in (CODE_ROOT, TRAINING_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import train_b210_small4k_uq_release as uq_release  # noqa: E402
import train_c208_npt1_member_sync_tke_loao as npt1  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d  # noqa: E402


dr1 = uq_release.dr1
ft12 = uq_release.ft12
smu1 = uq_release.smu1
ensemble_train = uq_release.ensemble_train

EXPERIMENT = "b210_clean716_exact_int8_package_verify"
SCHEMA = "b210_clean716_exact_int8_package_verify_v1"
WORKER_SCHEMA = "b210_clean716_exact_int8_replay_worker_v1"
FORMAL_BRANCH = "codex/b210-clean716-package-verify"
FORMAL_ACKNOWLEDGEMENT = "B210_CLEAN716_EXACT_INT8_PACKAGE_VERIFY"
WORKER_ACKNOWLEDGEMENT = "B210_CLEAN716_INTERNAL_REPLAY_WORKER"

MAX_ARCHIVE_BYTES = 256 * 1024**2
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3
EXPECTED_MEMBERS = (
    "submission.py",
    "models/__init__.py",
    "models/unet2d.py",
    "models/ensemble_unet2d.py",
    "ensemble_unet2d.pth",
)
TRUSTED_MEMBER_SOURCES = {
    "submission.py": CODE_ROOT / "submissions/track1_ensemble_unet2d.py",
    "models/__init__.py": CODE_ROOT / "submissions/ensemble_models_init.py",
    "models/unet2d.py": CODE_ROOT / "models/unet2d.py",
    "models/ensemble_unet2d.py": CODE_ROOT / "models/ensemble_unet2d.py",
}
STORAGE_FORMAT = "symmetric_per_dim0_int8_v1"
QUANTIZE_MIN_ELEMENTS = 1024
MIN_QUANTIZED_FLOAT_FRACTION = 0.99
EVAL_BATCH_SIZE = 16
WALL_BUDGET_SECONDS = 60 * 60

PACKAGING_FLOORS = {
    "rel_l2_score": -0.03,
    "tke_rel_l2_score": -0.05,
    "mvpe_rel_l2_score": -0.03,
    "native_sps_score": -0.10,
    "non_time_delta": -0.05,
}
FLOOR_COMPARISON_EPSILON = 1e-12
FORECAST_BUFFER_KEYS = (
    "persistence_scale",
    "persistence_bias",
    "target_std",
)

CONTRACT = {
    "question": (
        "Does the exact final INT8 archive preserve the raw all716 UQ-refit "
        "checkpoint closely enough on the identical clean716 replay?"
    ),
    "sources": {
        "archive_sha256": "required and checked before extraction",
        "uq_release_result_sha256": "required and checked before parsing",
        "refit_checkpoint": ("bound to the refit_endpoint seal inside the UQ result"),
        "point_identity": (
            "point checkpoint/model SHA, non-UQ partition SHA, and normalizer SHA"
        ),
        "cross_run_uq_point_sums": (
            "diagnostic only; fold-partitioned OOF and sequential replay are not "
            "an exact numerical identity gate"
        ),
        "clean_manifest_file_sha256": (uq_release.FORMAL_MANIFEST_FILE_SHA256),
        "clean_manifest_semantic_sha256": (uq_release.FORMAL_MANIFEST_SEMANTIC_SHA256),
    },
    "data": {
        "trajectories": uq_release.EXPECTED_TRAJECTORIES,
        "tail_windows": uq_release.EXPECTED_TAIL_WINDOWS,
        "excluded": dr1.EXCLUDED_BASENAME,
        "slice": "u/v[val_boundary:,::2,::2]",
        "order": "identical to UQ all716 refit diagnostic",
        "legacy_725_evaluator_forbidden": True,
    },
    "isolation": {
        "controller": "SHA validation and comparison only",
        "direct": "fresh subprocess",
        "package": "separate fresh subprocess and fresh temporary extraction",
        "sequential_workers": True,
    },
    "package": {
        "members": list(EXPECTED_MEMBERS),
        "max_archive_bytes": MAX_ARCHIVE_BYTES,
        "storage_format": STORAGE_FORMAT,
        "quantize_min_elements": QUANTIZE_MIN_ELEMENTS,
        "inference_dtype": "float16 on CUDA",
        "strict_b210_config": True,
        "strict_state_restore": True,
        "quantized_float_fraction_min": MIN_QUANTIZED_FLOAT_FRACTION,
    },
    "checks": [
        "input_bitwise_unchanged",
        "input_and_output_pressure_exact_zero",
        "finite_prediction_and_bounds",
        "lower_le_upper",
        "raw_target_and_scored_mask_alignment",
        "additive_sufficient_metric_state",
    ],
    "packaging_impact_floors": PACKAGING_FLOORS,
    "failure_policy": (
        "STOP; do not alter the INT8 threshold or select another package"
    ),
    "evidence_class": (
        "same-clean716 packaging-equivalence diagnostic; not generalization "
        "or official leaderboard evidence"
    ),
    "submission_authorized": False,
}
CONTRACT_SHA256 = dr1.canonical_sha256(CONTRACT)


def _read_json_mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON evidence must be a mapping: {path}")
    return value


def _locked_file(path: Path, expected_sha256: str, label: str) -> dict:
    expected = dr1.normalized_sha256(expected_sha256, f"{label} SHA")
    actual = dr1.sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA changed: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "bytes": path.stat().st_size,
    }


def _validate_uq_gate(row: Mapping[str, object]) -> dict:
    gate = row.get("promotion_gate")
    folds = row.get("folds")
    if not isinstance(gate, Mapping) or not isinstance(folds, list):
        raise RuntimeError("UQ release lacks its complete OOF gate evidence")
    conditions = gate.get("conditions")
    baseline = gate.get("baseline_oof")
    candidate = gate.get("candidate_oof")
    gate_folds = gate.get("folds")
    aoas = gate.get("aoas")
    if (
        gate.get("pass") is not True
        or gate.get("decision") != "PASS_B210_SMALL4K_CLEAN716_UQ_OOF_PROMOTION"
        or gate.get("aggregation") != "additive_sufficient_metric_state"
        or not isinstance(conditions, Mapping)
        or not all(conditions.values())
        or not isinstance(baseline, Mapping)
        or not isinstance(candidate, Mapping)
        or not isinstance(gate_folds, list)
        or len(gate_folds) != uq_release.FOLDS
        or not isinstance(aoas, list)
        or len(aoas) != len(dr1.AOA_VALUES)
        or len(folds) != uq_release.FOLDS
    ):
        raise RuntimeError("UQ release promotion gate is incomplete")
    baseline_states = [fold.get("baseline", {}).get("metric_state") for fold in folds]
    candidate_states = [fold.get("candidate", {}).get("metric_state") for fold in folds]
    if not all(
        isinstance(state, Mapping) for state in baseline_states + candidate_states
    ):
        raise RuntimeError("UQ release lacks five-fold sufficient metric state")
    merged_baseline = smu1.merge_metric_states(
        [dict(state) for state in baseline_states]
    )
    merged_candidate = smu1.merge_metric_states(
        [dict(state) for state in candidate_states]
    )
    if (
        merged_baseline != baseline.get("metric_state")
        or merged_candidate != candidate.get("metric_state")
        or smu1.finalize_metric_state(merged_baseline) != baseline.get("metrics")
        or smu1.finalize_metric_state(merged_candidate) != candidate.get("metrics")
    ):
        raise RuntimeError("UQ release pooled gate is not additive sufficient state")
    expected_fold_rows = [
        uq_release._delta_row(index, baseline_states[index], candidate_states[index])
        for index in range(uq_release.FOLDS)
    ]
    if expected_fold_rows != gate_folds:
        raise RuntimeError("UQ release fold gate rows changed")
    for fold, baseline_state, candidate_state in zip(
        folds, baseline_states, candidate_states, strict=True
    ):
        if fold.get(
            "point_metric_state_bitwise"
        ) is not True or not uq_release.point_state_equal(
            baseline_state, candidate_state
        ):
            raise RuntimeError("UQ release OOF point state changed")
    fold_deltas = [float(value["sps_delta"]) for value in gate_folds]
    aoa_deltas = []
    if {int(value.get("group", -1)) for value in aoas} != set(dr1.AOA_VALUES):
        raise RuntimeError("UQ release AoA gate groups changed")
    for value in aoas:
        before = value.get("baseline", {})
        after = value.get("candidate", {})
        delta = float(after.get("native_sps_score", math.nan)) - float(
            before.get("native_sps_score", math.nan)
        )
        if not math.isfinite(delta) or not math.isclose(
            delta, float(value.get("sps_delta", math.nan)), abs_tol=1e-12
        ):
            raise RuntimeError("UQ release AoA SPS delta changed")
        aoa_deltas.append(delta)
    pooled_delta = float(
        candidate["metrics"]["native_sps_score"]
        - baseline["metrics"]["native_sps_score"]
    )
    expected_conditions = {
        "pooled_sps_gain_ge_0_10": pooled_delta >= uq_release.POOLED_GAIN_GATE,
        "positive_folds_ge_4_of_5": sum(value > 0.0 for value in fold_deltas)
        >= uq_release.GROUP_POSITIVE_GATE,
        "worst_fold_ge_minus_0_03": min(fold_deltas) >= uq_release.FOLD_WORST_GATE,
        "positive_aoas_ge_4_of_5": sum(value > 0.0 for value in aoa_deltas)
        >= uq_release.GROUP_POSITIVE_GATE,
        "worst_aoa_ge_minus_0_05": min(aoa_deltas) >= uq_release.AOA_WORST_GATE,
    }
    if (
        not math.isclose(
            pooled_delta, float(gate.get("pooled_sps_delta", math.nan)), abs_tol=1e-12
        )
        or dict(conditions) != expected_conditions
        or not all(expected_conditions.values())
    ):
        raise RuntimeError("UQ release promotion conditions do not recompute")
    return {
        "aggregation": "additive_sufficient_metric_state",
        "folds": len(folds),
        "aoas": len(aoas),
        "pooled_sps_delta": pooled_delta,
        "recomputed": True,
    }


def validate_uq_result(path: Path, expected_sha256: str) -> tuple[dict, dict]:
    source = _locked_file(path, expected_sha256, "UQ release result")
    row = _read_json_mapping(path)
    gate_report = _validate_uq_gate(row)
    gate = row.get("promotion_gate")
    refit = row.get("refit_endpoint")
    point = row.get("point")
    manifest = row.get("manifest", {})
    tail_access = row.get("tail_access", {})
    training = row.get("refit_training", {})
    point_normalizer = point.get("normalizer", {}) if isinstance(point, Mapping) else {}
    if (
        row.get("schema") != uq_release.SCHEMA
        or row.get("experiment") != uq_release.EXPERIMENT
        or row.get("contract_sha256") != uq_release.CONTRACT_SHA256
        or not isinstance(gate, Mapping)
        or gate.get("pass") is not True
        or row.get("decision") != "PASS_B210_SMALL4K_CLEAN716_UQ_OOF_PROMOTION"
        or not isinstance(refit, Mapping)
        or refit.get("schema") != uq_release.REFIT_ENDPOINT_SCHEMA
        or refit.get("stage") != uq_release.REFIT_STAGE
        or int(refit.get("iteration", -1)) != uq_release.REFIT_UPDATES
        or refit.get("exact_verifier_required") is not True
        or refit.get("zip_written") is not False
        or refit.get("submission_authorized") is not False
        or row.get("uq_stage_checkpoint_written") is not True
        or row.get("exact_verifier_required") is not True
        or row.get("point_gain_attribution") is not False
        or row.get("zip_written") is not False
        or row.get("submission_authorized") is not False
        or not isinstance(point, Mapping)
    ):
        raise RuntimeError("UQ release result is not a verifier-ready PASS")
    if (
        manifest.get("manifest_file_sha256") != uq_release.FORMAL_MANIFEST_FILE_SHA256
        or manifest.get("clean2544", {}).get("semantic_sha256")
        != uq_release.FORMAL_MANIFEST_SEMANTIC_SHA256
        or manifest.get("clean2544", {}).get("trajectories")
        != uq_release.EXPECTED_TRAJECTORIES
        or manifest.get("clean716", {}).get("tail_windows")
        != uq_release.EXPECTED_TAIL_WINDOWS
        or manifest.get("clean716", {}).get("excluded") != dr1.EXCLUDED_BASENAME
    ):
        raise RuntimeError("UQ release clean716 manifest evidence changed")
    if (
        tail_access.get("allowlist_exact") is not True
        or tail_access.get("prefix_frames_read") != 0
        or tail_access.get("excluded_opened") is not False
        or len(tail_access.get("opened_file_allowlist", []))
        != uq_release.EXPECTED_TRAJECTORIES
    ):
        raise RuntimeError("UQ release tail-access evidence changed")
    if (
        int(training.get("updates", -1)) != uq_release.REFIT_UPDATES
        or training.get("frozen_state_bitwise") is not True
        or training.get("frozen_state_sha256_before")
        != training.get("frozen_state_sha256_after")
        or not isinstance(point_normalizer, Mapping)
        or training.get("frozen_state_sha256_before")
        != point.get("frozen_state_sha256")
    ):
        raise RuntimeError("UQ release refit training evidence changed")
    seal = {
        "path": str(refit.get("path")),
        "file_sha256": dr1.normalized_sha256(
            str(refit.get("file_sha256")), "refit checkpoint SHA"
        ),
        "model_state_sha256": dr1.normalized_sha256(
            str(refit.get("model_state_sha256")), "refit model-state SHA"
        ),
        "schema": refit["schema"],
        "stage": refit["stage"],
        "iteration": int(refit["iteration"]),
        "point_checkpoint_sha256": dr1.normalized_sha256(
            str(point.get("checkpoint_sha256")), "point checkpoint SHA"
        ),
        "point_model_state_sha256": dr1.normalized_sha256(
            str(point.get("model_state_sha256")), "point model-state SHA"
        ),
        "point_frozen_state_sha256": dr1.normalized_sha256(
            str(point.get("frozen_state_sha256")), "point frozen-state SHA"
        ),
        "point_normalizer_sha256": dr1.normalized_sha256(
            str(point_normalizer.get("sha256")), "point normalizer SHA"
        ),
        "refit_training_sha256": dr1.canonical_sha256(training),
        "promotion_gate_sha256": dr1.canonical_sha256(gate),
        "gate_validation": gate_report,
    }
    return row, {**source, "refit_endpoint": seal}


def validate_refit_checkpoint(
    path: Path, seal: Mapping[str, object], device: torch.device | None = None
) -> tuple[dict, dict, smu1.FoldNormalizer | None, EnsembleUNet2d | None]:
    identity = _locked_file(path, str(seal["file_sha256"]), "UQ refit checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    fixed = {
        "schema": uq_release.REFIT_ENDPOINT_SCHEMA,
        "experiment": uq_release.EXPERIMENT,
        "contract_sha256": uq_release.CONTRACT_SHA256,
        "stage": uq_release.REFIT_STAGE,
        "phase": "real",
        "iteration": uq_release.REFIT_UPDATES,
        "endpoint": "fixed_terminal_all716_u800",
        "primary": True,
        "optimizer_state_saved": False,
        "point_gain_attribution": False,
        "exact_verifier_required": True,
        "zip_written": False,
        "submission_authorized": False,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in fixed.items()
    ):
        raise RuntimeError("UQ refit checkpoint stage contract changed")
    if payload.get("oof_promotion_gate", {}).get("pass") is not True:
        raise RuntimeError("UQ refit checkpoint lacks its passing OOF gate")
    if (
        dr1.normalized_sha256(
            str(payload.get("point_checkpoint_sha256")), "refit point checkpoint SHA"
        )
        != seal["point_checkpoint_sha256"]
        or dr1.normalized_sha256(
            str(payload.get("point_model_state_sha256")), "refit point model-state SHA"
        )
        != seal["point_model_state_sha256"]
        or dr1.canonical_sha256(payload.get("training"))
        != seal["refit_training_sha256"]
        or dr1.canonical_sha256(payload.get("oof_promotion_gate"))
        != seal["promotion_gate_sha256"]
    ):
        raise RuntimeError("UQ refit checkpoint is not bound to its release summary")
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise RuntimeError("UQ refit checkpoint lacks config/state")
    parsed = ft12.validate_b210_model_config(config)
    state_sha = dr1.model_state_sha256(state)
    if state_sha != seal["model_state_sha256"] or state_sha != dr1.normalized_sha256(
        str(payload.get("model_state_sha256")), "payload model-state SHA"
    ):
        raise RuntimeError("UQ refit model-state SHA changed")
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ):
        raise TypeError("UQ refit floating state must be FP32")
    frozen_state_sha = dr1.model_state_sha256(
        {
            key: value
            for key, value in state.items()
            if not key.startswith(uq_release.UQ_PREFIXES)
        }
    )
    training = payload.get("training", {})
    if (
        frozen_state_sha != seal["point_frozen_state_sha256"]
        or training.get("frozen_state_sha256_before") != frozen_state_sha
        or training.get("frozen_state_sha256_after") != frozen_state_sha
    ):
        raise RuntimeError("UQ refit non-UQ state is not bound to the point endpoint")
    statistics, normalizer_report = uq_release.validate_flat_normalizer(payload)
    if normalizer_report["sha256"] != seal["point_normalizer_sha256"]:
        raise RuntimeError("UQ refit normalizer is not bound to the point endpoint")
    report = {
        **identity,
        "model_state_sha256": state_sha,
        "model_config_sha256": dr1.canonical_sha256(config),
        "normalizer": normalizer_report,
        "frozen_state_sha256": frozen_state_sha,
        "strict_b210_config": True,
        **fixed,
    }
    if device is None:
        return payload, report, None, None
    normalizer = smu1.FoldNormalizer(statistics, device)
    model = EnsembleUNet2d(parsed).float()
    model.load_state_dict(state, strict=True)
    ensemble_train.common.set_model_stats(model, normalizer)
    total = sum(value.numel() for value in model.parameters())
    controller = sum(value.numel() for value in model.energy_controller.parameters())
    if (
        total != uq_release.EXPECTED_PARAMETERS
        or controller != uq_release.EXPECTED_CONTROLLER_PARAMETERS
    ):
        raise RuntimeError("strict B210 refit parameter count changed")
    report.update(
        {
            "strict_load": True,
            "parameters": total,
            "controller_parameters": controller,
        }
    )
    return payload, report, normalizer, model.to(device)


def archive_identity(path: Path, expected_sha256: str) -> tuple[dict, dict]:
    identity = _locked_file(path, expected_sha256, "INT8 archive")
    if identity["bytes"] > MAX_ARCHIVE_BYTES:
        raise RuntimeError("INT8 archive exceeds 256 MiB")
    with zipfile.ZipFile(path) as archive:
        members = tuple(archive.namelist())
        bad = archive.testzip()
        uncompressed = sum(info.file_size for info in archive.infolist())
        if bad is not None:
            raise RuntimeError(f"corrupt archive member: {bad}")
        if members != EXPECTED_MEMBERS:
            raise RuntimeError(f"unexpected INT8 archive layout: {members}")
        if uncompressed > MAX_UNCOMPRESSED_BYTES:
            raise RuntimeError("INT8 archive uncompressed size is unreasonable")
        for name in members:
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe ZIP member: {name}")
        source_bindings = {}
        for name, source in TRUSTED_MEMBER_SOURCES.items():
            packaged = archive.read(name)
            trusted = source.read_bytes()
            if packaged != trusted:
                raise RuntimeError(
                    f"archive code member differs from trusted source: {name}"
                )
            source_bindings[name] = {
                "sha256": hashlib.sha256(packaged).hexdigest(),
                "trusted_path": str(source.resolve()),
            }
        payload = torch.load(
            io.BytesIO(archive.read("ensemble_unet2d.pth")),
            map_location="cpu",
            weights_only=True,
        )
    return {
        **identity,
        "members": list(members),
        "uncompressed_bytes": uncompressed,
        "within_256_mib": True,
        "trusted_code_sources": source_bindings,
    }, payload


def validate_source_normalizer_buffers(direct: Mapping[str, object]) -> dict:
    state = direct.get("model_state_dict")
    normalizer = direct.get("normalizer")
    statistic_keys = ("mean_input", "mean_target", "std_input", "std_target")
    if (
        not isinstance(state, Mapping)
        or not isinstance(normalizer, Mapping)
        or set(normalizer) != set(statistic_keys)
    ):
        raise RuntimeError("refit source lacks state or flat normalizer")
    actual = {key: state.get(key) for key in FORECAST_BUFFER_KEYS}
    if any(not isinstance(value, torch.Tensor) for value in actual.values()):
        raise RuntimeError("refit source lacks forecast normalizer buffers")
    statistics = {
        key: torch.as_tensor(normalizer[key], dtype=torch.float32)
        for key in statistic_keys
    }
    channels = int(actual["target_std"].numel())
    if any(value.shape != (3,) for value in statistics.values()) or channels != 3:
        raise RuntimeError("refit source normalizer shape changed")
    safe_target = statistics["std_target"][:channels].clamp_min(1e-8)
    expected = {
        "persistence_scale": statistics["std_input"][:channels] / safe_target,
        "persistence_bias": (
            statistics["mean_input"][:channels] - statistics["mean_target"][:channels]
        )
        / safe_target,
        "target_std": safe_target,
    }
    mismatches = sorted(
        key
        for key in FORECAST_BUFFER_KEYS
        if not torch.equal(
            actual[key].detach().cpu().float(), expected[key].detach().cpu()
        )
    )
    if mismatches:
        raise RuntimeError(
            "refit forecast buffers are inconsistent with flat normalizer: "
            f"{mismatches}"
        )
    return {
        "keys": list(FORECAST_BUFFER_KEYS),
        "source_fp32_exact": True,
        "flat_normalizer_consistent": True,
    }


def validate_int8_payload(
    payload: Mapping[str, object], direct: Mapping[str, object]
) -> dict:
    config = payload.get("model_config")
    direct_config = direct.get("model_config")
    stored = payload.get("quantized_state_dict")
    direct_state = direct.get("model_state_dict")
    if (
        set(payload)
        != {
            "phase",
            "model_config",
            "quantized_state_dict",
            "normalizer",
            "inference_dtype",
            "storage_format",
        }
        or payload.get("phase") != "real"
        or payload.get("storage_format") != STORAGE_FORMAT
        or payload.get("inference_dtype") != "float16"
        or "model_state_dict" in payload
        or not isinstance(config, Mapping)
        or config != direct_config
        or not isinstance(stored, Mapping)
        or not isinstance(direct_state, Mapping)
        or set(stored) != set(direct_state)
    ):
        raise RuntimeError("INT8 package payload contract changed")
    ft12.validate_b210_model_config(config)
    package_normalizer = payload.get("normalizer")
    if (
        not isinstance(package_normalizer, Mapping)
        or package_normalizer != direct.get("normalizer")
        or dr1.canonical_sha256(package_normalizer) != direct.get("normalizer_sha256")
    ):
        raise RuntimeError("INT8 package flat normalizer changed")
    source_normalizer_buffers = validate_source_normalizer_buffers(direct)

    quantized_tensors = 0
    quantized_elements = 0
    floating_elements = 0
    plain_fp16_tensors = 0
    for key, original in direct_state.items():
        if not isinstance(original, torch.Tensor):
            raise TypeError(f"direct state entry is not a tensor: {key}")
        value = stored[key]
        should_quantize = (
            original.is_floating_point()
            and original.ndim >= 2
            and original.numel() >= QUANTIZE_MIN_ELEMENTS
        )
        if original.is_floating_point():
            floating_elements += original.numel()
        if isinstance(value, Mapping):
            if not should_quantize:
                raise RuntimeError(f"ineligible tensor was INT8-quantized: {key}")
            if set(value) != {"quantized", "scale"}:
                raise RuntimeError(f"invalid quantized entry: {key}")
            quantized = value["quantized"]
            scale = value["scale"]
            expected_scale = (original.shape[0],) + (1,) * (original.ndim - 1)
            if (
                not isinstance(quantized, torch.Tensor)
                or quantized.dtype != torch.int8
                or quantized.shape != original.shape
                or int(quantized.min()) < -127
                or not isinstance(scale, torch.Tensor)
                or scale.dtype != torch.float32
                or tuple(scale.shape) != expected_scale
                or not bool(torch.isfinite(scale).all())
                or bool((scale <= 0.0).any())
            ):
                raise RuntimeError(f"invalid INT8 tensor/scale: {key}")
            restored = quantized.float() * scale
            error = (restored - original.float()).abs()
            reduction_dims = tuple(range(1, original.ndim))
            expected_scale_value = (
                original.detach()
                .cpu()
                .float()
                .abs()
                .amax(dim=reduction_dims, keepdim=True)
                / 127.0
            ).clamp_min(torch.finfo(torch.float32).tiny)
            expected_quantized = (
                torch.round(original.detach().cpu().float() / expected_scale_value)
                .clamp(-127, 127)
                .to(torch.int8)
            )
            if (
                not torch.equal(scale, expected_scale_value)
                or not torch.equal(quantized, expected_quantized)
                or not bool((error <= scale / 2.0 + 1e-6).all())
            ):
                raise RuntimeError(f"INT8 tensor is not bound to refit state: {key}")
            quantized_tensors += 1
            quantized_elements += original.numel()
        elif original.is_floating_point():
            if should_quantize:
                raise RuntimeError(f"eligible tensor was not INT8-quantized: {key}")
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float16
                or not torch.equal(value, original.to(torch.float16))
            ):
                raise RuntimeError(f"plain FP16 tensor changed: {key}")
            plain_fp16_tensors += 1
        elif not isinstance(value, torch.Tensor) or not torch.equal(value, original):
            raise RuntimeError(f"non-floating state changed: {key}")
    fraction = quantized_elements / max(floating_elements, 1)
    if quantized_tensors == 0 or fraction < MIN_QUANTIZED_FLOAT_FRACTION:
        raise RuntimeError(
            "archive does not contain the required INT8 storage coverage"
        )
    return {
        "storage_format": STORAGE_FORMAT,
        "inference_dtype": "float16",
        "state_keys": len(stored),
        "quantized_tensors": quantized_tensors,
        "quantized_elements": quantized_elements,
        "floating_elements": floating_elements,
        "quantized_float_fraction": fraction,
        "quantize_min_elements": QUANTIZE_MIN_ELEMENTS,
        "plain_fp16_tensors": plain_fp16_tensors,
        "strict_b210_config": True,
        "normalizer_bound_to_refit": True,
        "source_normalizer_buffers": source_normalizer_buffers,
        "per_dim0_error_bound": True,
    }


def safe_extract_exact(archive_path: Path, destination: Path) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("fresh extraction directory is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        if tuple(archive.namelist()) != EXPECTED_MEMBERS:
            raise RuntimeError("archive layout changed before extraction")
        for name in EXPECTED_MEMBERS:
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe ZIP member: {name}")
            target = (destination / member).resolve()
            if destination.resolve() not in target.parents:
                raise ValueError(f"ZIP member escapes extraction root: {name}")
        archive.extractall(destination)
    return {
        "fresh_directory": True,
        "members": list(EXPECTED_MEMBERS),
        "extracted_root": str(destination.resolve()),
    }


def load_clean716(
    real_dir: Path, clean_manifest: Path
) -> tuple[object, list, list, dict]:
    if dr1.sha256(clean_manifest) != uq_release.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean manifest file SHA changed")
    clean_rows, clean_report = dr1.load_manifest(
        clean_manifest, uq_release.FORMAL_MANIFEST_SEMANTIC_SHA256
    )
    tail_rows, tail_report = uq_release.derive_clean716_manifest(clean_rows)
    access = uq_release.TailAccessAudit()
    trajectories, metadata, windows = uq_release.load_tail_rows(
        real_dir, tail_rows, access
    )
    access_report = access.report([row["file"] for row in tail_rows])
    expected_files = sorted(str(row["file"]) for row in tail_rows)
    if (
        len(trajectories) != uq_release.EXPECTED_TRAJECTORIES
        or len(windows) != uq_release.EXPECTED_TAIL_WINDOWS
        or not access_report["allowlist_exact"]
        or access_report["prefix_frames_read"] != 0
        or access_report["excluded_opened"]
        or access_report["opened_file_allowlist"] != expected_files
        or access_report["expected_file_allowlist"] != expected_files
        or access_report["tail_arrays_read"] != uq_release.EXPECTED_TRAJECTORIES
    ):
        raise RuntimeError("clean716 replay data contract changed")
    ordered_keys = [
        {"file": row["file"], "start": int(start)}
        for row in tail_rows
        for start in row["val_starts"]
    ]
    dataset = ensemble_train.common.repro.WindowDataset(trajectories, windows)
    report = {
        "manifest_file_sha256": uq_release.FORMAL_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": clean_report["semantic_sha256"],
        "clean716_semantic_sha256": tail_report["semantic_sha256"],
        "trajectories": len(trajectories),
        "tail_windows": len(windows),
        "excluded": dr1.EXCLUDED_BASENAME,
        "ordered_key_sha256": dr1.canonical_sha256(ordered_keys),
        "tail_access": access_report,
    }
    return dataset, metadata, windows, report


def _update_numpy_hash(digest, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _new_replay_audit() -> dict:
    return {
        "input_before": hashlib.sha256(),
        "input_after": hashlib.sha256(),
        "target": hashlib.sha256(),
        "mask": hashlib.sha256(),
        "samples": 0,
        "scored": 0,
        "input_pressure_nonzero": 0,
        "output_pressure_nonzero": 0,
        "batches": 0,
    }


def _audit_batch_before(
    audit: dict, history: torch.Tensor, target: torch.Tensor
) -> np.ndarray:
    history_np = history.contiguous().numpy()
    target_np = target.contiguous().numpy()
    mask_np = target_np[..., :2] != 0.0
    _update_numpy_hash(audit["input_before"], history_np)
    _update_numpy_hash(audit["target"], target_np)
    _update_numpy_hash(audit["mask"], mask_np.astype(np.uint8, copy=False))
    audit["samples"] += len(history_np)
    audit["scored"] += int(np.count_nonzero(mask_np))
    audit["input_pressure_nonzero"] += int(np.count_nonzero(history_np[..., 2]))
    audit["batches"] += 1
    return history_np


def _finalize_replay_audit(audit: dict) -> dict:
    unchanged = audit["input_before"].hexdigest() == audit["input_after"].hexdigest()
    return {
        "ordered_raw_input_sha256": audit["input_before"].hexdigest(),
        "ordered_raw_input_after_sha256": audit["input_after"].hexdigest(),
        "input_bitwise_unchanged": unchanged,
        "ordered_raw_target_sha256": audit["target"].hexdigest(),
        "ordered_scored_mask_sha256": audit["mask"].hexdigest(),
        "samples": audit["samples"],
        "scored_elements": audit["scored"],
        "input_pressure_nonzero": audit["input_pressure_nonzero"],
        "output_pressure_nonzero": audit["output_pressure_nonzero"],
        "batches": audit["batches"],
    }


@torch.inference_mode()
def replay_direct(
    model: EnsembleUNet2d,
    dataset,
    normalizer: smu1.FoldNormalizer,
    device: torch.device,
    deadline: float,
) -> tuple[dict, dict]:
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    state = smu1.new_metric_state()
    audit = _new_replay_audit()
    model.eval()
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for history, target in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("direct clean716 replay exceeded wall budget")
        raw = _audit_batch_before(audit, history, target)
        history_before = raw.copy()
        history_device = history.to(device, non_blocking=True)
        target_device = target.to(device, non_blocking=True)
        history_norm, _ = normalizer.preprocess(history_device, target_device)
        prediction_norm, lower_norm, upper_norm = model.forward_distribution(
            history_norm
        )
        prediction = normalizer.postprocess_prediction(prediction_norm.float())
        prediction[..., 2] = 0.0
        lower_extent = lower_norm.float() * scale
        upper_extent = upper_norm.float() * scale
        lower = prediction[..., :2] - lower_extent
        upper = prediction[..., :2] + upper_extent
        if (
            not bool(torch.isfinite(prediction).all())
            or not bool(torch.isfinite(lower).all())
            or not bool(torch.isfinite(upper).all())
            or bool((lower > upper).any())
        ):
            raise FloatingPointError("direct refit produced invalid outputs")
        if not np.array_equal(raw, history_before):
            raise RuntimeError("direct replay mutated raw input")
        audit["output_pressure_nonzero"] += int(
            torch.count_nonzero(prediction[..., 2]).item()
        )
        _update_numpy_hash(audit["input_after"], raw)
        smu1.update_metric_state(
            state,
            prediction,
            target_device.float(),
            lower_extent,
            upper_extent,
        )
    report = _finalize_replay_audit(audit)
    if (
        not report["input_bitwise_unchanged"]
        or report["input_pressure_nonzero"] != 0
        or report["output_pressure_nonzero"] != 0
        or report["samples"] != uq_release.EXPECTED_TAIL_WINDOWS
        or int(state["samples"]) != uq_release.EXPECTED_TAIL_WINDOWS
    ):
        raise RuntimeError("direct clean716 replay audit failed")
    return state, report


def _purge_packaged_modules() -> None:
    for name in tuple(sys.modules):
        if (
            name == "models"
            or name.startswith("models.")
            or name == "evaluated_submission"
        ):
            sys.modules.pop(name, None)


def _module_source_report(package_dir: Path, submission) -> dict:
    root = package_dir.resolve()
    modules = {
        "submission": submission,
        "models": sys.modules.get("models"),
        "models.unet2d": sys.modules.get("models.unet2d"),
        "models.ensemble_unet2d": sys.modules.get("models.ensemble_unet2d"),
    }
    report = {}
    for name, module in modules.items():
        source = getattr(module, "__file__", None)
        if source is None:
            raise ImportError(f"packaged module lacks source path: {name}")
        resolved = Path(source).resolve()
        if resolved != root and root not in resolved.parents:
            raise ImportError(f"packaged module escaped fresh extraction: {name}")
        report[name] = str(resolved)
    return report


def load_packaged_submission(package_dir: Path):
    _purge_packaged_modules()
    sys.path.insert(0, str(package_dir))
    spec = importlib.util.spec_from_file_location(
        "evaluated_submission", package_dir / "submission.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load extracted submission.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluated_submission"] = module
    spec.loader.exec_module(module)
    return module, _module_source_report(package_dir, module)


def validate_runtime_restore(submission, payload: Mapping[str, object]) -> dict:
    if getattr(submission, "_BATCH_SIZE", None) != EVAL_BATCH_SIZE:
        raise RuntimeError("packaged B210 batch-size contract changed")
    runtime = submission._get_model()
    model = runtime.get("model")
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("packaged submission did not load a model")
    ft12.validate_b210_model_config(model.architecture_config())
    total = sum(value.numel() for value in model.parameters())
    controller = sum(value.numel() for value in model.energy_controller.parameters())
    if (
        total != uq_release.EXPECTED_PARAMETERS
        or controller != uq_release.EXPECTED_CONTROLLER_PARAMETERS
        or model.training
    ):
        raise RuntimeError("packaged strict B210 runtime changed")
    expected_dtype = (
        torch.float16 if runtime["device"].type == "cuda" else torch.float32
    )
    if runtime["dtype"] != expected_dtype:
        raise RuntimeError("packaged runtime restore dtype changed")
    restored = submission._restore_state_dict(payload, runtime["dtype"])
    loaded = model.state_dict()
    if set(restored) != set(loaded):
        raise RuntimeError("packaged runtime state keys differ from strict restore")
    mismatch_keys = sorted(
        key
        for key in loaded
        if not torch.equal(loaded[key].detach().cpu(), restored[key].detach().cpu())
    )
    if mismatch_keys:
        raise RuntimeError(
            f"packaged runtime state differs from strict restore: {mismatch_keys}"
        )
    del restored
    runtime_device = torch.device(runtime["device"])
    runtime_device_index = runtime_device.index
    if runtime_device.type == "cuda" and runtime_device_index is None:
        runtime_device_index = torch.cuda.current_device()
    return {
        "strict_load": True,
        "parameters": total,
        "controller_parameters": controller,
        "device": str(runtime["device"]),
        "device_type": runtime_device.type,
        "device_index": runtime_device_index,
        "dtype": str(runtime["dtype"]),
        "model_eval": True,
        "restored_state_matches_runtime": True,
    }


def replay_package(submission, dataset, deadline: float) -> tuple[dict, dict]:
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    state = smu1.new_metric_state()
    audit = _new_replay_audit()
    for history, target in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("package clean716 replay exceeded wall budget")
        raw = _audit_batch_before(audit, history, target)
        before = raw.copy()
        result = submission.predict(raw, metadata={})
        if not np.array_equal(raw, before):
            raise RuntimeError("submission mutated its input array")
        _update_numpy_hash(audit["input_after"], raw)
        if not isinstance(result, Mapping) or set(result) != {
            "prediction",
            "lower",
            "upper",
        }:
            raise ValueError("submission did not return point/lower/upper")
        prediction = np.asarray(result["prediction"])
        lower = np.asarray(result["lower"])
        upper = np.asarray(result["upper"])
        expected_shape = tuple(target.shape)
        if (
            prediction.shape != expected_shape
            or lower.shape != expected_shape
            or upper.shape != expected_shape
            or prediction.dtype != np.float32
            or lower.dtype != np.float32
            or upper.dtype != np.float32
            or not np.all(np.isfinite(prediction))
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise FloatingPointError("INT8 package returned invalid outputs")
        pressure_nonzero = sum(
            int(np.count_nonzero(value[..., 2])) for value in (prediction, lower, upper)
        )
        audit["output_pressure_nonzero"] += pressure_nonzero
        prediction_tensor = torch.from_numpy(np.ascontiguousarray(prediction))
        truth = target.float()
        lower_extent = torch.from_numpy(
            np.ascontiguousarray(prediction[..., :2] - lower[..., :2])
        )
        upper_extent = torch.from_numpy(
            np.ascontiguousarray(upper[..., :2] - prediction[..., :2])
        )
        smu1.update_metric_state(
            state, prediction_tensor, truth, lower_extent, upper_extent
        )
    report = _finalize_replay_audit(audit)
    if (
        not report["input_bitwise_unchanged"]
        or report["input_pressure_nonzero"] != 0
        or report["output_pressure_nonzero"] != 0
        or report["samples"] != uq_release.EXPECTED_TAIL_WINDOWS
        or int(state["samples"]) != uq_release.EXPECTED_TAIL_WINDOWS
    ):
        raise RuntimeError("INT8 package clean716 replay audit failed")
    return state, report


def worker_direct(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    hardware = ft12.validate_hardware(device, args.hardware_class)
    deadline = time.perf_counter() + WALL_BUDGET_SECONDS
    dataset, _metadata, _windows, data_report = load_clean716(
        args.real_dir, args.clean_manifest
    )
    seal = {
        "file_sha256": args.refit_checkpoint_sha256,
        "model_state_sha256": args.refit_model_state_sha256,
        "point_checkpoint_sha256": args.refit_point_checkpoint_sha256,
        "point_model_state_sha256": args.refit_point_model_state_sha256,
        "point_frozen_state_sha256": args.refit_point_frozen_state_sha256,
        "point_normalizer_sha256": args.refit_point_normalizer_sha256,
        "refit_training_sha256": args.refit_training_sha256,
        "promotion_gate_sha256": args.promotion_gate_sha256,
    }
    _payload, checkpoint, normalizer, model = validate_refit_checkpoint(
        args.refit_checkpoint, seal, device
    )
    assert normalizer is not None and model is not None
    state, audit = replay_direct(model, dataset, normalizer, device, deadline)
    result = {
        "schema": WORKER_SCHEMA,
        "experiment": EXPERIMENT,
        "kind": "direct_refit_clean716",
        "fresh_process": True,
        "checkpoint": checkpoint,
        "data": data_report,
        "metrics": smu1.finalize_metric_state(state),
        "metric_state": state,
        "replay_audit": audit,
        "hardware": hardware,
    }
    dr1.write_json_atomic(args.worker_output, result)
    return result


def worker_package(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    hardware = ft12.validate_hardware(device, args.hardware_class)
    deadline = time.perf_counter() + WALL_BUDGET_SECONDS
    dataset, _metadata, _windows, data_report = load_clean716(
        args.real_dir, args.clean_manifest
    )
    seal = {
        "file_sha256": args.refit_checkpoint_sha256,
        "model_state_sha256": args.refit_model_state_sha256,
        "point_checkpoint_sha256": args.refit_point_checkpoint_sha256,
        "point_model_state_sha256": args.refit_point_model_state_sha256,
        "point_frozen_state_sha256": args.refit_point_frozen_state_sha256,
        "point_normalizer_sha256": args.refit_point_normalizer_sha256,
        "refit_training_sha256": args.refit_training_sha256,
        "promotion_gate_sha256": args.promotion_gate_sha256,
    }
    direct, checkpoint, _normalizer, _model = validate_refit_checkpoint(
        args.refit_checkpoint, seal, None
    )
    archive, payload = archive_identity(args.archive, args.archive_sha256)
    storage = validate_int8_payload(payload, direct)
    extraction_root = args.worker_output.parent / "fresh_extract"
    extraction = safe_extract_exact(args.archive, extraction_root)
    submission, module_sources = load_packaged_submission(extraction_root)
    runtime = validate_runtime_restore(submission, payload)
    requested_index = device.index
    if device.type == "cuda" and requested_index is None:
        requested_index = torch.cuda.current_device()
    if (
        runtime["device_type"] != device.type
        or runtime["device_index"] != requested_index
    ):
        raise RuntimeError("packaged runtime selected the wrong CUDA device")
    state, audit = replay_package(submission, dataset, deadline)
    result = {
        "schema": WORKER_SCHEMA,
        "experiment": EXPERIMENT,
        "kind": "int8_package_clean716",
        "fresh_process": True,
        "checkpoint": checkpoint,
        "archive": archive,
        "storage": storage,
        "extraction": extraction,
        "module_sources": module_sources,
        "runtime": runtime,
        "data": data_report,
        "metrics": smu1.finalize_metric_state(state),
        "metric_state": state,
        "replay_audit": audit,
        "hardware": hardware,
    }
    dr1.write_json_atomic(args.worker_output, result)
    return result


def metric_comparison(
    package_state: Mapping[str, object], direct_state: Mapping[str, object]
) -> dict:
    package = smu1.finalize_metric_state(dict(package_state))
    direct = smu1.finalize_metric_state(dict(direct_state))
    keys = (
        "rel_l2_score",
        "tke_rel_l2_score",
        "mvpe_rel_l2_score",
        "native_sps_score",
    )
    delta = {key: float(package[key] - direct[key]) for key in keys}
    direct_delta = sum(
        npt1.DIRECT_COEFFICIENTS[key] * delta[key] for key in npt1.DIRECT_COEFFICIENTS
    )
    non_time = direct_delta + npt1.SPS_COEFFICIENT * delta["native_sps_score"]
    conditions = {
        "rel_ge_minus_0p03": delta["rel_l2_score"]
        >= PACKAGING_FLOORS["rel_l2_score"] - FLOOR_COMPARISON_EPSILON,
        "tke_ge_minus_0p05": delta["tke_rel_l2_score"]
        >= PACKAGING_FLOORS["tke_rel_l2_score"] - FLOOR_COMPARISON_EPSILON,
        "mvpe_ge_minus_0p03": delta["mvpe_rel_l2_score"]
        >= PACKAGING_FLOORS["mvpe_rel_l2_score"] - FLOOR_COMPARISON_EPSILON,
        "sps_ge_minus_0p10": delta["native_sps_score"]
        >= PACKAGING_FLOORS["native_sps_score"] - FLOOR_COMPARISON_EPSILON,
        "non_time_ge_minus_0p05": non_time
        >= PACKAGING_FLOORS["non_time_delta"] - FLOOR_COMPARISON_EPSILON,
    }
    passed = all(conditions.values())
    return {
        "direct": direct,
        "package": package,
        "delta": delta,
        "direct_delta": float(direct_delta),
        "non_time_delta": float(non_time),
        "floors": dict(PACKAGING_FLOORS),
        "conditions": conditions,
        "pass": passed,
        "decision": (
            "PASS_B210_CLEAN716_EXACT_INT8_PACKAGE_VERIFY"
            if passed
            else "STOP_B210_CLEAN716_EXACT_INT8_PACKAGE_DAMAGE"
        ),
    }


def validate_worker_alignment(
    direct: Mapping[str, object], package: Mapping[str, object]
) -> dict:
    if (
        direct.get("schema") != WORKER_SCHEMA
        or package.get("schema") != WORKER_SCHEMA
        or direct.get("kind") != "direct_refit_clean716"
        or package.get("kind") != "int8_package_clean716"
        or direct.get("fresh_process") is not True
        or package.get("fresh_process") is not True
    ):
        raise RuntimeError("replay worker identity changed")
    direct_data = direct.get("data", {})
    package_data = package.get("data", {})
    direct_audit = direct.get("replay_audit", {})
    package_audit = package.get("replay_audit", {})
    aligned = {
        "manifest_file_sha256": direct_data.get("manifest_file_sha256")
        == package_data.get("manifest_file_sha256")
        == uq_release.FORMAL_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": direct_data.get("manifest_semantic_sha256")
        == package_data.get("manifest_semantic_sha256")
        == uq_release.FORMAL_MANIFEST_SEMANTIC_SHA256,
        "clean716_semantic_sha256": direct_data.get("clean716_semantic_sha256")
        == package_data.get("clean716_semantic_sha256"),
        "ordered_key_sha256": direct_data.get("ordered_key_sha256")
        == package_data.get("ordered_key_sha256"),
        "ordered_raw_input_sha256": direct_audit.get("ordered_raw_input_sha256")
        == package_audit.get("ordered_raw_input_sha256"),
        "ordered_raw_target_sha256": direct_audit.get("ordered_raw_target_sha256")
        == package_audit.get("ordered_raw_target_sha256"),
        "ordered_scored_mask_sha256": direct_audit.get("ordered_scored_mask_sha256")
        == package_audit.get("ordered_scored_mask_sha256"),
        "samples_716": direct_audit.get("samples")
        == package_audit.get("samples")
        == uq_release.EXPECTED_TAIL_WINDOWS,
        "scored_elements": direct_audit.get("scored_elements")
        == package_audit.get("scored_elements"),
        "input_bitwise_unchanged": direct_audit.get("input_bitwise_unchanged") is True
        and package_audit.get("input_bitwise_unchanged") is True,
        "pressure_zero": direct_audit.get("input_pressure_nonzero") == 0
        and package_audit.get("input_pressure_nonzero") == 0
        and package_audit.get("output_pressure_nonzero") == 0,
    }
    if not all(aligned.values()):
        raise RuntimeError(
            f"direct/package clean716 replay streams are not aligned: {aligned}"
        )
    return aligned


def direct_uq_point_diagnostic(
    direct: Mapping[str, object], uq_result: Mapping[str, object]
) -> dict:
    direct_state = direct.get("metric_state")
    gate = uq_result.get("promotion_gate", {})
    reference = gate.get("candidate_oof", {}).get("metric_state")
    baseline = gate.get("baseline_oof", {}).get("metric_state")
    keys = ("samples", "rel_error_sum", "tke_error_sum", "mvpe_error_sum")
    if not all(
        isinstance(value, Mapping) for value in (direct_state, reference, baseline)
    ):
        raise RuntimeError("UQ point-state binding evidence is missing")
    if any(reference.get(key) != baseline.get(key) for key in keys):
        raise RuntimeError("UQ candidate OOF point state differs from native baseline")
    if (
        int(direct_state.get("samples", -1)) != uq_release.EXPECTED_TAIL_WINDOWS
        or int(reference.get("samples", -1)) != uq_release.EXPECTED_TAIL_WINDOWS
    ):
        raise RuntimeError("cross-run point diagnostic does not cover clean716")
    raw_deltas = {}
    for key in keys:
        actual = float(direct_state.get(key, math.nan))
        expected = float(reference.get(key, math.nan))
        delta = actual - expected
        if not math.isfinite(delta):
            raise RuntimeError(f"cross-run point diagnostic is non-finite: {key}")
        raw_deltas[key] = delta
    direct_metrics = smu1.finalize_metric_state(dict(direct_state))
    reference_metrics = smu1.finalize_metric_state(dict(reference))
    score_keys = ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")
    score_deltas = {
        key: float(direct_metrics[key] - reference_metrics[key]) for key in score_keys
    }
    return {
        "reference": "cross-run UQ candidate_oof additive point metric state",
        "direct": {
            "point_metric_state": {key: direct_state[key] for key in keys},
            "point_metrics": {key: direct_metrics[key] for key in score_keys},
        },
        "uq_oof": {
            "point_metric_state": {key: reference[key] for key in keys},
            "point_metrics": {key: reference_metrics[key] for key in score_keys},
        },
        "raw_deltas_direct_minus_uq_oof": raw_deltas,
        "score_deltas_direct_minus_uq_oof": score_deltas,
        "observed_raw_deltas_all_zero": all(
            value == 0.0 for value in raw_deltas.values()
        ),
        "hard_gate": False,
        "exact_equivalence_claimed": False,
        "decision_effect": "diagnostic_only",
        "explanation": (
            "The same frozen point function was replayed in different executions: "
            "five fold-partitioned batch streams versus one sequential clean716 stream. "
            "Batch shapes, accumulation order, CUDA kernels, and hardware may change "
            "floating cumulative sums; identity is enforced by state/normalizer SHA."
        ),
    }


def worker_command(args: argparse.Namespace, kind: str, output: Path) -> list[str]:
    if kind not in {"direct", "package"}:
        raise ValueError(f"unknown replay worker {kind}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        f"--worker-{kind}",
        "--worker-output",
        str(output),
        "--refit-checkpoint",
        str(args.refit_checkpoint),
        "--refit-checkpoint-sha256",
        args.refit_checkpoint_sha256,
        "--refit-model-state-sha256",
        args.refit_model_state_sha256,
        "--refit-point-checkpoint-sha256",
        args.refit_point_checkpoint_sha256,
        "--refit-point-model-state-sha256",
        args.refit_point_model_state_sha256,
        "--refit-point-frozen-state-sha256",
        args.refit_point_frozen_state_sha256,
        "--refit-point-normalizer-sha256",
        args.refit_point_normalizer_sha256,
        "--refit-training-sha256",
        args.refit_training_sha256,
        "--promotion-gate-sha256",
        args.promotion_gate_sha256,
        "--clean-manifest",
        str(args.clean_manifest),
        "--real-dir",
        str(args.real_dir),
        "--device",
        args.device,
        "--hardware-class",
        args.hardware_class,
        "--acknowledge-worker",
        WORKER_ACKNOWLEDGEMENT,
    ]
    if kind == "package":
        command.extend(
            [
                "--archive",
                str(args.archive),
                "--archive-sha256",
                args.archive_sha256,
            ]
        )
    return command


def run_worker(command: Sequence[str], timeout: float) -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fresh replay worker failed: "
            f"returncode={completed.returncode}; stderr={completed.stderr[-4000:]}"
        )
    return {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def run_verify(args: argparse.Namespace) -> dict:
    identity = dr1.validate_formal_git_identity(dr1.git_identity())
    if identity["branch"] != FORMAL_BRANCH:
        raise RuntimeError("formal package verifier is on the wrong branch")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verifier output: {args.output}")
    device = torch.device(args.device)
    hardware = ft12.validate_hardware(device, args.hardware_class)
    started = time.perf_counter()
    deadline = started + WALL_BUDGET_SECONDS

    uq_result, uq_source = validate_uq_result(args.uq_result, args.uq_result_sha256)
    seal = uq_source["refit_endpoint"]
    if dr1.sha256(args.refit_checkpoint) != seal["file_sha256"]:
        raise RuntimeError("supplied refit checkpoint is not the UQ-result endpoint")
    args.refit_checkpoint_sha256 = seal["file_sha256"]
    args.refit_model_state_sha256 = seal["model_state_sha256"]
    args.refit_point_checkpoint_sha256 = seal["point_checkpoint_sha256"]
    args.refit_point_model_state_sha256 = seal["point_model_state_sha256"]
    args.refit_point_frozen_state_sha256 = seal["point_frozen_state_sha256"]
    args.refit_point_normalizer_sha256 = seal["point_normalizer_sha256"]
    args.refit_training_sha256 = seal["refit_training_sha256"]
    args.promotion_gate_sha256 = seal["promotion_gate_sha256"]
    archive, _payload = archive_identity(args.archive, args.archive_sha256)
    if dr1.sha256(args.clean_manifest) != uq_release.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean manifest file SHA changed")
    clean_rows, clean_report = dr1.load_manifest(
        args.clean_manifest, uq_release.FORMAL_MANIFEST_SEMANTIC_SHA256
    )
    _tail_rows, tail_report = uq_release.derive_clean716_manifest(clean_rows)
    if (
        clean_report["trajectories"] != uq_release.EXPECTED_TRAJECTORIES
        or tail_report["tail_windows"] != uq_release.EXPECTED_TAIL_WINDOWS
        or tail_report["semantic_sha256"]
        != uq_result["manifest"]["clean716"]["semantic_sha256"]
    ):
        raise RuntimeError("verifier clean716 manifest differs from UQ result")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="b210-clean716-package-verify-", dir=args.temp_root
    ) as temporary:
        root = Path(temporary)
        direct_path = root / "direct_worker.json"
        package_path = root / "package_worker.json"
        direct_process = run_worker(
            worker_command(args, "direct", direct_path),
            max(1.0, deadline - time.perf_counter()),
        )
        direct = _read_json_mapping(direct_path)
        package_process = run_worker(
            worker_command(args, "package", package_path),
            max(1.0, deadline - time.perf_counter()),
        )
        package = _read_json_mapping(package_path)
    alignment = validate_worker_alignment(direct, package)
    uq_point_diagnostic = direct_uq_point_diagnostic(direct, uq_result)
    comparison = metric_comparison(package["metric_state"], direct["metric_state"])
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract_sha256": CONTRACT_SHA256,
        "evidence_class": CONTRACT["evidence_class"],
        "sources": {
            "archive": archive,
            "uq_release_result": uq_source,
            "refit_checkpoint": {
                "path": str(args.refit_checkpoint.resolve()),
                "file_sha256": seal["file_sha256"],
                "model_state_sha256": seal["model_state_sha256"],
                "frozen_state_sha256": seal["point_frozen_state_sha256"],
                "normalizer_sha256": seal["point_normalizer_sha256"],
            },
            "clean_manifest": {
                "path": str(args.clean_manifest.resolve()),
                "file_sha256": uq_release.FORMAL_MANIFEST_FILE_SHA256,
                "semantic_sha256": clean_report["semantic_sha256"],
                "clean716_semantic_sha256": tail_report["semantic_sha256"],
            },
        },
        "workers": {
            "direct": {"process": direct_process, "result": direct},
            "package": {"process": package_process, "result": package},
        },
        "alignment": alignment,
        "uq_point_cross_run_diagnostic": uq_point_diagnostic,
        "comparison": comparison,
        "git": identity,
        "hardware": hardware,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "elapsed_seconds": time.perf_counter() - started,
        "int8_threshold_tuning_authorized": False,
        "zip_written": False,
        "submission_authorized": False,
        "decision": comparison["decision"],
    }
    dr1.write_json_atomic(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--contract-only", action="store_true")
    mode.add_argument("--worker-direct", action="store_true")
    mode.add_argument("--worker-package", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--uq-result", type=Path)
    parser.add_argument("--uq-result-sha256")
    parser.add_argument("--refit-checkpoint", type=Path)
    parser.add_argument("--refit-checkpoint-sha256")
    parser.add_argument("--refit-model-state-sha256")
    parser.add_argument("--refit-point-checkpoint-sha256")
    parser.add_argument("--refit-point-model-state-sha256")
    parser.add_argument("--refit-point-frozen-state-sha256")
    parser.add_argument("--refit-point-normalizer-sha256")
    parser.add_argument("--refit-training-sha256")
    parser.add_argument("--promotion-gate-sha256")
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--temp-root", type=Path, default=Path("runs/tmp"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hardware-class",
        choices=("rtx4080_super", "rtx4090"),
        default="rtx4080_super",
    )
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    parser.add_argument("--acknowledge-worker", default="")
    return parser.parse_args()


def _required(args: argparse.Namespace, names: Sequence[str], label: str) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"{label} missing {missing}")


def validate_args(args: argparse.Namespace) -> None:
    if args.verify:
        _required(
            args,
            (
                "archive",
                "archive_sha256",
                "uq_result",
                "uq_result_sha256",
                "refit_checkpoint",
                "clean_manifest",
                "output",
            ),
            "package verifier",
        )
        args.archive_sha256 = dr1.normalized_sha256(args.archive_sha256, "archive SHA")
        args.uq_result_sha256 = dr1.normalized_sha256(
            args.uq_result_sha256, "UQ result SHA"
        )
        if (
            not args.execute_formal_gpu
            or args.acknowledge_protocol != FORMAL_ACKNOWLEDGEMENT
        ):
            raise ValueError("formal package verification requires acknowledgement")
        if any(
            (
                args.worker_output,
                args.acknowledge_worker,
                args.refit_checkpoint_sha256,
                args.refit_model_state_sha256,
                args.refit_point_checkpoint_sha256,
                args.refit_point_model_state_sha256,
                args.refit_point_frozen_state_sha256,
                args.refit_point_normalizer_sha256,
                args.refit_training_sha256,
                args.promotion_gate_sha256,
            )
        ):
            raise ValueError("controller received worker-only arguments")
    elif args.worker_direct or args.worker_package:
        names = [
            "worker_output",
            "refit_checkpoint",
            "refit_checkpoint_sha256",
            "refit_model_state_sha256",
            "refit_point_checkpoint_sha256",
            "refit_point_model_state_sha256",
            "refit_point_frozen_state_sha256",
            "refit_point_normalizer_sha256",
            "refit_training_sha256",
            "promotion_gate_sha256",
            "clean_manifest",
        ]
        if args.worker_package:
            names.extend(("archive", "archive_sha256"))
        _required(args, names, "replay worker")
        args.refit_checkpoint_sha256 = dr1.normalized_sha256(
            args.refit_checkpoint_sha256, "refit checkpoint SHA"
        )
        args.refit_model_state_sha256 = dr1.normalized_sha256(
            args.refit_model_state_sha256, "refit model-state SHA"
        )
        args.refit_point_checkpoint_sha256 = dr1.normalized_sha256(
            args.refit_point_checkpoint_sha256, "refit point checkpoint SHA"
        )
        args.refit_point_model_state_sha256 = dr1.normalized_sha256(
            args.refit_point_model_state_sha256, "refit point model-state SHA"
        )
        args.refit_point_frozen_state_sha256 = dr1.normalized_sha256(
            args.refit_point_frozen_state_sha256, "refit point frozen-state SHA"
        )
        args.refit_point_normalizer_sha256 = dr1.normalized_sha256(
            args.refit_point_normalizer_sha256, "refit point normalizer SHA"
        )
        args.refit_training_sha256 = dr1.normalized_sha256(
            args.refit_training_sha256, "refit training SHA"
        )
        args.promotion_gate_sha256 = dr1.normalized_sha256(
            args.promotion_gate_sha256, "promotion gate SHA"
        )
        if args.worker_package:
            args.archive_sha256 = dr1.normalized_sha256(
                args.archive_sha256, "archive SHA"
            )
        if args.acknowledge_worker != WORKER_ACKNOWLEDGEMENT:
            raise ValueError("replay worker lacks internal acknowledgement")
        if args.worker_direct and (args.archive or args.archive_sha256):
            raise ValueError("direct worker received package-only arguments")
        if any(
            (
                args.uq_result,
                args.uq_result_sha256,
                args.output,
                args.execute_formal_gpu,
                args.acknowledge_protocol,
            )
        ):
            raise ValueError("worker received controller-only arguments")
    elif any(
        (
            args.archive,
            args.archive_sha256,
            args.uq_result,
            args.uq_result_sha256,
            args.refit_checkpoint,
            args.refit_checkpoint_sha256,
            args.refit_model_state_sha256,
            args.refit_point_checkpoint_sha256,
            args.refit_point_model_state_sha256,
            args.refit_point_frozen_state_sha256,
            args.refit_point_normalizer_sha256,
            args.refit_training_sha256,
            args.promotion_gate_sha256,
            args.clean_manifest,
            args.output,
            args.worker_output,
            args.execute_formal_gpu,
            args.acknowledge_protocol,
            args.acknowledge_worker,
        )
    ):
        raise ValueError("contract-only received execution arguments")


def write_stop(path: Path, error: BaseException) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    dr1.write_json_atomic(
        path,
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "contract_sha256": CONTRACT_SHA256,
            "decision": "STOP_B210_CLEAN716_EXACT_INT8_PACKAGE_VERIFY_RUNTIME",
            "error_type": type(error).__name__,
            "error": str(error),
            "int8_threshold_tuning_authorized": False,
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
                    "schema": SCHEMA,
                    "experiment": EXPERIMENT,
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
        if args.worker_direct:
            result = worker_direct(args)
        elif args.worker_package:
            result = worker_package(args)
        else:
            result = run_verify(args)
    except BaseException as error:
        if args.verify and args.output is not None:
            write_stop(args.output, error)
        raise
    print(
        json.dumps(
            {
                "event": "b210_clean716_package_verify_complete",
                "kind": result.get("kind", "controller"),
                "decision": result.get("decision", "WORKER_COMPLETE"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
