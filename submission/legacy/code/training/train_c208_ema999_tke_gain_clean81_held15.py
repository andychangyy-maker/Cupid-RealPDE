#!/usr/bin/env python3
"""Same-run raw/EMA plus fit-only global TKE-gain held-AoA15 gate.

The raw and FP32 EMA endpoints are sealed first.  A single positive global
fluctuation gain is then fit from already-open other-four-AoA prefixes and
sealed as metadata.  Held-AoA15 prefixes cannot open until all three seals
exist.  This program never constructs legacy 2576/725 or clean716 windows.
"""

from __future__ import annotations

import argparse
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
import train_c208_ema999_clean81_held10 as ema10  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_ensemble_unet2d as ensemble_train  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


EXPERIMENT = "c208_ema999_global_tke_gain_clean81_held15"
SCHEMA = "c208_ema999_tke_gain_clean81_held15_v1"
SEALED_ENDPOINT_SCHEMA = "c208_ema999_tke_gain_endpoint_v1"
GAIN_METADATA_SCHEMA = "c208_ema999_global_tke_gain_fit_v1"
PRE_SCORING_SCHEMA = "c208_ema999_tke_gain_pre_scoring_seal_v1"
PARTIAL_RESULT_SCHEMA = "c208_ema999_tke_gain_partial_result_v1"
RUN_OWNER_SCHEMA = "c208_ema999_tke_gain_run_owner_v1"
FORMAL_BRANCH = "codex/c208-ema999-gain-held15"
FORMAL_START_SHA256 = ema10.FORMAL_START_SHA256
HELD_AOA = 15
EMA_DECAY = ema10.EMA_DECAY
UPDATES = dr1.UPDATES
SEED = dr1.SEED
OFFICIAL_TKE_DENOMINATOR_EPSILON = 1e-8
MEAN_INVARIANCE_ATOL = 2e-6
MVPE_SCORE_INVARIANCE_ATOL = 1e-5
TKE_SCALING_ATOL = 1e-10
TKE_SCALING_RTOL = 2e-5
BOUND_SHIFT_ATOL = 1e-12
WLS_OBJECTIVE_ATOL = 1e-9
WALL_BUDGET_SECONDS = 3600.0


CONTRACT = {
    "evidence_class": (
        "same-run raw/EMA plus fit-only global physical-TKE gain held-AoA15"),
    "start_sim_sha256": FORMAL_START_SHA256,
    "held_aoa": HELD_AOA,
    "training": {
        "updates": UPDATES,
        "batch_size": 8,
        "seed": SEED,
        "lr": smu1.LR,
        "weight_decay": smu1.WEIGHT_DECAY,
        "scheduler_t_max": smu1.SCHEDULER_T_MAX,
        "grad_clip": smu1.GRAD_CLIP,
    },
    "ema": {
        "decay": EMA_DECAY,
        "updates": UPDATES,
        "update_order": ema_lib.UPDATE_ORDER_CONTRACT,
        "shadow_dtype": str(torch.float32),
        "decay_sweep": False,
    },
    "gain": {
        "scope": "one global scalar fit on other-four prefix targets only",
        "parameterization": "s=g^2; g positive",
        "objective": (
            "sum_i ||s*x_i-y_i||_2^2 / max(||y_i||_2,1e-8)^2"),
        "closed_form": "A=sum_i w_i<x_i,x_i>; B=sum_i w_i<x_i,y_i>; s=B/A",
        "tke_field_dtype": "official float32 physical TKE",
        "accumulator_dtype": "float64",
        "grid": False,
        "range_clip": False,
        "objective_improvement_atol": WLS_OBJECTIVE_ATOL,
    },
    "transport": {
        "prediction": "mean_t(EMA)+g*(EMA-mean_t(EMA))",
        "ema_normalized_extents": "bitwise reused; signed and asymmetric",
        "pressure": "bitwise unchanged from EMA",
    },
    "data": {
        "trajectories": dr1.EXPECTED_TRAJECTORIES,
        "prefix_windows": dr1.EXPECTED_WINDOWS,
        "fit_windows": dr1.EXPECTED_FIT_COUNTS[HELD_AOA],
        "held_windows": dr1.EXPECTED_AOA_COUNTS[HELD_AOA],
        "manifest_sha256": dr1.FORMAL_MANIFEST_SHA256,
        "excluded": dr1.EXCLUDED_BASENAME,
        "tail_arrays_read": 0,
    },
    "gate": {
        "primary_vs_raw_tke_strictly_positive": True,
        "primary_vs_ema_tke_strictly_positive": True,
        "primary_vs_raw_point_strictly_positive": True,
        "primary_vs_raw_exact_fixed_time_final_strictly_positive": True,
        "primary_vs_raw_rel_floor": -0.05,
        "primary_vs_raw_mvpe_floor": -0.05,
        "primary_vs_ema_mvpe_abs_tolerance": MVPE_SCORE_INVARIANCE_ATOL,
        "primary_vs_raw_sps_floor": -0.10,
    },
    "release_checkpoint_written": False,
    "submission_authorized": False,
    "wall_budget_seconds": WALL_BUDGET_SECONDS,
}


def _float32_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _update_array_hash(digest, label: str, value: np.ndarray) -> None:
    array = _float32_array(value)
    digest.update(label.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _update_raw_tensor_hash(digest, value: torch.Tensor) -> None:
    """Match the inherited held-target audit byte contract exactly."""
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))


@torch.inference_mode()
def audit_ordered_targets(dataset, indices: Sequence[int]) -> dict:
    """Hash the complete held target/mask stream before loading an endpoint."""
    loader = DataLoader(
        dataset, batch_sampler=[
            list(indices[start:start + smu1.EVAL_BATCH_SIZE])
            for start in range(0, len(indices), smu1.EVAL_BATCH_SIZE)],
        num_workers=0, pin_memory=False)
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    audited_samples = 0
    audited_scored = 0
    for _history, target in loader:
        target_cpu = target.contiguous()
        scored_cpu = target_cpu[..., :2].ne(0.0).to(torch.uint8).contiguous()
        _update_raw_tensor_hash(target_hasher, target_cpu)
        _update_raw_tensor_hash(mask_hasher, scored_cpu)
        audited_samples += len(target_cpu)
        audited_scored += int(scored_cpu.sum())
    if audited_samples != len(indices):
        raise RuntimeError("held target audit did not consume one partition")
    return {
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
        "samples": audited_samples,
        "scored_elements": audited_scored,
    }


def wall_deadline_audit(started: float, deadline: float) -> dict:
    checked = time.perf_counter()
    if not all(math.isfinite(value) for value in (started, deadline, checked)):
        raise FloatingPointError("wall deadline audit is non-finite")
    if deadline < started:
        raise ValueError("wall deadline precedes run start")
    return {
        "wall_budget_seconds": float(deadline - started),
        "checked_elapsed_seconds": float(checked - started),
        "within_wall_budget": bool(checked <= deadline),
    }


def official_float32_tke_field(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return the official physical u/v TKE field in float32."""
    array = _float32_array(value)
    if array.ndim != 5 or array.shape[1] < 2 or array.shape[-1] < 2:
        raise ValueError("TKE input must have shape (N,T,H,W,C>=2)")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("TKE input is non-finite")
    result = np.asarray(
        smu1.full.OFFICIAL.kinetic_energy(array[..., :2]),
        dtype=np.float32)
    if result.shape != (array.shape[0], array.shape[2], array.shape[3]):
        raise RuntimeError("official TKE field shape changed")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("official TKE field is invalid")
    return result


def tke_wls_sufficient_statistics(
        predicted_tke: np.ndarray, target_tke: np.ndarray,
        ) -> dict[str, float | int]:
    """Accumulate the per-sample d^-2 WLS normal equation in FP64."""
    predicted = np.asarray(predicted_tke, dtype=np.float32)
    target = np.asarray(target_tke, dtype=np.float32)
    if predicted.shape != target.shape or predicted.ndim < 2:
        raise ValueError("predicted/target TKE fields must be aligned batches")
    if (not np.all(np.isfinite(predicted))
            or not np.all(np.isfinite(target))):
        raise FloatingPointError("TKE WLS fields are non-finite")
    if np.any(predicted < 0.0) or np.any(target < 0.0):
        raise ValueError("TKE WLS fields must be non-negative")
    x = predicted.reshape(predicted.shape[0], -1).astype(np.float64)
    y = target.reshape(target.shape[0], -1).astype(np.float64)
    target_norm = np.linalg.norm(y, axis=1).clip(
        min=OFFICIAL_TKE_DENOMINATOR_EPSILON)
    weight = np.square(np.reciprocal(target_norm))
    x_sq = np.einsum("ij,ij->i", x, x, dtype=np.float64)
    xy = np.einsum("ij,ij->i", x, y, dtype=np.float64)
    y_sq = np.einsum("ij,ij->i", y, y, dtype=np.float64)
    return {
        "samples": int(len(x)),
        "A": float(np.sum(weight * x_sq, dtype=np.float64)),
        "B": float(np.sum(weight * xy, dtype=np.float64)),
        "C": float(np.sum(weight * y_sq, dtype=np.float64)),
    }


def solve_positive_tke_gain(A: float, B: float, C: float) -> dict[str, float]:
    """Solve s=B/A and g=sqrt(max(0,s)) without a tuned range."""
    A, B, C = float(A), float(B), float(C)
    if not all(math.isfinite(value) for value in (A, B, C)):
        raise FloatingPointError("TKE gain WLS sufficient statistics are non-finite")
    if A <= 0.0:
        raise ValueError("TKE gain WLS denominator A must be positive")
    if B < 0.0:
        raise ValueError("TKE gain WLS numerator B must be non-negative")
    s0 = B / A
    if not math.isfinite(s0):
        raise FloatingPointError("unconstrained TKE scale is non-finite")
    s = max(0.0, s0)
    gain = math.sqrt(s)
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("closed-form TKE fluctuation gain must be positive")
    objective_identity = A - 2.0 * B + C
    objective_optimum = A * s * s - 2.0 * B * s + C
    if not all(math.isfinite(value) for value in (
            objective_identity, objective_optimum)):
        raise FloatingPointError("TKE gain WLS objective is non-finite")
    if objective_optimum > objective_identity + WLS_OBJECTIVE_ATOL:
        raise RuntimeError("closed-form TKE gain does not improve WLS objective")
    return {
        "A": A,
        "B": B,
        "C": C,
        "s_unconstrained": s0,
        "s": s,
        "g": gain,
        "objective_at_s_1": objective_identity,
        "objective_at_s": objective_optimum,
        "normal_equation_residual": A * s - B,
        "objective_improvement_atol": WLS_OBJECTIVE_ATOL,
    }


def apply_global_fluctuation_gain(
        prediction: torch.Tensor | np.ndarray, gain: float,
        ) -> np.ndarray:
    """Scale only physical u/v temporal fluctuations; preserve pressure."""
    if not math.isfinite(float(gain)) or float(gain) <= 0.0:
        raise ValueError("fluctuation gain must be finite and positive")
    source = _float32_array(prediction)
    if source.ndim != 5 or source.shape[-1] < 2:
        raise ValueError("gain input must have shape (N,T,H,W,C>=2)")
    result = source.copy()
    mean = np.mean(source[..., :2], axis=1, keepdims=True, dtype=np.float32)
    result[..., :2] = (
        mean + np.float32(gain) * (source[..., :2] - mean))
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("gained prediction is non-finite")
    return result


def new_transport_audit() -> dict[str, float | int | bool]:
    return {
        "samples": 0,
        "temporal_mean_max_abs_delta": 0.0,
        "tke_scale_max_abs_error": 0.0,
        "tke_scale_max_expected_abs": 0.0,
        "tke_scale_observed_numerator": 0.0,
        "tke_scale_observed_denominator": 0.0,
        "pressure_bitwise": True,
        "lower_extent_bitwise": True,
        "upper_extent_bitwise": True,
        "width_max_abs_delta": 0.0,
        "asymmetry_max_abs_delta": 0.0,
        "lower_bound_shift_max_abs_error": 0.0,
        "upper_bound_shift_max_abs_error": 0.0,
    }


def update_transport_audit(
        state: dict, ema_prediction: np.ndarray, gained_prediction: np.ndarray,
        lower_extent: np.ndarray, upper_extent: np.ndarray, s: float,
        ) -> None:
    ema_prediction = _float32_array(ema_prediction)
    gained_prediction = _float32_array(gained_prediction)
    lower_extent = _float32_array(lower_extent)
    upper_extent = _float32_array(upper_extent)
    if ema_prediction.shape != gained_prediction.shape:
        raise ValueError("EMA/gained predictions are not aligned")
    expected_extent_shape = (*ema_prediction.shape[:-1], 2)
    if (lower_extent.shape != expected_extent_shape
            or upper_extent.shape != expected_extent_shape):
        raise ValueError("EMA interval extents are not aligned")

    ema_mean = np.mean(
        ema_prediction[..., :2], axis=1, dtype=np.float32)
    gained_mean = np.mean(
        gained_prediction[..., :2], axis=1, dtype=np.float32)
    state["temporal_mean_max_abs_delta"] = max(
        float(state["temporal_mean_max_abs_delta"]),
        float(np.max(np.abs(gained_mean - ema_mean))))

    ema_tke = official_float32_tke_field(ema_prediction).astype(np.float64)
    gained_tke = official_float32_tke_field(
        gained_prediction).astype(np.float64)
    expected_tke = float(s) * ema_tke
    error = np.abs(gained_tke - expected_tke)
    state["tke_scale_max_abs_error"] = max(
        float(state["tke_scale_max_abs_error"]), float(np.max(error)))
    state["tke_scale_max_expected_abs"] = max(
        float(state["tke_scale_max_expected_abs"]),
        float(np.max(np.abs(expected_tke))))
    state["tke_scale_observed_numerator"] += float(np.sum(
        ema_tke * gained_tke, dtype=np.float64))
    state["tke_scale_observed_denominator"] += float(np.sum(
        ema_tke * ema_tke, dtype=np.float64))

    if ema_prediction.shape[-1] > 2:
        state["pressure_bitwise"] = bool(
            state["pressure_bitwise"]
            and np.array_equal(
                gained_prediction[..., 2:], ema_prediction[..., 2:]))

    candidate_lower = lower_extent
    candidate_upper = upper_extent
    state["lower_extent_bitwise"] = bool(
        state["lower_extent_bitwise"]
        and np.array_equal(candidate_lower, lower_extent))
    state["upper_extent_bitwise"] = bool(
        state["upper_extent_bitwise"]
        and np.array_equal(candidate_upper, upper_extent))
    state["width_max_abs_delta"] = max(
        float(state["width_max_abs_delta"]),
        float(np.max(np.abs(
            (candidate_lower + candidate_upper)
            - (lower_extent + upper_extent)))))
    state["asymmetry_max_abs_delta"] = max(
        float(state["asymmetry_max_abs_delta"]),
        float(np.max(np.abs(
            (candidate_upper - candidate_lower)
            - (upper_extent - lower_extent)))))

    ema_uv = ema_prediction[..., :2].astype(np.float64)
    gained_uv = gained_prediction[..., :2].astype(np.float64)
    lower64 = lower_extent.astype(np.float64)
    upper64 = upper_extent.astype(np.float64)
    point_shift = gained_uv - ema_uv
    lower_shift = (
        (gained_uv - lower64) - (ema_uv - lower64))
    upper_shift = (
        (gained_uv + upper64) - (ema_uv + upper64))
    state["lower_bound_shift_max_abs_error"] = max(
        float(state["lower_bound_shift_max_abs_error"]),
        float(np.max(np.abs(lower_shift - point_shift))))
    state["upper_bound_shift_max_abs_error"] = max(
        float(state["upper_bound_shift_max_abs_error"]),
        float(np.max(np.abs(upper_shift - point_shift))))
    state["samples"] += int(len(ema_prediction))


def finalize_transport_audit(state: Mapping[str, object], s: float) -> dict:
    result = dict(state)
    denominator = float(result["tke_scale_observed_denominator"])
    observed_s = (
        float(result["tke_scale_observed_numerator"]) / denominator
        if denominator > 0.0 else math.nan)
    bound = (
        TKE_SCALING_ATOL
        + TKE_SCALING_RTOL * float(result["tke_scale_max_expected_abs"]))
    conditions = {
        "samples_positive": int(result["samples"]) > 0,
        "mean_invariant": (
            float(result["temporal_mean_max_abs_delta"])
            <= MEAN_INVARIANCE_ATOL),
        "tke_scales_as_s": (
            float(result["tke_scale_max_abs_error"]) <= bound),
        "observed_s_finite": math.isfinite(observed_s),
        "pressure_bitwise": bool(result["pressure_bitwise"]),
        "lower_extent_bitwise": bool(result["lower_extent_bitwise"]),
        "upper_extent_bitwise": bool(result["upper_extent_bitwise"]),
        "width_unchanged": float(result["width_max_abs_delta"]) == 0.0,
        "asymmetry_unchanged": (
            float(result["asymmetry_max_abs_delta"]) == 0.0),
        "lower_bound_shift_equals_point_shift": (
            float(result["lower_bound_shift_max_abs_error"])
            <= BOUND_SHIFT_ATOL),
        "upper_bound_shift_equals_point_shift": (
            float(result["upper_bound_shift_max_abs_error"])
            <= BOUND_SHIFT_ATOL),
    }
    result.update({
        "expected_s": float(s),
        "observed_s_l2": observed_s,
        "tke_scale_error_bound": bound,
        "conditions": conditions,
        "pass": all(conditions.values()),
    })
    return result


def save_endpoint(
        path: Path, model: EnsembleUNet2d, arm: str, start: dict,
        normalizer: smu1.FoldNormalizer, stream_sha256: str,
        extra: dict | None = None,
        ) -> dict:
    state = ensemble_train.common.checkpoint_state(model)
    state_sha = dr1.model_state_sha256(state)
    payload = {
        "schema": SEALED_ENDPOINT_SCHEMA,
        "phase": "real",
        "iteration": UPDATES,
        "arm": arm,
        "held_aoa": HELD_AOA,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "start_sim_checkpoint_sha256": start["checkpoint_sha256"],
        "start_sim_model_state_sha256": start["model_state_sha256"],
        "sample_stream_sha256": stream_sha256,
        "normalizer": ensemble_train.common.normalizer_dict(normalizer),
        "held_prefix_opened_before_seal": False,
        "extra": extra or {},
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "iteration": UPDATES,
    }


def load_endpoint(
        path: Path, seal: Mapping[str, object],
        normalizer: smu1.FoldNormalizer, device: torch.device,
        ) -> EnsembleUNet2d:
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("sealed held15 endpoint file drifted")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != SEALED_ENDPOINT_SCHEMA
            or payload.get("held_aoa") != HELD_AOA
            or payload.get("iteration") != UPDATES):
        raise ValueError("invalid held15 endpoint contract")
    state = payload["model_state_dict"]
    if (payload.get("model_state_sha256") != seal["model_state_sha256"]
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError("sealed held15 endpoint tensor state drifted")
    stored = dict(payload["model_config"])
    if stored.pop("architecture", None) != "ensemble_unet":
        raise ValueError("sealed held15 endpoint architecture changed")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
    model.load_state_dict(state, strict=True)
    ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


@torch.inference_mode()
def fit_global_tke_gain(
        model: EnsembleUNet2d, dataset, indices: Sequence[int],
        normalizer: smu1.FoldNormalizer, device: torch.device,
        deadline: float,
        ) -> dict:
    loader = DataLoader(
        dataset, batch_sampler=[
            list(indices[start:start + smu1.EVAL_BATCH_SIZE])
            for start in range(0, len(indices), smu1.EVAL_BATCH_SIZE)],
        num_workers=0, pin_memory=True)
    totals = {"samples": 0, "A": 0.0, "B": 0.0, "C": 0.0}
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    prediction_hasher = hashlib.sha256()
    audited_scored = 0
    model.eval()
    for history, target in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("held15 TKE-gain fit exceeded wall budget")
        target_cpu = target.contiguous()
        scored_cpu = target_cpu[..., :2].ne(0.0).to(torch.uint8).contiguous()
        _update_raw_tensor_hash(target_hasher, target_cpu)
        _update_raw_tensor_hash(mask_hasher, scored_cpu)
        audited_scored += int(scored_cpu.sum())
        history = history.to(device, non_blocking=True)
        target_device = target.to(device, non_blocking=True)
        history_norm, _target_norm = normalizer.preprocess(
            history, target_device)
        prediction_norm, _lower, _upper = model.forward_distribution(
            history_norm)
        prediction = normalizer.postprocess_prediction(
            prediction_norm.float())
        prediction_np = _float32_array(prediction)
        target_np = _float32_array(target_cpu)
        _update_array_hash(prediction_hasher, "prediction", prediction_np)
        row = tke_wls_sufficient_statistics(
            official_float32_tke_field(prediction_np),
            official_float32_tke_field(target_np))
        totals["samples"] += int(row["samples"])
        for key in ("A", "B", "C"):
            totals[key] += float(row[key])
    if totals["samples"] != len(indices):
        raise RuntimeError("TKE-gain fit did not consume all fit windows")
    solution = solve_positive_tke_gain(
        totals["A"], totals["B"], totals["C"])
    return {
        "method": "global_positive_physical_tke_fluctuation_gain",
        "samples": totals["samples"],
        "official_tke_denominator_epsilon": (
            OFFICIAL_TKE_DENOMINATOR_EPSILON),
        "tke_field_dtype": "float32",
        "accumulator_dtype": "float64",
        "wls": solution,
        "ordered_prediction_sha256": prediction_hasher.hexdigest(),
        "target_audit": {
            "ordered_raw_target_sha256": target_hasher.hexdigest(),
            "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
            "samples": totals["samples"],
            "scored_elements": audited_scored,
        },
        "held_targets_used": False,
        "grid_used": False,
        "range_clip_used": False,
    }


def save_gain_metadata(
        path: Path, fit: dict, raw_seal: Mapping[str, object],
        ema_seal: Mapping[str, object], normalizer_sha256: str,
        stream_sha256: str, git_commit: str,
        ) -> dict:
    payload = {
        "schema": GAIN_METADATA_SCHEMA,
        "held_aoa": HELD_AOA,
        "fit_partition": "clean81 other-four prefix windows only",
        "fit_samples": dr1.EXPECTED_FIT_COUNTS[HELD_AOA],
        "held_targets_used": False,
        "raw_endpoint_file_sha256": raw_seal["file_sha256"],
        "raw_endpoint_model_state_sha256": raw_seal["model_state_sha256"],
        "ema_endpoint_file_sha256": ema_seal["file_sha256"],
        "ema_endpoint_model_state_sha256": ema_seal["model_state_sha256"],
        "normalizer_statistics_sha256": normalizer_sha256,
        "sample_stream_sha256": stream_sha256,
        "source_git_commit": git_commit,
        "fit": fit,
    }
    dr1.write_json_atomic(path, payload)
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "sealed_monotonic_ns": time.monotonic_ns(),
        "schema": GAIN_METADATA_SCHEMA,
        "g": float(fit["wls"]["g"]),
        "s": float(fit["wls"]["s"]),
    }


def load_gain_metadata(
        path: Path, seal: Mapping[str, object],
        raw_seal: Mapping[str, object], ema_seal: Mapping[str, object],
        normalizer_sha256: str, stream_sha256: str, git_commit: str,
        ) -> dict:
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("sealed TKE-gain metadata drifted")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema") != GAIN_METADATA_SCHEMA
            or payload.get("held_aoa") != HELD_AOA
            or payload.get("held_targets_used") is not False):
        raise ValueError("invalid sealed TKE-gain metadata")
    expected = {
        "raw_endpoint_file_sha256": raw_seal["file_sha256"],
        "raw_endpoint_model_state_sha256": raw_seal["model_state_sha256"],
        "ema_endpoint_file_sha256": ema_seal["file_sha256"],
        "ema_endpoint_model_state_sha256": ema_seal["model_state_sha256"],
        "normalizer_statistics_sha256": normalizer_sha256,
        "sample_stream_sha256": stream_sha256,
        "source_git_commit": git_commit,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("sealed TKE-gain provenance metadata drifted")
    fit = payload.get("fit")
    if (not isinstance(fit, dict)
            or int(fit.get("samples", -1))
            != dr1.EXPECTED_FIT_COUNTS[HELD_AOA]
            or fit.get("held_targets_used") is not False):
        raise ValueError("sealed TKE-gain fit contract changed")
    if (float(fit["wls"]["g"]) != float(seal["g"])
            or float(fit["wls"]["s"]) != float(seal["s"])):
        raise RuntimeError("sealed TKE-gain scalar metadata drifted")
    return payload


def write_partial_result(path: Path, stage: str, **evidence) -> dict:
    payload = {
        "schema": PARTIAL_RESULT_SCHEMA,
        "experiment": EXPERIMENT,
        "stage": stage,
        "held_aoa": HELD_AOA,
        **evidence,
        "release_checkpoint_written": False,
        "submission_authorized": False,
    }
    dr1.write_json_atomic(path, payload)
    return payload


def staged_file_evidence(path: Path) -> dict | None:
    """Read one file created by this run without hiding handler failures."""
    if not path.is_file():
        return None
    result = {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
    }
    try:
        result["payload"] = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # Preserve the file hash even if JSON is damaged.
        result["read_error_type"] = type(exc).__name__
        result["read_error"] = str(exc)
    return result


@torch.inference_mode()
def evaluate_ema_and_gain(
        model: EnsembleUNet2d, dataset, indices: Sequence[int],
        normalizer: smu1.FoldNormalizer, device: torch.device,
        deadline: float, gain: float, s: float,
        ) -> tuple[dict, dict, dict]:
    loader = DataLoader(
        dataset, batch_sampler=[
            list(indices[start:start + smu1.EVAL_BATCH_SIZE])
            for start in range(0, len(indices), smu1.EVAL_BATCH_SIZE)],
        num_workers=0, pin_memory=True)
    ema_state = smu1.new_metric_state()
    gained_state = smu1.new_metric_state()
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    extent_hasher = hashlib.sha256()
    ema_prediction_hasher = hashlib.sha256()
    gained_prediction_hasher = hashlib.sha256()
    audit_state = new_transport_audit()
    model.eval()
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    audited_scored = 0
    for history, target in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("held15 scoring exceeded wall budget")
        target_cpu = target.contiguous()
        scored_cpu = target_cpu[..., :2].ne(0.0).to(torch.uint8).contiguous()
        _update_raw_tensor_hash(target_hasher, target_cpu)
        _update_raw_tensor_hash(mask_hasher, scored_cpu)
        audited_scored += int(scored_cpu.sum())
        history = history.to(device, non_blocking=True)
        target_device = target.to(device, non_blocking=True)
        history_norm, _target_norm = normalizer.preprocess(
            history, target_device)
        prediction_norm, lower_norm, upper_norm = model.forward_distribution(
            history_norm)
        prediction = normalizer.postprocess_prediction(
            prediction_norm.float())
        lower_extent = lower_norm.float() * scale
        upper_extent = upper_norm.float() * scale
        prediction_np = _float32_array(prediction)
        gained_np = apply_global_fluctuation_gain(prediction_np, gain)
        lower_np = _float32_array(lower_extent)
        upper_np = _float32_array(upper_extent)
        _update_array_hash(
            ema_prediction_hasher, "ema_prediction", prediction_np)
        _update_array_hash(
            gained_prediction_hasher, "gained_prediction", gained_np)
        _update_array_hash(extent_hasher, "lower_normalized", lower_norm)
        _update_array_hash(extent_hasher, "upper_normalized", upper_norm)
        update_transport_audit(
            audit_state, prediction_np, gained_np, lower_np, upper_np, s)
        smu1.update_metric_state(
            ema_state, torch.from_numpy(prediction_np), target_cpu,
            torch.from_numpy(lower_np), torch.from_numpy(upper_np))
        smu1.update_metric_state(
            gained_state, torch.from_numpy(gained_np), target_cpu,
            torch.from_numpy(lower_np), torch.from_numpy(upper_np))
    if (int(ema_state["samples"]) != len(indices)
            or int(gained_state["samples"]) != len(indices)
            or int(ema_state["sps_scored"]) != audited_scored
            or int(gained_state["sps_scored"]) != audited_scored):
        raise RuntimeError("held15 EMA/gain scorer did not consume one partition")
    target_audit = {
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
        "samples": len(indices),
        "scored_elements": audited_scored,
    }
    transport = finalize_transport_audit(audit_state, s)
    transport.update({
        "ema_prediction_sha256": ema_prediction_hasher.hexdigest(),
        "gained_prediction_sha256": gained_prediction_hasher.hexdigest(),
        "ema_normalized_extent_sha256": extent_hasher.hexdigest(),
        "primary_normalized_extent_sha256": extent_hasher.hexdigest(),
    })
    if not transport["pass"]:
        raise RuntimeError("held15 gain transport invariants failed")
    return ({
        "metrics": smu1.finalize_metric_state(ema_state),
        "metric_state": ema_state,
        "held_target_audit": target_audit,
    }, {
        "metrics": smu1.finalize_metric_state(gained_state),
        "metric_state": gained_state,
        "held_target_audit": target_audit,
    }, transport)


def primary_gate(
        primary_vs_raw: Mapping[str, object],
        primary_vs_ema: Mapping[str, object], transport: Mapping[str, object],
        ) -> dict:
    raw_delta = primary_vs_raw["delta"]
    ema_delta = primary_vs_ema["delta"]
    conditions = {
        "tke_positive_vs_raw": float(raw_delta["tke_rel_l2_score"]) > 0.0,
        "tke_positive_vs_ema": float(ema_delta["tke_rel_l2_score"]) > 0.0,
        "point_positive_vs_raw": float(raw_delta["point_score"]) > 0.0,
        "exact_fixed_time_final_positive_vs_raw": (
            float(primary_vs_raw["exact_non_time_final_delta"]) > 0.0),
        "rel_floor_vs_raw": float(raw_delta["rel_l2_score"]) >= -0.05,
        "mvpe_floor_vs_raw": float(raw_delta["mvpe_rel_l2_score"]) >= -0.05,
        "mvpe_invariant_vs_ema": (
            abs(float(ema_delta["mvpe_rel_l2_score"]))
            <= MVPE_SCORE_INVARIANCE_ATOL),
        "sps_floor_vs_raw": float(raw_delta["native_sps_score"]) >= -0.10,
        "transport_invariants": bool(transport["pass"]),
    }
    passed = all(conditions.values())
    return {
        "pass": passed,
        "conditions": conditions,
        "decision": (
            "GO_EMA999_TKE_GAIN_FULL_CLEAN81_LOAO" if passed
            else "STOP_EMA999_TKE_GAIN_HELD15"),
    }


def run(args: argparse.Namespace) -> dict:
    manifest_sha = dr1.normalized_sha256(
        args.clean_manifest_sha256, "clean manifest SHA-256")
    if manifest_sha != dr1.FORMAL_MANIFEST_SHA256:
        raise ValueError("formal held15 run requires the frozen clean81 manifest")
    if float(args.wall_budget_seconds) != WALL_BUDGET_SECONDS:
        raise ValueError("formal held15 wall budget must be exactly 3600 seconds")
    identity = dr1.validate_formal_git_identity(dr1.git_identity())
    if identity["branch"] != FORMAL_BRANCH:
        raise RuntimeError("formal held15 run is on the wrong branch")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, manifest_sha)
    fit_rows = [row for row in rows if int(row["aoa"]) != HELD_AOA]
    held_rows = [row for row in rows if int(row["aoa"]) == HELD_AOA]
    if sum(len(row["train_starts"]) for row in fit_rows) != (
            dr1.EXPECTED_FIT_COUNTS[HELD_AOA]):
        raise RuntimeError("held15 other-four fit count changed")
    if sum(len(row["train_starts"]) for row in held_rows) != (
            dr1.EXPECTED_AOA_COUNTS[HELD_AOA]):
        raise RuntimeError("held15 sample count changed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal EMA999 TKE-gain held15 requires CUDA")
    args.output_dir.mkdir(parents=True)
    dr1.write_json_atomic(args.output_dir / "run_owner.json", {
        "schema": RUN_OWNER_SCHEMA,
        "experiment": EXPERIMENT,
        "git": identity,
        "release_checkpoint_written": False,
        "submission_authorized": False,
    })
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    access = dr1.PrefixAccessAudit(HELD_AOA)

    fit_trajectories, fit_metadata, fit_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", access)
    fit_dataset = ensemble_train.common.repro.WindowDataset(
        fit_trajectories, fit_windows)
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, fit_windows)
    statistics = smu1.fold_statistics(sufficient, range(len(fit_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    normalizer_sha = smu1.statistics_digest(statistics)
    schedule = smu1.paired_batch_schedule(
        range(len(fit_windows)), fit_windows, updates=UPDATES, seed=SEED)
    frozen_masks = dr1.bootstrap_mask_by_trajectory(len(fit_trajectories))

    ensemble_train.seed_everything(SEED)
    model, start = ema10.inspect_start_checkpoint(
        args.start_sim, args.start_sim_sha256, normalizer, device)
    training, ema = ema10.train_raw_with_ema(
        model, fit_dataset, schedule, normalizer, device, deadline,
        frozen_masks)
    if training["actual_stream_sha256"] != schedule["sha256"]:
        raise RuntimeError("held15 Real sample stream changed")

    raw_path = args.output_dir / "raw_sealed_u4000.pth"
    raw_seal = save_endpoint(
        raw_path, model, "control", start, normalizer, schedule["sha256"])
    if raw_seal["model_state_sha256"] != (
            training["final_raw_model_state_sha256"]):
        raise RuntimeError("raw endpoint differs from trained raw state")
    access.seal(
        "control", raw_path, raw_seal["file_sha256"],
        raw_seal["model_state_sha256"])
    raw_sha_before_ema_seal = dr1.model_state_sha256(model.state_dict())
    ema_payload = ema.seal(model)
    raw_sha_after_ema_seal = dr1.model_state_sha256(model.state_dict())
    if raw_sha_before_ema_seal != raw_sha_after_ema_seal:
        raise RuntimeError("materializing EMA mutated the raw model")
    ema_state = ema_payload.pop("model_state")
    if (int(ema_payload["updates"]) != UPDATES
            or float(ema_payload["decay"]) != EMA_DECAY):
        raise RuntimeError("EMA seal counter/decay changed")
    if ema_lib.tensor_state_sha256(ema_state) != (
            ema_payload["materialized_model_state_sha256"]):
        raise RuntimeError("materialized EMA state hash changed before load")
    ema_model = EnsembleUNet2d(EnsembleUNet2dConfig(
        **{key: value for key, value in model.architecture_config().items()
           if key != "architecture"}))
    ema_model.load_state_dict(ema_state, strict=True)
    if ema_lib.tensor_state_sha256(ema_model.state_dict()) != (
            ema_payload["materialized_model_state_sha256"]):
        raise RuntimeError("strict-loaded EMA model differs from sealed state")
    ema_model = ema_model.to(device)
    ema_path = args.output_dir / "ema999_sealed_u4000.pth"
    ema_seal = save_endpoint(
        ema_path, ema_model, "ema_diagnostic", start, normalizer,
        schedule["sha256"], extra={"ema": ema_payload})
    access.seal(
        "candidate", ema_path, ema_seal["file_sha256"],
        ema_seal["model_state_sha256"])
    del model, ema_model, ema, ema_state
    torch.cuda.empty_cache()
    gc.collect()
    if not access.both_sealed:
        raise RuntimeError("raw and EMA endpoints were not both sealed")

    fit_ema = load_endpoint(ema_path, ema_seal, normalizer, device)
    gain_fit = fit_global_tke_gain(
        fit_ema, fit_dataset, range(len(fit_windows)), normalizer,
        device, deadline)
    del fit_ema
    torch.cuda.empty_cache()
    gc.collect()
    gain_path = args.output_dir / "global_tke_gain_sealed.json"
    gain_seal = save_gain_metadata(
        gain_path, gain_fit, raw_seal, ema_seal, normalizer_sha,
        schedule["sha256"], identity["commit"])
    gain_payload = load_gain_metadata(
        gain_path, gain_seal, raw_seal, ema_seal, normalizer_sha,
        schedule["sha256"], identity["commit"])
    if dr1.sha256(gain_path) != gain_seal["file_sha256"]:
        raise RuntimeError("TKE-gain metadata was not durably sealed")

    fit_ordered_key_sha = smu1.ordered_held_key_digest(
        range(len(fit_windows)), fit_windows, fit_metadata)
    pre_scoring_path = args.output_dir / "pre_scoring_seal.json"
    partial_path = args.output_dir / "partial_result.json"
    pre_scoring_payload = {
        "schema": PRE_SCORING_SCHEMA,
        "experiment": EXPERIMENT,
        "stage": "PRE_SCORING_SEALED",
        "held_aoa": HELD_AOA,
        "held_prefix_opened": False,
        "manifest": manifest_report,
        "fit_samples": len(fit_windows),
        "expected_held_samples": dr1.EXPECTED_AOA_COUNTS[HELD_AOA],
        "normalizer_statistics_sha256": normalizer_sha,
        "sample_stream_sha256": schedule["sha256"],
        "fit_ordered_key_sha256": fit_ordered_key_sha,
        "start": start,
        "training": training,
        "raw_endpoint_seal": raw_seal,
        "ema_endpoint_seal": ema_seal,
        "ema_contract": ema_payload,
        "raw_model_unchanged_by_ema_seal": (
            raw_sha_before_ema_seal == raw_sha_after_ema_seal),
        "gain_metadata_seal": gain_seal,
        "gain": gain_payload,
        "fit_target_audit": gain_fit["target_audit"],
        "prefix_access_before_held": access.report(
            [row["file"] for row in rows]),
        "git": identity,
        "release_checkpoint_written": False,
        "submission_authorized": False,
    }
    dr1.write_json_atomic(pre_scoring_path, pre_scoring_payload)
    pre_scoring_seal = {
        "path": str(pre_scoring_path.resolve()),
        "file_sha256": dr1.sha256(pre_scoring_path),
        "sealed_monotonic_ns": time.monotonic_ns(),
        "schema": PRE_SCORING_SCHEMA,
    }
    partial = write_partial_result(
        partial_path, "PRE_SCORING_SEALED",
        pre_scoring_seal=pre_scoring_seal,
        raw_endpoint_seal=raw_seal,
        ema_endpoint_seal=ema_seal,
        gain_metadata_seal=gain_seal,
        fit_target_audit=gain_fit["target_audit"],
        prefix_access=access.report([row["file"] for row in rows]),
        git=identity)

    try:
        held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
            args.real_dir, held_rows, "held", access)
    except Exception:
        write_partial_result(
            partial_path, "HELD_LOAD_FAILED",
            pre_scoring_seal=pre_scoring_seal,
            raw_endpoint_seal=raw_seal,
            ema_endpoint_seal=ema_seal,
            gain_metadata_seal=gain_seal,
            fit_target_audit=gain_fit["target_audit"],
            prefix_access=access.report([row["file"] for row in rows]),
            git=identity)
        raise
    partial = write_partial_result(
        partial_path, "HELD_OPENED",
        pre_scoring_seal=pre_scoring_seal,
        raw_endpoint_seal=raw_seal,
        ema_endpoint_seal=ema_seal,
        gain_metadata_seal=gain_seal,
        fit_target_audit=gain_fit["target_audit"],
        held_samples=len(held_windows),
        prefix_access=access.report([row["file"] for row in rows]),
        git=identity)
    held_dataset = ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    evaluate_indices = list(range(len(held_windows)))
    held_ordered_key_sha = smu1.ordered_held_key_digest(
        evaluate_indices, held_windows, held_metadata)
    partial = write_partial_result(
        partial_path, "HELD_READY",
        pre_scoring_seal=pre_scoring_seal,
        raw_endpoint_seal=raw_seal,
        ema_endpoint_seal=ema_seal,
        gain_metadata_seal=gain_seal,
        fit_target_audit=gain_fit["target_audit"],
        held_samples=len(held_windows),
        held_ordered_key_sha256=held_ordered_key_sha,
        prefix_access=access.report([row["file"] for row in rows]),
        git=identity)
    held_target_audit = audit_ordered_targets(
        held_dataset, evaluate_indices)
    partial = write_partial_result(
        partial_path, "HELD_TARGETS_SEALED",
        pre_scoring_seal=pre_scoring_seal,
        raw_endpoint_seal=raw_seal,
        ema_endpoint_seal=ema_seal,
        gain_metadata_seal=gain_seal,
        fit_target_audit=gain_fit["target_audit"],
        held_samples=len(held_windows),
        held_ordered_key_sha256=held_ordered_key_sha,
        held_target_audit=held_target_audit,
        prefix_access=access.report([row["file"] for row in rows]),
        git=identity)

    raw_model = load_endpoint(raw_path, raw_seal, normalizer, device)
    try:
        raw_metrics, raw_state, raw_target_audit = smu1.evaluate_model(
            raw_model, held_dataset, evaluate_indices, normalizer,
            device, deadline)
    except Exception:
        write_partial_result(
            partial_path, "RAW_SCORING_FAILED",
            prior_partial=partial,
            prefix_access=access.report([row["file"] for row in rows]))
        raise
    partial = write_partial_result(
        partial_path, "RAW_SCORED",
        pre_scoring_seal=pre_scoring_seal,
        raw_endpoint_seal=raw_seal,
        ema_endpoint_seal=ema_seal,
        gain_metadata_seal=gain_seal,
        fit_target_audit=gain_fit["target_audit"],
        held_samples=len(held_windows),
        held_ordered_key_sha256=held_ordered_key_sha,
        held_target_audit=held_target_audit,
        raw_metrics=raw_metrics,
        raw_target_audit=raw_target_audit,
        prefix_access=access.report([row["file"] for row in rows]),
        git=identity)
    del raw_model
    torch.cuda.empty_cache()
    gc.collect()
    held_ema = load_endpoint(ema_path, ema_seal, normalizer, device)
    try:
        ema_report, primary_report, transport = evaluate_ema_and_gain(
            held_ema, held_dataset, evaluate_indices, normalizer, device,
            deadline, gain_seal["g"], gain_seal["s"])
    except Exception:
        write_partial_result(
            partial_path, "EMA_GAIN_SCORING_FAILED",
            prior_partial=partial,
            prefix_access=access.report([row["file"] for row in rows]))
        raise
    partial = write_partial_result(
        partial_path, "ALL_ARMS_SCORED_UNVERIFIED",
        pre_scoring_seal=pre_scoring_seal,
        raw_endpoint_seal=raw_seal,
        ema_endpoint_seal=ema_seal,
        gain_metadata_seal=gain_seal,
        fit_target_audit=gain_fit["target_audit"],
        held_samples=len(held_windows),
        held_ordered_key_sha256=held_ordered_key_sha,
        held_target_audit=held_target_audit,
        raw_metrics=raw_metrics,
        raw_target_audit=raw_target_audit,
        ema_metrics=ema_report["metrics"],
        ema_target_audit=ema_report["held_target_audit"],
        primary_metrics=primary_report["metrics"],
        primary_target_audit=primary_report["held_target_audit"],
        transport_audit=transport,
        prefix_access=access.report([row["file"] for row in rows]),
        git=identity)
    del held_ema
    torch.cuda.empty_cache()
    gc.collect()
    if not (
            held_target_audit == raw_target_audit
            == ema_report["held_target_audit"]
            == primary_report["held_target_audit"]):
        write_partial_result(
            partial_path, "TARGET_IDENTITY_FAILED",
            prior_partial=partial,
            target_identity_verified=False,
            prefix_access=access.report([row["file"] for row in rows]))
        raise RuntimeError("raw/EMA/gained held targets or masks differ")
    partial = write_partial_result(
        partial_path, "TARGETS_VERIFIED",
        prior_partial=partial,
        target_identity_verified=True,
        prefix_access=access.report([row["file"] for row in rows]))

    access_report = access.report([row["file"] for row in rows])
    held_events = [
        row for row in access_report["events"] if row["phase"] == "held"]
    held_after_gain = bool(
        held_events
        and int(held_events[0]["opened_monotonic_ns"])
        > int(gain_seal["sealed_monotonic_ns"]))
    held_after_pre_scoring = bool(
        held_events
        and int(held_events[0]["opened_monotonic_ns"])
        > int(pre_scoring_seal["sealed_monotonic_ns"]))
    access_report.update({
        "gain_metadata_seal": gain_seal,
        "gain_metadata_file_intact": (
            dr1.sha256(gain_path) == gain_seal["file_sha256"]),
        "pre_scoring_seal": pre_scoring_seal,
        "pre_scoring_seal_file_intact": (
            dr1.sha256(pre_scoring_path)
            == pre_scoring_seal["file_sha256"]),
        "held_first_open_after_gain_seal": held_after_gain,
        "held_first_open_after_pre_scoring_seal": held_after_pre_scoring,
        "held_first_open_after_all_seals": bool(
            access_report["held_first_open_after_both_seals"]
            and held_after_gain and held_after_pre_scoring),
    })
    if not (
            access_report["allowlist_exact"]
            and access_report["tail_arrays_read"] == 0
            and not access_report["excluded_opened"]
            and access_report["held_first_open_after_all_seals"]
            and access_report["gain_metadata_file_intact"]
            and access_report["pre_scoring_seal_file_intact"]):
        raise RuntimeError("held15 prefix/gain-seal access audit failed")

    raw_report = {
        "seal": raw_seal,
        "metrics": raw_metrics,
        "metric_state": raw_state,
        "held_target_audit": raw_target_audit,
    }
    ema_report["seal"] = ema_seal
    primary_report["gain_metadata_seal"] = gain_seal
    ema_vs_raw = dr1.compare_metrics(
        ema_report["metrics"], raw_report["metrics"])
    primary_vs_raw = dr1.compare_metrics(
        primary_report["metrics"], raw_report["metrics"])
    primary_vs_ema = dr1.compare_metrics(
        primary_report["metrics"], ema_report["metrics"])
    pre_gate_deadline = wall_deadline_audit(started, deadline)
    if not pre_gate_deadline["within_wall_budget"]:
        write_partial_result(
            partial_path, "FINAL_DEADLINE_EXCEEDED",
            prior_partial=partial,
            held_target_audit=held_target_audit,
            target_identity_verified=True,
            deadline_audit={"pre_gate": pre_gate_deadline},
            prefix_access=access_report)
        raise TimeoutError("held15 final gate exceeded 3600-second wall budget")
    gate = primary_gate(primary_vs_raw, primary_vs_ema, transport)
    partial = write_partial_result(
        partial_path, "GATE_COMPLETE",
        prior_partial=partial,
        target_identity_verified=True,
        comparisons={
            "ema_vs_raw": ema_vs_raw,
            "primary_vs_raw": primary_vs_raw,
            "primary_vs_ema": primary_vs_ema,
        },
        gate=gate,
        deadline_audit={"pre_gate": pre_gate_deadline},
        prefix_access=access_report)
    partial_result_seal = {
        "path": str(partial_path.resolve()),
        "file_sha256": dr1.sha256(partial_path),
        "stage": "GATE_COMPLETE",
    }
    pre_return_deadline = wall_deadline_audit(started, deadline)
    if not pre_return_deadline["within_wall_budget"]:
        write_partial_result(
            partial_path, "FINAL_DEADLINE_EXCEEDED",
            prior_partial=partial,
            held_target_audit=held_target_audit,
            target_identity_verified=True,
            deadline_audit={
                "pre_gate": pre_gate_deadline,
                "pre_return": pre_return_deadline,
            },
            prefix_access=access_report)
        raise TimeoutError("held15 final result exceeded 3600-second wall budget")
    deadline_audit = {
        "pre_gate": pre_gate_deadline,
        "pre_return": pre_return_deadline,
    }
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract": CONTRACT,
        "decision": gate["decision"],
        "gate": gate,
        "held_aoa": HELD_AOA,
        "fit_samples": len(fit_windows),
        "samples": len(held_windows),
        "manifest": manifest_report,
        "normalizer_statistics_sha256": normalizer_sha,
        "sample_stream_sha256": schedule["sha256"],
        "fit_ordered_key_sha256": fit_ordered_key_sha,
        "held_ordered_key_sha256": held_ordered_key_sha,
        "held_target_audit": held_target_audit,
        "start": start,
        "training": training,
        "ema_contract": ema_payload,
        "raw_model_unchanged_by_ema_seal": (
            raw_sha_before_ema_seal == raw_sha_after_ema_seal),
        "gain": gain_payload,
        "control_raw": raw_report,
        "ema_diagnostic": ema_report,
        "primary_ema_gain": primary_report,
        "transport_audit": transport,
        "comparisons": {
            "ema_vs_raw": ema_vs_raw,
            "primary_vs_raw": primary_vs_raw,
            "primary_vs_ema": primary_vs_ema,
        },
        "prefix_access_audit": access_report,
        "deadline_audit": deadline_audit,
        "pre_scoring_seal": pre_scoring_seal,
        "partial_result_seal": partial_result_seal,
        "git": identity,
        "elapsed_seconds": pre_return_deadline["checked_elapsed_seconds"],
        "release_checkpoint_written": False,
        "submission_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument(
        "--clean-manifest-sha256", default=dr1.FORMAL_MANIFEST_SHA256)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256", default=FORMAL_START_SHA256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--wall-budget-seconds", type=float, default=WALL_BUDGET_SECONDS)
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


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
    if args.clean_manifest is None or args.start_sim is None:
        raise ValueError("clean manifest and legacy Sim checkpoint are required")
    manifest_sha = dr1.normalized_sha256(
        args.clean_manifest_sha256, "clean manifest SHA-256")
    if manifest_sha != dr1.FORMAL_MANIFEST_SHA256:
        raise ValueError("formal held15 run requires the frozen clean81 manifest")
    if (not math.isfinite(args.wall_budget_seconds)
            or float(args.wall_budget_seconds) != WALL_BUDGET_SECONDS):
        raise ValueError("formal held15 wall budget must be exactly 3600 seconds")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    try:
        summary = run(args)
        dr1.write_json_atomic(args.output_dir / "result.json", summary)
        print(json.dumps({
            "event": "c208_ema999_tke_gain_held15_complete",
            "decision": summary["decision"],
        }), flush=True)
    except Exception as exc:
        staged = {
            name: staged_file_evidence(args.output_dir / filename)
            for name, filename in (
                ("run_owner", "run_owner.json"),
                ("pre_scoring_seal", "pre_scoring_seal.json"),
                ("partial_result", "partial_result.json"),
                ("gain_metadata", "global_tke_gain_sealed.json"),
            )
        }
        staged = {key: value for key, value in staged.items()
                  if value is not None}
        stop = {
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "decision": "STOP_EMA999_TKE_GAIN_INFRASTRUCTURE_OR_CONTRACT",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "staged_provenance": staged,
            "release_checkpoint_written": False,
            "submission_authorized": False,
        }
        owner = staged.get("run_owner", {}).get("payload", {})
        result_path = args.output_dir / "result.json"
        if (owner.get("schema") == RUN_OWNER_SCHEMA
                and owner.get("experiment") == EXPERIMENT
                and not result_path.exists()):
            dr1.write_json_atomic(result_path, stop)
        print(json.dumps(stop, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
