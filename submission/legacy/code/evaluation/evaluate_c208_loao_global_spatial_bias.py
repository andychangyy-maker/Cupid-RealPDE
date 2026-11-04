#!/usr/bin/env python3
"""Calibrator-level LOAO audit of one fixed C208 spatial residual bias.

The existing E0 OOF collector reconstructs the frozen point, target, and
absolute-bound streams.  For each held AoA, this audit averages the literal
dense target-minus-point residual over every sample from the other four AoAs
and all 20 forecast steps.  The resulting single ``b[y,x,uv]`` is broadcast
over time on the held AoA.  There is no mask, alpha, regularizer, parameter,
or metadata-conditioned deployment path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_c208_q50_mean_point_shift as e0  # noqa: E402


EXPERIMENT = "c208_loao_global_spatial_residual_bias"
PASS_DECISION = "PASS_GLOBAL_SPATIAL_BIAS_TO_ACTUAL_PACKAGE_PREFLIGHT"
STOP_DECISION = "STOP_GLOBAL_SPATIAL_BIAS_NO_GENERALIZABLE_SIGNAL"
ERROR_DECISION = "STOP_GLOBAL_SPATIAL_BIAS_ERROR"
AOA_VALUES = e0.AOA_VALUES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def dense_spatial_bias(point: np.ndarray, target: np.ndarray,
                       calibration_rows: np.ndarray) -> np.ndarray:
    """Return the literal dense mean(target-point) over sample and time."""
    calibration_rows = np.asarray(calibration_rows, dtype=bool)
    if (point.dtype != np.float32 or target.dtype != np.float32
            or point.shape != target.shape or point.ndim != 5
            or point.shape[1] != 20 or point.shape[-1] != 2):
        raise ValueError("point/target must be aligned float32 UV sequences")
    if (calibration_rows.shape != (len(point),)
            or not calibration_rows.any()):
        raise ValueError("calibration rows must be non-empty and aligned")
    residual = target[calibration_rows] - point[calibration_rows]
    bias = residual.mean(axis=(0, 1), keepdims=True, dtype=np.float32)
    if bias.shape != (1, 1, point.shape[2], point.shape[3], 2):
        raise RuntimeError("dense spatial bias has an unexpected shape")
    if not np.all(np.isfinite(bias)):
        raise FloatingPointError("dense spatial bias is non-finite")
    return np.ascontiguousarray(bias)


def apply_spatial_bias(point: np.ndarray, bias: np.ndarray) -> np.ndarray:
    expected = (1, 1, point.shape[2], point.shape[3], 2)
    if point.dtype != np.float32 or point.ndim != 5 or point.shape[-1] != 2:
        raise ValueError("point must be a float32 UV sequence")
    if bias.dtype != np.float32 or bias.shape != expected:
        raise ValueError("bias must be one fixed float32 spatial UV field")
    candidate = point + bias
    if candidate.dtype != np.float32 or not np.all(np.isfinite(candidate)):
        raise FloatingPointError("spatial-bias candidate is non-finite")
    return candidate


def build_loao_candidate(
        point: np.ndarray, target: np.ndarray, aoas: np.ndarray,
        groups: tuple[int, ...] = AOA_VALUES,
        ) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    """Fit exactly one other-four-AoA bias for each held AoA."""
    aoas = np.asarray(aoas)
    if aoas.shape != (len(point),) or set(aoas.tolist()) != set(groups):
        raise ValueError("AoA rows differ from the frozen five groups")
    candidate = point.copy()
    biases = {}
    calibration_counts = {}
    for held in groups:
        calibration = aoas != held
        held_rows = aoas == held
        bias = dense_spatial_bias(point, target, calibration)
        candidate[held_rows] = apply_spatial_bias(point[held_rows], bias)
        biases[held] = bias
        calibration_counts[held] = int(calibration.sum())
    if not np.all(np.isfinite(candidate)):
        raise FloatingPointError("LOAO spatial-bias candidate is non-finite")
    return candidate, biases, calibration_counts


def bias_summary(bias: np.ndarray) -> dict[str, object]:
    values = bias.astype(np.float64, copy=False)
    return {
        "sha256": e0.array_sha256(bias),
        "shape": list(bias.shape),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "rms": float(np.sqrt(np.mean(values ** 2))),
    }


def gate_decision(
        pooled_delta: dict[str, float], loao_rows: list[dict],
        hard_guards: dict[str, bool], elapsed: float, gpu_seconds: float,
        ) -> tuple[dict[str, bool], str]:
    held_aoas = [row.get("held_aoa") for row in loao_rows]
    if (len(loao_rows) != len(AOA_VALUES)
            or set(held_aoas) != set(AOA_VALUES)):
        raise ValueError("gate requires exactly the five frozen held AoAs")
    checks = e0.build_gate_checks(
        pooled_delta, loao_rows, hard_guards, elapsed, gpu_seconds)
    decision = PASS_DECISION if all(checks.values()) else STOP_DECISION
    return checks, decision


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    memory_available = e0.available_memory_bytes()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch_size != 16 or args.num_workers != 0:
        raise ValueError("formal spatial-bias audit requires batch-size=16 and num-workers=0")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal spatial-bias audit requires CUDA")

    e0.uc1.configure_replay(e0.uc1.EXPECTED_TRAINING_CONFIG["seed"])
    collected = e0._collect_oof(args, started)
    arrays = collected.pop("arrays")
    base = arrays["point"]
    target = arrays["target"]
    lower = arrays["lower"]
    upper = arrays["upper"]
    aoas = arrays["aoas"]
    folds = arrays["folds"]
    arrays.pop("bias", None)
    arrays.pop("solid_mask", None)

    initial_hashes = {
        key: e0.array_sha256(value)
        for key, value in {
            "point": base, "target": target,
            "lower": lower, "upper": upper,
        }.items()
    }
    for value in (base, target, lower, upper, aoas, folds):
        value.setflags(write=False)

    baseline = e0.score_bundle(base, target, lower, upper)
    baseline_folds = [
        e0.score_bundle(base[folds == fold], target[folds == fold],
                        lower[folds == fold], upper[folds == fold])
        for fold in range(5)
    ]
    baseline_replay = e0.uc1.baseline_replay_gate(
        collected["summary"]["selected_oof"], baseline,
        [row["selected"] for row in collected["summary"]["folds"]],
        baseline_folds)
    if not baseline_replay["pass"]:
        raise RuntimeError("spatial-bias baseline failed frozen OOF replay")

    loao_candidate, loao_biases, calibration_counts = build_loao_candidate(
        base, target, aoas)
    loao_metrics = e0.score_bundle(loao_candidate, target, lower, upper)
    loao_delta = e0.metric_delta(loao_metrics, baseline)
    loao_rows = []
    for held in AOA_VALUES:
        held_rows = aoas == held
        before = e0.score_bundle(
            base[held_rows], target[held_rows],
            lower[held_rows], upper[held_rows])
        after = e0.score_bundle(
            loao_candidate[held_rows], target[held_rows],
            lower[held_rows], upper[held_rows])
        delta = e0.metric_delta(after, before)
        loao_rows.append({
            "held_aoa": held,
            "calibration_samples": calibration_counts[held],
            "held_samples": int(held_rows.sum()),
            "bias": bias_summary(loao_biases[held]),
            "baseline": before,
            "candidate": after,
            "delta": delta,
            "non_time_shadow_delta": e0.non_time_shadow(delta),
        })

    baseline_outside = e0.point_outside_bounds_rate(
        base, target, lower, upper)
    loao_outside = e0.point_outside_bounds_rate(
        loao_candidate, target, lower, upper)
    loao_centered_max = e0.centered_fluctuation_max(base, loao_candidate)
    loao_candidate_sha256 = e0.array_sha256(loao_candidate)
    loao_finite = bool(np.all(np.isfinite(loao_candidate)))
    base_full = e0.pad_zero_pressure(base)
    loao_full = e0.pad_zero_pressure(loao_candidate)
    loao_pressure_contract = bool(
        np.array_equal(base_full[..., :2], base)
        and np.array_equal(loao_full[..., :2], loao_candidate)
        and np.count_nonzero(base_full[..., 2]) == 0
        and np.count_nonzero(loao_full[..., 2]) == 0)
    del base_full, loao_full, loao_candidate
    loao_gate_elapsed = time.perf_counter() - started

    full_bias = dense_spatial_bias(
        base, target, np.ones(len(base), dtype=bool))
    full_candidate = apply_spatial_bias(base, full_bias)
    full_metrics = e0.score_bundle(full_candidate, target, lower, upper)
    full_delta = e0.metric_delta(full_metrics, baseline)
    full_outside = e0.point_outside_bounds_rate(
        full_candidate, target, lower, upper)
    full_candidate_sha256 = e0.array_sha256(full_candidate)
    full_finite = bool(np.all(np.isfinite(full_candidate)))
    full_uv = e0.pad_zero_pressure(full_candidate)
    full_pressure_contract = bool(
        np.array_equal(full_uv[..., :2], full_candidate)
        and np.count_nonzero(full_uv[..., 2]) == 0)
    del full_uv
    full_diagnostic = {
        "participates_in_gate": False,
        "target_visible_same_data_fit": True,
        "calibration_samples": len(base),
        "bias": bias_summary(full_bias),
        "metrics": full_metrics,
        "delta": full_delta,
        "non_time_shadow_delta": e0.non_time_shadow(full_delta),
        "centered_fluctuation_max_abs": e0.centered_fluctuation_max(
            base, full_candidate),
        "tke_delta_within_1e_5": (
            abs(full_delta["tke_score"]) <= e0.TKE_TOLERANCE),
        "absolute_bounds_same_objects_reused": True,
        "coverage_exact": full_metrics["coverage"] == baseline["coverage"],
        "covered_elements_exact": (
            full_metrics["covered_elements"] == baseline["covered_elements"]),
        "scored_elements_exact": (
            full_metrics["scored_elements"] == baseline["scored_elements"]),
        "mean_width_exact": (
            full_metrics["mean_width"] == baseline["mean_width"]),
        "pressure_constructed_and_verified_zero": full_pressure_contract,
        "point_outside_bounds_rate": full_outside,
        "point_outside_bounds_rate_delta": full_outside - baseline_outside,
        "candidate_finite": full_finite,
    }
    del full_candidate

    final_hashes = {
        key: e0.array_sha256(value)
        for key, value in {
            "point": base, "target": target,
            "lower": lower, "upper": upper,
        }.items()
    }
    immutable_inputs = final_hashes == initial_hashes
    elapsed = time.perf_counter() - started
    hard_guards = {
        "baseline_replay_pass": bool(baseline_replay["pass"]),
        "fixed725_original_indices_permutation": bool(np.array_equal(
            np.sort(collected["fold_order_original_indices"]), np.arange(725))),
        "fold_point_stream_sha_frozen": (
            collected["stream_hashes"]["fold_order_point"]
            == e0.POINT_STREAM_SHA256),
        "fold_target_stream_sha_frozen": (
            collected["stream_hashes"]["fold_order_target"]
            == e0.TARGET_STREAM_SHA256),
        "history_input_unchanged_during_collection": True,
        "literal_dense_biases_and_candidates_finite": bool(
            loao_finite and all(np.all(np.isfinite(value))
                    for value in loao_biases.values())),
        "absolute_bounds_immutable": bool(
            initial_hashes["lower"] == final_hashes["lower"]
            and initial_hashes["upper"] == final_hashes["upper"]),
        "coverage_covered_scored_width_exact": bool(
            loao_metrics["coverage"] == baseline["coverage"]
            and loao_metrics["covered_elements"] == baseline["covered_elements"]
            and loao_metrics["scored_elements"] == baseline["scored_elements"]
            and loao_metrics["mean_width"] == baseline["mean_width"]),
        "loao_centered_fluctuation_within_1e_6": (
            loao_centered_max <= e0.CENTERED_FLUCTUATION_TOLERANCE),
        "immutable_point_target_and_bounds": bool(immutable_inputs),
        "pressure_constructed_and_verified_zero": bool(
            loao_pressure_contract),
        "no_mask_alpha_regularizer_training_checkpoint_or_zip": True,
    }
    checks, decision = gate_decision(
        loao_delta, loao_rows, hard_guards, loao_gate_elapsed,
        collected["gpu_seconds"])

    report = {
        "experiment": EXPERIMENT,
        "evidence_class": (
            "fixed725 fold-excluded UQ bounds plus calibrator-level LOAO; "
            "not official evidence and not full-model true LOAO"),
        "contract": {
            "formula": (
                "b_-a[y,x,uv]=dense_mean_{samples AoA!=a,time}(target-point); "
                "held_point'=point+b_-a"),
            "dense_mean_includes_zero_support": True,
            "mask": None,
            "alpha": None,
            "regularizer": None,
            "metadata_conditioned_deployment": False,
            "absolute_bounds": "built from base point first and frozen",
            "non_time_shadow_weights": e0.SHADOW_WEIGHTS,
            "full_fit_participates_in_gate": False,
            "loao_pooled_scoring": (
                "concatenate five held-AoA predictions, then call the pooled "
                "official scorer exactly once"),
            "trainable_parameters": 0,
        },
        "git": e0.git_identity(),
        "identities": collected["identities"],
        "fold_metadata": collected["fold_metadata"],
        "stream_hashes": {
            **collected["stream_hashes"],
            "original_order_point": initial_hashes["point"],
            "original_order_target": initial_hashes["target"],
            "original_order_lower": initial_hashes["lower"],
            "original_order_upper": initial_hashes["upper"],
            "loao_candidate": loao_candidate_sha256,
            "full_fit_candidate": full_candidate_sha256,
        },
        "baseline": baseline,
        "baseline_replay_gate": baseline_replay,
        "loao": {
            "pooled": loao_metrics,
            "pooled_delta": loao_delta,
            "pooled_non_time_shadow_delta": e0.non_time_shadow(loao_delta),
            "rows": loao_rows,
            "limitation": (
                "Only the spatial-bias calibrator excludes held AoA; the "
                "frozen point and UQ fold models may have seen that AoA."),
        },
        "full_fit_diagnostic": full_diagnostic,
        "invariants": {
            "loao_centered_fluctuation_max_abs": loao_centered_max,
            "loao_tke_delta_within_1e_5": (
                abs(loao_delta["tke_score"]) <= e0.TKE_TOLERANCE),
            "absolute_bounds_same_objects_reused": True,
            "baseline_point_outside_bounds_rate": baseline_outside,
            "loao_point_outside_bounds_rate": loao_outside,
            "loao_point_outside_bounds_rate_delta": (
                loao_outside - baseline_outside),
            "initial_hashes": initial_hashes,
            "final_hashes": final_hashes,
            "history_input_before_after_equal": True,
        },
        "gate_checks": checks,
        "decision": decision,
        "loao_gate_elapsed_seconds": loao_gate_elapsed,
        "elapsed_seconds": elapsed,
        "gpu_inference_seconds": collected["gpu_seconds"],
        "resources": {
            "memory_available_bytes_at_start": memory_available,
            "minimum_required_bytes": e0.MIN_AVAILABLE_MEMORY_BYTES,
        },
        "checkpoints_saved": [],
        "submission_created": False,
        "limitations": [
            "This is a target-visible estimator audit of the existing "
            "time-constant additive operator, not a new mechanism family.",
            "Fixed725 has been reused across campaign decisions; LOAO does not "
            "remove campaign-level adaptive selection risk.",
            "The full-fit diagnostic uses the same 725 targets and is never a gate.",
            "Literal dense mean mixes changing zero-support geometry; no masked "
            "rescue candidate is permitted.",
            "A PASS only authorizes actual-package preflight, not submission.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "global_spatial_bias_complete",
        "decision": decision,
        "loao_point_delta": loao_delta["point_score"],
        "loao_sps_delta": loao_delta["sps_score"],
        "loao_non_time_shadow_delta": e0.non_time_shadow(loao_delta),
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    try:
        if e0.available_memory_bytes() < e0.MIN_AVAILABLE_MEMORY_BYTES:
            raise MemoryError("formal spatial-bias audit requires 12 GiB RAM")
        main()
    except Exception as error:
        try:
            error_args = parse_args()
            if not error_args.output.exists():
                error_args.output.parent.mkdir(parents=True, exist_ok=True)
                error_args.output.write_text(
                    json.dumps({
                        "experiment": EXPERIMENT,
                        "decision": ERROR_DECISION,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "git": e0.git_identity(),
                        "checkpoints_saved": [],
                        "submission_created": False,
                    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
        except Exception:
            pass
        raise
