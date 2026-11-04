#!/usr/bin/env python3
"""Condition-OOF horizon/channel/side calibration for mean-expert intervals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_mean_expert_disagreement import OFFICIAL, sample_factor  # noqa: E402
from evaluate_mean_expert_extent_blend import (  # noqa: E402
    blend_extents,
    fixed_point,
)
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("runs/logs/split_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-weight-b", type=float, default=0.25)
    parser.add_argument("--mean-expert-weight", type=float, default=0.5)
    parser.add_argument("--extent-weights", type=float, nargs=3,
                        default=(0.75, 0.25, 0.0))
    parser.add_argument("--extent-scale", type=float, default=1.0)
    parser.add_argument("--extent-power", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--horizon-blocks", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--grid-min", type=float, default=0.65)
    parser.add_argument("--grid-max", type=float, default=1.45)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--spatial-stride", type=int, default=3)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--fit-residual-scales", type=float, nargs="+",
                        default=(1.0, 1.05, 1.10))
    parser.add_argument("--eval-residual-scales", type=float, nargs="+",
                        default=(1.0, 1.15, 1.30))
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if (args.folds < 2 or args.horizon_blocks <= 0 or args.rounds <= 0
            or args.grid_min <= 0 or args.grid_max < args.grid_min
            or args.grid_step <= 0 or args.spatial_stride <= 0
            or args.time_stride <= 0):
        raise ValueError("invalid positive calibration arguments")
    if 20 % args.horizon_blocks:
        raise ValueError("horizon blocks must divide 20")
    if (len(args.extent_weights) != 3
            or any(not math.isfinite(value) or value < 0
                   for value in args.extent_weights)
            or abs(sum(args.extent_weights) - 1.0) > 1e-6):
        raise ValueError("extent weights must be non-negative and sum to one")
    values = (
        args.point_weight_b, args.mean_expert_weight,
        args.extent_scale, args.extent_power,
        *args.fit_residual_scales, *args.eval_residual_scales)
    if any(not math.isfinite(value) or value <= 0 for value in values[2:]):
        raise ValueError("scales and power must be finite and positive")
    if args.extent_power < 1.0:
        raise ValueError("extent power must be at least one")
    if not 0 <= args.point_weight_b <= 1 or not 0 <= args.mean_expert_weight <= 1:
        raise ValueError("point weights must lie in [0,1]")


def expand(values: np.ndarray, horizon_blocks: int) -> np.ndarray:
    if values.shape != (horizon_blocks, 2, 2):
        raise ValueError(f"unexpected multiplier shape {values.shape}")
    return np.repeat(values, 20 // horizon_blocks, axis=0)


def metrics(cache: dict[str, np.ndarray], calibration: np.ndarray,
            residual_scale: float = 1.0,
            sample_indices: np.ndarray | None = None) -> dict[str, float]:
    residual = cache["residual"]
    lower = cache["lower"]
    upper = cache["upper"]
    scored = cache["scored"]
    factor = cache["factor"]
    if sample_indices is not None:
        residual = residual[sample_indices]
        lower = lower[sample_indices]
        upper = upper[sample_indices]
        scored = scored[sample_indices]
        factor = factor[sample_indices]
    lower = lower * calibration[None, :, None, None, :, 0]
    upper = upper * calibration[None, :, None, None, :, 1]
    inside = ((residual * np.float32(residual_scale) >= -lower)
              & (residual * np.float32(residual_scale) <= upper))
    penalty = np.exp(
        -(lower + upper) / np.float32(OFFICIAL.SIGMA_GLOBAL))
    count = int(np.count_nonzero(scored))
    score = np.sum(
        factor.reshape((-1, 1, 1, 1, 1)) * penalty * inside,
        where=scored, dtype=np.float64) / max(count, 1)
    return {
        "sps_score": float(100.0 * score),
        "coverage": float(np.count_nonzero(inside & scored) / max(count, 1)),
        "mean_width": float(np.sum(
            lower + upper, where=scored, dtype=np.float64) / max(count, 1)),
        "scored_elements": count,
    }


def tune(cache: dict[str, np.ndarray], sample_indices: np.ndarray,
         residual_scale: float, grid: np.ndarray, horizon_blocks: int,
         rounds: int, time_stride: int, spatial_stride: int,
         ) -> tuple[np.ndarray, list[dict]]:
    residual = cache["residual"][sample_indices][
        :, ::time_stride, ::2, ::spatial_stride]
    lower = cache["lower"][sample_indices][
        :, ::time_stride, ::2, ::spatial_stride]
    upper = cache["upper"][sample_indices][
        :, ::time_stride, ::2, ::spatial_stride]
    scored = cache["scored"][sample_indices][
        :, ::time_stride, ::2, ::spatial_stride]
    factor = cache["factor"][sample_indices].reshape((-1, 1, 1, 1))
    sampled_horizons = np.arange(0, 20, time_stride)
    block_size = 20 // horizon_blocks
    values = np.ones((horizon_blocks, 2, 2), dtype=np.float32)
    trace = []
    for round_index in range(rounds):
        for block in range(horizon_blocks):
            selected = np.flatnonzero(sampled_horizons // block_size == block)
            for channel in range(2):
                block_residual = np.take(
                    residual, selected, axis=1)[..., channel]
                block_lower = np.take(
                    lower, selected, axis=1)[..., channel]
                block_upper = np.take(
                    upper, selected, axis=1)[..., channel]
                block_scored = np.take(
                    scored, selected, axis=1)[..., channel]
                count = int(np.count_nonzero(block_scored))
                for side in range(2):
                    best_value = float(values[block, channel, side])
                    best_score = -np.inf
                    for candidate in grid:
                        lower_scale = (
                            candidate if side == 0
                            else values[block, channel, 0])
                        upper_scale = (
                            candidate if side == 1
                            else values[block, channel, 1])
                        changed_lower = block_lower * np.float32(lower_scale)
                        changed_upper = block_upper * np.float32(upper_scale)
                        changed_residual = (
                            block_residual * np.float32(residual_scale))
                        inside = ((changed_residual >= -changed_lower)
                                  & (changed_residual <= changed_upper))
                        penalty = np.exp(
                            -(changed_lower + changed_upper)
                            / np.float32(OFFICIAL.SIGMA_GLOBAL))
                        value = np.sum(
                            factor * penalty * inside,
                            where=block_scored, dtype=np.float64
                        ) / max(count, 1)
                        if value > best_score:
                            best_score = value
                            best_value = float(candidate)
                    values[block, channel, side] = best_value
                    trace.append({
                        "round": round_index + 1,
                        "horizon_block": block,
                        "channel": "uv"[channel],
                        "side": ("lower", "upper")[side],
                        "multiplier": best_value,
                        "subset_component_score": float(100.0 * best_score),
                    })
    return expand(values, horizon_blocks), trace


def aggregate(rows: list[dict], key: str) -> dict[str, float]:
    total = sum(row[key]["scored_elements"] for row in rows)
    return {
        metric: sum(
            row[key][metric] * row[key]["scored_elements"]
            for row in rows) / total
        for metric in ("sps_score", "coverage", "mean_width")
    } | {"scored_elements": total}


def main() -> None:
    args = parse_args()
    validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache, allow_pickle=False) as payload:
        raw = {key: payload[key] for key in payload.files}
    prediction = fixed_point(
        raw, args.point_weight_b, args.mean_expert_weight)
    lower, upper = blend_extents(
        raw, args.extent_weights[1], args.extent_weights[2],
        args.extent_scale, args.extent_power)
    target = raw["target"]
    cache = {
        "residual": target - prediction,
        "lower": lower,
        "upper": upper,
        "scored": target != 0.0,
        "factor": sample_factor(prediction, target).astype(np.float32),
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sample_folds, fold_manifest = trajectory_folds(manifest, args.folds)
    sample_folds = np.asarray(sample_folds, dtype=np.int64)
    if sample_folds.size != target.shape[0]:
        raise RuntimeError(
            f"manifest has {sample_folds.size} samples, cache has {target.shape[0]}")
    all_indices = np.arange(target.shape[0], dtype=np.int64)
    identity = np.ones((20, 2, 2), dtype=np.float32)
    grid = np.arange(
        args.grid_min, args.grid_max + 0.5 * args.grid_step,
        args.grid_step, dtype=np.float32)

    candidates = {}
    for fit_scale in args.fit_residual_scales:
        fold_rows = []
        fold_calibrations = []
        for fold in range(args.folds):
            train_indices = all_indices[sample_folds != fold]
            eval_indices = all_indices[sample_folds == fold]
            calibration, trace = tune(
                cache, train_indices, float(fit_scale), grid,
                args.horizon_blocks, args.rounds,
                args.time_stride, args.spatial_stride)
            fold_calibrations.append(calibration)
            row = {
                "fold": fold,
                "train_samples": int(train_indices.size),
                "eval_samples": int(eval_indices.size),
                "baseline": metrics(
                    cache, identity, sample_indices=eval_indices),
                "local": metrics(
                    cache, calibration, sample_indices=eval_indices),
                "stress_1.15": metrics(
                    cache, calibration, 1.15, eval_indices),
                "stress_1.30": metrics(
                    cache, calibration, 1.30, eval_indices),
                "calibration": calibration.tolist(),
                "trace": trace,
            }
            fold_rows.append(row)
            print(json.dumps({
                "event": "fold", "fit_residual_scale": fit_scale,
                "fold": fold, "baseline": row["baseline"],
                "local": row["local"], "stress_1.30": row["stress_1.30"],
            }), flush=True)
        oof = {
            key: aggregate(fold_rows, key)
            for key in ("baseline", "local", "stress_1.15", "stress_1.30")
        }
        final_calibration, final_trace = tune(
            cache, all_indices, float(fit_scale), grid,
            args.horizon_blocks, args.rounds,
            args.time_stride, args.spatial_stride)
        final_scenarios = {
            f"residual_scale_{float(scale):.2f}": metrics(
                cache, final_calibration, float(scale))
            for scale in args.eval_residual_scales
        }
        key = f"{float(fit_scale):.2f}"
        candidates[key] = {
            "fit_residual_scale": float(fit_scale),
            "oof": oof,
            "oof_local_delta": (
                oof["local"]["sps_score"] - oof["baseline"]["sps_score"]),
            "folds": fold_rows,
            "fold_calibration_mean": np.mean(
                fold_calibrations, axis=0).tolist(),
            "final_calibration": final_calibration.tolist(),
            "final_trace": final_trace,
            "final_scenarios": final_scenarios,
        }
        print(json.dumps({
            "event": "candidate", "fit_residual_scale": fit_scale,
            "oof": oof, "final_scenarios": final_scenarios,
        }), flush=True)

    best_local_key = max(
        candidates,
        key=lambda key: candidates[key]["oof"]["local"]["sps_score"])
    best_robust_key = max(
        candidates,
        key=lambda key: (
            candidates[key]["oof"]["local"]["sps_score"]
            + 0.25 * candidates[key]["oof"]["stress_1.30"]["sps_score"]))
    report = {
        "method": "condition_oof_horizon_channel_side_calibration",
        "cache": str(args.cache),
        "manifest": str(args.manifest),
        "point_weight_b": args.point_weight_b,
        "mean_expert_weight": args.mean_expert_weight,
        "extent_weights": args.extent_weights,
        "extent_scale": args.extent_scale,
        "extent_power": args.extent_power,
        "fold_manifest": fold_manifest,
        "best_local_key": best_local_key,
        "best_robust_key": best_robust_key,
        "best_local": candidates[best_local_key],
        "best_robust": candidates[best_robust_key],
        "candidates": candidates,
        "config": vars(args) | {"cache": str(args.cache),
                                 "manifest": str(args.manifest),
                                 "output": str(args.output)},
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "best_local_key": best_local_key,
        "best_robust_key": best_robust_key,
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
