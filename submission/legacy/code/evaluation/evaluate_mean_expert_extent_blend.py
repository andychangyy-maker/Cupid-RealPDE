#!/usr/bin/env python3
"""Search point-invariant interval blends from three distribution experts.

The deployed mean-expert point forecast is kept fixed.  A, B and C contribute
only positive lower/upper extents around that point, so interval diversity can
be optimized without changing rel-L2, TKE or MVPE.  A sparse deterministic
search ranks candidates before a small exact full-resolution evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from audit_mean_expert_disagreement import (  # noqa: E402
    OFFICIAL,
    collect_one,
    sample_factor,
)
from evaluate_flow_backbone_blend import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs=3, required=True,
                        metavar=("A", "B", "C"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--point-weight-b", type=float, default=0.25)
    parser.add_argument("--mean-expert-weight", type=float, default=0.5)
    parser.add_argument(
        "--extent-weights-b", type=float, nargs="+",
        default=(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75))
    parser.add_argument(
        "--extent-weights-c", type=float, nargs="+",
        default=(0.0, 0.125, 0.25, 0.375, 0.5))
    parser.add_argument(
        "--extent-scales", type=float, nargs="+",
        default=(0.875, 0.9375, 1.0, 1.0625))
    parser.add_argument(
        "--extent-powers", type=float, nargs="+", default=(1.0,),
        help="weighted power means; 1 is the ordinary convex blend")
    parser.add_argument("--search-stride", type=int, default=3)
    parser.add_argument("--finalists", type=int, default=12)
    parser.add_argument(
        "--residual-scales", type=float, nargs="+",
        default=(1.0, 1.15, 1.30))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.eval_batch_size <= 0 or args.search_stride <= 0 or args.finalists <= 0:
        raise ValueError("batch size, search stride and finalists must be positive")
    for name in ("point_weight_b", "mean_expert_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0,1]")
    for name in ("extent_weights_b", "extent_weights_c"):
        values = getattr(args, name)
        if not values or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in values):
            raise ValueError(f"{name} must contain finite values in [0,1]")
    if not args.extent_scales or any(
            not math.isfinite(value) or value <= 0
            for value in args.extent_scales):
        raise ValueError("extent scales must be finite and positive")
    if not args.extent_powers or any(
            not math.isfinite(value) or value < 1.0
            for value in args.extent_powers):
        raise ValueError("extent powers must be finite and at least one")
    if not args.residual_scales or any(
            not math.isfinite(value) or value <= 0
            for value in args.residual_scales):
        raise ValueError("residual scales must be finite and positive")


def collect_cache(args: argparse.Namespace, loader, normalizer,
                  device: torch.device) -> dict[str, np.ndarray]:
    arrays = []
    metadata = []
    for checkpoint in args.checkpoints:
        model, item = load_model(checkpoint, normalizer, device)
        arrays.append(collect_one(
            model, loader, normalizer, device, distribution=True))
        metadata.append(item)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    reference_target, reference_mask = arrays[0][-2:]
    for values in arrays[1:]:
        if not np.array_equal(reference_target, values[-2]):
            raise RuntimeError("target order changed between model evaluations")
        if not np.array_equal(reference_mask, values[-1]):
            raise RuntimeError("valid mask order changed between evaluations")
    result: dict[str, np.ndarray] = {
        "target": reference_target,
        "valid_mask": reference_mask,
        "model_metadata_json": np.asarray(json.dumps(metadata)),
    }
    for name, values in zip(("a", "b", "c"), arrays):
        prediction, lower, upper, _target, _mask = values
        result[f"prediction_{name}"] = prediction
        result[f"lower_{name}"] = lower
        result[f"upper_{name}"] = upper
    if args.cache is not None:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, **result)
    return result


def load_or_collect(args, loader, normalizer, device):
    if args.cache is not None and args.cache.is_file():
        with np.load(args.cache, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}
    return collect_cache(args, loader, normalizer, device)


def fixed_point(cache: dict[str, np.ndarray], weight_b: float,
                mean_weight: float) -> np.ndarray:
    prediction_a = cache["prediction_a"]
    base = prediction_a + np.float32(weight_b) * (
        cache["prediction_b"] - prediction_a)
    correction = np.float32(mean_weight) * (
        cache["prediction_c"].mean(axis=1, keepdims=True)
        - base.mean(axis=1, keepdims=True))
    return (base + correction).astype(np.float32, copy=False)


def sparse(array: np.ndarray, stride: int) -> np.ndarray:
    if array.ndim == 5:
        return array[:, ::2, ::2, ::stride, :]
    if array.ndim == 3:
        return array[:, ::2, ::stride]
    return array


def blend_extents(cache: dict[str, np.ndarray], weight_b: float,
                  weight_c: float, scale: float, power: float = 1.0,
                  sparse_stride: int | None = None,
                  ) -> tuple[np.ndarray, np.ndarray]:
    weight_a = 1.0 - weight_b - weight_c
    if weight_a < -1e-8:
        raise ValueError("extent weights must sum to at most one")
    if sparse_stride is None:
        get = lambda value: value
    else:
        get = lambda value: sparse(value, sparse_stride)
    weights = (max(weight_a, 0.0), weight_b, weight_c)
    names = ("a", "b", "c")
    active = [(weight, name) for weight, name in zip(weights, names)
              if weight > 0.0]
    if abs(power - 1.0) < 1e-8:
        lower = sum(
            np.float32(weight) * get(cache[f"lower_{name}"])
            for weight, name in active)
        upper = sum(
            np.float32(weight) * get(cache[f"upper_{name}"])
            for weight, name in active)
    else:
        reference = np.float32(0.01)
        exponent = np.float32(power)
        inverse_power = np.float32(1.0 / power)
        lower = reference * np.power(sum(
            np.float32(weight) * np.power(
                get(cache[f"lower_{name}"]) / reference, exponent)
            for weight, name in active), inverse_power)
        upper = reference * np.power(sum(
            np.float32(weight) * np.power(
                get(cache[f"upper_{name}"]) / reference, exponent)
            for weight, name in active), inverse_power)
    return (
        (np.float32(scale) * lower).astype(np.float32, copy=False),
        (np.float32(scale) * upper).astype(np.float32, copy=False),
    )


def sps_metrics(prediction: np.ndarray, target: np.ndarray,
                lower: np.ndarray, upper: np.ndarray,
                factor: np.ndarray, residual_scale: float) -> dict[str, float]:
    residual = (target - prediction) * np.float32(residual_scale)
    scored = target != 0.0
    inside = (residual >= -lower) & (residual <= upper)
    penalty = np.exp(
        -(lower + upper) / np.float32(OFFICIAL.SIGMA_GLOBAL))
    count = int(np.count_nonzero(scored))
    value = np.sum(
        factor.reshape((-1, 1, 1, 1, 1)) * penalty * inside,
        where=scored, dtype=np.float64) / max(count, 1)
    return {
        "sps_score": float(100.0 * value),
        "coverage": float(np.count_nonzero(inside & scored) / max(count, 1)),
        "mean_width": float(np.sum(
            lower + upper, where=scored, dtype=np.float64) / max(count, 1)),
        "scored_elements": count,
    }


def main() -> None:
    args = parse_args()
    validate(args)
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cache is not None and args.cache.is_file():
        with np.load(args.cache, allow_pickle=False) as payload:
            cache = {key: payload[key] for key in payload.files}
        train_windows = val_windows = trajectories = None
    else:
        data_args = SimpleNamespace(
            real_dir=args.real_dir, max_files=0, window_stride=20,
            val_fraction=0.2, batch_size=8,
            eval_batch_size=args.eval_batch_size, num_workers=0,
            real_stats=args.real_stats, output_dir=args.output.parent, seed=0)
        train_loader, val_loader, normalizer, train_windows, val_windows, trajectories = (
            common.prepare_real(data_args, device))
        cache = collect_cache(args, val_loader, normalizer, device)
        del train_loader, val_loader

    prediction = fixed_point(
        cache, args.point_weight_b, args.mean_expert_weight)
    target = cache["target"]
    factor = sample_factor(prediction, target)
    sparse_prediction = sparse(prediction, args.search_stride)
    sparse_target = sparse(target, args.search_stride)
    sparse_factor = sample_factor(sparse_prediction, sparse_target)

    candidates = []
    for weight_b in args.extent_weights_b:
        for weight_c in args.extent_weights_c:
            if weight_b + weight_c > 1.0 + 1e-8:
                continue
            for power in args.extent_powers:
                for scale in args.extent_scales:
                    candidates.append({
                        "extent_weight_a": float(1.0 - weight_b - weight_c),
                        "extent_weight_b": float(weight_b),
                        "extent_weight_c": float(weight_c),
                        "extent_scale": float(scale),
                        "extent_power": float(power),
                    })
    search_rows = []
    for candidate in candidates:
        lower, upper = blend_extents(
            cache, candidate["extent_weight_b"],
            candidate["extent_weight_c"], candidate["extent_scale"],
            candidate["extent_power"], sparse_stride=args.search_stride)
        local = sps_metrics(
            sparse_prediction, sparse_target, lower, upper,
            sparse_factor, residual_scale=1.0)
        stress = sps_metrics(
            sparse_prediction, sparse_target, lower, upper,
            sparse_factor, residual_scale=1.3)
        row = {
            **candidate, "local_sparse": local, "stress_1.30_sparse": stress,
            "selection_score": (
                local["sps_score"] + 0.25 * stress["sps_score"]),
        }
        search_rows.append(row)
        print(json.dumps({"event": "search", **row}), flush=True)

    selected = sorted(
        search_rows, key=lambda row: row["selection_score"], reverse=True
    )[:args.finalists]
    exact_rows = []
    for candidate in selected:
        lower, upper = blend_extents(
            cache, candidate["extent_weight_b"],
            candidate["extent_weight_c"], candidate["extent_scale"],
            candidate["extent_power"])
        scenarios = {
            f"residual_scale_{float(scale):.2f}": sps_metrics(
                prediction, target, lower, upper, factor, float(scale))
            for scale in args.residual_scales
        }
        row = {
            **{key: candidate[key] for key in (
                "extent_weight_a", "extent_weight_b", "extent_weight_c",
                "extent_scale", "extent_power")},
            "scenarios": scenarios,
        }
        exact_rows.append(row)
        print(json.dumps({"event": "exact", **row}), flush=True)

    best_local = max(
        exact_rows,
        key=lambda row: row["scenarios"]["residual_scale_1.00"]["sps_score"])
    best_robust = max(
        exact_rows,
        key=lambda row: sum(
            metrics["sps_score"] for metrics in row["scenarios"].values()))
    report = {
        "method": "point_invariant_three_expert_extent_blend",
        "checkpoints": [str(path) for path in args.checkpoints],
        "point_weight_b": args.point_weight_b,
        "mean_expert_weight": args.mean_expert_weight,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "best_local": best_local,
        "best_robust": best_robust,
        "search_rows": search_rows,
        "exact_finalists": exact_rows,
        "config": json.loads(json.dumps(vars(args), default=str)),
        "warning": (
            "Selection uses released real-tail labels. Prefer candidates "
            "stable under residual stress and across OOF-trained experts."),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "best_local": best_local,
        "best_robust": best_robust, "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
