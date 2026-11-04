#!/usr/bin/env python3
"""Evaluate a focused simplex of three independently trained flow models."""

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
from evaluate_flow_backbone_blend import collect, load_model, score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--checkpoint-c", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--conditional-b", type=float, nargs="+",
        default=(0.25, 0.375, 0.5, 0.625, 0.75),
        help="B fraction of the residual A/B budget after assigning C.")
    parser.add_argument(
        "--weights-c", type=float, nargs="+",
        default=(0.0, 0.125, 0.25, 0.375, 0.5))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_unit_interval(values: list[float], name: str) -> None:
    if (not values or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in values)):
        raise ValueError(f"{name} must contain finite values in [0,1]")


def weighted_sum(arrays, weights):
    result = np.zeros_like(arrays[0])
    for value, weight in zip(arrays, weights):
        if weight:
            result += np.float32(weight) * value
    return result


def main() -> None:
    args = parse_args()
    if args.eval_batch_size <= 0:
        raise ValueError("eval batch size must be positive")
    validate_unit_interval(args.conditional_b, "conditional-b")
    validate_unit_interval(args.weights_c, "weights-c")
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=args.output.parent, seed=0)
    (_train_loader, val_loader, normalizer, train_windows,
     val_windows, trajectories) = common.prepare_real(data_args, device)

    arrays = []
    metadata = []
    targets = []
    for checkpoint in (
            args.checkpoint_a, args.checkpoint_b, args.checkpoint_c):
        model, model_metadata = load_model(checkpoint, normalizer, device)
        prediction, lower, upper, target = collect(
            model, val_loader, normalizer, device)
        arrays.append((prediction, lower, upper))
        targets.append(target)
        metadata.append(model_metadata)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if (not np.array_equal(targets[0], targets[1])
            or not np.array_equal(targets[0], targets[2])):
        raise RuntimeError("target order changed between model evaluations")

    predictions, lowers, uppers = tuple(zip(*arrays))
    results = {}
    for weight_c in args.weights_c:
        for conditional_b in args.conditional_b:
            weight_c = float(weight_c)
            weight_b = (1.0 - weight_c) * float(conditional_b)
            weight_a = 1.0 - weight_b - weight_c
            weights = (weight_a, weight_b, weight_c)
            prediction = weighted_sum(predictions, weights)
            lower = weighted_sum(lowers, weights)
            upper = weighted_sum(uppers, weights)
            metrics = score(prediction, lower, upper, targets[0])
            metrics["non_time_mean"] = float(np.mean([
                metrics["rel_l2_score"], metrics["tke_score"],
                metrics["mvpe_score"], metrics["sps_score"],
            ]))
            key = "/".join(f"{weight:.5f}" for weight in weights)
            results[key] = {"weights_abc": weights, **metrics}
            print(json.dumps({
                "event": "triple_blend", "weights_abc": weights,
                **metrics,
            }), flush=True)
            del prediction, lower, upper

    best_point = max(
        results, key=lambda key: results[key]["point_score"])
    best_non_time = max(
        results, key=lambda key: results[key]["non_time_mean"])
    report = {
        "weight_definition": (
            "w_c is direct; w_b=(1-w_c)*conditional_b; "
            "w_a=1-w_b-w_c"),
        "models": metadata,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "best_point": results[best_point],
        "best_non_time": results[best_non_time],
        "results": results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "best_point": report["best_point"],
        "best_non_time": report["best_non_time"],
    }), flush=True)


if __name__ == "__main__":
    main()
