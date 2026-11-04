#!/usr/bin/env python3
"""Blend A/B forecasts while using C only as a temporal-mean expert."""

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
        "--weights-b", type=float, nargs="+",
        default=(0.25, 0.375, 0.5, 0.625))
    parser.add_argument(
        "--mean-expert-weights", type=float, nargs="+",
        default=(0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_values(values, name: str) -> None:
    if (not values or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in values)):
        raise ValueError(f"{name} must contain finite values in [0,1]")


def main() -> None:
    args = parse_args()
    if args.eval_batch_size <= 0:
        raise ValueError("eval batch size must be positive")
    validate_values(args.weights_b, "weights-b")
    validate_values(args.mean_expert_weights, "mean-expert-weights")
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
    targets = []
    metadata = []
    for checkpoint in (
            args.checkpoint_a, args.checkpoint_b, args.checkpoint_c):
        print(json.dumps({
            "event": "collect_start", "checkpoint": str(checkpoint),
        }), flush=True)
        model, model_metadata = load_model(checkpoint, normalizer, device)
        prediction, lower, upper, target = collect(
            model, val_loader, normalizer, device)
        arrays.append((prediction, lower, upper))
        targets.append(target)
        metadata.append(model_metadata)
        print(json.dumps({
            "event": "collect_complete", "checkpoint": str(checkpoint),
            "samples": int(target.shape[0]),
        }), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if (not np.array_equal(targets[0], targets[1])
            or not np.array_equal(targets[0], targets[2])):
        raise RuntimeError("target order changed between model evaluations")

    prediction_a, lower_a, upper_a = arrays[0]
    prediction_b, lower_b, upper_b = arrays[1]
    prediction_c = arrays[2][0]
    results = {}
    for weight_b in args.weights_b:
        weight_b = float(weight_b)
        base_prediction = prediction_a + weight_b * (
            prediction_b - prediction_a)
        base_lower = lower_a + weight_b * (lower_b - lower_a)
        base_upper = upper_a + weight_b * (upper_b - upper_a)
        mean_delta = (
            prediction_c.mean(axis=1, keepdims=True)
            - base_prediction.mean(axis=1, keepdims=True))
        reference_tke = None
        for mean_weight in args.mean_expert_weights:
            mean_weight = float(mean_weight)
            correction = np.float32(mean_weight) * mean_delta
            prediction = base_prediction + correction
            lower = base_lower + correction
            upper = base_upper + correction
            metrics = score(prediction, lower, upper, targets[0])
            metrics["non_time_mean"] = float(np.mean([
                metrics["rel_l2_score"], metrics["tke_score"],
                metrics["mvpe_score"], metrics["sps_score"],
            ]))
            if reference_tke is None:
                reference_tke = metrics["tke_score"]
            elif abs(metrics["tke_score"] - reference_tke) > 2e-5:
                raise RuntimeError(
                    "mean-only correction unexpectedly changed TKE")
            key = f"b={weight_b:.5f}/mean_c={mean_weight:.5f}"
            results[key] = {
                "weight_b": weight_b,
                "mean_expert_weight_c": mean_weight,
                **metrics,
            }
            print(json.dumps({
                "event": "mean_expert", "weight_b": weight_b,
                "mean_expert_weight_c": mean_weight, **metrics,
            }), flush=True)
            del prediction, lower, upper, correction
        del base_prediction, base_lower, base_upper, mean_delta

    best_point = max(
        results, key=lambda key: results[key]["point_score"])
    best_non_time = max(
        results, key=lambda key: results[key]["non_time_mean"])
    report = {
        "method": (
            "base=(1-w_b)A+w_bB; output=base+w_mean*"
            "mean_time(C-base); shift bounds by same correction"),
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
