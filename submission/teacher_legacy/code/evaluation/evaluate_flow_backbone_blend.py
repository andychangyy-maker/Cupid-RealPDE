#!/usr/bin/env python3
"""Evaluate prediction-level blends of two different flow backbones."""

from __future__ import annotations

import argparse
import importlib.util
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
from train_flow_backbone import construct_model  # noqa: E402


def load_official_scoring():
    path = REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"
    spec = importlib.util.spec_from_file_location(
        "realpde_track1_v9_blend_scoring", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import scoring from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL = load_official_scoring()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_model(checkpoint: Path, normalizer, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    architecture = config["architecture"]
    model = construct_model(architecture, config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    common.set_model_stats(model, normalizer)
    return model.eval(), {
        "checkpoint": str(checkpoint),
        "architecture": architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "iteration": payload.get("iteration"),
    }


@torch.inference_mode()
def collect(model, loader, normalizer, device: torch.device):
    predictions = []
    lower_bounds = []
    upper_bounds = []
    targets = []
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, _target_norm = normalizer.preprocess(history, target)
        prediction_norm, lower_extent, upper_extent = (
            model.forward_distribution(history_norm))
        prediction = common.to_physical_target(
            prediction_norm, normalizer)
        prediction[..., 2] = 0.0
        lower = prediction.clone()
        upper = prediction.clone()
        lower[..., :2] = prediction[..., :2] - lower_extent * scale
        upper[..., :2] = prediction[..., :2] + upper_extent * scale
        if (not torch.isfinite(prediction).all()
                or not torch.isfinite(lower).all()
                or not torch.isfinite(upper).all()
                or torch.any(lower > upper)):
            finite_prediction = bool(torch.isfinite(prediction).all())
            finite_lower = bool(torch.isfinite(lower).all())
            finite_upper = bool(torch.isfinite(upper).all())
            invalid_widths = int((lower > upper).sum())
            minimum_width = float((upper - lower).amin())
            raise FloatingPointError(
                "model produced invalid blend inputs: "
                f"finite=({finite_prediction},{finite_lower},{finite_upper}), "
                f"invalid_widths={invalid_widths}, "
                f"minimum_width={minimum_width:.9g}")
        predictions.append(prediction.cpu().numpy())
        lower_bounds.append(lower.cpu().numpy())
        upper_bounds.append(upper.cpu().numpy())
        targets.append(target.cpu().numpy())
    return tuple(np.concatenate(values) for values in (
        predictions, lower_bounds, upper_bounds, targets))


def score(prediction: np.ndarray, lower: np.ndarray,
          upper: np.ndarray, target: np.ndarray) -> dict[str, float]:
    channels = OFFICIAL.measured_channels(target)
    rel_error = float(np.mean(
        OFFICIAL.rel_l2_per_sample(prediction, target, channels)))
    tke_error = float(np.mean(
        OFFICIAL.tke_rel_l2_per_sample(prediction, target, channels)))
    mvpe_error = float(np.mean(
        OFFICIAL.mvpe_rel_l2_per_sample(prediction, target)))
    sps_value, coverage = OFFICIAL.aggregate_sps(
        prediction, target, channels, lower=lower, upper=upper)
    metrics = {
        "rel_l2_score": OFFICIAL.score_error(rel_error),
        "tke_score": OFFICIAL.score_error(tke_error),
        "mvpe_score": OFFICIAL.score_error(mvpe_error),
        "sps_score": OFFICIAL.score_sps(sps_value),
        "coverage": coverage,
        "mean_width": float(np.mean(
            (upper[..., :channels] - lower[..., :channels])[
                target[..., :channels] != 0.0])),
    }
    metrics["point_score"] = (
        0.5 * metrics["rel_l2_score"]
        + 0.3 * metrics["tke_score"]
        + 0.2 * metrics["mvpe_score"])
    return metrics


def main() -> None:
    args = parse_args()
    if args.eval_batch_size <= 0:
        raise ValueError("eval batch size must be positive")
    if (not args.alphas
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0
                   for value in args.alphas)):
        raise ValueError("alphas must be finite values in [0,1]")
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=args.output.parent, seed=0)
    (_train_loader, val_loader, normalizer, train_windows,
     val_windows, trajectories) = common.prepare_real(data_args, device)

    model_a, metadata_a = load_model(
        args.checkpoint_a, normalizer, device)
    prediction_a, lower_a, upper_a, target_a = collect(
        model_a, val_loader, normalizer, device)
    del model_a
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model_b, metadata_b = load_model(
        args.checkpoint_b, normalizer, device)
    prediction_b, lower_b, upper_b, target_b = collect(
        model_b, val_loader, normalizer, device)
    del model_b
    if not np.array_equal(target_a, target_b):
        raise RuntimeError("target order changed between model evaluations")

    results = {}
    for alpha in args.alphas:
        alpha = float(alpha)
        prediction = prediction_a + alpha * (prediction_b - prediction_a)
        lower = lower_a + alpha * (lower_b - lower_a)
        upper = upper_a + alpha * (upper_b - upper_a)
        metrics = score(prediction, lower, upper, target_a)
        results[f"{alpha:.3f}"] = metrics
        print(json.dumps(
            {"event": "blend", "alpha_b": alpha, **metrics}), flush=True)
    best_alpha = max(
        results, key=lambda key: results[key]["point_score"])
    report = {
        "alpha_definition": "output=(1-alpha_b)*a+alpha_b*b",
        "model_a": metadata_a,
        "model_b": metadata_b,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "best_alpha": float(best_alpha),
        "best_point_score": results[best_alpha]["point_score"],
        "results": results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "best_alpha": report["best_alpha"],
        "best_point_score": report["best_point_score"],
    }), flush=True)


if __name__ == "__main__":
    main()
