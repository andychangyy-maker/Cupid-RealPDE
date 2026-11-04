#!/usr/bin/env python3
"""Fuse A/B residual quantiles before selecting an SPS-optimal interval.

The deployed mean expert currently selects one interval independently inside
each model and then linearly blends the selected endpoints.  That operation is
not distributionally coherent when A and B choose different quantile pairs.
This audit blends all seven absolute outcome quantiles first, then performs one
utility selection.  Point predictions remain the chosen A/B/C mean expert.
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
    base_arrays,
    sample_factor,
)
from evaluate_flow_backbone_blend import load_model  # noqa: E402


SPS_SIGMA = 0.0563870259


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs=2, required=True,
                        metavar=("A", "B"))
    parser.add_argument("--ensemble-cache", type=Path, required=True)
    parser.add_argument("--quantile-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-b", type=float, default=0.25)
    parser.add_argument("--mean-expert-weight", type=float, default=0.5)
    parser.add_argument(
        "--quantile-weights-b", type=float, nargs="+",
        default=(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0))
    parser.add_argument(
        "--width-temperatures", type=float, nargs="+",
        default=(0.70, 0.85, 1.0, 1.15, 1.30))
    parser.add_argument(
        "--residual-scales", type=float, nargs="+",
        default=(1.0, 1.15, 1.30, 1.50))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    for name in ("weight_b", "mean_expert_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0,1]")
    if args.eval_batch_size <= 0 or args.score_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if not args.quantile_weights_b or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in args.quantile_weights_b):
        raise ValueError("quantile weights must be finite and in [0,1]")
    if not args.width_temperatures or any(
            not math.isfinite(value) or value <= 0
            for value in args.width_temperatures):
        raise ValueError("width temperatures must be finite and positive")
    if not args.residual_scales or any(
            not math.isfinite(value) or value <= 0
            for value in args.residual_scales):
        raise ValueError("residual scales must be finite and positive")


@torch.inference_mode()
def collect_quantiles(model, loader, normalizer,
                      device: torch.device) -> np.ndarray:
    values = []
    model.eval()
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, _ = normalizer.preprocess(history, target)
        _prediction, _members, quantiles, _lower, _upper = (
            model.forward_quantiles(history_norm))
        values.append(quantiles.detach().cpu().to(torch.float16).numpy())
    return np.concatenate(values)


def ensure_quantile_cache(args, device: torch.device) -> tuple[Path, Path]:
    args.quantile_cache_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        args.quantile_cache_dir / "quantiles_a_fp16.npy",
        args.quantile_cache_dir / "quantiles_b_fp16.npy",
    )
    if all(path.is_file() for path in paths):
        return paths
    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=args.quantile_cache_dir, seed=0)
    train_loader, val_loader, normalizer, *_ = common.prepare_real(
        data_args, device)
    del train_loader
    metadata = []
    for checkpoint, path in zip(args.checkpoints, paths):
        if path.is_file():
            continue
        model, item = load_model(checkpoint, normalizer, device)
        quantiles = collect_quantiles(model, val_loader, normalizer, device)
        np.save(path, quantiles, allow_pickle=False)
        metadata.append(item)
        del model, quantiles
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (args.quantile_cache_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    return paths


def candidate_pairs(levels: torch.Tensor) -> tuple[torch.Tensor, ...]:
    middle = levels.numel() // 2
    lower_indices = []
    upper_indices = []
    masses = []
    for lower in range(middle + 1):
        for upper in range(max(middle, lower + 1), levels.numel()):
            lower_indices.append(lower)
            upper_indices.append(upper)
            masses.append(float(levels[upper] - levels[lower]))
    return (
        torch.tensor(lower_indices, device=levels.device),
        torch.tensor(upper_indices, device=levels.device),
        torch.tensor(masses, device=levels.device),
    )


def select_interval(quantiles: torch.Tensor, levels: torch.Tensor,
                    width_temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    lower_indices, upper_indices, masses = candidate_pairs(levels)
    lower = quantiles.index_select(-1, lower_indices)
    upper = quantiles.index_select(-1, upper_indices)
    width = (upper - lower).clamp_min(1e-8)
    utility = masses.view(*([1] * (width.ndim - 1)), -1) * torch.exp(
        -width * float(width_temperature) / SPS_SIGMA)
    selected = utility.argmax(dim=-1, keepdim=True)
    return (
        lower.gather(-1, selected).squeeze(-1),
        upper.gather(-1, selected).squeeze(-1),
    )


def empty_accumulator(scales) -> dict:
    return {
        "scored_elements": 0,
        "width_sum": 0.0,
        "scenarios": {
            f"{float(scale):.2f}": {"numerator": 0.0, "covered": 0}
            for scale in scales
        },
    }


def update_accumulator(accumulator, prediction, target, factor,
                       lower_bound, upper_bound, scales) -> None:
    scored = target != 0.0
    width = upper_bound - lower_bound
    penalty = torch.exp(-width / SPS_SIGMA)
    count = int(scored.sum())
    accumulator["scored_elements"] += count
    accumulator["width_sum"] += float(width[scored].sum(dtype=torch.float64))
    residual = target - prediction
    shaped_factor = factor.view(-1, 1, 1, 1, 1)
    for scale in scales:
        scaled_target = prediction + residual * float(scale)
        inside = (scaled_target >= lower_bound) & (scaled_target <= upper_bound)
        row = accumulator["scenarios"][f"{float(scale):.2f}"]
        row["numerator"] += float(
            (shaped_factor * penalty * inside)[scored].sum(dtype=torch.float64))
        row["covered"] += int((inside & scored).sum())


def finalize(accumulator) -> dict:
    count = accumulator["scored_elements"]
    return {
        "mean_width": accumulator["width_sum"] / count,
        "scored_elements": count,
        "scenarios": {
            key: {
                "sps_score": 100.0 * value["numerator"] / count,
                "coverage": value["covered"] / count,
            }
            for key, value in accumulator["scenarios"].items()
        },
    }


def main() -> None:
    args = parse_args()
    validate(args)
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantile_paths = ensure_quantile_cache(args, device)
    quantiles_a = np.load(quantile_paths[0], mmap_mode="r", allow_pickle=False)
    quantiles_b = np.load(quantile_paths[1], mmap_mode="r", allow_pickle=False)
    with np.load(args.ensemble_cache, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    base = base_arrays(arrays, args.weight_b, args.mean_expert_weight)
    del arrays
    expected = (*base["prediction"].shape, 7)
    if quantiles_a.shape != expected or quantiles_b.shape != expected:
        raise RuntimeError(
            f"quantile cache shapes {quantiles_a.shape}/{quantiles_b.shape}, "
            f"expected {expected}")

    configs = [
        (float(weight_b), float(temperature))
        for weight_b in args.quantile_weights_b
        for temperature in args.width_temperatures
    ]
    accumulators = {config: empty_accumulator(args.residual_scales)
                    for config in configs}
    baseline_accumulator = empty_accumulator(args.residual_scales)
    levels = torch.tensor(
        (0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95),
        device=device, dtype=torch.float32)
    std = torch.tensor(
        (0.096810407937, 0.015963643789), device=device,
        dtype=torch.float32).view(1, 1, 1, 1, 2, 1)
    samples = base["prediction"].shape[0]
    for start in range(0, samples, args.score_batch_size):
        stop = min(samples, start + args.score_batch_size)
        prediction = torch.from_numpy(base["prediction"][start:stop]).to(device)
        target = torch.from_numpy(base["target"][start:stop]).to(device)
        prediction_a = torch.from_numpy(
            base["prediction_a"][start:stop]).to(device)
        prediction_b = torch.from_numpy(
            base["prediction_b"][start:stop]).to(device)
        correction = prediction - (
            prediction_a + args.weight_b * (prediction_b - prediction_a))
        factor = torch.from_numpy(sample_factor(
            base["prediction"][start:stop],
            base["target"][start:stop])).to(device)
        native_lower = prediction - torch.from_numpy(
            base["lower"][start:stop]).to(device)
        native_upper = prediction + torch.from_numpy(
            base["upper"][start:stop]).to(device)
        update_accumulator(
            baseline_accumulator, prediction, target, factor,
            native_lower, native_upper, args.residual_scales)

        qa = torch.from_numpy(np.asarray(
            quantiles_a[start:stop], dtype=np.float32)).to(device)
        qb = torch.from_numpy(np.asarray(
            quantiles_b[start:stop], dtype=np.float32)).to(device)
        outcome_a = prediction_a.unsqueeze(-1) + qa * std
        outcome_b = prediction_b.unsqueeze(-1) + qb * std
        for config in configs:
            quantile_weight_b, temperature = config
            fused = outcome_a.lerp(outcome_b, quantile_weight_b)
            fused = fused + correction.unsqueeze(-1)
            lower, upper = select_interval(fused, levels, temperature)
            update_accumulator(
                accumulators[config], prediction, target, factor,
                lower, upper, args.residual_scales)
        if stop == samples or stop % 80 == 0:
            print(json.dumps({
                "event": "score_progress", "samples": stop,
                "total": samples}), flush=True)

    baseline = finalize(baseline_accumulator)
    results = {}
    for config, accumulator in accumulators.items():
        weight_b, temperature = config
        key = f"quantile_b={weight_b:.5f}/width_temp={temperature:.5f}"
        results[key] = {
            "quantile_weight_b": weight_b,
            "width_temperature": temperature,
            **finalize(accumulator),
        }
    best_local_key = max(
        results,
        key=lambda key: results[key]["scenarios"]["1.00"]["sps_score"])
    best_robust_key = max(
        results,
        key=lambda key: (
            results[key]["scenarios"]["1.00"]["sps_score"]
            + 0.35 * results[key]["scenarios"]["1.30"]["sps_score"]))
    report = {
        "method": "blend_all_residual_quantiles_then_select_one_interval",
        "point_weight_b": args.weight_b,
        "mean_expert_weight": args.mean_expert_weight,
        "baseline_selected_endpoint_blend": baseline,
        "best_local": results[best_local_key],
        "best_local_delta": (
            results[best_local_key]["scenarios"]["1.00"]["sps_score"]
            - baseline["scenarios"]["1.00"]["sps_score"]),
        "best_robust": results[best_robust_key],
        "results": results,
        "config": json.loads(json.dumps(vars(args), default=str)),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "baseline": baseline,
        "best_local": report["best_local"],
        "best_local_delta": report["best_local_delta"],
        "best_robust": report["best_robust"],
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
