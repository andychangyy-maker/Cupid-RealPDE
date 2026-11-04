#!/usr/bin/env python3
"""Decompose native U-Net UQ into budget, allocation, and oracle gaps.

This is a diagnostic, not a calibrator. It asks whether SPS is limited by
the sample/horizon/channel amount of uncertainty or by where that budget is
placed spatially. Target-dependent variants are explicitly labelled oracle
upper bounds and must never be used by a submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy import stats
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from audit_unet_native_intervals import collect, import_module  # noqa: E402
from train_flow_backbone import construct_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--oracle-grid-min", type=float, default=0.20)
    parser.add_argument("--oracle-grid-max", type=float, default=3.00)
    parser.add_argument("--oracle-grid-step", type=float, default=0.05)
    return parser.parse_args()


def metrics(cache: dict[str, np.ndarray], lower: np.ndarray,
            upper: np.ndarray) -> dict[str, float]:
    residual = cache["residual"]
    scored = cache["scored"]
    factor = cache["factor"].reshape((-1, 1, 1, 1, 1))
    inside = (residual >= -lower) & (residual <= upper)
    penalty = np.exp(-(lower + upper) / common.SPS_SIGMA)
    width = np.broadcast_to(lower + upper, residual.shape)
    count = int(np.count_nonzero(scored))
    score = np.sum(
        factor * penalty * inside, where=scored,
        dtype=np.float64) / count
    return {
        "sps_score": float(100.0 * score),
        "coverage": float(np.count_nonzero(inside & scored) / count),
        "mean_width": float(np.sum(
            width, where=scored, dtype=np.float64) / count),
    }


def masked_mean(values: np.ndarray, mask: np.ndarray,
                axes: tuple[int, ...]) -> np.ndarray:
    numerator = np.sum(values, axis=axes, where=mask, keepdims=True,
                       dtype=np.float64)
    denominator = np.sum(mask, axis=axes, keepdims=True, dtype=np.int64)
    return (numerator / np.maximum(denominator, 1)).astype(np.float32)


def select_oracle_scales(
        cache: dict[str, np.ndarray], grid: np.ndarray,
        group: str) -> np.ndarray:
    """Select target-dependent multipliers on a 2x spatial subsample."""
    residual = cache["residual"][:, :, ::2, ::2]
    lower = cache["lower"][:, :, ::2, ::2]
    upper = cache["upper"][:, :, ::2, ::2]
    scored = cache["scored"][:, :, ::2, ::2]
    if group == "sample":
        reduction = (1, 2, 3, 4)
        shape = (residual.shape[0], 1, 1, 1, 1)
    elif group == "sample_horizon_channel":
        reduction = (2, 3)
        shape = (residual.shape[0], residual.shape[1], 1, 1,
                 residual.shape[4])
    else:
        raise ValueError(group)
    best_score = np.full(shape, -np.inf, dtype=np.float32)
    best_scale = np.ones(shape, dtype=np.float32)
    for scale in grid:
        inside = (residual >= -scale * lower) & (residual <= scale * upper)
        value = np.exp(-scale * (lower + upper) / common.SPS_SIGMA) * inside
        numerator = np.sum(
            value, axis=reduction, where=scored, keepdims=True,
            dtype=np.float64)
        denominator = np.sum(
            scored, axis=reduction, keepdims=True, dtype=np.int64)
        score = (numerator / np.maximum(denominator, 1)).astype(np.float32)
        improved = score > best_score
        best_score = np.where(improved, score, best_score)
        best_scale = np.where(improved, scale, best_scale)
    return best_scale


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def correlations(cache: dict[str, np.ndarray]) -> dict[str, float]:
    residual = cache["residual"]
    required = np.abs(residual)
    predicted = np.where(residual < 0.0, cache["lower"], cache["upper"])
    scored = cache["scored"]
    indices = np.flatnonzero(scored.reshape(-1))[::97]
    log_required = np.log1p(required.reshape(-1)[indices] / 1e-4)
    log_predicted = np.log1p(predicted.reshape(-1)[indices] / 1e-4)
    element_pearson = np.corrcoef(log_required, log_predicted)[0, 1]

    sample_required = masked_mean(required, scored, (1, 2, 3, 4)).reshape(-1)
    sample_predicted = masked_mean(
        predicted, scored, (1, 2, 3, 4)).reshape(-1)
    return {
        "sample_count": int(cache["factor"].shape[0]),
        "element_subsample_count": int(indices.size),
        "element_log_pearson": float(element_pearson),
        "element_log_spearman": safe_spearman(log_required, log_predicted),
        "sample_mean_pearson": float(np.corrcoef(
            sample_required, sample_predicted)[0, 1]),
        "sample_mean_spearman": safe_spearman(
            sample_required, sample_predicted),
    }


def main() -> None:
    args = parse_args()
    if (args.eval_batch_size <= 0 or args.oracle_grid_min <= 0.0
            or args.oracle_grid_step <= 0.0
            or args.oracle_grid_max < args.oracle_grid_min):
        raise ValueError("invalid positive diagnostic arguments")
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = payload["model_config"]["architecture"]
    model = construct_model(architecture, payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=args.output.parent, seed=0)
    (train_loader, val_loader, normalizer, train_windows,
     val_windows, trajectories) = common.prepare_real(data_args, device)
    common.set_model_stats(model, normalizer)
    official = import_module(args.kit_dir / "scoring.py", "uq_factor_scoring")
    cache = collect(model, val_loader, normalizer, device, official)
    del model, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    scored = cache["scored"]
    lower = cache["lower"]
    upper = cache["upper"]
    results = {"native": metrics(cache, lower, upper)}

    sample_hc_lower = masked_mean(lower, scored, (2, 3))
    sample_hc_upper = masked_mean(upper, scored, (2, 3))
    results["sample_horizon_channel_uniform_space"] = metrics(
        cache, sample_hc_lower, sample_hc_upper)

    hc_lower = masked_mean(lower, scored, (0, 2, 3))
    hc_upper = masked_mean(upper, scored, (0, 2, 3))
    results["horizon_channel_uniform_sample_space"] = metrics(
        cache, hc_lower, hc_upper)

    results["vertical_reflection_allocation"] = metrics(
        cache, np.flip(lower, axis=2), np.flip(upper, axis=2))

    grid = np.arange(
        args.oracle_grid_min,
        args.oracle_grid_max + 0.5 * args.oracle_grid_step,
        args.oracle_grid_step, dtype=np.float32)
    sample_scale = select_oracle_scales(cache, grid, "sample")
    results["oracle_sample_scalar"] = {
        **metrics(cache, lower * sample_scale, upper * sample_scale),
        "scale_quantiles": {
            str(q): float(np.quantile(sample_scale, q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
    }
    sample_hc_scale = select_oracle_scales(
        cache, grid, "sample_horizon_channel")
    results["oracle_sample_horizon_channel_scalar"] = {
        **metrics(cache, lower * sample_hc_scale, upper * sample_hc_scale),
        "scale_quantiles": {
            str(q): float(np.quantile(sample_hc_scale, q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
    }

    residual = cache["residual"]
    epsilon = np.float32(1e-7)
    oracle_lower = np.maximum(-residual, 0.0) + epsilon
    oracle_upper = np.maximum(residual, 0.0) + epsilon
    results["oracle_per_element_minimal"] = metrics(
        cache, oracle_lower, oracle_upper)

    report = {
        "checkpoint": str(args.checkpoint),
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "correlations": correlations(cache),
        "variants": results,
        "interpretation": {
            "uniform_space": (
                "Difference from native measures value of spatial allocation."),
            "oracle_sample_scalar": (
                "Target-dependent upper bound for a sample-conditioned budget."),
            "oracle_sample_horizon_channel_scalar": (
                "Target-dependent upper bound for a regime/temporal budget head."),
            "oracle_per_element_minimal": (
                "Unattainable target-dependent ceiling, never submission-safe."),
        },
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(args.output),
                      "variants": results}), flush=True)


if __name__ == "__main__":
    main()
