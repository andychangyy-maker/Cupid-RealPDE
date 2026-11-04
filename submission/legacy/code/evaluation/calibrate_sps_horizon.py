#!/usr/bin/env python3
"""Calibrate separate symmetric SPS widths for each forecast horizon and channel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_constant_spec = importlib.util.spec_from_file_location(
    "realpde_constant_calibration",
    REPO_ROOT / "code/evaluation/calibrate_sps_bounds.py")
if _constant_spec is None or _constant_spec.loader is None:
    raise ImportError("Cannot import calibrate_sps_bounds.py")
_constant = importlib.util.module_from_spec(_constant_spec)
sys.modules[_constant_spec.name] = _constant
_constant_spec.loader.exec_module(_constant)
_repro = _constant._repro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--stats", type=Path,
        default=Path("kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-width", type=float, default=0.04)
    parser.add_argument("--width-step", type=float, default=0.00025)
    parser.add_argument("--robust-residual-scale", type=float, default=1.65)
    return parser.parse_args()


def optimize_width(residual: np.ndarray, factor: np.ndarray, widths: np.ndarray,
                   residual_scale: float, sigma: float) -> tuple[float, float]:
    order = np.argsort(residual)
    residual = residual[order]
    cumulative = np.empty(residual.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(factor[order], dtype=np.float64, out=cumulative[1:])
    indices = np.searchsorted(residual, widths / residual_scale, side="right")
    values = cumulative[indices] * np.exp(-2.0 * widths / sigma)
    best = int(np.argmax(values))
    return float(widths[best]), float(values[best])


def stress_metrics(pred: np.ndarray, target: np.ndarray, factor: np.ndarray,
                   half_width: np.ndarray, residual_scale: float,
                   sigma: float) -> dict[str, float]:
    scored = target[..., :2] != 0.0
    error = np.abs(target[..., :2] - pred[..., :2]) * residual_scale
    width = np.broadcast_to(half_width[None, :, None, None, :], error.shape)
    inside = error <= width
    element_factor = np.broadcast_to(
        factor.reshape((-1, 1, 1, 1, 1)), error.shape)
    numerator = np.sum(
        element_factor * inside * np.exp(-2.0 * width / sigma),
        where=scored, dtype=np.float64)
    denominator = int(np.count_nonzero(scored))
    return {
        "sps_score": float(100.0 * numerator / denominator),
        "coverage": float(np.count_nonzero(inside & scored) / denominator),
        "mean_width": float(np.sum(2.0 * width, where=scored, dtype=np.float64)
                            / denominator),
    }


def main() -> None:
    args = parse_args()
    if (args.batch_size <= 0 or args.max_width <= 0 or args.width_step <= 0
            or not np.isfinite(args.robust_residual_scale)
            or args.robust_residual_scale <= 0):
        raise ValueError("Batch/width values and residual scale must be positive")
    device = torch.device(args.device)
    trajectories, metadata = _repro.load_trajectories(args.real_dir, 0)
    filenames = [entry["file"] for entry in metadata]
    windows = _constant.validation_windows(args.split_manifest, filenames)
    loader = DataLoader(
        _repro.WindowDataset(trajectories, windows), batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True)
    normalizer = _repro.Normalizer(args.stats, device)
    baseline = _repro.load_baseline_module(args.kit_dir)
    official = _constant.import_module(
        args.kit_dir / "scoring.py", "realpde_horizon_scoring")
    model, checkpoint_metadata = baseline.load_baseline(
        "fno", str(args.checkpoint), device=str(device))
    pred, target = _constant.predict_validation(model, loader, normalizer, device)
    channels = official.measured_channels(target)
    if channels != 2:
        raise ValueError(f"Expected two measured velocity channels, got {channels}")
    factor = _constant.sample_sps_factor(official, pred, target, channels)
    widths = np.arange(
        0.0, args.max_width + 0.5 * args.width_step,
        args.width_step, dtype=np.float64)

    selected = {}
    for label, scale in (("local", 1.0), ("robust", args.robust_residual_scale)):
        half_width = np.empty((pred.shape[1], channels), dtype=np.float64)
        objective_mass = 0.0
        for timestep in range(pred.shape[1]):
            for channel in range(channels):
                scored = target[:, timestep, ..., channel] != 0.0
                residual = np.abs(
                    target[:, timestep, ..., channel]
                    - pred[:, timestep, ..., channel])[scored]
                element_factor = np.broadcast_to(
                    factor[:, None, None], scored.shape)[scored]
                half_width[timestep, channel], mass = optimize_width(
                    residual, element_factor, widths, float(scale),
                    official.SIGMA_GLOBAL)
                objective_mass += mass
        metrics = stress_metrics(
            pred, target, factor, half_width, float(scale), official.SIGMA_GLOBAL)
        selected[label] = {
            "residual_scale": float(scale),
            "half_width_by_horizon_uv": half_width.tolist(),
            "metrics_at_scale": metrics,
            "objective_sps_score": float(
                100.0 * objective_mass
                / np.count_nonzero(target[..., :channels] != 0.0)),
        }

    local_width = np.asarray(selected["local"]["half_width_by_horizon_uv"])
    robust_width = np.asarray(selected["robust"]["half_width_by_horizon_uv"])
    local_lower = pred[..., :channels] - local_width[None, :, None, None, :]
    local_upper = pred[..., :channels] + local_width[None, :, None, None, :]
    robust_lower = pred[..., :channels] - robust_width[None, :, None, None, :]
    robust_upper = pred[..., :channels] + robust_width[None, :, None, None, :]
    local_sps, local_coverage = official.aggregate_sps(
        pred, target, channels, local_lower, local_upper)
    robust_local_sps, robust_local_coverage = official.aggregate_sps(
        pred, target, channels, robust_lower, robust_upper)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "split_manifest": str(args.split_manifest),
        "samples": int(pred.shape[0]),
        "width_step": args.width_step,
        "maximum_width": args.max_width,
        "local": {
            **selected["local"],
            "official_local_metrics": {
                "sps_score": float(100.0 * local_sps),
                "coverage": float(local_coverage),
            },
        },
        "robust": {
            **selected["robust"],
            "official_local_metrics": {
                "sps_score": float(100.0 * robust_local_sps),
                "coverage": float(robust_local_coverage),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
