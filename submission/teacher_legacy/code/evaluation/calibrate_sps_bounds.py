#!/usr/bin/env python3
"""Calibrate constant Track 1 SPS bounds without repeatedly recomputing metrics."""

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

_repro_spec = importlib.util.spec_from_file_location(
    "realpde_sps_reproduce", REPO_ROOT / "code/training/reproduce_finetune.py")
if _repro_spec is None or _repro_spec.loader is None:
    raise ImportError("Cannot import code/training/reproduce_finetune.py")
_repro = importlib.util.module_from_spec(_repro_spec)
sys.modules[_repro_spec.name] = _repro
_repro_spec.loader.exec_module(_repro)


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
    parser.add_argument("--max-width", type=float, default=0.03)
    parser.add_argument("--width-step", type=float, default=0.00025)
    parser.add_argument(
        "--residual-scales", type=float, nargs="+",
        default=(1.0, 1.25, 1.5, 1.75, 2.0),
        help="Also optimize after multiplying validation residuals by each scale.")
    return parser.parse_args()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validation_windows(manifest_path: Path, filenames: list[str]):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_file = {entry["file"]: entry for entry in manifest}
    if set(by_file) != set(filenames):
        raise ValueError("Split manifest and real-data files do not match")
    return [
        _repro.Window(index, int(start))
        for index, filename in enumerate(filenames)
        for start in by_file[filename]["val_starts"]
    ]


def predict_validation(model: torch.nn.Module, loader: DataLoader,
                       normalizer, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            x_norm = (x - normalizer.mean_input) / normalizer.std_input
            pred_norm = model(x_norm)
            pred = normalizer.postprocess_prediction(pred_norm).float().cpu().numpy()
            pred[..., 2] = 0.0
            predictions.append(pred)
            targets.append(target.numpy())
    return (
        np.concatenate(predictions).astype(np.float32, copy=False),
        np.concatenate(targets).astype(np.float32, copy=False),
    )


def sample_sps_factor(official, pred: np.ndarray, target: np.ndarray,
                      channels: int) -> np.ndarray:
    errors = (
        official.rel_l2_per_sample(pred, target, channels),
        official.tke_rel_l2_per_sample(pred, target, channels),
        official.mvpe_rel_l2_per_sample(pred, target),
    )
    factors = []
    for error in errors:
        normalized = error / (0.5 + error)
        factors.append(np.where(np.isfinite(normalized), 1.0 - normalized, 0.0))
    return 0.5 * factors[0] + 0.3 * factors[1] + 0.2 * factors[2]


def sorted_channel_residuals(pred: np.ndarray, target: np.ndarray, channel: int,
                             sample_factor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = target[..., channel] - pred[..., channel]
    scored = target[..., channel] != 0.0
    element_factor = np.broadcast_to(
        sample_factor.reshape((-1,) + (1,) * (residual.ndim - 1)), residual.shape)
    residual = residual[scored].astype(np.float32, copy=False)
    element_factor = element_factor[scored].astype(np.float64, copy=False)
    order = np.argsort(residual)
    sorted_residual = residual[order]
    cumulative_factor = np.empty(sorted_residual.size + 1, dtype=np.float64)
    cumulative_factor[0] = 0.0
    np.cumsum(element_factor[order], out=cumulative_factor[1:])
    return sorted_residual, cumulative_factor


def optimize_channel(sorted_residual: np.ndarray, cumulative_factor: np.ndarray,
                     widths: np.ndarray, residual_scale: float, sigma: float,
                     total_scored: int) -> dict[str, object]:
    lower_index = np.searchsorted(
        sorted_residual, -widths / residual_scale, side="left")
    upper_index = np.searchsorted(
        sorted_residual, widths / residual_scale, side="right")
    lower_prefix = cumulative_factor[lower_index]
    upper_prefix = cumulative_factor[upper_index]

    mass = upper_prefix[None, :] - lower_prefix[:, None]
    penalty = np.exp(-(widths[:, None] + widths[None, :]) / sigma)
    asymmetric = mass * penalty / total_scored
    asym_index = np.unravel_index(int(np.argmax(asymmetric)), asymmetric.shape)

    symmetric_mass = upper_prefix - lower_prefix
    symmetric = symmetric_mass * np.exp(-2.0 * widths / sigma) / total_scored
    sym_index = int(np.argmax(symmetric))
    return {
        "asymmetric": {
            "lower_width": float(widths[asym_index[0]]),
            "upper_width": float(widths[asym_index[1]]),
            "sps_contribution": float(asymmetric[asym_index]),
        },
        "symmetric": {
            "half_width": float(widths[sym_index]),
            "sps_contribution": float(symmetric[sym_index]),
        },
    }


def main() -> None:
    args = parse_args()
    if (args.batch_size <= 0 or args.max_width <= 0 or args.width_step <= 0
            or not np.all(np.isfinite(args.residual_scales))
            or np.any(np.asarray(args.residual_scales) <= 0)):
        raise ValueError("Batch/width values and residual scales must be positive and finite")
    device = torch.device(args.device)
    trajectories, metadata = _repro.load_trajectories(args.real_dir, 0)
    filenames = [entry["file"] for entry in metadata]
    windows = validation_windows(args.split_manifest, filenames)
    dataset = _repro.WindowDataset(trajectories, windows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    normalizer = _repro.Normalizer(args.stats, device)
    baseline = _repro.load_baseline_module(args.kit_dir)
    official = import_module(args.kit_dir / "scoring.py", "realpde_sps_scoring")
    model, checkpoint_metadata = baseline.load_baseline(
        "fno", str(args.checkpoint), device=str(device))
    pred, target = predict_validation(model, loader, normalizer, device)
    del model, trajectories
    if device.type == "cuda":
        torch.cuda.empty_cache()

    channels = official.measured_channels(target)
    factor = sample_sps_factor(official, pred, target, channels)
    scored_count = int(np.count_nonzero(target[..., :channels] != 0.0))
    default_sps, default_coverage = official.aggregate_sps(pred, target, channels)
    legacy_width = np.asarray([0.00975, 0.00525], dtype=np.float32)
    legacy_shape = (1,) * (pred.ndim - 1) + (channels,)
    legacy_width = legacy_width[:channels].reshape(legacy_shape)
    legacy_sps, legacy_coverage = official.aggregate_sps(
        pred, target, channels, pred[..., :channels] - legacy_width,
        pred[..., :channels] + legacy_width)

    widths = np.arange(
        0.0, args.max_width + args.width_step * 0.5,
        args.width_step, dtype=np.float64)
    sorted_channels = [
        sorted_channel_residuals(pred, target, channel, factor)
        for channel in range(channels)
    ]
    scenarios = []
    for scale in args.residual_scales:
        channel_results = [
            optimize_channel(*sorted_channel, widths, float(scale),
                             official.SIGMA_GLOBAL, scored_count)
            for sorted_channel in sorted_channels
        ]
        scenarios.append({
            "residual_scale": float(scale),
            "channels": channel_results,
            "best_symmetric_sps_score": 100.0 * sum(
                result["symmetric"]["sps_contribution"]
                for result in channel_results),
            "best_asymmetric_sps_score": 100.0 * sum(
                result["asymmetric"]["sps_contribution"]
                for result in channel_results),
        })

    accuracy_scores = {
        "rel_l2_score": official.score_error(float(np.mean(
            official.rel_l2_per_sample(pred, target, channels)))),
        "tke_score": official.score_error(float(np.mean(
            official.tke_rel_l2_per_sample(pred, target, channels)))),
        "mvpe_score": official.score_error(official.mvpe_rel_l2(pred, target)),
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "split_manifest": str(args.split_manifest),
        "samples": int(pred.shape[0]),
        "measured_channels": channels,
        "accuracy_scores": accuracy_scores,
        "default_interval": {
            "sps_score": 100.0 * default_sps, "coverage": default_coverage},
        "legacy_symmetric_interval": {
            "half_width_uv": [0.00975, 0.00525],
            "sps_score": 100.0 * legacy_sps, "coverage": legacy_coverage},
        "search": {
            "minimum_width": 0.0,
            "maximum_width": args.max_width,
            "step": args.width_step,
            "sigma_global": official.SIGMA_GLOBAL,
            "scenarios": scenarios,
        },
        "note": (
            "Residual scaling is a robustness stress test. Supplied bounds must still "
            "include finite lower and upper arrays under the v9 interface."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
