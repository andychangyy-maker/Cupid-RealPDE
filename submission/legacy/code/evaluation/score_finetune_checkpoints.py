#!/usr/bin/env python3
"""Score fine-tuning checkpoints with the bundled official Track 1 formulas."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_repro_spec = importlib.util.spec_from_file_location(
    "realpde_reproduce_finetune", REPO_ROOT / "code/training/reproduce_finetune.py")
if _repro_spec is None or _repro_spec.loader is None:
    raise ImportError("Cannot import code/training/reproduce_finetune.py")
_repro = importlib.util.module_from_spec(_repro_spec)
sys.modules[_repro_spec.name] = _repro
_repro_spec.loader.exec_module(_repro)
Normalizer = _repro.Normalizer
Window = _repro.Window
WindowDataset = _repro.WindowDataset
load_baseline_module = _repro.load_baseline_module
load_trajectories = _repro.load_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cno", "fno", "transolver"), required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--stats", type=Path,
                        default=Path("kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument("--kit-dir", type=Path,
                        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--interval-half-widths", type=float, nargs=2, metavar=("U", "V"),
        help=("Optional constant physical-space half-widths for the u/v prediction "
              "intervals. If omitted, the official scorer's default +/-5%% of "
              "abs(prediction) interval is used."))
    parser.add_argument("--continuity-weight", type=float, default=0.0,
                        help="Causal input/output boundary correction weight.")
    parser.add_argument("--continuity-power", type=float, default=4.0,
                        help="Temporal decay power for the boundary correction.")
    parser.add_argument("--interval-grid-u", type=float, nargs="+",
                        help="Optional local SPS calibration grid for u half-width.")
    parser.add_argument("--interval-grid-v", type=float, nargs="+",
                        help="Optional local SPS calibration grid for v half-width.")
    return parser.parse_args()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_checkpoints(items: list[str]) -> list[tuple[str, Path]]:
    result = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Checkpoint must be LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        if not label or not path:
            raise ValueError(f"Checkpoint must be LABEL=PATH, got {item!r}")
        result.append((label, Path(path)))
    return result


def validation_windows(manifest_path: Path, filenames: list[str]) -> list[Window]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    by_file = {entry["file"]: entry for entry in manifest}
    if set(by_file) != set(filenames):
        missing = sorted(set(filenames) - set(by_file))
        extra = sorted(set(by_file) - set(filenames))
        raise ValueError(f"Split/data mismatch; missing={missing}, extra={extra}")
    return [Window(index, int(start))
            for index, filename in enumerate(filenames)
            for start in by_file[filename]["val_starts"]]


def score_model(model: torch.nn.Module, loader: DataLoader, normalizer: Normalizer,
                official, device: torch.device,
                interval_half_widths: np.ndarray | None = None,
                continuity_weight: float = 0.0,
                continuity_power: float = 4.0,
                interval_grid: tuple[list[float], list[float]] | None = None,
                ) -> dict[str, object]:
    model.eval()
    predictions = []
    targets = []
    neural_seconds = 0.0
    samples = 0
    with torch.no_grad():
        for x, y in loader:
            observed_cpu = x
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            x = x.to(device, non_blocking=True)
            x_norm = (x - normalizer.mean_input) / normalizer.std_input
            pred_norm = model(x_norm)
            pred = normalizer.postprocess_prediction(pred_norm).float().cpu()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            neural_seconds += time.perf_counter() - started
            if continuity_weight:
                jump = observed_cpu[:, -1:, ..., :2] - pred[:, :1, ..., :2]
                ramp = torch.linspace(1.0, 0.0, pred.shape[1]).pow(continuity_power)
                ramp = ramp.reshape(1, -1, 1, 1, 1)
                pred[..., :2] += continuity_weight * ramp * jump
            pred[..., 2] = 0.0
            predictions.append(pred.numpy())
            targets.append(y.numpy())
            samples += x.shape[0]

    pred_all = np.concatenate(predictions).astype(np.float32, copy=False)
    target_all = np.concatenate(targets).astype(np.float32, copy=False)
    channels = official.measured_channels(target_all)
    rel_l2 = float(np.mean(official.rel_l2_per_sample(pred_all, target_all, channels)))
    tke = float(np.mean(official.tke_rel_l2_per_sample(pred_all, target_all, channels)))
    mvpe = float(official.mvpe_rel_l2(pred_all, target_all))
    lower = None
    upper = None
    if interval_half_widths is not None:
        half_width = np.zeros(pred_all.shape[-1], dtype=np.float32)
        half_width[:2] = interval_half_widths
        half_width = half_width.reshape((1,) * (pred_all.ndim - 1) + (-1,))
        lower = pred_all - half_width
        upper = pred_all + half_width
    sps, coverage = official.aggregate_sps(
        pred_all, target_all, channels, lower=lower, upper=upper)
    mean_time = neural_seconds / samples
    subscores = {
        "rel_l2_score": official.score_error(rel_l2),
        "tke_score": official.score_error(tke),
        "mvpe_score": official.score_error(mvpe),
        "time_score": official.score_time(mean_time),
        "sps_score": official.score_sps(sps),
    }
    result = {
        "raw_errors": {"rel_l2": rel_l2, "tke_rel_l2": tke, "mvpe_rel_l2": mvpe},
        "subscores": subscores,
        "subscore_mean_unofficial": float(np.mean(list(subscores.values()))),
        "mean_t_neural_s": mean_time,
        "total_timed_s": neural_seconds,
        "sps_raw": float(sps),
        "sps_coverage": float(coverage),
        "interval_half_widths_uv": (
            interval_half_widths.tolist() if interval_half_widths is not None else None),
        "continuity_weight": continuity_weight,
        "continuity_power": continuity_power,
        "samples": samples,
        "batch_size": loader.batch_size,
    }
    if interval_grid is not None:
        grid_results = []
        for u_width in interval_grid[0]:
            for v_width in interval_grid[1]:
                grid_width = np.asarray([u_width, v_width, 0.0], dtype=np.float32)
                grid_width = grid_width.reshape((1,) * (pred_all.ndim - 1) + (-1,))
                grid_sps, grid_coverage = official.aggregate_sps(
                    pred_all, target_all, channels,
                    lower=pred_all - grid_width, upper=pred_all + grid_width)
                grid_results.append({
                    "u_half_width": float(u_width),
                    "v_half_width": float(v_width),
                    "sps_raw": float(grid_sps),
                    "sps_score": official.score_sps(grid_sps),
                    "coverage": float(grid_coverage),
                })
        result["interval_grid"] = sorted(
            grid_results, key=lambda item: item["sps_score"], reverse=True)
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if (not np.isfinite(args.continuity_weight) or args.continuity_weight < 0 or
            not np.isfinite(args.continuity_power) or args.continuity_power <= 0):
        raise ValueError("continuity weight must be non-negative and power must be positive")
    interval_half_widths = None
    if args.interval_half_widths is not None:
        interval_half_widths = np.asarray(args.interval_half_widths, dtype=np.float32)
        if not np.all(np.isfinite(interval_half_widths)) or np.any(interval_half_widths < 0):
            raise ValueError("interval half-widths must be finite and non-negative")
    if (args.interval_grid_u is None) != (args.interval_grid_v is None):
        raise ValueError("interval-grid-u and interval-grid-v must be provided together")
    interval_grid = None
    if args.interval_grid_u is not None:
        grid_values = np.asarray(args.interval_grid_u + args.interval_grid_v, dtype=float)
        if not np.all(np.isfinite(grid_values)) or np.any(grid_values < 0):
            raise ValueError("interval grid values must be finite and non-negative")
        interval_grid = (args.interval_grid_u, args.interval_grid_v)
    device = torch.device(args.device)
    trajectories, metadata = load_trajectories(args.real_dir, 0)
    filenames = [item["file"] for item in metadata]
    windows = validation_windows(args.split_manifest, filenames)
    dataset = WindowDataset(trajectories, windows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    normalizer = Normalizer(args.stats, device)
    baseline_loader = load_baseline_module(args.kit_dir)
    official = import_module(args.kit_dir / "scoring.py", "realpde_official_scoring")

    result = {
        "formula_source": str(args.kit_dir / "scoring.py"),
        "split_manifest": str(args.split_manifest),
        "time_note": "Local hardware, Track-1 wrapper batch size; not leaderboard-comparable.",
        "interval_half_widths_uv": (
            interval_half_widths.tolist() if interval_half_widths is not None else None),
        "continuity_weight": args.continuity_weight,
        "continuity_power": args.continuity_power,
        "models": {},
    }
    for label, checkpoint in parse_checkpoints(args.checkpoint):
        model, metadata = baseline_loader.load_baseline(
            args.model, str(checkpoint), device=str(device))
        scored = score_model(
            model, loader, normalizer, official, device,
            interval_half_widths=interval_half_widths,
            continuity_weight=args.continuity_weight,
            continuity_power=args.continuity_power,
            interval_grid=interval_grid)
        scored["checkpoint"] = str(checkpoint)
        scored["checkpoint_metadata"] = metadata
        result["models"][label] = scored
        print(json.dumps({"label": label, **scored["subscores"]}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({"output": str(args.output), "models": list(result["models"])}), flush=True)


if __name__ == "__main__":
    main()
