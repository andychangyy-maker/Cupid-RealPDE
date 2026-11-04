#!/usr/bin/env python3
"""Recalibrate an already-trained FNO interval head without retraining it."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
_train_spec = importlib.util.spec_from_file_location(
    "realpde_interval_training",
    REPO_ROOT / "code/training/train_fno_interval_head.py")
if _train_spec is None or _train_spec.loader is None:
    raise ImportError("Cannot import train_fno_interval_head.py")
_train = importlib.util.module_from_spec(_train_spec)
sys.modules[_train_spec.name] = _train
_train_spec.loader.exec_module(_train)
_repro = _train._repro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-head", type=Path,
                        help="Optionally save a copy with the new calibration metadata.")
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--stats", type=Path,
        default=Path("kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--robust-residual-scale", type=float, default=1.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.batch_size <= 0 or not math.isfinite(args.robust_residual_scale)
            or args.robust_residual_scale <= 0):
        raise ValueError("Batch size and robust residual scale must be positive")
    device = torch.device(args.device)
    trajectories, metadata = _repro.load_trajectories(args.real_dir, 0)
    filenames = [entry["file"] for entry in metadata]
    _train_windows, val_windows = _train.validation_windows(
        args.split_manifest, filenames)
    loader = DataLoader(
        _repro.WindowDataset(trajectories, val_windows), batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True)
    normalizer = _repro.Normalizer(args.stats, device)
    baseline = _repro.load_baseline_module(args.kit_dir)
    point_model, point_metadata = baseline.load_baseline(
        "fno", str(args.point_checkpoint), device=str(device))
    point_model.eval()

    payload = torch.load(args.head_checkpoint, map_location=device, weights_only=False)
    head = _train.FNOIntervalHead(hidden=int(payload["head_hidden"])).to(device)
    head.load_state_dict(payload["head_state_dict"])
    cache = _train.collect_interval_cache(
        point_model, head, loader, normalizer, device)
    local_multiplier, local_grid = _train.calibrate_multipliers(cache, 1.0)
    robust_multiplier, robust_grid = _train.calibrate_multipliers(
        cache, args.robust_residual_scale)
    local_metrics = _train.exact_sps_for_cache(cache, tuple(local_multiplier))
    robust_on_local = _train.exact_sps_for_cache(cache, tuple(robust_multiplier))
    robust_stress = _train.exact_sps_for_cache(
        cache, tuple(robust_multiplier), args.robust_residual_scale)

    result = {
        "point_checkpoint": str(args.point_checkpoint),
        "point_checkpoint_metadata": point_metadata,
        "head_checkpoint": str(args.head_checkpoint),
        "samples": len(val_windows),
        "local_multiplier_uv": local_multiplier,
        "robust_multiplier_uv": robust_multiplier,
        "robust_residual_scale": args.robust_residual_scale,
        "local_metrics": local_metrics,
        "robust_multiplier_on_local_metrics": robust_on_local,
        "robust_stress_metrics": robust_stress,
        "local_grid": local_grid,
        "robust_grid": robust_grid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if args.output_head is not None:
        output_payload = {
            **payload,
            "point_checkpoint": str(args.point_checkpoint),
            "point_checkpoint_metadata": point_metadata,
            "local_multiplier_uv": local_multiplier,
            "robust_multiplier_uv": robust_multiplier,
            "robust_residual_scale": args.robust_residual_scale,
            "local_metrics": local_metrics,
            "robust_multiplier_on_local_metrics": robust_on_local,
            "robust_stress_metrics": robust_stress,
        }
        args.output_head.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_payload, args.output_head)
    print(json.dumps({
        "local_multiplier_uv": local_multiplier,
        "local_metrics": local_metrics,
        "robust_multiplier_uv": robust_multiplier,
        "robust_on_local": robust_on_local,
        "robust_stress": robust_stress,
    }), flush=True)


if __name__ == "__main__":
    main()
