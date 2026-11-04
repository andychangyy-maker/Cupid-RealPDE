#!/usr/bin/env python3
"""Evaluate width-only ensembles of trained structured SPS heads."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/uncertainty"))

from structured_interval_head import StructuredIntervalHead

_train_spec = importlib.util.spec_from_file_location(
    "realpde_structured_train",
    REPO_ROOT / "code/training/train_structured_interval_head.py")
if _train_spec is None or _train_spec.loader is None:
    raise ImportError("Cannot import train_structured_interval_head.py")
_train = importlib.util.module_from_spec(_train_spec)
sys.modules[_train_spec.name] = _train
_train_spec.loader.exec_module(_train)
_base = _train._base
_repro = _train._repro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--robust-residual-scale", type=float, default=1.65)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--stats", type=Path,
        default=Path("kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    return parser.parse_args()


def load_head(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    head = StructuredIntervalHead(
        hidden=int(payload["head_hidden"]),
        architecture=payload["architecture"],
        asymmetric=bool(payload["asymmetric"]),
        global_context=bool(payload.get("global_context", False)),
    ).to(device)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    return head, payload


def main() -> None:
    args = parse_args()
    if len(args.head_checkpoints) < 2:
        raise ValueError("At least two head checkpoints are required")
    device = torch.device(args.device)
    trajectories, metadata = _repro.load_trajectories(args.real_dir, 0)
    filenames = [entry["file"] for entry in metadata]
    _train_windows, val_windows = _base.validation_windows(
        args.split_manifest, filenames)
    loader = DataLoader(
        _repro.WindowDataset(trajectories, val_windows),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
        pin_memory=True)
    normalizer = _repro.Normalizer(args.stats, device)
    baseline = _repro.load_baseline_module(args.kit_dir)
    point_model, _metadata = baseline.load_baseline(
        "fno", str(args.point_checkpoint), device=str(device))
    point_model.eval()
    heads = [load_head(path, device)[0] for path in args.head_checkpoints]

    methods = ("arithmetic", "geometric", "minimum", "maximum")
    common = [dict(residual=[], factor=[]) for _ in range(2)]
    widths = {
        method: [dict(lower=[], upper=[]) for _ in range(2)]
        for method in methods
    }
    with torch.inference_mode():
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)[..., :2]
            prediction = _base.point_prediction(
                point_model, x, normalizer)[..., :2]
            head_widths = [head(prediction, x) for head in heads]
            lower_stack = torch.stack([pair[0] for pair in head_widths])
            upper_stack = torch.stack([pair[1] for pair in head_widths])
            combinations = {
                "arithmetic": (
                    lower_stack.mean(dim=0), upper_stack.mean(dim=0)),
                "geometric": (
                    lower_stack.log().mean(dim=0).exp(),
                    upper_stack.log().mean(dim=0).exp()),
                "minimum": (
                    lower_stack.min(dim=0).values,
                    upper_stack.min(dim=0).values),
                "maximum": (
                    lower_stack.max(dim=0).values,
                    upper_stack.max(dim=0).values),
            }
            factor = _base.sample_accuracy_factor(
                prediction, target).reshape(-1, 1, 1, 1)
            for channel in range(2):
                scored = target[..., channel] != 0.0
                expanded_factor = factor.expand_as(target[..., channel])
                common[channel]["residual"].append(
                    (target[..., channel] - prediction[..., channel])[scored])
                common[channel]["factor"].append(expanded_factor[scored])
                for method, (lower, upper) in combinations.items():
                    widths[method][channel]["lower"].append(
                        lower[..., channel][scored])
                    widths[method][channel]["upper"].append(
                        upper[..., channel][scored])

    common = [
        {key: torch.cat(value) for key, value in channel.items()}
        for channel in common
    ]
    results = {}
    for method in methods:
        cache = []
        for channel in range(2):
            cache.append({
                **common[channel],
                "lower": torch.cat(widths[method][channel]["lower"]),
                "upper": torch.cat(widths[method][channel]["upper"]),
            })
        local_lower, local_upper, _local_grid = _train.calibrate_multipliers(
            cache, True, 1.0)
        robust_lower, robust_upper, _robust_grid = _train.calibrate_multipliers(
            cache, True, args.robust_residual_scale)
        results[method] = {
            "local_lower_multiplier_uv": local_lower,
            "local_upper_multiplier_uv": local_upper,
            "robust_lower_multiplier_uv": robust_lower,
            "robust_upper_multiplier_uv": robust_upper,
            "local_metrics": _train.exact_sps_for_cache(
                cache, local_lower, local_upper),
            "robust_on_local_metrics": _train.exact_sps_for_cache(
                cache, robust_lower, robust_upper),
            "robust_stress_metrics": _train.exact_sps_for_cache(
                cache, robust_lower, robust_upper,
                args.robust_residual_scale),
        }
    output = {
        "point_checkpoint": str(args.point_checkpoint),
        "head_checkpoints": [str(path) for path in args.head_checkpoints],
        "validation_windows": len(val_windows),
        "robust_residual_scale": args.robust_residual_scale,
        "methods": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
