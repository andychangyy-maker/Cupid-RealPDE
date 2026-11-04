#!/usr/bin/env python3
"""Audit robust SPS calibration across hidden-residual scale hypotheses."""

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

_spec = importlib.util.spec_from_file_location(
    "realpde_structured_train",
    REPO_ROOT / "code/training/train_structured_interval_head.py")
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot import structured interval training module")
_train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train
_spec.loader.exec_module(_train)
_base = _train._base
_repro = _train._repro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--residual-scales", type=float, nargs="+",
        default=(1.65, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0))
    parser.add_argument(
        "--reference-multipliers", type=float, nargs=4,
        default=(1.4, 1.5, 1.4, 1.5),
        metavar=("LOWER_U", "LOWER_V", "UPPER_U", "UPPER_V"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--stats", type=Path,
        default=Path("kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(value <= 0 for value in args.residual_scales):
        raise ValueError("Residual scales must be positive")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    payload = torch.load(args.head_checkpoint, map_location=device, weights_only=False)
    head = StructuredIntervalHead(
        hidden=int(payload["head_hidden"]),
        architecture=payload["architecture"],
        asymmetric=bool(payload["asymmetric"]),
        global_context=bool(payload.get("global_context", False)),
    ).to(device)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    cache = _train.collect_interval_cache(
        point_model, head, loader, normalizer, device)

    reference_lower = list(args.reference_multipliers[:2])
    reference_upper = list(args.reference_multipliers[2:])
    results = {}
    for scale in args.residual_scales:
        lower, upper, _trace = _train.calibrate_multipliers(
            cache, asymmetric=True, residual_scale=scale)
        result = {
            "reference_multiplier_stress_metrics": _train.exact_sps_for_cache(
                cache, reference_lower, reference_upper, scale),
            "optimal_lower_multiplier_uv": lower,
            "optimal_upper_multiplier_uv": upper,
            "optimal_on_local_metrics": _train.exact_sps_for_cache(
                cache, lower, upper, 1.0),
            "optimal_stress_metrics": _train.exact_sps_for_cache(
                cache, lower, upper, scale),
        }
        results[str(scale)] = result
        candidate = dict(payload)
        candidate["robust_lower_multiplier_uv"] = lower
        candidate["robust_upper_multiplier_uv"] = upper
        candidate["robust_residual_scale"] = scale
        candidate["residual_scale_audit"] = result
        torch.save(candidate, args.output_dir / f"interval_head_scale_{scale:.2f}.pth")
        print(json.dumps({"event": "scale", "scale": scale, **result}), flush=True)

    report = {
        "point_checkpoint": str(args.point_checkpoint),
        "head_checkpoint": str(args.head_checkpoint),
        "validation_windows": len(val_windows),
        "reference_lower_multiplier_uv": reference_lower,
        "reference_upper_multiplier_uv": reference_upper,
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"event": "complete", "output_dir": str(args.output_dir)}),
          flush=True)


if __name__ == "__main__":
    main()
