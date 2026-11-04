#!/usr/bin/env python3
"""Compare condition-fold ensemble UQ decoders under residual inflation.

Residual scales above one are explicit diagnostics for Sim2Real domain shift;
they are not validation scores. Every candidate is evaluated on the exact Real
condition fold that was excluded while its UQ decoder was trained.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_ensemble_uq_oof as uq_train  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", required=True,
        help="Candidate in LABEL=OOF_DIRECTORY form; repeat as needed.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--residual-scales", type=float, nargs="+",
                        default=(1.0, 1.15, 1.30))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def candidates(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("candidate must use LABEL=OOF_DIRECTORY")
        if label in result:
            raise ValueError(f"duplicate candidate label: {label}")
        path = Path(raw_path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        result[label] = path
    return result


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    total = sum(row["scored_elements"] for row in rows)
    if total <= 0:
        raise RuntimeError("cannot aggregate an empty stress evaluation")
    return {
        key: sum(row[key] * row["scored_elements"] for row in rows) / total
        for key in ("sps_score", "coverage", "mean_width")
    } | {"scored_elements": total}


def main() -> None:
    args = parse_args()
    candidate_dirs = candidates(args.candidate)
    if args.folds < 2 or args.window_stride <= 0 or args.batch_size <= 0:
        raise ValueError("folds, stride and batch size must be positive")
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("val_fraction must lie in (0, 0.5)")
    if (not args.residual_scales
            or any(not math.isfinite(value) or value <= 0
                   for value in args.residual_scales)):
        raise ValueError("residual scales must be finite and positive")

    device = torch.device(args.device)
    payload = torch.load(
        args.point_checkpoint, map_location="cpu", weights_only=False)
    trajectories, metadata = common.repro.load_trajectories(
        args.real_dir, args.max_files)
    _point_windows, tail_windows, manifest = common.repro.make_split(
        metadata, args.window_stride, args.val_fraction)
    dataset = common.repro.WindowDataset(trajectories, tail_windows)
    sample_folds, fold_manifest = trajectory_folds(manifest, args.folds)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    all_indices = list(range(len(dataset)))
    rows = {label: {str(scale): [] for scale in args.residual_scales}
            for label in candidate_dirs}

    loader_args = argparse.Namespace(num_workers=args.num_workers)
    for fold in range(args.folds):
        eval_indices = [
            index for index in all_indices if sample_folds[index] == fold]
        loader = uq_train.make_loader(
            dataset, eval_indices, args.batch_size, False,
            loader_args, seed=0)
        for label, directory in candidate_dirs.items():
            checkpoint_path = directory / f"fold_{fold:02d}_best.pth"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False)
            model_config = checkpoint.get("model_config", {})
            model = uq_train.load_model(
                payload, normalizer, device, train_projections=False,
                add_flow_context=model_config.get("uq_flow_context", False),
                residual_quantile_intervals=model_config.get(
                    "residual_quantile_intervals", False),
                utility_refinement=model_config.get(
                    "utility_refinement", False),
                utility_refinement_log_limit=model_config.get(
                    "utility_refinement_log_limit", 0.75),
                utility_refinement_kernel=model_config.get(
                    "utility_refinement_kernel", 1),
                utility_refinement_blocks=model_config.get(
                    "utility_refinement_blocks", 0),
                temporal_uq_refinement=model_config.get(
                    "temporal_uq_refinement", False),
                temporal_uq_channels=model_config.get(
                    "temporal_uq_channels", 24),
                temporal_uq_blocks=model_config.get(
                    "temporal_uq_blocks", 2),
                temporal_uq_kernel=model_config.get(
                    "temporal_uq_kernel", 3),
                temporal_uq_logit_limit=model_config.get(
                    "temporal_uq_logit_limit", 1.5),
                coverage_calibration=model_config.get(
                    "coverage_calibration", False),
                coverage_calibration_kernel=model_config.get(
                    "coverage_calibration_kernel", 3),
                coverage_calibration_blocks=model_config.get(
                    "coverage_calibration_blocks", 1),
                coverage_calibration_logit_limit=model_config.get(
                    "coverage_calibration_logit_limit", 2.0),
                coverage_calibration_mode=model_config.get(
                    "coverage_calibration_mode", "independent"),
                coverage_calibration_temperature_limit=model_config.get(
                    "coverage_calibration_temperature_limit", 0.75))
            uq_train.load_uq_state(model, checkpoint["uq_state_dict"])
            fold_result = {}
            for scale in args.residual_scales:
                metrics = uq_train.exact_sps(
                    model, loader, normalizer, device,
                    residual_scale=scale)
                rows[label][str(scale)].append(metrics)
                fold_result[str(scale)] = metrics
            print(json.dumps({
                "event": "candidate_fold", "label": label,
                "fold": fold, "scenarios": fold_result,
            }), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = {
        "point_checkpoint": str(args.point_checkpoint),
        "folds": args.folds,
        "tail_windows": len(dataset),
        "fold_manifest": fold_manifest,
        "residual_scales": args.residual_scales,
        "candidates": {
            label: {
                "directory": str(candidate_dirs[label]),
                "scenarios": {
                    scale: aggregate(scale_rows)
                    for scale, scale_rows in scenarios.items()
                },
                "folds": scenarios,
            }
            for label, scenarios in rows.items()
        },
        "interpretation": (
            "Residual inflation is an explicit domain-shift stress test, "
            "not a held-out validation score."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "stress_complete",
        "output": str(args.output),
        "candidates": {
            label: value["scenarios"]
            for label, value in result["candidates"].items()
        },
    }), flush=True)


if __name__ == "__main__":
    main()
