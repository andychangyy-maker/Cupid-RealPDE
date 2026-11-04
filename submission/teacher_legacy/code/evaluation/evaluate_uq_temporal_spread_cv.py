#!/usr/bin/env python3
"""Cross-validate horizon/channel spread multipliers for OOF UQ models.

This diagnostic reduces every held-out fold to sufficient statistics over a
fixed multiplier grid.  A fold's multiplier for each forecast horizon and
velocity channel is selected only from the other folds, so ``cv`` is a genuine
condition-held-out estimate.  ``oracle_all`` uses all folds for selection and
is labelled as an upper bound rather than a validation result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
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
    parser.add_argument("--candidate-dir", type=Path, required=True)
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grid-min", type=float, default=0.75)
    parser.add_argument("--grid-max", type=float, default=1.30)
    parser.add_argument("--grid-step", type=float, default=0.025)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_fold_model(payload: dict, checkpoint: dict, normalizer,
                    device: torch.device):
    config = checkpoint["model_config"]
    model = uq_train.load_model(
        payload, normalizer, device, train_projections=False,
        add_flow_context=config.get("uq_flow_context", False),
        residual_quantile_intervals=config.get(
            "residual_quantile_intervals", False),
        utility_refinement=config.get("utility_refinement", False),
        utility_refinement_log_limit=config.get(
            "utility_refinement_log_limit", 0.75),
        utility_refinement_kernel=config.get(
            "utility_refinement_kernel", 1),
        utility_refinement_blocks=config.get(
            "utility_refinement_blocks", 0),
        temporal_uq_refinement=config.get(
            "temporal_uq_refinement", False),
        temporal_uq_channels=config.get("temporal_uq_channels", 24),
        temporal_uq_blocks=config.get("temporal_uq_blocks", 2),
        temporal_uq_kernel=config.get("temporal_uq_kernel", 3),
        temporal_uq_logit_limit=config.get(
            "temporal_uq_logit_limit", 1.5),
        coverage_calibration=config.get(
            "coverage_calibration", False),
        coverage_calibration_kernel=config.get(
            "coverage_calibration_kernel", 3),
        coverage_calibration_blocks=config.get(
            "coverage_calibration_blocks", 1),
        coverage_calibration_logit_limit=config.get(
            "coverage_calibration_logit_limit", 2.0),
        coverage_calibration_mode=config.get(
            "coverage_calibration_mode", "independent"),
        coverage_calibration_temperature_limit=config.get(
            "coverage_calibration_temperature_limit", 0.75))
    uq_train.load_uq_state(model, checkpoint["uq_state_dict"])
    return model


@torch.inference_mode()
def fold_statistics(model, loader, normalizer, device: torch.device,
                    grid: torch.Tensor) -> dict[str, np.ndarray]:
    steps = model.config.out_steps
    grid_size = grid.numel()
    numerator = torch.zeros(
        steps, 2, grid_size, dtype=torch.float64, device=device)
    covered = torch.zeros_like(numerator)
    width_sum = torch.zeros_like(numerator)
    count = torch.zeros(steps, 2, dtype=torch.float64, device=device)
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    grid_view = grid.view(1, 1, 1, 1, 1, grid_size)
    model.eval()
    for history, truth in loader:
        history = history.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        history_norm, _ = normalizer.preprocess(history, truth)
        prediction_norm, lower_norm, upper_norm = (
            model.forward_distribution(history_norm))
        prediction = common.to_physical_target(
            prediction_norm, normalizer)[..., :2]
        truth = truth[..., :2]
        lower = lower_norm * scale
        upper = upper_norm * scale
        residual = truth - prediction
        scored = truth != 0.0
        factor = common.sample_sps_factor(prediction, truth).view(
            -1, 1, 1, 1, 1, 1)
        lower_grid = lower.unsqueeze(-1) * grid_view
        upper_grid = upper.unsqueeze(-1) * grid_view
        inside = (
            (residual.unsqueeze(-1) >= -lower_grid)
            & (residual.unsqueeze(-1) <= upper_grid))
        value = factor * inside * torch.exp(
            -(lower_grid + upper_grid) / common.SPS_SIGMA)
        mask = scored.unsqueeze(-1)
        numerator += (value * mask).sum(dim=(0, 2, 3), dtype=torch.float64)
        covered += (inside & mask).sum(dim=(0, 2, 3), dtype=torch.float64)
        width_sum += (
            (lower_grid + upper_grid) * mask
        ).sum(dim=(0, 2, 3), dtype=torch.float64)
        count += scored.sum(dim=(0, 2, 3), dtype=torch.float64)
    return {
        "numerator": numerator.cpu().numpy(),
        "covered": covered.cpu().numpy(),
        "width_sum": width_sum.cpu().numpy(),
        "count": count.cpu().numpy(),
    }


def selected_metrics(stats: list[dict[str, np.ndarray]],
                     selected: list[np.ndarray]) -> dict[str, float]:
    numerator = covered = width_sum = count = 0.0
    for row, indices in zip(stats, selected):
        horizon = np.arange(indices.shape[0])[:, None]
        channel = np.arange(indices.shape[1])[None, :]
        numerator += row["numerator"][horizon, channel, indices].sum()
        covered += row["covered"][horizon, channel, indices].sum()
        width_sum += row["width_sum"][horizon, channel, indices].sum()
        count += row["count"].sum()
    return {
        "sps_score": float(100.0 * numerator / count),
        "coverage": float(covered / count),
        "mean_width": float(width_sum / count),
        "scored_elements": int(count),
    }


def main() -> None:
    args = parse_args()
    if (args.folds < 2 or args.batch_size <= 0 or args.grid_min <= 0
            or args.grid_max < args.grid_min or args.grid_step <= 0):
        raise ValueError("invalid fold, batch, or multiplier grid")
    device = torch.device(args.device)
    grid_np = np.arange(
        args.grid_min, args.grid_max + 0.5 * args.grid_step,
        args.grid_step, dtype=np.float32)
    grid = torch.from_numpy(grid_np).to(device)
    payload = torch.load(
        args.point_checkpoint, map_location="cpu", weights_only=False)
    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    _train, windows, manifest = common.repro.make_split(
        metadata, args.window_stride, args.val_fraction)
    dataset = common.repro.WindowDataset(trajectories, windows)
    sample_folds, fold_manifest = trajectory_folds(manifest, args.folds)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    loader_args = argparse.Namespace(num_workers=args.num_workers)
    stats = []
    for fold in range(args.folds):
        indices = [
            index for index, value in enumerate(sample_folds)
            if value == fold]
        loader = uq_train.make_loader(
            dataset, indices, args.batch_size, False,
            loader_args, seed=0)
        checkpoint = torch.load(
            args.candidate_dir / f"fold_{fold:02d}_best.pth",
            map_location="cpu", weights_only=False)
        model = load_fold_model(payload, checkpoint, normalizer, device)
        stats.append(fold_statistics(
            model, loader, normalizer, device, grid))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"event": "fold", "fold": fold,
                          "windows": len(indices)}), flush=True)

    scale_one = int(np.argmin(np.abs(grid_np - 1.0)))
    native_indices = [
        np.full(stats[0]["count"].shape, scale_one, dtype=np.int64)
        for _ in stats]
    cv_indices = []
    for held_out in range(args.folds):
        train_numerator = sum(
            row["numerator"] for index, row in enumerate(stats)
            if index != held_out)
        cv_indices.append(train_numerator.argmax(axis=-1))
    all_numerator = sum(row["numerator"] for row in stats)
    oracle_index = all_numerator.argmax(axis=-1)
    oracle_indices = [oracle_index for _ in stats]
    cv_scales = [grid_np[index] for index in cv_indices]
    report = {
        "point_checkpoint": str(args.point_checkpoint),
        "candidate_dir": str(args.candidate_dir),
        "fold_manifest": fold_manifest,
        "grid": grid_np.tolist(),
        "native": selected_metrics(stats, native_indices),
        "cv_horizon_channel": selected_metrics(stats, cv_indices),
        "oracle_all_horizon_channel": selected_metrics(
            stats, oracle_indices),
        "cv_scale_mean": float(np.mean(cv_scales)),
        "cv_scale_std": float(np.std(cv_scales)),
        "cv_scale_min": float(np.min(cv_scales)),
        "cv_scale_max": float(np.max(cv_scales)),
        "all_fit_scales": grid_np[oracle_index].tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"event": "complete", **{
        key: report[key] for key in (
            "native", "cv_horizon_channel",
            "oracle_all_horizon_channel")}}), flush=True)


if __name__ == "__main__":
    main()
