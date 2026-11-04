#!/usr/bin/env python3
"""Evaluate a fixed five-fold UQ output committee on one frozen point model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_ensemble_uq_oof as uq_train  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402


def exact_sps_update(
        prediction: torch.Tensor, truth: torch.Tensor,
        lower: torch.Tensor, upper: torch.Tensor) -> dict[str, float]:
    if not (prediction.shape == truth.shape == lower.shape == upper.shape):
        raise ValueError("SPS tensors must have the same shape")
    scored = truth != 0.0
    residual = truth - prediction
    inside = (residual >= -lower) & (residual <= upper)
    factor = common.sample_sps_factor(prediction, truth).view(
        -1, 1, 1, 1, 1)
    utility = factor * inside * torch.exp(
        -(lower + upper) / common.SPS_SIGMA)
    return {
        "numerator": float(utility[scored].sum(dtype=torch.float64)),
        "covered": int((inside & scored).sum()),
        "width_sum": float(
            (lower + upper)[scored].sum(dtype=torch.float64)),
        "count": int(scored.sum()),
    }


def merge_sps(rows: list[dict[str, float]]) -> dict[str, float]:
    count = sum(int(row["count"]) for row in rows)
    if count <= 0:
        raise ValueError("SPS accumulator is empty")
    return {
        "sps_score": 100.0 * sum(row["numerator"] for row in rows) / count,
        "coverage": sum(int(row["covered"]) for row in rows) / count,
        "mean_width": sum(row["width_sum"] for row in rows) / count,
        "scored_elements": count,
    }


def load_fold_model(base_payload: dict, fold_path: Path, normalizer,
                    device: torch.device):
    checkpoint = torch.load(fold_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = uq_train.load_model(
        base_payload, normalizer, device, train_projections=False,
        add_flow_context=config.get("uq_flow_context", False),
        residual_quantile_intervals=config.get(
            "residual_quantile_intervals", False),
        utility_refinement=config.get("utility_refinement", False),
        utility_refinement_log_limit=config.get(
            "utility_refinement_log_limit", 0.75),
        utility_refinement_kernel=config.get("utility_refinement_kernel", 1),
        utility_refinement_blocks=config.get("utility_refinement_blocks", 0),
        temporal_uq_refinement=config.get("temporal_uq_refinement", False),
        temporal_uq_channels=config.get("temporal_uq_channels", 24),
        temporal_uq_blocks=config.get("temporal_uq_blocks", 2),
        temporal_uq_kernel=config.get("temporal_uq_kernel", 3),
        temporal_uq_logit_limit=config.get("temporal_uq_logit_limit", 1.5),
        coverage_calibration=config.get("coverage_calibration", False),
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
    return model.eval()


def load_refit_model(payload: dict, normalizer, device: torch.device):
    model = uq_train.load_model(
        payload, normalizer, device, train_projections=False,
        add_flow_context=payload["model_config"].get("uq_flow_context", False),
        residual_quantile_intervals=payload["model_config"].get(
            "residual_quantile_intervals", False),
        utility_refinement=payload["model_config"].get(
            "utility_refinement", False),
        utility_refinement_log_limit=payload["model_config"].get(
            "utility_refinement_log_limit", 0.75),
        utility_refinement_kernel=payload["model_config"].get(
            "utility_refinement_kernel", 1),
        utility_refinement_blocks=payload["model_config"].get(
            "utility_refinement_blocks", 0),
        temporal_uq_refinement=payload["model_config"].get(
            "temporal_uq_refinement", False),
        temporal_uq_channels=payload["model_config"].get(
            "temporal_uq_channels", 24),
        temporal_uq_blocks=payload["model_config"].get(
            "temporal_uq_blocks", 2),
        temporal_uq_kernel=payload["model_config"].get(
            "temporal_uq_kernel", 3),
        temporal_uq_logit_limit=payload["model_config"].get(
            "temporal_uq_logit_limit", 1.5),
        coverage_calibration=payload["model_config"].get(
            "coverage_calibration", False),
        coverage_calibration_kernel=payload["model_config"].get(
            "coverage_calibration_kernel", 3),
        coverage_calibration_blocks=payload["model_config"].get(
            "coverage_calibration_blocks", 1),
        coverage_calibration_logit_limit=payload["model_config"].get(
            "coverage_calibration_logit_limit", 2.0),
        coverage_calibration_mode=payload["model_config"].get(
            "coverage_calibration_mode", "independent"),
        coverage_calibration_temperature_limit=payload["model_config"].get(
            "coverage_calibration_temperature_limit", 0.75))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--refit-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.folds < 2 or args.batch_size <= 0:
        raise ValueError("committee needs at least two folds and a batch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal fold committee audit requires CUDA")
    data_args = argparse.Namespace(
        real_dir=args.real_dir, max_files=0,
        window_stride=args.window_stride, val_fraction=args.val_fraction,
        batch_size=args.batch_size, eval_batch_size=args.batch_size,
        num_workers=0, real_stats=args.real_stats,
        output_dir=args.output.parent, seed=0)
    train_loader, loader, normalizer, _train_windows, val_windows, trajectories = (
        common.prepare_real(data_args, device))
    del train_loader
    point_payload = torch.load(
        args.point_checkpoint, map_location="cpu", weights_only=False)
    fold_paths = [
        args.fold_dir / f"fold_{fold:02d}_best.pth"
        for fold in range(args.folds)]
    if not all(path.is_file() for path in fold_paths):
        raise FileNotFoundError("fold committee checkpoint set is incomplete")
    models = [
        load_fold_model(point_payload, path, normalizer, device)
        for path in fold_paths]
    refit_payload = torch.load(
        args.refit_checkpoint, map_location="cpu", weights_only=False)
    refit = load_refit_model(refit_payload, normalizer, device)
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    accumulators: dict[str, list[dict[str, float]]] = {
        "refit": [], "arithmetic_committee": [], "median_committee": []}
    prediction_max_gap = 0.0
    with torch.inference_mode():
        for history, truth in loader:
            history = history.to(device, non_blocking=True)
            truth = truth.to(device, non_blocking=True)
            history_norm, _ = normalizer.preprocess(history, truth)
            truth = truth[..., :2]
            fold_values = [model.forward_distribution(history_norm)
                           for model in models]
            refit_prediction_norm, refit_lower_norm, refit_upper_norm = (
                refit.forward_distribution(history_norm))
            reference = fold_values[0][0]
            prediction_max_gap = max(prediction_max_gap, max(
                float((values[0] - reference).abs().max())
                for values in fold_values[1:]))
            prediction = common.to_physical_target(
                reference, normalizer)[..., :2]
            lower_stack = torch.stack([values[1] for values in fold_values])
            upper_stack = torch.stack([values[2] for values in fold_values])
            arithmetic_lower = lower_stack.mean(dim=0) * scale
            arithmetic_upper = upper_stack.mean(dim=0) * scale
            median_lower = lower_stack.median(dim=0).values * scale
            median_upper = upper_stack.median(dim=0).values * scale
            refit_prediction = common.to_physical_target(
                refit_prediction_norm, normalizer)[..., :2]
            accumulators["arithmetic_committee"].append(exact_sps_update(
                prediction, truth, arithmetic_lower, arithmetic_upper))
            accumulators["median_committee"].append(exact_sps_update(
                prediction, truth, median_lower, median_upper))
            accumulators["refit"].append(exact_sps_update(
                refit_prediction, truth,
                refit_lower_norm * scale, refit_upper_norm * scale))
    if prediction_max_gap > 2.0e-6:
        raise RuntimeError("fold UQ checkpoints changed frozen point output")
    metrics = {key: merge_sps(value) for key, value in accumulators.items()}
    result = {
        "schema": "ensemble_uq_fold_committee_audit_v1",
        "point_checkpoint": str(args.point_checkpoint.resolve()),
        "refit_checkpoint": str(args.refit_checkpoint.resolve()),
        "fold_checkpoints": [str(path.resolve()) for path in fold_paths],
        "folds": args.folds,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "prediction_max_abs_gap": prediction_max_gap,
        "metrics": metrics,
        "arithmetic_minus_refit_sps": (
            metrics["arithmetic_committee"]["sps_score"]
            - metrics["refit"]["sps_score"]),
        "median_minus_refit_sps": (
            metrics["median_committee"]["sps_score"]
            - metrics["refit"]["sps_score"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"event": "fold_committee_complete", **result}), flush=True)


if __name__ == "__main__":
    main()
