#!/usr/bin/env python3
"""Measure whether ensemble disagreement actually localizes forecast errors."""

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
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d  # noqa: E402
from train_flow_backbone import construct_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--element-stride", type=int, default=197,
        help="Keep one scored element in every N for rank diagnostics.")
    return parser.parse_args()


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    value = np.corrcoef(left, right)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return 0.0
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(level): float(np.quantile(values, level))
        for level in (0.1, 0.5, 0.9, 0.99)
    }


def diagnose(
        model: EnsembleUNet2d, loader, normalizer, device: torch.device,
        element_stride: int) -> dict:
    required_parts = []
    spread_parts = []
    interval_parts = []
    member_error_parts = []
    sample_required_parts = []
    sample_spread_parts = []
    sps_numerator = 0.0
    scored_count = 0
    covered_count = 0
    width_sum = 0.0
    zero_sps_numerator = 0.0
    zero_covered_count = 0
    zero_width_sum = 0.0
    zero_width_change_sum = 0.0
    model.eval()
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    member_scale = normalizer.std_target[:2].view(1, 1, 1, 1, 1, 2)
    member_mean = normalizer.mean_target[:2].view(1, 1, 1, 1, 1, 2)

    with torch.inference_mode():
        for history, target in loader:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            history_norm, _ = normalizer.preprocess(history, target)
            (prediction_norm, members_norm, _quantile_norm,
             lower_norm, upper_norm, features, _coverage_logits) = (
                model.forward_uq_training(history_norm))
            zero_members = prediction_norm.unsqueeze(1).expand_as(members_norm)
            zero_fused = model._fused_uq_features(
                features, zero_members, history_norm,
                prediction=prediction_norm)
            zero_quantile_norm = model._quantiles_from_fused(
                zero_fused, zero_members)
            zero_coverage_logits = model._coverage_logits(zero_fused)
            zero_lower_norm, zero_upper_norm = model._utility_interval(
                zero_quantile_norm, zero_coverage_logits)
            zero_lower_norm, zero_upper_norm = model._refine_utility_interval(
                zero_lower_norm, zero_upper_norm, zero_fused)
            zero_lower_norm, zero_upper_norm = model._ordered_interval_extents(
                zero_lower_norm, zero_upper_norm)
            prediction = common.to_physical_target(
                prediction_norm, normalizer)[..., :2]
            members = members_norm[..., :2] * member_scale + member_mean
            truth = target[..., :2]
            residual = truth - prediction
            required = residual.abs()
            spread = members.std(dim=1, unbiased=False)
            lower = lower_norm * scale
            upper = upper_norm * scale
            zero_lower = zero_lower_norm * scale
            zero_upper = zero_upper_norm * scale
            selected_interval = torch.where(residual < 0.0, lower, upper)
            scored = truth != 0.0

            flat_indices = torch.nonzero(
                scored.reshape(-1), as_tuple=False).flatten()[::element_stride]
            required_parts.append(
                required.reshape(-1)[flat_indices].float().cpu().numpy())
            spread_parts.append(
                spread.reshape(-1)[flat_indices].float().cpu().numpy())
            interval_parts.append(
                selected_interval.reshape(-1)[flat_indices].float().cpu().numpy())
            member_errors = (truth[:, None] - members).permute(
                0, 2, 3, 4, 5, 1).reshape(-1, members.shape[1])
            member_error_parts.append(
                member_errors[flat_indices].float().cpu().numpy())

            scored_float = scored.to(required.dtype)
            denominator = scored_float.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
            sample_required_parts.append(
                ((required * scored_float).sum(dim=(1, 2, 3, 4))
                 / denominator).cpu().numpy())
            sample_spread_parts.append(
                ((spread * scored_float).sum(dim=(1, 2, 3, 4))
                 / denominator).cpu().numpy())

            inside = (residual >= -lower) & (residual <= upper)
            factor = common.sample_sps_factor(prediction, truth).view(
                -1, 1, 1, 1, 1)
            utility = factor * inside * torch.exp(
                -(lower + upper) / common.SPS_SIGMA)
            sps_numerator += float(utility[scored].sum(dtype=torch.float64))
            scored_count += int(scored.sum())
            covered_count += int((inside & scored).sum())
            width_sum += float(
                (lower + upper)[scored].sum(dtype=torch.float64))

            zero_inside = (
                (residual >= -zero_lower) & (residual <= zero_upper))
            zero_utility = factor * zero_inside * torch.exp(
                -(zero_lower + zero_upper) / common.SPS_SIGMA)
            zero_sps_numerator += float(
                zero_utility[scored].sum(dtype=torch.float64))
            zero_covered_count += int((zero_inside & scored).sum())
            zero_width_sum += float(
                (zero_lower + zero_upper)[scored].sum(dtype=torch.float64))
            zero_width_change_sum += float((
                (zero_lower + zero_upper) - (lower + upper)).abs()[
                    scored].sum(dtype=torch.float64))
    required = np.concatenate(required_parts)
    spread = np.concatenate(spread_parts)
    interval = np.concatenate(interval_parts)
    member_errors = np.concatenate(member_error_parts)
    sample_required = np.concatenate(sample_required_parts)
    sample_spread = np.concatenate(sample_spread_parts)
    log_required = np.log1p(required / 1e-4)
    log_spread = np.log1p(spread / 1e-4)
    log_interval = np.log1p(interval / 1e-4)

    spread_threshold = np.quantile(spread, 0.9)
    error_threshold = np.quantile(required, 0.9)
    high_spread = spread >= spread_threshold
    high_error = required >= error_threshold
    pairwise = np.corrcoef(member_errors, rowvar=False)
    upper_triangle = pairwise[np.triu_indices(pairwise.shape[0], k=1)]
    return {
        "scored_elements": scored_count,
        "sampled_elements": int(required.size),
        "samples": int(sample_required.size),
        "native_sps_score": 100.0 * sps_numerator / scored_count,
        "native_sps_coverage": covered_count / scored_count,
        "native_sps_mean_width": width_sum / scored_count,
        "mean_absolute_residual": float(required.mean()),
        "zero_disagreement_native_sps_score": (
            100.0 * zero_sps_numerator / scored_count),
        "zero_disagreement_native_sps_coverage": (
            zero_covered_count / scored_count),
        "zero_disagreement_native_sps_mean_width": (
            zero_width_sum / scored_count),
        "zero_disagreement_mean_absolute_width_change": (
            zero_width_change_sum / scored_count),
        "mean_member_spread": float(spread.mean()),
        "spread_quantiles": quantiles(spread),
        "element_spread_log_pearson": safe_pearson(
            log_spread, log_required),
        "element_spread_log_spearman": safe_spearman(
            log_spread, log_required),
        "element_interval_log_spearman": safe_spearman(
            log_interval, log_required),
        "sample_spread_pearson": safe_pearson(
            sample_spread, sample_required),
        "sample_spread_spearman": safe_spearman(
            sample_spread, sample_required),
        "top_spread_decile_error_amplification": float(
            required[high_spread].mean() / required.mean()),
        "top_error_decile_recall_at_top_spread_decile": float(
            high_spread[high_error].mean()),
        "member_error_pairwise_correlation_mean": float(
            upper_triangle.mean()),
    }


def main() -> None:
    args = parse_args()
    if args.eval_batch_size <= 0 or args.element_stride <= 0:
        raise ValueError("batch size and element stride must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=args.output.parent, seed=0)
    (train_loader, val_loader, normalizer, _train_windows,
     val_windows, trajectories) = common.prepare_real(data_args, device)
    del train_loader

    results = {
        "element_stride": args.element_stride,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "models": {},
    }
    for checkpoint in args.checkpoints:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False)
        architecture = payload["model_config"]["architecture"]
        if architecture != "ensemble_unet":
            raise ValueError(f"{checkpoint} is not an ensemble_unet")
        model = construct_model(
            architecture, payload["model_config"]).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        common.set_model_stats(model, normalizer)
        label = checkpoint.parent.name
        results["models"][label] = {
            "checkpoint": str(checkpoint),
            **diagnose(
                model, val_loader, normalizer, device, args.element_stride),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({
            "event": "diagnosed", "label": label,
            **results["models"][label]}, ensure_ascii=False), flush=True)

    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
