#!/usr/bin/env python3
"""Fit a condition-CV SPS budget reallocator on the A/B/C mean expert.

The three frozen forecasters and their native distributions are loaded from a
reusable NPZ cache.  Only a small interval reallocator is optimized; the point
forecast remains exactly the deployed mean-expert prediction.  Model selection
uses held-out Reynolds/AoA folds, with both the selected stress scenario and
the unmodified local residuals reported for every fold.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from audit_mean_expert_disagreement import (  # noqa: E402
    base_arrays,
    sample_factor,
)
from train_uq_budget_reallocator import (  # noqa: E402
    aggregate_fold_metrics,
    exact_metrics,
    train_head,
    trajectory_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-cache", type=Path, required=True)
    parser.add_argument("--quantile-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-b", type=float, default=0.25)
    parser.add_argument("--mean-expert-weight", type=float, default=0.5)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-updates", type=int, default=1000)
    parser.add_argument("--final-updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--coverage-target", type=float, default=0.86)
    parser.add_argument("--interval-score-weight", type=float, default=0.02)
    parser.add_argument("--cvar-weight", type=float, default=0.25)
    parser.add_argument("--cvar-fraction", type=float, default=0.25)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--global-hidden", type=int, default=160)
    parser.add_argument("--residual-stress", type=float, default=1.0)
    parser.add_argument(
        "--use-dynamic-local", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--use-static-template", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--use-global-budget", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--use-regime-film", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--fit-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    positive = (
        args.folds, args.fold_updates, args.final_updates, args.batch_size,
        args.eval_batch_size, args.eval_interval, args.lr,
        args.hidden_channels, args.global_hidden, args.residual_stress,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("positive training arguments must be finite")
    if args.folds < 2:
        raise ValueError("at least two condition folds are required")
    if not 0.0 <= args.weight_b <= 1.0:
        raise ValueError("weight-b must be in [0,1]")
    if not 0.0 <= args.mean_expert_weight <= 1.0:
        raise ValueError("mean-expert-weight must be in [0,1]")
    if not 0.0 < args.coverage_target < 1.0:
        raise ValueError("coverage-target must be in (0,1)")
    if not 0.0 < args.cvar_fraction <= 1.0:
        raise ValueError("cvar-fraction must be in (0,1]")
    for value in (
            args.weight_decay, args.grad_clip, args.interval_score_weight,
            args.cvar_weight):
        if not math.isfinite(value) or value < 0:
            raise ValueError("loss and optimizer weights must be non-negative")
    if not (args.use_dynamic_local or args.use_static_template
            or args.use_global_budget):
        raise ValueError("at least one reallocation branch must be enabled")
    if not args.ensemble_cache.is_file():
        raise FileNotFoundError(args.ensemble_cache)
    for name in ("quantiles_a_fp16.npy", "quantiles_b_fp16.npy"):
        path = args.quantile_cache_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def histories_from_loader(loader, normalizer) -> tuple[torch.Tensor, torch.Tensor]:
    histories = []
    masks = []
    for history, target in loader:
        history = history.to(normalizer.mean_input.device, non_blocking=True)
        target = target.to(normalizer.mean_input.device, non_blocking=True)
        history_norm, _ = normalizer.preprocess(history, target)
        histories.append(history_norm[..., :2].cpu().to(torch.float16))
        masks.append((history[..., :2].abs().amax(dim=(1, 4)) > 1e-8).cpu())
    return torch.cat(histories), torch.cat(masks)


def _flatten_forecast(value: torch.Tensor) -> torch.Tensor:
    batch, steps, height, width, channels = value.shape
    return value.permute(0, 1, 4, 2, 3).reshape(
        batch, steps * channels, height, width)


def _flatten_quantiles(value: torch.Tensor) -> torch.Tensor:
    batch, steps, height, width, channels, levels = value.shape
    return value.permute(0, 1, 4, 5, 2, 3).reshape(
        batch, steps * channels * levels, height, width)


def build_auxiliary_features(args, arrays, base, std_target) -> torch.Tensor:
    scale = torch.from_numpy(std_target.astype(np.float16, copy=False))

    def normalized(value):
        tensor = torch.from_numpy(value.astype(np.float16, copy=False))
        return tensor / scale

    prediction = base["prediction"]
    delta_a = normalized(arrays["prediction_a"] - prediction)
    delta_b = normalized(arrays["prediction_b"] - prediction)
    delta_c = normalized(arrays["prediction_c"] - prediction)
    pieces = [
        _flatten_forecast(normalized(arrays[key]))
        for key in ("lower_a", "upper_a", "lower_b", "upper_b")
    ]
    pieces.extend(_flatten_forecast(value) for value in (
        delta_a, delta_b, delta_c,
        normalized(np.abs(arrays["prediction_a"] - arrays["prediction_b"])),
        normalized(np.abs(arrays["prediction_c"] - prediction)),
    ))
    quantile_paths = (
        args.quantile_cache_dir / "quantiles_a_fp16.npy",
        args.quantile_cache_dir / "quantiles_b_fp16.npy",
    )
    for path, delta in zip(quantile_paths, (delta_a, delta_b)):
        quantiles = torch.from_numpy(np.asarray(
            np.load(path, mmap_mode="r", allow_pickle=False)))
        if quantiles.shape[:5] != prediction.shape or quantiles.shape[-1] != 7:
            raise RuntimeError(f"unexpected quantile cache shape {quantiles.shape}")
        pieces.append(_flatten_quantiles(
            quantiles + delta.unsqueeze(-1)))
    auxiliary = torch.cat(pieces, dim=1).contiguous()
    if not torch.isfinite(auxiliary).all():
        raise FloatingPointError("non-finite expert auxiliary features")
    return auxiliary


def build_training_cache(args, val_loader, normalizer) -> dict[str, torch.Tensor]:
    history, valid_mask = histories_from_loader(val_loader, normalizer)
    with np.load(args.ensemble_cache, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    base = base_arrays(arrays, args.weight_b, args.mean_expert_weight)
    if history.shape[0] != base["prediction"].shape[0]:
        raise RuntimeError("history and ensemble-cache sample order differ")
    mean_target = normalizer.mean_target[:2].detach().cpu().numpy().reshape(
        (1, 1, 1, 1, 2))
    std_target = normalizer.std_target[:2].detach().cpu().numpy().reshape(
        (1, 1, 1, 1, 2))
    prediction_norm = (
        (base["prediction"] - mean_target) / std_target).astype(np.float16)
    factor = sample_factor(base["prediction"], base["target"])
    auxiliary = build_auxiliary_features(
        args, arrays, base, std_target)
    args.auxiliary_channels = int(auxiliary.shape[1])
    result = {
        "history": history,
        "prediction_norm": torch.from_numpy(prediction_norm),
        "prediction": torch.from_numpy(base["prediction"].astype(np.float16)),
        "target": torch.from_numpy(base["target"].astype(np.float16)),
        "native_lower": torch.from_numpy(base["lower"].astype(np.float16)),
        "native_upper": torch.from_numpy(base["upper"].astype(np.float16)),
        "valid_mask": valid_mask,
        "factor": torch.from_numpy(factor),
        "auxiliary": auxiliary,
    }
    del base, arrays, prediction_norm, factor, auxiliary
    gc.collect()
    return result


def checkpoint_payload(head, args, fold, baseline_local, selected_local,
                       baseline_stress, selected_stress, history):
    return {
        "model_config": head.architecture_config(),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in head.state_dict().items()},
        "source": {
            "ensemble_cache": args.ensemble_cache.name,
            "weight_b": args.weight_b,
            "mean_expert_weight": args.mean_expert_weight,
        },
        "fold": fold,
        "baseline_local": baseline_local,
        "selected_local": selected_local,
        "baseline_stress": baseline_stress,
        "selected_stress": selected_stress,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    validate(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_args = argparse.Namespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=args.num_workers,
        real_stats=args.real_stats, output_dir=args.output_dir, seed=args.seed)
    train_loader, val_loader, normalizer, train_windows, val_windows, trajectories = (
        common.prepare_real(data_args, device))
    cache = build_training_cache(args, val_loader, normalizer)
    del train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    manifest = json.loads(
        (args.output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    sample_folds, fold_manifest = trajectory_folds(manifest, args.folds)
    if len(sample_folds) != cache["factor"].shape[0]:
        raise RuntimeError("fold labels do not align with cached validation windows")
    fold_tensor = torch.tensor(sample_folds, dtype=torch.long)
    all_indices = torch.arange(fold_tensor.numel(), dtype=torch.long)
    fold_rows = []
    for fold in range(args.folds):
        train_indices = all_indices[fold_tensor != fold]
        eval_indices = all_indices[fold_tensor == fold]
        baseline_local = exact_metrics(
            None, cache, eval_indices, device, args.eval_batch_size, 1.0)
        baseline_stress = exact_metrics(
            None, cache, eval_indices, device, args.eval_batch_size,
            args.residual_stress)
        head, history, selected_stress = train_head(
            args, cache, train_indices, eval_indices,
            args.fold_updates, device, args.seed + fold + 1)
        if selected_stress is None:
            raise RuntimeError("fold training produced no selected checkpoint")
        selected_local = exact_metrics(
            head, cache, eval_indices, device, args.eval_batch_size, 1.0)
        checkpoint = args.output_dir / f"fold_{fold:02d}_best.pth"
        torch.save(checkpoint_payload(
            head, args, fold, baseline_local, selected_local,
            baseline_stress, selected_stress, history), checkpoint)
        row = {
            "fold": fold,
            "train_samples": int(train_indices.numel()),
            "eval_samples": int(eval_indices.numel()),
            "baseline_local": baseline_local,
            "selected_local": selected_local,
            "local_sps_delta": (
                selected_local["sps_score"] - baseline_local["sps_score"]),
            "baseline_stress": baseline_stress,
            "selected_stress": selected_stress,
            "stress_sps_delta": (
                selected_stress["sps_score"] - baseline_stress["sps_score"]),
        }
        fold_rows.append(row)
        print(json.dumps({"event": "fold_complete", **row}), flush=True)
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_local_oof = aggregate_fold_metrics(fold_rows, "baseline_local")
    selected_local_oof = aggregate_fold_metrics(fold_rows, "selected_local")
    baseline_stress_oof = aggregate_fold_metrics(fold_rows, "baseline_stress")
    selected_stress_oof = aggregate_fold_metrics(fold_rows, "selected_stress")
    final_checkpoint = None
    final_local = None
    final_stress = None
    if args.fit_final:
        final_head, final_history, _ = train_head(
            args, cache, all_indices, None, args.final_updates,
            device, args.seed + 1000)
        final_local = exact_metrics(
            final_head, cache, all_indices, device, args.eval_batch_size, 1.0)
        final_stress = exact_metrics(
            final_head, cache, all_indices, device, args.eval_batch_size,
            args.residual_stress)
        final_checkpoint = args.output_dir / "final.pth"
        torch.save(checkpoint_payload(
            final_head, args, -1, baseline_local_oof, final_local,
            baseline_stress_oof, final_stress, final_history), final_checkpoint)

    report = {
        "method": "mean_expert_condition_cv_spatial_budget_reallocation",
        "ensemble_cache": str(args.ensemble_cache),
        "weight_b": args.weight_b,
        "mean_expert_weight": args.mean_expert_weight,
        "train_windows_point_models": train_windows,
        "tail_windows_for_uq": val_windows,
        "trajectories": trajectories,
        "fold_manifest": fold_manifest,
        "folds": fold_rows,
        "oof_baseline_local": baseline_local_oof,
        "oof_selected_local": selected_local_oof,
        "oof_local_sps_delta": (
            selected_local_oof["sps_score"] - baseline_local_oof["sps_score"]),
        "oof_baseline_stress": baseline_stress_oof,
        "oof_selected_stress": selected_stress_oof,
        "oof_stress_sps_delta": (
            selected_stress_oof["sps_score"] - baseline_stress_oof["sps_score"]),
        "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
        "final_diagnostic_local": final_local,
        "final_diagnostic_stress": final_stress,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "warning": (
            "Fold deltas are condition-cross-validated. Final diagnostics use "
            "the same tail residuals used to fit final.pth."),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "oof_local_sps_delta": report["oof_local_sps_delta"],
        "oof_stress_sps_delta": report["oof_stress_sps_delta"],
        "summary": str(args.output_dir / "summary.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
