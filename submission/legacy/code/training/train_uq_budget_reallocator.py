#!/usr/bin/env python3
"""Condition-cross-validated spatial UQ reallocation on honest tail residuals.

The point forecaster is frozen. Its temporal-tail inputs, point predictions,
native intervals, and targets are cached in memory. A lightweight reallocator
is selected by trajectory-level Reynolds/AoA folds, then refit on all tail
residuals. The final head uses no labels, metadata, or cross-sample state at
inference time.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from models.uq_reallocator import (  # noqa: E402
    UQBudgetReallocator,
    UQReallocatorConfig,
)
from train_flow_backbone import construct_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-updates", type=int, default=1500)
    parser.add_argument("--final-updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--coverage-target", type=float, default=0.86)
    parser.add_argument("--interval-score-weight", type=float, default=0.02)
    parser.add_argument("--cvar-weight", type=float, default=0.25)
    parser.add_argument("--cvar-fraction", type=float, default=0.25)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--global-hidden", type=int, default=160)
    parser.add_argument(
        "--width-calibration-json", type=Path,
        help=("Optional audit summary containing calibrations; its selected "
              "multipliers are applied before learning spatial reallocation."))
    parser.add_argument(
        "--width-calibration-key", default="1.50",
        help="Key inside summary['calibrations'] when calibration JSON is used.")
    parser.add_argument(
        "--residual-stress", type=float, default=1.0,
        help=("Multiply held-out residuals only for an explicit Sim2Real stress "
              "test; the original per-sample accuracy factor is preserved."))
    parser.add_argument(
        "--use-dynamic-local", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--use-static-template", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--use-global-budget", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--fit-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-regime-film", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    positive = (
        args.folds, args.fold_updates, args.final_updates,
        args.batch_size, args.eval_batch_size, args.eval_interval,
        args.lr, args.hidden_channels, args.global_hidden,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("counts, learning rate, and dimensions must be positive")
    if args.folds < 2:
        raise ValueError("at least two condition folds are required")
    finite_nonnegative = (
        args.weight_decay, args.grad_clip, args.interval_score_weight,
        args.cvar_weight,
    )
    if any(not math.isfinite(value) or value < 0 for value in finite_nonnegative):
        raise ValueError("optimizer and loss weights must be finite/non-negative")
    if not 0 < args.coverage_target < 1:
        raise ValueError("coverage target must be in (0,1)")
    if not 0 < args.cvar_fraction <= 1:
        raise ValueError("CVaR fraction must be in (0,1]")
    if not math.isfinite(args.residual_stress) or args.residual_stress <= 0:
        raise ValueError("residual_stress must be finite and positive")
    if not (args.use_dynamic_local or args.use_static_template
            or args.use_global_budget):
        raise ValueError("at least one reallocation branch must be enabled")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def point_factor(prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    return common.sample_sps_factor(prediction, target)


def collect_cache(
        model, loader, normalizer, device: torch.device,
        width_multipliers: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> dict:
    values: dict[str, list[torch.Tensor]] = {
        key: [] for key in (
            "history", "prediction_norm", "prediction", "target",
            "native_lower", "native_upper", "valid_mask", "factor")
    }
    model.eval()
    with torch.inference_mode():
        for history, target in loader:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            history_norm, _ = normalizer.preprocess(history, target)
            prediction_norm, lower_norm, upper_norm = model.forward_distribution(
                history_norm)
            prediction = common.to_physical_target(
                prediction_norm, normalizer)[..., :2]
            scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
            lower = lower_norm * scale
            upper = upper_norm * scale
            if width_multipliers is not None:
                lower = lower * width_multipliers[0]
                upper = upper * width_multipliers[1]
            valid = history[..., :2].abs().amax(dim=(1, 4)) > 1e-8
            factor = point_factor(prediction, target[..., :2])
            batch = {
                "history": history_norm[..., :2],
                "prediction_norm": prediction_norm[..., :2],
                "prediction": prediction,
                "target": target[..., :2],
                "native_lower": lower,
                "native_upper": upper,
                "valid_mask": valid,
                "factor": factor,
            }
            for key, value in batch.items():
                dtype = torch.bool if key == "valid_mask" else (
                    torch.float32 if key == "factor" else torch.float16)
                values[key].append(value.detach().to(device="cpu", dtype=dtype))
    return {key: torch.cat(items, dim=0) for key, items in values.items()}


def batch_from_cache(cache: dict, indices: torch.Tensor,
                     device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in cache.items():
        selected = value.index_select(0, indices)
        if key == "valid_mask":
            result[key] = selected.to(device=device, non_blocking=True)
        else:
            result[key] = selected.to(
                device=device, dtype=torch.float32, non_blocking=True)
    return result


def reallocation_objective(
        args: argparse.Namespace, batch: dict[str, torch.Tensor],
        lower: torch.Tensor, upper: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, float]]:
    residual = (
        batch["target"] - batch["prediction"]) * args.residual_stress
    scored = batch["target"] != 0.0
    reference = lower.new_tensor([0.01075, 0.00550]).view(1, 1, 1, 1, 2)
    tau = 0.10 * reference
    soft_inside = (
        torch.sigmoid((residual + lower) / tau)
        * torch.sigmoid((upper - residual) / tau))
    efficiency_element = soft_inside * torch.exp(
        -(lower + upper) / common.SPS_SIGMA)
    count = scored.sum(dim=(1, 2, 3, 4)).clamp_min(1)
    efficiency = (
        (efficiency_element * scored).sum(dim=(1, 2, 3, 4)) / count)
    official_proxy = batch["factor"] * efficiency
    worst_count = max(1, math.ceil(efficiency.numel() * args.cvar_fraction))
    worst_efficiency = torch.topk(
        efficiency, k=worst_count, largest=False).values.mean()

    alpha = 1.0 - args.coverage_target
    miss_lower = (-lower - residual).clamp_min(0.0)
    miss_upper = (residual - upper).clamp_min(0.0)
    interval_score = (
        lower + upper + (2.0 / alpha) * (miss_lower + miss_upper)
    ) / common.SPS_SIGMA
    interval_score = interval_score[scored].mean()
    loss = (
        -official_proxy.mean()
        - args.cvar_weight * worst_efficiency
        + args.interval_score_weight * interval_score)
    return loss, {
        "loss": float(loss.detach()),
        "soft_official": float(official_proxy.mean().detach()),
        "soft_efficiency": float(efficiency.mean().detach()),
        "soft_worst_efficiency": float(worst_efficiency.detach()),
        "interval_score": float(interval_score.detach()),
    }


@torch.inference_mode()
def exact_metrics(
        head: UQBudgetReallocator | None, cache: dict,
        indices: torch.Tensor, device: torch.device,
        batch_size: int, residual_stress: float = 1.0,
        ) -> dict[str, float]:
    numerator = 0.0
    denominator = 0
    covered = 0
    width_sum = 0.0
    if head is not None:
        head.eval()
    for start in range(0, indices.numel(), batch_size):
        selected = indices[start:start + batch_size]
        batch = batch_from_cache(cache, selected, device)
        lower = batch["native_lower"]
        upper = batch["native_upper"]
        if head is not None:
            lower, upper = head(
                batch["history"], batch["prediction_norm"],
                lower, upper, batch["valid_mask"],
                auxiliary=batch.get("auxiliary"))
        residual = (
            batch["target"] - batch["prediction"]) * residual_stress
        scored = batch["target"] != 0.0
        inside = (residual >= -lower) & (residual <= upper)
        factor = batch["factor"].view(-1, 1, 1, 1, 1)
        value = factor * inside * torch.exp(
            -(lower + upper) / common.SPS_SIGMA)
        numerator += float(value[scored].sum(dtype=torch.float64))
        denominator += int(scored.sum())
        covered += int((inside & scored).sum())
        width_sum += float((lower + upper)[scored].sum(dtype=torch.float64))
    return {
        "sps_score": 100.0 * numerator / denominator,
        "coverage": covered / denominator,
        "mean_width": width_sum / denominator,
        "scored_elements": denominator,
    }


def make_head(args: argparse.Namespace, seed: int) -> UQBudgetReallocator:
    torch.manual_seed(seed)
    return UQBudgetReallocator(UQReallocatorConfig(
        hidden_channels=args.hidden_channels,
        global_hidden=args.global_hidden,
        use_dynamic_local=args.use_dynamic_local,
        use_static_template=args.use_static_template,
        use_global_budget=args.use_global_budget,
        use_regime_film=args.use_regime_film,
        auxiliary_channels=getattr(args, "auxiliary_channels", 0),
    ))


def train_head(
        args: argparse.Namespace, cache: dict,
        train_indices: torch.Tensor, eval_indices: torch.Tensor | None,
        updates: int, device: torch.device, seed: int,
        ) -> tuple[UQBudgetReallocator, list[dict], dict | None]:
    head = make_head(args, seed).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_metrics = None
    best_sps = -float("inf")
    history = []
    sums: dict[str, float] = {}
    count = 0
    started = time.time()
    for iteration in range(1, updates + 1):
        head.train()
        positions = torch.randint(
            0, train_indices.numel(), (args.batch_size,), generator=generator)
        selected = train_indices.index_select(0, positions)
        batch = batch_from_cache(cache, selected, device)
        optimizer.zero_grad(set_to_none=True)
        lower, upper = head(
            batch["history"], batch["prediction_norm"],
            batch["native_lower"], batch["native_upper"],
            batch["valid_mask"], auxiliary=batch.get("auxiliary"))
        loss, components = reallocation_objective(args, batch, lower, upper)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite UQ loss at update {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        for key, value in components.items():
            sums[key] = sums.get(key, 0.0) + value
        count += 1
        if iteration % args.eval_interval == 0 or iteration == updates:
            row = {
                "iteration": iteration,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
                **{f"train_{key}": value / count for key, value in sums.items()},
            }
            if eval_indices is not None:
                metrics = exact_metrics(
                    head, cache, eval_indices, device, args.eval_batch_size,
                    args.residual_stress)
                row.update({f"eval_{key}": value for key, value in metrics.items()})
                if metrics["sps_score"] > best_sps:
                    best_sps = metrics["sps_score"]
                    best_metrics = metrics
                    best_state = copy.deepcopy(head.state_dict())
            history.append(row)
            print(json.dumps({"event": "uq_eval", **row}), flush=True)
            sums.clear()
            count = 0
    if best_state is not None:
        head.load_state_dict(best_state, strict=True)
    return head, history, best_metrics


def trajectory_folds(manifest: list[dict], folds: int) -> tuple[list[int], dict]:
    re_values = sorted({int(item["re"]) for item in manifest})
    aoa_values = sorted({int(item["aoa"]) for item in manifest})
    re_rank = {value: index for index, value in enumerate(re_values)}
    aoa_rank = {value: index for index, value in enumerate(aoa_values)}
    sample_folds = []
    assignments = []
    for item in manifest:
        fold = (re_rank[int(item["re"])] + 2 * aoa_rank[int(item["aoa"])]) % folds
        count = len(item["val_starts"])
        sample_folds.extend([fold] * count)
        assignments.append({
            "file": item["file"], "re": item["re"], "aoa": item["aoa"],
            "fold": fold, "windows": count,
        })
    return sample_folds, {
        "formula": "(rank(re)+2*rank(aoa)) % folds",
        "assignments": assignments,
    }


def aggregate_fold_metrics(rows: list[dict], key: str) -> dict[str, float]:
    total = sum(row[key]["scored_elements"] for row in rows)
    return {
        "sps_score": sum(
            row[key]["sps_score"] * row[key]["scored_elements"]
            for row in rows) / total,
        "coverage": sum(
            row[key]["coverage"] * row[key]["scored_elements"]
            for row in rows) / total,
        "mean_width": sum(
            row[key]["mean_width"] * row[key]["scored_elements"]
            for row in rows) / total,
        "scored_elements": total,
    }


def load_width_multipliers(args, model, device: torch.device):
    if args.width_calibration_json is None:
        return None
    payload = json.loads(
        args.width_calibration_json.read_text(encoding="utf-8"))
    calibration = torch.as_tensor(
        payload["calibrations"][args.width_calibration_key],
        device=device, dtype=torch.float32)
    expected = (model.config.out_steps, 2, 2)
    if tuple(calibration.shape) != expected:
        raise ValueError(
            f"width calibration has {tuple(calibration.shape)}, expected {expected}")
    return (
        calibration[:, :, 0].view(1, expected[0], 1, 1, 2),
        calibration[:, :, 1].view(1, expected[0], 1, 1, 2),
    )


def main() -> None:
    args = parse_args()
    validate(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(
        args.point_checkpoint, map_location="cpu", weights_only=False)
    architecture = payload["model_config"]["architecture"]
    point_model = construct_model(architecture, payload["model_config"]).to(device)
    point_model.load_state_dict(payload["model_state_dict"], strict=True)
    width_multipliers = load_width_multipliers(args, point_model, device)
    data_args = argparse.Namespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=args.num_workers,
        real_stats=args.real_stats, output_dir=args.output_dir, seed=args.seed)
    (_train_loader, val_loader, normalizer, train_windows,
     val_windows, trajectories) = common.prepare_real(data_args, device)
    common.set_model_stats(point_model, normalizer)
    cache = collect_cache(
        point_model, val_loader, normalizer, device, width_multipliers)
    del point_model, val_loader, _train_loader
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
        baseline = exact_metrics(
            None, cache, eval_indices, device, args.eval_batch_size,
            args.residual_stress)
        head, history, best = train_head(
            args, cache, train_indices, eval_indices,
            args.fold_updates, device, args.seed + fold + 1)
        if best is None:
            raise RuntimeError("fold training produced no evaluation")
        checkpoint = args.output_dir / f"fold_{fold:02d}_best.pth"
        torch.save({
            "model_config": head.architecture_config(),
            "model_state_dict": {
                key: value.detach().cpu() for key, value in head.state_dict().items()},
            "fold": fold,
            "source_point_checkpoint": args.point_checkpoint.name,
            "baseline_metrics": baseline,
            "best_metrics": best,
            "history": history,
        }, checkpoint)
        row = {
            "fold": fold,
            "train_samples": int(train_indices.numel()),
            "eval_samples": int(eval_indices.numel()),
            "baseline": baseline,
            "reallocated": best,
            "sps_delta": best["sps_score"] - baseline["sps_score"],
        }
        fold_rows.append(row)
        print(json.dumps({"event": "fold_complete", **row}), flush=True)
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_oof = aggregate_fold_metrics(fold_rows, "baseline")
    reallocated_oof = aggregate_fold_metrics(fold_rows, "reallocated")
    final_checkpoint = None
    final_diagnostic = None
    if args.fit_final:
        final_head, final_history, _ = train_head(
            args, cache, all_indices, None, args.final_updates,
            device, args.seed + 1000)
        final_diagnostic = exact_metrics(
            final_head, cache, all_indices, device, args.eval_batch_size,
            args.residual_stress)
        final_checkpoint = args.output_dir / "final.pth"
        torch.save({
            "model_config": final_head.architecture_config(),
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in final_head.state_dict().items()},
            "source_point_checkpoint": args.point_checkpoint.name,
            "training_samples": int(all_indices.numel()),
            "history": final_history,
            "diagnostic_metrics_on_fit_data": final_diagnostic,
        }, final_checkpoint)

    report = {
        "method": "condition_cv_spatial_budget_reallocation",
        "source_point_checkpoint": str(args.point_checkpoint),
        "train_windows_point_model": train_windows,
        "tail_windows_for_uq": val_windows,
        "trajectories": trajectories,
        "fold_manifest": fold_manifest,
        "folds": fold_rows,
        "oof_baseline": baseline_oof,
        "oof_reallocated": reallocated_oof,
        "oof_sps_delta": (
            reallocated_oof["sps_score"] - baseline_oof["sps_score"]),
        "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
        "final_diagnostic_on_fit_data": final_diagnostic,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "warning": (
            "Fold metrics are condition-cross-validated. Final diagnostic uses "
            "the same tail residuals used to fit final.pth and is not held out."),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "oof_sps_delta": report["oof_sps_delta"],
        "summary": str(args.output_dir / "summary.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
