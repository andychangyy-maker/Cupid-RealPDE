#!/usr/bin/env python3
"""Distill the heavy UQ decoder into a fast direct multi-scale interval head."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for path in (CODE_ROOT, Path(__file__).parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from models.distilled_interval_head import (  # noqa: E402
    DistilledIntervalHead,
    DistilledIntervalHeadConfig,
)
from train_distilled_flow_corrector import (  # noqa: E402
    load_forecaster,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-a", type=Path, required=True)
    parser.add_argument("--teacher-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-b", type=float, default=0.25)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=150)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--real-all-windows", action=argparse.BooleanOptionalAction,
        default=False)

    parser.add_argument(
        "--feature-channels", type=int, nargs="+",
        default=(512, 256, 128))
    parser.add_argument("--projection-channels", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--max-log-width-ratio", type=float, default=2.0)
    parser.add_argument("--max-center-fraction", type=float, default=1.5)

    parser.add_argument("--interval-scale", type=float, default=0.025)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--soft-sps-weight", type=float, default=0.25)
    parser.add_argument("--coverage-weight", type=float, default=0.50)
    parser.add_argument("--coverage-target", type=float, default=0.90)
    parser.add_argument("--soft-tau-fraction", type=float, default=0.10)
    return parser.parse_args()


def head_config(args: argparse.Namespace) -> DistilledIntervalHeadConfig:
    return DistilledIntervalHeadConfig(
        feature_channels=tuple(args.feature_channels),
        projection_channels=args.projection_channels,
        hidden_channels=args.hidden_channels,
        blocks=args.blocks,
        groups=args.groups,
        kernel_size=args.kernel_size,
        max_log_width_ratio=args.max_log_width_ratio,
        max_center_fraction=args.max_center_fraction,
    )


def validate(args: argparse.Namespace) -> None:
    positive = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.eval_interval, args.checkpoint_interval, args.lr,
        args.window_stride, args.interval_scale,
        args.soft_tau_fraction)
    if any(value <= 0 for value in positive):
        raise ValueError("training values and scales must be positive")
    nonnegative = (
        args.weight_decay, args.grad_clip, args.distill_weight,
        args.soft_sps_weight, args.coverage_weight)
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        raise ValueError("loss and optimizer values must be finite/non-negative")
    if not math.isfinite(args.weight_b) or not 0 <= args.weight_b <= 1:
        raise ValueError("weight-b must lie in [0,1]")
    if not 0 < args.coverage_target < 1:
        raise ValueError("coverage target must lie in (0,1)")
    head_config(args).validate()


@torch.inference_mode()
def teacher_extents(
        x_normalized: torch.Tensor,
        teacher_a, teacher_b, weight_b: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    x_half = x_normalized.to(dtype=torch.float16)
    _prediction_a, lower_a, upper_a = (
        teacher_a.forward_distribution(x_half))
    _prediction_b, lower_b, upper_b = (
        teacher_b.forward_distribution(x_half))
    return (
        lower_a.lerp(lower_b, weight_b).float(),
        upper_a.lerp(upper_b, weight_b).float(),
    )


class FastDistribution:
    """Thin nn.Module-compatible facade for common evaluation functions."""

    def __init__(self, base, head):
        self.base = base
        self.head = head
        self.config = base.config

    def eval(self):
        self.base.eval()
        self.head.eval()
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.base(
                x.to(dtype=torch.float16)).float()

    def forward_distribution(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            prediction, features = self.base.forward_with_features(
                x.to(dtype=torch.float16))
        prediction = prediction.float()
        lower, upper = self.head(
            x.float(), prediction,
            [value.float() for value in features],
            self.base.target_std.float())
        return prediction, lower, upper


def evaluate(base, head, loader, normalizer, device) -> dict[str, float]:
    model = FastDistribution(base, head).eval()
    metrics = common.repro.evaluate(
        model, loader, normalizer, device)
    common.add_weighted(metrics, (5.0, 3.0, 2.0))
    metrics.update(common.evaluate_native_sps(
        model, loader, normalizer, device))
    metrics["non_time_mean"] = float(np.mean((
        metrics["rel_l2_score"],
        metrics["tke_rel_l2_score"],
        metrics["mvpe_rel_l2_score"],
        metrics["native_sps_score"],
    )))
    return metrics


def interval_loss(
        args: argparse.Namespace,
        prediction: torch.Tensor,
        lower: torch.Tensor, upper: torch.Tensor,
        teacher_lower: torch.Tensor, teacher_upper: torch.Tensor,
        truth_normalized: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    distill = 0.5 * (
        F.smooth_l1_loss(
            lower / args.interval_scale,
            teacher_lower / args.interval_scale)
        + F.smooth_l1_loss(
            upper / args.interval_scale,
            teacher_upper / args.interval_scale))

    pred_physical = common.to_physical_target(
        prediction, normalizer)[..., :2]
    truth_physical = common.to_physical_target(
        truth_normalized, normalizer)[..., :2]
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    lower_physical = lower * scale
    upper_physical = upper * scale
    residual = truth_physical - pred_physical
    scored = truth_physical != 0.0
    reference = prediction.new_tensor(
        [0.01075, 0.00550]).view(1, 1, 1, 1, 2)
    tau = args.soft_tau_fraction * reference
    soft_inside = (
        torch.sigmoid((residual + lower_physical) / tau)
        * torch.sigmoid((upper_physical - residual) / tau))
    factor = common.sample_sps_factor(
        pred_physical, truth_physical).view(
            -1, 1, 1, 1, 1)
    soft_sps = (
        factor * soft_inside
        * torch.exp(
            -(lower_physical + upper_physical)
            / common.SPS_SIGMA))[scored].mean()
    soft_coverage = soft_inside[scored].mean()
    coverage_penalty = F.relu(
        args.coverage_target - soft_coverage).square()
    loss = (
        args.distill_weight * distill
        - args.soft_sps_weight * soft_sps
        + args.coverage_weight * coverage_penalty)
    return loss, {
        "distill": distill.detach(),
        "soft_sps": soft_sps.detach(),
        "soft_coverage": soft_coverage.detach(),
        "coverage_penalty": coverage_penalty.detach(),
        "mean_width": (
            lower_physical + upper_physical)[scored].mean().detach(),
    }


def save_head(
        path: Path, head: DistilledIntervalHead,
        args: argparse.Namespace, iteration: int, metrics: dict,
        metadata: dict) -> None:
    torch.save({
        "phase": "real_interval_distillation",
        "iteration": iteration,
        "model_config": head.architecture_config(),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in head.state_dict().items()},
        "point_checkpoint": str(args.point_checkpoint),
        "teacher": {
            "a": metadata["a"],
            "b": metadata["b"],
            "weight_b": args.weight_b,
        },
        "metrics": metrics,
    }, path)


def main() -> None:
    args = parse_args()
    validate(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("direct interval distillation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=args.max_files,
        window_stride=args.window_stride,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        real_stats=args.real_stats,
        output_dir=args.output_dir,
        seed=args.seed,
        real_all_windows=args.real_all_windows,
    )
    (train_loader, val_loader, normalizer, train_windows,
     val_windows, trajectories) = common.prepare_real(data_args, device)
    base, metadata_point = load_forecaster(
        args.point_checkpoint, normalizer, device)
    teacher_a, metadata_a = load_forecaster(
        args.teacher_a, normalizer, device)
    teacher_b, metadata_b = load_forecaster(
        args.teacher_b, normalizer, device)
    metadata = {
        "point": metadata_point, "a": metadata_a, "b": metadata_b}
    head = DistilledIntervalHead(head_config(args)).to(device)
    parameters = sum(value.numel() for value in head.parameters())
    print(json.dumps({
        "event": "model",
        "head_config": head.architecture_config(),
        "head_parameters": parameters,
        "models": metadata,
    }), flush=True)
    initial = evaluate(base, head, val_loader, normalizer, device)
    print(json.dumps({"event": "initial", **initial}), flush=True)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    iterator = common.repro.cycling(train_loader)
    history = []
    sums: dict[str, float] = {}
    count = 0
    best_value = -float("inf")
    best_iteration = 0
    started = time.time()

    for iteration in range(1, args.updates + 1):
        head.train()
        history_batch, truth = next(iterator)
        history_batch = history_batch.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        history_norm, truth_norm = normalizer.preprocess(
            history_batch, truth)
        with torch.no_grad():
            prediction, features = base.forward_with_features(
                history_norm.to(dtype=torch.float16))
        prediction = prediction.float()
        features = [value.float() for value in features]
        target_lower, target_upper = teacher_extents(
            history_norm, teacher_a, teacher_b, args.weight_b)
        optimizer.zero_grad(set_to_none=True)
        lower, upper = head(
            history_norm, prediction, features,
            base.target_std.float())
        loss, components = interval_loss(
            args, prediction, lower, upper,
            target_lower, target_upper,
            truth_norm, normalizer)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite interval loss at {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        for key, value in {
                "loss": loss.detach(), **components}.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1

        if iteration % args.eval_interval == 0 or iteration == args.updates:
            row = {
                "iteration": iteration,
                **{
                    f"train_{key}_mean": value / count
                    for key, value in sums.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
            }
            metrics = evaluate(
                base, head, val_loader, normalizer, device)
            row.update(metrics)
            history.append(row)
            if metrics["non_time_mean"] > best_value:
                best_value = metrics["non_time_mean"]
                best_iteration = iteration
                save_head(
                    args.output_dir / "best.pth", head, args,
                    iteration, metrics, metadata)
            print(json.dumps(
                {"event": "eval", **row}), flush=True)
            sums.clear()
            count = 0
        if iteration % args.checkpoint_interval == 0:
            save_head(
                args.output_dir / f"model_{iteration:05d}.pth",
                head, args, iteration,
                history[-1] if history else {}, metadata)

    final_metrics = evaluate(
        base, head, val_loader, normalizer, device)
    save_head(
        args.output_dir / "final.pth", head, args,
        args.updates, final_metrics, metadata)
    summary = {
        "method": (
            "direct multi-scale interval head; deployment skips "
            "uq_projections, uq_fusion, quantile_head and utility scan"),
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
        "head_config": head.architecture_config(),
        "head_parameters": parameters,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "initial_metrics": initial,
        "final_metrics": final_metrics,
        "best_iteration": best_iteration,
        "best_non_time_mean": best_value,
        "history": history,
        "wall_seconds": time.time() - started,
        "models": metadata,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    with (args.output_dir / "learning_curve.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(
            key for row in history for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({
        "event": "complete",
        "best_iteration": best_iteration,
        "best_non_time_mean": best_value,
        "wall_seconds": summary["wall_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
