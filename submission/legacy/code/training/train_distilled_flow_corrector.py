#!/usr/bin/env python3
"""Distill a three-backbone Track 1 ensemble into one base plus a small head."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for path in (CODE_ROOT, Path(__file__).parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from models.distilled_flow_corrector import (  # noqa: E402
    DistilledFlowCorrector,
    DistilledFlowCorrectorConfig,
)
from train_flow_backbone import construct_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--checkpoint-c", type=Path, required=True)
    parser.add_argument("--start-corrector", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-b", type=float, default=0.25)
    parser.add_argument("--mean-expert-weight", type=float, default=0.50)
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

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--base-feature-channels", type=int, default=0)
    parser.add_argument(
        "--correct-fluctuation", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--correct-intervals", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--max-point-delta", type=float, default=0.75)
    parser.add_argument("--max-log-width-ratio", type=float, default=0.50)
    parser.add_argument("--max-center-fraction", type=float, default=0.50)

    parser.add_argument("--mean-scale", type=float, default=0.025)
    parser.add_argument("--fluctuation-scale", type=float, default=0.020)
    parser.add_argument("--interval-scale", type=float, default=0.025)
    parser.add_argument("--mean-loss-weight", type=float, default=2.0)
    parser.add_argument("--fluctuation-loss-weight", type=float, default=1.0)
    parser.add_argument("--profile-loss-weight", type=float, default=2.0)
    parser.add_argument("--interval-loss-weight", type=float, default=0.50)
    parser.add_argument("--truth-loss-weight", type=float, default=0.0)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    positive = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.eval_interval, args.checkpoint_interval, args.lr,
        args.window_stride, args.mean_scale, args.fluctuation_scale,
        args.interval_scale,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("training counts, scales, stride, and lr must be positive")
    nonnegative = (
        args.weight_decay, args.grad_clip, args.mean_loss_weight,
        args.fluctuation_loss_weight, args.profile_loss_weight,
        args.interval_loss_weight, args.truth_loss_weight,
    )
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        raise ValueError("loss and optimizer values must be finite/non-negative")
    for name in ("weight_b", "mean_expert_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0,1]")
    corrector_config(args).validate()


def corrector_config(args: argparse.Namespace) -> DistilledFlowCorrectorConfig:
    return DistilledFlowCorrectorConfig(
        hidden_channels=args.hidden_channels,
        blocks=args.blocks,
        groups=args.groups,
        kernel_size=args.kernel_size,
        base_feature_channels=args.base_feature_channels,
        max_point_delta=args.max_point_delta,
        max_log_width_ratio=args.max_log_width_ratio,
        max_center_fraction=args.max_center_fraction,
        correct_fluctuation=args.correct_fluctuation,
        correct_intervals=args.correct_intervals,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_forecaster(
        checkpoint: Path, normalizer, device: torch.device,
        ) -> tuple[nn.Module, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    architecture = config["architecture"]
    if architecture not in ("ensemble_unet", "unet"):
        raise ValueError(f"{checkpoint} is not a supported U-Net")
    model = construct_model(architecture, config).to(
        device=device, dtype=torch.float16)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    common.set_model_stats(model, normalizer)
    model.requires_grad_(False)
    model.eval()
    return model, {
        "checkpoint": str(checkpoint),
        "iteration": payload.get("iteration"),
        "parameters": sum(value.numel() for value in model.parameters()),
        "model_config": config,
    }


def build_corrector(
        args: argparse.Namespace, device: torch.device,
        ) -> DistilledFlowCorrector:
    requested = corrector_config(args)
    model = DistilledFlowCorrector(requested)
    if args.start_corrector is not None:
        payload = torch.load(
            args.start_corrector, map_location="cpu", weights_only=False)
        stored = dict(payload["model_config"])
        architecture = stored.pop(
            "architecture", "distilled_flow_corrector")
        if architecture != "distilled_flow_corrector":
            raise ValueError("start checkpoint is not a distilled corrector")
        if DistilledFlowCorrectorConfig(**stored) != requested:
            raise ValueError("start corrector config differs from requested config")
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)


@torch.inference_mode()
def teacher_targets(
        x_normalized: torch.Tensor,
        model_a: nn.Module, model_b: nn.Module, model_c: nn.Module,
        weight_b: float, mean_expert_weight: float,
        ) -> tuple[torch.Tensor, ...]:
    x_half = x_normalized.to(dtype=torch.float16)
    prediction_a, lower_a, upper_a, features_a = (
        model_a.forward_distribution_with_features(x_half))
    prediction_b, lower_b, upper_b = model_b.forward_distribution(x_half)
    base_prediction = prediction_a.lerp(prediction_b, weight_b)
    prediction_c = model_c(x_half)
    mean_delta = (
        prediction_c.mean(dim=1, keepdim=True)
        - base_prediction.mean(dim=1, keepdim=True))
    teacher_prediction = (
        base_prediction + mean_expert_weight * mean_delta)
    teacher_lower = lower_a.lerp(lower_b, weight_b)
    teacher_upper = upper_a.lerp(upper_b, weight_b)
    return tuple(value.float() for value in (
        prediction_a, lower_a, upper_a,
        teacher_prediction, teacher_lower, teacher_upper,
        features_a[-1].float()))


class DistilledForecast(nn.Module):
    """Evaluation facade exposing the normal model distribution interface."""

    def __init__(
            self, base: nn.Module, corrector: DistilledFlowCorrector):
        super().__init__()
        self.base = base
        self.corrector = corrector
        self.config = base.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_half = x.to(dtype=torch.float16)
        with torch.no_grad():
            prediction, features = self.base.forward_with_features(x_half)
        return self.corrector.correct_point(
            x.float(), prediction.float(),
            features[-1].float()).float()

    def forward_distribution(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_half = x.to(dtype=torch.float16)
        with torch.no_grad():
            prediction, lower, upper, features = (
                self.base.forward_distribution_with_features(x_half))
        return tuple(value.float() for value in (
            self.corrector.correct_distribution(
                x.float(), prediction.float(),
                lower.float(), upper.float(),
                features[-1].float())))


def evaluate(
        base: nn.Module, corrector: DistilledFlowCorrector,
        loader, normalizer, device: torch.device) -> dict[str, float]:
    model = DistilledForecast(base, corrector).eval()
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


def profile_distill_loss(
        predicted: torch.Tensor, target: torch.Tensor,
        scale: float) -> torch.Tensor:
    center_y = predicted.shape[2] // 2
    probe_y = [
        center_y + 2 * offset for offset in range(-4, 5)]
    probe_y = [
        value for value in probe_y if 0 <= value < predicted.shape[2]]
    values = []
    for index in range(4):
        probe_x = ((index + 1) * 16 + 10) // 2
        if 0 <= probe_x < predicted.shape[3]:
            values.append(F.smooth_l1_loss(
                predicted[:, :, probe_y, probe_x] / scale,
                target[:, :, probe_y, probe_x] / scale))
    if not values:
        raise ValueError("no valid profile probes")
    return torch.stack(values).mean()


def distill_loss(
        args: argparse.Namespace,
        student_prediction: torch.Tensor,
        student_lower: torch.Tensor, student_upper: torch.Tensor,
        base_prediction: torch.Tensor,
        teacher_prediction: torch.Tensor,
        teacher_lower: torch.Tensor, teacher_upper: torch.Tensor,
        truth_normalized: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    student_delta = (
        student_prediction[..., :2] - base_prediction[..., :2])
    teacher_delta = (
        teacher_prediction[..., :2] - base_prediction[..., :2])
    student_mean = student_delta.mean(dim=1, keepdim=True)
    teacher_mean = teacher_delta.mean(dim=1, keepdim=True)
    student_fluctuation = student_delta - student_mean
    teacher_fluctuation = teacher_delta - teacher_mean

    mean_loss = F.smooth_l1_loss(
        student_mean / args.mean_scale,
        teacher_mean / args.mean_scale)
    fluctuation_loss = (
        F.smooth_l1_loss(
            student_fluctuation / args.fluctuation_scale,
            teacher_fluctuation / args.fluctuation_scale)
        if args.correct_fluctuation else student_mean.new_zeros(()))
    profile_loss = profile_distill_loss(
        student_mean, teacher_mean, args.mean_scale)
    interval_loss = (
        0.5 * (
            F.smooth_l1_loss(
                student_lower / args.interval_scale,
                teacher_lower / args.interval_scale)
            + F.smooth_l1_loss(
                student_upper / args.interval_scale,
                teacher_upper / args.interval_scale))
        if args.correct_intervals else student_mean.new_zeros(()))
    truth_loss = (
        common.accuracy_objective(
            student_prediction, truth_normalized, normalizer,
            (5.0, 3.0, 2.0))[0]
        if args.truth_loss_weight else student_mean.new_zeros(()))
    loss = (
        args.mean_loss_weight * mean_loss
        + args.fluctuation_loss_weight * fluctuation_loss
        + args.profile_loss_weight * profile_loss
        + args.interval_loss_weight * interval_loss
        + args.truth_loss_weight * truth_loss)
    return loss, {
        "mean_distill": mean_loss.detach(),
        "fluctuation_distill": fluctuation_loss.detach(),
        "profile_distill": profile_loss.detach(),
        "interval_distill": interval_loss.detach(),
        "truth_accuracy": truth_loss.detach(),
        "point_delta_rms": student_delta.square().mean().sqrt().detach(),
    }


def save_corrector(
        path: Path, corrector: DistilledFlowCorrector,
        args: argparse.Namespace, iteration: int,
        metrics: dict, teacher_metadata: list[dict]) -> None:
    torch.save({
        "phase": "real_distillation",
        "iteration": iteration,
        "model_config": corrector.architecture_config(),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in corrector.state_dict().items()},
        "teacher": {
            "models": teacher_metadata,
            "weight_b": args.weight_b,
            "mean_expert_weight": args.mean_expert_weight,
        },
        "metrics": metrics,
    }, path)


def main() -> None:
    args = parse_args()
    validate(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("distillation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    data_args = SimpleNamespace(
        real_dir=args.real_dir,
        max_files=args.max_files,
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
    model_a, metadata_a = load_forecaster(
        args.checkpoint_a, normalizer, device)
    model_b, metadata_b = load_forecaster(
        args.checkpoint_b, normalizer, device)
    model_c, metadata_c = load_forecaster(
        args.checkpoint_c, normalizer, device)
    teacher_metadata = [metadata_a, metadata_b, metadata_c]
    corrector = build_corrector(args, device)
    parameters = sum(value.numel() for value in corrector.parameters())
    print(json.dumps({
        "event": "model",
        "corrector_config": corrector.architecture_config(),
        "corrector_parameters": parameters,
        "teacher": teacher_metadata,
    }), flush=True)

    initial = evaluate(
        model_a, corrector, val_loader, normalizer, device)
    print(json.dumps({"event": "initial", **initial}), flush=True)
    optimizer = torch.optim.AdamW(
        corrector.parameters(), lr=args.lr,
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
        corrector.train()
        history_batch, truth = next(iterator)
        history_batch = history_batch.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        history_norm, truth_norm = normalizer.preprocess(
            history_batch, truth)
        values = teacher_targets(
            history_norm, model_a, model_b, model_c,
            args.weight_b, args.mean_expert_weight)
        (prediction_a, lower_a, upper_a,
         teacher_prediction, teacher_lower, teacher_upper,
         features_a) = values
        optimizer.zero_grad(set_to_none=True)
        student_prediction, student_lower, student_upper = (
            corrector.correct_distribution(
                history_norm, prediction_a, lower_a, upper_a,
                features_a))
        loss, components = distill_loss(
            args, student_prediction, student_lower, student_upper,
            prediction_a, teacher_prediction, teacher_lower, teacher_upper,
            truth_norm, normalizer)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite distillation loss at update {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                corrector.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        values_to_log = {"loss": loss.detach(), **components}
        for key, value in values_to_log.items():
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
                model_a, corrector, val_loader, normalizer, device)
            row.update(metrics)
            history.append(row)
            if metrics["non_time_mean"] > best_value:
                best_value = metrics["non_time_mean"]
                best_iteration = iteration
                save_corrector(
                    args.output_dir / "best.pth", corrector, args,
                    iteration, metrics, teacher_metadata)
            print(json.dumps(
                {"event": "eval", **row}), flush=True)
            sums.clear()
            count = 0
        if iteration % args.checkpoint_interval == 0:
            save_corrector(
                args.output_dir / f"model_{iteration:05d}.pth",
                corrector, args, iteration,
                history[-1] if history else {}, teacher_metadata)

    final_metrics = evaluate(
        model_a, corrector, val_loader, normalizer, device)
    save_corrector(
        args.output_dir / "final.pth", corrector, args,
        args.updates, final_metrics, teacher_metadata)
    summary = {
        "method": (
            "single frozen A backbone plus flow-context corrector distilled "
            "from B blend and temporal-mean expert C"),
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
        "corrector_config": corrector.architecture_config(),
        "corrector_parameters": parameters,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "initial_metrics": initial,
        "final_metrics": final_metrics,
        "best_iteration": best_iteration,
        "best_non_time_mean": best_value,
        "history": history,
        "wall_seconds": time.time() - started,
        "teacher": teacher_metadata,
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
