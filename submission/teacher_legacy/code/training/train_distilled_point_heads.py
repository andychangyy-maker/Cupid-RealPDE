#!/usr/bin/env python3
"""Distill a three-model teacher directly into one C128 model's point heads."""

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
from train_distilled_flow_corrector import (  # noqa: E402
    load_forecaster,
    profile_distill_loss,
    seed_everything,
    teacher_targets,
)
from train_flow_backbone import construct_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--checkpoint-c", type=Path, required=True)
    parser.add_argument("--start-student", type=Path)
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
    parser.add_argument("--weight-decay", type=float, default=0.0)
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
    parser.add_argument("--mean-scale", type=float, default=0.025)
    parser.add_argument("--fluctuation-scale", type=float, default=0.020)
    parser.add_argument("--mean-loss-weight", type=float, default=2.0)
    parser.add_argument("--fluctuation-loss-weight", type=float, default=1.0)
    parser.add_argument("--profile-loss-weight", type=float, default=2.0)
    parser.add_argument("--truth-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--mean-only", action=argparse.BooleanOptionalAction,
        default=False)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    positive = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.eval_interval, args.checkpoint_interval, args.lr,
        args.window_stride, args.mean_scale, args.fluctuation_scale)
    if any(value <= 0 for value in positive):
        raise ValueError("training values and distillation scales must be positive")
    nonnegative = (
        args.weight_decay, args.grad_clip, args.mean_loss_weight,
        args.fluctuation_loss_weight, args.profile_loss_weight,
        args.truth_loss_weight)
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        raise ValueError("loss and optimizer values must be finite/non-negative")
    for name in ("weight_b", "mean_expert_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0,1]")


def load_student(
        checkpoint: Path, normalizer, device: torch.device,
        mean_only: bool):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    architecture = config["architecture"]
    if architecture not in ("ensemble_unet", "unet"):
        raise ValueError("student checkpoint must be ensemble_unet or unet")
    model = construct_model(architecture, config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    common.set_model_stats(model, normalizer)
    model.requires_grad_(False)
    head_groups = (
        model.member_point_heads
        if architecture == "ensemble_unet" else (model.point_heads,))
    for heads in head_groups:
        for parameter in heads.parameters():
            parameter.requires_grad_(True)
    if mean_only:
        def keep_uv_mean_rows(gradient: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(gradient)
            mask[:2] = 1.0
            return gradient * mask
        for heads in head_groups:
            for head in heads:
                head.weight.register_hook(keep_uv_mean_rows)
                head.bias.register_hook(keep_uv_mean_rows)
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("student exposes no trainable point heads")
    return model, payload, trainable


def evaluate(model, loader, normalizer, device) -> dict[str, float]:
    model.eval()
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


def point_head_distill_loss(
        args: argparse.Namespace, prediction: torch.Tensor,
        teacher_prediction: torch.Tensor,
        truth_normalized: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    student_mean = prediction[..., :2].mean(dim=1, keepdim=True)
    teacher_mean = teacher_prediction[..., :2].mean(
        dim=1, keepdim=True)
    student_fluctuation = prediction[..., :2] - student_mean
    teacher_fluctuation = (
        teacher_prediction[..., :2] - teacher_mean)
    mean_loss = F.smooth_l1_loss(
        student_mean / args.mean_scale,
        teacher_mean / args.mean_scale)
    fluctuation_loss = (
        F.smooth_l1_loss(
            student_fluctuation / args.fluctuation_scale,
            teacher_fluctuation / args.fluctuation_scale)
        if not args.mean_only else mean_loss.new_zeros(()))
    profile_loss = profile_distill_loss(
        student_mean, teacher_mean, args.mean_scale)
    truth_loss = (
        common.accuracy_objective(
            prediction, truth_normalized, normalizer,
            (5.0, 3.0, 2.0))[0]
        if args.truth_loss_weight else mean_loss.new_zeros(()))
    loss = (
        args.mean_loss_weight * mean_loss
        + args.fluctuation_loss_weight * fluctuation_loss
        + args.profile_loss_weight * profile_loss
        + args.truth_loss_weight * truth_loss)
    return loss, {
        "mean_distill": mean_loss.detach(),
        "fluctuation_distill": fluctuation_loss.detach(),
        "profile_distill": profile_loss.detach(),
        "truth_accuracy": truth_loss.detach(),
    }


def save_student(
        path: Path, model, source_payload: dict,
        args: argparse.Namespace, iteration: int,
        metrics: dict, teacher_metadata: list[dict]) -> None:
    output = dict(source_payload)
    output.update({
        "phase": "real",
        "iteration": iteration,
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()},
        "point_head_distillation": {
            "teacher_models": teacher_metadata,
            "weight_b": args.weight_b,
            "mean_expert_weight": args.mean_expert_weight,
            "metrics": metrics,
        },
    })
    torch.save(output, path)


def main() -> None:
    args = parse_args()
    validate(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("point-head distillation requires CUDA")
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
    fixed_a, metadata_a = load_forecaster(
        args.checkpoint_a, normalizer, device)
    model_b, metadata_b = load_forecaster(
        args.checkpoint_b, normalizer, device)
    model_c, metadata_c = load_forecaster(
        args.checkpoint_c, normalizer, device)
    student_checkpoint = (
        args.start_student
        if args.start_student is not None else args.checkpoint_a)
    student, source_payload, trainable = load_student(
        student_checkpoint, normalizer, device, args.mean_only)
    teacher_metadata = [metadata_a, metadata_b, metadata_c]
    trainable_parameters = sum(value.numel() for value in trainable)
    print(json.dumps({
        "event": "model",
        "total_parameters": sum(
            value.numel() for value in student.parameters()),
        "trainable_point_head_parameters": trainable_parameters,
        "teacher": teacher_metadata,
    }), flush=True)

    initial = evaluate(
        student, val_loader, normalizer, device)
    print(json.dumps({"event": "initial", **initial}), flush=True)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay)
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
        student.train()
        # Frozen normalization/backbone layers have no stochastic state, while
        # the point heads remain trainable.
        history_batch, truth = next(iterator)
        history_batch = history_batch.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        history_norm, truth_norm = normalizer.preprocess(
            history_batch, truth)
        teacher_values = teacher_targets(
            history_norm, fixed_a, model_b, model_c,
            args.weight_b, args.mean_expert_weight)
        teacher_prediction = teacher_values[3]
        with torch.no_grad():
            features = student._decode(history_norm)
        optimizer.zero_grad(set_to_none=True)
        detached_features = [value.detach() for value in features]
        if hasattr(student, "_members_from_features"):
            members = student._members_from_features(
                history_norm, detached_features)
            prediction = members.mean(dim=1)
        else:
            prediction = student._point_prediction(
                history_norm, detached_features)
        loss, components = point_head_distill_loss(
            args, prediction, teacher_prediction,
            truth_norm, normalizer)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite point-head loss at {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                trainable, args.grad_clip)
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
                student, val_loader, normalizer, device)
            row.update(metrics)
            history.append(row)
            if metrics["non_time_mean"] > best_value:
                best_value = metrics["non_time_mean"]
                best_iteration = iteration
                save_student(
                    args.output_dir / "best.pth", student,
                    source_payload, args, iteration,
                    metrics, teacher_metadata)
            print(json.dumps(
                {"event": "eval", **row}), flush=True)
            sums.clear()
            count = 0
        if iteration % args.checkpoint_interval == 0:
            save_student(
                args.output_dir / f"model_{iteration:05d}.pth",
                student, source_payload, args, iteration,
                history[-1] if history else {}, teacher_metadata)

    final_metrics = evaluate(
        student, val_loader, normalizer, device)
    save_student(
        args.output_dir / "final.pth", student,
        source_payload, args, args.updates,
        final_metrics, teacher_metadata)
    summary = {
        "method": (
            "three-model teacher distilled into existing multi-scale "
            "member point heads; no deployment-time modules added"),
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
        "total_parameters": sum(
            value.numel() for value in student.parameters()),
        "trainable_point_head_parameters": trainable_parameters,
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
