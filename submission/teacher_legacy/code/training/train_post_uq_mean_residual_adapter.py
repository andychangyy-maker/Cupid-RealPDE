#!/usr/bin/env python3
"""Distill a post-UQ temporal-mean residual without changing TKE or width."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_full_backbone_point_distillation as full  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
from train_distilled_flow_corrector import (  # noqa: E402
    profile_distill_loss,
    seed_everything,
)


FORMAL_BASE_SHA256 = (
    "758b44b2ce124d9c1bf72657d1a39247481c4794a6f2837bac8ffc072836ce07")
FORMAL_BASE_METRICS = {
    "rel_l2_score": 95.92538933854726,
    "tke_rel_l2_score": 80.96083157000368,
    "mvpe_rel_l2_score": 96.62538864624545,
    "point_score": 91.57602186952381,
}
FORMAL_BASE_TOLERANCE = 5e-4
STEP250_POINT_DELTA = 0.008
FINAL_POINT_DELTA = 0.015
TKE_INVARIANCE_TOLERANCE = 1e-4
WIDTH_INVARIANCE_TOLERANCE = 2e-7
MAX_POINT_DELTA = 0.75


class PostUQMeanResidualAdapter(nn.Module):
    """Predict one normalized u/v correction and broadcast it over time."""

    def __init__(self, input_channels: int, hidden_channels: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, (1, 9),
                padding=(0, 4), groups=hidden_channels),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, (9, 1),
                padding=(4, 0), groups=hidden_channels),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        correction = MAX_POINT_DELTA * torch.tanh(
            self.body(feature).permute(0, 2, 3, 1))
        return correction[:, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-student", type=Path, required=True)
    parser.add_argument("--teacher-archive", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mean-scale", type=float, default=0.025)
    parser.add_argument("--mean-loss-weight", type=float, default=2.0)
    parser.add_argument("--profile-loss-weight", type=float, default=2.0)
    parser.add_argument("--time-budget-seconds", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--expected-start-sha256")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    positive = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.eval_interval, args.hidden_channels, args.groups, args.lr,
        args.mean_scale, args.time_budget_seconds, args.window_stride)
    if any(value <= 0 for value in positive):
        raise ValueError("counts, scales, rates, and budgets must be positive")
    if args.updates % args.eval_interval:
        raise ValueError("updates must be divisible by eval_interval")
    if args.hidden_channels % args.groups:
        raise ValueError("hidden channels must be divisible by groups")
    if args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("weight decay and grad clip must be non-negative")
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("val fraction must lie in (0, 0.5)")


def corrected_prediction(
        base_prediction: torch.Tensor,
        correction_uv: torch.Tensor,
        ) -> torch.Tensor:
    if correction_uv.shape[1] != 1 or correction_uv.shape[-1] != 2:
        raise ValueError("adapter correction must have shape (B,1,H,W,2)")
    if base_prediction.shape[-1] == 2:
        correction = correction_uv
    else:
        correction = F.pad(
            correction_uv, (0, base_prediction.shape[-1] - 2))
    return base_prediction + correction


def corrected_distribution(
        base_prediction: torch.Tensor,
        lower_extent: torch.Tensor, upper_extent: torch.Tensor,
        correction_uv: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the adapter after UQ while preserving both residual extents."""
    return (
        corrected_prediction(base_prediction, correction_uv),
        lower_extent,
        upper_extent,
    )


def adapter_loss(
        correction_uv: torch.Tensor, base_prediction: torch.Tensor,
        teacher: torch.Tensor, args: argparse.Namespace,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_delta = (
        teacher[..., :2].mean(dim=1, keepdim=True)
        - base_prediction[..., :2].mean(dim=1, keepdim=True))
    mean_loss = F.smooth_l1_loss(
        correction_uv / args.mean_scale,
        target_delta / args.mean_scale)
    profile_loss = profile_distill_loss(
        correction_uv, target_delta, args.mean_scale)
    loss = (
        args.mean_loss_weight * mean_loss
        + args.profile_loss_weight * profile_loss)
    return loss, {
        "mean_distill": mean_loss.detach(),
        "profile_distill": profile_loss.detach(),
    }


def metric_delta(base: dict, corrected: dict) -> dict[str, float]:
    return {
        key: float(corrected[key] - base[key])
        for key in full.POINT_GATE
    }


@torch.inference_mode()
def evaluate(
        model, adapter, loader, normalizer, device, val_aoa,
        with_distribution: bool,
        ) -> dict:
    model.eval()
    adapter.eval()
    base_predictions = []
    corrected_predictions = []
    targets = []
    base_lowers = []
    base_uppers = []
    corrected_lowers = []
    corrected_uppers = []
    extents_bitwise_equal = True
    max_correction_temporal_span = 0.0
    target_std = normalizer.std_target[:2].view(1, 1, 1, 1, 2)

    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history, target)
        if with_distribution:
            base_norm, lower_extent, upper_extent, features = (
                model.forward_distribution_with_features(history_norm))
            base_lower_extent = lower_extent.detach().clone()
            base_upper_extent = upper_extent.detach().clone()
        else:
            base_norm, features = model.forward_with_features(history_norm)
        correction_uv = adapter(features[-1]).float()
        broadcast_correction = correction_uv.expand(
            -1, base_norm.shape[1], -1, -1, -1)
        correction_span = (
            broadcast_correction.amax(dim=1)
            - broadcast_correction.amin(dim=1))
        max_correction_temporal_span = max(
            max_correction_temporal_span,
            float(correction_span.abs().max()))
        if with_distribution:
            (corrected_norm, corrected_lower_extent,
             corrected_upper_extent) = corrected_distribution(
                 base_norm.float(), lower_extent, upper_extent,
                 correction_uv)
        else:
            corrected_norm = corrected_prediction(
                base_norm.float(), correction_uv)
        base_physical = normalizer.postprocess_prediction(
            base_norm.float())
        corrected_physical = normalizer.postprocess_prediction(
            corrected_norm)
        target_physical = normalizer.postprocess_target(target_norm)
        base_predictions.append(base_physical.cpu().numpy())
        corrected_predictions.append(corrected_physical.cpu().numpy())
        targets.append(target_physical.cpu().numpy())
        if with_distribution:
            extents_bitwise_equal = bool(
                extents_bitwise_equal
                and torch.equal(
                    corrected_lower_extent, base_lower_extent)
                and torch.equal(
                    corrected_upper_extent, base_upper_extent))
            lower_extent = base_lower_extent.float() * target_std
            upper_extent = base_upper_extent.float() * target_std
            corrected_lower_extent = (
                corrected_lower_extent.float() * target_std)
            corrected_upper_extent = (
                corrected_upper_extent.float() * target_std)
            base_point_uv = base_physical[..., :2]
            corrected_point_uv = corrected_physical[..., :2]
            base_lowers.append((base_point_uv - lower_extent).cpu().numpy())
            base_uppers.append((base_point_uv + upper_extent).cpu().numpy())
            corrected_lowers.append(
                (corrected_point_uv - corrected_lower_extent).cpu().numpy())
            corrected_uppers.append(
                (corrected_point_uv + corrected_upper_extent).cpu().numpy())

    base_prediction = np.concatenate(base_predictions)
    corrected_prediction_array = np.concatenate(corrected_predictions)
    target_array = np.concatenate(targets)
    if base_prediction.shape[0] != val_aoa.shape[0]:
        raise RuntimeError("AoA labels do not align with validation predictions")
    base_metrics = full.metric_bundle(base_prediction, target_array)
    corrected_metrics = full.metric_bundle(
        corrected_prediction_array, target_array)

    by_aoa = {}
    for aoa in sorted(np.unique(val_aoa)):
        selected = val_aoa == aoa
        base_group = full.metric_bundle(
            base_prediction[selected], target_array[selected])
        corrected_group = full.metric_bundle(
            corrected_prediction_array[selected], target_array[selected])
        by_aoa[f"{float(aoa):g}"] = {
            "base": base_group,
            "corrected": corrected_group,
            "delta": metric_delta(base_group, corrected_group),
        }

    result = {
        "base": base_metrics,
        "corrected": corrected_metrics,
        "delta": metric_delta(base_metrics, corrected_metrics),
        "by_aoa": by_aoa,
    }
    if with_distribution:
        base_lower = np.concatenate(base_lowers)
        base_upper = np.concatenate(base_uppers)
        corrected_lower = np.concatenate(corrected_lowers)
        corrected_upper = np.concatenate(corrected_uppers)
        channels = full.OFFICIAL.measured_channels(target_array)
        base_sps, base_coverage = full.OFFICIAL.aggregate_sps(
            base_prediction, target_array, channels,
            lower=base_lower, upper=base_upper)
        corrected_sps, corrected_coverage = full.OFFICIAL.aggregate_sps(
            corrected_prediction_array, target_array, channels,
            lower=corrected_lower, upper=corrected_upper)
        base_width = base_upper - base_lower
        corrected_width = corrected_upper - corrected_lower
        result["distribution"] = {
            "evidence_scope": "diagnostic_only_stale_uq",
            "uq_state_valid": False,
            "sps_actionable": False,
            "base_sps_score": full.OFFICIAL.score_sps(base_sps),
            "corrected_sps_score": full.OFFICIAL.score_sps(corrected_sps),
            "sps_delta": full.OFFICIAL.score_sps(corrected_sps)
            - full.OFFICIAL.score_sps(base_sps),
            "base_coverage": base_coverage,
            "corrected_coverage": corrected_coverage,
            "base_mean_width": float(np.mean(base_width)),
            "corrected_mean_width": float(np.mean(corrected_width)),
            "max_abs_width_drift": float(np.max(np.abs(
                corrected_width - base_width))),
            "extents_bitwise_equal": extents_bitwise_equal,
            "max_abs_correction_temporal_span": (
                max_correction_temporal_span),
        }
    return result


def formal_contract_errors(
        args: argparse.Namespace, initial: dict,
        start_sha256: str, teacher_sha256: str,
        split_manifest_sha256: str, real_stats_sha256: str,
        train_windows: int, val_windows: int, trajectories: int,
        source_payload: dict,
        ) -> list[str]:
    if not args.formal:
        return ["--formal was not requested"]
    expected_args = {
        "updates": 500,
        "eval_interval": 250,
        "max_files": 0,
        "seed": 0,
        "window_stride": 20,
        "val_fraction": 0.2,
        "batch_size": 8,
        "eval_batch_size": 16,
        "hidden_channels": 64,
        "groups": 8,
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "mean_scale": 0.025,
        "mean_loss_weight": 2.0,
        "profile_loss_weight": 2.0,
        "time_budget_seconds": 600.0,
    }
    errors = []
    for key, expected in expected_args.items():
        actual = getattr(args, key)
        matches = (
            bool(np.isclose(actual, expected, rtol=0.0, atol=1e-12))
            if isinstance(expected, float) else actual == expected)
        if not matches:
            errors.append(f"{key}={actual!r}, expected {expected!r}")
    actual_counts = {
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
    }
    for key, expected in full.FORMAL_COUNTS.items():
        if actual_counts[key] != expected:
            errors.append(
                f"{key}={actual_counts[key]}, expected {expected}")
    if teacher_sha256 != full.FORMAL_TEACHER_SHA256:
        errors.append("teacher archive sha256 mismatch")
    if real_stats_sha256 != full.FORMAL_REAL_STATS_SHA256:
        errors.append("real stats sha256 mismatch")
    if split_manifest_sha256 != full.FORMAL_SPLIT_MANIFEST_SHA256:
        errors.append("split manifest sha256 mismatch")
    if not args.expected_start_sha256:
        errors.append("--expected-start-sha256 is required with --formal")
    elif start_sha256.lower() != args.expected_start_sha256.lower():
        errors.append("start sha256 does not match the command contract")
    if start_sha256 != FORMAL_BASE_SHA256:
        errors.append("start checkpoint is not the frozen final_00800 baseline")
    base_record = source_payload.get("full_backbone_point_distillation", {})
    if base_record.get("status") != "STOP_FINAL_GATE":
        errors.append("start checkpoint lacks the expected STOP_FINAL_GATE record")
    if base_record.get("uq_state_valid") is not False:
        errors.append("start checkpoint must retain uq_state_valid=false")
    for key, expected in FORMAL_BASE_METRICS.items():
        actual = float(initial["corrected"][key])
        if abs(actual - expected) > FORMAL_BASE_TOLERANCE:
            errors.append(
                f"initial {key}={actual:.8f}, expected {expected:.8f} "
                f"within {FORMAL_BASE_TOLERANCE}")
    initial_delta = initial["delta"]
    if any(abs(value) > 1e-7 for value in initial_delta.values()):
        errors.append("zero-initialized adapter changed point metrics")
    return errors


def save_adapter(
        path: Path, adapter, args, iteration: int,
        evaluation: dict, status: str, start_sha256: str,
        ) -> None:
    torch.save({
        "method": "post_uq_temporal_mean_residual_adapter",
        "status": status,
        "iteration": iteration,
        "base_checkpoint": str(args.start_student.resolve()),
        "base_checkpoint_sha256": start_sha256,
        "teacher_archive": str(args.teacher_archive.resolve()),
        "teacher_archive_sha256": full.sha256(args.teacher_archive),
        "adapter_config": {
            "input_channels": 128,
            "hidden_channels": args.hidden_channels,
            "groups": args.groups,
            "output_channels": 2,
            "broadcast_future_steps": 20,
            "max_point_delta": MAX_POINT_DELTA,
            "application": (
                "after UQ; shift point only, preserve residual extents, "
                "so derived endpoints shift equally"),
        },
        "adapter_state_dict": {
            key: value.detach().cpu()
            for key, value in adapter.state_dict().items()},
        "metrics": evaluation,
        "uq_state_valid": False,
        "sps_actionable": False,
    }, path)


def main() -> None:
    args = parse_args()
    validate(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("mean residual distillation requires CUDA")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    (train_loader, cache_loader, val_loader, normalizer,
     train_keys, val_aoa, trajectories, split_manifest_sha256,
     real_stats_sha256, expected_teacher_shape) = full.prepare_data(
         args, device)
    start_sha256 = full.sha256(args.start_student)
    teacher_sha256 = full.sha256(args.teacher_archive)
    model, source_payload = full.load_student(
        args.start_student, normalizer, device)
    model.requires_grad_(False)
    model.eval()
    adapter = PostUQMeanResidualAdapter(
        input_channels=128,
        hidden_channels=args.hidden_channels,
        groups=args.groups).to(device)

    initial = evaluate(
        model, adapter, val_loader, normalizer, device, val_aoa,
        with_distribution=False)
    print(json.dumps({"event": "initial", **initial}), flush=True)
    contract_errors = formal_contract_errors(
        args, initial, start_sha256, teacher_sha256,
        split_manifest_sha256, real_stats_sha256,
        len(train_keys), int(val_aoa.shape[0]), trajectories,
        source_payload)
    formal_contract_pass = bool(args.formal and not contract_errors)
    if args.formal and contract_errors:
        rejection = {
            "status": "REJECT_FORMAL_CONTRACT",
            "errors": contract_errors,
            "initial": initial,
        }
        (args.output_dir / "contract_rejection.json").write_text(
            json.dumps(rejection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        raise RuntimeError(json.dumps(rejection, ensure_ascii=False))

    teacher_cache = full.build_or_load_teacher_cache(
        args, cache_loader, normalizer, train_keys,
        split_manifest_sha256, real_stats_sha256,
        expected_teacher_shape)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    iterator = common.repro.cycling(train_loader)
    history = []
    sums: dict[str, float] = {}
    count = 0
    status = "RUNNING"
    completed_iteration = 0
    training_started = time.time()

    for iteration in range(1, args.updates + 1):
        if time.time() - started > args.time_budget_seconds:
            status = "STOP_TIME_BUDGET"
            break
        indices, history_batch, truth = next(iterator)
        history_batch = history_batch.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        teacher = teacher_cache[indices].to(
            device=device, dtype=torch.float32, non_blocking=True)
        history_norm, _truth_norm = normalizer.preprocess(history_batch, truth)
        with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16):
            base_prediction, features = model.forward_with_features(
                history_norm)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            correction_uv = adapter(features[-1].detach())
        loss, components = adapter_loss(
            correction_uv.float(), base_prediction.detach().float(),
            teacher, args)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at iteration {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        completed_iteration = iteration
        for key, value in {"loss": loss.detach(), **components}.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1

        if iteration % args.eval_interval == 0:
            evaluation = evaluate(
                model, adapter, val_loader, normalizer, device, val_aoa,
                with_distribution=False)
            row = {
                "iteration": iteration,
                "elapsed_seconds": time.time() - started,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                **evaluation,
            }
            history.append(row)
            print(json.dumps({"event": "eval", **row}), flush=True)
            save_adapter(
                args.output_dir / f"adapter_{iteration:05d}.pth",
                adapter, args, iteration, evaluation, "CHECKPOINT",
                start_sha256)
            sums.clear()
            count = 0
            if iteration == 250:
                if (evaluation["delta"]["point_score"]
                        < STEP250_POINT_DELTA
                        or abs(evaluation["delta"]["tke_rel_l2_score"])
                        > TKE_INVARIANCE_TOLERANCE):
                    status = "STOP_STEP250_NO_SIGNAL"
                    break

    final = evaluate(
        model, adapter, val_loader, normalizer, device, val_aoa,
        with_distribution=True)
    positive_aoa = sum(
        item["delta"]["point_score"] >= 0.0
        for item in final["by_aoa"].values())
    pooled_pass = bool(
        final["delta"]["point_score"] >= FINAL_POINT_DELTA
        and final["delta"]["rel_l2_score"] >= 0.0
        and final["delta"]["mvpe_rel_l2_score"] >= 0.0
        and abs(final["delta"]["tke_rel_l2_score"])
        <= TKE_INVARIANCE_TOLERANCE)
    aoa_slice_pass = bool(
        len(final["by_aoa"]) == 5 and positive_aoa >= 4)
    width_pass = bool(
        final["distribution"]["max_abs_width_drift"]
        <= WIDTH_INVARIANCE_TOLERANCE
        and final["distribution"]["extents_bitwise_equal"]
        and final["distribution"][
            "max_abs_correction_temporal_span"] == 0.0)
    formal_gate_evaluated = bool(
        formal_contract_pass and completed_iteration == 500)
    provisional_gate_pass = bool(
        formal_gate_evaluated and pooled_pass
        and aoa_slice_pass and width_pass)
    if status == "RUNNING":
        if not args.formal:
            status = "NONFORMAL_COMPLETE_NO_GATE"
        elif provisional_gate_pass:
            status = "PROVISIONAL_GO_MEAN_ADAPTER_PENDING_DISTILL_LOAO"
        else:
            status = "STOP_FINAL_GATE"

    final_path = args.output_dir / f"final_adapter_{completed_iteration:05d}.pth"
    save_adapter(
        final_path, adapter, args, completed_iteration,
        final, status, start_sha256)
    summary = {
        "method": "post_uq_temporal_mean_residual_adapter",
        "status": status,
        "formal_contract_pass": formal_contract_pass,
        "formal_gate_evaluated": formal_gate_evaluated,
        "provisional_gate_pass": provisional_gate_pass,
        "promotion_gate_pass": False,
        "uq_state_valid": False,
        "sps_actionable": False,
        "required_next_gate": (
            "five-fold distillation-stage leave-one-AoA delta crossfit; "
            "base and teacher have seen all AoAs; UQ must be refit or "
            "revalidated before packaging"),
        "initial": initial,
        "final": final,
        "gates": {
            "step250_point_delta": STEP250_POINT_DELTA,
            "final_point_delta": FINAL_POINT_DELTA,
            "tke_invariance_tolerance": TKE_INVARIANCE_TOLERANCE,
            "width_invariance_tolerance": WIDTH_INVARIANCE_TOLERANCE,
            "required_positive_aoa_slices": 4,
            "pooled_pass": pooled_pass,
            "aoa_slice_pass": aoa_slice_pass,
            "width_pass": width_pass,
        },
        "positive_aoa_slices": positive_aoa,
        "completed_iteration": completed_iteration,
        "wall_seconds": time.time() - started,
        "training_wall_seconds": time.time() - training_started,
        "train_windows": len(train_keys),
        "val_windows": int(val_aoa.shape[0]),
        "trajectories": trajectories,
        "base_checkpoint_sha256": start_sha256,
        "teacher_archive_sha256": teacher_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "real_stats_sha256": real_stats_sha256,
        "script_sha256": full.sha256(Path(__file__)),
        "history": history,
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
        "final_adapter": str(final_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "status": status,
        "provisional_gate_pass": provisional_gate_pass,
        "completed_iteration": completed_iteration,
        "summary": str(args.output_dir / "summary.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
