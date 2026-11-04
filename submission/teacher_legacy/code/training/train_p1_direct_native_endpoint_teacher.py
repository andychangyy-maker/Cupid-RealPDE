#!/usr/bin/env python3
"""Train SHA758b native UQ from exact post-selection P0 endpoints.

This route is intentionally different from both the failed full-7q teacher
and the old 572-tail lightweight endpoint head.  The first 1000 updates train
the *native* heavy UQ decoder on exact physical endpoints reconstructed from
formal A/B fold models' ``forward_distribution`` outputs (therefore including
utility selection and post-selection refinement) plus the formal C mean shift.
The RAM-only 2032-point anchor is then released.  The final 3000 updates use
only folds0--3 tail truth.  Held point targets are never used and held tail
fold04 is evaluated once, after update 4000.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, Path(__file__).parent, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_mean_expert_oof_blend as s1  # noqa: E402
import evaluate_p1_observation_gated_point_blend as point_gate  # noqa: E402
import train_ensemble_uq_oof as uq_train  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
import train_p1_direct_native_quantile_teacher as quantile_teacher  # noqa: E402
from train_distilled_flow_corrector import load_forecaster  # noqa: E402
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


ARCHITECTURE = "p1_direct_native_exact_endpoint_teacher"
WEIGHT_B = 0.625
WEIGHT_C = 0.25
HELD_FOLD = 4
EXPECTED_POINT_WINDOWS = 2576
EXPECTED_TAIL_WINDOWS = 725
EXPECTED_TRAIN_POINT = 2032
EXPECTED_HELD_POINT = 544
EXPECTED_TRAIN_TAIL = 572
EXPECTED_HELD_TAIL = 153
REFERENCE_WIDTH = (0.01075, 0.00550)
ANCHOR_FORMAT = "p0_exact_postselection_absolute_endpoints_fp16_ram_v1"

FORMAL_POINT_SHA256 = quantile_teacher.FORMAL_POINT_SHA256
FORMAL_SOURCE_SHA256 = quantile_teacher.FORMAL_SOURCE_SHA256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-point-sha256", required=True)
    parser.add_argument("--point-checkpoint-a", type=Path, required=True)
    parser.add_argument("--oof-dir-a", type=Path, required=True)
    parser.add_argument("--point-checkpoint-b", type=Path, required=True)
    parser.add_argument("--oof-dir-b", type=Path, required=True)
    parser.add_argument("--mean-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--anchor-only-updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--reconstruction-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--truth-pinball-weight", type=float, default=0.25)
    parser.add_argument("--soft-sps-weight", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float, default=0.25)
    parser.add_argument("--coverage-target", type=float, default=(
        quantile_teacher.FOLD04_REFERENCE["coverage"]))
    parser.add_argument("--soft-tau-fraction", type=float, default=0.10)
    parser.add_argument("--time-budget-seconds", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    quantile_teacher.validate_args(args)
    if args.expected_point_sha256 != FORMAL_POINT_SHA256:
        raise ValueError("exact-endpoint teacher requires frozen SHA758b")
    if (args.updates != 4000 or args.anchor_only_updates != 1000
            or args.batch_size != 8 or args.eval_batch_size != 16
            or args.reconstruction_batch_size != 16
            or args.lr != 1e-4 or args.weight_decay != 1e-5
            or args.grad_clip != 1.0 or args.truth_pinball_weight != 0.25
            or args.soft_sps_weight != 1.0
            or args.coverage_weight != 0.25
            or args.coverage_target
            != quantile_teacher.FOLD04_REFERENCE["coverage"]
            or args.soft_tau_fraction != 0.10
            or args.time_budget_seconds != 1800.0
            or args.seed != 0 or args.num_workers != 0
            or args.device != "cuda:0"):
        raise ValueError("formal exact-endpoint hyperparameters are frozen")


def validate_formal_source_files(args: argparse.Namespace) -> dict:
    source_paths = {
        "point_checkpoint_a": args.point_checkpoint_a,
        "point_checkpoint_b": args.point_checkpoint_b,
        "mean_expert_checkpoint": args.mean_expert_checkpoint,
    }
    identities = {
        name: s1.file_identity(path) for name, path in source_paths.items()}
    for name, expected in FORMAL_SOURCE_SHA256.items():
        if identities[name]["sha256"] != expected:
            raise ValueError(f"formal source SHA-256 differs at {name}")
    return identities


def validate_formal_oof_contracts(
        args: argparse.Namespace, fold_manifest: dict) -> dict:
    return {
        "a": quantile_teacher.validate_oof_quantile_contract(
            args.point_checkpoint_a, args.oof_dir_a, fold_manifest, "a"),
        "b": quantile_teacher.validate_oof_quantile_contract(
            args.point_checkpoint_b, args.oof_dir_b, fold_manifest, "b"),
    }


@torch.inference_mode()
def collect_absolute_endpoints(model, loader, normalizer,
                               device: torch.device) -> tuple[
                                   torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions = []
    lowers = []
    uppers = []
    dtype = next(model.parameters()).dtype
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    model.eval()
    for history, _unused_target in loader:
        history = history.to(device, non_blocking=True)
        normalized = (
            history - normalizer.mean_input) / normalizer.std_input
        prediction_norm, lower_extent, upper_extent = (
            model.forward_distribution(normalized.to(dtype=dtype)))
        prediction = common.to_physical_target(
            prediction_norm.float(), normalizer)[..., :2]
        lower = prediction - lower_extent.float() * scale
        upper = prediction + upper_extent.float() * scale
        if (not torch.isfinite(lower).all()
                or not torch.isfinite(upper).all()
                or torch.any(lower > upper)):
            raise FloatingPointError("formal fold model produced invalid bounds")
        predictions.append(prediction.cpu().to(torch.float16))
        lowers.append(lower.cpu().to(torch.float16))
        uppers.append(upper.cpu().to(torch.float16))
    return torch.cat(predictions), torch.cat(lowers), torch.cat(uppers)


def compose_p0_absolute_endpoints(
        prediction_a: torch.Tensor, lower_a: torch.Tensor,
        upper_a: torch.Tensor, prediction_b: torch.Tensor,
        lower_b: torch.Tensor, upper_b: torch.Tensor,
        prediction_c: torch.Tensor,
        weight_b: float = WEIGHT_B,
        weight_c: float = WEIGHT_C) -> tuple[torch.Tensor, torch.Tensor]:
    shape = prediction_a.shape
    values = (
        lower_a, upper_a, prediction_b, lower_b, upper_b, prediction_c)
    if (prediction_a.ndim != 5 or prediction_a.shape[-1] != 2
            or any(value.shape != shape for value in values)):
        raise ValueError("A/B/C endpoint tensors must share (N,T,H,W,2)")
    if not (0.0 <= weight_b <= 1.0 and 0.0 <= weight_c <= 1.0):
        raise ValueError("P0 endpoint weights must lie in [0,1]")
    base_prediction = prediction_a.lerp(prediction_b, weight_b)
    correction = weight_c * (
        prediction_c.mean(dim=1, keepdim=True)
        - base_prediction.mean(dim=1, keepdim=True))
    lower = lower_a.lerp(lower_b, weight_b) + correction
    upper = upper_a.lerp(upper_b, weight_b) + correction
    if (not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all()
            or torch.any(lower > upper)):
        raise FloatingPointError("composed exact P0 endpoints are invalid")
    return lower, upper


def reconstruct_point_endpoint_anchor(
        args: argparse.Namespace, dataset, point_folds: list[int],
        fold_manifest: dict, normalizer, device: torch.device,
        train_indices: list[int]) -> tuple[torch.Tensor, dict]:
    sources = validate_formal_source_files(args)
    contracts = validate_formal_oof_contracts(args, fold_manifest)
    payload_a = torch.load(
        args.point_checkpoint_a, map_location="cpu", weights_only=False)
    payload_b = torch.load(
        args.point_checkpoint_b, map_location="cpu", weights_only=False)
    row_of_index = {index: row for row, index in enumerate(train_indices)}
    anchor = torch.empty(
        (len(train_indices), 20, 32, 64, 2, 2), dtype=torch.float16)

    model_c, metadata_c = point_gate.load_model(
        args.mean_expert_checkpoint, normalizer, device)
    c_loader = quantile_teacher.make_loader(
        dataset, train_indices, args.reconstruction_batch_size,
        args.num_workers)
    prediction_c = quantile_teacher.collect_point_predictions(
        model_c, c_loader, normalizer, device)
    del model_c
    gc.collect()
    torch.cuda.empty_cache()

    fold_rows = []
    for fold in range(HELD_FOLD):
        indices = [index for index in train_indices
                   if point_folds[index] == fold]
        loader = quantile_teacher.make_loader(
            dataset, indices, args.reconstruction_batch_size,
            args.num_workers)
        model_a, metadata_a = s1.load_fold_model(
            payload_a, args.oof_dir_a / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_a, normalizer, device)
        prediction_a, lower_a, upper_a = collect_absolute_endpoints(
            model_a, loader, normalizer, device)
        del model_a
        gc.collect()
        torch.cuda.empty_cache()

        model_b, metadata_b = s1.load_fold_model(
            payload_b, args.oof_dir_b / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_b, normalizer, device)
        prediction_b, lower_b, upper_b = collect_absolute_endpoints(
            model_b, loader, normalizer, device)
        del model_b
        c_rows = torch.stack([
            prediction_c[row_of_index[index]] for index in indices])
        lower, upper = compose_p0_absolute_endpoints(
            prediction_a.float(), lower_a.float(), upper_a.float(),
            prediction_b.float(), lower_b.float(), upper_b.float(),
            c_rows.float())
        output_rows = torch.as_tensor(
            [row_of_index[index] for index in indices], dtype=torch.long)
        anchor[output_rows, ..., 0] = lower.to(torch.float16)
        anchor[output_rows, ..., 1] = upper.to(torch.float16)
        fold_rows.append({
            "fold": fold,
            "samples": len(indices),
            "model_a": metadata_a,
            "model_b": metadata_b,
        })
        print(json.dumps({
            "event": "exact_endpoint_anchor_fold",
            "fold": fold, "samples": len(indices),
        }), flush=True)
        del prediction_a, lower_a, upper_a
        del prediction_b, lower_b, upper_b, c_rows, lower, upper, output_rows
        gc.collect()
        torch.cuda.empty_cache()
    del payload_a, payload_b, prediction_c
    if (not torch.isfinite(anchor).all()
            or torch.any(anchor[..., 0] > anchor[..., 1])):
        raise FloatingPointError("RAM exact-endpoint anchor is invalid")
    return anchor, {
        "format": ANCHOR_FORMAT,
        "evidence": (
            "exact A/B post-selection/refinement absolute endpoints, "
            "then exact P0 B=.625/C=.25 composition"),
        "partition": "point train_starts folds0-3 only",
        "weights": {"b": WEIGHT_B, "c": WEIGHT_C},
        "train_folds": [0, 1, 2, 3],
        "held_fold_reconstructed": False,
        "samples": len(anchor),
        "stored_bytes": anchor.numel() * anchor.element_size(),
        "formal_sources": {
            "source_checkpoints": sources,
            "oof": contracts,
        },
        "oof_contracts": contracts,
        "mean_expert": metadata_c,
        "folds": fold_rows,
    }


class EndpointAnchorDataset(Dataset):
    def __init__(self, windows, train_indices: list[int],
                 anchor: torch.Tensor):
        if len(train_indices) != len(anchor):
            raise ValueError("endpoint anchor rows do not align")
        self.windows = windows
        self.indices = train_indices
        self.anchor = anchor

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int):
        history, target = self.windows[self.indices[row]]
        return history, target, self.anchor[row, ..., 0], self.anchor[row, ..., 1]


def absolute_endpoints_from_native(
        prediction_norm: torch.Tensor, lower_extent: torch.Tensor,
        upper_extent: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    prediction = common.to_physical_target(
        prediction_norm.float(), normalizer)[..., :2]
    lower = prediction - lower_extent.float() * scale
    upper = prediction + upper_extent.float() * scale
    if (not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all()
            or torch.any(lower > upper)):
        raise FloatingPointError("native endpoint output is invalid")
    return lower, upper


def endpoint_distillation_objective(
        prediction_norm: torch.Tensor, lower_extent: torch.Tensor,
        upper_extent: torch.Tensor, teacher_lower: torch.Tensor,
        teacher_upper: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    lower, upper = absolute_endpoints_from_native(
        prediction_norm, lower_extent, upper_extent, normalizer)
    reference = lower.new_tensor(REFERENCE_WIDTH).view(1, 1, 1, 1, 2)
    lower_loss = F.smooth_l1_loss(
        (lower - teacher_lower.float()) / reference,
        torch.zeros_like(lower))
    upper_loss = F.smooth_l1_loss(
        (upper - teacher_upper.float()) / reference,
        torch.zeros_like(upper))
    loss = 0.5 * (lower_loss + upper_loss)
    return loss, {
        "endpoint_distill": loss.detach(),
        "mean_width": (upper - lower).mean().detach(),
    }


def formal_training_partitions(
        point_folds: list[int], tail_folds: list[int],
        ) -> tuple[list[int], list[int], list[int], list[int]]:
    point_train, point_held = quantile_teacher.fold_partition_indices(
        point_folds)
    tail_train, tail_held = quantile_teacher.fold_partition_indices(tail_folds)
    if (any(point_folds[index] == HELD_FOLD for index in point_train)
            or any(tail_folds[index] == HELD_FOLD for index in tail_train)
            or any(point_folds[index] != HELD_FOLD for index in point_held)
            or any(tail_folds[index] != HELD_FOLD for index in tail_held)):
        raise RuntimeError("fold04 leaked across the formal partition")
    return point_train, point_held, tail_train, tail_held


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal exact-endpoint teacher requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    point_file = s1.file_identity(args.point_checkpoint)
    if point_file["sha256"] != FORMAL_POINT_SHA256:
        raise ValueError("point checkpoint is not frozen SHA758b")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    quantile_teacher.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    process_started = time.perf_counter()

    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    point_windows, tail_windows, runtime_manifest = common.repro.make_split(
        metadata, 20, 0.2)
    if (len(point_windows) != EXPECTED_POINT_WINDOWS
            or len(tail_windows) != EXPECTED_TAIL_WINDOWS):
        raise RuntimeError("frozen 2576/725 split changed")
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    if runtime_manifest != frozen_manifest:
        raise RuntimeError("runtime split differs from frozen manifest")
    tail_folds, fold_manifest = trajectory_folds(runtime_manifest, 5)
    point_folds = quantile_teacher.partition_fold_labels(
        runtime_manifest, fold_manifest, "train_starts")
    point_train, point_held, tail_train, tail_held = (
        formal_training_partitions(point_folds, tail_folds))
    if (len(point_train), len(point_held), len(tail_train), len(tail_held)) != (
            EXPECTED_TRAIN_POINT, EXPECTED_HELD_POINT,
            EXPECTED_TRAIN_TAIL, EXPECTED_HELD_TAIL):
        raise RuntimeError("fixed fold counts differ from 2032/544/572/153")
    point_dataset = common.repro.WindowDataset(trajectories, point_windows)
    tail_dataset = common.repro.WindowDataset(trajectories, tail_windows)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    anchor, anchor_metadata = reconstruct_point_endpoint_anchor(
        args, point_dataset, point_folds, fold_manifest,
        normalizer, device, point_train)
    anchor_dataset = EndpointAnchorDataset(
        point_dataset, point_train, anchor)
    anchor_loader = DataLoader(
        anchor_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    truth_loader = quantile_teacher.make_loader(
        tail_dataset, tail_train, args.batch_size, 0,
        shuffle=True, seed=args.seed + 1)
    held_loader = quantile_teacher.make_loader(
        tail_dataset, tail_held, args.eval_batch_size, 0)

    payload = torch.load(
        args.point_checkpoint, map_location="cpu", weights_only=False)
    model = uq_train.load_model(
        payload, normalizer, device, train_projections=True)
    point_state_before = quantile_teacher.point_state_identity(model)
    parameters = uq_train.trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    anchor_iterator = common.repro.cycling(anchor_loader)
    truth_iterator = common.repro.cycling(truth_loader)
    history_rows = []
    sums: dict[str, float] = {}
    count = 0
    completed_updates = 0
    training_started = time.perf_counter()
    for iteration in range(1, args.updates + 1):
        if time.perf_counter() - training_started > args.time_budget_seconds:
            break
        stage = quantile_teacher.training_stage(
            iteration, args.anchor_only_updates, args.updates)
        if stage == "point2032_p0_7q_pretrain":
            stage = "point2032_exact_endpoint_pretrain"
            history, _unused_truth, teacher_lower, teacher_upper = next(
                anchor_iterator)
            history = history.to(device, non_blocking=True)
            teacher_lower = teacher_lower.to(
                device, non_blocking=True).float()
            teacher_upper = teacher_upper.to(
                device, non_blocking=True).float()
            normalized = (
                history - normalizer.mean_input) / normalizer.std_input
            truth = None
        else:
            history, truth = next(truth_iterator)
            history = history.to(device, non_blocking=True)
            truth = truth.to(device, non_blocking=True).float()
            normalized, _ = normalizer.preprocess(history, truth)
            teacher_lower = teacher_upper = None
        optimizer.zero_grad(set_to_none=True)
        values = model.forward_uq_training(normalized)
        prediction, quantiles, lower, upper = (
            values[0], values[2], values[3], values[4])
        if stage == "point2032_exact_endpoint_pretrain":
            loss, components = endpoint_distillation_objective(
                prediction, lower, upper, teacher_lower,
                teacher_upper, normalizer)
        else:
            loss, components = quantile_teacher.native_truth_objective(
                args, prediction, quantiles, lower, upper,
                truth, normalizer)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {iteration}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        scheduler.step()
        completed_updates = iteration
        if iteration == args.anchor_only_updates:
            del anchor_iterator, anchor_loader, anchor_dataset, anchor
            gc.collect()
        for key, value in {"loss": loss.detach(), **components}.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration % 250 == 0 or iteration == args.updates:
            row = {
                "iteration": iteration,
                "stage": stage,
                "elapsed_seconds": time.perf_counter() - training_started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
            }
            history_rows.append(row)
            print(json.dumps({"event": "exact_endpoint_train", **row}),
                  flush=True)
            sums.clear()
            count = 0

    point_state_after = quantile_teacher.point_state_identity(model)
    point_state_equal = point_state_before == point_state_after
    if not point_state_equal:
        raise RuntimeError("frozen point/backbone state changed")
    uq_state = uq_train.uq_state_dict(model)
    del model, parameters, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    deploy_model, deploy_metadata = load_forecaster(
        args.point_checkpoint, normalizer, device)
    uq_train.load_uq_state(deploy_model, uq_state)
    evaluation = quantile_teacher.evaluate_native_teacher(
        deploy_model, held_loader, normalizer, device)
    gate = quantile_teacher.teacher_gate(
        args, evaluation, point_state_equal, completed_updates)
    status = "PASS_FULL_NATIVE_EXACT_ENDPOINT_TEACHER" if gate["passed"] else (
        "STOP_FULL_NATIVE_EXACT_ENDPOINT_TEACHER")
    checkpoint_path = args.output_dir / "native_exact_endpoint_teacher.pth"
    torch.save({
        "architecture": ARCHITECTURE,
        "model_config": payload["model_config"],
        "uq_state_dict": uq_state,
        "point_checkpoint": point_file,
        "held_fold": HELD_FOLD,
        "updates": completed_updates,
        "evaluation": evaluation,
        "gate": gate,
    }, checkpoint_path)
    summary = {
        "status": status,
        "evidence_class": (
            "fold04-held full-native exact-postselection-endpoint teacher; "
            "point2032 endpoint distill then tail572 truth; no official evidence"),
        "mechanism_difference": {
            "vs_failed_full7q_teacher": (
                "distills exact post-selection/refinement endpoints, not an "
                "irreversible reconstructed seven-quantile bank"),
            "vs_old_tail572_endpoint_head": (
                "trains the full native UQ decoder after 2032 dense point "
                "anchors, not a lightweight head from 572 truths"),
        },
        "point_checkpoint": point_file,
        "deploy_model": deploy_metadata,
        "counts": {
            "point_endpoint_distill_folds0_to3": len(point_train),
            "point_held_fold04_unused": len(point_held),
            "tail_truth_folds0_to3": len(tail_train),
            "tail_held_fold04_terminal_eval": len(tail_held),
        },
        "held_fold_evaluations_during_training": 0,
        "held_point_targets_used": False,
        "completed_updates": completed_updates,
        "wall_seconds": time.perf_counter() - process_started,
        "fold_manifest": fold_manifest,
        "anchor": anchor_metadata,
        "point_state": {
            "before": point_state_before,
            "after": point_state_after,
            "bitwise_equal": point_state_equal,
        },
        "evaluation": evaluation,
        "reference": quantile_teacher.FOLD04_REFERENCE,
        "gate": gate,
        "training_history": history_rows,
        "checkpoint": s1.file_identity(checkpoint_path),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "native_exact_endpoint_complete",
        "status": status, "evaluation": evaluation,
        "gate": gate, "checkpoint": summary["checkpoint"],
    }), flush=True)


if __name__ == "__main__":
    main()
