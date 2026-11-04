#!/usr/bin/env python3
"""Five-fold distillation-stage leave-one-AoA-out point-model gate.

This is deliberately a screening driver, not a deployment trainer.  It loads
the frozen 2576/725 split and teacher cache once, then starts each held-AoA run
from the exact same H96 checkpoint.  A held AoA is absent from every optimizer
update and is touched only for the update-0 and terminal update-800 metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_full_backbone_point_distillation as base  # noqa: E402


HELD_AOAS = (0.0, 5.0, 10.0, 15.0, 20.0)
FORMAL_START_SHA256 = (
    "71fd8daad9893adddc1e8b0d46ce9972455700f7c82296e74adacf035705daef")
FORMAL_UPDATES = 800
FORMAL_BATCH_SIZE = 8
FORMAL_EVAL_BATCH_SIZE = 16
FORMAL_ENCODER_LR = 2e-6
FORMAL_DECODER_LR = 1e-5
FORMAL_POINT_HEAD_LR = 3e-5
FORMAL_WEIGHT_DECAY = 1e-5
FORMAL_GRAD_CLIP = 1.0
FORMAL_SEED = 0
FORMAL_WINDOW_STRIDE = 20
FORMAL_VAL_FRACTION = 0.2
FORMAL_FOLD_TIME_BUDGET_SECONDS = 1500.0
MIN_NONNEGATIVE_POINT_FOLDS = 4
WORST_POINT_DELTA_FLOOR = -0.02
WORST_TKE_DELTA_FLOOR = -0.10


class OriginalIndexSubset(Dataset):
    """Subset a dataset while retaining indices into the full teacher cache."""

    def __init__(self, base_dataset: Dataset, original_indices: list[int]):
        self.base_dataset = base_dataset
        self.original_indices = tuple(int(index) for index in original_indices)

    def __len__(self) -> int:
        return len(self.original_indices)

    def __getitem__(self, index: int):
        original_index = self.original_indices[index]
        history, target = self.base_dataset[original_index]
        return original_index, history, target


def window_aoa(window, metadata: list[dict]) -> float:
    return float(metadata[window.trajectory]["aoa"])


def build_loao_partition(
        train_windows: list, validation_windows: list,
        metadata: list[dict], held_aoa: float,
        ) -> dict:
    """Return original train/cache indices and held-tail indices for one fold."""
    held_aoa = float(held_aoa)
    train_indices = [
        index for index, window in enumerate(train_windows)
        if window_aoa(window, metadata) != held_aoa]
    excluded_train_indices = [
        index for index, window in enumerate(train_windows)
        if window_aoa(window, metadata) == held_aoa]
    held_validation_indices = [
        index for index, window in enumerate(validation_windows)
        if window_aoa(window, metadata) == held_aoa]
    train_aoas = [
        window_aoa(train_windows[index], metadata)
        for index in train_indices]
    validation_aoas = [
        window_aoa(validation_windows[index], metadata)
        for index in held_validation_indices]
    if not train_indices:
        raise ValueError(f"held AoA {held_aoa:g} leaves no training windows")
    if not excluded_train_indices:
        raise ValueError(f"held AoA {held_aoa:g} has no excluded train windows")
    if not held_validation_indices:
        raise ValueError(f"held AoA {held_aoa:g} has no fixed725 windows")
    if any(value == held_aoa for value in train_aoas):
        raise RuntimeError("held AoA leaked into optimizer windows")
    if any(value != held_aoa for value in validation_aoas):
        raise RuntimeError("non-held AoA leaked into terminal evaluation windows")
    return {
        "held_aoa": held_aoa,
        "train_indices": train_indices,
        "excluded_train_indices": excluded_train_indices,
        "held_validation_indices": held_validation_indices,
        "train_windows": len(train_indices),
        "excluded_held_train_windows": len(excluded_train_indices),
        "held_validation_windows": len(held_validation_indices),
        "train_aoas": sorted(set(train_aoas)),
        "validation_aoas": sorted(set(validation_aoas)),
    }


def metrics_delta(initial: dict, final: dict) -> dict[str, float]:
    names = (
        "rel_l2_score", "tke_rel_l2_score",
        "mvpe_rel_l2_score", "point_score")
    return {
        name: float(final[name]) - float(initial[name])
        for name in names}


def loao_gate(fold_rows: list[dict]) -> dict:
    expected = [float(value) for value in HELD_AOAS]
    observed = [float(row["held_aoa"]) for row in fold_rows]
    complete = observed == expected
    finite = all(bool(row["point_state_finite"]) for row in fold_rows)
    update_complete = all(
        int(row["completed_updates"]) == FORMAL_UPDATES for row in fold_rows)
    point_deltas = [float(row["delta"]["point_score"]) for row in fold_rows]
    tke_deltas = [
        float(row["delta"]["tke_rel_l2_score"]) for row in fold_rows]
    deltas_finite = (
        len(point_deltas) == len(expected)
        and len(tke_deltas) == len(expected)
        and all(np.isfinite(value) for value in (*point_deltas, *tke_deltas)))
    nonnegative = (
        sum(delta >= 0.0 for delta in point_deltas)
        if deltas_finite else 0)
    worst_point_safe = (
        deltas_finite and min(point_deltas) >= WORST_POINT_DELTA_FLOOR)
    worst_tke_safe = (
        deltas_finite and min(tke_deltas) >= WORST_TKE_DELTA_FLOOR)
    passed = bool(
        complete and finite and update_complete and deltas_finite
        and nonnegative >= MIN_NONNEGATIVE_POINT_FOLDS
        and worst_point_safe and worst_tke_safe)
    return {
        "held_aoa_order_complete": complete,
        "point_state_finite_all_folds": finite,
        "updates_800_all_folds": update_complete,
        "point_nonnegative_folds": nonnegative,
        "required_point_nonnegative_folds": MIN_NONNEGATIVE_POINT_FOLDS,
        "worst_point_delta": (
            min(point_deltas) if deltas_finite else None),
        "worst_point_delta_floor": WORST_POINT_DELTA_FLOOR,
        "worst_point_delta_safe": worst_point_safe,
        "worst_tke_delta": min(tke_deltas) if deltas_finite else None,
        "worst_tke_delta_floor": WORST_TKE_DELTA_FLOOR,
        "worst_tke_delta_safe": worst_tke_safe,
        "pass": passed,
    }


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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--fold-time-budget-seconds", type=float,
        default=FORMAL_FOLD_TIME_BUDGET_SECONDS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.fold_time_budget_seconds != FORMAL_FOLD_TIME_BUDGET_SECONDS:
        raise ValueError(
            "formal LOAO freezes each fold time budget at 1500 seconds")
    if args.output_dir.joinpath("summary.json").exists():
        raise FileExistsError(
            f"refusing to overwrite {args.output_dir / 'summary.json'}")


def prepare_fixed_data(args: argparse.Namespace, device: torch.device) -> dict:
    trajectories, metadata = base.common.repro.load_trajectories(
        args.real_dir, 0)
    train_windows, validation_windows, manifest = base.common.repro.make_split(
        metadata, FORMAL_WINDOW_STRIDE, FORMAL_VAL_FRACTION)
    train_dataset = base.common.repro.WindowDataset(
        trajectories, train_windows)
    validation_dataset = base.common.repro.WindowDataset(
        trajectories, validation_windows)
    normalizer = base.common.repro.Normalizer(args.real_stats, device)
    train_keys = [
        (metadata[window.trajectory]["file"], int(window.start))
        for window in train_windows]
    example_target = train_dataset[0][1]
    expected_teacher_shape = (
        len(train_windows), *example_target.shape[:-1], 2)
    split_manifest_sha256 = base.canonical_sha256(manifest)
    real_stats_sha256 = base.sha256(args.real_stats)
    aoas = sorted({float(item["aoa"]) for item in metadata})
    contract_errors = []
    actual_counts = {
        "train_windows": len(train_windows),
        "val_windows": len(validation_windows),
        "trajectories": len(trajectories),
    }
    for name, expected in base.FORMAL_COUNTS.items():
        if actual_counts[name] != expected:
            contract_errors.append(
                f"{name}={actual_counts[name]}, expected {expected}")
    if split_manifest_sha256 != base.FORMAL_SPLIT_MANIFEST_SHA256:
        contract_errors.append("fixed split manifest hash mismatch")
    if real_stats_sha256 != base.FORMAL_REAL_STATS_SHA256:
        contract_errors.append("real stats hash mismatch")
    if aoas != list(HELD_AOAS):
        contract_errors.append(f"AoA values {aoas} != {list(HELD_AOAS)}")
    if contract_errors:
        raise RuntimeError("; ".join(contract_errors))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return {
        "trajectories": trajectories,
        "metadata": metadata,
        "train_windows": train_windows,
        "validation_windows": validation_windows,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "normalizer": normalizer,
        "train_keys": train_keys,
        "expected_teacher_shape": expected_teacher_shape,
        "split_manifest_sha256": split_manifest_sha256,
        "real_stats_sha256": real_stats_sha256,
        "manifest": manifest,
    }


def state_finite_audit(model: torch.nn.Module) -> dict:
    floating = 0
    point = 0
    all_finite = True
    point_finite = True
    for name, value in model.state_dict().items():
        if not (torch.is_floating_point(value) or torch.is_complex(value)):
            continue
        floating += 1
        value_finite = base.tensor_is_finite(
            value.reshape(-1), chunk_size=1024 * 1024)
        all_finite = all_finite and value_finite
        if name.startswith("member_point_heads."):
            point += 1
            point_finite = point_finite and value_finite
    if point == 0:
        raise RuntimeError("student has no member_point_heads state")
    return {
        "floating_state_tensors": floating,
        "point_state_tensors": point,
        "all_state_finite": bool(all_finite),
        "point_state_finite": bool(point_finite),
    }


def make_train_loader(
        train_dataset: Dataset, original_indices: list[int],
        args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        OriginalIndexSubset(train_dataset, original_indices),
        batch_size=FORMAL_BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(FORMAL_SEED))


def make_validation_loader(
        validation_dataset: Dataset, held_indices: list[int],
        args: argparse.Namespace) -> DataLoader:
    subset = torch.utils.data.Subset(validation_dataset, held_indices)
    return DataLoader(
        subset, batch_size=FORMAL_EVAL_BATCH_SIZE, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)


def train_one_fold(
        args: argparse.Namespace, data: dict, teacher_cache: torch.Tensor,
        held_aoa: float, device: torch.device,
        ) -> dict:
    partition = build_loao_partition(
        data["train_windows"], data["validation_windows"],
        data["metadata"], held_aoa)
    base.seed_everything(FORMAL_SEED)
    model, _source_payload = base.load_student(
        args.start_student, data["normalizer"], device)
    train_loader = make_train_loader(
        data["train_dataset"], partition["train_indices"], args)
    held_loader = make_validation_loader(
        data["validation_dataset"],
        partition["held_validation_indices"], args)
    held_labels = np.full(
        partition["held_validation_windows"], held_aoa, dtype=np.float32)

    # These are the only two calls that touch held-tail samples: update 0 and
    # the terminal update.  No held loader is reachable from the update loop.
    initial = base.evaluate_point(
        model, held_loader, data["normalizer"], device, held_labels)
    print(json.dumps({
        "event": "loao_fold_start", "held_aoa": held_aoa,
        "initial": initial, "isolation": {
            key: value for key, value in partition.items()
            if not key.endswith("indices")}}, ensure_ascii=False), flush=True)

    groups = base.optimizer_groups(model)
    learning_rates = (
        FORMAL_ENCODER_LR, FORMAL_DECODER_LR, FORMAL_POINT_HEAD_LR)
    for group, learning_rate in zip(groups, learning_rates):
        group["lr"] = learning_rate
    optimizer = torch.optim.AdamW(groups, weight_decay=FORMAL_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FORMAL_UPDATES)
    trainable = [
        parameter for group in groups for parameter in group["params"]]
    iterator = base.common.repro.cycling(train_loader)
    started = time.time()
    completed_updates = 0
    running_loss = 0.0
    running_count = 0
    train_log = []
    status = "RUNNING"
    for iteration in range(1, FORMAL_UPDATES + 1):
        if time.time() - started > args.fold_time_budget_seconds:
            status = "STOP_FOLD_TIME_BUDGET"
            break
        model.train()
        original_indices, history, truth = next(iterator)
        if any(
                int(index) not in partition["train_indices"]
                for index in original_indices.tolist()):
            raise RuntimeError("optimizer loader emitted an unauthorized index")
        history = history.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        teacher_uv = teacher_cache[original_indices].to(
            device=device, dtype=torch.float32, non_blocking=True)
        history_norm, truth_norm = data["normalizer"].preprocess(
            history, truth)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(history_norm)
        loss, _components = base.distill_loss(
            prediction.float(), teacher_uv, truth_norm, data["normalizer"])
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss for held AoA {held_aoa:g} "
                f"at update {iteration}")
        loss.backward()
        if FORMAL_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(trainable, FORMAL_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        completed_updates = iteration
        running_loss += float(loss.detach())
        running_count += 1
        if iteration % 100 == 0:
            row = {
                "iteration": iteration,
                "train_loss_mean": running_loss / running_count,
                "elapsed_seconds": time.time() - started,
            }
            train_log.append(row)
            print(json.dumps({
                "event": "loao_train_progress", "held_aoa": held_aoa,
                **row}), flush=True)
            running_loss = 0.0
            running_count = 0

    finite_audit = state_finite_audit(model)
    if not finite_audit["point_state_finite"]:
        raise FloatingPointError(
            f"non-finite point state for held AoA {held_aoa:g}")
    final = base.evaluate_point(
        model, held_loader, data["normalizer"], device, held_labels)
    delta = metrics_delta(initial, final)
    if status == "RUNNING":
        status = (
            "FOLD_POINT_NONDECREASING"
            if delta["point_score"] >= 0.0
            else "FOLD_POINT_DECREASED")
    row = {
        "held_aoa": held_aoa,
        "status": status,
        "completed_updates": completed_updates,
        "initial": initial,
        "final": final,
        "delta": delta,
        "point_state_finite": finite_audit["point_state_finite"],
        "state_finite_audit": finite_audit,
        "isolation": {
            key: value for key, value in partition.items()
            if not key.endswith("indices")},
        "train_log_without_held_evaluation": train_log,
        "wall_seconds": time.time() - started,
    }
    print(json.dumps({
        "event": "loao_fold_terminal", **row}, ensure_ascii=False), flush=True)
    del optimizer, scheduler, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def write_summary(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal point LOAO requires CUDA")
    start_sha256 = base.sha256(args.start_student)
    teacher_sha256 = base.sha256(args.teacher_archive)
    if start_sha256 != FORMAL_START_SHA256:
        raise RuntimeError(
            f"start student sha256 {start_sha256} != {FORMAL_START_SHA256}")
    if teacher_sha256 != base.FORMAL_TEACHER_SHA256:
        raise RuntimeError(
            f"teacher archive sha256 {teacher_sha256} "
            f"!= {base.FORMAL_TEACHER_SHA256}")
    torch.set_float32_matmul_precision("high")
    data = prepare_fixed_data(args, device)
    cache_loader = DataLoader(
        data["train_dataset"], batch_size=FORMAL_EVAL_BATCH_SIZE,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    teacher_cache = base.build_or_load_teacher_cache(
        args, cache_loader, data["normalizer"], data["train_keys"],
        data["split_manifest_sha256"], data["real_stats_sha256"],
        data["expected_teacher_shape"])
    summary_path = args.output_dir / "summary.json"
    summary = {
        "method": "full_backbone_point_distillation_stage_true_loao",
        "status": "RUNNING",
        "evidence_class": (
            "fixed2576_training_exclusion_fixed725_held_aoa_terminal_delta"),
        "start_student_sha256": start_sha256,
        "teacher_archive_sha256": teacher_sha256,
        "teacher_cache": str(args.teacher_cache.resolve()),
        "teacher_cache_sha256": base.sha256(args.teacher_cache),
        "split_manifest_sha256": data["split_manifest_sha256"],
        "real_stats_sha256": data["real_stats_sha256"],
        "fixed_counts": dict(base.FORMAL_COUNTS),
        "held_aoa_order": list(HELD_AOAS),
        "folds": [],
        "training_contract": {
            "updates_per_fold": FORMAL_UPDATES,
            "batch_size": FORMAL_BATCH_SIZE,
            "eval_batch_size": FORMAL_EVAL_BATCH_SIZE,
            "encoder_lr": FORMAL_ENCODER_LR,
            "decoder_lr": FORMAL_DECODER_LR,
            "point_head_lr": FORMAL_POINT_HEAD_LR,
            "weight_decay": FORMAL_WEIGHT_DECAY,
            "grad_clip": FORMAL_GRAD_CLIP,
            "seed_reset_each_fold": FORMAL_SEED,
            "fold_time_budget_seconds": FORMAL_FOLD_TIME_BUDGET_SECONDS,
            "held_evaluations": ["update0", "terminal_update800"],
            "intermediate_held_evaluation": False,
            "checkpoint_saved": False,
        },
    }
    write_summary(summary_path, summary)
    for held_aoa in HELD_AOAS:
        fold_row = train_one_fold(
            args, data, teacher_cache, held_aoa, device)
        summary["folds"].append(fold_row)
        write_summary(summary_path, summary)
        if fold_row["completed_updates"] != FORMAL_UPDATES:
            summary["status"] = "STOP_INCOMPLETE_FOLD"
            summary["gate"] = loao_gate(summary["folds"])
            write_summary(summary_path, summary)
            break
    else:
        gate = loao_gate(summary["folds"])
        summary["gate"] = gate
        summary["status"] = (
            "PASS_POINT_DISTILLATION_STAGE_LOAO"
            if gate["pass"] else "STOP_POINT_DISTILLATION_STAGE_LOAO")
        write_summary(summary_path, summary)
    print(json.dumps({
        "event": "loao_complete", "status": summary["status"],
        "gate": summary.get("gate"), "summary": str(summary_path)},
        ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
