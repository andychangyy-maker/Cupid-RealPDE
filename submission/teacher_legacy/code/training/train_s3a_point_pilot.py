#!/usr/bin/env python3
"""Run the bounded S3a held-fold C128 point-distillation pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import train_full_backbone_point_distillation as base


FORMAL_INPUTS = {
    "start_student_sha256": (
        "f5d9f10ad9c764cc3f181a4f0b788b53ed15ba436f1d190bddda7152dbedd28b"),
    "teacher_archive_sha256": (
        "0dd0f66c8cc18b2ce83f2af05f64e1821b9df9fe0597d762eb2bfe05f42ff3a5"),
    "fold_manifest_sha256": (
        "8a4515a7cfd766d19a502aa3b45dc7a75340e43d80b28c8c6e1607ef119eb73f"),
    "real_stats_sha256": base.FORMAL_REAL_STATS_SHA256,
    "split_manifest_canonical_sha256": base.FORMAL_SPLIT_MANIFEST_SHA256,
}
FORMAL_COUNTS = {
    "trajectories": 82,
    "train_windows": 2032,
    "eval_windows": 544,
    "tail_windows_unused": 725,
    "gate_fold_tail_windows_unused": 153,
}
FORMAL_HELD_FOLD = 4
FORMAL_STRUCTURE = {
    "mean_profile_channels": 64,
    "multi_scale_regime_film": False,
}
FORMAL_GATE = {
    "point_delta_min": 0.010,
    "tke_delta_min": -0.050,
    "rel_or_mvpe_delta_min": 0.010,
}
FORMAL_SCORER_SHA256 = (
    "a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39")
BASELINE_TOLERANCE = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-student", type=Path, required=True)
    parser.add_argument("--teacher-archive", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-initial-summary", type=Path)
    parser.add_argument("--expected-teacher-cache-sha256")
    parser.add_argument("--expected-code-revision")
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--held-fold", type=int, default=FORMAL_HELD_FOLD)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, required=True)
    parser.add_argument("--encoder-lr", type=float, default=2e-6)
    parser.add_argument("--decoder-lr", type=float, default=1e-5)
    parser.add_argument("--point-head-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--time-budget-seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.updates, args.batch_size, args.eval_batch_size,
        args.eval_interval, args.encoder_lr, args.decoder_lr,
        args.point_head_lr, args.time_budget_seconds, args.window_stride,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("counts, learning rates, and budgets must be positive")
    if args.updates % args.eval_interval:
        raise ValueError("updates must be divisible by eval_interval")
    if args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("weight decay and grad clip must be non-negative")
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("val_fraction must lie in (0, 0.5)")
    if args.smoke and (args.updates != 20 or args.eval_interval != 20):
        raise ValueError("the registered smoke is exactly 20 updates")
    if args.formal and (args.updates != 400 or args.eval_interval != 400):
        raise ValueError("the registered formal pilot is exactly 400 updates")
    if args.formal and args.time_budget_seconds != 1200.0:
        raise ValueError("the registered formal training budget is 1200 seconds")
    if args.smoke and args.time_budget_seconds != 300.0:
        raise ValueError("the registered smoke training budget is 300 seconds")
    if args.formal and args.expected_initial_summary is None:
        raise ValueError("formal mode requires --expected-initial-summary")
    if args.held_fold != FORMAL_HELD_FOLD:
        raise ValueError(f"the registered gate fold is {FORMAL_HELD_FOLD}")
    if args.formal and not args.expected_teacher_cache_sha256:
        raise ValueError("formal mode requires --expected-teacher-cache-sha256")
    if args.formal and not args.expected_code_revision:
        raise ValueError("formal mode requires --expected-code-revision")


def current_code_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=base.REPO_ROOT,
        text=True).strip()


def load_fold_assignments(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = payload.get("fold_manifest", payload)
    rows = manifest.get("assignments") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("fold manifest has no assignments")
    result: dict[str, int] = {}
    for row in rows:
        filename = str(row["file"])
        fold = int(row["fold"])
        if filename in result:
            raise ValueError(f"duplicate fold assignment for {filename}")
        if fold not in range(5):
            raise ValueError(f"invalid fold {fold} for {filename}")
        result[filename] = fold
    return result


def select_held_fold_windows(
        metadata: list[dict[str, Any]], train_windows, val_windows,
        assignments: dict[str, int], held_fold: int):
    files = [str(item["file"]) for item in metadata]
    missing = sorted(set(files) - set(assignments))
    extra = sorted(set(assignments) - set(files))
    if missing or extra:
        raise ValueError(
            f"fold assignments do not match trajectories: missing={missing}, "
            f"extra={extra}")

    def fold_of(window) -> int:
        return assignments[files[window.trajectory]]

    selected_train = [window for window in train_windows
                      if fold_of(window) != held_fold]
    held_train = [window for window in train_windows
                  if fold_of(window) == held_fold]
    selected_val = [window for window in val_windows
                    if fold_of(window) == held_fold]
    return selected_train, held_train, selected_val


def prepare_data(args: argparse.Namespace, device: torch.device):
    trajectories, metadata = base.common.repro.load_trajectories(
        args.real_dir, args.max_files)
    all_train, all_val, split_manifest = base.common.repro.make_split(
        metadata, args.window_stride, args.val_fraction)
    assignments = load_fold_assignments(args.fold_manifest)
    train_windows, held_train, gate_fold_tail = select_held_fold_windows(
        metadata, all_train, all_val, assignments, args.held_fold)
    train_base = base.common.repro.WindowDataset(trajectories, train_windows)
    eval_dataset = base.common.repro.WindowDataset(trajectories, held_train)
    train_loader = DataLoader(
        base.IndexedDataset(train_base), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    cache_loader = DataLoader(
        train_base, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    normalizer = base.common.repro.Normalizer(args.real_stats, device)
    train_keys = [
        (str(metadata[window.trajectory]["file"]), int(window.start))
        for window in train_windows]
    eval_aoa = np.asarray([
        float(metadata[window.trajectory]["aoa"])
        for window in held_train], dtype=np.float32)
    example_target = train_base[0][1]
    expected_teacher_shape = (
        len(train_windows), *example_target.shape[:-1], 2)
    return {
        "train_loader": train_loader,
        "cache_loader": cache_loader,
        "eval_loader": eval_loader,
        "normalizer": normalizer,
        "train_keys": train_keys,
        "eval_aoa": eval_aoa,
        "trajectories": len(trajectories),
        "train_windows": len(train_windows),
        "eval_windows": len(held_train),
        "tail_windows_unused": len(all_val),
        "gate_fold_tail_windows_unused": len(gate_fold_tail),
        "split_manifest_sha256": base.canonical_sha256(split_manifest),
        "real_stats_sha256": base.sha256(args.real_stats),
        "fold_manifest_sha256": base.sha256(args.fold_manifest),
        "expected_teacher_shape": expected_teacher_shape,
    }


def target_model_config(source: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    if config.get("architecture") != "ensemble_unet":
        raise ValueError("S3a requires an ensemble_unet start checkpoint")
    if int(config.get("base", -1)) != 128:
        raise ValueError("S3a requires the C128 start checkpoint")
    if int(config.get("mean_profile_channels", 0)) != 0:
        raise ValueError("S3a requires a start checkpoint without a profile branch")
    if bool(config.get("multi_scale_regime_film", False)):
        raise ValueError("S3a changes only the profile branch, not decoder FiLM")
    config.update(FORMAL_STRUCTURE)
    return config


def load_upgraded_student(path: Path, normalizer, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_config = dict(payload.get("model_config", {}))
    config = target_model_config(source_config)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("start checkpoint has no model_state_dict")

    reference = base.construct_model("ensemble_unet", source_config)
    reference.load_state_dict(state, strict=True)
    model = base.construct_model("ensemble_unet", config)
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {
        key for key in model.state_dict()
        if key.startswith("mean_profile_refiner.")
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise ValueError(
            "unexpected S3a missing keys: "
            f"{sorted(set(incompatible.missing_keys) ^ expected_missing)}")
    if incompatible.unexpected_keys:
        raise ValueError(
            f"unexpected S3a checkpoint keys: {incompatible.unexpected_keys}")
    final_layer = model.mean_profile_refiner[-1]
    if (torch.count_nonzero(final_layer.weight).item() != 0
            or torch.count_nonzero(final_layer.bias).item() != 0):
        raise ValueError("mean-profile branch is not zero initialized")
    reference.to(device)
    model.to(device)
    base.common.set_model_stats(reference, normalizer)
    base.common.set_model_stats(model, normalizer)
    return model, reference, payload, config, sorted(expected_missing)


@torch.inference_mode()
def audit_initial_identity(model, reference, loader, normalizer,
                           device: torch.device) -> dict[str, Any]:
    model.eval()
    reference.eval()
    history, target = next(iter(loader))
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    history_norm, _ = normalizer.preprocess(history, target)
    expected = reference(history_norm)
    actual = model(history_norm)
    return {
        "batch": int(history.shape[0]),
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs_diff": float((actual - expected).abs().max()),
    }


def distill_loss(student, teacher_uv, truth, normalizer):
    student_uv = student[..., :2]
    student_mean = student_uv.mean(dim=1, keepdim=True)
    teacher_mean = teacher_uv.mean(dim=1, keepdim=True)
    student_fluctuation = student_uv - student_mean
    teacher_fluctuation = teacher_uv - teacher_mean
    mean_loss = F.smooth_l1_loss(student_mean / 0.025, teacher_mean / 0.025)
    fluctuation_loss = F.smooth_l1_loss(
        student_fluctuation / 0.020, teacher_fluctuation / 0.020)
    profile_loss = base.profile_distill_loss(
        student_mean, teacher_mean, 0.025)
    tke_loss = base.tke_map_distill_loss(
        student_uv, teacher_uv, normalizer)
    truth_loss = base.common.accuracy_objective(
        student, truth, normalizer, (5.0, 3.0, 2.0))[0]
    loss = (
        2.0 * mean_loss
        + fluctuation_loss
        + 2.0 * profile_loss
        + 0.05 * tke_loss
        + 0.10 * truth_loss)
    return loss, {
        "mean_distill": mean_loss.detach(),
        "fluctuation_distill": fluctuation_loss.detach(),
        "profile_distill": profile_loss.detach(),
        "tke_map_distill": tke_loss.detach(),
        "truth_accuracy": truth_loss.detach(),
    }


def metric_deltas(initial: dict[str, Any], final: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(final[key]) - float(initial[key])
        for key in (
            "rel_l2_score", "tke_rel_l2_score",
            "mvpe_rel_l2_score", "point_score")
    }


def decide_gate(deltas: dict[str, float]) -> dict[str, Any]:
    point = deltas["point_score"] >= FORMAL_GATE["point_delta_min"]
    tke = deltas["tke_rel_l2_score"] >= FORMAL_GATE["tke_delta_min"]
    rel_or_mvpe = max(
        deltas["rel_l2_score"], deltas["mvpe_rel_l2_score"]
    ) >= FORMAL_GATE["rel_or_mvpe_delta_min"]
    return {
        "passed": bool(point and tke and rel_or_mvpe),
        "point_delta_pass": bool(point),
        "tke_delta_pass": bool(tke),
        "rel_or_mvpe_delta_pass": bool(rel_or_mvpe),
    }


def expected_smoke_errors(
        initial: dict[str, Any], path: Path | None,
        data: dict[str, Any], actual_inputs: dict[str, str],
        teacher_cache_sha256: str | None, code_revision: str,
        scorer_sha256: str) -> list[str]:
    if path is None:
        return ["formal run has no expected smoke summary"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if payload.get("status") != "SMOKE_COMPLETE_NO_SCIENTIFIC_GATE":
        errors.append("expected summary is not a successful smoke")
    if payload.get("completed_iteration") != 20:
        errors.append("expected smoke did not complete exactly 20 updates")
    config = payload.get("training_config", {})
    expected_smoke_config = {
        "smoke": True,
        "formal": False,
        "updates": 20,
        "eval_interval": 20,
        "time_budget_seconds": 300.0,
        "held_fold": FORMAL_HELD_FOLD,
    }
    for key, expected_value in expected_smoke_config.items():
        if config.get(key) != expected_value:
            errors.append(
                f"expected smoke {key}={config.get(key)!r}, "
                f"expected {expected_value!r}")
    expected_counts = {key: data[key] for key in FORMAL_COUNTS}
    if payload.get("counts") != expected_counts:
        errors.append("expected smoke count contract differs from formal run")
    if payload.get("input_hashes") != actual_inputs:
        errors.append("expected smoke input hashes differ from formal run")
    identity = payload.get("structure_change", {}).get(
        "initialization_identity", {})
    if (identity.get("bitwise_equal") is not True
            or float(identity.get("max_abs_diff", float("inf"))) != 0.0):
        errors.append("expected smoke did not prove zero-init identity")
    if payload.get("teacher_cache_sha256") != teacher_cache_sha256:
        errors.append("expected smoke teacher-cache SHA differs from formal run")
    if payload.get("code_revision") != code_revision:
        errors.append("expected smoke code revision differs from formal run")
    if payload.get("official_scorer_sha256") != scorer_sha256:
        errors.append("expected smoke scorer SHA differs from formal run")
    expected = payload.get("initial_metrics")
    if not isinstance(expected, dict):
        return [*errors, "expected smoke has no initial_metrics"]
    for key in (
            "rel_l2_score", "tke_rel_l2_score",
            "mvpe_rel_l2_score", "point_score"):
        difference = abs(float(initial[key]) - float(expected[key]))
        if difference > BASELINE_TOLERANCE:
            errors.append(
                f"initial {key} differs from smoke by {difference:.3g}")
    return errors


def contract_errors(args: argparse.Namespace, data: dict[str, Any],
                    initial: dict[str, Any], start_sha256: str,
                    teacher_sha256: str,
                    teacher_cache_sha256: str | None,
                    code_revision: str, scorer_sha256: str) -> list[str]:
    errors = []
    actual_inputs = {
        "start_student_sha256": start_sha256,
        "teacher_archive_sha256": teacher_sha256,
        "fold_manifest_sha256": data["fold_manifest_sha256"],
        "real_stats_sha256": data["real_stats_sha256"],
        "split_manifest_canonical_sha256": data["split_manifest_sha256"],
    }
    for key, expected in FORMAL_INPUTS.items():
        if actual_inputs[key].lower() != expected.lower():
            errors.append(f"{key}={actual_inputs[key]}, expected {expected}")
    for key, expected in FORMAL_COUNTS.items():
        if data[key] != expected:
            errors.append(f"{key}={data[key]}, expected {expected}")
    expected_args = {
        "held_fold": FORMAL_HELD_FOLD,
        "batch_size": 8,
        "eval_batch_size": 16,
        "encoder_lr": 2e-6,
        "decoder_lr": 1e-5,
        "point_head_lr": 3e-5,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "seed": 0,
        "num_workers": 0,
        "max_files": 0,
        "window_stride": 20,
        "val_fraction": 0.2,
    }
    for key, expected in expected_args.items():
        actual = getattr(args, key)
        if isinstance(expected, float):
            matches = bool(np.isclose(actual, expected, rtol=0.0, atol=1e-12))
        else:
            matches = actual == expected
        if not matches:
            errors.append(f"{key}={actual!r}, expected {expected!r}")
    if scorer_sha256 != FORMAL_SCORER_SHA256:
        errors.append(
            f"official scorer SHA {scorer_sha256} != {FORMAL_SCORER_SHA256}")
    if args.formal:
        if code_revision != args.expected_code_revision:
            errors.append(
                f"code revision {code_revision} != {args.expected_code_revision}")
        if teacher_cache_sha256 is None:
            errors.append("formal run requires a pre-existing teacher cache")
        elif (teacher_cache_sha256.lower()
              != args.expected_teacher_cache_sha256.lower()):
            errors.append(
                "teacher-cache SHA does not match the registered formal cache")
        errors.extend(expected_smoke_errors(
            initial, args.expected_initial_summary, data, actual_inputs,
            teacher_cache_sha256, code_revision, scorer_sha256))
    return errors


def save_checkpoint(path: Path, model, source_payload, model_config,
                    args, iteration: int, metrics: dict[str, Any], status: str):
    torch.save({
        "phase": "real",
        "iteration": iteration,
        "model_config": model_config,
        "normalizer": source_payload.get("normalizer", {}),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metrics": metrics,
        "s3a_point_pilot": {
            "status": status,
            "held_fold": args.held_fold,
            "teacher_archive_sha256": base.sha256(args.teacher_archive),
            "warning": "UQ state is invalid after point-backbone training",
            "uq_state_valid": False,
        },
    }, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("S3a requires CUDA")
    base.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    total_started = time.time()

    data = prepare_data(args, device)
    start_sha256 = base.sha256(args.start_student)
    teacher_sha256 = base.sha256(args.teacher_archive)
    teacher_cache_sha256 = (
        base.sha256(args.teacher_cache) if args.teacher_cache.is_file()
        else None)
    code_revision = current_code_revision()
    scorer_path = (
        base.REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py")
    scorer_sha256 = base.sha256(scorer_path)
    model, reference, source_payload, model_config, missing_keys = (
        load_upgraded_student(args.start_student, data["normalizer"], device))
    identity = audit_initial_identity(
        model, reference, data["eval_loader"], data["normalizer"], device)
    del reference
    torch.cuda.empty_cache()
    if not identity["bitwise_equal"]:
        raise RuntimeError(f"zero-init identity failed: {identity}")

    initial = base.evaluate_point(
        model, data["eval_loader"], data["normalizer"],
        device, data["eval_aoa"])
    print(json.dumps({"event": "initial", **initial}), flush=True)
    errors = contract_errors(
        args, data, initial, start_sha256, teacher_sha256,
        teacher_cache_sha256, code_revision, scorer_sha256)
    if errors:
        rejection = {
            "status": "REJECT_S3A_CONTRACT",
            "errors": errors,
            "initial_metrics": initial,
            "initialization_identity": identity,
        }
        (args.output_dir / "contract_rejection.json").write_text(
            json.dumps(rejection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        raise RuntimeError(json.dumps(rejection, ensure_ascii=False))

    teacher_cache = base.build_or_load_teacher_cache(
        args, data["cache_loader"], data["normalizer"],
        data["train_keys"], data["split_manifest_sha256"],
        data["real_stats_sha256"], data["expected_teacher_shape"])
    teacher_cache_sha256 = base.sha256(args.teacher_cache)
    if (args.formal and teacher_cache_sha256.lower()
            != args.expected_teacher_cache_sha256.lower()):
        raise RuntimeError("teacher cache changed after formal contract check")
    groups = base.optimizer_groups(model)
    for group, lr in zip(
            groups, (args.encoder_lr, args.decoder_lr, args.point_head_lr)):
        group["lr"] = lr
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=400)
    iterator = base.common.repro.cycling(data["train_loader"])
    sums: dict[str, float] = {}
    count = 0
    completed_iteration = 0
    status = "RUNNING"
    training_started = time.time()

    for iteration in range(1, args.updates + 1):
        if time.time() - training_started > args.time_budget_seconds:
            status = "STOP_TIME_BUDGET"
            break
        model.train()
        indices, history_batch, truth = next(iterator)
        history_batch = history_batch.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        teacher_uv = teacher_cache[indices].to(
            device=device, dtype=torch.float32, non_blocking=True)
        history_norm, truth_norm = data["normalizer"].preprocess(
            history_batch, truth)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(history_norm)
        loss, components = distill_loss(
            prediction.float(), teacher_uv, truth_norm, data["normalizer"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at iteration {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in groups for parameter in group["params"]],
                args.grad_clip)
        optimizer.step()
        scheduler.step()
        completed_iteration = iteration
        for key, value in {"loss": loss.detach(), **components}.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1

    training_wall_seconds = time.time() - training_started
    if (status == "RUNNING"
            and training_wall_seconds > args.time_budget_seconds):
        status = "STOP_TIME_BUDGET"
    final = base.evaluate_point(
        model, data["eval_loader"], data["normalizer"],
        device, data["eval_aoa"])
    deltas = metric_deltas(initial, final)
    gate = decide_gate(deltas)
    if status == "RUNNING":
        if args.smoke:
            status = "SMOKE_COMPLETE_NO_SCIENTIFIC_GATE"
        elif completed_iteration != 400:
            status = "STOP_INCOMPLETE"
        elif gate["passed"]:
            status = "PASS_S3A_AUTHORIZE_S3B"
        else:
            status = "STOP_S3A_GATE"

    final_path = args.output_dir / f"final_{completed_iteration:05d}.pth"
    save_checkpoint(
        final_path, model, source_payload, model_config,
        args, completed_iteration, final, status)
    summary = {
        "schema_version": 1,
        "experiment": "track1_s3a_c128_p1_held_fold_point_pilot",
        "evidence_class": "single_pre_registered_held_fold_stress_pilot",
        "status": status,
        "scientific_gate_evaluated": bool(
            args.formal and completed_iteration == 400
            and status in {"PASS_S3A_AUTHORIZE_S3B", "STOP_S3A_GATE"}),
        "gate": {**FORMAL_GATE, **gate},
        "initial_metrics": initial,
        "final_metrics": final,
        "deltas": deltas,
        "held_fold": args.held_fold,
        "counts": {key: data[key] for key in FORMAL_COUNTS},
        "completed_iteration": completed_iteration,
        "input_hashes": {
            "start_student_sha256": start_sha256,
            "teacher_archive_sha256": teacher_sha256,
            "fold_manifest_sha256": data["fold_manifest_sha256"],
            "real_stats_sha256": data["real_stats_sha256"],
            "split_manifest_canonical_sha256": data["split_manifest_sha256"],
        },
        "teacher_cache_sha256": teacher_cache_sha256,
        "code_revision": code_revision,
        "official_scorer_sha256": scorer_sha256,
        "structure_change": {
            "only": "zero_initialized_mean_profile_branch",
            **FORMAL_STRUCTURE,
            "missing_keys_initialized": missing_keys,
            "initialization_identity": identity,
        },
        "loss_weights": {
            "temporal_mean": 2.0,
            "zero_mean_fluctuation": 1.0,
            "mean_profile": 2.0,
            "teacher_tke_map": 0.05,
            "real_truth_5_3_2": 0.10,
        },
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
        "train_component_means": {
            key: value / max(count, 1) for key, value in sums.items()},
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "training_wall_seconds": training_wall_seconds,
        "total_wall_seconds": time.time() - total_started,
        "final_checkpoint": str(final_path),
        "limitations": [
            "The start model and P1 teacher were trained with all conditions.",
            "This stress pilot is not model-level leave-one-condition-out evidence.",
            "All 725 tail windows are unused by training and update-400 gating.",
            "The gate uses the held fold's 544 point-train windows.",
            "No UQ training or CodaBench submission is authorized by this run alone.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "status": status,
        "completed_iteration": completed_iteration,
        "gate_passed": gate["passed"],
        "summary": str(summary_path),
    }), flush=True)


if __name__ == "__main__":
    main()
