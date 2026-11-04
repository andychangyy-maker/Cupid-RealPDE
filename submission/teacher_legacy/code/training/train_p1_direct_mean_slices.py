#!/usr/bin/env python3
"""Distill only the temporal-mean u/v rows of the C128 point heads.

This is a held-fold mechanism pilot.  The high-precision H96 checkpoint is a
nearby source for the fast Direct architecture, not an exact FP32 recovery of
the historical Direct ZIP.  Update-0 metrics from this exact checkpoint are
therefore the only baseline used by the scientific gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import train_full_backbone_point_distillation as base
import train_s3a_point_pilot as s3a


FORMAL_START_SHA256 = (
    "71fd8daad9893adddc1e8b0d46ce9972455700f7c82296e74adacf035705daef")
FORMAL_TEACHER_ARCHIVE_SHA256 = (
    "0dd0f66c8cc18b2ce83f2af05f64e1821b9df9fe0597d762eb2bfe05f42ff3a5")
FORMAL_FOLD_MANIFEST_SHA256 = (
    "8a4515a7cfd766d19a502aa3b45dc7a75340e43d80b28c8c6e1607ef119eb73f")
FORMAL_SCORER_SHA256 = (
    "a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39")
FORMAL_COUNTS = dict(s3a.FORMAL_COUNTS)
FORMAL_HELD_FOLD = 4
FORMAL_GATE = {
    "point_delta_min": 0.040,
    "oracle_gap_recovery_min": 0.70,
    "rel_l2_delta_min": 0.0,
    "mvpe_delta_min": 0.0,
    "tke_abs_drift_max": 1e-6,
    "positive_aoa_groups_min": 4,
    "fluctuation_drift_max": 1e-7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-student", type=Path, required=True)
    parser.add_argument("--teacher-archive", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-teacher-cache-sha256")
    parser.add_argument("--expected-script-sha256")
    parser.add_argument(
        "--oracle-point-score", type=float,
        help="Pre-registered held-fold P1 temporal-mean oracle point score")
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
        args.point_head_lr, args.time_budget_seconds, args.window_stride)
    if any(value <= 0 for value in positive):
        raise ValueError("counts, learning rates, and budgets must be positive")
    if args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("weight decay and grad clip must be non-negative")
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("val_fraction must lie in (0, 0.5)")
    if args.held_fold != FORMAL_HELD_FOLD:
        raise ValueError(f"the registered held fold is {FORMAL_HELD_FOLD}")
    if args.smoke and (args.updates != 2
                       or args.time_budget_seconds != 300.0):
        raise ValueError("the registered smoke is 2 updates / 300 seconds")
    if args.formal and (args.updates != 800
                        or args.time_budget_seconds != 600.0):
        raise ValueError("the registered formal is 800 updates / 600 seconds")
    if args.formal and not args.expected_teacher_cache_sha256:
        raise ValueError("formal mode requires --expected-teacher-cache-sha256")
    if args.formal and not args.expected_script_sha256:
        raise ValueError("formal mode requires --expected-script-sha256")
    if args.formal and args.oracle_point_score is None:
        raise ValueError("formal mode requires --oracle-point-score")


def _tensor_bytes(value: torch.Tensor) -> bytes:
    cpu = value.detach().contiguous().cpu()
    return cpu.reshape(-1).view(torch.uint8).numpy().tobytes()


class MeanSliceGuard:
    """Make only rows 0:2 trainable and restore all other rows after a step."""

    def __init__(self, model: torch.nn.Module):
        if not bool(model.config.mean_fluctuation):
            raise ValueError("mean-slice training requires mean_fluctuation=True")
        if int(model.config.out_channels) < 3:
            raise ValueError("expected u/v/pressure point-head outputs")
        model.requires_grad_(False)
        self.parameters: list[torch.nn.Parameter] = []
        self.parameter_names: dict[str, torch.nn.Parameter] = {}
        self.frozen_rows: dict[str, torch.Tensor] = {}
        self.initial_mean_rows: dict[str, torch.Tensor] = {}
        self._hooks = []
        for member_index, heads in enumerate(model.member_point_heads):
            for scale_index, head in enumerate(heads):
                for field in ("weight", "bias"):
                    name = (
                        f"member_point_heads.{member_index}.{scale_index}."
                        f"{field}")
                    parameter = getattr(head, field)
                    if parameter is None or parameter.shape[0] <= 2:
                        raise ValueError(f"invalid point-head parameter {name}")
                    parameter.requires_grad_(True)
                    self.parameters.append(parameter)
                    self.parameter_names[name] = parameter
                    self.frozen_rows[name] = parameter[2:].detach().clone()
                    self.initial_mean_rows[name] = parameter[:2].detach().clone()
                    self._hooks.append(parameter.register_hook(
                        self._mean_row_gradient))
        if not self.parameters:
            raise ValueError("model has no member point heads")

    @staticmethod
    def _mean_row_gradient(gradient: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(gradient)
        masked[:2].copy_(gradient[:2])
        return masked

    @torch.no_grad()
    def restore_frozen_rows(self) -> None:
        for name, parameter in self.parameter_names.items():
            parameter[2:].copy_(self.frozen_rows[name])

    def frozen_rows_bitwise_equal(self) -> bool:
        return all(torch.equal(
            self.parameter_names[name][2:].detach(), expected)
                   for name, expected in self.frozen_rows.items())

    def frozen_rows_max_abs_drift(self) -> float:
        return max(float((
            self.parameter_names[name][2:].detach() - expected
        ).abs().max()) for name, expected in self.frozen_rows.items())

    def capture_mean_rows(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter[:2].detach().clone()
            for name, parameter in self.parameter_names.items()}

    @torch.no_grad()
    def restore_mean_rows(self, rows: dict[str, torch.Tensor]) -> None:
        if rows.keys() != self.parameter_names.keys():
            raise ValueError("mean-row snapshot does not match point heads")
        for name, parameter in self.parameter_names.items():
            parameter[:2].copy_(rows[name])

    def immutable_state_sha256(self, model: torch.nn.Module) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            immutable = value[2:] if name in self.parameter_names else value
            digest.update(_tensor_bytes(immutable))
        return digest.hexdigest()


def load_student(path: Path, normalizer, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload.get("model_config", {}))
    required = {
        "architecture": "ensemble_unet",
        "base": 128,
        "uq_hidden_channels": 96,
        "ensemble_members": 4,
        "mean_fluctuation": True,
    }
    errors = [
        f"{key}={config.get(key)!r}, expected {expected!r}"
        for key, expected in required.items()
        if config.get(key) != expected]
    if errors:
        raise ValueError("invalid high-precision neighbor start: " + "; ".join(errors))
    model = base.construct_model("ensemble_unet", config)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("start checkpoint has no model_state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device)
    base.common.set_model_stats(model, normalizer)
    return model, payload, config


def mean_slice_loss(student, teacher_uv, truth):
    student_mean = student[..., :2].mean(dim=1, keepdim=True)
    teacher_mean = teacher_uv.mean(dim=1, keepdim=True)
    truth_mean = truth[..., :2].mean(dim=1, keepdim=True)
    mean_loss = F.smooth_l1_loss(
        student_mean / 0.025, teacher_mean / 0.025)
    profile_loss = base.profile_distill_loss(
        student_mean, teacher_mean, 0.025)
    truth_mean_loss = F.smooth_l1_loss(
        student_mean / 0.025, truth_mean / 0.025)
    loss = 2.0 * mean_loss + 2.0 * profile_loss + 0.10 * truth_mean_loss
    return loss, {
        "teacher_temporal_mean": mean_loss.detach(),
        "teacher_mean_profile": profile_loss.detach(),
        "truth_temporal_mean": truth_mean_loss.detach(),
    }


def metric_deltas(initial: dict[str, Any], final: dict[str, Any]):
    return {
        key: float(final[key]) - float(initial[key])
        for key in (
            "rel_l2_score", "tke_rel_l2_score",
            "mvpe_rel_l2_score", "point_score")}


def aoa_slice_gate(initial: dict[str, Any], final: dict[str, Any]):
    groups = sorted(set(initial["by_aoa"]) & set(final["by_aoa"]))
    rows = {}
    positive = 0
    for group in groups:
        delta = float(final["by_aoa"][group]["point_score"]) - float(
            initial["by_aoa"][group]["point_score"])
        rows[group] = {"point_delta": delta, "nonnegative": delta >= 0.0}
        positive += int(delta >= 0.0)
    passed = len(groups) == 5 and positive >= int(
        FORMAL_GATE["positive_aoa_groups_min"])
    return {
        "groups": rows,
        "positive_point_groups": positive,
        "required_positive_groups": int(
            FORMAL_GATE["positive_aoa_groups_min"]),
        "passed": bool(passed),
    }


def decide_gate(initial: dict[str, Any], final: dict[str, Any],
                oracle_point_score: float | None,
                fluctuation_drift: float,
                immutable_state_equal: bool,
                frozen_rows_equal: bool):
    deltas = metric_deltas(initial, final)
    initial_point = float(initial["point_score"])
    oracle_gap = (
        float(oracle_point_score) - initial_point
        if oracle_point_score is not None else None)
    recovery = (
        deltas["point_score"] / oracle_gap
        if oracle_gap is not None and oracle_gap > 0.0 else None)
    aoa = aoa_slice_gate(initial, final)
    checks = {
        "point_delta_pass": deltas["point_score"] >= FORMAL_GATE["point_delta_min"],
        "oracle_gap_positive": oracle_gap is not None and oracle_gap > 0.0,
        "oracle_gap_recovery_pass": (
            recovery is not None
            and recovery >= FORMAL_GATE["oracle_gap_recovery_min"]),
        "rel_l2_nonnegative_pass": (
            deltas["rel_l2_score"] >= FORMAL_GATE["rel_l2_delta_min"]),
        "mvpe_nonnegative_pass": (
            deltas["mvpe_rel_l2_score"] >= FORMAL_GATE["mvpe_delta_min"]),
        "tke_drift_pass": (
            abs(deltas["tke_rel_l2_score"])
            <= FORMAL_GATE["tke_abs_drift_max"]),
        "aoa_slice_pass": aoa["passed"],
        "fluctuation_drift_pass": (
            fluctuation_drift <= FORMAL_GATE["fluctuation_drift_max"]),
        "immutable_state_pass": immutable_state_equal,
        "frozen_rows_pass": frozen_rows_equal,
    }
    return {
        "passed": bool(all(checks.values())),
        **checks,
        "oracle_point_score": oracle_point_score,
        "oracle_gap": oracle_gap,
        "oracle_gap_recovery": recovery,
        "aoa_slices": aoa,
        "deltas": deltas,
    }


@torch.inference_mode()
def audit_prediction_fluctuation(
        model, loader, normalizer, device: torch.device,
        guard: MeanSliceGuard) -> dict[str, Any]:
    """Measure mean-free u/v change between update-0 and final mean rows."""
    model.eval()
    final_rows = guard.capture_mean_rows()
    maximum = 0.0
    samples = 0
    try:
        for history, target in loader:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            history_norm, _ = normalizer.preprocess(history, target)
            guard.restore_mean_rows(final_rows)
            final_prediction = model(history_norm)[..., :2]
            guard.restore_mean_rows(guard.initial_mean_rows)
            initial_prediction = model(history_norm)[..., :2]
            change = final_prediction.double() - initial_prediction.double()
            mean_free = change - change.mean(dim=1, keepdim=True)
            maximum = max(maximum, float(mean_free.abs().max()))
            samples += int(history.shape[0])
    finally:
        guard.restore_mean_rows(final_rows)
    return {
        "definition": (
            "max abs temporal-mean-free change in normalized u/v prediction"),
        "samples": samples,
        "max_abs_drift": maximum,
    }


def formal_contract_errors(
        args: argparse.Namespace, data: dict[str, Any],
        input_hashes: dict[str, str], initial: dict[str, Any]) -> list[str]:
    if not args.formal:
        return []
    errors = []
    expected_hashes = {
        "start_student_sha256": FORMAL_START_SHA256,
        "teacher_archive_sha256": FORMAL_TEACHER_ARCHIVE_SHA256,
        "fold_manifest_sha256": FORMAL_FOLD_MANIFEST_SHA256,
        "real_stats_sha256": base.FORMAL_REAL_STATS_SHA256,
        "split_manifest_canonical_sha256": base.FORMAL_SPLIT_MANIFEST_SHA256,
        "official_scorer_sha256": FORMAL_SCORER_SHA256,
    }
    for key, expected in expected_hashes.items():
        if input_hashes[key].lower() != expected.lower():
            errors.append(f"{key}={input_hashes[key]}, expected {expected}")
    for key, expected in FORMAL_COUNTS.items():
        if data[key] != expected:
            errors.append(f"{key}={data[key]}, expected {expected}")
    expected_args = {
        "updates": 800,
        "batch_size": 8,
        "eval_batch_size": 16,
        "point_head_lr": 3e-5,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "time_budget_seconds": 600.0,
        "seed": 0,
        "num_workers": 0,
        "max_files": 0,
        "window_stride": 20,
        "val_fraction": 0.2,
        "held_fold": FORMAL_HELD_FOLD,
    }
    for key, expected in expected_args.items():
        actual = getattr(args, key)
        matches = (
            bool(np.isclose(actual, expected, rtol=0.0, atol=1e-12))
            if isinstance(expected, float) else actual == expected)
        if not matches:
            errors.append(f"{key}={actual!r}, expected {expected!r}")
    cache_sha = input_hashes.get("teacher_cache_sha256")
    if cache_sha is None:
        errors.append("formal run requires a pre-existing teacher cache")
    elif cache_sha.lower() != args.expected_teacher_cache_sha256.lower():
        errors.append("teacher cache SHA does not match the registered value")
    if (input_hashes["script_sha256"].lower()
            != args.expected_script_sha256.lower()):
        errors.append("script SHA does not match the registered value")
    if (args.oracle_point_score is None
            or not np.isfinite(args.oracle_point_score)
            or args.oracle_point_score <= float(initial["point_score"])):
        errors.append(
            "oracle point score must be finite and above actual update-0 score")
    return errors


def save_checkpoint(path: Path, model, source_payload: dict[str, Any],
                    model_config: dict[str, Any], args: argparse.Namespace,
                    iteration: int, metrics: dict[str, Any], status: str) -> None:
    payload = {
        "phase": "real",
        "iteration": iteration,
        "model_config": model_config,
        "normalizer": source_payload.get("normalizer", {}),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()},
        "metrics": metrics,
        "p1_direct_mean_slices": {
            "status": status,
            "held_fold": args.held_fold,
            "start_kind": "high_precision_neighbor_start",
            "trained_parameters": (
                "rows 0:2 of every member/scale point-head weight and bias"),
            "warning": "native UQ is invalid after point-head mean changes",
            "uq_state_valid": False,
        },
    }
    if "point_head_distillation" in source_payload:
        payload["point_head_distillation"] = source_payload[
            "point_head_distillation"]
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("mean-slice distillation requires CUDA")
    base.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.time()

    data = s3a.prepare_data(args, device)
    scorer_path = (
        base.REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py")
    input_hashes = {
        "start_student_sha256": base.sha256(args.start_student),
        "teacher_archive_sha256": base.sha256(args.teacher_archive),
        "teacher_cache_sha256": (
            base.sha256(args.teacher_cache)
            if args.teacher_cache.is_file() else None),
        "fold_manifest_sha256": data["fold_manifest_sha256"],
        "real_stats_sha256": data["real_stats_sha256"],
        "split_manifest_canonical_sha256": data["split_manifest_sha256"],
        "official_scorer_sha256": base.sha256(scorer_path),
        "script_sha256": base.sha256(Path(__file__)),
    }
    model, source_payload, model_config = load_student(
        args.start_student, data["normalizer"], device)
    guard = MeanSliceGuard(model)
    immutable_sha_before = guard.immutable_state_sha256(model)
    initial = base.evaluate_point(
        model, data["eval_loader"], data["normalizer"],
        device, data["eval_aoa"])
    print(json.dumps({"event": "initial", **initial}), flush=True)

    errors = formal_contract_errors(args, data, input_hashes, initial)
    if errors:
        rejection = {
            "status": "REJECT_MEAN_SLICE_CONTRACT",
            "errors": errors,
            "initial_metrics": initial,
            "input_hashes": input_hashes,
        }
        (args.output_dir / "contract_rejection.json").write_text(
            json.dumps(rejection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        raise RuntimeError(json.dumps(rejection, ensure_ascii=False))

    teacher_cache = base.build_or_load_teacher_cache(
        args, data["cache_loader"], data["normalizer"],
        data["train_keys"], data["split_manifest_sha256"],
        data["real_stats_sha256"], data["expected_teacher_shape"])
    input_hashes["teacher_cache_sha256"] = base.sha256(args.teacher_cache)
    if (args.expected_teacher_cache_sha256
            and input_hashes["teacher_cache_sha256"].lower()
            != args.expected_teacher_cache_sha256.lower()):
        raise RuntimeError("teacher cache changed after the contract check")

    optimizer = torch.optim.AdamW(
        guard.parameters, lr=args.point_head_lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
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
        model.eval()
        indices, history, truth = next(iterator)
        history = history.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        teacher_uv = teacher_cache[indices].to(
            device=device, dtype=torch.float32, non_blocking=True)
        history_norm, truth_norm = data["normalizer"].preprocess(
            history, truth)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(history_norm)
        loss, components = mean_slice_loss(
            prediction.float(), teacher_uv, truth_norm)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at iteration {iteration}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(guard.parameters, args.grad_clip)
        optimizer.step()
        guard.restore_frozen_rows()
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
    fluctuation_audit = audit_prediction_fluctuation(
        model, data["eval_loader"], data["normalizer"], device, guard)
    immutable_sha_after = guard.immutable_state_sha256(model)
    frozen_equal = guard.frozen_rows_bitwise_equal()
    gate = decide_gate(
        initial, final, args.oracle_point_score,
        fluctuation_audit["max_abs_drift"],
        immutable_sha_before == immutable_sha_after, frozen_equal)
    scientific_gate_evaluated = bool(
        args.formal and completed_iteration == 800
        and status == "RUNNING")
    if status == "RUNNING":
        if args.smoke:
            status = "SMOKE_COMPLETE_NO_SCIENTIFIC_GATE"
        elif completed_iteration != 800:
            status = "STOP_INCOMPLETE"
        elif gate["passed"]:
            status = "PASS_MEAN_SLICE_AUTHORIZE_UQ"
        else:
            status = "STOP_MEAN_SLICE_GATE"

    final_path = args.output_dir / f"final_{completed_iteration:05d}.pth"
    save_checkpoint(
        final_path, model, source_payload, model_config,
        args, completed_iteration, final, status)
    summary = {
        "schema_version": 1,
        "experiment": "track1_p1_direct_mean_slice_held_fold4",
        "evidence_class": "single_pre_registered_held_fold_mechanism_pilot",
        "status": status,
        "scientific_gate_evaluated": scientific_gate_evaluated,
        "start_kind": "high_precision_neighbor_start",
        "start_warning": (
            "SHA71fd is not an exact FP32 recovery of historical fast Direct"),
        "held_fold": args.held_fold,
        "counts": {key: data[key] for key in FORMAL_COUNTS},
        "completed_iteration": completed_iteration,
        "initial_metrics": initial,
        "final_metrics": final,
        "deltas": metric_deltas(initial, final),
        "gate_thresholds": FORMAL_GATE,
        "gate": gate,
        "fluctuation_audit": fluctuation_audit,
        "parameter_audit": {
            "trainable_scope": (
                "rows 0:2 of all member/scale point-head weights and biases"),
            "trainable_tensor_count": len(guard.parameters),
            "frozen_rows_bitwise_equal": frozen_equal,
            "frozen_rows_max_abs_drift": guard.frozen_rows_max_abs_drift(),
            "immutable_state_sha256_before": immutable_sha_before,
            "immutable_state_sha256_after": immutable_sha_after,
            "immutable_state_bitwise_equal": (
                immutable_sha_before == immutable_sha_after),
        },
        "input_hashes": input_hashes,
        "loss_weights": {
            "teacher_temporal_mean": 2.0,
            "teacher_mean_profile": 2.0,
            "truth_temporal_mean": 0.10,
        },
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
        "train_component_means": {
            key: value / max(count, 1) for key, value in sums.items()},
        "training_wall_seconds": training_wall_seconds,
        "total_wall_seconds": time.time() - total_started,
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)),
        "peak_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)),
        "final_checkpoint": str(final_path),
        "limitations": [
            "The start model and P1 teacher have seen all conditions.",
            "The gate is held fold 4, not complete five-fold or LOAO evidence.",
            "Only the 2032 non-held-fold point windows are used for updates.",
            "All 725 tail windows, including fold-4 tail, remain unused.",
            "Native UQ must be retrained after this point-head change.",
            "No CodaBench submission is authorized by this run alone.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "status": status,
        "completed_iteration": completed_iteration,
        "gate_passed": gate["passed"],
        "summary": str(summary_path),
    }), flush=True)


if __name__ == "__main__":
    main()
