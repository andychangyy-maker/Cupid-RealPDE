#!/usr/bin/env python3
"""Run the exact standard C208 Real phase after a promoted SHG1b Sim start.

The SHG1b checkpoint is admitted only with its promoted Sim summary and both
artifact hashes.  Real optimization is the frozen standard baseline path.
The fixed 725 tail is inaccessible to the optimizer and is evaluated exactly
once, at update 4000.  A failed terminal gate saves no checkpoint; a passing
gate saves only ``model_04000.pth``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

import train_c208_real_sim_rehearsal as rehearsal


METHOD = "c208_shg1b_standard_real_unique_gate"
SHG_METHOD = "c208_shg1b_sim_hgrad_pretrain"
SHG_PREFLIGHT_STATUS = "GO_SHG1B_TO_SIM10000"
SHG_PROMOTE_STATUS = "PROMOTE_SHG1B_SIM_MODEL_10000"
SHG_TERMINAL_UPDATE = 10_000
SHG_HGRAD_WEIGHT = 0.020
SHG_HGRAD_RAMP_UPDATES = 200
SHG_CALIBRATION_SOURCE_SUMMARY_SHA256 = (
    "04fddf7fbf35820bc8e4a83aebd6483162119f9e8b26071ecf75cce6fbf3a7d3")
SHG_BASELINE_SUMMARY_SHA256 = (
    "9b57d155d914bbd3e152dbb5d82c7d1fa1b1d0b654c55cab41bb9df85e4296ef")
SHG_TEACHER_CACHE_SHA256 = (
    "70cdc0d0d50c273bcd0abeaa706a21d4ef18f4709abcad62325f2c7111e1592a")
SHG_ONLY_SCIENTIFIC_DELTA = (
    "physical_uv_wh_forward_difference_rms_hgrad_"
    "through_complete_aggregate_point_graph")
SHG_FROZEN_SIM_LOSS_KEYS = (
    "batch_size", "lr", "weight_decay", "grad_clip", "seed",
    "max_files", "window_stride", "input_noise_std", "point_loss_mode",
    "mse_weight", "mse_warmup_updates", "mean_loss_weight",
    "member_loss_weight", "member_loss_mode", "bootstrap_probability",
    "quantile_loss_weight", "soft_sps_loss_weight",
    "soft_sps_tau_fraction", "soft_sps_warmup_updates",
    "soft_sps_uq_only", "soft_sps_proxy_real_scale",
    "coverage_bce_weight", "profile_loss_weight", "accuracy_weights",
    "distill_weight", "scheduler_t_max")

TERMINAL_UPDATE = 4_000
GATE_POINT_DELTA_MIN = 0.10
GATE_COMPONENT_DELTA_FLOOR = -0.02
GATE_SPS_DELTA_MIN = 0.70
GATE_NON_TIME_SHADOW_MIN = 0.30
NATIVE_STEP4000_METRICS = {
    "rel_l2_score": 95.86352389426571,
    "tke_rel_l2_score": 80.95221179698406,
    "mvpe_rel_l2_score": 96.51432542559023,
    "point_score": 91.52029057134612,
    "native_sps_score": 53.07979215521265,
}
COMPONENT_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")

# This is an executable assertion about the imported standard loss helper, not
# a second implementation of that loss.  Any drift stops before GPU training.
STANDARD_IDENTITY_CONTRACT = {
    "phase": "real",
    "updates": rehearsal.FORMAL_REQUESTED_UPDATES,
    "batch_size": rehearsal.FORMAL_BATCH_SIZE,
    "lr": rehearsal.FORMAL_LR,
    "weight_decay": rehearsal.FORMAL_WEIGHT_DECAY,
    "grad_clip": rehearsal.FORMAL_GRAD_CLIP,
    "seed": rehearsal.FORMAL_SEED,
    "window_stride": rehearsal.FORMAL_WINDOW_STRIDE,
    "val_fraction": rehearsal.FORMAL_VAL_FRACTION,
    "real_all_windows": False,
    "input_noise_std": 0.0,
    "point_loss_mode": "official_accuracy",
    "mse_weight": 0.25,
    "mse_warmup_updates": rehearsal.REAL_MSE_WARMUP_UPDATES,
    "mean_loss_weight": 0.25,
    "member_loss_weight": 1.0,
    "member_loss_mode": "point",
    "bootstrap_probability": 0.75,
    "quantile_loss_weight": 0.05,
    "soft_sps_loss_weight": rehearsal.REAL_SOFT_SPS_WEIGHT,
    "soft_sps_tau_fraction": rehearsal.REAL_SOFT_SPS_TAU_FRACTION,
    "soft_sps_warmup_updates": rehearsal.REAL_SOFT_SPS_WARMUP_UPDATES,
    "soft_sps_uq_only": True,
    "accuracy_weights": list(rehearsal.REAL_ACCURACY_WEIGHTS),
    "checkpoint_interval": rehearsal.FORMAL_CHECKPOINT_INTERVAL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-shg", type=Path, required=True)
    parser.add_argument("--start-shg-sha256", required=True)
    parser.add_argument("--shg-summary", type=Path, required=True)
    parser.add_argument("--shg-summary-sha256", required=True)
    parser.add_argument("--identity-summary", type=Path, required=True)
    parser.add_argument("--identity-summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--updates", type=int, default=rehearsal.FORMAL_REQUESTED_UPDATES,
        help="Frozen at 5000 so update 4000 has the standard LR schedule.")
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.updates != rehearsal.FORMAL_REQUESTED_UPDATES:
        raise ValueError("formal SHG1 Real requires --updates 5000")
    args.start_shg_sha256 = rehearsal.normalized_sha256(
        args.start_shg_sha256, "start-shg-sha256")
    args.shg_summary_sha256 = rehearsal.normalized_sha256(
        args.shg_summary_sha256, "shg-summary-sha256")
    args.identity_summary_sha256 = rehearsal.normalized_sha256(
        args.identity_summary_sha256, "identity-summary-sha256")
    if args.identity_summary_sha256 != rehearsal.FORMAL_IDENTITY_SUMMARY_SHA256:
        raise ValueError("identity summary is not the frozen standard C208 control")
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse SHG1 Real output: {args.output_dir}")


def _require_mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_exact(mapping: dict, key: str, expected, scope: str) -> None:
    actual = mapping.get(key)
    if isinstance(expected, float):
        matched = isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    else:
        matched = actual == expected
    if not matched:
        raise ValueError(
            f"{scope}[{key}]={actual!r}, expected {expected!r}")


def _finite_numeric_leaves(value) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("baseline_sim anchor contains a non-finite value")
        return [number]
    if isinstance(value, dict):
        result = []
        for child in value.values():
            result.extend(_finite_numeric_leaves(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(_finite_numeric_leaves(child))
        return result
    return []


def load_shg_summary(args: argparse.Namespace) -> dict:
    actual = rehearsal.sha256(args.shg_summary)
    if actual != args.shg_summary_sha256:
        raise RuntimeError(
            f"SHG1 summary SHA-256 {actual} != expected "
            f"{args.shg_summary_sha256}")
    payload = json.loads(args.shg_summary.read_text(encoding="utf-8"))
    payload = _require_mapping(payload, "SHG1 summary")
    for key, expected in {
            "method": SHG_METHOD,
            "status": SHG_PROMOTE_STATUS,
            "completed_updates": SHG_TERMINAL_UPDATE,
            "parameters": rehearsal.EXPECTED_PARAMETER_COUNT,
            }.items():
        _require_exact(payload, key, expected, "SHG1 summary")
    if payload.get("model_config") != rehearsal.EXPECTED_MODEL_CONFIG:
        raise ValueError("SHG1 summary model_config is not the frozen C208 config")

    preflight = _require_mapping(payload.get("preflight"), "SHG1 preflight")
    _require_exact(preflight, "pass", True, "SHG1 preflight")
    _require_exact(preflight, "status", SHG_PREFLIGHT_STATUS, "SHG1 preflight")
    sim_gate = _require_mapping(payload.get("sim_gate"), "SHG1 Sim gate")
    _require_exact(sim_gate, "pass", True, "SHG1 Sim gate")
    _require_exact(sim_gate, "status", SHG_PROMOTE_STATUS, "SHG1 Sim gate")

    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, str) or Path(checkpoint).name != "model_10000.pth":
        raise ValueError("SHG1 summary must name only model_10000.pth")
    checkpoint_sha = rehearsal.normalized_sha256(
        payload.get("checkpoint_sha256", ""), "SHG1 checkpoint_sha256")
    if checkpoint_sha != args.start_shg_sha256:
        raise ValueError("SHG1 summary checkpoint SHA does not bind start-shg")

    baseline = _require_mapping(payload.get("baseline_sim"), "baseline_sim")
    if not isinstance(baseline.get("path"), str) or not baseline["path"]:
        raise ValueError("baseline_sim must record a checkpoint path")
    baseline_sha = rehearsal.normalized_sha256(
        baseline.get("sha256", ""), "baseline_sim sha256")
    if baseline_sha != rehearsal.FORMAL_START_SIM_SHA256:
        raise ValueError("baseline_sim is not the frozen standard C208 Sim10k")
    for key in ("model_config_sha256", "training_config_sha256"):
        rehearsal.normalized_sha256(
            baseline.get(key, ""), f"baseline_sim {key}")
    anchor = _require_mapping(
        baseline.get("anchor_metrics"), "baseline_sim anchor_metrics")
    if not _finite_numeric_leaves(anchor):
        raise ValueError("baseline_sim anchor_metrics must contain finite metrics")
    source_summary = _require_mapping(
        baseline.get("source_summary"), "baseline_sim source_summary")
    if not isinstance(source_summary.get("path"), str) \
            or not source_summary["path"]:
        raise ValueError("baseline_sim source_summary must record a path")
    source_summary_sha = rehearsal.normalized_sha256(
        source_summary.get("sha256", ""), "baseline_sim source_summary sha256")
    if source_summary_sha != SHG_BASELINE_SUMMARY_SHA256:
        raise ValueError("baseline_sim source_summary is not the frozen summary")
    for key in (
            "model_config_sha256", "training_config_sha256", "history_sha256"):
        rehearsal.normalized_sha256(
            source_summary.get(key, ""), f"baseline_sim source_summary {key}")
    for key in ("model_config_sha256", "training_config_sha256"):
        if source_summary[key] != baseline[key]:
            raise ValueError(
                f"baseline_sim source_summary {key} differs from checkpoint")
    _require_exact(
        source_summary, "teacher_cache_sha256", SHG_TEACHER_CACHE_SHA256,
        "baseline_sim source_summary")
    _require_exact(
        source_summary, "teacher_cache_method",
        "sim_current_p1_point_teacher", "baseline_sim source_summary")
    _require_exact(
        source_summary, "distill_weight", rehearsal.FORMAL_P1_DISTILL_WEIGHT,
        "baseline_sim source_summary")

    teacher_cache = _require_mapping(
        payload.get("teacher_cache"), "SHG1 teacher_cache")
    _require_exact(
        teacher_cache, "sha256", SHG_TEACHER_CACHE_SHA256,
        "SHG1 teacher_cache")
    training = _require_mapping(
        payload.get("training_contract"), "SHG1 training_contract")
    for key, expected in {
            "seed": 0,
            "control_updates": 200,
            "terminal_update": SHG_TERMINAL_UPDATE,
            "only_scientific_delta": SHG_ONLY_SCIENTIFIC_DELTA,
            "soft_sps_gradient_scope": "full_graph_uq_only_false",
            "checkpoint_policy": "only_after_terminal_sim_gate_pass",
            "calibration_source_summary_sha256": (
                SHG_CALIBRATION_SOURCE_SUMMARY_SHA256),
            }.items():
        _require_exact(training, key, expected, "SHG1 training_contract")
    _require_exact(training, "hgrad_weight", SHG_HGRAD_WEIGHT,
                   "SHG1 training_contract")
    _require_exact(training, "hgrad_ramp_updates", SHG_HGRAD_RAMP_UPDATES,
                   "SHG1 training_contract")
    frozen_sim = _require_mapping(
        training.get("frozen_sim_loss"), "SHG1 frozen_sim_loss")
    if set(frozen_sim) != set(SHG_FROZEN_SIM_LOSS_KEYS):
        missing = sorted(set(SHG_FROZEN_SIM_LOSS_KEYS) - set(frozen_sim))
        extra = sorted(set(frozen_sim) - set(SHG_FROZEN_SIM_LOSS_KEYS))
        raise ValueError(
            f"SHG1 frozen_sim_loss keys differ: missing={missing}, extra={extra}")
    for key, expected in {
            "seed": 0,
            "max_files": 0,
            "window_stride": 20,
            "input_noise_std": 0.0,
            "point_loss_mode": "official_accuracy",
            "mse_warmup_updates": 0,
            "soft_sps_loss_weight": 0.05,
            "soft_sps_warmup_updates": 1500,
            "soft_sps_uq_only": False,
            "soft_sps_proxy_real_scale": True,
            "accuracy_weights": list(rehearsal.SIM_ACCURACY_WEIGHTS),
            "distill_weight": rehearsal.FORMAL_P1_DISTILL_WEIGHT,
            "scheduler_t_max": SHG_TERMINAL_UPDATE,
            }.items():
        _require_exact(frozen_sim, key, expected, "SHG1 frozen_sim_loss")
    return {**payload, "path": str(args.shg_summary.resolve()), "sha256": actual}


def load_shg_start_payload(args: argparse.Namespace, summary: dict) -> dict:
    actual = rehearsal.sha256(args.start_shg)
    if actual != args.start_shg_sha256:
        raise RuntimeError(
            f"SHG1 checkpoint SHA-256 {actual} != expected "
            f"{args.start_shg_sha256}")
    payload = torch.load(args.start_shg, map_location="cpu", weights_only=False)
    payload = _require_mapping(payload, "SHG1 checkpoint")
    _require_exact(payload, "phase", "sim", "SHG1 checkpoint")
    _require_exact(
        payload, "iteration", SHG_TERMINAL_UPDATE, "SHG1 checkpoint")
    if payload.get("model_config") != rehearsal.EXPECTED_MODEL_CONFIG:
        raise ValueError("SHG1 checkpoint model_config is not the frozen C208 config")
    training = _require_mapping(
        payload.get("training_config"), "SHG1 checkpoint training_config")
    for key, expected in {
            "method": SHG_METHOD,
            "baseline_sim_sha256": summary["baseline_sim"]["sha256"],
            "baseline_summary_sha256": SHG_BASELINE_SUMMARY_SHA256,
            "teacher_cache_sha256": SHG_TEACHER_CACHE_SHA256,
            "hgrad_weight": SHG_HGRAD_WEIGHT,
            "hgrad_ramp_updates": SHG_HGRAD_RAMP_UPDATES,
            "calibration_source_summary_sha256": (
                SHG_CALIBRATION_SOURCE_SUMMARY_SHA256),
            "preflight_pass": True,
            "sim_gate_pass": True,
            }.items():
        _require_exact(training, key, expected, "SHG1 checkpoint training_config")
    baseline_summary_path = training.get("baseline_summary")
    if not isinstance(baseline_summary_path, str) or not baseline_summary_path:
        raise ValueError(
            "SHG1 checkpoint training_config must record baseline_summary")
    source_summary_path = summary["baseline_sim"]["source_summary"]["path"]
    if Path(baseline_summary_path).resolve() != Path(source_summary_path).resolve():
        raise ValueError(
            "SHG1 checkpoint baseline_summary path differs from Sim summary")
    frozen_sim = summary["training_contract"]["frozen_sim_loss"]
    for key in SHG_FROZEN_SIM_LOSS_KEYS:
        if key == "scheduler_t_max":
            continue
        _require_exact(
            training, key, frozen_sim[key], "SHG1 checkpoint training_config")
    metrics = _require_mapping(payload.get("metrics"), "SHG1 checkpoint metrics")
    _require_exact(metrics, "pass", True, "SHG1 checkpoint metrics")
    _require_exact(metrics, "status", SHG_PROMOTE_STATUS,
                   "SHG1 checkpoint metrics")
    return payload


def validate_standard_real_contract() -> None:
    if rehearsal.IDENTITY_TRAINING_CONTRACT != STANDARD_IDENTITY_CONTRACT:
        raise RuntimeError("imported standard Real training contract drifted")
    if rehearsal.FORMAL_SCHEDULER_T_MAX != rehearsal.FORMAL_REQUESTED_UPDATES:
        raise RuntimeError("standard Real scheduler T_max drifted from 5000")


def load_identity_step4000(args: argparse.Namespace) -> dict:
    """Validate the frozen standard control and exact update-4000 row."""
    proxy = argparse.Namespace(
        rehearsal_weight=rehearsal.FORMAL_REHEARSAL_WEIGHT,
        identity_summary=args.identity_summary,
        identity_summary_sha256=args.identity_summary_sha256,
    )
    identity = rehearsal.load_identity_step2000(proxy)
    if identity is None:
        raise RuntimeError("frozen identity summary unexpectedly resolved null")
    payload = json.loads(args.identity_summary.read_text(encoding="utf-8"))
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError("identity summary has no history")
    rows = [
        row for row in history if isinstance(row, dict)
        and int(row.get("iteration", -1)) == TERMINAL_UPDATE]
    if len(rows) != 1:
        raise ValueError("identity summary must have exactly one update-4000 row")
    metrics = {}
    for key, expected in NATIVE_STEP4000_METRICS.items():
        value = rows[0].get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"identity update-4000 row is missing finite {key}")
        metrics[key] = float(value)
        if not math.isclose(metrics[key], expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"identity update-4000 {key}={metrics[key]}, expected {expected}")
    return {**identity, "metrics_update4000": metrics}


def terminal_metric_gate(metrics: dict, baseline: dict | None = None) -> dict:
    if baseline is None:
        baseline = NATIVE_STEP4000_METRICS
    required = (*NATIVE_STEP4000_METRICS.keys(),)
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"terminal metrics missing {missing}")
    selected = {key: float(metrics[key]) for key in required}
    if not all(math.isfinite(value) for value in selected.values()):
        raise ValueError("terminal metrics contain non-finite values")
    delta = {key: selected[key] - float(baseline[key]) for key in required}
    shadow = sum(
        rehearsal.NON_TIME_COEFFICIENTS[key] * delta[key]
        for key in rehearsal.NON_TIME_COEFFICIENTS)
    component_floors = {
        key: delta[key] >= GATE_COMPONENT_DELTA_FLOOR - 1e-12
        for key in COMPONENT_KEYS}
    passed = bool(
        delta["point_score"] >= GATE_POINT_DELTA_MIN - 1e-12
        and all(component_floors.values())
        and delta["native_sps_score"] >= GATE_SPS_DELTA_MIN - 1e-12
        and shadow >= GATE_NON_TIME_SHADOW_MIN - 1e-12)
    return {
        "status": "PROMOTE_SHG1B_REAL_MODEL_04000" if passed
        else "STOP_SHG1B_REAL_UPDATE4000_GATE",
        "pass": passed,
        "metrics": selected,
        "baseline": dict(baseline),
        "delta": delta,
        "component_floors": component_floors,
        "non_time_shadow": shadow,
    }


def evaluate_terminal_gate(
        model, real: dict, device: torch.device, identity: dict) -> dict:
    if real["validation_access_updates"]:
        raise RuntimeError("fixed725 was accessed before its unique update4000 gate")
    metrics = rehearsal.ensemble_train.evaluate(
        model, real["validation_loader"], real["normalizer"], device,
        rehearsal.REAL_ACCURACY_WEIGHTS)
    real["validation_access_updates"].append(TERMINAL_UPDATE)
    result = terminal_metric_gate(metrics, identity["metrics_update4000"])
    result["identity_summary"] = identity
    return result


def train_endpoint(
        args: argparse.Namespace, model, real: dict, identity: dict,
        device: torch.device) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=rehearsal.FORMAL_LR,
        weight_decay=rehearsal.FORMAL_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    iterator = rehearsal.ensemble_train.common.repro.cycling(real["loader"])
    history = []
    sums: dict[str, float] = {}
    count = 0
    gate = None
    started = time.time()

    for iteration in range(1, TERMINAL_UPDATE + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = rehearsal.backward_real_supervised(
            model, next(iterator), real["normalizer"], device, iteration)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), rehearsal.FORMAL_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        values["learning_rate"] = optimizer.param_groups[0]["lr"]
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration % rehearsal.FORMAL_LOG_INTERVAL == 0:
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(json.dumps({
                "event": "c208_shg1b_standard_real_progress", **row,
            }), flush=True)
            sums.clear()
            count = 0
        if iteration == TERMINAL_UPDATE:
            gate = evaluate_terminal_gate(model, real, device, identity)
            print(json.dumps({
                "event": "c208_shg1b_standard_real_gate", **gate,
            }), flush=True)
            if not gate["pass"]:
                return {
                    "status": gate["status"],
                    "completed_updates": iteration,
                    "history": history,
                    "gate": gate,
                    "checkpoint": None,
                    "wall_seconds": time.time() - started,
                }

    checkpoint = args.output_dir / "model_04000.pth"
    stats = rehearsal.ensemble_train.common.normalizer_dict(real["normalizer"])
    rehearsal.ensemble_train.common.save_checkpoint(
        checkpoint, model, args, TERMINAL_UPDATE, history, stats, gate)
    return {
        "status": "COMPLETE_SHG1B_REAL_MODEL_04000_AFTER_GATE",
        "completed_updates": TERMINAL_UPDATE,
        "history": history,
        "gate": gate,
        "checkpoint": str(checkpoint),
        "wall_seconds": time.time() - started,
    }


def training_contract(checkpoint_saved: bool) -> dict:
    return {
        **STANDARD_IDENTITY_CONTRACT,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": rehearsal.FORMAL_SCHEDULER_T_MAX,
        "terminal_update": TERMINAL_UPDATE,
        "fixed725_evaluation_updates": [TERMINAL_UPDATE],
        "fixed725_optimizer_access": False,
        "standard_real_loss_helper": "backward_real_supervised",
        "native_soft_sps_gradient_scope": "uq_trainable_parameters_only",
        "shg_or_other_auxiliary_loss": None,
        "checkpoints_saved": ["model_04000.pth"] if checkpoint_saved else [],
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    validate_standard_real_contract()
    shg_summary = load_shg_summary(args)
    identity = load_identity_step4000(args)
    args.phase = "real"
    args.method = METHOD
    args.start_sim_sha256 = args.start_shg_sha256
    args.shg_summary_sha256 = shg_summary["sha256"]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal C208 SHG1 Real training requires CUDA")
    rehearsal.ensemble_train.seed_everything(rehearsal.FORMAL_SEED)
    torch.set_float32_matmul_precision("high")
    payload = load_shg_start_payload(args, shg_summary)
    model = rehearsal.load_model(payload, device)
    del payload
    args.output_dir.mkdir(parents=True, exist_ok=False)
    real = rehearsal.prepare_real_training(args, device)
    rehearsal.ensemble_train.common.set_model_stats(model, real["normalizer"])
    print(json.dumps({
        "event": "c208_shg1b_standard_real_contract",
        "parameters": rehearsal.EXPECTED_PARAMETER_COUNT,
        "start_shg_sha256": args.start_shg_sha256,
        "shg_summary_sha256": args.shg_summary_sha256,
        "identity_summary_sha256": args.identity_summary_sha256,
        "standard_real_loss_changed": False,
        "fixed725_access": "once_at_update4000_only",
        "fixed725_optimizer_access": False,
    }), flush=True)
    result = train_endpoint(args, model, real, identity, device)
    summary = {
        "method": METHOD,
        "evidence_class": (
            "fixed2576_standard_real_optimizer_single_fixed725_update4000_gate"),
        "start_shg": str(args.start_shg.resolve()),
        "start_shg_sha256": args.start_shg_sha256,
        "shg_summary": {
            "path": shg_summary["path"],
            "sha256": shg_summary["sha256"],
            "method": shg_summary["method"],
            "status": shg_summary["status"],
            "sim_gate": shg_summary["sim_gate"],
        },
        "identity_summary": identity,
        "model_config": rehearsal.EXPECTED_MODEL_CONFIG,
        "parameters": rehearsal.EXPECTED_PARAMETER_COUNT,
        "real_contract": {
            "counts": real["counts"],
            "split_manifest_sha256": real["manifest_sha256"],
            "real_stats_sha256": real["stats_sha256"],
            "validation_access_updates": real["validation_access_updates"],
            "optimizer_access": False,
        },
        "training_contract": training_contract(result["checkpoint"] is not None),
        "gate_contract": {
            "update": TERMINAL_UPDATE,
            "baseline": NATIVE_STEP4000_METRICS,
            "point_score_delta_min": GATE_POINT_DELTA_MIN,
            "component_delta_floor": GATE_COMPONENT_DELTA_FLOOR,
            "native_sps_delta_min": GATE_SPS_DELTA_MIN,
            "non_time_shadow_min": GATE_NON_TIME_SHADOW_MIN,
            "single_fixed725_access": True,
            "on_failure": "STOP_NO_CHECKPOINT",
        },
        **result,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "complete", "status": result["status"],
        "summary": str(summary_path), "checkpoint": result["checkpoint"],
    }), flush=True)


if __name__ == "__main__":
    main()
