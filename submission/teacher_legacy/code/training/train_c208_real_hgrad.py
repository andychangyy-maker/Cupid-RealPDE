#!/usr/bin/env python3
"""Run the single audit-calibrated C208-HG1b-small4k Real endpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import time

import torch
from torch.utils.data import DataLoader

import train_c208_real_sim_rehearsal as rehearsal


PREFLIGHT_UPDATE = 200
PREFLIGHT_AUDIT_UPDATES = (50, 100, 150, 200)
GATE_UPDATE = rehearsal.FORMAL_GATE_UPDATE
TERMINAL_UPDATE = rehearsal.FORMAL_TERMINAL_UPDATE
HGRAD_WEIGHT = 0.004
HGRAD_RAMP_UPDATES = 200
HGRAD_RMS_EPSILON = 1e-6
HGRAD_SCALE_FLOOR = 1e-6
PREFLIGHT_LOSS_RATIO_RANGE = (0.005, 0.06)
PREFLIGHT_GRAD_RATIO_RANGE = (0.01, 0.20)
PREFLIGHT_MEDIAN_COSINE_MIN = -0.10
PREFLIGHT_WORST_COSINE_MIN = -0.50
GATE_POINT_FLOOR = -0.10
GATE_REL_FLOOR = -0.10
GATE_TKE_FLOOR = -0.30
GATE_MVPE_FLOOR = -0.10
GATE_SPS_FLOOR = -0.10
GATE_SHADOW_MIN = 0.04
TERMINAL_SHADOW_MIN = 0.20
NATIVE_STEP4000_METRICS = {
    "rel_l2_score": 95.86352389426571,
    "tke_rel_l2_score": 80.95221179698406,
    "mvpe_rel_l2_score": 96.51432542559023,
    "point_score": 91.52029057134612,
    "native_sps_score": 53.07979215521265,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--start-sim-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int,
                        default=rehearsal.FORMAL_REQUESTED_UPDATES)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--identity-summary", type=Path, required=True)
    parser.add_argument("--identity-summary-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.updates != rehearsal.FORMAL_REQUESTED_UPDATES:
        raise ValueError("formal HG1 requires --updates 5000")
    args.start_sim_sha256 = rehearsal.normalized_sha256(
        args.start_sim_sha256, "start-sim-sha256")
    if args.start_sim_sha256 != rehearsal.FORMAL_START_SIM_SHA256:
        raise ValueError("start-sim-sha256 is not the frozen Sim10k start")
    args.identity_summary_sha256 = rehearsal.normalized_sha256(
        args.identity_summary_sha256, "identity-summary-sha256")
    if (args.identity_summary_sha256
            != rehearsal.FORMAL_IDENTITY_SUMMARY_SHA256):
        raise ValueError("identity summary is not the frozen native control")
    protected = (
        "summary.json", "model_04000.pth", "hgrad_scale_contract.json")
    if any((args.output_dir / name).exists() for name in protected):
        raise FileExistsError(
            f"refusing to overwrite a completed run in {args.output_dir}")


def load_identity_step2000(args: argparse.Namespace) -> dict:
    proxy = argparse.Namespace(
        **vars(args), rehearsal_weight=rehearsal.FORMAL_REHEARSAL_WEIGHT)
    identity = rehearsal.load_identity_step2000(proxy)
    if identity is None:
        raise RuntimeError("frozen identity summary unexpectedly resolved null")
    return identity


def forward_differences(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 5 or value.shape[-1] != 2:
        raise ValueError("velocity must have shape [B,T,H,W,2]")
    dx = value[:, :, :, 1:, :] - value[:, :, :, :-1, :]
    dy = value[:, :, 1:, :, :] - value[:, :, :-1, :, :]
    return dx, dy


def compute_hgrad_scale_contract(
        dataset, split_sha256: str, real_stats_sha256: str,
        expected_windows: int = rehearsal.FORMAL_REAL_COUNTS["train_windows"],
        ) -> dict:
    if len(dataset) != expected_windows:
        raise RuntimeError(
            f"hgrad stats windows={len(dataset)}, expected {expected_windows}")
    sums = {
        "dx": torch.zeros(2, dtype=torch.float64),
        "dy": torch.zeros(2, dtype=torch.float64),
    }
    counts = {"dx": 0, "dy": 0}
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(0))
    for batch in loader:
        target = batch[1][..., :2].to(dtype=torch.float64)
        if not bool(torch.isfinite(target).all()):
            raise FloatingPointError("non-finite train target in hgrad stats")
        dx, dy = forward_differences(target)
        sums["dx"].add_(dx.square().sum(dim=(0, 1, 2, 3)))
        sums["dy"].add_(dy.square().sum(dim=(0, 1, 2, 3)))
        counts["dx"] += dx.numel() // 2
        counts["dy"] += dy.numel() // 2
    scales = {
        key: torch.sqrt(sums[key] / counts[key]).clamp_min(
            HGRAD_SCALE_FLOOR).tolist()
        for key in ("dx", "dy")
    }
    payload = {
        "method": "train2576_physical_uv_forward_difference_rms",
        "train_windows": len(dataset),
        "split_manifest_sha256": split_sha256,
        "real_stats_sha256": real_stats_sha256,
        "scale_floor": HGRAD_SCALE_FLOOR,
        "counts_per_channel": counts,
        "sum_squares": {key: sums[key].tolist() for key in sums},
        "scales": scales,
        "validation_windows_used": 0,
    }
    payload["canonical_sha256"] = rehearsal.canonical_sha256(payload)
    return payload


def validate_hgrad_scale_contract(contract: dict, real: dict) -> None:
    payload = dict(contract)
    digest = payload.pop("canonical_sha256", None)
    if digest != rehearsal.canonical_sha256(payload):
        raise RuntimeError("hgrad scale canonical SHA-256 mismatch")
    if contract.get("train_windows") != rehearsal.FORMAL_REAL_COUNTS[
            "train_windows"]:
        raise RuntimeError("hgrad scale contract is not train2576")
    if contract.get("validation_windows_used") != 0:
        raise RuntimeError("hgrad scale contract accessed validation windows")
    if contract.get("split_manifest_sha256") != real["manifest_sha256"]:
        raise RuntimeError("hgrad scale split SHA-256 mismatch")
    if contract.get("real_stats_sha256") != real["stats_sha256"]:
        raise RuntimeError("hgrad scale Real stats SHA-256 mismatch")


def hgrad_weight(iteration: int) -> float:
    return HGRAD_WEIGHT * min(1.0, iteration / HGRAD_RAMP_UPDATES)


def hgrad_loss(
        prediction: torch.Tensor, target: torch.Tensor, normalizer,
        scale_contract: dict,
        ) -> torch.Tensor:
    pred = rehearsal.ensemble_train.common.to_physical_target(
        prediction, normalizer)[..., :2]
    truth = rehearsal.ensemble_train.common.to_physical_target(
        target, normalizer)[..., :2]
    pred_dx, pred_dy = forward_differences(pred)
    truth_dx, truth_dy = forward_differences(truth)
    scale_dx = pred.new_tensor(scale_contract["scales"]["dx"])
    scale_dy = pred.new_tensor(scale_contract["scales"]["dy"])
    error_dx = (pred_dx - truth_dx) / scale_dx
    error_dy = (pred_dy - truth_dy) / scale_dy
    rms_dx = torch.sqrt(
        error_dx.square().mean(dim=(1, 2, 3, 4)) + HGRAD_RMS_EPSILON)
    rms_dy = torch.sqrt(
        error_dy.square().mean(dim=(1, 2, 3, 4)) + HGRAD_RMS_EPSILON)
    return 0.5 * (rms_dx + rms_dy).mean()


def output_gradient_audit(
        point_term: torch.Tensor, weighted_hgrad: torch.Tensor,
        supervised_native: torch.Tensor, prediction: torch.Tensor,
        ) -> dict[str, float]:
    point_gradient = torch.autograd.grad(
        point_term, prediction, retain_graph=True)[0]
    hgrad_gradient = torch.autograd.grad(
        weighted_hgrad, prediction, retain_graph=True)[0]
    point_flat = point_gradient.reshape(-1)
    hgrad_flat = hgrad_gradient.reshape(-1)
    point_norm = torch.linalg.vector_norm(point_flat)
    hgrad_norm = torch.linalg.vector_norm(hgrad_flat)
    denominator = point_norm * hgrad_norm
    cosine = torch.dot(point_flat, hgrad_flat) / denominator.clamp_min(1e-12)
    values = {
        "loss_ratio": float(
            weighted_hgrad.detach()
            / supervised_native.detach().abs().clamp_min(1e-12)),
        "gradient_ratio": float(hgrad_norm / point_norm.clamp_min(1e-12)),
        "cosine": float(cosine),
        "point_gradient_norm": float(point_norm),
        "hgrad_gradient_norm": float(hgrad_norm),
    }
    values["finite_nonzero"] = bool(
        all(math.isfinite(value) for value in values.values())
        and values["point_gradient_norm"] > 0.0
        and values["hgrad_gradient_norm"] > 0.0)
    return values


def evaluate_preflight(audits: list[dict]) -> dict:
    updates = [row.get("iteration") for row in audits]
    exact = updates == list(PREFLIGHT_AUDIT_UPDATES)
    finite = exact and all(row.get("finite_nonzero") is True for row in audits)
    if finite:
        loss_ratio = median(row["loss_ratio"] for row in audits)
        gradient_ratio = median(row["gradient_ratio"] for row in audits)
        cosine = median(row["cosine"] for row in audits)
        worst_cosine = min(row["cosine"] for row in audits)
    else:
        loss_ratio = gradient_ratio = cosine = worst_cosine = None
    passed = bool(
        finite
        and PREFLIGHT_LOSS_RATIO_RANGE[0] <= loss_ratio
        <= PREFLIGHT_LOSS_RATIO_RANGE[1]
        and PREFLIGHT_GRAD_RATIO_RANGE[0] <= gradient_ratio
        <= PREFLIGHT_GRAD_RATIO_RANGE[1]
        and cosine >= PREFLIGHT_MEDIAN_COSINE_MIN
        and worst_cosine >= PREFLIGHT_WORST_COSINE_MIN)
    return {
        "status": "GO_HGRAD_TO_UPDATE2000" if passed
        else "STOP_HGRAD_PREFLIGHT",
        "pass": passed,
        "audits": audits,
        "median_loss_ratio": loss_ratio,
        "median_gradient_ratio": gradient_ratio,
        "median_cosine": cosine,
        "worst_cosine": worst_cosine,
    }


def metric_gate(metrics: dict, baseline: dict, shadow_min: float) -> dict:
    keys = rehearsal.IDENTITY_METRIC_KEYS
    delta = {key: float(metrics[key]) - float(baseline[key]) for key in keys}
    shadow = sum(
        rehearsal.NON_TIME_COEFFICIENTS[key] * delta[key]
        for key in rehearsal.NON_TIME_COEFFICIENTS)
    floors = {
        "point": delta["point_score"] >= GATE_POINT_FLOOR,
        "rel": delta["rel_l2_score"] >= GATE_REL_FLOOR,
        "tke": delta["tke_rel_l2_score"] >= GATE_TKE_FLOOR,
        "mvpe": delta["mvpe_rel_l2_score"] >= GATE_MVPE_FLOOR,
        "sps": delta["native_sps_score"] >= GATE_SPS_FLOOR,
    }
    passed = bool(all(floors.values()) and shadow >= shadow_min)
    return {
        "pass": passed,
        "delta": delta,
        "floors": floors,
        "non_time_shadow": shadow,
        "shadow_min": shadow_min,
    }


def evaluate_step2000_gate(model, real: dict, device, identity: dict) -> dict:
    if real["validation_access_updates"]:
        raise RuntimeError("fixed725 was already accessed before its only gate")
    metrics = rehearsal.ensemble_train.evaluate(
        model, real["validation_loader"], real["normalizer"], device,
        rehearsal.REAL_ACCURACY_WEIGHTS)
    real["validation_access_updates"].append(GATE_UPDATE)
    metrics = {key: float(value) for key, value in metrics.items()}
    result = metric_gate(metrics, identity["metrics"], GATE_SHADOW_MIN)
    result.update({
        "status": "GO_HGRAD_TO_UPDATE4000" if result["pass"]
        else "STOP_HGRAD_UPDATE2000_GATE",
        "metrics": metrics,
        "identity_summary": identity,
    })
    return result


def evaluate_terminal_promotion(metrics: dict) -> dict:
    baseline = dict(NATIVE_STEP4000_METRICS)
    for key in rehearsal.IDENTITY_METRIC_KEYS:
        baseline.setdefault(key, float(metrics[key]))
    result = metric_gate(metrics, baseline, TERMINAL_SHADOW_MIN)
    result["status"] = "PROMOTE_HGRAD_TO_OOF" if result["pass"] \
        else "STOP_HGRAD_BEFORE_OOF"
    return result


def backward_real_hgrad(
        model, batch, normalizer, scale_contract: dict,
        device: torch.device, iteration: int,
        ) -> tuple[dict[str, float], dict | None]:
    history, target, trajectory_ids, _sample_indices = batch
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    history_norm, target_norm = normalizer.preprocess(history, target)
    (prediction, members, quantiles, lower, upper, _features,
     _coverage_logits) = model.forward_uq_training(history_norm)
    point_loss, point_components = rehearsal.ensemble_train.point_loss(
        rehearsal.real_loss_args(), prediction, target_norm, normalizer,
        rehearsal.REAL_ACCURACY_WEIGHTS, iteration)
    member_loss, bootstrap_fraction = (
        rehearsal.ensemble_train.bootstrap_member_point_loss(
            rehearsal.real_loss_args(), members, target_norm, trajectory_ids,
            normalizer, rehearsal.REAL_ACCURACY_WEIGHTS, iteration))
    quantile_loss = rehearsal.ensemble_train.residual_quantile_loss(
        model, prediction, quantiles, target_norm, normalizer)
    soft_sps_loss, soft_sps_score = (
        rehearsal.ensemble_train.soft_native_sps_objective(
            prediction, target_norm, lower, upper, normalizer,
            tau_fraction=rehearsal.REAL_SOFT_SPS_TAU_FRACTION))
    soft_sps_weight = rehearsal.REAL_SOFT_SPS_WEIGHT * min(
        1.0, iteration / rehearsal.REAL_SOFT_SPS_WARMUP_UPDATES)
    supervised_native = (
        0.25 * point_loss + member_loss + 0.05 * quantile_loss)
    spatial_loss = hgrad_loss(
        prediction, target_norm, normalizer, scale_contract)
    spatial_weight = hgrad_weight(iteration)
    weighted_hgrad = spatial_weight * spatial_loss
    supervised = supervised_native + weighted_hgrad
    total = supervised + soft_sps_weight * soft_sps_loss
    if not bool(torch.isfinite(total)):
        raise FloatingPointError(f"non-finite HG1 loss at update {iteration}")
    audit = None
    if iteration in PREFLIGHT_AUDIT_UPDATES:
        audit = output_gradient_audit(
            0.25 * point_loss, weighted_hgrad,
            supervised_native, prediction)
        audit["iteration"] = iteration
    supervised.backward(retain_graph=True)
    interval_parameters = rehearsal.ensemble_train.uq_trainable_parameters(model)
    interval_gradients = torch.autograd.grad(
        soft_sps_weight * soft_sps_loss,
        interval_parameters, allow_unused=True)
    for parameter, gradient in zip(interval_parameters, interval_gradients):
        if gradient is None:
            continue
        if parameter.grad is None:
            parameter.grad = gradient
        else:
            parameter.grad.add_(gradient)
    values = {
        "real_loss": float(total.detach()),
        "real_supervised_native_loss": float(supervised_native.detach()),
        "hgrad_loss": float(spatial_loss.detach()),
        "hgrad_weight": spatial_weight,
        "hgrad_weighted_loss": float(weighted_hgrad.detach()),
        "real_soft_sps_score": float(soft_sps_score.detach()),
        "real_bootstrap_fraction": float(bootstrap_fraction.detach()),
        **{f"real_{key}": float(value.detach())
           for key, value in point_components.items()},
    }
    return values, audit


def train_endpoint(args, model, real, identity, scale_contract, device) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=rehearsal.FORMAL_LR,
        weight_decay=rehearsal.FORMAL_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)
    iterator = rehearsal.ensemble_train.common.repro.cycling(real["loader"])
    history, audits = [], []
    sums: dict[str, float] = {}
    count = 0
    preflight = gate = None
    started = time.time()
    for iteration in range(1, TERMINAL_UPDATE + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values, audit = backward_real_hgrad(
            model, next(iterator), real["normalizer"], scale_contract,
            device, iteration)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), rehearsal.FORMAL_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        if audit is not None:
            audits.append(audit)
        values["learning_rate"] = optimizer.param_groups[0]["lr"]
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value
        count += 1
        if iteration == PREFLIGHT_UPDATE:
            preflight = evaluate_preflight(audits)
            print(json.dumps({"event": "c208_hgrad_preflight", **preflight}),
                  flush=True)
            if not preflight["pass"]:
                return _stopped(preflight["status"], iteration, history,
                                preflight, None, started)
        if iteration % rehearsal.FORMAL_LOG_INTERVAL == 0:
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(json.dumps({"event": "c208_hgrad_progress", **row}),
                  flush=True)
            sums.clear()
            count = 0
        if iteration == GATE_UPDATE:
            gate = evaluate_step2000_gate(model, real, device, identity)
            print(json.dumps({"event": "c208_hgrad_gate", **gate}), flush=True)
            if not gate["pass"]:
                return _stopped(gate["status"], iteration, history,
                                preflight, gate, started)
    checkpoint = args.output_dir / "model_04000.pth"
    stats = rehearsal.ensemble_train.common.normalizer_dict(real["normalizer"])
    rehearsal.ensemble_train.common.save_checkpoint(
        checkpoint, model, args, TERMINAL_UPDATE, history, stats, history[-1])
    return {
        "status": "COMPLETE_HGRAD_UPDATE4000_AFTER_GATES",
        "completed_updates": TERMINAL_UPDATE,
        "history": history,
        "preflight": preflight,
        "gate": gate,
        "checkpoint": str(checkpoint),
        "wall_seconds": time.time() - started,
    }


def _stopped(status, iteration, history, preflight, gate, started) -> dict:
    return {
        "status": status,
        "completed_updates": iteration,
        "history": history,
        "preflight": preflight,
        "gate": gate,
        "checkpoint": None,
        "wall_seconds": time.time() - started,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.phase = "real"
    args.hgrad_weight = HGRAD_WEIGHT
    args.hgrad_ramp_updates = HGRAD_RAMP_UPDATES
    args.hgrad_definition = "physical_uv_forward_difference_rms"
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal C208 HG1 training requires CUDA")
    rehearsal.ensemble_train.seed_everything(rehearsal.FORMAL_SEED)
    torch.set_float32_matmul_precision("high")
    payload = rehearsal.load_start_payload(args)
    model = rehearsal.load_model(payload, device)
    del payload
    real = rehearsal.prepare_real_training(args, device)
    rehearsal.ensemble_train.common.set_model_stats(model, real["normalizer"])
    scale_contract = compute_hgrad_scale_contract(
        real["loader"].dataset, real["manifest_sha256"], real["stats_sha256"])
    scale_path = args.output_dir / "hgrad_scale_contract.json"
    scale_path.write_text(
        json.dumps(scale_contract, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n", encoding="utf-8")
    reloaded = json.loads(scale_path.read_text(encoding="utf-8"))
    validate_hgrad_scale_contract(reloaded, real)
    args.hgrad_scale_contract_sha256 = reloaded["canonical_sha256"]
    identity = load_identity_step2000(args)
    print(json.dumps({
        "event": "c208_hgrad_contract",
        "parameters": rehearsal.EXPECTED_PARAMETER_COUNT,
        "scale_contract_sha256": reloaded["canonical_sha256"],
        "preflight_update": PREFLIGHT_UPDATE,
        "gate_update": GATE_UPDATE,
        "terminal_update": TERMINAL_UPDATE,
        "hgrad_weight": HGRAD_WEIGHT,
        "hgrad_ramp_updates": HGRAD_RAMP_UPDATES,
        "validation_optimizer_access": False,
    }), flush=True)
    result = train_endpoint(args, model, real, identity, reloaded, device)
    summary = {
        "method": "c208_hg1b_small4k",
        "evidence_class": (
            "fixed2576_real_hgrad_train_only_preflight_"
            "single_fixed725_gate_update2000"),
        "artifact_contract": {
            "start_sim": str(args.start_sim),
            "start_sim_sha256": args.start_sim_sha256,
            "identity_summary": str(args.identity_summary),
            "identity_summary_sha256": args.identity_summary_sha256,
            "hgrad_scale_contract": str(scale_path),
            "hgrad_scale_contract_sha256": reloaded["canonical_sha256"],
        },
        "real_contract": {
            "counts": real["counts"],
            "split_manifest_sha256": real["manifest_sha256"],
            "real_stats_sha256": real["stats_sha256"],
            "validation_optimizer_access": real["validation_optimizer_access"],
            "validation_access_updates": real["validation_access_updates"],
        },
        "hgrad_contract": {
            "channels": ["u", "v"],
            "difference": "unpadded_forward_dx_dy",
            "vorticity": False,
            "member_scope": "ensemble_mean_only",
            "weight": HGRAD_WEIGHT,
            "ramp_updates": HGRAD_RAMP_UPDATES,
            "rms_epsilon": HGRAD_RMS_EPSILON,
            "scale_source": "fixed2576_targets_only",
        },
        "gate_contract": {
            "point_floor": GATE_POINT_FLOOR,
            "rel_floor": GATE_REL_FLOOR,
            "tke_floor": GATE_TKE_FLOOR,
            "mvpe_floor": GATE_MVPE_FLOOR,
            "sps_floor": GATE_SPS_FLOOR,
            "step2000_shadow_min": GATE_SHADOW_MIN,
            "terminal_shadow_min": TERMINAL_SHADOW_MIN,
            "terminal_evaluation": "offline_only_not_in_training",
        },
        "training_contract": {
            "requested_updates": args.updates,
            "terminal_update": TERMINAL_UPDATE,
            "scheduler_t_max": rehearsal.FORMAL_SCHEDULER_T_MAX,
            "batch_size": rehearsal.FORMAL_BATCH_SIZE,
            "lr": rehearsal.FORMAL_LR,
            "weight_decay": rehearsal.FORMAL_WEIGHT_DECAY,
            "grad_clip": rehearsal.FORMAL_GRAD_CLIP,
            "seed": rehearsal.FORMAL_SEED,
            "checkpoints_saved": (
                ["model_04000.pth"] if result["checkpoint"] else []),
        },
        **result,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "complete", "status": result["status"],
        "summary": str(summary_path), "checkpoint": result["checkpoint"],
    }), flush=True)


if __name__ == "__main__":
    main()
