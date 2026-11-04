#!/usr/bin/env python3
"""Fresh 23M DCT planned-forcing MS-Closure Sim12k experiment.

This is the planned-forcing counterpart of the scaled MS control.  It keeps
the control's held-AoA10 split, seeded sample stream, optimizer, 12k cosine
schedule, EMA, and raw/EMA evaluator.  Future target coefficients supervise
the causal history-only plan head through one bounded relative objective;
they are never accepted by or fed into the rollout.

Smoke examples (neither starts formal training)::

    python train_msclosure_dct_planned_forcing_scaled_long.py \
        --mode smoke --device cpu
    python train_msclosure_dct_planned_forcing_scaled_long.py \
        --mode smoke --device cuda:0
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_multiple_shooting_closure_scaled_long as control  # noqa: E402
import train_pyramid_causal3d_loao_screen as loao  # noqa: E402
import train_pyramid_causal3d_sim_long as atomic  # noqa: E402
from models.msclosure_dct_planned_forcing import (  # noqa: E402
    DCTPlannedForcingMultipleShootingClosure,
    DCTPlannedForcingMultipleShootingConfig,
)


SCHEMA = "msclosure_dct_planned_forcing_scaled_long_held10_v1"
ACKNOWLEDGEMENT = "RUN_MS_DCT_PLANNED_FORCING_SCALED_LONG_HELD10"
FIT_AOAS = control.FIT_AOAS
HELD_AOA = control.HELD_AOA
SEED = control.SEED
SCHEDULE_SEED = control.SCHEDULE_SEED
UPDATES = control.UPDATES
BATCH_SIZE = control.BATCH_SIZE
LR = control.LR
WEIGHT_DECAY = control.WEIGHT_DECAY
GRAD_CLIP = control.GRAD_CLIP
EMA_DECAY = control.EMA_DECAY
TRACE_INTERVAL = control.TRACE_INTERVAL
LOSS_WEIGHTS = control.LOSS_WEIGHTS
MSE_WEIGHT = control.MSE_WEIGHT

# Accuracy has six weight units (1, 4, 1); the bounded coefficient objective
# contributes one additional unit on the same convention.
COEFFICIENT_AUX_WEIGHT = 1.0 / sum(LOSS_WEIGHTS)
BOUNDED_RELATIVE_SCALE = 0.5

MODEL_CONFIG = DCTPlannedForcingMultipleShootingConfig(
    e0_channels=128,
    e1_channels=256,
    e2_channels=512,
    context_channels=128,
    cell_channels=192,
    cell_blocks=2,
)


def model_config(
        *, tiny: bool = False,
        ) -> DCTPlannedForcingMultipleShootingConfig:
    if not tiny:
        return MODEL_CONFIG
    return DCTPlannedForcingMultipleShootingConfig(
        e0_channels=8, e1_channels=16, e2_channels=24,
        e0_blocks=1, e1_blocks=1, e2_blocks=1,
        context_channels=8, cell_channels=16, cell_blocks=1,
    )


def construct_model(
        device: torch.device, *, tiny: bool = False,
        ) -> DCTPlannedForcingMultipleShootingClosure:
    control.seed_everything(device)
    return DCTPlannedForcingMultipleShootingClosure(
        model_config(tiny=tiny), rollout_mode="carry_detached",
    ).float().to(device)


def _coefficient_free_mask(
        free_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if (free_mask.ndim != 4 or free_mask.shape[-1] != 1
            or free_mask.shape[0] != reference.shape[0]
            or free_mask.shape[1:3] != reference.shape[-2:]):
        raise ValueError("free mask must have shape [B,H,W,1]")
    return free_mask.permute(0, 3, 1, 2).unsqueeze(2).to(reference)


def coefficient_relative_error(
        prediction: torch.Tensor, truth: torch.Tensor,
        free_mask: torch.Tensor) -> torch.Tensor:
    """Per-sample free-fluid coefficient Frobenius-relative error."""
    if (prediction.ndim != 5 or prediction.shape[1:3] != (2, 4)
            or prediction.shape != truth.shape):
        raise ValueError(
            "coefficient tensors must share shape [B,2,4,H,W]")
    prediction_float = prediction.float()
    truth_float = truth.to(dtype=torch.float32)
    free = _coefficient_free_mask(free_mask, prediction_float)
    difference = (prediction_float - truth_float) * free
    target = truth_float * free
    numerator = torch.linalg.vector_norm(
        difference.reshape(difference.shape[0], -1), dim=1)
    denominator = torch.linalg.vector_norm(
        target.reshape(target.shape[0], -1), dim=1)
    return numerator / denominator.clamp_min(
        torch.finfo(denominator.dtype).eps)


def coefficient_auxiliary_objective(
        predicted_coefficients: torch.Tensor,
        target_normalized: torch.Tensor,
        free_mask: torch.Tensor,
        model: DCTPlannedForcingMultipleShootingClosure,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build label coefficients after rollout and use them only in loss."""
    if (target_normalized.ndim != 5
            or target_normalized.shape[1] != model.config.out_steps
            or target_normalized.shape[-1] < 2):
        raise ValueError("target must have shape [B,20,H,W,C>=2]")
    truth = model.plan_head.temporal_coefficients(
        target_normalized[..., :2]).detach()
    relative = coefficient_relative_error(
        predicted_coefficients, truth, free_mask)
    bounded = 1.0 - 1.0 / (
        1.0 + BOUNDED_RELATIVE_SCALE * relative)
    return bounded.mean(), relative, truth


def _masked_rms(
        value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(value)
    count = expanded.sum().clamp_min(1.0)
    return ((value.float().square() * expanded).sum() / count).sqrt()


def _gradient_norm(parameters) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


@torch.no_grad()
def coefficient_diagnostics(
        values: dict[str, torch.Tensor], truth: torch.Tensor,
        relative: torch.Tensor,
        ) -> dict[str, float]:
    """Materialize plan diagnostics only at an explicitly traced update."""
    predicted = values["predicted_coefficients"].float()
    truth_float = truth.float()
    coefficient_mask = _coefficient_free_mask(
        values["free_mask"], predicted)
    predicted_masked = predicted * coefficient_mask
    truth_masked = truth_float * coefficient_mask
    cosine = F.cosine_similarity(
        predicted_masked.flatten(1), truth_masked.flatten(1),
        dim=1, eps=torch.finfo(torch.float32).eps)

    forcing_normalized = values["planned_forcing_normalized"].float()
    forcing_physical = values["planned_forcing_physical"].float()
    forcing_mask = values["free_mask"].unsqueeze(1).to(
        forcing_normalized)
    return {
        "predicted_coefficient_free_rms": float(
            _masked_rms(predicted, coefficient_mask)),
        "truth_coefficient_free_rms": float(
            _masked_rms(truth_float, coefficient_mask)),
        "coefficient_free_cosine": float(cosine.mean()),
        "coefficient_relative_frobenius": float(relative.mean()),
        "forcing_normalized_free_rms": float(
            _masked_rms(forcing_normalized, forcing_mask)),
        "forcing_physical_free_rms": float(
            _masked_rms(forcing_physical, forcing_mask)),
    }


def train(
        model: DCTPlannedForcingMultipleShootingClosure, dataset, normalizer,
        device: torch.device, *, updates: int, batch_size: int,
        trace_interval: int,
        ) -> tuple[dict[str, object], ema_lib.FP32EMAShadow]:
    schedule = control.make_schedule(len(dataset), updates, batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UPDATES if updates == UPDATES else updates)
    ema = ema_lib.FP32EMAShadow(model, EMA_DECAY)
    trace: list[dict[str, object]] = []
    interval_loss_sum = torch.zeros((), device=device)
    interval_count = 0
    started = time.perf_counter()

    for update, indices in enumerate(schedule, start=1):
        should_report = (
            update == 1 or update % trace_interval == 0 or update == updates)
        items = [dataset[int(index)] for index in indices]
        history_physical = torch.stack([item[0] for item in items]).to(device)
        target_physical = torch.stack([item[1] for item in items]).to(device)
        history, target = normalizer.preprocess(
            history_physical, target_physical)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        # The rollout receives history only.  Truth coefficients are formed
        # afterward and are consumed solely by the auxiliary objective.
        values = model.forward_components(history)
        prediction = values["prediction"]
        accuracy, details = base.accuracy_objective(
            prediction, target, normalizer, LOSS_WEIGHTS)
        mse_uv = F.mse_loss(prediction[..., :2], target[..., :2])
        coefficient_loss, coefficient_relative, truth_coefficients = (
            coefficient_auxiliary_objective(
                values["predicted_coefficients"], target,
                values["free_mask"], model))
        loss = (
            accuracy + MSE_WEIGHT * mse_uv
            + COEFFICIENT_AUX_WEIGHT * coefficient_loss)
        loss.backward()
        if should_report:
            plan_head_gradient_norm = _gradient_norm(
                model.plan_head.parameters())
            feedback_gradient = model.cell.plan_stem.weight.grad
            feedback_gradient_rms = (
                float(feedback_gradient.detach().float().square().mean().sqrt())
                if feedback_gradient is not None else 0.0)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=update)
        scheduler.step()
        interval_loss_sum = interval_loss_sum + loss.detach()
        interval_count += 1

        if should_report:
            finite_scalars = torch.stack((
                loss.detach().float(), grad_norm.detach().float(),
                accuracy.detach().float(), mse_uv.detach().float(),
                coefficient_loss.detach().float(),
                coefficient_relative.detach().float().mean(),
            ))
            if not bool(torch.isfinite(finite_scalars).all()):
                raise FloatingPointError(
                    f"non-finite training value at update {update}")
            diagnostic = coefficient_diagnostics(
                values, truth_coefficients, coefficient_relative)
            row = {
                "update": update,
                "loss_mean_since_last_report": float(
                    interval_loss_sum / interval_count),
                "loss_current": float(loss.detach()),
                "accuracy": float(accuracy.detach()),
                "mse_uv": float(mse_uv.detach()),
                "coefficient_auxiliary": float(coefficient_loss.detach()),
                "plan_head_gradient_norm_before_clip": (
                    plan_head_gradient_norm),
                "plan_feedback_gradient_rms_before_clip": (
                    feedback_gradient_rms),
                "grad_norm_before_clip": float(grad_norm.detach()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "wall_seconds": time.perf_counter() - started,
                **{name: float(value) for name, value in details.items()},
                **diagnostic,
            }
            trace.append(row)
            interval_loss_sum.zero_()
            interval_count = 0
            print(json.dumps(
                {"event": "ms_dct_planned_forcing_progress", **row},
                sort_keys=True), flush=True)

    terminal_finite = all(
        not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in model.state_dict().values())
    if not terminal_finite:
        raise FloatingPointError("terminal model state is non-finite")
    return ({
        "updates": updates,
        "sample_stream_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "trace": trace,
        "wall_seconds": time.perf_counter() - started,
    }, ema)


def save_endpoint(
        path: Path, *, kind: str, state: dict[str, torch.Tensor],
        git: dict[str, object], normalizer, updates: int,
        ) -> dict[str, str]:
    atomic.atomic_torch_save({
        "schema": SCHEMA,
        "endpoint_kind": kind,
        "updates": updates,
        "rollout_mode": "carry_detached",
        "model_config": asdict(MODEL_CONFIG),
        "normalizer": control.normalizer_record(normalizer),
        "loss": (
            "accuracy(1,4,1)+0.25*normalized_uv_mse"
            "+(1/6)*bounded_free_coefficient_frobenius_relative"),
        "truth_coefficients_enter_rollout": False,
        "git": git,
        "model_state": state,
    }, path)
    return {
        "path": str(path.resolve()),
        "sha256": atomic.file_sha256(path),
        "model_state_sha256": base.state_sha(state),
    }


def evaluate_endpoint(
        model: DCTPlannedForcingMultipleShootingClosure,
        state: dict[str, torch.Tensor], trajectories, windows, files,
        normalizer, device: torch.device,
        ) -> dict[str, object]:
    model.load_state_dict(state, strict=True)
    model.eval()
    return loao._evaluate_same_model(
        model, trajectories, windows, files, normalizer, device)


def formal(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError(
            f"formal run requires --execute --acknowledge {ACKNOWLEDGEMENT}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    if args.sim_dir is None or args.output_dir is None:
        raise ValueError("formal training requires --sim-dir and --output-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    git = base.git_report()
    inventory = base.inventory(args.sim_dir)
    fit_rows = [row for row in inventory if int(row["aoa"]) in FIT_AOAS]
    held_rows = [row for row in inventory if int(row["aoa"]) == HELD_AOA]
    fit_trajectories, fit_windows, _fit_files = base.load_fields(
        args.sim_dir, fit_rows)
    normalizer = base.ChannelNormalizer(
        *base.window_statistics(fit_trajectories, fit_windows), device=device)
    dataset = base.SimWindowDataset(fit_trajectories, fit_windows)

    model = construct_model(device)
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    training, ema = train(
        model, dataset, normalizer, device,
        updates=UPDATES, batch_size=BATCH_SIZE,
        trace_interval=TRACE_INTERVAL)
    raw_state = control.snapshot(model)
    ema_state = dict(ema.materialize_model_state(model))
    checkpoints = {
        "raw": save_endpoint(
            args.output_dir / f"raw_u{UPDATES:05d}.pt",
            kind=f"raw_u{UPDATES:05d}", state=raw_state, git=git,
            normalizer=normalizer, updates=UPDATES),
        "ema": save_endpoint(
            args.output_dir / f"ema999_u{UPDATES:05d}.pt",
            kind=f"ema999_u{UPDATES:05d}", state=ema_state, git=git,
            normalizer=normalizer, updates=UPDATES),
    }

    del dataset, fit_trajectories
    gc.collect()
    torch.cuda.empty_cache()
    held_trajectories, held_windows, held_files = base.load_fields(
        args.sim_dir, held_rows)
    evaluations = {
        "raw": evaluate_endpoint(
            model, raw_state, held_trajectories, held_windows, held_files,
            normalizer, device),
        "ema999": evaluate_endpoint(
            model, ema_state, held_trajectories, held_windows, held_files,
            normalizer, device),
    }
    raw_aggregate = evaluations["raw"]["aggregate"]
    ema_aggregate = evaluations["ema999"]["aggregate"]
    original_scaled = {
        "rel_l2_score": 94.26758827253985,
        "tke_rel_l2_score": 81.66777310182073,
        "mvpe_rel_l2_score": 95.244431540729,
        "point_contribution_46_10_9": 60.10186675421602,
    }
    mature_reference = {
        key: value for key, value in control.MATURE_B210_REFERENCE.items()
        if key != "name"}
    comparisons = {
        "raw_minus_original_scaled_MS": base.comparison(
            raw_aggregate, original_scaled),
        "ema999_minus_original_scaled_MS": base.comparison(
            ema_aggregate, original_scaled),
        "raw_minus_mature_B210": base.comparison(
            raw_aggregate, mature_reference),
        "ema999_minus_mature_B210": base.comparison(
            ema_aggregate, mature_reference),
        "ema999_minus_raw": base.comparison(ema_aggregate, raw_aggregate),
    }
    result = {
        "schema": SCHEMA,
        "git": git,
        "training": {
            "fit_aoas": list(FIT_AOAS), "held_aoa": HELD_AOA,
            "fresh_initialization": True,
            "rollout_mode": "carry_detached",
            "model_config": asdict(MODEL_CONFIG),
            "architecture": model.architecture_config(),
            "parameters": sum(value.numel() for value in model.parameters()),
            "updates": UPDATES, "batch_size": BATCH_SIZE,
            "seed": SEED, "schedule_seed": SCHEDULE_SEED,
            "sample_schedule": "seeded_with_replacement_updates_x_batch",
            "optimizer": "AdamW", "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR", "scheduler_t_max": UPDATES,
            "grad_clip": GRAD_CLIP,
            "loss": (
                "accuracy(1,4,1)+0.25*normalized_uv_mse"
                "+(1/6)*bounded_free_coefficient_frobenius_relative"),
            "coefficient_relative_bounding": (
                "1-1/(1+0.5*relative_error)"),
            "truth_coefficients_enter_rollout": False,
            "ema_decay": EMA_DECAY,
            "trace_interval": TRACE_INTERVAL,
        },
        "normalizer": control.normalizer_record(normalizer),
        "training_result": training,
        "checkpoints": checkpoints,
        "original_scaled_MS_reference": original_scaled,
        "mature_B210_reference": control.MATURE_B210_REFERENCE,
        "held_AoA10": evaluations,
        "comparisons": comparisons,
        "selection": "none; raw and EMA are both reported",
    }
    atomic.atomic_json_save(result, args.output_dir / "result.json")
    print(json.dumps({
        "event": "ms_dct_planned_forcing_complete",
        "result": str((args.output_dir / "result.json").resolve()),
        "raw": raw_aggregate,
        "ema999": ema_aggregate,
        "comparisons": comparisons,
    }, sort_keys=True), flush=True)
    return result


def smoke(device: torch.device) -> dict[str, object]:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    model = construct_model(device, tiny=True)
    trajectory = torch.randn(40, 16, 16, 3)
    dataset = base.SimWindowDataset(
        [trajectory], [base.SimWindow(0, 0), base.SimWindow(0, 0)])
    normalizer = base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), device)
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    training, ema = train(
        model, dataset, normalizer, device,
        updates=2, batch_size=1, trace_interval=1)
    raw = model(dataset[0][0][None].to(device))
    model.load_state_dict(ema.materialize_model_state(model), strict=True)
    averaged = model(dataset[0][0][None].to(device))
    return {
        "schema": SCHEMA,
        "smoke": "PASS",
        "device": str(device),
        "training": training,
        "raw_finite": bool(torch.isfinite(raw).all()),
        "ema_finite": bool(torch.isfinite(averaged).all()),
        "output_shape": list(raw.shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("contract", "smoke", "formal"),
        default="contract")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--device", default="cuda:0",
        help="cpu or cuda:N; both are supported by smoke mode")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        model = construct_model(torch.device("cpu"))
        print(json.dumps({
            "schema": SCHEMA,
            "acknowledgement": ACKNOWLEDGEMENT,
            "model_config": asdict(MODEL_CONFIG),
            "parameters": sum(value.numel() for value in model.parameters()),
            "training": {
                "fit_aoas": list(FIT_AOAS), "held_aoa": HELD_AOA,
                "updates": UPDATES, "batch_size": BATCH_SIZE,
                "seed": SEED, "schedule_seed": SCHEDULE_SEED,
                "lr": LR, "weight_decay": WEIGHT_DECAY,
                "scheduler_t_max": UPDATES, "ema_decay": EMA_DECAY,
                "coefficient_auxiliary_weight": COEFFICIENT_AUX_WEIGHT,
                "truth_coefficients_enter_rollout": False,
            },
        }, sort_keys=True, indent=2))
        return
    device = torch.device(args.device)
    result = smoke(device) if args.mode == "smoke" else formal(args, device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
