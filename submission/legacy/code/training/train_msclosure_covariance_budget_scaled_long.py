#!/usr/bin/env python3
"""Fresh 23M covariance-budget MS-Closure training and held-AoA10 report.

The generated trajectory carries its own Welford covariance state.  A causal
history head predicts a local PSD final covariance budget, while every closure
step reads the consumed and remaining budget.  Future truth never enters the
rollout; it supervises the budget head only through a fit-only, dimensionless
Frobenius-relative auxiliary objective.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
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
from models.msclosure_covariance_budget import (  # noqa: E402
    CovarianceBudgetMultipleShootingClosure,
    CovarianceBudgetMultipleShootingConfig,
)


SCHEMA = "msclosure_covariance_budget_scaled_long_held10_v1"
ACKNOWLEDGEMENT = "RUN_MS_COVARIANCE_BUDGET_SCALED_LONG_HELD10"
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

# The existing accuracy objective contains six weight units (1, 4, 1).  The
# covariance task uses the same bounded score mapping and contributes one
# additional unit, rather than introducing a scale-dependent raw MSE weight.
COVARIANCE_AUX_WEIGHT = 1.0 / sum(LOSS_WEIGHTS)

MODEL_CONFIG = CovarianceBudgetMultipleShootingConfig(
    e0_channels=128,
    e1_channels=256,
    e2_channels=512,
    context_channels=128,
    cell_channels=192,
    cell_blocks=2,
)


def model_config(*, tiny: bool = False) -> CovarianceBudgetMultipleShootingConfig:
    if not tiny:
        return MODEL_CONFIG
    return CovarianceBudgetMultipleShootingConfig(
        e0_channels=8, e1_channels=16, e2_channels=24,
        e0_blocks=1, e1_blocks=1, e2_blocks=1,
        context_channels=8, cell_channels=16, cell_blocks=1,
    )


def construct_model(
        device: torch.device, *, tiny: bool = False,
        ) -> CovarianceBudgetMultipleShootingClosure:
    control.seed_everything(device)
    return CovarianceBudgetMultipleShootingClosure(
        model_config(tiny=tiny), rollout_mode="carry_detached",
    ).float().to(device)


def _free_mask(history_uv: torch.Tensor) -> torch.Tensor:
    """Match the model's history-derived solid and outer-boundary mask."""
    if history_uv.ndim != 5 or history_uv.shape[-1] != 2:
        raise ValueError("history UV must have shape [B,T,H,W,2]")
    solid = history_uv.abs().amax(dim=(1, 4)).unsqueeze(-1) <= 1.0e-6
    outer = torch.zeros_like(solid)
    outer[:, 0, :, :] = True
    outer[:, -1, :, :] = True
    outer[:, :, 0, :] = True
    outer[:, :, -1, :] = True
    return ~(solid | outer)


def population_covariance_packed(value: torch.Tensor) -> torch.Tensor:
    """Return population covariance as packed [uu,uv,vv]."""
    if value.ndim != 5 or value.shape[-1] != 2:
        raise ValueError("velocity sequence must have shape [B,T,H,W,2]")
    working = value if value.dtype == torch.float64 else value.float()
    centered = working - working.mean(dim=1, keepdim=True)
    return torch.stack((
        centered[..., 0].square().mean(dim=1),
        (centered[..., 0] * centered[..., 1]).mean(dim=1),
        centered[..., 1].square().mean(dim=1),
    ), dim=-1)


def whiten_velocity(value: torch.Tensor, cholesky: torch.Tensor) -> torch.Tensor:
    """Whiten target-normalized UV; temporal translation is immaterial."""
    value = value.float()
    factor = cholesky.to(device=value.device, dtype=torch.float32)
    white_u = value[..., 0] / factor[0, 0]
    white_v = (value[..., 1] - factor[1, 0] * white_u) / factor[1, 1]
    return torch.stack((white_u, white_v), dim=-1)


def white_population_covariance_packed(
        value: torch.Tensor, cholesky: torch.Tensor) -> torch.Tensor:
    return population_covariance_packed(whiten_velocity(value, cholesky))


def packed_matrix(packed: torch.Tensor) -> torch.Tensor:
    row_u = torch.stack((packed[..., 0], packed[..., 1]), dim=-1)
    row_v = torch.stack((packed[..., 1], packed[..., 2]), dim=-1)
    return torch.stack((row_u, row_v), dim=-2)


@torch.no_grad()
def fit_covariance_reference(dataset, normalizer) -> tuple[torch.Tensor, dict[str, object]]:
    """Compute one FP64 covariance reference from fit targets/free pixels."""
    mean_target = normalizer.mean_target[:2].detach().cpu().double()
    std_target = normalizer.std_target[:2].detach().cpu().double()
    total = torch.zeros(3, dtype=torch.float64)
    free_pixel_windows = 0
    for index in range(len(dataset)):
        history, target = dataset[index]
        target_normalized = (
            (target[..., :2].double() - mean_target) / std_target)
        covariance = population_covariance_packed(
            target_normalized.unsqueeze(0)).double()[0]
        free = _free_mask(history[None, ..., :2])[0, ..., 0]
        total += covariance[free].sum(dim=0)
        free_pixel_windows += int(free.sum())
    if free_pixel_windows <= 0:
        raise RuntimeError("fit covariance reference has no free pixels")
    packed = total / free_pixel_windows
    matrix = packed_matrix(packed).double()
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    if float(eigenvalues.min()) <= 0:
        raise RuntimeError(f"fit covariance reference is not PD: {matrix}")
    cholesky = torch.linalg.cholesky(matrix)
    report = {
        "definition": (
            "mean fit-window population covariance of target-normalized UV "
            "over free pixels"),
        "fit_aoas": list(FIT_AOAS),
        "windows": len(dataset),
        "free_pixel_windows": free_pixel_windows,
        "matrix": matrix.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "cholesky": cholesky.tolist(),
        "accumulation_dtype": "torch.float64",
    }
    return matrix.float(), report


def _frob_vector(packed: torch.Tensor) -> torch.Tensor:
    # This vector norm is exactly the Frobenius norm of a symmetric 2x2 matrix.
    return torch.stack((
        packed[..., 0], packed[..., 2],
        math.sqrt(2.0) * packed[..., 1],
    ), dim=-1)


def covariance_relative_error(
        prediction: torch.Tensor, truth: torch.Tensor,
        free_mask: torch.Tensor) -> torch.Tensor:
    """Per-sample free-fluid Frobenius-relative covariance error."""
    free = free_mask.to(dtype=torch.float32)
    difference = _frob_vector(prediction - truth) * free
    target = _frob_vector(truth) * free
    # ``sqrt(sum(x**2))`` has an undefined hand-written derivative at x=0.
    # vector_norm deliberately supplies the correct zero subgradient there.
    numerator = torch.linalg.vector_norm(
        difference.reshape(difference.shape[0], -1), dim=1)
    denominator = torch.linalg.vector_norm(
        target.reshape(target.shape[0], -1), dim=1)
    valid = denominator > torch.finfo(torch.float32).tiny
    if not bool(valid.any()):
        raise RuntimeError("batch contains no nonzero free-fluid covariance")
    return (
        numerator[valid] / denominator[valid])


def covariance_auxiliary_objective(
        predicted_budget_white: torch.Tensor,
        target_normalized: torch.Tensor,
        free_mask: torch.Tensor,
        cholesky: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    truth_white = white_population_covariance_packed(
        target_normalized[..., :2], cholesky)
    relative = covariance_relative_error(
        predicted_budget_white, truth_white, free_mask)
    loss = (1.0 - 1.0 / (1.0 + 0.5 * relative)).mean()
    return loss, relative.mean().detach(), truth_white


def _gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(total)


@torch.no_grad()
def covariance_diagnostics(
        model: CovarianceBudgetMultipleShootingClosure,
        values: dict[str, torch.Tensor], target: torch.Tensor,
        truth_white: torch.Tensor,
        ) -> dict[str, object]:
    free = values["free_mask"].to(torch.float32)
    predicted = values["predicted_budget_white_packed"]
    realized = values["realized_covariance_white_packed"]
    direct = white_population_covariance_packed(
        values["prediction"][..., :2],
        model.covariance_reference_cholesky)
    budget_truth = covariance_relative_error(predicted, truth_white, free)
    realized_truth = covariance_relative_error(realized, truth_white, free)
    # Normalize closure tracking error by the nonzero predicted budget.  The
    # persistence initialization has exactly zero realized covariance.
    budget_realized = covariance_relative_error(realized, predicted, free)
    selected = free[..., 0].bool()
    welford_difference = (realized - direct).abs()[selected]

    residual_matrix = packed_matrix(values["budget_residual_white_packed"])
    residual_minimum = torch.linalg.eigvalsh(residual_matrix)[..., 0]
    checkpoint_trace = values["budget_residual_checkpoints"].diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)
    checkpoint_means = [
        float(checkpoint_trace[:, index][selected].mean())
        for index in range(checkpoint_trace.shape[1])
    ]
    return {
        "budget_truth_relative_frobenius": float(budget_truth.mean()),
        "realized_truth_relative_frobenius": float(realized_truth.mean()),
        "budget_realized_relative_frobenius": float(budget_realized.mean()),
        "welford_vs_direct_white_max_abs": (
            float(welford_difference.max()) if welford_difference.numel() else 0.0),
        "residual_min_eigenvalue_free_mean": float(
            residual_minimum[selected].mean()),
        "residual_overspent_free_fraction": float(
            (residual_minimum[selected] < 0).float().mean()),
        "target_normalized_residual_trace_free_mean_at_5_10_15_20": (
            checkpoint_means),
        "predicted_budget_white_trace_free_mean": float(
            (predicted[..., 0] + predicted[..., 2])[selected].mean()),
        "truth_budget_white_trace_free_mean": float(
            (truth_white[..., 0] + truth_white[..., 2])[selected].mean()),
        "realized_white_trace_free_mean": float(
            (realized[..., 0] + realized[..., 2])[selected].mean()),
        "target_shape": list(target.shape),
    }


def train(
        model: CovarianceBudgetMultipleShootingClosure, dataset, normalizer,
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
    interval: list[float] = []
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
        values = model.forward_components(history)
        prediction = values["prediction"]
        accuracy, details = base.accuracy_objective(
            prediction, target, normalizer, LOSS_WEIGHTS)
        mse_uv = F.mse_loss(prediction[..., :2], target[..., :2])
        covariance_loss, covariance_relative, truth_white = (
            covariance_auxiliary_objective(
                values["predicted_budget_white_packed"], target,
                values["free_mask"],
                model.covariance_reference_cholesky))
        loss = (
            accuracy + MSE_WEIGHT * mse_uv
            + COVARIANCE_AUX_WEIGHT * covariance_loss)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        if should_report:
            budget_gradient_norm = _gradient_norm(
                model.covariance_budget.parameters())
            stem_gradient = model.cell.stem.weight.grad
            feedback_start = 2 + model.config.context_channels
            feedback_end = feedback_start + 8
            feedback_stem_gradient_rms = (
                float(stem_gradient[:, feedback_start:feedback_end]
                      .detach().float().square().mean().sqrt())
                if stem_gradient is not None else 0.0)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient at update {update}")
        optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=update)
        scheduler.step()
        interval.append(float(loss.detach()))

        if should_report:
            diagnostic = covariance_diagnostics(
                model, values, target, truth_white)
            row = {
                "update": update,
                "loss_mean_since_last_report": float(np.mean(interval)),
                "loss_current": float(loss.detach()),
                "accuracy": float(accuracy.detach()),
                "mse_uv": float(mse_uv.detach()),
                "covariance_auxiliary": float(covariance_loss.detach()),
                "covariance_relative_frobenius": float(covariance_relative),
                "budget_head_gradient_norm_before_clip": budget_gradient_norm,
                "feedback_stem_gradient_rms_before_clip": (
                    feedback_stem_gradient_rms),
                "grad_norm_before_clip": float(grad_norm.detach()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "wall_seconds": time.perf_counter() - started,
                **{name: float(value) for name, value in details.items()},
                **diagnostic,
            }
            trace.append(row)
            interval.clear()
            print(json.dumps(
                {"event": "ms_covariance_budget_progress", **row},
                sort_keys=True), flush=True)

    return ({
        "updates": updates,
        "sample_stream_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "trace": trace,
        "wall_seconds": time.perf_counter() - started,
    }, ema)


def save_endpoint(
        path: Path, *, kind: str, state: dict[str, torch.Tensor],
        git: dict[str, object], normalizer, covariance_reference: dict[str, object],
        updates: int,
        ) -> dict[str, str]:
    atomic.atomic_torch_save({
        "schema": SCHEMA,
        "endpoint_kind": kind,
        "updates": updates,
        "rollout_mode": "carry_detached",
        "model_config": asdict(MODEL_CONFIG),
        "normalizer": control.normalizer_record(normalizer),
        "covariance_reference": covariance_reference,
        "loss": (
            "accuracy(1,4,1)+0.25*normalized_uv_mse"
            "+(1/6)*bounded_relative_covariance_frobenius"),
        "git": git,
        "model_state": state,
    }, path)
    return {
        "path": str(path.resolve()),
        "sha256": atomic.file_sha256(path),
        "model_state_sha256": base.state_sha(state),
    }


def evaluate_endpoint(
        model: CovarianceBudgetMultipleShootingClosure,
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
    covariance_reference, reference_report = fit_covariance_reference(
        dataset, normalizer)

    model = construct_model(device)
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    model.set_covariance_reference(covariance_reference)
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
            normalizer=normalizer, covariance_reference=reference_report,
            updates=UPDATES),
        "ema": save_endpoint(
            args.output_dir / f"ema999_u{UPDATES:05d}.pt",
            kind=f"ema999_u{UPDATES:05d}", state=ema_state, git=git,
            normalizer=normalizer, covariance_reference=reference_report,
            updates=UPDATES),
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
    mature_reference = {
        key: value for key, value in control.MATURE_B210_REFERENCE.items()
        if key != "name"}
    original_scaled = {
        "rel_l2_score": 94.26758827253985,
        "tke_rel_l2_score": 81.66777310182073,
        "mvpe_rel_l2_score": 95.244431540729,
        "point_contribution_46_10_9": 60.10186675421602,
    }
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
            "optimizer": "AdamW", "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR", "scheduler_t_max": UPDATES,
            "grad_clip": GRAD_CLIP,
            "loss": (
                "accuracy(1,4,1)+0.25*normalized_uv_mse"
                "+(1/6)*bounded_relative_covariance_frobenius"),
            "covariance_truth_enters_rollout": False,
            "ema_decay": EMA_DECAY,
            "trace_interval": TRACE_INTERVAL,
        },
        "normalizer": control.normalizer_record(normalizer),
        "covariance_reference": reference_report,
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
        "event": "ms_covariance_budget_complete",
        "result": str((args.output_dir / "result.json").resolve()),
        "raw": raw_aggregate,
        "ema999": ema_aggregate,
        "comparisons": comparisons,
    }, sort_keys=True), flush=True)
    return result


def smoke(device: torch.device) -> dict[str, object]:
    model = construct_model(device, tiny=True)
    model.set_covariance_reference(torch.tensor(
        ((0.20, 0.04), (0.04, 0.35)), device=device))
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
    parser.add_argument("--device", default="cuda:0")
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
                "updates": UPDATES, "batch_size": BATCH_SIZE,
                "lr": LR, "weight_decay": WEIGHT_DECAY,
                "scheduler_t_max": UPDATES, "ema_decay": EMA_DECAY,
                "covariance_auxiliary_weight": COVARIANCE_AUX_WEIGHT,
            },
        }, sort_keys=True, indent=2))
        return
    device = torch.device(args.device)
    result = smoke(device) if args.mode == "smoke" else formal(args, device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
