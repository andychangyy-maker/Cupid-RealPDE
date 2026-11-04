#!/usr/bin/env python3
"""Fresh scaled carry-only MS-Closure training and held-AoA10 report.

This is intentionally a small experiment driver: one model, one fixed recipe,
and no score gate.  It trains only on AoA 0/5/15/20, saves the raw and FP32
EMA endpoints, then evaluates both endpoints on held AoA10 with the existing
LOAO evaluator.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_pyramid_causal3d_loao_screen as loao  # noqa: E402
import train_pyramid_causal3d_sim_long as atomic  # noqa: E402
from models.multiple_shooting_closure import (  # noqa: E402
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
)


SCHEMA = "multiple_shooting_closure_scaled_long_held10_v1"
ACKNOWLEDGEMENT = "RUN_MS_CLOSURE_SCALED_LONG_HELD10"
FIT_AOAS = (0, 5, 15, 20)
HELD_AOA = 10
SEED = 210
SCHEDULE_SEED = 220
UPDATES = 12_000
BATCH_SIZE = 8
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
TRACE_INTERVAL = 1_000
LOSS_WEIGHTS = (1.0, 4.0, 1.0)
MSE_WEIGHT = 0.25

MODEL_CONFIG = MultipleShootingClosureConfig(
    e0_channels=128,
    e1_channels=256,
    e2_channels=512,
    context_channels=128,
    cell_channels=192,
    cell_blocks=2,
)

SMALL_M17_REFERENCE = {
    "name": "M17_small_carry_u2000_held_AoA10",
    "rel_l2_score": 93.78858160389613,
    "tke_rel_l2_score": 80.90057543724068,
    "mvpe_rel_l2_score": 94.702652522688,
    "point_contribution_46_10_9": 59.75604380855821,
}

MATURE_B210_REFERENCE = {
    "name": "mature_B210_same_held_AoA10_protocol",
    "rel_l2_score": 94.218598,
    "tke_rel_l2_score": 82.846022,
    "mvpe_rel_l2_score": 95.230467,
    "point_contribution_46_10_9": 60.195899,
}


def seed_everything(device: torch.device) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def model_config(*, tiny: bool = False) -> MultipleShootingClosureConfig:
    if not tiny:
        return MODEL_CONFIG
    return MultipleShootingClosureConfig(
        e0_channels=8, e1_channels=16, e2_channels=24,
        e0_blocks=1, e1_blocks=1, e2_blocks=1,
        context_channels=8, cell_channels=16, cell_blocks=1,
    )


def construct_model(device: torch.device, *, tiny: bool = False) -> MultipleShootingClosure:
    seed_everything(device)
    return MultipleShootingClosure(
        model_config(tiny=tiny), rollout_mode="carry_detached").float().to(device)


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def normalizer_record(normalizer) -> dict[str, list[float]]:
    return {
        name: getattr(normalizer, name).detach().cpu().tolist()
        for name in ("mean_input", "std_input", "mean_target", "std_target")
    }


def make_schedule(length: int, updates: int, batch_size: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("training dataset is empty")
    return np.random.default_rng(SCHEDULE_SEED).integers(
        0, length, size=(updates, batch_size), dtype=np.int64)


def train(
        model: MultipleShootingClosure, dataset, normalizer,
        device: torch.device, *, updates: int, batch_size: int,
        trace_interval: int,
        ) -> tuple[dict[str, object], ema_lib.FP32EMAShadow]:
    schedule = make_schedule(len(dataset), updates, batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UPDATES if updates == UPDATES else updates)
    ema = ema_lib.FP32EMAShadow(model, EMA_DECAY)
    trace: list[dict[str, object]] = []
    interval: list[float] = []
    started = time.perf_counter()

    for update, indices in enumerate(schedule, start=1):
        items = [dataset[int(index)] for index in indices]
        history = torch.stack([item[0] for item in items]).to(device)
        target = torch.stack([item[1] for item in items]).to(device)
        history, target = normalizer.preprocess(history, target)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        accuracy, details = base.accuracy_objective(
            prediction, target, normalizer, LOSS_WEIGHTS)
        mse_uv = (prediction[..., :2] - target[..., :2]).square().mean()
        loss = accuracy + MSE_WEIGHT * mse_uv
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient at update {update}")
        optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=update)
        scheduler.step()
        interval.append(float(loss.detach()))

        if update == 1 or update % trace_interval == 0 or update == updates:
            row = {
                "update": update,
                "loss_mean_since_last_report": float(np.mean(interval)),
                "accuracy": float(accuracy.detach()),
                "mse_uv": float(mse_uv.detach()),
                "grad_norm_before_clip": float(grad_norm.detach()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "wall_seconds": time.perf_counter() - started,
                **{name: float(value) for name, value in details.items()},
            }
            trace.append(row)
            interval.clear()
            print(json.dumps(
                {"event": "ms_closure_scaled_long_progress", **row},
                sort_keys=True), flush=True)

    if not all(
            not value.is_floating_point() or bool(torch.isfinite(value).all())
            for value in model.state_dict().values()):
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
        "normalizer": normalizer_record(normalizer),
        "git": git,
        "model_state": state,
    }, path)
    return {
        "path": str(path.resolve()),
        "sha256": atomic.file_sha256(path),
        "model_state_sha256": base.state_sha(state),
    }


def evaluate_endpoint(
        model: MultipleShootingClosure, state: dict[str, torch.Tensor],
        trajectories, windows, files, normalizer, device: torch.device,
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
    raw_state = snapshot(model)
    ema_state = ema.materialize_model_state(model)
    checkpoints = {
        "raw": save_endpoint(
            args.output_dir / f"raw_u{UPDATES:05d}.pt",
            kind=f"raw_u{UPDATES:05d}",
            state=raw_state, git=git, normalizer=normalizer, updates=UPDATES),
        "ema": save_endpoint(
            args.output_dir / f"ema999_u{UPDATES:05d}.pt",
            kind=f"ema999_u{UPDATES:05d}",
            state=dict(ema_state), git=git, normalizer=normalizer,
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
            model, dict(ema_state), held_trajectories, held_windows, held_files,
            normalizer, device),
    }
    raw_aggregate = evaluations["raw"]["aggregate"]
    ema_aggregate = evaluations["ema999"]["aggregate"]
    reference = {key: value for key, value in SMALL_M17_REFERENCE.items()
                 if key != "name"}
    mature_reference = {
        key: value for key, value in MATURE_B210_REFERENCE.items()
        if key != "name"}
    comparisons = {
        "raw_minus_small_M17": base.comparison(raw_aggregate, reference),
        "ema999_minus_small_M17": base.comparison(ema_aggregate, reference),
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
            "parameters": sum(value.numel() for value in model.parameters()),
            "updates": UPDATES, "batch_size": BATCH_SIZE,
            "optimizer": "AdamW", "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR", "scheduler_t_max": UPDATES,
            "grad_clip": GRAD_CLIP,
            "loss": "accuracy(1,4,1)+0.25*normalized_uv_mse",
            "ema_decay": EMA_DECAY,
            "trace_interval": TRACE_INTERVAL,
        },
        "normalizer": normalizer_record(normalizer),
        "training_result": training,
        "checkpoints": checkpoints,
        "small_M17_reference": SMALL_M17_REFERENCE,
        "mature_B210_reference": MATURE_B210_REFERENCE,
        "held_AoA10": evaluations,
        "comparisons": comparisons,
        "selection": "none; raw and EMA are both reported",
    }
    atomic.atomic_json_save(result, args.output_dir / "result.json")
    print(json.dumps({
        "event": "ms_closure_scaled_long_complete",
        "result": str((args.output_dir / "result.json").resolve()),
        "raw": raw_aggregate,
        "ema999": ema_aggregate,
        "comparisons": comparisons,
    }, sort_keys=True), flush=True)
    return result


def smoke(device: torch.device) -> dict[str, object]:
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
            },
        }, sort_keys=True, indent=2))
        return
    device = torch.device(args.device)
    result = smoke(device) if args.mode == "smoke" else formal(args, device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
