#!/usr/bin/env python3
"""Fresh 23M MS-Closure training with dense fit-window phase coverage.

The scaled MS control observes 20->20 windows whose starts are spaced by 20
frames.  This experiment changes only the training sample support to stride 5.
It retains the sparse fit-window normalizer, model, seed, optimizer, 12k
schedule, loss, EMA, and stride-20 held-AoA10 evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_multiple_shooting_closure_scaled_long as control  # noqa: E402
import train_pyramid_causal3d_sim_long as atomic  # noqa: E402


SCHEMA = "multiple_shooting_closure_dense_stride5_long_held10_v1"
ACKNOWLEDGEMENT = "RUN_MS_CLOSURE_DENSE_STRIDE5_LONG_HELD10"
TRAIN_STRIDE = 5
HELD_STRIDE = 20


def dense_windows(trajectories: list[torch.Tensor]) -> list[base.SimWindow]:
    """Use the control's exclusive final-start convention at stride five."""
    windows = []
    for trajectory_index, trajectory in enumerate(trajectories):
        for start in range(0, len(trajectory) - 40, TRAIN_STRIDE):
            windows.append(base.SimWindow(trajectory_index, start))
    if not windows:
        raise ValueError("dense training window set is empty")
    return windows


def window_manifest(windows: list[base.SimWindow]) -> dict[str, object]:
    array = np.asarray(
        [(window.trajectory, window.start) for window in windows],
        dtype=np.int64)
    return {
        "windows": len(windows),
        "stride": TRAIN_STRIDE,
        "trajectory_start_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "first": array[0].tolist(),
        "last": array[-1].tolist(),
    }


def save_endpoint(
        path: Path, *, kind: str, state: dict[str, torch.Tensor],
        git: dict[str, object], normalizer, updates: int,
        manifest: dict[str, object],
        ) -> dict[str, str]:
    atomic.atomic_torch_save({
        "schema": SCHEMA,
        "endpoint_kind": kind,
        "updates": updates,
        "rollout_mode": "carry_detached",
        "model_config": asdict(control.MODEL_CONFIG),
        "normalizer": control.normalizer_record(normalizer),
        "training_window_manifest": manifest,
        "git": git,
        "model_state": state,
    }, path)
    return {
        "path": str(path.resolve()),
        "sha256": atomic.file_sha256(path),
        "model_state_sha256": base.state_sha(state),
    }


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
    fit_rows = [
        row for row in inventory if int(row["aoa"]) in control.FIT_AOAS]
    held_rows = [
        row for row in inventory if int(row["aoa"]) == control.HELD_AOA]
    fit_trajectories, sparse_fit_windows, _fit_files = base.load_fields(
        args.sim_dir, fit_rows)
    # Preserve the control's normalization coordinates exactly.  Only the
    # optimizer's set of sampled window starts changes in this experiment.
    normalizer = base.ChannelNormalizer(
        *base.window_statistics(fit_trajectories, sparse_fit_windows),
        device=device)
    training_windows = dense_windows(fit_trajectories)
    manifest = window_manifest(training_windows)
    manifest["fit_inventory_sha256"] = base.canonical_sha(fit_rows)
    dataset = base.SimWindowDataset(fit_trajectories, training_windows)

    model = control.construct_model(device)
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    training, ema = control.train(
        model, dataset, normalizer, device,
        updates=control.UPDATES, batch_size=control.BATCH_SIZE,
        trace_interval=control.TRACE_INTERVAL)
    raw_state = control.snapshot(model)
    ema_state = dict(ema.materialize_model_state(model))
    checkpoints = {
        "raw": save_endpoint(
            args.output_dir / f"raw_u{control.UPDATES:05d}.pt",
            kind=f"raw_u{control.UPDATES:05d}", state=raw_state, git=git,
            normalizer=normalizer, updates=control.UPDATES,
            manifest=manifest),
        "ema": save_endpoint(
            args.output_dir / f"ema999_u{control.UPDATES:05d}.pt",
            kind=f"ema999_u{control.UPDATES:05d}", state=ema_state, git=git,
            normalizer=normalizer, updates=control.UPDATES,
            manifest=manifest),
    }

    del dataset, fit_trajectories
    gc.collect()
    torch.cuda.empty_cache()
    held_trajectories, held_windows, held_files = base.load_fields(
        args.sim_dir, held_rows)
    evaluations = {
        "raw": control.evaluate_endpoint(
            model, raw_state, held_trajectories, held_windows, held_files,
            normalizer, device),
        "ema999": control.evaluate_endpoint(
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
        "raw_minus_sparse_scaled_MS": base.comparison(
            raw_aggregate, original_scaled),
        "ema999_minus_sparse_scaled_MS": base.comparison(
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
            "fit_aoas": list(control.FIT_AOAS),
            "held_aoa": control.HELD_AOA,
            "fresh_initialization": True,
            "training_stride": TRAIN_STRIDE,
            "held_stride": HELD_STRIDE,
            "normalizer_window_stride": HELD_STRIDE,
            "training_window_manifest": manifest,
            "sparse_control_windows": len(sparse_fit_windows),
            "released_fit_trajectories_only": True,
            "rollout_mode": "carry_detached",
            "model_config": asdict(control.MODEL_CONFIG),
            "architecture": model.architecture_config(),
            "parameters": sum(value.numel() for value in model.parameters()),
            "updates": control.UPDATES,
            "batch_size": control.BATCH_SIZE,
            "seed": control.SEED,
            "schedule_seed": control.SCHEDULE_SEED,
            "optimizer": "AdamW",
            "lr": control.LR,
            "weight_decay": control.WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": control.UPDATES,
            "grad_clip": control.GRAD_CLIP,
            "loss": "accuracy(1,4,1)+0.25*normalized_uv_mse",
            "ema_decay": control.EMA_DECAY,
            "trace_interval": control.TRACE_INTERVAL,
        },
        "normalizer": control.normalizer_record(normalizer),
        "training_result": training,
        "checkpoints": checkpoints,
        "sparse_scaled_MS_reference": original_scaled,
        "mature_B210_reference": control.MATURE_B210_REFERENCE,
        "held_AoA10": evaluations,
        "comparisons": comparisons,
        "selection": "none; raw and EMA are both reported",
    }
    atomic.atomic_json_save(result, args.output_dir / "result.json")
    print(json.dumps({
        "event": "ms_closure_dense_stride5_complete",
        "result": str((args.output_dir / "result.json").resolve()),
        "raw": raw_aggregate,
        "ema999": ema_aggregate,
        "comparisons": comparisons,
    }, sort_keys=True), flush=True)
    return result


def smoke(device: torch.device) -> dict[str, object]:
    model = control.construct_model(device, tiny=True)
    trajectory = torch.randn(60, 16, 16, 3)
    trajectories = [trajectory]
    windows = dense_windows(trajectories)
    dataset = base.SimWindowDataset(trajectories, windows)
    normalizer = base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), device)
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    training, ema = control.train(
        model, dataset, normalizer, device,
        updates=2, batch_size=1, trace_interval=1)
    raw = model(dataset[0][0][None].to(device))
    model.load_state_dict(ema.materialize_model_state(model), strict=True)
    averaged = model(dataset[0][0][None].to(device))
    return {
        "schema": SCHEMA,
        "smoke": "PASS",
        "device": str(device),
        "training_window_manifest": window_manifest(windows),
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
        model = control.construct_model(torch.device("cpu"))
        print(json.dumps({
            "schema": SCHEMA,
            "acknowledgement": ACKNOWLEDGEMENT,
            "training_stride": TRAIN_STRIDE,
            "held_stride": HELD_STRIDE,
            "normalizer_window_stride": HELD_STRIDE,
            "model_config": asdict(control.MODEL_CONFIG),
            "parameters": sum(value.numel() for value in model.parameters()),
            "training": {
                "fit_aoas": list(control.FIT_AOAS),
                "held_aoa": control.HELD_AOA,
                "updates": control.UPDATES,
                "batch_size": control.BATCH_SIZE,
                "seed": control.SEED,
                "schedule_seed": control.SCHEDULE_SEED,
                "lr": control.LR,
                "weight_decay": control.WEIGHT_DECAY,
                "scheduler_t_max": control.UPDATES,
                "ema_decay": control.EMA_DECAY,
            },
        }, sort_keys=True, indent=2))
        return
    device = torch.device(args.device)
    result = smoke(device) if args.mode == "smoke" else formal(args, device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
