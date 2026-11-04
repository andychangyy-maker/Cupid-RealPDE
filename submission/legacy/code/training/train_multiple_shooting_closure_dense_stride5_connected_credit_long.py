#!/usr/bin/env python3
"""Fresh dense-stride5 MS-Closure training with connected segment credit.

The completed dense-stride5 run is the matched control.  This candidate keeps
its released-data inventory, stride-five training windows, sparse-window
normalizer, deterministic sample stream, 23M model, loss, optimizer, schedule,
EMA, and stride-twenty held-AoA10 evaluation.  Only the autograd connection of
the carried physical state across shooting segments changes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_multiple_shooting_closure_connected_credit_scaled_long as connected  # noqa: E402
import train_multiple_shooting_closure_dense_stride5_long as dense  # noqa: E402
import train_pyramid_causal3d_sim_long as atomic  # noqa: E402


control = dense.control
SCHEMA = "multiple_shooting_dense_stride5_connected_credit_held10_v1"
ACKNOWLEDGEMENT = "RUN_MS_DENSE5_CONNECTED_CREDIT_HELD10"
TRAIN_STRIDE = dense.TRAIN_STRIDE
HELD_STRIDE = dense.HELD_STRIDE

DENSE_CONTROL_REFERENCE = {
    "name": "dense_stride5_MS_23M_u12000_held_AoA10",
    "training_window_manifest_sha256": (
        "ac3969636101f6363bd04380aee8c8e3e0abf1fd5af2ba7576000e62a2d9af89"),
    "normalizer_sha256": (
        "bc8f8123745de6e1d8a1cc898283c48cf77e62c749ea0aaf2cfe4066690c99cf"),
    "sample_stream_sha256": (
        "118d296c3421ff75ec71ffd867aea35bf218622bf258477aadba9db820658cde"),
    "raw": {
        "rel_l2_score": 94.62055367684444,
        "tke_rel_l2_score": 82.72618109070918,
        "mvpe_rel_l2_score": 95.6187558497288,
        "point_contribution_46_10_9": 60.40376082689495,
    },
    "ema999": {
        "rel_l2_score": 94.62621547576379,
        "tke_rel_l2_score": 82.75426431149157,
        "mvpe_rel_l2_score": 95.62265625273531,
        "point_contribution_46_10_9": 60.40952461274668,
    },
}


def construct_model(
        device: torch.device, *, tiny: bool = False,
        ) -> connected.MultipleShootingClosure:
    return connected.construct_model(device, tiny=tiny)


def construct_matched_candidate(
        device: torch.device, *, tiny: bool = False,
        ) -> tuple[connected.MultipleShootingClosure, dict[str, object]]:
    return connected.construct_matched_candidate(device, tiny=tiny)


def train(model, dataset, normalizer, device: torch.device, *, updates: int,
          batch_size: int, trace_interval: int):
    return connected.train(
        model, dataset, normalizer, device, updates=updates,
        batch_size=batch_size, trace_interval=trace_interval)


def dense_control_identity(
        manifest: dict[str, object], normalizer,
        training_window_count: int) -> dict[str, object]:
    normalizer_record = control.normalizer_record(normalizer)
    schedule = control.make_schedule(
        training_window_count, control.UPDATES, control.BATCH_SIZE)
    observed = {
        "training_window_manifest_sha256": base.canonical_sha(manifest),
        "normalizer_sha256": base.canonical_sha(normalizer_record),
        "sample_stream_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
    }
    expected = {
        name: DENSE_CONTROL_REFERENCE[name] for name in observed}
    matches = {
        name: observed[name] == expected[name] for name in observed}
    if not all(matches.values()):
        raise RuntimeError("dense connected candidate is not control-matched")
    return {
        "observed": observed,
        "expected": expected,
        "all_matched": True,
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
        "segment_gradient_mode": "connected",
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
    normalizer = base.ChannelNormalizer(
        *base.window_statistics(fit_trajectories, sparse_fit_windows),
        device=device)
    training_windows = dense.dense_windows(fit_trajectories)
    manifest = dense.window_manifest(training_windows)
    manifest["fit_inventory_sha256"] = base.canonical_sha(fit_rows)
    control_identity = dense_control_identity(
        manifest, normalizer, len(training_windows))
    dataset = base.SimWindowDataset(fit_trajectories, training_windows)

    model, matched_initialization = construct_matched_candidate(device)
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    training, ema = train(
        model, dataset, normalizer, device, updates=control.UPDATES,
        batch_size=control.BATCH_SIZE,
        trace_interval=control.TRACE_INTERVAL)
    if (training["sample_stream_sha256"]
            != control_identity["observed"]["sample_stream_sha256"]):
        raise RuntimeError("realized sample stream changed from dense control")
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
    mature_reference = {
        key: value for key, value in control.MATURE_B210_REFERENCE.items()
        if key != "name"}
    comparisons = {
        "raw_minus_dense_raw": base.comparison(
            raw_aggregate, DENSE_CONTROL_REFERENCE["raw"]),
        "ema999_minus_dense_ema999": base.comparison(
            ema_aggregate, DENSE_CONTROL_REFERENCE["ema999"]),
        "raw_minus_mature_B210": base.comparison(
            raw_aggregate, mature_reference),
        "ema999_minus_mature_B210": base.comparison(
            ema_aggregate, mature_reference),
        "ema999_minus_raw": base.comparison(ema_aggregate, raw_aggregate),
    }
    result = {
        "schema": SCHEMA,
        "git": git,
        "matched_initialization": matched_initialization,
        "dense_control_identity": control_identity,
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
            "segment_gradient_mode": "connected",
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
        "dense_control_reference": DENSE_CONTROL_REFERENCE,
        "mature_B210_reference": control.MATURE_B210_REFERENCE,
        "held_AoA10": evaluations,
        "comparisons": comparisons,
        "selection_gate": None,
        "selection": "none; raw and EMA are both reported",
    }
    atomic.atomic_json_save(result, args.output_dir / "result.json")
    print(json.dumps({
        "event": "ms_dense5_connected_credit_complete",
        "result": str((args.output_dir / "result.json").resolve()),
        "raw": raw_aggregate,
        "ema999": ema_aggregate,
        "comparisons": comparisons,
        "selection_gate": None,
    }, sort_keys=True), flush=True)
    return result


def smoke(device: torch.device) -> dict[str, object]:
    model, matched_initialization = construct_matched_candidate(
        device, tiny=True)
    trajectory = torch.randn(60, 16, 16, 3)
    trajectories = [trajectory]
    windows = dense.dense_windows(trajectories)
    dataset = base.SimWindowDataset(trajectories, windows)
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
        "matched_initialization": matched_initialization,
        "training_window_manifest": dense.window_manifest(windows),
        "training": training,
        "raw_finite": bool(torch.isfinite(raw).all()),
        "ema_finite": bool(torch.isfinite(averaged).all()),
        "output_shape": list(raw.shape),
        "selection_gate": None,
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
        model, identity = construct_matched_candidate(torch.device("cpu"))
        print(json.dumps({
            "schema": SCHEMA,
            "acknowledgement": ACKNOWLEDGEMENT,
            "training_stride": TRAIN_STRIDE,
            "held_stride": HELD_STRIDE,
            "model_config": asdict(control.MODEL_CONFIG),
            "parameters": sum(value.numel() for value in model.parameters()),
            "matched_initialization": identity,
            "dense_control_reference": DENSE_CONTROL_REFERENCE,
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
            "selection_gate": None,
        }, sort_keys=True, indent=2))
        return
    device = torch.device(args.device)
    result = smoke(device) if args.mode == "smoke" else formal(args, device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
