#!/usr/bin/env python3
"""Fresh connected-credit MS-Closure training and held-AoA10 report.

This is a matched training-graph intervention on the original scaled 23M
MS-Closure.  It keeps the same physical rollout, parameters, data stream,
objective, optimizer, schedule, EMA, and held evaluation as the truncated
control.  The only difference is that predicted physical state remains
differentiable across the second through fourth shooting segments.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_multiple_shooting_closure_scaled_long as control  # noqa: E402
import train_pyramid_causal3d_loao_screen as loao  # noqa: E402
import train_pyramid_causal3d_sim_long as atomic  # noqa: E402
from models.multiple_shooting_closure import (  # noqa: E402
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
)


SCHEMA = "multiple_shooting_connected_credit_scaled_long_held10_v1"
ACKNOWLEDGEMENT = "RUN_MS_CONNECTED_CREDIT_SCALED_LONG_HELD10"
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
MODEL_CONFIG = control.MODEL_CONFIG
SMALL_M17_REFERENCE = control.SMALL_M17_REFERENCE
MATURE_B210_REFERENCE = control.MATURE_B210_REFERENCE
SPARSE_SCALED_MS_REFERENCE = {
    "rel_l2_score": 94.26758827253985,
    "tke_rel_l2_score": 81.66777310182073,
    "mvpe_rel_l2_score": 95.244431540729,
    "point_contribution_46_10_9": 60.10186675421602,
}


def model_config(*, tiny: bool = False) -> MultipleShootingClosureConfig:
    return control.model_config(tiny=tiny) if tiny else MODEL_CONFIG


def construct_model(
        device: torch.device, *, tiny: bool = False,
        ) -> MultipleShootingClosure:
    control.seed_everything(device)
    return MultipleShootingClosure(
        model_config(tiny=tiny), rollout_mode="carry_detached",
        segment_gradient_mode="connected").float().to(device)


def initialization_identity(
        truncated: MultipleShootingClosure,
        connected: MultipleShootingClosure) -> dict[str, object]:
    """Prove that the matched arms differ only in a non-tensor graph flag."""
    truncated_state = truncated.state_dict()
    connected_state = connected.state_dict()
    keys_equal = truncated_state.keys() == connected_state.keys()
    common_names = sorted(set(truncated_state) & set(connected_state))
    common_equal = keys_equal and all(
        torch.equal(truncated_state[name], connected_state[name])
        for name in common_names)
    truncated_config = truncated.architecture_config()
    connected_config = connected.architecture_config()
    config_differences = sorted(
        name for name in set(truncated_config) | set(connected_config)
        if truncated_config.get(name) != connected_config.get(name))
    parameter_counts = {
        "truncated": sum(value.numel() for value in truncated.parameters()),
        "connected": sum(value.numel() for value in connected.parameters()),
    }
    report = {
        "truncated_state_sha256": base.state_sha(truncated_state),
        "connected_state_sha256": base.state_sha(connected_state),
        "state_key_sets_equal": keys_equal,
        "common_state_tensor_count": len(common_names),
        "common_state_bitwise_identical": common_equal,
        "parameter_counts": parameter_counts,
        "only_architecture_config_difference": config_differences,
        "paired_seed": SEED,
    }
    if (not common_equal
            or parameter_counts["truncated"] != parameter_counts["connected"]
            or config_differences != ["segment_gradient_mode"]):
        raise RuntimeError(
            "connected-credit initialization is not matched to control")
    return report


def construct_matched_candidate(
        device: torch.device, *, tiny: bool = False,
        ) -> tuple[MultipleShootingClosure, dict[str, object]]:
    truncated = control.construct_model(device, tiny=tiny)
    connected = construct_model(device, tiny=tiny)
    identity = initialization_identity(truncated, connected)
    del truncated
    return connected, identity


def train(
        model: MultipleShootingClosure, dataset, normalizer,
        device: torch.device, *, updates: int, batch_size: int,
        trace_interval: int,
        ) -> tuple[dict[str, object], ema_lib.FP32EMAShadow]:
    """Delegate the otherwise unchanged recipe to the scaled-long control."""
    if model.segment_gradient_mode != "connected":
        raise ValueError("connected-credit trainer requires connected graph")
    return control.train(
        model, dataset, normalizer, device, updates=updates,
        batch_size=batch_size, trace_interval=trace_interval)


def save_endpoint(
        path: Path, *, kind: str, state: dict[str, torch.Tensor],
        git: dict[str, object], normalizer, updates: int,
        ) -> dict[str, str]:
    atomic.atomic_torch_save({
        "schema": SCHEMA,
        "endpoint_kind": kind,
        "updates": updates,
        "rollout_mode": "carry_detached",
        "segment_gradient_mode": "connected",
        "model_config": asdict(MODEL_CONFIG),
        "normalizer": control.normalizer_record(normalizer),
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

    model, matched_initialization = construct_matched_candidate(device)
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    training, ema = train(
        model, dataset, normalizer, device, updates=UPDATES,
        batch_size=BATCH_SIZE, trace_interval=TRACE_INTERVAL)
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
    reference = {
        key: value for key, value in SMALL_M17_REFERENCE.items()
        if key != "name"}
    mature_reference = {
        key: value for key, value in MATURE_B210_REFERENCE.items()
        if key != "name"}
    comparisons = {
        "raw_minus_sparse_scaled_MS": base.comparison(
            raw_aggregate, SPARSE_SCALED_MS_REFERENCE),
        "ema999_minus_sparse_scaled_MS": base.comparison(
            ema_aggregate, SPARSE_SCALED_MS_REFERENCE),
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
        "matched_initialization": matched_initialization,
        "training": {
            "fit_aoas": list(FIT_AOAS),
            "held_aoa": HELD_AOA,
            "fresh_initialization": True,
            "rollout_mode": "carry_detached",
            "segment_gradient_mode": "connected",
            "model_config": asdict(MODEL_CONFIG),
            "parameters": sum(value.numel() for value in model.parameters()),
            "updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": UPDATES,
            "grad_clip": GRAD_CLIP,
            "loss": "accuracy(1,4,1)+0.25*normalized_uv_mse",
            "ema_decay": EMA_DECAY,
            "trace_interval": TRACE_INTERVAL,
            "sample_stream": "identical deterministic control schedule",
        },
        "normalizer": control.normalizer_record(normalizer),
        "training_result": training,
        "checkpoints": checkpoints,
        "small_M17_reference": SMALL_M17_REFERENCE,
        "sparse_scaled_MS_reference": SPARSE_SCALED_MS_REFERENCE,
        "mature_B210_reference": MATURE_B210_REFERENCE,
        "held_AoA10": evaluations,
        "comparisons": comparisons,
        "selection_gate": None,
        "selection": "none; raw and EMA are both reported",
    }
    atomic.atomic_json_save(result, args.output_dir / "result.json")
    print(json.dumps({
        "event": "ms_connected_credit_scaled_long_complete",
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
        "matched_initialization": matched_initialization,
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
            "model_config": asdict(MODEL_CONFIG),
            "parameters": sum(value.numel() for value in model.parameters()),
            "matched_initialization": identity,
            "training": {
                "updates": UPDATES,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "scheduler_t_max": UPDATES,
                "ema_decay": EMA_DECAY,
                "loss": "accuracy(1,4,1)+0.25*normalized_uv_mse",
            },
            "selection_gate": None,
        }, sort_keys=True, indent=2))
        return
    device = torch.device(args.device)
    result = smoke(device) if args.mode == "smoke" else formal(args, device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
