#!/usr/bin/env python3
"""Fresh stateful-subgrid MS-Closure training and held-AoA10 report.

The optimization and data protocol are exactly the existing scaled-long
multiple-shooting control.  The only trained-model change is the bounded
twelve-channel recurrent subgrid state.  Raw and FP32 EMA endpoints are
reported normally; the EMA endpoint is additionally evaluated from the same
checkpoint with recurrent, static-h, and segment-reset-h execution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_multiple_shooting_closure_scaled_long as control  # noqa: E402
import train_pyramid_causal3d_loao_screen as loao  # noqa: E402
import train_pyramid_causal3d_sim_long as atomic  # noqa: E402
from models.msclosure_stateful_subgrid import (  # noqa: E402
    SUBGRID_STATE_OVERRIDES,
    StatefulSubgridMultipleShootingClosure,
    StatefulSubgridMultipleShootingConfig,
)


SCHEMA = "msclosure_stateful_subgrid_scaled_long_held10_v1"
ACKNOWLEDGEMENT = "RUN_MS_STATEFUL_SUBGRID_SCALED_LONG_HELD10"
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

MODEL_CONFIG = StatefulSubgridMultipleShootingConfig(
    e0_channels=128,
    e1_channels=256,
    e2_channels=512,
    context_channels=128,
    cell_channels=192,
    cell_blocks=2,
    subgrid_channels=12,
)

SMALL_M17_REFERENCE = control.SMALL_M17_REFERENCE
MATURE_B210_REFERENCE = control.MATURE_B210_REFERENCE


def model_config(
        *, tiny: bool = False) -> StatefulSubgridMultipleShootingConfig:
    if not tiny:
        return MODEL_CONFIG
    return StatefulSubgridMultipleShootingConfig(
        e0_channels=8, e1_channels=16, e2_channels=24,
        e0_blocks=1, e1_blocks=1, e2_blocks=1,
        context_channels=8, cell_channels=16, cell_blocks=1,
        subgrid_channels=12)


def construct_model(
        device: torch.device, *, tiny: bool = False,
        ) -> StatefulSubgridMultipleShootingClosure:
    control.seed_everything(device)
    return StatefulSubgridMultipleShootingClosure(
        model_config(tiny=tiny), rollout_mode="carry_detached"
    ).float().to(device)


def train(
        model: StatefulSubgridMultipleShootingClosure, dataset, normalizer,
        device: torch.device, *, updates: int, batch_size: int,
        trace_interval: int,
        ) -> tuple[dict[str, object], ema_lib.FP32EMAShadow]:
    """Use the control trainer verbatim, with no stateful-specific loss."""
    model.set_subgrid_override("recurrent")
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
        "subgrid_default_mode": "recurrent",
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
        model: StatefulSubgridMultipleShootingClosure,
        state: dict[str, torch.Tensor], trajectories, windows, files,
        normalizer, device: torch.device, *, subgrid_mode: str,
        ) -> dict[str, object]:
    model.load_state_dict(state, strict=True)
    model.set_subgrid_override(subgrid_mode)
    model.eval()
    return loao._evaluate_same_model(
        model, trajectories, windows, files, normalizer, device)


def evaluate_ema_mechanisms(
        model: StatefulSubgridMultipleShootingClosure,
        state: dict[str, torch.Tensor], trajectories, windows, files,
        normalizer, device: torch.device,
        ) -> dict[str, object]:
    """Evaluate three h flows without changing weights or physical carry."""
    evaluations = {}
    try:
        for mode in SUBGRID_STATE_OVERRIDES:
            evaluations[mode] = evaluate_endpoint(
                model, state, trajectories, windows, files, normalizer,
                device, subgrid_mode=mode)
    finally:
        model.set_subgrid_override("recurrent")
    recurrent = evaluations["recurrent"]["aggregate"]
    return {
        "checkpoint_model_state_sha256": base.state_sha(state),
        "same_checkpoint_all_arms": True,
        "physical_rollout_all_arms": "carry_detached",
        "subgrid_modes": list(SUBGRID_STATE_OVERRIDES),
        "evaluations": evaluations,
        "comparisons": {
            "static_minus_recurrent": base.comparison(
                evaluations["static"]["aggregate"], recurrent),
            "reset_minus_recurrent": base.comparison(
                evaluations["reset"]["aggregate"], recurrent),
        },
        "selection": "none; same-checkpoint mechanism diagnostic only",
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
    raw_evaluation = evaluate_endpoint(
        model, raw_state, held_trajectories, held_windows, held_files,
        normalizer, device, subgrid_mode="recurrent")
    ema_mechanisms = evaluate_ema_mechanisms(
        model, ema_state, held_trajectories, held_windows, held_files,
        normalizer, device)
    ema_evaluation = ema_mechanisms["evaluations"]["recurrent"]
    raw_aggregate = raw_evaluation["aggregate"]
    ema_aggregate = ema_evaluation["aggregate"]
    reference = {
        key: value for key, value in SMALL_M17_REFERENCE.items()
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
            "fit_aoas": list(FIT_AOAS),
            "held_aoa": HELD_AOA,
            "fresh_initialization": True,
            "rollout_mode": "carry_detached",
            "subgrid_training_mode": "recurrent",
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
        },
        "normalizer": control.normalizer_record(normalizer),
        "training_result": training,
        "checkpoints": checkpoints,
        "small_M17_reference": SMALL_M17_REFERENCE,
        "mature_B210_reference": MATURE_B210_REFERENCE,
        "held_AoA10": {
            "raw": raw_evaluation,
            "ema999": ema_evaluation,
        },
        "ema999_subgrid_mechanism_diagnostic": ema_mechanisms,
        "comparisons": comparisons,
        "selection": "none; raw and EMA are both reported",
    }
    atomic.atomic_json_save(result, args.output_dir / "result.json")
    print(json.dumps({
        "event": "msclosure_stateful_subgrid_scaled_long_complete",
        "result": str((args.output_dir / "result.json").resolve()),
        "raw": raw_aggregate,
        "ema999": ema_aggregate,
        "ema999_subgrid_comparisons": ema_mechanisms["comparisons"],
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
    ema_state = dict(ema.materialize_model_state(model))
    predictions = {}
    try:
        for mode in SUBGRID_STATE_OVERRIDES:
            model.load_state_dict(ema_state, strict=True)
            model.set_subgrid_override(mode)
            predictions[mode] = model(dataset[0][0][None].to(device))
    finally:
        model.set_subgrid_override("recurrent")
    return {
        "schema": SCHEMA,
        "smoke": "PASS",
        "training": training,
        "raw_finite": bool(torch.isfinite(raw).all()),
        "ema_override_finite": {
            mode: bool(torch.isfinite(value).all())
            for mode, value in predictions.items()},
        "output_shapes": {
            mode: list(value.shape) for mode, value in predictions.items()},
        "same_checkpoint_override_modes": list(SUBGRID_STATE_OVERRIDES),
        "physical_rollout_all_modes": "carry_detached",
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
                "updates": UPDATES,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "scheduler_t_max": UPDATES,
                "ema_decay": EMA_DECAY,
                "loss": "accuracy(1,4,1)+0.25*normalized_uv_mse",
            },
            "ema_held_diagnostic_modes": list(SUBGRID_STATE_OVERRIDES),
            "selection_gate": None,
        }, sort_keys=True, indent=2))
        return
    device = torch.device(args.device)
    result = smoke(device) if args.mode == "smoke" else formal(args, device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
