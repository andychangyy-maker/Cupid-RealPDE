#!/usr/bin/env python3
"""Train a K=4 zero-mean DCT residual on the frozen scaled MS closure."""
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
import train_hato_loao_screen as shared  # noqa: E402
import train_msclosure_frozen_local_energy as common  # noqa: E402
import train_pyramid_causal3d_sim_long as atomic  # noqa: E402
from models.msclosure_temporal_basis_residual import (  # noqa: E402
    FrozenMSClosureTemporalBasisResidual,
    MSTemporalBasisResidualConfig,
    MSTemporalBasisResidualController,
)


SCHEMA = "msclosure_frozen_dct4_residual_held10_v1"
FIT_AOAS = (0, 5, 15, 20)
HELD_AOA = 10
UPDATES = 4_000
BATCH_SIZE = 8
SEED = 321
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
TRACE_INTERVAL = 250
ACCURACY_WEIGHTS = (1.0, 4.0, 1.0)
MSE_WEIGHT = 0.25


@torch.no_grad()
def identity_report(wrapper: FrozenMSClosureTemporalBasisResidual,
                    history: torch.Tensor) -> dict[str, object]:
    baseline = wrapper.base(history)
    values = wrapper.forward_with_aux(history)
    candidate = values["prediction"]
    mean_delta = (
        candidate[..., :2].mean(dim=1)
        - baseline[..., :2].mean(dim=1))
    return {
        "prediction_bitwise_equal": bool(torch.equal(baseline, candidate)),
        "prediction_max_abs_delta": float((baseline - candidate).abs().max()),
        "normalized_temporal_mean_max_abs_delta": float(mean_delta.abs().max()),
        "normalized_delta_max_abs": float(
            values["normalized_delta"].abs().max()),
        "pressure_bitwise_equal": bool(torch.equal(
            baseline[..., 2:], candidate[..., 2:])),
    }


def train_head(wrapper: FrozenMSClosureTemporalBasisResidual, dataset,
               normalizer, device: torch.device):
    schedule = np.random.default_rng(SEED).integers(
        0, len(dataset), size=(UPDATES, BATCH_SIZE), dtype=np.int64)
    parameters = tuple(wrapper.controller.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UPDATES)
    ema = ema_lib.FP32EMAShadow(wrapper.controller, EMA_DECAY)
    trace: list[dict[str, float | int]] = []
    interval: list[float] = []
    started = time.perf_counter()

    first_history, _ = common.batch_from(
        dataset, schedule[0], normalizer, device)
    identity = identity_report(wrapper, first_history)
    if not (identity["prediction_bitwise_equal"]
            and identity["pressure_bitwise_equal"]
            and identity["normalized_temporal_mean_max_abs_delta"] <= 1e-6):
        raise RuntimeError(f"zero-head identity failure: {identity}")

    for update, indices in enumerate(schedule, start=1):
        history, target = common.batch_from(
            dataset, indices, normalizer, device)
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        values = wrapper.forward_with_aux(history)
        accuracy, details = base.accuracy_objective(
            values["prediction"], target, normalizer, ACCURACY_WEIGHTS)
        mse_uv = F.mse_loss(
            values["prediction"][..., :2], target[..., :2])
        loss = accuracy + MSE_WEIGHT * mse_uv
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        if any(parameter.grad is not None
               for parameter in wrapper.base.parameters()):
            raise RuntimeError("frozen MS-Closure received a gradient")
        grad_norm = nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient at update {update}")
        optimizer.step()
        scheduler.step()
        ema.update_after_optimizer_step(
            wrapper.controller, optimizer_step=update)
        interval.append(float(loss.detach()))

        if update == 1 or update % TRACE_INTERVAL == 0 or update == UPDATES:
            with torch.no_grad():
                coefficients = values["coefficients"]
                delta = values["normalized_delta"]
                row = {
                    "update": update,
                    "loss_mean_since_last_report": float(np.mean(interval)),
                    "loss_current": float(loss.detach()),
                    "accuracy_current": float(accuracy.detach()),
                    "mse_uv_current": float(mse_uv.detach()),
                    "rel_l2": float(details["rel_l2"]),
                    "tke_rel_l2": float(details["tke_rel_l2"]),
                    "mvpe_rel_l2": float(details["mvpe_rel_l2"]),
                    "coefficient_rms": float(
                        coefficients.float().square().mean().sqrt()),
                    "coefficient_abs_max": float(coefficients.abs().max()),
                    "normalized_delta_rms": float(
                        delta.float().square().mean().sqrt()),
                    "grad_norm_before_clip": float(grad_norm),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "wall_seconds": time.perf_counter() - started,
                }
            trace.append(row)
            interval.clear()
            print(json.dumps(
                {"event": "ms_dct4_residual_progress", **row},
                sort_keys=True), flush=True)

    raw_state = common.state_snapshot(wrapper.controller)
    ema_state = dict(ema.materialize_model_state(wrapper.controller))
    return raw_state, ema_state, {
        "updates": UPDATES,
        "sample_stream_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "identity_at_update0": identity,
        "trace": trace,
        "wall_seconds": time.perf_counter() - started,
    }


def save_head(path: Path, state: dict[str, torch.Tensor], *, kind: str,
              controller: MSTemporalBasisResidualController,
              source: Path, source_sha: str) -> dict[str, object]:
    atomic.atomic_torch_save({
        "schema": SCHEMA,
        "endpoint_kind": kind,
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": source_sha,
        "controller_config": controller.architecture_config(),
        "controller_state": state,
    }, path)
    return {
        "path": str(path.resolve()),
        "sha256": common.file_sha256(path),
        "state_sha256": base.state_sha(state),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal DCT residual training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = common.file_sha256(args.source_checkpoint)
    model, normalizer, source_payload = common.load_source(
        args.source_checkpoint, device)
    source_state_sha = base.state_sha(common.state_snapshot(model))

    rows = base.inventory(args.sim_dir)
    fit_rows = [row for row in rows if int(row["aoa"]) in FIT_AOAS]
    held_rows = [row for row in rows if int(row["aoa"]) == HELD_AOA]
    fit_trajectories, fit_windows, _fit_files = base.load_fields(
        args.sim_dir, fit_rows)
    dataset = base.SimWindowDataset(fit_trajectories, fit_windows)
    energy_reference = common.fit_energy_reference(
        model, dataset, normalizer, device)
    controller_config = MSTemporalBasisResidualConfig(
        modes=4,
        energy_reference=max(
            energy_reference["future_tke_free_pixel_mean"], 1.0e-8))
    controller = MSTemporalBasisResidualController(
        model.config.context_channels, model.config.out_steps,
        controller_config).to(device)
    wrapper = FrozenMSClosureTemporalBasisResidual(
        model, controller=controller)
    trainable_parameters = sum(
        parameter.numel() for parameter in wrapper.parameters()
        if parameter.requires_grad)
    raw_state, ema_state, training = train_head(
        wrapper, dataset, normalizer, device)
    checkpoints = {
        "raw": save_head(
            args.output_dir / f"raw_head_u{UPDATES:05d}.pt", raw_state,
            kind=f"raw_u{UPDATES:05d}", controller=controller,
            source=args.source_checkpoint, source_sha=source_sha),
        "ema999": save_head(
            args.output_dir / f"ema999_head_u{UPDATES:05d}.pt", ema_state,
            kind=f"ema999_u{UPDATES:05d}", controller=controller,
            source=args.source_checkpoint, source_sha=source_sha),
    }

    del dataset, fit_trajectories
    gc.collect()
    torch.cuda.empty_cache()
    held_trajectories, held_windows, held_files = base.load_fields(
        args.sim_dir, held_rows)
    evaluations = {}
    for kind, state in (("raw", raw_state), ("ema999", ema_state)):
        wrapper.controller.load_state_dict(state, strict=True)
        wrapper.eval()
        evaluations[kind] = shared.evaluate_pair({
            "spatial_window_control": model,
            "temporal_candidate": wrapper,
        }, held_trajectories, held_windows, held_files, normalizer, device)

    if base.state_sha(common.state_snapshot(model)) != source_state_sha:
        raise RuntimeError("frozen source state changed during DCT experiment")
    result = {
        "schema": SCHEMA,
        "git": base.git_report(),
        "source": {
            "path": str(args.source_checkpoint.resolve()),
            "sha256": source_sha,
            "endpoint_kind": source_payload.get("endpoint_kind"),
            "state_sha256": source_state_sha,
            "state_unchanged": True,
        },
        "split": {"fit_aoas": list(FIT_AOAS), "held_aoa": HELD_AOA},
        "energy_reference": energy_reference,
        "controller": {
            "config": asdict(controller_config),
            "architecture": controller.architecture_config(),
            "parameters": trainable_parameters,
            "base_frozen": True,
            "fixed_non_dc_dct_basis": True,
            "changes_temporal_waveform": True,
        },
        "training_contract": {
            "updates": UPDATES, "batch_size": BATCH_SIZE,
            "optimizer": "AdamW", "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR", "scheduler_t_max": UPDATES,
            "grad_clip": GRAD_CLIP, "ema_decay": EMA_DECAY,
            "loss": "accuracy(1,4,1)+0.25*normalized_uv_mse",
            "score_gate": None,
        },
        "oracle_context": {
            "held_dct4_zero_mean_residual_energy_recovery":
                0.9279659000314789,
            "oracle_is_not_used_for_training": True,
        },
        "training_result": training,
        "checkpoints": checkpoints,
        "held_AoA10": evaluations,
        "selection": "none; raw and EMA are both reported",
    }
    atomic.atomic_json_save(result, args.output_dir / "result.json")
    print(json.dumps({
        "event": "ms_dct4_residual_complete",
        "result": str((args.output_dir / "result.json").resolve()),
        "raw_delta": evaluations["raw"]["delta_candidate_minus_control"],
        "ema999_delta": evaluations["ema999"]["delta_candidate_minus_control"],
    }, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
