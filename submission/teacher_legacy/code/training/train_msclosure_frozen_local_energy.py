#!/usr/bin/env python3
"""Train a local-energy allocator on the frozen scaled MS-Closure endpoint.

This is a diagnostic before an expensive from-scratch integration.  It runs
the complete fixed update schedule without score gates and reports raw and EMA
heads against the exact same frozen backbone on held AoA10.
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
import train_hato_loao_screen as shared  # noqa: E402
import train_pyramid_causal3d_sim_long as atomic  # noqa: E402
from models.msclosure_local_energy import (  # noqa: E402
    FrozenMSClosureLocalEnergy,
    MSLocalEnergyConfig,
    MSLocalEnergyController,
)
from models.multiple_shooting_closure import (  # noqa: E402
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
)


SCHEMA = "msclosure_frozen_local_energy_held10_v1"
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
POINT_WEIGHTS = (46.0, 10.0, 9.0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def load_source(path: Path, device: torch.device):
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    required = {"model_config", "rollout_mode", "normalizer", "model_state"}
    if not required.issubset(payload):
        raise ValueError(f"source checkpoint is missing {sorted(required - set(payload))}")
    config = MultipleShootingClosureConfig(**payload["model_config"])
    model = MultipleShootingClosure(
        config, rollout_mode=payload["rollout_mode"]).float()
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(device).eval()
    stats = payload["normalizer"]
    normalizer = base.ChannelNormalizer(
        torch.tensor(stats["mean_input"]),
        torch.tensor(stats["mean_target"]),
        torch.tensor(stats["std_input"]),
        torch.tensor(stats["std_target"]), device)
    expected = {
        "mean_input": normalizer.mean_input,
        "mean_target": normalizer.mean_target,
        "std_input": normalizer.std_input,
        "std_target": normalizer.std_target,
    }
    if not all(torch.equal(getattr(model, name), value)
               for name, value in expected.items()):
        raise RuntimeError("source checkpoint normalizer buffers disagree")
    return model, normalizer, payload


def batch_from(dataset, indices, normalizer, device: torch.device):
    items = [dataset[int(index)] for index in indices]
    history = torch.stack([item[0] for item in items]).to(device)
    target = torch.stack([item[1] for item in items]).to(device)
    return normalizer.preprocess(history, target)


@torch.inference_mode()
def fit_energy_reference(model: MultipleShootingClosure, dataset,
                         normalizer, device: torch.device) -> dict[str, float]:
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)
    history_total = torch.zeros((), device=device, dtype=torch.float64)
    for start in range(0, len(dataset), 16):
        indices = range(start, min(start + 16, len(dataset)))
        history, _target = batch_from(dataset, indices, normalizer, device)
        components = model.forward_components(history)
        future_tke = MSLocalEnergyController.physical_tke(
            components["physical_uv"]).double()
        physical_history = (
            history[..., :2] * model.std_input[:2]
            + model.mean_input[:2])
        history_tke = MSLocalEnergyController.physical_tke(
            physical_history).double()
        free = components["free_mask"][..., 0].double()
        total += (future_tke * free).sum()
        history_total += (history_tke * free).sum()
        count += free.sum()
    if not bool(count > 0) or not bool(torch.isfinite(total)):
        raise RuntimeError("invalid fit-only energy reference")
    return {
        "future_tke_free_pixel_mean": float(total / count),
        "history_tke_free_pixel_mean": float(history_total / count),
        "free_pixel_observations": int(count),
    }


@torch.no_grad()
def identity_report(wrapper: FrozenMSClosureLocalEnergy,
                    history: torch.Tensor) -> dict[str, object]:
    baseline = wrapper.base(history)
    values = wrapper.forward_with_aux(history)
    candidate = values["prediction"]
    base_mean = values["base_physical_uv"].mean(dim=1)
    candidate_mean = values["corrected_physical_uv"].mean(dim=1)
    return {
        "prediction_bitwise_equal": bool(torch.equal(baseline, candidate)),
        "prediction_max_abs_delta": float((baseline - candidate).abs().max()),
        "physical_temporal_mean_max_abs_delta": float(
            (base_mean - candidate_mean).abs().max()),
        "delta_logk_max_abs": float(values["delta_logk"].abs().max()),
    }


def train_head(wrapper: FrozenMSClosureLocalEnergy, dataset, normalizer,
               device: torch.device):
    rng = np.random.default_rng(SEED)
    schedule = rng.integers(
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

    first_history, _ = batch_from(
        dataset, schedule[0], normalizer, device)
    identity = identity_report(wrapper, first_history)
    if (not identity["prediction_bitwise_equal"]
            or identity["physical_temporal_mean_max_abs_delta"] > 1.0e-6):
        raise RuntimeError(f"zero-head identity failure: {identity}")

    for update, indices in enumerate(schedule, start=1):
        history, target = batch_from(dataset, indices, normalizer, device)
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        values = wrapper.forward_with_aux(history)
        loss, details = base.accuracy_objective(
            values["prediction"], target, normalizer, POINT_WEIGHTS)
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
                delta = values["delta_logk"]
                row = {
                    "update": update,
                    "loss_mean_since_last_report": float(np.mean(interval)),
                    "loss_current": float(loss.detach()),
                    "rel_l2": float(details["rel_l2"]),
                    "tke_rel_l2": float(details["tke_rel_l2"]),
                    "mvpe_rel_l2": float(details["mvpe_rel_l2"]),
                    "delta_logk_mean": float(delta.mean()),
                    "delta_logk_std": float(delta.std(unbiased=False)),
                    "delta_logk_abs_max": float(delta.abs().max()),
                    "grad_norm_before_clip": float(grad_norm),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "wall_seconds": time.perf_counter() - started,
                }
            trace.append(row)
            interval.clear()
            print(json.dumps(
                {"event": "ms_local_energy_progress", **row},
                sort_keys=True), flush=True)

    raw_state = state_snapshot(wrapper.controller)
    ema_state = dict(ema.materialize_model_state(wrapper.controller))
    return raw_state, ema_state, {
        "updates": UPDATES,
        "sample_stream_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "identity_at_update0": identity,
        "trace": trace,
        "wall_seconds": time.perf_counter() - started,
    }


def evaluate_pair(base_model: MultipleShootingClosure,
                  wrapper: FrozenMSClosureLocalEnergy,
                  trajectories, windows, files, normalizer,
                  device: torch.device) -> dict[str, object]:
    return shared.evaluate_pair({
        "spatial_window_control": base_model,
        "temporal_candidate": wrapper,
    }, trajectories, windows, files, normalizer, device)


def save_head(path: Path, state: dict[str, torch.Tensor], *, kind: str,
              controller: MSLocalEnergyController, source: Path,
              source_sha: str) -> dict[str, object]:
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
        "sha256": file_sha256(path),
        "state_sha256": base.state_sha(state),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal local-energy training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = file_sha256(args.source_checkpoint)
    model, normalizer, source_payload = load_source(
        args.source_checkpoint, device)
    source_state_sha = base.state_sha(state_snapshot(model))

    rows = base.inventory(args.sim_dir)
    fit_rows = [row for row in rows if int(row["aoa"]) in FIT_AOAS]
    held_rows = [row for row in rows if int(row["aoa"]) == HELD_AOA]
    fit_trajectories, fit_windows, _fit_files = base.load_fields(
        args.sim_dir, fit_rows)
    dataset = base.SimWindowDataset(fit_trajectories, fit_windows)
    energy_reference = fit_energy_reference(
        model, dataset, normalizer, device)
    controller_config = MSLocalEnergyConfig(
        energy_reference=max(
            energy_reference["future_tke_free_pixel_mean"], 1.0e-8))
    wrapper = FrozenMSClosureLocalEnergy(
        model, MSLocalEnergyController(
            model.config.context_channels, controller_config).to(device))
    trainable_parameters = sum(
        parameter.numel() for parameter in wrapper.parameters()
        if parameter.requires_grad)
    raw_state, ema_state, training = train_head(
        wrapper, dataset, normalizer, device)
    checkpoints = {
        "raw": save_head(
            args.output_dir / f"raw_head_u{UPDATES:05d}.pt", raw_state,
            kind=f"raw_u{UPDATES:05d}", controller=wrapper.controller,
            source=args.source_checkpoint, source_sha=source_sha),
        "ema999": save_head(
            args.output_dir / f"ema999_head_u{UPDATES:05d}.pt", ema_state,
            kind=f"ema999_u{UPDATES:05d}", controller=wrapper.controller,
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
        evaluations[kind] = evaluate_pair(
            model, wrapper, held_trajectories, held_windows, held_files,
            normalizer, device)

    terminal_source_sha = base.state_sha(state_snapshot(model))
    if terminal_source_sha != source_state_sha:
        raise RuntimeError("frozen source state changed during head experiment")
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
            "parameters": trainable_parameters,
            "input_channels": model.config.context_channels + 2,
            "conditioning": [
                "causal_multiscale_context", "generated_future_tke",
                "observed_history_tke"],
            "one_map_shared_across_time_and_uv": True,
            "base_frozen": True,
        },
        "training_contract": {
            "updates": UPDATES, "batch_size": BATCH_SIZE,
            "optimizer": "AdamW", "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR", "scheduler_t_max": UPDATES,
            "grad_clip": GRAD_CLIP, "ema_decay": EMA_DECAY,
            "loss": "accuracy_objective_only",
            "point_weights": list(POINT_WEIGHTS),
            "score_gate": None,
        },
        "training_result": training,
        "checkpoints": checkpoints,
        "held_AoA10": evaluations,
        "selection": "none; raw and EMA are both reported",
    }
    atomic.atomic_json_save(result, args.output_dir / "result.json")
    print(json.dumps({
        "event": "ms_local_energy_complete",
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
