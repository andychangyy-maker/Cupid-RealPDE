#!/usr/bin/env python3
"""M14 paired Sim2k screen for two-scale deformable history memory.

The aligned candidate and spatially-permuted control have identical tensors,
initial values, sample streams, optimizer settings, and executed operators.
Held-AoA10 field values remain unopened until both in-memory endpoint hashes
exist.  This discovery screen never trains UQ or writes model checkpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Mapping

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


SCHEMA = "b210_m14_deformable_history_memory_sim_discovery_v1"
ACKNOWLEDGEMENT = "RUN_M14_DEFORMABLE_HISTORY_MEMORY_SIM2K"
ARMS = ("control", "candidate")
ARM_MODE = {"control": "spatially_permuted", "candidate": "aligned"}
SEED = 210
SCHEDULE_SEED = 220
UPDATES = 2_000
BATCH_SIZE = 8
LR = 3.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
HELD_AOA = 10
EXPECTED_PARAMETER_INCREMENT = 114_320
GATE = {
    "delta_point_min": .05,
    "component_floor": -.03,
    "positive_trajectories_min": 15,
    "worst_trajectory_delta_point_min": -.05,
    "aligned_minus_zero_point_min": .01,
    "latency_ratio_max": 1.35,
    "train_peak_vram_ratio_max": 1.25,
    "parameter_fraction_max": .001,
}
STRUCTURE = {
    "deformable_history_memory_channels": 32,
    "deformable_history_memory_anchors": (0, 6, 13, 19),
    "deformable_history_memory_max_offset": 1.5,
}


def required_runtime_api() -> dict[str, object]:
    fields = set(EnsembleUNet2dConfig.__dataclass_fields__)
    required = {"deformable_history_memory_mode", *STRUCTURE}
    return {
        "config_fields": sorted(required),
        "missing_config_fields": sorted(required - fields),
        "zero_offset_method": "set_deformable_history_memory_zero_offset",
    }


def model_config(mode: str, *, tiny: bool = False) -> EnsembleUNet2dConfig:
    """Return the mature B210 topology without previously failed M12 state."""
    if mode not in (*ARM_MODE.values(), "none"):
        raise ValueError(f"unknown M14 memory mode {mode!r}")
    missing = required_runtime_api()["missing_config_fields"]
    if missing:
        raise RuntimeError(
            "M14 deformable-memory core unavailable; missing fields: "
            + ", ".join(str(value) for value in missing))
    values = asdict(base.model_config("causal_temporal", tiny=tiny))
    values.update(
        history_dynamics_state_mode="none",
        deformable_history_memory_mode=mode,
        **STRUCTURE,
    )
    if tiny:
        # The memory contract requires four pyramid levels.  Spatial 16x16
        # and base8 keep smoke inexpensive while retaining both injection sites.
        values.update(
            base=8,
            depth=4,
            groups=4,
            cross_conditioned_channels=8,
            uq_hidden_channels=8,
            member_adapter_channels=8,
            mean_profile_channels=8,
            temporal_uq_channels=8,
            physical_energy_hidden_channels=8,
            deformable_history_memory_channels=8,
        )
    return EnsembleUNet2dConfig(**values)


def _seed(device: torch.device) -> None:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _config_differences(left: EnsembleUNet2dConfig,
                        right: EnsembleUNet2dConfig) -> list[str]:
    return sorted(
        key for key in left.__dataclass_fields__
        if getattr(left, key) != getattr(right, key))


def construct_pair(device: torch.device, *, tiny: bool = False):
    models: dict[str, EnsembleUNet2d] = {}
    for arm in ARMS:
        _seed(device)
        models[arm] = EnsembleUNet2d(
            model_config(ARM_MODE[arm], tiny=tiny)).float().to(device)
    left = models["control"].state_dict()
    right = models["candidate"].state_dict()
    equal = left.keys() == right.keys() and all(
        torch.equal(value, right[key]) for key, value in left.items())
    if not equal:
        raise RuntimeError("M14 arms do not start bitwise identical")
    differences = _config_differences(
        models["control"].config, models["candidate"].config)
    if differences != ["deformable_history_memory_mode"]:
        raise RuntimeError(f"M14 config differences changed: {differences}")
    counts = {arm: _parameter_count(model) for arm, model in models.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"M14 parameter counts differ: {counts}")
    return models, {
        "state_sha256": base.state_sha(left),
        "state_tensors": len(left),
        "bitwise_identical": True,
        "parameter_counts": counts,
        "only_config_difference": differences[0],
    }


def set_pair_normalizer_statistics(
        models: Mapping[str, EnsembleUNet2d],
        normalizer) -> dict[str, object]:
    """Install the same fit-only persistence map without inherited arm names."""
    values = (
        normalizer.mean_input,
        normalizer.mean_target,
        normalizer.std_input,
        normalizer.std_target,
    )
    for model in models.values():
        model.set_normalizer_statistics(*values)
    buffer_names = ("persistence_scale", "persistence_bias", "target_std")
    control = models["control"]
    candidate = models["candidate"]
    if not all(torch.equal(getattr(control, name), getattr(candidate, name))
               for name in buffer_names):
        raise RuntimeError("paired normalizer-dependent buffers differ")
    return {
        "state_sha256": base.state_sha({
            name: getattr(control, name) for name in buffer_names}),
        **{
            name: getattr(control, name).detach().cpu().tolist()
            for name in buffer_names
        },
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latency_preflight(device: torch.device) -> dict[str, object]:
    """Measure update-zero active overhead against the exact legacy topology."""
    if device.type != "cuda":
        return {"measured": False, "reason": "CUDA required"}
    _seed(device)
    legacy = EnsembleUNet2d(model_config("none")).float().to(device).eval()
    _seed(device)
    active = EnsembleUNet2d(model_config("aligned")).float().to(device).eval()
    history = torch.zeros(16, 20, 64, 128, 3, device=device)
    with torch.inference_mode():
        legacy_output = legacy(history)
        active_output = active(history)
        if not torch.equal(legacy_output, active_output):
            raise RuntimeError("active memory is not an update-zero identity")
        for _ in range(5):
            legacy(history)
            active(history)
        _synchronize(device)
        samples: dict[str, list[float]] = {"legacy": [], "active": []}
        for repeat in range(20):
            order = (("legacy", legacy), ("active", active))
            if repeat % 2:
                order = tuple(reversed(order))
            for name, model in order:
                _synchronize(device)
                started = time.perf_counter()
                model(history)
                _synchronize(device)
                samples[name].append(time.perf_counter() - started)
    counts = {"legacy": _parameter_count(legacy),
              "active": _parameter_count(active)}
    medians = {key: statistics.median(values)
               for key, values in samples.items()}
    result = {
        "measured": True,
        "batch": 16,
        "input_shape_per_sample": [20, 64, 128, 3],
        "warmups_per_model": 6,
        "timed_repeats_per_model": 20,
        "median_seconds": medians,
        "active_over_legacy_ratio": (
            medians["active"] / max(medians["legacy"], 1.0e-12)),
        "parameter_counts": counts,
        "parameter_increment": counts["active"] - counts["legacy"],
        "parameter_fraction": (
            (counts["active"] - counts["legacy"])
            / max(counts["legacy"], 1)),
        "update_zero_output_bitwise_equal": True,
    }
    del history, legacy_output, active_output, legacy, active
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _train_peak_bytes(mode: str, device: torch.device) -> int:
    """One full batch8 AdamW step, including parameters, grads, and states."""
    _seed(device)
    model = EnsembleUNet2d(model_config(mode)).float().to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = torch.randn(8, 20, 64, 128, 3, device=device)
    target = torch.randn(8, 20, 64, 128, 3, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(history)
    loss = (prediction - target).square().mean()
    loss.backward()
    optimizer.step()
    _synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    del prediction, loss, target, history, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return peak


def resource_preflight(device: torch.device) -> dict[str, object]:
    latency = _latency_preflight(device)
    if device.type != "cuda":
        return {"latency": latency, "train_peak_vram": {
            "measured": False, "reason": "CUDA required"}}
    peaks = {
        "legacy": _train_peak_bytes("none", device),
        "active": _train_peak_bytes("aligned", device),
    }
    return {
        "latency": latency,
        "train_peak_vram": {
            "measured": True,
            "batch": 8,
            "bytes": peaks,
            "active_over_legacy_ratio": (
                peaks["active"] / max(peaks["legacy"], 1)),
        },
    }


def update_zero_audit(models: Mapping[str, EnsembleUNet2d], dataset,
                      normalizer, device: torch.device) -> dict[str, object]:
    history, target = dataset[0]
    history, target = normalizer.preprocess(
        history.unsqueeze(0).to(device), target.unsqueeze(0).to(device))
    outputs = {}
    losses = {}
    for arm in ARMS:
        models[arm].eval()
        with torch.inference_mode():
            outputs[arm] = models[arm](history)
            losses[arm] = float((outputs[arm] - target).square().mean())
    equal = torch.equal(outputs["control"], outputs["candidate"])
    if not equal or losses["control"] != losses["candidate"]:
        raise RuntimeError("M14 update-zero predictions/losses differ")
    return {
        "prediction_bitwise_equal": True,
        "prediction_sha256": base.state_sha({"prediction": outputs["control"]}),
        "mse": losses["control"],
    }


def _memory_gradient_groups(model: nn.Module) -> dict[str, bool]:
    groups = {"frame_encoder": False, "query_heads": False,
              "value_projections": False}
    for name, parameter in model.named_parameters():
        if "deformable_history_memory." not in name or parameter.grad is None:
            continue
        for group in groups:
            if f"deformable_history_memory.{group}" in name:
                groups[group] = groups[group] or bool(
                    torch.count_nonzero(parameter.grad.detach()).item())
    return groups


def _finite_state(model: nn.Module) -> bool:
    return all(
        not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in model.state_dict().values())


def train_arm(name: str, model: EnsembleUNet2d, dataset,
              schedule: np.ndarray, normalizer, device: torch.device,
              updates: int) -> dict[str, object]:
    """Train one point-only arm with a reset stochastic stream."""
    _seed(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates)
    trace = []
    interval: list[float] = []
    gradient_seen = {"frame_encoder": False, "query_heads": False,
                     "value_projections": False}
    started = time.perf_counter()
    for update, (history, target) in enumerate(
            base.batch_stream(dataset, schedule), 1):
        model.train()
        history, target = normalizer.preprocess(
            history.to(device), target.to(device))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        accuracy, details = base.accuracy_objective(
            prediction, target, normalizer, (1.0, 4.0, 1.0))
        mse = (prediction - target).square().mean()
        loss = accuracy + .25 * mse
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"{name} loss is non-finite at {update}")
        loss.backward()
        for group, seen in _memory_gradient_groups(model).items():
            gradient_seen[group] = gradient_seen[group] or seen
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(
                f"{name} gradient is non-finite at {update}")
        optimizer.step()
        scheduler.step()
        interval.append(float(loss.detach()))
        if update == 1 or update % 250 == 0 or update == updates:
            row = {
                "update": update,
                "loss_mean": float(np.mean(interval)),
                "mse": float(mse.detach()),
                "grad_norm_before_clip": float(grad_norm.detach()),
                **{key: float(value) for key, value in details.items()},
            }
            trace.append(row)
            print(json.dumps({"event": "m14_progress", "arm": name,
                              **row,
                              "wall_seconds": time.perf_counter() - started},
                             sort_keys=True), flush=True)
            interval.clear()
    if not _finite_state(model):
        raise FloatingPointError(f"{name} endpoint contains non-finite state")
    return {
        "endpoint_state_sha256": base.state_sha(model.state_dict()),
        "updates": updates,
        "wall_seconds": time.perf_counter() - started,
        "point_forward_only": True,
        "uq_forward_executed": False,
        "memory_gradient_seen_nonzero": gradient_seen,
        "trace": trace,
    }


def _validate_held_panel(window_files: list[str]) -> None:
    counts = {
        name: window_files.count(name) for name in sorted(set(window_files))}
    if (len(counts) != base.EXPECTED_FILES_PER_AOA
            or set(counts.values())
            != {base.EXPECTED_WINDOWS_PER_TRAJECTORY}):
        raise RuntimeError(f"held panel shape changed: {counts}")


def evaluate_model(model: EnsembleUNet2d, trajectories, windows,
                   window_files: list[str], normalizer,
                   device: torch.device) -> dict[str, object]:
    _validate_held_panel(window_files)
    dataset = base.SimWindowDataset(trajectories, windows)
    error_parts = []
    labels: list[str] = []
    digest = hashlib.sha256()
    model.eval()
    for start in range(0, len(dataset), 16):
        indices = range(start, min(start + 16, len(dataset)))
        samples = [dataset[index] for index in indices]
        history = torch.stack([item[0] for item in samples]).to(device)
        target = torch.stack([item[1] for item in samples]).numpy()
        history = ((history - normalizer.mean_input)
                   / normalizer.std_input)
        with torch.inference_mode():
            prediction = model(history)
            prediction = (
                prediction * normalizer.std_target
                + normalizer.mean_target)
        array = np.ascontiguousarray(prediction.float().cpu().numpy())
        if not np.isfinite(array).all():
            raise FloatingPointError("held prediction is non-finite")
        digest.update(array.tobytes())
        error_parts.append(base.error_rows(array, target))
        labels.extend(window_files[index] for index in indices)
    errors = np.concatenate(error_parts)
    label_array = np.asarray(labels)
    per_trajectory = {
        name: base.score(errors[label_array == name])
        for name in sorted(set(labels))
    }
    return {
        "aggregate": base.score(errors),
        "per_trajectory": per_trajectory,
        "prediction_sha256": digest.hexdigest(),
    }


def evaluate_variants(models: Mapping[str, EnsembleUNet2d], trajectories,
                      windows, window_files: list[str], normalizer,
                      device: torch.device) -> dict[str, object]:
    variants = {
        "control": evaluate_model(
            models["control"], trajectories, windows, window_files,
            normalizer, device),
        "candidate": evaluate_model(
            models["candidate"], trajectories, windows, window_files,
            normalizer, device),
    }
    before_ablation_sha = base.state_sha(models["candidate"].state_dict())
    models["candidate"].set_deformable_history_memory_zero_offset(True)
    try:
        variants["candidate_zero_offset"] = evaluate_model(
            models["candidate"], trajectories, windows, window_files,
            normalizer, device)
    finally:
        models["candidate"].set_deformable_history_memory_zero_offset(False)
    after_ablation_sha = base.state_sha(models["candidate"].state_dict())
    if before_ablation_sha != after_ablation_sha:
        raise RuntimeError("zero-offset ablation changed candidate state")

    candidate_control = base.comparison(
        variants["candidate"]["aggregate"],
        variants["control"]["aggregate"])
    candidate_zero = base.comparison(
        variants["candidate"]["aggregate"],
        variants["candidate_zero_offset"]["aggregate"])
    per_trajectory = {}
    paired_deltas = []
    for name in variants["candidate"]["per_trajectory"]:
        scores = {
            key: value["per_trajectory"][name]
            for key, value in variants.items()
        }
        delta_control = base.comparison(
            scores["candidate"], scores["control"])
        delta_zero = base.comparison(
            scores["candidate"], scores["candidate_zero_offset"])
        paired_deltas.append(
            float(delta_control["point_contribution_46_10_9"]))
        per_trajectory[name] = {
            **scores,
            "delta_candidate_minus_control": delta_control,
            "delta_candidate_minus_zero_offset": delta_zero,
        }
    return {
        "aggregate": {
            key: value["aggregate"] for key, value in variants.items()},
        "prediction_sha256": {
            key: value["prediction_sha256"]
            for key, value in variants.items()},
        "delta_candidate_minus_control": candidate_control,
        "delta_candidate_minus_zero_offset": candidate_zero,
        "per_trajectory": per_trajectory,
        "trajectory_count": len(per_trajectory),
        "windows_per_trajectory": base.EXPECTED_WINDOWS_PER_TRAJECTORY,
        "positive_trajectories_candidate_minus_control": sum(
            value > 0.0 for value in paired_deltas),
        "worst_trajectory_delta_candidate_minus_control": min(
            paired_deltas),
        "candidate_state_unchanged_by_zero_ablation": True,
    }


def decide_gate(evaluation: Mapping[str, object],
                resources: Mapping[str, object],
                initial_identity: Mapping[str, object],
                endpoints: Mapping[str, object]) -> dict[str, object]:
    delta = evaluation["delta_candidate_minus_control"]
    zero = evaluation["delta_candidate_minus_zero_offset"]
    latency = resources["latency"]
    vram = resources["train_peak_vram"]
    gradients = all(
        all(bool(value) for value in endpoint[
            "memory_gradient_seen_nonzero"].values())
        for endpoint in endpoints.values())
    checks = {
        "initial_state_bitwise_equal": bool(
            initial_identity["bitwise_identical"]),
        "only_mode_differs": (
            initial_identity["only_config_difference"]
            == "deformable_history_memory_mode"),
        "all_memory_gradient_groups_live": gradients,
        "point_delta": (
            float(delta["point_contribution_46_10_9"])
            >= GATE["delta_point_min"]),
        "component_floor": all(
            float(delta[key]) >= GATE["component_floor"]
            for key in base.POINT_WEIGHTS),
        "positive_trajectories": (
            int(evaluation[
                "positive_trajectories_candidate_minus_control"])
            >= GATE["positive_trajectories_min"]),
        "worst_trajectory": (
            float(evaluation[
                "worst_trajectory_delta_candidate_minus_control"])
            >= GATE["worst_trajectory_delta_point_min"]),
        "same_checkpoint_zero_offset": (
            float(zero["point_contribution_46_10_9"])
            >= GATE["aligned_minus_zero_point_min"]),
        "parameter_increment": (
            int(latency["parameter_increment"])
            == EXPECTED_PARAMETER_INCREMENT),
        "parameter_fraction": (
            float(latency["parameter_fraction"])
            <= GATE["parameter_fraction_max"]),
        "latency": (
            float(latency["active_over_legacy_ratio"])
            <= GATE["latency_ratio_max"]),
        "train_peak_vram": (
            float(vram["active_over_legacy_ratio"])
            <= GATE["train_peak_vram_ratio_max"]),
        "complete_panel": (
            int(evaluation["trajectory_count"])
            == base.EXPECTED_FILES_PER_AOA
            and int(evaluation["windows_per_trajectory"])
            == base.EXPECTED_WINDOWS_PER_TRAJECTORY),
        "ablation_state_unchanged": bool(
            evaluation["candidate_state_unchanged_by_zero_ablation"]),
    }
    return {**GATE, "checks": checks, "pass": all(checks.values())}


def smoke(device: torch.device) -> dict[str, object]:
    models, identity = construct_pair(device, tiny=True)
    normalizer = base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), device)
    buffers = set_pair_normalizer_statistics(models, normalizer)
    trajectory = torch.randn(40, 16, 16, 3)
    dataset = base.SimWindowDataset(
        [trajectory], [base.SimWindow(0, 0)])
    update_zero = update_zero_audit(
        models, dataset, normalizer, device)
    schedule = np.zeros((3, 1), dtype=np.int64)
    endpoints = {
        arm: train_arm(arm, models[arm], dataset, schedule,
                       normalizer, device, len(schedule))
        for arm in ARMS
    }
    return {
        "schema": SCHEMA,
        "smoke": "PASS",
        "identity": identity,
        "normalizer_buffers": buffers,
        "update_zero": update_zero,
        "endpoints": endpoints,
    }


def run(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError(
            "formal M14 requires --execute and exact acknowledgement")
    if (args.sim_dir is None or args.output_dir is None
            or args.held_aoa is None):
        raise ValueError(
            "formal M14 requires sim-dir, output-dir, and held-aoa")
    if (args.updates != UPDATES or args.batch_size != BATCH_SIZE
            or args.held_aoa != HELD_AOA):
        raise ValueError(
            "M14 contract is updates2000, batch8, held-AoA10")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal M14 requires CUDA resource measurements")
    missing = required_runtime_api()["missing_config_fields"]
    if missing:
        raise RuntimeError(f"M14 core API is incomplete: {missing}")
    git = base.git_report()
    if not git["clean"]:
        raise RuntimeError(
            f"formal M14 requires clean Git: {git['porcelain']}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "refusing to overwrite non-empty M14 output directory")

    resources = resource_preflight(device)
    rows = base.inventory(args.sim_dir)
    fit_rows = [row for row in rows if int(row["aoa"]) != HELD_AOA]
    held_rows = [row for row in rows if int(row["aoa"]) == HELD_AOA]
    if len(fit_rows) != 80 or len(held_rows) != 20:
        raise RuntimeError("M14 whole-AoA split count changed")

    # Only fit u/v/p values are opened here.  Held inventory metadata above is
    # allowed, but held flow-field arrays stay unopened until both endpoints.
    fit_trajectories, fit_windows, _unused = base.load_fields(
        args.sim_dir, fit_rows)
    statistics_values = base.window_statistics(
        fit_trajectories, fit_windows)
    normalizer = base.ChannelNormalizer(
        *statistics_values, device=device)
    fit_dataset = base.SimWindowDataset(
        fit_trajectories, fit_windows)
    schedule = np.random.default_rng(SCHEDULE_SEED).integers(
        0, len(fit_dataset), size=(UPDATES, BATCH_SIZE), dtype=np.int64)
    models, initial_identity = construct_pair(device)
    normalizer_buffers = set_pair_normalizer_statistics(
        models, normalizer)
    update_zero = update_zero_audit(
        models, fit_dataset, normalizer, device)
    endpoints = {
        arm: train_arm(
            arm, models[arm], fit_dataset, schedule,
            normalizer, device, UPDATES)
        for arm in ARMS
    }
    if not all(endpoint.get("endpoint_state_sha256")
               for endpoint in endpoints.values()):
        raise RuntimeError("both endpoint seals do not exist")

    # Free fit arrays before this first held u/v/p materialization.  The two
    # endpoint SHA seals above remain in-memory in the result record.
    del fit_dataset, fit_trajectories
    gc.collect()
    held_trajectories, held_windows, held_window_files = base.load_fields(
        args.sim_dir, held_rows)
    evaluation = evaluate_variants(
        models, held_trajectories, held_windows, held_window_files,
        normalizer, device)
    gate = decide_gate(
        evaluation, resources, initial_identity, endpoints)
    result = {
        "schema": SCHEMA,
        "evidence_class": "selection_reused_whole_AoA10_Sim",
        "git": git,
        "held_aoa": HELD_AOA,
        "fit_aoas": [aoa for aoa in base.AOAS if aoa != HELD_AOA],
        "training": {
            "updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "model_and_dropout_seed": SEED,
            "schedule_seed": SCHEDULE_SEED,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR_Tmax_2000",
            "grad_clip": GRAD_CLIP,
            "point_loss": "accuracy_1_4_1_plus_.25_mse",
            "augmentation": "none",
            "uq_forward_executed": False,
        },
        "structure": {
            **STRUCTURE,
            "banks": [[8, 16], [16, 32]],
            "injection_channels": [1680, 840],
            "control": "half_grid_spatial_roll_of_predicted_offsets",
        },
        "runtime_api": required_runtime_api(),
        "source_sha256": {
            "model": file_sha256(
                REPO_ROOT / "code/models/unet2d.py"),
            "trainer": file_sha256(Path(__file__).resolve()),
            "base_harness": file_sha256(
                Path(base.__file__).resolve()),
            "scoring": file_sha256(
                REPO_ROOT
                / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"),
        },
        "fit_inventory_sha256": base.canonical_sha(fit_rows),
        "held_inventory_sha256": base.canonical_sha(held_rows),
        "fit_normalizer_sha256": base.state_sha({
            str(index): value
            for index, value in enumerate(statistics_values)}),
        "sample_stream_sha256_both_arms": hashlib.sha256(
            schedule.tobytes()).hexdigest(),
        "initial_identity": initial_identity,
        "normalizer_dependent_buffers": normalizer_buffers,
        "update_zero": update_zero,
        "resources": resources,
        "endpoints": endpoints,
        "held_field_values_loaded_before_both_endpoint_seals": False,
        "evaluation": evaluation,
        "gate": gate,
        "decision": (
            "GO_M14_CONFIRMATORY_SIM"
            if gate["pass"] else
            "STOP_M14_DEFORMABLE_HISTORY_MEMORY"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"),
                   allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "delta_candidate_minus_control": evaluation[
            "delta_candidate_minus_control"],
        "delta_candidate_minus_zero_offset": evaluation[
            "delta_candidate_minus_zero_offset"],
        "result": str(result_path),
    }, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("contract", "smoke", "formal"),
        default="contract")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--held-aoa", type=int, default=HELD_AOA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        print(json.dumps({
            "schema": SCHEMA,
            "acknowledgement": ACKNOWLEDGEMENT,
            "runtime_api": required_runtime_api(),
            "structure": STRUCTURE,
            "gate": GATE,
        }, sort_keys=True))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if args.mode == "smoke":
        print(json.dumps(smoke(device), sort_keys=True))
        return
    run(args, device)


if __name__ == "__main__":
    main()
