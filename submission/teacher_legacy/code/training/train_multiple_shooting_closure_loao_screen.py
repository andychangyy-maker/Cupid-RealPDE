#!/usr/bin/env python3
"""Held-AoA10 paired Sim2k screen for MS-Closure-4x5.

Both arms run the same causal UV encoder and the same twenty bounded physical
closure calls.  The sole causal variable is the detached state selected at
the starts of segments 2--4: terminal-history reset versus previous-segment
carry.  Held field values are loaded only after both endpoints are sealed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
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
import train_hato_loao_screen as shared  # noqa: E402
import train_pyramid_causal3d_loao_screen as loao  # noqa: E402
import train_pyramid_causal3d_sim_long as atomic  # noqa: E402


SCHEMA = "multiple_shooting_closure_4x5_held10_sim2k_v1"
ACKNOWLEDGEMENT = "RUN_MS_CLOSURE_4X5_HELD10_SIM2K"
ARMS = ("reset_control", "carry_candidate")
ARM_MODE = {"reset_control": "reset_anchor",
            "carry_candidate": "carry_detached"}
SEED = 210
SCHEDULE_SEED = 220
UPDATES = 2_000
BATCH_SIZE = 8
LR = 3.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
HELD_AOA = 10
FIT_AOAS = (0, 5, 15, 20)
EXPECTED_PARAMETERS = 1_441_112
EXPECTED_STATE_TENSORS = 166
EXPECTED_INITIAL_SHA256 = (
    "0fa91adc04890122238239bc00e3ea62d359ff6c2fc3499c709698ae4b638bc1")
EXPECTED_ARCHITECTURE_SHA256 = (
    "1fbe97acbb057f0bbbef92f28bba65799bc832188f358048927cef836633281a")
EXPECTED_SAMPLE_STREAM_SHA256 = (
    "f0a9ad185eabb0c45c067a2be595378aad58beab5cb2116e5ab4faa65fdca960")
GATE = {
    "delta_point_min": 0.10,
    "component_floor": -0.03,
    "positive_trajectories_min": 16,
    "worst_trajectory_delta_point_min": -0.05,
    "candidate_absolute_point_min": 60.225899,
    "all_cumulative_prefix_delta_min": 0.0,
    "native_minus_reset_override_min": 0.03,
    "late_native_minus_reset_override_min": 0.05,
}
GRADIENT_GROUPS = {
    "causal_encoder": ("stem.", "stage0.", "stage1.", "stage2.",
                       "down1.", "down2."),
    "history_context": ("context0.", "context1.", "context2.",
                        "context_refine."),
    "step_coordinates": ("step_embedding",),
    "closure_body": ("cell.stem.", "cell.blocks."),
    "bounded_advection": ("cell.displacement_head.",),
    "finite_volume_flux": ("cell.flux_head.",),
    "bounded_source": ("cell.source_head.",),
}


def _classes():
    from models.multiple_shooting_closure import (  # noqa: PLC0415
        MultipleShootingClosure, MultipleShootingClosureConfig)
    return MultipleShootingClosureConfig, MultipleShootingClosure


def _seed(device: torch.device) -> None:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def _count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_config(*, tiny: bool = False):
    config_type, _operator = _classes()
    if not tiny:
        return config_type()
    return config_type(
        e0_channels=8, e1_channels=16, e2_channels=24,
        e0_blocks=1, e1_blocks=1, e2_blocks=1,
        context_channels=8, cell_channels=16, cell_blocks=1,
        step_embedding_channels=8)


def construct_pair(device: torch.device, *, tiny: bool = False):
    _config, operator = _classes()
    models, states = {}, {}
    for arm in ARMS:
        _seed(device)
        models[arm] = operator(
            model_config(tiny=tiny), ARM_MODE[arm]).float().to(device)
        states[arm] = _snapshot(models[arm])
    left, right = (states[arm] for arm in ARMS)
    equal = left.keys() == right.keys() and all(
        torch.equal(value, right[key]) for key, value in left.items())
    if not equal:
        raise RuntimeError("MS-Closure paired initial states differ")
    counts = {arm: _count(model) for arm, model in models.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"MS-Closure paired parameter counts differ: {counts}")
    architectures = {arm: model.architecture_config()
                     for arm, model in models.items()}
    differing = sorted(key for key in architectures[ARMS[0]]
                       if architectures[ARMS[0]][key]
                       != architectures[ARMS[1]][key])
    if differing != ["rollout_mode"]:
        raise RuntimeError(f"unexpected paired architecture differences: {differing}")
    common = dict(architectures[ARMS[0]])
    common.pop("rollout_mode")
    architecture_sha = _canonical_sha(common)
    initial_sha = base.state_sha(left)
    if not tiny:
        if set(counts.values()) != {EXPECTED_PARAMETERS}:
            raise RuntimeError(f"formal parameter count changed: {counts}")
        if len(left) != EXPECTED_STATE_TENSORS:
            raise RuntimeError("formal state tensor count changed")
        if initial_sha != EXPECTED_INITIAL_SHA256:
            raise RuntimeError("formal initial state SHA changed")
        if architecture_sha != EXPECTED_ARCHITECTURE_SHA256:
            raise RuntimeError("formal architecture SHA changed")
    return models, {
        "bitwise_identical": True,
        "parameter_counts": counts,
        "state_tensors": len(left),
        "initial_state_sha256": initial_sha,
        "architecture_sha256_without_rollout_mode": architecture_sha,
        "only_architecture_difference": "rollout_mode",
        "architecture": architectures,
    }


def set_normalizer(models: Mapping[str, nn.Module], normalizer) -> dict[str, object]:
    values = (normalizer.mean_input, normalizer.mean_target,
              normalizer.std_input, normalizer.std_target)
    for model in models.values():
        model.set_normalizer_statistics(*values)
    names = ("mean_input", "std_input", "mean_target", "std_target")
    left, right = (models[arm] for arm in ARMS)
    equal = {name: bool(torch.equal(getattr(left, name), getattr(right, name)))
             for name in names}
    if not all(equal.values()):
        raise RuntimeError("paired normalizer buffers differ")
    return {"bitwise_equal": equal,
            "state_sha256": base.state_sha(
                {name: getattr(left, name) for name in names}),
            **{name: getattr(left, name).detach().cpu().tolist()
               for name in names}}


def _gradient_groups(model: nn.Module) -> tuple[dict[str, bool], list[str]]:
    live = {group: False for group in GRADIENT_GROUPS}
    nonfinite = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            nonfinite.append(name)
            continue
        if not bool(torch.count_nonzero(gradient)):
            continue
        for group, prefixes in GRADIENT_GROUPS.items():
            if any(name == prefix or name.startswith(prefix) for prefix in prefixes):
                live[group] = True
    return live, nonfinite


def update_zero_contract(models, dataset, normalizer, device) -> dict[str, object]:
    history, target = dataset[0]
    history, _target = normalizer.preprocess(
        history[None].to(device), target[None].to(device))
    for model in models.values():
        model.eval()
    with torch.inference_mode():
        components = {arm: model.forward_components(history)
                      for arm, model in models.items()}
        changed_pressure = history.clone()
        changed_pressure[..., 2] = torch.randn_like(changed_pressure[..., 2])
        pressure_changed = models["carry_candidate"](changed_pressure)
    control = components["reset_control"]
    candidate = components["carry_candidate"]
    checks = {
        "prediction_bitwise_equal": torch.equal(
            control["prediction"], candidate["prediction"]),
        "first_segment_bitwise_equal": torch.equal(
            control["prediction"][:, :5], candidate["prediction"][:, :5]),
        "pressure_input_invariant": torch.equal(
            candidate["prediction"], pressure_changed),
        "physical_output_pressure_zero": bool(
            torch.count_nonzero(candidate["prediction"][..., 2]
                                * normalizer.std_target[2]
                                + normalizer.mean_target[2]) == 0),
        "flux_telescopes": float(
            candidate["flux_delta_global_sum"].abs().max()) <= 1e-6,
        "twenty_transition_calls": candidate["physical_uv"].shape[1] == 20,
        "four_segment_states": candidate["segment_starts"].shape[1] == 4,
    }
    if not all(checks.values()):
        raise RuntimeError(f"update-zero contract failed: {checks}")
    return {"checks": checks, "pass": True,
            "max_flux_global_sum": float(
                candidate["flux_delta_global_sum"].abs().max())}


def train_arm(name: str, model: nn.Module, dataset, schedule: np.ndarray,
              normalizer, device: torch.device) -> dict[str, object]:
    _seed(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(schedule))
    seen = {group: False for group in GRADIENT_GROUPS}
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()}
    u200_updated = None
    nonfinite_seen: set[str] = set()
    trace, interval = [], []
    started = time.perf_counter()
    for update, row in enumerate(schedule, 1):
        samples = [dataset[int(index)] for index in row]
        history = torch.stack([item[0] for item in samples]).to(device)
        target = torch.stack([item[1] for item in samples]).to(device)
        history, target = normalizer.preprocess(history, target)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        accuracy, details = base.accuracy_objective(
            prediction, target, normalizer, (1.0, 4.0, 1.0))
        mse_uv = (prediction[..., :2] - target[..., :2]).square().mean()
        loss = accuracy + 0.25 * mse_uv
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"{name} non-finite loss at {update}")
        loss.backward()
        live, nonfinite = _gradient_groups(model)
        for group, value in live.items():
            seen[group] |= value
        nonfinite_seen.update(nonfinite)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"{name} non-finite grad at {update}")
        optimizer.step()
        scheduler.step()
        if update == min(200, len(schedule)):
            current = dict(model.named_parameters())
            u200_updated = {
                group: any(
                    not torch.equal(initial_parameters[name], parameter.detach().cpu())
                    for name, parameter in current.items()
                    if any(name == prefix or name.startswith(prefix)
                           for prefix in prefixes))
                for group, prefixes in GRADIENT_GROUPS.items()}
        interval.append(float(loss.detach()))
        if update == 1 or update % 250 == 0 or update == len(schedule):
            trace.append({
                "update": update, "loss_mean": float(np.mean(interval)),
                "mse_uv": float(mse_uv.detach()),
                "grad_norm_before_clip": float(grad_norm.detach()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{key: float(value) for key, value in details.items()},
            })
            print(json.dumps({"event": "ms_closure_progress", "arm": name,
                              **trace[-1],
                              "wall_seconds": time.perf_counter() - started},
                             sort_keys=True), flush=True)
            interval.clear()
    endpoint_sha = base.state_sha(_snapshot(model))
    return {
        "endpoint_state_sha256": endpoint_sha,
        "updates": len(schedule),
        "wall_seconds": time.perf_counter() - started,
        "gradient_groups_seen": seen,
        "all_gradient_groups_seen": all(seen.values()),
        "u200_parameter_groups_updated": u200_updated,
        "all_u200_parameter_groups_updated": bool(
            u200_updated is not None and all(u200_updated.values())),
        "nonfinite_gradient_tensors_seen": sorted(nonfinite_seen),
        "trace": trace,
    }


def endpoint_mechanical_health(models, endpoints, dataset, normalizer,
                               device: torch.device) -> dict[str, object]:
    """Fit-only fail-fast checks that must precede any held field load."""
    items = [dataset[index] for index in (0, 1)]
    history = torch.stack([item[0] for item in items]).to(device)
    target = torch.stack([item[1] for item in items]).to(device)
    history, _target = normalizer.preprocess(history, target)
    reports = {}
    for arm, model in models.items():
        model.eval()
        with torch.inference_mode():
            components = model.forward_components(history)
            singles = torch.cat((model(history[:1]), model(history[1:2])))
            batched = components["prediction"]
            reversed_batch = model(history.flip(0)).flip(0)
            native_context = model.encode_history(history)
        geometry = components["solid_mask"] | components["outer_boundary_mask"]
        anchor = (history[..., :2] * model.std_input[:2]
                  + model.mean_input[:2])[:, -1]
        anchored = anchor[:, None].expand_as(components["physical_uv"])
        selected_prediction = torch.masked_select(
            components["physical_uv"],
            geometry[:, None].expand_as(components["physical_uv"]))
        selected_anchor = torch.masked_select(
            anchored, geometry[:, None].expand_as(anchored))
        bounds = {
            "displacement": float(components["displacement"].abs().max()),
            "raw_flux": float(components["raw_flux_normalized"].abs().max()),
            "source": float(components["source_normalized"].abs().max()),
            "flux_global_sum": float(
                components["flux_delta_global_sum"].abs().max()),
        }
        state_before = base.state_sha(_snapshot(model))
        model.set_reset_override(True)
        try:
            with torch.inference_mode():
                reset_components = model.forward_components(history)
                reset_context = model.encode_history(history)
        finally:
            model.set_reset_override(False)
        state_after = base.state_sha(_snapshot(model))
        batch_invariance_max_abs = max(
            float((singles - batched).abs().max()),
            float((reversed_batch - batched).abs().max()))
        checks = {
            "finite_state": bool(torch.isfinite(
                components["physical_uv"]).all()),
            "displacement_bounded": bounds["displacement"] <= 1.0 + 1e-6,
            "raw_flux_bounded": bounds["raw_flux"] <= 0.025 + 1e-7,
            "source_bounded": bounds["source"] <= 0.10 + 1e-6,
            "flux_telescopes": bounds["flux_global_sum"] <= 1e-6,
            "geometry_cells_exact_anchor": torch.equal(
                selected_prediction, selected_anchor),
            "solid_mask_present": bool(components["solid_mask"].any()),
            "outer_boundary_present": bool(
                components["outer_boundary_mask"].any()),
            "singleton_and_permutation_batch_invariant_max_abs_le_1e_5":
                batch_invariance_max_abs <= 1e-5,
            "override_state_unchanged": state_before == state_after,
            "override_context_bitwise_unchanged": torch.equal(
                native_context, reset_context),
            "override_first_segment_bitwise_unchanged": torch.equal(
                components["prediction"][:, :5],
                reset_components["prediction"][:, :5]),
            "endpoint_state_matches_training_seal": state_after
                == endpoints[arm]["endpoint_state_sha256"],
            "all_gradient_groups_seen": bool(
                endpoints[arm]["all_gradient_groups_seen"]),
            "all_u200_groups_updated": bool(
                endpoints[arm]["all_u200_parameter_groups_updated"]),
            "no_nonfinite_gradient": not bool(
                endpoints[arm]["nonfinite_gradient_tensors_seen"]),
        }
        reports[arm] = {
            "checks": checks,
            "pass": all(checks.values()),
            "bounds": bounds,
            "batch_invariance_max_abs": batch_invariance_max_abs,
            "geometry_fraction": float(geometry.float().mean()),
        }
    overall = all(report["pass"] for report in reports.values())
    return {"arms": reports, "pass": overall,
            "held_load_authorized": overall}


def seal_endpoint_files(models, endpoints, output_dir: Path,
                        schedule_sha: str) -> None:
    """Atomically persist both endpoint states before any held field load."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm, model in models.items():
        endpoint_path = output_dir / f"{arm}_u{UPDATES:05d}.pt"
        state = _snapshot(model)
        declared_sha = base.state_sha(state)
        if declared_sha != endpoints[arm]["endpoint_state_sha256"]:
            raise RuntimeError(f"{arm} endpoint changed before file seal")
        atomic.atomic_torch_save({
            "schema": SCHEMA, "arm": arm, "updates": UPDATES,
            "sample_stream_sha256": schedule_sha,
            "model_state_sha256": declared_sha,
            "model_state": state,
        }, endpoint_path)
        endpoints[arm]["endpoint_file"] = str(endpoint_path.resolve())
        endpoints[arm]["endpoint_file_sha256"] = atomic.file_sha256(
            endpoint_path)


def seal_preheld_report(
        output_dir: Path, *, git: Mapping[str, object],
        inventory_seals: Mapping[str, object], schedule_sha: str,
        identity: Mapping[str, object], normalizer: Mapping[str, object],
        update_zero: Mapping[str, object], endpoints: Mapping[str, object],
        mechanical: Mapping[str, object]) -> dict[str, object]:
    """Persist and verify every training-side fact before opening held fields."""
    payload = {
        "schema": SCHEMA,
        "stage": "PREHELD_ENDPOINTS_AND_MECHANICS_SEALED",
        "held_field_values_loaded": False,
        "git": git,
        "inventory_seals": inventory_seals,
        "sample_stream_sha256_both_arms": schedule_sha,
        "identity": identity,
        "normalizer": normalizer,
        "update_zero_contract": update_zero,
        "endpoints_with_training_traces": endpoints,
        "fit_endpoint_mechanical_health": mechanical,
    }
    path = output_dir / "preheld_seal.json"
    atomic.atomic_json_save(payload, path)
    digest = atomic.file_sha256(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if _canonical_sha(loaded) != _canonical_sha(payload):
        raise RuntimeError("preheld seal read-back verification failed")
    sha_path = output_dir / "preheld_seal.sha256"
    sha_path.write_text(f"{digest}  preheld_seal.json\n", encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "sha_file": str(sha_path.resolve()),
        "readback_canonical_json_verified": True,
    }


def _rename(value: object) -> object:
    if isinstance(value, dict):
        return {("reset_control" if key == "spatial_window_control"
                 else "carry_candidate" if key == "temporal_candidate"
                 else key): _rename(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename(item) for item in value]
    return value


def evaluate_pair(models, trajectories, windows, files, normalizer, device):
    raw = shared.evaluate_pair({
        "spatial_window_control": models["reset_control"],
        "temporal_candidate": models["carry_candidate"],
    }, trajectories, windows, files, normalizer, device)
    result = _rename(raw)
    trajectory_deltas = [
        float(row["delta_candidate_minus_control"]
              ["point_contribution_46_10_9"])
        for row in result["per_trajectory"].values()]
    # The preregistered gate says nonnegative, whereas the inherited helper
    # reports strictly positive trajectories.  Freeze the stated >= 0 rule
    # locally instead of silently weakening/changing the written contract.
    result["positive_trajectories"] = sum(
        value >= 0.0 for value in trajectory_deltas)
    result["positive_trajectories_definition"] = ">= 0.0"
    return result


def evaluate_slice(models: Mapping[str, nn.Module], trajectories, windows,
                   normalizer, device: torch.device,
                   start: int, stop: int) -> dict[str, object]:
    """Exact point metrics for one fixed future-frame slice."""
    dataset = base.SimWindowDataset(trajectories, windows)
    errors = {name: [] for name in models}
    for offset in range(0, len(dataset), 16):
        items = [dataset[index] for index in range(
            offset, min(offset + 16, len(dataset)))]
        history = torch.stack([item[0] for item in items]).to(device)
        target = torch.stack([item[1] for item in items]).numpy()
        history = (history - normalizer.mean_input) / normalizer.std_input
        for name, model in models.items():
            model.eval()
            with torch.inference_mode():
                prediction = model(history)
                prediction = (prediction * normalizer.std_target
                              + normalizer.mean_target)
            array = prediction.float().cpu().numpy()
            errors[name].append(base.error_rows(
                array[:, start:stop], target[:, start:stop]))
    scores = {name: shared._score_with_errors(np.concatenate(parts))
              for name, parts in errors.items()}
    names = tuple(models)
    result = {**scores,
              "start_horizon_one_based": start + 1,
              "stop_horizon_one_based_inclusive": stop}
    if len(names) == 2:
        result["delta_second_minus_first"] = shared._comparison(
            scores[names[1]], scores[names[0]])
    return result


def mechanism_override(candidate, native_evaluation, trajectories, windows,
                       files, normalizer, device) -> dict[str, object]:
    state_before = base.state_sha(_snapshot(candidate))
    late_native = evaluate_slice(
        {"carry_native": candidate}, trajectories, windows,
        normalizer, device, 10, 20)["carry_native"]
    candidate.set_reset_override(True)
    try:
        override = loao._evaluate_same_model(
            candidate, trajectories, windows, files, normalizer, device)
        late_override = evaluate_slice(
            {"carry_reset_override": candidate}, trajectories, windows,
            normalizer, device, 10, 20)["carry_reset_override"]
    finally:
        candidate.set_reset_override(False)
    state_after = base.state_sha(_snapshot(candidate))
    if state_before != state_after:
        raise RuntimeError("reset override changed endpoint state")
    native = native_evaluation["aggregate"]["carry_candidate"]
    return {
        "same_checkpoint_state_unchanged": True,
        "override": override,
        "delta_native_minus_reset_override": shared._comparison(
            native, override["aggregate"]),
        "late_horizons_11_20": {
            "native": late_native,
            "reset_override": late_override,
            "delta_native_minus_reset": shared._comparison(
                late_native, late_override),
        },
        "causal_gain_claim": False,
    }


def state_diagnostics(models, trajectories, windows, normalizer,
                      device, panel_windows: int = 32) -> dict[str, object]:
    dataset = base.SimWindowDataset(trajectories, windows)
    result = {}
    for arm, model in models.items():
        sum_square = torch.zeros(20, dtype=torch.float64)
        count = 0
        maxima = {"displacement": 0.0, "raw_flux": 0.0,
                  "flux_delta": 0.0, "source": 0.0,
                  "flux_global_sum": 0.0}
        first_five_override_rms = None
        for start in range(0, min(panel_windows, len(dataset)), 8):
            items = [dataset[index] for index in range(
                start, min(start + 8, panel_windows, len(dataset)))]
            history = torch.stack([item[0] for item in items]).to(device)
            target = torch.stack([item[1] for item in items]).to(device)
            history, _target = normalizer.preprocess(history, target)
            model.eval()
            with torch.inference_mode():
                components = model.forward_components(history)
            physical = components["physical_uv"].double().cpu()
            sum_square += physical.square().sum(dim=(0, 2, 3, 4))
            count += int(physical.shape[0] * physical.shape[2]
                         * physical.shape[3] * physical.shape[4])
            maxima["displacement"] = max(
                maxima["displacement"],
                float(components["displacement"].abs().max()))
            maxima["raw_flux"] = max(
                maxima["raw_flux"],
                float(components["raw_flux_normalized"].abs().max()))
            maxima["flux_delta"] = max(
                maxima["flux_delta"],
                float(components["flux_delta_normalized"].abs().max()))
            maxima["source"] = max(
                maxima["source"],
                float(components["source_normalized"].abs().max()))
            maxima["flux_global_sum"] = max(
                maxima["flux_global_sum"],
                float(components["flux_delta_global_sum"].abs().max()))
            if arm == "carry_candidate" and start == 0:
                model.set_reset_override(True)
                try:
                    with torch.inference_mode():
                        reset = model(history)
                finally:
                    model.set_reset_override(False)
                native = components["prediction"]
                first_five_override_rms = float(
                    (native[:, :5] - reset[:, :5]).square().mean().sqrt())
        result[arm] = {
            "panel_windows": min(panel_windows, len(dataset)),
            "physical_state_rms_by_horizon": (sum_square / count).sqrt().tolist(),
            "maxima": maxima,
            "first5_native_vs_reset_override_rms": first_five_override_rms,
        }
    return result


def decide(evaluation, endpoints, override) -> dict[str, object]:
    delta = evaluation["delta_candidate_minus_control"]
    prefix_deltas = [
        float(evaluation["per_prefix_horizon"][str(index)]
              ["delta_candidate_minus_control"]
              ["point_contribution_46_10_9"])
        for index in range(1, 21)]
    override_delta = override["delta_native_minus_reset_override"]
    late_override_delta = override["late_horizons_11_20"][
        "delta_native_minus_reset"]
    checks = {
        "delta_point": float(delta["point_contribution_46_10_9"])
            >= GATE["delta_point_min"],
        "component_floor": all(float(delta[key]) >= GATE["component_floor"]
                               for key in base.POINT_WEIGHTS),
        "positive_trajectories": int(evaluation["positive_trajectories"])
            >= GATE["positive_trajectories_min"],
        "worst_trajectory": float(evaluation["worst_trajectory_delta_point"])
            >= GATE["worst_trajectory_delta_point_min"],
        "candidate_absolute_point": float(
            evaluation["aggregate"]["carry_candidate"]
            ["point_contribution_46_10_9"])
            >= GATE["candidate_absolute_point_min"],
        "all_cumulative_prefix_deltas_nonnegative": min(prefix_deltas)
            >= GATE["all_cumulative_prefix_delta_min"],
        "native_minus_reset_override": float(
            override_delta["point_contribution_46_10_9"])
            >= GATE["native_minus_reset_override_min"],
        "late_native_minus_reset_override": float(
            late_override_delta["point_contribution_46_10_9"])
            >= GATE["late_native_minus_reset_override_min"],
        "gradient_health": all(
            endpoint["all_gradient_groups_seen"]
            and endpoint["all_u200_parameter_groups_updated"]
            and not endpoint["nonfinite_gradient_tensors_seen"]
            for endpoint in endpoints.values()),
    }
    return {**GATE, "checks": checks, "pass": all(checks.values()),
            "minimum_cumulative_prefix_delta": min(prefix_deltas)}


def smoke(device: torch.device) -> dict[str, object]:
    models, identity = construct_pair(device, tiny=True)
    normalizer = base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), device)
    buffers = set_normalizer(models, normalizer)
    trajectory = torch.randn(40, 16, 16, 3)
    trajectory[:, 4:8, 5:9, :2] = 0
    dataset = base.SimWindowDataset(
        [trajectory], [base.SimWindow(0, 0), base.SimWindow(0, 0)])
    update_zero = update_zero_contract(models, dataset, normalizer, device)
    schedule = np.zeros((2, 1), dtype=np.int64)
    endpoints = {arm: train_arm(
        arm, model, dataset, schedule, normalizer, device)
        for arm, model in models.items()}
    mechanical = endpoint_mechanical_health(
        models, endpoints, dataset, normalizer, device)
    if not mechanical["pass"]:
        raise RuntimeError(f"CPU smoke mechanical health failed: {mechanical}")
    return {"schema": SCHEMA, "smoke": "PASS", "identity": identity,
            "normalizer": buffers, "update_zero": update_zero,
            "endpoints": endpoints, "endpoint_mechanical": mechanical}


def resource_estimate() -> dict[str, object]:
    return {
        "basis": "measured_synthetic_batch8_cuda_preflight_2026_09_04",
        "parameters_per_arm": EXPECTED_PARAMETERS,
        "fp32_parameter_mib": EXPECTED_PARAMETERS * 4 / 1024 ** 2,
        "device": "NVIDIA_GeForce_RTX_4090",
        "synthetic_shape": [8, 20, 32, 64, 3],
        "measured_peak_allocated_gib": 1.7775917053222656,
        "measured_first_update_seconds": 1.021781999617815,
        "measured_warm_update_median_seconds": 0.16396280005574226,
        "estimated_2000_update_minutes_per_arm_excluding_io": 5.47,
        "largest_cost": "20 shared closure calls and 40 grid_sample calls",
        "preflight_status": "PASS",
    }


def run(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError(
            f"formal run requires --execute --acknowledge {ACKNOWLEDGEMENT}")
    if args.updates != UPDATES or args.batch_size != BATCH_SIZE:
        raise ValueError("formal contract freezes updates2000/batch8")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal screen requires CUDA")
    if args.output_dir is None or args.sim_dir is None:
        raise ValueError("formal screen requires sim/output paths")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    git = base.git_report()
    if not git["clean"]:
        raise RuntimeError(f"formal screen requires clean Git: {git['porcelain']}")

    inventory = base.inventory(args.sim_dir)
    fit_rows = [row for row in inventory if int(row["aoa"]) in FIT_AOAS]
    held_rows = [row for row in inventory if int(row["aoa"]) == HELD_AOA]
    protocol = loao.protocol_for(HELD_AOA)
    inventory_seals = loao.validate_inventory_seals(protocol, fit_rows, held_rows)
    fit_trajectories, fit_windows, _files = base.load_fields(args.sim_dir, fit_rows)
    normalizer = base.ChannelNormalizer(
        *base.window_statistics(fit_trajectories, fit_windows), device=device)
    dataset = base.SimWindowDataset(fit_trajectories, fit_windows)
    schedule = np.random.default_rng(SCHEDULE_SEED).integers(
        0, len(dataset), size=(UPDATES, BATCH_SIZE), dtype=np.int64)
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()
    if schedule_sha != EXPECTED_SAMPLE_STREAM_SHA256:
        raise RuntimeError("formal paired sample stream SHA changed")
    models, identity = construct_pair(device)
    buffers = set_normalizer(models, normalizer)
    update_zero = update_zero_contract(models, dataset, normalizer, device)
    endpoints = {arm: train_arm(
        arm, model, dataset, schedule, normalizer, device)
        for arm, model in models.items()}
    if not all(endpoint["endpoint_state_sha256"]
               for endpoint in endpoints.values()):
        raise RuntimeError("held load blocked: endpoint seal missing")
    seal_endpoint_files(models, endpoints, args.output_dir, schedule_sha)
    if not all(endpoint.get("endpoint_file_sha256")
               for endpoint in endpoints.values()):
        raise RuntimeError("held load blocked: endpoint file seal missing")
    mechanical = endpoint_mechanical_health(
        models, endpoints, dataset, normalizer, device)
    if not mechanical["pass"]:
        raise RuntimeError(
            "STOP_M17_IMPLEMENTATION before held load: endpoint mechanical "
            + json.dumps(mechanical, sort_keys=True))
    preheld_seal = seal_preheld_report(
        args.output_dir, git=git, inventory_seals=inventory_seals,
        schedule_sha=schedule_sha, identity=identity, normalizer=buffers,
        update_zero=update_zero, endpoints=endpoints,
        mechanical=mechanical)
    del dataset, fit_trajectories
    gc.collect(); torch.cuda.empty_cache()

    held_trajectories, held_windows, held_files = base.load_fields(
        args.sim_dir, held_rows)
    evaluation = evaluate_pair(
        models, held_trajectories, held_windows, held_files,
        normalizer, device)
    evaluation["horizons_11_20"] = evaluate_slice(
        {"reset_control": models["reset_control"],
         "carry_candidate": models["carry_candidate"]},
        held_trajectories, held_windows, normalizer, device, 10, 20)
    diagnostics = state_diagnostics(
        models, held_trajectories, held_windows, normalizer, device)
    override = mechanism_override(
        models["carry_candidate"], evaluation, held_trajectories,
        held_windows, held_files, normalizer, device)
    gate = decide(evaluation, endpoints, override)
    result = {
        "schema": SCHEMA,
        "evidence_class": "selection_reused_whole_AoA10_Sim",
        "git": git,
        "protocol": {
            "held_aoa": HELD_AOA, "fit_aoas": list(FIT_AOAS),
            "single_causal_variable": "detached_cross_segment_state_carry",
            "updates": UPDATES, "batch_size": BATCH_SIZE,
            "optimizer": "AdamW", "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR_Tmax_2000",
            "loss": "accuracy_1_4_1_uv_plus_0.25_mse_uv",
            "augmentation": "none", "teacher": "none", "uq": "none",
            "pressure_input": "structurally_sliced_before_encoder",
        },
        **inventory_seals,
        "sample_stream_sha256_both_arms": schedule_sha,
        "held_field_values_loaded_before_both_endpoint_seals": False,
        "preheld_seal": preheld_seal,
        "identity": identity, "normalizer": buffers,
        "update_zero_contract": update_zero,
        "fit_endpoint_mechanical_health_before_held_load": mechanical,
        "resource_estimate": resource_estimate(),
        "endpoints": endpoints,
        "evaluation": evaluation,
        "state_and_bound_diagnostics": diagnostics,
        "carry_reset_same_checkpoint_override": override,
        "gate": gate,
        "decision": ("PASS_M17_SIM_DISCOVERY"
                     if gate["pass"] else "STOP_M17_SIM_DISCOVERY"),
        "next_action": ("HELD20_CONFIRMATION_ALLOWED"
                        if gate["pass"] else "NO_HELD20_REAL_UQ_OR_SCAN"),
    }
    result_path = args.output_dir / "result.json"
    atomic.atomic_json_save(result, result_path)
    result_sha = atomic.file_sha256(result_path)
    (args.output_dir / "result.sha256").write_text(
        f"{result_sha}  result.json\n", encoding="utf-8")
    print(json.dumps({
        "event": "ms_closure_complete", "result": str(result_path),
        "result_sha256": result_sha,
        "decision": result["decision"],
    }, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "smoke", "formal"),
                        default="contract")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.mode == "contract":
        models, identity = construct_pair(torch.device("cpu"))
        del models
        print(json.dumps({"schema": SCHEMA, "identity": identity,
                          "resource_estimate": resource_estimate()},
                         sort_keys=True, indent=2))
    elif args.mode == "smoke":
        print(json.dumps(smoke(device), sort_keys=True, indent=2))
    else:
        run(args, device)


if __name__ == "__main__":
    main()
