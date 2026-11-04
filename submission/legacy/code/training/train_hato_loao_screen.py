#!/usr/bin/env python3
"""M15 paired whole-AoA Sim2k screen for the HATO backbone.

This is deliberately a backbone discovery experiment, not a B210 adapter.
Both arms retain the same HATO tensors, fit-only normalizer, dropout stream,
and pre-generated batches.  The candidate lets the hierarchical operator mix
the observed-time axis; the control runs its otherwise matched window mixer
on the spatial axis.  Held AoA10 *field values* are opened only after both
in-memory endpoint state hashes have been sealed.  No checkpoint, UQ path,
augmentation, or teacher is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
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


SCHEMA = "m15_hato_whole_aoa_sim2k_v1"
ACKNOWLEDGEMENT = "RUN_M15_HATO_SIM2K"
ARMS = ("spatial_window_control", "temporal_candidate")
ARM_MODE = {
    "spatial_window_control": "spatial_window_control",
    "temporal_candidate": "temporal_candidate",
}
SEED = 210
SCHEDULE_SEED = 220
UPDATES = 2_000
BATCH_SIZE = 8
LR = 3.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
HELD_AOA = 10
FIT_AOAS = (0, 5, 15, 20)
POINT_WEIGHTS = {
    "rel_l2_score": .46,
    "tke_rel_l2_score": .10,
    "mvpe_rel_l2_score": .09,
}
NORMALIZER_BUFFER_NAMES = ("persistence_scale", "persistence_bias", "target_std")
REQUIRED_GRADIENT_GROUPS = {
    "attention_or_future_query": ("attention", "_future"),
    "observed_time_embedding": ("observed_time",),
    "mean_head": ("mean_head",),
    "fluctuation_head": ("fluctuation_head",),
}
GATE = {
    "delta_point_min": .08,
    "component_floor": -.05,
    "positive_trajectories_min": 15,
    "worst_trajectory_delta_point_min": -.05,
}


def _model_classes():
    """Delay the new-backbone import so contract tests do not mask its work."""
    try:
        from models.hierarchical_axial_temporal_operator import (  # noqa: PLC0415
            HATOConfig, HierarchicalAxialTemporalOperator)
    except ImportError as error:
        raise RuntimeError(
            "M15 HATO core is unavailable; expected "
            "models.hierarchical_axial_temporal_operator") from error
    return HATOConfig, HierarchicalAxialTemporalOperator


def _config_field_names(config_type) -> set[str]:
    if hasattr(config_type, "__dataclass_fields__"):
        return set(config_type.__dataclass_fields__)
    # A lightweight fallback keeps this harness useful if the model chooses a
    # plain typed constructor.  Its default configuration remains authoritative.
    return set()


def model_config(*, tiny: bool = False):
    """Construct one explicit-time HATO topology without UQ outputs.

    HATO owns the exact dataclass field vocabulary.  The fixed values below
    are applied only where that vocabulary exposes them; this avoids turning a
    trainer import into a second, divergent architecture specification.
    """
    HATOConfig, _unused = _model_classes()
    # The formal topology is defined by the model's frozen defaults, rather
    # than re-declared here with a stale, parallel list of widths/blocks.
    # Smoke only shrinks the three explicitly documented encoder widths.
    if not tiny:
        return HATOConfig()
    # Preserve default group/head divisibility: E1 / 6 and E2 / 8.
    values = {"e0_channels": 8, "e1_channels": 24, "e2_channels": 32}
    fields = _config_field_names(HATOConfig)
    return HATOConfig(**{key: value for key, value in values.items()
                         if not fields or key in fields})


def _architecture_config(model: nn.Module) -> object:
    value = model.architecture_config()
    if is_dataclass(value):
        return asdict(value)
    return value


def _seed(device: torch.device) -> None:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def construct_pair(device: torch.device, *, tiny: bool = False):
    """Build bitwise-identical tensors; mode is the sole intervention."""
    _config, HATO = _model_classes()
    models: dict[str, nn.Module] = {}
    starts: dict[str, dict[str, torch.Tensor]] = {}
    for arm in ARMS:
        _seed(device)
        model = HATO(
            model_config(tiny=tiny), interaction_mode=ARM_MODE[arm])
        models[arm] = model.float().to(device)
        starts[arm] = _snapshot(models[arm])
    control, candidate = starts[ARMS[0]], starts[ARMS[1]]
    if control.keys() != candidate.keys() or any(
            not torch.equal(value, candidate[key])
            for key, value in control.items()):
        raise RuntimeError("M15 HATO paired arms are not bitwise identical")
    counts = {arm: _count_parameters(model) for arm, model in models.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"M15 HATO paired parameter counts differ: {counts}")
    configs = {arm: _architecture_config(model) for arm, model in models.items()}
    return models, {
        "state_sha256": base.state_sha(control),
        "state_tensors": len(control),
        "bitwise_identical": True,
        "parameter_counts": counts,
        "architecture_config": configs,
        "only_interaction_difference": "interaction_mode",
    }


def set_pair_normalizer_statistics(models: Mapping[str, nn.Module], normalizer):
    values = (normalizer.mean_input, normalizer.mean_target,
              normalizer.std_input, normalizer.std_target)
    for model in models.values():
        model.set_normalizer_statistics(*values)
    missing = [name for name in NORMALIZER_BUFFER_NAMES
               if not all(hasattr(model, name) for model in models.values())]
    if missing:
        raise RuntimeError("HATO misses normalizer buffers: " + ", ".join(missing))
    left, right = (models[arm] for arm in ARMS)
    equal = {name: bool(torch.equal(getattr(left, name), getattr(right, name)))
             for name in NORMALIZER_BUFFER_NAMES}
    if not all(equal.values()):
        raise RuntimeError("paired normalizer buffers differ: " + repr(equal))
    return {
        "fit_normalizer_sha256": base.state_sha({
            "mean_input": normalizer.mean_input,
            "mean_target": normalizer.mean_target,
            "std_input": normalizer.std_input,
            "std_target": normalizer.std_target,
        }),
        "setter_called_both_arms": True,
        "normalizer_buffers_bitwise_equal": equal,
        "normalizer_buffer_sha256": base.state_sha({
            name: getattr(left, name) for name in NORMALIZER_BUFFER_NAMES}),
    }


def batch_stream(dataset, schedule: np.ndarray):
    for row in schedule:
        samples = [dataset[int(index)] for index in row]
        yield (torch.stack([item[0] for item in samples]),
               torch.stack([item[1] for item in samples]))


def _finite(model: nn.Module) -> bool:
    return all(not value.is_floating_point() or bool(torch.isfinite(value).all())
               for value in model.state_dict().values())


def _gradient_audit(model: nn.Module) -> dict[str, object]:
    """Record whether the new temporal path and both output roles are live."""
    totals = {"tensor_count": 0, "element_count": 0,
              "nonzero_tensor_count": 0, "nonzero_element_count": 0,
              "nonfinite_gradient_tensors": []}
    groups = {key: False for key in REQUIRED_GRADIENT_GROUPS}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        totals["tensor_count"] += 1
        totals["element_count"] += parameter.numel()
        grad = parameter.grad
        if grad is None:
            continue
        if not bool(torch.isfinite(grad).all()):
            totals["nonfinite_gradient_tensors"].append(name)
            continue
        nonzero = int(torch.count_nonzero(grad.detach()).item())
        if nonzero:
            totals["nonzero_tensor_count"] += 1
            totals["nonzero_element_count"] += nonzero
            for group, markers in REQUIRED_GRADIENT_GROUPS.items():
                if any(marker in name for marker in markers):
                    groups[group] = True
    return {
        "all_seen_gradients_finite": not totals["nonfinite_gradient_tensors"],
        "trainable_tensor_coverage": totals["nonzero_tensor_count"] / max(totals["tensor_count"], 1),
        "trainable_element_coverage": totals["nonzero_element_count"] / max(totals["element_count"], 1),
        "nonzero_tensor_count": totals["nonzero_tensor_count"],
        "trainable_tensor_count": totals["tensor_count"],
        "required_groups_live": groups,
        "all_required_groups_live": all(groups.values()),
        "nonfinite_gradient_tensors": totals["nonfinite_gradient_tensors"],
    }


def _component_contract(model: nn.Module, history: torch.Tensor) -> dict[str, object]:
    """Ensure the new model exposes, and can execute, its decomposition API."""
    with torch.inference_mode():
        components = model.forward_components(history)
    if not isinstance(components, Mapping):
        raise TypeError("HATO forward_components must return a mapping")
    return {"forward_components_mapping": True,
            "component_keys": sorted(str(key) for key in components)}


def attention_execution_contract(models: Mapping[str, nn.Module],
                                 device: torch.device, *, formal: bool) -> dict[str, object]:
    """Execute the paired private sequence layouts at formal pyramid sizes.

    E1 is 16x32 but pads to 16x35, hence 560 length-20 sequences; E2 is
    8x16 but pads to 8x20, hence 160.  This catches the subtle failure mode
    where the spatial control gets a different attention batch or sequence
    length even though its parameter tensors match the candidate's.
    """
    records = {}
    expected = {"e1": [560, 20, 192], "e2": [160, 20, 256]}
    for level, height, width, name in (("e1", 16, 32, "e1_blocks"),
                                       ("e2", 8, 16, "e2_blocks")):
        sequences = {}
        outputs = {}
        for arm in ARMS:
            model = models[arm]
            block = getattr(model, name)[0]
            channels = int(getattr(model.config, f"{level}_channels"))
            values = torch.zeros(1, 20, channels, height, width, device=device)
            if arm == "temporal_candidate":
                packed, padded_height, padded_width = block._temporal_sequences(values)
            else:
                packed, padded_height, padded_width = block._spatial_sequences(values)
            with torch.inference_mode():
                output = block(values)
            sequences[arm] = list(packed.shape)
            outputs[arm] = {"shape": list(output.shape),
                            "finite": bool(torch.isfinite(output).all()),
                            "padded_height": int(padded_height),
                            "padded_width": int(padded_width)}
        equal = sequences[ARMS[0]] == sequences[ARMS[1]]
        length_twenty = all(shape[1] == 20 for shape in sequences.values())
        record = {"sequence_shape": sequences, "sequence_shape_bitwise_equal": equal,
                  "sequence_length_twenty": length_twenty, "block_execution": outputs}
        if formal:
            record["expected_formal_shape"] = expected[level]
            record["matches_expected_formal_shape"] = (
                sequences[ARMS[0]] == expected[level]
                and sequences[ARMS[1]] == expected[level])
        records[level] = record
    passed = all(value["sequence_shape_bitwise_equal"]
                 and value["sequence_length_twenty"]
                 and all(item["finite"] for item in value["block_execution"].values())
                 and (not formal or value["matches_expected_formal_shape"])
                 for value in records.values())
    return {"formal": formal, "levels": records, "pass": passed}


def train_arm(name: str, model: nn.Module, dataset, schedule: np.ndarray,
              normalizer, device: torch.device, updates: int) -> dict[str, object]:
    _seed(device)  # same dropout/random stream for sequential paired arms
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates)
    trace, interval = [], []
    gradient_seen = {key: False for key in REQUIRED_GRADIENT_GROUPS}
    nonfinite_gradient_tensors: set[str] = set()
    latest_gradient_audit: dict[str, object] | None = None
    started = time.perf_counter()
    for update, (history, target) in enumerate(batch_stream(dataset, schedule), 1):
        model.train()
        history, target = normalizer.preprocess(history.to(device), target.to(device))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        accuracy, details = base.accuracy_objective(
            prediction, target, normalizer, (1.0, 4.0, 1.0))
        mse = (prediction - target).square().mean()
        loss = accuracy + .25 * mse
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"{name} non-finite loss at update {update}")
        loss.backward()
        latest_gradient_audit = _gradient_audit(model)
        for key, value in latest_gradient_audit["required_groups_live"].items():
            gradient_seen[key] = gradient_seen[key] or bool(value)
        nonfinite_gradient_tensors.update(
            latest_gradient_audit["nonfinite_gradient_tensors"])
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"{name} non-finite gradient at update {update}")
        optimizer.step()
        scheduler.step()
        interval.append(float(loss.detach()))
        if update == 1 or update % 250 == 0 or update == updates:
            row = {"update": update, "loss_mean": float(np.mean(interval)),
                   "mse": float(mse.detach()),
                   "grad_norm_before_clip": float(grad_norm.detach()),
                   **{key: float(value) for key, value in details.items()}}
            trace.append(row)
            print(json.dumps({"event": "m15_hato_progress", "arm": name,
                              **row, "wall_seconds": time.perf_counter() - started},
                             sort_keys=True), flush=True)
            interval.clear()
    if not _finite(model):
        raise FloatingPointError(f"{name} endpoint contains non-finite state")
    assert latest_gradient_audit is not None
    return {"endpoint_state_sha256": base.state_sha(_snapshot(model)),
            "updates": updates, "wall_seconds": time.perf_counter() - started,
            "point_forward_only": True, "uq_forward_executed": False,
            "gradient_audit": {
                **latest_gradient_audit,
                "required_groups_seen_over_training": gradient_seen,
                "all_required_groups_seen_over_training": all(gradient_seen.values()),
                "nonfinite_gradient_tensors_over_training": sorted(nonfinite_gradient_tensors),
            },
            "trace": trace}


def _physical_error_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return base.error_rows(prediction, target)


def _score_with_errors(errors: np.ndarray) -> dict[str, float | int]:
    result = base.score(errors)
    # Keep raw physical errors beside the starting-kit score transform.
    means = errors.mean(axis=0, dtype=np.float64)
    return {**result, "mean_physical_errors": {
        "rel_l2_error": float(means[0]), "tke_rel_l2_error": float(means[1]),
        "mvpe_rel_l2_error": float(means[2]),
    }}


def _comparison(candidate: Mapping[str, object], control: Mapping[str, object]):
    scores = base.comparison(candidate, control)
    raw_c, raw_k = candidate["mean_physical_errors"], control["mean_physical_errors"]
    return {**scores, "physical_error_delta_candidate_minus_control": {
        key: float(raw_c[key]) - float(raw_k[key]) for key in raw_c}}


def evaluate_pair(models: Mapping[str, nn.Module], trajectories, windows,
                  window_files: list[str], normalizer, device: torch.device):
    counts = {name: window_files.count(name) for name in sorted(set(window_files))}
    if (len(counts) != base.EXPECTED_FILES_PER_AOA
            or set(counts.values()) != {base.EXPECTED_WINDOWS_PER_TRAJECTORY}):
        raise RuntimeError(f"held panel changed: {counts}")
    dataset = base.SimWindowDataset(trajectories, windows)
    errors = {arm: [] for arm in ARMS}
    prefix_errors = {arm: [[] for _ in range(20)] for arm in ARMS}
    labels: list[str] = []
    digests = {arm: hashlib.sha256() for arm in ARMS}
    for start in range(0, len(dataset), 16):
        indices = range(start, min(start + 16, len(dataset)))
        items = [dataset[index] for index in indices]
        history = torch.stack([item[0] for item in items]).to(device)
        target = torch.stack([item[1] for item in items]).numpy()
        history = (history - normalizer.mean_input) / normalizer.std_input
        for arm, model in models.items():
            model.eval()
            with torch.inference_mode():
                predicted = model(history)
                predicted = predicted * normalizer.std_target + normalizer.mean_target
            array = np.ascontiguousarray(predicted.float().cpu().numpy())
            if not np.isfinite(array).all():
                raise FloatingPointError(f"{arm} held prediction is non-finite")
            digests[arm].update(array.tobytes())
            errors[arm].append(_physical_error_rows(array, target))
            # TKE is a temporal-fluctuation metric: a single output frame has
            # no temporal fluctuation.  Report cumulative prefixes instead of
            # pretending that 20 separate h=1 TKE values are meaningful.
            for prefix in range(1, 21):
                prefix_errors[arm][prefix - 1].append(_physical_error_rows(
                    array[:, :prefix], target[:, :prefix]))
        labels.extend(window_files[index] for index in indices)
    joined = {arm: np.concatenate(parts) for arm, parts in errors.items()}
    aggregate = {arm: _score_with_errors(value) for arm, value in joined.items()}
    per_prefix_horizon = {}
    for prefix in range(1, 21):
        rows = {arm: np.concatenate(prefix_errors[arm][prefix - 1]) for arm in ARMS}
        control, candidate = (_score_with_errors(rows[arm]) for arm in ARMS)
        per_prefix_horizon[str(prefix)] = {
            ARMS[0]: control, ARMS[1]: candidate,
            "delta_candidate_minus_control": _comparison(candidate, control),
            "prefix_frames": prefix,
            "tke_degenerate_single_frame": prefix == 1,
        }
    label_array = np.asarray(labels)
    per_trajectory = {}
    deltas = []
    for name in sorted(set(labels)):
        selected = label_array == name
        control, candidate = (_score_with_errors(joined[arm][selected]) for arm in ARMS)
        delta = _comparison(candidate, control)
        deltas.append(float(delta["point_contribution_46_10_9"]))
        per_trajectory[name] = {ARMS[0]: control, ARMS[1]: candidate,
                                "delta_candidate_minus_control": delta}
    delta = _comparison(aggregate[ARMS[1]], aggregate[ARMS[0]])
    return {
        "aggregate": aggregate,
        "delta_candidate_minus_control": delta,
        "per_trajectory": per_trajectory,
        "per_prefix_horizon": per_prefix_horizon,
        "prediction_sha256": {arm: digest.hexdigest() for arm, digest in digests.items()},
        "trajectory_count": len(per_trajectory),
        "windows_per_trajectory": base.EXPECTED_WINDOWS_PER_TRAJECTORY,
        "positive_trajectories": sum(value > 0.0 for value in deltas),
        "worst_trajectory_delta_point": min(deltas),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _one_model(arm: str, device: torch.device) -> nn.Module:
    _config, HATO = _model_classes()
    _seed(device)
    return HATO(model_config(), interaction_mode=ARM_MODE[arm]).float().to(device)


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _latency_arm(arm: str, device: torch.device) -> float:
    model = None
    history = None
    try:
        model = _one_model(arm, device).eval()
        history = torch.zeros(16, 20, 64, 128, 3, device=device)
        with torch.inference_mode():
            for _ in range(3):
                model(history)
            samples = []
            for _ in range(12):
                _sync(device)
                started = time.perf_counter()
                model(history)
                _sync(device)
                samples.append(time.perf_counter() - started)
        return statistics.median(samples)
    finally:
        del history, model
        _clear_cuda()


def _train_peak_arm(arm: str, device: torch.device) -> int:
    model = optimizer = x = y = loss = None
    try:
        model = _one_model(arm, device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        x = torch.randn(8, 20, 64, 128, 3, device=device)
        y = torch.randn(8, 20, 64, 128, 3, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - y).square().mean()
        loss.backward()
        optimizer.step()
        _sync(device)
        return int(torch.cuda.max_memory_allocated(device))
    finally:
        del loss, y, x, optimizer, model
        _clear_cuda()


def resource_preflight(device: torch.device) -> dict[str, object]:
    """Health record only: resource values do not veto a structural signal."""
    counts = {}
    for arm in ARMS:
        model = None
        try:
            model = _one_model(arm, device)
            counts[arm] = _count_parameters(model)
        finally:
            del model
            _clear_cuda()
    result: dict[str, object] = {"parameter_counts": counts,
                                 "models_resident_simultaneously": 1}
    if device.type != "cuda":
        return {**result, "latency": {"measured": False, "reason": "CUDA required"},
                "train_peak_vram": {"measured": False, "reason": "CUDA required"}}
    try:
        latency = {arm: _latency_arm(arm, device) for arm in ARMS}
        peaks = {arm: _train_peak_arm(arm, device) for arm in ARMS}
    except RuntimeError as error:
        _clear_cuda()
        if not _is_oom(error):
            raise
        return {**result,
                "health": {"status": "FAIL_OOM", "message": str(error),
                           "models_resident_simultaneously": 1},
                "latency": {"measured": False, "reason": "CUDA OOM during sequential preflight"},
                "train_peak_vram": {"measured": False, "reason": "CUDA OOM during sequential preflight"}}
    return {**result, "health": {"status": "PASS",
                                    "models_resident_simultaneously": 1},
            "latency": {"measured": True, "batch": 16,
                        "input_shape_per_sample": [20, 64, 128, 3],
                        "median_seconds": latency},
            "train_peak_vram": {"measured": True, "batch": 8, "bytes": peaks}}


def decide_gate(evaluation: Mapping[str, object], identity: Mapping[str, object],
                attention_contract: Mapping[str, object]):
    delta = evaluation["delta_candidate_minus_control"]
    checks = {
        "initial_state_bitwise_equal": bool(identity["bitwise_identical"]),
        "only_interaction_differs": identity["only_interaction_difference"] == "interaction_mode",
        "attention_execution_contract": bool(attention_contract["pass"]),
        "delta_point": float(delta["point_contribution_46_10_9"]) >= GATE["delta_point_min"],
        "component_floor": all(float(delta[key]) >= GATE["component_floor"]
                               for key in POINT_WEIGHTS),
        "positive_trajectories": int(evaluation["positive_trajectories"])
            >= GATE["positive_trajectories_min"],
        "worst_trajectory": float(evaluation["worst_trajectory_delta_point"])
            >= GATE["worst_trajectory_delta_point_min"],
        "complete_panel": (int(evaluation["trajectory_count"]) == base.EXPECTED_FILES_PER_AOA
                            and int(evaluation["windows_per_trajectory"])
                            == base.EXPECTED_WINDOWS_PER_TRAJECTORY),
    }
    return {**GATE, "checks": checks, "pass": all(checks.values())}


def require_endpoint_seals(endpoints: Mapping[str, Mapping[str, object]]) -> None:
    """The sole gate immediately before the first held field-value load."""
    missing = [arm for arm in ARMS
               if not endpoints.get(arm, {}).get("endpoint_state_sha256")]
    if missing:
        raise RuntimeError(
            "held seal guard: endpoint state hash missing for " + ", ".join(missing))


def smoke(device: torch.device) -> dict[str, object]:
    models, identity = construct_pair(device, tiny=True)
    normalizer = base.ChannelNormalizer(torch.zeros(3), torch.zeros(3),
                                        torch.ones(3), torch.ones(3), device)
    buffers = set_pair_normalizer_statistics(models, normalizer)
    trajectory = torch.randn(40, 16, 16, 3)
    dataset = base.SimWindowDataset([trajectory], [base.SimWindow(0, 0)])
    history, _target = dataset[0]
    contract = _component_contract(models[ARMS[0]], history.unsqueeze(0).to(device))
    attention_contract = attention_execution_contract(models, device, formal=False)
    schedule = np.zeros((3, 1), dtype=np.int64)
    endpoints = {arm: train_arm(arm, models[arm], dataset, schedule,
                                normalizer, device, len(schedule))
                 for arm in ARMS}
    return {"schema": SCHEMA, "smoke": "PASS", "identity": identity,
            "normalizer": buffers, "forward_components": contract,
            "attention_execution_contract": attention_contract,
            "endpoints": endpoints}


def run(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError("formal M15 requires --execute and exact acknowledgement")
    if (args.held_aoa != HELD_AOA or args.updates != UPDATES
            or args.batch_size != BATCH_SIZE):
        raise ValueError("M15 contract is held10, updates2000, batch8")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal M15 requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    git = base.git_report()
    if not git["clean"]:
        raise RuntimeError(f"formal M15 requires clean Git: {git['porcelain']}")
    resources = resource_preflight(device)
    inventory = base.inventory(args.sim_dir)  # metadata only
    fit_rows = [row for row in inventory if int(row["aoa"]) in FIT_AOAS]
    held_rows = [row for row in inventory if int(row["aoa"]) == HELD_AOA]
    if len(fit_rows) != 80 or len(held_rows) != 20:
        raise RuntimeError("M15 whole-AoA split count changed")
    # Held field arrays remain unopened through fit construction/training.
    fit_trajectories, fit_windows, _ = base.load_fields(args.sim_dir, fit_rows)
    statistics_values = base.window_statistics(fit_trajectories, fit_windows)
    normalizer = base.ChannelNormalizer(*statistics_values, device=device)
    fit_dataset = base.SimWindowDataset(fit_trajectories, fit_windows)
    schedule = np.random.default_rng(SCHEDULE_SEED).integers(
        0, len(fit_dataset), size=(UPDATES, BATCH_SIZE), dtype=np.int64)
    models, identity = construct_pair(device)
    normalizer_record = set_pair_normalizer_statistics(models, normalizer)
    history, _ = fit_dataset[0]
    components = _component_contract(models[ARMS[0]], history.unsqueeze(0).to(device))
    attention_contract = attention_execution_contract(models, device, formal=True)
    endpoints = {arm: train_arm(arm, models[arm], fit_dataset, schedule,
                                normalizer, device, UPDATES)
                 for arm in ARMS}
    require_endpoint_seals(endpoints)
    del fit_dataset, fit_trajectories
    gc.collect(); torch.cuda.empty_cache()
    # This is the first held u/v/p value materialization in the run.
    held_trajectories, held_windows, held_window_files = base.load_fields(
        args.sim_dir, held_rows)
    evaluation = evaluate_pair(models, held_trajectories, held_windows,
                               held_window_files, normalizer, device)
    gate = decide_gate(evaluation, identity, attention_contract)
    result = {
        "schema": SCHEMA, "evidence_class": "selection_reused_whole_AoA10_Sim",
        "git": git, "held_aoa": HELD_AOA, "fit_aoas": list(FIT_AOAS),
        "training": {"updates": UPDATES, "batch_size": BATCH_SIZE,
                     "model_seed": SEED, "schedule_seed": SCHEDULE_SEED,
                     "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
                     "scheduler": "CosineAnnealingLR_Tmax_2000", "grad_clip": GRAD_CLIP,
                     "point_loss": "accuracy_1_4_1_plus_.25_mse",
                     "augmentation": "none", "teacher": "none", "uq": "none"},
        "fit_inventory_sha256": base.canonical_sha(fit_rows),
        "held_inventory_sha256": base.canonical_sha(held_rows),
        "sample_stream_sha256_both_arms": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "initial_identity": identity, "normalizer": normalizer_record,
        "forward_components": components, "attention_execution_contract": attention_contract,
        "resources": resources,
        "endpoints": endpoints,
        "held_field_values_loaded_before_both_endpoint_seals": False,
        "evaluation": evaluation, "gate": gate,
        "decision": "GO_M15_HATO_CONFIRMATORY_SIM" if gate["pass"] else "STOP_M15_HATO",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "result.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "result": str(path),
                      "delta": evaluation["delta_candidate_minus_control"]}, sort_keys=True),
          flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "smoke", "formal"), default="contract")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--held-aoa", type=int, default=HELD_AOA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", "--acknowledge-protocol",
                        dest="acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        print(json.dumps({"schema": SCHEMA, "acknowledgement": ACKNOWLEDGEMENT,
                          "arms": ARM_MODE, "gate": GATE}, sort_keys=True))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if args.mode == "smoke":
        print(json.dumps(smoke(device), sort_keys=True))
    else:
        run(args, device)


if __name__ == "__main__":
    main()
