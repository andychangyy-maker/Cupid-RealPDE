#!/usr/bin/env python3
"""M16 strict paired Sim2k screen for a pyramid causal-3D operator.

The candidate and matched control have bitwise-identical tensors, consume an
identical fit-only sample/dropout stream, and differ only in whether their
matched 3-D mixer acts on observed time or a framewise spatial axis.  The CLI
defaults to the independent held-AoA20 confirmation; the earlier held-AoA10
discovery remains available only through its own split and acknowledgement.
Held field values are not materialised until both endpoint states have been
SHA-sealed.  Both protocols are point-only and write no checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
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
import train_hato_loao_screen as shared  # noqa: E402


SCHEMA = "m16_pyramid_causal3d_whole_aoa_sim2k_v1"
ACKNOWLEDGEMENT = "RUN_M16_PYRAMID_CAUSAL3D_SIM2K"
CONFIRMATION_SCHEMA = "m16_pyramid_causal3d_held20_confirmation_v1"
CONFIRMATION_ACKNOWLEDGEMENT = (
    "RUN_M16_PYRAMID_CAUSAL3D_HELD20_CONFIRMATION")
ARMS = ("spatial_axis_control", "temporal_candidate")
ARM_MODE = {
    "spatial_axis_control": "framewise_spatial_control",
    "temporal_candidate": "causal_temporal_candidate",
}
SEED = 210
SCHEDULE_SEED = 220
SAMPLE_INDEX_SHA256 = (
    "f0a9ad185eabb0c45c067a2be595378aad58beab5cb2116e5ab4faa65fdca960")
UPDATES = 2_000
BATCH_SIZE = 8
LR = 3.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
HELD_AOA = 10
FIT_AOAS = (0, 5, 15, 20)
CONFIRMATION_HELD_AOA = 20
CONFIRMATION_FIT_AOAS = (0, 5, 10, 15)
DEFAULT_FORMAL_HELD_AOA = CONFIRMATION_HELD_AOA
POINT_WEIGHTS = {
    "rel_l2_score": .46,
    "tke_rel_l2_score": .10,
    "mvpe_rel_l2_score": .09,
}
GATE = {
    "delta_point_min": .08,
    "component_floor": -.05,
    "positive_trajectories_min": 15,
    "worst_trajectory_delta_point_min": -.05,
    "candidate_absolute_point_min": 59.85,
}


def protocol_for(held_aoa: int) -> dict[str, object]:
    """Bind every split-sensitive value so held10/held20 cannot be mixed."""
    if held_aoa == HELD_AOA:
        return {
            "name": "held10_discovery", "held_aoa": HELD_AOA,
            "fit_aoas": FIT_AOAS, "schema": SCHEMA,
            "acknowledgement": ACKNOWLEDGEMENT,
            "evidence_class": "selection_reused_whole_AoA10_Sim",
            "gate": dict(GATE),
            "fit_inventory_sha256": (
                "c8dbbc06b61701d0b259a32c0d6a8451d4b5a0ae7c12966af27e097ccba4aab8"),
            "held_inventory_sha256": (
                "21ed38ef4de7007ac2fc835a145406322bde85b5c4afcc88c8a257b8344d2b79"),
            "go_decision": "GO_M16_CAUSAL3D_CONFIRMATORY_SIM",
            "stop_decision": "STOP_M16_CAUSAL3D",
        }
    if held_aoa == CONFIRMATION_HELD_AOA:
        return {
            "name": "held20_independent_confirmation",
            "held_aoa": CONFIRMATION_HELD_AOA,
            "fit_aoas": CONFIRMATION_FIT_AOAS,
            "schema": CONFIRMATION_SCHEMA,
            "acknowledgement": CONFIRMATION_ACKNOWLEDGEMENT,
            "evidence_class": "independent_whole_AoA20_confirmation_Sim",
            "gate": {**GATE, "candidate_absolute_point_min": 59.18},
            "fit_inventory_sha256": (
                "9b5763d8a7ea6ebc870c90031733cd4049bc76d6a556da4e3c1cb1a36573cd93"),
            "held_inventory_sha256": (
                "d5a68f5c3c9ff093099b942aff9630f8f178bf7127171ed730c4608bf2907f9b"),
            "threshold_provenance": {
                "reference": "standalone_temporal_transport_held20",
                "reference_point": 59.126205,
                "candidate_absolute_point_min": 59.18,
                "margin": 0.053795,
            },
            "go_decision": "PASS_M16_CAUSAL3D_HELD20_CONFIRMATION",
            "mechanism_only_decision": "MECHANISM_ONLY_STOP_M16_INVESTMENT",
            "stop_decision": "FAIL_M16_CAUSAL3D_HELD20_CONFIRMATION",
        }
    raise ValueError(f"unsupported M16 held AoA {held_aoa}; expected 10 or 20")


NORMALIZER_BUFFER_NAMES = ("persistence_scale", "persistence_bias", "target_std")
REQUIRED_GRADIENT_GROUPS = {
    "axis_mixer": (".axis.", "depthwise", "axis_mixer", "mixer"),
    "bounded_displacement": ("transport_head", "displacement", "warp", "offset"),
    "future_projection": ("observed_to_future_logits", "horizon", "future_projection"),
    "multiscale_decoder": ("coarse_spatial", "up21", "fuse1", "up10",
                           "fuse0", "output_spatial", "multiscale_decoder"),
    "mean_head": ("mean_head",),
    "fluctuation_head": ("fluctuation_head",),
}


def _model_classes():
    try:
        from models.pyramid_causal3d_operator import (  # noqa: PLC0415
            PyramidCausal3DOperatorConfig, PyramidCausal3DOperator)
    except ImportError as error:
        raise RuntimeError(
            "M16 model unavailable; expected models.pyramid_causal3d_operator") from error
    return PyramidCausal3DOperatorConfig, PyramidCausal3DOperator


def model_config(*, tiny: bool = False):
    config_type, _operator = _model_classes()
    if not tiny:
        return config_type()
    names = {field.name for field in fields(config_type)} if is_dataclass(config_type) else set()
    candidates = {
        "e0_channels": 8, "e1_channels": 24, "e2_channels": 32,
        "c0_channels": 8, "c1_channels": 24, "c2_channels": 32,
        "base_channels": 8, "hidden_channels": 24,
    }
    return config_type(**{key: value for key, value in candidates.items() if key in names})


def _seed(device: torch.device) -> None:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def _count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _architecture(model: nn.Module) -> object:
    value = model.architecture_config()
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return value


def construct_pair(device: torch.device, *, tiny: bool = False):
    _config, operator = _model_classes()
    models, starts = {}, {}
    for arm in ARMS:
        _seed(device)
        models[arm] = operator(
            model_config(tiny=tiny), ARM_MODE[arm]).float().to(device)
        starts[arm] = _snapshot(models[arm])
    left, right = starts[ARMS[0]], starts[ARMS[1]]
    equal = left.keys() == right.keys() and all(
        torch.equal(value, right[key]) for key, value in left.items())
    if not equal:
        raise RuntimeError("M16 paired initial states are not bitwise identical")
    counts = {arm: _count(model) for arm, model in models.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"M16 paired parameter counts differ: {counts}")
    return models, {
        "state_sha256": base.state_sha(left), "state_tensors": len(left),
        "bitwise_identical": True, "parameter_counts": counts,
        "only_interaction_difference": "interaction_mode",
        "architecture_config": {arm: _architecture(model)
                                for arm, model in models.items()},
    }


def set_pair_normalizer_statistics(models: Mapping[str, nn.Module], normalizer):
    values = (normalizer.mean_input, normalizer.mean_target,
              normalizer.std_input, normalizer.std_target)
    for model in models.values():
        model.set_normalizer_statistics(*values)
    missing = [name for name in NORMALIZER_BUFFER_NAMES
               if not all(hasattr(model, name) for model in models.values())]
    if missing:
        raise RuntimeError("M16 normalizer buffers missing: " + ", ".join(missing))
    left, right = (models[arm] for arm in ARMS)
    equal = {name: bool(torch.equal(getattr(left, name), getattr(right, name)))
             for name in NORMALIZER_BUFFER_NAMES}
    if not all(equal.values()):
        raise RuntimeError(f"M16 paired normalizer buffers differ: {equal}")
    return {"normalizer_buffers_bitwise_equal": equal,
            "normalizer_buffer_sha256": base.state_sha({
                name: getattr(left, name) for name in NORMALIZER_BUFFER_NAMES})}


def _gradient_audit(model: nn.Module) -> dict[str, object]:
    groups = {group: False for group in REQUIRED_GRADIENT_GROUPS}
    total_tensors = nonzero_tensors = total_elements = nonzero_elements = 0
    missing_grad, nonfinite = [], []
    parameters = dict(model.named_parameters())
    required_names = [
        name
        for stage, blocks in enumerate((model.config.e0_blocks,
                                        model.config.e1_blocks,
                                        model.config.e2_blocks))
        for block in range(blocks)
        for name in (f"stage{stage}.{block}.axis.depthwise.weight",
                     f"stage{stage}.{block}.axis.projection.weight")
    ] + ["observed_to_future_logits", "horizon0", "horizon1", "horizon2"]
    required_live = {}
    required_finite = {}
    for name in required_names:
        parameter = parameters.get(name)
        gradient = None if parameter is None else parameter.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all())
        required_finite[name] = finite
        required_live[name] = finite and bool(torch.count_nonzero(gradient.detach()).item())
    for name, parameter in parameters.items():
        if not parameter.requires_grad:
            continue
        total_tensors += 1
        total_elements += parameter.numel()
        if parameter.grad is None:
            missing_grad.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
            continue
        count = int(torch.count_nonzero(parameter.grad.detach()).item())
        nonzero_elements += count
        if count:
            nonzero_tensors += 1
            for group, markers in REQUIRED_GRADIENT_GROUPS.items():
                groups[group] |= any(marker in name for marker in markers)
    return {
        "all_seen_gradients_finite": not nonfinite,
        "missing_gradient_tensors": missing_grad,
        "nonfinite_gradient_tensors": nonfinite,
        "trainable_tensor_coverage": nonzero_tensors / max(total_tensors, 1),
        "trainable_element_coverage": nonzero_elements / max(total_elements, 1),
        "nonzero_tensor_count": nonzero_tensors,
        "trainable_tensor_count": total_tensors,
        "required_groups_live": groups,
        "all_required_groups_live": all(groups.values()),
        "required_tensor_nonzero_finite": required_live,
        "required_tensor_gradient_finite": required_finite,
        "all_required_tensors_nonzero_finite": all(required_live.values()),
        "required_tensor_count": len(required_names),
    }


def train_arm(name: str, model: nn.Module, dataset, schedule: np.ndarray,
              normalizer, device: torch.device, updates: int):
    _seed(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
    trace, interval = [], []
    seen_groups = {group: False for group in REQUIRED_GRADIENT_GROUPS}
    required_tensor_seen: dict[str, bool] | None = None
    required_tensor_nonfinite: set[str] = set()
    nonfinite_over_training: set[str] = set()
    latest = None
    started = time.perf_counter()
    for update, (history, target) in enumerate(shared.batch_stream(dataset, schedule), 1):
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
        latest = _gradient_audit(model)
        for group, live in latest["required_groups_live"].items():
            seen_groups[group] |= bool(live)
        if required_tensor_seen is None:
            required_tensor_seen = {
                name: False for name in latest["required_tensor_nonzero_finite"]}
        for tensor_name, live in latest["required_tensor_nonzero_finite"].items():
            required_tensor_seen[tensor_name] |= bool(live)
        required_tensor_nonfinite.update(
            name for name, finite in latest["required_tensor_gradient_finite"].items()
            if not finite and dict(model.named_parameters()).get(name) is not None
            and dict(model.named_parameters())[name].grad is not None)
        nonfinite_over_training.update(latest["nonfinite_gradient_tensors"])
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"{name} non-finite gradient at update {update}")
        optimizer.step(); scheduler.step()
        interval.append(float(loss.detach()))
        if update == 1 or update % 250 == 0 or update == updates:
            row = {"update": update, "loss_mean": float(np.mean(interval)),
                   "mse": float(mse.detach()),
                   "grad_norm_before_clip": float(grad_norm.detach()),
                   **{key: float(value) for key, value in details.items()}}
            trace.append(row); interval.clear()
            print(json.dumps({"event": "m16_causal3d_progress", "arm": name,
                              **row, "wall_seconds": time.perf_counter() - started},
                             sort_keys=True), flush=True)
    if latest is None:
        raise RuntimeError("M16 arm executed zero updates")
    assert required_tensor_seen is not None
    latest = {**latest, "required_groups_seen_over_training": seen_groups,
              "all_required_groups_seen_over_training": all(seen_groups.values()),
              "required_tensors_seen_over_training": required_tensor_seen,
              "all_required_tensors_seen_over_training": all(required_tensor_seen.values()),
              "required_tensors_nonfinite_over_training": sorted(required_tensor_nonfinite),
              "nonfinite_gradient_tensors_over_training": sorted(nonfinite_over_training)}
    return {"endpoint_state_sha256": base.state_sha(_snapshot(model)),
            "updates": updates, "wall_seconds": time.perf_counter() - started,
            "point_forward_only": True, "uq_forward_executed": False,
            "gradient_audit": latest, "trace": trace}


def _tensor_shape(value: object) -> list[int] | None:
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        for item in value:
            shape = _tensor_shape(item)
            if shape is not None:
                return shape
    return None


def _component_tensors(value: object):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _component_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _component_tensors(item)


def _displacement_contract(model: nn.Module,
                           components: Mapping[str, object]) -> dict[str, object]:
    keys = [str(key) for key in components
            if any(marker in str(key).lower()
                   for marker in ("displacement", "warp_offset", "flow_offset"))]
    tensors = [tensor for key in keys for tensor in _component_tensors(components[key])]
    finite = bool(tensors) and all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
    max_abs = max((float(tensor.detach().abs().max()) for tensor in tensors), default=float("inf"))
    bound = None
    for owner in (getattr(model, "config", object()), model):
        for name in ("max_displacement", "displacement_bound", "displacement_limit"):
            if hasattr(owner, name):
                bound = float(getattr(owner, name)); break
        if bound is not None: break
    return {"component_keys": keys, "tensors_nonempty": bool(tensors),
            "finite": finite, "max_abs": max_abs, "declared_bound": bound,
            "within_declared_bound": bound is not None and max_abs <= bound + 1.0e-6}


def endpoint_component_health(models: Mapping[str, nn.Module], dataset,
                              normalizer, device: torch.device) -> dict[str, object]:
    history, _target = dataset[0]
    history = (history.unsqueeze(0).to(device) - normalizer.mean_input) / normalizer.std_input
    arms = {}
    for arm, model in models.items():
        before = base.state_sha(_snapshot(model))
        model.eval()
        with torch.inference_mode():
            components = model.forward_components(history)
        displacement = _displacement_contract(model, components)
        fluctuation = components.get("fluctuation")
        fluctuation_zero_mean = (
            torch.is_tensor(fluctuation)
            and float(fluctuation.mean(dim=1).abs().max()) <= 1.0e-5)
        prediction = components.get("prediction")
        state_unchanged = before == base.state_sha(_snapshot(model))
        passed = (displacement["finite"] and displacement["within_declared_bound"]
                  and fluctuation_zero_mean and torch.is_tensor(prediction)
                  and bool(torch.isfinite(prediction).all()) and state_unchanged)
        arms[arm] = {"bounded_displacement": displacement,
                     "fluctuation_zero_time_mean": bool(fluctuation_zero_mean),
                     "prediction_finite": bool(torch.is_tensor(prediction)
                                               and torch.isfinite(prediction).all()),
                     "model_state_unchanged": state_unchanged, "pass": passed}
    return {"arms": arms, "pass": all(value["pass"] for value in arms.values())}


def axis_execution_contract(models: Mapping[str, nn.Module], device: torch.device,
                            *, formal: bool) -> dict[str, object]:
    """Execute and verify all six matched kernel-5 axial blocks."""
    records = {}
    for arm, model in models.items():
        before_sha = base.state_sha(_snapshot(model))
        captured = []
        handles = []
        for name, module in model.named_modules():
            if module is model or not hasattr(module, "interaction_mode"):
                continue
            def hook(_module, inputs, output, *, module_name=name):
                captured.append({"module": module_name,
                                 "declared_mode": str(_module.interaction_mode),
                                 "input_shape": _tensor_shape(inputs),
                                 "output_shape": _tensor_shape(output)})
            handles.append(module.register_forward_hook(hook))
        height, width = ((64, 128) if formal else (16, 16))
        history = torch.zeros(1, 20, height, width, 3, device=device)
        model.eval()
        try:
            with torch.inference_mode():
                components = model.forward_components(history)
        finally:
            for handle in handles:
                handle.remove()
        if not isinstance(components, Mapping) or "prediction" not in components:
            raise RuntimeError("M16 forward_components contract changed")
        after_sha = base.state_sha(_snapshot(model))
        displacement = _displacement_contract(model, components)
        public = model.axis_execution_contract()
        if not isinstance(public, list):
            raise TypeError("model.axis_execution_contract() must return a list")
        records[arm] = {"captured_interaction_blocks": captured,
                        "public_axis_contract": public,
                        "prediction_shape": list(components["prediction"].shape),
                        "prediction_finite": bool(torch.isfinite(components["prediction"]).all()),
                        "model_state_unchanged": before_sha == after_sha,
                        "bounded_displacement": displacement}
    left, right = (records[arm]["captured_interaction_blocks"] for arm in ARMS)
    left_shapes = [(row["input_shape"], row["output_shape"]) for row in left]
    right_shapes = [(row["input_shape"], row["output_shape"]) for row in right]
    nonempty = bool(left) and bool(right)
    same_shapes = left_shapes == right_shapes
    declared = (all(row["declared_mode"] == ARM_MODE[ARMS[0]] for row in left)
                and all(row["declared_mode"] == ARM_MODE[ARMS[1]] for row in right))
    height, width = ((64, 128) if formal else (16, 16))
    output_shape = [1, 20, height, width, 3]
    indexed = {
        arm: {(int(row["stage"]), int(row["block"])): row
              for row in records[arm]["public_axis_contract"]}
        for arm in ARMS}
    config = models[ARMS[0]].config
    expected_keys = {(stage, block)
                     for stage, blocks in enumerate((config.e0_blocks,
                                                     config.e1_blocks,
                                                     config.e2_blocks))
                     for block in range(blocks)}
    six_blocks = len(expected_keys) == 6 and all(set(value) == expected_keys
                                                for value in indexed.values())
    expected_widths = (config.e0_channels, config.e1_channels, config.e2_channels)
    kernel_padding = True
    canonical = True
    packed_shapes = True
    compute_matched = True
    mode_semantics = True
    derived = {}
    for key in sorted(expected_keys):
        stage, _block = key
        expected_shape = [1, 20, int(expected_widths[stage]),
                          height // (2 ** (stage + 1)),
                          width // (2 ** (stage + 1))]
        pair = {arm: indexed[arm].get(key, {}) for arm in ARMS}
        if not all(pair.values()):
            kernel_padding = canonical = packed_shapes = compute_matched = mode_semantics = False
            continue
        kernel_padding &= all(
            int(row["kernel_size"]) == 5 and int(row["left_pad"]) == 4
            and int(row["right_pad"]) == 0 for row in pair.values())
        shapes = {arm: list(row["canonical_shape"]) for arm, row in pair.items()}
        canonical &= (shapes[ARMS[0]] == shapes[ARMS[1]] == expected_shape)
        _batch, steps, channels, stage_height, stage_width = expected_shape
        expected_packed = {
            "temporal_candidate": [_batch, channels, steps,
                                   stage_height, stage_width],
            "spatial_axis_control": [_batch * steps, channels,
                                     stage_width, stage_height, 1],
        }
        actual_packed = {arm: list(row["packed_shape"])
                         for arm, row in pair.items()}
        packed_shapes &= all(actual_packed[arm] == expected_packed[arm]
                             for arm in ARMS)
        output_elements = {arm: int(row["output_elements"]) for arm, row in pair.items()}
        macs = {arm: int(row["macs"]) for arm, row in pair.items()}
        channels = expected_shape[2]
        expected_elements = int(np.prod(expected_shape))
        expected_macs = expected_elements * (5 + 2 * channels)
        compute_matched &= (len(set(output_elements.values())) == 1
                            and len(set(macs.values())) == 1
                            and next(iter(output_elements.values())) == expected_elements
                            and next(iter(macs.values())) == expected_macs)
        candidate = pair["temporal_candidate"]
        control = pair["spatial_axis_control"]
        mode_semantics &= (bool(candidate["cross_time"])
                           and not bool(candidate["control_time_folded"])
                           and not bool(control["cross_time"])
                           and bool(control["control_time_folded"]))
        derived[f"stage{stage}_block{key[1]}"] = {
            "expected_canonical_shape": expected_shape,
            "expected_packed_shape": expected_packed,
            "actual_packed_shape": actual_packed,
            "output_elements": output_elements, "macs": macs}
    passed = (nonempty and same_shapes and declared and six_blocks
              and kernel_padding and canonical and packed_shapes
              and compute_matched and mode_semantics
              and all(record["prediction_shape"] == output_shape
                      and record["prediction_finite"]
                      and record["model_state_unchanged"]
                      and record["bounded_displacement"]["finite"]
                      and record["bounded_displacement"]["within_declared_bound"]
                      for record in records.values()))
    return {"formal": formal, "arms": records,
            "interaction_blocks_nonempty": nonempty,
            "paired_block_io_shapes_equal": same_shapes,
            "mode_declarations_correct": declared,
            "six_blocks_stages_0_1_2": six_blocks,
            "kernel5_left4_right0": kernel_padding,
            "canonical_shapes_equal_and_expected": canonical,
            "packed_shapes_independently_expected": packed_shapes,
            "output_elements_and_macs_equal_and_expected": compute_matched,
            "candidate_control_axis_semantics": mode_semantics,
            "derived_per_block": derived, "pass": passed}


def _single_model(arm: str, device: torch.device) -> nn.Module:
    _config, operator = _model_classes()
    _seed(device)
    return operator(model_config(), ARM_MODE[arm]).float().to(device)


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _latency_arm(arm: str, device: torch.device) -> float:
    model = history = None
    try:
        model = _single_model(arm, device).eval()
        history = torch.zeros(16, 20, 64, 128, 3, device=device)
        with torch.inference_mode():
            for _ in range(3): model(history)
            samples = []
            for _ in range(12):
                torch.cuda.synchronize(device); started = time.perf_counter()
                model(history); torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - started)
        return statistics.median(samples)
    finally:
        del history, model; _clear_cuda()


def _peak_arm(arm: str, device: torch.device) -> int:
    model = optimizer = history = target = loss = None
    try:
        model = _single_model(arm, device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        history = torch.randn(8, 20, 64, 128, 3, device=device)
        target = torch.randn(8, 20, 64, 128, 3, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(history) - target).square().mean()
        loss.backward(); optimizer.step(); torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))
    finally:
        del loss, target, history, optimizer, model; _clear_cuda()


def resource_preflight(device: torch.device) -> dict[str, object]:
    counts = {}
    for arm in ARMS:
        model = None
        try:
            model = _single_model(arm, device); counts[arm] = _count(model)
        finally:
            del model; _clear_cuda()
    if device.type != "cuda":
        return {"parameter_counts": counts, "models_resident_simultaneously": 1,
                "latency": {"measured": False, "reason": "CUDA required"},
                "train_peak_vram": {"measured": False, "reason": "CUDA required"}}
    try:
        latency = {arm: _latency_arm(arm, device) for arm in ARMS}
        peaks = {arm: _peak_arm(arm, device) for arm in ARMS}
    except RuntimeError as error:
        _clear_cuda()
        if not _is_oom(error): raise
        return {"parameter_counts": counts, "models_resident_simultaneously": 1,
                "health": {"status": "FAIL_OOM", "message": str(error)},
                "latency": {"measured": False, "reason": "CUDA OOM"},
                "train_peak_vram": {"measured": False, "reason": "CUDA OOM"}}
    return {"parameter_counts": counts, "models_resident_simultaneously": 1,
            "health": {"status": "PASS"},
            "latency": {"measured": True, "batch": 16,
                        "input_shape_per_sample": [20, 64, 128, 3],
                        "median_seconds": latency},
            "train_peak_vram": {"measured": True, "batch": 8, "bytes": peaks}}


def require_endpoint_seals(endpoints: Mapping[str, Mapping[str, object]]) -> None:
    missing = [arm for arm in ARMS
               if not endpoints.get(arm, {}).get("endpoint_state_sha256")]
    if missing:
        raise RuntimeError("held seal guard missing endpoint: " + ", ".join(missing))


def validate_inventory_seals(protocol: Mapping[str, object], fit_rows,
                             held_rows) -> dict[str, str]:
    """Validate metadata-only inventory before any split field is loaded."""
    computed = {"fit_inventory_sha256": base.canonical_sha(fit_rows),
                "held_inventory_sha256": base.canonical_sha(held_rows)}
    mismatches = [name for name, value in computed.items()
                  if value != protocol[name]]
    if mismatches:
        details = ", ".join(
            f"{name}={computed[name]} expected={protocol[name]}"
            for name in mismatches)
        raise RuntimeError("M16 inventory metadata seal mismatch: " + details)
    return computed


def validate_sample_index_schedule(schedule: np.ndarray) -> str:
    if schedule.shape != (UPDATES, BATCH_SIZE) or schedule.dtype != np.int64:
        raise RuntimeError(
            f"M16 sample schedule contract changed: {schedule.shape}/{schedule.dtype}")
    digest = hashlib.sha256(schedule.tobytes()).hexdigest()
    if digest != SAMPLE_INDEX_SHA256:
        raise RuntimeError(
            f"M16 sample-index SHA mismatch: {digest} expected={SAMPLE_INDEX_SHA256}")
    return digest


def _rename_shared_control(value: object) -> object:
    """Translate the generic M15 evaluator's control label without changing math."""
    if isinstance(value, dict):
        return {("spatial_axis_control" if key == "spatial_window_control" else key):
                _rename_shared_control(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename_shared_control(item) for item in value]
    return value


def evaluate_pair(models: Mapping[str, nn.Module], trajectories, windows,
                  window_files: list[str], normalizer, device: torch.device):
    generic_models = {
        "spatial_window_control": models["spatial_axis_control"],
        "temporal_candidate": models["temporal_candidate"],
    }
    return _rename_shared_control(shared.evaluate_pair(
        generic_models, trajectories, windows, window_files, normalizer, device))


def _evaluate_same_model(model: nn.Module, trajectories, windows,
                         window_files: list[str], normalizer,
                         device: torch.device) -> dict[str, object]:
    # Reuse the frozen evaluator math.  Passing the same eval-mode module into
    # both slots costs an extra inference but cannot alter the metric protocol.
    raw = shared.evaluate_pair(
        {"spatial_window_control": model, "temporal_candidate": model},
        trajectories, windows, window_files, normalizer, device)
    return {
        "aggregate": raw["aggregate"]["temporal_candidate"],
        "per_prefix_horizon": {
            key: value["temporal_candidate"]
            for key, value in raw["per_prefix_horizon"].items()},
        "per_trajectory": {
            key: value["temporal_candidate"]
            for key, value in raw["per_trajectory"].items()},
        "prediction_sha256": raw["prediction_sha256"]["temporal_candidate"],
    }


def endpoint_dependency_diagnostics(
        candidate: nn.Module, native_evaluation: Mapping[str, object],
        trajectories, windows, window_files: list[str], normalizer,
        device: torch.device) -> dict[str, object]:
    """Same-checkpoint dependence probes; never treated as causal GO claims."""
    native = native_evaluation["aggregate"]["temporal_candidate"]
    probes = {
        "axis_bypass": ("set_axis_bypass_override",
                         "set_axis_mixer_bypass_override"),
        "zero_warp": ("set_zero_displacement_override",),
    }
    result = {"interpretation": (
        "same-checkpoint path-dependence diagnostic only; not an independent "
        "causal benefit estimate")}
    for probe, setter_names in probes.items():
        setter_name = next((name for name in setter_names
                            if callable(getattr(candidate, name, None))), None)
        if setter_name is None:
            result[probe] = {"available": False,
                             "reason": "model exposes no safe override"}
            continue
        state_before = base.state_sha(_snapshot(candidate))
        setter = getattr(candidate, setter_name)
        setter(True)
        try:
            override = _evaluate_same_model(
                candidate, trajectories, windows, window_files,
                normalizer, device)
        finally:
            setter(False)
        state_after = base.state_sha(_snapshot(candidate))
        if state_before != state_after:
            raise RuntimeError(f"{probe} diagnostic changed candidate state")
        result[probe] = {
            "available": True, "setter": setter_name,
            "same_checkpoint_state_unchanged": True,
            "override": override,
            "delta_native_minus_override": shared._comparison(
                native, override["aggregate"]),
            "causal_go_claim": False,
        }
    return result


def mechanical_health(identity: Mapping[str, object], normalizer: Mapping[str, object],
                      axis_contract: Mapping[str, object], resources: Mapping[str, object],
                      endpoints: Mapping[str, Mapping[str, object]],
                      endpoint_components: Mapping[str, object]) -> dict[str, object]:
    initial_counts = identity["parameter_counts"]
    resource_counts = resources["parameter_counts"]
    checks = {
        "initial_state_bitwise_equal": bool(identity["bitwise_identical"]),
        "paired_parameter_counts_equal": len(set(initial_counts.values())) == 1,
        "resource_parameter_counts_match": initial_counts == resource_counts,
        "normalizer_buffers_bitwise_equal": all(
            bool(value) for value in normalizer["normalizer_buffers_bitwise_equal"].values()),
        "axis_execution_contract": bool(axis_contract["pass"]),
        "endpoint_component_health": bool(endpoint_components["pass"]),
        "resource_health": resources.get("health", {}).get("status") == "PASS",
        "all_required_gradient_groups_seen": all(
            bool(endpoint["gradient_audit"]["all_required_groups_seen_over_training"])
            for endpoint in endpoints.values()),
        "all_required_gradient_tensors_seen": all(
            bool(endpoint["gradient_audit"]["all_required_tensors_seen_over_training"])
            and not endpoint["gradient_audit"]["required_tensors_nonfinite_over_training"]
            for endpoint in endpoints.values()),
        "all_gradients_finite": all(
            bool(endpoint["gradient_audit"]["all_seen_gradients_finite"])
            and not endpoint["gradient_audit"]["nonfinite_gradient_tensors_over_training"]
            for endpoint in endpoints.values()),
    }
    return {"checks": checks, "pass": all(checks.values())}


def require_mechanical_health(report: Mapping[str, object]) -> None:
    if not bool(report["pass"]):
        failed = [key for key, value in report["checks"].items() if not value]
        raise RuntimeError("M16 mechanical health failed before held load: "
                           + ", ".join(failed))


def load_held_fields_after_guards(
        sim_dir: Path, held_rows, endpoints: Mapping[str, Mapping[str, object]],
        mechanical: Mapping[str, object], *, loader=None):
    """The only formal gateway from sealed fit training to held field values."""
    require_endpoint_seals(endpoints)
    require_mechanical_health(mechanical)
    loader = base.load_fields if loader is None else loader
    return loader(sim_dir, held_rows)


def decide_gate(evaluation: Mapping[str, object], identity: Mapping[str, object],
                axis_contract: Mapping[str, object],
                mechanical: Mapping[str, object],
                gate: Mapping[str, float | int] | None = None) -> dict[str, object]:
    gate = GATE if gate is None else gate
    delta = evaluation["delta_candidate_minus_control"]
    causal_checks = {
        "initial_state_bitwise_equal": bool(identity["bitwise_identical"]),
        "only_interaction_differs": identity["only_interaction_difference"] == "interaction_mode",
        "axis_execution_contract": bool(axis_contract["pass"]),
        "mechanical_health": bool(mechanical["pass"]),
        "delta_point": float(delta["point_contribution_46_10_9"]) >= gate["delta_point_min"],
        "component_floor": all(float(delta[key]) >= gate["component_floor"]
                               for key in POINT_WEIGHTS),
        "positive_trajectories": int(evaluation["positive_trajectories"])
            >= gate["positive_trajectories_min"],
        "worst_trajectory": float(evaluation["worst_trajectory_delta_point"])
            >= gate["worst_trajectory_delta_point_min"],
        "complete_panel": (int(evaluation["trajectory_count"]) == 20
                            and int(evaluation["windows_per_trajectory"]) == 48),
    }
    candidate_point = float(evaluation["aggregate"]["temporal_candidate"][
        "point_contribution_46_10_9"])
    investment_checks = {
        "candidate_absolute_point": candidate_point >= gate["candidate_absolute_point_min"]}
    causal_pass = all(causal_checks.values())
    investment_pass = all(investment_checks.values())
    return {**gate, "causal_attribution_checks": causal_checks,
            "causal_attribution_pass": causal_pass,
            "investment_checks": investment_checks,
            "candidate_absolute_point": candidate_point,
            "investment_pass": investment_pass,
            "pass": causal_pass and investment_pass}


def decision_for(protocol: Mapping[str, object], gate: Mapping[str, object]) -> str:
    """Held20 preserves a mechanism-only outcome instead of collapsing to FAIL."""
    if protocol["name"] == "held20_independent_confirmation":
        if not bool(gate["causal_attribution_pass"]):
            return str(protocol["stop_decision"])
        if not bool(gate["investment_pass"]):
            return str(protocol["mechanism_only_decision"])
        return str(protocol["go_decision"])
    return str(protocol["go_decision"] if gate["pass"]
               else protocol["stop_decision"])


def smoke(device: torch.device,
          protocol: Mapping[str, object] | None = None) -> dict[str, object]:
    protocol = protocol_for(HELD_AOA) if protocol is None else protocol
    models, identity = construct_pair(device, tiny=True)
    normalizer = base.ChannelNormalizer(torch.zeros(3), torch.zeros(3),
                                        torch.ones(3), torch.ones(3), device)
    buffers = set_pair_normalizer_statistics(models, normalizer)
    axis = axis_execution_contract(models, device, formal=False)
    trajectory = torch.randn(40, 16, 16, 3)
    dataset = base.SimWindowDataset([trajectory], [base.SimWindow(0, 0)])
    schedule = np.zeros((3, 1), dtype=np.int64)
    endpoints = {arm: train_arm(arm, models[arm], dataset, schedule,
                                normalizer, device, len(schedule)) for arm in ARMS}
    return {"schema": protocol["schema"], "protocol": protocol["name"],
            "smoke": "PASS", "identity": identity,
            "normalizer": buffers, "axis_execution_contract": axis,
            "endpoints": endpoints}


def run(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    protocol = protocol_for(args.held_aoa)
    if not args.execute or args.acknowledge != protocol["acknowledgement"]:
        raise PermissionError(
            f"formal M16 {protocol['name']} requires --execute and its exact "
            "protocol acknowledgement")
    if args.updates != UPDATES or args.batch_size != BATCH_SIZE:
        raise ValueError("M16 protocols freeze updates2000 and batch8")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal M16 requires CUDA")
    if args.output_dir is None or args.sim_dir is None:
        raise ValueError("formal M16 requires sim-dir and output-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty M16 output directory")
    git = base.git_report()
    if not git["clean"]:
        raise RuntimeError(f"formal M16 requires clean Git: {git['porcelain']}")
    resources = resource_preflight(device)
    inventory = base.inventory(args.sim_dir)
    fit_aoas = tuple(int(value) for value in protocol["fit_aoas"])
    held_aoa = int(protocol["held_aoa"])
    fit_rows = [row for row in inventory if int(row["aoa"]) in fit_aoas]
    held_rows = [row for row in inventory if int(row["aoa"]) == held_aoa]
    if len(fit_rows) != 80 or len(held_rows) != 20:
        raise RuntimeError("M16 whole-AoA split count changed")
    inventory_seals = validate_inventory_seals(protocol, fit_rows, held_rows)
    fit_trajectories, fit_windows, _ = base.load_fields(args.sim_dir, fit_rows)
    statistics_values = base.window_statistics(fit_trajectories, fit_windows)
    normalizer = base.ChannelNormalizer(*statistics_values, device=device)
    fit_dataset = base.SimWindowDataset(fit_trajectories, fit_windows)
    schedule = np.random.default_rng(SCHEDULE_SEED).integers(
        0, len(fit_dataset), size=(UPDATES, BATCH_SIZE), dtype=np.int64)
    schedule_sha256 = validate_sample_index_schedule(schedule)
    models, identity = construct_pair(device)
    buffers = set_pair_normalizer_statistics(models, normalizer)
    axis = axis_execution_contract(models, device, formal=True)
    endpoints = {arm: train_arm(arm, models[arm], fit_dataset, schedule,
                                normalizer, device, UPDATES) for arm in ARMS}
    require_endpoint_seals(endpoints)
    endpoint_components = endpoint_component_health(
        models, fit_dataset, normalizer, device)
    mechanical = mechanical_health(
        identity, buffers, axis, resources, endpoints, endpoint_components)
    del fit_dataset, fit_trajectories
    gc.collect(); torch.cuda.empty_cache()
    held_trajectories, held_windows, held_files = load_held_fields_after_guards(
        args.sim_dir, held_rows, endpoints, mechanical)
    evaluation = evaluate_pair(
        models, held_trajectories, held_windows, held_files, normalizer, device)
    dependency_diagnostics = endpoint_dependency_diagnostics(
        models["temporal_candidate"], evaluation, held_trajectories,
        held_windows, held_files, normalizer, device)
    gate = decide_gate(evaluation, identity, axis, mechanical, protocol["gate"])
    result = {
        "schema": protocol["schema"], "evidence_class": protocol["evidence_class"],
        "protocol": {"name": protocol["name"],
                     "acknowledgement": protocol["acknowledgement"]},
        "git": git, "held_aoa": held_aoa, "fit_aoas": list(fit_aoas),
        "training": {"updates": UPDATES, "batch_size": BATCH_SIZE,
                     "model_seed": SEED, "schedule_seed": SCHEDULE_SEED,
                     "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
                     "scheduler": "CosineAnnealingLR_Tmax_2000", "grad_clip": GRAD_CLIP,
                     "point_loss": "accuracy_1_4_1_plus_.25_mse",
                     "augmentation": "none", "teacher": "none", "uq": "none"},
        **inventory_seals,
        "sample_stream_sha256_both_arms": schedule_sha256,
        "initial_identity": identity, "normalizer": buffers,
        "axis_execution_contract": axis, "resources": resources,
        "endpoints": endpoints, "endpoint_component_health": endpoint_components,
        "mechanical_health": mechanical,
        "held_field_values_loaded_before_both_endpoint_seals": False,
        "evaluation": evaluation,
        "candidate_endpoint_dependency_diagnostics": dependency_diagnostics,
        "historical_absolute_point_references": {
            "standalone_temporal_transport_candidate": 59.792207,
            "hato_spatial_control": 59.217947,
            "same_protocol_b210": 60.192570,
            "reference_split": "held_AoA10",
            "directly_comparable_to_current_split": held_aoa == 10,
            "interpretation": (
                "capacity references only; held10 values are not direct absolute "
                "comparators for held20 confirmation"),
        },
        "threshold_provenance": protocol.get("threshold_provenance"),
        "gate": gate,
        "decision": decision_for(protocol, gate),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "result": str(result_path),
                      "delta": evaluation["delta_candidate_minus_control"]}, sort_keys=True),
          flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "smoke", "formal"), default="contract")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--held-aoa", type=int, choices=(10, 20),
                        default=DEFAULT_FORMAL_HELD_AOA)
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
        protocol = protocol_for(args.held_aoa)
        print(json.dumps({"schema": protocol["schema"],
                          "protocol": protocol["name"],
                          "held_aoa": protocol["held_aoa"],
                          "fit_aoas": protocol["fit_aoas"],
                          "acknowledgement": protocol["acknowledgement"],
                          "arms": ARM_MODE, "gate": protocol["gate"]}, sort_keys=True))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    result = (smoke(device, protocol_for(args.held_aoa))
              if args.mode == "smoke" else run(args, device))
    if args.mode == "smoke":
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
