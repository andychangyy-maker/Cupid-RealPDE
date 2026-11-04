#!/usr/bin/env python3
"""Strict paired Sim2k screen for the explicit-time ST-B210 backbone.

The two arms have identical tensors, normalizer and fit-only sample stream.
They differ only in the matched local axis mixer: causal observed time versus
framewise spatial-x.  The runner intentionally has no flattened-history
bypass, checkpoint selection, teacher, augmentation, UQ, or held-data access
before both update-2000 endpoints are sealed.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
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

SCHEMA = "st_b210_explicit_time_whole_aoa_sim2k_v1"
ACKNOWLEDGEMENT = "RUN_ST_B210_EXPLICIT_TIME_HELD10_SIM2K"
ARMS = ("spatial_axis_control", "temporal_candidate")
ARM_MODE = {
    "spatial_axis_control": "framewise_spatial_control",
    "temporal_candidate": "causal_temporal_candidate",
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
EXPECTED_PARAMETERS = 29_187_608
EXPECTED_STATE_TENSORS = 238
EXPECTED_INITIAL_STATE_SHA256 = (
    "4c256ee8397a42e2826e88dd3c83bbb09e5aa10081f0caff566713220d9cda67")
EXPECTED_ARCHITECTURE_SHA256 = (
    "7344a29050ad0cf45297f2383a7308c0594e346a85c102d07e29578b344a2a99")
EXPECTED_SAMPLE_STREAM_SHA256 = (
    "f0a9ad185eabb0c45c067a2be595378aad58beab5cb2116e5ab4faa65fdca960")
EXPECTED_FIT_INVENTORY_SHA256 = (
    "c8dbbc06b61701d0b259a32c0d6a8451d4b5a0ae7c12966af27e097ccba4aab8")
EXPECTED_HELD_INVENTORY_SHA256 = (
    "21ed38ef4de7007ac2fc835a145406322bde85b5c4afcc88c8a257b8344d2b79")
MAX_TRAIN_PEAK_BYTES = 21 * 1024 ** 3
POINT_WEIGHTS = dict(base.POINT_WEIGHTS)
GATE = {
    "delta_point_min": .10,
    "component_floor": -.05,
    "positive_trajectories_min": 16,
    "worst_trajectory_delta_point_min": -.05,
    "candidate_absolute_point_min": 60.225899,
}
NORMALIZER_BUFFER_NAMES = ("persistence_scale", "persistence_bias", "target_std")


def _model_classes():
    """Keep the runner importable before the new model module is installed."""
    try:
        from models.spatiotemporal_b210 import (  # noqa: PLC0415
            SpatiotemporalB210, SpatiotemporalB210Config)
    except ImportError as error:
        raise RuntimeError(
            "ST-B210 model unavailable; expected models.spatiotemporal_b210") from error
    return SpatiotemporalB210Config, SpatiotemporalB210


def _seed(device: torch.device) -> None:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _architecture(model: nn.Module) -> dict[str, object]:
    value = model.architecture_config()
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return dict(value)


def _canonical_json_sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_config(*, tiny: bool = False):
    config_type, _operator = _model_classes()
    if not tiny:
        return config_type()
    names = set(getattr(config_type, "__dataclass_fields__", {}))
    candidates = {
        "e0_channels": 8, "e1_channels": 16, "e2_channels": 24, "e3_channels": 32,
        "base_channels": 8, "decoder_channels": 16,
        "e0_blocks": 1, "e1_blocks": 1, "e2_blocks": 1, "e3_blocks": 1,
        "blocks": 1, "groups": 4,
    }
    return config_type(**{key: value for key, value in candidates.items() if key in names})


def construct_pair(device: torch.device, *, tiny: bool = False):
    _config, operator = _model_classes()
    models, starts = {}, {}
    for arm in ARMS:
        _seed(device)
        models[arm] = operator(model_config(tiny=tiny), ARM_MODE[arm]).float().to(device)
        starts[arm] = _snapshot(models[arm])
    left, right = (starts[arm] for arm in ARMS)
    bitwise = left.keys() == right.keys() and all(torch.equal(value, right[key])
                                                   for key, value in left.items())
    if not bitwise:
        raise RuntimeError("ST-B210 paired initial states are not bitwise identical")
    counts = {arm: _count(model) for arm, model in models.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"ST-B210 paired parameter counts differ: {counts}")
    if not tiny and set(counts.values()) != {EXPECTED_PARAMETERS}:
        raise RuntimeError(f"ST-B210 parameter identity changed: {counts}")
    architecture = {arm: _architecture(model) for arm, model in models.items()}
    differing = sorted(key for key in architecture[ARMS[0]]
                       if architecture[ARMS[0]].get(key) != architecture[ARMS[1]].get(key))
    if differing != ["interaction_mode"]:
        raise RuntimeError(f"unexpected paired ST-B210 differences: {differing}")
    if any(model.interaction_mode != ARM_MODE[arm] for arm, model in models.items()):
        raise RuntimeError("ST-B210 model interaction_mode does not match arm")
    if any(bool(architecture[arm].get("flattened_history_bypass", False)) for arm in ARMS):
        raise RuntimeError("ST-B210 screen forbids a flattened-history bypass")
    common_architecture = dict(architecture[ARMS[0]])
    common_architecture.pop("interaction_mode")
    architecture_sha = _canonical_json_sha(common_architecture)
    if not tiny and architecture_sha != EXPECTED_ARCHITECTURE_SHA256:
        raise RuntimeError(
            "ST-B210 frozen architecture changed: " + architecture_sha)
    state_sha = base.state_sha(left)
    if (not tiny and (len(left) != EXPECTED_STATE_TENSORS
                      or state_sha != EXPECTED_INITIAL_STATE_SHA256)):
        raise RuntimeError(
            f"ST-B210 frozen initial state changed: tensors={len(left)}, sha={state_sha}")
    return models, {
        "state_sha256": state_sha, "state_tensors": len(left),
        "bitwise_identical": True, "parameter_counts": counts,
        "architecture_sha256_without_interaction_mode": architecture_sha,
        "only_architecture_difference": "interaction_mode",
        "differing_architecture_keys": differing, "architecture_config": architecture,
    }


def set_pair_normalizer_statistics(models: Mapping[str, nn.Module], normalizer):
    values = (normalizer.mean_input, normalizer.mean_target,
              normalizer.std_input, normalizer.std_target)
    for model in models.values():
        model.set_normalizer_statistics(*values)
    missing = [name for name in NORMALIZER_BUFFER_NAMES
               if not all(hasattr(model, name) for model in models.values())]
    if missing:
        raise RuntimeError("ST-B210 normalizer buffers missing: " + ", ".join(missing))
    left, right = (models[arm] for arm in ARMS)
    equal = {name: bool(torch.equal(getattr(left, name), getattr(right, name)))
             for name in NORMALIZER_BUFFER_NAMES}
    if not all(equal.values()):
        raise RuntimeError(f"ST-B210 normalizer buffers differ: {equal}")
    return {"normalizer_buffers_bitwise_equal": equal,
            "normalizer_buffer_sha256": base.state_sha(
                {name: getattr(left, name) for name in NORMALIZER_BUFFER_NAMES})}


def batch_stream(dataset, schedule: np.ndarray):
    for row in schedule:
        samples = [dataset[int(index)] for index in row]
        yield torch.stack([item[0] for item in samples]), torch.stack([item[1] for item in samples])


def _required_parameter_names(model: nn.Module) -> list[str]:
    encoder = ["stem.weight", "down1.weight", "down2.weight", "down3.weight"]
    encoder.extend(
        f"stage{stage}.{block}.{suffix}"
        for stage in range(4)
        for block in range(int(getattr(model.config, f"e{stage}_blocks", 0)))
        for suffix in (
            "axis.depthwise.weight", "axis.projection.weight",
            "channel_mlp.2.weight", "spatial.body.2.weight"))
    temporal = [
        "observed_to_future_logits",
        *(f"observed_time{stage}" for stage in range(4)),
        *(f"horizon{stage}" for stage in range(4)),
    ]
    decoder = [
        "coarse_spatial.0.body.2.weight",
        "up32.weight", "fuse2.0.body.2.weight",
        "up21.weight", "fuse1.0.body.2.weight",
        "up10.weight", "fuse0.0.body.2.weight",
        "output_spatial.0.body.2.weight",
        "mean_spatial.0.body.2.weight",
    ]
    output = [
        "mean_head.weight", "fluctuation_head.weight", "transport_head.weight"]
    return [*encoder, *temporal, *decoder, *output]


def _gradient_audit(model: nn.Module) -> dict[str, object]:
    required = {"axis": False, "future_projection": False, "decoder": False,
                "output": False}
    parameters = dict(model.named_parameters())
    expected = _required_parameter_names(model)
    strict_present = all(name in parameters for name in expected)
    required_names = expected if strict_present else []
    required_live, required_finite = {}, {}
    nonfinite, seen = [], {key: False for key in required}
    total_tensors = nonzero_tensors = total_elements = nonzero_elements = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        total_tensors += 1; total_elements += parameter.numel()
        if parameter.grad is None:
            if name in required_names: required_live[name] = False; required_finite[name] = False
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name); continue
        count = int(torch.count_nonzero(parameter.grad).item())
        if name in required_names:
            required_finite[name] = True; required_live[name] = bool(count)
        if not count:
            continue
        nonzero_tensors += 1; nonzero_elements += count
        if any(token in name for token in ("axis", "mixer", "depthwise")):
            seen["axis"] = True
        if any(token in name for token in ("observed_to_future", "future_projection", "horizon")):
            seen["future_projection"] = True
        if any(token in name for token in ("decoder", "up", "fuse", "spatial")):
            seen["decoder"] = True
        if any(token in name for token in ("head", "output")):
            seen["output"] = True
    for name in required_names:
        required_live.setdefault(name, False); required_finite.setdefault(name, False)
    return {"required_groups_live": seen, "all_required_groups_live": all(seen.values()),
            "nonfinite_gradient_tensors": nonfinite,
            "all_seen_gradients_finite": not nonfinite,
            "strict_required_tensor_contract_present": strict_present,
            "required_tensor_nonzero_finite": required_live,
            "required_tensor_gradient_finite": required_finite,
            "all_required_tensors_nonzero_finite": bool(required_names) and all(required_live.values()),
            "required_tensor_count": len(required_names),
            "trainable_tensor_coverage": nonzero_tensors / max(total_tensors, 1),
            "trainable_element_coverage": nonzero_elements / max(total_elements, 1),
            "nonzero_tensor_count": nonzero_tensors, "trainable_tensor_count": total_tensors}


def train_arm(name: str, model: nn.Module, dataset, schedule: np.ndarray, normalizer,
              device: torch.device, updates: int) -> dict[str, object]:
    _seed(device)
    initial_state = _snapshot(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
    trace, interval = [], []
    seen = {"axis": False, "future_projection": False, "decoder": False, "output": False}
    required_seen: dict[str, bool] = {}
    strict_contract_seen = False
    nonfinite: set[str] = set(); started = time.perf_counter()
    for update, (history, target) in enumerate(batch_stream(dataset, schedule), 1):
        model.train(); history, target = normalizer.preprocess(history.to(device), target.to(device))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        accuracy, details = base.accuracy_objective(prediction, target, normalizer, (1., 4., 1.))
        mse = (prediction - target).square().mean(); loss = accuracy + .25 * mse
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"{name} non-finite loss at {update}")
        loss.backward(); audit = _gradient_audit(model)
        for key, value in audit["required_groups_live"].items(): seen[key] |= bool(value)
        nonfinite.update(audit["nonfinite_gradient_tensors"])
        strict_contract_seen |= bool(audit["strict_required_tensor_contract_present"])
        for key, value in audit["required_tensor_nonzero_finite"].items():
            required_seen[key] = required_seen.get(key, False) or bool(value)
        norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError(f"{name} non-finite gradient at {update}")
        optimizer.step(); scheduler.step(); interval.append(float(loss.detach()))
        if update == 1 or update % 250 == 0 or update == updates:
            row = {"update": update, "loss_mean": float(np.mean(interval)),
                   "mse": float(mse.detach()), "grad_norm_before_clip": float(norm.detach()),
                   **{key: float(value) for key, value in details.items()}}
            trace.append(row); interval.clear()
            print(json.dumps({"event": "st_b210_progress", "arm": name, **row,
                              "wall_seconds": time.perf_counter() - started}, sort_keys=True), flush=True)
    endpoint_state = _snapshot(model)
    endpoint_finite = all(not value.is_floating_point() or bool(torch.isfinite(value).all())
                          for value in endpoint_state.values())
    if not endpoint_finite:
        raise FloatingPointError(f"{name} endpoint state is non-finite")
    required_names = [key for key in _required_parameter_names(model)
                      if key in initial_state and key in endpoint_state]
    required_changed = {
        key: bool(not torch.equal(initial_state[key], endpoint_state[key]))
        for key in required_names}
    model.eval()
    with torch.inference_mode():
        components = (
            model.forward_components(history[:1])
            if hasattr(model, "forward_components")
            else {"prediction": model(history[:1])})
    prediction_finite = bool(torch.isfinite(components["prediction"]).all())
    displacement = components.get("displacement")
    fluctuation = components.get("fluctuation")
    displacement_finite = (
        True if displacement is None else bool(torch.isfinite(displacement).all()))
    displacement_max_abs = (
        0.0 if displacement is None else float(displacement.abs().max().cpu()))
    fluctuation_mean_max_abs = float(
        0.0 if fluctuation is None else
        fluctuation.mean(dim=1).abs().max().cpu())
    component_health = {
        "prediction_finite": prediction_finite,
        "displacement_finite": displacement_finite,
        "displacement_max_abs": displacement_max_abs,
        "displacement_within_declared_bound": (
            displacement_max_abs <= float(model.config.max_displacement) + 1.0e-6),
        "fluctuation_temporal_mean_max_abs": fluctuation_mean_max_abs,
        "fluctuation_zero_temporal_mean": fluctuation_mean_max_abs <= 1.0e-5,
    }
    return {"endpoint_state_sha256": base.state_sha(endpoint_state), "endpoint_state_finite": endpoint_finite, "updates": updates,
            "wall_seconds": time.perf_counter() - started, "trace": trace,
            "required_endpoint_parameters_changed": required_changed,
            "all_required_endpoint_parameters_changed": all(required_changed.values()),
            "component_health": component_health,
            "gradient_audit": {"required_groups_seen_over_training": seen,
                               "all_required_groups_seen_over_training": all(seen.values()),
                               "strict_required_tensor_contract_present": strict_contract_seen,
                               "required_tensors_seen_over_training": required_seen,
                               "all_required_tensors_seen_over_training": bool(required_seen) and all(required_seen.values()),
                               "nonfinite_gradient_tensors_over_training": sorted(nonfinite),
                               "all_seen_gradients_finite": not nonfinite,
                               "last_step": audit}}


def _axis_contract(models: Mapping[str, nn.Module], device: torch.device, *, formal: bool):
    height, width = ((32, 64) if formal else (16, 16))
    records = {}
    for arm, model in models.items():
        before = base.state_sha(_snapshot(model)); model.eval()
        with torch.inference_mode(): prediction = model(torch.zeros(1, 20, height, width, 3, device=device))
        rows = model.axis_execution_contract()
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("ST-B210 must expose non-empty axis_execution_contract")
        records[arm] = {"rows": rows, "prediction_shape": list(prediction.shape),
                        "prediction_finite": bool(torch.isfinite(prediction).all()),
                        "state_unchanged": before == base.state_sha(_snapshot(model))}
    indexed = {arm: {(int(row["stage"]), int(row["block"])): row for row in records[arm]["rows"]}
               for arm in ARMS}
    expected_keys = {
        (stage, block)
        for stage in range(4)
        for block in range(int(getattr(models[ARMS[0]].config, f"e{stage}_blocks", 0)))}
    same_keys = (
        set(indexed[ARMS[0]]) == set(indexed[ARMS[1]]) == expected_keys)
    checks = {"same_and_complete_blocks": same_keys,
              "expected_block_count": len(expected_keys) == (8 if formal else len(expected_keys)),
              "kernel_and_padding": True, "canonical_shapes_equal": True,
              "output_elements_and_macs_equal": True, "axis_semantics": True,
              "prediction_shape_and_finite": True, "state_unchanged": True}
    for key in set(indexed[ARMS[0]]) | set(indexed[ARMS[1]]):
        left, right = indexed[ARMS[0]].get(key), indexed[ARMS[1]].get(key)
        if left is None or right is None:
            for name in ("kernel_and_padding", "canonical_shapes_equal", "output_elements_and_macs_equal", "axis_semantics"):
                checks[name] = False
            continue
        checks["kernel_and_padding"] &= all(int(row["kernel_size"]) == 5 and int(row["left_pad"]) == 4
                                                and int(row["right_pad"]) == 0 for row in (left, right))
        checks["canonical_shapes_equal"] &= list(left["canonical_shape"]) == list(right["canonical_shape"])
        checks["output_elements_and_macs_equal"] &= (int(left["output_elements"]) == int(right["output_elements"])
                                                       and int(left["macs"]) == int(right["macs"]))
        control, candidate = left, right
        checks["axis_semantics"] &= (not bool(control["cross_time"]) and bool(control["control_time_folded"])
                                     and bool(candidate["cross_time"]) and not bool(candidate["control_time_folded"]))
    for record in records.values():
        checks["prediction_shape_and_finite"] &= (record["prediction_shape"] == [1, 20, height, width, 3]
                                                   and record["prediction_finite"])
        checks["state_unchanged"] &= record["state_unchanged"]
    return {"formal": formal, "arms": records, "expected_block_keys": sorted(expected_keys),
            "checks": checks, "pass": all(checks.values())}


def validate_sample_stream(schedule: np.ndarray, dataset_len: int) -> str:
    if schedule.shape != (UPDATES, BATCH_SIZE) or schedule.dtype != np.int64:
        raise RuntimeError("ST-B210 schedule shape/dtype changed")
    if int(schedule.min()) < 0 or int(schedule.max()) >= dataset_len:
        raise RuntimeError("ST-B210 schedule indices out of fit dataset")
    return hashlib.sha256(schedule.tobytes()).hexdigest()


def require_endpoint_seals(endpoints: Mapping[str, Mapping[str, object]]) -> None:
    missing = [arm for arm in ARMS if not endpoints.get(arm, {}).get("endpoint_state_sha256")]
    if missing: raise RuntimeError("held seal guard missing endpoint: " + ", ".join(missing))


def mechanical_health(identity, normalizer, axis, endpoints, resources=None) -> dict[str, object]:
    checks = {
        "paired_initial_state": bool(identity["bitwise_identical"]),
        "paired_parameter_counts": len(set(identity["parameter_counts"].values())) == 1,
        "normalizer_equal": all(normalizer["normalizer_buffers_bitwise_equal"].values()),
        "axis_contract": bool(axis["pass"]),
        "all_required_gradients": all(value["gradient_audit"]["all_required_groups_seen_over_training"]
                                      for value in endpoints.values()),
        "strict_required_tensor_contract": all(value["gradient_audit"]["strict_required_tensor_contract_present"]
                                                 and value["gradient_audit"]["all_required_tensors_seen_over_training"]
                                                 for value in endpoints.values()),
        "finite_gradients": all(not value["gradient_audit"]["nonfinite_gradient_tensors_over_training"]
                                for value in endpoints.values()),
        "finite_endpoint_state": all(bool(value["endpoint_state_finite"]) for value in endpoints.values()),
        "required_endpoint_parameters_changed": all(
            bool(value["all_required_endpoint_parameters_changed"])
            for value in endpoints.values()),
        "endpoint_components_healthy": all(
            bool(value["component_health"][key])
            for value in endpoints.values()
            for key in (
                "prediction_finite", "displacement_finite",
                "displacement_within_declared_bound",
                "fluctuation_zero_temporal_mean")),
    }
    if resources is not None:
        checks["resource_preflight"] = bool(resources["health"]["pass"])
    return {"checks": checks, "pass": all(checks.values())}


def load_held_fields_after_guards(sim_dir: Path, held_rows, endpoints, mechanical, *, loader=None):
    require_endpoint_seals(endpoints)
    if not mechanical["pass"]:
        raise RuntimeError("ST-B210 mechanical health failed before held load")
    return (base.load_fields if loader is None else loader)(sim_dir, held_rows)


def _rename(value: object) -> object:
    if isinstance(value, dict):
        return {("spatial_axis_control" if key == "spatial_window_control" else key): _rename(item)
                for key, item in value.items()}
    if isinstance(value, list): return [_rename(item) for item in value]
    return value


def evaluate_pair(models, trajectories, windows, files, normalizer, device):
    return _rename(shared.evaluate_pair({"spatial_window_control": models[ARMS[0]],
                                         "temporal_candidate": models[ARMS[1]]},
                                        trajectories, windows, files, normalizer, device))


def decide_gate(evaluation, identity, axis, mechanical) -> dict[str, object]:
    delta = evaluation["delta_candidate_minus_control"]
    causal = {
        "initial_state": bool(identity["bitwise_identical"]), "axis_contract": bool(axis["pass"]),
        "mechanical": bool(mechanical["pass"]),
        "delta_point": float(delta["point_contribution_46_10_9"]) >= GATE["delta_point_min"],
        "components": all(float(delta[key]) >= GATE["component_floor"] for key in POINT_WEIGHTS),
        "positive_trajectories": int(evaluation["positive_trajectories"]) >= GATE["positive_trajectories_min"],
        "worst_trajectory": float(evaluation["worst_trajectory_delta_point"]) >= GATE["worst_trajectory_delta_point_min"],
        "complete_panel": int(evaluation["trajectory_count"]) == 20 and int(evaluation["windows_per_trajectory"]) == 48,
    }
    candidate = float(evaluation["aggregate"]["temporal_candidate"]["point_contribution_46_10_9"])
    investment = {"candidate_absolute_point": candidate >= GATE["candidate_absolute_point_min"]}
    return {**GATE, "causal_attribution_checks": causal, "causal_attribution_pass": all(causal.values()),
            "investment_checks": investment, "investment_pass": all(investment.values()),
            "diagnostics": {"weighted_rel_mvpe_delta": (.46 * float(delta["rel_l2_score"])
                                                           + .09 * float(delta["mvpe_rel_l2_score"]))},
            "candidate_absolute_point": candidate, "pass": all(causal.values()) and all(investment.values())}


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def resource_preflight(device: torch.device) -> dict[str, object]:
    models, identity = construct_pair(device)
    try:
        counts = identity["parameter_counts"]
        if device.type != "cuda":
            return {"parameter_counts": counts,
                    "health": {"status": "CPU_ONLY", "pass": False}}
        peaks, probe_losses = {}, {}
        for arm in ARMS:
            _clear_cuda()
            torch.cuda.reset_peak_memory_stats(device)
            model = models[arm].train()
            x, y = (torch.randn(
                BATCH_SIZE, 20, 32, 64, 3, device=device) for _ in range(2))
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            identity_normalizer = base.ChannelNormalizer(
                torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3),
                device)
            accuracy, _details = base.accuracy_objective(
                prediction, y, identity_normalizer, (1.0, 4.0, 1.0))
            loss = accuracy + .25 * (prediction - y).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            torch.cuda.synchronize(device)
            probe_losses[arm] = float(loss.detach())
            peaks[arm] = int(torch.cuda.max_memory_allocated(device))
            model.zero_grad(set_to_none=True)
            del optimizer, x, y, prediction, accuracy, loss, identity_normalizer
        latency = {}
        for arm in ARMS:
            model = models[arm].eval()
            x = torch.randn(BATCH_SIZE, 20, 32, 64, 3, device=device)
            with torch.inference_mode():
                for _ in range(2):
                    model(x)
                torch.cuda.synchronize(device)
                elapsed = []
                for _ in range(5):
                    started = time.perf_counter()
                    model(x)
                    torch.cuda.synchronize(device)
                    elapsed.append(time.perf_counter() - started)
            median = float(np.median(elapsed))
            latency[arm] = {
                "batch": BATCH_SIZE, "median_batch_seconds": median,
                "median_per_sample_seconds": median / BATCH_SIZE,
                "repeats": len(elapsed)}
            del x
        raw64 = {}
        try:
            for arm in ARMS:
                with torch.inference_mode():
                    output = models[arm].eval()(torch.zeros(1, 20, 64, 128, 3, device=device))
                raw64[arm] = {"shape": list(output.shape), "finite": bool(torch.isfinite(output).all())}
        except RuntimeError as error:
            if not _is_oom(error):
                raise
            raw64 = {"status": "OOM_DIAGNOSTIC_ONLY", "message": str(error)}
            _clear_cuda()
        within_limit = all(value <= MAX_TRAIN_PEAK_BYTES for value in peaks.values())
        finite_losses = all(np.isfinite(value) for value in probe_losses.values())
        health = {
            "status": "PASS" if within_limit and finite_losses else "FAIL",
            "pass": bool(within_limit and finite_losses),
            "train_peak_limit_bytes": MAX_TRAIN_PEAK_BYTES,
            "train_peak_within_limit": within_limit,
            "official_probe_loss_finite": finite_losses,
        }
        return {"parameter_counts": counts, "health": health,
                "train_peak_vram": {"batch": 8, "bytes": peaks},
                "official_probe_loss": probe_losses,
                "paired_inference_latency": latency,
                "raw_grid_inference_only": {"batch": 1, "input_shape": [20, 64, 128, 3], "arms": raw64}}
    finally:
        del models; _clear_cuda()


def preflight(device: torch.device) -> dict[str, object]:
    models, identity = construct_pair(device); normalizer = base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), device)
    buffers = set_pair_normalizer_statistics(models, normalizer)
    axis = _axis_contract(models, device, formal=True)
    del models, normalizer
    _clear_cuda()
    resources = resource_preflight(device)
    passed = bool(axis["pass"] and resources["health"]["pass"])
    return {"schema": SCHEMA, "preflight": "PASS" if passed else "FAIL",
            "identity": identity, "normalizer": buffers, "axis_execution_contract": axis,
            "resources": resources}


def run(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError("formal ST-B210 requires --execute and exact acknowledgement")
    if args.updates != UPDATES or args.batch_size != BATCH_SIZE:
        raise ValueError("ST-B210 protocol freezes updates2000 and batch8")
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("formal ST-B210 requires CUDA")
    if args.sim_dir is None or args.output_dir is None: raise ValueError("formal requires sim-dir/output-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError("refusing non-empty output")
    git = base.git_report()
    if not git["clean"]: raise RuntimeError(f"formal ST-B210 requires clean Git: {git['porcelain']}")
    inventory = base.inventory(args.sim_dir)
    fit_rows = [row for row in inventory if int(row["aoa"]) in FIT_AOAS]
    held_rows = [row for row in inventory if int(row["aoa"]) == HELD_AOA]
    if len(fit_rows) != 80 or len(held_rows) != 20: raise RuntimeError("whole-AoA split count changed")
    fit_inventory_sha, held_inventory_sha = base.canonical_sha(fit_rows), base.canonical_sha(held_rows)
    if fit_inventory_sha != EXPECTED_FIT_INVENTORY_SHA256:
        raise RuntimeError("ST-B210 fit inventory SHA changed: " + fit_inventory_sha)
    if held_inventory_sha != EXPECTED_HELD_INVENTORY_SHA256:
        raise RuntimeError("ST-B210 held inventory SHA changed: " + held_inventory_sha)
    fit_trajectories, fit_windows, _ = base.load_fields(args.sim_dir, fit_rows)
    if len(fit_windows) != 3840: raise RuntimeError("fit window count changed")
    normalizer = base.ChannelNormalizer(*base.window_statistics(fit_trajectories, fit_windows), device=device)
    dataset = base.SimWindowDataset(fit_trajectories, fit_windows)
    schedule = np.random.default_rng(SCHEDULE_SEED).integers(0, len(dataset), size=(UPDATES, BATCH_SIZE), dtype=np.int64)
    schedule_sha = validate_sample_stream(schedule, len(dataset))
    if schedule_sha != EXPECTED_SAMPLE_STREAM_SHA256:
        raise RuntimeError("ST-B210 sample stream SHA changed: " + schedule_sha)
    resources = resource_preflight(device); models, identity = construct_pair(device)
    buffers = set_pair_normalizer_statistics(models, normalizer); axis = _axis_contract(models, device, formal=True)
    endpoints = {arm: train_arm(arm, models[arm], dataset, schedule, normalizer, device, UPDATES) for arm in ARMS}
    mechanical = mechanical_health(identity, buffers, axis, endpoints, resources)
    del dataset, fit_trajectories; _clear_cuda()
    held_trajectories, held_windows, held_files = load_held_fields_after_guards(
        args.sim_dir, held_rows, endpoints, mechanical)
    evaluation = evaluate_pair(models, held_trajectories, held_windows, held_files, normalizer, device)
    gate = decide_gate(evaluation, identity, axis, mechanical)
    if gate["pass"]:
        decision = "GO_ST_B210_HELD20_CONFIRMATION"
    elif gate["causal_attribution_pass"] and not gate["investment_pass"]:
        decision = "MECHANISM_ONLY_STOP_ST_B210_INVESTMENT"
    else:
        decision = "STOP_ST_B210"
    result = {"schema": SCHEMA, "evidence_class": "selection_reused_whole_AoA10_Sim",
              "protocol": {"acknowledgement": ACKNOWLEDGEMENT}, "git": git,
              "fit_aoas": list(FIT_AOAS), "held_aoa": HELD_AOA,
              "fit_inventory_sha256": fit_inventory_sha, "held_inventory_sha256": held_inventory_sha,
              "training": {"updates": UPDATES, "batch_size": BATCH_SIZE, "model_seed": SEED,
                           "schedule_seed": SCHEDULE_SEED, "optimizer": "AdamW", "lr": LR,
                           "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
                           "point_loss": "accuracy_1_4_1_plus_.25_mse", "teacher": "none", "uq": "none"},
              "sample_stream_sha256_both_arms": schedule_sha, "initial_identity": identity,
              "normalizer": buffers, "axis_execution_contract": axis, "resources": resources,
              "endpoints": endpoints, "mechanical_health": mechanical,
              "held_field_values_loaded_before_both_endpoint_seals": False,
              "evaluation": evaluation, "gate": gate, "decision": decision}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "result": str(result_path),
                      "delta": evaluation["delta_candidate_minus_control"]}, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "preflight", "formal"), default="contract")
    parser.add_argument("--sim-dir", type=Path); parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--held-aoa", type=int, choices=(10,), default=HELD_AOA)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE); parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", "--acknowledge-protocol", dest="acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        print(json.dumps({"schema": SCHEMA, "held_aoa": HELD_AOA, "fit_aoas": FIT_AOAS,
                          "acknowledgement": ACKNOWLEDGEMENT, "arms": ARM_MODE, "gate": GATE}, sort_keys=True)); return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("requested CUDA unavailable")
    result = preflight(device) if args.mode == "preflight" else run(args, device)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
