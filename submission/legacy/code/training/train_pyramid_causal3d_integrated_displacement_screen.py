#!/usr/bin/env python3
"""Strict paired screen for absolute versus integrated PC3D displacement.

Both arms use the same causal PC3D, initial tensors, sample stream and point
loss.  The only architecture value that differs is whether the raw transport
head is interpreted independently at every horizon or cumulatively along the
forecast horizon.  Held field values remain inaccessible until both trained
endpoints and all mechanical checks have been sealed.
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
import train_pyramid_causal3d_loao_screen as pc3d_screen  # noqa: E402


SCHEMA = "pc3d_integrated_displacement_held10_sim2k_v1"
ACKNOWLEDGEMENT = "RUN_PC3D_INTEGRATED_DISPLACEMENT_HELD10"
CONFIRMATION_SCHEMA = "pc3d_integrated_displacement_held20_sim2k_v1"
CONFIRMATION_ACKNOWLEDGEMENT = (
    "RUN_PC3D_INTEGRATED_DISPLACEMENT_HELD20")
ARMS = ("absolute_control", "integrated_candidate")
ARM_SEMANTICS = {
    "absolute_control": "absolute",
    "integrated_candidate": "horizon_integrated",
}
SEED = 210
SCHEDULE_SEED = 220
SAMPLE_INDEX_SHA256 = pc3d_screen.SAMPLE_INDEX_SHA256
UPDATES = 2_000
BATCH_SIZE = 8
LR = 3.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
EXPECTED_PARAMETERS = 8_929_552
EXPECTED_STATE_TENSORS = 184
HELD_AOA = 10
FIT_AOAS = (0, 5, 15, 20)
CONFIRMATION_HELD_AOA = 20
CONFIRMATION_FIT_AOAS = (0, 5, 10, 15)
POINT_WEIGHTS = dict(pc3d_screen.POINT_WEIGHTS)
NORMALIZER_BUFFER_NAMES = pc3d_screen.NORMALIZER_BUFFER_NAMES
GATE = {
    "delta_point_min": .08,
    "component_floor": -.05,
    "positive_trajectories_min": 15,
    "worst_trajectory_delta_point_min": -.05,
    "candidate_absolute_point_min": 59.85,
    "absolute_control_reference": 59.932626120502356,
    "absolute_control_fidelity_tolerance": .02,
}


def protocol_for(held_aoa: int) -> dict[str, object]:
    if held_aoa == HELD_AOA:
        return {
            "name": "held10_discovery",
            "held_aoa": HELD_AOA,
            "fit_aoas": FIT_AOAS,
            "schema": SCHEMA,
            "acknowledgement": ACKNOWLEDGEMENT,
            "evidence_class": "selection_reused_whole_AoA10_Sim",
            "fit_inventory_sha256": (
                "c8dbbc06b61701d0b259a32c0d6a8451d4b5a0ae7c12966af27e097ccba4aab8"),
            "held_inventory_sha256": (
                "21ed38ef4de7007ac2fc835a145406322bde85b5c4afcc88c8a257b8344d2b79"),
            "gate": dict(GATE),
            "go_decision": "GO_INTEGRATED_DISPLACEMENT_HELD20",
            "stop_decision": "STOP_INTEGRATED_DISPLACEMENT",
            "inconclusive_decision": "INCONCLUSIVE_ABSOLUTE_CONTROL_FIDELITY",
        }
    if held_aoa == CONFIRMATION_HELD_AOA:
        return {
            "name": "held20_independent_confirmation",
            "held_aoa": CONFIRMATION_HELD_AOA,
            "fit_aoas": CONFIRMATION_FIT_AOAS,
            "schema": CONFIRMATION_SCHEMA,
            "acknowledgement": CONFIRMATION_ACKNOWLEDGEMENT,
            "evidence_class": "independent_whole_AoA20_confirmation_Sim",
            "fit_inventory_sha256": (
                "9b5763d8a7ea6ebc870c90031733cd4049bc76d6a556da4e3c1cb1a36573cd93"),
            "held_inventory_sha256": (
                "d5a68f5c3c9ff093099b942aff9630f8f178bf7127171ed730c4608bf2907f9b"),
            "gate": {
                **GATE,
                "candidate_absolute_point_min": 59.18,
                "absolute_control_reference": 59.27145154391036,
            },
            "go_decision": "PASS_INTEGRATED_DISPLACEMENT_HELD20",
            "stop_decision": "FAIL_INTEGRATED_DISPLACEMENT_HELD20",
            "inconclusive_decision": "INCONCLUSIVE_ABSOLUTE_CONTROL_FIDELITY_HELD20",
        }
    raise ValueError(f"unsupported held AoA {held_aoa}; expected 10 or 20")


def _model_classes():
    try:
        from models.pyramid_causal3d_integrated_displacement import (  # noqa: PLC0415
            PyramidCausal3DIntegratedDisplacementOperator)
        from models.pyramid_causal3d_operator import (  # noqa: PLC0415
            PyramidCausal3DOperatorConfig)
    except ImportError as error:
        raise RuntimeError("integrated-displacement PC3D model unavailable") from error
    return PyramidCausal3DOperatorConfig, PyramidCausal3DIntegratedDisplacementOperator


def model_config(*, tiny: bool = False):
    config_type, _operator = _model_classes()
    if not tiny:
        return config_type()
    names = {field.name for field in fields(config_type)} if is_dataclass(config_type) else set()
    candidates = {
        "e0_channels": 8, "e1_channels": 24, "e2_channels": 32,
        "e0_blocks": 1, "e1_blocks": 1, "e2_blocks": 1,
    }
    return config_type(**{key: value for key, value in candidates.items()
                         if key in names})


def _seed(device: torch.device) -> None:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def _count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _architecture(model: nn.Module) -> dict[str, object]:
    value = model.architecture_config()
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return dict(value)


def construct_pair(device: torch.device, *, tiny: bool = False):
    _config, operator = _model_classes()
    models, starts = {}, {}
    for arm in ARMS:
        _seed(device)
        models[arm] = operator(
            model_config(tiny=tiny), ARM_SEMANTICS[arm]).float().to(device)
        starts[arm] = _snapshot(models[arm])
    left, right = starts[ARMS[0]], starts[ARMS[1]]
    bitwise = left.keys() == right.keys() and all(
        torch.equal(value, right[key]) for key, value in left.items())
    if not bitwise:
        raise RuntimeError("paired initial states are not bitwise identical")
    counts = {arm: _count(model) for arm, model in models.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"paired parameter counts differ: {counts}")
    if not tiny and (set(counts.values()) != {EXPECTED_PARAMETERS}
                     or len(left) != EXPECTED_STATE_TENSORS):
        raise RuntimeError(
            "default PC3D size contract changed: "
            f"parameters={counts}, state_tensors={len(left)}")
    architecture = {arm: _architecture(model) for arm, model in models.items()}
    differing = sorted(
        key for key in architecture[ARMS[0]]
        if architecture[ARMS[0]][key] != architecture[ARMS[1]][key])
    if differing != ["displacement_semantics"]:
        raise RuntimeError(f"unexpected paired architecture differences: {differing}")
    if any(model.interaction_mode != "causal_temporal_candidate"
           for model in models.values()):
        raise RuntimeError("both displacement arms must use the same causal mixer")
    return models, {
        "state_sha256": base.state_sha(left),
        "state_tensors": len(left),
        "bitwise_identical": True,
        "parameter_counts": counts,
        "only_architecture_difference": "displacement_semantics",
        "differing_architecture_keys": differing,
        "common_interaction_mode": "causal_temporal_candidate",
        "architecture_config": architecture,
    }


def set_pair_normalizer_statistics(models: Mapping[str, nn.Module], normalizer):
    values = (normalizer.mean_input, normalizer.mean_target,
              normalizer.std_input, normalizer.std_target)
    for model in models.values():
        model.set_normalizer_statistics(*values)
    missing = [name for name in NORMALIZER_BUFFER_NAMES
               if not all(hasattr(model, name) for model in models.values())]
    if missing:
        raise RuntimeError("normalizer buffers missing: " + ", ".join(missing))
    left, right = (models[arm] for arm in ARMS)
    equal = {name: bool(torch.equal(getattr(left, name), getattr(right, name)))
             for name in NORMALIZER_BUFFER_NAMES}
    if not all(equal.values()):
        raise RuntimeError(f"paired normalizer buffers differ: {equal}")
    return {
        "normalizer_buffers_bitwise_equal": equal,
        "normalizer_buffer_sha256": base.state_sha({
            name: getattr(left, name) for name in NORMALIZER_BUFFER_NAMES}),
    }


def displacement_execution_contract(models: Mapping[str, nn.Module],
                                    device: torch.device, *, formal: bool):
    before = {arm: base.state_sha(_snapshot(model))
              for arm, model in models.items()}
    height, width = ((64, 128) if formal else (16, 16))
    history = torch.zeros(1, 20, height, width, 3, device=device)
    records = {}
    with torch.inference_mode():
        for arm, model in models.items():
            model.eval()
            components = model.forward_components(history)
            records[arm] = {
                "prediction_shape": list(components["prediction"].shape),
                "prediction_finite": bool(torch.isfinite(
                    components["prediction"]).all()),
                "bounded_displacement": pc3d_screen._displacement_contract(
                    model, components),
                "axis_contract": model.axis_execution_contract(),
                "components": {key: value.detach().cpu()
                               for key, value in components.items()},
            }
    left_components = records[ARMS[0]].pop("components")
    right_components = records[ARMS[1]].pop("components")
    update0_equal = (
        left_components.keys() == right_components.keys()
        and all(torch.equal(value, right_components[key])
                for key, value in left_components.items()))

    raw = torch.linspace(-.25, .25, 2 * 20 * 8 * 2 * 3,
                         device=device).reshape(2, 20, 8, 2, 3)
    absolute_basis = models["absolute_control"].apply_displacement_basis(raw)
    integrated_basis = models["integrated_candidate"].apply_displacement_basis(raw)
    pure_basis = {
        "absolute_exact_identity": bool(torch.equal(absolute_basis, raw)),
        "integrated_exact_cumsum": bool(torch.equal(
            integrated_basis, torch.cumsum(raw, dim=1))),
        "horizon_one_bitwise_equal": bool(torch.equal(
            absolute_basis[:, 0], integrated_basis[:, 0])),
    }
    impulse = torch.zeros_like(raw)
    impulse[:, 7] = 1.0
    impulse_response = models["integrated_candidate"].apply_displacement_basis(impulse)
    pure_basis["future_impulse_causality"] = bool(
        torch.count_nonzero(impulse_response[:, :7]) == 0
        and torch.equal(impulse_response[:, 7:],
                        torch.ones_like(impulse_response[:, 7:])))

    axis_rows = {arm: records[arm]["axis_contract"] for arm in ARMS}
    expected_blocks = sum((models[ARMS[0]].config.e0_blocks,
                           models[ARMS[0]].config.e1_blocks,
                           models[ARMS[0]].config.e2_blocks))
    causal_axis = all(
        len(rows) == expected_blocks
        and all(bool(row["cross_time"])
                and not bool(row["control_time_folded"])
                for row in rows)
        for rows in axis_rows.values())
    paired_axis = len(axis_rows[ARMS[0]]) == len(axis_rows[ARMS[1]]) and all(
        left["canonical_shape"] == right["canonical_shape"]
        and left["packed_shape"] == right["packed_shape"]
        and left["output_elements"] == right["output_elements"]
        and left["macs"] == right["macs"]
        for left, right in zip(axis_rows[ARMS[0]], axis_rows[ARMS[1]], strict=True))
    state_unchanged = all(
        before[arm] == base.state_sha(_snapshot(models[arm])) for arm in ARMS)
    checks = {
        "update0_complete_components_bitwise_equal": update0_equal,
        "absolute_basis_exact_identity": pure_basis["absolute_exact_identity"],
        "integrated_basis_exact_cumsum": pure_basis["integrated_exact_cumsum"],
        "horizon_one_bitwise_equal": pure_basis["horizon_one_bitwise_equal"],
        "future_impulse_causality": pure_basis["future_impulse_causality"],
        "both_arms_same_causal_axis": causal_axis,
        "paired_axis_shapes_and_compute_equal": paired_axis,
        "predictions_finite_and_expected_shape": all(
            record["prediction_finite"]
            and record["prediction_shape"] == [1, 20, height, width, 3]
            for record in records.values()),
        "bounded_displacements": all(
            record["bounded_displacement"]["finite"]
            and record["bounded_displacement"]["within_declared_bound"]
            for record in records.values()),
        "model_state_unchanged": state_unchanged,
    }
    return {"formal": formal, "checks": checks, "pass": all(checks.values()),
            "pure_basis": pure_basis, "arms": records}


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _single_model(arm: str, device: torch.device) -> nn.Module:
    _config, operator = _model_classes()
    _seed(device)
    return operator(model_config(), ARM_SEMANTICS[arm]).float().to(device)


def _latency_arm(arm: str, device: torch.device) -> float:
    model = history = None
    try:
        model = _single_model(arm, device).eval()
        history = torch.zeros(16, 20, 64, 128, 3, device=device)
        with torch.inference_mode():
            for _ in range(3):
                model(history)
            samples = []
            for _ in range(12):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                model(history)
                torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - started)
        return statistics.median(samples)
    finally:
        del history, model
        _clear_cuda()


def _peak_arm(arm: str, device: torch.device) -> int:
    model = optimizer = history = target = loss = None
    try:
        model = _single_model(arm, device).train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        history = torch.randn(8, 20, 64, 128, 3, device=device)
        target = torch.randn(8, 20, 64, 128, 3, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(history) - target).square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))
    finally:
        del loss, target, history, optimizer, model
        _clear_cuda()


def resource_preflight(device: torch.device) -> dict[str, object]:
    counts = {}
    for arm in ARMS:
        model = None
        try:
            model = _single_model(arm, device)
            counts[arm] = _count(model)
        finally:
            del model
            _clear_cuda()
    if set(counts.values()) != {EXPECTED_PARAMETERS}:
        raise RuntimeError(
            f"default PC3D parameter contract changed before data load: {counts}")
    if device.type != "cuda":
        return {
            "parameter_counts": counts,
            "models_resident_simultaneously": 1,
            "latency": {"measured": False, "reason": "CUDA required"},
            "train_peak_vram": {"measured": False, "reason": "CUDA required"},
        }
    try:
        latency = {arm: _latency_arm(arm, device) for arm in ARMS}
        peaks = {arm: _peak_arm(arm, device) for arm in ARMS}
    except RuntimeError as error:
        _clear_cuda()
        if not (isinstance(error, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(error).lower()):
            raise
        return {
            "parameter_counts": counts,
            "models_resident_simultaneously": 1,
            "health": {"status": "FAIL_OOM", "message": str(error)},
            "latency": {"measured": False, "reason": "CUDA OOM"},
            "train_peak_vram": {"measured": False, "reason": "CUDA OOM"},
        }
    return {
        "parameter_counts": counts,
        "models_resident_simultaneously": 1,
        "health": {"status": "PASS"},
        "latency": {"measured": True, "batch": 16,
                    "input_shape_per_sample": [20, 64, 128, 3],
                    "median_seconds": latency},
        "train_peak_vram": {"measured": True, "batch": 8, "bytes": peaks},
    }


def require_endpoint_seals(endpoints: Mapping[str, Mapping[str, object]]) -> None:
    missing = [arm for arm in ARMS
               if not endpoints.get(arm, {}).get("endpoint_state_sha256")]
    if missing:
        raise RuntimeError("held seal guard missing endpoint: " + ", ".join(missing))


def mechanical_health(identity: Mapping[str, object], normalizer: Mapping[str, object],
                      displacement_contract: Mapping[str, object],
                      resources: Mapping[str, object],
                      endpoints: Mapping[str, Mapping[str, object]],
                      endpoint_components: Mapping[str, object]) -> dict[str, object]:
    checks = {
        "initial_state_bitwise_equal": bool(identity["bitwise_identical"]),
        "only_displacement_semantics_differs": (
            identity["only_architecture_difference"] == "displacement_semantics"
            and identity["differing_architecture_keys"] == ["displacement_semantics"]),
        "same_causal_interaction_mode": (
            identity["common_interaction_mode"] == "causal_temporal_candidate"),
        "paired_parameter_counts_equal": len(set(
            identity["parameter_counts"].values())) == 1,
        "resource_parameter_counts_match": (
            identity["parameter_counts"] == resources["parameter_counts"]),
        "normalizer_buffers_bitwise_equal": all(
            bool(value) for value in
            normalizer["normalizer_buffers_bitwise_equal"].values()),
        "displacement_execution_contract": bool(displacement_contract["pass"]),
        "endpoint_component_health": bool(endpoint_components["pass"]),
        "resource_health": resources.get("health", {}).get("status") == "PASS",
        "all_required_gradient_groups_seen": all(
            bool(endpoint["gradient_audit"][
                "all_required_groups_seen_over_training"])
            for endpoint in endpoints.values()),
        "all_required_gradient_tensors_seen": all(
            bool(endpoint["gradient_audit"][
                "all_required_tensors_seen_over_training"])
            and not endpoint["gradient_audit"][
                "required_tensors_nonfinite_over_training"]
            for endpoint in endpoints.values()),
        "all_gradients_finite": all(
            bool(endpoint["gradient_audit"]["all_seen_gradients_finite"])
            and not endpoint["gradient_audit"][
                "nonfinite_gradient_tensors_over_training"]
            for endpoint in endpoints.values()),
    }
    return {"checks": checks, "pass": all(checks.values())}


def require_mechanical_health(report: Mapping[str, object]) -> None:
    if not bool(report["pass"]):
        failed = [key for key, value in report["checks"].items() if not value]
        raise RuntimeError("mechanical health failed before held load: "
                           + ", ".join(failed))


def load_held_fields_after_guards(
        sim_dir: Path, held_rows, endpoints: Mapping[str, Mapping[str, object]],
        mechanical: Mapping[str, object], *, loader=None):
    require_endpoint_seals(endpoints)
    require_mechanical_health(mechanical)
    loader = base.load_fields if loader is None else loader
    return loader(sim_dir, held_rows)


def _rename_evaluator_arms(value: object) -> object:
    rename = {
        "spatial_window_control": "absolute_control",
        "temporal_candidate": "integrated_candidate",
    }
    if isinstance(value, dict):
        return {rename.get(key, key): _rename_evaluator_arms(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_rename_evaluator_arms(item) for item in value]
    return value


def evaluate_pair(models: Mapping[str, nn.Module], trajectories, windows,
                  window_files: list[str], normalizer, device: torch.device):
    generic = {
        "spatial_window_control": models["absolute_control"],
        "temporal_candidate": models["integrated_candidate"],
    }
    return _rename_evaluator_arms(shared.evaluate_pair(
        generic, trajectories, windows, window_files, normalizer, device))


def decide_gate(evaluation: Mapping[str, object], identity: Mapping[str, object],
                displacement_contract: Mapping[str, object],
                mechanical: Mapping[str, object],
                gate: Mapping[str, float | int]) -> dict[str, object]:
    delta = evaluation["delta_candidate_minus_control"]
    control_point = float(evaluation["aggregate"]["absolute_control"][
        "point_contribution_46_10_9"])
    candidate_point = float(evaluation["aggregate"]["integrated_candidate"][
        "point_contribution_46_10_9"])
    control_error = abs(control_point - float(gate["absolute_control_reference"]))
    control_fidelity = control_error <= float(
        gate["absolute_control_fidelity_tolerance"])
    science_checks = {
        "initial_state_bitwise_equal": bool(identity["bitwise_identical"]),
        "only_displacement_semantics_differs": (
            identity["only_architecture_difference"] == "displacement_semantics"),
        "displacement_execution_contract": bool(displacement_contract["pass"]),
        "mechanical_health": bool(mechanical["pass"]),
        "delta_point": float(delta["point_contribution_46_10_9"])
            >= float(gate["delta_point_min"]),
        "component_floor": all(float(delta[key]) >= float(gate["component_floor"])
                               for key in POINT_WEIGHTS),
        "positive_trajectories": int(evaluation["positive_trajectories"])
            >= int(gate["positive_trajectories_min"]),
        "worst_trajectory": float(evaluation["worst_trajectory_delta_point"])
            >= float(gate["worst_trajectory_delta_point_min"]),
        "complete_panel": (int(evaluation["trajectory_count"]) == 20
                            and int(evaluation["windows_per_trajectory"]) == 48),
        "candidate_absolute_point": candidate_point
            >= float(gate["candidate_absolute_point_min"]),
    }
    return {
        **gate,
        "absolute_control_point": control_point,
        "absolute_control_fidelity_error": control_error,
        "absolute_control_fidelity_pass": control_fidelity,
        "candidate_absolute_point": candidate_point,
        "science_checks": science_checks,
        "science_pass": all(science_checks.values()),
        "pass": control_fidelity and all(science_checks.values()),
    }


def decision_for(protocol: Mapping[str, object], gate: Mapping[str, object]) -> str:
    if not bool(gate["absolute_control_fidelity_pass"]):
        return str(protocol["inconclusive_decision"])
    return str(protocol["go_decision"] if gate["science_pass"]
               else protocol["stop_decision"])


def validate_sample_index_schedule(schedule: np.ndarray) -> str:
    return pc3d_screen.validate_sample_index_schedule(schedule)


def validate_discovery_prerequisite(
        path: Path | None, git: Mapping[str, object]) -> dict[str, object]:
    """Fail closed before held20 unless a same-code held10 result declared GO."""
    if path is None or not path.is_file():
        raise RuntimeError(
            "held20 requires --discovery-result pointing to a completed held10 GO")
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "schema": payload.get("schema") == SCHEMA,
        "held_aoa": payload.get("held_aoa") == HELD_AOA,
        "decision": payload.get("decision") == "GO_INTEGRATED_DISPLACEMENT_HELD20",
        "gate_pass": payload.get("gate", {}).get("pass") is True,
        "same_git_commit": payload.get("git", {}).get("commit") == git.get("commit"),
        "same_git_branch": payload.get("git", {}).get("branch") == git.get("branch"),
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError("held10 discovery prerequisite failed: " + ", ".join(failed))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checks": checks,
    }


def smoke(device: torch.device) -> dict[str, object]:
    models, identity = construct_pair(device, tiny=True)
    normalizer = base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), device)
    buffers = set_pair_normalizer_statistics(models, normalizer)
    displacement = displacement_execution_contract(models, device, formal=False)
    trajectory = torch.randn(40, 16, 16, 3)
    dataset = base.SimWindowDataset([trajectory], [base.SimWindow(0, 0)])
    schedule = np.zeros((3, 1), dtype=np.int64)
    endpoints = {arm: pc3d_screen.train_arm(
        arm, models[arm], dataset, schedule, normalizer, device, len(schedule))
        for arm in ARMS}
    return {"schema": SCHEMA, "smoke": "PASS", "identity": identity,
            "normalizer": buffers, "displacement_execution_contract": displacement,
            "endpoints": endpoints}


def run(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    protocol = protocol_for(args.held_aoa)
    if not args.execute or args.acknowledge != protocol["acknowledgement"]:
        raise PermissionError(
            f"formal {protocol['name']} requires --execute and its exact "
            "protocol acknowledgement")
    if args.updates != UPDATES or args.batch_size != BATCH_SIZE:
        raise ValueError("integrated-displacement protocols freeze updates2000 and batch8")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal integrated-displacement screen requires CUDA")
    if args.output_dir is None or args.sim_dir is None:
        raise ValueError("formal screen requires sim-dir and output-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    git = base.git_report()
    if not git["clean"]:
        raise RuntimeError(f"formal screen requires clean Git: {git['porcelain']}")

    prerequisite = None
    if int(protocol["held_aoa"]) == CONFIRMATION_HELD_AOA:
        prerequisite = validate_discovery_prerequisite(
            args.discovery_result, git)

    resources = resource_preflight(device)
    if resources.get("health", {}).get("status") != "PASS":
        raise RuntimeError(
            "resource preflight failed before data load: "
            f"{resources.get('health')}")
    inventory = base.inventory(args.sim_dir)
    fit_aoas = tuple(int(value) for value in protocol["fit_aoas"])
    held_aoa = int(protocol["held_aoa"])
    fit_rows = [row for row in inventory if int(row["aoa"]) in fit_aoas]
    held_rows = [row for row in inventory if int(row["aoa"]) == held_aoa]
    if len(fit_rows) != 80 or len(held_rows) != 20:
        raise RuntimeError("whole-AoA split count changed")
    inventory_seals = pc3d_screen.validate_inventory_seals(
        protocol, fit_rows, held_rows)
    fit_trajectories, fit_windows, _ = base.load_fields(args.sim_dir, fit_rows)
    normalizer = base.ChannelNormalizer(
        *base.window_statistics(fit_trajectories, fit_windows), device=device)
    fit_dataset = base.SimWindowDataset(fit_trajectories, fit_windows)
    schedule = np.random.default_rng(SCHEDULE_SEED).integers(
        0, len(fit_dataset), size=(UPDATES, BATCH_SIZE), dtype=np.int64)
    schedule_sha256 = validate_sample_index_schedule(schedule)

    models, identity = construct_pair(device)
    buffers = set_pair_normalizer_statistics(models, normalizer)
    displacement = displacement_execution_contract(models, device, formal=True)
    endpoints = {arm: pc3d_screen.train_arm(
        arm, models[arm], fit_dataset, schedule, normalizer, device, UPDATES)
        for arm in ARMS}
    require_endpoint_seals(endpoints)
    endpoint_components = pc3d_screen.endpoint_component_health(
        models, fit_dataset, normalizer, device)
    mechanical = mechanical_health(
        identity, buffers, displacement, resources, endpoints,
        endpoint_components)

    del fit_dataset, fit_trajectories
    gc.collect()
    torch.cuda.empty_cache()
    held_trajectories, held_windows, held_files = load_held_fields_after_guards(
        args.sim_dir, held_rows, endpoints, mechanical)
    evaluation = evaluate_pair(
        models, held_trajectories, held_windows, held_files, normalizer, device)
    gate = decide_gate(evaluation, identity, displacement, mechanical,
                       protocol["gate"])
    result = {
        "schema": protocol["schema"],
        "evidence_class": protocol["evidence_class"],
        "protocol": {"name": protocol["name"],
                     "acknowledgement": protocol["acknowledgement"]},
        "discovery_prerequisite": prerequisite,
        "git": git,
        "held_aoa": held_aoa,
        "fit_aoas": list(fit_aoas),
        "training": {
            "updates": UPDATES, "batch_size": BATCH_SIZE,
            "model_seed": SEED, "schedule_seed": SCHEDULE_SEED,
            "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR_Tmax_2000", "grad_clip": GRAD_CLIP,
            "point_loss": "accuracy_1_4_1_plus_.25_mse",
            "augmentation": "none", "teacher": "none", "uq": "none",
            "common_interaction_mode": "causal_temporal_candidate",
            "arm_displacement_semantics": dict(ARM_SEMANTICS),
        },
        **inventory_seals,
        "sample_stream_sha256_both_arms": schedule_sha256,
        "initial_identity": identity,
        "normalizer": buffers,
        "displacement_execution_contract": displacement,
        "resources": resources,
        "endpoints": endpoints,
        "endpoint_component_health": endpoint_components,
        "mechanical_health": mechanical,
        "held_field_values_loaded_before_both_endpoint_seals": False,
        "evaluation": evaluation,
        "gate": gate,
        "decision": decision_for(protocol, gate),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"decision": result["decision"],
                      "result": str(result_path),
                      "delta": evaluation["delta_candidate_minus_control"]},
                     sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "smoke", "formal"),
                        default="contract")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--discovery-result", type=Path)
    parser.add_argument("--held-aoa", type=int, choices=(10, 20), default=10)
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
        print(json.dumps({
            "schema": protocol["schema"], "protocol": protocol["name"],
            "held_aoa": protocol["held_aoa"],
            "fit_aoas": protocol["fit_aoas"],
            "acknowledgement": protocol["acknowledgement"],
            "arms": ARM_SEMANTICS, "gate": protocol["gate"],
        }, sort_keys=True))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    result = smoke(device) if args.mode == "smoke" else run(args, device)
    if args.mode == "smoke":
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
