#!/usr/bin/env python3
"""Held-AoA10 paired native-resolution residual-refiner PC3D screen.

The causal PC3D base is trained once using its exact historical contract and
then frozen.  Two small refiners start from the same tensors and are trained
on that frozen base output: phase-pack-2 is the matched control and native
full-resolution mixing is the candidate.  This is a held10 selection-reuse
screen.  The effective grid is the existing ``::2`` loader grid.
"""
from __future__ import annotations

import argparse
import gc
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
from models.full_resolution_residual_refiner import (  # noqa: E402
    FullResolutionResidualRefiner, FullResolutionResidualRefinerConfig,
    phase_pack_2, phase_unpack_2)
from models.pyramid_causal3d_operator import (  # noqa: E402
    PyramidCausal3DOperator, PyramidCausal3DOperatorConfig)

SCHEMA = "pc3d_native_refiner_phasepack_held10_sim2k_v1"
ACKNOWLEDGEMENT = "RUN_PC3D_NATIVE_REFINER_PHASEPACK_HELD10"
HELD_AOA, FIT_AOAS = 10, (0, 5, 15, 20)
EVIDENCE_CLASS = "selection_reused_whole_AoA10_Sim"
BASE_SEED, REFINER_SEED, SCHEDULE_SEED = 210, 610, 220
UPDATES, BATCH_SIZE = 2_000, 8
LR, WEIGHT_DECAY, GRAD_CLIP = 3.0e-4, 1.0e-5, 1.0
EXPECTED_BASE_PARAMETERS, EXPECTED_BASE_STATE_TENSORS = 8_929_552, 184
EXPECTED_BASE_INITIAL_SHA256 = (
    "102108a0c53241f6ac9cc2a3aa33075cd713af89786c09e01e4536dfe5a56658")
EXPECTED_REFINER_PARAMETERS, EXPECTED_REFINER_STATE_TENSORS = 8_771, 22
ARMS = ("phase_pack_2_control", "full_resolution_candidate")
ARM_MODE = {"phase_pack_2_control": "phase_pack_2",
            "full_resolution_candidate": "full_resolution"}
GATE = {"delta_point_min": .04, "weighted_rel_mvpe_min": .025,
        "component_floor": -.05, "positive_trajectories_min": 15,
        "worst_trajectory_delta_point_min": -.05,
        "candidate_minus_base_point_min": .04,
        "base_reference": 59.932626120502356, "base_fidelity_tolerance": .02}


def protocol() -> dict[str, object]:
    return {"name": "held10_selection_reuse", "held_aoa": HELD_AOA,
            "fit_aoas": FIT_AOAS, "schema": SCHEMA,
            "acknowledgement": ACKNOWLEDGEMENT, "evidence_class": EVIDENCE_CLASS,
            "fit_inventory_sha256": "c8dbbc06b61701d0b259a32c0d6a8451d4b5a0ae7c12966af27e097ccba4aab8",
            "held_inventory_sha256": "21ed38ef4de7007ac2fc835a145406322bde85b5c4afcc88c8a257b8344d2b79",
            "gate": dict(GATE), "go_decision": "GO_NATIVE_REFINER_HELD20",
            "stop_decision": "STOP_NATIVE_REFINER_PHASEPACK",
            "inconclusive_decision": "INCONCLUSIVE_BASE_FIDELITY"}


def _seed(value: int, device: torch.device) -> None:
    torch.manual_seed(value)
    if device.type == "cuda": torch.cuda.manual_seed_all(value)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def base_config(*, tiny: bool = False) -> PyramidCausal3DOperatorConfig:
    if not tiny: return PyramidCausal3DOperatorConfig()
    return PyramidCausal3DOperatorConfig(e0_channels=8, e1_channels=24, e2_channels=32,
                                         e0_blocks=1, e1_blocks=1, e2_blocks=1)


def construct_base(device: torch.device, *, tiny: bool = False) -> tuple[nn.Module, dict[str, object]]:
    _seed(BASE_SEED, device)
    model = PyramidCausal3DOperator(base_config(tiny=tiny), "causal_temporal_candidate").float().to(device)
    state = _snapshot(model)
    state_sha256 = base.state_sha(state)
    initial_matches = state_sha256 == EXPECTED_BASE_INITIAL_SHA256
    if not tiny and (_count(model) != EXPECTED_BASE_PARAMETERS
                     or len(state) != EXPECTED_BASE_STATE_TENSORS
                     or not initial_matches):
        raise RuntimeError(
            "historical causal PC3D initial-state contract changed: "
            f"parameters={_count(model)}, tensors={len(state)}, "
            f"sha256={state_sha256}")
    return model, {"initial_state_sha256": state_sha256,
                   "expected_initial_state_sha256": (
                       EXPECTED_BASE_INITIAL_SHA256 if not tiny else None),
                   "initial_state_matches_expected": (
                       initial_matches if not tiny else None),
                   "parameter_count": _count(model),
                   "state_tensors": len(state),
                   "interaction_mode": "causal_temporal_candidate"}


def construct_refiner_pair(device: torch.device) -> tuple[dict[str, nn.Module], dict[str, object]]:
    models, starts, configs = {}, {}, {}
    for arm in ARMS:
        _seed(REFINER_SEED, device)
        config = FullResolutionResidualRefinerConfig(execution_mode=ARM_MODE[arm])
        model = FullResolutionResidualRefiner(config).float().to(device)
        models[arm], starts[arm], configs[arm] = model, _snapshot(model), model.architecture_config()
    left, right = starts[ARMS[0]], starts[ARMS[1]]
    equal = left.keys() == right.keys() and all(torch.equal(v, right[k]) for k, v in left.items())
    differing = sorted(k for k in configs[ARMS[0]] if configs[ARMS[0]][k] != configs[ARMS[1]][k])
    counts = {arm: _count(model) for arm, model in models.items()}
    if not equal or differing != ["execution_mode"] or set(counts.values()) != {EXPECTED_REFINER_PARAMETERS} or len(left) != EXPECTED_REFINER_STATE_TENSORS:
        raise RuntimeError("paired refiner identity contract changed")
    return models, {"state_sha256": base.state_sha(left), "state_tensors": len(left),
                    "parameter_counts": counts, "bitwise_identical": True,
                    "only_architecture_difference": "execution_mode",
                    "differing_architecture_keys": differing, "architecture_config": configs}


def set_base_normalizer(base_model: nn.Module, normalizer) -> dict[str, object]:
    base_model.set_normalizer_statistics(normalizer.mean_input, normalizer.mean_target,
                                         normalizer.std_input, normalizer.std_target)
    names = pc3d_screen.NORMALIZER_BUFFER_NAMES
    return {"base_normalizer_sha256": base.state_sha({n: getattr(base_model, n) for n in names}),
            "normalizer_source": "fit_only_existing_effective_grid_loader_::2"}


def freeze_base(base_model: nn.Module) -> dict[str, object]:
    base_model.eval()
    for parameter in base_model.parameters(): parameter.requires_grad_(False)
    frozen = not any(parameter.requires_grad for parameter in base_model.parameters())
    if not frozen: raise RuntimeError("base freeze failed")
    return {"base_eval": not base_model.training, "base_requires_grad_false": frozen,
            "base_state_sha256": base.state_sha(_snapshot(base_model))}


def refiner_execution_contract(models: Mapping[str, nn.Module], device: torch.device, *, formal: bool) -> dict[str, object]:
    # Field files are raw 64x128 and the existing loader materialises ::2,
    # so the formal effective-grid execution is 32x64.
    height, width = ((32, 64) if formal else (16, 16))
    generator = torch.Generator(device=device).manual_seed(611)
    future = torch.randn(
        2, 20, height, width, 3, device=device, generator=generator)
    packed = phase_pack_2(future)
    unpacked = phase_unpack_2(packed, batch_size=2)
    impulse = torch.zeros(1, 20, height, width, 3, device=device)
    phase_values = {}
    for phase_index, (phase_y, phase_x) in enumerate(
            ((0, 0), (0, 1), (1, 0), (1, 1))):
        time_index = 3 + phase_index
        y_index, x_index = 4 + phase_y, 6 + phase_x
        impulse[0, time_index, y_index, x_index, 0] = phase_index + 1
        phase_values[phase_index] = (
            time_index, y_index // 2, x_index // 2, phase_index + 1)
    packed_impulse = phase_pack_2(impulse)
    restored_impulse = phase_unpack_2(packed_impulse, batch_size=1)
    phase_order_exact = all(
        float(packed_impulse[phase, time_index, y_index, x_index, 0]) == value
        for phase, (time_index, y_index, x_index, value)
        in phase_values.items())
    contracts, update0, bounded = {}, {}, {}
    for arm, model in models.items():
        before = base.state_sha(_snapshot(model)); model.eval()
        with torch.inference_mode(): corrected, correction = model(future), model.correction(future)
        contracts[arm] = model.last_execution_contract
        update0[arm] = bool(torch.equal(corrected, future))
        bounded[arm] = bool(torch.isfinite(correction).all() and correction.abs().max() <= model.config.residual_limit + 1e-6)
        if before != base.state_sha(_snapshot(model)): raise RuntimeError("refiner contract mutated model state")
    left, right = (contracts[arm] for arm in ARMS)
    checks = {"random_pack_unpack_exact": bool(torch.equal(future, unpacked)),
              "four_phase_impulse_pack_unpack_exact": bool(torch.equal(
                  impulse, restored_impulse)),
              "four_phase_impulse_order_exact": phase_order_exact,
              "update0_identity_both_modes": all(update0.values()),
              "feature_elements_equal": left["feature_elements"] == right["feature_elements"],
              "convolutional_macs_equal": left["theoretical_macs"] == right["theoretical_macs"],
              "finite_bounded_correction": all(bounded.values()),
              "execution_modes_unique": left["execution_mode"] != right["execution_mode"]}
    return {"formal": formal, "raw_grid_before_existing_loader": [64, 128] if formal else None,
            "effective_grid_from_existing_loader_::2": [height, width],
            "checks": checks, "pass": all(checks.values()),
            "arms": contracts, "update0_identity": update0, "bounded_correction": bounded}


def _refiner_gradient_audit(model: nn.Module) -> dict[str, object]:
    rows = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.requires_grad}
    finite = {name: grad is not None and bool(torch.isfinite(grad).all()) for name, grad in rows.items()}
    live = {name: finite[name] and bool(torch.count_nonzero(grad).item()) for name, grad in rows.items()}
    return {"all_seen_gradients_finite": all(finite.values()), "all_trainable_tensors_live": all(live.values()),
            "live_tensors": live, "finite_tensors": finite,
            "nonzero_tensor_count": sum(live.values()), "trainable_tensor_count": len(rows)}


def train_base(model: nn.Module, dataset, schedule: np.ndarray, normalizer, device: torch.device) -> dict[str, object]:
    return pc3d_screen.train_arm("causal_pc3d_base", model, dataset, schedule, normalizer, device, UPDATES)


def train_refiner(name: str, refiner: nn.Module, base_model: nn.Module, dataset, schedule: np.ndarray,
                  normalizer, device: torch.device) -> dict[str, object]:
    _seed(REFINER_SEED, device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    trace, latest = [], None
    seen = {name: False for name, parameter in refiner.named_parameters()
            if parameter.requires_grad}
    nonfinite_over_training: set[str] = set()
    base_sha_before = base.state_sha(_snapshot(base_model))
    started = time.perf_counter()
    for update, (history, target) in enumerate(shared.batch_stream(dataset, schedule), 1):
        refiner.train(); history, target = normalizer.preprocess(history.to(device), target.to(device))
        optimizer.zero_grad(set_to_none=True)
        # ``inference_mode`` tensors cannot be retained by Conv3d backward.
        # The base remains frozen, while this ordinary no-grad value is a safe
        # leaf input for refiner autograd.
        with torch.no_grad(): base_prediction = base_model(history).detach()
        prediction = refiner(base_prediction)
        accuracy, details = base.accuracy_objective(prediction, target, normalizer, (1., 4., 1.))
        mse = (prediction - target).square().mean(); loss = accuracy + .25 * mse
        if not bool(torch.isfinite(loss)): raise FloatingPointError(f"{name} non-finite loss at {update}")
        loss.backward(); latest = _refiner_gradient_audit(refiner)
        for tensor_name, live in latest["live_tensors"].items(): seen[tensor_name] |= bool(live)
        nonfinite_over_training.update(name for name, finite in latest["finite_tensors"].items() if not finite)
        norm = nn.utils.clip_grad_norm_(refiner.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(norm)): raise FloatingPointError(f"{name} non-finite gradient at {update}")
        optimizer.step(); scheduler.step()
        if update == 1 or update % 250 == 0 or update == UPDATES:
            row = {"update": update, "loss": float(loss.detach()),
                   "mse": float(mse.detach()),
                   "grad_norm_before_clip": float(norm.detach()),
                   **{k: float(v) for k, v in details.items()}}
            trace.append(row)
            print(json.dumps({"event": "pc3d_native_refiner_progress",
                              "arm": name, **row}, sort_keys=True), flush=True)
    base_sha_after = base.state_sha(_snapshot(base_model))
    if base_sha_before != base_sha_after:
        raise RuntimeError(f"{name} changed frozen base state")
    if latest is None or not all(seen.values()) or nonfinite_over_training:
        raise RuntimeError(f"{name} refiner gradients are not live and finite")
    return {"endpoint_state_sha256": base.state_sha(_snapshot(refiner)), "updates": UPDATES,
            "wall_seconds": time.perf_counter() - started, "point_forward_only": True,
            "base_state_sha256_before": base_sha_before,
            "base_state_sha256_after": base_sha_after,
            "base_state_unchanged_during_refiner_training": True,
            "gradient_audit": {**latest, "live_tensors_seen_over_training": seen,
                                "all_trainable_tensors_seen_over_training": all(seen.values()),
                                "nonfinite_tensors_over_training": sorted(nonfinite_over_training)},
            "trace": trace}


class RefinedModel(nn.Module):
    def __init__(self, base_model: nn.Module, refiner: nn.Module):
        super().__init__(); self.base_model, self.refiner = base_model, refiner
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(): prediction = self.base_model(history)
        return self.refiner(prediction)


def _rename(value: object) -> object:
    mapping = {"spatial_window_control": ARMS[0], "temporal_candidate": ARMS[1]}
    if isinstance(value, dict): return {mapping.get(k, k): _rename(v) for k, v in value.items()}
    if isinstance(value, list): return [_rename(v) for v in value]
    return value


def evaluate(base_model: nn.Module, refiners: Mapping[str, nn.Module], trajectories, windows, files, normalizer, device: torch.device):
    refined = {arm: RefinedModel(base_model, refiners[arm]) for arm in ARMS}
    paired = _rename(shared.evaluate_pair({"spatial_window_control": refined[ARMS[0]],
                                           "temporal_candidate": refined[ARMS[1]]}, trajectories, windows, files, normalizer, device))
    raw = shared.evaluate_pair({"spatial_window_control": base_model, "temporal_candidate": base_model}, trajectories, windows, files, normalizer, device)
    base_eval = {"aggregate": raw["aggregate"]["temporal_candidate"],
                 "per_prefix_horizon": {k: v["temporal_candidate"] for k, v in raw["per_prefix_horizon"].items()},
                 "per_trajectory": {k: v["temporal_candidate"] for k, v in raw["per_trajectory"].items()},
                 "prediction_sha256": raw["prediction_sha256"]["temporal_candidate"]}
    return {"base": base_eval, "control_vs_candidate": paired,
            "prediction_sha256": {"base": base_eval["prediction_sha256"], **paired["prediction_sha256"]}}


def decide_gate(evaluation: Mapping[str, object], identity: Mapping[str, object], execution: Mapping[str, object], mechanical: Mapping[str, object]) -> dict[str, object]:
    paired, base_point = evaluation["control_vs_candidate"], float(evaluation["base"]["aggregate"]["point_contribution_46_10_9"])
    delta = paired["delta_candidate_minus_control"]
    candidate_point = float(paired["aggregate"][ARMS[1]]["point_contribution_46_10_9"])
    weighted = .46 * float(delta["rel_l2_score"]) + .09 * float(delta["mvpe_rel_l2_score"])
    fidelity_error = abs(base_point - GATE["base_reference"])
    science = {"paired_initial_state": bool(identity["bitwise_identical"]), "execution_contract": bool(execution["pass"]),
               "mechanical_health": bool(mechanical["pass"]), "candidate_minus_control_point": float(delta["point_contribution_46_10_9"]) >= GATE["delta_point_min"],
               "weighted_rel_mvpe": weighted >= GATE["weighted_rel_mvpe_min"], "components_floor": all(float(delta[k]) >= GATE["component_floor"] for k in ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")),
               "positive_trajectories": int(paired["positive_trajectories"]) >= GATE["positive_trajectories_min"],
               "worst_trajectory": float(paired["worst_trajectory_delta_point"]) >= GATE["worst_trajectory_delta_point_min"],
               "candidate_minus_base_point": candidate_point - base_point >= GATE["candidate_minus_base_point_min"],
               "complete_panel": int(paired["trajectory_count"]) == 20 and int(paired["windows_per_trajectory"]) == 48}
    return {**GATE, "base_point": base_point, "candidate_point": candidate_point, "base_fidelity_error": fidelity_error,
            "base_fidelity_pass": fidelity_error <= GATE["base_fidelity_tolerance"], "weighted_rel_mvpe": weighted,
            "science_checks": science, "science_pass": all(science.values()), "pass": all(science.values()) and fidelity_error <= .02}


def decision_for(gate: Mapping[str, object]) -> str:
    if not bool(gate["base_fidelity_pass"]): return "INCONCLUSIVE_BASE_FIDELITY"
    return "GO_NATIVE_REFINER_HELD20" if bool(gate["science_pass"]) else "STOP_NATIVE_REFINER_PHASEPACK"


def require_endpoint_seals(endpoints: Mapping[str, Mapping[str, object]]) -> None:
    missing = [name for name in ("base", *ARMS) if not endpoints.get(name, {}).get("endpoint_state_sha256")]
    if missing: raise RuntimeError("held seal guard missing endpoint: " + ", ".join(missing))


def require_mechanical_health(report: Mapping[str, object]) -> None:
    if not report.get("pass"):
        raise RuntimeError("mechanical health failed before held load: " + ", ".join(k for k, v in report["checks"].items() if not v))


def load_held_fields_after_guards(sim_dir: Path, rows, endpoints, mechanical, *, loader=None):
    require_endpoint_seals(endpoints); require_mechanical_health(mechanical)
    return (base.load_fields if loader is None else loader)(sim_dir, rows)


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _resource_arm(arm: str, device: torch.device) -> tuple[float, int]:
    """Measure sequential full-base + one-refiner execution at the real effective grid."""
    base_model = refiner = history = target = loss = None
    try:
        base_model, _ = construct_base(device)
        freeze_base(base_model)
        refiner = construct_refiner_pair(device)[0][arm]
        history = torch.zeros(8, 20, 32, 64, 3, device=device)
        target = torch.zeros_like(history)
        optimizer = torch.optim.AdamW(refiner.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(): base_prediction = base_model(history).detach()
        loss = (refiner(base_prediction) - target).square().mean()
        loss.backward(); optimizer.step(); torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
        refiner.eval(); history16 = torch.zeros(16, 20, 32, 64, 3, device=device)
        with torch.inference_mode():
            for _ in range(3): refiner(base_model(history16))
            samples = []
            for _ in range(12):
                torch.cuda.synchronize(device); started = time.perf_counter()
                refiner(base_model(history16)); torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - started)
        del history16, optimizer
        return statistics.median(samples), peak
    finally:
        del loss, target, history, refiner, base_model
        _clear_cuda()


def resource_preflight(device: torch.device) -> dict[str, object]:
    base_model, base_identity = construct_base(device)
    refiners, refiner_identity = construct_refiner_pair(device)
    del base_model, refiners; _clear_cuda()
    result = {"base": base_identity, "refiners": refiner_identity,
              "models_resident_simultaneously": 2,
              "formal_pre_data_resource_check": True,
              "effective_grid_from_existing_loader_::2": [32, 64]}
    if device.type != "cuda":
        return {**result, "health": {"status": "NOT_MEASURED_NO_CUDA"},
                "latency": {"measured": False, "reason": "CUDA required"},
                "peak_vram": {"measured": False, "reason": "CUDA required"}}
    try:
        measured = {arm: _resource_arm(arm, device) for arm in ARMS}
    except RuntimeError as error:
        if not (isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()): raise
        return {**result, "health": {"status": "FAIL_OOM", "message": str(error)},
                "latency": {"measured": False, "reason": "CUDA OOM"},
                "peak_vram": {"measured": False, "reason": "CUDA OOM"}}
    return {**result, "health": {"status": "PASS"},
            "latency": {"measured": True, "batch": 16, "full_base_no_grad_plus_refiner": True,
                        "median_seconds": {arm: item[0] for arm, item in measured.items()}},
            "peak_vram": {"measured": True, "batch": 8, "full_base_no_grad_plus_refiner_backward": True,
                          "bytes": {arm: item[1] for arm, item in measured.items()}}}


def endpoint_health(base_model: nn.Module, refiners: Mapping[str, nn.Module], dataset,
                    normalizer, device: torch.device, frozen: Mapping[str, object]) -> dict[str, object]:
    history, _target = dataset[0]
    history = (history.unsqueeze(0).to(device) - normalizer.mean_input) / normalizer.std_input
    before = base.state_sha(_snapshot(base_model)); base_model.eval()
    with torch.no_grad(): components = base_model.forward_components(history)
    prediction, fluctuation = components["prediction"], components.get("fluctuation")
    base_state_unchanged = before == base.state_sha(_snapshot(base_model)) == frozen["base_state_sha256"]
    fluctuation_zero_mean = (torch.is_tensor(fluctuation)
                             and float(fluctuation.mean(dim=1).abs().max()) <= 1.0e-5)
    arms = {}
    for arm, refiner in refiners.items():
        refiner.eval(); refiner_before = base.state_sha(_snapshot(refiner))
        with torch.no_grad(): correction = refiner.correction(prediction)
        arms[arm] = {"correction_finite": bool(torch.isfinite(correction).all()),
                     "correction_max_abs": float(correction.abs().max()),
                     "correction_bound": float(refiner.config.residual_limit),
                     "correction_bounded": bool(correction.abs().max() <= refiner.config.residual_limit + 1e-6),
                     "refiner_state_unchanged": refiner_before == base.state_sha(_snapshot(refiner))}
    checks = {"base_state_unchanged": base_state_unchanged,
              "base_fluctuation_zero_time_mean": bool(fluctuation_zero_mean),
              "base_prediction_finite": bool(torch.isfinite(prediction).all()),
              "refiner_corrections_finite_bounded": all(row["correction_finite"] and row["correction_bounded"] and row["refiner_state_unchanged"] for row in arms.values())}
    return {"checks": checks, "pass": all(checks.values()), "arms": arms}


def mechanical_health(base_identity, base_frozen, refiner_identity, execution,
                      resources, endpoints, endpoint_checks) -> dict[str, object]:
    checks = {"base_initial_state_matches_expected": bool(
                  base_identity["initial_state_matches_expected"]),
              "base_frozen": bool(base_frozen["base_eval"] and base_frozen["base_requires_grad_false"]),
              "refiner_paired_identity": bool(refiner_identity["bitwise_identical"] and refiner_identity["differing_architecture_keys"] == ["execution_mode"]),
              "refiner_execution_contract": bool(execution["pass"]), "resource_preflight": resources.get("health", {}).get("status") == "PASS",
              "base_and_refiner_endpoint_health": bool(endpoint_checks["pass"]),
              "base_state_unchanged_during_refiner_training": all(endpoints[a]["base_state_unchanged_during_refiner_training"] for a in ARMS),
              "all_refiner_gradients_live_finite": all(endpoints[a]["gradient_audit"]["all_trainable_tensors_seen_over_training"] and not endpoints[a]["gradient_audit"]["nonfinite_tensors_over_training"] for a in ARMS)}
    return {"checks": checks, "pass": all(checks.values())}


def smoke(device: torch.device) -> dict[str, object]:
    base_model, base_identity = construct_base(device, tiny=True); refiners, identity = construct_refiner_pair(device)
    normalizer = base.ChannelNormalizer(torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), device)
    set_base_normalizer(base_model, normalizer); execution = refiner_execution_contract(refiners, device, formal=False)
    frozen = freeze_base(base_model)
    return {"schema": SCHEMA, "smoke": "PASS", "base": base_identity, "refiners": identity,
            "base_frozen": frozen, "refiner_execution_contract": execution}


def run(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT: raise PermissionError("formal held10 requires --execute and exact acknowledgement")
    if args.held_aoa != HELD_AOA or args.updates != UPDATES or args.batch_size != BATCH_SIZE: raise ValueError("contract is held10, updates2000, batch8")
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("formal screen requires CUDA")
    if args.sim_dir is None or args.output_dir is None: raise ValueError("formal screen requires sim-dir and output-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError("refusing to overwrite non-empty output directory")
    git = base.git_report()
    if not git["clean"]: raise RuntimeError(f"formal screen requires clean Git: {git['porcelain']}")
    resources = resource_preflight(device)  # strictly before even inventory metadata
    if resources.get("health", {}).get("status") != "PASS":
        raise RuntimeError(f"resource preflight failed before data load: {resources.get('health')}")
    inventory = base.inventory(args.sim_dir); fit_rows = [r for r in inventory if int(r["aoa"]) in FIT_AOAS]; held_rows = [r for r in inventory if int(r["aoa"]) == HELD_AOA]
    if len(fit_rows) != 80 or len(held_rows) != 20: raise RuntimeError("whole-AoA held10 split count changed")
    seals = pc3d_screen.validate_inventory_seals(protocol(), fit_rows, held_rows)
    fit_traj, fit_windows, _ = base.load_fields(args.sim_dir, fit_rows)
    normalizer = base.ChannelNormalizer(*base.window_statistics(fit_traj, fit_windows), device=device); dataset = base.SimWindowDataset(fit_traj, fit_windows)
    schedule = np.random.default_rng(SCHEDULE_SEED).integers(0, len(dataset), size=(UPDATES, BATCH_SIZE), dtype=np.int64)
    schedule_sha = pc3d_screen.validate_sample_index_schedule(schedule)
    base_model, base_identity = construct_base(device); base_normalizer = set_base_normalizer(base_model, normalizer)
    base_endpoint = train_base(base_model, dataset, schedule, normalizer, device); frozen = freeze_base(base_model)
    refiners, refiner_identity = construct_refiner_pair(device); execution = refiner_execution_contract(refiners, device, formal=True)
    endpoints = {"base": base_endpoint, **{arm: train_refiner(arm, refiners[arm], base_model, dataset, schedule, normalizer, device) for arm in ARMS}}
    endpoint_checks = endpoint_health(base_model, refiners, dataset, normalizer, device, frozen)
    mechanical = mechanical_health(base_identity, frozen, refiner_identity,
                                   execution, resources, endpoints,
                                   endpoint_checks)
    del dataset, fit_traj; gc.collect(); torch.cuda.empty_cache()
    held_traj, held_windows, held_files = load_held_fields_after_guards(args.sim_dir, held_rows, endpoints, mechanical)
    evaluation = evaluate(base_model, refiners, held_traj, held_windows, held_files, normalizer, device); gate = decide_gate(evaluation, refiner_identity, execution, mechanical)
    result = {"schema": SCHEMA, "evidence_class": EVIDENCE_CLASS,
              "protocol": {"name": protocol()["name"],
                           "acknowledgement": ACKNOWLEDGEMENT},
              "git": git, "held_aoa": HELD_AOA,
              "fit_aoas": list(FIT_AOAS),
              "data_grid": {"raw_h5_field": [64, 128],
                            "loader_spatial_stride": [2, 2],
                            "effective_model_grid": [32, 64]},
              "training": {"base_seed": BASE_SEED,
                           "refiner_seed": REFINER_SEED,
                           "schedule_seed": SCHEDULE_SEED,
                           "base_updates": UPDATES,
                           "updates_per_refiner": UPDATES,
                           "batch_size": BATCH_SIZE,
                           "optimizer": "AdamW", "lr": LR,
                           "weight_decay": WEIGHT_DECAY,
                           "scheduler": "CosineAnnealingLR_Tmax_2000",
                           "grad_clip": GRAD_CLIP,
                           "point_loss": "accuracy_1_4_1_plus_.25_mse",
                           "augmentation": "none", "teacher": "none",
                           "ema": "none", "uq": "none"},
              **seals, "sample_stream_sha256": schedule_sha,
              "base_initial_identity": base_identity,
              "base_normalizer": base_normalizer,
              "refiner_initial_identity": refiner_identity,
              "base_frozen": frozen,
              "refiner_execution_contract": execution,
              "resources": resources, "endpoints": endpoints,
              "endpoint_health": endpoint_checks,
              "mechanical_health": mechanical,
              "held_field_values_loaded_before_base_and_two_refiner_seals_and_mechanical_seal": False,
              "evaluation": evaluation, "gate": gate,
              "decision": decision_for(gate)}
    args.output_dir.mkdir(parents=True, exist_ok=True); path = args.output_dir / "result.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"], "result": str(path),
        "candidate_minus_control": evaluation["control_vs_candidate"][
            "delta_candidate_minus_control"],
        "candidate_minus_base_point": gate["candidate_point"] - gate["base_point"],
    }, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode", choices=("contract", "smoke", "formal"), default="contract")
    parser.add_argument("--sim-dir", type=Path); parser.add_argument("--output-dir", type=Path); parser.add_argument("--held-aoa", type=int, default=HELD_AOA)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--updates", type=int, default=UPDATES); parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--acknowledge", "--acknowledge-protocol", dest="acknowledge", default=""); return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract": print(json.dumps({"schema": SCHEMA, "held_aoa": HELD_AOA, "fit_aoas": FIT_AOAS, "evidence_class": EVIDENCE_CLASS, "acknowledgement": ACKNOWLEDGEMENT, "arms": ARM_MODE, "gate": GATE, "effective_grid": "existing loader ::2"}, sort_keys=True)); return
    device = torch.device(args.device)
    if args.mode == "smoke": print(json.dumps(smoke(device), sort_keys=True))
    else: run(args, device)


if __name__ == "__main__": main()
