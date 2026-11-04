#!/usr/bin/env python3
"""Condition-held Real4k with isolated base-member and CRPS adapter gradients."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import time
from types import MethodType
from typing import Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

import train_b210_condition_held_real as held
import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_np_functional_sim as b210_sim
import train_b210_scratch_real_dense_aug as dense4k
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1
from models.ensemble_unet2d import EnsembleUNet2d
from probabilistic_objectives import pointwise_ensemble_crps


EXPERIMENT = "b210_distribution_preserving_real4k_loao"
SCHEMA = "b210_distribution_preserving_real4k_fold_v1"
ENDPOINT_SCHEMA = "b210_distribution_preserving_real4k_endpoint_v1"
ARM = "distribution_preserving4k"
ENDPOINT = "ema_u4000"
UPDATES = 4_000
CRPS_WEIGHT = 0.05
EXPECTED_PARAMETERS = b210_sim.EXPECTED_BUDGETED_DISAGREEMENT_PARAMETERS
START_METHOD = b210_sim.BUDGETED_DISAGREEMENT_ALLOCATION_METHOD
GATE = {
    "point_contribution_min": -0.05,
    "sps_delta_min": 0.20,
    "non_time_contribution_min": 0.05,
    "native_minus_collapsed_sps_min": 0.30,
    "member_error_correlation_max": 0.80,
    "spread_error_spearman_min": 0.50,
}


def expected_config():
    return b210_sim.b210_config(
        "candidate",
        member_feature_adapters=True,
        mean_preserving_member_adapters=True,
        detach_member_adapter_context=True,
        detach_uq_context=True,
        budgeted_disagreement_allocation=True,
    )


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters if parameter.grad is not None]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


@contextmanager
def base_member_supervision_forward(
        model: EnsembleUNet2d) -> Iterator[dict[str, torch.Tensor]]:
    """Return base members to native supervision while caching real members.

    The interval tower is evaluated by the original forward before the member
    substitution, so it still receives the true detached disagreement.  Only
    the tensor exposed to the legacy bootstrap member loss is replaced.
    """
    original = model.forward_uq_training
    cache: dict[str, torch.Tensor] = {}

    def isolated(self: EnsembleUNet2d, history_norm: torch.Tensor):
        values = original(history_norm)
        prediction, adapted, quantiles, lower, upper, features, coverage = values
        base = self._members_from_features(
            history_norm, features, use_member_adapters=False)
        if self.config.physical_energy_member_sync:
            base = self._energy_controlled_members(
                base, features, history_norm)
        cache["adapted_members"] = adapted
        cache["base_members"] = base
        return prediction, base, quantiles, lower, upper, features, coverage

    model.forward_uq_training = MethodType(isolated, model)
    try:
        yield cache
    finally:
        model.forward_uq_training = original


ORIGINAL_REAL_BACKWARD = ft12.npt1.backward_real_supervised_npt1


def backward_distribution_preserving(
        model: EnsembleUNet2d, batch, normalizer,
        device: torch.device, iteration: int, **kwargs) -> dict[str, float]:
    """Route native supervision to base members and CRPS only to adapters."""
    if (model.member_feature_adapters is None
            or not model.config.mean_preserving_member_adapters
            or not model.config.detach_uq_context
            or not model.config.budgeted_disagreement_allocation):
        raise ValueError("distribution-preserving backward needs exact v7")
    with base_member_supervision_forward(model) as cache:
        values = ORIGINAL_REAL_BACKWARD(
            model, batch, normalizer, device, iteration, **kwargs)
    if set(cache) != {"adapted_members", "base_members"}:
        raise RuntimeError("distribution member cache is incomplete")

    _history, target, _trajectory_ids, _sample_indices = batch
    target = target.to(device, non_blocking=True)
    target_norm = (
        target - normalizer.mean_target.view(1, 1, 1, 1, -1)
    ) / normalizer.std_target.view(1, 1, 1, 1, -1)
    crps, attraction, repulsion = pointwise_ensemble_crps(
        cache["adapted_members"], target_norm)
    adapter_parameters = tuple(
        parameter for parameter in model.member_feature_adapters.parameters()
        if parameter.requires_grad)
    native_adapter_gradient_norm = _gradient_norm(adapter_parameters)
    adapter_gradients = torch.autograd.grad(
        CRPS_WEIGHT * crps, adapter_parameters,
        allow_unused=False, retain_graph=False)
    crps_gradient_norm = float(torch.stack([
        value.detach().float().square().sum()
        for value in adapter_gradients]).sum().sqrt())
    if not math.isfinite(crps_gradient_norm) or crps_gradient_norm <= 0.0:
        raise FloatingPointError("Real CRPS adapter gradient is invalid")
    for parameter, gradient in zip(
            adapter_parameters, adapter_gradients, strict=True):
        if parameter.grad is None:
            parameter.grad = gradient.detach().clone()
        else:
            parameter.grad.add_(gradient.detach())
    member_mean_gap = float((
        cache["adapted_members"].mean(dim=1)
        - cache["base_members"].mean(dim=1)).detach().abs().max())
    values.update({
        "distribution_preserving_real": 1.0,
        "native_member_supervision_uses_base": 1.0,
        "real_pointwise_crps": float(crps.detach()),
        "real_pointwise_crps_attraction": float(attraction.detach()),
        "real_pointwise_crps_repulsion": float(repulsion.detach()),
        "real_pointwise_crps_weight": CRPS_WEIGHT,
        "native_adapter_gradient_norm": native_adapter_gradient_norm,
        "crps_adapter_gradient_norm": crps_gradient_norm,
        "member_mean_prediction_max_abs_gap": member_mean_gap,
    })
    return values


@contextmanager
def endpoint_contract(held_aoa: int) -> Iterator[None]:
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "PRIMARY_ENDPOINT": ENDPOINT,
        "UPDATES": UPDATES,
        "SCHEDULER_T_MAX": 12_000,
        "HELD_AOA": held_aoa,
    }
    original = {name: getattr(small4k, name) for name in replacements}
    original_backward = ft12.npt1.backward_real_supervised_npt1
    try:
        for name, value in replacements.items():
            setattr(small4k, name, value)
        ft12.npt1.backward_real_supervised_npt1 = (
            backward_distribution_preserving)
        yield
    finally:
        ft12.npt1.backward_real_supervised_npt1 = original_backward
        for name, value in original.items():
            setattr(small4k, name, value)


def load_endpoint(
        seal: Mapping[str, object], normalizer: smu1.FoldNormalizer,
        device: torch.device, held_aoa: int) -> EnsembleUNet2d:
    path = Path(str(seal["path"]))
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("distribution-preserving endpoint file changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != ENDPOINT_SCHEMA
            or payload.get("experiment") != EXPERIMENT
            or payload.get("endpoint") != ENDPOINT
            or int(payload.get("iteration", -1)) != UPDATES
            or int(payload.get("held_aoa", -1)) != held_aoa
            or not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError("distribution-preserving endpoint payload changed")
    parsed = ft12.validate_b210_model_config(
        payload["model_config"], expected_config())
    model = EnsembleUNet2d(parsed).float()
    model.load_state_dict(state, strict=True)
    ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def decision_gate(comparison: Mapping[str, object], diagnostic: Mapping[str, float]) -> dict:
    conditions = {
        "point_contribution": float(
            comparison["point_final_contribution_delta"])
        >= GATE["point_contribution_min"],
        "sps_delta": float(comparison["delta"]["native_sps_score"])
        >= GATE["sps_delta_min"],
        "non_time_contribution": float(
            comparison["non_time_final_contribution_delta"])
        >= GATE["non_time_contribution_min"],
        "native_beats_collapsed": float(
            diagnostic["native_minus_collapsed_sps"])
        >= GATE["native_minus_collapsed_sps_min"],
        "members_not_collapsed": float(
            diagnostic["member_error_pairwise_correlation_mean"])
        <= GATE["member_error_correlation_max"],
        "spread_localizes_error": float(
            diagnostic["element_spread_log_spearman"])
        >= GATE["spread_error_spearman_min"],
    }
    passed = all(conditions.values())
    return {
        "thresholds": dict(GATE),
        "conditions": conditions,
        "pass": passed,
        "decision": (
            "PASS_DISTRIBUTION_PRESERVING_REAL4K_HELD_FOLD"
            if passed else "STOP_DISTRIBUTION_PRESERVING_REAL4K_HELD_FOLD"),
    }


def run_fold(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.held_aoa not in held.AOA_VALUES:
        raise ValueError("held AoA is outside released conditions")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("distribution-preserving Real4k requires CUDA")
    control_row, control = held.load_frozen_control(
        args.control_fold_result, args.held_aoa)
    rows, manifest = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    if manifest != control["manifest"]:
        raise RuntimeError("manifest differs from frozen Dense4k control")
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    start_sha = dr1.sha256(args.start_sim)
    if start_sha != args.start_sim_sha256.lower():
        raise RuntimeError("v7 Sim start SHA changed")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    audit = held.PrefixAccessAudit(args.held_aoa, [ARM])
    fit_trajectories, fit_metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", audit)
    dense_windows = dense4k.dense_windows(
        fit_trajectories, dense4k.CANDIDATE_STRIDE)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        fit_trajectories, dense_windows)
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, sparse_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    statistics_sha = smu1.statistics_digest(statistics)
    if statistics_sha != control["normalizer"]["statistics_sha256"]:
        raise RuntimeError("candidate normalizer differs from frozen control")

    ft12.npt1.ensemble_train.seed_everything(small4k.SEED)
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha, normalizer, device,
        expected_config=expected_config(),
        expected_parameters=EXPECTED_PARAMETERS,
        expected_controller_parameters=ft12.EXPECTED_CONTROLLER_PARAMETERS)
    schedule = held.uniform_batch_schedule(dense_windows, UPDATES)
    arm_dir = args.output_dir / ARM
    arm_dir.mkdir()
    seal_adapter = held.ArmSealAdapter(audit, ARM)
    with endpoint_contract(args.held_aoa):
        training, seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            arm_dir, start_report, seal_adapter,
            input_augmentation=dense4k.AUGMENTATION,
            endpoint_filename=f"{ENDPOINT}.pth",
            expected_parameters=EXPECTED_PARAMETERS)
    if seal_adapter.value != seal:
        raise RuntimeError("candidate endpoint is not access-audit bound")
    del model, dataset, fit_trajectories
    gc.collect()
    torch.cuda.empty_cache()

    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", audit)
    held_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    model = load_endpoint(seal, normalizer, device, args.held_aoa)
    metrics, metric_state, target_audit = smu1.evaluate_model(
        model, held_dataset, list(range(len(held_windows))),
        normalizer, device, deadline)
    if target_audit != control["target_audit"]:
        raise RuntimeError("held targets differ from frozen control")
    diagnostic_loader = DataLoader(
        held_dataset, batch_size=8, shuffle=False, num_workers=0)
    from diagnose_ensemble_disagreement import diagnose
    raw_diagnostic = diagnose(
        model, diagnostic_loader, normalizer, device, element_stride=197)
    diagnostic = {
        key: float(raw_diagnostic[key]) for key in (
            "native_sps_score", "zero_disagreement_native_sps_score",
            "native_sps_mean_width",
            "zero_disagreement_native_sps_mean_width",
            "element_spread_log_spearman", "sample_spread_spearman",
            "member_error_pairwise_correlation_mean",
            "top_spread_decile_error_amplification",
            "top_error_decile_recall_at_top_spread_decile")}
    diagnostic["native_minus_collapsed_sps"] = (
        diagnostic["native_sps_score"]
        - diagnostic["zero_disagreement_native_sps_score"])
    comparison = held.metric_comparison(metrics, control["metrics"])
    gate = decision_gate(comparison, diagnostic)
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": args.held_aoa,
        "manifest": manifest,
        "normalizer": {
            "kind": "fit-condition sparse prefixes only",
            "statistics_sha256": statistics_sha,
            "matches_frozen_control": True,
            "held_condition_in_normalizer": False,
        },
        "start": {
            "path": str(args.start_sim.resolve()),
            "sha256": start_sha,
            "method": START_METHOD,
            "parameters": EXPECTED_PARAMETERS,
        },
        "training_contract": {
            "updates": UPDATES,
            "sampling": schedule["sampling"],
            "sample_stream_sha256": schedule["sha256"],
            "input_augmentation": dict(dense4k.AUGMENTATION),
            "native_member_supervision": "base members only",
            "stochastic_adapter_supervision": "pointwise CRPS only",
            "crps_weight": CRPS_WEIGHT,
            "uq_context": "detached",
        },
        "control": control,
        "arm": {
            "training": training,
            "seal": seal,
            "metrics": metrics,
            "metric_state": metric_state,
            "target_audit": target_audit,
        },
        "comparison": comparison,
        "disagreement_diagnostic": diagnostic,
        "gate": gate,
        "access_audit": audit.report(),
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_class": (
            "Real-stage whole-AoA-held validation against SHA-frozen "
            "Dense4k control; no temporal-tail labels"),
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-aoa", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--start-sim-sha256", required=True)
    parser.add_argument(
        "--control-fold-result", nargs=2, required=True,
        metavar=("PATH", "SHA256"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_fold(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dr1.write_json_atomic(args.output_dir / "stop.json", {
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "error_type": type(error).__name__,
            "error": str(error),
            "submission_authorized": False,
        })
        raise
    print(json.dumps({
        "event": "b210_distribution_preserving_real4k_complete",
        "held_aoa": result["held_aoa"],
        "decision": result["gate"]["decision"],
        "output": str(args.output_dir / "fold_result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
