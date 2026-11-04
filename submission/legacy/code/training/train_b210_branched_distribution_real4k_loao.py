#!/usr/bin/env python3
"""Condition-held Real4k with separate point and branched-future roles."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

import train_b210_distribution_preserving_real4k_loao as same_target
import train_b210_np_functional_sim as b210_sim
import train_c208_dr1_clean81_paired_loao as dr1
from physics.closure_augmentation import ClosureResidualAugmentor
from probabilistic_objectives import pointwise_ensemble_crps


EXPERIMENT = "b210_branched_distribution_real4k_loao"
SCHEMA = "b210_branched_distribution_real4k_fold_v1"
ENDPOINT_SCHEMA = "b210_branched_distribution_real4k_endpoint_v1"
ARM = "branched_distribution4k"
CRPS_WEIGHT = 0.05
FUTURE_PROBABILITY = 1.0
FUTURE_AMPLITUDE = 0.05
FUTURE_RNG_OFFSET = 104_729
FUTURE_RNG_BASE = 28_082_901
POINT_IDENTITY_TOLERANCE = 1.0e-5
GATE = {
    "sps_vs_same_target_min": 0.15,
    "native_minus_collapsed_sps_min": 0.30,
    "member_error_correlation_max": 0.65,
    "spread_error_spearman_min": 0.55,
    "point_vs_dense_min": -0.02,
    "non_time_vs_dense_min": 0.03,
}

_FUTURE_AUGMENTOR: ClosureResidualAugmentor | None = None
_FUTURE_BANK_AUDIT: dict | None = None


def _adapter_gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters if parameter.grad is not None]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt())


def prepare_fit_prefix_augmentor(
        bank_path: Path, bank_sha256: str, fit_metadata: list[dict],
        held_aoa: int, device: torch.device,
        ) -> tuple[ClosureResidualAugmentor, dict]:
    """Load only fit-condition closure records for target-only futures."""
    actual_sha = dr1.sha256(bank_path)
    if actual_sha != bank_sha256.lower():
        raise RuntimeError("Real closure bank SHA-256 changed")
    payload = torch.load(bank_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "realpde_clean_real_closure_bank_v2":
        raise ValueError("branched Real4k needs the clean81 v2 closure bank")
    source = payload.get("source_contract")
    if (not isinstance(source, Mapping)
            or source.get("manifest_sha256")
            != b210_sim.CLEAN_REAL_MANIFEST_SHA256
            or source.get("records") != 81
            or not source.get("known_invalid_trajectory_excluded")
            or source.get("validation_or_test_labels_read") is not False):
        raise RuntimeError("closure bank source contract changed")

    fit_files = [str(row["file"]) for row in fit_metadata]
    if len(fit_files) != len(set(fit_files)):
        raise ValueError("fit metadata contains duplicate trajectories")
    records_by_file = {
        str(record["file"]): record for record in payload["records"]}
    missing = sorted(set(fit_files) - set(records_by_file))
    if missing:
        raise RuntimeError(f"closure bank misses fit trajectories: {missing}")
    records = [records_by_file[name] for name in fit_files]
    if any(int(record["aoa"]) == held_aoa for record in records):
        raise RuntimeError("held-condition closure record entered fit bank")
    if {str(record["file"]) for record in records} != set(fit_files):
        raise RuntimeError("fit closure record identity mismatch")

    augmentor = ClosureResidualAugmentor(
        records, fit_metadata, candidate_count=3,
        source_contract=source).to(device)
    audit = {
        "path": str(bank_path.resolve()),
        "sha256": actual_sha,
        "source_manifest_sha256": source["manifest_sha256"],
        "source_prefix_rule": source.get("prefix_rule"),
        "validation_or_test_labels_read": False,
        "held_aoa": held_aoa,
        "held_records": sum(
            int(record["aoa"]) == held_aoa for record in records),
        "fit_records": len(records),
        "fit_files_sha256": dr1.canonical_sha256(sorted(fit_files)),
        "candidate_count": 3,
    }
    return augmentor, audit


def backward_branched_distribution(
        model, batch, normalizer, device: torch.device,
        iteration: int, **kwargs) -> dict[str, float]:
    """Train point/base UQ on truth and stochastic adapters on a future."""
    if _FUTURE_AUGMENTOR is None:
        raise RuntimeError("fit-prefix future augmentor was not prepared")
    if (model.member_feature_adapters is None
            or not model.config.mean_preserving_member_adapters
            or not model.config.detach_uq_context
            or not model.config.budgeted_disagreement_allocation):
        raise ValueError("branched Real4k needs the exact v7 role split")

    with same_target.base_member_supervision_forward(model) as cache:
        values = same_target.ORIGINAL_REAL_BACKWARD(
            model, batch, normalizer, device, iteration, **kwargs)
    if set(cache) != {"adapted_members", "base_members"}:
        raise RuntimeError("branched member cache is incomplete")

    history, target, trajectory_ids, _sample_indices = batch
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    cuda_devices = (
        [device.index if device.index is not None
         else torch.cuda.current_device()]
        if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=cuda_devices):
        branch_seed = FUTURE_RNG_BASE + FUTURE_RNG_OFFSET * iteration
        torch.manual_seed(branch_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(branch_seed)
        with torch.no_grad():
            future, selected = _FUTURE_AUGMENTOR.augment_future(
                history, target, trajectory_ids,
                probability=FUTURE_PROBABILITY,
                amplitude=FUTURE_AMPLITUDE,
                closure_decay=b210_sim.REAL_CLOSURE_DECAY,
                perturbation_decay=b210_sim.REAL_CLOSURE_PERTURBATION_DECAY,
                advection_cells=b210_sim.REAL_CLOSURE_ADVECTION_CELLS)
    if not bool(selected.all()):
        raise RuntimeError("probability-one future branching did not select all")
    _unused_history, future_norm = normalizer.preprocess(history, future)

    crps, attraction, repulsion = pointwise_ensemble_crps(
        cache["adapted_members"], future_norm)
    adapter_parameters = tuple(
        parameter for parameter in model.member_feature_adapters.parameters()
        if parameter.requires_grad)
    native_adapter_gradient_norm = _adapter_gradient_norm(adapter_parameters)
    gradients = torch.autograd.grad(
        CRPS_WEIGHT * crps, adapter_parameters,
        allow_unused=False, retain_graph=False)
    crps_gradient_norm = float(torch.stack([
        gradient.detach().float().square().sum()
        for gradient in gradients]).sum().sqrt())
    if not math.isfinite(crps_gradient_norm) or crps_gradient_norm <= 0.0:
        raise FloatingPointError("branched-future CRPS gradient is invalid")
    for parameter, gradient in zip(
            adapter_parameters, gradients, strict=True):
        if parameter.grad is None:
            parameter.grad = gradient.detach().clone()
        else:
            parameter.grad.add_(gradient.detach())

    member_mean_gap = float((
        cache["adapted_members"].mean(dim=1)
        - cache["base_members"].mean(dim=1)).detach().abs().max())
    future_uv_rms = float((
        future[..., :2].float() - target[..., :2].float()
    ).square().mean().sqrt())
    values.update({
        "branched_distribution_real": 1.0,
        "native_member_supervision_uses_base": 1.0,
        "future_probability": FUTURE_PROBABILITY,
        "future_amplitude": FUTURE_AMPLITUDE,
        "future_selected_fraction": float(selected.float().mean()),
        "future_target_delta_uv_rms": future_uv_rms,
        "future_pointwise_crps": float(crps.detach()),
        "future_pointwise_crps_attraction": float(attraction.detach()),
        "future_pointwise_crps_repulsion": float(repulsion.detach()),
        "future_pointwise_crps_weight": CRPS_WEIGHT,
        "native_adapter_gradient_norm": native_adapter_gradient_norm,
        "future_crps_adapter_gradient_norm": crps_gradient_norm,
        "member_mean_prediction_max_abs_gap": member_mean_gap,
    })
    return values


def _load_attested_json(spec: tuple[Path, str], label: str) -> dict:
    path, expected_sha = spec
    actual_sha = dr1.sha256(path)
    if actual_sha != expected_sha.lower():
        raise RuntimeError(f"{label} SHA-256 changed")
    value = dr1.load_json_mapping(path, label)
    value = dict(value)
    value["path"] = str(path.resolve())
    value["sha256"] = actual_sha
    return value


def decision_gate(
        result: Mapping[str, object], same_target_result: Mapping[str, object],
        mechanism_comparison: Mapping[str, object],
        ) -> dict:
    candidate_metrics = result["arm"]["metrics"]
    same_metrics = same_target_result["arm"]["metrics"]
    point_keys = (
        "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
        "point_score")
    point_differences = {
        key: float(candidate_metrics[key]) - float(same_metrics[key])
        for key in point_keys}
    point_identity_max_abs = max(map(abs, point_differences.values()))
    diagnostic = result["disagreement_diagnostic"]
    conditions = {
        "point_function_identity": (
            point_identity_max_abs <= POINT_IDENTITY_TOLERANCE),
        "same_sample_stream": (
            result["training_contract"]["sample_stream_sha256"]
            == same_target_result["training_contract"][
                "sample_stream_sha256"]),
        "same_normalizer": (
            result["normalizer"]["statistics_sha256"]
            == same_target_result["normalizer"]["statistics_sha256"]),
        "same_held_targets": (
            result["arm"]["target_audit"]
            == same_target_result["arm"]["target_audit"]),
        "held_bank_excluded": _FUTURE_BANK_AUDIT["held_records"] == 0,
        "sps_vs_same_target": float(
            mechanism_comparison["delta"]["native_sps_score"])
            >= GATE["sps_vs_same_target_min"],
        "native_beats_collapsed": float(
            diagnostic["native_minus_collapsed_sps"])
            >= GATE["native_minus_collapsed_sps_min"],
        "members_not_collapsed": float(
            diagnostic["member_error_pairwise_correlation_mean"])
            <= GATE["member_error_correlation_max"],
        "spread_localizes_error": float(
            diagnostic["element_spread_log_spearman"])
            >= GATE["spread_error_spearman_min"],
        "point_vs_dense": float(
            result["comparison"]["point_final_contribution_delta"])
            >= GATE["point_vs_dense_min"],
        "non_time_vs_dense": float(
            result["comparison"]["non_time_final_contribution_delta"])
            >= GATE["non_time_vs_dense_min"],
    }
    passed = all(conditions.values())
    return {
        "thresholds": {
            **GATE,
            "point_identity_tolerance": POINT_IDENTITY_TOLERANCE,
        },
        "point_metric_differences_vs_same_target": point_differences,
        "point_identity_max_abs": point_identity_max_abs,
        "conditions": conditions,
        "pass": passed,
        "decision": (
            "PASS_BRANCHED_DISTRIBUTION_REAL4K_HELD10"
            if passed else "STOP_BRANCHED_DISTRIBUTION_REAL4K_HELD10"),
    }


def run(args: argparse.Namespace) -> dict:
    global _FUTURE_AUGMENTOR, _FUTURE_BANK_AUDIT
    rows, _manifest = dr1.load_manifest(
        args.clean_manifest,
        same_target.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    fit_metadata = [
        {
            "file": str(row["file"]),
            "frames": int(row["val_boundary"]),
            "source_frames": int(row["frames"]),
            "re": int(row["re"]),
            "aoa": int(row["aoa"]),
            "val_boundary": int(row["val_boundary"]),
        }
        for row in rows if int(row["aoa"]) != args.held_aoa]
    device = torch.device(args.device)
    _FUTURE_AUGMENTOR, _FUTURE_BANK_AUDIT = prepare_fit_prefix_augmentor(
        args.real_closure_bank, args.real_closure_bank_sha256,
        fit_metadata, args.held_aoa, device)
    same_result = _load_attested_json(
        args.same_target_control_fold_result, "same-target CRPS control")

    patched = {
        "EXPERIMENT": EXPERIMENT,
        "SCHEMA": SCHEMA,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "ARM": ARM,
        "backward_distribution_preserving": backward_branched_distribution,
    }
    original = {name: getattr(same_target, name) for name in patched}
    try:
        for name, value in patched.items():
            setattr(same_target, name, value)
        result = same_target.run_fold(args)
    finally:
        for name, value in original.items():
            setattr(same_target, name, value)

    mechanism = same_target.held.metric_comparison(
        result["arm"]["metrics"], same_result["arm"]["metrics"])
    result["schema"] = SCHEMA
    result["experiment"] = EXPERIMENT
    result["training_contract"].update({
        "deterministic_point_target": "released Real prefix truth",
        "base_interval_target": "released Real prefix truth",
        "stochastic_adapter_supervision": (
            "pointwise CRPS against target-only fit-prefix closure future"),
        "future_probability": FUTURE_PROBABILITY,
        "future_amplitude": FUTURE_AMPLITUDE,
        "future_rng": (
            "independent fork; base + offset * iteration"),
        "crps_weight": CRPS_WEIGHT,
    })
    result["future_bank_audit"] = dict(_FUTURE_BANK_AUDIT)
    result["same_target_control"] = {
        "path": same_result["path"],
        "sha256": same_result["sha256"],
        "metrics": same_result["arm"]["metrics"],
        "disagreement_diagnostic": same_result["disagreement_diagnostic"],
    }
    result["mechanism_comparison_vs_same_target"] = mechanism
    result["gate"] = decision_gate(result, same_result, mechanism)
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    _FUTURE_AUGMENTOR = None
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
        type=Path, metavar=("PATH", "SHA256"))
    parser.add_argument(
        "--same-target-control-fold-result", nargs=2, required=True,
        type=Path, metavar=("PATH", "SHA256"))
    parser.add_argument("--real-closure-bank", type=Path, required=True)
    parser.add_argument("--real-closure-bank-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    # SHA values are parsed as paths only because argparse applies one type to
    # both nargs fields.  Convert them back to plain hexadecimal strings.
    args.control_fold_result = (
        args.control_fold_result[0], str(args.control_fold_result[1]))
    args.same_target_control_fold_result = (
        args.same_target_control_fold_result[0],
        str(args.same_target_control_fold_result[1]))
    return args


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
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
        "event": "b210_branched_distribution_real4k_complete",
        "held_aoa": result["held_aoa"],
        "decision": result["gate"]["decision"],
        "output": str(args.output_dir / "fold_result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
