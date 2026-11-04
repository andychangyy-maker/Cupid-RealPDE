#!/usr/bin/env python3
"""Whole-AoA OOF validation for fresh Native-UQ on 3,341 Real windows.

For one held AoA, the point model is the SHA-bound Real4k endpoint whose
training excluded that AoA.  Every native UQ-only tensor is reset from the
formal pure-Sim10k checkpoint.  A fixed u800 endpoint is then fit only on the
other four AoAs' point-OOF residuals.  The held AoA is scored only after that
endpoint is sealed; it never contributes an optimizer sample.

The aggregate combines all five held-AoA sufficient metric states and reports
native-before/native-after SPS without a hand-authored promotion threshold.
It is mandatory provenance for the later all-3,341 fixed 4x800 refit.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

import train_b210_full_real_oof_native_uq as full_uq


point_fold = full_uq.point_fold
cache_builder = full_uq.cache_builder
dr1 = full_uq.dr1
ft12 = full_uq.ft12
smu1 = full_uq.smu1
uq_release = full_uq.uq_release
uq_train = full_uq.uq_train

EXPERIMENT = "b210_full_real_loao_native_uq"
FOLD_SCHEMA = "b210_full_real_loao_native_uq_fold_v1"
AGGREGATE_SCHEMA = full_uq.LOAO_AGGREGATE_SCHEMA
ENDPOINT_SCHEMA = "b210_full_real_loao_native_uq_endpoint_v1"
ACKNOWLEDGEMENT = "RUN_B210_FULL_REAL_LOAO_NATIVE_UQ_V1"
UPDATES = full_uq.UPDATES_PER_CYCLE
FOLDS = point_fold.FOLDS


def partition_indices(
        identities: Sequence[Mapping[str, object]], held_aoa: int,
        ) -> tuple[list[int], list[int], dict]:
    point_fold.aoa_fold(held_aoa)
    fit = [
        index for index, row in enumerate(identities)
        if int(row["aoa"]) != int(held_aoa)]
    held = [
        index for index, row in enumerate(identities)
        if int(row["aoa"]) == int(held_aoa)]
    expected_held = point_fold.EXPECTED_FOLDS[held_aoa]["held_oof_windows"]
    if (len(fit) + len(held) != full_uq.EXPECTED_SAMPLES
            or len(held) != expected_held
            or set(fit).intersection(held)
            or sorted(fit + held) != list(range(full_uq.EXPECTED_SAMPLES))):
        raise RuntimeError("whole-AoA UQ fold partition changed")
    fit_identities = [identities[index] for index in fit]
    held_identities = [identities[index] for index in held]
    return fit, held, {
        "fit_samples": len(fit),
        "held_samples": len(held),
        "fit_identity_sha256": dr1.canonical_sha256(fit_identities),
        "held_identity_sha256": dr1.canonical_sha256(held_identities),
        "held_optimizer_samples": 0,
        "held_aoa": int(held_aoa),
    }


def load_point_fold(
        result_path: Path, expected_result_sha256: str, held_aoa: int,
        cache: Mapping[str, object], device: torch.device,
        ) -> tuple[dict, torch.nn.Module, smu1.FoldNormalizer, dict]:
    expected_result_sha256 = dr1.normalized_sha256(
        expected_result_sha256, "point-OOF fold result SHA")
    if (not result_path.is_file()
            or dr1.sha256(result_path) != expected_result_sha256):
        raise RuntimeError("point-OOF fold result SHA changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    point_fold.verify_json_payload(result)
    fold = point_fold.aoa_fold(held_aoa)
    if (result.get("schema") != point_fold.SCHEMA
            or int(result.get("held_aoa", -1)) != held_aoa
            or int(result.get("held_fold", -1)) != fold
            or result.get("decision") != "COMPLETE_FULL_REAL_POINT_OOF_FOLD"
            or result.get("held_data_opened") is not False
            or result.get("clean716_manifest_read") is not False
            or result.get("start", {}).get("checkpoint_sha256")
            != point_fold.START_SHA256
            or result.get("manifest_file_sha256")
            != ft12.FORMAL_MANIFEST_FILE_SHA256):
        raise RuntimeError("point-OOF fold result contract changed")
    cache_rows = [
        row for row in cache.get("fold_results", [])
        if int(row.get("held_aoa", -1)) == held_aoa]
    if (len(cache_rows) != 1
            or cache_rows[0].get("result_sha256") != expected_result_sha256
            or Path(str(cache_rows[0].get("result", ""))).resolve()
            != result_path.resolve()):
        raise RuntimeError("point fold is not the one bound into OOF cache")
    seal = result.get("fit", {}).get("seal", {})
    checkpoint = Path(str(seal.get("path", "")))
    accepted_schemas = (
        point_fold.ENDPOINT_SCHEMA, point_fold.support.ENDPOINT_SCHEMA)
    accepted_experiments = (
        point_fold.EXPERIMENT, point_fold.support.EXPERIMENT)
    verified = point_fold.validate_endpoint(
        checkpoint, seal, held_aoa=held_aoa,
        accepted_schemas=accepted_schemas,
        accepted_experiments=accepted_experiments)
    if (cache_rows[0].get("checkpoint_sha256")
            != verified["file_sha256"]
            or cache_rows[0].get("model_state_sha256")
            != verified["model_state_sha256"]):
        raise RuntimeError("point fold checkpoint differs from OOF cache binding")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    normalizer, normalizer_report = full_uq.normalizer_from_payload(
        payload, device)
    model = uq_train.load_model(
        payload, normalizer, device, train_projections=True)
    point_partition_sha = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    payload_point_sha = dr1.model_state_sha256({
        key: value for key, value in payload["model_state_dict"].items()
        if not key.startswith(uq_release.UQ_PREFIXES)})
    if point_partition_sha != payload_point_sha:
        raise RuntimeError("loaded point-OOF non-UQ partition changed")
    return payload, model, normalizer, {
        "held_aoa": held_aoa,
        "result": str(result_path.resolve()),
        "result_sha256": expected_result_sha256,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": verified["file_sha256"],
        "model_state_sha256": verified["model_state_sha256"],
        "point_partition_sha256": point_partition_sha,
        "normalizer": normalizer_report,
        "source_kind": result["source_kind"],
    }


def load_reference_final_uq(
        result_path: Path, expected_result_sha256: str,
        final_point_loader: Callable | None = None,
        ) -> tuple[dict[str, torch.Tensor], dict]:
    """Read the all-81 Real UQ partition for reference arm A only."""
    loader = (
        full_uq.load_final_point
        if final_point_loader is None else final_point_loader)
    _payload, model, _normalizer, point_report = loader(
        result_path, expected_result_sha256, torch.device("cpu"))
    state = {
        key: value.detach().cpu().clone()
        for key, value in uq_release.partition_state(model, uq=True).items()}
    if not state:
        raise RuntimeError("reference final point has no native UQ state")
    report = {
        "result": str(result_path.resolve()),
        "result_sha256": expected_result_sha256,
        "checkpoint": point_report["checkpoint"],
        "checkpoint_sha256": point_report["checkpoint_sha256"],
        "uq_state_sha256": dr1.model_state_sha256(state),
        "uq_state_tensors": len(state),
        "real_in_sample_initialization": True,
        "reference_only": True,
        "optimizer_source": False,
        "checkpoint_selection_source": False,
    }
    del model
    gc.collect()
    return state, report


def replace_uq_state(
        model: torch.nn.Module, state: Mapping[str, torch.Tensor],
        expected_point_sha256: str, label: str,
        ) -> dict:
    destination = model.state_dict()
    expected = {
        key for key in destination if key.startswith(uq_release.UQ_PREFIXES)}
    if not state or set(state) != expected:
        raise RuntimeError(f"{label} UQ state-key partition changed")
    for key, value in state.items():
        if (destination[key].shape != value.shape
                or destination[key].dtype != value.dtype):
            raise RuntimeError(f"{label} UQ tensor contract changed at {key}")
    point_before = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    if point_before != expected_point_sha256:
        raise RuntimeError(f"{label} did not start from held-fold point state")
    for key, value in state.items():
        destination[key] = value
    model.load_state_dict(destination, strict=True)
    point_after = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    uq_after = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=True))
    if point_after != point_before:
        raise RuntimeError(f"{label} UQ load changed held-fold point state")
    expected_uq_sha = dr1.model_state_sha256(state)
    if uq_after != expected_uq_sha:
        raise RuntimeError(f"{label} UQ load changed source tensors")
    return {
        "label": label,
        "uq_state_sha256": uq_after,
        "point_partition_sha256_before": point_before,
        "point_partition_sha256_after": point_after,
        "point_partition_bitwise": True,
    }


@torch.inference_mode()
def evaluate_oof_native(
        model: torch.nn.Module, dataset: full_uq.OOFResidualDataset,
        indices: Sequence[int], normalizer: smu1.FoldNormalizer,
        device: torch.device, deadline: float,
        ) -> tuple[dict, dict, dict, dict]:
    """Score intervals with cached held-fold OOF residual/factor tensors."""
    if not indices:
        raise ValueError("OOF Native-UQ evaluation partition is empty")
    loader = DataLoader(
        Subset(dataset, list(indices)), batch_size=smu1.EVAL_BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    state = smu1.new_metric_state()
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    override_numerator = 0.0
    override_scored = 0
    override_covered = 0
    override_width_sum = 0.0
    residual_drift_sum = 0.0
    residual_drift_count = 0
    residual_drift_max = 0.0
    factor_drift_sum = 0.0
    factor_drift_max = 0.0
    samples = 0
    model.eval()
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    sigma = float(smu1.full.OFFICIAL.SIGMA_GLOBAL)
    for history, target, residual, factor in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("LOAO-UQ wall budget expired during held scoring")
        target_cpu = target.contiguous()
        scored_cpu = target_cpu[..., :2].ne(0.0).to(torch.uint8).contiguous()
        target_hasher.update(str(tuple(target_cpu.shape)).encode("ascii"))
        target_hasher.update(target_cpu.numpy().tobytes(order="C"))
        mask_hasher.update(str(tuple(scored_cpu.shape)).encode("ascii"))
        mask_hasher.update(scored_cpu.numpy().tobytes(order="C"))
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        residual = residual.to(device, dtype=torch.float32, non_blocking=True)
        factor = factor.to(device, dtype=torch.float32, non_blocking=True)
        history_norm, _target_norm = normalizer.preprocess(history, target)
        prediction_norm, lower_norm, upper_norm = model.forward_distribution(
            history_norm)
        prediction = normalizer.postprocess_prediction(prediction_norm.float())
        truth = target.float()
        lower = lower_norm.float() * scale
        upper = upper_norm.float() * scale
        # update_metric_state supplies point sufficient statistics; its SPS
        # fields are replaced below by the explicitly cached OOF calculation.
        smu1.update_metric_state(state, prediction, truth, lower, upper)
        scored = truth[..., :2] != 0.0
        inside = (residual >= -lower) & (residual <= upper)
        penalty = torch.exp(-(lower + upper) / sigma)
        shaped_factor = factor.view((-1,) + (1,) * (residual.ndim - 1))
        value = shaped_factor * inside * penalty
        override_numerator += float(value[scored].sum(dtype=torch.float64))
        override_scored += int(scored.sum())
        override_covered += int((inside & scored).sum())
        override_width_sum += float(
            (lower + upper)[scored].sum(dtype=torch.float64))

        actual_residual = truth[..., :2] - prediction[..., :2]
        residual_drift = (actual_residual - residual).abs()
        actual_factor = ft12.npt1.ensemble_train.common.sample_sps_factor(
            prediction[..., :2], truth[..., :2]).float()
        factor_drift = (actual_factor - factor).abs()
        residual_drift_sum += float(residual_drift.sum(dtype=torch.float64))
        residual_drift_count += residual_drift.numel()
        residual_drift_max = max(
            residual_drift_max, float(residual_drift.max()))
        factor_drift_sum += float(factor_drift.sum(dtype=torch.float64))
        factor_drift_max = max(factor_drift_max, float(factor_drift.max()))
        samples += len(target)
    state["sps_numerator"] = override_numerator
    state["sps_scored"] = override_scored
    state["sps_covered"] = override_covered
    state["sps_width_sum"] = override_width_sum
    if samples != len(indices) or int(state["samples"]) != len(indices):
        raise RuntimeError("LOAO-UQ scorer did not consume complete held fold")
    metrics = smu1.finalize_metric_state(state)
    target_audit = {
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
        "samples": samples,
        "scored_elements": override_scored,
    }
    override_audit = {
        "source": "held-AoA point-OOF residual/factor cache",
        "cached_residual_used_for_inside_test": True,
        "cached_point_factor_used_for_sps": True,
        "raw_target_used_for_sps_residual_or_factor": False,
        "raw_target_used_for_scored_mask_and_point_metrics": True,
        "residual_fp16_reconstruction_mean_abs": (
            residual_drift_sum / residual_drift_count),
        "residual_fp16_reconstruction_max_abs": residual_drift_max,
        "point_factor_reconstruction_mean_abs": factor_drift_sum / samples,
        "point_factor_reconstruction_max_abs": factor_drift_max,
    }
    return metrics, state, target_audit, override_audit


def save_fold_endpoint(
        path: Path, model: torch.nn.Module, held_aoa: int,
        point_report: Mapping[str, object],
        cache_report: Mapping[str, object],
        initialization: Mapping[str, object],
        training: Mapping[str, object],
        partition: Mapping[str, object],
        ) -> dict:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    uq_state = {
        key: value.detach().cpu().clone()
        for key, value in uq_release.partition_state(model, uq=True).items()}
    uq_sha = dr1.model_state_sha256(uq_state)
    frozen_sha = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    if frozen_sha != point_report["point_partition_sha256"]:
        raise RuntimeError("LOAO-UQ endpoint changed point/source partition")
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": held_aoa,
        "held_fold": point_fold.aoa_fold(held_aoa),
        "endpoint": "fixed_oof_u800",
        "iteration": UPDATES,
        "uq_state_dict": uq_state,
        "uq_state_sha256": uq_sha,
        "point": dict(point_report),
        "point_partition_sha256": frozen_sha,
        "oof_residual_cache": dict(cache_report),
        "uq_initialization": dict(initialization),
        "partition": dict(partition),
        "training": dict(training),
        "held_scored_after_seal": False,
        "held_checkpoint_selection": False,
        "clean716_manifest_read": False,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "uq_state_sha256": uq_sha,
        "point_partition_sha256": frozen_sha,
        "endpoint": "fixed_oof_u800",
        "iteration": UPDATES,
    }


def load_fold_endpoint(
        model: torch.nn.Module, seal: Mapping[str, object], held_aoa: int,
        expected_point_sha256: str) -> dict:
    path = Path(str(seal.get("path", "")))
    if not path.is_file() or dr1.sha256(path) != seal.get("file_sha256"):
        raise RuntimeError("LOAO-UQ endpoint file SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("uq_state_dict")
    if (payload.get("schema") != ENDPOINT_SCHEMA
            or payload.get("experiment") != EXPERIMENT
            or int(payload.get("held_aoa", -1)) != held_aoa
            or payload.get("endpoint") != "fixed_oof_u800"
            or int(payload.get("iteration", -1)) != UPDATES
            or payload.get("held_checkpoint_selection") is not False
            or not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal.get("uq_state_sha256")
            or payload.get("point_partition_sha256")
            != expected_point_sha256):
        raise RuntimeError("LOAO-UQ endpoint payload changed")
    destination = model.state_dict()
    expected_keys = {
        key for key in destination if key.startswith(uq_release.UQ_PREFIXES)}
    if set(state) != expected_keys:
        raise RuntimeError("LOAO-UQ endpoint state-key partition changed")
    for key, value in state.items():
        if destination[key].shape != value.shape:
            raise RuntimeError(f"LOAO-UQ endpoint shape changed at {key}")
        destination[key] = value
    point_before = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    model.load_state_dict(destination, strict=True)
    point_after = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    if point_before != point_after or point_after != expected_point_sha256:
        raise RuntimeError("loading LOAO-UQ endpoint changed point state")
    return payload


def sps_comparison(
        before: Mapping[str, float], after: Mapping[str, float]) -> dict:
    delta = float(after["native_sps_score"] - before["native_sps_score"])
    return {
        "native_before": dict(before),
        "native_after": dict(after),
        "native_sps_before": float(before["native_sps_score"]),
        "native_sps_after": float(after["native_sps_score"]),
        "native_sps_delta": delta,
        "point_metrics_bitwise": True,
    }


def run_fold(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal whole-AoA Native-UQ fold requires CUDA")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 runtime manifest file identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    point_fold.validate_inventory(rows)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")

    cache, residuals, factors, cache_report = full_uq.load_oof_cache(
        args.oof_residual_cache, args.oof_residual_cache_sha256)
    dataset, dataset_report = full_uq.build_dataset(
        rows, args.real_dir, cache, residuals, factors)
    fit_indices, held_indices, partition = partition_indices(
        cache["sample_identities"], args.held_aoa)
    reference_uq_state, reference_uq_report = load_reference_final_uq(
        args.reference_final_point_result,
        args.reference_final_point_result_sha256)
    point_payload, candidate, normalizer, point_report = load_point_fold(
        args.point_fold_result, args.point_fold_result_sha256,
        args.held_aoa, cache, device)
    initialization = full_uq.reset_uq_from_pure_sim(
        candidate, args.uq_init_sim, args.uq_init_sim_sha256,
        point_payload["model_config"])
    if (initialization["point_partition_sha256_after_reset"]
            != point_report["point_partition_sha256"]):
        raise RuntimeError("LOAO fresh-UQ reset changed point fold state")
    training = full_uq.train_cycle(
        candidate, dataset, normalizer, device, 1, deadline,
        indices=fit_indices,
        seed=full_uq.SEED + point_fold.aoa_fold(args.held_aoa) + 1,
        expected_point_partition_sha256=point_report[
            "point_partition_sha256"])
    if training["optimizer_samples"] != partition["fit_samples"]:
        raise RuntimeError("held AoA entered LOAO-UQ optimizer partition")
    endpoint_path = args.output_dir / "fixed_oof_u800_uq_only.pth"
    seal = save_fold_endpoint(
        endpoint_path, candidate, args.held_aoa, point_report, cache_report,
        initialization, training, partition)
    del candidate
    gc.collect()
    torch.cuda.empty_cache()

    # Held truth is used only after the fixed terminal endpoint is immutable.
    # A: original all-81 Real UQ (leaky reference only).
    _payload, reference, _normalizer, reference_point = load_point_fold(
        args.point_fold_result, args.point_fold_result_sha256,
        args.held_aoa, cache, device)
    reference_load = replace_uq_state(
        reference, reference_uq_state,
        point_report["point_partition_sha256"],
        "A_final_point_original_uq_reference")
    reference_metrics, reference_state, reference_target_audit, reference_override = (
        evaluate_oof_native(
            reference, dataset, held_indices, normalizer, device, deadline))
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    # B: genuinely held-independent pure-Sim UQ reset.
    _payload, baseline, _normalizer, baseline_point = load_point_fold(
        args.point_fold_result, args.point_fold_result_sha256,
        args.held_aoa, cache, device)
    baseline_initialization = full_uq.reset_uq_from_pure_sim(
        baseline, args.uq_init_sim, args.uq_init_sim_sha256,
        point_payload["model_config"])
    before_metrics, before_state, before_target_audit, before_override = (
        evaluate_oof_native(
            baseline, dataset, held_indices, normalizer, device, deadline))
    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    _payload, candidate, _normalizer, candidate_point = load_point_fold(
        args.point_fold_result, args.point_fold_result_sha256,
        args.held_aoa, cache, device)
    candidate_initialization = full_uq.reset_uq_from_pure_sim(
        candidate, args.uq_init_sim, args.uq_init_sim_sha256,
        point_payload["model_config"])
    load_fold_endpoint(
        candidate, seal, args.held_aoa,
        point_report["point_partition_sha256"])
    after_metrics, after_state, after_target_audit, after_override = (
        evaluate_oof_native(
            candidate, dataset, held_indices, normalizer, device, deadline))
    if (reference_point["point_partition_sha256"]
            != baseline_point["point_partition_sha256"]
            or baseline_point["point_partition_sha256"]
            != candidate_point["point_partition_sha256"]
            or baseline_initialization["uq_sha256_after_reset"]
            != candidate_initialization["uq_sha256_after_reset"]
            or reference_target_audit != before_target_audit
            or before_target_audit != after_target_audit
            or not uq_release.point_state_equal(reference_state, before_state)
            or not uq_release.point_state_equal(before_state, after_state)):
        raise RuntimeError("LOAO-UQ paired held evaluation contract failed")
    comparison = sps_comparison(before_metrics, after_metrics)
    comparisons = {
        "primary_c_minus_b": comparison,
        "reference_c_minus_a": sps_comparison(
            reference_metrics, after_metrics),
        "reset_b_minus_a": sps_comparison(
            reference_metrics, before_metrics),
    }
    result = {
        "schema": FOLD_SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": args.held_aoa,
        "held_fold": point_fold.aoa_fold(args.held_aoa),
        "manifest": manifest_report,
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "point": point_report,
        "oof_residual_cache": cache_report,
        "dataset": dataset_report,
        "partition": partition,
        "uq_initialization": initialization,
        "reference_final_uq": reference_uq_report,
        "training": training,
        "endpoint": seal,
        "held_evaluation": {
            "arms": {
                "A_final_point_original_uq_reference": {
                    "metrics": reference_metrics,
                    "metric_state": reference_state,
                    "uq_load": reference_load,
                    "oof_override_audit": reference_override,
                    "real_in_sample_uq_reference_only": True,
                },
                "B_pure_sim_reset_native": {
                    "metrics": before_metrics,
                    "metric_state": before_state,
                    "uq_initialization": baseline_initialization,
                    "oof_override_audit": before_override,
                },
                "C_loao_u800": {
                    "metrics": after_metrics,
                    "metric_state": after_state,
                    "oof_override_audit": after_override,
                },
            },
            "native_before": {
                "metrics": before_metrics, "metric_state": before_state},
            "native_after": {
                "metrics": after_metrics, "metric_state": after_state},
            "target_audit": after_target_audit,
            "endpoint_was_sealed_before_held_scoring": True,
        },
        "comparison": comparison,
        "comparisons": comparisons,
        "point_state_bitwise": True,
        "held_optimizer_samples": 0,
        "held_checkpoint_selection": False,
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
        "decision": "COMPLETE_FIXED_U800_HELD_AOA_SCORING",
    }
    result = point_fold.seal_json_payload(result)
    result_path = args.output_dir / "fold_result.json"
    dr1.write_json_atomic(result_path, result)
    point_fold.verify_json_payload(json.loads(
        result_path.read_text(encoding="utf-8")))
    del candidate, dataset, residuals, factors, cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def aggregate_results(pairs: Sequence[Sequence[str]]) -> dict:
    if len(pairs) != FOLDS:
        raise ValueError("LOAO-UQ aggregate requires exactly five fold results")
    rows, sources = [], []
    for raw_path, raw_sha in pairs:
        path = Path(raw_path)
        expected_sha = dr1.normalized_sha256(raw_sha, "LOAO-UQ fold result SHA")
        if not path.is_file() or dr1.sha256(path) != expected_sha:
            raise RuntimeError("LOAO-UQ fold result SHA changed")
        row = json.loads(path.read_text(encoding="utf-8"))
        point_fold.verify_json_payload(row)
        if (row.get("schema") != FOLD_SCHEMA
                or row.get("experiment") != EXPERIMENT
                or int(row.get("training", {}).get("updates", -1)) != UPDATES
                or row.get("endpoint", {}).get("endpoint") != "fixed_oof_u800"
                or row.get("held_evaluation", {}).get(
                    "endpoint_was_sealed_before_held_scoring") is not True
                or int(row.get("held_optimizer_samples", -1)) != 0
                or row.get("point_state_bitwise") is not True
                or row.get("held_checkpoint_selection") is not False
                or row.get("clean716_manifest_read") is not False
                or row.get("uq_initialization", {}).get(
                    "full_real_uq_state_inherited") is not False
                or row.get("reference_final_uq", {}).get(
                    "real_in_sample_initialization") is not True
                or row.get("reference_final_uq", {}).get(
                    "reference_only") is not True
                or set(row.get("held_evaluation", {}).get("arms", {})) != {
                    "A_final_point_original_uq_reference",
                    "B_pure_sim_reset_native", "C_loao_u800"}):
            raise RuntimeError("LOAO-UQ fold result contract changed")
        rows.append(row)
        sources.append({"path": str(path.resolve()), "sha256": expected_sha})
    rows.sort(key=lambda row: int(row["held_aoa"]))
    held_aoas = [int(row["held_aoa"]) for row in rows]
    if held_aoas != list(point_fold.AOA_VALUES):
        raise RuntimeError("LOAO-UQ aggregate does not cover each AoA once")
    cache_shas = {
        row["oof_residual_cache"]["file_sha256"] for row in rows}
    init_shas = {
        row["uq_initialization"]["source_file_sha256"] for row in rows}
    reference_result_shas = {
        row["reference_final_uq"]["result_sha256"] for row in rows}
    reference_uq_shas = {
        row["reference_final_uq"]["uq_state_sha256"] for row in rows}
    if (len(cache_shas) != 1 or len(init_shas) != 1
            or len(reference_result_shas) != 1
            or len(reference_uq_shas) != 1):
        raise RuntimeError("LOAO-UQ folds used different cache/init sources")
    reference_state = smu1.merge_metric_states([
        row["held_evaluation"]["arms"][
            "A_final_point_original_uq_reference"]["metric_state"]
        for row in rows])
    before_state = smu1.merge_metric_states([
        row["held_evaluation"]["arms"][
            "B_pure_sim_reset_native"]["metric_state"]
        for row in rows])
    after_state = smu1.merge_metric_states([
        row["held_evaluation"]["arms"]["C_loao_u800"]["metric_state"]
        for row in rows])
    if (not uq_release.point_state_equal(reference_state, before_state)
            or not uq_release.point_state_equal(before_state, after_state)):
        raise RuntimeError("pooled LOAO-UQ point metric state changed")
    reference_metrics = smu1.finalize_metric_state(reference_state)
    before_metrics = smu1.finalize_metric_state(before_state)
    after_metrics = smu1.finalize_metric_state(after_state)
    comparison = sps_comparison(before_metrics, after_metrics)
    reference_comparison = sps_comparison(reference_metrics, after_metrics)
    reset_comparison = sps_comparison(reference_metrics, before_metrics)
    per_fold = [{
        "held_aoa": int(row["held_aoa"]),
        "held_samples": int(row["partition"]["held_samples"]),
        "A_final_point_original_uq_reference": float(
            row["held_evaluation"]["arms"][
                "A_final_point_original_uq_reference"]["metrics"][
                    "native_sps_score"]),
        "B_pure_sim_reset_native": float(
            row["held_evaluation"]["arms"][
                "B_pure_sim_reset_native"]["metrics"]["native_sps_score"]),
        "C_loao_u800": float(
            row["held_evaluation"]["arms"][
                "C_loao_u800"]["metrics"]["native_sps_score"]),
        "native_sps_before": float(
            row["comparison"]["native_sps_before"]),
        "native_sps_after": float(row["comparison"]["native_sps_after"]),
        "native_sps_delta": float(row["comparison"]["native_sps_delta"]),
    } for row in rows]
    return {
        "schema": AGGREGATE_SCHEMA,
        "experiment": EXPERIMENT,
        "folds": FOLDS,
        "held_aoas": held_aoas,
        "sources": sources,
        "oof_residual_cache": {"file_sha256": next(iter(cache_shas))},
        "uq_initialization": {
            "policy": "reset_all_existing_uq_only_tensors_from_pure_sim10k",
            "source_file_sha256": next(iter(init_shas)),
            "full_real_uq_state_inherited": False,
        },
        "reference_final_uq": {
            "result_sha256": next(iter(reference_result_shas)),
            "uq_state_sha256": next(iter(reference_uq_shas)),
            "real_in_sample_initialization": True,
            "reference_only": True,
        },
        "updates_per_fold": UPDATES,
        "total_crossfit_updates": FOLDS * UPDATES,
        "fixed_terminal_only": True,
        "per_fold_native_sps": per_fold,
        "pooled": {
            **comparison,
            "arms": {
                "A_final_point_original_uq_reference": reference_metrics,
                "B_pure_sim_reset_native": before_metrics,
                "C_loao_u800": after_metrics,
            },
            "comparisons": {
                "primary_c_minus_b": comparison,
                "reference_c_minus_a": reference_comparison,
                "reset_b_minus_a": reset_comparison,
            },
        },
        "positive_sps_folds": sum(
            row["native_sps_delta"] > 0.0 for row in per_fold),
        "worst_fold_native_sps_delta": min(
            row["native_sps_delta"] for row in per_fold),
        "point_metrics_bitwise": True,
        "automatic_promotion_threshold": None,
        "checkpoint_selection": False,
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
        "submission_authorized": False,
        "decision": "COMPLETE_LOAO_EVIDENCE_NO_AUTOMATIC_GATE",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fold-worker", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    mode.add_argument("--contract-only", action="store_true")
    parser.add_argument("--held-aoa", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--point-fold-result", type=Path)
    parser.add_argument("--point-fold-result-sha256")
    parser.add_argument("--reference-final-point-result", type=Path)
    parser.add_argument("--reference-final-point-result-sha256")
    parser.add_argument("--oof-residual-cache", type=Path)
    parser.add_argument("--oof-residual-cache-sha256")
    parser.add_argument("--uq-init-sim", type=Path)
    parser.add_argument("--uq-init-sim-sha256")
    parser.add_argument(
        "--fold-result", nargs=2, action="append", metavar=("PATH", "SHA256"))
    parser.add_argument("--aggregate-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 60 * 60)
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    fold_values = (
        args.held_aoa, args.output_dir, args.real_dir, args.clean_manifest,
        args.point_fold_result, args.point_fold_result_sha256,
        args.reference_final_point_result,
        args.reference_final_point_result_sha256,
        args.oof_residual_cache, args.oof_residual_cache_sha256,
        args.uq_init_sim, args.uq_init_sim_sha256)
    if args.contract_only:
        if (any(value is not None for value in fold_values)
                or args.fold_result or args.aggregate_output is not None
                or args.execute_formal_gpu or args.acknowledge_protocol):
            raise ValueError("contract-only mode received execution inputs")
        return
    if args.aggregate_only:
        if (not args.fold_result or args.aggregate_output is None):
            raise ValueError("LOAO-UQ aggregate requires fold results/output")
        if (any(value is not None for value in fold_values)
                or args.execute_formal_gpu or args.acknowledge_protocol):
            raise ValueError("LOAO-UQ aggregate received fold-only inputs")
        if args.aggregate_output.exists():
            raise FileExistsError(args.aggregate_output)
        return
    if any(value is None for value in fold_values):
        raise ValueError("LOAO-UQ fold worker is missing required input")
    point_fold.aoa_fold(args.held_aoa)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fold_result or args.aggregate_output is not None:
        raise ValueError("LOAO-UQ fold worker received aggregate inputs")
    if (not args.execute_formal_gpu
            or args.acknowledge_protocol != ACKNOWLEDGEMENT):
        raise PermissionError("formal LOAO-UQ fold needs acknowledgement")
    args.point_fold_result_sha256 = dr1.normalized_sha256(
        args.point_fold_result_sha256, "point-OOF fold result SHA")
    args.reference_final_point_result_sha256 = dr1.normalized_sha256(
        args.reference_final_point_result_sha256,
        "reference final point result SHA")
    args.oof_residual_cache_sha256 = dr1.normalized_sha256(
        args.oof_residual_cache_sha256, "OOF residual cache SHA")
    args.uq_init_sim_sha256 = dr1.normalized_sha256(
        args.uq_init_sim_sha256, "pure-Sim10k UQ initialization SHA")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        print(json.dumps({
            "experiment": EXPERIMENT,
            "fold_schema": FOLD_SCHEMA,
            "aggregate_schema": AGGREGATE_SCHEMA,
            "held_aoas": list(point_fold.AOA_VALUES),
            "updates_per_fold": UPDATES,
            "total_crossfit_updates": FOLDS * UPDATES,
            "held_scoring_arms": {
                "A": "final-point original UQ; in-sample reference only",
                "B": "pure-Sim10k reset UQ",
                "C": "other-four-AoA OOF-residual u800 UQ",
            },
            "primary_comparison": "C-B",
            "reference_comparison": "C-A",
            "held_sps_source": "point-OOF cached residual/factor override",
            "uq_initialization": (
                "all existing UQ-only tensors reset from pure-Sim10k"),
            "held_optimizer_samples": 0,
            "automatic_promotion_threshold": None,
            "clean716_manifest_accepted": False,
        }, indent=2, sort_keys=True))
        return
    if args.aggregate_only:
        result = aggregate_results(args.fold_result)
        result = point_fold.seal_json_payload(result)
        dr1.write_json_atomic(args.aggregate_output, result)
        point_fold.verify_json_payload(json.loads(
            args.aggregate_output.read_text(encoding="utf-8")))
        print(json.dumps({
            "event": "b210_full_real_loao_native_uq_aggregate_complete",
            "pooled": result["pooled"],
            "output": str(args.aggregate_output.resolve()),
        }), flush=True)
        return
    result = run_fold(args)
    print(json.dumps({
        "event": "b210_full_real_loao_native_uq_fold_complete",
        "held_aoa": args.held_aoa,
        "comparison": result["comparison"],
        "output": str((args.output_dir / "fold_result.json").resolve()),
    }), flush=True)


if __name__ == "__main__":
    main()
