#!/usr/bin/env python3
"""Fresh Native-UQ cross-fit/refit on trajectory-disjoint all-AoA folds.

The five point models and residual cache come from the fixed
``(rank(Re)+2*rank(AoA)) % 5`` trajectory plan.  A fold worker resets every
UQ-only tensor from the formal pure-Sim10k checkpoint, trains a fixed 800
updates on the other four folds' point-OOF residuals, seals the endpoint, and
only then scores the held trajectories.  The aggregate is descriptive and
has no promotion threshold.  The refit resets UQ again and runs four fixed
800-update cycles on all 3,341 cross-fit residuals while the final all81 point
partition remains bitwise frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

import b210_same_aoa_trajectory_folds as fold_plan
import build_b210_same_aoa_point_oof_residual_cache as cache_builder
import train_b210_full_real_loao_native_uq as whole_loao
import train_b210_full_real_oof_native_uq as whole_refit
import train_b210_same_aoa_point_oof_fold as point_fold


dr1 = point_fold.dr1
ft12 = point_fold.ft12
smu1 = point_fold.smu1
uq_release = whole_refit.uq_release
uq_train = whole_refit.uq_train

EXPERIMENT = "b210_same_aoa_trajectory_native_uq"
FOLD_SCHEMA = "b210_same_aoa_trajectory_native_uq_fold_v1"
AGGREGATE_SCHEMA = "b210_same_aoa_trajectory_native_uq_aggregate_v1"
FOLD_ENDPOINT_SCHEMA = "b210_same_aoa_trajectory_native_uq_endpoint_v1"
REFIT_SCHEMA = "b210_same_aoa_trajectory_native_uq_refit_v1"
REFIT_RESULT_SCHEMA = "b210_same_aoa_trajectory_native_uq_run_v1"
REFIT_CYCLE_SCHEMA = "b210_same_aoa_trajectory_native_uq_cycle_v1"
ACKNOWLEDGEMENT = "RUN_B210_SAME_AOA_TRAJECTORY_NATIVE_UQ_V1"
FOLD_COMPLETION_EVENT = "b210_same_aoa_native_uq_fold_complete"
AGGREGATE_COMPLETION_EVENT = "b210_same_aoa_native_uq_aggregate_complete"
REFIT_COMPLETION_EVENT = "b210_same_aoa_native_uq_refit_complete"
EXPECTED_SAMPLES = cache_builder.EXPECTED_SAMPLES
UPDATES_PER_FOLD = whole_refit.UPDATES_PER_CYCLE
CYCLES = whole_refit.CYCLES
TOTAL_UPDATES = CYCLES * UPDATES_PER_FOLD
FINAL_ENDPOINT = "fixed_same_aoa_oof_u3200"


def load_final_point_artifact(
        result_path: Path, expected_result_sha256: str,
        device: torch.device):
    """Load the all81 point artifact under the active lineage contract."""
    expected_result_sha256 = dr1.normalized_sha256(
        expected_result_sha256, "final point result SHA")
    formal = point_fold.FORMAL_FINAL_POINT_RESULT_SHA256
    if formal is not None and expected_result_sha256 != formal:
        raise RuntimeError("final point result is outside the formal lineage")
    return whole_refit.load_final_point(
        result_path, expected_result_sha256, device,
        expected_result_schema=point_fold.FINAL_POINT_RESULT_SCHEMA,
        expected_endpoint_schema=point_fold.FINAL_POINT_ENDPOINT_SCHEMA,
        expected_start_sha256=point_fold.START_SHA256)


def load_reference_final_uq(
        result_path: Path, expected_result_sha256: str):
    return whole_loao.load_reference_final_uq(
        result_path, expected_result_sha256,
        final_point_loader=load_final_point_artifact)


def reset_uq_from_lineage_sim(
        model: torch.nn.Module, path: Path, expected_sha256: str,
        expected_model_config: Mapping[str, object]):
    """Reset UQ from the active point lineage's exact Sim endpoint."""
    return whole_refit.reset_uq_from_pure_sim(
        model, path, expected_sha256, expected_model_config,
        formal_start_sha256=point_fold.START_SHA256,
        expected_iteration=point_fold.START_EXPECTED_ITERATION,
        expected_status=point_fold.START_EXPECTED_STATUS,
        source_label=point_fold.UQ_SOURCE_LABEL)


def _lineage_matches(payload: Mapping[str, object]) -> bool:
    observed = payload.get("model_lineage")
    return (observed == point_fold.MODEL_LINEAGE
            or (observed is None and not point_fold.LINEAGE_FIELD_REQUIRED))


def _sha_tensor_bytes(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def build_dataset(
        rows: Sequence[Mapping[str, object]], real_dir: Path,
        cache: Mapping[str, object]) -> tuple[whole_refit.OOFResidualDataset, dict]:
    evaluation_rows = cache_builder.full_evaluation_rows(rows)
    assignments = fold_plan.assignment_map(rows)
    identities = [
        {"file": str(row["file"]), "start": int(start),
         "re": int(row["re"]), "aoa": int(row["aoa"]),
         "fold": assignments[str(row["file"])]}
        for row in evaluation_rows for start in row["val_starts"]]
    if (identities != cache.get("sample_identities")
            or dr1.canonical_sha256(identities)
            != cache.get("sample_identity_sha256")):
        raise RuntimeError("same-AoA dataset order differs from OOF cache")
    trajectories = [
        cache_builder.read_trajectory(real_dir, row) for row in evaluation_rows]
    window_type = ft12.npt1.ensemble_train.common.repro.Window
    windows = [
        window_type(trajectory, int(start))
        for trajectory, row in enumerate(evaluation_rows)
        for start in row["val_starts"]]
    base = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    history_hashes, target_hashes = [], []
    expected_history = cache["ordered_history_sample_sha256"]
    expected_target = cache["ordered_target_sample_sha256"]
    for index in range(len(base)):
        history, target = base[index]
        history_hashes.append(_sha_tensor_bytes(history))
        target_hashes.append(_sha_tensor_bytes(target))
        if (history_hashes[-1] != expected_history[index]
                or target_hashes[-1] != expected_target[index]):
            raise RuntimeError(f"same-AoA cache content changed at {index}")
    if (dr1.canonical_sha256(history_hashes)
            != cache["ordered_history_stream_sha256"]
            or dr1.canonical_sha256(target_hashes)
            != cache["ordered_target_stream_sha256"]):
        raise RuntimeError("same-AoA cache sample stream digest changed")
    residuals = cache["signed_residual_physical_uv"]
    factors = cache["oof_point_factor"]
    return whole_refit.OOFResidualDataset(base, residuals, factors), {
        "trajectories": len(trajectories),
        "windows": len(windows),
        "window_stride": 20,
        "fold_formula": fold_plan.FORMULA,
        "fold_plan_assignment_sha256": fold_plan.EXPECTED_ASSIGNMENT_SHA256,
        "ordered_history_stream_sha256": cache[
            "ordered_history_stream_sha256"],
        "ordered_target_stream_sha256": cache[
            "ordered_target_stream_sha256"],
        "complete_trajectory_arrays_read": True,
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
    }


def partition_indices(
        identities: Sequence[Mapping[str, object]], held_fold: int,
        ) -> tuple[list[int], list[int], dict]:
    if held_fold not in fold_plan.FOLD_VALUES:
        raise ValueError(f"unsupported held fold {held_fold}")
    fit = [index for index, row in enumerate(identities)
           if int(row["fold"]) != held_fold]
    held = [index for index, row in enumerate(identities)
            if int(row["fold"]) == held_fold]
    expected = fold_plan.EXPECTED_FOLDS[held_fold]
    if (len(fit) != expected["fit_oof_windows"]
            or len(held) != expected["held_oof_windows"]
            or sorted(fit + held) != list(range(EXPECTED_SAMPLES))
            or set(fit).intersection(held)):
        raise RuntimeError("same-AoA UQ fold partition changed")
    held_rows = [identities[index] for index in held]
    held_aoa_windows = dict(sorted(
        Counter(int(row["aoa"]) for row in held_rows).items()))
    if set(held_aoa_windows) != set(fold_plan.AOA_VALUES):
        raise RuntimeError("same-AoA UQ held fold lost an AoA")
    return fit, held, {
        "held_fold": held_fold,
        "fit_samples": len(fit),
        "held_samples": len(held),
        "held_aoa_windows": held_aoa_windows,
        "fit_identity_sha256": dr1.canonical_sha256(
            [identities[index] for index in fit]),
        "held_identity_sha256": dr1.canonical_sha256(held_rows),
        "held_optimizer_samples": 0,
    }


def load_point_fold(
        result_path: Path, expected_result_sha256: str, held_fold: int,
        cache: Mapping[str, object], device: torch.device,
        ) -> tuple[dict, torch.nn.Module, object, dict]:
    expected_result_sha256 = dr1.normalized_sha256(
        expected_result_sha256, "same-AoA point fold result SHA")
    if (not result_path.is_file()
            or dr1.sha256(result_path) != expected_result_sha256):
        raise RuntimeError("same-AoA point fold result SHA changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    point_fold.whole_aoa.verify_json_payload(result)
    if (result.get("schema") != point_fold.SCHEMA
            or int(result.get("held_fold", -1)) != held_fold
            or result.get("fold_plan_assignment_sha256")
            != fold_plan.EXPECTED_ASSIGNMENT_SHA256
            or not _lineage_matches(result)
            or result.get("decision")
            != "COMPLETE_SAME_AOA_TRAJECTORY_POINT_OOF_FOLD"
            or result.get("held_data_opened") is not False
            or result.get("clean716_manifest_read") is not False
            or result.get("evaluation_manifest_read") is not False
            or result.get("start", {}).get("checkpoint_sha256")
            != point_fold.START_SHA256):
        raise RuntimeError("same-AoA point fold result contract changed")
    cache_rows = [row for row in cache.get("fold_results", [])
                  if int(row.get("fold", -1)) == held_fold]
    if (len(cache_rows) != 1
            or cache_rows[0].get("result_sha256")
            != expected_result_sha256
            or Path(str(cache_rows[0].get("result", ""))).resolve()
            != result_path.resolve()):
        raise RuntimeError("point fold differs from cache binding")
    seal = result.get("fit", {}).get("seal", {})
    checkpoint = Path(str(seal.get("path", "")))
    verified = point_fold.validate_endpoint(
        checkpoint, seal, held_fold=held_fold)
    if (cache_rows[0].get("checkpoint_sha256") != verified["file_sha256"]
            or cache_rows[0].get("model_state_sha256")
            != verified["model_state_sha256"]):
        raise RuntimeError("point checkpoint differs from cache binding")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    normalizer, normalizer_report = whole_refit.normalizer_from_payload(
        payload, device)
    model = uq_train.load_model(
        payload, normalizer, device, train_projections=True)
    point_sha = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    payload_point_sha = dr1.model_state_sha256({
        key: value for key, value in payload["model_state_dict"].items()
        if not key.startswith(uq_release.UQ_PREFIXES)})
    if point_sha != payload_point_sha:
        raise RuntimeError("same-AoA loaded point partition changed")
    return payload, model, normalizer, {
        "held_fold": held_fold,
        "model_lineage": point_fold.MODEL_LINEAGE,
        "result": str(result_path.resolve()),
        "result_sha256": expected_result_sha256,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": verified["file_sha256"],
        "model_state_sha256": verified["model_state_sha256"],
        "point_partition_sha256": point_sha,
        "normalizer": normalizer_report,
    }


def save_fold_endpoint(
        path: Path, model: torch.nn.Module, held_fold: int,
        point_report: Mapping[str, object], cache_report: Mapping[str, object],
        initialization: Mapping[str, object], training: Mapping[str, object],
        partition: Mapping[str, object]) -> dict:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    uq_state = {key: value.detach().cpu().clone()
                for key, value in uq_release.partition_state(model, uq=True).items()}
    uq_sha = dr1.model_state_sha256(uq_state)
    point_sha = dr1.model_state_sha256(
        uq_release.partition_state(model, uq=False))
    if point_sha != point_report["point_partition_sha256"]:
        raise RuntimeError("same-AoA UQ fold changed point state")
    payload = {
        "schema": FOLD_ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "model_lineage": point_fold.MODEL_LINEAGE,
        "held_fold": held_fold,
        "endpoint": "fixed_oof_u800",
        "iteration": UPDATES_PER_FOLD,
        "uq_state_dict": uq_state,
        "uq_state_sha256": uq_sha,
        "point": dict(point_report),
        "point_partition_sha256": point_sha,
        "oof_residual_cache": dict(cache_report),
        "uq_initialization": dict(initialization),
        "training": dict(training),
        "partition": dict(partition),
        "held_scored_after_seal": False,
        "held_checkpoint_selection": False,
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "uq_state_sha256": uq_sha,
        "point_partition_sha256": point_sha,
        "held_fold": held_fold,
        "endpoint": "fixed_oof_u800",
        "iteration": UPDATES_PER_FOLD,
    }


def load_fold_endpoint(
        model: torch.nn.Module, seal: Mapping[str, object], held_fold: int,
        expected_point_sha256: str) -> dict:
    path = Path(str(seal["path"]))
    if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("same-AoA UQ fold endpoint SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("uq_state_dict")
    if (payload.get("schema") != FOLD_ENDPOINT_SCHEMA
            or int(payload.get("held_fold", -1)) != held_fold
            or payload.get("endpoint") != "fixed_oof_u800"
            or int(payload.get("iteration", -1)) != UPDATES_PER_FOLD
            or payload.get("point_partition_sha256") != expected_point_sha256
            or not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal["uq_state_sha256"]):
        raise RuntimeError("same-AoA UQ fold endpoint contract changed")
    return whole_loao.replace_uq_state(
        model, state, expected_point_sha256, "C_trajectory_crossfit_u800")


def load_inputs(args: argparse.Namespace):
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 runtime manifest file identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    fold_plan.build_plan(rows)
    cache, cache_report = cache_builder.load_cache(
        args.oof_residual_cache, args.oof_residual_cache_sha256)
    dataset, dataset_report = build_dataset(rows, args.real_dir, cache)
    return rows, manifest_report, cache, cache_report, dataset, dataset_report


def run_fold(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal same-AoA Native-UQ fold requires CUDA")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    (_rows, manifest_report, cache, cache_report,
     dataset, dataset_report) = load_inputs(args)
    fit_indices, held_indices, partition = partition_indices(
        cache["sample_identities"], args.held_fold)
    reference_uq, reference_report = load_reference_final_uq(
        args.reference_final_point_result,
        args.reference_final_point_result_sha256)
    point_payload, candidate, normalizer, point_report = load_point_fold(
        args.point_fold_result, args.point_fold_result_sha256,
        args.held_fold, cache, device)
    initialization = reset_uq_from_lineage_sim(
        candidate, args.uq_init_sim, args.uq_init_sim_sha256,
        point_payload["model_config"])
    if (initialization["point_partition_sha256_after_reset"]
            != point_report["point_partition_sha256"]):
        raise RuntimeError("same-AoA fresh-UQ reset changed point state")
    training = whole_refit.train_cycle(
        candidate, dataset, normalizer, device, 1, deadline,
        indices=fit_indices, seed=whole_refit.SEED + args.held_fold + 1,
        expected_point_partition_sha256=point_report[
            "point_partition_sha256"])
    training["supervision"] = (
        "other-four trajectory folds' point-OOF signed residual/factor")
    if training["optimizer_samples"] != partition["fit_samples"]:
        raise RuntimeError("held trajectory fold entered UQ optimizer")
    seal = save_fold_endpoint(
        args.output_dir / "fixed_oof_u800_uq_only.pth", candidate,
        args.held_fold, point_report, cache_report, initialization,
        training, partition)
    del candidate
    gc.collect()
    torch.cuda.empty_cache()

    def fresh_point_model():
        return load_point_fold(
            args.point_fold_result, args.point_fold_result_sha256,
            args.held_fold, cache, device)

    _payload, reference, _norm, ref_point = fresh_point_model()
    reference_load = whole_loao.replace_uq_state(
        reference, reference_uq, point_report["point_partition_sha256"],
        "A_final_point_original_uq_reference")
    ref_metrics, ref_state, ref_target, ref_override = (
        whole_loao.evaluate_oof_native(
            reference, dataset, held_indices, normalizer, device, deadline))
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    _payload, baseline, _norm, base_point = fresh_point_model()
    baseline_init = reset_uq_from_lineage_sim(
        baseline, args.uq_init_sim, args.uq_init_sim_sha256,
        point_payload["model_config"])
    before_metrics, before_state, before_target, before_override = (
        whole_loao.evaluate_oof_native(
            baseline, dataset, held_indices, normalizer, device, deadline))
    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    _payload, candidate, _norm, cand_point = fresh_point_model()
    candidate_init = reset_uq_from_lineage_sim(
        candidate, args.uq_init_sim, args.uq_init_sim_sha256,
        point_payload["model_config"])
    candidate_load = load_fold_endpoint(
        candidate, seal, args.held_fold,
        point_report["point_partition_sha256"])
    after_metrics, after_state, after_target, after_override = (
        whole_loao.evaluate_oof_native(
            candidate, dataset, held_indices, normalizer, device, deadline))
    if (ref_point["point_partition_sha256"]
            != base_point["point_partition_sha256"]
            or base_point["point_partition_sha256"]
            != cand_point["point_partition_sha256"]
            or baseline_init["uq_sha256_after_reset"]
            != candidate_init["uq_sha256_after_reset"]
            or ref_target != before_target or before_target != after_target
            or not uq_release.point_state_equal(ref_state, before_state)
            or not uq_release.point_state_equal(before_state, after_state)):
        raise RuntimeError("same-AoA paired UQ evaluation contract failed")
    comparisons = {
        "primary_c_minus_b": whole_loao.sps_comparison(
            before_metrics, after_metrics),
        "reference_c_minus_a": whole_loao.sps_comparison(
            ref_metrics, after_metrics),
        "reset_b_minus_a": whole_loao.sps_comparison(
            ref_metrics, before_metrics),
    }
    result = {
        "schema": FOLD_SCHEMA,
        "experiment": EXPERIMENT,
        "model_lineage": point_fold.MODEL_LINEAGE,
        "held_fold": args.held_fold,
        "folds": fold_plan.FOLDS,
        "fold_formula": fold_plan.FORMULA,
        "fold_plan_assignment_sha256": fold_plan.EXPECTED_ASSIGNMENT_SHA256,
        "manifest": manifest_report,
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "point": point_report,
        "oof_residual_cache": cache_report,
        "dataset": dataset_report,
        "partition": partition,
        "uq_initialization": initialization,
        "reference_final_uq": reference_report,
        "training": training,
        "endpoint": seal,
        "held_evaluation": {
            "arms": {
                "A_final_point_original_uq_reference": {
                    "metrics": ref_metrics, "metric_state": ref_state,
                    "uq_load": reference_load,
                    "oof_override_audit": ref_override,
                    "real_in_sample_uq_reference_only": True,
                },
                "B_pure_sim_reset_native": {
                    "metrics": before_metrics, "metric_state": before_state,
                    "uq_initialization": baseline_init,
                    "oof_override_audit": before_override,
                },
                "C_trajectory_crossfit_u800": {
                    "metrics": after_metrics, "metric_state": after_state,
                    "uq_load": candidate_load,
                    "oof_override_audit": after_override,
                },
            },
            "target_audit": after_target,
            "endpoint_was_sealed_before_held_scoring": True,
        },
        "comparison": comparisons["primary_c_minus_b"],
        "comparisons": comparisons,
        "point_state_bitwise": True,
        "held_optimizer_samples": 0,
        "held_checkpoint_selection": False,
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
        "decision": "COMPLETE_FIXED_U800_HELD_TRAJECTORY_FOLD_SCORING",
    }
    result = point_fold.whole_aoa.seal_json_payload(result)
    result_path = args.output_dir / "fold_result.json"
    dr1.write_json_atomic(result_path, result)
    point_fold.whole_aoa.verify_json_payload(json.loads(
        result_path.read_text(encoding="utf-8")))
    del candidate, dataset, cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def aggregate_results(pairs: Sequence[Sequence[str]]) -> dict:
    if len(pairs) != fold_plan.FOLDS:
        raise ValueError("same-AoA aggregate requires exactly five fold results")
    rows, sources = [], []
    for raw_path, raw_sha in pairs:
        path = Path(raw_path)
        expected_sha = dr1.normalized_sha256(raw_sha, "UQ fold result SHA")
        if not path.is_file() or dr1.sha256(path) != expected_sha:
            raise RuntimeError("same-AoA UQ fold result SHA changed")
        row = json.loads(path.read_text(encoding="utf-8"))
        point_fold.whole_aoa.verify_json_payload(row)
        if (row.get("schema") != FOLD_SCHEMA
                or row.get("experiment") != EXPERIMENT
                or not _lineage_matches(row)
                or int(row.get("training", {}).get("updates", -1))
                != UPDATES_PER_FOLD
                or row.get("endpoint", {}).get("endpoint")
                != "fixed_oof_u800"
                or row.get("held_evaluation", {}).get(
                    "endpoint_was_sealed_before_held_scoring") is not True
                or int(row.get("held_optimizer_samples", -1)) != 0
                or row.get("point_state_bitwise") is not True
                or row.get("held_checkpoint_selection") is not False
                or row.get("clean716_manifest_read") is not False
                or row.get("evaluation_manifest_read") is not False
                or row.get("uq_initialization", {}).get(
                    "full_real_uq_state_inherited") is not False
                or set(row.get("held_evaluation", {}).get("arms", {})) != {
                    "A_final_point_original_uq_reference",
                    "B_pure_sim_reset_native",
                    "C_trajectory_crossfit_u800"}):
            raise RuntimeError("same-AoA UQ fold result contract changed")
        rows.append(row)
        sources.append({"path": str(path.resolve()), "sha256": expected_sha})
    rows.sort(key=lambda row: int(row["held_fold"]))
    held_folds = [int(row["held_fold"]) for row in rows]
    if held_folds != list(fold_plan.FOLD_VALUES):
        raise RuntimeError("same-AoA aggregate does not cover all folds")
    cache_shas = {row["oof_residual_cache"]["file_sha256"] for row in rows}
    init_shas = {row["uq_initialization"]["source_file_sha256"] for row in rows}
    init_policies = {row["uq_initialization"]["policy"] for row in rows}
    reference_shas = {row["reference_final_uq"]["result_sha256"] for row in rows}
    reference_uq_shas = {row["reference_final_uq"]["uq_state_sha256"]
                         for row in rows}
    if any(len(values) != 1 for values in (
            cache_shas, init_shas, init_policies,
            reference_shas, reference_uq_shas)):
        raise RuntimeError("same-AoA UQ folds used different sources")
    arm_names = (
        "A_final_point_original_uq_reference",
        "B_pure_sim_reset_native", "C_trajectory_crossfit_u800")
    states = {
        arm: smu1.merge_metric_states([
            row["held_evaluation"]["arms"][arm]["metric_state"]
            for row in rows])
        for arm in arm_names}
    if (not uq_release.point_state_equal(states[arm_names[0]], states[arm_names[1]])
            or not uq_release.point_state_equal(
                states[arm_names[1]], states[arm_names[2]])):
        raise RuntimeError("same-AoA pooled point metric state changed")
    metrics = {arm: smu1.finalize_metric_state(state)
               for arm, state in states.items()}
    primary = whole_loao.sps_comparison(
        metrics[arm_names[1]], metrics[arm_names[2]])
    reference = whole_loao.sps_comparison(
        metrics[arm_names[0]], metrics[arm_names[2]])
    reset = whole_loao.sps_comparison(
        metrics[arm_names[0]], metrics[arm_names[1]])
    per_fold = [{
        "held_fold": int(row["held_fold"]),
        "held_samples": int(row["partition"]["held_samples"]),
        "held_aoa_windows": row["partition"]["held_aoa_windows"],
        **{arm: float(row["held_evaluation"]["arms"][arm]["metrics"][
            "native_sps_score"]) for arm in arm_names},
        "native_sps_delta_c_minus_b": float(
            row["comparisons"]["primary_c_minus_b"]["native_sps_delta"]),
        "native_sps_delta_c_minus_a": float(
            row["comparisons"]["reference_c_minus_a"]["native_sps_delta"]),
    } for row in rows]
    return {
        "schema": AGGREGATE_SCHEMA,
        "experiment": EXPERIMENT,
        "model_lineage": point_fold.MODEL_LINEAGE,
        "folds": fold_plan.FOLDS,
        "held_folds": held_folds,
        "fold_formula": fold_plan.FORMULA,
        "fold_plan_assignment_sha256": fold_plan.EXPECTED_ASSIGNMENT_SHA256,
        "sources": sources,
        "oof_residual_cache": {"file_sha256": next(iter(cache_shas))},
        "uq_initialization": {
            "policy": next(iter(init_policies)),
            "source_file_sha256": next(iter(init_shas)),
            "full_real_uq_state_inherited": False,
        },
        "reference_final_uq": {
            "result_sha256": next(iter(reference_shas)),
            "uq_state_sha256": next(iter(reference_uq_shas)),
            "real_in_sample_initialization": True,
            "reference_only": True,
        },
        "updates_per_fold": UPDATES_PER_FOLD,
        "total_crossfit_updates": fold_plan.FOLDS * UPDATES_PER_FOLD,
        "fixed_terminal_only": True,
        "per_fold_native_sps": per_fold,
        "pooled": {
            **primary,
            "arms": metrics,
            "comparisons": {
                "primary_c_minus_b": primary,
                "reference_c_minus_a": reference,
                "reset_b_minus_a": reset,
            },
        },
        "point_metrics_bitwise": True,
        "automatic_promotion_threshold": None,
        "checkpoint_selection": False,
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
        "submission_authorized": False,
        "decision": "COMPLETE_TRAJECTORY_CROSSFIT_EVIDENCE_NO_AUTOMATIC_GATE",
    }


def load_aggregate(path: Path, expected_sha256: str,
                   cache_report: Mapping[str, object],
                   initialization: Mapping[str, object]) -> dict:
    expected_sha256 = dr1.normalized_sha256(
        expected_sha256, "same-AoA UQ aggregate SHA")
    if not path.is_file() or dr1.sha256(path) != expected_sha256:
        raise RuntimeError("same-AoA UQ aggregate SHA changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    point_fold.whole_aoa.verify_json_payload(payload)
    if (payload.get("schema") != AGGREGATE_SCHEMA
            or not _lineage_matches(payload)
            or payload.get("held_folds") != list(fold_plan.FOLD_VALUES)
            or int(payload.get("folds", -1)) != fold_plan.FOLDS
            or int(payload.get("updates_per_fold", -1)) != UPDATES_PER_FOLD
            or payload.get("fixed_terminal_only") is not True
            or payload.get("checkpoint_selection") is not False
            or payload.get("clean716_manifest_read") is not False
            or payload.get("oof_residual_cache", {}).get("file_sha256")
            != cache_report["file_sha256"]
            or payload.get("uq_initialization", {}).get("source_file_sha256")
            != initialization["source_file_sha256"]):
        raise RuntimeError("same-AoA UQ aggregate contract changed")
    return {
        "path": str(path.resolve()),
        "file_sha256": expected_sha256,
        "schema": AGGREGATE_SCHEMA,
        "held_folds": list(fold_plan.FOLD_VALUES),
        "folds": fold_plan.FOLDS,
        "updates_per_fold": UPDATES_PER_FOLD,
        "pooled": payload["pooled"],
        "per_fold_native_sps": payload["per_fold_native_sps"],
        "fixed_terminal_only": True,
        "checkpoint_selection": False,
    }


def save_cycle(path: Path, model: torch.nn.Module, cycle: int,
               point_sha: str, training: Mapping[str, object]) -> dict:
    state = {key: value.detach().cpu().clone()
             for key, value in uq_release.partition_state(model, uq=True).items()}
    payload = {
        "schema": REFIT_CYCLE_SCHEMA,
        "experiment": EXPERIMENT,
        "model_lineage": point_fold.MODEL_LINEAGE,
        "cycle": cycle,
        "iteration": cycle * UPDATES_PER_FOLD,
        "endpoint": f"fixed_same_aoa_oof_u{cycle * UPDATES_PER_FOLD}",
        "uq_state_dict": state,
        "uq_state_sha256": dr1.model_state_sha256(state),
        "point_partition_sha256": point_sha,
        "training": dict(training),
        "checkpoint_selection": False,
        "clean716_manifest_read": False,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()), "file_sha256": dr1.sha256(path),
        "uq_state_sha256": payload["uq_state_sha256"],
        "cycle": cycle, "iteration": payload["iteration"],
        "endpoint": payload["endpoint"],
    }


def save_final(
        path: Path, model: torch.nn.Module, point_payload: Mapping[str, object],
        point_report: Mapping[str, object], cache_report: Mapping[str, object],
        dataset_report: Mapping[str, object], initialization: Mapping[str, object],
        validation: Mapping[str, object], cycles: Sequence[Mapping[str, object]],
        seals: Sequence[Mapping[str, object]]) -> dict:
    state = {key: value.detach().cpu().clone()
             for key, value in model.state_dict().items()}
    state_sha = dr1.model_state_sha256(state)
    point_sha = dr1.model_state_sha256({
        key: value for key, value in state.items()
        if not key.startswith(uq_release.UQ_PREFIXES)})
    if point_sha != point_report["point_partition_sha256"]:
        raise RuntimeError("same-AoA final refit changed point state")
    output = {
        "schema": REFIT_SCHEMA,
        "experiment": EXPERIMENT,
        "model_lineage": point_fold.MODEL_LINEAGE,
        "phase": "real",
        "iteration": TOTAL_UPDATES,
        "endpoint": FINAL_ENDPOINT,
        "primary": True,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "normalizer": dict(point_report["normalizer"]["values"]),
        "normalizer_sha256": point_report["normalizer"]["sha256"],
        "point_checkpoint": dict(point_report),
        "point_partition_sha256": point_sha,
        "point_state_bitwise": True,
        "uq_initialization": dict(initialization),
        "trajectory_crossfit_validation": dict(validation),
        "oof_residual_cache": dict(cache_report),
        "dataset": dict(dataset_report),
        "training": {
            "cycles": [dict(row) for row in cycles],
            "cycle_seals": [dict(row) for row in seals],
            "cycles_fixed_before_training": CYCLES,
            "updates_per_cycle": UPDATES_PER_FOLD,
            "total_updates": TOTAL_UPDATES,
            "checkpoint_selection": False,
        },
        "source_point_model_config_sha256": dr1.canonical_sha256(
            point_payload["model_config"]),
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
        "checkpoint_selection": False,
        "optimizer_state_saved": False,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    torch.save(output, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()), "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "point_partition_sha256": point_sha,
        "schema": REFIT_SCHEMA, "endpoint": FINAL_ENDPOINT,
        "iteration": TOTAL_UPDATES,
    }


def run_refit(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal same-AoA Native-UQ refit requires CUDA")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    (_rows, manifest_report, cache, cache_report,
     dataset, dataset_report) = load_inputs(args)
    point_payload, model, normalizer, point_report = load_final_point_artifact(
        args.point_result, args.point_result_sha256, device)
    initialization = reset_uq_from_lineage_sim(
        model, args.uq_init_sim, args.uq_init_sim_sha256,
        point_payload["model_config"])
    if (initialization["point_partition_sha256_after_reset"]
            != point_report["point_partition_sha256"]):
        raise RuntimeError("same-AoA refit UQ reset changed point state")
    validation = load_aggregate(
        args.uq_aggregate, args.uq_aggregate_sha256,
        cache_report, initialization)
    cycle_reports, cycle_seals = [], []
    for cycle in range(1, CYCLES + 1):
        report = whole_refit.train_cycle(
            model, dataset, normalizer, device, cycle, deadline,
            expected_point_partition_sha256=point_report[
                "point_partition_sha256"])
        report["supervision"] = (
            "all3341 trajectory-cross-fit signed residual/factor")
        seal = save_cycle(
            args.output_dir / f"native_uq_u{cycle * UPDATES_PER_FOLD}_uq_only.pth",
            model, cycle, point_report["point_partition_sha256"], report)
        cycle_reports.append(report)
        cycle_seals.append(seal)
        print(json.dumps({
            "event": "b210_same_aoa_native_uq_cycle_sealed",
            "cycle": cycle, "seal": seal}), flush=True)
    endpoint_path = args.output_dir / "same_aoa_oof_native_uq_u3200.pth"
    seal = save_final(
        endpoint_path, model, point_payload, point_report, cache_report,
        dataset_report, initialization, validation, cycle_reports, cycle_seals)
    verified_payload = torch.load(
        endpoint_path, map_location="cpu", weights_only=False)
    if (dr1.sha256(endpoint_path) != seal["file_sha256"]
            or dr1.model_state_sha256(verified_payload["model_state_dict"])
            != seal["model_state_sha256"]
            or verified_payload.get("point_state_bitwise") is not True
            or verified_payload.get("checkpoint_selection") is not False):
        raise RuntimeError("same-AoA final endpoint verification failed")
    result = {
        "schema": REFIT_RESULT_SCHEMA,
        "experiment": EXPERIMENT,
        "model_lineage": point_fold.MODEL_LINEAGE,
        "manifest": manifest_report,
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "point": point_report,
        "uq_initialization": initialization,
        "trajectory_crossfit_validation": validation,
        "oof_residual_cache": cache_report,
        "dataset": dataset_report,
        "fixed_cycles": CYCLES,
        "updates_per_cycle": UPDATES_PER_FOLD,
        "total_updates": TOTAL_UPDATES,
        "cycle_reports": cycle_reports,
        "cycle_seals": cycle_seals,
        "primary_endpoint": FINAL_ENDPOINT,
        "endpoint": seal,
        "endpoint_verification": {"exact": True, **seal},
        "point_state_bitwise": True,
        "checkpoint_selection": False,
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
        "decision": "COMPLETE_FIXED_SAME_AOA_OOF_NATIVE_UQ_U3200",
    }
    result = point_fold.whole_aoa.seal_json_payload(result)
    result_path = args.output_dir / "result.json"
    dr1.write_json_atomic(result_path, result)
    point_fold.whole_aoa.verify_json_payload(json.loads(
        result_path.read_text(encoding="utf-8")))
    del model, dataset, cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fold-worker", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    mode.add_argument("--run-refit", action="store_true")
    mode.add_argument("--contract-only", action="store_true")
    parser.add_argument("--held-fold", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--aggregate-output", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--point-fold-result", type=Path)
    parser.add_argument("--point-fold-result-sha256")
    parser.add_argument("--reference-final-point-result", type=Path)
    parser.add_argument("--reference-final-point-result-sha256")
    parser.add_argument("--point-result", type=Path)
    parser.add_argument("--point-result-sha256")
    parser.add_argument("--uq-init-sim", type=Path)
    parser.add_argument("--uq-init-sim-sha256")
    parser.add_argument("--oof-residual-cache", type=Path)
    parser.add_argument("--oof-residual-cache-sha256")
    parser.add_argument("--fold-result", nargs=2, action="append",
                        metavar=("PATH", "SHA256"))
    parser.add_argument("--uq-aggregate", type=Path)
    parser.add_argument("--uq-aggregate-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 60 * 60)
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def _require(values: Sequence[object], label: str) -> None:
    if any(value is None for value in values):
        raise ValueError(f"{label} is missing required input")


def validate_args(args: argparse.Namespace) -> None:
    if args.contract_only:
        return
    if args.aggregate:
        _require((args.fold_result, args.aggregate_output), "aggregate")
        if args.aggregate_output.exists():
            raise FileExistsError(args.aggregate_output)
        return
    common = (args.output_dir, args.real_dir, args.clean_manifest,
              args.uq_init_sim, args.uq_init_sim_sha256,
              args.oof_residual_cache, args.oof_residual_cache_sha256)
    _require(common, "Native-UQ execution")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (not args.execute_formal_gpu
            or args.acknowledge_protocol != ACKNOWLEDGEMENT):
        raise PermissionError("formal same-AoA Native-UQ needs acknowledgement")
    args.uq_init_sim_sha256 = dr1.normalized_sha256(
        args.uq_init_sim_sha256, "Sim-lineage UQ source SHA")
    args.oof_residual_cache_sha256 = dr1.normalized_sha256(
        args.oof_residual_cache_sha256, "same-AoA OOF cache SHA")
    if args.fold_worker:
        _require((args.held_fold, args.point_fold_result,
                  args.point_fold_result_sha256,
                  args.reference_final_point_result,
                  args.reference_final_point_result_sha256), "fold worker")
        if args.held_fold not in fold_plan.FOLD_VALUES:
            raise ValueError(f"unsupported held fold {args.held_fold}")
        args.point_fold_result_sha256 = dr1.normalized_sha256(
            args.point_fold_result_sha256, "point fold result SHA")
        args.reference_final_point_result_sha256 = dr1.normalized_sha256(
            args.reference_final_point_result_sha256,
            "reference final point result SHA")
    else:
        _require((args.point_result, args.point_result_sha256,
                  args.uq_aggregate, args.uq_aggregate_sha256), "refit")
        args.point_result_sha256 = dr1.normalized_sha256(
            args.point_result_sha256, "final point result SHA")
        args.uq_aggregate_sha256 = dr1.normalized_sha256(
            args.uq_aggregate_sha256, "UQ aggregate SHA")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        print(json.dumps({
            "experiment": EXPERIMENT,
            "fold_formula": fold_plan.FORMULA,
            "fold_plan_assignment_sha256": fold_plan.EXPECTED_ASSIGNMENT_SHA256,
            "folds": fold_plan.FOLDS,
            "samples": EXPECTED_SAMPLES,
            "updates_per_fold": UPDATES_PER_FOLD,
            "refit_cycles": CYCLES,
            "refit_total_updates": TOTAL_UPDATES,
            "model_lineage": point_fold.MODEL_LINEAGE,
            "fresh_uq_source": point_fold.UQ_SOURCE_LABEL,
            "fresh_uq_source_sha256": point_fold.START_SHA256,
            "fresh_uq_source_iteration": point_fold.START_EXPECTED_ITERATION,
            "fresh_uq_source_status": point_fold.START_EXPECTED_STATUS,
            "formal_final_point_result_sha256": (
                point_fold.FORMAL_FINAL_POINT_RESULT_SHA256),
            "point_partition_frozen": True,
            "checkpoint_selection": False,
            "clean716_manifest_accepted": False,
            "evaluation_manifest_accepted": False,
        }, indent=2, sort_keys=True))
        return
    if args.aggregate:
        result = point_fold.whole_aoa.seal_json_payload(
            aggregate_results(args.fold_result))
        dr1.write_json_atomic(args.aggregate_output, result)
        point_fold.whole_aoa.verify_json_payload(json.loads(
            args.aggregate_output.read_text(encoding="utf-8")))
        print(json.dumps({
            "event": AGGREGATE_COMPLETION_EVENT,
            "output": str(args.aggregate_output.resolve()),
            "pooled": result["pooled"],
        }), flush=True)
        return
    result = run_fold(args) if args.fold_worker else run_refit(args)
    print(json.dumps({
        "event": (FOLD_COMPLETION_EVENT
                  if args.fold_worker
                  else REFIT_COMPLETION_EVENT),
        "result": str((args.output_dir / "fold_result.json").resolve()
                      if args.fold_worker
                      else (args.output_dir / "result.json").resolve()),
        "decision": result["decision"],
    }), flush=True)


if __name__ == "__main__":
    main()
