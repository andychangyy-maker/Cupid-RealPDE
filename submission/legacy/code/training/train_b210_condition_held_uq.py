#!/usr/bin/env python3
"""Validate frozen-point UQ-only adaptation on whole held Real AoAs.

Each worker reproduces the SHA-bound Dense4k point endpoint using only the
other four AoAs, verifies its model-state digest against the frozen control,
and then trains only the native UQ modules on the same fit-condition sparse
prefix windows.  Both point and fixed-u800 UQ endpoints are sealed before the
held AoA is opened.  Temporal tails are never constructed or read.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

import train_b210_condition_held_real as held
import train_b210_small4k_uq_release as uq_release


dr1 = held.dr1
smu1 = held.smu1
ft12 = held.ft12
dense4k = held.dense4k

EXPERIMENT = "b210_condition_held_uq_only"
FOLD_SCHEMA = "b210_condition_held_uq_only_fold_v1"
AGGREGATE_SCHEMA = "b210_condition_held_uq_only_aggregate_v1"
UQ_ENDPOINT_SCHEMA = "b210_condition_held_uq_only_endpoint_v1"
UQ_ARM = "uqonly_u800"
POINT_ARM = "dense4k"
UQ_GATE = {
    "pooled_sps_gain_min_inclusive": 0.10,
    "minimum_positive_folds": 4,
    "worst_fold_min_inclusive": -0.03,
}
MAX_POINT_REPLAY_CONTRIBUTION_ABS = 0.05


def save_uq_endpoint(
        path: Path, model: torch.nn.Module, held_aoa: int,
        point_state_sha256: str, point_partition_sha256: str,
        training: Mapping[str, object]) -> dict:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(path)
    state = {
        key: value.detach().cpu().clone()
        for key, value in uq_release.partition_state(model, uq=True).items()
    }
    state_sha = dr1.model_state_sha256(state)
    payload = {
        "schema": UQ_ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": held_aoa,
        "endpoint": UQ_ARM,
        "iteration": uq_release.FOLD_UPDATES,
        "point_model_state_sha256": point_state_sha256,
        "point_partition_sha256": point_partition_sha256,
        "uq_state_dict": state,
        "uq_state_sha256": state_sha,
        "training": dict(training),
        "held_checkpoint_selection": False,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "uq_state_sha256": state_sha,
        "endpoint": UQ_ARM,
        "iteration": uq_release.FOLD_UPDATES,
    }


def sps_comparison(
        candidate: Mapping[str, float], control: Mapping[str, float]) -> dict:
    delta = float(
        candidate["native_sps_score"] - control["native_sps_score"])
    return {
        "control": dict(control),
        "candidate": dict(candidate),
        "native_sps_delta": delta,
        "non_time_final_contribution_delta": (
            held.FINAL_WEIGHTS["native_sps_score"] * delta),
        "point_metrics_bitwise": True,
    }


def point_replay_check(
        current: Mapping[str, float], frozen: Mapping[str, float]) -> dict:
    comparison = held.metric_comparison(current, frozen)
    magnitude = abs(float(comparison["point_final_contribution_delta"]))
    return {
        "comparison": comparison,
        "maximum_absolute_point_final_contribution": (
            MAX_POINT_REPLAY_CONTRIBUTION_ABS),
        "absolute_point_final_contribution": magnitude,
        "pass": magnitude <= MAX_POINT_REPLAY_CONTRIBUTION_ABS,
        "native_sps_not_used_for_point_replay_identity": True,
    }


def run_fold(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.held_aoa not in held.AOA_VALUES:
        raise ValueError("held AoA is outside the released conditions")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("condition-held UQ training requires CUDA")
    control_row, control = held.load_frozen_control(
        args.control_fold_result, args.held_aoa)
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, held.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    if manifest_report != control["manifest"]:
        raise RuntimeError("manifest differs from frozen Dense4k control")
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    if not fit_rows or not held_rows:
        raise RuntimeError("condition-held UQ fold has empty fit or held rows")
    start_sha = dr1.sha256(args.start_sim)
    if (args.start_sim_sha256 is not None
            and start_sha != args.start_sim_sha256.lower()):
        raise RuntimeError("condition-held UQ Sim start SHA changed")
    if start_sha != control["start"]["sha256"]:
        raise RuntimeError("UQ candidate does not share control Sim start")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    audit = held.PrefixAccessAudit(args.held_aoa, [POINT_ARM, UQ_ARM])
    fit_trajectories, fit_metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", audit)
    dense_windows = dense4k.dense_windows(
        fit_trajectories, dense4k.CANDIDATE_STRIDE)
    point_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        fit_trajectories, dense_windows)
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, sparse_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    statistics_sha = smu1.statistics_digest(statistics)
    if statistics_sha != control["normalizer"]["statistics_sha256"]:
        raise RuntimeError("UQ normalizer differs from frozen control")
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "fit_normalizer.pt")

    point_report = held.train_arm(
        POINT_ARM, args, audit, point_dataset, dense_windows, fit_metadata,
        normalizer, start_sha, deadline)
    expected_point_sha = control_row["arms"][POINT_ARM]["seal"][
        "model_state_sha256"]
    actual_point_sha = point_report["seal"]["model_state_sha256"]
    point_replay_identity = {
        "expected_model_state_sha256": expected_point_sha,
        "actual_model_state_sha256": actual_point_sha,
        "model_state_bitwise_across_runs": (
            actual_point_sha == expected_point_sha),
        "sample_stream_sha256": point_report["sample_stream_sha256"],
        "sample_stream_matches_frozen_control": (
            point_report["sample_stream_sha256"]
            == control_row["arms"][POINT_ARM]["seal"][
                "sample_stream_sha256"]),
        "cuda_training_declared_bitwise_deterministic": False,
    }
    if not point_replay_identity["sample_stream_matches_frozen_control"]:
        raise RuntimeError("replayed Dense4k sample stream changed")
    candidate = held.load_endpoint(
        point_report["seal"], normalizer, device, args.held_aoa)
    point_partition_sha = dr1.model_state_sha256(
        uq_release.partition_state(candidate, uq=False))
    sparse_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        fit_trajectories, sparse_windows)
    fold_index = held.AOA_VALUES.index(args.held_aoa)
    uq_training = uq_release.train_uq_terminal(
        candidate, sparse_dataset, list(range(len(sparse_windows))),
        normalizer, device, deadline, uq_release.SEED + fold_index + 1,
        f"held_aoa{args.held_aoa}")
    if (uq_training["frozen_state_sha256_before"] != point_partition_sha
            or uq_training["frozen_state_sha256_after"] != point_partition_sha):
        raise RuntimeError("UQ-only stage changed the frozen point state")
    uq_dir = args.output_dir / UQ_ARM
    uq_dir.mkdir()
    uq_seal = save_uq_endpoint(
        uq_dir / "fixed_terminal_u800.pth", candidate, args.held_aoa,
        actual_point_sha, point_partition_sha, uq_training)
    audit.seal(UQ_ARM, uq_seal)

    del point_dataset, sparse_dataset, fit_trajectories
    gc.collect()
    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", audit)
    held_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    baseline = held.load_endpoint(
        point_report["seal"], normalizer, device, args.held_aoa)
    baseline_metrics, baseline_state, baseline_target_audit = smu1.evaluate_model(
        baseline, held_dataset, list(range(len(held_windows))), normalizer,
        device, deadline)
    replay_check = point_replay_check(baseline_metrics, control["metrics"])
    if not replay_check["pass"]:
        raise RuntimeError("replayed Dense4k point quality drift exceeded tolerance")
    del baseline
    gc.collect()
    torch.cuda.empty_cache()
    candidate_metrics, candidate_state, candidate_target_audit = smu1.evaluate_model(
        candidate, held_dataset, list(range(len(held_windows))), normalizer,
        device, deadline)
    if baseline_target_audit != candidate_target_audit:
        raise RuntimeError("held target audit changed during UQ comparison")
    if baseline_target_audit != control["target_audit"]:
        raise RuntimeError("held targets differ from frozen Dense4k control")
    if not uq_release.point_state_equal(baseline_state, candidate_state):
        raise RuntimeError("UQ-only candidate changed point metric state")
    comparison = sps_comparison(candidate_metrics, baseline_metrics)
    del candidate
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "schema": FOLD_SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": args.held_aoa,
        "fit_trajectories": len(fit_rows),
        "held_trajectories": len(held_rows),
        "fit_sparse_windows": len(sparse_windows),
        "fit_dense_windows": len(dense_windows),
        "held_prefix_windows": len(held_windows),
        "manifest": manifest_report,
        "normalizer": {
            "statistics_sha256": statistics_sha,
            "matches_frozen_control": True,
            "held_condition_in_normalizer": False,
        },
        "start": {"sha256": start_sha, "matches_frozen_control": True},
        "control": control,
        "point": point_report,
        "point_replay_identity": point_replay_identity,
        "point_replay_quality_check": replay_check,
        "uq": {
            "training": uq_training,
            "seal": uq_seal,
            "baseline_metrics": baseline_metrics,
            "baseline_metric_state": baseline_state,
            "candidate_metrics": candidate_metrics,
            "candidate_metric_state": candidate_state,
            "target_audit": candidate_target_audit,
        },
        "comparison": comparison,
        "access_audit": audit.report(),
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_class": (
            "Real-stage whole-AoA-held UQ-only validation; point state is "
            "SHA-identical and held AoA is excluded from point and UQ fit"),
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    return result


def aggregate_results(pairs: Sequence[Sequence[str]]) -> dict:
    rows = []
    sources = []
    for raw_path, raw_sha in pairs:
        path = Path(raw_path)
        expected = dr1.normalized_sha256(raw_sha, "UQ fold result SHA")
        actual = dr1.sha256(path)
        if actual != expected:
            raise RuntimeError(f"UQ fold result changed: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("schema") != FOLD_SCHEMA
                or row.get("experiment") != EXPERIMENT):
            raise ValueError("condition-held UQ fold identity changed")
        _control_row, control = held.load_frozen_control(
            (row["control"]["path"], row["control"]["sha256"]),
            int(row["held_aoa"]))
        for key in ("metrics", "metric_state", "target_audit"):
            if control[key] != row["control"][key]:
                raise RuntimeError("embedded frozen Dense4k control changed")
        if (not row["point_replay_identity"][
                "sample_stream_matches_frozen_control"]
                or not row["point_replay_quality_check"]["pass"]):
            raise RuntimeError("point replay identity gate did not pass")
        rows.append(row)
        sources.append({"path": str(path.resolve()), "sha256": actual})
    held_aoas = [int(row["held_aoa"]) for row in rows]
    if len(held_aoas) != len(set(held_aoas)):
        raise ValueError("condition-held UQ aggregate repeats an AoA")
    baseline_state = smu1.merge_metric_states([
        row["uq"]["baseline_metric_state"] for row in rows])
    candidate_state = smu1.merge_metric_states([
        row["uq"]["candidate_metric_state"] for row in rows])
    if not uq_release.point_state_equal(baseline_state, candidate_state):
        raise RuntimeError("pooled UQ candidate changed point metric state")
    baseline_metrics = smu1.finalize_metric_state(baseline_state)
    candidate_metrics = smu1.finalize_metric_state(candidate_state)
    comparison = sps_comparison(candidate_metrics, baseline_metrics)
    fold_deltas = [
        float(row["comparison"]["native_sps_delta"]) for row in rows]
    exact_full_loao = set(held_aoas) == set(held.AOA_VALUES)
    pooled_pass = bool(
        comparison["native_sps_delta"]
        >= UQ_GATE["pooled_sps_gain_min_inclusive"])
    positive_folds = sum(value > 0.0 for value in fold_deltas)
    positive_pass = bool(
        exact_full_loao
        and positive_folds >= UQ_GATE["minimum_positive_folds"])
    worst_fold = min(fold_deltas)
    worst_pass = bool(worst_fold >= UQ_GATE["worst_fold_min_inclusive"])
    gate_pass = bool(
        exact_full_loao and pooled_pass and positive_pass and worst_pass)
    return {
        "schema": AGGREGATE_SCHEMA,
        "experiment": EXPERIMENT,
        "sources": sources,
        "held_aoas": held_aoas,
        "arms": {
            "native_baseline": {
                "metric_state": baseline_state, "metrics": baseline_metrics},
            UQ_ARM: {
                "metric_state": candidate_state, "metrics": candidate_metrics},
        },
        "comparison": comparison,
        "gate": {
            "thresholds": dict(UQ_GATE),
            "point_replay_max_absolute_contribution": max(
                row["point_replay_quality_check"][
                    "absolute_point_final_contribution"] for row in rows),
            "point_replay_tolerance": MAX_POINT_REPLAY_CONTRIBUTION_ABS,
            "exact_full_loao": exact_full_loao,
            "pooled_pass": pooled_pass,
            "positive_folds": positive_folds,
            "positive_folds_pass": positive_pass,
            "worst_fold": worst_fold,
            "worst_fold_pass": worst_pass,
            "validation_gate_pass": gate_pass,
        },
        "decision": (
            "PASS_CONDITION_HELD_UQ_ONLY_GATE" if gate_pass
            else "STOP_CONDITION_HELD_UQ_ONLY_GATE"),
        "submission_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fold-worker", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--held-aoa", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256")
    parser.add_argument(
        "--control-fold-result", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument(
        "--fold-result", nargs=2, action="append", metavar=("PATH", "SHA256"))
    parser.add_argument("--aggregate-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=2400.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.fold_worker:
        required = {
            "held_aoa": args.held_aoa,
            "output_dir": args.output_dir,
            "real_dir": args.real_dir,
            "clean_manifest": args.clean_manifest,
            "start_sim": args.start_sim,
            "control_fold_result": args.control_fold_result,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise ValueError(f"condition-held UQ fold is missing {missing}")
        if args.fold_result or args.aggregate_output:
            raise ValueError("UQ fold worker received aggregate arguments")
    else:
        if not args.fold_result or args.aggregate_output is None:
            raise ValueError("UQ aggregate requires fold results and output")
        if args.aggregate_output.exists():
            raise FileExistsError(args.aggregate_output)
        if any((args.held_aoa, args.output_dir, args.real_dir,
                args.clean_manifest, args.start_sim, args.start_sim_sha256,
                args.control_fold_result)):
            raise ValueError("UQ aggregate received fold-only arguments")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.aggregate_only:
        result = aggregate_results(args.fold_result)
        dr1.write_json_atomic(args.aggregate_output, result)
        print(json.dumps({
            "event": "b210_condition_held_uq_aggregate_complete",
            "decision": result["decision"],
            "output": str(args.aggregate_output),
        }), flush=True)
        return
    result = run_fold(args)
    print(json.dumps({
        "event": "b210_condition_held_uq_fold_complete",
        "held_aoa": result["held_aoa"],
        "comparison": result["comparison"],
        "output": str(args.output_dir / "fold_result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
