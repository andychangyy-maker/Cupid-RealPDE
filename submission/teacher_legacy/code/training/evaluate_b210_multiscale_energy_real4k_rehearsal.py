#!/usr/bin/env python3
"""Recover the sealed Real4k+rehearsal endpoint evaluation after training.

This evaluator never trains or overwrites an endpoint.  It exists so a late
report-assembly failure cannot force a second stochastic training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import train_b210_multiscale_energy_real4k_rehearsal as experiment
import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_scratch_real_dense_aug as dense4k
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1


SCHEMA = "b210_multiscale_energy_real4k_sim_rehearsal_recovered_eval_v1"


def endpoint_seal(path: Path, expected_sha256: str) -> tuple[dict, dict]:
    actual = dr1.sha256(path)
    if actual != dr1.normalized_sha256(expected_sha256, "endpoint SHA"):
        raise RuntimeError("sealed rehearsal endpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != small4k.ENDPOINT_SCHEMA
            or payload.get("experiment") != small4k.EXPERIMENT
            or payload.get("endpoint") != small4k.PRIMARY_ENDPOINT
            or payload.get("primary") is not True
            or int(payload.get("iteration", -1)) != small4k.UPDATES
            or not isinstance(state, dict)):
        raise RuntimeError("sealed rehearsal endpoint payload changed")
    state_sha256 = dr1.model_state_sha256(state)
    if state_sha256 != payload.get("model_state_sha256"):
        raise RuntimeError("sealed rehearsal endpoint state changed")
    seal = {
        "endpoint": small4k.PRIMARY_ENDPOINT,
        "path": str(path.resolve()),
        "file_sha256": actual,
        "model_state_sha256": state_sha256,
        "model_config_sha256": dr1.canonical_sha256(payload["model_config"]),
        "sample_stream_sha256": payload["sample_stream_sha256"],
        "iteration": small4k.UPDATES,
        "primary": True,
        "optimizer_state_saved": False,
    }
    return seal, payload


def run(args: argparse.Namespace) -> dict:
    if args.output.is_file():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("recovered evaluation requires CUDA")
    torch.set_float32_matmul_precision("high")

    seal, payload = endpoint_seal(args.endpoint, args.endpoint_sha256)
    start_sha256 = str(payload.get("start", {}).get("checkpoint_sha256"))
    reference = experiment.load_reference(
        args.reference_result, args.reference_result_sha256,
        args.evaluation_manifest, start_sha256)

    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 prefix manifest changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = dense4k.load_evaluation_rows(
        args.evaluation_manifest, rows)

    prefix_audit = dense4k.PrefixAudit()
    trajectories, _metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", prefix_audit)
    statistics = smu1.fold_statistics(
        smu1.window_sufficient_statistics(trajectories, sparse_windows),
        range(len(sparse_windows)))
    statistics_sha256 = smu1.statistics_digest(statistics)
    if statistics_sha256 != reference["normalizer"].get("statistics_sha256"):
        raise RuntimeError("recovered evaluation normalizer changed")
    normalizer = smu1.FoldNormalizer(statistics, device)
    normalizer_report = ft12.normalizer_payload(normalizer)
    if (normalizer_report["sha256"] != payload.get("normalizer_sha256")
            or normalizer_report["values"] != payload.get("normalizer")):
        raise RuntimeError("recovered endpoint normalizer values changed")

    config, parameters, controller_parameters = (
        dense4k.architecture_contract(False, True))
    model = small4k.load_primary_endpoint(
        seal, normalizer, device, expected_model_config=config,
        expected_parameters=parameters,
        expected_controller_parameters=controller_parameters)

    tail_audit = experiment.EndpointTailAudit(seal)
    tail_trajectories, tail_metadata, tail_windows = dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", tail_audit)
    tail_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        tail_trajectories, tail_windows)
    metrics, state, target_audit = smu1.evaluate_model(
        model, tail_dataset, list(range(len(tail_windows))),
        normalizer, device, started + args.wall_budget_seconds)
    metrics_by_aoa = {}
    indices_by_aoa = dense4k.group_window_indices_by_aoa(
        tail_windows, tail_metadata)
    for aoa, indices in indices_by_aoa.items():
        aoa_metrics, _state, _audit = smu1.evaluate_model(
            model, tail_dataset, indices, normalizer, device,
            started + args.wall_budget_seconds)
        metrics_by_aoa[str(aoa)] = aoa_metrics
    if target_audit != reference["target_audit"]:
        raise RuntimeError("recovered/reference clean716 target audit differs")

    reference_comparison = dense4k.compare(metrics, reference["metrics"])
    reference_comparison["by_aoa"] = {
        str(aoa): dense4k.compare(
            metrics_by_aoa[str(aoa)],
            reference["metrics_by_aoa"][str(aoa)])
        for aoa in dr1.AOA_VALUES
    }
    old_comparison = dense4k.compare(
        metrics, experiment.OLD_SHARED_DENSE4K)
    gate = experiment.release_gate(reference_comparison, old_comparison)
    result = {
        "schema": SCHEMA,
        "experiment": experiment.EXPERIMENT,
        "recovered_after_report_assembly_failure": True,
        "endpoint": seal,
        "endpoint_training_identity": {
            "start": payload["start"],
            "normalizer_sha256": payload["normalizer_sha256"],
            "sample_stream_sha256": payload["sample_stream_sha256"],
            "ema": payload["ema"],
        },
        "manifest": manifest_report,
        "evaluation_manifest": {
            "path": str(args.evaluation_manifest.resolve()),
            "sha256": dr1.sha256(args.evaluation_manifest),
            "tail_windows": len(tail_windows),
        },
        "candidate": {
            "metrics": metrics,
            "metric_state": state,
            "metrics_by_aoa": metrics_by_aoa,
            "target_audit": target_audit,
        },
        "reference": reference,
        "reference_comparison": reference_comparison,
        "old_shared_dense4k": dict(experiment.OLD_SHARED_DENSE4K),
        "old_shared_comparison": old_comparison,
        "gate": gate,
        "access_audit": {
            "prefix_opened": prefix_audit.opened,
            "tail_opened_after_seal": tail_audit.opened,
            "tail_target_read_before_seal": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--endpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--reference-result-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_multiscale_rehearsal_recovered_eval_complete",
        "candidate": result["candidate"]["metrics"],
        "reference_delta": result["reference_comparison"]["delta"],
        "old_shared_delta": result["old_shared_comparison"]["delta"],
        "gate": result["gate"],
        "result": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
