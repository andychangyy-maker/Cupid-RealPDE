#!/usr/bin/env python3
"""Train the member-integrated LGF2 Real candidate against sealed LGF1 control."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

import torch

import train_b210_legacy_tke_floor_real_paired as lgf1


dense = lgf1.dense
ft12 = lgf1.ft12
small4k = lgf1.small4k
dr1 = lgf1.dr1
smu1 = lgf1.smu1

EXPERIMENT = "b210_member_integrated_tke_floor_real"
SCHEMA = "b210_member_integrated_tke_floor_real_v1"
PRIOR_RESULT_SHA256 = (
    "ec2a683cf00fcaff4bfcbbe871136ca1ce87960646b2b6bff4febff2bf1469aa")
PRIOR_CONTROL_FILE_SHA256 = (
    "cefef7f1a8498f66edbdec8af0c4ec03cb8473b4c35214d883f9d3148c61d98c")
PRIOR_STREAM_SHA256 = (
    "14331e0ff2d61c7f62a384061b47f629f6a71999fad6ba4e97f37d414f17a7fb")


def candidate_contract() -> dict:
    return {
        "point_accuracy_weights": lgf1.OFFICIAL_POINT_WEIGHTS,
        "member_accuracy_weights": lgf1.OFFICIAL_POINT_WEIGHTS,
        "legacy_tke_gradient_floor": True,
        "legacy_tke_gradient_floor_on_members": True,
        "legacy_point_accuracy_weights": lgf1.LEGACY_REAL_POINT_WEIGHTS,
        "legacy_member_accuracy_weights": lgf1.LEGACY_REAL_POINT_WEIGHTS,
    }


def load_prior(path: Path) -> dict:
    if dr1.sha256(path) != PRIOR_RESULT_SHA256:
        raise RuntimeError("sealed LGF1 Real result SHA changed")
    prior = json.loads(path.read_text(encoding="utf-8"))
    control = prior.get("arms", {}).get("control", {})
    candidate = prior.get("arms", {}).get("candidate", {})
    seal = control.get("seal", {})
    if (prior.get("schema") != lgf1.SCHEMA
            or prior.get("decision") != "STOP_REAL_LGF1_PAIRED"
            or seal.get("file_sha256") != PRIOR_CONTROL_FILE_SHA256
            or seal.get("sample_stream_sha256") != PRIOR_STREAM_SHA256
            or not Path(str(seal.get("path", ""))).is_file()
            or dr1.sha256(Path(str(seal["path"])))
            != PRIOR_CONTROL_FILE_SHA256
            or not isinstance(control.get("metrics"), dict)
            or not isinstance(control.get("metrics_by_aoa"), dict)
            or not isinstance(candidate.get("metrics"), dict)):
        raise RuntimeError("sealed LGF1 control/result contract changed")
    return prior


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    prior = load_prior(args.prior_result)
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 prefix manifest byte identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = dense.load_evaluation_rows(
        args.evaluation_manifest, rows)
    start_sha = dr1.sha256(args.start_sim)
    if start_sha != lgf1.START_SIM_SHA256:
        raise RuntimeError("LGF1 Sim10k checkpoint SHA changed")
    device = torch.device(args.device)
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("LGF2 Real experiment requires CUDA")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    expected_config, expected_parameters, expected_controller_parameters = (
        dense.architecture_contract(False, False))

    prefix_audit = dense.PrefixAudit()
    trajectories, metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", prefix_audit)
    windows = dense.dense_windows(trajectories, dense.CANDIDATE_STRIDE)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    sufficient = smu1.window_sufficient_statistics(
        trajectories, sparse_windows)
    statistics_values = smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics_values, device)
    statistics_sha = smu1.statistics_digest(statistics_values)
    if statistics_sha != prior["normalizer"]["statistics_sha256"]:
        raise RuntimeError("LGF2 normalizer differs from paired LGF1")
    torch.save(tuple(value.detach().cpu() for value in statistics_values),
               args.output_dir / "point_normalizer.pt")
    schedule = smu1.paired_batch_schedule(
        range(len(windows)), windows, updates=small4k.UPDATES,
        seed=small4k.SEED)
    schedule["updates"] = small4k.UPDATES
    if schedule["sha256"] != PRIOR_STREAM_SHA256:
        raise RuntimeError("LGF2 schedule differs from paired LGF1")

    ft12.npt1.ensemble_train.seed_everything(small4k.SEED)
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha, normalizer, device,
        expected_config=expected_config,
        expected_parameters=expected_parameters,
        expected_controller_parameters=expected_controller_parameters)
    candidate_dir = args.output_dir / "candidate"
    candidate_dir.mkdir()
    access = dense.ArmSeal("candidate")
    contract = candidate_contract()
    training, candidate_seal = small4k.train_candidate(
        model, dataset, schedule, normalizer, device, deadline,
        candidate_dir, start_report, access,
        input_augmentation=dense.AUGMENTATION,
        expected_parameters=expected_parameters,
        **contract)
    if access.value != candidate_seal:
        raise RuntimeError("LGF2 candidate endpoint seal was not bound")
    del model, dataset, trajectories
    gc.collect()
    torch.cuda.empty_cache()

    prior_control = prior["arms"]["control"]
    seals = {
        "control": prior_control["seal"],
        "candidate": candidate_seal,
    }
    tail_audit = dense.TailAudit(seals)
    tail_trajectories, tail_metadata, tail_windows = dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", tail_audit)
    tail_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        tail_trajectories, tail_windows)
    indices_by_aoa = dense.group_window_indices_by_aoa(
        tail_windows, tail_metadata)
    model = small4k.load_primary_endpoint(
        candidate_seal, normalizer, device,
        expected_model_config=expected_config,
        expected_parameters=expected_parameters,
        expected_controller_parameters=expected_controller_parameters)
    metrics, state, target_audit = smu1.evaluate_model(
        model, tail_dataset, list(range(len(tail_windows))),
        normalizer, device, deadline)
    metrics_by_aoa = {}
    for aoa, indices in indices_by_aoa.items():
        aoa_metrics, _state, _audit = smu1.evaluate_model(
            model, tail_dataset, indices, normalizer, device, deadline)
        metrics_by_aoa[str(aoa)] = aoa_metrics
    if target_audit != prior_control["target_audit"]:
        raise RuntimeError("LGF2/prior-control clean716 target audits differ")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    comparison = dense.compare(metrics, prior_control["metrics"])
    comparison["by_aoa"] = {
        str(aoa): dense.compare(
            metrics_by_aoa[str(aoa)],
            prior_control["metrics_by_aoa"][str(aoa)])
        for aoa in dr1.AOA_VALUES
    }
    gate = lgf1.promotion_gate(comparison)
    if not math.isfinite(gate["official_point_contribution"]):
        raise FloatingPointError("LGF2 comparison is non-finite")
    versus_lgf1 = dense.compare(
        metrics, prior["arms"]["candidate"]["metrics"])
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "start": start_report,
        "prior_result": {
            "path": str(args.prior_result.resolve()),
            "sha256": PRIOR_RESULT_SHA256,
            "control_file_sha256": PRIOR_CONTROL_FILE_SHA256,
        },
        "manifest": manifest_report,
        "evaluation_manifest": {
            "path": str(args.evaluation_manifest.resolve()),
            "sha256": dr1.sha256(args.evaluation_manifest),
            "tail_windows": len(tail_windows),
        },
        "normalizer": {
            "statistics_sha256": statistics_sha,
            "path": str((args.output_dir / "point_normalizer.pt").resolve()),
        },
        "candidate_contract": contract,
        "training": training,
        "candidate": {
            "seal": candidate_seal,
            "metrics": metrics,
            "metric_state": state,
            "target_audit": target_audit,
            "metrics_by_aoa": metrics_by_aoa,
        },
        "comparison_vs_paired_control": comparison,
        "comparison_vs_lgf1_candidate": versus_lgf1,
        "promotion_gate": gate,
        "access_audit": {
            "prefix_opened": prefix_audit.opened,
            "tail_opened_after_control_and_candidate_seals": (
                tail_audit.opened),
            "tail_target_read_before_candidate_seal": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "decision": (
            "PASS_REAL_LGF2_MEMBER_INTEGRATED" if gate["pass"]
            else "STOP_REAL_LGF2_MEMBER_INTEGRATED"),
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=2 * 3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_member_integrated_tke_floor_real_complete",
        "decision": result["decision"],
        "promotion_gate": result["promotion_gate"],
        "comparison": result["comparison_vs_paired_control"],
        "result": str(args.output_dir / "result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
