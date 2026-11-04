#!/usr/bin/env python3
"""Pair the current Real objective against official-point LGF1.

Both arms start from the same sealed LGF1 Sim10k checkpoint and consume the
same dense-stride5 Real prefix schedule, augmentation RNG, optimizer, EMA and
normalizer.  The clean716 tail is opened only after both endpoints are sealed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time
from typing import Mapping

import torch

import train_b210_scratch_real_dense_aug as dense


ft12 = dense.ft12
small4k = dense.small4k
dr1 = dense.dr1
smu1 = dense.smu1

EXPERIMENT = "b210_legacy_tke_floor_real_paired"
SCHEMA = "b210_legacy_tke_floor_real_paired_v1"
START_SIM_SHA256 = (
    "385896809a1c35a23b405956e36252b58671ee2e0c6c4201fd4a2f65d87595e1")
OFFICIAL_POINT_WEIGHTS = (46.0, 10.0, 9.0)
LEGACY_REAL_POINT_WEIGHTS = tuple(
    float(value) for value in ft12.npt1.joint_real.REAL_ACCURACY_WEIGHTS)
UPDATES = small4k.UPDATES
TAIL_WINDOWS = dense.TAIL_WINDOWS

PROMOTION_GATE = {
    "official_point_contribution": 0.04,
    "rel_l2_score": -0.03,
    "tke_rel_l2_score": -0.10,
    "mvpe_rel_l2_score": -0.05,
    "native_sps_score": -0.10,
    "positive_aoas": 3,
}


def official_point_contribution(delta: Mapping[str, float]) -> float:
    """Return the published-weight contribution of the three point scores."""
    return (
        0.46 * float(delta["rel_l2_score"])
        + 0.10 * float(delta["tke_rel_l2_score"])
        + 0.09 * float(delta["mvpe_rel_l2_score"])
    )


def promotion_gate(comparison: Mapping[str, object]) -> dict:
    delta = comparison["delta"]
    by_aoa = comparison["by_aoa"]
    contribution = official_point_contribution(delta)
    positive_aoas = sum(
        official_point_contribution(row["delta"]) > 0.0
        for row in by_aoa.values())
    checks = {
        "official_point_contribution": (
            contribution >= PROMOTION_GATE["official_point_contribution"]),
        "rel_l2_floor": (
            float(delta["rel_l2_score"])
            >= PROMOTION_GATE["rel_l2_score"]),
        "tke_floor": (
            float(delta["tke_rel_l2_score"])
            >= PROMOTION_GATE["tke_rel_l2_score"]),
        "mvpe_floor": (
            float(delta["mvpe_rel_l2_score"])
            >= PROMOTION_GATE["mvpe_rel_l2_score"]),
        "sps_proxy_floor": (
            float(delta["native_sps_score"])
            >= PROMOTION_GATE["native_sps_score"]),
        "positive_aoas": positive_aoas >= PROMOTION_GATE["positive_aoas"],
    }
    return {
        "official_point_contribution": contribution,
        "positive_aoas": positive_aoas,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _arm_contract(arm: str) -> dict:
    if arm == "control":
        return {
            "point_accuracy_weights": LEGACY_REAL_POINT_WEIGHTS,
            "legacy_tke_gradient_floor": False,
            "legacy_point_accuracy_weights": None,
        }
    if arm == "candidate":
        return {
            "point_accuracy_weights": OFFICIAL_POINT_WEIGHTS,
            "legacy_tke_gradient_floor": True,
            "legacy_point_accuracy_weights": LEGACY_REAL_POINT_WEIGHTS,
        }
    raise ValueError(f"unknown Real LGF1 arm {arm}")


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 prefix manifest byte identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = dense.load_evaluation_rows(
        args.evaluation_manifest, rows)
    start_sha = dr1.sha256(args.start_sim)
    if start_sha != START_SIM_SHA256:
        raise RuntimeError("LGF1 Sim10k checkpoint SHA changed")
    device = torch.device(args.device)
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("paired LGF1 Real experiment requires CUDA")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    expected_config, expected_parameters, expected_controller_parameters = (
        dense.architecture_contract(False, False))

    prefix_audit = dense.PrefixAudit()
    trajectories, metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", prefix_audit)
    if len(sparse_windows) != dr1.EXPECTED_WINDOWS:
        raise RuntimeError("clean81 sparse prefix count changed")
    windows = dense.dense_windows(trajectories, dense.CANDIDATE_STRIDE)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    sufficient = smu1.window_sufficient_statistics(
        trajectories, sparse_windows)
    statistics_values = smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics_values, device)
    torch.save(tuple(value.detach().cpu() for value in statistics_values),
               args.output_dir / "point_normalizer.pt")
    schedule = smu1.paired_batch_schedule(
        range(len(windows)), windows, updates=UPDATES, seed=small4k.SEED)
    schedule["updates"] = UPDATES

    reports: dict[str, dict] = {}
    seals: dict[str, dict] = {}
    for arm in ("control", "candidate"):
        ft12.npt1.ensemble_train.seed_everything(small4k.SEED)
        model, start_report = ft12.validate_start_checkpoint(
            args.start_sim, start_sha, normalizer, device,
            expected_config=expected_config,
            expected_parameters=expected_parameters,
            expected_controller_parameters=expected_controller_parameters)
        arm_dir = args.output_dir / arm
        arm_dir.mkdir()
        access = dense.ArmSeal(arm)
        contract = _arm_contract(arm)
        training, seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            arm_dir, start_report, access,
            input_augmentation=dense.AUGMENTATION,
            expected_parameters=expected_parameters,
            **contract)
        if access.value != seal:
            raise RuntimeError(f"{arm} endpoint seal was not bound")
        reports[arm] = {
            "fit_windows": len(windows),
            "window_stride": dense.CANDIDATE_STRIDE,
            "input_augmentation": dense.AUGMENTATION,
            "objective": contract,
            "training": training,
            "seal": seal,
        }
        seals[arm] = seal
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if (reports["control"]["training"]["actual_stream_sha256"]
            != reports["candidate"]["training"]["actual_stream_sha256"]):
        raise RuntimeError("paired LGF1 Real sample streams differ")

    del dataset, trajectories
    gc.collect()
    tail_audit = dense.TailAudit(seals)
    tail_trajectories, tail_metadata, tail_windows = dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", tail_audit)
    tail_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        tail_trajectories, tail_windows)
    indices_by_aoa = dense.group_window_indices_by_aoa(
        tail_windows, tail_metadata)
    for arm in ("control", "candidate"):
        model = small4k.load_primary_endpoint(
            seals[arm], normalizer, device,
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
        reports[arm].update({
            "metrics": metrics,
            "metric_state": state,
            "target_audit": target_audit,
            "metrics_by_aoa": metrics_by_aoa,
        })
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if reports["control"]["target_audit"] != reports["candidate"][
            "target_audit"]:
        raise RuntimeError("paired LGF1 clean716 target audit differs")

    comparison = dense.compare(
        reports["candidate"]["metrics"], reports["control"]["metrics"])
    comparison["by_aoa"] = {
        str(aoa): dense.compare(
            reports["candidate"]["metrics_by_aoa"][str(aoa)],
            reports["control"]["metrics_by_aoa"][str(aoa)])
        for aoa in dr1.AOA_VALUES
    }
    gate = promotion_gate(comparison)
    if not math.isfinite(gate["official_point_contribution"]):
        raise FloatingPointError("paired LGF1 Real comparison is non-finite")
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "start": {
            "checkpoint": str(args.start_sim.resolve()),
            "checkpoint_sha256": start_sha,
            "sim_direction_gate": "PASS_SIM_DIRECTION_GATE",
        },
        "manifest": manifest_report,
        "evaluation_manifest": {
            "path": str(args.evaluation_manifest.resolve()),
            "sha256": dr1.sha256(args.evaluation_manifest),
            "tail_windows": len(tail_windows),
        },
        "normalizer": {
            "kind": "clean2544 sparse-stride20 prefix only",
            "statistics_sha256": smu1.statistics_digest(statistics_values),
            "path": str((args.output_dir / "point_normalizer.pt").resolve()),
        },
        "paired_contract": {
            "same_start": True,
            "same_dense_stride5_data": True,
            "same_augmentation_and_rng": True,
            "same_optimizer_scheduler_ema": True,
            "only_difference": "point objective and LGF1 correction",
        },
        "parameters": expected_parameters,
        "controller_parameters": expected_controller_parameters,
        "arms": reports,
        "comparison": comparison,
        "promotion_gate": gate,
        "access_audit": {
            "prefix_opened": prefix_audit.opened,
            "tail_opened_after_both_seals": tail_audit.opened,
            "tail_target_read_before_both_seals": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "decision": (
            "PASS_REAL_LGF1_PAIRED" if gate["pass"]
            else "STOP_REAL_LGF1_PAIRED"),
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_legacy_tke_floor_real_paired_complete",
        "decision": result["decision"],
        "promotion_gate": result["promotion_gate"],
        "comparison": result["comparison"],
        "result": str(args.output_dir / "result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
