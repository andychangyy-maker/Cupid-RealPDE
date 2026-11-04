#!/usr/bin/env python3
"""Run one preregistered multiscale B210 Real4k + Sim-rehearsal endpoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time
from typing import Mapping

import torch
from torch.utils.data import DataLoader

from build_sim_p1_point_teacher_cache import load_and_validate_cache
import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_np_functional_sim as sim_tke
import train_b210_scratch_real_dense_aug as dense4k
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_real_sim_rehearsal as rehearsal
import train_c208_smu1_paired_loao as smu1


EXPERIMENT = "b210_multiscale_energy_real4k_sim_rehearsal"
SCHEMA = "b210_multiscale_energy_real4k_sim_rehearsal_v1"
REHEARSAL_WEIGHT = 0.05
SIM_STRIDE = 20
SIM_TRAJECTORIES = 100
REFERENCE_SCHEMA = dense4k.MULTISCALE_ENERGY_SCHEMA
REFERENCE_EXPERIMENT = dense4k.MULTISCALE_ENERGY_EXPERIMENT
OLD_SHARED_DENSE4K = {
    "rel_l2_score": 96.08181821913679,
    "tke_rel_l2_score": 81.38940436498885,
    "mvpe_rel_l2_score": 96.70708155944949,
    "point_score": 91.79914673095496,
    "native_sps_score": 53.489587893276315,
    "native_sps_coverage": 0.9050327882104381,
    "native_sps_mean_width": 0.017006921992774888,
}
THRESHOLDS = {
    "point_score": 0.05,
    "tke_rel_l2_score": 0.15,
    "rel_l2_score": -0.03,
    "mvpe_rel_l2_score": -0.03,
    "native_sps_score_reference": -0.10,
    "native_sps_score_old": 0.0,
    "positive_aoas": 4,
    "worst_aoa_point": -0.06,
}


class EndpointTailAudit:
    def __init__(self, seal: Mapping[str, object]) -> None:
        self.seal = dict(seal)
        self.opened: list[str] = []

    def before_open(self, row: dict, phase: str) -> None:
        if phase != "held" or row["file"] == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("invalid rehearsal clean-tail open")
        path = Path(str(self.seal["path"]))
        if not path.is_file() or dr1.sha256(path) != self.seal["file_sha256"]:
            raise RuntimeError("rehearsal endpoint changed before tail open")

    def record_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        if phase != "held" or filename in self.opened:
            raise RuntimeError("invalid or duplicate rehearsal tail record")
        self.opened.append(filename)


def load_reference(path: Path, expected_sha256: str,
                   evaluation_manifest: Path, start_sha256: str) -> dict:
    actual = dr1.sha256(path)
    if actual != dr1.normalized_sha256(expected_sha256, "reference SHA"):
        raise RuntimeError("Real4k reference result changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema") != REFERENCE_SCHEMA
            or payload.get("experiment") != REFERENCE_EXPERIMENT
            or payload.get("start", {}).get("checkpoint_sha256")
            != start_sha256
            or payload.get("horizon_local_energy_controller") is not False
            or payload.get("multiscale_energy_controller") is not True):
        raise RuntimeError("Real4k reference identity changed")
    evaluation = payload.get("evaluation_manifest", {})
    if (evaluation.get("sha256") != dr1.sha256(evaluation_manifest)
            or int(evaluation.get("tail_windows", -1))
            != dense4k.TAIL_WINDOWS):
        raise RuntimeError("Real4k reference tail identity changed")
    candidate = payload.get("arms", {}).get("candidate", {})
    if (not isinstance(candidate.get("metrics"), Mapping)
            or not isinstance(candidate.get("metrics_by_aoa"), Mapping)
            or not isinstance(candidate.get("target_audit"), Mapping)):
        raise RuntimeError("Real4k reference candidate is incomplete")
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "metrics": dict(candidate["metrics"]),
        "metrics_by_aoa": {
            key: dict(value)
            for key, value in candidate["metrics_by_aoa"].items()},
        "target_audit": dict(candidate["target_audit"]),
        "normalizer": dict(payload.get("normalizer", {})),
        "seal": dict(candidate.get("seal", {})),
    }


def prepare_sim(args: argparse.Namespace, start_payload: Mapping[str, object],
                device: torch.device) -> tuple[dict, object]:
    sim_normalizer, normalizer_values, normalizer_sha = (
        rehearsal.frozen_sim_normalizer(dict(start_payload), device))
    actual_cache_sha = dr1.sha256(args.p1_cache)
    if actual_cache_sha != dr1.normalized_sha256(
            args.p1_cache_sha256, "P1 cache SHA"):
        raise RuntimeError("P1 rehearsal cache changed")
    trajectories, windows, manifest = (
        rehearsal.ensemble_train.common.scratch.load_simulations(
            args.sim_dir, 0, SIM_STRIDE, True))
    if len(trajectories) != SIM_TRAJECTORIES or len(windows) != 4800:
        raise RuntimeError("released-Sim rehearsal inventory changed")
    dataset = rehearsal.ensemble_train.common.scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    cache, cache_metadata = load_and_validate_cache(
        args.p1_cache, manifest, dataset)
    if cache_metadata["sha256"] != actual_cache_sha:
        raise RuntimeError("validated P1 cache identity changed")
    tagged = rehearsal.ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(
        tagged, batch_size=small4k.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(small4k.SEED))
    return ({
        "loader": loader,
        "normalizer": sim_normalizer,
        "cache": cache,
        "manifest": manifest,
        "manifest_sha256": dr1.canonical_sha256(manifest),
        "cache_metadata": cache_metadata,
        "normalizer_values": normalizer_values,
        "normalizer_sha256": normalizer_sha,
        "windows": len(windows),
        "trajectories": len(trajectories),
    }, dataset)


class SimRehearsalBackward:
    def __init__(self, sim: dict, real_normalizer, device: torch.device) -> None:
        self.sim = sim
        self.real_normalizer = real_normalizer
        self.device = device
        self.iterator = rehearsal.ensemble_train.common.repro.cycling(
            sim["loader"])
        self.rows: list[dict] = []

    def __call__(self, iteration: int, model) -> Mapping[str, float]:
        batch = next(self.iterator)
        if not isinstance(batch, (tuple, list)) or len(batch) != 4:
            raise RuntimeError("Sim rehearsal batch is not tuple4")
        _history, _target, trajectory_ids, sample_indices = batch
        self.rows.append({
            "iteration": iteration,
            "trajectory_ids": trajectory_ids.to(dtype=torch.long).tolist(),
            "sample_indices": sample_indices.to(dtype=torch.long).tolist(),
        })
        return rehearsal.backward_sim_rehearsal(
            model, batch, self.sim["normalizer"], self.real_normalizer,
            self.sim["cache"], self.device, REHEARSAL_WEIGHT)

    def contract(self) -> dict:
        return {
            "order": "Real backward -> Sim backward -> clip -> optimizer -> EMA",
            "outer_weight": REHEARSAL_WEIGHT,
            "inference_graph_changed": False,
        }

    def report(self) -> dict:
        if len(self.rows) != small4k.UPDATES:
            raise RuntimeError("Sim rehearsal stream did not reach 4000")
        return {
            "updates": len(self.rows),
            "stream_sha256": dr1.canonical_sha256(self.rows),
            "outer_weight": REHEARSAL_WEIGHT,
            "truth_weights": list(rehearsal.SIM_ACCURACY_WEIGHTS),
            "p1_distill_weight": rehearsal.FORMAL_P1_DISTILL_WEIGHT,
            "manifest_sha256": self.sim["manifest_sha256"],
            "cache_sha256": self.sim["cache_metadata"]["sha256"],
            "normalizer_sha256": self.sim["normalizer_sha256"],
            "windows": self.sim["windows"],
            "trajectories": self.sim["trajectories"],
        }


def release_gate(reference_comparison: Mapping[str, object],
                 old_comparison: Mapping[str, object]) -> dict:
    ref = reference_comparison["delta"]
    old = old_comparison["delta"]
    by_aoa = reference_comparison["by_aoa"]
    positive_aoas = sum(
        float(row["candidate_non_time_proxy_delta"]) > 0.0
        for row in by_aoa.values())
    worst_aoa_point = min(
        float(row["delta"]["point_score"]) for row in by_aoa.values())
    checks = {
        "finite": all(math.isfinite(float(value))
                      for value in (*ref.values(), *old.values())),
        "reference_point": float(ref["point_score"]) >= 0.05,
        "reference_tke": float(ref["tke_rel_l2_score"]) >= 0.15,
        "reference_rel": float(ref["rel_l2_score"]) >= -0.03,
        "reference_mvpe": float(ref["mvpe_rel_l2_score"]) >= -0.03,
        "reference_sps": float(ref["native_sps_score"]) >= -0.10,
        "old_point": float(old["point_score"]) >= 0.05,
        "old_tke": float(old["tke_rel_l2_score"]) >= 0.15,
        "old_rel": float(old["rel_l2_score"]) >= -0.03,
        "old_mvpe": float(old["mvpe_rel_l2_score"]) >= -0.03,
        "old_sps": float(old["native_sps_score"]) >= 0.0,
        "aoa_count": positive_aoas >= 4,
        "aoa_floor": worst_aoa_point >= -0.06,
    }
    passed = all(checks.values())
    return {
        "pass": passed,
        "decision": (
            "PASS_B210_MULTISCALE_REAL4K_REHEARSAL_TO_UQ"
            if passed else "STOP_B210_MULTISCALE_REAL4K_REHEARSAL"),
        "checks": checks,
        "thresholds": dict(THRESHOLDS),
        "positive_aoas": positive_aoas,
        "worst_aoa_point_delta": worst_aoa_point,
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(args.device)
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("B210 rehearsal requires CUDA")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 prefix manifest changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = dense4k.load_evaluation_rows(
        args.evaluation_manifest, rows)
    start_sha = dr1.sha256(args.start_sim)
    if start_sha != dr1.normalized_sha256(
            args.start_sim_sha256, "start Sim SHA"):
        raise RuntimeError("multiscale Sim start changed")
    reference = load_reference(
        args.reference_result, args.reference_result_sha256,
        args.evaluation_manifest, start_sha)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")

    prefix_audit = dense4k.PrefixAudit()
    trajectories, metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", prefix_audit)
    dense_windows = dense4k.dense_windows(
        trajectories, dense4k.CANDIDATE_STRIDE)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, dense_windows)
    sufficient = smu1.window_sufficient_statistics(
        trajectories, sparse_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    statistics_sha = smu1.statistics_digest(statistics)
    if reference["normalizer"].get("statistics_sha256") != statistics_sha:
        raise RuntimeError("rehearsal/reference Real normalizer changed")
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "point_normalizer.pt")

    config, parameters, controller_parameters = (
        dense4k.architecture_contract(False, True))
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha, normalizer, device,
        expected_config=config, expected_parameters=parameters,
        expected_controller_parameters=controller_parameters)
    start_payload = torch.load(
        args.start_sim, map_location="cpu", weights_only=False)
    sim, sim_dataset = prepare_sim(args, start_payload, device)
    callback = SimRehearsalBackward(sim, normalizer, device)
    schedule = smu1.paired_batch_schedule(
        range(len(dense_windows)), dense_windows,
        updates=small4k.UPDATES, seed=small4k.SEED)
    schedule["updates"] = small4k.UPDATES
    ft12.npt1.ensemble_train.seed_everything(small4k.SEED)
    access = dense4k.ArmSeal("rehearsal")
    training, seal = small4k.train_candidate(
        model, dataset, schedule, normalizer, device, deadline,
        args.output_dir, start_report, access,
        input_augmentation=dense4k.AUGMENTATION,
        expected_parameters=parameters,
        extra_backward=callback,
        extra_backward_contract=callback.contract())
    if access.value != seal:
        raise RuntimeError("rehearsal endpoint seal was not bound")
    rehearsal_report = callback.report()
    del (model, dataset, sim_dataset, trajectories, sim, callback,
         start_payload)
    gc.collect()
    torch.cuda.empty_cache()

    tail_audit = EndpointTailAudit(seal)
    tail_trajectories, tail_metadata, tail_windows = dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", tail_audit)
    tail_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        tail_trajectories, tail_windows)
    evaluated = small4k.load_primary_endpoint(
        seal, normalizer, device, expected_model_config=config,
        expected_parameters=parameters,
        expected_controller_parameters=controller_parameters)
    metrics, state, target_audit = smu1.evaluate_model(
        evaluated, tail_dataset, list(range(len(tail_windows))),
        normalizer, device, deadline)
    indices_by_aoa = dense4k.group_window_indices_by_aoa(
        tail_windows, tail_metadata)
    metrics_by_aoa = {}
    for aoa, indices in indices_by_aoa.items():
        aoa_metrics, _state, _audit = smu1.evaluate_model(
            evaluated, tail_dataset, indices, normalizer, device, deadline)
        metrics_by_aoa[str(aoa)] = aoa_metrics
    if target_audit != reference["target_audit"]:
        raise RuntimeError("rehearsal/reference clean716 target audit differs")

    reference_comparison = dense4k.compare(metrics, reference["metrics"])
    reference_comparison["by_aoa"] = {
        str(aoa): dense4k.compare(
            metrics_by_aoa[str(aoa)],
            reference["metrics_by_aoa"][str(aoa)])
        for aoa in dr1.AOA_VALUES
    }
    old_comparison = dense4k.compare(metrics, OLD_SHARED_DENSE4K)
    gate = release_gate(reference_comparison, old_comparison)
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "start": {
            "checkpoint": str(args.start_sim.resolve()),
            "checkpoint_sha256": start_sha,
            "from_random_initialization_sim10k": True,
            "real_checkpoint_continuation": False,
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
        "parameters": parameters,
        "controller_parameters": controller_parameters,
        "training": training,
        "sim_rehearsal": rehearsal_report,
        "seal": seal,
        "candidate": {
            "metrics": metrics,
            "metric_state": state,
            "metrics_by_aoa": metrics_by_aoa,
            "target_audit": target_audit,
        },
        "reference": reference,
        "reference_comparison": reference_comparison,
        "old_shared_dense4k": dict(OLD_SHARED_DENSE4K),
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
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--start-sim-sha256", required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--reference-result-sha256", required=True)
    parser.add_argument("--p1-cache", type=Path, required=True)
    parser.add_argument("--p1-cache-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_multiscale_energy_real4k_rehearsal_complete",
        "candidate": result["candidate"]["metrics"],
        "reference_delta": result["reference_comparison"]["delta"],
        "old_shared_delta": result["old_shared_comparison"]["delta"],
        "gate": result["gate"],
        "result": str(args.output_dir / "result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
