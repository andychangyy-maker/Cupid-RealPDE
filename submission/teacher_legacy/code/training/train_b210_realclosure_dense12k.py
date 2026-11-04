#!/usr/bin/env python3
"""Train one Real-closure B210 on dense clean81 support for fixed 12k.

The run starts again from the sealed seed-1 closure-Sim10k checkpoint.  It
does not continue the existing Real4k model and it never evaluates an
intermediate Real endpoint.  Only EMA-u12000 is written before the fixed
clean716 tail is opened once.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import time
from typing import Iterator, Mapping, Sequence

import torch

import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_np_functional_sim as sim_tke
import train_b210_scratch_real_dense_aug as dense4k
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1


EXPERIMENT = "b210_realclosure_dense12k"
SCHEMA = "b210_realclosure_dense12k_v1"
ENDPOINT_SCHEMA = "b210_realclosure_dense12k_endpoint_v1"
MULTISCALE_EXPERIMENT = "b210_multiscale_tke_realclosure_dense12k"
MULTISCALE_SCHEMA = "b210_multiscale_tke_realclosure_dense12k_v1"
MULTISCALE_ENDPOINT_SCHEMA = (
    "b210_multiscale_tke_realclosure_dense12k_endpoint_v1")
MULTISCALE_ENERGY_EXPERIMENT = "b210_multiscale_energy_realclosure_dense12k"
MULTISCALE_ENERGY_SCHEMA = "b210_multiscale_energy_realclosure_dense12k_v1"
MULTISCALE_ENERGY_ENDPOINT_SCHEMA = (
    "b210_multiscale_energy_realclosure_dense12k_endpoint_v1")
UPDATES = 12_000
PRIMARY_ENDPOINT = "ema_u12000"
ENDPOINT_FILENAME = "ema999_u12000_primary.pth"
EXPECTED_START_SHA256 = (
    "505f36e0445d8d8541c3b974f37164a01e45b7e5c256f614eb54d2d8ff2a3afd")
THRESHOLDS = {
    "point_score": 0.04,
    "tke_rel_l2_score_alternative": 0.12,
    "rel_l2_score_alternative_floor": -0.02,
    "mvpe_rel_l2_score_alternative_floor": -0.02,
    "native_sps_score": -0.10,
    "positive_aoas": 4,
    "worst_aoa_point": -0.08,
}


class EndpointTailAudit:
    """Open the fixed tail only after the sole endpoint is SHA sealed."""

    def __init__(self, seal: Mapping[str, object]) -> None:
        self.seal = dict(seal)
        self.opened: list[str] = []

    def _validate_seal(self) -> None:
        path = Path(str(self.seal["path"]))
        if (not path.is_file()
                or dr1.sha256(path) != self.seal["file_sha256"]):
            raise RuntimeError("dense12k endpoint changed before tail open")

    def before_open(self, row: dict, phase: str) -> None:
        if phase != "held" or row["file"] == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("invalid dense12k clean-tail open")
        self._validate_seal()

    def record_open(self, row: dict, phase: str) -> None:
        filename = str(row["file"])
        if phase != "held" or filename in self.opened:
            raise RuntimeError("invalid or duplicate dense12k tail record")
        self.opened.append(filename)


@contextmanager
def dense12k_training_contract(
        multiscale_tke: bool = False,
        multiscale_energy: bool = False,
        ) -> Iterator[None]:
    """Temporarily bind the generic trainer to the preregistered endpoint."""

    if multiscale_tke and multiscale_energy:
        raise ValueError("select only one multiscale Dense12k mode")
    experiment = (
        MULTISCALE_ENERGY_EXPERIMENT if multiscale_energy
        else MULTISCALE_EXPERIMENT if multiscale_tke else EXPERIMENT)
    endpoint_schema = (
        MULTISCALE_ENERGY_ENDPOINT_SCHEMA if multiscale_energy
        else MULTISCALE_ENDPOINT_SCHEMA if multiscale_tke
        else ENDPOINT_SCHEMA)

    replacements = {
        "EXPERIMENT": experiment,
        "ENDPOINT_SCHEMA": endpoint_schema,
        "PRIMARY_ENDPOINT": PRIMARY_ENDPOINT,
        "UPDATES": UPDATES,
        "SCHEDULER_T_MAX": UPDATES,
    }
    original = {name: getattr(small4k, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(small4k, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(small4k, name, value)


def compare(candidate: Mapping[str, float],
            control: Mapping[str, float]) -> dict:
    return dense4k.compare(candidate, control)


def release_gate(comparison: Mapping[str, object]) -> dict:
    delta = comparison["delta"]
    by_aoa = comparison["by_aoa"]
    direct = float(delta["point_score"]) >= THRESHOLDS["point_score"]
    alternative = (
        float(delta["tke_rel_l2_score"])
        >= THRESHOLDS["tke_rel_l2_score_alternative"]
        and float(delta["rel_l2_score"])
        >= THRESHOLDS["rel_l2_score_alternative_floor"]
        and float(delta["mvpe_rel_l2_score"])
        >= THRESHOLDS["mvpe_rel_l2_score_alternative_floor"])
    positive_aoas = sum(
        float(row["candidate_non_time_proxy_delta"]) > 0.0
        for row in by_aoa.values())
    worst_point = min(
        float(row["delta"]["point_score"]) for row in by_aoa.values())
    checks = {
        "finite": all(math.isfinite(float(value)) for value in delta.values()),
        "point_or_tke_signal": direct or alternative,
        "native_sps_safe": (
            float(delta["native_sps_score"])
            >= THRESHOLDS["native_sps_score"]),
        "aoa_count": positive_aoas >= THRESHOLDS["positive_aoas"],
        "aoa_floor": worst_point >= THRESHOLDS["worst_aoa_point"],
    }
    passed = all(checks.values())
    return {
        "pass": passed,
        "decision": (
            "PASS_B210_DENSE12K_TO_NATIVE_UQ"
            if passed else "STOP_B210_DENSE12K_GATE"),
        "checks": checks,
        "thresholds": dict(THRESHOLDS),
        "positive_aoas": positive_aoas,
        "worst_aoa_point_delta": worst_point,
    }


def validate_baseline(
        baseline: Mapping[str, object], start_sha: str,
        evaluation_manifest: Path) -> dict:
    if baseline.get("schema") != dense4k.SCHEMA:
        raise RuntimeError("Dense4k baseline schema changed")
    start = baseline.get("start", {})
    if start.get("checkpoint_sha256") != start_sha:
        raise RuntimeError("Dense4k baseline did not use this Sim checkpoint")
    evaluation = baseline.get("evaluation_manifest", {})
    if (evaluation.get("sha256") != dr1.sha256(evaluation_manifest)
            or int(evaluation.get("tail_windows", -1))
            != dense4k.TAIL_WINDOWS):
        raise RuntimeError("Dense4k baseline evaluation manifest changed")
    candidate = baseline.get("arms", {}).get("candidate", {})
    metrics = candidate.get("metrics")
    metrics_by_aoa = candidate.get("metrics_by_aoa")
    target_audit = candidate.get("target_audit")
    if (not isinstance(metrics, Mapping)
            or not isinstance(metrics_by_aoa, Mapping)
            or set(metrics_by_aoa) != {str(value) for value in dr1.AOA_VALUES}
            or not isinstance(target_audit, Mapping)):
        raise RuntimeError("Dense4k baseline metrics are incomplete")
    return {
        "metrics": dict(metrics),
        "metrics_by_aoa": {
            key: dict(value) for key, value in metrics_by_aoa.items()},
        "target_audit": dict(target_audit),
        "seal": dict(candidate.get("seal", {})),
        "normalizer": dict(baseline.get("normalizer", {})),
    }


def validate_dense12k_reference(
        reference: Mapping[str, object], evaluation_manifest: Path,
        ) -> dict:
    if (reference.get("schema") != SCHEMA
            or reference.get("experiment") != EXPERIMENT):
        raise RuntimeError("Dense12k reference identity changed")
    evaluation = reference.get("evaluation_manifest", {})
    if (evaluation.get("sha256") != dr1.sha256(evaluation_manifest)
            or int(evaluation.get("tail_windows", -1))
            != dense4k.TAIL_WINDOWS):
        raise RuntimeError("Dense12k reference evaluation manifest changed")
    candidate = reference.get("candidate", {})
    metrics = candidate.get("metrics")
    metrics_by_aoa = candidate.get("metrics_by_aoa")
    target_audit = candidate.get("target_audit")
    if (not isinstance(metrics, Mapping)
            or not isinstance(metrics_by_aoa, Mapping)
            or set(metrics_by_aoa) != {str(value) for value in dr1.AOA_VALUES}
            or not isinstance(target_audit, Mapping)):
        raise RuntimeError("Dense12k reference metrics are incomplete")
    return {
        "metrics": dict(metrics),
        "metrics_by_aoa": {
            key: dict(value) for key, value in metrics_by_aoa.items()},
        "target_audit": dict(target_audit),
        "seal": dict(reference.get("seal", {})),
        "normalizer": dict(reference.get("normalizer", {})),
        "start": dict(reference.get("start", {})),
    }


def load_multiscale_tke_supervisor(
        path: Path, expected_sha256: str, start_sha256: str,
        model, device: torch.device,
        ) -> tuple[sim_tke.MultiScaleTKEDeepSupervisor, dict]:
    actual_sha = dr1.sha256(path)
    if actual_sha != dr1.normalized_sha256(
            expected_sha256, "multiscale TKE supervisor SHA"):
        raise RuntimeError("multiscale TKE supervisor sidecar changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("method") != sim_tke.MULTISCALE_TKE_METHOD
            or payload.get("model_checkpoint_sha256") != start_sha256
            or payload.get("deployable") is not False
            or float(payload.get("physical_scale", -1.0))
            != sim_tke.MULTISCALE_TKE_PHYSICAL_SCALE
            or tuple(payload.get("level_weights", ()))
            != sim_tke.MULTISCALE_TKE_LEVEL_WEIGHTS
            or float(payload.get("weight", -1.0))
            != sim_tke.MULTISCALE_TKE_WEIGHT):
        raise RuntimeError("multiscale TKE supervisor contract changed")
    supervisor = sim_tke.build_multiscale_tke_supervisor(model, device)
    supervisor.load_state_dict(payload["state_dict"], strict=True)
    return supervisor, {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "parameters": sum(
            parameter.numel() for parameter in supervisor.parameters()),
        "deployable": False,
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("Dense12k requires CUDA")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 prefix manifest byte identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = dense4k.load_evaluation_rows(
        args.evaluation_manifest, rows)
    start_sha = dr1.sha256(args.start_sim)
    expected_start = (args.start_sim_sha256 or EXPECTED_START_SHA256).lower()
    if start_sha != expected_start:
        raise RuntimeError("closure seed1 Sim checkpoint SHA changed")
    multiscale_tke = args.multiscale_tke_supervisor is not None
    multiscale_energy = bool(args.multiscale_energy_controller)
    if multiscale_tke != (args.multiscale_tke_supervisor_sha256 is not None):
        raise ValueError("multiscale TKE supervisor path/SHA must be paired")
    if multiscale_tke and multiscale_energy:
        raise ValueError("training-only TKE supervisor and deployed multiscale "
                         "energy controller are separate architecture arms")
    if (args.baseline_result is None) == (args.reference_result is None):
        raise ValueError("select exactly one baseline or Dense12k reference")
    reference_path = args.reference_result or args.baseline_result
    reference_payload = json.loads(
        reference_path.read_text(encoding="utf-8"))
    reference = (
        validate_dense12k_reference(
            reference_payload, args.evaluation_manifest)
        if args.reference_result is not None else validate_baseline(
            reference_payload, start_sha, args.evaluation_manifest))
    if (multiscale_tke or multiscale_energy) != (
            args.reference_result is not None):
        raise ValueError(
            "multiscale Real training requires a Dense12k reference")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    device = torch.device(args.device)
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
    if (reference["normalizer"].get("statistics_sha256")
            != smu1.statistics_digest(statistics)):
        raise RuntimeError("Dense12k normalizer differs from Dense4k")
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "point_normalizer.pt")

    schedule = smu1.paired_batch_schedule(
        range(len(dense_windows)), dense_windows,
        updates=UPDATES, seed=small4k.SEED)
    schedule["updates"] = UPDATES
    ft12.npt1.ensemble_train.seed_everything(small4k.SEED)
    expected_config = sim_tke.b210_config(
        "candidate", multiscale_energy_controller=multiscale_energy)
    expected_parameters = (
        sim_tke.EXPECTED_MULTISCALE_ENERGY_PARAMETERS
        if multiscale_energy else ft12.EXPECTED_PARAMETERS)
    expected_controller_parameters = (
        sim_tke.EXPECTED_MULTISCALE_ENERGY_CONTROLLER_PARAMETERS
        if multiscale_energy else ft12.EXPECTED_CONTROLLER_PARAMETERS)
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha, normalizer, device,
        expected_config=expected_config,
        expected_parameters=expected_parameters,
        expected_controller_parameters=expected_controller_parameters)
    supervisor = None
    supervisor_start = None
    supervisor_terminal = None
    if multiscale_tke:
        supervisor, supervisor_start = load_multiscale_tke_supervisor(
            args.multiscale_tke_supervisor,
            args.multiscale_tke_supervisor_sha256,
            start_sha, model, device)
    access = dense4k.ArmSeal("dense12k")
    with dense12k_training_contract(multiscale_tke, multiscale_energy):
        training, seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            args.output_dir, start_report, access,
            input_augmentation=dense4k.AUGMENTATION,
            endpoint_filename=ENDPOINT_FILENAME,
            full_backbone_auxiliary=supervisor,
            full_backbone_auxiliary_weight=(
                sim_tke.MULTISCALE_TKE_WEIGHT if multiscale_tke else 0.0),
            full_backbone_gradient_audit_update=(
                sim_tke.MULTISCALE_TKE_GRADIENT_AUDIT_UPDATE
                if multiscale_tke else 0),
            full_backbone_gradient_ratio_range=(
                sim_tke.MULTISCALE_TKE_GRADIENT_RATIO_RANGE
                if multiscale_tke else None),
            expected_parameters=expected_parameters)
        if access.value != seal:
            raise RuntimeError("Dense12k endpoint seal was not bound")
        if supervisor is not None:
            supervisor_path = (
                args.output_dir / "multiscale_tke_supervisor_u12000.pth")
            torch.save({
                "experiment": MULTISCALE_EXPERIMENT,
                "model_endpoint_sha256": seal["file_sha256"],
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in supervisor.state_dict().items()},
                "deployable": False,
            }, supervisor_path)
            supervisor_terminal = {
                "path": str(supervisor_path.resolve()),
                "sha256": dr1.sha256(supervisor_path),
                "deployable": False,
            }
        del dataset, trajectories, model
        if supervisor is not None:
            del supervisor
        gc.collect()
        torch.cuda.empty_cache()

        tail_audit = EndpointTailAudit(seal)
        tail_trajectories, tail_metadata, tail_windows = dr1.load_prefix_rows(
            args.real_dir, evaluation_rows, "held", tail_audit)
        tail_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
            tail_trajectories, tail_windows)
        evaluated = small4k.load_primary_endpoint(
            seal, normalizer, device,
            expected_model_config=expected_config,
            expected_parameters=expected_parameters,
            expected_controller_parameters=expected_controller_parameters)
        metrics, state, target_audit = smu1.evaluate_model(
            evaluated, tail_dataset, list(range(len(tail_windows))),
            normalizer, device, deadline)
        indices_by_aoa = dense4k.group_window_indices_by_aoa(
            tail_windows, tail_metadata)
        metrics_by_aoa = {}
        for aoa, indices in indices_by_aoa.items():
            aoa_metrics, _state, _audit = smu1.evaluate_model(
                evaluated, tail_dataset, indices, normalizer,
                device, deadline)
            metrics_by_aoa[str(aoa)] = aoa_metrics
        del evaluated, tail_dataset, tail_trajectories
        gc.collect()
        torch.cuda.empty_cache()

    if target_audit != reference["target_audit"]:
        raise RuntimeError("Dense12k/Dense4k clean716 target audit differs")
    comparison = compare(metrics, reference["metrics"])
    comparison["by_aoa"] = {
        str(aoa): compare(
            metrics_by_aoa[str(aoa)],
            reference["metrics_by_aoa"][str(aoa)])
        for aoa in dr1.AOA_VALUES
    }
    gate = release_gate(comparison)
    result = {
        "schema": (
            MULTISCALE_ENERGY_SCHEMA if multiscale_energy
            else MULTISCALE_SCHEMA if multiscale_tke else SCHEMA),
        "experiment": (
            MULTISCALE_ENERGY_EXPERIMENT if multiscale_energy
            else MULTISCALE_EXPERIMENT if multiscale_tke else EXPERIMENT),
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
            "kind": "sparse clean2544 prefix only",
            "statistics_sha256": smu1.statistics_digest(statistics),
            "path": str((args.output_dir / "point_normalizer.pt").resolve()),
        },
        "training": training,
        "training_only_multiscale_tke_supervisor": (
            None if not multiscale_tke else {
                "start": supervisor_start,
                "terminal": supervisor_terminal,
                "saved_in_deploy_endpoint": False,
            }),
        "deployed_multiscale_energy_controller": multiscale_energy,
        "seal": seal,
        "candidate": {
            "fit_windows": len(dense_windows),
            "window_stride": dense4k.CANDIDATE_STRIDE,
            "input_augmentation": dict(dense4k.AUGMENTATION),
            "metrics": metrics,
            "metric_state": state,
            "metrics_by_aoa": metrics_by_aoa,
            "target_audit": target_audit,
        },
        "reference": {
            "path": str(reference_path.resolve()),
            "kind": (
                "prior_dense12k_endpoint"
                if args.reference_result is not None
                else "same_start_dense4k_endpoint"),
            **reference,
        },
        "comparison": comparison,
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
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--start-sim-sha256")
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--multiscale-tke-supervisor", type=Path)
    parser.add_argument("--multiscale-tke-supervisor-sha256")
    parser.add_argument("--multiscale-energy-controller", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_realclosure_dense12k_complete",
        "candidate": result["candidate"]["metrics"],
        "delta": result["comparison"]["delta"],
        "gate": result["gate"],
        "result": str(args.output_dir / "result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
