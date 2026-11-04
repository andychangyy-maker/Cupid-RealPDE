#!/usr/bin/env python3
"""Compare exact and denser augmented Real training from one scratch B210 Sim.

Both arms start from the same terminal Sim10k checkpoint.  Training may only
open the clean81 prefix frames.  The fixed clean716 tail is opened after both
EMA-u4000 endpoints have been written and sealed, and is used once for a
paired research comparison rather than endpoint selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_np_functional_sim as sim_tke
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1


EXPERIMENT = "b210_scratch_real_dense_aug"
SCHEMA = "b210_scratch_real_dense_aug_v1"
HORIZON_LOCAL_EXPERIMENT = "b210_horizon_local_real_dense_aug"
HORIZON_LOCAL_SCHEMA = "b210_horizon_local_real_dense_aug_v1"
MULTISCALE_ENERGY_EXPERIMENT = "b210_multiscale_energy_real_dense_aug"
MULTISCALE_ENERGY_SCHEMA = "b210_multiscale_energy_real_dense_aug_v1"
CONTROL_STRIDE = 20
CANDIDATE_STRIDE = 5
AUGMENTATION = {
    "input_gain_std": 0.01,
    "input_noise_std": 0.01,
    "input_blur_probability": 0.15,
    "input_blur_strength": 0.20,
}
TAIL_WINDOWS = 716


def arm_augmentation_contract() -> dict[str, dict[str, float]]:
    """Return the explicit input augmentation assigned to each paired arm."""
    return {"control": {}, "candidate": dict(AUGMENTATION)}


def arm_window_contract(
        trajectories: Sequence[torch.Tensor],
        manifest_windows: Sequence,
        ) -> dict[str, list]:
    """Assign the frozen manifest and dense windows to the default arms."""
    return {
        "control": list(manifest_windows),
        "candidate": dense_windows(trajectories, CANDIDATE_STRIDE),
    }


def architecture_contract(
        horizon_local_energy_controller: bool,
        multiscale_energy_controller: bool = False,
        ) -> tuple:
    """Return the exact model/parameter contract for the selected Sim start."""
    if horizon_local_energy_controller and multiscale_energy_controller:
        raise ValueError("select only one deployed energy-controller variant")
    config = sim_tke.b210_config(
        "candidate",
        multiscale_energy_controller=multiscale_energy_controller,
        horizon_local_energy_controller=horizon_local_energy_controller)
    if horizon_local_energy_controller:
        return (
            config,
            sim_tke.EXPECTED_HORIZON_LOCAL_ENERGY_PARAMETERS,
            sim_tke.EXPECTED_HORIZON_LOCAL_ENERGY_CONTROLLER_PARAMETERS,
        )
    if multiscale_energy_controller:
        return (
            config,
            sim_tke.EXPECTED_MULTISCALE_ENERGY_PARAMETERS,
            sim_tke.EXPECTED_MULTISCALE_ENERGY_CONTROLLER_PARAMETERS,
        )
    return (
        config,
        ft12.EXPECTED_PARAMETERS,
        ft12.EXPECTED_CONTROLLER_PARAMETERS,
    )


class PrefixAudit:
    def __init__(self) -> None:
        self.opened: list[str] = []

    def before_open(self, row: dict, phase: str) -> None:
        if phase != "fit" or row["file"] == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("invalid clean-prefix open")
        if row["file"] in self.opened:
            raise RuntimeError("clean prefix opened twice")

    def record_open(self, row: dict, phase: str) -> None:
        if phase != "fit" or row["file"] in self.opened:
            raise RuntimeError("invalid or duplicate clean-prefix record")
        self.opened.append(str(row["file"]))


class ArmSeal:
    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.value: dict | None = None

    def seal(self, value: Mapping[str, object]) -> None:
        if self.value is not None:
            raise RuntimeError(f"{self.arm} endpoint sealed twice")
        self.value = dict(value)


class TailAudit:
    def __init__(self, seals: Mapping[str, Mapping[str, object]]) -> None:
        if set(seals) != {"control", "candidate"}:
            raise ValueError("both endpoint seals are required before tail open")
        self.seals = {key: dict(value) for key, value in seals.items()}
        self.opened: list[str] = []

    def before_open(self, row: dict, phase: str) -> None:
        if phase != "held" or row["file"] == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("invalid clean-tail open")
        for seal in self.seals.values():
            path = Path(str(seal["path"]))
            if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
                raise RuntimeError("endpoint changed before tail open")

    def record_open(self, row: dict, phase: str) -> None:
        if phase != "held" or row["file"] in self.opened:
            raise RuntimeError("invalid or duplicate clean-tail record")
        self.opened.append(str(row["file"]))


def dense_windows(trajectories: Sequence[torch.Tensor], stride: int) -> list:
    if stride <= 0:
        raise ValueError("dense stride must be positive")
    windows = []
    window_type = ft12.npt1.ensemble_train.common.repro.Window
    for trajectory, values in enumerate(trajectories):
        windows.extend(
            window_type(trajectory, start)
            for start in range(0, int(values.shape[0]) - 40 + 1, stride))
    return windows


def load_evaluation_rows(path: Path, fit_rows: Sequence[dict]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != len(fit_rows):
        raise ValueError("evaluation manifest does not cover clean81")
    fit_by_file = {str(row["file"]): row for row in fit_rows}
    result = []
    for raw in payload:
        if not isinstance(raw, dict) or raw.get("file") not in fit_by_file:
            raise ValueError("evaluation manifest membership changed")
        fit = fit_by_file[str(raw["file"])]
        for key in ("frames", "re", "aoa", "val_boundary", "train_starts"):
            if raw.get(key) != fit.get(key):
                raise ValueError(
                    f"evaluation manifest {key} differs for {raw.get('file')}")
        starts = raw.get("val_starts")
        if (not isinstance(starts, list) or not starts
                or any(not isinstance(value, int) for value in starts)
                or any(value < int(fit["val_boundary"])
                       or value + 40 > int(fit["frames"])
                       for value in starts)):
            raise ValueError("evaluation starts are not a valid fixed tail")
        result.append({
            **fit,
            "train_starts": list(starts),
            "val_boundary": int(fit["frames"]),
        })
    if sum(len(row["train_starts"]) for row in result) != TAIL_WINDOWS:
        raise RuntimeError("fixed clean tail is not 716 windows")
    return result


def compare(candidate: Mapping[str, float],
            control: Mapping[str, float]) -> dict:
    keys = (
        "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
        "point_score", "native_sps_score", "native_sps_coverage",
        "native_sps_mean_width",
    )
    delta = {key: float(candidate[key]) - float(control[key]) for key in keys}
    return {
        "control": dict(control),
        "candidate": dict(candidate),
        "delta": delta,
        "candidate_point_improved": delta["point_score"] > 0.0,
        "candidate_non_time_proxy_delta": (
            0.5 * delta["rel_l2_score"]
            + 0.1 * delta["tke_rel_l2_score"]
            + 0.1 * delta["mvpe_rel_l2_score"]
            + 0.2 * delta["native_sps_score"]),
    }


def group_window_indices_by_aoa(windows: Sequence,
                                metadata: Sequence[Mapping[str, object]],
                                ) -> dict[int, list[int]]:
    """Partition evaluation windows by trajectory AoA without reordering."""
    grouped = {aoa: [] for aoa in dr1.AOA_VALUES}
    for index, window in enumerate(windows):
        trajectory = int(window.trajectory)
        if trajectory < 0 or trajectory >= len(metadata):
            raise IndexError("evaluation window trajectory is out of range")
        aoa = int(metadata[trajectory]["aoa"])
        if aoa not in grouped:
            raise ValueError(f"unexpected evaluation AoA {aoa}")
        grouped[aoa].append(index)
    if sum(map(len, grouped.values())) != len(windows):
        raise RuntimeError("AoA groups do not cover all evaluation windows")
    if any(not indices for indices in grouped.values()):
        raise RuntimeError("an expected AoA has no evaluation windows")
    return grouped


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 prefix manifest byte identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = load_evaluation_rows(args.evaluation_manifest, rows)
    start_sha = dr1.sha256(args.start_sim)
    if args.start_sim_sha256 and start_sha != args.start_sim_sha256.lower():
        raise RuntimeError("scratch Sim checkpoint SHA changed")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("paired scratch Real experiment requires CUDA")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    horizon_local = bool(args.horizon_local_energy_controller)
    multiscale_energy = bool(args.multiscale_energy_controller)
    expected_config, expected_parameters, expected_controller_parameters = (
        architecture_contract(horizon_local, multiscale_energy))
    prefix_audit = PrefixAudit()
    trajectories, metadata, manifest_windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", prefix_audit)
    if len(manifest_windows) != dr1.EXPECTED_WINDOWS:
        raise RuntimeError("control prefix window count changed")
    # Preserve the historical standard-prefix normalizer even when a research
    # wrapper assigns a different (but released-data-only) window stream.
    sufficient = smu1.window_sufficient_statistics(
        trajectories, manifest_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(manifest_windows)))
    normalizer = smu1.FoldNormalizer(statistics, torch.device(args.device))
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "point_normalizer.pt")

    windows_by_arm = arm_window_contract(trajectories, manifest_windows)
    if set(windows_by_arm) != {"control", "candidate"}:
        raise RuntimeError("paired arm window contract is incomplete")
    datasets = {
        arm: ft12.npt1.ensemble_train.common.repro.WindowDataset(
            trajectories, windows)
        for arm, windows in windows_by_arm.items()
    }
    augmentation_by_arm = arm_augmentation_contract()
    if set(augmentation_by_arm) != {"control", "candidate"}:
        raise RuntimeError("paired arm augmentation contract is incomplete")
    reports: dict[str, dict] = {}
    seals: dict[str, dict] = {}
    for arm in ("control", "candidate"):
        ft12.npt1.ensemble_train.seed_everything(small4k.SEED)
        model, start_report = ft12.validate_start_checkpoint(
            args.start_sim, start_sha, normalizer, torch.device(args.device),
            expected_config=expected_config,
            expected_parameters=expected_parameters,
            expected_controller_parameters=expected_controller_parameters)
        windows = windows_by_arm[arm]
        schedule = smu1.paired_batch_schedule(
            range(len(windows)), windows, updates=small4k.UPDATES,
            seed=small4k.SEED)
        schedule["updates"] = small4k.UPDATES
        arm_dir = args.output_dir / arm
        arm_dir.mkdir()
        access = ArmSeal(arm)
        training, seal = small4k.train_candidate(
            model, datasets[arm], schedule, normalizer,
            torch.device(args.device), deadline, arm_dir, start_report, access,
            input_augmentation=augmentation_by_arm[arm],
            expected_parameters=expected_parameters)
        if access.value != seal:
            raise RuntimeError(f"{arm} endpoint seal was not bound")
        reports[arm] = {
            "fit_windows": len(windows),
            "window_stride": (
                CONTROL_STRIDE if arm == "control" else CANDIDATE_STRIDE),
            "input_augmentation": augmentation_by_arm[arm],
            "training": training,
            "seal": seal,
        }
        seals[arm] = seal
        del model
        gc.collect()
        torch.cuda.empty_cache()

    del datasets, trajectories
    gc.collect()
    tail_audit = TailAudit(seals)
    tail_trajectories, tail_metadata, tail_windows = dr1.load_prefix_rows(
        args.real_dir, evaluation_rows, "held", tail_audit)
    tail_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        tail_trajectories, tail_windows)
    indices_by_aoa = group_window_indices_by_aoa(
        tail_windows, tail_metadata)
    for arm in ("control", "candidate"):
        seal = seals[arm]
        model = small4k.load_primary_endpoint(
            seal, normalizer, torch.device(args.device),
            expected_model_config=expected_config,
            expected_parameters=expected_parameters,
            expected_controller_parameters=expected_controller_parameters)
        metrics, state, target_audit = smu1.evaluate_model(
            model, tail_dataset, list(range(len(tail_windows))), normalizer,
            torch.device(args.device), deadline)
        reports[arm].update({
            "metrics": metrics,
            "metric_state": state,
            "target_audit": target_audit,
        })
        metrics_by_aoa = {}
        for aoa, indices in indices_by_aoa.items():
            aoa_metrics, _aoa_state, _aoa_target_audit = (
                smu1.evaluate_model(
                    model, tail_dataset, indices, normalizer,
                    torch.device(args.device), deadline))
            metrics_by_aoa[str(aoa)] = aoa_metrics
        reports[arm]["metrics_by_aoa"] = metrics_by_aoa
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if reports["control"]["target_audit"] != reports["candidate"]["target_audit"]:
        raise RuntimeError("paired clean716 target audit differs")
    comparison = compare(
        reports["candidate"]["metrics"], reports["control"]["metrics"])
    comparison_by_aoa = {
        str(aoa): compare(
            reports["candidate"]["metrics_by_aoa"][str(aoa)],
            reports["control"]["metrics_by_aoa"][str(aoa)])
        for aoa in dr1.AOA_VALUES
    }
    comparison["by_aoa"] = comparison_by_aoa
    comparison["candidate_positive_aoas"] = sum(
        row["candidate_non_time_proxy_delta"] > 0.0
        for row in comparison_by_aoa.values())
    if not math.isfinite(comparison["candidate_non_time_proxy_delta"]):
        raise FloatingPointError("paired comparison is non-finite")
    result = {
        "schema": (
            HORIZON_LOCAL_SCHEMA if horizon_local else
            MULTISCALE_ENERGY_SCHEMA if multiscale_energy else SCHEMA),
        "experiment": (
            HORIZON_LOCAL_EXPERIMENT if horizon_local else
            MULTISCALE_ENERGY_EXPERIMENT if multiscale_energy else EXPERIMENT),
        "start": {
            "checkpoint": str(args.start_sim.resolve()),
            "checkpoint_sha256": start_sha,
            "from_random_initialization_sim10k": True,
        },
        "manifest": manifest_report,
        "evaluation_manifest": {
            "path": str(args.evaluation_manifest.resolve()),
            "sha256": dr1.sha256(args.evaluation_manifest),
            "tail_windows": len(tail_windows),
        },
        "normalizer": {
            "kind": "control clean2544 prefix only",
            "statistics_sha256": smu1.statistics_digest(statistics),
            "path": str((args.output_dir / "point_normalizer.pt").resolve()),
        },
        "horizon_local_energy_controller": horizon_local,
        "multiscale_energy_controller": multiscale_energy,
        "parameters": expected_parameters,
        "controller_parameters": expected_controller_parameters,
        "arms": reports,
        "comparison": comparison,
        "access_audit": {
            "prefix_opened": prefix_audit.opened,
            "tail_opened_after_both_seals": tail_audit.opened,
            "tail_target_read_before_both_seals": False,
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
    parser.add_argument(
        "--horizon-local-energy-controller", action="store_true",
        help=("Require the passed horizon-local Sim10k architecture and keep "
              "that deployed controller trainable through both Real4k arms."))
    parser.add_argument(
        "--multiscale-energy-controller", action="store_true",
        help=("Require the passed multiscale-energy Sim10k architecture and "
              "keep its deployed controller trainable in both Real4k arms."))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_scratch_real_dense_aug_complete",
        "comparison": result["comparison"],
        "result": str(args.output_dir / "result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
