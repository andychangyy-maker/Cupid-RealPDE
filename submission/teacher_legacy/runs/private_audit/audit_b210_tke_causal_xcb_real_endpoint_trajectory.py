#!/usr/bin/env python3
"""Audit when causal-XCB Sim gains disappear during fixed Real4k adaptation.

The standard and causal Sim24k starts receive the exact same dense Real stream,
augmentation, optimizer, scheduler and EMA.  Intermediate EMA states remain in
CPU memory.  The clean716 research tail is opened only after both u4000
endpoints are sealed, so this script cannot use a held metric to affect either
training trajectory.  The result is diagnostic only and cannot select a
submission endpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
        REPO_ROOT,
        REPO_ROOT / "runs" / "private_audit",
        REPO_ROOT / "code",
        REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_real_small4k_confirm_held0 as small4k  # noqa: E402
import train_b210_scratch_real_dense_aug as dense4k  # noqa: E402
import train_b210_self_describing_real_dense_aug as self_describing  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d  # noqa: E402


SCHEMA = "b210_tke_causal_xcb_real_endpoint_trajectory_audit_v1"
EXPERIMENT = "b210_tke_causal_xcb_real_endpoint_trajectory_audit"
SNAPSHOT_UPDATES = (500, 1_000, 1_500, 2_000, 3_000)
TERMINAL_UPDATE = 4_000
ARMS = ("standard", "causal")
EXPECTED = {
    "standard": {
        "status": "PASS_B210_XCB_MEMBER_STANDARD24K_SIM",
        "terminal_model_state_sha256": (
            "285454cd7a7fcd0d08b18aef4f33d6c706444cc758707e7615e4226de288ec87"),
        "terminal_metrics": {
            "rel_l2_score": 96.09728433814325,
            "tke_rel_l2_score": 81.55927780127305,
            "mvpe_rel_l2_score": 96.73793213001692,
            "native_sps_score": 53.65558502067784,
        },
    },
    "causal": {
        "status": "PASS_B210_TKE_CAUSAL_XCB_MEMBER_SIM24K",
        "terminal_model_state_sha256": (
            "0f4bc9743b2cb68ea36fadfafcf0a80d6bf395a3277ca7c066a5db2d5c68ad1e"),
        "terminal_metrics": {
            "rel_l2_score": 96.11830596433195,
            "tke_rel_l2_score": 81.53111767666195,
            "mvpe_rel_l2_score": 96.7675748846564,
            "native_sps_score": 53.611457809516395,
        },
    },
}
TERMINAL_REPRODUCTION_TOLERANCE = {
    "rel_l2_score": 0.02,
    "tke_rel_l2_score": 0.02,
    "mvpe_rel_l2_score": 0.02,
    "native_sps_score": 0.05,
}


def official_delta(candidate: Mapping[str, float],
                   control: Mapping[str, float]) -> dict[str, float]:
    delta = {
        key: float(candidate[key]) - float(control[key])
        for key in (
            "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
            "native_sps_score")
    }
    point65 = (
        0.46 * delta["rel_l2_score"]
        + 0.10 * delta["tke_rel_l2_score"]
        + 0.09 * delta["mvpe_rel_l2_score"])
    return {
        **delta,
        "official_point65_delta": point65,
        "official_non_time_delta": (
            point65 + 0.25 * delta["native_sps_score"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--standard-sim", type=Path, required=True)
    parser.add_argument("--standard-sim-sha256", required=True)
    parser.add_argument("--causal-sim", type=Path, required=True)
    parser.add_argument("--causal-sim-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 3600)
    return parser.parse_args()


def architecture_contract(paths: Mapping[str, Path]) -> tuple:
    contracts = {}
    for arm, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        contracts[arm] = self_describing.architecture_from_payload(payload)
        del payload
        gc.collect()
    first = contracts[ARMS[0]]
    for arm in ARMS[1:]:
        if contracts[arm] != first:
            raise RuntimeError(f"{arm} architecture differs from standard")
    return first


def evaluate_state(
        state: Mapping[str, torch.Tensor], config, dataset, indices,
        normalizer, device: torch.device, deadline: float,
        ) -> tuple[dict, dict]:
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    model.to(device)
    metrics, _metric_state, target_audit = dense4k.smu1.evaluate_model(
        model, dataset, indices, normalizer, device, deadline)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, target_audit


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("endpoint trajectory audit requires CUDA")
    if dense4k.dr1.sha256(args.clean_manifest) != (
            dense4k.ft12.FORMAL_MANIFEST_FILE_SHA256):
        raise RuntimeError("clean81 prefix manifest byte identity changed")
    paths = {
        "standard": args.standard_sim,
        "causal": args.causal_sim,
    }
    hashes = {
        "standard": args.standard_sim_sha256.lower(),
        "causal": args.causal_sim_sha256.lower(),
    }
    config, expected_parameters, expected_controller_parameters = (
        architecture_contract(paths))
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")

    rows, manifest_report = dense4k.dr1.load_manifest(
        args.clean_manifest, dense4k.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = dense4k.load_evaluation_rows(
        args.evaluation_manifest, rows)
    prefix_audit = dense4k.PrefixAudit()
    trajectories, _metadata, sparse_windows = dense4k.dr1.load_prefix_rows(
        args.real_dir, rows, "fit", prefix_audit)
    dense_windows = dense4k.dense_windows(
        trajectories, dense4k.CANDIDATE_STRIDE)
    dataset = dense4k.ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, dense_windows)
    sufficient = dense4k.smu1.window_sufficient_statistics(
        trajectories, sparse_windows)
    statistics = dense4k.smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = dense4k.smu1.FoldNormalizer(statistics, device)
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "point_normalizer.pt")
    schedule = dense4k.smu1.paired_batch_schedule(
        range(len(dense_windows)), dense_windows,
        updates=TERMINAL_UPDATE, seed=small4k.SEED)
    schedule["updates"] = TERMINAL_UPDATE

    snapshots: dict[str, dict[int, Mapping[str, torch.Tensor]]] = {
        arm: {} for arm in ARMS}
    seals = {}
    state_reproduction = {}
    training_reports = {}
    for arm in ARMS:
        dense4k.ft12.npt1.ensemble_train.seed_everything(small4k.SEED)
        model, start_report = dense4k.ft12.validate_start_checkpoint(
            paths[arm], hashes[arm], normalizer, device,
            expected_config=config,
            expected_parameters=expected_parameters,
            expected_controller_parameters=expected_controller_parameters,
            expected_iteration=TERMINAL_UPDATE,
            expected_status=EXPECTED[arm]["status"])
        access = dense4k.ArmSeal(arm)

        def retain(update: int, state: Mapping[str, torch.Tensor],
                   *, selected_arm: str = arm) -> None:
            if update in snapshots[selected_arm]:
                raise RuntimeError("EMA trajectory snapshot duplicated")
            snapshots[selected_arm][update] = state

        arm_dir = args.output_dir / arm
        arm_dir.mkdir()
        training, seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            arm_dir, start_report, access,
            input_augmentation=dense4k.AUGMENTATION,
            expected_parameters=expected_parameters,
            ema_snapshot_updates=SNAPSHOT_UPDATES,
            ema_snapshot_sink=retain)
        if access.value != seal:
            raise RuntimeError(f"{arm} terminal endpoint was not sealed")
        # CUDA convolution/FFT kernels are not guaranteed bitwise identical
        # across independent processes.  Preserve both hashes as evidence and
        # defer the hard reproduction gate to the scored terminal metrics.
        state_reproduction[arm] = {
            "reference_model_state_sha256": (
                EXPECTED[arm]["terminal_model_state_sha256"]),
            "observed_model_state_sha256": seal["model_state_sha256"],
            "bitwise_equal": (
                seal["model_state_sha256"]
                == EXPECTED[arm]["terminal_model_state_sha256"]),
        }
        if tuple(snapshots[arm]) != SNAPSHOT_UPDATES:
            raise RuntimeError(f"{arm} EMA snapshot sequence is incomplete")
        training_reports[arm] = training
        seals[arm] = seal
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # No held array is opened until both deterministic u4000 states have been
    # reproduced and sealed above.
    del dataset, trajectories
    gc.collect()
    tail_audit = dense4k.TailAudit({
        "control": seals["standard"],
        "candidate": seals["causal"],
    })
    tail_trajectories, _tail_metadata, tail_windows = (
        dense4k.dr1.load_prefix_rows(
            args.real_dir, evaluation_rows, "held", tail_audit))
    tail_dataset = dense4k.ft12.npt1.ensemble_train.common.repro.WindowDataset(
        tail_trajectories, tail_windows)
    indices = list(range(len(tail_windows)))
    metrics: dict[str, dict[str, dict]] = {arm: {} for arm in ARMS}
    audits = []
    for arm in ARMS:
        for update in SNAPSHOT_UPDATES:
            values, audit = evaluate_state(
                snapshots[arm].pop(update), config, tail_dataset, indices,
                normalizer, device, deadline)
            metrics[arm][str(update)] = values
            audits.append(audit)
        terminal = torch.load(
            seals[arm]["path"], map_location="cpu", weights_only=False)
        values, audit = evaluate_state(
            terminal["model_state_dict"], config, tail_dataset, indices,
            normalizer, device, deadline)
        metrics[arm][str(TERMINAL_UPDATE)] = values
        audits.append(audit)
        del terminal
        gc.collect()
    if any(value != audits[0] for value in audits[1:]):
        raise RuntimeError("trajectory endpoint target audits differ")
    comparisons = {
        str(update): official_delta(
            metrics["causal"][str(update)],
            metrics["standard"][str(update)])
        for update in (*SNAPSHOT_UPDATES, TERMINAL_UPDATE)
    }
    metric_reproduction = {}
    for arm in ARMS:
        observed = metrics[arm][str(TERMINAL_UPDATE)]
        reference = EXPECTED[arm]["terminal_metrics"]
        absolute_delta = {
            key: abs(float(observed[key]) - float(reference[key]))
            for key in TERMINAL_REPRODUCTION_TOLERANCE
        }
        passed = all(
            absolute_delta[key] <= TERMINAL_REPRODUCTION_TOLERANCE[key]
            for key in absolute_delta)
        metric_reproduction[arm] = {
            "reference": dict(reference),
            "observed": {
                key: float(observed[key])
                for key in TERMINAL_REPRODUCTION_TOLERANCE},
            "absolute_delta": absolute_delta,
            "tolerance": dict(TERMINAL_REPRODUCTION_TOLERANCE),
            "pass": passed,
        }
        if not passed:
            raise RuntimeError(
                f"{arm} u4000 terminal metrics do not reproduce history")
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "diagnostics_cannot_select_endpoint": True,
        "submission_authorized": False,
        "starts": {
            arm: {
                "path": str(paths[arm].resolve()),
                "sha256": hashes[arm],
                "required_status": EXPECTED[arm]["status"],
            }
            for arm in ARMS
        },
        "protocol": {
            "updates": TERMINAL_UPDATE,
            "ema_snapshot_updates": list(SNAPSHOT_UPDATES),
            "snapshots_persisted_to_disk": False,
            "held_opened_after_both_terminal_seals": True,
            "window_stride": dense4k.CANDIDATE_STRIDE,
            "input_augmentation": dict(dense4k.AUGMENTATION),
            "sample_stream_sha256": schedule["sha256"],
        },
        "manifest": manifest_report,
        "evaluation_manifest": {
            "path": str(args.evaluation_manifest.resolve()),
            "sha256": dense4k.dr1.sha256(args.evaluation_manifest),
            "windows": len(tail_windows),
        },
        "training": training_reports,
        "seals": seals,
        "terminal_state_reproduction": state_reproduction,
        "terminal_metric_reproduction": metric_reproduction,
        "metrics": metrics,
        "causal_minus_standard": comparisons,
        "target_audit": audits[0],
        "access_audit": {
            "prefix_opened": prefix_audit.opened,
            "tail_opened_after_both_seals": tail_audit.opened,
            "tail_target_read_before_both_seals": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    dense4k.dr1.write_json_atomic(args.output_dir / "result.json", result)
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_tke_causal_real_endpoint_trajectory_complete",
        "causal_minus_standard": result["causal_minus_standard"],
        "result": str((args.output_dir / "result.json").resolve()),
    }), flush=True)


if __name__ == "__main__":
    main()
