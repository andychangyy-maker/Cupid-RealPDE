#!/usr/bin/env python3
"""Single-run raw-vs-EMA clean81 held-AoA10 gate for C208.

Only other-four-AoA prefixes are opened while the raw model is trained.  The
raw and FP32 EMA endpoints are both sealed before any held-AoA prefix is
opened.  This program never constructs legacy 2576/725 or clean716 windows.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fp32_ema_shadow as ema_lib  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_real_sim_rehearsal as c208  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_ensemble_unet2d as ensemble_train  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


EXPERIMENT = "c208_ema999_clean81_same_run_held10"
SCHEMA = "c208_ema999_clean81_held10_v1"
SEALED_SCHEMA = "c208_ema999_clean81_endpoint_v1"
FORMAL_START_SHA256 = c208.FORMAL_START_SIM_SHA256
HELD_AOA = 10
EMA_DECAY = 0.999
UPDATES = dr1.UPDATES
SEED = dr1.SEED


CONTRACT = {
    "evidence_class": "same-run raw-vs-EMA clean81 held-AoA10",
    "start_sim_sha256": FORMAL_START_SHA256,
    "held_aoa": HELD_AOA,
    "training": {
        "updates": UPDATES,
        "batch_size": 8,
        "seed": SEED,
        "lr": smu1.LR,
        "weight_decay": smu1.WEIGHT_DECAY,
        "scheduler_t_max": smu1.SCHEDULER_T_MAX,
        "grad_clip": smu1.GRAD_CLIP,
    },
    "ema": {
        "decay": EMA_DECAY,
        "updates": UPDATES,
        "update_order": ema_lib.UPDATE_ORDER_CONTRACT,
        "shadow_dtype": str(torch.float32),
    },
    "data": {
        "trajectories": dr1.EXPECTED_TRAJECTORIES,
        "prefix_windows": dr1.EXPECTED_WINDOWS,
        "manifest_sha256": dr1.FORMAL_MANIFEST_SHA256,
        "excluded": dr1.EXCLUDED_BASENAME,
        "tail_arrays_read": 0,
    },
    "gate": {
        "point_delta_strictly_positive": True,
        "exact_non_time_final_delta_strictly_positive": True,
        "rel_tke_mvpe_floor": -0.02,
        "sps_floor": -0.10,
    },
    "release_checkpoint_written": False,
    "submission_authorized": False,
}


def inspect_start_checkpoint(
        path: Path, expected_sha256: str,
        normalizer: smu1.FoldNormalizer, device: torch.device,
        ) -> tuple[EnsembleUNet2d, dict]:
    expected = dr1.normalized_sha256(expected_sha256, "start Sim checkpoint")
    actual = dr1.sha256(path)
    if actual != expected or actual != FORMAL_START_SHA256:
        raise RuntimeError("legacy C208 Sim checkpoint SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (not isinstance(payload, dict) or payload.get("phase") != "sim"
            or int(payload.get("iteration", -1)) != 10000):
        raise ValueError("start checkpoint is not terminal Sim10k")
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, Mapping):
        raise ValueError("start checkpoint lacks model config/state")
    stored = dict(config)
    if stored.pop("architecture", None) != "ensemble_unet":
        raise ValueError("start checkpoint is not an ensemble U-Net")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
    model.load_state_dict(state, strict=True)
    if sum(value.numel() for value in model.parameters()) != dr1.EXPECTED_PARAMETERS:
        raise RuntimeError("C208 parameter count changed")
    ensemble_train.common.set_model_stats(model, normalizer)
    report = {
        "path": str(path.resolve()),
        "checkpoint_sha256": actual,
        "model_state_sha256": dr1.model_state_sha256(state),
        "model_config_sha256": dr1.canonical_sha256(config),
        "iteration": 10000,
        "parameters": dr1.EXPECTED_PARAMETERS,
    }
    return model.to(device), report


def train_raw_with_ema(
        model: EnsembleUNet2d, dataset, schedule: dict,
        normalizer: smu1.FoldNormalizer, device: torch.device,
        deadline: float, bootstrap_masks: Mapping[int, Sequence[bool]],
        ) -> tuple[dict, ema_lib.FP32EMAShadow]:
    tagged = ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(
        tagged, batch_sampler=schedule["batches"], num_workers=0,
        pin_memory=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=smu1.LR, weight_decay=smu1.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=smu1.SCHEDULER_T_MAX)
    ema = ema_lib.FP32EMAShadow(model, EMA_DECAY)
    initial_raw_sha = dr1.model_state_sha256(model.state_dict())
    actual_rows = []
    history = []
    sums: dict[str, float] = {}
    count = 0
    started = time.perf_counter()
    for iteration, batch in enumerate(loader, start=1):
        if iteration > UPDATES:
            break
        if time.perf_counter() > deadline:
            raise TimeoutError("EMA999 held10 wall budget expired during training")
        expected = schedule["rows"][iteration - 1]
        trajectory_ids = batch[2].to(dtype=torch.long, device="cpu")
        sample_indices = batch[3].to(dtype=torch.long, device="cpu")
        try:
            mask = torch.as_tensor([
                bootstrap_masks[int(trajectory)]
                for trajectory in trajectory_ids.tolist()
            ], dtype=torch.bool)
        except KeyError as exc:
            raise RuntimeError(
                f"trajectory {exc.args[0]} has no frozen mask") from exc
        actual = {
            "sample_indices": sample_indices.tolist(),
            "trajectory_ids": trajectory_ids.tolist(),
            "bootstrap_mask": mask.to(torch.uint8).tolist(),
        }
        if actual != expected:
            raise RuntimeError(
                f"EMA999 sample stream mismatch at update {iteration}")
        actual_rows.append(actual)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = c208.backward_real_supervised(
            model, batch, normalizer, device, iteration,
            exact_physical_uq_target=True, member_mask=mask)
        torch.nn.utils.clip_grad_norm_(model.parameters(), smu1.GRAD_CLIP)
        optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=iteration)
        scheduler.step()
        values["learning_rate"] = optimizer.param_groups[0]["lr"]
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration % smu1.LOG_INTERVAL == 0:
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps({"event": "ema999_progress", **row}), flush=True)
            sums.clear()
            count = 0
    if len(actual_rows) != UPDATES or ema.updates != UPDATES:
        raise RuntimeError("EMA999 did not complete exactly 4000 ordered updates")
    actual_sha = dr1.canonical_sha256(actual_rows)
    if actual_sha != schedule["sha256"]:
        raise RuntimeError("EMA999 sample stream digest mismatch")
    return ({
        "updates": UPDATES,
        "initial_raw_model_state_sha256": initial_raw_sha,
        "final_raw_model_state_sha256": dr1.model_state_sha256(
            model.state_dict()),
        "actual_stream_sha256": actual_sha,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "ema_updates": ema.updates,
        "ema_decay": ema.decay,
        "ema_parameter_manifest_sha256": ema.parameter_manifest_sha256,
    }, ema)


def save_endpoint(
        path: Path, model: EnsembleUNet2d, arm: str, start: dict,
        normalizer: smu1.FoldNormalizer, stream_sha256: str,
        extra: dict | None = None,
        ) -> dict:
    state = ensemble_train.common.checkpoint_state(model)
    state_sha = dr1.model_state_sha256(state)
    payload = {
        "schema": SEALED_SCHEMA,
        "phase": "real",
        "iteration": UPDATES,
        "arm": arm,
        "held_aoa": HELD_AOA,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": model.architecture_config(),
        "start_sim_checkpoint_sha256": start["checkpoint_sha256"],
        "start_sim_model_state_sha256": start["model_state_sha256"],
        "sample_stream_sha256": stream_sha256,
        "normalizer": ensemble_train.common.normalizer_dict(normalizer),
        "held_prefix_opened_before_seal": False,
        "extra": extra or {},
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": state_sha,
        "iteration": UPDATES,
    }


def load_endpoint(
        path: Path, seal: dict, normalizer: smu1.FoldNormalizer,
        device: torch.device,
        ) -> EnsembleUNet2d:
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("sealed EMA999 endpoint file drifted")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SEALED_SCHEMA:
        raise ValueError("invalid EMA999 endpoint schema")
    state = payload["model_state_dict"]
    if (payload.get("model_state_sha256") != seal["model_state_sha256"]
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError("sealed EMA999 endpoint tensor state drifted")
    stored = dict(payload["model_config"])
    if stored.pop("architecture", None) != "ensemble_unet":
        raise ValueError("sealed EMA999 endpoint architecture changed")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
    model.load_state_dict(state, strict=True)
    ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def quick_gate(comparison: Mapping[str, object]) -> dict:
    delta = comparison["delta"]
    conditions = {
        "point_positive": float(delta["point_score"]) > 0.0,
        "exact_non_time_final_positive": (
            float(comparison["exact_non_time_final_delta"]) > 0.0),
        "rel_floor": float(delta["rel_l2_score"]) >= -0.02,
        "tke_floor": float(delta["tke_rel_l2_score"]) >= -0.02,
        "mvpe_floor": float(delta["mvpe_rel_l2_score"]) >= -0.02,
        "sps_floor": float(delta["native_sps_score"]) >= -0.10,
    }
    passed = all(conditions.values())
    return {
        "pass": passed,
        "conditions": conditions,
        "decision": (
            "GO_EMA999_FULL_CLEAN81_LOAO" if passed
            else "STOP_EMA999_HELD10"),
    }


def run(args: argparse.Namespace) -> dict:
    identity = dr1.validate_formal_git_identity(dr1.git_identity())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, args.clean_manifest_sha256)
    fit_rows = [row for row in rows if int(row["aoa"]) != HELD_AOA]
    held_rows = [row for row in rows if int(row["aoa"]) == HELD_AOA]
    if sum(len(row["train_starts"]) for row in fit_rows) != (
            dr1.EXPECTED_FIT_COUNTS[HELD_AOA]):
        raise RuntimeError("held10 other-four fit count changed")
    if sum(len(row["train_starts"]) for row in held_rows) != (
            dr1.EXPECTED_AOA_COUNTS[HELD_AOA]):
        raise RuntimeError("held10 sample count changed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal EMA999 held10 requires CUDA")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    access = dr1.PrefixAccessAudit(HELD_AOA)

    fit_trajectories, fit_metadata, fit_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", access)
    fit_dataset = ensemble_train.common.repro.WindowDataset(
        fit_trajectories, fit_windows)
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, fit_windows)
    statistics = smu1.fold_statistics(sufficient, range(len(fit_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    schedule = smu1.paired_batch_schedule(
        range(len(fit_windows)), fit_windows, updates=UPDATES, seed=SEED)
    frozen_masks = dr1.bootstrap_mask_by_trajectory(len(fit_trajectories))

    ensemble_train.seed_everything(SEED)
    model, start = inspect_start_checkpoint(
        args.start_sim, args.start_sim_sha256, normalizer, device)
    training, ema = train_raw_with_ema(
        model, fit_dataset, schedule, normalizer, device, deadline,
        frozen_masks)
    if training["actual_stream_sha256"] != schedule["sha256"]:
        raise RuntimeError("EMA999 Real sample stream changed")

    raw_path = args.output_dir / "raw_sealed_u4000.pth"
    raw_seal = save_endpoint(
        raw_path, model, "control", start, normalizer, schedule["sha256"])
    if raw_seal["model_state_sha256"] != (
            training["final_raw_model_state_sha256"]):
        raise RuntimeError("raw endpoint state differs from trained raw state")
    access.seal("control", raw_path, raw_seal["file_sha256"],
                raw_seal["model_state_sha256"])
    raw_sha_before_ema_seal = dr1.model_state_sha256(model.state_dict())
    ema_payload = ema.seal(model)
    raw_sha_after_ema_seal = dr1.model_state_sha256(model.state_dict())
    if raw_sha_before_ema_seal != raw_sha_after_ema_seal:
        raise RuntimeError("materializing EMA mutated the raw model")
    ema_state = ema_payload.pop("model_state")
    if (int(ema_payload["updates"]) != UPDATES
            or float(ema_payload["decay"]) != EMA_DECAY):
        raise RuntimeError("EMA seal counter/decay changed")
    if ema_lib.tensor_state_sha256(ema_state) != (
            ema_payload["materialized_model_state_sha256"]):
        raise RuntimeError("materialized EMA state hash changed before load")
    ema_model = EnsembleUNet2d(EnsembleUNet2dConfig(
        **{key: value for key, value in model.architecture_config().items()
           if key != "architecture"}))
    ema_model.load_state_dict(ema_state, strict=True)
    if ema_lib.tensor_state_sha256(ema_model.state_dict()) != (
            ema_payload["materialized_model_state_sha256"]):
        raise RuntimeError("strict-loaded EMA model differs from sealed state")
    ema_model = ema_model.to(device)
    ema_path = args.output_dir / "ema999_sealed_u4000.pth"
    ema_seal = save_endpoint(
        ema_path, ema_model, "candidate", start, normalizer,
        schedule["sha256"], extra={"ema": ema_payload})
    access.seal("candidate", ema_path, ema_seal["file_sha256"],
                ema_seal["model_state_sha256"])
    del model, ema_model, ema, ema_state
    torch.cuda.empty_cache()
    gc.collect()
    if not access.both_sealed:
        raise RuntimeError("raw and EMA endpoints were not both sealed")

    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", access)
    held_dataset = ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    evaluate_indices = list(range(len(held_windows)))
    reports = {}
    for arm, seal in (("control", raw_seal), ("candidate", ema_seal)):
        evaluated = load_endpoint(Path(seal["path"]), seal, normalizer, device)
        metrics, state, held_audit = smu1.evaluate_model(
            evaluated, held_dataset, evaluate_indices, normalizer,
            device, deadline)
        reports[arm] = {
            "seal": seal,
            "metrics": metrics,
            "metric_state": state,
            "held_target_audit": held_audit,
        }
        del evaluated
        torch.cuda.empty_cache()
        gc.collect()
    if reports["control"]["held_target_audit"] != (
            reports["candidate"]["held_target_audit"]):
        raise RuntimeError("raw and EMA held targets/masks differ")
    access_report = access.report([row["file"] for row in rows])
    if not (
            access_report["allowlist_exact"]
            and access_report["tail_arrays_read"] == 0
            and not access_report["excluded_opened"]
            and access_report["held_first_open_after_both_seals"]):
        raise RuntimeError("EMA999 prefix access audit failed")
    comparison = dr1.compare_metrics(
        reports["candidate"]["metrics"], reports["control"]["metrics"])
    gate = quick_gate(comparison)
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "contract": CONTRACT,
        "decision": gate["decision"],
        "gate": gate,
        "held_aoa": HELD_AOA,
        "fit_samples": len(fit_windows),
        "samples": len(held_windows),
        "manifest": manifest_report,
        "normalizer_statistics_sha256": smu1.statistics_digest(statistics),
        "sample_stream_sha256": schedule["sha256"],
        "fit_ordered_key_sha256": smu1.ordered_held_key_digest(
            range(len(fit_windows)), fit_windows, fit_metadata),
        "held_ordered_key_sha256": smu1.ordered_held_key_digest(
            range(len(held_windows)), held_windows, held_metadata),
        "start": start,
        "training": training,
        "ema_contract": ema_payload,
        "raw_model_unchanged_by_ema_seal": (
            raw_sha_before_ema_seal == raw_sha_after_ema_seal),
        "control": reports["control"],
        "candidate": reports["candidate"],
        "comparison": comparison,
        "prefix_access_audit": access_report,
        "git": identity,
        "elapsed_seconds": time.perf_counter() - started,
        "release_checkpoint_written": False,
        "submission_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument(
        "--clean-manifest-sha256", default=dr1.FORMAL_MANIFEST_SHA256)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256", default=FORMAL_START_SHA256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3600.0)
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.contract_only:
        print(json.dumps({
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "contract": CONTRACT,
            "decision": "CONTRACT_ONLY_NO_DATA_NO_GPU",
        }, sort_keys=True))
        return
    if args.clean_manifest is None or args.start_sim is None:
        raise ValueError("clean manifest and legacy Sim checkpoint are required")
    if (not math.isfinite(args.wall_budget_seconds)
            or args.wall_budget_seconds <= 0):
        raise ValueError("wall budget must be positive and finite")
    try:
        summary = run(args)
        dr1.write_json_atomic(args.output_dir / "result.json", summary)
        print(json.dumps({
            "event": "c208_ema999_held10_complete",
            "decision": summary["decision"],
        }), flush=True)
    except Exception as exc:
        stop = {
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "decision": "STOP_EMA999_INFRASTRUCTURE_OR_CONTRACT",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "release_checkpoint_written": False,
            "submission_authorized": False,
        }
        if args.output_dir.exists():
            dr1.write_json_atomic(args.output_dir / "result.json", stop)
        print(json.dumps(stop, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
