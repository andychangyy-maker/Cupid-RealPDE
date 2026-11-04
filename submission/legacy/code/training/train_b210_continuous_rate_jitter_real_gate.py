#!/usr/bin/env python3
"""Screen coherent continuous-time rate jitter on condition-held Real data.

Both arms start from the same SHA-bound Standard24k Sim checkpoint, consume
the same dense source-window and bootstrap streams, and use the same ordinary
input augmentation.  The only difference is that the candidate resamples each
complete 40-frame history/target sequence at a deterministic temporal rate.
The held AoA is opened only after both EMA endpoints have been sealed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import time
from typing import Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset

import train_b210_condition_held_real as held
import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_np_functional_sim as sim_tke
import train_b210_scratch_real_dense_aug as dense4k
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1


EXPERIMENT = "b210_continuous_rate_jitter_real_gate"
SCHEMA = "b210_continuous_rate_jitter_real_gate_fold_v1"
AGGREGATE_SCHEMA = "b210_continuous_rate_jitter_real_gate_aggregate_v1"
ENDPOINT_SCHEMA = "b210_continuous_rate_jitter_real_gate_endpoint_v1"
ARMS = ("control", "rate_jitter")
ENDPOINT = {
    "control": "ema_u1500_control",
    "rate_jitter": "ema_u1500_rate_jitter",
}
UPDATES = 1_500
SCHEDULER_T_MAX = 12_000
WINDOW_FRAMES = 40
SOURCE_FRAMES = 44
WINDOW_STRIDE = 5
RATE_NUMERATORS = (10, 9, 11)
RATE_DENOMINATOR = 10
RATE_NAMES = ("identity_1.0", "slow_0.9", "fast_1.1")
RATE_SEED = 41_027
EXPECTED_SIM_ITERATION = 4_000
EXPECTED_SIM_STATUS = "PASS_B210_STANDARD24K_MATCHED_COMPUTE_SIM4000"
EXPECTED_STANDARD24K_SHA256 = (
    "4b56454ce64e301a9898dc1c066d23180de587cbebc7c53786fd3ca126981e68")
DISCOVERY_POINT_MIN = 0.04
POOLED_POINT_MIN = 0.12
SAFETY_METRIC_FLOOR = -0.50


def rate_capable_windows(
        trajectories: Sequence[torch.Tensor], stride: int = WINDOW_STRIDE,
        ) -> list:
    """Return a common inventory with all 44 source frames available."""
    if stride <= 0:
        raise ValueError("rate-jitter stride must be positive")
    window_type = ft12.npt1.ensemble_train.common.repro.Window
    windows = []
    for trajectory, values in enumerate(trajectories):
        frames = int(values.shape[0])
        windows.extend(
            window_type(trajectory, start)
            for start in range(0, frames - SOURCE_FRAMES + 1, stride))
    if not windows:
        raise ValueError("rate-jitter inventory is empty")
    return windows


def interpolation_plan(rate_code: int) -> tuple[torch.Tensor, ...]:
    """Build exact rational floor/ceil/weight vectors for one rate."""
    if rate_code < 0 or rate_code >= len(RATE_NUMERATORS):
        raise ValueError(f"unknown rate code {rate_code}")
    numerator = RATE_NUMERATORS[rate_code]
    step = torch.arange(WINDOW_FRAMES, dtype=torch.long)
    scaled = step * numerator
    lower = torch.div(scaled, RATE_DENOMINATOR, rounding_mode="floor")
    remainder = scaled.remainder(RATE_DENOMINATOR)
    upper = lower + remainder.ne(0).to(torch.long)
    weight = remainder.to(torch.float32).div(float(RATE_DENOMINATOR))
    if int(upper.max()) >= SOURCE_FRAMES:
        raise RuntimeError("rate interpolation exceeds its source inventory")
    return lower, upper, weight


INTERPOLATION_PLANS = tuple(
    interpolation_plan(code) for code in range(len(RATE_NUMERATORS)))


class ScheduledContinuousRateDataset(Dataset):
    """Materialize a deterministic per-occurrence rate schedule without RNG."""

    def __init__(
            self, trajectories: Sequence[torch.Tensor],
            source_windows: Sequence, source_indices: Sequence[int],
            rate_codes: Sequence[int]) -> None:
        if len(source_indices) != len(rate_codes) or not source_indices:
            raise ValueError("source-index and rate schedules must align")
        if any(index < 0 or index >= len(source_windows)
               for index in source_indices):
            raise IndexError("scheduled source window is out of range")
        if any(code < 0 or code >= len(RATE_NUMERATORS)
               for code in rate_codes):
            raise ValueError("scheduled rate code is invalid")
        self.trajectories = trajectories
        self.source_windows = source_windows
        self.source_indices = tuple(int(value) for value in source_indices)
        self.rate_codes = tuple(int(value) for value in rate_codes)
        # TrajectoryTaggedDataset reads this attribute to bind every occurrence
        # to its trajectory.  Duplicates are intentional and schedule-specific.
        self.windows = [source_windows[index] for index in self.source_indices]
        for window in self.windows:
            available = int(trajectories[window.trajectory].shape[0]) - int(
                window.start)
            if available < SOURCE_FRAMES:
                raise ValueError("scheduled window lacks 44 source frames")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        trajectory = self.trajectories[window.trajectory]
        source = trajectory[window.start:window.start + SOURCE_FRAMES]
        rate_code = self.rate_codes[index]
        if rate_code == 0:
            # Preserve the historical identity arm bit-for-bit.
            uv = source[:WINDOW_FRAMES]
        else:
            lower, upper, weight = INTERPOLATION_PLANS[rate_code]
            shape = (WINDOW_FRAMES,) + (1,) * (source.ndim - 1)
            fraction = weight.to(dtype=source.dtype).view(shape)
            uv = source[lower] + fraction * (source[upper] - source[lower])
        pressure = torch.zeros((*uv.shape[:-1], 1), dtype=uv.dtype)
        uvp = torch.cat((uv, pressure), dim=-1)
        return uvp[:20], uvp[20:]


def candidate_rate_codes(count: int, seed: int = RATE_SEED) -> list[int]:
    """Return a shuffled exact 50/25/25 identity/slow/fast allocation."""
    if count <= 0:
        raise ValueError("rate schedule must contain occurrences")
    slow = count // 4
    fast = count // 4
    identity = count - slow - fast
    values = torch.tensor(
        [0] * identity + [1] * slow + [2] * fast, dtype=torch.long)
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    return values[order].tolist()


def expand_occurrence_schedule(base_schedule: Mapping[str, object]) -> dict:
    """Give every scheduled draw a unique dataset index for fixed rate lookup."""
    batches = []
    rows = []
    source_indices = []
    cursor = 0
    for base_row in base_schedule["rows"]:
        base_indices = [int(value) for value in base_row["sample_indices"]]
        occurrence_indices = list(range(cursor, cursor + len(base_indices)))
        cursor += len(base_indices)
        batches.append(occurrence_indices)
        source_indices.extend(base_indices)
        rows.append({
            "sample_indices": occurrence_indices,
            "trajectory_ids": [
                int(value) for value in base_row["trajectory_ids"]],
            "bootstrap_mask": base_row["bootstrap_mask"],
        })
    if len(rows) != UPDATES:
        raise RuntimeError("base schedule does not have the frozen update count")
    return {
        "batches": batches,
        "rows": rows,
        "masks": [],
        "updates": UPDATES,
        "sha256": dr1.canonical_sha256(rows),
        "source_indices": source_indices,
        "base_stream_sha256": str(base_schedule["sha256"]),
    }


def rate_schedule_report(
        source_indices: Sequence[int], rate_codes: Sequence[int]) -> dict:
    if len(source_indices) != len(rate_codes):
        raise ValueError("rate report schedules are not aligned")
    rows = [
        {"occurrence": index, "source_index": int(source),
         "rate_code": int(rate)}
        for index, (source, rate) in enumerate(zip(source_indices, rate_codes))
    ]
    counts = Counter(int(value) for value in rate_codes)
    return {
        "sha256": dr1.canonical_sha256(rows),
        "occurrences": len(rows),
        "counts": {
            RATE_NAMES[code]: int(counts.get(code, 0))
            for code in range(len(RATE_NAMES))
        },
        "formula": "x_tilde[k] = linear_interp(x[start + rate*k])",
        "rates": [value / RATE_DENOMINATOR for value in RATE_NUMERATORS],
        "source_frames": SOURCE_FRAMES,
        "output_frames": WINDOW_FRAMES,
        "maximum_source_offset": 43,
    }


def validate_standard24k_identity(
        path: Path, claimed_sha256: str | None = None) -> str:
    """Require the exact preregistered Standard24k Sim artifact."""
    actual = dr1.sha256(path)
    if actual != EXPECTED_STANDARD24K_SHA256:
        raise RuntimeError("preregistered Standard24k Sim SHA changed")
    if claimed_sha256 is not None:
        claimed = dr1.normalized_sha256(
            claimed_sha256, "claimed Standard24k Sim SHA")
        if claimed != EXPECTED_STANDARD24K_SHA256:
            raise RuntimeError("CLI Standard24k SHA differs from preregistration")
    return actual


@contextmanager
def endpoint_contract(arm: str, held_aoa: int) -> Iterator[None]:
    if arm not in ARMS:
        raise ValueError(f"unknown rate-jitter arm {arm}")
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "PRIMARY_ENDPOINT": ENDPOINT[arm],
        "UPDATES": UPDATES,
        "SCHEDULER_T_MAX": SCHEDULER_T_MAX,
        "HELD_AOA": held_aoa,
    }
    original = {name: getattr(small4k, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(small4k, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(small4k, name, value)


def load_endpoint(
        seal: Mapping[str, object], normalizer: smu1.FoldNormalizer,
        device: torch.device, held_aoa: int, arm: str):
    with endpoint_contract(arm, held_aoa):
        return small4k.load_primary_endpoint(
            seal, normalizer, device,
            expected_model_config=sim_tke.b210_config("candidate"),
            expected_parameters=ft12.EXPECTED_PARAMETERS,
            expected_controller_parameters=ft12.EXPECTED_CONTROLLER_PARAMETERS)


def train_arm(
        arm: str, args: argparse.Namespace, audit: held.PrefixAccessAudit,
        dataset: ScheduledContinuousRateDataset, schedule: Mapping[str, object],
        normalizer: smu1.FoldNormalizer, start_sha: str,
        rate_report: Mapping[str, object], deadline: float) -> dict:
    device = torch.device(args.device)
    ft12.npt1.ensemble_train.seed_everything(small4k.SEED)
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha, normalizer, device,
        expected_config=sim_tke.b210_config("candidate"),
        expected_parameters=ft12.EXPECTED_PARAMETERS,
        expected_controller_parameters=ft12.EXPECTED_CONTROLLER_PARAMETERS,
        expected_iteration=EXPECTED_SIM_ITERATION,
        expected_status=EXPECTED_SIM_STATUS)
    arm_dir = args.output_dir / arm
    arm_dir.mkdir()
    adapter = held.ArmSealAdapter(audit, arm)
    with endpoint_contract(arm, args.held_aoa):
        training, seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            arm_dir, start_report, adapter,
            input_augmentation=dense4k.AUGMENTATION,
            endpoint_filename=f"{ENDPOINT[arm]}.pth",
            expected_parameters=ft12.EXPECTED_PARAMETERS,
            scheduler_t_max=SCHEDULER_T_MAX)
    if adapter.value != seal:
        raise RuntimeError(f"{arm} endpoint is not bound to the access audit")
    result = {
        "updates": UPDATES,
        "source_inventory_stride": WINDOW_STRIDE,
        "occurrence_stream_sha256": schedule["sha256"],
        "base_stream_sha256": schedule["base_stream_sha256"],
        "rate_schedule": dict(rate_report),
        "input_augmentation": dict(dense4k.AUGMENTATION),
        "training": training,
        "seal": seal,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_fold(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.held_aoa not in (10, 20):
        raise ValueError("discovery gate only authorizes held AoA 10 or 20")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("rate-jitter gate requires CUDA")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    if not fit_rows or not held_rows:
        raise RuntimeError("condition-held fold has empty fit or held rows")
    start_sha = validate_standard24k_identity(
        args.start_sim, args.start_sim_sha256)

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    audit = held.PrefixAccessAudit(args.held_aoa, ARMS)
    fit_trajectories, fit_metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", audit)
    source_windows = rate_capable_windows(fit_trajectories)
    base_schedule = smu1.paired_batch_schedule(
        range(len(source_windows)), source_windows,
        updates=UPDATES, seed=small4k.SEED)
    schedule = expand_occurrence_schedule(base_schedule)
    control_rates = [0] * len(schedule["source_indices"])
    candidate_rates = candidate_rate_codes(len(schedule["source_indices"]))
    datasets = {
        "control": ScheduledContinuousRateDataset(
            fit_trajectories, source_windows, schedule["source_indices"],
            control_rates),
        "rate_jitter": ScheduledContinuousRateDataset(
            fit_trajectories, source_windows, schedule["source_indices"],
            candidate_rates),
    }
    rate_reports = {
        "control": rate_schedule_report(
            schedule["source_indices"], control_rates),
        "rate_jitter": rate_schedule_report(
            schedule["source_indices"], candidate_rates),
    }
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, sparse_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics, torch.device(args.device))
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "fit_normalizer.pt")

    reports = {}
    for arm in ARMS:
        reports[arm] = train_arm(
            arm, args, audit, datasets[arm], schedule, normalizer,
            start_sha, rate_reports[arm], deadline)
    del datasets, fit_trajectories
    gc.collect()

    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", audit)
    held_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    target_audit = None
    for arm in ARMS:
        model = load_endpoint(
            reports[arm]["seal"], normalizer, torch.device(args.device),
            args.held_aoa, arm)
        metrics, metric_state, current_target_audit = smu1.evaluate_model(
            model, held_dataset, list(range(len(held_windows))), normalizer,
            torch.device(args.device), deadline)
        if target_audit is None:
            target_audit = current_target_audit
        elif target_audit != current_target_audit:
            raise RuntimeError("held target audit changed between arms")
        reports[arm].update({
            "metrics": metrics,
            "metric_state": metric_state,
            "target_audit": current_target_audit,
        })
        del model
        gc.collect()
        torch.cuda.empty_cache()

    comparison = held.metric_comparison(
        reports["rate_jitter"]["metrics"], reports["control"]["metrics"])
    point_delta = float(comparison["point_final_contribution_delta"])
    metric_delta = comparison["delta"]
    safety_pass = all(
        math.isfinite(float(metric_delta[key]))
        and float(metric_delta[key]) >= SAFETY_METRIC_FLOOR
        for key in ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score"))
    discovery_pass = bool(point_delta >= DISCOVERY_POINT_MIN and safety_pass)
    decision = (
        "RUN_FROZEN_HELD20_CONFIRMATION"
        if args.held_aoa == 10 and discovery_pass else
        "AWAIT_SHA_BOUND_TWO_FOLD_AGGREGATION"
        if args.held_aoa == 20 and discovery_pass else
        "STOP_RATE_JITTER_AFTER_FOLD")
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": args.held_aoa,
        "fit_trajectories": len(fit_rows),
        "held_trajectories": len(held_rows),
        "fit_sparse_windows": len(sparse_windows),
        "fit_rate_capable_windows": len(source_windows),
        "scheduled_occurrences": len(schedule["source_indices"]),
        "held_prefix_windows": len(held_windows),
        "manifest": manifest_report,
        "normalizer": {
            "kind": "fit-condition sparse prefixes only",
            "statistics_sha256": smu1.statistics_digest(statistics),
            "held_condition_in_normalizer": False,
        },
        "start": {
            "path": str(args.start_sim.resolve()),
            "sha256": start_sha,
            "iteration": EXPECTED_SIM_ITERATION,
            "status": EXPECTED_SIM_STATUS,
            "expected_sha256": EXPECTED_STANDARD24K_SHA256,
            "identity_validated": True,
            "shared_by_both_arms": True,
        },
        "paired_contract": {
            "same_source_inventory": True,
            "same_occurrence_stream": True,
            "same_bootstrap_stream": True,
            "same_input_augmentation": True,
            "same_optimizer_and_rng_seed": True,
            "candidate_dataset_consumes_no_global_rng": True,
            "only_difference": "whole-window continuous temporal rate",
        },
        "arms": reports,
        "comparison": comparison,
        "discovery_gate": {
            "point_min_inclusive": DISCOVERY_POINT_MIN,
            "individual_metric_safety_floor": SAFETY_METRIC_FLOOR,
            "safety_pass": safety_pass,
            "pass": discovery_pass,
            "decision": decision,
        },
        "access_audit": audit.report(),
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_class": (
            "Real-stage whole-AoA-held discovery; Sim start contains all "
            "released Sim conditions"),
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    return result


def aggregate_results(pairs: Sequence[Sequence[str]]) -> dict:
    """SHA-bind held10/20 folds and apply the preregistered pooled gate."""
    rows = []
    sources = []
    for raw_path, raw_sha in pairs:
        path = Path(raw_path)
        expected = dr1.normalized_sha256(raw_sha, "rate-jitter fold SHA")
        actual = dr1.sha256(path)
        if actual != expected:
            raise RuntimeError(f"rate-jitter fold result changed: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("schema") != SCHEMA
                or row.get("experiment") != EXPERIMENT
                or row.get("start", {}).get("sha256")
                != EXPECTED_STANDARD24K_SHA256
                or row.get("start", {}).get("expected_sha256")
                != EXPECTED_STANDARD24K_SHA256
                or row.get("start", {}).get("identity_validated") is not True):
            raise ValueError("rate-jitter fold identity changed")
        rows.append(row)
        sources.append({"path": str(path.resolve()), "sha256": actual})
    held_aoas = [int(row["held_aoa"]) for row in rows]
    if sorted(held_aoas) != [10, 20]:
        raise ValueError("aggregate requires exactly one held10 and held20 fold")
    control_state = smu1.merge_metric_states([
        row["arms"]["control"]["metric_state"] for row in rows])
    candidate_state = smu1.merge_metric_states([
        row["arms"]["rate_jitter"]["metric_state"] for row in rows])
    control_metrics = smu1.finalize_metric_state(control_state)
    candidate_metrics = smu1.finalize_metric_state(candidate_state)
    comparison = held.metric_comparison(candidate_metrics, control_metrics)
    fold_point_deltas = {
        str(row["held_aoa"]): float(
            row["comparison"]["point_final_contribution_delta"])
        for row in rows
    }
    folds_pass = all(
        value >= DISCOVERY_POINT_MIN for value in fold_point_deltas.values())
    pooled_point = float(comparison["point_final_contribution_delta"])
    pooled_safety = all(
        math.isfinite(float(comparison["delta"][key]))
        and float(comparison["delta"][key]) >= SAFETY_METRIC_FLOOR
        for key in ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score"))
    gate_pass = bool(
        folds_pass and pooled_safety and pooled_point >= POOLED_POINT_MIN)
    return {
        "schema": AGGREGATE_SCHEMA,
        "experiment": EXPERIMENT,
        "sources": sources,
        "held_aoas": held_aoas,
        "start_sha256": EXPECTED_STANDARD24K_SHA256,
        "arms": {
            "control": {
                "metric_state": control_state, "metrics": control_metrics},
            "rate_jitter": {
                "metric_state": candidate_state,
                "metrics": candidate_metrics},
        },
        "comparison": comparison,
        "gate": {
            "per_fold_point_min_inclusive": DISCOVERY_POINT_MIN,
            "pooled_point_min_inclusive": POOLED_POINT_MIN,
            "individual_metric_safety_floor": SAFETY_METRIC_FLOOR,
            "fold_point_deltas": fold_point_deltas,
            "folds_pass": folds_pass,
            "pooled_safety_pass": pooled_safety,
            "pass": gate_pass,
        },
        "decision": (
            "PROMOTE_RATE_JITTER_TO_STRICT_FULL_LIFECYCLE_VALIDATION"
            if gate_pass else "STOP_RATE_JITTER_AFTER_TWO_FOLD_GATE"),
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3600)
    parser.add_argument(
        "--fold-result", nargs=2, action="append", metavar=("PATH", "SHA256"))
    parser.add_argument("--aggregate-output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.fold_worker:
        required = {
            "held_aoa": args.held_aoa, "output_dir": args.output_dir,
            "real_dir": args.real_dir, "clean_manifest": args.clean_manifest,
            "start_sim": args.start_sim,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"rate-jitter fold is missing {missing}")
        if args.fold_result or args.aggregate_output:
            raise ValueError("fold worker received aggregate arguments")
    else:
        if args.fold_result is None or len(args.fold_result) != 2:
            raise ValueError("aggregate requires exactly two fold results")
        if args.aggregate_output is None:
            raise ValueError("aggregate output path is required")
        if any(value is not None for value in (
                args.held_aoa, args.output_dir, args.real_dir,
                args.clean_manifest, args.start_sim, args.start_sim_sha256)):
            raise ValueError("aggregate received fold-worker arguments")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.fold_worker:
        result = run_fold(args)
        payload = {
            "event": "b210_continuous_rate_jitter_real_gate_complete",
            "held_aoa": result["held_aoa"],
            "comparison": result["comparison"],
            "decision": result["discovery_gate"]["decision"],
            "result": str(args.output_dir / "fold_result.json"),
        }
    else:
        result = aggregate_results(args.fold_result)
        if args.aggregate_output.exists():
            raise FileExistsError(args.aggregate_output)
        dr1.write_json_atomic(args.aggregate_output, result)
        payload = {
            "event": "b210_continuous_rate_jitter_aggregate_complete",
            "comparison": result["comparison"],
            "decision": result["decision"],
            "result": str(args.aggregate_output),
        }
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
