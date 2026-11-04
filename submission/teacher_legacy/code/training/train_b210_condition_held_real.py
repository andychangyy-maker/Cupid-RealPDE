#!/usr/bin/env python3
"""Train B210 Real arms with whole conditions excluded from optimization.

The fold worker loads only the other-condition Real prefixes, trains fixed
endpoints from one immutable Sim10k checkpoint, seals them, and only then opens
the held-condition prefixes for evaluation.  Temporal-tail labels are never
constructed or read.

The primary validation arm compares the hidden-set-positive Dense4k recipe
with the hidden-set-negative Dense12k recipe.  This tests whether genuine
condition-held validation can recover a known official transfer ordering.  An
optional hierarchical Dense4k arm samples AoA -> trajectory -> phase nearly
uniformly and is the first candidate for the next full-data retraining.
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

import numpy as np
import torch

import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_np_functional_sim as sim_tke
import train_b210_scratch_real_dense_aug as dense4k
import train_c208_dr1_clean81_paired_loao as dr1
import train_c208_smu1_paired_loao as smu1


EXPERIMENT = "b210_condition_held_real"
SCHEMA = "b210_condition_held_real_fold_v1"
AGGREGATE_SCHEMA = "b210_condition_held_real_aggregate_v1"
BALANCED_SCHEMA = "b210_condition_held_balanced_fold_v1"
BALANCED_AGGREGATE_SCHEMA = "b210_condition_held_balanced_aggregate_v1"
ENDPOINT_SCHEMA = "b210_condition_held_real_endpoint_v1"
AOA_VALUES = (0, 5, 10, 15, 20)
ARM_ORDER = ("dense4k", "dense12k")
BALANCED_ARM = "balanced4k"
UPDATES = {"dense4k": 4_000, "dense12k": 12_000, BALANCED_ARM: 4_000}
ENDPOINT = {
    "dense4k": "ema_u4000",
    "dense12k": "ema_u12000",
    BALANCED_ARM: "ema_u4000_balanced",
}
FINAL_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_rel_l2_score": 0.10,
    "mvpe_rel_l2_score": 0.09,
    "native_sps_score": 0.25,
}
BALANCED_GATE = {
    "pooled_non_time_min_exclusive": 0.0,
    "minimum_positive_folds": 4,
    "worst_fold_min_inclusive": -0.05,
}


class PrefixAccessAudit:
    """Forbid held-condition opens until every requested arm is sealed."""

    def __init__(self, held_aoa: int, arms: Sequence[str]) -> None:
        self.held_aoa = held_aoa
        self.arms = tuple(arms)
        self.seals: dict[str, dict] = {}
        self.fit_opened: list[str] = []
        self.held_opened: list[str] = []

    def before_open(self, row: dict, phase: str) -> None:
        aoa = int(row["aoa"])
        filename = str(row["file"])
        if filename == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("excluded trajectory entered condition-held fold")
        if phase == "fit":
            if aoa == self.held_aoa or self.seals:
                raise RuntimeError("fit access contains held condition or is late")
            if filename in self.fit_opened:
                raise RuntimeError("fit trajectory opened twice")
            return
        if phase != "held" or aoa != self.held_aoa:
            raise RuntimeError("held access is not the requested condition")
        if set(self.seals) != set(self.arms):
            raise RuntimeError("held condition opened before all arm seals")
        for seal in self.seals.values():
            path = Path(str(seal["path"]))
            if not path.is_file() or dr1.sha256(path) != seal["file_sha256"]:
                raise RuntimeError("condition-held endpoint changed before scoring")
        if filename in self.held_opened:
            raise RuntimeError("held trajectory opened twice")

    def record_open(self, row: dict, phase: str) -> None:
        target = self.fit_opened if phase == "fit" else self.held_opened
        filename = str(row["file"])
        if filename in target:
            raise RuntimeError("condition trajectory record duplicated")
        target.append(filename)

    def seal(self, arm: str, value: Mapping[str, object]) -> None:
        if arm not in self.arms or arm in self.seals:
            raise RuntimeError(f"invalid or duplicate arm seal: {arm}")
        self.seals[arm] = dict(value)

    def report(self) -> dict:
        return {
            "held_aoa": self.held_aoa,
            "fit_opened": self.fit_opened,
            "held_opened_after_all_seals": self.held_opened,
            "sealed_arms": sorted(self.seals),
            "held_target_opened_before_all_seals": False,
            "temporal_tail_constructed_or_read": False,
        }


class ArmSealAdapter:
    def __init__(self, audit: PrefixAccessAudit, arm: str) -> None:
        self.audit = audit
        self.arm = arm
        self.value: dict | None = None

    def seal(self, value: Mapping[str, object]) -> None:
        if self.value is not None:
            raise RuntimeError(f"{self.arm} endpoint sealed twice")
        self.value = dict(value)
        self.audit.seal(self.arm, value)


@contextmanager
def endpoint_contract(arm: str, held_aoa: int) -> Iterator[None]:
    """Bind the generic fixed-endpoint trainer for one research arm."""
    if arm not in UPDATES:
        raise ValueError(f"unknown condition-held arm: {arm}")
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "PRIMARY_ENDPOINT": ENDPOINT[arm],
        "UPDATES": UPDATES[arm],
        "SCHEDULER_T_MAX": 12_000,
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


def hierarchical_batch_schedule(
        windows: Sequence, metadata: Sequence[Mapping[str, object]],
        updates: int, seed: int, batch_size: int = small4k.BATCH_SIZE) -> dict:
    """Sample AoA, trajectory, then temporal phase with deterministic balance."""
    if updates <= 0 or batch_size <= 0:
        raise ValueError("hierarchical schedule sizes must be positive")
    by_aoa: dict[int, dict[int, list[int]]] = {}
    for index, window in enumerate(windows):
        trajectory = int(window.trajectory)
        aoa = int(metadata[trajectory]["aoa"])
        by_aoa.setdefault(aoa, {}).setdefault(trajectory, []).append(index)
    if len(by_aoa) < 2 or any(not values for values in by_aoa.values()):
        raise ValueError("hierarchical schedule needs multiple non-empty AoAs")
    generator = torch.Generator().manual_seed(seed)
    aoas = sorted(by_aoa)
    rows = []
    batches = []
    aoa_counts = {aoa: 0 for aoa in aoas}
    trajectory_counts = {
        trajectory: 0
        for values in by_aoa.values() for trajectory in values
    }
    cursor = 0
    for _update in range(updates):
        selected = []
        for _slot in range(batch_size):
            aoa = aoas[cursor % len(aoas)]
            cursor += 1
            trajectories = sorted(by_aoa[aoa])
            trajectory = trajectories[int(torch.randint(
                len(trajectories), (1,), generator=generator))]
            choices = by_aoa[aoa][trajectory]
            index = choices[int(torch.randint(
                len(choices), (1,), generator=generator))]
            selected.append(index)
            aoa_counts[aoa] += 1
            trajectory_counts[trajectory] += 1
        order = torch.randperm(batch_size, generator=generator).tolist()
        batch = [selected[index] for index in order]
        trajectories = [int(windows[index].trajectory) for index in batch]
        mask = ft12.npt1.ensemble_train.trajectory_member_mask(
            torch.as_tensor(trajectories), 0.75, seed, 4)
        batches.append(batch)
        rows.append({
            "sample_indices": batch,
            "trajectory_ids": trajectories,
            "bootstrap_mask": mask.to(torch.uint8).tolist(),
        })
    return {
        "batches": batches,
        "rows": rows,
        "masks": [],
        "updates": updates,
        "sha256": dr1.canonical_sha256(rows),
        "sampling": "AoA -> trajectory/Re -> temporal phase",
        "aoa_draw_counts": {str(key): value for key, value in aoa_counts.items()},
        "trajectory_draw_counts": {
            str(key): value for key, value in trajectory_counts.items()},
    }


def uniform_batch_schedule(windows: Sequence, updates: int) -> dict:
    schedule = smu1.paired_batch_schedule(
        range(len(windows)), windows, updates=updates, seed=small4k.SEED)
    schedule["updates"] = updates
    schedule["sampling"] = "uniform shuffled dense windows"
    return schedule


def load_endpoint(
        seal: Mapping[str, object], normalizer: smu1.FoldNormalizer,
        device: torch.device, expected_held_aoa: int):
    path = Path(str(seal["path"]))
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("condition-held endpoint file changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != ENDPOINT_SCHEMA
            or payload.get("experiment") != EXPERIMENT
            or payload.get("endpoint") != seal["endpoint"]
            or int(payload.get("iteration", -1)) != int(seal["iteration"])
            or int(payload.get("held_aoa", -1)) != expected_held_aoa
            or not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError("condition-held endpoint payload changed")
    config = ft12.validate_b210_model_config(
        payload["model_config"], sim_tke.b210_config("candidate"))
    model = small4k.EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def metric_comparison(candidate: Mapping[str, float], control: Mapping[str, float]) -> dict:
    keys = tuple(FINAL_WEIGHTS)
    delta = {key: float(candidate[key]) - float(control[key]) for key in keys}
    point = sum(
        FINAL_WEIGHTS[key] * delta[key]
        for key in ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score"))
    non_time = point + FINAL_WEIGHTS["native_sps_score"] * delta[
        "native_sps_score"]
    return {
        "control": dict(control),
        "candidate": dict(candidate),
        "delta": delta,
        "point_final_contribution_delta": point,
        "non_time_final_contribution_delta": non_time,
    }


def train_arm(
        arm: str, args: argparse.Namespace, audit: PrefixAccessAudit,
        dataset, windows: Sequence, metadata: Sequence[Mapping[str, object]],
        normalizer: smu1.FoldNormalizer, start_sha: str,
        deadline: float) -> dict:
    device = torch.device(args.device)
    ft12.npt1.ensemble_train.seed_everything(small4k.SEED)
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha, normalizer, device,
        expected_config=sim_tke.b210_config("candidate"),
        expected_parameters=ft12.EXPECTED_PARAMETERS,
        expected_controller_parameters=ft12.EXPECTED_CONTROLLER_PARAMETERS)
    schedule = (
        hierarchical_batch_schedule(
            windows, metadata, UPDATES[arm], small4k.SEED)
        if arm == BALANCED_ARM else
        uniform_batch_schedule(windows, UPDATES[arm]))
    arm_dir = args.output_dir / arm
    arm_dir.mkdir()
    adapter = ArmSealAdapter(audit, arm)
    with endpoint_contract(arm, args.held_aoa):
        training, seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            arm_dir, start_report, adapter,
            input_augmentation=dense4k.AUGMENTATION,
            endpoint_filename=f"{ENDPOINT[arm]}.pth",
            expected_parameters=ft12.EXPECTED_PARAMETERS)
    if adapter.value != seal:
        raise RuntimeError(f"{arm} endpoint is not bound to access audit")
    result = {
        "updates": UPDATES[arm],
        "sampling": schedule["sampling"],
        "sample_stream_sha256": schedule["sha256"],
        "schedule_balance": {
            key: schedule[key]
            for key in ("aoa_draw_counts", "trajectory_draw_counts")
            if key in schedule
        },
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
    if args.held_aoa not in AOA_VALUES:
        raise ValueError("held AoA is outside the released conditions")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("condition-held B210 training requires CUDA")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    if not fit_rows or not held_rows:
        raise RuntimeError("condition-held fold has empty fit or held rows")
    start_sha = dr1.sha256(args.start_sim)
    if (args.start_sim_sha256 is not None
            and start_sha != args.start_sim_sha256.lower()):
        raise RuntimeError("condition-held Sim start SHA changed")
    arms = list(ARM_ORDER)
    if args.include_balanced:
        arms.append(BALANCED_ARM)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    audit = PrefixAccessAudit(args.held_aoa, arms)
    fit_trajectories, fit_metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", audit)
    dense_windows = dense4k.dense_windows(
        fit_trajectories, dense4k.CANDIDATE_STRIDE)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        fit_trajectories, dense_windows)
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, sparse_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics, torch.device(args.device))
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "fit_normalizer.pt")
    reports = {}
    for arm in arms:
        reports[arm] = train_arm(
            arm, args, audit, dataset, dense_windows, fit_metadata,
            normalizer, start_sha, deadline)
    del dataset, fit_trajectories
    gc.collect()
    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", audit)
    held_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    target_audit = None
    for arm in arms:
        model = load_endpoint(
            reports[arm]["seal"], normalizer, torch.device(args.device),
            args.held_aoa)
        metrics, metric_state, current_target_audit = smu1.evaluate_model(
            model, held_dataset, list(range(len(held_windows))), normalizer,
            torch.device(args.device), deadline)
        if target_audit is None:
            target_audit = current_target_audit
        elif target_audit != current_target_audit:
            raise RuntimeError("held target audit changed between arms")
        reports[arm]["metrics"] = metrics
        reports[arm]["metric_state"] = metric_state
        reports[arm]["target_audit"] = current_target_audit
        del model
        gc.collect()
        torch.cuda.empty_cache()
    comparisons = {
        "dense12k_minus_dense4k": metric_comparison(
            reports["dense12k"]["metrics"], reports["dense4k"]["metrics"]),
    }
    if BALANCED_ARM in reports:
        comparisons["balanced4k_minus_dense4k"] = metric_comparison(
            reports[BALANCED_ARM]["metrics"], reports["dense4k"]["metrics"])
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": args.held_aoa,
        "fit_trajectories": len(fit_rows),
        "held_trajectories": len(held_rows),
        "fit_sparse_windows": len(sparse_windows),
        "fit_dense_windows": len(dense_windows),
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
            "sim_checkpoint_shared_by_all_arms": True,
        },
        "arms": reports,
        "comparisons": comparisons,
        "access_audit": audit.report(),
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_class": (
            "Real-stage whole-AoA-held validation; Sim start contains all "
            "released Sim conditions, so this is not full-lifecycle LOAO"),
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    return result


def load_frozen_control(
        pair: Sequence[str], expected_held_aoa: int) -> tuple[dict, dict]:
    """Load one SHA-bound Dense4k fold without reopening its checkpoint."""
    raw_path, raw_sha = pair
    path = Path(raw_path)
    expected_sha = dr1.normalized_sha256(raw_sha, "control fold result SHA")
    actual_sha = dr1.sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"frozen Dense4k control changed: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row.get("schema") != SCHEMA
            or row.get("experiment") != EXPERIMENT
            or int(row.get("held_aoa", -1)) != expected_held_aoa
            or "dense4k" not in row.get("arms", {})):
        raise ValueError("frozen Dense4k control identity changed")
    arm = row["arms"]["dense4k"]
    required = ("metrics", "metric_state", "target_audit")
    if any(key not in arm for key in required):
        raise ValueError("frozen Dense4k control is incomplete")
    source = {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "manifest": row["manifest"],
        "normalizer": row["normalizer"],
        "start": row["start"],
        "metrics": arm["metrics"],
        "metric_state": arm["metric_state"],
        "target_audit": arm["target_audit"],
    }
    return row, source


def run_balanced_fold(args: argparse.Namespace) -> dict:
    """Train only Balanced4k and compare with an immutable Dense4k result."""
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.held_aoa not in AOA_VALUES:
        raise ValueError("held AoA is outside the released conditions")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("condition-held B210 training requires CUDA")
    control_row, control = load_frozen_control(
        args.control_fold_result, args.held_aoa)
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    if manifest_report != control["manifest"]:
        raise RuntimeError("manifest report differs from frozen Dense4k control")
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    if not fit_rows or not held_rows:
        raise RuntimeError("condition-held fold has empty fit or held rows")
    start_sha = dr1.sha256(args.start_sim)
    if (args.start_sim_sha256 is not None
            and start_sha != args.start_sim_sha256.lower()):
        raise RuntimeError("condition-held Sim start SHA changed")
    if start_sha != control["start"]["sha256"]:
        raise RuntimeError("balanced candidate does not share control Sim start")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    torch.set_float32_matmul_precision("high")
    audit = PrefixAccessAudit(args.held_aoa, [BALANCED_ARM])
    fit_trajectories, fit_metadata, sparse_windows = dr1.load_prefix_rows(
        args.real_dir, fit_rows, "fit", audit)
    dense_windows = dense4k.dense_windows(
        fit_trajectories, dense4k.CANDIDATE_STRIDE)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        fit_trajectories, dense_windows)
    sufficient = smu1.window_sufficient_statistics(
        fit_trajectories, sparse_windows)
    statistics = smu1.fold_statistics(
        sufficient, range(len(sparse_windows)))
    normalizer = smu1.FoldNormalizer(statistics, torch.device(args.device))
    statistics_sha = smu1.statistics_digest(statistics)
    if statistics_sha != control["normalizer"]["statistics_sha256"]:
        raise RuntimeError("balanced normalizer differs from frozen control")
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "fit_normalizer.pt")
    candidate = train_arm(
        BALANCED_ARM, args, audit, dataset, dense_windows, fit_metadata,
        normalizer, start_sha, deadline)
    del dataset, fit_trajectories
    gc.collect()
    held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows(
        args.real_dir, held_rows, "held", audit)
    held_dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        held_trajectories, held_windows)
    model = load_endpoint(
        candidate["seal"], normalizer, torch.device(args.device),
        args.held_aoa)
    metrics, metric_state, target_audit = smu1.evaluate_model(
        model, held_dataset, list(range(len(held_windows))), normalizer,
        torch.device(args.device), deadline)
    if target_audit != control["target_audit"]:
        raise RuntimeError("held targets differ from frozen Dense4k control")
    candidate["metrics"] = metrics
    candidate["metric_state"] = metric_state
    candidate["target_audit"] = target_audit
    del model
    gc.collect()
    torch.cuda.empty_cache()
    comparison = metric_comparison(metrics, control["metrics"])
    result = {
        "schema": BALANCED_SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": args.held_aoa,
        "fit_trajectories": len(fit_rows),
        "held_trajectories": len(held_rows),
        "fit_sparse_windows": len(sparse_windows),
        "fit_dense_windows": len(dense_windows),
        "held_prefix_windows": len(held_windows),
        "manifest": manifest_report,
        "normalizer": {
            "kind": "fit-condition sparse prefixes only",
            "statistics_sha256": statistics_sha,
            "matches_frozen_control": True,
            "held_condition_in_normalizer": False,
        },
        "start": {
            "path": str(args.start_sim.resolve()),
            "sha256": start_sha,
            "matches_frozen_control": True,
        },
        "control": control,
        "arm": candidate,
        "comparison": comparison,
        "access_audit": audit.report(),
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_class": control_row["evidence_class"],
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    return result


def aggregate_balanced_results(pairs: Sequence[Sequence[str]]) -> dict:
    """Merge matched Balanced4k folds and apply the preregistered gate."""
    rows = []
    sources = []
    for raw_path, raw_sha in pairs:
        path = Path(raw_path)
        expected = dr1.normalized_sha256(raw_sha, "balanced fold result SHA")
        actual = dr1.sha256(path)
        if actual != expected:
            raise RuntimeError(f"balanced fold result changed: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("schema") != BALANCED_SCHEMA
                or row.get("experiment") != EXPERIMENT):
            raise ValueError("balanced fold identity changed")
        control_row, control = load_frozen_control(
            (row["control"]["path"], row["control"]["sha256"]),
            int(row["held_aoa"]))
        del control_row
        for key in ("metrics", "metric_state", "target_audit"):
            if control[key] != row["control"][key]:
                raise RuntimeError("embedded frozen control state changed")
        rows.append(row)
        sources.append({"path": str(path.resolve()), "sha256": actual})
    held = [int(row["held_aoa"]) for row in rows]
    if len(held) != len(set(held)):
        raise ValueError("balanced aggregate repeats an AoA")
    control_state = smu1.merge_metric_states([
        row["control"]["metric_state"] for row in rows])
    candidate_state = smu1.merge_metric_states([
        row["arm"]["metric_state"] for row in rows])
    control_metrics = smu1.finalize_metric_state(control_state)
    candidate_metrics = smu1.finalize_metric_state(candidate_state)
    comparison = metric_comparison(candidate_metrics, control_metrics)
    fold_deltas = [
        float(row["comparison"]["non_time_final_contribution_delta"])
        for row in rows
    ]
    exact_full_loao = set(held) == set(AOA_VALUES)
    pooled_pass = bool(
        comparison["non_time_final_contribution_delta"]
        > BALANCED_GATE["pooled_non_time_min_exclusive"])
    positive_folds = sum(value > 0.0 for value in fold_deltas)
    positive_folds_pass = bool(
        exact_full_loao
        and positive_folds >= BALANCED_GATE["minimum_positive_folds"])
    worst_fold = min(fold_deltas)
    worst_fold_pass = bool(
        worst_fold >= BALANCED_GATE["worst_fold_min_inclusive"])
    gate_pass = bool(
        exact_full_loao and pooled_pass and positive_folds_pass
        and worst_fold_pass)
    return {
        "schema": BALANCED_AGGREGATE_SCHEMA,
        "experiment": EXPERIMENT,
        "sources": sources,
        "held_aoas": held,
        "arms": {
            "dense4k": {
                "metric_state": control_state,
                "metrics": control_metrics,
                "source": "SHA-frozen condition-held controls",
            },
            BALANCED_ARM: {
                "metric_state": candidate_state,
                "metrics": candidate_metrics,
            },
        },
        "comparison": comparison,
        "gate": {
            "thresholds": dict(BALANCED_GATE),
            "exact_full_loao": exact_full_loao,
            "pooled_pass": pooled_pass,
            "positive_folds": positive_folds,
            "positive_folds_pass": positive_folds_pass,
            "worst_fold": worst_fold,
            "worst_fold_pass": worst_fold_pass,
            "validation_gate_pass": gate_pass,
        },
        "proxy_decision": (
            "PASS_BALANCED4K_CONDITION_HELD_GATE"
            if gate_pass else "STOP_BALANCED4K_CONDITION_HELD_GATE"),
        "submission_authorized": False,
    }


def aggregate_results(pairs: Sequence[Sequence[str]]) -> dict:
    rows = []
    sources = []
    for raw_path, raw_sha in pairs:
        path = Path(raw_path)
        expected = dr1.normalized_sha256(raw_sha, "fold result SHA")
        actual = dr1.sha256(path)
        if actual != expected:
            raise RuntimeError(f"fold result changed: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema") != SCHEMA or row.get("experiment") != EXPERIMENT:
            raise ValueError("condition-held fold identity changed")
        rows.append(row)
        sources.append({"path": str(path.resolve()), "sha256": actual})
    held = [int(row["held_aoa"]) for row in rows]
    if len(held) != len(set(held)):
        raise ValueError("condition-held aggregate repeats an AoA")
    common_arms = set(rows[0]["arms"])
    if any(set(row["arms"]) != common_arms for row in rows):
        raise ValueError("condition-held folds do not contain the same arms")
    arms = {}
    for arm in sorted(common_arms):
        state = smu1.merge_metric_states([
            row["arms"][arm]["metric_state"] for row in rows])
        arms[arm] = {
            "metric_state": state,
            "metrics": smu1.finalize_metric_state(state),
        }
    comparisons = {
        "dense12k_minus_dense4k": metric_comparison(
            arms["dense12k"]["metrics"], arms["dense4k"]["metrics"]),
    }
    if BALANCED_ARM in arms:
        comparisons["balanced4k_minus_dense4k"] = metric_comparison(
            arms[BALANCED_ARM]["metrics"], arms["dense4k"]["metrics"])
    dense12_fold_deltas = [
        row["comparisons"]["dense12k_minus_dense4k"][
            "non_time_final_contribution_delta"]
        for row in rows
    ]
    known_ordering = {
        "official_reference": "scratch Dense4k quality > Dense12k quality",
        "pooled_dense12k_minus_dense4k_non_time_delta": comparisons[
            "dense12k_minus_dense4k"]["non_time_final_contribution_delta"],
        "negative_folds": sum(value < 0.0 for value in dense12_fold_deltas),
        "folds": len(rows),
        "exact_full_loao": set(held) == set(AOA_VALUES),
    }
    known_ordering["direction_matches_official"] = bool(
        known_ordering["pooled_dense12k_minus_dense4k_non_time_delta"] < 0.0)
    known_ordering["minimum_negative_folds"] = (
        3 if known_ordering["exact_full_loao"] else None)
    known_ordering["fold_direction_pass"] = (
        known_ordering["negative_folds"] >= 3
        if known_ordering["exact_full_loao"] else None)
    known_ordering["validation_gate_pass"] = bool(
        known_ordering["direction_matches_official"]
        and (not known_ordering["exact_full_loao"]
             or known_ordering["fold_direction_pass"]))
    return {
        "schema": AGGREGATE_SCHEMA,
        "experiment": EXPERIMENT,
        "sources": sources,
        "held_aoas": held,
        "arms": arms,
        "comparisons": comparisons,
        "known_official_ordering_check": known_ordering,
        "proxy_decision": (
            "PASS_REAL_STAGE_CONDITION_HELD_DIRECTION"
            if known_ordering["validation_gate_pass"]
            else "STOP_REAL_STAGE_CONDITION_HELD_DIRECTION"),
        "submission_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fold-worker", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    mode.add_argument("--balanced-fold-worker", action="store_true")
    mode.add_argument("--balanced-aggregate-only", action="store_true")
    parser.add_argument("--held-aoa", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256")
    parser.add_argument("--include-balanced", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 3600)
    parser.add_argument(
        "--fold-result", nargs=2, action="append", metavar=("PATH", "SHA256"))
    parser.add_argument(
        "--control-fold-result", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--aggregate-output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.fold_worker or args.balanced_fold_worker:
        required = {
            "held_aoa": args.held_aoa,
            "output_dir": args.output_dir,
            "real_dir": args.real_dir,
            "clean_manifest": args.clean_manifest,
            "start_sim": args.start_sim,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"condition-held fold is missing {missing}")
        if args.fold_result or args.aggregate_output:
            raise ValueError("fold worker received aggregate arguments")
        if args.fold_worker and args.control_fold_result:
            raise ValueError("ordinary fold received a frozen control")
        if args.balanced_fold_worker and not args.control_fold_result:
            raise ValueError("balanced fold requires a frozen control")
        if args.balanced_fold_worker and args.include_balanced:
            raise ValueError("balanced-only fold received legacy include flag")
    else:
        if not args.fold_result or args.aggregate_output is None:
            raise ValueError("aggregate requires fold results and output")
        if args.aggregate_output.exists():
            raise FileExistsError(args.aggregate_output)
        if any((args.held_aoa, args.output_dir, args.real_dir,
                args.clean_manifest, args.start_sim, args.start_sim_sha256,
                args.include_balanced, args.control_fold_result)):
            raise ValueError("aggregate received fold-only arguments")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.aggregate_only or args.balanced_aggregate_only:
        result = (
            aggregate_balanced_results(args.fold_result)
            if args.balanced_aggregate_only else
            aggregate_results(args.fold_result))
        dr1.write_json_atomic(args.aggregate_output, result)
        print(json.dumps({
            "event": "b210_condition_held_aggregate_complete",
            "decision": result["proxy_decision"],
            "output": str(args.aggregate_output),
        }), flush=True)
        return
    result = (
        run_balanced_fold(args) if args.balanced_fold_worker else
        run_fold(args))
    print(json.dumps({
        "event": "b210_condition_held_fold_complete",
        "held_aoa": result["held_aoa"],
        "comparisons": (
            {"balanced4k_minus_dense4k": result["comparison"]}
            if args.balanced_fold_worker else result["comparisons"]),
        "output": str(args.output_dir / "fold_result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
