#!/usr/bin/env python3
"""Run the single preregistered C208-S4K1 Sim continuation refinement."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import train_c208_real_sim_rehearsal as rehearsal
import train_c208_sim_hgrad_pretrain as shg
from models.ensemble_unet2d import EnsembleUNet2d


METHOD = "c208_s4k1_sim_low_lr_refine"
PASS_STATUS = "PROMOTE_S4K1_SIM_MODEL_14000"
PREFIX_PASS_STATUS = "GO_S4K1_TO_SIM4000"
PREFIX_STOP_STATUS = "STOP_S4K1_PREFIX200"
TERMINAL_STOP_STATUS = "STOP_S4K1_SIM4000"

START_GLOBAL_UPDATE = 10_000
PREFIX_LOCAL_UPDATE = 200
TERMINAL_LOCAL_UPDATE = 4_000
TERMINAL_GLOBAL_UPDATE = START_GLOBAL_UPDATE + TERMINAL_LOCAL_UPDATE
REFINEMENT_LR = 3e-5
SCHEDULER_T_MAX = TERMINAL_LOCAL_UPDATE
ANCHOR_SIZE = 128
EXPECTED_SIM_WINDOWS = 4_800
EXPECTED_SIM_CONDITIONS = 100
EXPECTED_WINDOWS_PER_CONDITION = 48

BASELINE_SIM_SHA256 = rehearsal.FORMAL_START_SIM_SHA256
BASELINE_SUMMARY_SHA256 = shg.BASELINE_SUMMARY_SHA256
BASELINE_SPLIT_MANIFEST_SHA256 = (
    "01c9ed5694832e0a486dd483711f7621a1078e38755de8f1edc8cbec0e9ad38e")
TEACHER_CACHE_SHA256 = rehearsal.FORMAL_P1_CACHE_SHA256
REAL_STATS_SHA256 = rehearsal.FORMAL_REAL_STATS_SHA256
BASELINE_MODEL_CONFIG_SHA256 = (
    "686278197c641e9c4b907e66e95ab898e4f94850276deca50c9af8c48a1836dc")
BASELINE_TRAINING_CONFIG_SHA256 = (
    "b13f24643dd8d5e142c79e9e525f1603f3248f8b1304c3ccc093f36da484373e")
BASELINE_NORMALIZER_SHA256 = (
    "ab8065e45470fd4c24eadaecad647d0f62b557c3e19e9a235e6f4c6867c21c65")

POINT_COMPONENT_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")
SEEN_P1_RATIO_MAX = 0.995
SEEN_POINT_DELTA_MIN = 0.01
UNSEEN_P1_RATIO_MAX = 0.9975
UNSEEN_POINT_DELTA_MIN = 0.0
PREFIX_COMPONENT_DELTA_FLOOR = -0.02
PREFIX_SOFT_SPS_DELTA_FLOOR = -0.002
UNSEEN_POINT_GAP_FROM_SEEN_FLOOR = -0.05
UNSEEN_P1_DECLINE_FRACTION_OF_SEEN_MIN = 0.5
TERMINAL_P1_RATIO_MAX = 0.99
TERMINAL_POINT_DELTA_MIN = 0.03
TERMINAL_COMPONENT_DELTA_FLOOR = -0.05
TERMINAL_SOFT_SPS_DELTA_FLOOR = -0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sim", type=Path, required=True)
    parser.add_argument("--baseline-sim-sha256", required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-summary-sha256", required=True)
    parser.add_argument(
        "--baseline-split-manifest", type=Path, required=True)
    parser.add_argument(
        "--baseline-split-manifest-sha256", required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--teacher-cache-sha256", required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--real-stats-sha256", required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers != 0:
        raise ValueError("formal S4K1 freezes num-workers=0")
    frozen_hashes = {
        "baseline_sim_sha256": BASELINE_SIM_SHA256,
        "baseline_summary_sha256": BASELINE_SUMMARY_SHA256,
        "baseline_split_manifest_sha256": BASELINE_SPLIT_MANIFEST_SHA256,
        "teacher_cache_sha256": TEACHER_CACHE_SHA256,
        "real_stats_sha256": REAL_STATS_SHA256,
    }
    for name, expected in frozen_hashes.items():
        actual = rehearsal.normalized_sha256(
            getattr(args, name), name.replace("_", "-"))
        setattr(args, name, actual)
        if actual != expected:
            raise ValueError(f"{name} is not the frozen S4K1 artifact")
    protected = ("summary.json", "model_14000.pth")
    if any((args.output_dir / name).exists() for name in protected):
        raise FileExistsError(
            f"refusing to overwrite a completed S4K1 run in {args.output_dir}")


def validate_baseline_contract(
        payload: dict, metadata: dict | None = None) -> dict:
    metadata = metadata or shg.validate_baseline_payload(payload)
    if metadata["model_config_sha256"] != BASELINE_MODEL_CONFIG_SHA256:
        raise ValueError("baseline model_config hash differs from frozen S4K1")
    if metadata["training_config_sha256"] != BASELINE_TRAINING_CONFIG_SHA256:
        raise ValueError("baseline training_config hash differs from frozen S4K1")
    stored = payload.get("normalizer")
    if not isinstance(stored, dict):
        raise ValueError("baseline Sim checkpoint has no normalizer")
    normalizer_sha = rehearsal.canonical_sha256(stored)
    if normalizer_sha != BASELINE_NORMALIZER_SHA256:
        raise ValueError("baseline normalizer hash differs from frozen S4K1")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or "target_std" not in state:
        raise ValueError("baseline model state has no target_std")
    target_std = torch.as_tensor(state["target_std"], dtype=torch.float64)
    stored_std = torch.as_tensor(stored.get("std_target"), dtype=torch.float64)
    if target_std.shape != stored_std.shape or not torch.equal(
            target_std, stored_std):
        raise ValueError("baseline model target_std differs from normalizer")
    metadata["normalizer_sha256"] = normalizer_sha
    return metadata


def validate_baseline_summary_contract(metadata: dict) -> None:
    if metadata["model_config_sha256"] != BASELINE_MODEL_CONFIG_SHA256:
        raise ValueError("summary model_config hash differs from frozen S4K1")
    if metadata["training_config_sha256"] != BASELINE_TRAINING_CONFIG_SHA256:
        raise ValueError("summary training_config hash differs from frozen S4K1")


def load_teacher_cache(args: argparse.Namespace, manifest: list, dataset):
    if shg.sha256(args.teacher_cache) != TEACHER_CACHE_SHA256:
        raise RuntimeError("teacher cache SHA differs before deserialization")
    cache, metadata = rehearsal.load_and_validate_cache(
        args.teacher_cache, manifest, dataset)
    if metadata["sha256"] != TEACHER_CACHE_SHA256:
        raise RuntimeError("validated teacher cache SHA differs from frozen P1")
    return cache, metadata


def training_args(
        args: argparse.Namespace, baseline_training: dict) -> argparse.Namespace:
    result = shg.training_args(args, baseline_training)
    overrides = {
        "updates": TERMINAL_LOCAL_UPDATE, "lr": REFINEMENT_LR,
        "start_checkpoint": args.baseline_sim, "method": METHOD,
        "start_global_update": START_GLOBAL_UPDATE,
        "terminal_global_update": TERMINAL_GLOBAL_UPDATE,
        "scheduler_t_max": SCHEDULER_T_MAX,
        "optimizer_reset": True, "batch_sequence_restart": True,
        "baseline_sim_sha256": BASELINE_SIM_SHA256,
        "baseline_summary_sha256": BASELINE_SUMMARY_SHA256,
        "baseline_split_manifest_sha256": BASELINE_SPLIT_MANIFEST_SHA256,
        "teacher_cache_sha256": TEACHER_CACHE_SHA256,
        "real_stats_sha256": REAL_STATS_SHA256,
    }
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


class NaturalIndexDataset(Dataset):
    def __init__(self, length: int):
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return int(index)


def _training_order_loader(dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))


def derive_prefix_contract(dataset, args: argparse.Namespace) -> dict:
    if len(dataset) < EXPECTED_SIM_WINDOWS:
        raise RuntimeError(
            f"Sim dataset has {len(dataset)} windows, expected "
            f"{EXPECTED_SIM_WINDOWS}")
    if PREFIX_LOCAL_UPDATE * args.batch_size > len(dataset):
        raise RuntimeError("prefix selection unexpectedly spans multiple epochs")
    preview = _training_order_loader(NaturalIndexDataset(len(dataset)), args)
    batch_indices = []
    batch_keys = []
    for update, indices_tensor in enumerate(preview, start=1):
        if update > PREFIX_LOCAL_UPDATE:
            break
        indices = [int(value) for value in indices_tensor.tolist()]
        trajectories = torch.tensor([
            int(dataset.windows[index].trajectory) for index in indices],
            dtype=torch.long)
        sample_indices = torch.tensor(indices, dtype=torch.long)
        batch_indices.append(indices)
        batch_keys.append(shg.batch_key(trajectories, sample_indices))
    if len(batch_indices) != PREFIX_LOCAL_UPDATE:
        raise RuntimeError("could not derive all 200 formal prefix batches")

    prefix_set = {index for batch in batch_indices for index in batch}
    seen = []
    seen_set = set()
    for batch in batch_indices:
        for index in batch:
            if index not in seen_set and len(seen) < ANCHOR_SIZE:
                seen.append(index)
                seen_set.add(index)
    unseen = [
        index for index in range(len(dataset)) if index not in prefix_set
    ][:ANCHOR_SIZE]
    if len(seen) != ANCHOR_SIZE or len(unseen) != ANCHOR_SIZE:
        raise RuntimeError("formal Seen/Unseen anchors are incomplete")
    contract = {
        "method": "seed0_dataloader_exact_prefix_natural_indices_v1",
        "updates": PREFIX_LOCAL_UPDATE,
        "batch_size": int(args.batch_size),
        "selection_reads_labels_or_metrics": False,
        "batch_sample_indices": batch_indices,
        "batch_sample_indices_sha256": rehearsal.canonical_sha256(
            batch_indices),
        "batch_keys": batch_keys,
        "batch_keys_sha256": rehearsal.canonical_sha256(batch_keys),
        "seen_indices": seen,
        "seen_indices_sha256": rehearsal.canonical_sha256(seen),
        "unseen_indices": unseen,
        "unseen_indices_sha256": rehearsal.canonical_sha256(unseen),
    }
    contract["canonical_sha256"] = rehearsal.canonical_sha256(contract)
    return contract


def condition_balanced_contract(dataset) -> dict:
    if len(dataset) != EXPECTED_SIM_WINDOWS:
        raise RuntimeError(
            f"condition-balanced Sim requires {EXPECTED_SIM_WINDOWS} windows")
    counts = Counter(int(window.trajectory) for window in dataset.windows)
    if len(counts) != EXPECTED_SIM_CONDITIONS or set(counts.values()) != {
            EXPECTED_WINDOWS_PER_CONDITION}:
        raise RuntimeError(
            "Sim windows are not 100 conditions x 48 natural windows")
    indices = list(range(len(dataset)))
    contract = {
        "method": "all_natural_sim_windows_condition_balanced_v1",
        "indices": indices,
        "indices_sha256": rehearsal.canonical_sha256(indices),
        "condition_counts": {
            str(key): counts[key] for key in sorted(counts)},
        "conditions": len(counts),
        "windows_per_condition": EXPECTED_WINDOWS_PER_CONDITION,
        "windows": len(indices),
    }
    contract["canonical_sha256"] = rehearsal.canonical_sha256(contract)
    return contract


@torch.no_grad()
def evaluate_anchor(
        model: EnsembleUNet2d, dataset, indices: list[int], normalizer,
        soft_sps_normalizer, teacher_cache: torch.Tensor,
        args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    point_loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(0))
    metrics = rehearsal.ensemble_train.common.repro.evaluate(
        model, point_loader, shg.SimEvaluationNormalizer(normalizer), device)
    rehearsal.ensemble_train.common.add_weighted(
        metrics, tuple(float(value) for value in args.accuracy_weights))
    indexed_loader = DataLoader(
        shg.IndexedDataset(dataset, indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(0))
    sums: dict[str, float] = {}
    count = 0
    model.eval()
    for history, target, sample_indices in indexed_loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history, target)
        prediction, _members, _quantiles, lower, upper, _features, _coverage = (
            model.forward_uq_training(history_norm))
        teacher = teacher_cache[sample_indices].to(
            device=device, dtype=torch.float32, non_blocking=True)
        distillation, components = (
            rehearsal.ensemble_train.component_point_distillation_loss(
                prediction, teacher, normalizer))
        _soft_loss, soft_score = (
            rehearsal.ensemble_train.soft_native_sps_objective(
                prediction, target_norm, lower, upper,
                soft_sps_normalizer, args.soft_sps_tau_fraction,
                metric_normalizer=normalizer))
        batch = history.shape[0]
        values = {
            "p1_distill_loss": float(distillation),
            "native_soft_sps_proxy": float(soft_score),
            **{key: float(value) for key, value in components.items()},
        }
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + batch * value
        count += batch
    metrics.update({key: value / count for key, value in sums.items()})
    result = {key: float(value) for key, value in metrics.items()}
    if not result or not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("non-finite S4K1 anchor metric")
    return result


def make_optimizer(model: EnsembleUNet2d, args: argparse.Namespace):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=REFINEMENT_LR,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX)
    return optimizer, scheduler


def sim_update(
        model: EnsembleUNet2d, batch, normalizer, soft_sps_normalizer,
        teacher_cache: torch.Tensor, args: argparse.Namespace,
        device: torch.device, local_update: int, optimizer, scheduler,
        ) -> tuple[dict[str, float], str, list[int]]:
    global_update = START_GLOBAL_UPDATE + local_update
    model.train()
    history, target, trajectory_ids, sample_indices = batch
    key = shg.batch_key(trajectory_ids, sample_indices)
    natural_indices = [int(value) for value in sample_indices.tolist()]
    history = history.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    history_norm, target_norm = normalizer.preprocess(history, target)
    optimizer.zero_grad(set_to_none=True)
    (prediction, members, quantiles, lower, upper, _features,
     coverage_logits) = model.forward_uq_training(history_norm)
    weights = tuple(float(value) for value in args.accuracy_weights)
    point_loss, point_components = rehearsal.ensemble_train.point_loss(
        args, prediction, target_norm, normalizer, weights, global_update)
    if args.member_loss_mode == "point":
        member_loss, bootstrap_fraction = (
            rehearsal.ensemble_train.bootstrap_member_point_loss(
                args, members, target_norm, trajectory_ids, normalizer,
                weights, global_update))
    else:
        member_loss, bootstrap_fraction = (
            rehearsal.ensemble_train.bootstrap_member_mse(
                members, target_norm, trajectory_ids,
                args.bootstrap_probability, args.seed))
    quantile_loss = rehearsal.ensemble_train.residual_quantile_loss(
        model, prediction, quantiles, target_norm, normalizer)
    soft_sps_loss, soft_sps_score = (
        rehearsal.ensemble_train.soft_native_sps_objective(
            prediction, target_norm, lower, upper, soft_sps_normalizer,
            args.soft_sps_tau_fraction, metric_normalizer=normalizer))
    coverage_bce = (
        rehearsal.ensemble_train.coverage_calibration_objective(
            model, prediction, quantiles, coverage_logits,
            target_norm, normalizer)
        if coverage_logits is not None else prediction.new_zeros(()))
    profile_loss = (
        rehearsal.ensemble_train.mean_profile_objective(
            prediction, target_norm, normalizer)
        if args.profile_loss_weight else prediction.new_zeros(()))
    base_loss = (
        args.mean_loss_weight * point_loss
        + args.member_loss_weight * member_loss
        + args.quantile_loss_weight * quantile_loss
        + args.coverage_bce_weight * coverage_bce
        + args.profile_loss_weight * profile_loss)
    teacher = teacher_cache[sample_indices].to(
        device=device, dtype=torch.float32, non_blocking=True)
    distillation, distillation_components = (
        rehearsal.ensemble_train.component_point_distillation_loss(
            prediction, teacher, normalizer))
    supervised = base_loss + args.distill_weight * distillation
    soft_sps_weight = args.soft_sps_loss_weight * min(
        1.0, global_update / max(args.soft_sps_warmup_updates, 1))
    total = supervised + soft_sps_weight * soft_sps_loss
    if not torch.isfinite(total):
        raise FloatingPointError(
            f"non-finite S4K1 loss at local update {local_update}")
    total.backward()
    if args.grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    scheduler.step()
    values = {
        "local_update": local_update,
        "global_update": global_update,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "loss": float(total.detach()),
        "supervised_loss": float(supervised.detach()),
        "point_loss": float(point_loss.detach()),
        "member_loss": float(member_loss.detach()),
        "quantile_loss": float(quantile_loss.detach()),
        "soft_sps_loss": float(soft_sps_loss.detach()),
        "soft_sps_score": float(soft_sps_score.detach()),
        "soft_sps_weight": float(soft_sps_weight),
        "p1_distill_loss": float(distillation.detach()),
        "bootstrap_fraction": float(bootstrap_fraction.detach()),
        **{key: float(value.detach())
           for key, value in point_components.items()},
        **{key: float(value.detach())
           for key, value in distillation_components.items()},
    }
    return values, key, natural_indices


def _relative_at_most(value: float, reference: float, multiplier: float) -> bool:
    if reference <= 1e-12:
        return value <= reference + 1e-12
    return value <= multiplier * reference + 1e-12


def _relative_decline(initial: float, terminal: float) -> float:
    if initial <= 1e-12:
        return 0.0 if terminal <= initial + 1e-12 else -float("inf")
    return (initial - terminal) / initial


def prefix_gate(
        seen_initial: dict, seen_terminal: dict,
        unseen_initial: dict, unseen_terminal: dict) -> dict:
    seen_components = {
        key: seen_terminal[key] - seen_initial[key]
        for key in POINT_COMPONENT_KEYS}
    unseen_components = {
        key: unseen_terminal[key] - unseen_initial[key]
        for key in POINT_COMPONENT_KEYS}
    seen_point_gain = seen_terminal["point_score"] - seen_initial["point_score"]
    unseen_point_gain = (
        unseen_terminal["point_score"] - unseen_initial["point_score"])
    seen_decline = _relative_decline(
        seen_initial["p1_distill_loss"], seen_terminal["p1_distill_loss"])
    unseen_decline = _relative_decline(
        unseen_initial["p1_distill_loss"], unseen_terminal["p1_distill_loss"])
    checks = {
        "seen_p1_decline": _relative_at_most(
            seen_terminal["p1_distill_loss"],
            seen_initial["p1_distill_loss"], SEEN_P1_RATIO_MAX),
        "seen_point_gain": seen_point_gain >= SEEN_POINT_DELTA_MIN - 1e-12,
        "seen_components": all(
            value >= PREFIX_COMPONENT_DELTA_FLOOR - 1e-12
            for value in seen_components.values()),
        "seen_soft_sps": (
            seen_terminal["native_soft_sps_proxy"]
            - seen_initial["native_soft_sps_proxy"]
            >= PREFIX_SOFT_SPS_DELTA_FLOOR - 1e-12),
        "unseen_p1_decline": _relative_at_most(
            unseen_terminal["p1_distill_loss"],
            unseen_initial["p1_distill_loss"], UNSEEN_P1_RATIO_MAX),
        "unseen_point_gain": unseen_point_gain >= UNSEEN_POINT_DELTA_MIN - 1e-12,
        "unseen_components": all(
            value >= PREFIX_COMPONENT_DELTA_FLOOR - 1e-12
            for value in unseen_components.values()),
        "unseen_soft_sps": (
            unseen_terminal["native_soft_sps_proxy"]
            - unseen_initial["native_soft_sps_proxy"]
            >= PREFIX_SOFT_SPS_DELTA_FLOOR - 1e-12),
        "unseen_point_tracks_seen": (
            unseen_point_gain
            >= seen_point_gain + UNSEEN_POINT_GAP_FROM_SEEN_FLOOR - 1e-12),
        "unseen_p1_tracks_seen": (
            unseen_decline
            >= UNSEEN_P1_DECLINE_FRACTION_OF_SEEN_MIN * seen_decline - 1e-12),
    }
    passed = all(checks.values())
    return {
        "status": PREFIX_PASS_STATUS if passed else PREFIX_STOP_STATUS,
        "pass": passed,
        "checks": checks,
        "seen_initial": seen_initial,
        "seen_terminal": seen_terminal,
        "unseen_initial": unseen_initial,
        "unseen_terminal": unseen_terminal,
        "seen_component_delta": seen_components,
        "unseen_component_delta": unseen_components,
        "seen_point_gain": seen_point_gain,
        "unseen_point_gain": unseen_point_gain,
        "seen_p1_relative_decline": seen_decline,
        "unseen_p1_relative_decline": unseen_decline,
    }


def terminal_gate(baseline: dict, candidate: dict) -> dict:
    component_delta = {
        key: candidate[key] - baseline[key] for key in POINT_COMPONENT_KEYS}
    checks = {
        "p1_decline": _relative_at_most(
            candidate["p1_distill_loss"], baseline["p1_distill_loss"],
            TERMINAL_P1_RATIO_MAX),
        "point_gain": (
            candidate["point_score"] - baseline["point_score"]
            >= TERMINAL_POINT_DELTA_MIN - 1e-12),
        "components": all(
            value >= TERMINAL_COMPONENT_DELTA_FLOOR - 1e-12
            for value in component_delta.values()),
        "soft_sps": (
            candidate["native_soft_sps_proxy"]
            - baseline["native_soft_sps_proxy"]
            >= TERMINAL_SOFT_SPS_DELTA_FLOOR - 1e-12),
    }
    passed = all(checks.values())
    return {
        "status": PASS_STATUS if passed else TERMINAL_STOP_STATUS,
        "pass": passed,
        "checks": checks,
        "baseline": baseline,
        "candidate": candidate,
        "component_delta": component_delta,
        "point_delta": candidate["point_score"] - baseline["point_score"],
        "soft_sps_proxy_delta": (
            candidate["native_soft_sps_proxy"]
            - baseline["native_soft_sps_proxy"]),
    }


def save_terminal_checkpoint(
        gate: dict, output_dir: Path, model: EnsembleUNet2d,
        args: argparse.Namespace, history: list[dict], normalizer,
        ) -> tuple[str | None, str | None]:
    if not gate.get("pass"):
        return None, None
    args.prefix_gate_pass = True
    args.sim_gate_pass = True
    args.checkpoint_policy = "only_after_terminal_sim_gate_pass"
    checkpoint = output_dir / "model_14000.pth"
    rehearsal.ensemble_train.common.save_checkpoint(
        checkpoint, model, args, TERMINAL_GLOBAL_UPDATE, history,
        rehearsal.ensemble_train.common.normalizer_dict(normalizer), gate)
    return str(checkpoint), shg.sha256(checkpoint)


def run_experiment(
        args: argparse.Namespace, payload: dict, baseline_metadata: dict,
        baseline_summary_metadata: dict, sim_args: argparse.Namespace,
        raw_loader, normalizer, soft_sps_normalizer,
        teacher_cache: torch.Tensor, teacher_metadata: dict,
        normalizer_metadata: dict, device: torch.device,
        ) -> dict:
    started = time.time()
    dataset = raw_loader.dataset
    prefix_contract = derive_prefix_contract(dataset, sim_args)
    balanced_contract = condition_balanced_contract(dataset)
    print(json.dumps({
        "event": "c208_s4k1_prefix_contract",
        "canonical_sha256": prefix_contract["canonical_sha256"],
        "batch_keys_sha256": prefix_contract["batch_keys_sha256"],
        "seen_indices_sha256": prefix_contract["seen_indices_sha256"],
        "unseen_indices_sha256": prefix_contract["unseen_indices_sha256"],
    }), flush=True)

    model = shg.build_model(payload, device, load_weights=True)
    rehearsal.ensemble_train.common.set_model_stats(model, normalizer)
    seen_initial = evaluate_anchor(
        model, dataset, prefix_contract["seen_indices"], normalizer,
        soft_sps_normalizer, teacher_cache, sim_args, device)
    unseen_initial = evaluate_anchor(
        model, dataset, prefix_contract["unseen_indices"], normalizer,
        soft_sps_normalizer, teacher_cache, sim_args, device)
    balanced_initial = evaluate_anchor(
        model, dataset, balanced_contract["indices"], normalizer,
        soft_sps_normalizer, teacher_cache, sim_args, device)

    rehearsal.ensemble_train.seed_everything(sim_args.seed)
    training_loader = rehearsal.ensemble_train.tag_training_loader(
        raw_loader, sim_args)
    training_iterator = rehearsal.ensemble_train.common.repro.cycling(
        training_loader)
    optimizer, scheduler = make_optimizer(model, sim_args)
    history = []
    batch_keys = []
    log_interval = int(sim_args.eval_interval)
    last_values = None
    for local_update in range(1, PREFIX_LOCAL_UPDATE + 1):
        last_values, key, natural_indices = sim_update(
            model, next(training_iterator), normalizer, soft_sps_normalizer,
            teacher_cache, sim_args, device, local_update,
            optimizer, scheduler)
        expected_indices = prefix_contract["batch_sample_indices"][
            local_update - 1]
        expected_key = prefix_contract["batch_keys"][local_update - 1]
        if natural_indices != expected_indices or key != expected_key:
            raise RuntimeError(
                f"actual S4K1 prefix differs at local update {local_update}")
        batch_keys.append(key)
    history.append(dict(last_values))
    seen_terminal = evaluate_anchor(
        model, dataset, prefix_contract["seen_indices"], normalizer,
        soft_sps_normalizer, teacher_cache, sim_args, device)
    unseen_terminal = evaluate_anchor(
        model, dataset, prefix_contract["unseen_indices"], normalizer,
        soft_sps_normalizer, teacher_cache, sim_args, device)
    preflight = prefix_gate(
        seen_initial, seen_terminal, unseen_initial, unseen_terminal)
    print(json.dumps({
        "event": "c208_s4k1_prefix_gate", **preflight,
    }), flush=True)

    terminal = None
    checkpoint = checkpoint_sha = None
    completed = PREFIX_LOCAL_UPDATE
    if preflight["pass"]:
        for local_update in range(
                PREFIX_LOCAL_UPDATE + 1, TERMINAL_LOCAL_UPDATE + 1):
            last_values, key, _natural_indices = sim_update(
                model, next(training_iterator), normalizer,
                soft_sps_normalizer, teacher_cache, sim_args, device,
                local_update, optimizer, scheduler)
            batch_keys.append(key)
            if local_update % log_interval == 0 \
                    or local_update == TERMINAL_LOCAL_UPDATE:
                history.append(dict(last_values))
                print(json.dumps({
                    "event": "c208_s4k1_progress", **last_values,
                }), flush=True)
        completed = TERMINAL_LOCAL_UPDATE
        balanced_terminal = evaluate_anchor(
            model, dataset, balanced_contract["indices"], normalizer,
            soft_sps_normalizer, teacher_cache, sim_args, device)
        terminal = terminal_gate(balanced_initial, balanced_terminal)
        print(json.dumps({
            "event": "c208_s4k1_terminal_gate", **terminal,
        }), flush=True)
        checkpoint, checkpoint_sha = save_terminal_checkpoint(
            terminal, args.output_dir, model, sim_args, history, normalizer)

    status = preflight["status"] if not preflight["pass"] else terminal["status"]
    frozen_loss = {
        key: baseline_metadata["training_config"][key]
        for key in shg.FROZEN_SIM_LOSS_KEYS}
    frozen_loss["lr"] = REFINEMENT_LR
    frozen_loss["scheduler_t_max"] = SCHEDULER_T_MAX
    summary = {
        "method": METHOD,
        "status": status,
        "completed_local_updates": completed,
        "completed_global_update": START_GLOBAL_UPDATE + completed,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "model_config": payload["model_config"],
        "parameters": rehearsal.EXPECTED_PARAMETER_COUNT,
        "baseline_sim": {
            **{key: baseline_metadata[key] for key in (
                "path", "sha256", "model_config_sha256",
                "training_config_sha256", "normalizer_sha256")},
            "source_summary": baseline_summary_metadata,
            "condition_balanced_anchor": balanced_initial,
        },
        "teacher_cache": teacher_metadata,
        "sim_normalizer": normalizer_metadata,
        "prefix_contract": prefix_contract,
        "condition_balanced_contract": balanced_contract,
        "prefix_gate": preflight,
        "sim_gate": terminal,
        "batch_sequence_sha256_completed": rehearsal.canonical_sha256(
            batch_keys),
        "history": history,
        "training_contract": {
            "only_scientific_delta": (
                "fresh_adamw_low_lr_continuation_from_exact_sim10000"),
            "optimizer": "fresh_adamw",
            "lr": REFINEMENT_LR,
            "weight_decay": float(sim_args.weight_decay),
            "grad_clip": float(sim_args.grad_clip),
            "scheduler": "cosine_annealing",
            "scheduler_t_max": SCHEDULER_T_MAX,
            "start_global_update": START_GLOBAL_UPDATE,
            "terminal_local_update": TERMINAL_LOCAL_UPDATE,
            "terminal_global_update": TERMINAL_GLOBAL_UPDATE,
            "loss_iteration": "global_10000_plus_local_update",
            "optimizer_reset": True,
            "scheduler_reset": True,
            "loader_restart_seed": int(sim_args.seed),
            "same_process_after_prefix_pass": True,
            "frozen_sim_loss": frozen_loss,
            "checkpoint_policy": "only_after_terminal_sim_gate_pass",
            "fixed725_windows_used": 0,
        },
        "evidence_class": (
            "released_sim_train_only_seen_unseen_and_condition_balanced_"
            "diagnostic"),
        "wall_seconds": time.time() - started,
    }
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal C208-S4K1 training requires CUDA")
    if shg.sha256(args.real_stats) != REAL_STATS_SHA256:
        raise RuntimeError("Real stats SHA differs from frozen SPS proxy scale")
    if shg.sha256(args.baseline_split_manifest) \
            != BASELINE_SPLIT_MANIFEST_SHA256:
        raise RuntimeError("baseline Sim split manifest SHA differs")
    payload, baseline_metadata = shg.load_baseline(args)
    validate_baseline_contract(payload, baseline_metadata)
    baseline_summary_metadata = shg.load_baseline_summary(
        args, payload, baseline_metadata)
    validate_baseline_summary_contract(baseline_summary_metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim_args = training_args(args, baseline_metadata["training_config"])
    rehearsal.ensemble_train.seed_everything(sim_args.seed)
    torch.set_float32_matmul_precision("high")
    raw_loader, normalizer, train_windows, trajectories = (
        rehearsal.ensemble_train.common.prepare_sim(sim_args, device))
    if train_windows != EXPECTED_SIM_WINDOWS \
            or trajectories != EXPECTED_SIM_CONDITIONS:
        raise RuntimeError(
            f"Sim shape={train_windows}/{trajectories}, expected "
            f"{EXPECTED_SIM_WINDOWS}/{EXPECTED_SIM_CONDITIONS}")
    runtime_manifest_path = args.output_dir / "split_manifest.json"
    if shg.sha256(runtime_manifest_path) != BASELINE_SPLIT_MANIFEST_SHA256:
        raise RuntimeError("runtime Sim split manifest differs from baseline")
    normalizer_metadata = shg.validate_runtime_normalizer(payload, normalizer)
    if normalizer_metadata["baseline_sha256"] != BASELINE_NORMALIZER_SHA256 \
            or normalizer_metadata["runtime_sha256"] != BASELINE_NORMALIZER_SHA256:
        raise RuntimeError("runtime Sim normalizer hash differs from baseline")
    manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    teacher_cache, teacher_metadata = load_teacher_cache(
        args, manifest, raw_loader.dataset)
    shg.validate_teacher_summary_binding(
        teacher_metadata, baseline_summary_metadata)
    soft_sps_normalizer = rehearsal.ensemble_train.common.repro.Normalizer(
        args.real_stats, device)
    print(json.dumps({
        "event": "c208_s4k1_contract",
        "method": METHOD,
        "baseline_sim_sha256": BASELINE_SIM_SHA256,
        "baseline_summary_sha256": BASELINE_SUMMARY_SHA256,
        "baseline_split_manifest_sha256": BASELINE_SPLIT_MANIFEST_SHA256,
        "teacher_cache_sha256": TEACHER_CACHE_SHA256,
        "real_stats_sha256": REAL_STATS_SHA256,
        "lr": REFINEMENT_LR,
        "scheduler_t_max": SCHEDULER_T_MAX,
    }), flush=True)
    summary = run_experiment(
        args, payload, baseline_metadata, baseline_summary_metadata,
        sim_args, raw_loader, normalizer, soft_sps_normalizer,
        teacher_cache, teacher_metadata, normalizer_metadata, device)
    shg.write_summary(args.output_dir, summary)
    print(json.dumps({
        "event": "c208_s4k1_complete",
        "status": summary["status"],
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
    }), flush=True)


if __name__ == "__main__":
    main()
