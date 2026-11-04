#!/usr/bin/env python3
"""LOAO audit of a hidden-coordinate-free interval-width allocator."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset

import train_b210_point_uq_function_distill_oof as source_oof
import train_b210_small4k_uq_release as uq
import train_b210_full_prefix_uq_refit as full


EXPERIMENT = "b210_output_space_uq_allocator_oof"
SCHEMA = "b210_output_space_uq_allocator_oof_v1"
ENDPOINT_SCHEMA = "b210_output_space_uq_allocator_endpoint_v1"
CHANNELS = 48
BLOCKS = 2
LOG_LIMIT = 0.75
PROMOTION = {
    "pooled_sps_delta_vs_source_min": 0.10,
    "positive_folds_vs_source_min": 4,
    "worst_fold_vs_source_min": -0.05,
    "pooled_sps_delta_vs_native_continue_min": 0.03,
    "positive_folds_vs_native_continue_min": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--point-checkpoint-sha256", required=True)
    parser.add_argument("--source-oof-summary", type=Path, required=True)
    parser.add_argument("--source-oof-summary-sha256", required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=7200.0)
    return parser.parse_args()


def frozen_without_allocator_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("output_space_uq_allocator.")
    }


def frozen_without_output_space_state(
        model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith((
            "output_space_uq_allocator.",
            "output_space_uq_budgeter."))
    }


def upgrade_output_allocator(
        point_payload: Mapping[str, object], normalizer,
        source_uq_state: Mapping[str, torch.Tensor],
        device: torch.device) -> torch.nn.Module:
    base_config = full.ft12.validate_b210_model_config(
        point_payload["model_config"], full.sim_tke.b210_config("candidate"))
    config = replace(
        base_config,
        output_space_uq_channels=CHANNELS,
        output_space_uq_blocks=BLOCKS,
        output_space_uq_log_limit=LOG_LIMIT,
    )
    model = full.EnsembleUNet2d(config).float()
    destination = model.state_dict()
    source = dict(point_payload["model_state_dict"])
    source.update(source_uq_state)
    missing = set(destination) - set(source)
    unexpected = set(source) - set(destination)
    if unexpected or any(
            not key.startswith("output_space_uq_allocator.") for key in missing):
        raise RuntimeError(
            "output-space allocator migration contract changed: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, value in source.items():
        if destination[key].shape != value.shape:
            raise RuntimeError(f"output-space allocator shape drifted at {key}")
        destination[key] = value
    model.load_state_dict(destination, strict=True)
    full.ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def configure_output_space_only(
        model: torch.nn.Module, *, train_allocator: bool,
        train_budgeter: bool) -> tuple[torch.nn.Parameter, ...]:
    if not train_allocator and not train_budgeter:
        raise ValueError("at least one output-space UQ module must train")
    modules = []
    if train_allocator:
        if model.output_space_uq_allocator is None:
            raise RuntimeError("output-space allocator is disabled")
        modules.append(model.output_space_uq_allocator)
    if train_budgeter:
        if model.output_space_uq_budgeter is None:
            raise RuntimeError("output-space budgeter is disabled")
        modules.append(model.output_space_uq_budgeter)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    allowed = tuple(
        prefix for prefix, enabled in (
            ("output_space_uq_allocator.", train_allocator),
            ("output_space_uq_budgeter.", train_budgeter)) if enabled)
    if not names or any(not name.startswith(allowed) for name in names):
        raise RuntimeError("output-space-only trainability contract changed")
    return tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad)


def configure_allocator_only(model: torch.nn.Module) -> tuple[torch.nn.Parameter, ...]:
    return configure_output_space_only(
        model, train_allocator=True, train_budgeter=False)


def set_output_space_training_mode(
        model: torch.nn.Module, *, train_allocator: bool,
        train_budgeter: bool) -> None:
    model.eval()
    if train_allocator:
        model.output_space_uq_allocator.train()
    if train_budgeter:
        model.output_space_uq_budgeter.train()
    if model.training:
        raise RuntimeError("frozen point model left eval mode")


def set_allocator_training_mode(model: torch.nn.Module) -> None:
    set_output_space_training_mode(
        model, train_allocator=True, train_budgeter=False)


def train_output_space_terminal(
        model: torch.nn.Module, dataset, indices: Sequence[int], normalizer,
        device: torch.device, deadline: float, seed: int, label: str, *,
        train_allocator: bool, train_budgeter: bool) -> dict:
    if not indices:
        raise ValueError("output-space optimizer partition is empty")
    uq.uq_train.seed_everything(seed)
    parameters = configure_output_space_only(
        model, train_allocator=train_allocator,
        train_budgeter=train_budgeter)
    frozen_before = full.dr1.model_state_sha256(
        frozen_without_output_space_state(model))
    loader = DataLoader(
        Subset(dataset, list(indices)), batch_size=uq.BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed))
    iterator = uq.ensemble_train.common.repro.cycling(loader)
    optimizer = torch.optim.AdamW(
        parameters, lr=uq.LR, weight_decay=uq.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=uq.SCHEDULER_T_MAX)
    objective_args = uq.sharp2_args()
    history = []
    sums: dict[str, float] = {}
    count = 0
    started = time.perf_counter()
    gradient_preflight = None
    for iteration in range(1, uq.FOLD_UPDATES + 1):
        if time.perf_counter() > deadline:
            raise TimeoutError("output-space allocator wall budget expired")
        set_output_space_training_mode(
            model, train_allocator=train_allocator,
            train_budgeter=train_budgeter)
        history_batch, target = next(iterator)
        history_batch = history_batch.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history_batch, target)
        optimizer.zero_grad(set_to_none=True)
        prediction, quantiles, extents, logits = uq.uq_train.quantile_forward(
            model, history_norm)
        loss, components = uq.uq_train.uq_objective(
            objective_args, model, prediction, quantiles, target_norm,
            normalizer, interval_extents=extents, coverage_logits=logits)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"non-finite allocator loss at {label} u{iteration}")
        loss.backward()
        gradients = [value.grad for value in parameters if value.grad is not None]
        if not gradients or any(
                not bool(torch.isfinite(value).all()) for value in gradients):
            raise FloatingPointError(
                f"invalid allocator gradient at {label} u{iteration}")
        if iteration == 1:
            squared = sum(
                float(value.detach().float().square().sum())
                for value in gradients)
            if not math.isfinite(squared) or squared <= 0.0:
                raise RuntimeError("allocator gradient preflight is zero")
            gradient_preflight = {
                "iteration": 1, "finite": True, "nonzero": True,
                "l2_norm": math.sqrt(squared)}
        clipped = torch.nn.utils.clip_grad_norm_(parameters, uq.GRAD_CLIP)
        if not bool(torch.isfinite(clipped)):
            raise FloatingPointError("allocator clipped gradient is non-finite")
        optimizer.step()
        scheduler.step()
        for key, value in components.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration % uq.LOG_INTERVAL == 0:
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps({
                "event": "b210_output_space_uq_progress",
                "label": label, **row}), flush=True)
            sums.clear()
            count = 0
    frozen_after = full.dr1.model_state_sha256(
        frozen_without_output_space_state(model))
    if frozen_after != frozen_before:
        raise RuntimeError("point/base-UQ state drifted during allocator fit")
    model.eval()
    return {
        "updates": uq.FOLD_UPDATES,
        "seed": seed,
        "batch_size": uq.BATCH_SIZE,
        "optimizer": "AdamW",
        "lr": uq.LR,
        "weight_decay": uq.WEIGHT_DECAY,
        "scheduler_t_max": uq.SCHEDULER_T_MAX,
        "objective": {
            key: value for key, value in vars(objective_args).items()
            if isinstance(value, (bool, float, int, str)) or value is None},
        "gradient_preflight": gradient_preflight,
        "train_allocator": train_allocator,
        "train_budgeter": train_budgeter,
        "trainable_parameters": sum(value.numel() for value in parameters),
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "frozen_state_bitwise": True,
        "history": history,
        "wall_seconds": time.perf_counter() - started,
    }


def train_allocator_terminal(
        model: torch.nn.Module, dataset, indices: Sequence[int], normalizer,
        device: torch.device, deadline: float, seed: int, label: str) -> dict:
    return train_output_space_terminal(
        model, dataset, indices, normalizer, device, deadline, seed, label,
        train_allocator=True, train_budgeter=False)


def save_endpoint(
        path: Path, model: torch.nn.Module, *, held_aoa: int,
        arm: str, training: Mapping[str, object]) -> dict:
    state = {
        key: value.detach().cpu().clone()
        for key, value in uq.partition_state(model, uq=True).items()}
    state_sha = full.dr1.model_state_sha256(state)
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": held_aoa,
        "arm": arm,
        "iteration": uq.FOLD_UPDATES,
        "uq_state_dict": state,
        "uq_state_sha256": state_sha,
        "training": dict(training),
        "held_checkpoint_selection": False,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": full.dr1.sha256(path),
        "uq_state_sha256": state_sha,
    }


def load_endpoint(path: Path, expected_sha: str) -> Mapping[str, object]:
    if full.dr1.sha256(path) != expected_sha:
        raise RuntimeError("output-space UQ endpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("uq_state_dict")
    if (payload.get("schema") != ENDPOINT_SCHEMA
            or not isinstance(state, Mapping)
            or full.dr1.model_state_sha256(state)
            != payload.get("uq_state_sha256")):
        raise RuntimeError("output-space UQ endpoint seal is invalid")
    return payload


def pooled(rows: Sequence[Mapping[str, object]], arm: str) -> dict:
    state = full.smu1.merge_metric_states(
        [row["arms"][arm]["metric_state"] for row in rows])
    return full.smu1.finalize_metric_state(state)


def compare(rows: Sequence[Mapping[str, object]], candidate: str,
            control: str) -> dict:
    control_metrics = pooled(rows, control)
    candidate_metrics = pooled(rows, candidate)
    deltas = [
        float(row["arms"][candidate]["metrics"]["native_sps_score"])
        - float(row["arms"][control]["metrics"]["native_sps_score"])
        for row in rows]
    return {
        "control": control_metrics,
        "candidate": candidate_metrics,
        "native_sps_delta": (
            candidate_metrics["native_sps_score"]
            - control_metrics["native_sps_score"]),
        "fold_deltas": deltas,
        "positive_folds": sum(value > 0.0 for value in deltas),
        "worst_fold": min(deltas),
    }


def promotion(rows: Sequence[Mapping[str, object]]) -> dict:
    versus_source = compare(rows, "output_allocator", "source_u800")
    versus_native = compare(rows, "output_allocator", "native_continue_u1600")
    conditions = {
        "pooled_vs_source": versus_source["native_sps_delta"]
        >= PROMOTION["pooled_sps_delta_vs_source_min"],
        "positive_folds_vs_source": versus_source["positive_folds"]
        >= PROMOTION["positive_folds_vs_source_min"],
        "worst_fold_vs_source": versus_source["worst_fold"]
        >= PROMOTION["worst_fold_vs_source_min"],
        "pooled_vs_native_continue": versus_native["native_sps_delta"]
        >= PROMOTION["pooled_sps_delta_vs_native_continue_min"],
        "positive_folds_vs_native_continue": versus_native["positive_folds"]
        >= PROMOTION["positive_folds_vs_native_continue_min"],
    }
    return {
        "versus_source": versus_source,
        "versus_native_continue": versus_native,
        "conditions": conditions,
        "pass": all(conditions.values()),
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("output-space UQ OOF requires CUDA")
    if args.wall_budget_seconds <= 0.0 or not math.isfinite(
            args.wall_budget_seconds):
        raise ValueError("wall budget must be finite and positive")
    point_sha = full.dr1.sha256(args.point_checkpoint)
    summary_sha = full.dr1.sha256(args.source_oof_summary)
    if point_sha != args.point_checkpoint_sha256.lower():
        raise RuntimeError("point checkpoint SHA changed")
    if summary_sha != args.source_oof_summary_sha256.lower():
        raise RuntimeError("source OOF summary SHA changed")
    source_summary = json.loads(args.source_oof_summary.read_text())
    if (source_summary.get("schema") != source_oof.SCHEMA
            or len(source_summary.get("folds", ())) != len(full.dr1.AOA_VALUES)):
        raise RuntimeError("source OOF summary contract changed")
    source_folds = {
        int(row["held_aoa"]): row for row in source_summary["folds"]}
    rows, manifest = full.dr1.load_manifest(
        args.clean_manifest, full.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    point_payload, point_model, normalizer, point_report = full.load_point(
        args.point_checkpoint, point_sha, device)
    del point_model
    gc.collect()

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    fold_rows = []
    for held_aoa in full.dr1.AOA_VALUES:
        fold_dir = args.output_dir / f"held{held_aoa}"
        fold_dir.mkdir()
        fit_rows = [row for row in rows if int(row["aoa"]) != held_aoa]
        held_rows = [row for row in rows if int(row["aoa"]) == held_aoa]
        audit = source_oof.FoldAccessAudit(held_aoa)
        fit_trajectories, fit_metadata, fit_windows = full.dr1.load_prefix_rows(
            args.real_dir, fit_rows, "fit", audit)
        fit_dataset = full.ft12.npt1.ensemble_train.common.repro.WindowDataset(
            fit_trajectories, fit_windows)
        source_seal = source_folds[held_aoa]["arms"]["control"]["seal"]
        source_endpoint = source_oof.load_uq_endpoint(
            Path(source_seal["path"]), source_seal["file_sha256"])
        source_uq = source_endpoint["uq_state_dict"]

        native = source_oof.clone_point_model(point_payload, normalizer, device)
        current = native.state_dict()
        current.update(source_uq)
        native.load_state_dict(current, strict=True)
        native_training = uq.train_uq_terminal(
            native, fit_dataset, list(range(len(fit_windows))), normalizer,
            device, deadline,
            uq.SEED + full.dr1.AOA_VALUES.index(held_aoa) + 101,
            f"held{held_aoa}_native_continue")
        native_seal = save_endpoint(
            fold_dir / "native_continue_u1600.pth", native,
            held_aoa=held_aoa, arm="native_continue_u1600",
            training=native_training)
        del native
        gc.collect()
        torch.cuda.empty_cache()

        allocator = upgrade_output_allocator(
            point_payload, normalizer, source_uq, device)
        allocator_training = train_allocator_terminal(
            allocator, fit_dataset, list(range(len(fit_windows))), normalizer,
            device, deadline,
            uq.SEED + full.dr1.AOA_VALUES.index(held_aoa) + 201,
            f"held{held_aoa}_output_allocator")
        allocator_seal = save_endpoint(
            fold_dir / "output_allocator_u800.pth", allocator,
            held_aoa=held_aoa, arm="output_allocator",
            training=allocator_training)
        del allocator, fit_dataset, fit_trajectories
        gc.collect()
        torch.cuda.empty_cache()

        audit.seal()
        held_trajectories, held_metadata, held_windows = full.dr1.load_prefix_rows(
            args.real_dir, held_rows, "held", audit)
        held_dataset = full.ft12.npt1.ensemble_train.common.repro.WindowDataset(
            held_trajectories, held_windows)
        arms = {}
        for arm in ("source_u800", "native_continue_u1600", "output_allocator"):
            if arm == "source_u800":
                model = source_oof.clone_point_model(
                    point_payload, normalizer, device)
                current = model.state_dict()
                current.update(source_uq)
                model.load_state_dict(current, strict=True)
                seal = source_seal
            elif arm == "native_continue_u1600":
                endpoint = load_endpoint(
                    Path(native_seal["path"]), native_seal["file_sha256"])
                model = source_oof.clone_point_model(
                    point_payload, normalizer, device)
                current = model.state_dict()
                current.update(endpoint["uq_state_dict"])
                model.load_state_dict(current, strict=True)
                seal = native_seal
            else:
                endpoint = load_endpoint(
                    Path(allocator_seal["path"]),
                    allocator_seal["file_sha256"])
                model = upgrade_output_allocator(
                    point_payload, normalizer,
                    endpoint["uq_state_dict"], device)
                seal = allocator_seal
            metric_state, _by_aoa = uq.evaluate_grouped(
                model, held_dataset, list(range(len(held_windows))),
                held_metadata, normalizer, device, deadline)
            arms[arm] = {
                "metrics": full.smu1.finalize_metric_state(metric_state),
                "metric_state": metric_state,
                "seal": seal,
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()
        fold_row = {
            "held_aoa": held_aoa,
            "fit_trajectories": len(fit_rows),
            "fit_windows": len(fit_windows),
            "held_trajectories": len(held_rows),
            "held_windows": len(held_windows),
            "arms": arms,
            "access": {"sealed_before_held": audit.sealed,
                       "events": audit.events},
        }
        full.dr1.write_json_atomic(fold_dir / "result.json", fold_row)
        fold_rows.append(fold_row)
        del held_dataset, held_trajectories
        gc.collect()

    gate = promotion(fold_rows)
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "architecture": {
            "channels": CHANNELS, "blocks": BLOCKS,
            "log_limit": LOG_LIMIT,
            "inputs": "history_motion+mean_forecast+statistics+coordinates",
            "hidden_backbone_features": False,
            "mean_spatial_width_budget_preserved": True,
        },
        "promotion": dict(PROMOTION),
        "point": point_report,
        "source_oof": {
            "path": str(args.source_oof_summary.resolve()),
            "file_sha256": summary_sha,
            "decision": source_summary["decision"],
        },
        "manifest": manifest,
        "folds": fold_rows,
        "gate": gate,
        "elapsed_seconds": time.perf_counter() - started,
        "decision": (
            "PASS_OUTPUT_SPACE_UQ_ALLOCATOR_OOF"
            if gate["pass"] else "STOP_OUTPUT_SPACE_UQ_ALLOCATOR_OOF"),
        "submission_authorized": False,
    }
    full.dr1.write_json_atomic(args.output_dir / "summary.json", result)
    print(json.dumps({
        "event": "b210_output_space_uq_allocator_oof_complete",
        "decision": result["decision"], "gate": gate,
        "output": str(args.output_dir / "summary.json")}), flush=True)
    return result


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    run(args)


if __name__ == "__main__":
    main()
