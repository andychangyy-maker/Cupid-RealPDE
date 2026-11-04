#!/usr/bin/env python3
"""Separate global coverage budget from spatial interval allocation."""

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

import train_b210_full_prefix_uq_refit as full
import train_b210_output_space_uq_allocator_oof as spatial
import train_b210_point_uq_function_distill_oof as source_oof
import train_b210_small4k_uq_release as uq


EXPERIMENT = "b210_output_space_uq_budget_oof"
SCHEMA = "b210_output_space_uq_budget_oof_v1"
ENDPOINT_SCHEMA = "b210_output_space_uq_budget_endpoint_v1"
BUDGET_CHANNELS = 48
BUDGET_BLOCKS = 1
BUDGET_LOG_LIMIT = 0.50
ARMS = ("budget_only", "budget_plus_spatial")
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
    parser.add_argument("--spatial-oof-summary", type=Path, required=True)
    parser.add_argument("--spatial-oof-summary-sha256", required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=7200.0)
    return parser.parse_args()


def upgrade_model(
        point_payload: Mapping[str, object], normalizer,
        source_uq_state: Mapping[str, torch.Tensor], device: torch.device, *,
        spatial_allocator: bool) -> torch.nn.Module:
    base_config = full.ft12.validate_b210_model_config(
        point_payload["model_config"], full.sim_tke.b210_config("candidate"))
    config = replace(
        base_config,
        output_space_uq_channels=(spatial.CHANNELS if spatial_allocator else 0),
        output_space_uq_blocks=spatial.BLOCKS,
        output_space_uq_log_limit=spatial.LOG_LIMIT,
        output_space_uq_budget_channels=BUDGET_CHANNELS,
        output_space_uq_budget_blocks=BUDGET_BLOCKS,
        output_space_uq_budget_log_limit=BUDGET_LOG_LIMIT,
    )
    model = full.EnsembleUNet2d(config).float()
    destination = model.state_dict()
    source = dict(point_payload["model_state_dict"])
    source.update(source_uq_state)
    missing = set(destination) - set(source)
    unexpected = set(source) - set(destination)
    allowed_missing = ("output_space_uq_budgeter.",)
    if spatial_allocator:
        allowed_missing += ("output_space_uq_allocator.",)
    if unexpected or any(not key.startswith(allowed_missing) for key in missing):
        raise RuntimeError(
            "output-space budget migration contract changed: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, value in source.items():
        if destination[key].shape != value.shape:
            raise RuntimeError(f"output-space budget shape drifted at {key}")
        destination[key] = value
    model.load_state_dict(destination, strict=True)
    full.ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


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
        raise RuntimeError("output-space budget endpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("uq_state_dict")
    if (payload.get("schema") != ENDPOINT_SCHEMA
            or not isinstance(state, Mapping)
            or full.dr1.model_state_sha256(state)
            != payload.get("uq_state_sha256")):
        raise RuntimeError("output-space budget endpoint seal is invalid")
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


def candidate_gate(rows: Sequence[Mapping[str, object]], arm: str) -> dict:
    versus_source = compare(rows, arm, "source_u800")
    versus_native = compare(rows, arm, "native_continue_u1600")
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


def select(rows: Sequence[Mapping[str, object]]) -> dict:
    candidates = {arm: candidate_gate(rows, arm) for arm in ARMS}
    passing = [
        (row["versus_source"]["native_sps_delta"], arm)
        for arm, row in candidates.items() if row["pass"]]
    selected = max(passing)[1] if passing else None
    return {
        "candidates": candidates,
        "spatial_increment": compare(
            rows, "budget_plus_spatial", "budget_only"),
        "selected": selected,
        "pass": selected is not None,
        "selection_rule": (
            "largest pooled SPS gain versus source among passing arms"),
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("output-space budget OOF requires CUDA")
    if args.wall_budget_seconds <= 0.0 or not math.isfinite(
            args.wall_budget_seconds):
        raise ValueError("wall budget must be finite and positive")
    point_sha = full.dr1.sha256(args.point_checkpoint)
    summary_sha = full.dr1.sha256(args.spatial_oof_summary)
    if point_sha != args.point_checkpoint_sha256.lower():
        raise RuntimeError("point checkpoint SHA changed")
    if summary_sha != args.spatial_oof_summary_sha256.lower():
        raise RuntimeError("spatial OOF summary SHA changed")
    source_summary = json.loads(args.spatial_oof_summary.read_text())
    if (source_summary.get("schema") != spatial.SCHEMA
            or len(source_summary.get("folds", ())) != len(full.dr1.AOA_VALUES)):
        raise RuntimeError("spatial OOF summary contract changed")
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
        fit_trajectories, _fit_metadata, fit_windows = full.dr1.load_prefix_rows(
            args.real_dir, fit_rows, "fit", audit)
        fit_dataset = full.ft12.npt1.ensemble_train.common.repro.WindowDataset(
            fit_trajectories, fit_windows)
        source_fold = source_folds[held_aoa]
        source_seal = source_fold["arms"]["source_u800"]["seal"]
        source_endpoint = source_oof.load_uq_endpoint(
            Path(source_seal["path"]), source_seal["file_sha256"])
        source_uq = source_endpoint["uq_state_dict"]
        candidate_seals = {}
        for arm in ARMS:
            joint = arm == "budget_plus_spatial"
            model = upgrade_model(
                point_payload, normalizer, source_uq, device,
                spatial_allocator=joint)
            training = spatial.train_output_space_terminal(
                model, fit_dataset, list(range(len(fit_windows))), normalizer,
                device, deadline,
                uq.SEED + full.dr1.AOA_VALUES.index(held_aoa)
                + (301 if joint else 251),
                f"held{held_aoa}_{arm}",
                train_allocator=joint, train_budgeter=True)
            candidate_seals[arm] = save_endpoint(
                fold_dir / f"{arm}_u800.pth", model,
                held_aoa=held_aoa, arm=arm, training=training)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        del fit_dataset, fit_trajectories
        audit.seal()
        held_trajectories, held_metadata, held_windows = full.dr1.load_prefix_rows(
            args.real_dir, held_rows, "held", audit)
        held_dataset = full.ft12.npt1.ensemble_train.common.repro.WindowDataset(
            held_trajectories, held_windows)
        arms = {}
        for arm in ("source_u800", "native_continue_u1600", *ARMS):
            if arm == "source_u800":
                model = source_oof.clone_point_model(
                    point_payload, normalizer, device)
                current = model.state_dict()
                current.update(source_uq)
                model.load_state_dict(current, strict=True)
                seal = source_seal
            elif arm == "native_continue_u1600":
                seal = source_fold["arms"][arm]["seal"]
                endpoint = spatial.load_endpoint(
                    Path(seal["path"]), seal["file_sha256"])
                model = source_oof.clone_point_model(
                    point_payload, normalizer, device)
                current = model.state_dict()
                current.update(endpoint["uq_state_dict"])
                model.load_state_dict(current, strict=True)
            else:
                seal = candidate_seals[arm]
                endpoint = load_endpoint(
                    Path(seal["path"]), seal["file_sha256"])
                model = upgrade_model(
                    point_payload, normalizer,
                    endpoint["uq_state_dict"], device,
                    spatial_allocator=arm == "budget_plus_spatial")
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

    selection = select(fold_rows)
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "architecture": {
            "budget_channels": BUDGET_CHANNELS,
            "budget_blocks": BUDGET_BLOCKS,
            "budget_log_limit": BUDGET_LOG_LIMIT,
            "spatial_channels": spatial.CHANNELS,
            "spatial_blocks": spatial.BLOCKS,
            "spatial_log_limit": spatial.LOG_LIMIT,
            "hidden_backbone_features": False,
        },
        "promotion": dict(PROMOTION),
        "point": point_report,
        "spatial_oof": {
            "path": str(args.spatial_oof_summary.resolve()),
            "file_sha256": summary_sha,
            "decision": source_summary["decision"],
        },
        "manifest": manifest,
        "folds": fold_rows,
        "selection": selection,
        "elapsed_seconds": time.perf_counter() - started,
        "decision": (
            "PASS_OUTPUT_SPACE_UQ_BUDGET_OOF"
            if selection["pass"] else "STOP_OUTPUT_SPACE_UQ_BUDGET_OOF"),
        "submission_authorized": False,
    }
    full.dr1.write_json_atomic(args.output_dir / "summary.json", result)
    print(json.dumps({
        "event": "b210_output_space_uq_budget_oof_complete",
        "decision": result["decision"], "selection": selection,
        "output": str(args.output_dir / "summary.json")}), flush=True)
    return result


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    run(args)


if __name__ == "__main__":
    main()
