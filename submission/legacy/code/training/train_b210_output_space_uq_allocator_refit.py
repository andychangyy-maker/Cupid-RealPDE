#!/usr/bin/env python3
"""Refit an OOF-approved output-space UQ allocator on all Real prefixes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import time
from typing import Mapping

import torch

import train_b210_full_prefix_uq_refit as full
import train_b210_output_space_uq_allocator_oof as oof


EXPERIMENT = "b210_output_space_uq_allocator_full_prefix_refit"
SCHEMA = "b210_output_space_uq_allocator_full_prefix_refit_v1"
SOURCE_SCHEMA = "b210_full_prefix_native_uq_refit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-summary", type=Path, required=True)
    parser.add_argument("--oof-summary-sha256", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=1800.0)
    return parser.parse_args()


def validate_oof_summary(path: Path, expected_sha: str) -> dict:
    actual = full.dr1.sha256(path)
    if actual != expected_sha.lower():
        raise RuntimeError("output-space allocator OOF summary SHA changed")
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row.get("schema") != oof.SCHEMA
            or row.get("decision") != "PASS_OUTPUT_SPACE_UQ_ALLOCATOR_OOF"
            or row.get("gate", {}).get("pass") is not True):
        raise RuntimeError("output-space allocator OOF gate did not pass")
    architecture = row.get("architecture", {})
    expected = {
        "channels": oof.CHANNELS, "blocks": oof.BLOCKS,
        "log_limit": oof.LOG_LIMIT}
    if any(architecture.get(key) != value for key, value in expected.items()):
        raise RuntimeError("OOF-approved allocator architecture changed")
    return row


def validate_source(path: Path, expected_sha: str) -> dict:
    actual = full.dr1.sha256(path)
    if actual != expected_sha.lower():
        raise RuntimeError("full-prefix source checkpoint SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != SOURCE_SCHEMA
            or payload.get("endpoint") != "fixed_terminal_full_prefix_u800"
            or int(payload.get("iteration", -1)) != 800
            or not isinstance(state, Mapping)
            or full.dr1.model_state_sha256(state)
            != payload.get("model_state_sha256")):
        raise RuntimeError("full-prefix source checkpoint contract changed")
    if any(key.startswith("output_space_uq_allocator.") for key in state):
        raise RuntimeError("source checkpoint already contains allocator state")
    return payload


def normalizer_from_source(payload: Mapping[str, object], device: torch.device):
    return full.normalizer_from_point(payload, device)


def save_endpoint(
        path: Path, model: torch.nn.Module, *, source: Mapping[str, object],
        source_sha: str, normalizer_report: Mapping[str, object],
        training: Mapping[str, object], oof_summary: Mapping[str, object],
        oof_summary_sha: str) -> dict:
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()}
    state_sha = full.dr1.model_state_sha256(state)
    frozen_sha = full.dr1.model_state_sha256(
        oof.frozen_without_allocator_state(model))
    if frozen_sha != training["frozen_state_sha256_before"]:
        raise RuntimeError("allocator refit changed source point/base-UQ state")
    payload = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real",
        "iteration": oof.uq.FOLD_UPDATES,
        "endpoint": "fixed_terminal_full_prefix_allocator_u800",
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": asdict(model.config) | {"architecture": "ensemble_unet"},
        "normalizer": dict(normalizer_report["values"]),
        "normalizer_sha256": normalizer_report["sha256"],
        "source_checkpoint": {
            "file_sha256": source_sha,
            "schema": source["schema"],
            "model_state_sha256": source["model_state_sha256"],
        },
        "source_frozen_state_sha256": frozen_sha,
        "training": dict(training),
        "oof_selection": {
            "summary_sha256": oof_summary_sha,
            "gate": oof_summary["gate"],
            "architecture": oof_summary["architecture"],
        },
        "checkpoint_selection": False,
        "tail_arrays_read": 0,
        "point_and_base_uq_bitwise": True,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": full.dr1.sha256(path),
        "model_state_sha256": state_sha,
        "source_frozen_state_sha256": frozen_sha,
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.wall_budget_seconds <= 0.0 or not math.isfinite(
            args.wall_budget_seconds):
        raise ValueError("wall budget must be finite and positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-prefix output-space allocator requires CUDA")
    oof_summary = validate_oof_summary(
        args.oof_summary, args.oof_summary_sha256)
    source_sha = full.dr1.sha256(args.source_checkpoint)
    source = validate_source(args.source_checkpoint, source_sha)
    if source_sha != args.source_checkpoint_sha256.lower():
        raise RuntimeError("source checkpoint argument SHA changed")
    normalizer, normalizer_report = normalizer_from_source(source, device)
    model = oof.upgrade_output_allocator(
        source, normalizer, {}, device)
    rows, manifest = full.dr1.load_manifest(
        args.clean_manifest, full.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    audit = full.PrefixOnlyAudit()
    trajectories, _metadata, windows = full.dr1.load_prefix_rows(
        args.real_dir, rows, "fit", audit)
    access = audit.report(rows)
    if (not access["allowlist_exact"] or access["tail_arrays_read"] != 0
            or access["excluded_opened"]
            or len(windows) != full.EXPECTED_WINDOWS):
        raise RuntimeError("full-prefix allocator data contract failed")
    dataset = full.ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    training = oof.train_allocator_terminal(
        model, dataset, list(range(len(windows))), normalizer, device,
        deadline, oof.uq.SEED, "full_prefix_output_allocator")
    endpoint = save_endpoint(
        args.output_dir / "full_prefix_output_allocator_u800.pth", model,
        source=source, source_sha=source_sha,
        normalizer_report=normalizer_report, training=training,
        oof_summary=oof_summary,
        oof_summary_sha=args.oof_summary_sha256.lower())
    del model, dataset, trajectories
    gc.collect()
    torch.cuda.empty_cache()
    result = {
        "schema": f"{SCHEMA}_summary",
        "experiment": EXPERIMENT,
        "decision": "FULL_PREFIX_OUTPUT_ALLOCATOR_COMPLETE",
        "source_checkpoint": {
            "path": str(args.source_checkpoint.resolve()),
            "sha256": source_sha},
        "oof_summary": {
            "path": str(args.oof_summary.resolve()),
            "sha256": args.oof_summary_sha256.lower()},
        "manifest": manifest,
        "access": access,
        "fit_windows": len(windows),
        "training": training,
        "endpoint": endpoint,
        "elapsed_seconds": time.perf_counter() - started,
        "tail_arrays_read": 0,
        "submission_authorized": False,
    }
    full.dr1.write_json_atomic(args.output_dir / "summary.json", result)
    print(json.dumps({
        "event": "full_prefix_output_allocator_complete",
        "endpoint": endpoint,
        "output": str(args.output_dir / "summary.json")}), flush=True)
    return result


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    run(args)


if __name__ == "__main__":
    main()
