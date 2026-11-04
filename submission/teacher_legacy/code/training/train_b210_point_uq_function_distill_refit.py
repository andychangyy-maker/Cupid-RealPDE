#!/usr/bin/env python3
"""Refit the OOF-selected point-interface UQ objective on all Real prefixes."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time
from typing import Mapping

import torch

import train_b210_full_prefix_uq_refit as full
import train_b210_point_uq_function_distill_oof as oof
import train_b210_small4k_uq_release as uq
from train_distilled_flow_corrector import load_forecaster


EXPERIMENT = "b210_point_uq_function_distill_full_prefix_refit"
SCHEMA = "b210_point_uq_function_distill_full_prefix_refit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-summary", type=Path, required=True)
    parser.add_argument("--oof-summary-sha256", required=True)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--point-checkpoint-sha256", required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint-sha256", required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=1800.0)
    return parser.parse_args()


def validate_oof_summary(path: Path, expected_sha: str) -> tuple[dict, float]:
    actual = full.dr1.sha256(path)
    if actual != expected_sha.lower():
        raise RuntimeError("function-distill OOF summary SHA changed")
    row = json.loads(path.read_text(encoding="utf-8"))
    selection = row.get("selection", {})
    if (
        row.get("schema") != oof.SCHEMA
        or row.get("decision") != "PASS_FUNCTION_LEVEL_UQ_DISTILL_OOF"
        or selection.get("pass") is not True
        or selection.get("selected") is None
    ):
        raise RuntimeError("function-distill OOF gate did not pass")
    weight = float(selection["selected_weight"])
    if weight not in oof.WEIGHTS or weight <= 0.0:
        raise RuntimeError("OOF selected a non-preregistered distillation weight")
    return row, weight


def save_endpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    point_payload: Mapping[str, object],
    point_report: Mapping[str, object],
    weight: float,
    training: Mapping[str, object],
    oof_summary: Mapping[str, object],
    oof_summary_sha: str,
) -> dict:
    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    state_sha = full.dr1.model_state_sha256(state)
    point_partition_sha = full.dr1.model_state_sha256(
        {key: value for key, value in state.items() if not key.startswith(uq.UQ_PREFIXES)}
    )
    if point_partition_sha != training["frozen_state_sha256_before"]:
        raise RuntimeError("full-prefix function distill changed point state")
    payload = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "phase": "real",
        "iteration": uq.FOLD_UPDATES,
        "endpoint": "fixed_terminal_full_prefix_u800",
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": dict(point_payload["model_config"]),
        "normalizer": dict(point_payload["normalizer"]),
        "normalizer_sha256": point_payload["normalizer_sha256"],
        "point_checkpoint": dict(point_report),
        "point_frozen_state_sha256": point_partition_sha,
        "extent_distill_weight": weight,
        "training": dict(training),
        "oof_selection": {
            "summary_sha256": oof_summary_sha,
            "selected": oof_summary["selection"]["selected"],
            "selected_weight": weight,
            "selected_comparison": oof_summary["selection"][
                "selected_comparison"
            ],
        },
        "checkpoint_selection": False,
        "tail_arrays_read": 0,
        "point_state_bitwise": True,
        "submission_authorized": False,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": full.dr1.sha256(path),
        "model_state_sha256": state_sha,
        "point_frozen_state_sha256": point_partition_sha,
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.wall_budget_seconds <= 0.0 or not math.isfinite(args.wall_budget_seconds):
        raise ValueError("wall budget must be finite and positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-prefix function distill requires CUDA")
    oof_summary, weight = validate_oof_summary(
        args.oof_summary, args.oof_summary_sha256
    )
    point_sha = full.dr1.sha256(args.point_checkpoint)
    teacher_sha = full.dr1.sha256(args.teacher_checkpoint)
    if point_sha != args.point_checkpoint_sha256.lower():
        raise RuntimeError("point checkpoint SHA changed")
    if teacher_sha != args.teacher_checkpoint_sha256.lower():
        raise RuntimeError("teacher checkpoint SHA changed")
    point_payload, model, normalizer, point_report = full.load_point(
        args.point_checkpoint, point_sha, device
    )
    teacher_payload = torch.load(
        args.teacher_checkpoint, map_location="cpu", weights_only=False
    )
    if teacher_payload.get("normalizer_sha256") != point_report["normalizer"]["sha256"]:
        raise RuntimeError("teacher and point normalizers differ")
    teacher, teacher_report = load_forecaster(
        args.teacher_checkpoint, normalizer, device
    )
    teacher_report["file_sha256"] = teacher_sha
    rows, manifest = full.dr1.load_manifest(
        args.clean_manifest, full.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256
    )
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    audit = full.PrefixOnlyAudit()
    trajectories, _metadata, windows = full.dr1.load_prefix_rows(
        args.real_dir, rows, "fit", audit
    )
    access = audit.report(rows)
    if (
        not access["allowlist_exact"]
        or access["tail_arrays_read"] != 0
        or access["excluded_opened"]
        or len(windows) != full.EXPECTED_WINDOWS
    ):
        raise RuntimeError("full-prefix function-distill data contract failed")
    dataset = full.ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows
    )
    training = uq.train_uq_terminal(
        model,
        dataset,
        list(range(len(windows))),
        normalizer,
        device,
        deadline,
        uq.SEED,
        f"full_prefix_distill_{weight:.3f}",
        objective_args=oof.objective(weight),
        extent_teachers=(teacher, teacher),
        extent_teacher_weight_b=0.0,
    )
    endpoint = save_endpoint(
        args.output_dir / "full_prefix_function_distill_u800.pth",
        model,
        point_payload=point_payload,
        point_report=point_report,
        weight=weight,
        training=training,
        oof_summary=oof_summary,
        oof_summary_sha=args.oof_summary_sha256.lower(),
    )
    del model, teacher, dataset, trajectories
    gc.collect()
    torch.cuda.empty_cache()
    result = {
        "schema": f"{SCHEMA}_summary",
        "experiment": EXPERIMENT,
        "decision": "FULL_PREFIX_FUNCTION_DISTILL_COMPLETE",
        "selected_weight": weight,
        "point": point_report,
        "teacher": teacher_report,
        "oof_summary": {
            "path": str(args.oof_summary.resolve()),
            "sha256": args.oof_summary_sha256.lower(),
        },
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
    print(
        json.dumps(
            {
                "event": "full_prefix_function_distill_complete",
                "weight": weight,
                "endpoint": endpoint,
                "output": str(args.output_dir / "summary.json"),
            }
        ),
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    run(args)


if __name__ == "__main__":
    main()
