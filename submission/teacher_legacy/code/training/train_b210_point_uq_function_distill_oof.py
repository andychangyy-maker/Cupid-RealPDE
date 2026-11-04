#!/usr/bin/env python3
"""Transfer a strong B210 interval function onto a stronger frozen point model."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

import train_b210_full_prefix_uq_refit as full
import train_b210_small4k_uq_release as uq
from train_distilled_flow_corrector import load_forecaster


EXPERIMENT = "b210_point_uq_function_distill_oof"
SCHEMA = "b210_point_uq_function_distill_oof_v1"
ENDPOINT_SCHEMA = "b210_point_uq_function_distill_endpoint_v1"
WEIGHTS = (0.0, 0.5, 1.5, 4.0)
PROMOTION = {
    "pooled_sps_delta_min": 0.05,
    "positive_folds_min": 4,
    "worst_fold_min": -0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--point-checkpoint-sha256", required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint-sha256", required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=7200.0)
    return parser.parse_args()


class FoldAccessAudit:
    def __init__(self, held_aoa: int) -> None:
        self.held_aoa = held_aoa
        self.sealed = False
        self.events: list[dict] = []
        self.opened: set[str] = set()

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        name = str(row["file"])
        aoa = int(row["aoa"])
        if name in self.opened or name == full.dr1.EXCLUDED_BASENAME:
            raise RuntimeError("invalid or repeated condition-split trajectory")
        if phase == "fit" and aoa == self.held_aoa:
            raise RuntimeError("held AoA entered UQ fit data")
        if phase == "held" and (aoa != self.held_aoa or not self.sealed):
            raise RuntimeError("held AoA opened before all UQ endpoints were sealed")
        if phase not in ("fit", "held"):
            raise RuntimeError("unexpected condition-split data phase")

    def record_open(self, row: Mapping[str, object], phase: str) -> None:
        self.before_open(row, phase)
        name = str(row["file"])
        self.opened.add(name)
        self.events.append({"file": name, "aoa": int(row["aoa"]), "phase": phase})

    def seal(self) -> None:
        self.sealed = True


def objective(weight: float) -> argparse.Namespace:
    args = uq.sharp2_args()
    args.extent_distill_weight = float(weight)
    return args


def arm_label(weight: float) -> str:
    return "control" if weight == 0.0 else f"distill_{weight:.3f}"


def clone_point_model(
    payload: Mapping[str, object], normalizer, device: torch.device
) -> torch.nn.Module:
    config = full.ft12.validate_b210_model_config(
        payload["model_config"], full.sim_tke.b210_config("candidate")
    )
    model = full.EnsembleUNet2d(config).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    full.ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def save_uq_endpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    held_aoa: int,
    weight: float,
    training: Mapping[str, object],
) -> dict:
    state = {
        key: value.detach().cpu().clone()
        for key, value in uq.partition_state(model, uq=True).items()
    }
    state_sha = full.dr1.model_state_sha256(state)
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": held_aoa,
        "extent_distill_weight": weight,
        "iteration": uq.FOLD_UPDATES,
        "endpoint": "fixed_terminal_u800",
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
        "weight": weight,
    }


def load_uq_endpoint(path: Path, expected_sha: str) -> dict:
    if full.dr1.sha256(path) != expected_sha:
        raise RuntimeError("UQ function-distill endpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("uq_state_dict")
    if (
        payload.get("schema") != ENDPOINT_SCHEMA
        or not isinstance(state, Mapping)
        or full.dr1.model_state_sha256(state) != payload.get("uq_state_sha256")
    ):
        raise RuntimeError("UQ function-distill endpoint seal is invalid")
    return payload


def pooled(rows: Sequence[Mapping[str, object]], arm: str) -> tuple[dict, dict]:
    state = full.smu1.merge_metric_states(
        [row["arms"][arm]["metric_state"] for row in rows]
    )
    return state, full.smu1.finalize_metric_state(state)


def comparison(rows: Sequence[Mapping[str, object]], arm: str) -> dict:
    _control_state, control = pooled(rows, "control")
    _candidate_state, candidate = pooled(rows, arm)
    deltas = [
        float(row["arms"][arm]["metrics"]["native_sps_score"])
        - float(row["arms"]["control"]["metrics"]["native_sps_score"])
        for row in rows
    ]
    delta = float(candidate["native_sps_score"] - control["native_sps_score"])
    conditions = {
        "pooled_sps_delta": delta >= PROMOTION["pooled_sps_delta_min"],
        "positive_folds": sum(value > 0.0 for value in deltas)
        >= PROMOTION["positive_folds_min"],
        "worst_fold": min(deltas) >= PROMOTION["worst_fold_min"],
    }
    return {
        "control": control,
        "candidate": candidate,
        "native_sps_delta": delta,
        "non_time_delta": 0.25 * delta,
        "fold_deltas": deltas,
        "positive_folds": sum(value > 0.0 for value in deltas),
        "worst_fold": min(deltas),
        "conditions": conditions,
        "pass": all(conditions.values()),
    }


def select_weight(rows: Sequence[Mapping[str, object]]) -> dict:
    candidates = {
        arm_label(weight): comparison(rows, arm_label(weight))
        for weight in WEIGHTS
        if weight > 0.0
    }
    passing = [
        (row["native_sps_delta"], arm, row)
        for arm, row in candidates.items()
        if row["pass"]
    ]
    if not passing:
        return {"selected": None, "candidates": candidates, "pass": False}
    _delta, selected, selected_row = max(passing)
    return {
        "selected": selected,
        "selected_weight": float(selected.removeprefix("distill_")),
        "selected_comparison": selected_row,
        "candidates": candidates,
        "selection_rule": "largest pooled SPS delta among preregistered passing weights",
        "pass": True,
    }


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("UQ function distillation requires CUDA")
    if args.wall_budget_seconds <= 0.0 or not math.isfinite(args.wall_budget_seconds):
        raise ValueError("wall budget must be finite and positive")
    point_sha = full.dr1.sha256(args.point_checkpoint)
    teacher_sha = full.dr1.sha256(args.teacher_checkpoint)
    if point_sha != args.point_checkpoint_sha256.lower():
        raise RuntimeError("point checkpoint SHA changed")
    if teacher_sha != args.teacher_checkpoint_sha256.lower():
        raise RuntimeError("teacher checkpoint SHA changed")
    rows, manifest = full.dr1.load_manifest(
        args.clean_manifest, full.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256
    )
    point_payload, point_model, normalizer, point_report = full.load_point(
        args.point_checkpoint, point_sha, device
    )
    del point_model
    gc.collect()
    teacher_payload = torch.load(
        args.teacher_checkpoint, map_location="cpu", weights_only=False
    )
    if teacher_payload.get("normalizer_sha256") != point_report["normalizer"]["sha256"]:
        raise RuntimeError("teacher and point normalizers differ")
    teacher, teacher_report = load_forecaster(
        args.teacher_checkpoint, normalizer, device
    )
    teacher_report["file_sha256"] = teacher_sha

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    fold_rows = []
    for held_aoa in full.dr1.AOA_VALUES:
        fold_dir = args.output_dir / f"held{held_aoa}"
        fold_dir.mkdir()
        fit_rows = [row for row in rows if int(row["aoa"]) != held_aoa]
        held_rows = [row for row in rows if int(row["aoa"]) == held_aoa]
        audit = FoldAccessAudit(held_aoa)
        fit_trajectories, fit_metadata, fit_windows = full.dr1.load_prefix_rows(
            args.real_dir, fit_rows, "fit", audit
        )
        fit_dataset = full.ft12.npt1.ensemble_train.common.repro.WindowDataset(
            fit_trajectories, fit_windows
        )
        seals = {}
        for weight in WEIGHTS:
            label = arm_label(weight)
            model = clone_point_model(point_payload, normalizer, device)
            training = uq.train_uq_terminal(
                model,
                fit_dataset,
                list(range(len(fit_windows))),
                normalizer,
                device,
                deadline,
                uq.SEED + full.dr1.AOA_VALUES.index(held_aoa) + 1,
                f"held{held_aoa}_{label}",
                objective_args=objective(weight),
                extent_teachers=(teacher, teacher) if weight > 0.0 else None,
                extent_teacher_weight_b=0.0,
            )
            seals[label] = save_uq_endpoint(
                fold_dir / f"{label}_u800.pth",
                model,
                held_aoa=held_aoa,
                weight=weight,
                training=training,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        del fit_dataset, fit_trajectories
        audit.seal()
        held_trajectories, held_metadata, held_windows = full.dr1.load_prefix_rows(
            args.real_dir, held_rows, "held", audit
        )
        held_dataset = full.ft12.npt1.ensemble_train.common.repro.WindowDataset(
            held_trajectories, held_windows
        )
        arms = {}
        for weight in WEIGHTS:
            label = arm_label(weight)
            endpoint = load_uq_endpoint(
                Path(seals[label]["path"]), seals[label]["file_sha256"]
            )
            model = clone_point_model(point_payload, normalizer, device)
            current = model.state_dict()
            current.update(endpoint["uq_state_dict"])
            model.load_state_dict(current, strict=True)
            metric_state, _by_aoa = uq.evaluate_grouped(
                model,
                held_dataset,
                list(range(len(held_windows))),
                held_metadata,
                normalizer,
                device,
                deadline,
            )
            arms[label] = {
                "weight": weight,
                "metrics": full.smu1.finalize_metric_state(metric_state),
                "metric_state": metric_state,
                "seal": seals[label],
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
            "access": {"sealed_before_held": audit.sealed, "events": audit.events},
        }
        full.dr1.write_json_atomic(fold_dir / "result.json", fold_row)
        fold_rows.append(fold_row)
        del held_dataset, held_trajectories
        gc.collect()

    selection = select_weight(fold_rows)
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "weights": list(WEIGHTS),
        "promotion": dict(PROMOTION),
        "point": point_report,
        "teacher": teacher_report,
        "manifest": manifest,
        "folds": fold_rows,
        "selection": selection,
        "elapsed_seconds": time.perf_counter() - started,
        "decision": (
            "PASS_FUNCTION_LEVEL_UQ_DISTILL_OOF"
            if selection["pass"]
            else "STOP_FUNCTION_LEVEL_UQ_DISTILL_OOF"
        ),
        "submission_authorized": False,
    }
    full.dr1.write_json_atomic(args.output_dir / "summary.json", result)
    print(
        json.dumps(
            {
                "event": "b210_point_uq_function_distill_oof_complete",
                "decision": result["decision"],
                "selection": selection,
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
