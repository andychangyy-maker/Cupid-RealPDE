#!/usr/bin/env python3
"""Nested point-OOF audit of partial Sim parameter anchoring after Real fit."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
        REPO_ROOT / "code", REPO_ROOT / "code" / "training",
        REPO_ROOT / "code" / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_sim_real_disagreement_transfer as transfer  # noqa: E402
import build_b210_xcb_point_oof_residual_cache as oof_builder  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_ensemble_uq_oof as uq  # noqa: E402
import reproduce_finetune as repro  # noqa: E402


SCHEMA = "b210_xcb_sim_anchor_point_oof_audit_v1"
REPRODUCTION_ABS_TOLERANCE = 5.0e-5
POINT_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_rel_l2_score": 0.10,
    "mvpe_rel_l2_score": 0.09,
}
XCB_PREFIXES = (
    "regime_film.", "decoder_regime_film.", "cross_conditioner.",
    "member_point_heads.", "member_feature_adapters.",
    "mean_profile_refiner.", "energy_controller.",
)


def parse_alphas(raw: str) -> list[float]:
    values = sorted(set(float(value.strip()) for value in raw.split(",") if value.strip()))
    if (not values or any(not math.isfinite(value) or not 0.0 < value < 1.0
                          for value in values)):
        raise ValueError("anchor alphas must be finite and inside (0,1)")
    return values


def parameter_in_scope(name: str, scope: str) -> bool:
    if name.startswith(uq.UQ_PREFIXES):
        return False
    if scope == "all_point":
        return True
    if scope == "xcb":
        return name.startswith(XCB_PREFIXES)
    raise ValueError(f"unknown Sim anchor scope {scope}")


@torch.no_grad()
def apply_anchor(
        model, real_state: Mapping[str, torch.Tensor],
        sim_state: Mapping[str, torch.Tensor], scope: str,
        alpha: float, device: torch.device) -> None:
    if not 0.0 <= alpha < 1.0:
        raise ValueError("anchor alpha must be in [0,1)")
    for name, parameter in model.named_parameters():
        real = real_state[name].to(device=device, dtype=parameter.dtype)
        if alpha and parameter_in_scope(name, scope):
            sim = sim_state[name].to(device=device, dtype=parameter.dtype)
            parameter.copy_(torch.lerp(real, sim, alpha))
        else:
            parameter.copy_(real)


def point_contribution(metrics: Mapping[str, float]) -> float:
    return sum(POINT_WEIGHTS[key] * float(metrics[key]) for key in POINT_WEIGHTS)


def build_held_dataset(
        real_dir: Path, rows: Sequence[Mapping[str, object]]):
    trajectories = []
    windows = []
    for row in rows:
        trajectory = len(trajectories)
        trajectories.append(oof_builder.read_trajectory(real_dir, row))
        windows.extend(
            repro.Window(trajectory, int(start))
            for start in row["val_starts"])
    return repro.WindowDataset(trajectories, windows)


def embedded_normalizer(payload: Mapping[str, object], device: torch.device):
    raw = payload.get("normalizer")
    if not isinstance(raw, Mapping):
        raise TypeError("Real fold checkpoint has no embedded normalizer")
    return smu1.FoldNormalizer((
        torch.as_tensor(raw["mean_input"]),
        torch.as_tensor(raw["mean_target"]),
        torch.as_tensor(raw["std_input"]),
        torch.as_tensor(raw["std_target"]),
    ), device)


def metric_delta(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> dict:
    delta = {
        key: float(candidate[key]) - float(baseline[key])
        for key in POINT_WEIGHTS}
    delta["official_point_contribution"] = sum(
        POINT_WEIGHTS[key] * delta[key] for key in POINT_WEIGHTS)
    return delta


def evaluate_fold(
        args: argparse.Namespace, fold: int, rows: Sequence[Mapping[str, object]],
        sim_payload: Mapping[str, object], candidates: Sequence[tuple[str, float]],
        ) -> dict:
    result_path = Path(args.fold_result_template.format(fold=fold))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if int(result["held_fold"]) != fold or not bool(result["gate"]["pass"]):
        raise ValueError(f"point-OOF fold {fold} result is not admissible")
    checkpoint = Path(result["fit"]["seal"]["path"])
    if oof_builder.sha256(checkpoint) != result["fit"]["seal"]["file_sha256"]:
        raise RuntimeError(f"point-OOF fold {fold} checkpoint SHA changed")
    real_payload = torch.load(
        str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
    if real_payload["model_config"] != sim_payload["model_config"]:
        raise ValueError("Sim and Real point architecture differ")
    device = torch.device(args.device)
    normalizer = embedded_normalizer(real_payload, device)
    model = transfer.load_model(real_payload, device)
    dataset = build_held_dataset(args.real_dir, rows)
    indices = list(range(len(dataset)))
    rows_out = []
    real_state = real_payload["model_state_dict"]
    sim_state = sim_payload["model_state_dict"]
    for scope, alpha in candidates:
        apply_anchor(model, real_state, sim_state, scope, alpha, device)
        metrics, state, target_audit = smu1.evaluate_model(
            model, dataset, indices, normalizer, device,
            time.perf_counter() + args.fold_wall_seconds)
        rows_out.append({
            "scope": scope, "alpha": alpha,
            "metrics": metrics, "metric_state": state,
            "point_contribution": point_contribution(metrics),
            "target_audit": target_audit,
        })
    baseline = rows_out[0]
    expected = result["held"]["fold_metrics"]
    reproduction = {
        key: baseline["metrics"][key] - float(expected[key])
        for key in POINT_WEIGHTS}
    if (max(abs(value) for value in reproduction.values())
            > REPRODUCTION_ABS_TOLERANCE):
        raise RuntimeError("alpha-zero fold metric did not reproduce")
    for row in rows_out:
        if row["target_audit"] != baseline["target_audit"]:
            raise RuntimeError("anchor candidates saw different held targets")
        row["delta_vs_real"] = metric_delta(
            row["metrics"], baseline["metrics"])
    del model, dataset, real_payload
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "fold": fold, "windows": len(indices),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": oof_builder.sha256(checkpoint),
        "baseline_reproduction_delta": reproduction,
        "candidates": rows_out,
    }


def candidate_key(row: Mapping[str, object]) -> tuple[str, float]:
    return str(row["scope"]), float(row["alpha"])


def merge_candidate(folds: Sequence[Mapping[str, object]], key: tuple[str, float]) -> dict:
    states = []
    for fold in folds:
        match = [row for row in fold["candidates"] if candidate_key(row) == key]
        if len(match) != 1:
            raise RuntimeError("anchor candidate missing from a fold")
        states.append(match[0]["metric_state"])
    return smu1.finalize_metric_state(smu1.merge_metric_states(states))


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    alphas = parse_alphas(args.alphas)
    candidates = [("all_point", 0.0)] + [
        (scope, alpha) for scope in ("all_point", "xcb") for alpha in alphas]
    manifest = oof_builder.validate_manifest(json.loads(
        args.evaluation_manifest.read_text(encoding="utf-8")))
    assignments = oof_builder.condition_fold_assignments(manifest, args.folds)
    sim_payload = torch.load(
        str(args.sim_checkpoint), map_location="cpu", weights_only=False,
        mmap=True)
    fold_rows = []
    for fold in range(args.folds):
        held = [row for row in manifest if assignments[str(row["file"])] == fold]
        fold_rows.append(evaluate_fold(
            args, fold, held, sim_payload, candidates))
    identity = ("all_point", 0.0)
    nested = []
    for held in range(args.folds):
        train_folds = [row for row in fold_rows if int(row["fold"]) != held]
        held_fold = fold_rows[held]
        selected = max(
            candidates,
            key=lambda key: point_contribution(merge_candidate(train_folds, key)))
        baseline = merge_candidate([held_fold], identity)
        candidate = merge_candidate([held_fold], selected)
        nested.append({
            "held_fold": held,
            "selected_scope": selected[0],
            "selected_alpha": selected[1],
            "baseline": baseline,
            "candidate": candidate,
            "delta": metric_delta(candidate, baseline),
        })
    baseline_oof = merge_candidate([
        {**fold, "candidates": [next(
            row for row in fold["candidates"]
            if candidate_key(row) == identity)]}
        for fold in fold_rows], identity)
    nested_state = []
    for held, row in enumerate(nested):
        key = (row["selected_scope"], row["selected_alpha"])
        match = next(
            item for item in fold_rows[held]["candidates"]
            if candidate_key(item) == key)
        nested_state.append(match["metric_state"])
    candidate_oof = smu1.finalize_metric_state(
        smu1.merge_metric_states(nested_state))
    delta = metric_delta(candidate_oof, baseline_oof)
    positive = sum(
        row["delta"]["official_point_contribution"] > 0.0
        for row in nested)
    conditions = {
        "nested_point_contribution_improves": (
            delta["official_point_contribution"] >= args.min_point_delta),
        "fold_gain_is_consistent": positive >= args.min_positive_folds,
    }
    report = {
        "schema": SCHEMA,
        "sim_checkpoint": str(args.sim_checkpoint.resolve()),
        "sim_checkpoint_sha256": oof_builder.sha256(args.sim_checkpoint),
        "evaluation_manifest_sha256": oof_builder.sha256(args.evaluation_manifest),
        "candidates": [{"scope": scope, "alpha": alpha} for scope, alpha in candidates],
        "folds": fold_rows,
        "nested": nested,
        "nested_oof_baseline": baseline_oof,
        "nested_oof_candidate": candidate_oof,
        "nested_oof_delta": delta,
        "positive_folds": positive,
        "gate": {"conditions": conditions, "pass": all(conditions.values())},
        "decision": (
            "GO_REAL_SIM_FUNCTIONAL_ANCHOR_TRAINING"
            if all(conditions.values()) else "STOP_SIMPLE_SIM_PARAMETER_ANCHOR"),
        "submission_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sim-checkpoint", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument(
        "--fold-result-template", required=True,
        help="Path template containing {fold}, ending in each result.json.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alphas", default="0.01,0.025,0.05,0.10,0.20")
    parser.add_argument("--min-point-delta", type=float, default=0.005)
    parser.add_argument("--min-positive-folds", type=int, default=4)
    parser.add_argument("--fold-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({
        "event": "b210_xcb_sim_anchor_point_oof_complete",
        "delta": report["nested_oof_delta"],
        "positive_folds": report["positive_folds"],
        "decision": report["decision"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
