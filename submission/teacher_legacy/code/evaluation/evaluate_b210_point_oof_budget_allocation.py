#!/usr/bin/env python3
"""Nested-CV audit of budget-preserving member-spread UQ allocation.

The base intervals come from the temporal-tail OOF UQ checkpoints.  Accuracy
factors and residual labels come from condition-excluded point checkpoints.
Only two global allocator hyperparameters are selected on the other four
condition folds, so the held-fold delta is a genuine method-selection audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
        REPO_ROOT / "code", REPO_ROOT / "code" / "training",
        REPO_ROOT / "code" / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_b210_xcb_point_oof_residual_cache as oof_builder  # noqa: E402
import train_ensemble_uq_oof as uq  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402


SCHEMA = "b210_xcb_point_oof_budget_allocation_audit_v1"
FEATURE_SCHEMA = "b210_xcb_oof_uq_allocation_features_v1"


def spatial_standardize(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 5:
        raise ValueError("allocation signal must have B,T,H,W,C shape")
    centered = value - value.mean(dim=(2, 3), keepdim=True)
    variance = centered.square().mean(dim=(2, 3), keepdim=True)
    return centered / torch.sqrt(variance + 1.0e-4)


def budgeted_scale(
        lower: torch.Tensor, upper: torch.Tensor, signal: torch.Tensor,
        strength: float, sharpness: float) -> torch.Tensor:
    if lower.shape != upper.shape or lower.shape != signal.shape:
        raise ValueError("base intervals and signal must have equal shapes")
    if not math.isfinite(strength) or not math.isfinite(sharpness):
        raise ValueError("allocator parameters must be finite")
    if sharpness <= 0.0:
        raise ValueError("allocator sharpness must be positive")
    localized = torch.tanh(sharpness * spatial_standardize(signal))
    raw = torch.exp(strength * localized)
    width = (lower + upper).clamp_min(0.0)
    budget = width.mean(dim=(2, 3), keepdim=True)
    proposed = (width * raw).mean(dim=(2, 3), keepdim=True)
    correction = torch.where(
        budget > 0.0,
        budget / proposed.clamp_min(torch.finfo(proposed.dtype).tiny),
        torch.ones_like(budget))
    return raw * correction


def parse_grid(raw: str, *, positive: bool = False) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("allocator grid must contain finite values")
    if positive and any(value <= 0.0 for value in values):
        raise ValueError("sharpness grid must be positive")
    return sorted(set(values))


def load_fold_uq_model(
        point_payload: dict, fold_checkpoint: Path, training: Mapping[str, object],
        normalizer, device: torch.device):
    model = uq.load_model(
        point_payload, normalizer, device,
        bool(training["train_projections"]),
        bool(training["add_flow_context"]),
        bool(training["residual_quantile_intervals"]),
        bool(training["utility_refinement"]),
        float(training["utility_refinement_log_limit"]),
        int(training["utility_refinement_kernel"]),
        int(training["utility_refinement_blocks"]),
        bool(training["temporal_uq_refinement"]),
        int(training["temporal_uq_channels"]),
        int(training["temporal_uq_blocks"]),
        int(training["temporal_uq_kernel"]),
        float(training["temporal_uq_logit_limit"]),
        bool(training["coverage_calibration"]),
        int(training["coverage_calibration_kernel"]),
        int(training["coverage_calibration_blocks"]),
        float(training["coverage_calibration_logit_limit"]),
        str(training["coverage_calibration_mode"]),
        float(training["coverage_calibration_temperature_limit"]),
    )
    fold_payload = torch.load(
        fold_checkpoint, map_location="cpu", weights_only=False)
    uq.load_uq_state(model, fold_payload["uq_state_dict"])
    return model.eval(), fold_payload


def validate_identity(
        oof_cache: Mapping[str, object], manifest: Sequence[Mapping[str, object]],
        folds: int) -> tuple[list[dict], dict[tuple[str, int], int]]:
    assignments = oof_builder.condition_fold_assignments(manifest, folds)
    expected = [
        {
            "file": str(row["file"]), "start": int(start),
            "re": int(row["re"]), "aoa": int(row["aoa"]),
            "fold": assignments[str(row["file"])],
        }
        for row in manifest for start in row["val_starts"]]
    observed = oof_cache.get("sample_identities")
    if observed != expected:
        raise ValueError("OOF residual cache and evaluation manifest differ")
    if oof_cache.get("sample_identity_sha256") != oof_builder.canonical_sha256(expected):
        raise ValueError("OOF residual sample identity SHA changed")
    return expected, {
        (str(item["file"]), int(item["start"])): index
        for index, item in enumerate(expected)}


def collect_features(args: argparse.Namespace, oof_cache: Mapping[str, object]) -> dict:
    manifest = oof_builder.validate_manifest(json.loads(
        args.evaluation_manifest.read_text(encoding="utf-8")))
    identities, positions = validate_identity(oof_cache, manifest, args.folds)
    shape = (len(identities), 20, 32, 64, 2)
    lower = torch.empty(shape, dtype=torch.float16)
    upper = torch.empty(shape, dtype=torch.float16)
    signal = torch.empty(shape, dtype=torch.float16)
    full_residual = torch.empty(shape, dtype=torch.float16)
    scored = torch.empty(shape, dtype=torch.bool)
    full_factor = torch.full((len(identities),), float("nan"), dtype=torch.float32)
    written = torch.zeros(len(identities), dtype=torch.bool)
    device = torch.device(args.device)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    point_payload = torch.load(
        str(args.point_checkpoint), map_location="cpu", weights_only=False,
        mmap=True)
    summary = json.loads(args.uq_summary.read_text(encoding="utf-8"))
    training = summary["training_config"]
    target_std = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    fold_reports = []
    for fold in range(args.folds):
        checkpoint = args.uq_fold_dir / f"fold_{fold:02d}_best.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, fold_payload = load_fold_uq_model(
            point_payload, checkpoint, training, normalizer, device)
        if (int(fold_payload["fold"]) != fold
                or bool(fold_payload["held_fold_checkpoint_selection"])):
            raise ValueError("UQ fold checkpoint selection contract changed")
        fold_rows = [
            row for row in manifest
            if int(identities[positions[(str(row["file"]), int(row["val_starts"][0]))]]["fold"]) == fold]
        with torch.inference_mode():
            for row in fold_rows:
                starts = [int(value) for value in row["val_starts"]]
                history, target = oof_builder.uvp_windows(
                    oof_builder.read_trajectory(args.real_dir, row), starts)
                for local, start in enumerate(starts):
                    position = positions[(str(row["file"]), start)]
                    history_sha = hashlib.sha256(
                        history[local].numpy().tobytes(order="C")).hexdigest()
                    target_sha = hashlib.sha256(
                        target[local].numpy().tobytes(order="C")).hexdigest()
                    if (history_sha
                            != oof_cache["ordered_history_sample_sha256"][position]
                            or target_sha
                            != oof_cache["ordered_target_sample_sha256"][position]):
                        raise RuntimeError(
                            "UQ features and OOF residual raw samples differ")
                for offset in range(0, len(starts), args.batch_size):
                    batch_starts = starts[offset:offset + args.batch_size]
                    batch_history = history[offset:offset + args.batch_size].to(device)
                    batch_target = target[offset:offset + args.batch_size].to(device)
                    history_norm, _ = normalizer.preprocess(
                        batch_history, batch_target)
                    values = model.forward_uq_training(history_norm)
                    prediction_norm, members, _quantiles, lower_norm, upper_norm = values[:5]
                    prediction = common.to_physical_target(
                        prediction_norm, normalizer)[..., :2]
                    truth = batch_target[..., :2]
                    physical_std = members[..., :2].std(
                        dim=1, unbiased=False) * target_std
                    physical_range = (
                        members[..., :2].amax(dim=1)
                        - members[..., :2].amin(dim=1)) * target_std
                    reference = physical_std.new_tensor(
                        [0.01075, 0.00550]).view(1, 1, 1, 1, 2)
                    batch_signal = 0.5 * (
                        torch.log1p(physical_std / reference)
                        + torch.log1p(physical_range / (2.0 * reference)))
                    batch_lower = lower_norm * target_std
                    batch_upper = upper_norm * target_std
                    batch_factor = common.sample_sps_factor(prediction, truth)
                    for local, start in enumerate(batch_starts):
                        position = positions[(str(row["file"]), start)]
                        if written[position]:
                            raise RuntimeError("UQ allocation feature written twice")
                        lower[position].copy_(batch_lower[local].half().cpu())
                        upper[position].copy_(batch_upper[local].half().cpu())
                        signal[position].copy_(batch_signal[local].half().cpu())
                        full_residual[position].copy_(
                            (truth[local] - prediction[local]).half().cpu())
                        scored[position].copy_((truth[local] != 0.0).cpu())
                        full_factor[position] = batch_factor[local].float().cpu()
                        written[position] = True
        fold_reports.append({
            "fold": fold,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": oof_builder.sha256(checkpoint),
            "selected_iteration": int(fold_payload["selected_iteration"]),
        })
        del model, fold_payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not written.all() or not torch.isfinite(full_factor).all():
        raise RuntimeError("UQ allocation feature cache is incomplete")
    payload = {
        "schema": FEATURE_SCHEMA,
        "sample_identity_sha256": oof_cache["sample_identity_sha256"],
        "evaluation_manifest_sha256": oof_builder.sha256(args.evaluation_manifest),
        "point_checkpoint_sha256": oof_builder.sha256(args.point_checkpoint),
        "fold_uq": fold_reports,
        "native_lower": lower,
        "native_upper": upper,
        "member_disagreement_signal": signal,
        "full_point_residual": full_residual,
        "scored_mask": scored,
        "full_point_factor": full_factor,
    }
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.feature_cache.with_suffix(args.feature_cache.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.feature_cache)
    return payload


def cache_tensor(payload: Mapping[str, object], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"feature cache has no tensor {key}")
    return value


@torch.inference_mode()
def exact_metrics(
        features: Mapping[str, object], residual: torch.Tensor,
        factor: torch.Tensor, indices: torch.Tensor,
        strength: float, sharpness: float, device: torch.device,
        batch_size: int) -> dict[str, float]:
    numerator = 0.0
    denominator = 0
    coverage = 0
    width_sum = 0.0
    for offset in range(0, indices.numel(), batch_size):
        chosen = indices[offset:offset + batch_size]
        lower = cache_tensor(features, "native_lower").index_select(
            0, chosen).to(device=device, dtype=torch.float32)
        upper = cache_tensor(features, "native_upper").index_select(
            0, chosen).to(device=device, dtype=torch.float32)
        signal = cache_tensor(features, "member_disagreement_signal").index_select(
            0, chosen).to(device=device, dtype=torch.float32)
        batch_residual = residual.index_select(0, chosen).to(
            device=device, dtype=torch.float32)
        scored = cache_tensor(features, "scored_mask").index_select(
            0, chosen).to(device=device)
        batch_factor = factor.index_select(0, chosen).to(
            device=device, dtype=torch.float32).view(-1, 1, 1, 1, 1)
        scale = budgeted_scale(lower, upper, signal, strength, sharpness)
        lower = lower * scale
        upper = upper * scale
        inside = (batch_residual >= -lower) & (batch_residual <= upper)
        value = batch_factor * inside * torch.exp(
            -(lower + upper) / common.SPS_SIGMA)
        numerator += float(value[scored].sum(dtype=torch.float64))
        denominator += int(scored.sum())
        coverage += int((inside & scored).sum())
        width_sum += float((lower + upper)[scored].sum(dtype=torch.float64))
    return {
        "sps_score": 100.0 * numerator / denominator,
        "coverage": coverage / denominator,
        "mean_width": width_sum / denominator,
        "scored_elements": denominator,
    }


def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    total = sum(int(row["scored_elements"]) for row in rows)
    return {
        key: sum(float(row[key]) * int(row["scored_elements"]) for row in rows) / total
        for key in ("sps_score", "coverage", "mean_width")
    } | {"scored_elements": total}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    strengths = parse_grid(args.strengths)
    sharpnesses = parse_grid(args.sharpnesses, positive=True)
    if 0.0 not in strengths:
        strengths.append(0.0)
        strengths.sort()
    oof_cache = torch.load(
        str(args.oof_residual_cache), map_location="cpu", weights_only=False,
        mmap=True)
    if oof_cache.get("schema") != oof_builder.SCHEMA:
        raise ValueError("unexpected point-OOF residual cache schema")
    if args.feature_cache.exists():
        features = torch.load(
            str(args.feature_cache), map_location="cpu", weights_only=False,
            mmap=True)
    else:
        features = collect_features(args, oof_cache)
    if features.get("schema") != FEATURE_SCHEMA:
        raise ValueError("unexpected UQ allocation feature cache schema")
    if features.get("sample_identity_sha256") != oof_cache.get("sample_identity_sha256"):
        raise ValueError("OOF residual and UQ feature cache identities differ")
    residual = cache_tensor(oof_cache, "signed_residual_physical_uv")
    factor = cache_tensor(oof_cache, "oof_point_factor")
    full_residual = cache_tensor(features, "full_point_residual")
    full_factor = cache_tensor(features, "full_point_factor")
    identities = oof_cache["sample_identities"]
    folds = torch.tensor([int(item["fold"]) for item in identities])
    device = torch.device(args.device)
    candidates = [(strength, sharpness) for strength in strengths for sharpness in sharpnesses]
    metrics: dict[tuple[int, float, float], dict[str, float]] = {}
    for fold in range(args.folds):
        indices = torch.nonzero(folds == fold, as_tuple=False).flatten()
        for strength, sharpness in candidates:
            metrics[(fold, strength, sharpness)] = exact_metrics(
                features, residual, factor, indices, strength, sharpness,
                device, args.batch_size)
    nested = []
    for held in range(args.folds):
        train_folds = [fold for fold in range(args.folds) if fold != held]
        selected = max(
            candidates,
            key=lambda candidate: aggregate([
                metrics[(fold, *candidate)] for fold in train_folds
            ])["sps_score"])
        baseline = metrics[(held, 0.0, sharpnesses[0])]
        candidate = metrics[(held, *selected)]
        held_indices = torch.nonzero(folds == held, as_tuple=False).flatten()
        full_baseline = exact_metrics(
            features, full_residual, full_factor, held_indices,
            0.0, sharpnesses[0], device, args.batch_size)
        full_candidate = exact_metrics(
            features, full_residual, full_factor, held_indices,
            selected[0], selected[1], device, args.batch_size)
        nested.append({
            "held_fold": held,
            "selected_strength": selected[0],
            "selected_sharpness": selected[1],
            "baseline": baseline,
            "candidate": candidate,
            "sps_delta": candidate["sps_score"] - baseline["sps_score"],
            "mean_width_delta": candidate["mean_width"] - baseline["mean_width"],
            "full_point_baseline": full_baseline,
            "full_point_candidate": full_candidate,
            "full_point_sps_delta": (
                full_candidate["sps_score"] - full_baseline["sps_score"]),
        })
    baseline_oof = aggregate([row["baseline"] for row in nested])
    candidate_oof = aggregate([row["candidate"] for row in nested])
    full_baseline_oof = aggregate([
        row["full_point_baseline"] for row in nested])
    full_candidate_oof = aggregate([
        row["full_point_candidate"] for row in nested])
    positive = sum(row["sps_delta"] > 0.0 for row in nested)
    full_positive = sum(row["full_point_sps_delta"] > 0.0 for row in nested)
    uq_summary = json.loads(args.uq_summary.read_text(encoding="utf-8"))
    expected_full_point_oof = uq_summary["final_oof"]
    reproduction_delta = {
        key: full_baseline_oof[key] - float(expected_full_point_oof[key])
        for key in ("sps_score", "coverage", "mean_width")}
    gate = {
        "nested_oof_sps_delta_min": 0.10,
        "nested_full_point_sps_delta_min": 0.0,
        "positive_folds_min": 4,
        "mean_width_abs_delta_max": 2.0e-6,
        "base_uq_sps_reproduction_abs_error_max": 0.03,
    }
    conditions = {
        "nested_oof_sps_improves": (
            candidate_oof["sps_score"] - baseline_oof["sps_score"]
            >= gate["nested_oof_sps_delta_min"]),
        "fold_gain_is_consistent": positive >= gate["positive_folds_min"],
        "allocation_transfers_to_full_point_residual": (
            full_candidate_oof["sps_score"] - full_baseline_oof["sps_score"]
            >= gate["nested_full_point_sps_delta_min"]),
        "width_budget_is_preserved": abs(
            candidate_oof["mean_width"] - baseline_oof["mean_width"]
        ) <= gate["mean_width_abs_delta_max"],
        "base_uq_metric_is_reproduced": abs(
            reproduction_delta["sps_score"]
        ) <= gate["base_uq_sps_reproduction_abs_error_max"],
    }
    report = {
        "schema": SCHEMA,
        "oof_residual_cache": str(args.oof_residual_cache.resolve()),
        "oof_residual_cache_sha256": oof_builder.sha256(args.oof_residual_cache),
        "feature_cache": str(args.feature_cache.resolve()),
        "feature_cache_sha256": oof_builder.sha256(args.feature_cache),
        "grid": {"strength": strengths, "sharpness": sharpnesses},
        "nested_folds": nested,
        "nested_oof_baseline": baseline_oof,
        "nested_oof_candidate": candidate_oof,
        "nested_oof_sps_delta": (
            candidate_oof["sps_score"] - baseline_oof["sps_score"]),
        "positive_folds": positive,
        "nested_full_point_baseline": full_baseline_oof,
        "nested_full_point_candidate": full_candidate_oof,
        "nested_full_point_sps_delta": (
            full_candidate_oof["sps_score"]
            - full_baseline_oof["sps_score"]),
        "full_point_positive_folds": full_positive,
        "expected_full_point_uq_oof": expected_full_point_oof,
        "base_uq_reproduction_delta": reproduction_delta,
        "gate": {"thresholds": gate, "conditions": conditions,
                 "pass": all(conditions.values())},
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
    parser.add_argument("--oof-residual-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--uq-fold-dir", type=Path, required=True)
    parser.add_argument("--uq-summary", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--strengths", default="-0.60,-0.45,-0.30,-0.20,-0.10,0,0.10,0.20,0.30,0.45,0.60")
    parser.add_argument("--sharpnesses", default="0.25,0.50,1.0,2.0,4.0")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({
        "event": "b210_xcb_point_oof_budget_allocation_complete",
        "nested_oof_sps_delta": report["nested_oof_sps_delta"],
        "positive_folds": report["positive_folds"],
        "gate": report["gate"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
