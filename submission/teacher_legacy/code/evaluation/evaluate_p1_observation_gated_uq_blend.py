#!/usr/bin/env python3
"""Bounded observation-gated A/B/C uncertainty blend for exact P1.

The packaged P1 point is loaded from a frozen parity cache and never changed.
Only the two existing P0 uncertainty-composition scalars vary per sample.  A
deterministic sampled hard-SPS oracle supplies two labels, and one fixed ridge
model predicts them from the eight raw-history statistics preregistered by the
observation-gated point experiment.  This runner does not train a neural model,
add a model forward, calibrate interval centers/widths, package, or submit.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO_ROOT / "code" / "evaluation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

import evaluate_p1_observation_gated_point_blend as point_gate  # noqa: E402


EVIDENCE_CLASS = "fixed725_condition5_true_loao_observation_gated_uq_blend"
POINT_CACHE_SHA256 = (
    "d205224b93d59f79b476ec6c22715aa1a262af79b0ee1c88f769e43d810769c7")
POINT_ARRAY_SHA256 = (
    "9b2d1f11e6017f6e505d52a62ec2158aa67c2bb86d02bdb10a66823dfbb46c52")
TARGET_ARRAY_SHA256 = (
    "88d334a29e9adc540471ab3cba859c8f8d07cf2bed4f3377a18eccd1ca8b2cc9")
BASELINE_WEIGHT_B = 0.625
BASELINE_WEIGHT_C = 0.25
WEIGHT_B_BOUNDS = (0.25, 1.0)
WEIGHT_C_BOUNDS = (0.0, 0.5)
ROUND_DELTAS = (
    (-0.20, -0.10, 0.0, 0.10, 0.20),
    (-0.05, -0.025, 0.0, 0.025, 0.05),
)
MAX_SAMPLE_ELEMENTS = 32768
RIDGE_ALPHA = 1.0
ORACLE_MIN_SPS_GAIN = 0.20
POOLED_MIN_SPS_GAIN = 0.08
CONDITION_MIN_POSITIVE = 4
CONDITION_WORST_FLOOR = -0.03
LOAO_MIN_POSITIVE = 4
LOAO_WORST_FLOOR = -0.05
MAX_COVERAGE_DROP = 0.0025
MAX_WIDTH_RATIO = 1.02
RECONSTRUCTION_BUDGET_SECONDS = 12.0 * 60.0
TOTAL_BUDGET_SECONDS = 40.0 * 60.0
METRIC_TOLERANCE = 5e-6
AOA_VALUES = point_gate.AOA_VALUES
FEATURE_NAMES = point_gate.FEATURE_NAMES
COMPONENT_KEYS = (
    "a_prediction", "a_lower", "a_upper",
    "b_prediction", "b_lower", "b_upper", "c_prediction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint-a", type=Path, required=True)
    parser.add_argument("--oof-dir-a", type=Path, required=True)
    parser.add_argument("--point-checkpoint-b", type=Path, required=True)
    parser.add_argument("--oof-dir-b", type=Path, required=True)
    parser.add_argument("--mean-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--p0-archive", type=Path, required=True)
    parser.add_argument("--p1-archive", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--point-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_point_cache(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    identity = point_gate.s1.file_identity(path)
    if identity["sha256"] != POINT_CACHE_SHA256:
        raise ValueError("exact P1 point cache SHA-256 differs from contract")
    # PyTorch 2.1 accepts mmap only when the filename is a plain string.
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if set(payload) != {
            "prediction", "target", "fold", "true_aoa",
            "observation_uv_fp16", "metadata"}:
        raise ValueError("exact P1 point cache keys differ from contract")
    metadata = payload["metadata"]
    if (metadata.get("archive_sha256") != point_gate.P1_ARCHIVE_SHA256
            or metadata.get("point_identity", {}).get("sha256")
            != POINT_ARRAY_SHA256
            or metadata.get("target_identity", {}).get("sha256")
            != TARGET_ARRAY_SHA256
            or metadata.get("parity") != {"1": True, "16": True, "33": True}):
        raise ValueError("exact P1 point cache provenance differs from contract")
    arrays = {
        key: payload[key].numpy()
        for key in ("prediction", "target", "fold", "true_aoa")
    }
    expected_shapes = {
        "prediction": (725, 20, 32, 64, 3),
        "target": (725, 20, 32, 64, 3),
        "fold": (725,),
        "true_aoa": (725,),
    }
    if any(arrays[key].shape != shape
           for key, shape in expected_shapes.items()):
        raise ValueError("exact P1 point cache shapes differ from contract")
    if (point_gate.s1.array_identity(arrays["prediction"])["sha256"]
            != POINT_ARRAY_SHA256
            or point_gate.s1.array_identity(arrays["target"])["sha256"]
            != TARGET_ARRAY_SHA256):
        raise ValueError("exact P1 cached arrays differ from frozen identities")
    return arrays, {"file": identity, "metadata": metadata}


def compose_sample_bounds(
        components: dict[str, np.ndarray], weight_b: np.ndarray,
        weight_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = components["a_prediction"].shape
    if (len(shape) != 5 or any(components[key].shape != shape
                               for key in COMPONENT_KEYS)):
        raise ValueError("A/B/C components must share shape (N,T,H,W,C)")
    n = shape[0]
    wb = np.asarray(weight_b, dtype=np.float32)
    wc = np.asarray(weight_c, dtype=np.float32)
    if wb.shape != (n,) or wc.shape != (n,):
        raise ValueError("sample weight arrays must have shape (N,)")
    if (not np.all(np.isfinite(wb)) or not np.all(np.isfinite(wc))
            or np.any(wb < WEIGHT_B_BOUNDS[0])
            or np.any(wb > WEIGHT_B_BOUNDS[1])
            or np.any(wc < WEIGHT_C_BOUNDS[0])
            or np.any(wc > WEIGHT_C_BOUNDS[1])):
        raise ValueError("sample uncertainty weights are outside contract")
    broadcast = (n, 1, 1, 1, 1)
    wb = wb.reshape(broadcast)
    wc = wc.reshape(broadcast)
    base_prediction = (
        components["a_prediction"] + wb * (
            components["b_prediction"] - components["a_prediction"]))
    lower = components["a_lower"] + wb * (
        components["b_lower"] - components["a_lower"])
    upper = components["a_upper"] + wb * (
        components["b_upper"] - components["a_upper"])
    correction = wc * (
        components["c_prediction"].mean(axis=1, keepdims=True)
        - base_prediction.mean(axis=1, keepdims=True))
    lower = (lower + correction).astype(np.float32, copy=False)
    upper = (upper + correction).astype(np.float32, copy=False)
    if (not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(lower > upper)):
        raise FloatingPointError("composed uncertainty bounds are invalid")
    return lower, upper


def official_sample_weights(
        prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    official = point_gate.OFFICIAL
    channels = official.measured_channels(target)
    dm = official.rel_l2_per_sample(prediction, target, channels)
    tke = official.tke_rel_l2_per_sample(prediction, target, channels)
    mvpe = official.mvpe_rel_l2_per_sample(prediction, target)
    normalized = [value / (0.5 + value) for value in (dm, tke, mvpe)]
    weights = (
        0.5 * (1.0 - normalized[0])
        + 0.3 * (1.0 - normalized[1])
        + 0.2 * (1.0 - normalized[2]))
    if weights.shape != (len(prediction),) or not np.all(np.isfinite(weights)):
        raise FloatingPointError("official per-sample SPS weights are invalid")
    return weights.astype(np.float64, copy=False)


def stratified_sample(
        components: dict[str, np.ndarray], target: np.ndarray,
        sample_index: int, sample_weight: float,
        max_elements: int = MAX_SAMPLE_ELEMENTS) -> dict[str, np.ndarray]:
    channels = point_gate.OFFICIAL.measured_channels(target)
    horizon = target.shape[1]
    per_stratum = max(1, max_elements // (horizon * channels))
    parts = {key: [] for key in (
        "target", "a_lower", "lower_delta", "a_upper", "upper_delta",
        "correction_index")}
    spatial_width = target.shape[3]
    full_channels = target.shape[4]
    for time_index in range(horizon):
        for channel in range(channels):
            truth = target[
                sample_index, time_index, :, :, channel].reshape(-1)
            valid = np.flatnonzero(truth != 0.0)
            count = min(per_stratum, len(valid))
            if count == 0:
                continue
            selected = valid[np.linspace(
                0, len(valid) - 1, count, dtype=np.int64)]
            row = selected // spatial_width
            column = selected % spatial_width
            correction_index = (
                (row * spatial_width + column) * full_channels + channel)
            a_lower = components["a_lower"][
                sample_index, time_index, :, :, channel].reshape(-1)[selected]
            b_lower = components["b_lower"][
                sample_index, time_index, :, :, channel].reshape(-1)[selected]
            a_upper = components["a_upper"][
                sample_index, time_index, :, :, channel].reshape(-1)[selected]
            b_upper = components["b_upper"][
                sample_index, time_index, :, :, channel].reshape(-1)[selected]
            parts["target"].append(truth[selected])
            parts["a_lower"].append(a_lower)
            parts["lower_delta"].append(b_lower - a_lower)
            parts["a_upper"].append(a_upper)
            parts["upper_delta"].append(b_upper - a_upper)
            parts["correction_index"].append(correction_index)
    if not parts["target"]:
        raise ValueError("sample has no scored target element")
    result = {key: np.concatenate(value)[:max_elements]
              for key, value in parts.items()}
    result["a_prediction"] = components["a_prediction"][sample_index]
    result["b_prediction"] = components["b_prediction"][sample_index]
    result["c_mean"] = components["c_prediction"][sample_index].mean(axis=0)
    result["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
    if len(result["target"]) > max_elements:
        raise RuntimeError("stratified sampler exceeded its fixed budget")
    return result


def hard_sps_objective(sample: dict[str, np.ndarray],
                       weight_b: float, weight_c: float) -> float:
    wb = np.float32(weight_b)
    wc = np.float32(weight_c)
    base_prediction = sample["a_prediction"] + wb * (
        sample["b_prediction"] - sample["a_prediction"])
    correction = wc * (sample["c_mean"] - base_prediction.mean(axis=0))
    correction = correction.reshape(-1)[sample["correction_index"]]
    lower = sample["a_lower"] + wb * sample["lower_delta"] + correction
    upper = sample["a_upper"] + wb * sample["upper_delta"] + correction
    if np.any(lower > upper):
        return float("-inf")
    inside = (sample["target"] >= lower) & (sample["target"] <= upper)
    width = (upper - lower) / point_gate.OFFICIAL.SIGMA_GLOBAL
    element = float(sample["sample_weight"]) * np.exp(-width)
    value = float(np.sum(element, where=inside, dtype=np.float64)
                  / len(element))
    return value if np.isfinite(value) else float("-inf")


def choose_trial(trials: list[dict]) -> dict:
    """Highest objective; exact ties prefer zero then smaller |delta|."""
    return max(trials, key=lambda row: (
        row["objective"], row["delta"] == 0.0,
        -abs(row["delta"]), -row["delta"]))


def coordinate_oracle(sample: dict[str, np.ndarray]) -> dict[str, object]:
    weights = np.asarray(
        [BASELINE_WEIGHT_B, BASELINE_WEIGHT_C], dtype=np.float64)
    identity = hard_sps_objective(sample, *weights)
    objective = identity
    trace = []
    bounds = (WEIGHT_B_BOUNDS, WEIGHT_C_BOUNDS)
    names = ("bounds_weight_b", "bounds_mean_expert_weight")
    for round_index, deltas in enumerate(ROUND_DELTAS, start=1):
        for coordinate, name in enumerate(names):
            anchor = float(weights[coordinate])
            trials = []
            for delta in deltas:
                candidate = weights.copy()
                candidate[coordinate] = np.clip(
                    anchor + delta, *bounds[coordinate])
                trials.append({
                    "delta": float(delta),
                    "value": float(candidate[coordinate]),
                    "objective": hard_sps_objective(sample, *candidate),
                    "weights": candidate,
                })
            selected = choose_trial(trials)
            weights = selected["weights"]
            objective = selected["objective"]
            trace.append({
                "round": round_index,
                "coordinate": name,
                "anchor": anchor,
                "selected_delta": selected["delta"],
                "selected_value": selected["value"],
                "objective": objective,
            })
    return {
        "weight_b": float(weights[0]),
        "weight_c": float(weights[1]),
        "identity_objective": identity,
        "selected_objective": objective,
        "sampled_objective_gain": objective - identity,
        "sample_elements": int(len(sample["target"])),
        "trace": trace,
    }


def oracle_labels(
        components: dict[str, np.ndarray], prediction: np.ndarray,
        target: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    sample_weights = official_sample_weights(prediction, target)
    labels = np.empty((len(target), 2), dtype=np.float64)
    records = []
    for index in range(len(target)):
        sampled = stratified_sample(
            components, target, index, sample_weights[index])
        result = coordinate_oracle(sampled)
        labels[index] = (result["weight_b"], result["weight_c"])
        records.append({"sample": index, **result})
    return labels, records


def fit_ridge(features: np.ndarray,
              labels: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or y.shape != (len(x), 2):
        raise ValueError("ridge inputs have unexpected shape")
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-12, scale, 1.0)
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict_ridge(model: dict[str, np.ndarray],
                  features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    design = np.column_stack((
        np.ones(len(x)), (x - model["mean"]) / model["scale"]))
    prediction = design @ model["coefficients"]
    prediction[:, 0] = np.clip(prediction[:, 0], *WEIGHT_B_BOUNDS)
    prediction[:, 1] = np.clip(prediction[:, 1], *WEIGHT_C_BOUNDS)
    return prediction


def serialize_ridge(model: dict[str, np.ndarray]) -> dict:
    return {
        "feature_mean": model["mean"].tolist(),
        "feature_scale": model["scale"].tolist(),
        "coefficients_intercept_first": model["coefficients"].tolist(),
    }


def crossfit_ridge(
        features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
        expected_groups: tuple[int, ...]) -> tuple[np.ndarray, list[dict]]:
    if tuple(sorted(np.unique(groups).tolist())) != expected_groups:
        raise ValueError("crossfit groups differ from contract")
    predicted = np.full_like(labels, np.nan, dtype=np.float64)
    records = []
    for held in expected_groups:
        train = groups != held
        test = groups == held
        model = fit_ridge(features[train], labels[train])
        predicted[test] = predict_ridge(model, features[test])
        records.append({
            "held": int(held),
            "train_samples": int(np.count_nonzero(train)),
            "test_samples": int(np.count_nonzero(test)),
            **serialize_ridge(model),
        })
    if not np.all(np.isfinite(predicted)):
        raise RuntimeError("crossfit ridge left unpredicted weights")
    return predicted, records


def metric_comparison(
        prediction: np.ndarray, target: np.ndarray,
        baseline_bounds: tuple[np.ndarray, np.ndarray],
        candidate_bounds: tuple[np.ndarray, np.ndarray],
        mask: np.ndarray | None = None) -> dict[str, object]:
    if mask is None:
        mask = np.ones(len(prediction), dtype=bool)
    before = point_gate.score(
        prediction[mask], baseline_bounds[0][mask],
        baseline_bounds[1][mask], target[mask])
    after = point_gate.score(
        prediction[mask], candidate_bounds[0][mask],
        candidate_bounds[1][mask], target[mask])
    return {
        "samples": int(np.count_nonzero(mask)),
        "baseline": before,
        "candidate": after,
        "sps_delta": after["sps_score"] - before["sps_score"],
        "coverage_delta": after["coverage"] - before["coverage"],
        "width_ratio": after["mean_width"] / before["mean_width"],
    }


def partition_rows(
        prediction: np.ndarray, target: np.ndarray,
        baseline_bounds: tuple[np.ndarray, np.ndarray],
        candidate_bounds: tuple[np.ndarray, np.ndarray], groups: np.ndarray,
        expected_groups: tuple[int, ...], name: str) -> list[dict]:
    rows = []
    for held in expected_groups:
        rows.append({name: int(held), **metric_comparison(
            prediction, target, baseline_bounds, candidate_bounds,
            groups == held)})
    return rows


def gate_checks(oracle_gain: float, pooled: dict,
                condition_rows: list[dict], loao_rows: list[dict]) -> dict:
    condition_gains = [row["sps_delta"] for row in condition_rows]
    loao_gains = [row["sps_delta"] for row in loao_rows]
    return {
        "oracle_sps_gain_ge_0_20": oracle_gain >= ORACLE_MIN_SPS_GAIN,
        "deployable_pooled_sps_gain_ge_0_08": (
            pooled["sps_delta"] >= POOLED_MIN_SPS_GAIN),
        "condition_positive_ge_4_of_5": (
            sum(value > 0.0 for value in condition_gains)
            >= CONDITION_MIN_POSITIVE),
        "condition_worst_ge_minus_0_03": (
            min(condition_gains) >= CONDITION_WORST_FLOOR),
        "loao_positive_ge_4_of_5": (
            sum(value > 0.0 for value in loao_gains) >= LOAO_MIN_POSITIVE),
        "loao_worst_ge_minus_0_05": min(loao_gains) >= LOAO_WORST_FLOOR,
        "deployable_coverage_drop_le_0_0025": (
            pooled["coverage_delta"] >= -MAX_COVERAGE_DROP),
        "deployable_mean_width_le_1_02x": (
            pooled["width_ratio"] <= MAX_WIDTH_RATIO),
    }


def reconstruct_components(
        args: argparse.Namespace, dataset, sample_folds, fold_manifest,
        raw_features: np.ndarray, raw_aoas: np.ndarray,
        device: torch.device) -> dict[str, object]:
    summaries = {
        "a": point_gate.s1.validate_oof_summary(
            args.oof_dir_a / "summary.json", args.point_checkpoint_a,
            fold_manifest),
        "b": point_gate.s1.validate_oof_summary(
            args.oof_dir_b / "summary.json", args.point_checkpoint_b,
            fold_manifest),
    }
    payload_a = torch.load(
        args.point_checkpoint_a, map_location="cpu", weights_only=False)
    payload_b = torch.load(
        args.point_checkpoint_b, map_location="cpu", weights_only=False)
    normalizer = point_gate.common.repro.Normalizer(args.real_stats, device)
    mean_expert, mean_metadata = point_gate.load_model(
        args.mean_expert_checkpoint, normalizer, device)
    loader_args = argparse.Namespace(num_workers=args.num_workers)
    parts = {key: [] for key in COMPONENT_KEYS}
    target_parts = []
    feature_parts = []
    aoa_parts = []
    fold_parts = []
    fold_metadata = []
    all_indices = list(range(len(dataset)))
    for fold in range(5):
        indices = [index for index in all_indices if sample_folds[index] == fold]
        loader = point_gate.uq_train.make_loader(
            dataset, indices, args.batch_size, False, loader_args, seed=0)
        model_a, metadata_a = point_gate.s1.load_fold_model(
            payload_a, args.oof_dir_a / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_a, normalizer, device)
        values_a = point_gate.collect(model_a, loader, normalizer, device)
        del model_a
        model_b, metadata_b = point_gate.s1.load_fold_model(
            payload_b, args.oof_dir_b / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_b, normalizer, device)
        values_b = point_gate.collect(model_b, loader, normalizer, device)
        del model_b
        values_c = point_gate.collect(mean_expert, loader, normalizer, device)
        if (not np.array_equal(values_a[3], values_b[3])
                or not np.array_equal(values_a[3], values_c[3])):
            raise RuntimeError(f"target order changed in fold {fold}")
        for key, value in zip(
                ("a_prediction", "a_lower", "a_upper"), values_a[:3]):
            parts[key].append(value)
        for key, value in zip(
                ("b_prediction", "b_lower", "b_upper"), values_b[:3]):
            parts[key].append(value)
        parts["c_prediction"].append(values_c[0])
        target_parts.append(values_a[3])
        feature_parts.append(raw_features[indices])
        aoa_parts.append(raw_aoas[indices])
        fold_parts.append(np.full(len(indices), fold, dtype=np.int64))
        fold_metadata.append({
            "fold": fold, "samples": len(indices),
            "model_a": metadata_a, "model_b": metadata_b,
        })
        print(json.dumps({
            "event": "observation_uq_reconstruct", "fold": fold,
            "samples": len(indices)}), flush=True)
        del values_a, values_b, values_c
        torch.cuda.empty_cache()
    del mean_expert, payload_a, payload_b, normalizer
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "components": {
            key: np.concatenate(value) for key, value in parts.items()},
        "target": np.concatenate(target_parts),
        "features": np.concatenate(feature_parts),
        "aoas": np.concatenate(aoa_parts),
        "folds": np.concatenate(fold_parts),
        "fold_metadata": fold_metadata,
        "oof_summaries": summaries,
        "mean_expert": mean_metadata,
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    identities = point_gate.validate_fixed_contract(args)
    cache, cache_identity = load_point_cache(args.point_cache)
    device = torch.device(args.device)
    reconstruction_started = time.perf_counter()
    trajectories, metadata = point_gate.common.repro.load_trajectories(
        args.real_dir, 0)
    point_windows, tail_windows, runtime_manifest = (
        point_gate.common.repro.make_split(metadata, 20, 0.2))
    if len(point_windows) != 2576 or len(tail_windows) != 725:
        raise ValueError("frozen 2576/725 split size changed")
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    dataset = point_gate.common.repro.WindowDataset(trajectories, tail_windows)
    sample_folds, fold_manifest = point_gate.trajectory_folds(
        runtime_manifest, 5)
    point_gate.s1.validate_s1_split(
        runtime_manifest, frozen_manifest, len(dataset), sample_folds)
    raw_features = []
    raw_aoas = []
    for index, window in enumerate(tail_windows):
        history, _target = dataset[index]
        raw_features.append(point_gate.observation_features(history.numpy()))
        raw_aoas.append(int(metadata[window.trajectory]["aoa"]))
    reconstructed = reconstruct_components(
        args, dataset, sample_folds, fold_manifest,
        np.asarray(raw_features, dtype=np.float64),
        np.asarray(raw_aoas, dtype=np.int64), device)
    del trajectories, dataset
    reconstruction_elapsed = time.perf_counter() - reconstruction_started
    target = reconstructed["target"]
    features = reconstructed["features"]
    folds = reconstructed["folds"]
    aoas = reconstructed["aoas"]
    components = reconstructed["components"]
    prediction = cache["prediction"]
    if (not np.array_equal(cache["target"], target)
            or not np.array_equal(cache["fold"], folds)
            or not np.array_equal(cache["true_aoa"], aoas)):
        raise RuntimeError("point cache order differs from OOF reconstruction")
    point_identity_before = point_gate.s1.array_identity(prediction)
    prediction.setflags(write=False)
    target.setflags(write=False)
    baseline_weights_b = np.full(
        len(target), BASELINE_WEIGHT_B, dtype=np.float64)
    baseline_weights_c = np.full(
        len(target), BASELINE_WEIGHT_C, dtype=np.float64)
    baseline_bounds = compose_sample_bounds(
        components, baseline_weights_b, baseline_weights_c)
    baseline_bounds_identity = {
        side: point_gate.s1.array_identity(value)
        for side, value in zip(("lower", "upper"), baseline_bounds)}
    if any(baseline_bounds_identity[side]["sha256"]
           != point_gate.P0_BOUNDS_SHA256[side]
           for side in ("lower", "upper")):
        raise RuntimeError("P0 baseline bounds differ from bitwise contract")
    baseline = point_gate.score(
        prediction, baseline_bounds[0], baseline_bounds[1], target)
    report = {
        "experiment": "track1_p1_observation_gated_uq_blend",
        "evidence_class": EVIDENCE_CLASS,
        "contract": {
            "point": "frozen packaged P1 bitwise cache; never recomposed",
            "baseline_bounds_weight_b": BASELINE_WEIGHT_B,
            "baseline_bounds_mean_expert_weight": BASELINE_WEIGHT_C,
            "candidate_weight_b_bounds": list(WEIGHT_B_BOUNDS),
            "candidate_weight_c_bounds": list(WEIGHT_C_BOUNDS),
            "feature_names": list(FEATURE_NAMES),
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_features": "train-fold z-score only",
            "ridge_output": "raw two-weight prediction clipped to each range",
            "coordinate_order": ["bounds_weight_b", "bounds_mean_expert_weight"],
            "round_deltas": [list(value) for value in ROUND_DELTAS],
            "tie_break": "zero delta, then smaller absolute delta",
            "oracle_sampling": (
                "per sample deterministic 20-horizon x 2-channel strata; "
                "at most 32768 measured elements"),
            "oracle_objective": "starting-kit hard-SPS local additive term",
            "no_posthoc_center_or_width_calibration": True,
            "no_extra_forward_at_deployment": True,
        },
        "gates": {
            "oracle_min_sps_gain": ORACLE_MIN_SPS_GAIN,
            "deployable_pooled_min_sps_gain": POOLED_MIN_SPS_GAIN,
            "condition_min_positive": CONDITION_MIN_POSITIVE,
            "condition_worst_floor": CONDITION_WORST_FLOOR,
            "loao_min_positive": LOAO_MIN_POSITIVE,
            "loao_worst_floor": LOAO_WORST_FLOOR,
            "max_coverage_drop": MAX_COVERAGE_DROP,
            "max_mean_width_ratio": MAX_WIDTH_RATIO,
            "reconstruction_budget_seconds": RECONSTRUCTION_BUDGET_SECONDS,
            "total_budget_seconds": TOTAL_BUDGET_SECONDS,
        },
        "identities": identities,
        "point_cache": cache_identity,
        "point_identity_before": point_identity_before,
        "target_identity": point_gate.s1.array_identity(target),
        "feature_identity": point_gate.s1.array_identity(features),
        "baseline_bounds_identity": baseline_bounds_identity,
        "baseline": baseline,
        "oof_summaries": reconstructed["oof_summaries"],
        "mean_expert": reconstructed["mean_expert"],
        "fold_metadata": reconstructed["fold_metadata"],
        "reconstruction_elapsed_seconds": reconstruction_elapsed,
    }
    if reconstruction_elapsed > RECONSTRUCTION_BUDGET_SECONDS:
        report["decision"] = "STOP_OBSERVATION_UQ_RECONSTRUCTION_BUDGET"
        write_report(args.output, report)
        return

    labels, oracle_records = oracle_labels(components, prediction, target)
    oracle_bounds = compose_sample_bounds(
        components, labels[:, 0], labels[:, 1])
    oracle = metric_comparison(
        prediction, target, baseline_bounds, oracle_bounds)
    report["oracle"] = {
        **oracle,
        "labels": labels.tolist(),
        "weight_b_summary": {
            "min": float(labels[:, 0].min()),
            "mean": float(labels[:, 0].mean()),
            "max": float(labels[:, 0].max()),
        },
        "weight_c_summary": {
            "min": float(labels[:, 1].min()),
            "mean": float(labels[:, 1].mean()),
            "max": float(labels[:, 1].max()),
        },
        "sample_records": oracle_records,
    }
    if oracle["sps_delta"] < ORACLE_MIN_SPS_GAIN:
        report["point_identity_after"] = point_gate.s1.array_identity(prediction)
        report["point_bitwise_frozen"] = (
            report["point_identity_after"] == point_identity_before)
        report["elapsed_seconds"] = time.perf_counter() - started
        report["decision"] = "STOP_OBSERVATION_UQ_ORACLE_BELOW_0P20"
        write_report(args.output, report)
        print(json.dumps({
            "event": "observation_uq_complete",
            "decision": report["decision"],
            "oracle_sps_gain": oracle["sps_delta"]}), flush=True)
        return

    condition_weights, condition_models = crossfit_ridge(
        features, labels, folds, tuple(range(5)))
    condition_bounds = compose_sample_bounds(
        components, condition_weights[:, 0], condition_weights[:, 1])
    condition_pooled = metric_comparison(
        prediction, target, baseline_bounds, condition_bounds)
    condition_rows = partition_rows(
        prediction, target, baseline_bounds, condition_bounds,
        folds, tuple(range(5)), "held_fold")
    loao_weights, loao_models = crossfit_ridge(
        features, labels, aoas, AOA_VALUES)
    loao_bounds = compose_sample_bounds(
        components, loao_weights[:, 0], loao_weights[:, 1])
    loao_pooled = metric_comparison(
        prediction, target, baseline_bounds, loao_bounds)
    loao_rows = partition_rows(
        prediction, target, baseline_bounds, loao_bounds,
        aoas, AOA_VALUES, "held_aoa")
    checks = gate_checks(
        oracle["sps_delta"], condition_pooled, condition_rows, loao_rows)
    full_model = fit_ridge(features, labels)
    point_identity_after = point_gate.s1.array_identity(prediction)
    point_frozen = point_identity_after == point_identity_before
    elapsed = time.perf_counter() - started
    checks.update({
        "point_bitwise_frozen": point_frozen,
        "reconstruction_within_12m": (
            reconstruction_elapsed <= RECONSTRUCTION_BUDGET_SECONDS),
        "total_within_40m": elapsed <= TOTAL_BUDGET_SECONDS,
    })
    report.update({
        "condition5": {
            "pooled": condition_pooled,
            "rows": condition_rows,
            "predicted_weights": condition_weights.tolist(),
            "models": condition_models,
            "positive_folds": sum(
                row["sps_delta"] > 0.0 for row in condition_rows),
            "worst_fold_sps_delta": min(
                row["sps_delta"] for row in condition_rows),
        },
        "true_loao": {
            "pooled": loao_pooled,
            "rows": loao_rows,
            "predicted_weights": loao_weights.tolist(),
            "models": loao_models,
            "positive_aoas": sum(row["sps_delta"] > 0.0 for row in loao_rows),
            "worst_aoa_sps_delta": min(
                row["sps_delta"] for row in loao_rows),
        },
        "full_fixed725_model": serialize_ridge(full_model),
        "point_identity_after": point_identity_after,
        "point_bitwise_frozen": point_frozen,
        "gate_checks": checks,
        "elapsed_seconds": elapsed,
        "decision": (
            "PASS_OBSERVATION_GATED_UQ_BLEND"
            if all(checks.values())
            else "STOP_OBSERVATION_GATED_UQ_BLEND"),
        "limitations": [
            "Fixed725 OOF diagnostic, not official evidence.",
            "Oracle labels see each sample target and are not deployable.",
            "Only condition-five and true LOAO rows estimate held behavior.",
            "No latency benchmark, ZIP, or official submission is produced.",
        ],
    })
    write_report(args.output, report)
    print(json.dumps({
        "event": "observation_uq_complete",
        "decision": report["decision"],
        "oracle_sps_gain": oracle["sps_delta"],
        "condition_pooled_sps_gain": condition_pooled["sps_delta"],
        "condition_positive": report["condition5"]["positive_folds"],
        "condition_worst": report["condition5"]["worst_fold_sps_delta"],
        "loao_positive": report["true_loao"]["positive_aoas"],
        "loao_worst": report["true_loao"]["worst_aoa_sps_delta"],
    }), flush=True)


if __name__ == "__main__":
    main()
