#!/usr/bin/env python3
"""Evaluate fixed P0 ZIP bounds with three packaged point blends on 725 tail.

This is a deployment-path, fixed-tail post-selection diagnostic.  It does not
train, estimate Time, or provide an independent OOF estimate of blend-weight
generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from evaluate_flow_backbone_blend import OFFICIAL, score  # noqa: E402
from evaluate_mean_expert_oof_blend import (  # noqa: E402
    S1_BASELINE_ZIP_SHA256,
    S1_CANDIDATES,
    S1_FOLD_COUNTS,
    S1_REAL_STATS_SHA256,
    S1_SCORER_SHA256,
    S1_SPLIT_MANIFEST_SHA256,
    file_identity,
    point_bounds_diagnostic,
    validate_s1_split,
)
from evaluate_track1_submission_archive import (  # noqa: E402
    load_submission,
    normalized_result,
    safe_extract,
)
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


EVIDENCE_CLASS = "deployment_zip_fixed_tail_post_selection_diagnostic"
SCORER_PATH = REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"
P0_EXPECTED_METRICS = {
    "rel_l2_score": 96.005444,
    "tke_score": 81.007462,
    "mvpe_score": 96.701618,
    "sps_score": 54.002325,
}
P0_METRIC_TOLERANCE = 5e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temp-root", type=Path, default=Path("runs/tmp"))
    return parser.parse_args()


def locked_identity(path: Path, expected_sha256: str, label: str) -> dict:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 {identity['sha256']} differs from the contract")
    return identity


def validate_archive_config(package_dir: Path) -> dict:
    config = json.loads(
        (package_dir / "blend_config.json").read_text(encoding="utf-8"))
    if config.get("weight_b") != S1_CANDIDATES[0][1]:
        raise ValueError("packaged weight_b is not P0")
    if config.get("mean_expert_weight") != S1_CANDIDATES[0][2]:
        raise ValueError("packaged mean_expert_weight is not P0")
    if config.get("checkpoints") != [
            "model_a.pth", "model_b.pth", "model_c.pth"]:
        raise ValueError("packaged checkpoint order is not A/B/C")
    return config


def compose_packaged_candidate(
        distribution_a, distribution_b, prediction_c,
        weight_b: float, mean_weight: float):
    """Mirror the packaged torch dtype and operation order exactly."""
    base = tuple(
        value_a.lerp(value_b, weight_b)
        for value_a, value_b in zip(distribution_a, distribution_b))
    mean_delta = (
        prediction_c.mean(dim=1, keepdim=True)
        - base[0].mean(dim=1, keepdim=True))
    correction = mean_delta.mul(mean_weight)
    return tuple(value + correction for value in base)


def compose_packaged_point(
        prediction_a, prediction_b, prediction_c,
        weight_b: float, mean_weight: float):
    base = prediction_a.lerp(prediction_b, weight_b)
    mean_delta = (
        prediction_c.mean(dim=1, keepdim=True)
        - base.mean(dim=1, keepdim=True))
    return base + mean_delta.mul(mean_weight)


def assert_p0_matches_submission(custom_p0, packaged) -> None:
    if any(not np.array_equal(left, right)
           for left, right in zip(custom_p0, packaged)):
        raise RuntimeError("custom P0 differs bitwise from submission.predict")


def validate_p0_metrics(metrics: dict[str, float]) -> None:
    differences = {
        key: abs(metrics[key] - expected)
        for key, expected in P0_EXPECTED_METRICS.items()
    }
    failed = {
        key: difference for key, difference in differences.items()
        if difference > P0_METRIC_TOLERANCE
    }
    if failed:
        raise RuntimeError(
            f"P0 frozen-tail metric gate failed: {failed}; "
            f"tolerance={P0_METRIC_TOLERANCE}")


def bounds_reuse_evidence(
        control_lower: np.ndarray, control_upper: np.ndarray,
        candidate_lower: np.ndarray, candidate_upper: np.ndarray) -> dict:
    return {
        "same_objects_as_control": bool(
            candidate_lower is control_lower and candidate_upper is control_upper),
        "bitwise_equal_to_control": bool(
            np.array_equal(candidate_lower, control_lower)
            and np.array_equal(candidate_upper, control_upper)),
        "control_arrays_read_only": bool(
            not control_lower.flags.writeable and not control_upper.flags.writeable),
    }


def run_packaged_batch(submission, raw_input: np.ndarray):
    """Run submission.predict once while recording its packaged A/B/C values."""
    distributions = []
    expert_points = []
    original_distribution = submission._physical_distribution
    original_point = submission._physical_point

    def record_distribution(*args, **kwargs):
        value = original_distribution(*args, **kwargs)
        distributions.append(value)
        return value

    def record_point(*args, **kwargs):
        value = original_point(*args, **kwargs)
        expert_points.append(value)
        return value

    submission._physical_distribution = record_distribution
    submission._physical_point = record_point
    try:
        packaged = normalized_result(submission.predict(raw_input, metadata={}))
    finally:
        submission._physical_distribution = original_distribution
        submission._physical_point = original_point

    if len(distributions) != 2 or len(expert_points) != 1:
        raise RuntimeError(
            "one evaluator batch must map to one packaged A/B/C batch")
    packaged = tuple(np.asarray(value, dtype=np.float32) for value in packaged)
    with torch.inference_mode():
        custom_p0 = compose_packaged_candidate(
            distributions[0], distributions[1], expert_points[0],
            S1_CANDIDATES[0][1], S1_CANDIDATES[0][2])
        custom_p0 = tuple(
            value.float().cpu().numpy() for value in custom_p0)
        assert_p0_matches_submission(custom_p0, packaged)
        candidates = {"P0": custom_p0[0]}
        for name, weight_b, mean_weight in S1_CANDIDATES[1:]:
            candidates[name] = compose_packaged_point(
                distributions[0][0], distributions[1][0], expert_points[0],
                weight_b, mean_weight).float().cpu().numpy()
    return candidates, packaged[1], packaged[2]


def new_stream_identity() -> dict:
    return {
        "digest": hashlib.sha256(), "rows": 0, "tail_shape": None,
        "dtype": None, "bytes": 0,
    }


def update_stream_identity(state: dict, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    tail_shape = tuple(contiguous.shape[1:])
    dtype = str(contiguous.dtype)
    if state["tail_shape"] is None:
        state["tail_shape"] = tail_shape
        state["dtype"] = dtype
    elif state["tail_shape"] != tail_shape or state["dtype"] != dtype:
        raise ValueError("streamed array shape or dtype changed between batches")
    state["digest"].update(memoryview(contiguous).cast("B"))
    state["rows"] += len(contiguous)
    state["bytes"] += contiguous.nbytes


def finalize_stream_identity(state: dict) -> dict:
    if state["tail_shape"] is None:
        raise ValueError("cannot finalize an empty streamed array")
    return {
        "shape": [state["rows"], *state["tail_shape"]],
        "dtype": state["dtype"],
        "bytes": state["bytes"],
        "sha256": state["digest"].hexdigest(),
    }


def new_metric_group() -> dict:
    return {
        "rel_errors": [], "tke_errors": [], "mvpe_errors": [],
        "sps_numerators": np.zeros(3, dtype=np.float64),
        "scored_elements": 0, "inside_elements": 0,
        "width_sum": np.float64(0.0), "outside_elements": 0,
    }


def new_stream_accumulator() -> dict:
    return {
        "pooled": new_metric_group(),
        "folds": [new_metric_group() for _ in range(5)],
    }


def update_metric_group(
        group: dict, prediction: np.ndarray, lower: np.ndarray,
        upper: np.ndarray, target: np.ndarray,
        errors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    channels = OFFICIAL.measured_channels(target)
    p = prediction[..., :channels]
    t = target[..., :channels]
    lo = lower[..., :channels]
    hi = upper[..., :channels]
    scored = t != 0.0
    inside = (t >= lo) & (t <= hi)
    nil = (hi - lo) / OFFICIAL.SIGMA_GLOBAL
    n_scored = int(np.count_nonzero(scored))
    if n_scored == 0:
        raise ValueError("stream batch has no scored target elements")

    for key, values in zip(
            ("rel_errors", "tke_errors", "mvpe_errors"), errors):
        group[key].append(np.asarray(values))
    normalized = tuple(values / (0.5 + values) for values in errors)
    for index, values in enumerate(normalized):
        shaped = values.reshape((len(values),) + (1,) * (p.ndim - 1))
        elem = (1.0 - shaped) * np.exp(-nil)
        elem = np.where(inside, elem, 0.0)
        elem = np.where(np.isfinite(shaped), elem, 0.0)
        group["sps_numerators"][index] += np.sum(
            elem, where=scored, dtype=np.float64)
    group["scored_elements"] += n_scored
    group["inside_elements"] += int(np.count_nonzero(inside & scored))
    group["width_sum"] += np.sum(
        hi - lo, where=scored, dtype=np.float64)
    outside = (p < lo) | (p > hi)
    group["outside_elements"] += int(np.count_nonzero(outside & scored))


def update_stream_accumulator(
        accumulator: dict, prediction: np.ndarray, lower: np.ndarray,
        upper: np.ndarray, target: np.ndarray,
        sample_folds: np.ndarray) -> None:
    channels = OFFICIAL.measured_channels(target)
    errors = (
        OFFICIAL.rel_l2_per_sample(prediction, target, channels),
        OFFICIAL.tke_rel_l2_per_sample(prediction, target, channels),
        OFFICIAL.mvpe_rel_l2_per_sample(prediction, target),
    )
    update_metric_group(
        accumulator["pooled"], prediction, lower, upper, target, errors)
    for fold in range(5):
        selected = sample_folds == fold
        if not np.any(selected):
            continue
        update_metric_group(
            accumulator["folds"][fold], prediction[selected],
            lower[selected], upper[selected], target[selected],
            tuple(values[selected] for values in errors))


def finalize_metric_group(group: dict) -> dict[str, float]:
    errors = [
        np.concatenate(group[key])
        for key in ("rel_errors", "tke_errors", "mvpe_errors")
    ]
    rel_l2_score, tke_score, mvpe_score = (
        OFFICIAL.score_error(float(np.mean(values))) for values in errors)
    n_scored = group["scored_elements"]
    branch_values = group["sps_numerators"] / n_scored
    sps_value = (
        0.5 * branch_values[0]
        + 0.3 * branch_values[1]
        + 0.2 * branch_values[2])
    result = {
        "rel_l2_score": rel_l2_score,
        "tke_score": tke_score,
        "mvpe_score": mvpe_score,
        "sps_score": OFFICIAL.score_sps(float(sps_value)),
        "coverage": group["inside_elements"] / n_scored,
        "mean_width": float(group["width_sum"] / n_scored),
        "point_outside_bounds_rate": group["outside_elements"] / n_scored,
    }
    result["point_score"] = (
        0.5 * rel_l2_score + 0.3 * tke_score + 0.2 * mvpe_score)
    return result


def finalize_stream_accumulator(accumulator: dict) -> dict:
    result = finalize_metric_group(accumulator["pooled"])
    result["folds"] = [
        {"fold": fold, **finalize_metric_group(group)}
        for fold, group in enumerate(accumulator["folds"])
    ]
    return result


def point_outside_bounds_rate(
        prediction: np.ndarray, lower: np.ndarray, upper: np.ndarray,
        target: np.ndarray) -> float:
    channels = OFFICIAL.measured_channels(target)
    scored = target[..., :channels] != 0.0
    outside = (
        (prediction[..., :channels] < lower[..., :channels])
        | (prediction[..., :channels] > upper[..., :channels]))
    return float(np.count_nonzero(outside & scored) / np.count_nonzero(scored))


def validate_packaged_batch(
        predictions: dict[str, np.ndarray], lower: np.ndarray,
        upper: np.ndarray, target: np.ndarray) -> None:
    if lower.shape != target.shape or upper.shape != target.shape:
        raise ValueError("packaged P0 bounds do not match target shape")
    if (not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))):
        raise FloatingPointError("packaged P0 bounds contain non-finite values")
    if np.any(lower > upper):
        raise FloatingPointError("packaged P0 bounds are reversed")
    for name, prediction in predictions.items():
        OFFICIAL.validate_shapes(prediction, target)
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError(
                f"packaged point candidate {name} contains non-finite values")


def score_candidate(prediction, lower, upper, target, sample_folds) -> dict:
    pooled = score(prediction, lower, upper, target)
    pooled["point_outside_bounds_rate"] = point_outside_bounds_rate(
        prediction, lower, upper, target)
    folds = np.asarray(sample_folds)
    pooled["folds"] = []
    for fold in range(5):
        selected = folds == fold
        fold_metrics = score(
            prediction[selected], lower[selected], upper[selected],
            target[selected])
        fold_metrics["point_outside_bounds_rate"] = (
            point_outside_bounds_rate(
                prediction[selected], lower[selected], upper[selected],
                target[selected]))
        pooled["folds"].append({"fold": fold, **fold_metrics})
    return pooled


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    if args.batch_size != 16:
        raise ValueError("formal S1-ZIP requires batch size 16")
    if not torch.cuda.is_available():
        raise RuntimeError("the packaged deployment path requires CUDA")

    archive_identity = locked_identity(
        args.archive, S1_BASELINE_ZIP_SHA256, "baseline ZIP")
    split_identity = locked_identity(
        args.split_manifest, S1_SPLIT_MANIFEST_SHA256, "split manifest")
    stats_identity = locked_identity(
        args.real_stats, S1_REAL_STATS_SHA256, "real stats")
    scorer_identity = locked_identity(
        SCORER_PATH, S1_SCORER_SHA256, "official scorer")

    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    _train_windows, tail_windows, runtime_manifest = common.repro.make_split(
        metadata, 20, 0.2)
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    sample_folds, fold_manifest = trajectory_folds(runtime_manifest, 5)
    validate_s1_split(
        runtime_manifest, frozen_manifest, len(tail_windows), sample_folds)
    dataset = common.repro.WindowDataset(trajectories, tail_windows)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    accumulators = {
        name: new_stream_accumulator() for name, *_ in S1_CANDIDATES}
    point_identity_states = {
        name: new_stream_identity() for name, *_ in S1_CANDIDATES}
    lower_identity_state = new_stream_identity()
    upper_identity_state = new_stream_identity()
    target_identity_state = new_stream_identity()
    bounds_reuse_checks = []
    with tempfile.TemporaryDirectory(
            prefix="s1-zip-", dir=args.temp_root) as temporary:
        package_dir = Path(temporary)
        safe_extract(args.archive, package_dir)
        blend_config = validate_archive_config(package_dir)
        submission = load_submission(package_dir)
        sample_offset = 0
        for batch, (history, batch_target) in enumerate(loader):
            predictions, lower, upper = run_packaged_batch(
                submission, history.numpy())
            target = batch_target.numpy().astype(np.float32, copy=False)
            stop = sample_offset + len(target)
            batch_folds = np.asarray(sample_folds[sample_offset:stop])
            if len(batch_folds) != len(target):
                raise RuntimeError("fold assignment ended before tail loader")
            validate_packaged_batch(predictions, lower, upper, target)
            lower.setflags(write=False)
            upper.setflags(write=False)
            target.setflags(write=False)
            update_stream_identity(lower_identity_state, lower)
            update_stream_identity(upper_identity_state, upper)
            update_stream_identity(target_identity_state, target)
            candidate_bound_ids = []
            for name, prediction in predictions.items():
                candidate_lower, candidate_upper = lower, upper
                candidate_bound_ids.append(
                    (id(candidate_lower), id(candidate_upper)))
                update_stream_identity(point_identity_states[name], prediction)
                update_stream_accumulator(
                    accumulators[name], prediction,
                    candidate_lower, candidate_upper,
                    target, batch_folds)
            bounds_reuse_checks.append(bool(
                len(candidate_bound_ids) == len(S1_CANDIDATES)
                and len(set(candidate_bound_ids)) == 1
                and not lower.flags.writeable and not upper.flags.writeable))
            print(json.dumps({
                "event": "s1_zip_stream_batch", "batch": batch,
                "windows": len(target), "start": sample_offset,
                "stop": stop, "retained_full_arrays": 0,
            }), flush=True)
            sample_offset = stop
        if sample_offset != len(dataset):
            raise RuntimeError("tail loader did not produce exactly 725 windows")

    bounds_identity = {
        "lower": finalize_stream_identity(lower_identity_state),
        "upper": finalize_stream_identity(upper_identity_state),
    }
    target_identity = finalize_stream_identity(target_identity_state)
    results = {}
    for name, weight_b, mean_weight in S1_CANDIDATES:
        metrics = finalize_stream_accumulator(accumulators[name])
        if name == "P0":
            validate_p0_metrics(metrics)
        reuse = {
            "same_batch_objects_for_all_candidates": bool(
                all(bounds_reuse_checks)),
            "batch_arrays_read_only": True,
            "stream_identity_shared": True,
        }
        results[name] = {
            "name": name,
            "weight_b": weight_b,
            "mean_expert_weight_c": mean_weight,
            "point_identity": finalize_stream_identity(
                point_identity_states[name]),
            "bounds_equal_to_control": bool(all(reuse.values())),
            "bounds_reuse_evidence": reuse,
            "bounds_identity": bounds_identity,
            **metrics,
        }
        print(json.dumps({
            "event": "s1_zip_candidate_finalized", **results[name],
        }), flush=True)

    diagnostic = point_bounds_diagnostic(results, "P0")
    diagnostic["evidence_class"] = EVIDENCE_CLASS
    report = {
        "evidence_class": EVIDENCE_CLASS,
        "protocol": (
            "Exact packaged A/B/C deployment numerics on the frozen 725-tail "
            "split. P0 absolute bounds are reused bitwise for P0/P1/P2. "
            "Each batch is scored by exact sufficient-statistic accumulators "
            "and then released; no full point, bounds, or target array is "
            "materialized. This is post-selection and is not an independent "
            "OOF estimate."),
        "memory_strategy": {
            "tail_traversals": 1,
            "packaged_predict_calls": (
                (len(dataset) + args.batch_size - 1) // args.batch_size),
            "formal_batch_size": 16,
            "full_arrays_materialized": False,
            "retained_after_each_batch": (
                "per-sample error vectors, float64 SPS sums/counts, hashes"),
        },
        "time_measured": False,
        "training_performed": False,
        "tail_windows": len(dataset),
        "fold_counts": list(S1_FOLD_COUNTS),
        "fold_manifest": fold_manifest,
        "archive": {**archive_identity, "blend_config": blend_config},
        "split_manifest": split_identity,
        "real_stats": {
            **stats_identity,
            "role": "reference_identity_unused_by_packaged_inference",
        },
        "official_scorer": scorer_identity,
        "runner": file_identity(Path(__file__)),
        "bounds_source": "submission.predict P0 absolute arrays",
        "bounds_identity": bounds_identity,
        "target_identity": target_identity,
        "p0_custom_equals_submission_predict_bitwise": True,
        "point_proxy_definition": (
            "0.5*rel_l2_score + 0.3*tke_score + 0.2*mvpe_score"),
        "results": results,
        "decision": diagnostic,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "s1_zip_complete",
        "decision": diagnostic["decision"],
        "selected_point_teacher": diagnostic["selected_point_teacher"],
    }), flush=True)


if __name__ == "__main__":
    main()
