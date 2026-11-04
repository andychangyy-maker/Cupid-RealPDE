#!/usr/bin/env python3
"""Cross-fit a small hard-SPS calibration on a packaged Track 1 model.

The point prediction is never changed.  Lower and upper bounds receive one
scale and physical-unit offset for each of four forecast-horizon blocks and
each measured channel.  Parameters are fitted with the starting-kit hard SPS
element objective and checked both by five condition folds and leave-one-AoA-
out evaluation before a deployable all-clean fit is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time

import numpy as np
import torch


import evaluate_track1_submission_archive as archive_eval


OFFICIAL = archive_eval.OFFICIAL
HORIZON_BLOCKS = ((0, 5), (5, 10), (10, 15), (15, 20))
MEASURED_CHANNELS = (0, 1)
PARAMETER_NAMES = (
    "lower_scale", "lower_offset", "upper_scale", "upper_offset")
IDENTITY_PARAMETERS = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
SCALE_BOUNDS = (0.80, 1.20)
OFFSET_BOUNDS = (-0.002, 0.002)
ROUND_SPECS = (
    {
        "round": 1,
        "scale_deltas": (-0.08, -0.04, 0.0, 0.04, 0.08),
        "offset_deltas": (-0.0008, -0.0004, 0.0, 0.0004, 0.0008),
    },
    {
        "round": 2,
        "scale_deltas": (-0.02, -0.01, 0.0, 0.01, 0.02),
        "offset_deltas": (-0.0002, -0.0001, 0.0, 0.0001, 0.0002),
    },
)
MAX_FIT_POINTS = 32768
FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def condition_labels(manifest: list[dict]) -> dict[str, np.ndarray]:
    re_values = sorted({int(row["re"]) for row in manifest})
    aoa_values = sorted({int(row["aoa"]) for row in manifest})
    re_rank = {value: index for index, value in enumerate(re_values)}
    aoa_rank = {value: index for index, value in enumerate(aoa_values)}
    values = {key: [] for key in ("fold", "re", "aoa", "trajectory")}
    assignments = []
    for trajectory, row in enumerate(manifest):
        re_value = int(row["re"])
        aoa_value = int(row["aoa"])
        fold = (re_rank[re_value] + 2 * aoa_rank[aoa_value]) % FOLDS
        count = len(row["val_starts"])
        values["fold"].extend([fold] * count)
        values["re"].extend([re_value] * count)
        values["aoa"].extend([aoa_value] * count)
        values["trajectory"].extend([trajectory] * count)
        assignments.append({
            "file": row["file"], "re": re_value, "aoa": aoa_value,
            "fold": fold, "windows": count,
        })
    result = {
        key: np.asarray(value, dtype=np.int32) for key, value in values.items()
    }
    result["assignments"] = assignments
    return result


def build_cache(args: argparse.Namespace, archive_sha: str) -> dict:
    cache = args.cache_dir
    if cache.exists() and any(cache.iterdir()):
        if not args.rebuild_cache:
            raise FileExistsError(f"cache directory is not empty: {cache}")
        raise RuntimeError("refusing to destructively rebuild a non-empty cache")
    cache.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=cache, seed=0)
    (_train_loader, val_loader, _normalizer, train_windows,
     val_windows, trajectories) = archive_eval.common.prepare_real(
         data_args, device)
    manifest = json.loads((cache / "split_manifest.json").read_text(
        encoding="utf-8"))
    labels = condition_labels(manifest)
    arrays = {key: [] for key in ("prediction", "lower", "upper", "target")}
    with tempfile.TemporaryDirectory(prefix="b210-calibration-") as temporary:
        package_dir = Path(temporary)
        archive_eval.safe_extract(args.archive.resolve(), package_dir)
        submission = archive_eval.load_submission(package_dir)
        for history, target in val_loader:
            result = submission.predict(history.numpy(), metadata={})
            prediction, lower, upper = archive_eval.normalized_result(result)
            if lower is None or upper is None:
                raise ValueError("archive does not return learned bounds")
            batch = {
                "prediction": np.asarray(prediction, dtype=np.float32)[..., :2],
                "lower": np.asarray(lower, dtype=np.float32)[..., :2],
                "upper": np.asarray(upper, dtype=np.float32)[..., :2],
                "target": target.numpy().astype(np.float32, copy=False)[..., :2],
            }
            OFFICIAL.validate_shapes(batch["prediction"], batch["target"])
            if (not all(np.all(np.isfinite(value)) for value in batch.values())
                    or np.any(batch["lower"] > batch["upper"])):
                raise FloatingPointError("archive cache contains invalid arrays")
            for key, value in batch.items():
                arrays[key].append(value)
    shape = None
    for key, parts in arrays.items():
        value = np.concatenate(parts)
        shape = value.shape
        np.save(cache / f"{key}.npy", value, allow_pickle=False)
    if shape is None or shape[0] != val_windows:
        raise RuntimeError("cache window count changed")
    for key in ("fold", "re", "aoa", "trajectory"):
        if len(labels[key]) != val_windows:
            raise RuntimeError(f"{key} labels do not cover the cache")
        np.save(cache / f"{key}.npy", labels[key], allow_pickle=False)
    metadata = {
        "schema": "b210_archive_calibration_cache_v1",
        "archive": {
            "path": str(args.archive.resolve()), "sha256": archive_sha,
            "bytes": args.archive.stat().st_size,
        },
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "shape": list(shape),
        "manifest_sha256": canonical_sha256(manifest),
        "fold_formula": "(rank(re)+2*rank(aoa)) % 5",
        "fold_assignments": labels["assignments"],
    }
    (cache / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return metadata


def load_cache(args: argparse.Namespace, archive_sha: str) -> tuple[dict, dict]:
    metadata_path = args.cache_dir / "metadata.json"
    if not metadata_path.exists():
        metadata = build_cache(args, archive_sha)
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("schema") != "b210_archive_calibration_cache_v1"
            or metadata.get("archive", {}).get("sha256") != archive_sha):
        raise RuntimeError("cache is not bound to the requested archive")
    arrays = {
        key: np.load(args.cache_dir / f"{key}.npy", mmap_mode="r")
        for key in (
            "prediction", "lower", "upper", "target",
            "fold", "re", "aoa", "trajectory")
    }
    expected = tuple(metadata["shape"])
    if any(tuple(arrays[key].shape) != expected
           for key in ("prediction", "lower", "upper", "target")):
        raise RuntimeError("cached field shapes changed")
    if any(len(arrays[key]) != expected[0]
           for key in ("fold", "re", "aoa", "trajectory")):
        raise RuntimeError("cached condition labels changed")
    return metadata, arrays


def official_sample_weights(prediction: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
    dm = OFFICIAL.rel_l2_per_sample(prediction, target, 2)
    tke = OFFICIAL.tke_rel_l2_per_sample(prediction, target, 2)
    mvpe = OFFICIAL.mvpe_rel_l2_per_sample(prediction, target)
    weights = (
        0.5 * (1.0 - dm / (0.5 + dm))
        + 0.3 * (1.0 - tke / (0.5 + tke))
        + 0.2 * (1.0 - mvpe / (0.5 + mvpe)))
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError("sample SPS weights are non-finite")
    return weights.astype(np.float64, copy=False)


def stratified_group_sample(
        prediction: np.ndarray, target: np.ndarray,
        lower: np.ndarray, upper: np.ndarray, sample_weights: np.ndarray,
        fit_mask: np.ndarray, start: int, stop: int, channel: int,
        max_points: int = MAX_FIT_POINTS) -> dict[str, np.ndarray]:
    sample_indices = np.flatnonzero(fit_mask)
    per_sample = max(1, max_points // max(len(sample_indices), 1))
    parts = {key: [] for key in ("point", "target", "lower", "upper", "weight")}
    for sample in sample_indices:
        values = {
            "point": prediction[sample, start:stop, :, :, channel].reshape(-1),
            "target": target[sample, start:stop, :, :, channel].reshape(-1),
            "lower": lower[sample, start:stop, :, :, channel].reshape(-1),
            "upper": upper[sample, start:stop, :, :, channel].reshape(-1),
        }
        valid = values["target"] != 0.0
        count = min(per_sample, int(np.count_nonzero(valid)))
        if count == 0:
            continue
        for key in values:
            values[key] = values[key][valid]
        selected = np.linspace(
            0, len(values["point"]) - 1, count, dtype=np.int64)
        for key in values:
            parts[key].append(values[key][selected])
        parts["weight"].append(np.full(
            count, sample_weights[sample], dtype=np.float64))
    if not parts["point"]:
        raise ValueError("calibration group has no measured elements")
    return {key: np.concatenate(value)[:max_points]
            for key, value in parts.items()}


def transform_sample_bounds(sample: dict[str, np.ndarray],
                            parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower_scale, lower_offset, upper_scale, upper_offset = parameters
    point = sample["point"]
    raw_lower = (
        point + np.float32(lower_scale) * (sample["lower"] - point)
        + np.float32(lower_offset))
    raw_upper = (
        point + np.float32(upper_scale) * (sample["upper"] - point)
        + np.float32(upper_offset))
    return np.minimum(raw_lower, raw_upper), np.maximum(raw_lower, raw_upper)


def direct_hard_sps_objective(sample: dict[str, np.ndarray],
                              parameters: np.ndarray) -> float:
    lower, upper = transform_sample_bounds(sample, parameters)
    inside = (sample["target"] >= lower) & (sample["target"] <= upper)
    element = sample["weight"] * np.exp(
        -(upper - lower) / OFFICIAL.SIGMA_GLOBAL)
    return float(np.sum(element, where=inside, dtype=np.float64) / len(element))


def coordinate_search(sample: dict[str, np.ndarray]) -> dict:
    parameters = IDENTITY_PARAMETERS.copy()
    identity = direct_hard_sps_objective(sample, parameters)
    score = identity
    for round_spec in ROUND_SPECS:
        for index, _name in enumerate(PARAMETER_NAMES):
            deltas = (round_spec["scale_deltas"] if index in (0, 2)
                      else round_spec["offset_deltas"])
            low, high = SCALE_BOUNDS if index in (0, 2) else OFFSET_BOUNDS
            anchor = float(parameters[index])
            trials = []
            for delta in deltas:
                candidate = parameters.copy()
                candidate[index] = np.clip(anchor + delta, low, high)
                trials.append((
                    direct_hard_sps_objective(sample, candidate),
                    -abs(delta), -float(candidate[index]), candidate))
            score, _distance, _value, parameters = max(
                trials, key=lambda row: row[:3])
    return {
        "parameters": {
            name: float(parameters[index])
            for index, name in enumerate(PARAMETER_NAMES)
        },
        "sample_points": int(len(sample["point"])),
        "identity_objective": identity,
        "selected_objective": score,
    }


def fit_groups(arrays: dict, sample_weights: np.ndarray,
               fit_mask: np.ndarray) -> list[dict]:
    rows = []
    for block, (start, stop) in enumerate(HORIZON_BLOCKS):
        for channel in MEASURED_CHANNELS:
            sample = stratified_group_sample(
                arrays["prediction"], arrays["target"],
                arrays["lower"], arrays["upper"], sample_weights,
                fit_mask, start, stop, channel)
            row = coordinate_search(sample)
            row.update({
                "block": block, "horizon_start": start,
                "horizon_stop": stop, "channel": channel,
            })
            rows.append(row)
    return rows


def allocate_bounds(arrays: dict) -> dict[str, np.ndarray]:
    return {
        "lower": np.array(arrays["lower"], dtype=np.float32, copy=True),
        "upper": np.array(arrays["upper"], dtype=np.float32, copy=True),
    }


def apply_groups(arrays: dict, groups: list[dict], apply_mask: np.ndarray,
                 output: dict[str, np.ndarray]) -> dict:
    reversed_count = 0
    elements = 0
    for group in groups:
        start, stop = group["horizon_start"], group["horizon_stop"]
        channel = group["channel"]
        values = group["parameters"]
        point = arrays["prediction"][apply_mask, start:stop, :, :, channel]
        raw_lower = point + np.float32(values["lower_scale"]) * (
            arrays["lower"][apply_mask, start:stop, :, :, channel] - point)
        raw_lower += np.float32(values["lower_offset"])
        raw_upper = point + np.float32(values["upper_scale"]) * (
            arrays["upper"][apply_mask, start:stop, :, :, channel] - point)
        raw_upper += np.float32(values["upper_offset"])
        reversed_count += int(np.count_nonzero(raw_lower > raw_upper))
        elements += raw_lower.size
        output["lower"][apply_mask, start:stop, :, :, channel] = np.minimum(
            raw_lower, raw_upper)
        output["upper"][apply_mask, start:stop, :, :, channel] = np.maximum(
            raw_lower, raw_upper)
    return {
        "reversed_before_projection": reversed_count,
        "elements": elements,
    }


def metric_row(arrays: dict, bounds: dict[str, np.ndarray],
               mask: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(arrays["prediction"][mask])
    target = np.asarray(arrays["target"][mask])
    lower = np.asarray(bounds["lower"][mask])
    upper = np.asarray(bounds["upper"][mask])
    value, coverage = OFFICIAL.aggregate_sps(
        prediction, target, 2, lower=lower, upper=upper)
    scored = target != 0.0
    return {
        "sps_score": OFFICIAL.score_sps(value),
        "coverage": coverage,
        "mean_width": float(np.mean((upper - lower)[scored])),
        "samples": int(np.count_nonzero(mask)),
    }


def crossfit(arrays: dict, sample_weights: np.ndarray,
             labels: np.ndarray, groups: list[int]) -> dict:
    output = allocate_bounds(arrays)
    rows = []
    for group in groups:
        held = labels == group
        fitted = fit_groups(arrays, sample_weights, ~held)
        projection = apply_groups(arrays, fitted, held, output)
        rows.append({"group": int(group), "parameters": fitted,
                     "projection": projection})
    baseline = {"lower": arrays["lower"], "upper": arrays["upper"]}
    full = np.ones(len(labels), dtype=bool)
    for row in rows:
        held = labels == row["group"]
        before = metric_row(arrays, baseline, held)
        after = metric_row(arrays, output, held)
        row.update({
            "baseline": before, "candidate": after,
            "sps_delta": after["sps_score"] - before["sps_score"],
        })
    before = metric_row(arrays, baseline, full)
    after = metric_row(arrays, output, full)
    return {
        "baseline": before,
        "candidate": after,
        "sps_delta": after["sps_score"] - before["sps_score"],
        "groups": rows,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    archive_sha = sha256(args.archive)
    metadata, arrays = load_cache(args, archive_sha)
    sample_weights = official_sample_weights(
        arrays["prediction"], arrays["target"])
    fold = crossfit(
        arrays, sample_weights, np.asarray(arrays["fold"]), list(range(FOLDS)))
    aoa_values = sorted(map(int, np.unique(arrays["aoa"])))
    loao = crossfit(
        arrays, sample_weights, np.asarray(arrays["aoa"]), aoa_values)
    full_mask = np.ones(len(arrays["fold"]), dtype=bool)
    full_groups = fit_groups(arrays, sample_weights, full_mask)
    full_bounds = allocate_bounds(arrays)
    projection = apply_groups(arrays, full_groups, full_mask, full_bounds)
    baseline_bounds = {"lower": arrays["lower"], "upper": arrays["upper"]}
    baseline = metric_row(arrays, baseline_bounds, full_mask)
    full = metric_row(arrays, full_bounds, full_mask)
    fold_deltas = [row["sps_delta"] for row in fold["groups"]]
    aoa_deltas = [row["sps_delta"] for row in loao["groups"]]
    conditions = {
        "condition5_delta_ge_0p03": fold["sps_delta"] >= 0.03,
        "positive_condition_folds_ge_4": sum(x > 0.0 for x in fold_deltas) >= 4,
        "worst_condition_fold_ge_minus_0p03": min(fold_deltas) >= -0.03,
        "loao_delta_ge_0p03": loao["sps_delta"] >= 0.03,
        "positive_aoas_ge_4": sum(x > 0.0 for x in aoa_deltas) >= 4,
        "worst_aoa_ge_minus_0p05": min(aoa_deltas) >= -0.05,
    }
    report = {
        "schema": "b210_archive_bound_calibration_v1",
        "archive": metadata["archive"],
        "cache": {
            "path": str(args.cache_dir.resolve()),
            "manifest_sha256": metadata["manifest_sha256"],
            "samples": metadata["val_windows"],
        },
        "method": {
            "formula": "side'=point+scale*(side-point)+offset",
            "horizon_blocks": HORIZON_BLOCKS,
            "channels": MEASURED_CHANNELS,
            "scale_bounds": SCALE_BOUNDS,
            "offset_bounds": OFFSET_BOUNDS,
            "max_fit_points_per_group": MAX_FIT_POINTS,
            "optimizer": "deterministic two-round coordinate search",
            "point_prediction_changed": False,
        },
        "baseline": baseline,
        "condition5_crossfit": fold,
        "leave_one_aoa_out": loao,
        "all_clean_fit": {
            "candidate": full,
            "sps_delta": full["sps_score"] - baseline["sps_score"],
            "parameters": full_groups,
            "projection": projection,
        },
        "promotion_gate": {
            "conditions": conditions,
            "pass": all(conditions.values()),
            "decision": (
                "PASS_B210_BOUND_CALIBRATION"
                if all(conditions.values())
                else "STOP_B210_BOUND_CALIBRATION"),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "b210_bound_calibration_complete",
        "decision": report["promotion_gate"]["decision"],
        "condition5_sps_delta": fold["sps_delta"],
        "loao_sps_delta": loao["sps_delta"],
        "all_clean_sps_delta": report["all_clean_fit"]["sps_delta"],
        "output": str(args.output.resolve()),
    }), flush=True)


if __name__ == "__main__":
    main()
