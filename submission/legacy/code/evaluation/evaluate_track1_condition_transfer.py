#!/usr/bin/env python3
"""Audit Track 1 models under trajectory-level condition partitions.

The command has two deliberately separate responsibilities:

1. Build target-free, whole-trajectory partitions for future training:
   leave-one-AoA-out (LOAO), leave-one-Re-bin-out (LORO), and joint
   condition folds.  Held trajectories never appear in the corresponding fit
   set.  Their prefix windows can therefore be used for genuine condition-held
   validation without touching the temporal-tail labels.
2. Re-score already trained submission archives by AoA/Re groups on the clean
   temporal tail.  This is only a condition-stratified transfer audit because
   those archived models saw prefixes from the same trajectories during
   training; the report never labels it condition-held evidence.

Only per-window sufficient statistics are retained.  Raw predictions, targets,
and competition data are not written to the report or Git worktree.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


import evaluate_track1_submission_archive as archive_eval


OFFICIAL = archive_eval.OFFICIAL
SCHEMA = "track1_condition_transfer_audit_v1"
PARTITION_SCHEMA = "track1_condition_partitions_v1"
RE_BINS = 5
JOINT_FOLDS = 5
FINAL_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_score": 0.10,
    "mvpe_score": 0.09,
    "time_score": 0.10,
    "sps_score": 0.25,
}
NON_TIME_WEIGHT = sum(
    weight for name, weight in FINAL_WEIGHTS.items() if name != "time_score")
POINT_WEIGHT = sum(
    FINAL_WEIGHTS[name]
    for name in ("rel_l2_score", "tke_score", "mvpe_score"))
# Frozen before archive evaluation.  The trajectory p10 term prevents a large
# easy-condition majority from hiding a small collection of failed regimes.
TRANSFER_PROXY_MACRO_WEIGHT = 0.80
TRANSFER_PROXY_P10_WEIGHT = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("archive must be NAME=/path/to/file.zip")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
        raise argparse.ArgumentTypeError(
            "archive NAME may contain only letters, digits, '_' and '-'")
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", action="append", type=parse_named_path, default=[],
        metavar="NAME=ZIP", help="submission archive to audit; repeatable")
    parser.add_argument(
        "--official-results", type=Path,
        help="optional JSON mapping archive names to official subscores")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temp-root", type=Path, default=Path("runs/tmp"))
    return parser.parse_args()


def normalized_manifest(raw: object) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("split manifest must be a non-empty list")
    rows = []
    seen = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("split manifest rows must be mappings")
        filename = str(item["file"])
        if not filename or Path(filename).name != filename or filename in seen:
            raise ValueError(f"unsafe or duplicate trajectory name: {filename}")
        seen.add(filename)
        train_starts = [int(value) for value in item["train_starts"]]
        val_starts = [int(value) for value in item["val_starts"]]
        if (not train_starts or not val_starts
                or train_starts != sorted(set(train_starts))
                or val_starts != sorted(set(val_starts))):
            raise ValueError(f"invalid window starts for {filename}")
        rows.append({
            "file": filename,
            "frames": int(item["frames"]),
            "re": int(item["re"]),
            "aoa": int(item["aoa"]),
            "train_starts": train_starts,
            "val_starts": val_starts,
            "val_boundary": int(item["val_boundary"]),
        })
    return rows


def rank_bins(values: Iterable[int], bins: int = RE_BINS) -> dict[int, int]:
    unique = sorted({int(value) for value in values})
    if len(unique) < bins:
        raise ValueError(f"need at least {bins} unique Reynolds numbers")
    groups = np.array_split(np.asarray(unique, dtype=np.int64), bins)
    return {
        int(value): index
        for index, group in enumerate(groups)
        for value in group.tolist()
    }


def condition_labels(rows: Sequence[Mapping[str, object]]) -> tuple[list[dict], dict]:
    re_bin = rank_bins(int(row["re"]) for row in rows)
    re_rank = {
        value: index
        for index, value in enumerate(sorted(re_bin))
    }
    aoa_rank = {
        value: index
        for index, value in enumerate(sorted({int(row["aoa"]) for row in rows}))
    }
    labels = []
    assignments = []
    for trajectory, row in enumerate(rows):
        re_value = int(row["re"])
        aoa_value = int(row["aoa"])
        bin_value = re_bin[re_value]
        joint_fold = (
            re_rank[re_value] + 2 * aoa_rank[aoa_value]) % JOINT_FOLDS
        assignment = {
            "trajectory": trajectory,
            "file": str(row["file"]),
            "re": re_value,
            "aoa": aoa_value,
            "re_bin": bin_value,
            "aoa_re_bin": f"aoa{aoa_value}_rebin{bin_value}",
            "joint_fold": joint_fold,
            "train_windows": len(row["train_starts"]),
            "tail_windows": len(row["val_starts"]),
        }
        assignments.append(assignment)
        labels.extend([assignment] * len(row["val_starts"]))
    return labels, {
        "re_bin_count": RE_BINS,
        "joint_fold_count": JOINT_FOLDS,
        "re_bin_assignment": {str(key): value for key, value in re_bin.items()},
        "trajectory_assignments": assignments,
    }


def fold_record(rows: Sequence[Mapping[str, object]], held: set[str]) -> dict:
    fit = [row for row in rows if str(row["file"]) not in held]
    validation = [row for row in rows if str(row["file"]) in held]
    fit_names = {str(row["file"]) for row in fit}
    held_names = {str(row["file"]) for row in validation}
    if not fit_names or not held_names or fit_names & held_names:
        raise RuntimeError("condition partition is empty or leaks trajectories")
    return {
        "fit_trajectories": len(fit),
        "held_trajectories": len(validation),
        "fit_prefix_windows": sum(len(row["train_starts"]) for row in fit),
        "held_prefix_windows": sum(
            len(row["train_starts"]) for row in validation),
        "held_tail_windows": sum(len(row["val_starts"]) for row in validation),
        "fit_files": sorted(fit_names),
        "held_files": sorted(held_names),
        "trajectory_overlap": [],
    }


def build_partitions(rows: Sequence[Mapping[str, object]], assignments: Sequence[Mapping[str, object]]) -> dict:
    by_name = {str(row["file"]): row for row in rows}
    if set(by_name) != {str(row["file"]) for row in assignments}:
        raise RuntimeError("condition assignments do not cover the manifest")
    values = {
        "loao": sorted({int(row["aoa"]) for row in assignments}),
        "loro_re_bin": list(range(RE_BINS)),
        "joint_condition_cv": list(range(JOINT_FOLDS)),
    }
    field = {
        "loao": "aoa",
        "loro_re_bin": "re_bin",
        "joint_condition_cv": "joint_fold",
    }
    schemes = {}
    for scheme, fold_values in values.items():
        folds = []
        held_union = set()
        for value in fold_values:
            held = {
                str(row["file"]) for row in assignments
                if int(row[field[scheme]]) == value
            }
            held_union.update(held)
            folds.append({
                "held_value": value,
                **fold_record(rows, held),
            })
        counts = {
            name: sum(name in fold["held_files"] for fold in folds)
            for name in by_name
        }
        if held_union != set(by_name) or set(counts.values()) != {1}:
            raise RuntimeError(f"{scheme} does not hold every trajectory once")
        schemes[scheme] = {
            "field": field[scheme],
            "folds": folds,
            "every_trajectory_held_exactly_once": True,
        }
    result = {
        "schema": PARTITION_SCHEMA,
        "evidence_contract": {
            "fit_uses_only_fit_files_train_starts": True,
            "condition_held_selection_uses_held_files_train_starts": True,
            "tail_labels_reserved_for_final_replay": True,
            "whole_trajectory_partition": True,
        },
        "schemes": schemes,
    }
    result["sha256"] = canonical_sha256(result)
    return result


def validate_prediction_bundle(
        prediction: np.ndarray, lower: np.ndarray, upper: np.ndarray,
        target: np.ndarray) -> int:
    OFFICIAL.validate_shapes(prediction, target)
    if lower.shape != prediction.shape or upper.shape != prediction.shape:
        raise ValueError("archive bounds do not match prediction shape")
    if (not np.all(np.isfinite(prediction))
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)):
        raise FloatingPointError("archive returned invalid prediction bundle")
    return OFFICIAL.measured_channels(target)


def sufficient_statistics(
        prediction: np.ndarray, lower: np.ndarray, upper: np.ndarray,
        target: np.ndarray) -> dict[str, np.ndarray]:
    channels = validate_prediction_bundle(prediction, lower, upper, target)
    rel = OFFICIAL.rel_l2_per_sample(prediction, target, channels).astype(
        np.float64)
    tke = OFFICIAL.tke_rel_l2_per_sample(prediction, target, channels).astype(
        np.float64)
    mvpe = OFFICIAL.mvpe_rel_l2_per_sample(prediction, target).astype(
        np.float64)
    p = prediction[..., :channels]
    t = target[..., :channels]
    lo = lower[..., :channels]
    hi = upper[..., :channels]
    scored = t != 0.0
    inside = (t >= lo) & (t <= hi)
    width = hi - lo
    penalty = np.exp(-width / OFFICIAL.SIGMA_GLOBAL)
    shaped = (
        0.5 * (1.0 - rel / (0.5 + rel))
        + 0.3 * (1.0 - tke / (0.5 + tke))
        + 0.2 * (1.0 - mvpe / (0.5 + mvpe)))
    axes = tuple(range(1, prediction.ndim))
    utility_sum = np.sum(
        shaped.reshape((-1,) + (1,) * (p.ndim - 1)) * penalty * inside,
        axis=axes, where=scored, dtype=np.float64)
    return {
        "rel_error": rel,
        "tke_error": tke,
        "mvpe_error": mvpe,
        "sps_utility_sum": utility_sum,
        "scored_count": np.sum(scored, axis=axes, dtype=np.int64),
        "inside_count": np.sum(inside & scored, axis=axes, dtype=np.int64),
        "width_sum": np.sum(width, axis=axes, where=scored, dtype=np.float64),
    }


def concatenate_statistics(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("cannot concatenate empty statistics")
    keys = set(parts[0])
    if any(set(part) != keys for part in parts):
        raise RuntimeError("statistic keys changed between batches")
    return {key: np.concatenate([part[key] for part in parts]) for key in keys}


def score_statistics(stats: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, float]:
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("score group must contain one or more sample indices")
    rel = OFFICIAL.score_error(float(np.mean(stats["rel_error"][indices])))
    tke = OFFICIAL.score_error(float(np.mean(stats["tke_error"][indices])))
    mvpe = OFFICIAL.score_error(float(np.mean(stats["mvpe_error"][indices])))
    scored = int(np.sum(stats["scored_count"][indices], dtype=np.int64))
    if scored <= 0:
        raise ValueError("score group has no measured elements")
    sps = 100.0 * float(
        np.sum(stats["sps_utility_sum"][indices], dtype=np.float64) / scored)
    coverage = float(
        np.sum(stats["inside_count"][indices], dtype=np.int64) / scored)
    mean_width = float(
        np.sum(stats["width_sum"][indices], dtype=np.float64) / scored)
    point_contribution = 0.46 * rel + 0.10 * tke + 0.09 * mvpe
    non_time_contribution = point_contribution + 0.25 * sps
    return {
        "windows": int(len(indices)),
        "rel_l2_score": rel,
        "tke_score": tke,
        "mvpe_score": mvpe,
        "sps_score": sps,
        "coverage": coverage,
        "mean_width": mean_width,
        "point_score_46_10_9": point_contribution / POINT_WEIGHT,
        "point_final_contribution": point_contribution,
        "non_time_final_contribution": non_time_contribution,
        "non_time_score_normalized": non_time_contribution / NON_TIME_WEIGHT,
    }


def grouped_scores(
        stats: Mapping[str, np.ndarray], labels: Sequence[Mapping[str, object]],
        key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(str(label[key]), []).append(index)
    return {
        value: score_statistics(stats, np.asarray(indices, dtype=np.int64))
        for value, indices in sorted(groups.items())
    }


def macro_summary(groups: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    values = np.asarray([
        float(group["non_time_final_contribution"])
        for group in groups.values()
    ], dtype=np.float64)
    return {
        "groups": len(values),
        "macro_non_time_final_contribution": float(np.mean(values)),
        "worst_non_time_final_contribution": float(np.min(values)),
        "p10_non_time_final_contribution": float(np.quantile(
            values, 0.10, method="linear")),
        "spread_non_time_final_contribution": float(np.max(values) - np.min(values)),
    }


def summarize_transfer(
        stats: Mapping[str, np.ndarray], labels: Sequence[Mapping[str, object]]) -> dict:
    if len(labels) != len(stats["rel_error"]):
        raise RuntimeError("condition labels do not align with predictions")
    all_indices = np.arange(len(labels), dtype=np.int64)
    grouped = {
        key: grouped_scores(stats, labels, key)
        for key in (
            "aoa", "re_bin", "aoa_re_bin", "joint_fold", "file")
    }
    summaries = {key: macro_summary(value) for key, value in grouped.items()}
    macro_cell = summaries["aoa_re_bin"][
        "macro_non_time_final_contribution"]
    trajectory_p10 = summaries["file"]["p10_non_time_final_contribution"]
    transfer_proxy = (
        TRANSFER_PROXY_MACRO_WEIGHT * macro_cell
        + TRANSFER_PROXY_P10_WEIGHT * trajectory_p10)
    return {
        "evidence_class": (
            "condition-stratified temporal-tail audit; archived models saw "
            "same-trajectory prefixes, so this is not condition-held evidence"),
        "pooled": score_statistics(stats, all_indices),
        "groups": grouped,
        "group_summaries": summaries,
        "condition_transfer_proxy": transfer_proxy,
        "condition_transfer_proxy_formula": (
            "0.80 * macro(AoA x Re-bin non-Time contribution) + "
            "0.20 * trajectory-p10 non-Time contribution"),
    }


def collect_archive(
        name: str, archive: Path, val_loader, labels: Sequence[Mapping[str, object]],
        temp_root: Path) -> dict:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    parts = []
    durations = []
    with tempfile.TemporaryDirectory(
            prefix=f"condition-audit-{name}-", dir=temp_root) as temporary:
        package_dir = Path(temporary)
        archive_eval.safe_extract(archive, package_dir)
        submission = archive_eval.load_submission(package_dir)
        for history, target in val_loader:
            history_array = history.numpy()
            started = time.perf_counter()
            result = submission.predict(history_array, metadata={})
            durations.append(time.perf_counter() - started)
            prediction, lower, upper = archive_eval.normalized_result(result)
            if lower is None or upper is None:
                raise ValueError(f"archive {name} does not return learned bounds")
            parts.append(sufficient_statistics(
                np.asarray(prediction, dtype=np.float32),
                np.asarray(lower, dtype=np.float32),
                np.asarray(upper, dtype=np.float32),
                target.numpy().astype(np.float32, copy=False)))
        del submission
    stats = concatenate_statistics(parts)
    report = summarize_transfer(stats, labels)
    report["archive"] = {
        "name": name,
        "path": str(archive),
        "sha256": sha256(archive),
        "bytes": archive.stat().st_size,
    }
    report["latency_diagnostic"] = {
        "predict_calls": len(durations),
        "predict_seconds_total": float(sum(durations)),
        "predict_seconds_per_window": float(sum(durations) / max(len(labels), 1)),
        "not_comparable_to_official_time": True,
    }
    del stats, parts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def validate_official_results(raw: object, names: set[str]) -> dict[str, dict]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("official results must be a JSON object")
    result = {}
    required = set(FINAL_WEIGHTS)
    for name, row in raw.items():
        if name not in names:
            raise ValueError(f"official result has no matching archive: {name}")
        if not isinstance(row, Mapping) or not required <= set(row):
            raise ValueError(f"official result {name} lacks the five subscores")
        values = {key: float(row[key]) for key in required}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"official result {name} is non-finite")
        non_time = sum(
            FINAL_WEIGHTS[key] * values[key]
            for key in required if key != "time_score")
        result[name] = {
            **values,
            "final_score": (
                None if row.get("final_score") is None
                else float(row["final_score"])),
            "non_time_final_contribution": non_time,
        }
    return result


def rank(values: Mapping[str, float]) -> list[str]:
    return [
        name for name, _value in sorted(
            values.items(), key=lambda item: (-item[1], item[0]))
    ]


def compare_rankings(models: Mapping[str, Mapping[str, object]], official: Mapping[str, Mapping[str, object]]) -> dict:
    common = sorted(set(models) & set(official))
    if len(common) < 2:
        return {
            "available": False,
            "reason": "need at least two archives with official subscores",
        }
    local_pooled = {
        name: float(models[name]["pooled"]["non_time_final_contribution"])
        for name in common
    }
    local_transfer = {
        name: float(models[name]["condition_transfer_proxy"])
        for name in common
    }
    official_non_time = {
        name: float(official[name]["non_time_final_contribution"])
        for name in common
    }
    official_rank = rank(official_non_time)
    pooled_rank = rank(local_pooled)
    transfer_rank = rank(local_transfer)
    return {
        "available": True,
        "models": common,
        "official_non_time_rank": official_rank,
        "local_pooled_rank": pooled_rank,
        "condition_transfer_proxy_rank": transfer_rank,
        "local_pooled_exact_rank_match": pooled_rank == official_rank,
        "condition_transfer_exact_rank_match": transfer_rank == official_rank,
        "interpretation": (
            "An exact match is supportive only. A mismatch rejects this local "
            "archive proxy and requires genuine held-condition retraining; it "
            "must not be repaired by tuning the proxy on these official rows."),
    }


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.eval_batch_size <= 0:
        raise ValueError("eval batch size must be positive")
    names = [name for name, _path in args.archive]
    if len(names) != len(set(names)):
        raise ValueError("archive names must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=args.output.parent, seed=0)
    (train_loader, val_loader, _normalizer, train_windows,
     val_windows, trajectories) = archive_eval.common.prepare_real(
         data_args, device)
    del train_loader
    manifest_path = args.output.parent / "split_manifest.json"
    rows = normalized_manifest(json.loads(manifest_path.read_text(
        encoding="utf-8")))
    labels, condition_manifest = condition_labels(rows)
    if len(labels) != val_windows:
        raise RuntimeError("manifest tail windows do not match validation loader")
    partitions = build_partitions(
        rows, condition_manifest["trajectory_assignments"])
    models = {}
    for name, path in args.archive:
        models[name] = collect_archive(
            name, path, val_loader, labels, args.temp_root)
    raw_official = None
    if args.official_results is not None:
        raw_official = json.loads(args.official_results.read_text(
            encoding="utf-8"))
    official = validate_official_results(raw_official, set(names))
    result = {
        "schema": SCHEMA,
        "data": {
            "real_dir": str(args.real_dir.resolve()),
            "trajectories": trajectories,
            "train_prefix_windows": train_windows,
            "tail_windows": val_windows,
            "split_manifest_sha256": canonical_sha256(rows),
            "competition_arrays_written": False,
        },
        "condition_manifest": condition_manifest,
        "partitions": partitions,
        "models": models,
        "official_results": official,
        "ranking_check": compare_rankings(models, official),
        "evidence_boundary": {
            "partitions_are_valid_for_future_condition_held_training": True,
            "archive_scores_are_condition_stratified_not_condition_held": True,
            "tail_proxy_mismatch_must_not_be_repaired_with_official_tuning": True,
            "submission_authorized": False,
        },
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "event": "track1_condition_transfer_audit_complete",
        "models": sorted(result["models"]),
        "ranking_check": result["ranking_check"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
