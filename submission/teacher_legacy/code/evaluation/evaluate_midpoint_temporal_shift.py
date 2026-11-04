#!/usr/bin/env python3
"""Evaluate a zero-parameter temporal-mean point shift from fixed UQ bounds.

This archive-level run is a descriptive preflight only: all-tail interval heads
may have seen the evaluation targets.  A GO decision requires fold-held-out
bounds evaluated by the same scorer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from evaluate_track1_submission_archive import (  # noqa: E402
    OFFICIAL,
    load_submission,
    normalized_result,
    safe_extract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temp-root", type=Path, default=Path("runs/tmp"))
    return parser.parse_args()


def midpoint_temporal_shift(
        prediction: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        channels: int,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Shift measured point channels by the bounds' time-mean midpoint bias."""
    candidate = prediction.copy()
    midpoint_bias = (
        (lower[..., :channels] - prediction[..., :channels])
        + (upper[..., :channels] - prediction[..., :channels])
    ).mean(axis=1, keepdims=True, dtype=np.float32) * np.float32(0.5)
    candidate[..., :channels] += midpoint_bias
    return candidate, midpoint_bias


def score_bundle(
        prediction: np.ndarray,
        target: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        channels: int,
        ) -> dict[str, float]:
    rel_error = float(np.mean(
        OFFICIAL.rel_l2_per_sample(prediction, target, channels)))
    tke_error = float(np.mean(
        OFFICIAL.tke_rel_l2_per_sample(prediction, target, channels)))
    mvpe_error = float(np.mean(
        OFFICIAL.mvpe_rel_l2_per_sample(prediction, target)))
    sps_value, coverage = OFFICIAL.aggregate_sps(
        prediction, target, channels, lower=lower, upper=upper)
    scored = target[..., :channels] != 0.0
    result = {
        "rel_l2_score": OFFICIAL.score_error(rel_error),
        "tke_score": OFFICIAL.score_error(tke_error),
        "mvpe_score": OFFICIAL.score_error(mvpe_error),
        "sps_score": OFFICIAL.score_sps(sps_value),
        "coverage": float(coverage),
        "mean_width": float(np.mean(
            (upper[..., :channels] - lower[..., :channels])[scored])),
    }
    result["point_score"] = (
        0.5 * result["rel_l2_score"]
        + 0.3 * result["tke_score"]
        + 0.2 * result["mvpe_score"])
    return result


def main() -> None:
    args = parse_args()
    if args.eval_batch_size <= 0:
        raise ValueError("eval batch size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)

    data_args = SimpleNamespace(
        real_dir=args.real_dir, max_files=0, window_stride=20,
        val_fraction=0.2, batch_size=8,
        eval_batch_size=args.eval_batch_size, num_workers=0,
        real_stats=args.real_stats, output_dir=args.output.parent, seed=0)
    (_train_loader, val_loader, _normalizer, train_windows,
     val_windows, trajectories) = common.prepare_real(data_args, device)

    predictions = []
    lower_bounds = []
    upper_bounds = []
    targets = []
    durations = []
    archive = args.archive.resolve()
    with tempfile.TemporaryDirectory(
            prefix="midpoint-shift-", dir=args.temp_root) as temporary:
        package_dir = Path(temporary)
        safe_extract(archive, package_dir)
        submission = load_submission(package_dir)
        for history, target in val_loader:
            history_array = history.cpu().numpy()
            started = time.perf_counter()
            result = submission.predict(history_array, metadata={})
            durations.append(time.perf_counter() - started)
            prediction, lower, upper = normalized_result(result)
            prediction = np.asarray(prediction, dtype=np.float32)
            lower = np.asarray(lower, dtype=np.float32)
            upper = np.asarray(upper, dtype=np.float32)
            target_array = target.cpu().numpy().astype(np.float32, copy=False)
            OFFICIAL.validate_shapes(prediction, target_array)
            if (lower.shape != prediction.shape or upper.shape != prediction.shape
                    or not np.all(np.isfinite(prediction))
                    or not np.all(np.isfinite(lower))
                    or not np.all(np.isfinite(upper))
                    or np.any(lower > upper)):
                raise ValueError("archive returned invalid point or bounds")
            predictions.append(prediction)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            targets.append(target_array)

    prediction = np.concatenate(predictions)
    lower = np.concatenate(lower_bounds)
    upper = np.concatenate(upper_bounds)
    target = np.concatenate(targets)
    channels = OFFICIAL.measured_channels(target)
    candidate, bias = midpoint_temporal_shift(
        prediction, lower, upper, channels)
    baseline = score_bundle(prediction, target, lower, upper, channels)
    shifted = score_bundle(candidate, target, lower, upper, channels)
    delta = {key: shifted[key] - baseline[key] for key in baseline}

    invariants = {
        "unmeasured_channels_bitwise_equal": bool(np.array_equal(
            candidate[..., channels:], prediction[..., channels:])),
        "tke_score_abs_delta": abs(delta["tke_score"]),
        "coverage_abs_delta": abs(delta["coverage"]),
        "mean_width_abs_delta": abs(delta["mean_width"]),
        "bounds_unchanged_by_construction": True,
    }
    if (not invariants["unmeasured_channels_bitwise_equal"]
            or invariants["tke_score_abs_delta"] > 1e-5
            or invariants["coverage_abs_delta"] != 0.0
            or invariants["mean_width_abs_delta"] != 0.0):
        raise AssertionError(f"midpoint-shift invariant failed: {invariants}")

    report = {
        "method": "alpha1_temporal_mean_interval_midpoint_point_shift",
        "evidence_scope": "all-tail archive preflight; STOP-capable, not GO-capable",
        "archive": str(archive),
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "predict_calls": len(durations),
        "predict_seconds_total": float(sum(durations)),
        "baseline": baseline,
        "shifted": shifted,
        "delta": delta,
        "invariants": invariants,
        "bias_rms": float(np.sqrt(np.mean(bias * bias))),
        "bias_abs_max": float(np.max(np.abs(bias))),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
