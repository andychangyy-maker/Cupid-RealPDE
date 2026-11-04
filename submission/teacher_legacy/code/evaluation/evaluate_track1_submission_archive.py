#!/usr/bin/env python3
"""Run a Track 1 ZIP itself on the standard temporal-tail validation split."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402


def load_official_scoring():
    path = REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"
    spec = importlib.util.spec_from_file_location(
        "realpde_track1_v9_archive_scoring", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import scoring from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL = load_official_scoring()


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
    parser.add_argument(
        "--interval-scales", type=float, nargs="*", default=(),
        help="optional positive scales around prediction for local SPS audit")
    return parser.parse_args()


def safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe ZIP member: {name}")
        archive.extractall(destination)


def load_submission(package_dir: Path):
    # train_hybrid_fno2d imports the repository's models package while preparing
    # data.  Remove only the modules shadowed by a submission archive so this
    # evaluation really executes the packaged source files.
    for name in ("models.ensemble_unet2d", "models.unet2d", "models"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(package_dir))
    spec = importlib.util.spec_from_file_location(
        "evaluated_submission", package_dir / "submission.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load packaged submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_result(result):
    if isinstance(result, dict):
        return result["prediction"], result.get("lower"), result.get("upper")
    return result, None, None


def main() -> None:
    args = parse_args()
    if args.eval_batch_size <= 0:
        raise ValueError("eval batch size must be positive")
    if any(not np.isfinite(value) or value <= 0
           for value in args.interval_scales):
        raise ValueError("interval scales must be finite and positive")
    device = torch.device(args.device)
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
            prefix="archive-eval-", dir=args.temp_root) as temporary:
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
            target_array = target.cpu().numpy().astype(np.float32, copy=False)
            OFFICIAL.validate_shapes(prediction, target_array)
            if not np.all(np.isfinite(prediction)):
                raise FloatingPointError("archive returned non-finite prediction")
            if lower is None or upper is None:
                raise ValueError("archive must return both lower and upper")
            lower = np.asarray(lower, dtype=np.float32)
            upper = np.asarray(upper, dtype=np.float32)
            if lower.shape != prediction.shape or upper.shape != prediction.shape:
                raise ValueError("archive bounds do not match prediction shape")
            if (not np.all(np.isfinite(lower))
                    or not np.all(np.isfinite(upper))
                    or np.any(lower > upper)):
                raise FloatingPointError("archive returned invalid bounds")
            predictions.append(prediction)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            targets.append(target_array)

    prediction = np.concatenate(predictions)
    lower = np.concatenate(lower_bounds)
    upper = np.concatenate(upper_bounds)
    target = np.concatenate(targets)
    channels = OFFICIAL.measured_channels(target)
    rel_error = float(np.mean(
        OFFICIAL.rel_l2_per_sample(prediction, target, channels)))
    tke_error = float(np.mean(
        OFFICIAL.tke_rel_l2_per_sample(prediction, target, channels)))
    mvpe_error = float(np.mean(
        OFFICIAL.mvpe_rel_l2_per_sample(prediction, target)))
    sps_value, coverage = OFFICIAL.aggregate_sps(
        prediction, target, channels, lower=lower, upper=upper)
    metrics = {
        "rel_l2_score": OFFICIAL.score_error(rel_error),
        "tke_score": OFFICIAL.score_error(tke_error),
        "mvpe_score": OFFICIAL.score_error(mvpe_error),
        "sps_score": OFFICIAL.score_sps(sps_value),
        "coverage": coverage,
        "mean_width": float(np.mean(
            (upper[..., :channels] - lower[..., :channels])[
                target[..., :channels] != 0.0])),
    }
    metrics["point_score"] = (
        0.5 * metrics["rel_l2_score"]
        + 0.3 * metrics["tke_score"]
        + 0.2 * metrics["mvpe_score"])
    scale_grid = []
    scored = target[..., :channels] != 0.0
    for scale in args.interval_scales:
        scaled_lower = prediction + scale * (lower - prediction)
        scaled_upper = prediction + scale * (upper - prediction)
        value, scaled_coverage = OFFICIAL.aggregate_sps(
            prediction, target, channels,
            lower=scaled_lower, upper=scaled_upper)
        scale_grid.append({
            "scale": float(scale),
            "sps_score": OFFICIAL.score_sps(value),
            "coverage": scaled_coverage,
            "mean_width": float(np.mean(
                (scaled_upper[..., :channels]
                 - scaled_lower[..., :channels])[scored])),
        })
    elapsed = float(sum(durations))
    seconds_per_window = elapsed / max(val_windows, 1)
    report = {
        "archive": str(archive),
        "train_windows": train_windows,
        "val_windows": val_windows,
        "trajectories": trajectories,
        "predict_calls": len(durations),
        "predict_seconds_total": elapsed,
        "predict_seconds_mean": float(np.mean(durations)),
        "predict_seconds_per_window": seconds_per_window,
        "local_proxy_time_score": OFFICIAL.score_time(seconds_per_window),
        "metrics": metrics,
        "interval_scale_grid": scale_grid,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
