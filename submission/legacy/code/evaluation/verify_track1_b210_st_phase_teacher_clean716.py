#!/usr/bin/env python3
"""Strict clean716 archive-to-archive verification for the phase teacher.

Only rows and validation starts in the supplied frozen manifest are opened.
Each ZIP is imported independently and its public ``predict`` function is run
on every batch.  Metrics are accumulated without prediction caches, while
ordered input/target/output/bound byte streams are SHA-bound.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import reproduce_finetune as data_lib  # noqa: E402


def _load_official():
    path = REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"
    spec = importlib.util.spec_from_file_location("strict_clean716_scoring", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


OFFICIAL = _load_official()
EXPECTED_TRAJECTORIES = 81
EXPECTED_WINDOWS = 716
FINAL_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_score": 0.10,
    "mvpe_score": 0.09,
    "time_score": 0.10,
    "sps_score": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_frozen_clean716(real_dir: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or len(manifest) != EXPECTED_TRAJECTORIES:
        raise RuntimeError("frozen manifest must contain exactly 81 trajectories")
    trajectories = []
    windows = []
    opened = []
    aoas = []
    for trajectory_index, row in enumerate(manifest):
        if not isinstance(row, dict):
            raise TypeError("manifest rows must be mappings")
        path = real_dir / str(row["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            u = np.asarray(handle["u"][:, ::2, ::2], dtype=np.float32)
            v = np.asarray(handle["v"][:, ::2, ::2], dtype=np.float32)
            re_value = int(np.asarray(handle["re"]))
            aoa_value = int(np.asarray(handle["aoa"]))
        if (
            u.shape != v.shape
            or int(u.shape[0]) != int(row["frames"])
            or re_value != int(row["re"])
            or aoa_value != int(row["aoa"])
        ):
            raise RuntimeError(f"H5/manifest contract changed: {path.name}")
        trajectories.append(
            torch.from_numpy(np.ascontiguousarray(np.stack((u, v), axis=-1)))
        )
        opened.append(path.name)
        for start in row["val_starts"]:
            start = int(start)
            if start < 0 or start + 40 > u.shape[0]:
                raise RuntimeError(f"invalid validation start in {path.name}")
            windows.append(data_lib.Window(trajectory_index, start))
            aoas.append(aoa_value)
    if len(windows) != EXPECTED_WINDOWS or len(set(opened)) != EXPECTED_TRAJECTORIES:
        raise RuntimeError("frozen panel is not exactly clean716/81")
    dataset = data_lib.WindowDataset(trajectories, windows)
    return dataset, manifest, opened, aoas


def safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe ZIP member: {name}")
        archive.extractall(destination)


def _purge_submission_modules() -> None:
    for name in tuple(sys.modules):
        if name == "models" or name.startswith("models.") or name.startswith(
            "strict_archive_submission_"
        ):
            sys.modules.pop(name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_submission(package_dir: Path, label: str):
    _purge_submission_modules()
    sys.path.insert(0, str(package_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"strict_archive_submission_{label}", package_dir / "submission.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(package_dir))
    return module


def normalized_result(result):
    if isinstance(result, dict):
        return result["prediction"], result.get("lower"), result.get("upper")
    return result, None, None


def evaluate_archive(
    archive: Path,
    label: str,
    loader: DataLoader,
    temp_root: Path,
) -> dict:
    digests = {
        name: hashlib.sha256()
        for name in ("input", "target", "prediction", "lower", "upper")
    }
    samples = scored_elements = covered = 0
    rel_sum = tke_sum = mvpe_sum = 0.0
    sps_numerator = width_sum = 0.0
    durations = []
    with tempfile.TemporaryDirectory(prefix=f"strict-{label}-", dir=temp_root) as tmp:
        package_dir = Path(tmp)
        safe_extract(archive, package_dir)
        submission = load_submission(package_dir, label)
        for history, target in loader:
            input_array = history.numpy().astype(np.float32, copy=False)
            target_array = target.numpy().astype(np.float32, copy=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = submission.predict(input_array, metadata={})
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            prediction, lower, upper = normalized_result(result)
            prediction = np.asarray(prediction, dtype=np.float32)
            lower = np.asarray(lower, dtype=np.float32)
            upper = np.asarray(upper, dtype=np.float32)
            OFFICIAL.validate_shapes(prediction, target_array)
            if (
                lower.shape != prediction.shape
                or upper.shape != prediction.shape
                or not np.all(np.isfinite(prediction))
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower > upper)
            ):
                raise RuntimeError(f"invalid output from {label} archive")
            for name, value in (
                ("input", input_array),
                ("target", target_array),
                ("prediction", prediction),
                ("lower", lower),
                ("upper", upper),
            ):
                digests[name].update(np.ascontiguousarray(value).tobytes())

            channels = OFFICIAL.measured_channels(target_array)
            rel_sum += float(
                np.sum(
                    OFFICIAL.rel_l2_per_sample(prediction, target_array, channels),
                    dtype=np.float64,
                )
            )
            tke_sum += float(
                np.sum(
                    OFFICIAL.tke_rel_l2_per_sample(
                        prediction, target_array, channels
                    ),
                    dtype=np.float64,
                )
            )
            mvpe_sum += float(
                np.sum(
                    OFFICIAL.mvpe_rel_l2_per_sample(prediction, target_array),
                    dtype=np.float64,
                )
            )
            sps_value, coverage = OFFICIAL.aggregate_sps(
                prediction, target_array, channels, lower=lower, upper=upper
            )
            scored = target_array[..., :channels] != 0.0
            count = int(np.count_nonzero(scored))
            scored_elements += count
            sps_numerator += float(sps_value) * count
            covered += int(round(float(coverage) * count))
            width_sum += float(
                np.sum(
                    (upper[..., :channels] - lower[..., :channels])[scored],
                    dtype=np.float64,
                )
            )
            samples += int(prediction.shape[0])
        del submission
    _purge_submission_modules()
    if samples != EXPECTED_WINDOWS:
        raise RuntimeError(f"archive evaluated {samples}, expected 716")
    mean_seconds = float(sum(durations) / samples)
    metrics = {
        "rel_l2_score": OFFICIAL.score_error(rel_sum / samples),
        "tke_score": OFFICIAL.score_error(tke_sum / samples),
        "mvpe_score": OFFICIAL.score_error(mvpe_sum / samples),
        "time_score": OFFICIAL.score_time(mean_seconds),
        "sps_score": OFFICIAL.score_sps(sps_numerator / scored_elements),
    }
    return {
        "archive": str(archive.resolve()),
        "archive_sha256": file_sha256(archive),
        "samples": samples,
        "predict_calls": len(durations),
        "predict_seconds_total": float(sum(durations)),
        "predict_seconds_per_sample": mean_seconds,
        "metrics": metrics,
        "weighted_final_local": float(
            sum(FINAL_WEIGHTS[key] * metrics[key] for key in FINAL_WEIGHTS)
        ),
        "coverage": covered / scored_elements,
        "mean_width": width_sum / scored_elements,
        "ordered_sha256": {name: digest.hexdigest() for name, digest in digests.items()},
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    dataset, manifest, opened, aoas = load_frozen_clean716(
        args.real_dir.resolve(), args.manifest.resolve()
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    baseline = evaluate_archive(
        args.baseline_archive.resolve(), "baseline", loader, args.temp_root.resolve()
    )
    candidate = evaluate_archive(
        args.candidate_archive.resolve(), "candidate", loader, args.temp_root.resolve()
    )
    for name in ("input", "target"):
        if baseline["ordered_sha256"][name] != candidate["ordered_sha256"][name]:
            raise RuntimeError(f"ordered {name} changed between archive replays")
    bounds_bitwise = all(
        baseline["ordered_sha256"][name] == candidate["ordered_sha256"][name]
        for name in ("lower", "upper")
    )
    if not bounds_bitwise:
        raise RuntimeError("candidate did not preserve B210 bounds bitwise")
    metric_delta = {
        key: candidate["metrics"][key] - baseline["metrics"][key]
        for key in FINAL_WEIGHTS
    }
    report = {
        "schema": "track1_b210_st_phase_teacher_clean716_verify_v1",
        "evidence_class": "posthoc_clean716_transfer_probe_not_unbiased_selection",
        "submission_authorized": False,
        "panel": {
            "manifest": str(args.manifest.resolve()),
            "manifest_file_sha256": file_sha256(args.manifest.resolve()),
            "manifest_canonical_sha256": canonical_sha256(manifest),
            "trajectories": len(opened),
            "windows": len(dataset),
            "opened_files_sha256": canonical_sha256(opened),
            "excluded_7575_0": "7575_0.h5" not in opened,
            "aoa_counts": {
                str(aoa): int(sum(value == aoa for value in aoas))
                for aoa in sorted(set(aoas))
            },
            "ordered_input_sha256": baseline["ordered_sha256"]["input"],
            "ordered_target_sha256": baseline["ordered_sha256"]["target"],
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "metric_delta_candidate_minus_baseline": metric_delta,
            "weighted_final_delta_local": (
                candidate["weighted_final_local"] - baseline["weighted_final_local"]
            ),
            "lower_upper_bitwise_passthrough": bounds_bitwise,
            "lower_sha256": baseline["ordered_sha256"]["lower"],
            "upper_sha256": baseline["ordered_sha256"]["upper"],
        },
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
