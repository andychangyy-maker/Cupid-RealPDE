#!/usr/bin/env python3
"""Benchmark a Track 1 ZIP from cold import through warmed real inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--measured-calls", type=int, default=5)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe ZIP member: {name}")
        archive.extractall(destination)


def output_sha256(result) -> str:
    prediction = result["prediction"] if isinstance(result, dict) else result
    value = np.ascontiguousarray(np.asarray(prediction, dtype=np.float32))
    return hashlib.sha256(value.tobytes()).hexdigest()


def timed_predict(submission, value: np.ndarray) -> tuple[float, str]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = submission.predict(value, metadata={})
    torch.cuda.synchronize()
    return time.perf_counter() - started, output_sha256(result)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.batch_size <= 0 or args.measured_calls <= 0:
        raise ValueError("batch size and measured calls must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    archive = args.archive.resolve()
    process_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="archive-bench-", dir=args.temp_root) as tmp:
        package_dir = Path(tmp)
        safe_extract(archive, package_dir)
        sys.path.insert(0, str(package_dir))
        import_started = time.perf_counter()
        spec = importlib.util.spec_from_file_location(
            "benchmarked_track1_submission", package_dir / "submission.py"
        )
        submission = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(submission)
        torch.cuda.synchronize()
        import_seconds = time.perf_counter() - import_started

        generator = np.random.default_rng(210)
        value = generator.normal(
            loc=0.0,
            scale=0.05,
            size=(args.batch_size, 20, 32, 64, 3),
        ).astype(np.float32)
        value[..., 0] += 0.15
        value[..., 2] = 0.0
        first_seconds, first_hash = timed_predict(submission, value)
        cold_to_first_output = time.perf_counter() - process_started

        durations = []
        hashes = []
        for index in range(args.measured_calls):
            # A different history on every call proves predictions are not
            # served from an input/output cache while retaining equal shapes.
            changed = value.copy()
            changed[:, -1, ..., 0] += (index + 1) * 1.0e-3
            duration, digest = timed_predict(submission, changed)
            durations.append(duration)
            hashes.append(digest)
    result = {
        "schema": "track1_archive_cold_warm_benchmark_v1",
        "archive": str(archive),
        "archive_sha256": file_sha256(archive),
        "device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "module_import_including_builtin_prewarm_seconds": import_seconds,
        "first_explicit_call_seconds": first_seconds,
        "first_explicit_call_seconds_per_sample": first_seconds / args.batch_size,
        "cold_process_to_first_output_seconds": cold_to_first_output,
        "measured_calls": args.measured_calls,
        "warm_call_seconds": durations,
        "warm_seconds_per_sample_mean": float(np.mean(durations)) / args.batch_size,
        "warm_seconds_per_sample_median": float(np.median(durations)) / args.batch_size,
        "first_output_sha256": first_hash,
        "warm_output_sha256": hashes,
        "distinct_warm_outputs": len(set(hashes)),
        "prediction_cache_used": False,
    }
    if result["distinct_warm_outputs"] != args.measured_calls:
        raise RuntimeError("changed benchmark inputs did not produce distinct outputs")
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
