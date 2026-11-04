#!/usr/bin/env python3
"""Compare two Track 1 ZIPs in isolated directories and fresh processes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np


def load_input(path: Path, batch_size: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        u = np.asarray(handle["u"][:, ::2, ::2], dtype=np.float32)
        v = np.asarray(handle["v"][:, ::2, ::2], dtype=np.float32)
    starts = [20 * index for index in range(batch_size)]
    if starts[-1] + 20 > len(u):
        raise ValueError(f"{path} has too few frames for batch size {batch_size}")
    uv = np.stack((u, v), axis=-1)
    arrays = [uv[start:start + 20] for start in starts]
    uv_batch = np.stack(arrays)
    pressure = np.zeros((*uv_batch.shape[:-1], 1), dtype=np.float32)
    return np.concatenate((uv_batch, pressure), axis=-1)


def safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe ZIP member: {name}")
        archive.extractall(destination)


def normalize_result(result):
    if isinstance(result, dict):
        return result["prediction"], result.get("lower"), result.get("upper")
    return result, None, None


def worker(package_dir: Path, input_path: Path, output_path: Path,
           repeats: int) -> None:
    sys.path.insert(0, str(package_dir))
    spec = importlib.util.spec_from_file_location(
        "isolated_submission", package_dir / "submission.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = np.load(input_path)

    # Exercise both a singleton call and a larger call in the same process.
    normalize_result(module.predict(x[:1], metadata={}))
    timings = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = module.predict(x, metadata={})
        timings.append(time.perf_counter() - started)
    prediction, lower, upper = normalize_result(result)
    fields = {
        "prediction": np.asarray(prediction),
        "timings": np.asarray(timings, dtype=np.float64),
    }
    if lower is not None:
        fields["lower"] = np.asarray(lower)
    if upper is not None:
        fields["upper"] = np.asarray(upper)
    np.savez(output_path, **fields)


def describe(path: Path) -> dict:
    with np.load(path) as payload:
        result = {}
        for name in ("prediction", "lower", "upper"):
            if name not in payload:
                continue
            array = payload[name]
            result[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite": bool(np.isfinite(array).all()),
            }
        timings = payload["timings"]
        result["timing_seconds"] = {
            "values": timings.tolist(),
            "median": float(np.median(timings)),
            "minimum": float(np.min(timings)),
        }
        if "lower" in payload and "upper" in payload:
            result["bounds_ordered"] = bool(
                np.all(payload["lower"] <= payload["upper"]))
            result["bounds_contain_prediction"] = bool(
                np.all(payload["lower"] <= payload["prediction"])
                and np.all(payload["prediction"] <= payload["upper"]))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--example-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--temp-root", type=Path, default=Path("runs/tmp"))
    args = parser.parse_args()
    if args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("batch size and repeats must be positive")
    args.temp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
            prefix="submission-pair-", dir=args.temp_root) as temporary:
        root = Path(temporary)
        reference_dir = root / "reference"
        candidate_dir = root / "candidate"
        reference_dir.mkdir()
        candidate_dir.mkdir()
        safe_extract(args.reference, reference_dir)
        safe_extract(args.candidate, candidate_dir)
        input_path = root / "input.npy"
        np.save(input_path, load_input(args.example_h5, args.batch_size))
        outputs = {
            "reference": root / "reference_output.npz",
            "candidate": root / "candidate_output.npz",
        }
        for name, package_dir in (
                ("reference", reference_dir), ("candidate", candidate_dir)):
            subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "--worker",
                str(package_dir), str(input_path), str(outputs[name]),
                str(args.repeats),
            ], check=True)

        with np.load(outputs["reference"]) as reference, np.load(
                outputs["candidate"]) as candidate:
            difference = np.abs(
                reference["prediction"].astype(np.float64)
                - candidate["prediction"].astype(np.float64))
            comparison = {
                "prediction_exactly_equal": bool(np.array_equal(
                    reference["prediction"], candidate["prediction"])),
                "prediction_max_abs_difference": float(difference.max()),
            }
        report = {
            "reference_archive": str(args.reference),
            "candidate_archive": str(args.candidate),
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "reference": describe(outputs["reference"]),
            "candidate": describe(outputs["candidate"]),
            "comparison": comparison,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]),
               int(sys.argv[5]))
    else:
        main()
