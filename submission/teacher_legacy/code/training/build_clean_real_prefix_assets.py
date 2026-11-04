#!/usr/bin/env python3
"""Build clean81 train-prefix manifest and normalizer without reading tail arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch


EXCLUDED_BASENAME = "7575_0.h5"
FORMAL_COUNTS = {
    "trajectories": 81,
    "train_windows": 2544,
    "aoa_windows": {0: 502, 5: 433, 10: 576, 15: 544, 20: 489},
}
FORMAL_MANIFEST_SHA256 = (
    "d0f52e7ffdec0fad0cbd653ebb41a3d0d3c76dd2ee99a2b927cc9b0d14a76d09")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--exclude", default=EXCLUDED_BASENAME)
    parser.add_argument("--formal", action="store_true")
    return parser.parse_args()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_row(path: Path, stride: int, val_fraction: float) -> dict:
    if stride <= 0:
        raise ValueError("window stride must be positive")
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("validation fraction must lie in (0, 0.5)")
    with h5py.File(path, "r") as handle:
        if handle["u"].shape != handle["v"].shape:
            raise ValueError(f"u/v shapes differ in {path.name}")
        frames = int(handle["u"].shape[0])
        re_value = int(np.asarray(handle["re"]))
        aoa_value = int(np.asarray(handle["aoa"]))
    starts = list(range(0, frames - 40 + 1, stride))
    if len(starts) < 2:
        raise ValueError(f"trajectory {path.name} is too short")
    validation_count = max(1, math.ceil(len(starts) * val_fraction))
    val_boundary = starts[-validation_count]
    train_starts = [start for start in starts if start + 40 <= val_boundary]
    if not train_starts:
        raise ValueError(f"no training windows remain for {path.name}")
    return {
        "file": path.name,
        "frames": frames,
        "re": re_value,
        "aoa": aoa_value,
        "train_starts": train_starts,
        "val_boundary": val_boundary,
    }


def prefix_channel_moments(path: Path, boundary: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Read only [0:boundary] u/v array slices and return float64 moments."""
    if boundary <= 0:
        raise ValueError("training prefix boundary must be positive")
    with h5py.File(path, "r") as handle:
        u = np.asarray(handle["u"][:boundary, ::2, ::2], dtype=np.float32)
        v = np.asarray(handle["v"][:boundary, ::2, ::2], dtype=np.float32)
    if u.shape != v.shape or u.shape[0] != boundary:
        raise ValueError(f"prefix shape mismatch in {path.name}")
    values = np.stack((u, v), axis=-1).astype(np.float64, copy=False)
    count = int(np.prod(values.shape[:-1]))
    sums = values.sum(axis=tuple(range(values.ndim - 1)), dtype=np.float64)
    squares = np.square(values).sum(
        axis=tuple(range(values.ndim - 1)), dtype=np.float64)
    return sums, squares, count


def validate_formal(manifest: list[dict], manifest_sha256: str) -> None:
    aoa_windows = {
        aoa: sum(len(row["train_starts"]) for row in manifest
                 if row["aoa"] == aoa)
        for aoa in FORMAL_COUNTS["aoa_windows"]
    }
    checks = {
        "trajectories": len(manifest) == FORMAL_COUNTS["trajectories"],
        "train_windows": sum(len(row["train_starts"]) for row in manifest)
        == FORMAL_COUNTS["train_windows"],
        "aoa_windows": aoa_windows == FORMAL_COUNTS["aoa_windows"],
        "manifest_sha256": manifest_sha256 == FORMAL_MANIFEST_SHA256,
        "excluded_absent": all(
            row["file"] != EXCLUDED_BASENAME for row in manifest),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"formal clean81 contract failed: {failed}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    files = sorted(args.real_dir.glob("*.h5"))
    files = [path for path in files if path.name != args.exclude]
    if not files:
        raise FileNotFoundError(f"no clean Real H5 files in {args.real_dir}")

    manifest = [
        split_row(path, args.window_stride, args.val_fraction)
        for path in files]
    manifest_sha256 = canonical_sha256(manifest)
    if args.formal:
        validate_formal(manifest, manifest_sha256)

    total_sum = np.zeros(2, dtype=np.float64)
    total_squares = np.zeros(2, dtype=np.float64)
    total_count = 0
    by_name = {path.name: path for path in files}
    for index, row in enumerate(manifest, start=1):
        sums, squares, count = prefix_channel_moments(
            by_name[row["file"]], row["val_boundary"])
        total_sum += sums
        total_squares += squares
        total_count += count
        if index % 10 == 0 or index == len(manifest):
            print(json.dumps({
                "event": "clean_prefix_stats",
                "files": index,
                "total": len(manifest),
            }), flush=True)
    mean_uv = total_sum / total_count
    variance_uv = np.maximum(
        total_squares / total_count - np.square(mean_uv), 0.0)
    std_uv = np.sqrt(variance_uv)
    mean = torch.tensor([*mean_uv, 0.0], dtype=torch.float32)
    std = torch.tensor([*std_uv, 0.0], dtype=torch.float32)
    stats = (mean.clone(), mean.clone(), std.clone(), std.clone())

    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "train_manifest.json"
    stats_path = args.output_dir / "mean_std_clean81_prefix.pt"
    result_path = args.output_dir / "result.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    torch.save(stats, stats_path)
    result = {
        "method": "clean81_prefix_only_real_normalizer",
        "excluded": args.exclude,
        "tail_arrays_read": False,
        "prefix_slice": "u/v[:val_boundary,::2,::2]",
        "trajectories": len(manifest),
        "train_windows": sum(len(row["train_starts"]) for row in manifest),
        "aoa_windows": {
            str(aoa): sum(len(row["train_starts"]) for row in manifest
                          if row["aoa"] == aoa)
            for aoa in sorted({row["aoa"] for row in manifest})
        },
        "manifest_sha256": manifest_sha256,
        "normalizer": {
            "mean_input": mean.tolist(),
            "mean_target": mean.tolist(),
            "std_input": std.tolist(),
            "std_target": std.tolist(),
            "population_elements_per_channel": total_count,
        },
        "formal": bool(args.formal),
        "formal_manifest_sha256": FORMAL_MANIFEST_SHA256,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "clean_prefix_assets_complete",
        "manifest": str(manifest_path),
        "stats": str(stats_path),
        "result": str(result_path),
        "manifest_sha256": manifest_sha256,
    }), flush=True)


if __name__ == "__main__":
    main()
