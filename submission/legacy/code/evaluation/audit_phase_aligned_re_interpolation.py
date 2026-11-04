#!/usr/bin/env python3
"""Audit whether phase-aligned adjacent-Re interpolation stays on-manifold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase-radius", type=int, default=30)
    parser.add_argument("--phase-stride", type=int, default=2)
    parser.add_argument("--window-stride", type=int, default=40)
    return parser.parse_args()


def load_uv(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.stack([
            np.asarray(handle[name][:, ::2, ::2], dtype=np.float32)
            for name in ("u", "v")
        ], axis=-1)


def signature(value: np.ndarray) -> np.ndarray:
    result = value[::2, ::4, ::4].copy()
    result -= result.mean(axis=(0, 1, 2), keepdims=True)
    result /= np.sqrt(np.mean(result * result)) + 1.0e-6
    return result


def aligned_start(anchor: np.ndarray, other: np.ndarray, start: int,
                  radius: int, stride: int) -> int:
    reference = signature(anchor[start + 10:start + 20])
    low = max(0, start - radius)
    high = min(len(other) - 40, start + radius)
    candidates = list(range(low, high + 1, stride))
    if start not in candidates:
        candidates.append(start)
        candidates.sort()
    errors = [
        np.mean((signature(other[value + 10:value + 20]) - reference) ** 2)
        for value in candidates
    ]
    return int(candidates[int(np.argmin(errors))])


def tke(value: np.ndarray) -> np.ndarray:
    fluctuation = value - value.mean(axis=0, keepdims=True)
    return 0.5 * np.mean(fluctuation * fluctuation, axis=0).sum(axis=-1)


def relative_rms(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((prediction - target) ** 2))
        / (np.sqrt(np.mean(target ** 2)) + 1.0e-8))


def main() -> None:
    args = parse_args()
    if args.phase_radius < 0 or args.phase_stride <= 0:
        raise ValueError("invalid phase search")
    files = sorted(args.sim_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(args.sim_dir)
    reynolds = sorted({int(path.stem.split("_")[0]) for path in files})
    aoas = sorted({int(path.stem.split("_")[1]) for path in files})
    starts = list(range(40, 921, args.window_stride))
    rows = []
    started = time.time()
    for aoa in aoas:
        # Disjoint triples avoid reporting the same trajectory repeatedly.
        for index in range(1, len(reynolds) - 1, 3):
            low, middle, high = reynolds[index - 1:index + 2]
            if high - middle != middle - low:
                continue
            anchor = load_uv(args.sim_dir / f"{low}_{aoa}.h5")
            truth_trajectory = load_uv(args.sim_dir / f"{middle}_{aoa}.h5")
            upper = load_uv(args.sim_dir / f"{high}_{aoa}.h5")
            for start in starts:
                middle_start = aligned_start(
                    anchor, truth_trajectory, start,
                    args.phase_radius, args.phase_stride)
                upper_start = aligned_start(
                    anchor, upper, start,
                    args.phase_radius, args.phase_stride)
                truth = truth_trajectory[
                    middle_start + 20:middle_start + 40]
                lower_prediction = anchor[start + 20:start + 40]
                interpolation = 0.5 * (
                    lower_prediction
                    + upper[upper_start + 20:upper_start + 40])
                truth_tke = tke(truth)
                rows.append({
                    "aoa": aoa,
                    "middle_re": middle,
                    "start": start,
                    "middle_shift": middle_start - start,
                    "upper_shift": upper_start - start,
                    "rel_mix": relative_rms(interpolation, truth),
                    "rel_lower": relative_rms(lower_prediction, truth),
                    "tke_mix": relative_rms(tke(interpolation), truth_tke),
                    "tke_lower": relative_rms(tke(lower_prediction), truth_tke),
                })
    if not rows:
        raise RuntimeError("no equally spaced Reynolds triples were found")

    def aggregate(subset: list[dict]) -> dict[str, float]:
        return {
            "samples": len(subset),
            "rel_mix_mean": float(np.mean([row["rel_mix"] for row in subset])),
            "rel_lower_mean": float(np.mean([row["rel_lower"] for row in subset])),
            "rel_mix_win_fraction": float(np.mean([
                row["rel_mix"] < row["rel_lower"] for row in subset])),
            "tke_mix_mean": float(np.mean([row["tke_mix"] for row in subset])),
            "tke_lower_mean": float(np.mean([row["tke_lower"] for row in subset])),
            "tke_mix_win_fraction": float(np.mean([
                row["tke_mix"] < row["tke_lower"] for row in subset])),
            "middle_abs_shift_mean": float(np.mean([
                abs(row["middle_shift"]) for row in subset])),
            "upper_abs_shift_mean": float(np.mean([
                abs(row["upper_shift"]) for row in subset])),
        }

    result = {
        "schema": "realpde_phase_aligned_re_interpolation_audit_v1",
        "contract": {
            "sim_dir": str(args.sim_dir),
            "phase_radius": args.phase_radius,
            "phase_stride": args.phase_stride,
            "window_stride": args.window_stride,
            "spatial_downsample": 2,
            "signature": "last10_every2_frames_spatial_stride4_centered_rms",
            "interpolation": "equal-Re midpoint with low-Re phase anchor",
        },
        "aggregate": aggregate(rows),
        "by_aoa": {
            str(aoa): aggregate([row for row in rows if row["aoa"] == aoa])
            for aoa in aoas
        },
        "elapsed_seconds": time.time() - started,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
