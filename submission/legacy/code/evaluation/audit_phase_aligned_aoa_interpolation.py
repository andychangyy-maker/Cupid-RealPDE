#!/usr/bin/env python3
"""Audit phase-aligned interpolation along the released Sim AoA axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from audit_phase_aligned_re_interpolation import (
    aligned_start, load_uv, relative_rms, tke,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase-radius", type=int, default=30)
    parser.add_argument("--phase-stride", type=int, default=2)
    parser.add_argument("--window-stride", type=int, default=40)
    return parser.parse_args()


def aggregate(rows: list[dict]) -> dict[str, float]:
    return {
        "samples": len(rows),
        "rel_mix_mean": float(np.mean([row["rel_mix"] for row in rows])),
        "rel_lower_mean": float(np.mean([row["rel_lower"] for row in rows])),
        "rel_mix_win_fraction": float(np.mean([
            row["rel_mix"] < row["rel_lower"] for row in rows])),
        "tke_mix_mean": float(np.mean([row["tke_mix"] for row in rows])),
        "tke_lower_mean": float(np.mean([row["tke_lower"] for row in rows])),
        "tke_mix_win_fraction": float(np.mean([
            row["tke_mix"] < row["tke_lower"] for row in rows])),
    }


def main() -> None:
    args = parse_args()
    files = sorted(args.sim_dir.glob("*.h5"))
    reynolds = sorted({int(path.stem.split("_")[0]) for path in files})
    aoas = sorted({int(path.stem.split("_")[1]) for path in files})
    if len(aoas) < 3:
        raise RuntimeError("AoA interpolation needs at least three values")
    starts = list(range(40, 921, args.window_stride))
    rows = []
    started = time.time()
    # Two disjoint triples for the released five-value AoA grid.
    middle_positions = list(range(1, len(aoas) - 1, 2))
    for reynolds_value in reynolds:
        for position in middle_positions:
            low, middle, high = aoas[position - 1:position + 2]
            if high - middle != middle - low:
                continue
            anchor = load_uv(args.sim_dir / f"{reynolds_value}_{low}.h5")
            truth_trajectory = load_uv(
                args.sim_dir / f"{reynolds_value}_{middle}.h5")
            upper = load_uv(args.sim_dir / f"{reynolds_value}_{high}.h5")
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
                    "middle_aoa": middle,
                    "re": reynolds_value,
                    "rel_mix": relative_rms(interpolation, truth),
                    "rel_lower": relative_rms(lower_prediction, truth),
                    "tke_mix": relative_rms(tke(interpolation), truth_tke),
                    "tke_lower": relative_rms(tke(lower_prediction), truth_tke),
                })
    result = {
        "schema": "realpde_phase_aligned_aoa_interpolation_audit_v1",
        "contract": {
            "sim_dir": str(args.sim_dir),
            "phase_radius": args.phase_radius,
            "phase_stride": args.phase_stride,
            "window_stride": args.window_stride,
            "interpolation": "equal-AoA midpoint with low-AoA phase anchor",
        },
        "aggregate": aggregate(rows),
        "by_middle_aoa": {
            str(value): aggregate([
                row for row in rows if row["middle_aoa"] == value])
            for value in sorted({row["middle_aoa"] for row in rows})
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
