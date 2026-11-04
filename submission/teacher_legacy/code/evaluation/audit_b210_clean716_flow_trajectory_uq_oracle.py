#!/usr/bin/env python3
"""Fixed-point, target-visible flow-trajectory interval oracle on clean716."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/evaluation"))
import audit_b210_clean716_smooth_flow_oracle as smooth  # noqa: E402


SCHEMA = "b210_clean716_fixed_point_flow_trajectory_uq_oracle_v1"
EXPECTED_SHAPE = smooth.EXPECTED_SHAPE
GO_DELTA_SPS = 0.50
GO_POSITIVE_AOA = 4
GO_WORST_AOA = -0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def trajectory_envelope(point: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Envelope of point and the centered-sign smooth-warp orbit."""
    plus = smooth.smooth_flow_warp(point, flow)
    minus = smooth.smooth_flow_warp(point, -flow)
    lower = torch.minimum(point, torch.minimum(plus, minus))
    upper = torch.maximum(point, torch.maximum(plus, minus))
    return lower, upper


def interval_metrics(prediction: np.ndarray, target: np.ndarray,
                     lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    if lower.shape != prediction.shape or upper.shape != prediction.shape:
        raise ValueError("interval arrays must match point shape")
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise FloatingPointError("interval arrays must be finite")
    if np.any(lower > upper):
        raise ValueError("lower exceeds upper")
    value, coverage = smooth._integer.OFFICIAL.aggregate_sps(
        prediction, target, 2, lower, upper)
    scored = target != 0.0
    width = upper - lower
    return {
        "sps_score": float(100.0 * value),
        "coverage": float(coverage),
        "scored_mean_width": float(np.mean(width[scored], dtype=np.float64)),
        "all_mean_width": float(np.mean(width, dtype=np.float64)),
    }


def grouped_metrics(prediction: np.ndarray, target: np.ndarray,
                    lower: np.ndarray, upper: np.ndarray,
                    aoa: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        str(int(value)): interval_metrics(
            prediction[aoa == value], target[aoa == value],
            lower[aoa == value], upper[aoa == value])
        for value in sorted(np.unique(aoa).tolist())
    }


def horizon_diagnostics(target: np.ndarray, lower: np.ndarray,
                        upper: np.ndarray) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for left, right in smooth.HORIZON_BLOCKS:
        local_target = target[:, left:right]
        local_lower = lower[:, left:right]
        local_upper = upper[:, left:right]
        scored = local_target != 0.0
        inside = (local_target >= local_lower) & (local_target <= local_upper)
        rows[f"{left + 1:02d}_{right:02d}"] = {
            "coverage": float(np.count_nonzero(inside & scored) / np.count_nonzero(scored)),
            "scored_mean_width": float(np.mean(
                (local_upper - local_lower)[scored], dtype=np.float64)),
        }
    return rows


def delta_report(candidate: dict, native: dict) -> dict:
    by_aoa = {
        key: float(candidate["by_aoa"][key]["sps_score"]
                   - native["by_aoa"][key]["sps_score"])
        for key in native["by_aoa"]
    }
    values = list(by_aoa.values())
    return {
        "sps_score": float(candidate["overall"]["sps_score"]
                           - native["overall"]["sps_score"]),
        "coverage": float(candidate["overall"]["coverage"]
                          - native["overall"]["coverage"]),
        "scored_mean_width": float(candidate["overall"]["scored_mean_width"]
                                   - native["overall"]["scored_mean_width"]),
        "by_aoa_sps": by_aoa,
        "positive_aoa": int(sum(value > 0.0 for value in values)),
        "worst_aoa_sps": float(min(values)),
    }


def run(args: argparse.Namespace) -> dict:
    if args.batch_size != 16:
        raise ValueError("frozen oracle contract requires batch-size=16")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty result directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, metadata, _manifest = smooth._integer.validate_inputs(args.cache_dir)
    device = smooth._integer.resolve_device(args.device)
    prediction = np.load(paths["prediction"], mmap_mode="r")
    target = np.load(paths["target"], mmap_mode="r")
    native_lower = np.load(args.cache_dir / "lower.npy", mmap_mode="r")
    native_upper = np.load(args.cache_dir / "upper.npy", mmap_mode="r")
    aoa = np.load(paths["aoa"], mmap_mode="r")
    arrays = (prediction, target, native_lower, native_upper)
    if any(tuple(value.shape) != EXPECTED_SHAPE or value.dtype != np.float32
           for value in arrays):
        raise ValueError("clean716 point/target/bounds contract changed")
    if np.any(native_lower > native_upper):
        raise ValueError("native lower exceeds upper")

    config = {
        "schema": SCHEMA,
        "evidence_class": "oracle",
        "target_visible": True,
        "deployable": False,
        "fixed_point": True,
        "point_sha256": sha256_file(paths["prediction"]),
        "shape": list(EXPECTED_SHAPE),
        "batch_size": args.batch_size,
        "device": str(device),
        "flow_contract": {
            "horizon_blocks": [[left + 1, right] for left, right in smooth.HORIZON_BLOCKS],
            "control_grid": list(smooth.CONTROL_GRID),
            "max_displacement_pixels_per_component": smooth.MAX_DISPLACEMENT_PIXELS,
            "adam_steps": smooth.ADAM_STEPS,
            "adam_lr": smooth.ADAM_LR,
            "tv_weight": smooth.TV_WEIGHT,
        },
        "arms": ["native", "orbit_envelope", "native_union_orbit"],
        "go_gate": {
            "delta_sps": GO_DELTA_SPS,
            "positive_aoa": GO_POSITIVE_AOA,
            "worst_aoa_sps": GO_WORST_AOA,
            "rule": "either preregistered flow arm passes all conditions",
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")

    def summarize(lower: np.ndarray, upper: np.ndarray) -> dict:
        return {
            "overall": interval_metrics(prediction, target, lower, upper),
            "by_aoa": grouped_metrics(prediction, target, lower, upper, aoa),
            "by_horizon": horizon_diagnostics(target, lower, upper),
        }

    native = summarize(native_lower, native_upper)
    orbit_lower = np.empty(EXPECTED_SHAPE, dtype=np.float32)
    orbit_upper = np.empty(EXPECTED_SHAPE, dtype=np.float32)
    flow_magnitude_sum = 0.0
    flow_elements = 0
    flow_max = 0.0
    chosen_blocks = 0
    started = time.perf_counter()
    for start in range(0, EXPECTED_SHAPE[0], args.batch_size):
        stop = min(start + args.batch_size, EXPECTED_SHAPE[0])
        point = torch.from_numpy(np.array(prediction[start:stop], copy=True)).to(device)
        truth = torch.from_numpy(np.array(target[start:stop], copy=True)).to(device)
        for left, right in smooth.HORIZON_BLOCKS:
            _selected, chosen, flow = smooth.optimize_block_oracle(
                point[:, left:right], truth[:, left:right])
            lower, upper = trajectory_envelope(point[:, left:right], flow)
            orbit_lower[start:stop, left:right] = lower.cpu().numpy()
            orbit_upper[start:stop, left:right] = upper.cpu().numpy()
            magnitude = torch.linalg.vector_norm(flow, dim=-1)
            flow_magnitude_sum += float(magnitude.sum().cpu())
            flow_elements += magnitude.numel()
            flow_max = max(flow_max, float(magnitude.max().cpu()))
            chosen_blocks += int(chosen.sum().cpu())
        if stop == EXPECTED_SHAPE[0] or stop % 160 == 0:
            print(json.dumps({"event": "flow_uq_oracle_progress",
                              "samples_complete": stop,
                              "samples_total": EXPECTED_SHAPE[0],
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)

    if not (np.isfinite(orbit_lower).all() and np.isfinite(orbit_upper).all()
            and np.all(orbit_lower <= prediction) and np.all(prediction <= orbit_upper)):
        raise RuntimeError("orbit envelope contract failed")
    orbit = summarize(orbit_lower, orbit_upper)
    orbit["delta_vs_native"] = delta_report(orbit, native)

    np.minimum(orbit_lower, native_lower, out=orbit_lower)
    np.maximum(orbit_upper, native_upper, out=orbit_upper)
    union = summarize(orbit_lower, orbit_upper)
    union["delta_vs_native"] = delta_report(union, native)
    del orbit_lower, orbit_upper
    gc.collect()

    def passes(row: dict) -> bool:
        delta = row["delta_vs_native"]
        return (delta["sps_score"] >= GO_DELTA_SPS
                and delta["positive_aoa"] >= GO_POSITIVE_AOA
                and delta["worst_aoa_sps"] >= GO_WORST_AOA)

    passed = [name for name, row in (
        ("orbit_envelope", orbit), ("native_union_orbit", union)) if passes(row)]
    result = {
        "schema": SCHEMA,
        "evidence_class": "oracle",
        "target_visible": True,
        "deployable": False,
        "fixed_point": {
            "bitwise_unchanged": True,
            "sha256_before": config["point_sha256"],
            "sha256_after": sha256_file(paths["prediction"]),
        },
        "source_archive": metadata.get("archive"),
        "native": native,
        "orbit_envelope": orbit,
        "native_union_orbit": union,
        "flow": {
            "chosen_blocks": chosen_blocks,
            "total_blocks": EXPECTED_SHAPE[0] * len(smooth.HORIZON_BLOCKS),
            "mean_magnitude_pixels": flow_magnitude_sum / flow_elements,
            "max_magnitude_pixels": flow_max,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "passed_arms": passed,
        "decision": ("GO_HISTORY_FLOW_UQ_ZERO_TRAIN_REPLAY" if passed
                     else "STOP_FLOW_TRAJECTORY_UQ_REPRESENTATION"),
    }
    if result["fixed_point"]["sha256_before"] != result["fixed_point"]["sha256_after"]:
        raise RuntimeError("point cache changed during interval oracle")
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = {
        "config.json": sha256_file(args.output_dir / "config.json"),
        "result.json": sha256_file(result_path),
        "script": sha256_file(Path(__file__)),
        "prediction": sha256_file(paths["prediction"]),
        "target": sha256_file(paths["target"]),
        "native_lower": sha256_file(args.cache_dir / "lower.npy"),
        "native_upper": sha256_file(args.cache_dir / "upper.npy"),
    }
    (args.output_dir / "sha256.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"event": "flow_uq_oracle_complete",
                      "decision": result["decision"],
                      "passed_arms": result["passed_arms"]}), flush=True)


if __name__ == "__main__":
    main()
