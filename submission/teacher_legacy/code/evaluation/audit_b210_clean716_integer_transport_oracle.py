#!/usr/bin/env python3
"""Target-visible finite-translation headroom audit on the clean716 cache.

This program is deliberately an oracle, not a deployable predictor.  For each
sample and each of four five-frame horizon blocks it reads the target and picks
the integer (dy, dx) translation with the smallest full-field UV SSE.  It never
trains a parameter, loads a checkpoint, or writes a prediction array.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shlex
import sys
import time
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
SCORING_PATH = (
    REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py")
SCHEMA = "b210_clean716_integer_transport_oracle_v1"
EVIDENCE_CLASS = "oracle"
EXPECTED_SHAPE = (716, 20, 32, 64, 2)
EXPECTED_MANIFEST_SHA256 = (
    "c3443b99f290dbfb08b0b1e6e8eaa2bc84e37f552b087c4d3a7ecbe57cff0f20")
HORIZON_BLOCKS = ((0, 5), (5, 10), (10, 15), (15, 20))
WAKE_X_SLICE = (13, 64)
POINT_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_score": 0.10,
    "mvpe_score": 0.09,
}


def _candidate_order(shift: tuple[int, int]) -> tuple[int, int, int, int, int]:
    dy, dx = shift
    return abs(dy) + abs(dx), abs(dy), abs(dx), dy, dx


CANDIDATE_SHIFTS = tuple(sorted(
    ((dy, dx) for dy in range(-1, 2) for dx in range(-2, 3)),
    key=_candidate_order,
))
IDENTITY_INDEX = CANDIDATE_SHIFTS.index((0, 0))


def load_official_scoring():
    spec = importlib.util.spec_from_file_location(
        "realpde_track1_v9_transport_oracle_scoring", SCORING_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official scorer from {SCORING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL = load_official_scoring()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def integer_translate(value: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate B,T,H,W,C; positive dx/dy moves content right/down.

    Border values are replicated.  Identity is a bitwise bypass rather than a
    call through an interpolation operator.
    """
    if value.ndim != 5:
        raise ValueError("translation input must have shape (B,T,H,W,C)")
    if not isinstance(dy, int) or not isinstance(dx, int):
        raise TypeError("integer transport requires integer dy and dx")
    if dy == 0 and dx == 0:
        return value.clone()
    batch, steps, height, width, channels = value.shape
    image = value.permute(0, 1, 4, 2, 3).reshape(
        batch * steps, channels, height, width)
    pad_y, pad_x = abs(dy), abs(dx)
    padded = F.pad(image, (pad_x, pad_x, pad_y, pad_y), mode="replicate")
    start_y = pad_y - dy
    start_x = pad_x - dx
    shifted = padded[
        :, :, start_y:start_y + height, start_x:start_x + width]
    return shifted.reshape(
        batch, steps, channels, height, width).permute(0, 1, 3, 4, 2)


@torch.inference_mode()
def target_visible_block_oracle(
        prediction: torch.Tensor, target: torch.Tensor,
        shifts: Sequence[tuple[int, int]] = CANDIDATE_SHIFTS,
        blocks: Sequence[tuple[int, int]] = HORIZON_BLOCKS,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a target-visible UV-SSE-minimizing shift per sample/block."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must be aligned 5D tensors")
    if prediction.dtype != torch.float32 or target.dtype != torch.float32:
        raise TypeError("oracle selection requires float32 tensors")
    if prediction.shape[-1] != 2:
        raise ValueError("clean716 oracle requires exactly the UV channels")
    if not shifts or (0, 0) not in shifts:
        raise ValueError("finite candidate set must contain identity")
    if tuple(blocks) != HORIZON_BLOCKS:
        raise ValueError("oracle requires the frozen four horizon blocks")

    selected = torch.empty_like(prediction)
    indices = torch.empty(
        (len(prediction), len(blocks)), dtype=torch.int16,
        device=prediction.device)
    for block_index, (start, stop) in enumerate(blocks):
        source = prediction[:, start:stop]
        truth = target[:, start:stop]
        errors = []
        for dy, dx in shifts:
            candidate = integer_translate(source, int(dy), int(dx))
            errors.append((candidate - truth).square().sum(
                dim=(1, 2, 3, 4)))
        error_grid = torch.stack(errors, dim=1)
        best = torch.argmin(error_grid, dim=1)
        indices[:, block_index] = best.to(torch.int16)
        block_prediction = torch.empty_like(source)
        for shift_index, (dy, dx) in enumerate(shifts):
            rows = best == shift_index
            if bool(rows.any()):
                candidate = integer_translate(source[rows], int(dy), int(dx))
                block_prediction[rows] = candidate
        selected[:, start:stop] = block_prediction
    return selected, indices


def official_error_triples(
        prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    OFFICIAL.validate_shapes(prediction, target)
    channels = OFFICIAL.measured_channels(target)
    if channels != 2:
        raise ValueError(f"clean716 target must have two measured channels, got {channels}")
    return np.stack((
        OFFICIAL.rel_l2_per_sample(prediction, target, channels),
        OFFICIAL.tke_rel_l2_per_sample(prediction, target, channels),
        OFFICIAL.mvpe_rel_l2_per_sample(prediction, target),
    ), axis=1).astype(np.float64, copy=False)


def metrics_from_errors(errors: np.ndarray) -> dict[str, object]:
    if errors.ndim != 2 or errors.shape[1] != 3 or len(errors) == 0:
        raise ValueError("official errors must have non-empty shape (N,3)")
    means = np.mean(errors, axis=0, dtype=np.float64)
    scores = {
        "rel_l2_score": OFFICIAL.score_error(float(means[0])),
        "tke_score": OFFICIAL.score_error(float(means[1])),
        "mvpe_score": OFFICIAL.score_error(float(means[2])),
    }
    contribution = float(sum(
        POINT_WEIGHTS[key] * scores[key] for key in POINT_WEIGHTS))
    return {
        "samples": int(len(errors)),
        "mean_errors": {
            "rel_l2": float(means[0]),
            "tke_rel_l2": float(means[1]),
            "mvpe_rel_l2": float(means[2]),
        },
        **scores,
        "point_final_contribution_46_10_9": contribution,
        "point_score_normalized_46_10_9": contribution / sum(
            POINT_WEIGHTS.values()),
    }


def metric_comparison(
        baseline_errors: np.ndarray, oracle_errors: np.ndarray) -> dict[str, object]:
    baseline = metrics_from_errors(baseline_errors)
    oracle = metrics_from_errors(oracle_errors)
    keys = (
        "rel_l2_score", "tke_score", "mvpe_score",
        "point_final_contribution_46_10_9",
        "point_score_normalized_46_10_9",
    )
    return {
        "baseline": baseline,
        "oracle": oracle,
        "delta_oracle_minus_baseline": {
            key: float(oracle[key] - baseline[key]) for key in keys},
    }


def per_sample_sse(
        prediction: np.ndarray, target: np.ndarray,
        x_slice: tuple[int, int] | None = None) -> np.ndarray:
    if x_slice is not None:
        start, stop = x_slice
        prediction = prediction[:, :, :, start:stop, :]
        target = target[:, :, :, start:stop, :]
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    return np.sum(difference * difference, axis=(1, 2, 3, 4), dtype=np.float64)


def sse_comparison(
        baseline_sse: np.ndarray, oracle_sse: np.ndarray) -> dict[str, float | int]:
    if baseline_sse.shape != oracle_sse.shape or baseline_sse.ndim != 1:
        raise ValueError("SSE arrays must be aligned vectors")
    baseline = float(np.sum(baseline_sse, dtype=np.float64))
    oracle = float(np.sum(oracle_sse, dtype=np.float64))
    reduction = baseline - oracle
    return {
        "samples": int(len(baseline_sse)),
        "baseline_uv_sse": baseline,
        "oracle_uv_sse": oracle,
        "uv_sse_reduction": reduction,
        "uv_sse_reduction_fraction": (
            reduction / baseline if baseline > 0.0 else 0.0),
    }


def selection_distribution(
        indices: np.ndarray,
        shifts: Sequence[tuple[int, int]] = CANDIDATE_SHIFTS,
        blocks: Sequence[tuple[int, int]] = HORIZON_BLOCKS) -> dict[str, object]:
    if indices.ndim != 2 or indices.shape[1] != len(blocks):
        raise ValueError("selection indices must have shape (N,4)")

    def summarize(values: np.ndarray) -> dict[str, object]:
        counts = np.bincount(values.reshape(-1), minlength=len(shifts))
        histogram = {
            f"dy={dy:+d},dx={dx:+d}": int(counts[index])
            for index, (dy, dx) in enumerate(shifts) if counts[index]
        }
        total = int(values.size)
        identity = int(counts[shifts.index((0, 0))])
        chosen = np.asarray([shifts[int(index)] for index in values.reshape(-1)])
        return {
            "choices": total,
            "identity_choices": identity,
            "non_identity_fraction": float((total - identity) / max(total, 1)),
            "mean_abs_dy": float(np.mean(np.abs(chosen[:, 0]))),
            "mean_abs_dx": float(np.mean(np.abs(chosen[:, 1]))),
            "histogram": histogram,
        }

    return {
        "all_blocks": summarize(indices),
        "by_horizon": {
            f"{start + 1:02d}_{stop:02d}": summarize(indices[:, block:block + 1])
            for block, (start, stop) in enumerate(blocks)
        },
    }


def _delta_sse_group(
        baseline: np.ndarray, oracle: np.ndarray,
        indices: np.ndarray) -> dict[str, float | int]:
    return sse_comparison(baseline[indices], oracle[indices])


def validate_inputs(cache_dir: Path) -> tuple[dict[str, Path], dict, list]:
    paths = {
        "prediction": cache_dir / "prediction.npy",
        "target": cache_dir / "target.npy",
        "aoa": cache_dir / "aoa.npy",
        "metadata": cache_dir / "metadata.json",
        "split_manifest": cache_dir / "split_manifest.json",
        "official_scoring": SCORING_PATH,
        "audit_script": Path(__file__).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"clean716 cache is incomplete: {missing}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    if metadata.get("schema") != "b210_archive_calibration_cache_v1":
        raise ValueError("unexpected clean716 metadata schema")
    if tuple(metadata.get("shape", ())) != EXPECTED_SHAPE:
        raise ValueError("clean716 metadata shape changed")
    if int(metadata.get("val_windows", -1)) != EXPECTED_SHAPE[0]:
        raise ValueError("clean716 metadata sample count changed")
    manifest_sha = canonical_sha256(manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"clean716 manifest changed: {manifest_sha} != {EXPECTED_MANIFEST_SHA256}")
    if metadata.get("manifest_sha256") != manifest_sha:
        raise ValueError("metadata and split manifest hashes disagree")
    return paths, metadata, manifest


def environment_report(device: torch.device) -> str:
    rows = [
        f"utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"cwd={Path.cwd().resolve()}",
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"device={device}",
    ]
    if device.type == "cuda":
        rows.extend((
            f"cuda_runtime={torch.version.cuda}",
            f"gpu={torch.cuda.get_device_name(device)}",
        ))
    return "\n".join(rows) + "\n"


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    output_files = (
        "config.json", "command.txt", "environment.txt", "result.json",
        "sha256.txt")
    if args.output_dir.exists() and any(
            (args.output_dir / name).exists() for name in output_files):
        raise FileExistsError("refusing to overwrite an existing oracle audit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, metadata, _manifest = validate_inputs(args.cache_dir)
    device = resolve_device(args.device)
    prediction = np.load(paths["prediction"], mmap_mode="r")
    target = np.load(paths["target"], mmap_mode="r")
    aoa = np.load(paths["aoa"], mmap_mode="r")
    if (tuple(prediction.shape) != EXPECTED_SHAPE
            or tuple(target.shape) != EXPECTED_SHAPE
            or prediction.dtype != np.float32 or target.dtype != np.float32):
        raise ValueError("prediction/target arrays violate the clean716 contract")
    if tuple(aoa.shape) != (EXPECTED_SHAPE[0],):
        raise ValueError("AoA labels do not align with clean716")

    config = {
        "schema": SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "target_visible": True,
        "deployable": False,
        "training_performed": False,
        "checkpoint_loaded": False,
        "cache_dir": str(args.cache_dir.resolve()),
        "shape": list(EXPECTED_SHAPE),
        "batch_size": int(args.batch_size),
        "requested_device": args.device,
        "resolved_device": str(device),
        "horizon_blocks_1_based": [
            [start + 1, stop] for start, stop in HORIZON_BLOCKS],
        "candidate_integer_shifts_dy_dx": [
            [dy, dx] for dy, dx in CANDIDATE_SHIFTS],
        "selection": "per-sample/per-block minimum full-field UV SSE",
        "translation": {
            "positive_dx": "content moves right: out[y,x]=in[y,x-dx]",
            "positive_dy": "content moves down: out[y,x]=in[y-dy,x]",
            "padding": "replicate border",
            "identity": "bitwise bypass",
            "shared_across_uv": True,
        },
        "wake_region": {
            "x_zero_based_half_open": list(WAKE_X_SLICE),
            "definition": "fixed geometry-only streamwise slice",
        },
        "point_weights": POINT_WEIGHTS,
        "large_prediction_written": False,
    }
    config_path = args.output_dir / "config.json"
    command_path = args.output_dir / "command.txt"
    environment_path = args.output_dir / "environment.txt"
    write_json(config_path, config)
    command_path.write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    environment_path.write_text(environment_report(device), encoding="utf-8")

    started = time.perf_counter()
    samples = EXPECTED_SHAPE[0]
    baseline_errors = np.empty((samples, 3), dtype=np.float64)
    oracle_errors = np.empty_like(baseline_errors)
    baseline_horizon_errors = np.empty((samples, 4, 3), dtype=np.float64)
    oracle_horizon_errors = np.empty_like(baseline_horizon_errors)
    baseline_overall_sse = np.empty(samples, dtype=np.float64)
    oracle_overall_sse = np.empty(samples, dtype=np.float64)
    baseline_wake_sse = np.empty(samples, dtype=np.float64)
    oracle_wake_sse = np.empty(samples, dtype=np.float64)
    baseline_horizon_sse = np.empty((samples, 4), dtype=np.float64)
    oracle_horizon_sse = np.empty_like(baseline_horizon_sse)
    selected_indices = np.empty((samples, 4), dtype=np.int16)

    for start in range(0, samples, args.batch_size):
        stop = min(start + args.batch_size, samples)
        prediction_batch = np.array(prediction[start:stop], copy=True)
        target_batch = np.array(target[start:stop], copy=True)
        if not (np.all(np.isfinite(prediction_batch))
                and np.all(np.isfinite(target_batch))):
            raise FloatingPointError("clean716 prediction/target contains non-finite values")
        prediction_device = torch.from_numpy(prediction_batch).to(device)
        target_device = torch.from_numpy(target_batch).to(device)
        oracle_device, indices_device = target_visible_block_oracle(
            prediction_device, target_device)
        oracle_batch = oracle_device.cpu().numpy()
        selected_indices[start:stop] = indices_device.cpu().numpy()

        baseline_errors[start:stop] = official_error_triples(
            prediction_batch, target_batch)
        oracle_errors[start:stop] = official_error_triples(
            oracle_batch, target_batch)
        baseline_overall_sse[start:stop] = per_sample_sse(
            prediction_batch, target_batch)
        oracle_overall_sse[start:stop] = per_sample_sse(
            oracle_batch, target_batch)
        baseline_wake_sse[start:stop] = per_sample_sse(
            prediction_batch, target_batch, WAKE_X_SLICE)
        oracle_wake_sse[start:stop] = per_sample_sse(
            oracle_batch, target_batch, WAKE_X_SLICE)
        for block, (block_start, block_stop) in enumerate(HORIZON_BLOCKS):
            baseline_block = prediction_batch[:, block_start:block_stop]
            oracle_block = oracle_batch[:, block_start:block_stop]
            target_block = target_batch[:, block_start:block_stop]
            baseline_horizon_errors[start:stop, block] = official_error_triples(
                baseline_block, target_block)
            oracle_horizon_errors[start:stop, block] = official_error_triples(
                oracle_block, target_block)
            baseline_horizon_sse[start:stop, block] = per_sample_sse(
                baseline_block, target_block)
            oracle_horizon_sse[start:stop, block] = per_sample_sse(
                oracle_block, target_block)
        del prediction_device, target_device, oracle_device

    overall = {
        "official_point_metrics": metric_comparison(
            baseline_errors, oracle_errors),
        "overall_sse": sse_comparison(
            baseline_overall_sse, oracle_overall_sse),
        "wake_sse": sse_comparison(baseline_wake_sse, oracle_wake_sse),
    }
    by_horizon = {}
    for block, (start, stop) in enumerate(HORIZON_BLOCKS):
        label = f"{start + 1:02d}_{stop:02d}"
        by_horizon[label] = {
            "official_formula_on_horizon_diagnostic": metric_comparison(
                baseline_horizon_errors[:, block],
                oracle_horizon_errors[:, block]),
            "sse": sse_comparison(
                baseline_horizon_sse[:, block], oracle_horizon_sse[:, block]),
        }

    aoa_array = np.asarray(aoa, dtype=np.int64)
    by_aoa = {}
    for value in sorted(int(item) for item in np.unique(aoa_array)):
        rows = np.flatnonzero(aoa_array == value)
        by_aoa[str(value)] = {
            "official_point_metrics": metric_comparison(
                baseline_errors[rows], oracle_errors[rows]),
            "overall_sse": _delta_sse_group(
                baseline_overall_sse, oracle_overall_sse, rows),
            "wake_sse": _delta_sse_group(
                baseline_wake_sse, oracle_wake_sse, rows),
            "selection": selection_distribution(selected_indices[rows]),
        }

    input_hashes = {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for name, path in paths.items()
    }
    result = {
        "schema": SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "target_visible": True,
        "deployable": False,
        "interpretation": (
            "Target-visible translation headroom only. These shifts cannot be "
            "used at inference and are not evidence of held-condition gain."),
        "training_performed": False,
        "checkpoint_loaded": False,
        "large_prediction_written": False,
        "config_canonical_sha256": canonical_sha256(config),
        "inputs": input_hashes,
        "source_archive": metadata.get("archive"),
        "elapsed_seconds": time.perf_counter() - started,
        "overall": overall,
        "by_horizon": by_horizon,
        "by_aoa": by_aoa,
        "selection": selection_distribution(selected_indices),
    }
    result_path = args.output_dir / "result.json"
    write_json(result_path, result)

    checksums = []
    for name, item in input_hashes.items():
        checksums.append(f"{item['sha256']}  input:{name}")
    for path in (config_path, command_path, environment_path, result_path):
        checksums.append(f"{file_sha256(path)}  output:{path.name}")
    (args.output_dir / "sha256.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_clean716_integer_transport_oracle_complete",
        "evidence_class": result["evidence_class"],
        "target_visible": result["target_visible"],
        "output": str(args.output_dir.resolve()),
        "point_delta": result["overall"]["official_point_metrics"][
            "delta_oracle_minus_baseline"][
                "point_final_contribution_46_10_9"],
        "wake_sse_reduction_fraction": result["overall"]["wake_sse"][
            "uv_sse_reduction_fraction"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
