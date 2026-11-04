#!/usr/bin/env python3
"""LOAO whole-future POD headroom audit for the B210 clean716 cache.

This is a released-training-data, target-visible oracle.  For each held angle
of attack, the POD basis is fitted only from the other four AoA groups.  One
POD sample is one complete, zero-time-mean ``(20,H,W,2)`` future, rather than
one frame.  Held targets are used only to project their future fluctuation
onto the already-fitted basis.  The resulting candidate retains the frozen
baseline's temporal mean and replaces its fluctuation with that projection.

The audit is not a predictor, checkpoint, or submission recipe: it observes
the held future's modal coefficients and therefore measures only a low-rank
representability upper bound.  Ranks 8/16/32 are fixed in code and are all
reported; no held result selects a rank.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCORING_PATH = (
    REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py")


def _load_official_scoring():
    spec = importlib.util.spec_from_file_location(
        "realpde_track1_v9_modal_oracle_scoring", SCORING_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official scoring from {SCORING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL = _load_official_scoring()

SCHEMA = "b210_clean716_loao_whole_future_pod_oracle_v1"
EXPERIMENT = "ttdo_p0d_b210_clean716_modal_headroom"
EVIDENCE_CLASS = "oracle"
RANKS = (8, 16, 32)
AOA_VALUES = (0, 5, 10, 15, 20)
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_STEPS = 20
EXPECTED_CHANNELS = 2
EXPECTED_SHAPE = (716, 20, 32, 64, 2)
EXPECTED_MANIFEST_SHA256 = (
    "c3443b99f290dbfb08b0b1e6e8eaa2bc84e37f552b087c4d3a7ecbe57cff0f20")
WAKE_X_START = 13
WAKE_X_STOP = 38
POINT_WEIGHTS = {
    "rel_l2_score": 46.0,
    "tke_score": 10.0,
    "mvpe_score": 9.0,
}
POINT_WEIGHT_SUM = sum(POINT_WEIGHTS.values())
REQUIRED_CACHE_FILES = (
    "prediction.npy", "target.npy", "aoa.npy", "fold.npy",
    "metadata.json", "split_manifest.json",
)

EXPERIMENT_CONTRACT = {
    "schema": SCHEMA,
    "experiment": EXPERIMENT,
    "evidence_class": EVIDENCE_CLASS,
    "target_visible": True,
    "deployable": False,
    "data_scope": (
        "released Real training data represented by the frozen clean716 "
        "cache; no competition evaluation target"),
    "condition_holdout": (
        "five leave-one-AoA-out fits; the held AoA contributes zero target "
        "rows to its POD basis"),
    "cached_fold_role": (
        "diagnostic aggregation label only; it is an existing crossfit "
        "assignment and is not called condition-held"),
    "pod_sample": "one complete zero-time-mean (20,H,W,2) future window",
    "pod_fit": "uncentered dual POD after per-window temporal centering",
    "fixed_ranks": list(RANKS),
    "rank_selection": "none; all fixed ranks are reported",
    "oracle_candidate": (
        "baseline temporal mean + held-truth fluctuation projected onto the "
        "other-four-AoA train basis"),
    "interpretation": (
        "train-domain low-rank representability upper bound; the oracle "
        "replaces held future amplitude and phase and cannot be deployed"),
    "wake_region": {
        "name": "measured_wake_x13_37",
        "x_start_inclusive": WAKE_X_START,
        "x_stop_exclusive": WAKE_X_STOP,
        "mask_source": "fixed coordinates only",
    },
    "reported_point": {
        "formula": "(46*Rel-L2 + 10*TKE + 9*MVPE) / 65",
        "weights": POINT_WEIGHTS,
        "time_and_sps_included": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--projection-batch-size", type=int, default=32)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8")


def _git_report() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True).stdout.strip()

    porcelain = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(porcelain),
        "porcelain_lines": len(porcelain.splitlines()) if porcelain else 0,
    }


def _environment_text(device: torch.device) -> str:
    rows = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_request": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available() else "none"),
        "pid": os.getpid(),
    }
    return "".join(f"{key}={value}\n" for key, value in rows.items())


def _validate_fixed_ranks(ranks: Sequence[int], fit_samples: int) -> None:
    if tuple(ranks) != RANKS:
        raise ValueError(f"formal ranks must remain exactly {RANKS}")
    if fit_samples < RANKS[-1]:
        raise ValueError("fit partition has fewer rows than the maximum rank")


def fit_indices_for_held_aoa(
        aoas: np.ndarray, held_aoa: int,
        groups: Sequence[int] = AOA_VALUES) -> np.ndarray:
    """Return only other-AoA rows and fail closed on malformed groups."""
    labels = np.asarray(aoas)
    if labels.ndim != 1 or set(labels.tolist()) != set(groups):
        raise ValueError("AoA labels differ from the fixed five-group contract")
    if held_aoa not in groups:
        raise ValueError("held AoA is outside the fixed group contract")
    fit = np.flatnonzero(labels != held_aoa).astype(np.int64)
    held = np.flatnonzero(labels == held_aoa).astype(np.int64)
    if not len(fit) or not len(held):
        raise ValueError("LOAO fit or held partition is empty")
    if np.any(labels[fit] == held_aoa):
        raise RuntimeError("held AoA leaked into POD fit indices")
    return fit


def zero_time_mean_future(values: np.ndarray) -> np.ndarray:
    """Return FP32 per-window future fluctuations without mutating input."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 5 or array.shape[1] < 2 or array.shape[-1] != 2:
        raise ValueError("future must have shape (N,T,H,W,2)")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("future contains non-finite values")
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    result -= result.mean(axis=1, keepdims=True, dtype=np.float32)
    return result


def materialize_fluctuation_matrix(
        target: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Materialize selected whole futures as rows of one POD matrix."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("POD indices must be a non-empty vector")
    if int(indices.min()) < 0 or int(indices.max()) >= len(target):
        raise IndexError("POD indices leave target bounds")
    selected = zero_time_mean_future(target[indices])
    return selected.reshape(len(selected), -1)


def dual_pod_basis(
        fit_matrix: np.ndarray, max_rank: int, device: torch.device,
        ) -> tuple[torch.Tensor, dict[str, object]]:
    """Fit an uncentered POD basis through a sample Gram matrix.

    The returned rows are orthonormal whole-future modes.  QR only repairs
    roundoff in the dual reconstruction; its prefix preserves the nested
    leading-eigenspace ordering.
    """
    matrix = np.asarray(fit_matrix, dtype=np.float32, order="C")
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("fit matrix must be a non-empty 2D array")
    if max_rank <= 0 or max_rank > min(matrix.shape):
        raise ValueError("POD rank exceeds matrix dimensions")
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("POD fit matrix contains non-finite values")

    x = torch.from_numpy(matrix).to(device=device, dtype=torch.float32)
    gram = x @ x.transpose(0, 1)
    gram = 0.5 * (gram + gram.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    leading = eigenvalues[:max_rank]
    tolerance = (
        torch.finfo(torch.float32).eps * matrix.shape[0]
        * torch.clamp(eigenvalues[0], min=1.0))
    if bool((leading <= tolerance).any()):
        raise RuntimeError("POD fit has insufficient numerical rank")
    singular = torch.sqrt(leading)
    raw_basis = (
        eigenvectors[:, :max_rank].transpose(0, 1) @ x
    ) / singular[:, None]
    q, _r = torch.linalg.qr(raw_basis.transpose(0, 1), mode="reduced")
    basis = q.transpose(0, 1).contiguous()
    orthogonality = basis @ basis.transpose(0, 1)
    identity = torch.eye(max_rank, device=device, dtype=torch.float32)
    total_energy = float(torch.clamp(eigenvalues, min=0.0).sum().item())
    leading_cpu = leading.detach().cpu().numpy().astype(np.float64)
    cumulative = np.cumsum(leading_cpu)
    report = {
        "fit_samples": int(matrix.shape[0]),
        "flattened_future_dimensions": int(matrix.shape[1]),
        "max_rank": int(max_rank),
        "numerical_rank_at_least_max_rank": True,
        "leading_eigenvalues": leading_cpu.tolist(),
        "fit_energy_fraction_by_rank": {
            str(rank): float(cumulative[rank - 1] / max(total_energy, 1e-30))
            for rank in RANKS if rank <= max_rank
        },
        "basis_orthogonality_max_abs_error": float(
            torch.max(torch.abs(orthogonality - identity)).item()),
    }
    del x, gram, eigenvalues, eigenvectors, raw_basis, q
    return basis, report


def project_held_fluctuations(
        target: np.ndarray, held_indices: np.ndarray, basis: torch.Tensor,
        ranks: Sequence[int], device: torch.device, batch_size: int,
        ) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """Use held truth only to obtain oracle coefficients in a frozen basis."""
    indices = np.asarray(held_indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("held indices must be a non-empty vector")
    if batch_size <= 0:
        raise ValueError("projection batch size must be positive")
    ranks = tuple(int(rank) for rank in ranks)
    if not ranks or any(rank <= 0 or rank > basis.shape[0] for rank in ranks):
        raise ValueError("projection rank leaves fitted basis")
    output_shape = (len(indices), *target.shape[1:])
    projected = {
        rank: np.empty(output_shape, dtype=np.float32) for rank in ranks}
    truth_energy = 0.0
    residual_energy = {rank: 0.0 for rank in ranks}
    maximum_temporal_mean = {rank: 0.0 for rank in ranks}
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        fluctuation = zero_time_mean_future(target[indices[start:stop]])
        flat = torch.from_numpy(fluctuation.reshape(len(fluctuation), -1)).to(
            device=device, dtype=torch.float32)
        coefficients = flat @ basis.transpose(0, 1)
        truth_energy += float(torch.sum(flat.square(), dtype=torch.float64))
        for rank in ranks:
            reconstruction = (
                coefficients[:, :rank] @ basis[:rank]
            ).reshape(fluctuation.shape)
            # Every fitted row is time-centered.  Remove the remaining FP32
            # roundoff explicitly so the oracle cannot alter the frozen
            # baseline's temporal mean through numerical drift.
            reconstruction = reconstruction - reconstruction.mean(
                dim=1, keepdim=True)
            residual_energy[rank] += float(torch.sum(
                (flat - reconstruction.reshape(len(fluctuation), -1)).square(),
                dtype=torch.float64))
            value = reconstruction.detach().cpu().numpy().astype(
                np.float32, copy=False)
            projected[rank][start:stop] = value
            maximum_temporal_mean[rank] = max(
                maximum_temporal_mean[rank],
                float(np.max(np.abs(value.mean(
                    axis=1, dtype=np.float64)))))
        del flat, coefficients
    report = {
        "held_samples": int(len(indices)),
        "held_truth_fluctuation_sse": truth_energy,
        "by_rank": {
            str(rank): {
                "held_projection_residual_sse": residual_energy[rank],
                "held_fluctuation_sse_removed_fraction": (
                    1.0 - residual_energy[rank] / max(truth_energy, 1e-30)),
                "projected_temporal_mean_max_abs": maximum_temporal_mean[rank],
            }
            for rank in ranks
        },
    }
    return projected, report


def official_error_triples(
        prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the three official per-sample point errors."""
    OFFICIAL.validate_shapes(prediction, target)
    channels = OFFICIAL.measured_channels(target)
    if channels != EXPECTED_CHANNELS:
        raise ValueError(
            f"expected {EXPECTED_CHANNELS} measured channels, got {channels}")
    return np.stack((
        OFFICIAL.rel_l2_per_sample(prediction, target, channels),
        OFFICIAL.tke_rel_l2_per_sample(prediction, target, channels),
        OFFICIAL.mvpe_rel_l2_per_sample(prediction, target),
    ), axis=1).astype(np.float64, copy=False)


def metrics_from_errors(errors: np.ndarray) -> dict[str, object]:
    """Aggregate official errors without averaging scores per sample."""
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("official error matrix must have shape (N,3)")
    means = values.mean(axis=0, dtype=np.float64)
    scores = {
        "rel_l2_score": OFFICIAL.score_error(float(means[0])),
        "tke_score": OFFICIAL.score_error(float(means[1])),
        "mvpe_score": OFFICIAL.score_error(float(means[2])),
    }
    weighted_sum = float(sum(
        POINT_WEIGHTS[key] * scores[key] for key in POINT_WEIGHTS))
    return {
        "samples": int(len(values)),
        "mean_errors": {
            "rel_l2": float(means[0]),
            "tke_rel_l2": float(means[1]),
            "mvpe_rel_l2": float(means[2]),
        },
        **scores,
        "point_final_contribution_46_10_9": weighted_sum / 100.0,
        "point_score_normalized_46_10_9": weighted_sum / POINT_WEIGHT_SUM,
    }


def metric_comparison(
        baseline_errors: np.ndarray,
        candidate_errors: np.ndarray) -> dict[str, object]:
    baseline = metrics_from_errors(baseline_errors)
    candidate = metrics_from_errors(candidate_errors)
    keys = (
        "rel_l2_score", "tke_score", "mvpe_score",
        "point_final_contribution_46_10_9",
        "point_score_normalized_46_10_9",
    )
    return {
        "baseline": baseline,
        "oracle": candidate,
        "delta_oracle_minus_baseline": {
            key: float(candidate[key] - baseline[key]) for key in keys},
    }


def per_sample_sse(
        prediction: np.ndarray, target: np.ndarray,
        x_slice: tuple[int, int] | None = None) -> np.ndarray:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("SSE inputs must be aligned future tensors")
    if x_slice is not None:
        start, stop = x_slice
        prediction = prediction[:, :, :, start:stop, :]
        target = target[:, :, :, start:stop, :]
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    return np.sum(
        difference * difference, axis=(1, 2, 3, 4), dtype=np.float64)


def sse_comparison(
        baseline: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("SSE values must be aligned per-sample vectors")
    baseline_sum = float(baseline.sum(dtype=np.float64))
    candidate_sum = float(candidate.sum(dtype=np.float64))
    reduction = baseline_sum - candidate_sum
    return {
        "samples": int(len(baseline)),
        "baseline_uv_sse": baseline_sum,
        "oracle_uv_sse": candidate_sum,
        "uv_sse_reduction": reduction,
        "uv_sse_reduction_fraction": (
            reduction / baseline_sum if baseline_sum > 0.0 else 0.0),
    }


def evaluate_pair(
        baseline: np.ndarray, candidate: np.ndarray,
        target: np.ndarray) -> dict[str, object]:
    baseline_errors = official_error_triples(baseline, target)
    candidate_errors = official_error_triples(candidate, target)
    return {
        "official_point_metrics": metric_comparison(
            baseline_errors, candidate_errors),
        "overall_sse": sse_comparison(
            per_sample_sse(baseline, target),
            per_sample_sse(candidate, target)),
        "wake_sse": sse_comparison(
            per_sample_sse(
                baseline, target, (WAKE_X_START, WAKE_X_STOP)),
            per_sample_sse(
                candidate, target, (WAKE_X_START, WAKE_X_STOP))),
    }


def validate_cache(cache_dir: Path) -> tuple[dict[str, Path], dict]:
    paths = {name.removesuffix(".npy").removesuffix(".json"): cache_dir / name
             for name in REQUIRED_CACHE_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"clean716 cache is incomplete: {missing}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    if metadata.get("schema") != "b210_archive_calibration_cache_v1":
        raise ValueError("unexpected clean716 metadata schema")
    if tuple(metadata.get("shape", ())) != EXPECTED_SHAPE:
        raise ValueError("clean716 cache shape changed")
    manifest_hash = canonical_sha256(manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("clean716 split manifest changed")
    if metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("metadata and split-manifest hashes disagree")
    return paths, metadata


def _resolve_device(raw: str) -> torch.device:
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _input_hashes(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    extended = {
        **paths,
        "official_scoring": SCORING_PATH,
        "audit_script": Path(__file__).resolve(),
    }
    return {
        name: {
            "path": str(path.resolve()),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for name, path in extended.items()
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.projection_batch_size <= 0:
        raise ValueError("projection batch size must be positive")
    reserved = (
        "config.json", "command.txt", "environment.txt", "result.json",
        "sha256.txt")
    if args.output_dir.exists() and any(
            (args.output_dir / name).exists() for name in reserved):
        raise FileExistsError("refusing to overwrite an existing modal audit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, metadata = validate_cache(args.cache_dir)
    device = _resolve_device(args.device)
    prediction = np.load(paths["prediction"], mmap_mode="r")
    target = np.load(paths["target"], mmap_mode="r")
    aoa = np.load(paths["aoa"], mmap_mode="r")
    folds = np.load(paths["fold"], mmap_mode="r")
    if (tuple(prediction.shape) != EXPECTED_SHAPE
            or tuple(target.shape) != EXPECTED_SHAPE
            or prediction.dtype != np.float32 or target.dtype != np.float32):
        raise ValueError("prediction/target violate the clean716 contract")
    if tuple(aoa.shape) != (EXPECTED_SHAPE[0],):
        raise ValueError("AoA labels do not align with clean716")
    if tuple(folds.shape) != (EXPECTED_SHAPE[0],):
        raise ValueError("fold labels do not align with clean716")
    if set(np.asarray(aoa).tolist()) != set(AOA_VALUES):
        raise ValueError("AoA groups differ from the frozen contract")
    if set(np.asarray(folds).tolist()) != set(EXPECTED_FOLDS):
        raise ValueError("cached folds differ from the frozen contract")

    config = {
        **EXPERIMENT_CONTRACT,
        "cache_dir": str(args.cache_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "projection_batch_size": int(args.projection_batch_size),
        "shape": list(EXPECTED_SHAPE),
    }
    config_path = args.output_dir / "config.json"
    command_path = args.output_dir / "command.txt"
    environment_path = args.output_dir / "environment.txt"
    _write_json(config_path, config)
    command_path.write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    environment_path.write_text(
        _environment_text(device), encoding="utf-8")

    started = time.perf_counter()
    candidates = {
        rank: np.empty(EXPECTED_SHAPE, dtype=np.float32) for rank in RANKS}
    by_aoa: dict[str, object] = {}
    basis_reports: dict[str, object] = {}
    labels = np.asarray(aoa, dtype=np.int64)
    for held_aoa in AOA_VALUES:
        fit_indices = fit_indices_for_held_aoa(labels, held_aoa)
        held_indices = np.flatnonzero(labels == held_aoa).astype(np.int64)
        _validate_fixed_ranks(RANKS, len(fit_indices))
        fit_matrix = materialize_fluctuation_matrix(target, fit_indices)
        basis, basis_report = dual_pod_basis(
            fit_matrix, RANKS[-1], device)
        projected, projection_report = project_held_fluctuations(
            target, held_indices, basis, RANKS, device,
            args.projection_batch_size)
        baseline_held = np.asarray(prediction[held_indices], dtype=np.float32)
        target_held = np.asarray(target[held_indices], dtype=np.float32)
        baseline_mean = baseline_held.mean(
            axis=1, keepdims=True, dtype=np.float32)
        rank_results = {}
        for rank in RANKS:
            candidate = baseline_mean + projected[rank]
            candidates[rank][held_indices] = candidate
            rank_results[str(rank)] = evaluate_pair(
                baseline_held, candidate, target_held)
        by_aoa[str(held_aoa)] = {
            "fit_samples": int(len(fit_indices)),
            "held_samples": int(len(held_indices)),
            "held_target_rows_used_in_basis": 0,
            "projection": projection_report,
            "by_rank": rank_results,
        }
        basis_reports[str(held_aoa)] = basis_report
        del fit_matrix, basis, projected, baseline_held, target_held
        if device.type == "cuda":
            torch.cuda.empty_cache()

    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    overall = {
        str(rank): evaluate_pair(
            prediction_array, candidates[rank], target_array)
        for rank in RANKS
    }
    by_cached_fold: dict[str, object] = {}
    fold_labels = np.asarray(folds, dtype=np.int64)
    for fold in EXPECTED_FOLDS:
        rows = np.flatnonzero(fold_labels == fold)
        by_cached_fold[str(fold)] = {
            "role": "diagnostic_existing_crossfit_fold_not_condition_holdout",
            "samples": int(len(rows)),
            "by_rank": {
                str(rank): evaluate_pair(
                    prediction_array[rows], candidates[rank][rows],
                    target_array[rows])
                for rank in RANKS
            },
        }

    input_hashes = _input_hashes(paths)
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "evidence_class": EVIDENCE_CLASS,
        "target_visible": True,
        "deployable": False,
        "training_performed": False,
        "rank_selected": None,
        "condition_holdout": "leave_one_AoA_out",
        "interpretation": EXPERIMENT_CONTRACT["interpretation"],
        "config_canonical_sha256": canonical_sha256(config),
        "git": _git_report(),
        "source_archive": metadata.get("archive"),
        "inputs": input_hashes,
        "basis_by_held_aoa": basis_reports,
        "by_held_aoa": by_aoa,
        "overall": overall,
        "by_cached_fold": by_cached_fold,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "large_prediction_written": False,
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    checksum_rows = [
        f"{item['sha256']}  input:{name}"
        for name, item in input_hashes.items()]
    for path in (config_path, command_path, environment_path, result_path):
        checksum_rows.append(f"{sha256_file(path)}  output:{path.name}")
    (args.output_dir / "sha256.txt").write_text(
        "\n".join(checksum_rows) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    rank32 = result["overall"]["32"]
    print(json.dumps({
        "event": "b210_clean716_loao_modal_oracle_complete",
        "evidence_class": result["evidence_class"],
        "target_visible": result["target_visible"],
        "output": str(args.output_dir.resolve()),
        "rank32_point_delta": rank32["official_point_metrics"][
            "delta_oracle_minus_baseline"][
                "point_final_contribution_46_10_9"],
        "rank32_wake_sse_reduction_fraction": rank32["wake_sse"][
            "uv_sse_reduction_fraction"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
