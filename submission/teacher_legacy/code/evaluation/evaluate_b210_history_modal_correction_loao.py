#!/usr/bin/env python3
"""Evaluate the preregistered history-only rank-32 modal controller.

This is controller-only leave-one-AoA-out evidence on top of a frozen point
baseline that was trained with all five AoAs.  It is therefore *not* an
end-to-end LOAO model result.  Every fitted object receives already-sliced
fit-side arrays.  Held targets may be read before fitting only for frozen
SHA/bitwise raw-H5 integrity checks; they first enter any model-dependent
calculation after the deployable candidate has been formed, for scoring and
for the separately labelled matched oracle.

The program intentionally writes only JSON/text provenance.  It never writes
the 716-window candidate or oracle tensors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCORING_PATH = (
    REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py")
SCHEMA = "b210_history_modal_controller_loao_v1"
EXPERIMENT = "ttdo_p0f_history_only_coherent_modal_correction"
EVIDENCE_CLASS = "exploratory_clean716_controller_only_loao"
EXPECTED_SHAPE = (716, 20, 32, 64, 2)
EXPECTED_AOAS = (0, 5, 10, 15, 20)
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_MANIFEST_FILE_SHA256 = (
    "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958")
EXPECTED_MANIFEST_CANONICAL_SHA256 = (
    "c3443b99f290dbfb08b0b1e6e8eaa2bc84e37f552b087c4d3a7ecbe57cff0f20")
EXPECTED_TARGET_RAW_SHA256 = (
    "311dac57a255a835292b1af21ce352727e69b3e42d825f07e6b557b8f5bfbd3c")
EXPECTED_PREDICTION_RAW_SHA256 = (
    "cfbb4d3b05d40d27f6bc57c43083d29288ebe21b086acbae285948491dec0dde")
EXPECTED_FILE_SHA256 = {
    "prediction": "4e373860669ebaa4368833cbf705bf07452fee7fae0d5682d2370d9ceab5c075",
    "target": "533d913297b969613db8ebb3e41a4cb519c207c4270b11fb6bc96be4ff8a622b",
    "aoa": "3610334507b7b876f59efbebb99fc84639b66d02af47f5ffc694e84688cdfe4e",
    "fold": "65dc5ee31a4ff26cc67fbc6881827f40ef52adee320adf9b2bd496b5c7947666",
}
RANK = 32
RIDGE_LAMBDA = 1.0
STD_FLOOR = 1.0e-6
CLIP_QUANTILES = (0.005, 0.995)
HORIZON_BLOCKS = ((0, 5), (5, 10), (10, 15), (15, 20))
WAKE_X_SLICE = (13, 38)
POINT_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_score": 0.10,
    "mvpe_score": 0.09,
}
REQUIRED_CACHE_FILES = (
    "prediction.npy", "target.npy", "aoa.npy", "fold.npy",
    "metadata.json", "split_manifest.json",
)

CONTRACT = {
    "schema": SCHEMA,
    "experiment": EXPERIMENT,
    "evidence_class": EVIDENCE_CLASS,
    "baseline_training_scope": "full_AoA_frozen_baseline",
    "controller_holdout": "five_leave_one_AoA_out_fits",
    "end_to_end_condition_holdout": False,
    "deployable_feature": (
        "concat(POD(zero-time-mean history), "
        "POD(zero-time-mean frozen prediction))"),
    "feature_dimensions": 64,
    "basis_rank": RANK,
    "ridge_lambda": RIDGE_LAMBDA,
    "pod_and_projection_dtype": "float32",
    "linear_controller_dtype": "float64",
    "standard_deviation": {"ddof": 0, "floor": STD_FLOOR},
    "standardized_target_clip": {
        "quantiles": list(CLIP_QUANTILES),
        "method": "linear",
    },
    "candidate": "frozen prediction + zero-time-mean reconstructed correction",
    "matched_oracle": (
        "frozen prediction + POD(zero-time-mean(target-frozen prediction)); "
        "target-visible and non-deployable"),
    "point_contribution": (
        "0.46*Rel-L2 + 0.10*TKE + 0.09*MVPE; pooled errors are "
        "scored before weighting"),
    "wake": {"x_start_inclusive": 13, "x_stop_exclusive": 38},
    "large_candidate_written": False,
    "pre_candidate_target_reads": (
        "integrity-only SHA, finite, and raw-H5 bitwise validation; no fitted "
        "object or hyperparameter receives held values"),
}


def _load_official_scoring():
    spec = importlib.util.spec_from_file_location(
        "realpde_track1_v9_history_modal_scoring", SCORING_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official scoring from {SCORING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL = _load_official_scoring()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--history-path", type=Path, required=True)
    parser.add_argument(
        "--history-seal", type=Path,
        help="defaults to prediction_seal.json next to --history-path")
    parser.add_argument(
        "--history-result", type=Path,
        help="defaults to result.json next to --history-path")
    parser.add_argument("--real-dir", type=Path, required=True)
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


def sha256_array_raw(value: np.ndarray) -> str:
    """Hash C-order array bytes, deliberately excluding the .npy header."""
    array = np.asarray(value)
    digest = hashlib.sha256()
    if array.flags.c_contiguous:
        view = memoryview(array).cast("B")
        for start in range(0, len(view), 8 * 1024 * 1024):
            digest.update(view[start:start + 8 * 1024 * 1024])
    else:
        iterator = np.nditer(
            array, flags=["external_loop", "buffered"], order="C",
            op_flags=["readonly"], buffersize=2 * 1024 * 1024)
        for chunk in iterator:
            digest.update(np.asarray(chunk).tobytes(order="C"))
    return digest.hexdigest()


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


def _nested(mapping: Mapping[str, object], *keys: str) -> object:
    value: object = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"missing required provenance field: {'.'.join(keys)}")
        value = value[key]
    return value


def _resolve_device(raw: str) -> torch.device:
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


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
        "h5py": h5py.__version__,
        "device_request": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available() else "none"),
        "pid": os.getpid(),
    }
    return "".join(f"{key}={value}\n" for key, value in rows.items())


def zero_time_mean(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 5 or array.shape[1] < 2 or array.shape[-1] != 2:
        raise ValueError("flow windows must have shape (N,T,H,W,2)")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("flow windows contain non-finite values")
    output = np.array(array, dtype=np.float32, copy=True, order="C")
    output -= output.mean(axis=1, keepdims=True, dtype=np.float32)
    return output


def fit_indices_for_held_aoa(
        aoa: np.ndarray, held_aoa: int,
        groups: Sequence[int] = EXPECTED_AOAS) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(aoa)
    if labels.ndim != 1 or set(labels.tolist()) != set(groups):
        raise ValueError("AoA labels differ from the frozen five-group contract")
    if held_aoa not in groups:
        raise ValueError("held AoA is outside the frozen contract")
    fit = np.flatnonzero(labels != held_aoa).astype(np.int64)
    held = np.flatnonzero(labels == held_aoa).astype(np.int64)
    if not len(fit) or not len(held) or np.any(labels[fit] == held_aoa):
        raise RuntimeError("invalid or leaking LOAO split")
    return fit, held


def dual_pod_basis(
        fit_target: np.ndarray, rank: int, device: torch.device,
        ) -> tuple[torch.Tensor, dict[str, object]]:
    """Fit uncentered dual POD to already-sliced, time-centred fit futures."""
    matrix = zero_time_mean(fit_target).reshape(len(fit_target), -1)
    if rank <= 0 or rank > min(matrix.shape):
        raise ValueError("basis rank exceeds fit matrix dimensions")
    x = torch.from_numpy(matrix).to(device=device, dtype=torch.float32)
    gram = x @ x.transpose(0, 1)
    gram = 0.5 * (gram + gram.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    leading = eigenvalues[:rank]
    tolerance = (
        torch.finfo(torch.float32).eps * matrix.shape[0]
        * torch.clamp(eigenvalues[0], min=1.0))
    if bool((leading <= tolerance).any()):
        raise RuntimeError("fit target has insufficient numerical POD rank")
    raw = (eigenvectors[:, :rank].transpose(0, 1) @ x) / torch.sqrt(
        leading)[:, None]
    q, _ = torch.linalg.qr(raw.transpose(0, 1), mode="reduced")
    basis = q.transpose(0, 1).contiguous()
    orthogonality = basis @ basis.transpose(0, 1)
    identity = torch.eye(rank, device=device, dtype=torch.float32)
    total_energy = float(torch.clamp(eigenvalues, min=0.0).sum().item())
    report = {
        "fit_samples": int(len(fit_target)),
        "flattened_dimensions": int(matrix.shape[1]),
        "rank": int(rank),
        "fit_energy_fraction": float(
            torch.clamp(leading, min=0.0).sum().item()
            / max(total_energy, 1.0e-30)),
        "orthogonality_max_abs_error": float(torch.max(torch.abs(
            orthogonality - identity)).item()),
        "basis_raw_sha256": sha256_array_raw(
            basis.detach().cpu().numpy()),
    }
    del matrix, x, gram, eigenvalues, eigenvectors, raw, q
    return basis, report


def project_fluctuation(
        values: np.ndarray, basis: torch.Tensor, device: torch.device,
        batch_size: int, subtract: np.ndarray | None = None) -> np.ndarray:
    """Project already-sliced windows; all returned coefficients are FP64."""
    if batch_size <= 0:
        raise ValueError("projection batch size must be positive")
    if subtract is not None and np.shape(subtract) != np.shape(values):
        raise ValueError("projection subtraction arrays do not align")
    coefficients = np.empty((len(values), basis.shape[0]), dtype=np.float64)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        block = np.asarray(values[start:stop], dtype=np.float32)
        if subtract is not None:
            block = block - np.asarray(subtract[start:stop], dtype=np.float32)
        flat = zero_time_mean(block).reshape(len(block), -1)
        tensor = torch.from_numpy(flat).to(device=device, dtype=torch.float32)
        projected = tensor @ basis.transpose(0, 1)
        coefficients[start:stop] = projected.detach().cpu().numpy().astype(
            np.float64)
    return coefficients


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean


def fit_standardizer(fit_values: np.ndarray) -> Standardizer:
    values = np.asarray(fit_values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("fit standardizer input must be a finite non-empty matrix")
    mean = values.mean(axis=0, dtype=np.float64)
    standard_deviation = values.std(axis=0, ddof=0, dtype=np.float64)
    scale = np.maximum(standard_deviation, STD_FLOOR)
    return Standardizer(mean=mean, scale=scale)


@dataclass(frozen=True)
class ModalController:
    basis: torch.Tensor
    feature_scaler: Standardizer
    target_scaler: Standardizer
    ridge_weight: np.ndarray
    clip_lower_standardized: np.ndarray
    clip_upper_standardized: np.ndarray
    report: dict[str, object]


def fit_modal_controller(
        *, fit_history: np.ndarray, fit_prediction: np.ndarray,
        fit_target: np.ndarray, rank: int, ridge_lambda: float,
        device: torch.device, projection_batch_size: int,
        ) -> ModalController:
    """Fit only from pre-sliced fit arrays; there is no index/held argument."""
    shapes = {tuple(np.shape(x)) for x in (
        fit_history, fit_prediction, fit_target)}
    if len(shapes) != 1:
        raise ValueError("fit history/prediction/target arrays do not align")
    if ridge_lambda != RIDGE_LAMBDA:
        raise ValueError(f"formal ridge lambda must remain {RIDGE_LAMBDA}")
    basis, basis_report = dual_pod_basis(fit_target, rank, device)
    history_coeff = project_fluctuation(
        fit_history, basis, device, projection_batch_size)
    baseline_coeff = project_fluctuation(
        fit_prediction, basis, device, projection_batch_size)
    target_coeff = project_fluctuation(
        fit_target, basis, device, projection_batch_size,
        subtract=fit_prediction)
    feature = np.concatenate((history_coeff, baseline_coeff), axis=1)
    feature_scaler = fit_standardizer(feature)
    target_scaler = fit_standardizer(target_coeff)
    x = feature_scaler.transform(feature).astype(np.float64, copy=False)
    y = target_scaler.transform(target_coeff).astype(np.float64, copy=False)
    normal = x.T @ x + ridge_lambda * np.eye(
        x.shape[1], dtype=np.float64)
    right = x.T @ y
    condition_number = float(np.linalg.cond(normal))
    weight = np.linalg.solve(normal, right)
    clip_lower, clip_upper = np.quantile(
        y, CLIP_QUANTILES, axis=0, method="linear")
    report = {
        "fit_samples": int(len(feature)),
        "basis": basis_report,
        "feature_dimensions": int(feature.shape[1]),
        "target_dimensions": int(target_coeff.shape[1]),
        "numeric_dtype": "float64",
        "feature_standardizer": {
            "ddof": 0,
            "floor": STD_FLOOR,
            "mean_raw_sha256": sha256_array_raw(feature_scaler.mean),
            "scale_raw_sha256": sha256_array_raw(feature_scaler.scale),
            "floored_dimensions": int(np.sum(
                feature.std(axis=0, ddof=0) < STD_FLOOR)),
        },
        "target_standardizer": {
            "ddof": 0,
            "floor": STD_FLOOR,
            "mean_raw_sha256": sha256_array_raw(target_scaler.mean),
            "scale_raw_sha256": sha256_array_raw(target_scaler.scale),
            "floored_dimensions": int(np.sum(
                target_coeff.std(axis=0, ddof=0) < STD_FLOOR)),
        },
        "ridge": {
            "lambda": ridge_lambda,
            "normal_equation_condition_number": condition_number,
            "weight_frobenius_norm": float(np.linalg.norm(weight)),
            "weight_raw_sha256": sha256_array_raw(weight),
        },
        "clipping": {
            "space": "fit_target_standardized_coefficients",
            "quantiles": list(CLIP_QUANTILES),
            "method": "linear",
            "lower_raw_sha256": sha256_array_raw(clip_lower),
            "upper_raw_sha256": sha256_array_raw(clip_upper),
        },
        "held_target_access_counter": {
            "basis": 0,
            "feature_scaler": 0,
            "target_scaler": 0,
            "ridge": 0,
            "clipping": 0,
        },
    }
    return ModalController(
        basis=basis,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        ridge_weight=weight,
        clip_lower_standardized=clip_lower,
        clip_upper_standardized=clip_upper,
        report=report,
    )


def predict_modal_coefficients(
        controller: ModalController, *, history: np.ndarray,
        prediction: np.ndarray, device: torch.device,
        projection_batch_size: int,
        ) -> tuple[np.ndarray, dict[str, object]]:
    """Predict without any target argument."""
    if np.shape(history) != np.shape(prediction):
        raise ValueError("history and prediction do not align")
    history_coeff = project_fluctuation(
        history, controller.basis, device, projection_batch_size)
    baseline_coeff = project_fluctuation(
        prediction, controller.basis, device, projection_batch_size)
    feature = np.concatenate((history_coeff, baseline_coeff), axis=1)
    x = controller.feature_scaler.transform(feature)
    standardized_unclipped = x @ controller.ridge_weight
    standardized = np.clip(
        standardized_unclipped,
        controller.clip_lower_standardized,
        controller.clip_upper_standardized)
    coefficients = controller.target_scaler.inverse(standardized)
    clipped = standardized != standardized_unclipped
    return coefficients, {
        "samples": int(len(coefficients)),
        "coefficient_values": int(coefficients.size),
        "clipped_values": int(np.sum(clipped)),
        "clipped_fraction": float(np.mean(clipped)),
        "standardized_prediction_min": float(np.min(standardized)),
        "standardized_prediction_max": float(np.max(standardized)),
    }


def reconstruct_correction(
        coefficients: np.ndarray, basis: torch.Tensor,
        output_shape: Sequence[int], device: torch.device,
        batch_size: int) -> np.ndarray:
    shape = tuple(int(x) for x in output_shape)
    if len(shape) != 4 or shape[-1] != 2:
        raise ValueError("one correction shape must be (T,H,W,2)")
    if coefficients.shape != (len(coefficients), basis.shape[0]):
        raise ValueError("coefficient matrix does not match basis")
    if int(np.prod(shape)) != basis.shape[1]:
        raise ValueError("basis dimensions do not match output shape")
    output = np.empty((len(coefficients), *shape), dtype=np.float32)
    for start in range(0, len(coefficients), batch_size):
        stop = min(start + batch_size, len(coefficients))
        tensor = torch.from_numpy(
            np.asarray(coefficients[start:stop], dtype=np.float32)).to(
                device=device)
        correction = (tensor @ basis).reshape(stop - start, *shape)
        correction -= correction.mean(dim=1, keepdim=True)
        output[start:stop] = correction.detach().cpu().numpy()
    return output


def official_error_triples(
        prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    OFFICIAL.validate_shapes(prediction, target)
    channels = OFFICIAL.measured_channels(target)
    if channels != 2:
        raise ValueError(f"expected two measured UV channels, got {channels}")
    return np.stack((
        OFFICIAL.rel_l2_per_sample(prediction, target, channels),
        OFFICIAL.tke_rel_l2_per_sample(prediction, target, channels),
        OFFICIAL.mvpe_rel_l2_per_sample(prediction, target),
    ), axis=1).astype(np.float64, copy=False)


def metrics_from_errors(errors: np.ndarray) -> dict[str, object]:
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("official error matrix must have non-empty shape (N,3)")
    means = values.mean(axis=0, dtype=np.float64)
    scores = {
        "rel_l2_score": OFFICIAL.score_error(float(means[0])),
        "tke_score": OFFICIAL.score_error(float(means[1])),
        "mvpe_score": OFFICIAL.score_error(float(means[2])),
    }
    contribution = float(sum(
        POINT_WEIGHTS[key] * scores[key] for key in POINT_WEIGHTS))
    return {
        "samples": int(len(values)),
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
        baseline_errors: np.ndarray, candidate_errors: np.ndarray,
        candidate_label: str) -> dict[str, object]:
    baseline = metrics_from_errors(baseline_errors)
    candidate = metrics_from_errors(candidate_errors)
    keys = (
        "rel_l2_score", "tke_score", "mvpe_score",
        "point_final_contribution_46_10_9",
        "point_score_normalized_46_10_9",
    )
    return {
        "aggregation": "pooled_per_sample_errors_then_official_score",
        "baseline": baseline,
        candidate_label: candidate,
        f"delta_{candidate_label}_minus_baseline": {
            key: float(candidate[key] - baseline[key]) for key in keys
        },
    }


def per_sample_sse(
        prediction: np.ndarray, target: np.ndarray,
        *, time_slice: tuple[int, int] | None = None,
        x_slice: tuple[int, int] | None = None) -> np.ndarray:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("SSE arrays must be aligned 5D flow windows")
    if time_slice is not None:
        prediction = prediction[:, time_slice[0]:time_slice[1]]
        target = target[:, time_slice[0]:time_slice[1]]
    if x_slice is not None:
        prediction = prediction[:, :, :, x_slice[0]:x_slice[1], :]
        target = target[:, :, :, x_slice[0]:x_slice[1], :]
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    return np.sum(difference * difference, axis=(1, 2, 3, 4), dtype=np.float64)


def sse_comparison(
        baseline: np.ndarray, candidate: np.ndarray,
        candidate_label: str) -> dict[str, object]:
    base = np.asarray(baseline, dtype=np.float64)
    trial = np.asarray(candidate, dtype=np.float64)
    if base.ndim != 1 or base.shape != trial.shape or not len(base):
        raise ValueError("SSE vectors must be non-empty and aligned")
    base_sum = float(base.sum(dtype=np.float64))
    trial_sum = float(trial.sum(dtype=np.float64))
    reduction = base_sum - trial_sum
    return {
        "samples": int(len(base)),
        "baseline_uv_sse": base_sum,
        f"{candidate_label}_uv_sse": trial_sum,
        f"{candidate_label}_uv_sse_reduction": reduction,
        f"{candidate_label}_uv_sse_reduction_fraction": (
            reduction / base_sum if base_sum > 0.0 else 0.0),
    }


def coefficient_diagnostics(
        target: np.ndarray, prediction: np.ndarray,
        groups: np.ndarray | None = None) -> dict[str, object]:
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 2 or truth.shape != estimate.shape or not len(truth):
        raise ValueError("coefficient matrices must be non-empty and aligned")
    if groups is None:
        centered = truth - truth.mean(axis=0, keepdims=True, dtype=np.float64)
        r2_centering = "pooled_column_mean"
    else:
        labels = np.asarray(groups)
        if labels.shape != (len(truth),):
            raise ValueError("coefficient diagnostic groups do not align")
        centered = np.empty_like(truth)
        for label in np.unique(labels):
            rows = labels == label
            centered[rows] = truth[rows] - truth[rows].mean(
                axis=0, keepdims=True, dtype=np.float64)
        r2_centering = "within_controller_held_AoA_column_mean"
    residual_sse = float(np.sum((estimate - truth) ** 2, dtype=np.float64))
    centered_sse = float(np.sum(centered ** 2, dtype=np.float64))
    truth_norm = np.linalg.norm(truth, axis=1)
    estimate_norm = np.linalg.norm(estimate, axis=1)
    valid = (truth_norm > 1.0e-12) & (estimate_norm > 1.0e-12)
    cosine = np.sum(truth[valid] * estimate[valid], axis=1) / (
        truth_norm[valid] * estimate_norm[valid])
    ratios = estimate_norm[truth_norm > 1.0e-12] / truth_norm[
        truth_norm > 1.0e-12]
    return {
        "samples": int(len(truth)),
        "dimensions": int(truth.shape[1]),
        "r2": (
            1.0 - residual_sse / centered_sse
            if centered_sse > 0.0 else None),
        "r2_centering": r2_centering,
        "residual_sse": residual_sse,
        "target_centered_sse": centered_sse,
        "cosine_valid_samples": int(np.sum(valid)),
        "cosine_mean": float(np.mean(cosine)) if len(cosine) else None,
        "cosine_median": float(np.median(cosine)) if len(cosine) else None,
        "frobenius_norm_ratio": float(
            np.linalg.norm(estimate) / max(np.linalg.norm(truth), 1.0e-30)),
        "sample_norm_ratio_mean": (
            float(np.mean(ratios)) if len(ratios) else None),
        "sample_norm_ratio_median": (
            float(np.median(ratios)) if len(ratios) else None),
    }


def temporal_mean_drift(
        baseline: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    if baseline.shape != candidate.shape or baseline.ndim != 5:
        raise ValueError("mean-drift arrays must be aligned flow windows")
    drift = (
        candidate.astype(np.float64).mean(axis=1)
        - baseline.astype(np.float64).mean(axis=1))
    return {
        "maximum_absolute": float(np.max(np.abs(drift))),
        "root_mean_square": float(np.sqrt(np.mean(drift ** 2))),
        "mean_absolute": float(np.mean(np.abs(drift))),
    }


def _empty_diagnostic_vectors(samples: int) -> dict[str, object]:
    branches = ("baseline", "history_controller", "matched_oracle")
    return {
        "official_errors": {
            name: np.empty((samples, 3), dtype=np.float64) for name in branches},
        "overall_sse": {
            name: np.empty(samples, dtype=np.float64) for name in branches},
        "wake_sse": {
            name: np.empty(samples, dtype=np.float64) for name in branches},
        "horizon_sse": {
            name: np.empty((samples, len(HORIZON_BLOCKS)), dtype=np.float64)
            for name in branches},
        "wake_horizon_sse": {
            name: np.empty((samples, len(HORIZON_BLOCKS)), dtype=np.float64)
            for name in branches},
        "coefficient_target": np.empty((samples, RANK), dtype=np.float64),
        "coefficient_prediction": np.empty((samples, RANK), dtype=np.float64),
    }


def fill_diagnostic_vectors(
        vectors: dict[str, object], rows: np.ndarray, baseline: np.ndarray,
        controller: np.ndarray, oracle: np.ndarray, target: np.ndarray,
        coefficient_target: np.ndarray,
        coefficient_prediction: np.ndarray) -> None:
    branches = {
        "baseline": baseline,
        "history_controller": controller,
        "matched_oracle": oracle,
    }
    for name, values in branches.items():
        vectors["official_errors"][name][rows] = official_error_triples(
            values, target)
        vectors["overall_sse"][name][rows] = per_sample_sse(values, target)
        vectors["wake_sse"][name][rows] = per_sample_sse(
            values, target, x_slice=WAKE_X_SLICE)
        for block_index, block in enumerate(HORIZON_BLOCKS):
            vectors["horizon_sse"][name][rows, block_index] = per_sample_sse(
                values, target, time_slice=block)
            vectors["wake_horizon_sse"][name][rows, block_index] = (
                per_sample_sse(
                    values, target, time_slice=block, x_slice=WAKE_X_SLICE))
    vectors["coefficient_target"][rows] = coefficient_target
    vectors["coefficient_prediction"][rows] = coefficient_prediction


def summarize_rows(
        vectors: dict[str, object], rows: np.ndarray,
        aoa_labels: np.ndarray) -> dict[str, object]:
    selected = np.asarray(rows, dtype=np.int64)
    official = vectors["official_errors"]
    result: dict[str, object] = {
        "samples": int(len(selected)),
        "history_controller": {
            "official_point_metrics": metric_comparison(
                official["baseline"][selected],
                official["history_controller"][selected],
                "history_controller"),
            "overall_sse": sse_comparison(
                vectors["overall_sse"]["baseline"][selected],
                vectors["overall_sse"]["history_controller"][selected],
                "history_controller"),
            "wake_sse": sse_comparison(
                vectors["wake_sse"]["baseline"][selected],
                vectors["wake_sse"]["history_controller"][selected],
                "history_controller"),
        },
        "matched_target_visible_oracle": {
            "deployable": False,
            "official_point_metrics": metric_comparison(
                official["baseline"][selected],
                official["matched_oracle"][selected],
                "matched_oracle"),
            "overall_sse": sse_comparison(
                vectors["overall_sse"]["baseline"][selected],
                vectors["overall_sse"]["matched_oracle"][selected],
                "matched_oracle"),
            "wake_sse": sse_comparison(
                vectors["wake_sse"]["baseline"][selected],
                vectors["wake_sse"]["matched_oracle"][selected],
                "matched_oracle"),
        },
        "coefficient_diagnostics": coefficient_diagnostics(
            vectors["coefficient_target"][selected],
            vectors["coefficient_prediction"][selected],
            groups=aoa_labels[selected]),
    }
    for vector_branch, result_branch in (
            ("history_controller", "history_controller"),
            ("matched_oracle", "matched_target_visible_oracle")):
        result[result_branch]["horizon_blocks"] = {}
        for block_index, (start, stop) in enumerate(HORIZON_BLOCKS):
            result[result_branch]["horizon_blocks"][f"{start}_{stop}"] = {
                "overall": sse_comparison(
                    vectors["horizon_sse"]["baseline"][selected, block_index],
                    vectors["horizon_sse"][vector_branch][selected, block_index],
                    vector_branch),
                "wake": sse_comparison(
                    vectors["wake_horizon_sse"]["baseline"][
                        selected, block_index],
                    vectors["wake_horizon_sse"][vector_branch][
                        selected, block_index],
                    vector_branch),
            }
    return result


def validate_raw_h5_alignment(
        *, real_dir: Path, manifest: object, history: np.ndarray,
        target: np.ndarray, aoa: np.ndarray) -> dict[str, object]:
    """Stream every manifest window from raw H5 and require exact ordering.

    Manifest order, then each item's ``val_starts`` order, is the sole row
    order.  Equality is bitwise after the frozen float32/downsample decode.
    """
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("split manifest must be a non-empty trajectory list")
    if not real_dir.is_dir():
        raise FileNotFoundError(f"raw Real directory does not exist: {real_dir}")
    manifest_names = [str(_nested(item, "file")) for item in manifest]
    if len(manifest_names) != len(set(manifest_names)):
        raise ValueError("split manifest contains duplicate trajectory files")
    selected_paths = []
    for name in manifest_names:
        path = real_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"manifest H5 is missing: {path}")
        selected_paths.append(path)
    sorted_selected = [
        path.name for path in sorted(real_dir.glob("*.h5"))
        if path.name in set(manifest_names)]
    if sorted_selected != manifest_names:
        raise ValueError("manifest trajectory order differs from sorted raw loader")

    history_digest = hashlib.sha256()
    target_digest = hashlib.sha256()
    row = 0
    maximum_history_abs_difference = 0.0
    maximum_target_abs_difference = 0.0
    for item, path in zip(manifest, selected_paths):
        if not isinstance(item, Mapping):
            raise ValueError("split manifest trajectory entry must be an object")
        starts = _nested(item, "val_starts")
        if not isinstance(starts, list) or not all(
                isinstance(start, int) for start in starts):
            raise ValueError("manifest val_starts must be an integer list")
        with h5py.File(path, "r") as handle:
            for dataset in ("u", "v", "re", "aoa"):
                if dataset not in handle:
                    raise ValueError(f"{path.name} is missing H5 dataset {dataset}")
            if int(handle["u"].shape[0]) != int(_nested(item, "frames")):
                raise ValueError(f"manifest frame count differs for {path.name}")
            if int(np.asarray(handle["aoa"]).reshape(-1)[0]) != int(
                    _nested(item, "aoa")):
                raise ValueError(f"manifest AoA differs for {path.name}")
            if int(np.asarray(handle["re"]).reshape(-1)[0]) != int(
                    _nested(item, "re")):
                raise ValueError(f"manifest Reynolds number differs for {path.name}")
            for start in starts:
                stop = start + 40
                if start < 0 or stop > handle["u"].shape[0]:
                    raise ValueError(f"manifest window leaves {path.name} bounds")
                u = np.asarray(
                    handle["u"][start:stop, ::2, ::2], dtype=np.float32)
                v = np.asarray(
                    handle["v"][start:stop, ::2, ::2], dtype=np.float32)
                decoded = np.stack((u, v), axis=-1)
                raw_history = np.ascontiguousarray(decoded[:20])
                raw_target = np.ascontiguousarray(decoded[20:40])
                if row >= len(history):
                    raise ValueError("manifest produces more rows than cached arrays")
                cached_history = np.asarray(history[row])
                cached_target = np.asarray(target[row])
                history_difference = float(np.max(np.abs(
                    raw_history.astype(np.float64)
                    - cached_history.astype(np.float64))))
                target_difference = float(np.max(np.abs(
                    raw_target.astype(np.float64)
                    - cached_target.astype(np.float64))))
                maximum_history_abs_difference = max(
                    maximum_history_abs_difference, history_difference)
                maximum_target_abs_difference = max(
                    maximum_target_abs_difference, target_difference)
                if not np.array_equal(raw_history, cached_history):
                    raise ValueError(
                        f"history row {row} differs from raw {path.name}:{start}")
                if not np.array_equal(raw_target, cached_target):
                    raise ValueError(
                        f"target row {row} differs from raw {path.name}:{start}")
                if int(aoa[row]) != int(_nested(item, "aoa")):
                    raise ValueError(
                        f"cache AoA row {row} differs from manifest {path.name}")
                history_digest.update(memoryview(raw_history).cast("B"))
                target_digest.update(memoryview(raw_target).cast("B"))
                row += 1
    if row != len(history) or row != len(target) or row != len(aoa):
        raise ValueError("manifest row count differs from cached arrays")
    raw_history_sha = history_digest.hexdigest()
    raw_target_sha = target_digest.hexdigest()
    cached_history_sha = sha256_array_raw(history)
    cached_target_sha = sha256_array_raw(target)
    if raw_history_sha != cached_history_sha:
        raise ValueError("streamed raw-history SHA differs from history array SHA")
    if raw_target_sha != cached_target_sha:
        raise ValueError("streamed raw-target SHA differs from cache target SHA")
    return {
        "mode": "full_streaming_every_manifest_validation_window",
        "trajectory_order": "sorted Real H5 intersect manifest filenames",
        "window_order": "manifest list then val_starts list",
        "decode": "float32(stack(u[:,::2,::2],v[:,::2,::2]),last_axis)",
        "comparison": "bitwise_array_equal_per_window",
        "trajectories": int(len(manifest)),
        "windows": int(row),
        "history_raw_array_sha256": raw_history_sha,
        "target_raw_array_sha256": raw_target_sha,
        "maximum_history_absolute_difference": maximum_history_abs_difference,
        "maximum_target_absolute_difference": maximum_target_abs_difference,
        "passed": True,
    }


def validate_cached_labels(
        *, manifest: object, metadata: Mapping[str, object],
        aoa: np.ndarray, fold: np.ndarray) -> dict[str, object]:
    """Bind cached AoA/fold rows to manifest windows and metadata assignments."""
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("split manifest must be a non-empty list")
    assignments = metadata.get("fold_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("cache metadata has no fold assignments")
    by_file: dict[str, Mapping[str, object]] = {}
    for item in assignments:
        if not isinstance(item, Mapping):
            raise ValueError("fold assignment must be an object")
        filename = str(_nested(item, "file"))
        if filename in by_file:
            raise ValueError("duplicate metadata fold assignment")
        by_file[filename] = item
    expected_aoa: list[int] = []
    expected_fold: list[int] = []
    for item in manifest:
        if not isinstance(item, Mapping):
            raise ValueError("manifest item must be an object")
        filename = str(_nested(item, "file"))
        if filename not in by_file:
            raise ValueError(f"missing fold assignment for {filename}")
        assignment = by_file[filename]
        starts = _nested(item, "val_starts")
        if not isinstance(starts, list):
            raise ValueError("manifest val_starts must be a list")
        windows = len(starts)
        if int(_nested(assignment, "windows")) != windows:
            raise ValueError(f"fold-assignment window count differs for {filename}")
        manifest_aoa = int(_nested(item, "aoa"))
        if int(_nested(assignment, "aoa")) != manifest_aoa:
            raise ValueError(f"fold-assignment AoA differs for {filename}")
        if int(_nested(assignment, "re")) != int(_nested(item, "re")):
            raise ValueError(f"fold-assignment Reynolds differs for {filename}")
        expected_aoa.extend([manifest_aoa] * windows)
        expected_fold.extend([int(_nested(assignment, "fold"))] * windows)
    expected_aoa_array = np.asarray(expected_aoa, dtype=np.asarray(aoa).dtype)
    expected_fold_array = np.asarray(expected_fold, dtype=np.asarray(fold).dtype)
    if not np.array_equal(np.asarray(aoa), expected_aoa_array):
        raise ValueError("cached AoA rows differ from manifest order")
    if not np.array_equal(np.asarray(fold), expected_fold_array):
        raise ValueError("cached fold rows differ from metadata/manifest order")
    return {
        "windows": int(len(expected_aoa)),
        "trajectories": int(len(manifest)),
        "assignment_files": int(len(by_file)),
        "aoa_rows_raw_sha256": sha256_array_raw(np.asarray(aoa)),
        "fold_rows_raw_sha256": sha256_array_raw(np.asarray(fold)),
        "passed": True,
    }


def validate_inputs(
        *, cache_dir: Path, history_path: Path, history_seal_path: Path,
        history_result_path: Path, real_dir: Path,
        expected_shape: tuple[int, ...] = EXPECTED_SHAPE,
        expected_manifest_file_sha256: str = EXPECTED_MANIFEST_FILE_SHA256,
        expected_manifest_canonical_sha256: str = EXPECTED_MANIFEST_CANONICAL_SHA256,
        expected_target_raw_sha256: str = EXPECTED_TARGET_RAW_SHA256,
        expected_prediction_raw_sha256: str = EXPECTED_PREDICTION_RAW_SHA256,
        expected_file_sha256: Mapping[str, str] = EXPECTED_FILE_SHA256,
        ) -> tuple[dict[str, Path], dict[str, object], dict[str, np.ndarray]]:
    paths = {
        name.removesuffix(".npy").removesuffix(".json"): cache_dir / name
        for name in REQUIRED_CACHE_FILES
    }
    paths.update({
        "history": history_path,
        "history_seal": history_seal_path,
        "history_result": history_result_path,
    })
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required audit inputs are missing: {missing}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    seal = json.loads(paths["history_seal"].read_text(encoding="utf-8"))
    history_result = json.loads(
        paths["history_result"].read_text(encoding="utf-8"))
    manifest_file_sha = sha256_file(paths["split_manifest"])
    manifest_canonical_sha = canonical_sha256(manifest)
    if manifest_file_sha != expected_manifest_file_sha256:
        raise ValueError("split manifest file SHA differs from frozen contract")
    if manifest_canonical_sha != expected_manifest_canonical_sha256:
        raise ValueError("split manifest canonical-JSON SHA differs from contract")
    if metadata.get("schema") != "b210_archive_calibration_cache_v1":
        raise ValueError("unexpected clean716 cache metadata schema")
    if tuple(metadata.get("shape", ())) != expected_shape:
        raise ValueError("cache metadata shape differs from frozen contract")
    if metadata.get("manifest_sha256") != manifest_canonical_sha:
        raise ValueError("cache metadata canonical manifest SHA disagrees")
    for name, expected_sha in expected_file_sha256.items():
        if sha256_file(paths[name]) != expected_sha:
            raise ValueError(f"{name} file SHA differs from frozen contract")

    result_manifest_values = {
        "data_contract": _nested(
            history_result, "data_contract", "split_manifest_sha256"),
        "identity": _nested(
            history_result, "identity", "split_manifest_sha256"),
        "g1_artifact": _nested(
            history_result, "identity", "g1_artifact_sha256",
            "split_manifest.json"),
    }
    if any(value != manifest_file_sha for value in result_manifest_values.values()):
        raise ValueError("G2 result manifest file SHA does not bind to main cache")
    embedded_seal = _nested(history_result, "prediction_seal")
    if embedded_seal != seal:
        raise ValueError("prediction_seal file differs from G2 result embedding")
    if "split_manifest_sha256" in seal and (
            seal["split_manifest_sha256"] != manifest_file_sha):
        raise ValueError("prediction seal manifest SHA disagrees with main cache")

    arrays = {
        "prediction": np.load(paths["prediction"], mmap_mode="r"),
        "target": np.load(paths["target"], mmap_mode="r"),
        "aoa": np.load(paths["aoa"], mmap_mode="r"),
        "fold": np.load(paths["fold"], mmap_mode="r"),
        "history": np.load(paths["history"], mmap_mode="r"),
    }
    for name in ("prediction", "target", "history"):
        array = arrays[name]
        if tuple(array.shape) != expected_shape or array.dtype != np.float32:
            raise ValueError(f"{name} shape/dtype differs from frozen contract")
    for name in ("aoa", "fold"):
        if tuple(arrays[name].shape) != (expected_shape[0],):
            raise ValueError(f"{name} labels do not align with frozen arrays")
    label_alignment = validate_cached_labels(
        manifest=manifest, metadata=metadata, aoa=arrays["aoa"],
        fold=arrays["fold"])

    history_meta = _nested(seal, "histories")
    if not isinstance(history_meta, Mapping):
        raise ValueError("history seal metadata must be an object")
    history_file_sha = sha256_file(history_path)
    if _nested(history_meta, "sha256") != history_file_sha:
        raise ValueError("history file SHA differs from prediction seal")
    if tuple(_nested(history_meta, "shape")) != expected_shape:
        raise ValueError("history seal shape differs from frozen contract")
    if _nested(history_meta, "dtype") != "float32":
        raise ValueError("history seal dtype differs from frozen contract")
    if int(_nested(history_meta, "windows")) != expected_shape[0]:
        raise ValueError("history seal window count differs")
    if int(_nested(history_meta, "history_frames_per_window")) != 20:
        raise ValueError("history seal frame count differs")
    if int(_nested(history_meta, "future_target_values_decoded")) != 0:
        raise ValueError("history generation seal reports future target access")

    target_raw_sha = sha256_array_raw(arrays["target"])
    prediction_raw_sha = sha256_array_raw(arrays["prediction"])
    result_target_sha = _nested(
        history_result, "data_contract", "target_uv_sha256")
    if (target_raw_sha != expected_target_raw_sha256
            or result_target_sha != target_raw_sha):
        raise ValueError("cache target raw-array SHA differs from G2 contract")
    if prediction_raw_sha != expected_prediction_raw_sha256:
        raise ValueError("baseline prediction raw-array SHA differs from contract")
    raw_alignment = validate_raw_h5_alignment(
        real_dir=real_dir, manifest=manifest, history=arrays["history"],
        target=arrays["target"], aoa=arrays["aoa"])
    provenance = {
        "manifest": {
            "file_sha256": manifest_file_sha,
            "canonical_json_sha256": manifest_canonical_sha,
            "metadata_uses": "canonical_json_sha256",
            "g2_result_uses": "file_sha256",
            "g2_result_references": result_manifest_values,
        },
        "history": {
            "file_sha256": history_file_sha,
            "raw_array_sha256": sha256_array_raw(arrays["history"]),
            "seal_canonical_sha256": canonical_sha256(seal),
            "seal_file_sha256": sha256_file(history_seal_path),
            "seal_binding": "exact JSON object embedded in G2 result",
            "shape": list(arrays["history"].shape),
        },
        "target": {
            "raw_array_sha256": target_raw_sha,
            "g2_result_raw_array_sha256": result_target_sha,
        },
        "frozen_baseline_prediction": {
            "raw_array_sha256": prediction_raw_sha,
            "file_sha256": expected_file_sha256["prediction"],
        },
        "cached_label_alignment": label_alignment,
        "raw_h5_alignment": raw_alignment,
    }
    return paths, provenance, arrays


def evaluate_gate(
        pooled_summary: Mapping[str, object],
        per_aoa_summary: Mapping[str, Mapping[str, object]],
        ) -> dict[str, object]:
    """Apply the one preregistered deployable-controller decision gate."""
    pooled_metrics = _nested(
        pooled_summary, "history_controller", "official_point_metrics")
    pooled_delta = _nested(
        pooled_metrics, "delta_history_controller_minus_baseline")
    if not isinstance(pooled_delta, Mapping):
        raise ValueError("pooled controller delta must be an object")
    held_deltas: dict[str, float] = {}
    for label, summary in per_aoa_summary.items():
        delta = _nested(
            summary, "history_controller", "official_point_metrics",
            "delta_history_controller_minus_baseline",
            "point_final_contribution_46_10_9")
        held_deltas[str(label)] = float(delta)
    positive = sum(value > 0.0 for value in held_deltas.values())
    worst = min(held_deltas.values())
    checks = {
        "pooled_delta_point_ge_plus_0p030": float(
            pooled_delta["point_final_contribution_46_10_9"]) >= 0.030,
        "positive_held_aoa_ge_4_of_5": positive >= 4,
        "worst_held_aoa_delta_point_ge_minus_0p010": worst >= -0.010,
        "pooled_delta_rel_ge_0": float(
            pooled_delta["rel_l2_score"]) >= 0.0,
        "pooled_delta_tke_ge_0": float(
            pooled_delta["tke_score"]) >= 0.0,
    }
    if all(checks.values()):
        decision = "GO_MODAL_CONTROLLER"
    elif float(pooled_delta["point_final_contribution_46_10_9"]) > 0.0:
        decision = "INCONCLUSIVE_MODAL_CONTROLLER"
    else:
        decision = "STOP_LINEAR_MODAL_CONTROLLER"
    return {
        "decision": decision,
        "all_preregistered_checks_passed": bool(all(checks.values())),
        "checks": checks,
        "pooled_delta": {
            key: float(pooled_delta[key]) for key in (
                "rel_l2_score", "tke_score", "mvpe_score",
                "point_final_contribution_46_10_9")
        },
        "held_aoa_delta_point": held_deltas,
        "positive_held_aoa": int(positive),
        "worst_held_aoa_delta_point": float(worst),
        "interpretation": (
            "controller-only LOAO transfer on a frozen full-AoA baseline; "
            "not end-to-end condition holdout"),
    }


def _input_sha_report(paths: Mapping[str, Path]) -> dict[str, object]:
    return {
        name: {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "file_sha256": sha256_file(path),
        }
        for name, path in sorted(paths.items())
    }


def _formal_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"formal output directory already contains files: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    device = _resolve_device(args.device)
    git = _git_report()
    if git["dirty"]:
        raise RuntimeError(
            "formal audit requires a clean committed Git worktree")
    if args.projection_batch_size <= 0:
        raise ValueError("projection batch size must be positive")
    history_seal = (
        args.history_seal
        if args.history_seal is not None
        else args.history_path.parent / "prediction_seal.json")
    history_result = (
        args.history_result
        if args.history_result is not None
        else args.history_path.parent / "result.json")
    _formal_output_dir(args.output_dir)
    command = shlex.join([sys.executable, *sys.argv])
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (args.output_dir / "environment.txt").write_text(
        _environment_text(device), encoding="utf-8")
    config = {
        "contract": CONTRACT,
        "arguments": {
            "cache_dir": str(args.cache_dir),
            "history_path": str(args.history_path),
            "history_seal": str(history_seal),
            "history_result": str(history_result),
            "real_dir": str(args.real_dir),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "projection_batch_size": int(args.projection_batch_size),
        },
        "git": git,
    }
    _write_json(args.output_dir / "config.json", config)

    paths, provenance, arrays = validate_inputs(
        cache_dir=args.cache_dir,
        history_path=args.history_path,
        history_seal_path=history_seal,
        history_result_path=history_result,
        real_dir=args.real_dir)
    prediction = arrays["prediction"]
    target = arrays["target"]
    history = arrays["history"]
    aoa = np.asarray(arrays["aoa"])
    fold = np.asarray(arrays["fold"])
    if set(aoa.tolist()) != set(EXPECTED_AOAS):
        raise ValueError("cache AoA labels differ from formal contract")
    if set(fold.tolist()) != set(EXPECTED_FOLDS):
        raise ValueError("cache fold labels differ from formal contract")
    if not all(np.all(np.isfinite(array)) for array in (
            prediction, target, history)):
        raise FloatingPointError("formal input arrays contain non-finite values")

    vectors = _empty_diagnostic_vectors(len(prediction))
    held_reports: dict[str, object] = {}
    with torch.no_grad():
        for held_aoa in EXPECTED_AOAS:
            fit_rows, held_rows = fit_indices_for_held_aoa(aoa, held_aoa)
            controller = fit_modal_controller(
                fit_history=np.asarray(history[fit_rows]),
                fit_prediction=np.asarray(prediction[fit_rows]),
                fit_target=np.asarray(target[fit_rows]),
                rank=RANK,
                ridge_lambda=RIDGE_LAMBDA,
                device=device,
                projection_batch_size=args.projection_batch_size)

            # This call is the deployable boundary: no held target is accepted.
            predicted_coefficients, prediction_report = (
                predict_modal_coefficients(
                    controller,
                    history=np.asarray(history[held_rows]),
                    prediction=np.asarray(prediction[held_rows]),
                    device=device,
                    projection_batch_size=args.projection_batch_size))
            correction = reconstruct_correction(
                predicted_coefficients, controller.basis,
                EXPECTED_SHAPE[1:], device, args.projection_batch_size)
            held_baseline = np.asarray(prediction[held_rows], dtype=np.float32)
            candidate = held_baseline + correction
            if not np.all(np.isfinite(candidate)):
                raise FloatingPointError("history controller produced non-finite flow")

            # Integrity checks above may read raw held bytes.  This is the first
            # model-dependent use of held future: scoring and the separately
            # labelled target-visible oracle begin only after candidate freeze.
            held_target = np.asarray(target[held_rows], dtype=np.float32)
            target_coefficients = project_fluctuation(
                held_target, controller.basis, device,
                args.projection_batch_size, subtract=held_baseline)
            oracle_correction = reconstruct_correction(
                target_coefficients, controller.basis,
                EXPECTED_SHAPE[1:], device, args.projection_batch_size)
            oracle = held_baseline + oracle_correction
            fill_diagnostic_vectors(
                vectors, held_rows, held_baseline, candidate, oracle,
                held_target, target_coefficients, predicted_coefficients)
            held_reports[str(held_aoa)] = {
                "held_aoa": int(held_aoa),
                "fit_samples": int(len(fit_rows)),
                "held_samples": int(len(held_rows)),
                "fit_indices_sha256": sha256_array_raw(fit_rows),
                "held_indices_sha256": sha256_array_raw(held_rows),
                "fit": controller.report,
                "prediction": prediction_report,
                "target_access_order": (
                    "integrity-only raw reads occurred before fitting; candidate "
                    "coefficients and tensor were finalized before held target "
                    "entered scoring/oracle projection"),
                "candidate_temporal_mean_drift": temporal_mean_drift(
                    held_baseline, candidate),
                "oracle_temporal_mean_drift": temporal_mean_drift(
                    held_baseline, oracle),
            }
            print(json.dumps({
                "event": "held_aoa_complete",
                "held_aoa": held_aoa,
                "fit_samples": len(fit_rows),
                "held_samples": len(held_rows),
                "elapsed_s": round(time.time() - started, 3),
            }), flush=True)
            del controller, correction, candidate, oracle_correction, oracle
            if device.type == "cuda":
                torch.cuda.empty_cache()

    all_rows = np.arange(len(prediction), dtype=np.int64)
    pooled = summarize_rows(vectors, all_rows, aoa)
    by_aoa = {
        str(value): summarize_rows(
            vectors, np.flatnonzero(aoa == value), aoa)
        for value in EXPECTED_AOAS
    }
    by_fold = {
        str(value): summarize_rows(
            vectors, np.flatnonzero(fold == value), aoa)
        for value in EXPECTED_FOLDS
    }
    gate = evaluate_gate(pooled, by_aoa)
    elapsed = time.time() - started
    result: dict[str, object] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "evidence_class": EVIDENCE_CLASS,
        "contract": CONTRACT,
        "git": git,
        "data_provenance": provenance,
        "input_identity": _input_sha_report(paths),
        "execution": {
            "elapsed_seconds": float(elapsed),
            "peak_rss_mib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            "device": str(device),
            "projection_batch_size": int(args.projection_batch_size),
            "large_candidate_written": False,
        },
        "held_controller_fits": held_reports,
        "pooled": pooled,
        "by_held_aoa": by_aoa,
        "by_cached_fold_diagnostic_only": by_fold,
        "preregistered_gate": gate,
        "access_audit": {
            "target_integrity_validation_rows_before_candidate": int(len(target)),
            "target_integrity_validation_allowed_operations": [
                "frozen raw-array SHA-256",
                "finite check",
                "bitwise raw-H5 alignment",
                "cached label alignment",
            ],
            "held_target_rows_entering_each_controller_fit": 0,
            "held_target_rows_entering_feature_scaler_ridge_clip_or_hyperparameters": 0,
            "held_target_first_model_dependent_use": (
                "after each held candidate tensor was finalized"),
            "held_target_access_after_candidate_for_scoring": int(len(target)),
            "held_target_access_after_candidate_for_matched_oracle": int(
                len(target)),
            "baseline_training_scope": "full_AoA",
            "end_to_end_loao_claim_allowed": False,
        },
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    identities = {
        "result.json": sha256_file(result_path),
        "config.json": sha256_file(args.output_dir / "config.json"),
        "command.txt": sha256_file(args.output_dir / "command.txt"),
        "environment.txt": sha256_file(args.output_dir / "environment.txt"),
        "script": sha256_file(Path(__file__).resolve()),
    }
    (args.output_dir / "sha256.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in identities.items()),
        encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "decision": gate["decision"],
        "pooled_delta": gate["pooled_delta"],
        "positive_held_aoa": gate["positive_held_aoa"],
        "worst_held_aoa_delta_point": gate[
            "worst_held_aoa_delta_point"],
        "elapsed_s": round(elapsed, 3),
        "result_sha256": identities["result.json"],
    }, sort_keys=True), flush=True)
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
