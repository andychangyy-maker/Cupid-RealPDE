#!/usr/bin/env python3
"""Bounded observation-gated A/B/C point blend with frozen P0 bounds.

This runner tests one preregistered mechanism only.  Eight fixed statistics of
the observed history predict two per-window scalar weights.  The weights blend
the already-computed A/B/C point predictions, so the candidate adds no model
forward.  Absolute P0 lower/upper arrays are reused bitwise and point outputs
are never clipped to those bounds.

The target-aware bounded optimizer is an oracle diagnostic, not deployable.
Only its two scalar labels are learned by one fixed ridge model.  Condition-fold
cross-fitting and true leave-one-AoA-out fitting are reported separately.  The
fold UQ checkpoints excluded their held fold from gradient training, but their
iteration was selected on that fold; consequently this remains a local
fixed-725 diagnostic and is not an official score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
import sys

for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_ensemble_uq_oof as uq_train  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
import evaluate_mean_expert_oof_blend as s1  # noqa: E402
from evaluate_flow_backbone_blend import (  # noqa: E402
    OFFICIAL,
    collect,
    load_model,
    score,
)
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


EVIDENCE_CLASS = "fixed725_condition5_and_true_loao_point_blend_screen"
AB_HEADROOM_EVIDENCE_CLASS = (
    "fixed725_target_visible_ab_scalar_and_four_horizon_headroom")
P0_ARCHIVE_SHA256 = (
    "4131dea445da0747dc1cdc49548d9afe7bc726620d6d4f62d925c79f66e09803")
P1_ARCHIVE_SHA256 = (
    "0dd0f66c8cc18b2ce83f2af05f64e1821b9df9fe0597d762eb2bfe05f42ff3a5")
S1_ARTIFACT_MANIFEST_SHA256 = (
    "10c40ed421e84785fdf7fbe2192eb2ebf053bb45e7f980995659ec2df23d3b71")
S1_EVALUATOR_SHA256 = (
    "62c6d9889b98542585d06f97a8d14122aef14ab669be75e445c743249ab130af")
P0_BOUNDS_SHA256 = {
    "lower": "d14aa4304e3e26829a5deede05f74315fbcc801f2eb169b30789a6eb363c8671",
    "upper": "b2926bec51dac7d3aff02f2523dc1dc204379bf1f5bf07160fe979087f672e45",
}
P1_SOURCE_MATCHED_OOF_REFERENCE = {
    "rel_l2_score": 96.01891691961723,
    "tke_score": 81.0143022300871,
    "mvpe_score": 96.72395178786445,
    "point_score": 91.65853948640763,
    "sps_score": 53.78283459174445,
}
P1_SOURCE_MATCHED_OOF_RESULT_SHA256 = (
    "8a4515a7cfd766d19a502aa3b45dc7a75340e43d80b28c8c6e1607ef119eb73f")
P1_DEPLOYMENT_REFERENCE = {
    "rel_l2_score": 96.01867100426779,
    "tke_score": 81.00767577318524,
    "mvpe_score": 96.7223531512805,
    "point_score": 91.65610886434555,
    "sps_score": 54.02429413375911,
}
P1_POINT_TOLERANCE = 5e-4

AOA_VALUES = (0, 5, 10, 15, 20)
FEATURE_NAMES = (
    "last_masked_mean_u",
    "last_masked_mean_v",
    "last_masked_std_u",
    "last_masked_std_v",
    "history_spatial_mean_u_temporal_std",
    "history_spatial_mean_v_temporal_std",
    "last5_frame_difference_u_rms",
    "last5_frame_difference_v_rms",
)
BASE_WEIGHT = 0.5
MIN_WEIGHT = 0.25
MAX_WEIGHT = 0.75
WEIGHT_RADIUS = 0.25
RIDGE_ALPHA = 1.0
ORACLE_WC_GRID_POINTS = 33
ORACLE_COORDINATE_STEPS = (0.125, 0.0625, 0.03125, 0.015625,
                           0.0078125, 0.00390625)
AB_HEADROOM_HORIZON_BLOCKS = ((0, 5), (5, 10), (10, 15), (15, 20))
AB_HEADROOM_LOW_GATE = 0.15
AB_HEADROOM_GO_GATE = 0.20
AB_HEADROOM_TKE_DELTA_FLOOR = -0.03
AB_HEADROOM_FIXED_MEAN_TOLERANCE = 1e-5
AB_HEADROOM_HYPOTHESIS = (
    "Fixed P1 leaves target-visible non-Time headroom in condition-dependent "
    "A/B zero-mean fluctuation routing, especially across four 5-step horizons.")
AB_HEADROOM_EXACT_BASELINE = (
    "The reconstructed fixed P1 point array (wB=0.5, C mean weight=0.5), "
    "with its temporal mean and the bitwise-frozen P0 absolute bounds held "
    "fixed for every candidate delta.")

# Empirical local Final-score shadow; the official aggregation is unpublished.
FINAL_SHADOW_DERIVATIVES = {
    "rel_l2_score": 0.47201126,
    "tke_score": 0.08586562,
    "mvpe_score": 0.09410214,
    "time_score": 0.10102825,
    "sps_score": 0.24699273,
}
SHADOW_DERIVATIVES = {
    key: FINAL_SHADOW_DERIVATIVES[key]
    for key in (
        "rel_l2_score", "tke_score", "mvpe_score", "sps_score")
}
ORACLE_SHADOW_GAIN_GATE = 0.18
POOLED_SHADOW_GAIN_GATE = 0.12
CONDITION_POSITIVE_MIN = 4
CONDITION_WORST_FLOOR = -0.03
LOAO_POSITIVE_MIN = 4
LOAO_WORST_FLOOR = -0.05
RECONSTRUCTION_BUDGET_SECONDS = 40.0 * 60.0
FORMAL_BUDGET_SECONDS = 10.0 * 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint-a", type=Path, required=True)
    parser.add_argument("--oof-dir-a", type=Path, required=True)
    parser.add_argument("--point-checkpoint-b", type=Path, required=True)
    parser.add_argument("--oof-dir-b", type=Path, required=True)
    parser.add_argument("--mean-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--p0-archive", type=Path, required=True)
    parser.add_argument("--p1-archive", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ab-headroom-only", action="store_true",
        help=(
            "stop after target-visible scalar and four-block A/B fluctuation "
            "headroom oracles; do not fit a router"))
    return parser.parse_args()


def current_s1_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "baseline_zip": args.p0_archive,
        "split_manifest": args.split_manifest,
        "real_stats": args.real_stats,
        "official_scorer": (
            REPO_ROOT / "kits/track1/realpde_t1_starting_kit_v9/scoring.py"),
        "evaluator_source": Path(s1.__file__),
        "point_checkpoint_a": args.point_checkpoint_a,
        "point_checkpoint_b": args.point_checkpoint_b,
        "mean_expert_checkpoint": args.mean_expert_checkpoint,
        "oof_summary_a": args.oof_dir_a / "summary.json",
        "oof_summary_b": args.oof_dir_b / "summary.json",
    }
    for fold in range(5):
        paths[f"oof_a_fold_{fold:02d}"] = (
            args.oof_dir_a / f"fold_{fold:02d}_best.pth")
        paths[f"oof_b_fold_{fold:02d}"] = (
            args.oof_dir_b / f"fold_{fold:02d}_best.pth")
    return paths


def validate_portable_s1_manifest(args: argparse.Namespace) -> dict:
    manifest_identity = s1.file_identity(args.artifact_manifest)
    if manifest_identity["sha256"] != S1_ARTIFACT_MANIFEST_SHA256:
        raise ValueError("S1 artifact manifest SHA-256 differs from contract")
    payload = json.loads(args.artifact_manifest.read_text(encoding="utf-8"))
    if (payload.get("schema_version") != 1
            or payload.get("experiment")
            != "track1_s1_point_bounds_decoupling"
            or payload.get("candidate_contract")
            != [list(row) for row in s1.S1_CANDIDATES]):
        raise ValueError("unexpected S1 artifact manifest schema")
    frozen = payload.get("artifacts", {})
    paths = current_s1_artifact_paths(args)
    if set(frozen) != set(paths):
        raise ValueError("S1 artifact manifest keys differ from contract")
    current = s1.artifact_identities(paths)
    differences = {}
    for name in sorted(paths):
        expected = frozen[name]
        actual = current[name]
        if (actual["bytes"] != expected["bytes"]
                or actual["sha256"] != expected["sha256"]):
            differences[name] = {
                "expected": {
                    "bytes": expected["bytes"], "sha256": expected["sha256"]},
                "actual": {
                    "bytes": actual["bytes"], "sha256": actual["sha256"]},
            }
    if differences:
        raise ValueError(f"S1 artifacts differ from manifest: {differences}")
    if current["evaluator_source"]["sha256"] != S1_EVALUATOR_SHA256:
        raise ValueError("S1 evaluator SHA-256 differs from contract")
    return {
        "manifest": manifest_identity,
        "path_portability_rule": "bytes_and_sha256_equal; absolute paths ignored",
        "artifacts": current,
    }


def checkpoint_member_identities(path: Path) -> dict[str, dict]:
    result = {}
    with zipfile.ZipFile(path) as archive:
        for member in ("model_a.pth", "model_b.pth", "model_c.pth"):
            digest = hashlib.sha256()
            with archive.open(member) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            info = archive.getinfo(member)
            result[member] = {
                "bytes": info.file_size, "sha256": digest.hexdigest()}
    return result


def validate_archive_blend(path: Path, expected_sha256: str,
                           expected_b: float, expected_c: float,
                           *, decoupled_p1: bool = False) -> dict:
    identity = s1.file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"archive SHA-256 differs from contract: {path}")
    config = s1.read_blend_config(path)
    if decoupled_p1:
        weights_match = (
            config.get("point_weight_b") == expected_b
            and config.get("point_mean_expert_weight") == expected_c
            and config.get("bounds_weight_b") == 0.625
            and config.get("bounds_mean_expert_weight") == 0.25
            and config.get("source_archive_sha256") == P0_ARCHIVE_SHA256)
    else:
        weights_match = (
            config.get("weight_b") == expected_b
            and config.get("mean_expert_weight") == expected_c)
    if (not weights_match or config.get("checkpoints")
            != ["model_a.pth", "model_b.pth", "model_c.pth"]):
        raise ValueError(f"archive blend_config differs from contract: {path}")
    return {**identity, "blend_config": config,
            "checkpoint_members": checkpoint_member_identities(path)}


def validate_fixed_contract(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    if (args.batch_size != 16 or args.num_workers != 0
            or args.device != "cuda:0"):
        raise ValueError(
            "formal contract requires batch-size=16, num-workers=0, cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("formal contract requires CUDA")
    portable_manifest = validate_portable_s1_manifest(args)
    p0 = validate_archive_blend(
        args.p0_archive, P0_ARCHIVE_SHA256, 0.625, 0.25)
    p1 = validate_archive_blend(
        args.p1_archive, P1_ARCHIVE_SHA256, 0.5, 0.5,
        decoupled_p1=True)
    if p0["checkpoint_members"] != p1["checkpoint_members"]:
        raise ValueError("P0/P1 checkpoint ZIP members differ bytewise")
    return {"portable_s1_manifest": portable_manifest, "p0_archive": p0,
            "p1_archive": p1}


def observation_features(history: np.ndarray) -> np.ndarray:
    """Return the eight frozen raw-history features for one sample."""
    value = np.asarray(history)
    if value.ndim != 4 or value.shape[0] < 6 or value.shape[-1] < 2:
        raise ValueError("history must have shape (T,H,W,C), T>=6 and C>=2")
    uv = np.asarray(value[..., :2], dtype=np.float64)
    valid = np.any(uv != 0.0, axis=(0, 3))
    if not np.any(valid):
        raise ValueError("history has no measured velocity cell")
    measured = uv[:, valid, :]
    last = measured[-1]
    spatial_means = np.mean(measured, axis=1, dtype=np.float64)
    recent_difference = np.diff(measured[-6:], axis=0)
    features = np.asarray([
        np.mean(last[:, 0], dtype=np.float64),
        np.mean(last[:, 1], dtype=np.float64),
        np.std(last[:, 0], dtype=np.float64),
        np.std(last[:, 1], dtype=np.float64),
        np.std(spatial_means[:, 0], dtype=np.float64),
        np.std(spatial_means[:, 1], dtype=np.float64),
        np.sqrt(np.mean(recent_difference[..., 0] ** 2, dtype=np.float64)),
        np.sqrt(np.mean(recent_difference[..., 1] ** 2, dtype=np.float64)),
    ], dtype=np.float64)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("non-finite observation feature")
    return features


def blend_sample_weights(prediction_a: np.ndarray,
                         prediction_b: np.ndarray,
                         prediction_c: np.ndarray,
                         weight_b: np.ndarray,
                         weight_c: np.ndarray) -> np.ndarray:
    if (prediction_a.shape != prediction_b.shape
            or prediction_a.shape != prediction_c.shape
            or prediction_a.ndim != 5):
        raise ValueError("A/B/C predictions must share shape (N,T,H,W,C)")
    n = prediction_a.shape[0]
    wb = np.asarray(weight_b, dtype=np.float32)
    wc = np.asarray(weight_c, dtype=np.float32)
    if wb.shape != (n,) or wc.shape != (n,):
        raise ValueError("weight arrays must have shape (N,)")
    if (not np.all(np.isfinite(wb)) or not np.all(np.isfinite(wc))
            or np.any(wb < MIN_WEIGHT) or np.any(wb > MAX_WEIGHT)
            or np.any(wc < MIN_WEIGHT) or np.any(wc > MAX_WEIGHT)):
        raise ValueError("sample weights are outside the frozen bounds")
    shape = (n, 1, 1, 1, 1)
    base = prediction_a + wb.reshape(shape) * (prediction_b - prediction_a)
    mean_delta = (
        prediction_c.mean(axis=1, keepdims=True)
        - base.mean(axis=1, keepdims=True))
    return base + wc.reshape(shape) * mean_delta


def _ab_headroom_components(
        prediction_a: np.ndarray, prediction_b: np.ndarray,
        baseline_prediction: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (prediction_a.shape != prediction_b.shape
            or prediction_a.shape != baseline_prediction.shape
            or prediction_a.ndim != 5
            or prediction_a.shape[1] != AB_HEADROOM_HORIZON_BLOCKS[-1][1]):
        raise ValueError(
            "A/B/baseline predictions must share shape (N,20,H,W,C)")
    a = np.asarray(prediction_a, dtype=np.float32)
    b = np.asarray(prediction_b, dtype=np.float32)
    baseline = np.asarray(baseline_prediction, dtype=np.float32)
    fixed_mean = np.asarray(
        baseline.mean(axis=1, keepdims=True), dtype=np.float32)
    baseline_fluctuation = np.asarray(
        baseline - fixed_mean, dtype=np.float32)
    role_delta = np.asarray(
        (b - b.mean(axis=1, keepdims=True))
        - (a - a.mean(axis=1, keepdims=True)),
        dtype=np.float32)
    return fixed_mean, baseline_fluctuation, role_delta


def _ab_time_weights(weight_b: np.ndarray, samples: int) -> np.ndarray:
    weights = np.asarray(weight_b, dtype=np.float32)
    if weights.shape == (samples,):
        values = np.broadcast_to(
            weights[:, None], (samples, AB_HEADROOM_HORIZON_BLOCKS[-1][1]))
    elif weights.shape == (samples, len(AB_HEADROOM_HORIZON_BLOCKS)):
        values = np.empty(
            (samples, AB_HEADROOM_HORIZON_BLOCKS[-1][1]), dtype=np.float32)
        for block, (start, stop) in enumerate(AB_HEADROOM_HORIZON_BLOCKS):
            values[:, start:stop] = weights[:, block, None]
    else:
        raise ValueError(
            "A/B weights must have shape (N,) or (N,4 horizon blocks)")
    if (not np.all(np.isfinite(values))
            or np.any(values < MIN_WEIGHT) or np.any(values > MAX_WEIGHT)):
        raise ValueError("A/B weights are outside the frozen bounds")
    return values.reshape(samples, values.shape[1], 1, 1, 1)


def _ab_prediction_from_components(
        baseline_prediction: np.ndarray, fixed_mean: np.ndarray,
        baseline_fluctuation: np.ndarray, role_delta: np.ndarray,
        weight_b: np.ndarray) -> np.ndarray:
    weights = np.asarray(weight_b, dtype=np.float32)
    time_weights = _ab_time_weights(weights, baseline_prediction.shape[0])
    dynamic_fluctuation = np.asarray(
        baseline_fluctuation
        + (time_weights - BASE_WEIGHT) * role_delta,
        dtype=np.float32)
    dynamic_fluctuation -= dynamic_fluctuation.mean(
        axis=1, keepdims=True, dtype=np.float32)
    candidate = np.asarray(fixed_mean + dynamic_fluctuation, dtype=np.float32)
    fallback = (
        weights == BASE_WEIGHT
        if weights.ndim == 1
        else np.all(weights == BASE_WEIGHT, axis=1))
    candidate[fallback] = baseline_prediction[fallback]
    return candidate


def ab_headroom_prediction(
        prediction_a: np.ndarray, prediction_b: np.ndarray,
        baseline_prediction: np.ndarray, weight_b: np.ndarray) -> np.ndarray:
    """Route only A/B fluctuations and restore the fixed P1 temporal mean."""
    components = _ab_headroom_components(
        prediction_a, prediction_b, baseline_prediction)
    return _ab_prediction_from_components(
        baseline_prediction, *components, weight_b)


def _masked_dot_per_sample(left: np.ndarray, right: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.shape != mask.shape:
        raise ValueError("masked dot inputs must share shape")
    result = np.empty(left.shape[0], dtype=np.float64)
    for index in range(left.shape[0]):
        result[index] = np.sum(
            left[index][mask[index]] * right[index][mask[index]],
            dtype=np.float64)
    return result


def mse_oracle_initialization(prediction_a: np.ndarray,
                              prediction_b: np.ndarray,
                              prediction_c: np.ndarray,
                              target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed 33-point wC search with analytic bounded wB at each point."""
    channels = OFFICIAL.measured_channels(target)
    a = np.asarray(prediction_a[..., :channels], dtype=np.float32)
    d = np.asarray(
        prediction_b[..., :channels] - prediction_a[..., :channels],
        dtype=np.float32)
    target_value = np.asarray(target[..., :channels], dtype=np.float32)
    c_mean = np.asarray(
        prediction_c[..., :channels].mean(axis=1, keepdims=True),
        dtype=np.float32)
    a_mean = np.asarray(a.mean(axis=1, keepdims=True), dtype=np.float32)
    d_mean = np.asarray(d.mean(axis=1, keepdims=True), dtype=np.float32)
    mask = target_value != 0.0
    r0 = target_value - a
    g = np.broadcast_to(c_mean - a_mean, a.shape)
    dm = np.broadcast_to(d_mean, d.shape)

    rr = _masked_dot_per_sample(r0, r0, mask)
    r0g = _masked_dot_per_sample(r0, g, mask)
    gg = _masked_dot_per_sample(g, g, mask)
    r0d = _masked_dot_per_sample(r0, d, mask)
    r0dm = _masked_dot_per_sample(r0, dm, mask)
    gd = _masked_dot_per_sample(g, d, mask)
    gdm = _masked_dot_per_sample(g, dm, mask)
    dd = _masked_dot_per_sample(d, d, mask)
    ddm = _masked_dot_per_sample(d, dm, mask)
    dmdm = _masked_dot_per_sample(dm, dm, mask)

    best_sse = np.full(a.shape[0], np.inf, dtype=np.float64)
    best_b = np.full(a.shape[0], BASE_WEIGHT, dtype=np.float64)
    best_c = np.full(a.shape[0], BASE_WEIGHT, dtype=np.float64)
    for wc in np.linspace(
            MIN_WEIGHT, MAX_WEIGHT, ORACLE_WC_GRID_POINTS,
            dtype=np.float64):
        r_dot_r = rr - 2.0 * wc * r0g + wc * wc * gg
        r_dot_q = r0d - wc * (r0dm + gd) + wc * wc * gdm
        q_dot_q = dd - 2.0 * wc * ddm + wc * wc * dmdm
        wb = np.full_like(r_dot_q, BASE_WEIGHT)
        valid = q_dot_q > 1e-20
        wb[valid] = r_dot_q[valid] / q_dot_q[valid]
        wb = np.clip(wb, MIN_WEIGHT, MAX_WEIGHT)
        candidate_sse = r_dot_r - 2.0 * wb * r_dot_q + wb * wb * q_dot_q
        selected = candidate_sse < best_sse
        best_sse[selected] = candidate_sse[selected]
        best_b[selected] = wb[selected]
        best_c[selected] = wc
    return best_b, best_c


def ab_scalar_mse_initialization(
        prediction_a: np.ndarray, prediction_b: np.ndarray,
        baseline_prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Analytic scored-UV MSE initializer for one A/B weight per sample."""
    channels = OFFICIAL.measured_channels(target)
    components = _ab_headroom_components(
        prediction_a, prediction_b, baseline_prediction)
    role_delta = np.asarray(
        components[2][..., :channels], dtype=np.float32)
    role_delta -= role_delta.mean(axis=1, keepdims=True, dtype=np.float32)
    residual = np.asarray(
        target[..., :channels] - baseline_prediction[..., :channels],
        dtype=np.float32)
    mask = target[..., :channels] != 0.0
    numerator = _masked_dot_per_sample(residual, role_delta, mask)
    denominator = _masked_dot_per_sample(role_delta, role_delta, mask)
    weights = np.full(target.shape[0], BASE_WEIGHT, dtype=np.float64)
    valid = denominator > 1e-20
    weights[valid] += numerator[valid] / denominator[valid]
    return np.clip(weights, MIN_WEIGHT, MAX_WEIGHT)


def sample_interval_quality(lower: np.ndarray, upper: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
    """Frozen per-window interval term shared by every point candidate."""
    channels = OFFICIAL.measured_channels(target)
    measured_target = target[..., :channels]
    measured_lower = lower[..., :channels]
    measured_upper = upper[..., :channels]
    scored = measured_target != 0.0
    inside = ((measured_target >= measured_lower)
              & (measured_target <= measured_upper))
    interval_factor = np.exp(
        -(measured_upper - measured_lower) / OFFICIAL.SIGMA_GLOBAL)
    axes = tuple(range(1, scored.ndim))
    numerator = np.sum(
        np.where(inside & scored, interval_factor, 0.0),
        axis=axes, dtype=np.float64)
    denominator = np.sum(scored, axis=axes, dtype=np.float64)
    if np.any(denominator <= 0.0):
        raise ValueError("sample has no scored target element")
    return numerator / denominator


def sample_shadow_objective(prediction: np.ndarray, lower: np.ndarray,
                            upper: np.ndarray, target: np.ndarray,
                            interval_quality: np.ndarray | None = None,
                            ) -> np.ndarray:
    """Per-window exact-metric surrogate using the frozen Final derivatives."""
    channels = OFFICIAL.measured_channels(target)
    rel_error = OFFICIAL.rel_l2_per_sample(prediction, target, channels)
    tke_error = OFFICIAL.tke_rel_l2_per_sample(prediction, target, channels)
    mvpe_error = OFFICIAL.mvpe_rel_l2_per_sample(prediction, target)

    def error_score(error: np.ndarray) -> np.ndarray:
        return 100.0 / (1.0 + 0.5 * np.maximum(error, 0.0))

    if interval_quality is None:
        interval_quality = sample_interval_quality(lower, upper, target)
    if interval_quality.shape != (target.shape[0],):
        raise ValueError("interval quality must have shape (N,)")
    normalized = [
        error / (0.5 + error) for error in (rel_error, tke_error, mvpe_error)]
    sps_score = 100.0 * interval_quality * (
        0.5 * (1.0 - normalized[0])
        + 0.3 * (1.0 - normalized[1])
        + 0.2 * (1.0 - normalized[2]))
    return (
        SHADOW_DERIVATIVES["rel_l2_score"] * error_score(rel_error)
        + SHADOW_DERIVATIVES["tke_score"] * error_score(tke_error)
        + SHADOW_DERIVATIVES["mvpe_score"] * error_score(mvpe_error)
        + SHADOW_DERIVATIVES["sps_score"] * sps_score)


def bounded_target_oracle(prediction_a: np.ndarray,
                          prediction_b: np.ndarray,
                          prediction_c: np.ndarray,
                          lower: np.ndarray, upper: np.ndarray,
                          target: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                       dict]:
    """Deterministic target-aware optimizer for the bounded two-scalar form."""
    baseline_weights = np.full(target.shape[0], BASE_WEIGHT, dtype=np.float64)
    baseline_prediction = blend_sample_weights(
        prediction_a, prediction_b, prediction_c,
        baseline_weights, baseline_weights)
    interval_quality = sample_interval_quality(lower, upper, target)
    baseline_objective = sample_shadow_objective(
        baseline_prediction, lower, upper, target, interval_quality)
    wb, wc = mse_oracle_initialization(
        prediction_a, prediction_b, prediction_c, target)
    initial_prediction = blend_sample_weights(
        prediction_a, prediction_b, prediction_c, wb, wc)
    objective = sample_shadow_objective(
        initial_prediction, lower, upper, target, interval_quality)
    fallback = baseline_objective > objective
    wb[fallback] = BASE_WEIGHT
    wc[fallback] = BASE_WEIGHT
    objective[fallback] = baseline_objective[fallback]
    del baseline_prediction, initial_prediction

    evaluations = 2
    for step in ORACLE_COORDINATE_STEPS:
        for axis, direction in (("b", -1.0), ("b", 1.0),
                                ("c", -1.0), ("c", 1.0)):
            proposal_b = wb.copy()
            proposal_c = wc.copy()
            if axis == "b":
                proposal_b = np.clip(
                    proposal_b + direction * step, MIN_WEIGHT, MAX_WEIGHT)
            else:
                proposal_c = np.clip(
                    proposal_c + direction * step, MIN_WEIGHT, MAX_WEIGHT)
            proposal_prediction = blend_sample_weights(
                prediction_a, prediction_b, prediction_c,
                proposal_b, proposal_c)
            proposal_objective = sample_shadow_objective(
                proposal_prediction, lower, upper, target, interval_quality)
            selected = proposal_objective > objective + 1e-12
            wb[selected] = proposal_b[selected]
            wc[selected] = proposal_c[selected]
            objective[selected] = proposal_objective[selected]
            evaluations += 1
            del proposal_prediction
    if np.any(objective + 1e-10 < baseline_objective):
        raise RuntimeError("target oracle worsened its per-sample objective")
    return wb, wc, {
        "optimizer": (
            "33-point fixed wC / analytic wB scored-UV-MSE initialization; "
            "baseline fallback; deterministic bounded coordinate ascent on "
            "per-sample official-metric shadow surrogate"),
        "coordinate_steps": list(ORACLE_COORDINATE_STEPS),
        "objective_evaluations": evaluations,
        "samples_using_baseline_fallback": int(np.count_nonzero(fallback)),
        "mean_local_objective_gain": float(np.mean(
            objective - baseline_objective)),
    }


def bounded_ab_target_oracle(
        prediction_a: np.ndarray, prediction_b: np.ndarray,
        baseline_prediction: np.ndarray, lower: np.ndarray, upper: np.ndarray,
        target: np.ndarray, *, horizon_blocks: int,
        initial_weights: np.ndarray | None = None,
        ) -> tuple[np.ndarray, dict]:
    """Target-visible deterministic A/B fluctuation oracle with fallback."""
    if horizon_blocks not in (1, len(AB_HEADROOM_HORIZON_BLOCKS)):
        raise ValueError("horizon_blocks must be 1 or 4")
    samples = target.shape[0]
    expected_shape = (samples, horizon_blocks)
    if initial_weights is None:
        if horizon_blocks == 1:
            weights = ab_scalar_mse_initialization(
                prediction_a, prediction_b, baseline_prediction,
                target)[:, None]
            initialization = "analytic_scored_uv_mse_with_baseline_fallback"
        else:
            weights = np.full(expected_shape, BASE_WEIGHT, dtype=np.float64)
            initialization = "fixed_p1_weight_0.5"
    else:
        weights = np.asarray(initial_weights, dtype=np.float64).copy()
        if weights.shape != expected_shape:
            raise ValueError(
                f"initial A/B weights must have shape {expected_shape}")
        if (not np.all(np.isfinite(weights))
                or np.any(weights < MIN_WEIGHT)
                or np.any(weights > MAX_WEIGHT)):
            raise ValueError("initial A/B weights are outside the frozen bounds")
        initialization = "provided_bounded_weights_with_baseline_fallback"

    components = _ab_headroom_components(
        prediction_a, prediction_b, baseline_prediction)
    interval_quality = sample_interval_quality(lower, upper, target)
    baseline_objective = sample_shadow_objective(
        baseline_prediction, lower, upper, target, interval_quality)
    candidate = _ab_prediction_from_components(
        baseline_prediction, *components,
        weights[:, 0] if horizon_blocks == 1 else weights)
    objective = sample_shadow_objective(
        candidate, lower, upper, target, interval_quality)
    initial_fallback = objective <= baseline_objective + 1e-12
    weights[initial_fallback] = BASE_WEIGHT
    objective[initial_fallback] = baseline_objective[initial_fallback]
    del candidate

    evaluations = 1
    for step in ORACLE_COORDINATE_STEPS:
        for block in range(horizon_blocks):
            for direction in (-1.0, 1.0):
                proposal = weights.copy()
                proposal[:, block] = np.clip(
                    proposal[:, block] + direction * step,
                    MIN_WEIGHT, MAX_WEIGHT)
                proposal_value = (
                    proposal[:, 0] if horizon_blocks == 1 else proposal)
                proposal_prediction = _ab_prediction_from_components(
                    baseline_prediction, *components, proposal_value)
                proposal_objective = sample_shadow_objective(
                    proposal_prediction, lower, upper, target,
                    interval_quality)
                selected = proposal_objective > objective + 1e-12
                weights[selected] = proposal[selected]
                objective[selected] = proposal_objective[selected]
                evaluations += 1
                del proposal_prediction

    if np.any(objective + 1e-10 < baseline_objective):
        raise RuntimeError("A/B target oracle violated its baseline fallback")
    baseline_fallback = np.all(weights == BASE_WEIGHT, axis=1)
    result = weights[:, 0] if horizon_blocks == 1 else weights
    return result, {
        "optimizer": (
            "baseline fallback plus deterministic bounded coordinate ascent "
            "on the per-sample official-metric shadow surrogate"),
        "initialization": initialization,
        "coordinate_steps": list(ORACLE_COORDINATE_STEPS),
        "horizon_blocks": (
            [[start + 1, stop] for start, stop in AB_HEADROOM_HORIZON_BLOCKS]
            if horizon_blocks > 1 else [[1, 20]]),
        "objective_evaluations": evaluations,
        "samples_using_baseline_fallback": int(np.count_nonzero(
            baseline_fallback)),
        "samples_using_initial_fallback": int(np.count_nonzero(
            initial_fallback)),
        "mean_local_objective_gain": float(np.mean(
            objective - baseline_objective)),
    }


def fit_fixed_ridge(features: np.ndarray,
                    oracle_weights: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    weights = np.asarray(oracle_weights, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES)
            or weights.shape != (x.shape[0], 2)):
        raise ValueError("ridge inputs have unexpected shape")
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-12, scale, 1.0)
    design = np.column_stack((np.ones(x.shape[0]), (x - mean) / scale))
    target = (weights - BASE_WEIGHT) / WEIGHT_RADIUS
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ target)
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict_fixed_ridge(model: dict[str, np.ndarray],
                        features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    design = np.column_stack((
        np.ones(x.shape[0]), (x - model["mean"]) / model["scale"]))
    normalized = np.clip(design @ model["coefficients"], -1.0, 1.0)
    return BASE_WEIGHT + WEIGHT_RADIUS * normalized


def serialize_ridge(model: dict[str, np.ndarray]) -> dict:
    return {
        "feature_mean": model["mean"].tolist(),
        "feature_scale": model["scale"].tolist(),
        "coefficients_intercept_first": model["coefficients"].tolist(),
    }


def crossfit_weights(features: np.ndarray, oracle_weights: np.ndarray,
                     groups: np.ndarray,
                     expected_groups: tuple[int, ...]) -> tuple[np.ndarray, list[dict]]:
    group_values = np.asarray(groups)
    if tuple(sorted(np.unique(group_values).tolist())) != expected_groups:
        raise ValueError(
            f"cross-fit groups differ: {np.unique(group_values)}, "
            f"expected {expected_groups}")
    prediction = np.full_like(oracle_weights, np.nan, dtype=np.float64)
    models = []
    for held in expected_groups:
        train = group_values != held
        validation = group_values == held
        model = fit_fixed_ridge(features[train], oracle_weights[train])
        prediction[validation] = predict_fixed_ridge(model, features[validation])
        models.append({
            "held_group": int(held),
            "train_samples": int(np.count_nonzero(train)),
            "held_samples": int(np.count_nonzero(validation)),
            **serialize_ridge(model),
        })
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("cross-fit left unpredicted weights")
    return prediction, models


def shadow_delta(baseline: dict, candidate: dict) -> dict:
    deltas = {
        key: float(candidate[key] - baseline[key])
        for key in SHADOW_DERIVATIVES
    }
    contributions = {
        key: float(SHADOW_DERIVATIVES[key] * deltas[key])
        for key in SHADOW_DERIVATIVES
    }
    return {
        "metric_deltas": deltas,
        "contributions": contributions,
        "time_score_delta_assumed": 0.0,
        "shadow_final_gain": float(sum(contributions.values())),
    }


def bounds_risk(baseline_prediction: np.ndarray,
                candidate_prediction: np.ndarray,
                lower: np.ndarray, upper: np.ndarray,
                target: np.ndarray) -> dict:
    channels = OFFICIAL.measured_channels(target)
    scored = target[..., :channels] != 0.0
    baseline_outside = (
        (baseline_prediction[..., :channels] < lower[..., :channels])
        | (baseline_prediction[..., :channels] > upper[..., :channels]))
    candidate_outside = (
        (candidate_prediction[..., :channels] < lower[..., :channels])
        | (candidate_prediction[..., :channels] > upper[..., :channels]))
    scored_count = np.count_nonzero(scored)
    baseline_all_within = bool(np.all(
        (baseline_prediction >= lower) & (baseline_prediction <= upper)))
    candidate_all_within = bool(np.all(
        (candidate_prediction >= lower) & (candidate_prediction <= upper)))
    return {
        "point_clipping_used": False,
        "baseline_all_elements_within_bounds": baseline_all_within,
        "candidate_all_elements_within_bounds": candidate_all_within,
        "baseline_scored_outside_rate": float(
            np.count_nonzero(baseline_outside & scored) / scored_count),
        "candidate_scored_outside_rate": float(
            np.count_nonzero(candidate_outside & scored) / scored_count),
        "new_scored_outside_rate": float(np.count_nonzero(
            candidate_outside & ~baseline_outside & scored) / scored_count),
        "conditional_within_bounds_contract_pass": bool(
            not baseline_all_within or candidate_all_within),
    }


def comparison_row(name: str, baseline_prediction: np.ndarray,
                   candidate_prediction: np.ndarray,
                   lower: np.ndarray, upper: np.ndarray,
                   target: np.ndarray) -> dict:
    baseline = score(baseline_prediction, lower, upper, target)
    candidate = score(candidate_prediction, lower, upper, target)
    if (candidate["coverage"] != baseline["coverage"]
            or candidate["mean_width"] != baseline["mean_width"]):
        raise RuntimeError("fixed bounds changed coverage or mean width")
    risk = bounds_risk(
        baseline_prediction, candidate_prediction, lower, upper, target)
    return {
        "name": name,
        "samples": int(target.shape[0]),
        "baseline": baseline,
        "candidate": candidate,
        **shadow_delta(baseline, candidate),
        "bounds_risk": risk,
    }


def p1_reconstruction_control(actual: dict) -> dict:
    source_deltas = {
        key: float(actual[key] - expected)
        for key, expected in P1_SOURCE_MATCHED_OOF_REFERENCE.items()
    }
    deployment_drift = {
        key: float(actual[key] - expected)
        for key, expected in P1_DEPLOYMENT_REFERENCE.items()
    }
    absolute_source_differences = {
        key: abs(value) for key, value in source_deltas.items()}
    return {
        "guard": (
            "strict fp32 OOF reconstruction versus the source-matched frozen "
            "reference; every candidate delta uses this same reconstructed "
            "baseline array"),
        "source_matched_result_sha256": P1_SOURCE_MATCHED_OOF_RESULT_SHA256,
        "source_matched_reference": P1_SOURCE_MATCHED_OOF_REFERENCE,
        "source_matched_deltas": source_deltas,
        "source_matched_absolute_differences": absolute_source_differences,
        "tolerance": P1_POINT_TOLERANCE,
        "pass": all(
            value <= P1_POINT_TOLERANCE
            for value in absolute_source_differences.values()),
        "deployment_reference_drift_only": {
            "used_as_gate": False,
            "reference": P1_DEPLOYMENT_REFERENCE,
            "deltas": deployment_drift,
            "absolute_differences": {
                key: abs(value) for key, value in deployment_drift.items()},
        },
    }


def ab_headroom_decision(shadow_gain: float, tke_delta: float,
                         *, within_budget: bool = True) -> str:
    if not within_budget:
        return "STOP_AB_ROUTING_TIME_BUDGET"
    if shadow_gain < AB_HEADROOM_LOW_GATE:
        return "STOP_AB_ROUTING_NO_HEADROOM"
    if shadow_gain < AB_HEADROOM_GO_GATE:
        return "STOP_AB_ROUTING_INSUFFICIENT_FOR_PGR"
    if tke_delta < AB_HEADROOM_TKE_DELTA_FLOOR:
        return "STOP_TKE"
    return "GO_AB_ROUTER_CROSSFIT"


def _ab_weight_report(weights: np.ndarray) -> dict:
    value = np.asarray(weights, dtype=np.float64)
    report = {
        "values": value.tolist(),
        "identity": s1.array_identity(value),
        "minimum": float(np.min(value)),
        "mean": float(np.mean(value)),
        "maximum": float(np.max(value)),
    }
    if value.ndim == 2:
        report["per_block"] = [
            {
                "horizon": [start + 1, stop],
                "minimum": float(np.min(value[:, block])),
                "mean": float(np.mean(value[:, block])),
                "maximum": float(np.max(value[:, block])),
            }
            for block, (start, stop) in enumerate(AB_HEADROOM_HORIZON_BLOCKS)
        ]
    return report


def evaluate_ab_headroom(
        prediction_a: np.ndarray, prediction_b: np.ndarray,
        prediction_c: np.ndarray, lower: np.ndarray, upper: np.ndarray,
        target: np.ndarray) -> dict:
    """Run only the two frozen target-visible A/B headroom oracles."""
    started = time.perf_counter()
    baseline_weights = np.full(target.shape[0], BASE_WEIGHT, dtype=np.float64)
    baseline_prediction = blend_sample_weights(
        prediction_a, prediction_b, prediction_c,
        baseline_weights, baseline_weights)
    baseline_metrics = score(baseline_prediction, lower, upper, target)
    reconstruction = p1_reconstruction_control(baseline_metrics)
    fixed_mean = np.asarray(
        baseline_prediction.mean(axis=1, keepdims=True), dtype=np.float32)
    result = {
        "p1_reconstruction_control": reconstruction,
        "baseline_prediction_identity": s1.array_identity(
            baseline_prediction),
        "fixed_baseline_temporal_mean_identity": s1.array_identity(fixed_mean),
    }
    if not reconstruction["pass"]:
        result.update({
            "formal_elapsed_seconds": time.perf_counter() - started,
            "decision": "STOP_P1_SOURCE_MATCHED_RECONSTRUCTION",
        })
        return result

    scalar_weights, scalar_optimizer = bounded_ab_target_oracle(
        prediction_a, prediction_b, baseline_prediction,
        lower, upper, target, horizon_blocks=1)
    scalar_prediction = ab_headroom_prediction(
        prediction_a, prediction_b, baseline_prediction, scalar_weights)
    scalar_row = comparison_row(
        "TARGET_VISIBLE_AB_SCALAR_ORACLE", baseline_prediction,
        scalar_prediction, lower, upper, target)
    scalar_row.update({
        "optimizer": scalar_optimizer,
        "weight_b": _ab_weight_report(scalar_weights),
    })

    block_initial = np.repeat(
        scalar_weights[:, None], len(AB_HEADROOM_HORIZON_BLOCKS), axis=1)
    block_weights, block_optimizer = bounded_ab_target_oracle(
        prediction_a, prediction_b, baseline_prediction,
        lower, upper, target,
        horizon_blocks=len(AB_HEADROOM_HORIZON_BLOCKS),
        initial_weights=block_initial)
    block_prediction = ab_headroom_prediction(
        prediction_a, prediction_b, baseline_prediction, block_weights)
    block_row = comparison_row(
        "TARGET_VISIBLE_AB_FOUR_BLOCK_ORACLE", baseline_prediction,
        block_prediction, lower, upper, target)
    block_row.update({
        "optimizer": block_optimizer,
        "weight_b": _ab_weight_report(block_weights),
    })

    scalar_mean_difference = float(np.max(np.abs(
        scalar_prediction.mean(axis=1, keepdims=True) - fixed_mean)))
    block_mean_difference = float(np.max(np.abs(
        block_prediction.mean(axis=1, keepdims=True) - fixed_mean)))
    if max(scalar_mean_difference, block_mean_difference) \
            > AB_HEADROOM_FIXED_MEAN_TOLERANCE:
        raise RuntimeError("A/B headroom candidate changed the fixed P1 mean")
    elapsed = time.perf_counter() - started
    block_gain = block_row["shadow_final_gain"]
    block_tke_delta = block_row["metric_deltas"]["tke_score"]
    result.update({
        "scalar_oracle": scalar_row,
        "four_block_oracle": block_row,
        "fixed_temporal_mean_guard": {
            "tolerance": AB_HEADROOM_FIXED_MEAN_TOLERANCE,
            "scalar_max_abs_difference": scalar_mean_difference,
            "four_block_max_abs_difference": block_mean_difference,
            "pass": True,
        },
        "formal_elapsed_seconds": elapsed,
        "gate_checks": {
            "four_block_shadow_ge_0p15": (
                block_gain >= AB_HEADROOM_LOW_GATE),
            "four_block_shadow_ge_0p20": (
                block_gain >= AB_HEADROOM_GO_GATE),
            "four_block_tke_delta_ge_minus_0p03": (
                block_tke_delta >= AB_HEADROOM_TKE_DELTA_FLOOR),
            "formal_time_budget_pass": elapsed <= FORMAL_BUDGET_SECONDS,
        },
        "decision": ab_headroom_decision(
            block_gain, block_tke_delta,
            within_budget=elapsed <= FORMAL_BUDGET_SECONDS),
        "artifacts": {
            "checkpoint": None,
            "checkpoints_saved": [],
            "zip": None,
            "router_trained": False,
        },
    })
    return result


def subset_rows(prefix: str, baseline_prediction: np.ndarray,
                candidate_prediction: np.ndarray, lower: np.ndarray,
                upper: np.ndarray, target: np.ndarray,
                groups: np.ndarray,
                expected_groups: tuple[int, ...]) -> list[dict]:
    rows = []
    for group in expected_groups:
        selected = groups == group
        rows.append(comparison_row(
            f"{prefix}_{group}", baseline_prediction[selected],
            candidate_prediction[selected], lower[selected], upper[selected],
            target[selected]))
    return rows


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> None:
    args = parse_args()
    identities = validate_fixed_contract(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reconstruction_started = time.perf_counter()
    device = torch.device(args.device)

    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    _point_windows, tail_windows, runtime_manifest = common.repro.make_split(
        metadata, 20, 0.2)
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    dataset = common.repro.WindowDataset(trajectories, tail_windows)
    sample_folds, fold_manifest = trajectory_folds(runtime_manifest, 5)
    s1.validate_s1_split(
        runtime_manifest, frozen_manifest, len(dataset), sample_folds)
    oof_summaries = {
        "a": s1.validate_oof_summary(
            args.oof_dir_a / "summary.json", args.point_checkpoint_a,
            fold_manifest),
        "b": s1.validate_oof_summary(
            args.oof_dir_b / "summary.json", args.point_checkpoint_b,
            fold_manifest),
    }

    raw_features = []
    raw_aoas = []
    for index, window in enumerate(tail_windows):
        history, _target = dataset[index]
        raw_features.append(observation_features(history.numpy()))
        aoa = int(metadata[window.trajectory]["aoa"])
        if aoa not in AOA_VALUES:
            raise ValueError(f"unexpected AoA {aoa}")
        raw_aoas.append(aoa)
    raw_features = np.asarray(raw_features, dtype=np.float64)
    raw_aoas = np.asarray(raw_aoas, dtype=np.int64)

    payload_a = torch.load(
        args.point_checkpoint_a, map_location="cpu", weights_only=False)
    payload_b = torch.load(
        args.point_checkpoint_b, map_location="cpu", weights_only=False)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    mean_expert, mean_metadata = load_model(
        args.mean_expert_checkpoint, normalizer, device)
    loader_args = argparse.Namespace(num_workers=args.num_workers)
    all_indices = list(range(len(dataset)))
    prediction_a_folds = []
    prediction_b_folds = []
    prediction_c_folds = []
    p0_lower_folds = []
    p0_upper_folds = []
    target_folds = []
    feature_folds = []
    aoa_folds = []
    condition_folds = []
    fold_metadata = []

    for fold in range(5):
        indices = [index for index in all_indices if sample_folds[index] == fold]
        loader = uq_train.make_loader(
            dataset, indices, args.batch_size, False, loader_args, seed=0)
        model_a, metadata_a = s1.load_fold_model(
            payload_a, args.oof_dir_a / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_a, normalizer, device)
        values_a = collect(model_a, loader, normalizer, device)
        del model_a
        model_b, metadata_b = s1.load_fold_model(
            payload_b, args.oof_dir_b / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_b, normalizer, device)
        values_b = collect(model_b, loader, normalizer, device)
        del model_b
        values_c = collect(mean_expert, loader, normalizer, device)
        if (not np.array_equal(values_a[3], values_b[3])
                or not np.array_equal(values_a[3], values_c[3])):
            raise RuntimeError(f"target order changed in fold {fold}")
        arrays_a = dict(zip(("prediction", "lower", "upper"), values_a[:3]))
        arrays_b = dict(zip(("prediction", "lower", "upper"), values_b[:3]))
        p0 = s1.blend_candidate(
            arrays_a, arrays_b, values_c[0], weight_b=0.625,
            mean_weight=0.25)
        prediction_a_folds.append(arrays_a["prediction"])
        prediction_b_folds.append(arrays_b["prediction"])
        prediction_c_folds.append(values_c[0])
        p0_lower_folds.append(p0["lower"])
        p0_upper_folds.append(p0["upper"])
        target_folds.append(values_a[3])
        feature_folds.append(raw_features[indices])
        aoa_folds.append(raw_aoas[indices])
        condition_folds.append(np.full(len(indices), fold, dtype=np.int64))
        fold_metadata.append({
            "fold": fold, "samples": len(indices),
            "model_a": metadata_a, "model_b": metadata_b,
        })
        print(json.dumps({
            "event": "observation_gate_reconstruct", "fold": fold,
            "samples": len(indices),
        }), flush=True)
        del values_a, values_b, values_c, arrays_a, arrays_b, p0
        torch.cuda.empty_cache()

    del mean_expert, payload_a, payload_b, normalizer, trajectories
    torch.cuda.empty_cache()
    prediction_a = np.concatenate(prediction_a_folds)
    prediction_b = np.concatenate(prediction_b_folds)
    prediction_c = np.concatenate(prediction_c_folds)
    lower = np.concatenate(p0_lower_folds)
    upper = np.concatenate(p0_upper_folds)
    target = np.concatenate(target_folds)
    features = np.concatenate(feature_folds)
    aoas = np.concatenate(aoa_folds)
    condition_groups = np.concatenate(condition_folds)
    for value in (lower, upper, target):
        value.setflags(write=False)
    bounds_identity_before = {
        "lower": s1.array_identity(lower), "upper": s1.array_identity(upper)}
    if any(bounds_identity_before[key]["sha256"] != expected
           for key, expected in P0_BOUNDS_SHA256.items()):
        raise RuntimeError(
            "reconstructed P0 bounds differ from the frozen bitwise contract")
    reconstruction_elapsed = time.perf_counter() - reconstruction_started

    if args.ab_headroom_only:
        report = {
            "mode": "ab_headroom_only",
            "evidence_class": AB_HEADROOM_EVIDENCE_CLASS,
            "protocol": (
                "Reuse the exact fold-held A/B/C reconstruction, fixed P1 "
                "point baseline, and bitwise-frozen P0 bounds. Target-visible "
                "oracles change only A/B zero-mean fluctuation weights; no "
                "router, checkpoint, ZIP, or official score is produced."),
            "contract": {
                "hypothesis": AB_HEADROOM_HYPOTHESIS,
                "exact_baseline": AB_HEADROOM_EXACT_BASELINE,
                "exact_baseline_point": {
                    "weight_b": BASE_WEIGHT,
                    "mean_expert_weight_c": BASE_WEIGHT,
                },
                "candidate": (
                    "scalar wB per sample and four contiguous 5-step wB "
                    "values per sample; dynamic A/B fluctuation is recentered "
                    "over all 20 steps before restoring the fixed P1 mean"),
                "horizon_blocks_1_based_inclusive": [
                    [start + 1, stop]
                    for start, stop in AB_HEADROOM_HORIZON_BLOCKS],
                "weight_bounds": [MIN_WEIGHT, MAX_WEIGHT],
                "coordinate_steps": list(ORACLE_COORDINATE_STEPS),
                "baseline_fallback": True,
                "fixed_bounds": {"weight_b": 0.625,
                                 "mean_expert_weight_c": 0.25},
                "shadow_derivatives": SHADOW_DERIVATIVES,
                "time_score_delta_assumed": 0.0,
                "delta_comparison": (
                    "every metric delta compares candidate and baseline on "
                    "the same reconstructed arrays and fixed bounds"),
                "router_training": False,
            },
            "gates": {
                "no_headroom_if_shadow_below": AB_HEADROOM_LOW_GATE,
                "go_router_if_shadow_at_least": AB_HEADROOM_GO_GATE,
                "go_router_tke_delta_floor": AB_HEADROOM_TKE_DELTA_FLOOR,
                "source_matched_reconstruction_tolerance": (
                    P1_POINT_TOLERANCE),
            },
            "budget": {
                "reconstruction_seconds": RECONSTRUCTION_BUDGET_SECONDS,
                "headroom_oracles_seconds": FORMAL_BUDGET_SECONDS,
                "training_updates": 0,
                "checkpoints": 0,
                "zips": 0,
            },
            "stop_contract": [
                "STOP_RECONSTRUCTION_TIME_BUDGET",
                "STOP_P1_SOURCE_MATCHED_RECONSTRUCTION",
                "STOP_AB_ROUTING_TIME_BUDGET",
                "STOP_AB_ROUTING_NO_HEADROOM if block shadow < 0.15",
                "STOP_AB_ROUTING_INSUFFICIENT_FOR_PGR if 0.15 <= block "
                "shadow < 0.20",
                "STOP_TKE if block shadow >= 0.20 and TKE delta < -0.03",
                "GO_AB_ROUTER_CROSSFIT otherwise",
            ],
            "limitations": [
                "The target-visible weights use fixed725 labels and are not "
                "deployable predictions or official/OOD evidence.",
                "Deterministic bounded coordinate ascent is a preregistered "
                "diagnostic, not a proof of the continuous global optimum.",
                "The oracle uses the FP32 source reconstruction path; any "
                "promoted router must be replayed through the exact packaged "
                "INT8-storage/CUDA-FP16 deployment path.",
                "A GO only permits a history-visible cross-fit router; it "
                "does not permit adapters, full training, or a ZIP.",
            ],
            "identities": identities,
            "oof_summaries": oof_summaries,
            "mean_expert": mean_metadata,
            "folds": fold_metadata,
            "tail_windows": int(target.shape[0]),
            "target_identity": s1.array_identity(target),
            "fixed_p0_bounds_identity_before": bounds_identity_before,
            "reconstruction_elapsed_seconds": reconstruction_elapsed,
            "artifacts": {
                "checkpoint": None,
                "checkpoints_saved": [],
                "zip": None,
                "router_trained": False,
            },
        }
        if reconstruction_elapsed > RECONSTRUCTION_BUDGET_SECONDS:
            report["decision"] = "STOP_RECONSTRUCTION_TIME_BUDGET"
        else:
            report.update(evaluate_ab_headroom(
                prediction_a, prediction_b, prediction_c,
                lower, upper, target))
        report["fixed_p0_bounds_identity_after"] = {
            "lower": s1.array_identity(lower), "upper": s1.array_identity(upper)}
        report["bounds_bitwise_frozen"] = (
            report["fixed_p0_bounds_identity_after"] == bounds_identity_before)
        write_report(args.output, report)
        print(json.dumps({
            "event": "decision",
            "decision": report["decision"],
            "scalar_shadow_gain": report.get(
                "scalar_oracle", {}).get("shadow_final_gain"),
            "four_block_shadow_gain": report.get(
                "four_block_oracle", {}).get("shadow_final_gain"),
            "four_block_tke_delta": report.get(
                "four_block_oracle", {}).get(
                    "metric_deltas", {}).get("tke_score"),
        }), flush=True)
        return

    report = {
        "evidence_class": EVIDENCE_CLASS,
        "protocol": (
            "Exact A/B/C P1 point formula with bitwise-frozen fold-held P0 "
            "absolute bounds. Condition5 and LOAO ridge fits use only raw "
            "observation features at inference. Local fixed725 diagnostic only."),
        "contract": {
            "exact_baseline_point": {"weight_b": 0.5,
                                     "mean_expert_weight_c": 0.5},
            "fixed_bounds": {"weight_b": 0.625,
                             "mean_expert_weight_c": 0.25},
            "feature_names": list(FEATURE_NAMES),
            "feature_mask": (
                "spatial cell measured iff any nonzero u/v exists over the "
                "20-frame input history"),
            "recent_difference_definition": (
                "RMS over the last five frame-to-frame differences"),
            "weight_bounds": [MIN_WEIGHT, MAX_WEIGHT],
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_target": "(oracle_weight - 0.5) / 0.25; clipped to [-1,1]",
            "shadow_derivatives": SHADOW_DERIVATIVES,
            "time_score_delta_assumed": 0.0,
            "no_extra_model_forward": True,
            "point_clipping_forbidden": True,
        },
        "gates": {
            "oracle_pooled_shadow_gain_min": ORACLE_SHADOW_GAIN_GATE,
            "ridge_pooled_shadow_gain_min": POOLED_SHADOW_GAIN_GATE,
            "condition_positive_min": CONDITION_POSITIVE_MIN,
            "condition_worst_floor": CONDITION_WORST_FLOOR,
            "loao_positive_min": LOAO_POSITIVE_MIN,
            "loao_worst_floor": LOAO_WORST_FLOOR,
            "if_baseline_all_within_bounds_candidate_must_also_be": True,
            "reconstruction_budget_seconds": RECONSTRUCTION_BUDGET_SECONDS,
            "formal_budget_seconds": FORMAL_BUDGET_SECONDS,
        },
        "identities": identities,
        "oof_summaries": oof_summaries,
        "mean_expert": mean_metadata,
        "folds": fold_metadata,
        "tail_windows": int(target.shape[0]),
        "aoa_counts": {
            str(aoa): int(np.count_nonzero(aoas == aoa)) for aoa in AOA_VALUES},
        "feature_identity": s1.array_identity(features),
        "target_identity": s1.array_identity(target),
        "fixed_p0_bounds_identity_before": bounds_identity_before,
        "reconstruction_elapsed_seconds": reconstruction_elapsed,
    }
    if reconstruction_elapsed > RECONSTRUCTION_BUDGET_SECONDS:
        report["decision"] = "STOP_RECONSTRUCTION_TIME_BUDGET"
        write_report(args.output, report)
        print(json.dumps({"event": "decision", "decision": report["decision"]}),
              flush=True)
        return

    formal_started = time.perf_counter()
    baseline_weights = np.full(target.shape[0], BASE_WEIGHT, dtype=np.float64)
    baseline_prediction = blend_sample_weights(
        prediction_a, prediction_b, prediction_c,
        baseline_weights, baseline_weights)
    reconstruction = p1_reconstruction_control(
        score(baseline_prediction, lower, upper, target))
    report["p1_reconstruction_control"] = reconstruction
    if not reconstruction["pass"]:
        raise RuntimeError(
            f"P1 source-matched reconstruction control failed: "
            f"{reconstruction['source_matched_absolute_differences']}")
    oracle_b, oracle_c, oracle_optimizer = bounded_target_oracle(
        prediction_a, prediction_b, prediction_c, lower, upper, target)
    oracle_prediction = blend_sample_weights(
        prediction_a, prediction_b, prediction_c, oracle_b, oracle_c)
    oracle_row = comparison_row(
        "TARGET_AWARE_BOUNDED_ORACLE", baseline_prediction,
        oracle_prediction, lower, upper, target)
    oracle_row.update({
        "optimizer": oracle_optimizer,
        "weight_b_summary": {
            "min": float(np.min(oracle_b)), "mean": float(np.mean(oracle_b)),
            "max": float(np.max(oracle_b))},
        "weight_c_summary": {
            "min": float(np.min(oracle_c)), "mean": float(np.mean(oracle_c)),
            "max": float(np.max(oracle_c))},
        "p1_reconstruction_control": reconstruction,
        "baseline_prediction_identity": s1.array_identity(
            baseline_prediction),
    })
    report["oracle"] = oracle_row
    if oracle_row["shadow_final_gain"] < ORACLE_SHADOW_GAIN_GATE:
        report["formal_elapsed_seconds"] = time.perf_counter() - formal_started
        report["fixed_p0_bounds_identity_after"] = {
            "lower": s1.array_identity(lower), "upper": s1.array_identity(upper)}
        report["bounds_bitwise_frozen"] = (
            report["fixed_p0_bounds_identity_after"] == bounds_identity_before)
        report["decision"] = "STOP_ORACLE_CEILING_BELOW_0P18"
        write_report(args.output, report)
        print(json.dumps({"event": "decision", "decision": report["decision"],
                          "oracle_shadow_gain": oracle_row["shadow_final_gain"]}),
              flush=True)
        return

    oracle_weights = np.column_stack((oracle_b, oracle_c))
    condition_weights, condition_models = crossfit_weights(
        features, oracle_weights, condition_groups, tuple(range(5)))
    condition_prediction = blend_sample_weights(
        prediction_a, prediction_b, prediction_c,
        condition_weights[:, 0], condition_weights[:, 1])
    condition_pooled = comparison_row(
        "CONDITION5_CROSSFIT_POOLED", baseline_prediction,
        condition_prediction, lower, upper, target)
    condition_rows = subset_rows(
        "CONDITION_FOLD", baseline_prediction, condition_prediction,
        lower, upper, target, condition_groups, tuple(range(5)))

    loao_weights, loao_models = crossfit_weights(
        features, oracle_weights, aoas, AOA_VALUES)
    loao_prediction = blend_sample_weights(
        prediction_a, prediction_b, prediction_c,
        loao_weights[:, 0], loao_weights[:, 1])
    loao_pooled = comparison_row(
        "TRUE_LOAO_CROSSFIT_POOLED", baseline_prediction,
        loao_prediction, lower, upper, target)
    loao_rows = subset_rows(
        "HELD_AOA", baseline_prediction, loao_prediction,
        lower, upper, target, aoas, AOA_VALUES)

    condition_gains = [row["shadow_final_gain"] for row in condition_rows]
    loao_gains = [row["shadow_final_gain"] for row in loao_rows]
    full_model = fit_fixed_ridge(features, oracle_weights)
    formal_elapsed = time.perf_counter() - formal_started
    bounds_identity_after = {
        "lower": s1.array_identity(lower), "upper": s1.array_identity(upper)}
    bounds_bitwise_frozen = bounds_identity_after == bounds_identity_before
    boundary_contract_pass = bool(
        condition_pooled["bounds_risk"][
            "conditional_within_bounds_contract_pass"]
        and loao_pooled["bounds_risk"][
            "conditional_within_bounds_contract_pass"])
    gate_checks = {
        "oracle_pass": oracle_row["shadow_final_gain"]
        >= ORACLE_SHADOW_GAIN_GATE,
        "pooled_pass": condition_pooled["shadow_final_gain"]
        >= POOLED_SHADOW_GAIN_GATE,
        "condition_positive_pass": sum(gain > 0.0 for gain in condition_gains)
        >= CONDITION_POSITIVE_MIN,
        "condition_worst_pass": min(condition_gains)
        >= CONDITION_WORST_FLOOR,
        "loao_positive_pass": sum(gain > 0.0 for gain in loao_gains)
        >= LOAO_POSITIVE_MIN,
        "loao_worst_pass": min(loao_gains) >= LOAO_WORST_FLOOR,
        "bounds_bitwise_frozen": bounds_bitwise_frozen,
        "conditional_point_within_bounds_pass": boundary_contract_pass,
        "formal_time_budget_pass": formal_elapsed <= FORMAL_BUDGET_SECONDS,
    }
    report.update({
        "condition5": {
            "pooled": condition_pooled,
            "held_fold_rows": condition_rows,
            "models": condition_models,
            "positive_folds": sum(gain > 0.0 for gain in condition_gains),
            "worst_fold_shadow_gain": min(condition_gains),
        },
        "true_loao": {
            "pooled": loao_pooled,
            "held_aoa_rows": loao_rows,
            "models": loao_models,
            "positive_aoas": sum(gain > 0.0 for gain in loao_gains),
            "worst_aoa_shadow_gain": min(loao_gains),
        },
        "full_fixed725_deployment_fit": serialize_ridge(full_model),
        "fixed_p0_bounds_identity_after": bounds_identity_after,
        "bounds_bitwise_frozen": bounds_bitwise_frozen,
        "formal_elapsed_seconds": formal_elapsed,
        "gate_checks": gate_checks,
        "decision": (
            "PASS_OBSERVATION_GATED_POINT_BLEND"
            if all(gate_checks.values())
            else "STOP_OBSERVATION_GATED_POINT_BLEND"),
        "limitations": [
            "Public package filenames motivated only a hypothesis; their "
            "semantics are not treated as evidence.",
            "The target-aware oracle is a deterministic numerical optimizer, "
            "not a mathematical proof of the continuous global maximum.",
            "The final ridge is fit to oracle labels from fixed725; only "
            "cross-fit rows estimate held sample/condition behavior.",
            "OOF UQ checkpoint iteration used the held fold for selection; "
            "point backbones are fixed, but bounds remain post-selection local.",
            "No official score, ZIP packaging, latency claim, or submission is "
            "produced by this runner.",
        ],
    })
    write_report(args.output, report)
    print(json.dumps({
        "event": "decision", "decision": report["decision"],
        "oracle_shadow_gain": oracle_row["shadow_final_gain"],
        "condition_pooled_shadow_gain": condition_pooled["shadow_final_gain"],
        "condition_positive": report["condition5"]["positive_folds"],
        "condition_worst": report["condition5"]["worst_fold_shadow_gain"],
        "loao_positive": report["true_loao"]["positive_aoas"],
        "loao_worst": report["true_loao"]["worst_aoa_shadow_gain"],
    }), flush=True)


if __name__ == "__main__":
    main()
