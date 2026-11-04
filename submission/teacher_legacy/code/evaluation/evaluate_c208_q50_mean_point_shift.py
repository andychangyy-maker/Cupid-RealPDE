#!/usr/bin/env python3
"""Gate E0: frozen-C208 OOF q50 temporal-mean point correction.

The five existing fold-excluded UQ checkpoints are evaluated exactly once.
Their explicit tau=0.5 residual quantile is averaged over the 20 forecast
steps and added to the frozen point prediction on a preregistered alpha grid.
Absolute uncertainty bounds are built from the baseline point first and never
move.  Alpha is selected with calibrator-level leave-one-AoA-out evaluation.
No parameter, checkpoint, or submission archive is written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_c208_uc1_coverage_temperature as uc1  # noqa: E402
import evaluate_mean_expert_oof_blend as s1  # noqa: E402
import train_ensemble_uq_oof as uq_train  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
from evaluate_flow_backbone_blend import score as official_score  # noqa: E402
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


EXPERIMENT = "c208_e0_oof_q50_temporal_mean_point_shift"
POINT_CHECKPOINT_SHA256 = uc1.POINT_CHECKPOINT_SHA256
OOF_SUMMARY_SHA256 = uc1.OOF_SUMMARY_SHA256
POINT_STREAM_SHA256 = (
    "6421136347632f9929be134bffbe28d25feeac4a01eaec0ba0babca47866354c")
TARGET_STREAM_SHA256 = (
    "2365a2ba5d9e788445a631d17207d664964c6855e6b606221cca4e132cc51b66")
FOLD_CHECKPOINT_SHA256 = {
    0: "05699963378de94e9c5eea8846af4ec24d26a39971f218ef19063de1939b8596",
    1: "f025cb52bb454059dfa2c83e5b486e4b4d59a89aaf5b6915b7b4d196efc40a3b",
    2: "32ff859040e8f27b5d8fe1a50c92ecda72c48e05141abb61d6397df8e22b31bc",
    3: "2e704f03cdfde65203aba23915dc17e1e230336fca4510e6a89257e0b351cc79",
    4: "addee57d1171d58f33e0b7498471730eaaac146d5453c7b346cd104bab944876",
}
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
AOA_VALUES = uc1.AOA_VALUES
SHADOW_WEIGHTS = {
    "rel_l2_score": 0.47201126,
    "tke_score": 0.08586562,
    "mvpe_score": 0.09410214,
    "sps_score": 0.24699273,
}
POINT_GAIN_GATE = 0.03
NON_TIME_GAIN_GATE = 0.05
AOA_POSITIVE_GATE = 4
AOA_WORST_GATE = -0.01
TKE_TOLERANCE = 1e-5
CENTERED_FLUCTUATION_TOLERANCE = 1e-6
GPU_BUDGET_SECONDS = 5.0 * 60.0
TOTAL_BUDGET_SECONDS = 15.0 * 60.0
MIN_AVAILABLE_MEMORY_BYTES = 12 * 1024 ** 3
METRIC_KEYS = (
    "rel_l2_score", "tke_score", "mvpe_score", "sps_score",
    "point_score", "coverage", "mean_width")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def git_identity() -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "porcelain": status}


def available_memory_bytes() -> int:
    path = Path("/proc/meminfo")
    if not path.exists():
        return MIN_AVAILABLE_MEMORY_BYTES
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable row")


def temporal_mean_bias(q50_norm: torch.Tensor,
                       target_std: torch.Tensor) -> torch.Tensor:
    """Convert normalized residual q50 to a time-constant physical UV bias."""
    if (q50_norm.ndim != 5 or q50_norm.shape[-1] != 2
            or q50_norm.dtype != torch.float32):
        raise ValueError("q50 must have shape (N,T,H,W,2)")
    if target_std.shape != (2,) or target_std.dtype != torch.float32:
        raise ValueError("target_std must contain exactly the UV scales")
    if (not bool(torch.isfinite(q50_norm).all())
            or not bool(torch.isfinite(target_std).all())):
        raise FloatingPointError("q50 or target_std is non-finite")
    return (
        q50_norm.mean(dim=1, keepdim=True)
        * target_std.view(1, 1, 1, 1, 2))


def candidate_point(base: np.ndarray, bias: np.ndarray, alpha: float) -> np.ndarray:
    if alpha not in ALPHAS:
        raise ValueError("alpha differs from the frozen E0 grid")
    if base.ndim != 5 or bias.shape != (
            len(base), 1, base.shape[2], base.shape[3], 2):
        raise ValueError("base/bias shapes do not match the E0 contract")
    if alpha == 0.0:
        return base.copy()
    return base + np.float32(alpha) * bias


def pad_zero_pressure(uv: np.ndarray) -> np.ndarray:
    if uv.ndim != 5 or uv.shape[-1] != 2 or uv.dtype != np.float32:
        raise ValueError("pressure padding requires float32 UV fields")
    result = np.zeros((*uv.shape[:-1], 3), dtype=np.float32)
    result[..., :2] = uv
    return result


def score_bundle(prediction: np.ndarray, target: np.ndarray,
                 lower: np.ndarray, upper: np.ndarray) -> dict[str, float | int]:
    """Call the official NumPy scorer on one complete pooled partition."""
    metrics = official_score(prediction, lower, upper, target)
    channels = 2
    scored = target[..., :channels] != 0.0
    inside = (
        (target[..., :channels] >= lower[..., :channels])
        & (target[..., :channels] <= upper[..., :channels]))
    metrics["covered_elements"] = int(np.count_nonzero(inside & scored))
    metrics["scored_elements"] = int(np.count_nonzero(scored))
    return metrics


def metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in METRIC_KEYS
    }


def non_time_shadow(delta: dict[str, float]) -> float:
    return float(sum(SHADOW_WEIGHTS[key] * delta[key]
                     for key in SHADOW_WEIGHTS))


def select_alpha_on_mask(
        candidates: dict[float, np.ndarray], target: np.ndarray,
        lower: np.ndarray, upper: np.ndarray, calibration_mask: np.ndarray,
        ) -> dict:
    """Select alpha using only the explicitly supplied calibration rows."""
    calibration_mask = np.asarray(calibration_mask, dtype=bool)
    if calibration_mask.shape != (len(target),) or not calibration_mask.any():
        raise ValueError("calibration mask is empty or misaligned")
    baseline = score_bundle(
        candidates[0.0][calibration_mask], target[calibration_mask],
        lower[calibration_mask], upper[calibration_mask])
    rows = []
    best_alpha = None
    best_shadow = -float("inf")
    for alpha in ALPHAS:
        metrics = score_bundle(
            candidates[alpha][calibration_mask], target[calibration_mask],
            lower[calibration_mask], upper[calibration_mask])
        delta = metric_delta(metrics, baseline)
        shadow = non_time_shadow(delta)
        rows.append({
            "alpha": alpha,
            "metrics": metrics,
            "delta": delta,
            "non_time_shadow_delta": shadow,
        })
        if shadow > best_shadow:
            best_alpha = alpha
            best_shadow = shadow
    if best_alpha is None:
        raise RuntimeError("alpha selection produced no candidate")
    return {
        "selected_alpha": float(best_alpha),
        "selected_non_time_shadow_delta": float(best_shadow),
        "calibration_samples": int(calibration_mask.sum()),
        "baseline": baseline,
        "rows": rows,
    }


def select_loao_alphas(
        candidates: dict[float, np.ndarray], target: np.ndarray,
        lower: np.ndarray, upper: np.ndarray, aoas: np.ndarray,
        groups: tuple[int, ...] = AOA_VALUES,
        ) -> dict[int, dict]:
    """Fit each held-AoA alpha without passing any held row to its selector."""
    aoas = np.asarray(aoas)
    if aoas.shape != (len(target),) or set(aoas.tolist()) != set(groups):
        raise ValueError("AoA rows differ from the frozen E0 groups")
    return {
        held: select_alpha_on_mask(
            candidates, target, lower, upper, aoas != held)
        for held in groups
    }


def centered_fluctuation_max(base: np.ndarray, candidate: np.ndarray) -> float:
    base_centered = base - base.mean(axis=1, keepdims=True, dtype=np.float32)
    candidate_centered = candidate - candidate.mean(
        axis=1, keepdims=True, dtype=np.float32)
    return float(np.max(np.abs(candidate_centered - base_centered)))


def point_outside_bounds_rate(point: np.ndarray, target: np.ndarray,
                              lower: np.ndarray, upper: np.ndarray) -> float:
    scored = target != 0.0
    outside = (point < lower) | (point > upper)
    return float(np.count_nonzero(outside & scored) / np.count_nonzero(scored))


def build_gate_checks(
        pooled_delta: dict[str, float], loao_rows: list[dict],
        hard_guards: dict[str, bool], elapsed: float, gpu_seconds: float,
        ) -> dict[str, bool]:
    row_shadows = [float(row["non_time_shadow_delta"]) for row in loao_rows]
    checks = dict(hard_guards)
    checks.update({
        "loao_pooled_point_gain_ge_0_03": (
            pooled_delta["point_score"] >= POINT_GAIN_GATE),
        "loao_pooled_non_time_gain_ge_0_05": (
            non_time_shadow(pooled_delta) >= NON_TIME_GAIN_GATE),
        "loao_pooled_rel_nonnegative": pooled_delta["rel_l2_score"] >= 0.0,
        "loao_pooled_mvpe_nonnegative": pooled_delta["mvpe_score"] >= 0.0,
        "loao_pooled_sps_nonnegative": pooled_delta["sps_score"] >= 0.0,
        "loao_pooled_tke_within_1e_5": (
            abs(pooled_delta["tke_score"]) <= TKE_TOLERANCE),
        "loao_positive_aoa_ge_4_of_5": (
            sum(value > 0.0 for value in row_shadows) >= AOA_POSITIVE_GATE),
        "loao_worst_aoa_ge_minus_0_01": min(row_shadows) >= AOA_WORST_GATE,
        "gpu_within_5_minutes": gpu_seconds <= GPU_BUDGET_SECONDS,
        "wall_within_15_minutes": elapsed <= TOTAL_BUDGET_SECONDS,
    })
    return checks


def _ordered_digest() -> "hashlib._Hash":
    return hashlib.sha256()


def _digest_tensor(digest, value: torch.Tensor) -> None:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest.update(memoryview(array).cast("B"))


def _validate_terminal_checkpoints(summary: dict) -> None:
    rows = summary.get("folds", [])
    if len(rows) != 5:
        raise ValueError("E0 requires exactly five OOF folds")
    for fold, row in enumerate(rows):
        history = row.get("history", [])
        if (row.get("fold") != fold or row.get("best_iteration") != 800
                or len(history) != 1 or history[0].get("iteration") != 800):
            raise ValueError("OOF folds are not terminal-update-800 checkpoints")


def _collect_oof(args: argparse.Namespace, started: float) -> dict:
    identities = {
        "point_checkpoint": s1.file_identity(args.point_checkpoint),
        "oof_summary": s1.file_identity(args.oof_dir / "summary.json"),
        "split_manifest": s1.file_identity(args.split_manifest),
        "real_stats": s1.file_identity(args.real_stats),
        "fold_checkpoints": {
            str(fold): s1.file_identity(
                args.oof_dir / f"fold_{fold:02d}_best.pth")
            for fold in range(5)},
    }
    if identities["point_checkpoint"]["sha256"] != POINT_CHECKPOINT_SHA256:
        raise ValueError("point checkpoint SHA differs from E0")
    if identities["oof_summary"]["sha256"] != OOF_SUMMARY_SHA256:
        raise ValueError("OOF summary SHA differs from E0")
    if identities["split_manifest"]["sha256"] != s1.S1_SPLIT_MANIFEST_SHA256:
        raise ValueError("split manifest SHA differs from E0")
    if identities["real_stats"]["sha256"] != s1.S1_REAL_STATS_SHA256:
        raise ValueError("real stats SHA differs from E0")
    for fold, expected in FOLD_CHECKPOINT_SHA256.items():
        if identities["fold_checkpoints"][str(fold)]["sha256"] != expected:
            raise ValueError(f"fold {fold} checkpoint SHA differs from E0")

    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    point_windows, tail_windows, runtime_manifest = common.repro.make_split(
        metadata, 20, 0.2)
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    sample_folds, fold_manifest = trajectory_folds(runtime_manifest, 5)
    s1.validate_s1_split(
        runtime_manifest, frozen_manifest, len(tail_windows), sample_folds)
    if len(point_windows) != 2576 or len(tail_windows) != 725:
        raise ValueError("frozen split is not 2576/725")
    summary = uc1.validate_summary(
        args.oof_dir / "summary.json", args.point_checkpoint, fold_manifest)
    _validate_terminal_checkpoints(summary)

    dataset = common.repro.WindowDataset(trajectories, tail_windows)
    original_aoas = np.asarray([
        int(metadata[window.trajectory]["aoa"]) for window in tail_windows],
        dtype=np.int64)
    if set(original_aoas.tolist()) != set(AOA_VALUES):
        raise ValueError("tail AoA values differ from E0")
    payload = torch.load(
        args.point_checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    loader_args = argparse.Namespace(num_workers=0)

    chunks: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "point", "target", "lower", "upper", "bias", "solid_mask")}
    ordered_indices: list[int] = []
    point_digest = _ordered_digest()
    target_digest = _ordered_digest()
    q50_digest = _ordered_digest()
    lower_digest = _ordered_digest()
    upper_digest = _ordered_digest()
    history_before_digest = _ordered_digest()
    history_after_digest = _ordered_digest()
    target_after_digest = _ordered_digest()
    fold_metadata = []
    reference_config = None
    reference_levels = None
    gpu_seconds = 0.0

    with torch.inference_mode():
        for fold in range(5):
            if time.perf_counter() - started > TOTAL_BUDGET_SECONDS:
                raise TimeoutError("E0 exceeded the wall budget during OOF inference")
            indices = [
                index for index in range(len(dataset))
                if sample_folds[index] == fold]
            loader = uq_train.make_loader(
                dataset, indices, args.batch_size, False, loader_args, seed=0)
            model, model_metadata = s1.load_fold_model(
                payload, args.oof_dir / f"fold_{fold:02d}_best.pth",
                fold, args.point_checkpoint, normalizer, device)
            config = model_metadata["model_config"]
            levels = tuple(float(value) for value in model.config.quantile_levels)
            middle = len(levels) // 2
            if (len(levels) % 2 != 1 or levels[middle] != 0.5
                    or any(a >= b for a, b in zip(levels, levels[1:]))):
                raise ValueError("fold quantile levels lack a unique middle tau=0.5")
            if reference_config is None:
                reference_config = config
                reference_levels = levels
            elif config != reference_config or levels != reference_levels:
                raise ValueError("fold model configs or quantile levels differ")
            fold_metadata.append(model_metadata)
            offset = 0
            for history, target_full in loader:
                _digest_tensor(history_before_digest, history)
                _digest_tensor(target_digest, target_full[..., :2])
                history_device = history.to(device, non_blocking=True)
                target_device = target_full.to(device, non_blocking=True)
                history_norm, _ = normalizer.preprocess(
                    history_device, target_device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                gpu_started = time.perf_counter()
                (prediction_norm, _members, quantiles, lower_norm,
                 upper_norm) = model.forward_quantiles(history_norm)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                gpu_seconds += time.perf_counter() - gpu_started

                prediction = common.to_physical_target(
                    prediction_norm, normalizer)[..., :2]
                target = target_device[..., :2]
                scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
                q50 = quantiles[..., middle]
                q50_physical = q50 * scale
                bias = temporal_mean_bias(q50, normalizer.std_target[:2])
                lower_extent = lower_norm * scale
                upper_extent = upper_norm * scale
                lower = prediction - lower_extent
                upper = prediction + upper_extent
                if (not bool(torch.isfinite(q50).all())
                        or not bool(torch.isfinite(bias).all())
                        or not bool(torch.isfinite(lower).all())
                        or not bool(torch.isfinite(upper).all())
                        or bool((lower > upper).any())):
                    raise FloatingPointError("non-finite or unordered E0 tensors")
                solid = (history_device[..., :2] == 0).all(dim=1).all(dim=-1)
                _digest_tensor(point_digest, prediction)
                _digest_tensor(target_after_digest, target)
                _digest_tensor(q50_digest, q50_physical)
                _digest_tensor(lower_digest, lower)
                _digest_tensor(upper_digest, upper)
                _digest_tensor(history_after_digest, history)
                for key, tensor in (
                        ("point", prediction), ("target", target),
                        ("lower", lower), ("upper", upper),
                        ("bias", bias), ("solid_mask", solid)):
                    chunks[key].append(
                        np.ascontiguousarray(tensor.detach().cpu().numpy()))
                count = len(target)
                ordered_indices.extend(indices[offset:offset + count])
                offset += count
            if offset != len(indices):
                raise RuntimeError("fold loader did not preserve all original indices")
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if point_digest.hexdigest() != POINT_STREAM_SHA256:
        raise RuntimeError("fold-order point stream SHA differs from E0")
    if target_digest.hexdigest() != TARGET_STREAM_SHA256:
        raise RuntimeError("fold-order target stream SHA differs from E0")
    if target_after_digest.hexdigest() != target_digest.hexdigest():
        raise RuntimeError("target changed during E0 inference")
    if history_after_digest.hexdigest() != history_before_digest.hexdigest():
        raise RuntimeError("history changed during E0 inference")

    original_indices = np.asarray(ordered_indices, dtype=np.int64)
    if (len(original_indices) != 725
            or not np.array_equal(np.sort(original_indices), np.arange(725))):
        raise RuntimeError("fixed725 original indices are duplicated or missing")
    order = np.argsort(original_indices)
    arrays = {
        key: np.concatenate(values, axis=0)[order]
        for key, values in chunks.items()
    }
    for key in ("point", "target", "lower", "upper", "bias"):
        if arrays[key].dtype != np.float32 or not np.all(np.isfinite(arrays[key])):
            raise FloatingPointError(f"{key} cache is not finite float32")
    if not np.all(arrays["lower"] <= arrays["upper"]):
        raise RuntimeError("absolute bounds are unordered after reindexing")
    arrays["aoas"] = original_aoas
    arrays["folds"] = np.asarray(sample_folds, dtype=np.int64)
    return {
        "arrays": arrays,
        "identities": identities,
        "summary": summary,
        "fold_metadata": fold_metadata,
        "quantile_levels": reference_levels,
        "fold_order_original_indices": original_indices,
        "stream_hashes": {
            "fold_order_point": point_digest.hexdigest(),
            "fold_order_target": target_digest.hexdigest(),
            "fold_order_history": history_before_digest.hexdigest(),
            "fold_order_q50_physical": q50_digest.hexdigest(),
            "fold_order_absolute_lower": lower_digest.hexdigest(),
            "fold_order_absolute_upper": upper_digest.hexdigest(),
        },
        "gpu_seconds": gpu_seconds,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    memory_available = available_memory_bytes()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch_size != 16 or args.num_workers != 0:
        raise ValueError("formal E0 requires batch-size=16 and num-workers=0")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal E0 requires CUDA")
    uc1.configure_replay(uc1.EXPECTED_TRAINING_CONFIG["seed"])
    collected = _collect_oof(args, started)
    arrays = collected.pop("arrays")
    base = arrays["point"]
    target = arrays["target"]
    lower = arrays["lower"]
    upper = arrays["upper"]
    bias = arrays["bias"]
    aoas = arrays["aoas"]
    folds = arrays["folds"]
    solid_mask = arrays["solid_mask"]

    initial_hashes = {
        key: array_sha256(value)
        for key, value in {
            "point": base, "target": target, "lower": lower,
            "upper": upper, "bias": bias,
        }.items()
    }
    for value in (base, target, lower, upper, bias, aoas, folds, solid_mask):
        value.setflags(write=False)
    candidates = {
        alpha: candidate_point(base, bias, alpha) for alpha in ALPHAS}
    if not np.array_equal(candidates[0.0], base):
        raise RuntimeError("alpha=0 does not reproduce the point baseline")

    baseline = score_bundle(base, target, lower, upper)
    baseline_folds = [
        score_bundle(base[folds == fold], target[folds == fold],
                     lower[folds == fold], upper[folds == fold])
        for fold in range(5)
    ]
    baseline_replay = uc1.baseline_replay_gate(
        collected["summary"]["selected_oof"], baseline,
        [row["selected"] for row in collected["summary"]["folds"]],
        baseline_folds)
    if not baseline_replay["pass"]:
        raise RuntimeError("E0 baseline failed the frozen OOF replay gate")

    full_grid = []
    invariant_rows = []
    for alpha in ALPHAS:
        metrics = score_bundle(candidates[alpha], target, lower, upper)
        delta = metric_delta(metrics, baseline)
        fluctuation = centered_fluctuation_max(base, candidates[alpha])
        full_grid.append({
            "alpha": alpha, "metrics": metrics, "delta": delta,
            "non_time_shadow_delta": non_time_shadow(delta),
        })
        invariant_rows.append({
            "alpha": alpha,
            "lower_sha256": initial_hashes["lower"],
            "upper_sha256": initial_hashes["upper"],
            "coverage_exact": metrics["coverage"] == baseline["coverage"],
            "covered_elements_exact": (
                metrics["covered_elements"] == baseline["covered_elements"]),
            "scored_elements_exact": (
                metrics["scored_elements"] == baseline["scored_elements"]),
            "mean_width_exact": metrics["mean_width"] == baseline["mean_width"],
            "tke_delta_within_1e_5": (
                abs(delta["tke_score"]) <= TKE_TOLERANCE),
            "centered_fluctuation_max_abs": fluctuation,
            "point_outside_bounds_rate": point_outside_bounds_rate(
                candidates[alpha], target, lower, upper),
        })

    loao_selection = select_loao_alphas(
        candidates, target, lower, upper, aoas)
    loao_candidate = base.copy()
    loao_rows = []
    for held in AOA_VALUES:
        held_mask = aoas == held
        alpha = loao_selection[held]["selected_alpha"]
        loao_candidate[held_mask] = candidates[alpha][held_mask]
        before = score_bundle(
            base[held_mask], target[held_mask],
            lower[held_mask], upper[held_mask])
        after = score_bundle(
            candidates[alpha][held_mask], target[held_mask],
            lower[held_mask], upper[held_mask])
        delta = metric_delta(after, before)
        loao_rows.append({
            "held_aoa": held,
            "selected_alpha": alpha,
            "baseline": before,
            "candidate": after,
            "delta": delta,
            "non_time_shadow_delta": non_time_shadow(delta),
            "selection": loao_selection[held],
        })
    loao_metrics = score_bundle(loao_candidate, target, lower, upper)
    loao_delta = metric_delta(loao_metrics, baseline)
    loao_centered_max = centered_fluctuation_max(base, loao_candidate)

    full_best = None
    full_best_shadow = -float("inf")
    for row in full_grid:
        if row["non_time_shadow_delta"] > full_best_shadow:
            full_best = row
            full_best_shadow = row["non_time_shadow_delta"]
    if full_best is None:
        raise RuntimeError("full OOF alpha diagnostic has no row")
    full_selection = {
        "selected_alpha": full_best["alpha"],
        "selected_non_time_shadow_delta": full_best_shadow,
        "calibration_samples": len(target),
        "baseline": baseline,
        "rows": full_grid,
    }
    final_hashes = {
        key: array_sha256(value)
        for key, value in {
            "point": base, "target": target, "lower": lower,
            "upper": upper, "bias": bias,
        }.items()
    }
    immutable_inputs = final_hashes == initial_hashes
    all_alpha_invariants = all(
        row["coverage_exact"] and row["covered_elements_exact"]
        and row["scored_elements_exact"] and row["mean_width_exact"]
        and row["lower_sha256"] == initial_hashes["lower"]
        and row["upper_sha256"] == initial_hashes["upper"]
        and row["tke_delta_within_1e_5"]
        and row["centered_fluctuation_max_abs"]
        <= CENTERED_FLUCTUATION_TOLERANCE
        for row in invariant_rows)
    base_full = pad_zero_pressure(base)
    loao_full = pad_zero_pressure(loao_candidate)
    pressure_contract = bool(
        np.array_equal(base_full[..., :2], base)
        and np.array_equal(loao_full[..., :2], loao_candidate)
        and np.count_nonzero(base_full[..., 2]) == 0
        and np.count_nonzero(loao_full[..., 2]) == 0)
    del base_full, loao_full
    masked_bias = bias[:, 0][solid_mask]
    masked_bias_rms = (
        float(np.sqrt(np.mean(masked_bias.astype(np.float64) ** 2)))
        if masked_bias.size else 0.0)
    elapsed = time.perf_counter() - started
    hard_guards = {
        "baseline_replay_pass": bool(baseline_replay["pass"]),
        "fixed725_original_indices_permutation": bool(
            np.array_equal(
                np.sort(collected["fold_order_original_indices"]),
                np.arange(725))),
        "fold_point_stream_sha_frozen": (
            collected["stream_hashes"]["fold_order_point"]
            == POINT_STREAM_SHA256),
        "fold_target_stream_sha_frozen": (
            collected["stream_hashes"]["fold_order_target"]
            == TARGET_STREAM_SHA256),
        "fold_configs_and_q50_levels_consistent": bool(
            collected["quantile_levels"] is not None
            and len(collected["quantile_levels"]) % 2 == 1
            and collected["quantile_levels"][
                len(collected["quantile_levels"]) // 2] == 0.5),
        "q50_bias_candidate_finite": bool(
            np.all(np.isfinite(bias))
            and all(np.all(np.isfinite(value)) for value in candidates.values())),
        "alpha0_point_bitwise_identity": bool(
            np.array_equal(candidates[0.0], base)),
        "all_alpha_bounds_coverage_tke_centered_invariants": bool(
            all_alpha_invariants),
        "immutable_point_target_bounds_bias": bool(immutable_inputs),
        "loao_centered_fluctuation_within_1e_6": (
            loao_centered_max <= CENTERED_FLUCTUATION_TOLERANCE),
        "pressure_constructed_and_verified_zero": pressure_contract,
        "no_training_checkpoint_or_zip": True,
    }
    checks = build_gate_checks(
        loao_delta, loao_rows, hard_guards, elapsed,
        collected["gpu_seconds"])
    decision = (
        "PASS_Q50_MEAN_TO_ACTUAL_PACKAGE_PREFLIGHT"
        if all(checks.values()) else "STOP_Q50_MEAN_NO_GENERALIZABLE_SIGNAL")
    report = {
        "experiment": EXPERIMENT,
        "evidence_class": (
            "fixed725 fold-excluded UQ plus calibrator-level LOAO; not "
            "official evidence and not full-model true LOAO"),
        "contract": {
            "formula": (
                "point'=base_point+alpha*mean_t(q50_norm)*std_target_uv"),
            "alphas": list(ALPHAS),
            "tie_break": "strict unrounded shadow maximum; ascending alpha",
            "absolute_bounds": "built from base point first and frozen",
            "non_time_shadow_weights": SHADOW_WEIGHTS,
            "full_oof_alpha_participates_in_gate": False,
            "trainable_parameters": 0,
        },
        "git": git_identity(),
        "identities": collected["identities"],
        "fold_metadata": collected["fold_metadata"],
        "quantile_levels": list(collected["quantile_levels"]),
        "stream_hashes": {
            **collected["stream_hashes"],
            "original_order_point": initial_hashes["point"],
            "original_order_target": initial_hashes["target"],
            "original_order_lower": initial_hashes["lower"],
            "original_order_upper": initial_hashes["upper"],
            "original_order_q50_mean_bias": initial_hashes["bias"],
        },
        "baseline": baseline,
        "baseline_replay_gate": baseline_replay,
        "full_oof_grid_diagnostic": full_grid,
        "full_oof_deployment_alpha_diagnostic": full_selection,
        "loao": {
            "pooled": loao_metrics,
            "pooled_delta": loao_delta,
            "pooled_non_time_shadow_delta": non_time_shadow(loao_delta),
            "rows": loao_rows,
            "limitation": (
                "Only alpha excludes the held AoA; frozen UQ fold models may "
                "have seen other trajectories at that AoA."),
        },
        "invariants": {
            "rows": invariant_rows,
            "loao_centered_fluctuation_max_abs": loao_centered_max,
            "history_observation_zero_support_bias_rms": masked_bias_rms,
            "history_observation_zero_support_values": int(masked_bias.size),
            "initial_hashes": initial_hashes,
            "final_hashes": final_hashes,
        },
        "gate_checks": checks,
        "decision": decision,
        "elapsed_seconds": elapsed,
        "gpu_inference_seconds": collected["gpu_seconds"],
        "resources": {
            "memory_available_bytes_at_start": memory_available,
            "minimum_required_bytes": MIN_AVAILABLE_MEMORY_BYTES,
        },
        "checkpoints_saved": [],
        "submission_created": False,
        "limitations": [
            "Fixed725 OOF diagnostic, not an official score.",
            "The deployment alpha is same-data OOF diagnostic and not a GO input.",
            "A GO only authorizes actual-package preflight and latency checks.",
            "No q-level, alpha, horizon, channel, region, or sign expansion is allowed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "e0_complete",
        "decision": decision,
        "full_oof_alpha": full_selection["selected_alpha"],
        "loao_point_delta": loao_delta["point_score"],
        "loao_sps_delta": loao_delta["sps_score"],
        "loao_non_time_shadow_delta": non_time_shadow(loao_delta),
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    try:
        if available_memory_bytes() < MIN_AVAILABLE_MEMORY_BYTES:
            raise MemoryError("formal E0 requires at least 12 GiB available RAM")
        main()
    except Exception as error:
        try:
            error_args = parse_args()
            if not error_args.output.exists():
                error_args.output.parent.mkdir(parents=True, exist_ok=True)
                error_args.output.write_text(
                    json.dumps({
                        "experiment": EXPERIMENT,
                        "decision": "STOP_Q50_MEAN_ERROR",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "git": git_identity(),
                        "checkpoints_saved": [],
                        "submission_created": False,
                    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
        except Exception:
            pass
        raise
