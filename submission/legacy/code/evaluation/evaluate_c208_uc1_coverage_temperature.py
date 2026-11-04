#!/usr/bin/env python3
"""UC1: frozen-C208 OOF coverage-temperature calibration.

The five existing fold-excluded UQ models are never updated.  Pass one fits
only horizon/channel coverage sufficient statistics.  Pass two evaluates the
resulting 80-scalar calibrator with outer-fold and leave-one-AoA-out fitting.
No point value, checkpoint, or submission archive is written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_mean_expert_oof_blend as s1  # noqa: E402
import train_ensemble_uq_oof as uq_train  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402
from train_uq_budget_reallocator import trajectory_folds  # noqa: E402


EXPERIMENT = "c208_uc1_oof_coverage_temperature"
POINT_CHECKPOINT_SHA256 = (
    "79b3f68272580f80f1d9e09a54e9f9f5291e74524c8b11cdfd78b77a1566cbc3")
OOF_SUMMARY_SHA256 = (
    "995eb108b2d3e56324690b0fb73c80f78ddd2f3c4a1a8fff08d1219da91f7668")
BASELINE_OOF_SPS = 53.4471644789
AOA_VALUES = (0, 5, 10, 15, 20)
BIAS_LIMIT = 2.0
LOG_SLOPE_LIMIT = 0.75
POOLED_GAIN_GATE = 0.10
FOLD_POSITIVE_GATE = 4
FOLD_WORST_GATE = -0.03
LOAO_POSITIVE_GATE = 4
LOAO_WORST_GATE = -0.05
TOTAL_BUDGET_SECONDS = 15.0 * 60.0
BASELINE_SPS_TOLERANCE = 1e-4
BASELINE_MEAN_WIDTH_TOLERANCE = 5e-7
BASELINE_POOLED_COVERAGE_COUNT_TOLERANCE = 128
BASELINE_FOLD_COVERAGE_COUNT_TOLERANCE = 32

EXPECTED_TRAINING_CONFIG = {
    "folds": 5,
    "fold_updates": 800,
    "final_updates": 800,
    "fit_final": True,
    "batch_size": 8,
    "eval_batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "seed": 12,
    "pinball_weight": 0.05,
    "soft_sps_weight": 2.0,
    "soft_tau_fraction": 0.035,
    "cvar_weight": 0.35,
    "cvar_fraction": 0.25,
    "interval_score_weight": 0.0025,
    "coverage_target": 0.85,
    "utility_refinement": True,
    "utility_refinement_kernel": 3,
    "utility_refinement_blocks": 0,
    "utility_refinement_log_limit": 0.75,
    "temporal_uq_refinement": True,
    "temporal_uq_channels": 24,
    "temporal_uq_blocks": 2,
    "temporal_uq_kernel": 3,
    "temporal_uq_logit_limit": 1.5,
    "coverage_calibration": False,
    "train_projections": True,
    "window_stride": 20,
    "val_fraction": 0.2,
    "max_files": 0,
}


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


def configure_replay(seed: int) -> None:
    """Restore the numerical preamble used by the frozen OOF runner."""
    uq_train.seed_everything(seed)
    torch.set_float32_matmul_precision("high")


def nominal_masses(levels: tuple[float, ...],
                   pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    masses = np.asarray(
        [levels[upper] - levels[lower] for lower, upper in pairs],
        dtype=np.float64)
    if (masses.ndim != 1 or len(masses) == 0
            or not np.all((masses > 0.0) & (masses < 1.0))):
        raise ValueError("invalid quantile candidate masses")
    return masses


def fit_temperature(hits: np.ndarray, totals: np.ndarray,
                    masses: np.ndarray) -> dict[str, np.ndarray]:
    """Fit bounded Platt parameters from binomial sufficient statistics."""
    hits = np.asarray(hits, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if (hits.shape != totals.shape or hits.ndim != 3
            or hits.shape[-1] != len(masses)):
        raise ValueError("coverage statistics must have shape (T,C,K)")
    if (not np.all(np.isfinite(hits))
            or not np.all(np.isfinite(totals))
            or np.any(hits < 0.0) or np.any(hits > totals)
            or np.any(totals <= 0.0)):
        raise ValueError("invalid coverage sufficient statistics")

    hit_tensor = torch.from_numpy(hits)
    total_tensor = torch.from_numpy(totals)
    base_logits = torch.logit(torch.from_numpy(masses)).view(1, 1, -1)
    raw = torch.zeros((*hits.shape[:2], 2), dtype=torch.float64,
                      requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [raw], lr=0.8, max_iter=64, tolerance_grad=1e-11,
        tolerance_change=1e-13, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        bias = BIAS_LIMIT * torch.tanh(raw[..., 0:1])
        slope = torch.exp(LOG_SLOPE_LIMIT * torch.tanh(raw[..., 1:2]))
        logits = bias + slope * base_logits
        nll = torch.nn.functional.softplus(logits) * total_tensor
        nll = nll - logits * hit_tensor
        loss = (nll.sum(dim=-1) / total_tensor.sum(dim=-1)).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        bias = BIAS_LIMIT * torch.tanh(raw[..., 0])
        slope = torch.exp(LOG_SLOPE_LIMIT * torch.tanh(raw[..., 1]))
    result = {
        "bias": bias.numpy(),
        "slope": slope.numpy(),
    }
    if (not np.all(np.isfinite(result["bias"]))
            or not np.all(np.isfinite(result["slope"]))):
        raise FloatingPointError("non-finite fitted calibration")
    return result


def calibrated_probabilities(parameters: dict[str, np.ndarray],
                             masses: np.ndarray) -> np.ndarray:
    bias = np.asarray(parameters["bias"], dtype=np.float64)
    slope = np.asarray(parameters["slope"], dtype=np.float64)
    if bias.shape != slope.shape or bias.ndim != 2:
        raise ValueError("calibration parameters must have shape (T,C)")
    logits = np.log(masses / (1.0 - masses)).reshape(1, 1, -1)
    values = 1.0 / (1.0 + np.exp(-(bias[..., None]
                                   + slope[..., None] * logits)))
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite calibrated probabilities")
    return values


def leave_one_group_out(
        hits: dict[int, np.ndarray], totals: dict[int, np.ndarray],
        groups: tuple[int, ...], masses: np.ndarray,
        ) -> dict[int, dict[str, np.ndarray]]:
    if set(hits) != set(groups) or set(totals) != set(groups):
        raise ValueError("coverage groups differ from the frozen contract")
    return {
        held: fit_temperature(
            sum((hits[group] for group in groups if group != held),
                np.zeros_like(hits[held])),
            sum((totals[group] for group in groups if group != held),
                np.zeros_like(totals[held])),
            masses)
        for held in groups
    }


def identity_select(raw_physical_width: torch.Tensor,
                    valid: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    """Exact baseline path: use nominal mass directly, never logit round-trip."""
    probability = masses.view(1, 1, 1, 1, 1, -1)
    utility = probability * torch.exp(-raw_physical_width / common.SPS_SIGMA)
    utility = torch.where(valid, utility, utility.new_full((), -1.0))
    return utility.argmax(dim=-1, keepdim=True)


def calibrated_select(raw_physical_width: torch.Tensor,
                      valid: torch.Tensor,
                      probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim == 3:
        probabilities = probabilities.unsqueeze(0)
    if probabilities.ndim != 4:
        raise ValueError("probabilities must have shape (T,C,K) or (B,T,C,K)")
    probability = probabilities[:, :, None, None, :, :]
    utility = probability * torch.exp(-raw_physical_width / common.SPS_SIGMA)
    utility = torch.where(valid, utility, utility.new_full((), -1.0))
    return utility.argmax(dim=-1, keepdim=True)


def gather_candidate(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    return values.gather(-1, selected).squeeze(-1)


class SPSAccumulator:
    def __init__(self) -> None:
        self.numerator = 0.0
        self.scored = 0
        self.inside = 0
        self.width = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor,
               lower: torch.Tensor, upper: torch.Tensor) -> None:
        if len(prediction) == 0:
            return
        scored = target != 0.0
        inside = (target - prediction >= -lower) & (target - prediction <= upper)
        factor = common.sample_sps_factor(prediction, target).view(
            -1, 1, 1, 1, 1)
        value = factor * inside * torch.exp(
            -(lower + upper) / common.SPS_SIGMA)
        self.numerator += float(value[scored].sum(dtype=torch.float64))
        self.scored += int(scored.sum())
        self.inside += int((inside & scored).sum())
        self.width += float(
            (lower + upper)[scored].sum(dtype=torch.float64))

    def result(self) -> dict[str, float | int]:
        if self.scored <= 0:
            raise RuntimeError("empty SPS accumulator")
        return {
            "sps_score": 100.0 * self.numerator / self.scored,
            "coverage": self.inside / self.scored,
            "covered_elements": self.inside,
            "mean_width": self.width / self.scored,
            "scored_elements": self.scored,
        }


def baseline_partition_checks(
        expected: dict, actual: dict, coverage_count_tolerance: int,
        ) -> dict[str, object]:
    expected_scored = int(expected["scored_elements"])
    actual_scored = int(actual["scored_elements"])
    expected_covered = round(float(expected["coverage"]) * expected_scored)
    actual_covered = int(actual.get(
        "covered_elements", round(float(actual["coverage"]) * actual_scored)))
    deltas = {
        "sps_score": float(actual["sps_score"] - expected["sps_score"]),
        "coverage_count": actual_covered - expected_covered,
        "coverage": float(actual["coverage"] - expected["coverage"]),
        "mean_width": float(actual["mean_width"] - expected["mean_width"]),
    }
    checks = {
        "scored_elements_exact": actual_scored == expected_scored,
        "sps_within_tolerance": (
            abs(deltas["sps_score"]) <= BASELINE_SPS_TOLERANCE),
        "coverage_count_within_tolerance": (
            abs(deltas["coverage_count"]) <= coverage_count_tolerance),
        "mean_width_within_tolerance": (
            abs(deltas["mean_width"]) <= BASELINE_MEAN_WIDTH_TOLERANCE),
    }
    return {
        "expected": expected,
        "actual": actual,
        "deltas": deltas,
        "coverage_count_tolerance": coverage_count_tolerance,
        "checks": checks,
        "pass": all(checks.values()),
    }


def baseline_replay_gate(
        expected_pooled: dict, actual_pooled: dict,
        expected_folds: list[dict], actual_folds: list[dict],
        ) -> dict[str, object]:
    if len(expected_folds) != 5 or len(actual_folds) != 5:
        raise ValueError("baseline replay requires exactly five folds")
    pooled = baseline_partition_checks(
        expected_pooled, actual_pooled,
        BASELINE_POOLED_COVERAGE_COUNT_TOLERANCE)
    folds = [
        {
            "fold": fold,
            **baseline_partition_checks(
                expected_folds[fold], actual_folds[fold],
                BASELINE_FOLD_COVERAGE_COUNT_TOLERANCE),
        }
        for fold in range(5)
    ]
    metric_checks = {
        "sps": (
            pooled["checks"]["sps_within_tolerance"]
            and all(row["checks"]["sps_within_tolerance"] for row in folds)),
        "coverage": (
            pooled["checks"]["scored_elements_exact"]
            and pooled["checks"]["coverage_count_within_tolerance"]
            and all(
                row["checks"]["scored_elements_exact"]
                and row["checks"]["coverage_count_within_tolerance"]
                for row in folds)),
        "mean_width": (
            pooled["checks"]["mean_width_within_tolerance"]
            and all(
                row["checks"]["mean_width_within_tolerance"]
                for row in folds)),
    }
    return {
        "tolerances": {
            "sps_abs": BASELINE_SPS_TOLERANCE,
            "mean_width_abs": BASELINE_MEAN_WIDTH_TOLERANCE,
            "pooled_coverage_count_abs": (
                BASELINE_POOLED_COVERAGE_COUNT_TOLERANCE),
            "fold_coverage_count_abs": (
                BASELINE_FOLD_COVERAGE_COUNT_TOLERANCE),
        },
        "pooled": pooled,
        "folds": folds,
        "metric_checks": metric_checks,
        "pass": all(metric_checks.values()),
    }


def new_count_table(shape: tuple[int, int, int],
                    groups: tuple[int, ...]) -> tuple[dict, dict]:
    return (
        {group: np.zeros(shape, dtype=np.float64) for group in groups},
        {group: np.zeros(shape, dtype=np.float64) for group in groups},
    )


def update_counts(hits_out: np.ndarray, totals_out: np.ndarray,
                  residual: torch.Tensor, scored: torch.Tensor,
                  lower: torch.Tensor, upper: torch.Tensor,
                  valid: torch.Tensor) -> None:
    hit = ((residual.unsqueeze(-1) >= -lower)
           & (residual.unsqueeze(-1) <= upper)
           & scored.unsqueeze(-1) & valid)
    eligible = scored.unsqueeze(-1) & valid
    dims = (0, 2, 3)
    hits_out += hit.sum(dim=dims, dtype=torch.float64).cpu().numpy()
    totals_out += eligible.sum(dim=dims, dtype=torch.float64).cpu().numpy()


def candidate_tensors(model, history_norm: torch.Tensor,
                      normalizer) -> tuple:
    values = model.forward_uq_training(history_norm)
    (prediction, members, quantiles, baseline_lower, baseline_upper,
     features, coverage) = values
    if coverage is not None or model.coverage_calibration_head is not None:
        raise ValueError("UC1 baseline must not already contain coverage calibration")
    fused = model._fused_uq_features(features, members, history_norm)
    lower, upper, valid, masses = model._interval_candidates(quantiles)
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2, 1)
    raw_physical_width = (lower + upper) * scale
    selected = identity_select(raw_physical_width, valid, masses)
    if model.utility_refinement_head is not None:
        batch, steps, height, width, channels = baseline_lower.shape
        raw = model.utility_refinement_head(fused).reshape(
            batch, steps, channels, 2, height, width
        ).permute(0, 1, 4, 5, 2, 3)
        log_scale = model.config.utility_refinement_log_limit * torch.tanh(raw)
        lower = lower * torch.exp(log_scale[..., 0:1])
        upper = upper * torch.exp(log_scale[..., 1:2])
    invalid = lower + upper < 0.0
    lower, upper = (
        torch.where(invalid, -upper, lower),
        torch.where(invalid, -lower, upper),
    )
    identity_lower = gather_candidate(lower, selected)
    identity_upper = gather_candidate(upper, selected)
    if (not torch.equal(identity_lower, baseline_lower)
            or not torch.equal(identity_upper, baseline_upper)):
        raise RuntimeError("identity calibration does not reproduce baseline bounds")
    return (
        prediction, lower * scale, upper * scale, valid,
        raw_physical_width, masses, baseline_lower * scale.squeeze(-1),
        baseline_upper * scale.squeeze(-1))


def validate_summary(path: Path, point_checkpoint: Path,
                     fold_manifest: dict) -> dict:
    if s1.file_identity(path)["sha256"] != OOF_SUMMARY_SHA256:
        raise ValueError("OOF summary SHA differs from the UC1 contract")
    summary = json.loads(path.read_text(encoding="utf-8"))
    s1.validate_oof_summary(path, point_checkpoint, fold_manifest)
    config = summary.get("training_config", {})
    mismatch = {
        key: (config.get(key), expected)
        for key, expected in EXPECTED_TRAINING_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"OOF training config differs from UC1: {mismatch}")
    if abs(summary["selected_oof"]["sps_score"] - BASELINE_OOF_SPS) > 1e-10:
        raise ValueError("OOF selected SPS differs from the UC1 contract")
    return summary


def stream_digest() -> "hashlib._Hash":
    return hashlib.sha256()


def digest_tensor(digest, value: torch.Tensor) -> None:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest.update(memoryview(array).cast("B"))


def serialize_parameters(value: dict[str, np.ndarray]) -> dict[str, list]:
    return {key: array.tolist() for key, array in value.items()}


def metric_row(candidate: SPSAccumulator, baseline: SPSAccumulator,
               label: int) -> dict:
    after = candidate.result()
    before = baseline.result()
    return {
        "group": label,
        "baseline": before,
        "candidate": after,
        "sps_delta": after["sps_score"] - before["sps_score"],
    }


def gate(condition: dict, condition_rows: list[dict],
         loao_rows: list[dict], identity_parity: bool,
         point_unchanged: bool, baseline_match: bool,
         elapsed: float) -> dict[str, bool]:
    condition_deltas = [row["sps_delta"] for row in condition_rows]
    loao_deltas = [row["sps_delta"] for row in loao_rows]
    return {
        "identity_bounds_bitwise": identity_parity,
        "point_stream_bitwise_unchanged": point_unchanged,
        "baseline_matches_frozen_oof": baseline_match,
        "condition_pooled_gain_ge_0_10": condition["sps_delta"] >= POOLED_GAIN_GATE,
        "condition_positive_ge_4_of_5": sum(x > 0.0 for x in condition_deltas) >= FOLD_POSITIVE_GATE,
        "condition_worst_ge_minus_0_03": min(condition_deltas) >= FOLD_WORST_GATE,
        "loao_positive_ge_4_of_5": sum(x > 0.0 for x in loao_deltas) >= LOAO_POSITIVE_GATE,
        "loao_worst_ge_minus_0_05": min(loao_deltas) >= LOAO_WORST_GATE,
        "total_within_15_minutes": elapsed <= TOTAL_BUDGET_SECONDS,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch_size != 16 or args.num_workers != 0:
        raise ValueError("formal UC1 requires batch-size=16 and num-workers=0")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal UC1 requires CUDA")
    configure_replay(EXPECTED_TRAINING_CONFIG["seed"])
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
        raise ValueError("point checkpoint SHA differs from the UC1 contract")
    if identities["split_manifest"]["sha256"] != s1.S1_SPLIT_MANIFEST_SHA256:
        raise ValueError("split manifest SHA differs from the UC1 contract")
    if identities["real_stats"]["sha256"] != s1.S1_REAL_STATS_SHA256:
        raise ValueError("real stats SHA differs from the UC1 contract")

    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    point_windows, tail_windows, runtime_manifest = common.repro.make_split(
        metadata, 20, 0.2)
    frozen_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    sample_folds, fold_manifest = trajectory_folds(runtime_manifest, 5)
    s1.validate_s1_split(
        runtime_manifest, frozen_manifest, len(tail_windows), sample_folds)
    if len(point_windows) != 2576:
        raise ValueError("frozen point split is not 2576 windows")
    summary = validate_summary(
        args.oof_dir / "summary.json", args.point_checkpoint, fold_manifest)
    dataset = common.repro.WindowDataset(trajectories, tail_windows)
    aoas = np.asarray([
        int(metadata[window.trajectory]["aoa"]) for window in tail_windows],
        dtype=np.int64)
    if set(aoas.tolist()) != set(AOA_VALUES):
        raise ValueError("tail AoA values differ from the UC1 contract")
    payload = torch.load(
        args.point_checkpoint, map_location="cpu", weights_only=False)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    loader_args = argparse.Namespace(num_workers=0)
    fold_groups = tuple(range(5))
    masses_np = None
    fold_hits = fold_totals = aoa_hits = aoa_totals = None
    baseline = SPSAccumulator()
    baseline_folds = [SPSAccumulator() for _ in fold_groups]
    baseline_aoas = {aoa: SPSAccumulator() for aoa in AOA_VALUES}
    point_digest_first = stream_digest()
    target_digest_first = stream_digest()
    all_indices = list(range(len(dataset)))

    with torch.inference_mode():
        for fold in fold_groups:
            indices = [i for i in all_indices if sample_folds[i] == fold]
            loader = uq_train.make_loader(
                dataset, indices, args.batch_size, False, loader_args, seed=0)
            model, _metadata = s1.load_fold_model(
                payload, args.oof_dir / f"fold_{fold:02d}_best.pth",
                fold, args.point_checkpoint, normalizer, device)
            offset = 0
            for history, target_full in loader:
                history = history.to(device, non_blocking=True)
                target_full = target_full.to(device, non_blocking=True)
                history_norm, _ = normalizer.preprocess(history, target_full)
                target = target_full[..., :2]
                tensors = candidate_tensors(model, history_norm, normalizer)
                prediction_norm, lower, upper, valid, raw_width, masses, base_lower, base_upper = tensors
                prediction = common.to_physical_target(
                    prediction_norm, normalizer)[..., :2]
                digest_tensor(point_digest_first, prediction)
                digest_tensor(target_digest_first, target)
                baseline.update(prediction, target, base_lower, base_upper)
                baseline_folds[fold].update(
                    prediction, target, base_lower, base_upper)
                batch_aoas = aoas[indices[offset:offset + len(target)]]
                if masses_np is None:
                    masses_np = masses.detach().cpu().double().numpy()
                    shape = (target.shape[1], target.shape[-1], len(masses_np))
                    fold_hits, fold_totals = new_count_table(shape, fold_groups)
                    aoa_hits, aoa_totals = new_count_table(shape, AOA_VALUES)
                elif not np.array_equal(
                        masses_np, masses.detach().cpu().double().numpy()):
                    raise RuntimeError("candidate masses changed between folds")
                residual = target - prediction
                scored = target != 0.0
                update_counts(
                    fold_hits[fold], fold_totals[fold], residual, scored,
                    lower, upper, valid)
                for aoa in AOA_VALUES:
                    selected = torch.from_numpy(batch_aoas == aoa).to(device)
                    if bool(selected.any()):
                        update_counts(
                            aoa_hits[aoa], aoa_totals[aoa],
                            residual[selected], scored[selected], lower[selected],
                            upper[selected], valid[selected])
                        baseline_aoas[aoa].update(
                            prediction[selected], target[selected],
                            base_lower[selected], base_upper[selected])
                offset += len(target)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    baseline_result = baseline.result()
    baseline_fold_results = [row.result() for row in baseline_folds]
    baseline_replay = baseline_replay_gate(
        summary["selected_oof"], baseline_result,
        [row["selected"] for row in summary["folds"]],
        baseline_fold_results)
    baseline_match = bool(baseline_replay["pass"])
    if not baseline_match:
        replay_failure = {
            "metric_checks": baseline_replay["metric_checks"],
            "pooled": baseline_replay["pooled"]["deltas"],
            "folds": [
                {"fold": row["fold"], "deltas": row["deltas"]}
                for row in baseline_replay["folds"]],
        }
        raise RuntimeError(
            "reconstructed OOF baseline failed the three-axis replay gate: "
            f"{json.dumps(replay_failure, sort_keys=True)}")
    condition_parameters = leave_one_group_out(
        fold_hits, fold_totals, fold_groups, masses_np)
    loao_parameters = leave_one_group_out(
        aoa_hits, aoa_totals, AOA_VALUES, masses_np)
    full_parameters = fit_temperature(
        sum(fold_hits.values()), sum(fold_totals.values()), masses_np)

    condition = SPSAccumulator()
    condition_folds = [SPSAccumulator() for _ in fold_groups]
    loao = SPSAccumulator()
    loao_groups = {aoa: SPSAccumulator() for aoa in AOA_VALUES}
    point_digest_second = stream_digest()
    target_digest_second = stream_digest()
    with torch.inference_mode():
        for fold in fold_groups:
            indices = [i for i in all_indices if sample_folds[i] == fold]
            loader = uq_train.make_loader(
                dataset, indices, args.batch_size, False, loader_args, seed=0)
            model, _metadata = s1.load_fold_model(
                payload, args.oof_dir / f"fold_{fold:02d}_best.pth",
                fold, args.point_checkpoint, normalizer, device)
            fold_probability = torch.from_numpy(calibrated_probabilities(
                condition_parameters[fold], masses_np)).to(
                    device=device, dtype=torch.float32)
            offset = 0
            for history, target_full in loader:
                history = history.to(device, non_blocking=True)
                target_full = target_full.to(device, non_blocking=True)
                history_norm, _ = normalizer.preprocess(history, target_full)
                target = target_full[..., :2]
                (prediction_norm, lower, upper, valid, raw_width,
                 _masses, _bl, _bu) = candidate_tensors(
                     model, history_norm, normalizer)
                prediction = common.to_physical_target(
                    prediction_norm, normalizer)[..., :2]
                digest_tensor(point_digest_second, prediction)
                digest_tensor(target_digest_second, target)
                selected = calibrated_select(raw_width, valid, fold_probability)
                selected_lower = gather_candidate(lower, selected)
                selected_upper = gather_candidate(upper, selected)
                condition.update(
                    prediction, target, selected_lower, selected_upper)
                condition_folds[fold].update(
                    prediction, target, selected_lower, selected_upper)

                batch_aoas = aoas[indices[offset:offset + len(target)]]
                probabilities = torch.stack([
                    torch.from_numpy(calibrated_probabilities(
                        loao_parameters[int(aoa)], masses_np))
                    for aoa in batch_aoas]).to(device=device, dtype=torch.float32)
                selected = calibrated_select(raw_width, valid, probabilities)
                selected_lower = gather_candidate(lower, selected)
                selected_upper = gather_candidate(upper, selected)
                loao.update(prediction, target, selected_lower, selected_upper)
                for aoa in AOA_VALUES:
                    in_aoa = torch.from_numpy(batch_aoas == aoa).to(device)
                    if bool(in_aoa.any()):
                        loao_groups[aoa].update(
                            prediction[in_aoa], target[in_aoa],
                            selected_lower[in_aoa], selected_upper[in_aoa])
                offset += len(target)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    point_unchanged = (
        point_digest_first.hexdigest() == point_digest_second.hexdigest()
        and target_digest_first.hexdigest() == target_digest_second.hexdigest())
    condition_result = condition.result()
    condition_result["sps_delta"] = (
        condition_result["sps_score"] - baseline_result["sps_score"])
    loao_result = loao.result()
    loao_result["sps_delta"] = (
        loao_result["sps_score"] - baseline_result["sps_score"])
    condition_rows = [
        metric_row(condition_folds[fold], baseline_folds[fold], fold)
        for fold in fold_groups]
    loao_rows = [
        metric_row(loao_groups[aoa], baseline_aoas[aoa], aoa)
        for aoa in AOA_VALUES]
    elapsed = time.perf_counter() - started
    checks = gate(
        condition_result, condition_rows, loao_rows, True,
        point_unchanged, baseline_match, elapsed)
    report = {
        "experiment": EXPERIMENT,
        "evidence_class": (
            "fixed725 fold-excluded UQ; outer-fold and calibrator-level LOAO; "
            "not official evidence and not full-model true LOAO"),
        "contract": {
            "point_training": False,
            "uq_training": False,
            "trainable_values": 80,
            "formula": "sigmoid(b[h,c] + s[h,c]*logit(nominal_mass))",
            "bias_limit": BIAS_LIMIT,
            "slope_range": [math.exp(-LOG_SLOPE_LIMIT),
                            math.exp(LOG_SLOPE_LIMIT)],
            "identity_path": "direct nominal mass; no logit round-trip",
            "candidate_endpoints": "frozen; only pair selection changes",
        },
        "identities": identities,
        "point_stream_first_sha256": point_digest_first.hexdigest(),
        "point_stream_second_sha256": point_digest_second.hexdigest(),
        "target_stream_first_sha256": target_digest_first.hexdigest(),
        "target_stream_second_sha256": target_digest_second.hexdigest(),
        "baseline": baseline_result,
        "baseline_replay_gate": baseline_replay,
        "condition5": {
            "pooled": condition_result,
            "rows": condition_rows,
            "parameters": {
                str(key): serialize_parameters(value)
                for key, value in condition_parameters.items()},
        },
        "calibrator_loao": {
            "pooled": loao_result,
            "rows": loao_rows,
            "parameters": {
                str(key): serialize_parameters(value)
                for key, value in loao_parameters.items()},
            "limitation": (
                "The incremental calibrator excludes held AoA; frozen base "
                "fold models are not retrained with that AoA removed."),
        },
        "full_oof_parameters": serialize_parameters(full_parameters),
        "gate_checks": checks,
        "decision": (
            "PASS_UC1_TO_REFIT_ATTACHMENT"
            if all(checks.values()) else "STOP_UC1_COVERAGE_TEMPERATURE"),
        "elapsed_seconds": elapsed,
        "checkpoints_saved": [],
        "submission_created": False,
        "limitations": [
            "Fixed725 OOF diagnostic, not an official score.",
            "All-tail refit attachment is forbidden unless every gate passes.",
            "No retry, parameter sweep, or spatial calibration head is allowed.",
        ],
        "frozen_summary_selected_oof": summary["selected_oof"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "uc1_complete", "decision": report["decision"],
        "condition_sps_delta": condition_result["sps_delta"],
        "loao_sps_delta": loao_result["sps_delta"],
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
