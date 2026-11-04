#!/usr/bin/env python3
"""VF1: target-visible physical-zero oracle for the frozen C208 baseline.

This train-side upper bound changes only point-prediction UV scalars whose
matching future truth scalar is exactly physical zero.  Native C208 bounds are
computed once and reused unchanged.  The oracle is not deployable and cannot
produce a checkpoint or submission archive.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_c208_history_zero_projection as history_zero  # noqa: E402
import evaluate_c208_q50_mean_point_shift as q50  # noqa: E402
import evaluate_c208_streamwise_transport_oracle as transport  # noqa: E402
import train_c208_phase_residual_probe as phase  # noqa: E402
import train_c208_real_sim_rehearsal as c208  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402


EXPERIMENT = "c208_vf1_observation_factor_oracle"
FORMAL_ACKNOWLEDGEMENT = "VF1_REAL2576_TARGET_VISIBLE_ONLY"
FORMAL_C208_SHA256 = transport.FORMAL_C208_SHA256
FORMAL_SPLIT_MANIFEST_SHA256 = transport.FORMAL_SPLIT_MANIFEST_SHA256
FORMAL_REAL_STATS_SHA256 = transport.FORMAL_REAL_STATS_SHA256
FORMAL_COUNTS = dict(transport.FORMAL_COUNTS)
AOA_VALUES = transport.AOA_VALUES
FORMAL_EVAL_BATCH_SIZE = transport.FORMAL_EVAL_BATCH_SIZE
FORMAL_NUM_WORKERS = transport.FORMAL_NUM_WORKERS
POINT_DELTA_MIN = 0.15
WALL_BUDGET_SECONDS = 180.0
GPU_BUDGET_SECONDS = 180.0

EXPERIMENT_CONTRACT = {
    "hypothesis": (
        "A material part of the promoted C208 point deficit may come from "
        "predicting non-zero velocity on future physical-zero support; an "
        "impossible target-visible selector gives an upper bound before any "
        "observation-only factor model is trained."),
    "evidence_class": "target-visible upper-bound only; not deployable",
    "exact_baseline": {
        "checkpoint_sha256": FORMAL_C208_SHA256,
        "checkpoint_semantics": "promoted C208 Real iteration 4000",
        "split_manifest_canonical_sha256": FORMAL_SPLIT_MANIFEST_SHA256,
        "real_stats_sha256": FORMAL_REAL_STATS_SHA256,
        "inference": "frozen C208 Real u4000; CUDA BF16 autocast",
        "bounds": "native absolute UV bounds from the same frozen forward",
    },
    "data_boundary": {
        "train_windows": FORMAL_COUNTS["train_windows"],
        "tail_windows_manifest_count_only": FORMAL_COUNTS["tail_windows"],
        "tail_permission": (
            "no tail Window or Dataset construction, indexing, target read, "
            "or scoring"),
        "legacy_loader": (
            "raw H5 trajectories may be preloaded before train-window "
            "construction; no tail window is materialized"),
    },
    "single_intervention": {
        "selector": (
            "elementwise target[...,u:v] == physical 0.0 exactly; u and v "
            "are selected independently"),
        "point": "set only selected frozen-baseline UV scalars to 0.0",
        "bounds": "reuse identical baseline lower/upper arrays",
        "pressure": "reuse baseline pressure bitwise; it remains zero",
        "forbidden": [
            "threshold", "history-derived mask", "morphology", "soft blend",
            "parameter fit", "checkpoint", "ZIP", "fixed725", "tail target",
        ],
    },
    "gate": {
        "pooled_point_delta_min": POINT_DELTA_MIN,
        "pooled_rel_l2_delta": "strictly greater than zero",
        "pooled_mvpe_delta": "strictly greater than zero",
        "aoa20_non_time_delta_min": 0.0,
        "all_invariants": True,
    },
    "budget": {
        "wall_seconds": WALL_BUDGET_SECONDS,
        "gpu_seconds": GPU_BUDGET_SECONDS,
    },
    "decision": {
        "pass": "GO_VF1_OBSERVATION_FACTOR_HEADROOM",
        "fail": "STOP_VF1_NO_OBSERVATION_HEADROOM",
    },
}


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, started: float) -> None:
        self.started = started
        self.gpu_started: float | None = None

    def start_gpu(self) -> None:
        if self.gpu_started is not None:
            raise RuntimeError("VF1 GPU budget was started twice")
        self.gpu_started = time.perf_counter()

    def elapsed(self) -> dict[str, float]:
        now = time.perf_counter()
        return {
            "wall_seconds": now - self.started,
            "gpu_seconds": (
                0.0 if self.gpu_started is None else now - self.gpu_started),
        }

    def check(self, stage: str) -> None:
        elapsed = self.elapsed()
        if elapsed["wall_seconds"] > WALL_BUDGET_SECONDS:
            raise BudgetExpired(f"VF1 wall budget expired during {stage}")
        if elapsed["gpu_seconds"] > GPU_BUDGET_SECONDS:
            raise BudgetExpired(f"VF1 GPU budget expired during {stage}")


def truth_zero_factor_oracle(
        baseline: torch.Tensor, truth: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float | bool]]:
    """Apply the one preregistered target-visible transform without mutation."""
    if (baseline.ndim != 5 or baseline.shape[1] != transport.OUTPUT_STEPS
            or baseline.shape[-1] != 3 or truth.shape != baseline.shape):
        raise ValueError("baseline/truth must have shape (N,20,H,W,3)")
    if (baseline.dtype != torch.float32 or truth.dtype != torch.float32
            or baseline.device.type != "cpu" or truth.device.type != "cpu"):
        raise TypeError("VF1 requires CPU FP32 baseline and truth tensors")
    if not baseline.is_contiguous() or not truth.is_contiguous():
        raise TypeError("VF1 requires contiguous baseline and truth tensors")
    if not bool(torch.isfinite(baseline).all()) \
            or not bool(torch.isfinite(truth).all()):
        raise FloatingPointError("VF1 baseline or truth is non-finite")
    if bool(torch.count_nonzero(baseline[..., 2:])) \
            or bool(torch.count_nonzero(truth[..., 2:])):
        raise ValueError("VF1 requires exactly zero baseline/truth pressure")

    zero_mask = truth[..., :2] == 0.0
    candidate = baseline.clone()
    candidate[..., :2].masked_fill_(zero_mask, 0.0)
    outside = ~zero_mask
    base_outside = baseline[..., :2][outside].contiguous()
    candidate_outside = candidate[..., :2][outside].contiguous()
    pressure_equal = torch.equal(
        candidate[..., 2:].contiguous().view(torch.int32),
        baseline[..., 2:].contiguous().view(torch.int32))
    selected = int(torch.count_nonzero(zero_mask).item())
    report: dict[str, int | float | bool] = {
        "selected_future_uv_elements": selected,
        "total_future_uv_elements": int(zero_mask.numel()),
        "selected_fraction": float(selected / zero_mask.numel()),
        "selected_u_elements": int(torch.count_nonzero(zero_mask[..., 0])),
        "selected_v_elements": int(torch.count_nonzero(zero_mask[..., 1])),
        "selected_candidate_elements_nonzero": int(torch.count_nonzero(
            candidate[..., :2][zero_mask])),
        "selected_candidate_elements_are_zero": bool(
            torch.count_nonzero(candidate[..., :2][zero_mask]).item() == 0),
        "outside_mask_uv_bitwise_equal": bool(torch.equal(
            base_outside.view(torch.int32),
            candidate_outside.view(torch.int32))),
        "pressure_bitwise_equal": bool(pressure_equal),
        "candidate_finite": bool(torch.isfinite(candidate).all()),
        "target_visible_selector": True,
    }
    return candidate, zero_mask.contiguous(), report


def metric_comparison_fixed_bounds(
        baseline: np.ndarray, candidate: np.ndarray, target: np.ndarray,
        lower: np.ndarray, upper: np.ndarray) -> dict:
    """Score both points with the exact same absolute lower/upper arrays."""
    lower_before = q50.array_sha256(lower)
    upper_before = q50.array_sha256(upper)
    baseline_metrics = q50.score_bundle(baseline, target, lower, upper)
    candidate_metrics = q50.score_bundle(candidate, target, lower, upper)
    lower_after = q50.array_sha256(lower)
    upper_after = q50.array_sha256(upper)
    delta = q50.metric_delta(candidate_metrics, baseline_metrics)
    return {
        "samples": int(len(target)),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "non_time_shadow_delta": q50.non_time_shadow(delta),
        "bounds": {
            "lower_sha256_before": lower_before,
            "lower_sha256_after": lower_after,
            "upper_sha256_before": upper_before,
            "upper_sha256_after": upper_after,
            "bitwise_unchanged": (
                lower_before == lower_after and upper_before == upper_after),
        },
    }


def aoa_diagnostics_fixed_bounds(
        baseline: np.ndarray, candidate: np.ndarray, target: np.ndarray,
        lower: np.ndarray, upper: np.ndarray, aoas: np.ndarray) -> dict[str, dict]:
    if aoas.shape != (len(target),) or set(aoas.tolist()) != set(AOA_VALUES):
        raise ValueError("VF1 AoA labels differ from the frozen five groups")
    result = {}
    for aoa in AOA_VALUES:
        selected = aoas == aoa
        result[str(aoa)] = metric_comparison_fixed_bounds(
            baseline[selected], candidate[selected], target[selected],
            lower[selected], upper[selected])
    return result


def gate_result(
        pooled: dict, by_aoa: dict[str, dict], invariants: dict[str, bool],
        elapsed: dict[str, float]) -> dict[str, object]:
    if set(by_aoa) != {str(aoa) for aoa in AOA_VALUES}:
        raise ValueError("VF1 gate requires all five frozen AoA groups")
    if not invariants or any(type(value) is not bool for value in invariants.values()):
        raise TypeError("VF1 invariants must be a non-empty bool mapping")
    delta = pooled["delta"]
    checks = {
        "pooled_point_delta_ge_0_15": delta["point_score"] >= POINT_DELTA_MIN,
        "pooled_rel_l2_delta_strictly_positive": delta["rel_l2_score"] > 0.0,
        "pooled_mvpe_delta_strictly_positive": delta["mvpe_score"] > 0.0,
        "aoa20_non_time_delta_nonnegative": (
            by_aoa["20"]["non_time_shadow_delta"] >= 0.0),
        "all_invariants_pass": all(invariants.values()),
        "wall_within_180_seconds": (
            elapsed["wall_seconds"] <= WALL_BUDGET_SECONDS),
        "gpu_within_180_seconds": (
            elapsed["gpu_seconds"] <= GPU_BUDGET_SECONDS),
    }
    passed = all(checks.values())
    return {
        "pass": passed,
        "decision": (
            "GO_VF1_OBSERVATION_FACTOR_HEADROOM" if passed
            else "STOP_VF1_NO_OBSERVATION_HEADROOM"),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--eval-batch-size", type=int, default=FORMAL_EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=FORMAL_NUM_WORKERS)
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-standard-real2576-only")
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def validate_formal_args(args: argparse.Namespace) -> None:
    if not args.execute_formal_gpu:
        raise ValueError("formal VF1 requires --execute-formal-gpu")
    if args.acknowledge_standard_real2576_only != FORMAL_ACKNOWLEDGEMENT:
        raise ValueError("formal VF1 acknowledgement differs")
    if args.eval_batch_size != FORMAL_EVAL_BATCH_SIZE:
        raise ValueError("formal VF1 requires eval-batch-size=16")
    if args.num_workers != FORMAL_NUM_WORKERS:
        raise ValueError("formal VF1 requires num-workers=0")
    if torch.device(args.device).type != "cuda":
        raise ValueError("formal VF1 requires a CUDA device")


@torch.inference_mode()
def build_point_bound_cache(
        model, dataset, normalizer, device: torch.device, budget: Budget,
        eval_batch_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one frozen forward per train window and cache point/truth/bounds."""
    if len(dataset) != FORMAL_COUNTS["train_windows"]:
        raise ValueError("VF1 cache accepts exactly 2576 train windows")
    full_shape = (len(dataset), transport.OUTPUT_STEPS, 32, 64, 3)
    uv_shape = (*full_shape[:-1], 2)
    baseline = torch.empty(full_shape, dtype=torch.float32)
    truth_cache = torch.empty(full_shape, dtype=torch.float32)
    lower_cache = torch.empty(uv_shape, dtype=torch.float32)
    upper_cache = torch.empty(uv_shape, dtype=torch.float32)
    loader = DataLoader(
        dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=FORMAL_NUM_WORKERS, pin_memory=True)
    extent_scale = normalizer.std_target[:2].float().view(1, 1, 1, 1, 2)
    offset = 0
    model.eval()
    for history, target in loader:
        budget.check("2576 point/bound cache construction")
        history_device = history.to(device, non_blocking=True)
        target_device = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(
            history_device, target_device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = model.forward_uq_training(history_norm)
        prediction_norm = values[0]
        lower_extent_norm = values[3]
        upper_extent_norm = values[4]
        prediction = normalizer.postprocess_prediction(prediction_norm.float())
        truth = normalizer.postprocess_target(target_norm)
        prediction[..., 2:] = 0.0
        if bool(torch.count_nonzero(truth[..., 2:])):
            raise RuntimeError("VF1 Real truth pressure is not exactly zero")
        prediction_uv = prediction[..., :2]
        lower_extent = lower_extent_norm.float() * extent_scale
        upper_extent = upper_extent_norm.float() * extent_scale
        lower = prediction_uv - lower_extent
        upper = prediction_uv + upper_extent
        if not all(bool(torch.isfinite(value).all()) for value in (
                prediction, truth, lower, upper)):
            raise FloatingPointError("VF1 frozen forward produced non-finite data")
        if bool((lower > upper).any()):
            raise FloatingPointError("VF1 frozen bounds are reversed")
        batch = len(history)
        stop = offset + batch
        baseline[offset:stop].copy_(prediction.cpu())
        truth_cache[offset:stop].copy_(truth.cpu())
        lower_cache[offset:stop].copy_(lower.cpu())
        upper_cache[offset:stop].copy_(upper.cpu())
        offset = stop
    if offset != len(dataset):
        raise RuntimeError("VF1 cache did not consume exactly 2576 windows")
    return baseline, truth_cache, lower_cache, upper_cache


def _failure_result(
        decision: str, exc: Exception, contract_sha: str, identities: dict,
        counts: dict, raw_frames_preloaded: bool,
        tail: transport.TailAccessCounters, budget: Budget) -> dict:
    return {
        "experiment": EXPERIMENT,
        "decision": decision,
        "contract": EXPERIMENT_CONTRACT,
        "contract_canonical_sha256": contract_sha,
        "provenance": identities,
        "counts": counts,
        "raw_real_frames_preloaded_before_window_gate": raw_frames_preloaded,
        **tail.report(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "elapsed": budget.elapsed(),
        "checkpoint_written": False,
        "zip_written": False,
    }


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    budget = Budget(started)
    tail = transport.TailAccessCounters()
    identities: dict = {}
    counts: dict = {}
    raw_frames_preloaded = False
    contract_sha = transport.canonical_sha256(EXPERIMENT_CONTRACT)

    if args.contract_only:
        history_zero.write_result(result_path, {
            "experiment": EXPERIMENT,
            "decision": "CONTRACT_ONLY_NO_DATA_NO_GPU",
            "contract": EXPERIMENT_CONTRACT,
            "contract_canonical_sha256": contract_sha,
            **tail.report(),
            "checkpoint_written": False,
            "zip_written": False,
        })
        return

    try:
        validate_formal_args(args)
        identities = transport.validate_formal_inputs(args)
        budget.check("provenance validation")
        device = torch.device(args.device)
        if not torch.cuda.is_available():
            raise RuntimeError("formal VF1 requires CUDA")
        budget.start_gpu()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")

        trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
        raw_frames_preloaded = True
        train_windows, runtime_manifest, tail_count = phase.make_train_side_split(
            metadata)
        counts = {
            "train_windows": len(train_windows),
            "tail_windows_in_manifest_only": tail_count,
            "trajectories": len(trajectories),
        }
        expected_counts = {
            "train_windows": FORMAL_COUNTS["train_windows"],
            "tail_windows_in_manifest_only": FORMAL_COUNTS["tail_windows"],
            "trajectories": FORMAL_COUNTS["trajectories"],
        }
        if counts != expected_counts:
            raise RuntimeError(f"VF1 fixed split counts changed: {counts}")
        runtime_manifest_sha = transport.canonical_sha256(runtime_manifest)
        if runtime_manifest_sha != FORMAL_SPLIT_MANIFEST_SHA256:
            raise RuntimeError("VF1 runtime split canonical SHA changed")
        supplied_manifest = identities.pop("split_manifest")
        if runtime_manifest != supplied_manifest:
            raise RuntimeError("VF1 supplied split differs from runtime split")
        train_key_digest = phase.ordered_window_key_digest(
            train_windows, metadata)
        aoas = phase.window_aoas(train_windows, metadata)

        dataset = common.repro.WindowDataset(trajectories, train_windows)
        normalizer = common.repro.Normalizer(args.real_stats, device)
        payload = torch.load(
            args.point_checkpoint, map_location="cpu", weights_only=False)
        if (payload.get("phase") != "real"
                or int(payload.get("iteration", -1)) != 4000):
            raise ValueError("VF1 checkpoint is not C208 Real iteration 4000")
        config = payload.get("model_config", {})
        mismatch = {
            key: {"actual": config.get(key), "expected": value}
            for key, value in c208.EXPECTED_MODEL_CONFIG.items()
            if config.get(key) != value}
        if mismatch:
            raise ValueError(f"VF1 C208 model config changed: {mismatch}")
        model = c208.load_model(payload, device)
        common.set_model_stats(model, normalizer)
        model.requires_grad_(False)
        model.eval()
        baseline, truth, lower, upper = build_point_bound_cache(
            model, dataset, normalizer, device, budget,
            args.eval_batch_size)
        del model, payload, dataset, trajectories
        gc.collect()
        torch.cuda.empty_cache()

        input_hash_before = {
            "baseline": transport.tensor_sha256(baseline),
            "truth": transport.tensor_sha256(truth),
            "lower": transport.tensor_sha256(lower),
            "upper": transport.tensor_sha256(upper),
        }
        candidate, zero_mask, projection = truth_zero_factor_oracle(
            baseline, truth)
        target_zero_digest = transport.tensor_sha256(zero_mask)
        baseline_array = baseline.numpy()
        candidate_array = candidate.numpy()
        target_array = truth.numpy()
        lower_array = lower.numpy()
        upper_array = upper.numpy()
        pooled = metric_comparison_fixed_bounds(
            baseline_array, candidate_array, target_array,
            lower_array, upper_array)
        by_aoa = aoa_diagnostics_fixed_bounds(
            baseline_array, candidate_array, target_array,
            lower_array, upper_array, aoas)
        budget.check("official pooled and AoA scoring")
        input_hash_after = {
            "baseline": transport.tensor_sha256(baseline),
            "truth": transport.tensor_sha256(truth),
            "lower": transport.tensor_sha256(lower),
            "upper": transport.tensor_sha256(upper),
        }
        input_hashes_unchanged = input_hash_before == input_hash_after
        coverage_width_fixed = (
            pooled["baseline"]["coverage"] == pooled["candidate"]["coverage"]
            and pooled["baseline"]["mean_width"]
            == pooled["candidate"]["mean_width"])
        aoa_coverage_width_fixed = all(
            row["baseline"]["coverage"] == row["candidate"]["coverage"]
            and row["baseline"]["mean_width"]
            == row["candidate"]["mean_width"]
            for row in by_aoa.values())
        aoa_partition_exact = sum(
            row["samples"] for row in by_aoa.values()) == len(truth)
        invariants = {
            "checkpoint_hash": (
                identities["point_checkpoint_sha256"] == FORMAL_C208_SHA256),
            "real_stats_hash": (
                identities["real_stats_sha256"] == FORMAL_REAL_STATS_SHA256),
            "split_manifest_hash": (
                runtime_manifest_sha == FORMAL_SPLIT_MANIFEST_SHA256),
            "counts_exact": counts == expected_counts,
            "tail_access_zero": not tail.report()["tail_accessed"],
            "target_selector_exact_physical_zero": bool(torch.equal(
                zero_mask, truth[..., :2] == 0.0)),
            "selected_candidate_elements_are_zero": bool(
                projection["selected_candidate_elements_are_zero"]),
            "outside_mask_uv_bitwise_equal": bool(
                projection["outside_mask_uv_bitwise_equal"]),
            "pressure_bitwise_equal": bool(
                projection["pressure_bitwise_equal"]),
            "candidate_finite": bool(projection["candidate_finite"]),
            "input_hashes_unchanged": input_hashes_unchanged,
            "pooled_bounds_bitwise_unchanged": bool(
                pooled["bounds"]["bitwise_unchanged"]),
            "all_aoa_bounds_bitwise_unchanged": all(
                bool(row["bounds"]["bitwise_unchanged"])
                for row in by_aoa.values()),
            "coverage_and_width_unchanged": coverage_width_fixed,
            "all_aoa_coverage_and_width_unchanged": (
                aoa_coverage_width_fixed),
            "aoa_partition_exact": aoa_partition_exact,
        }
        torch.cuda.synchronize(device)
        budget.check("post-score invariant hashing")
        elapsed = budget.elapsed()
        gate = gate_result(pooled, by_aoa, invariants, elapsed)
        result = {
            "experiment": EXPERIMENT,
            "decision": gate["decision"],
            "contract": EXPERIMENT_CONTRACT,
            "contract_canonical_sha256": contract_sha,
            "provenance": identities,
            "run_manifest": {
                "split_manifest_canonical_sha256": runtime_manifest_sha,
                "ordered_train_window_key_digest": train_key_digest,
                "target_zero_mask_sha256": target_zero_digest,
                "counts": counts,
            },
            "counts": counts,
            "raw_real_frames_preloaded_before_window_gate": (
                raw_frames_preloaded),
            "tail_windows_constructed": False,
            "tail_targets_used": False,
            **tail.report(),
            "projection": projection,
            "input_hash_before": input_hash_before,
            "input_hash_after": input_hash_after,
            "official_exact_pooled": pooled,
            "official_exact_by_aoa": by_aoa,
            "invariants": invariants,
            "gate": gate,
            "elapsed": elapsed,
            "checkpoint_written": False,
            "zip_written": False,
        }
        history_zero.write_result(result_path, result)
        print(json.dumps({
            "event": "complete",
            "decision": result["decision"],
            "point_delta": pooled["delta"]["point_score"],
            "aoa20_non_time_delta": by_aoa["20"]["non_time_shadow_delta"],
        }), flush=True)
    except BudgetExpired as exc:
        result = _failure_result(
            "STOP_VF1_BUDGET", exc, contract_sha, identities, counts,
            raw_frames_preloaded, tail, budget)
        history_zero.write_result(result_path, result)
        print(json.dumps({
            "event": "complete", "decision": result["decision"]}),
            flush=True)
    except Exception as exc:
        result = _failure_result(
            "STOP_VF1_ERROR", exc, contract_sha, identities, counts,
            raw_frames_preloaded, tail, budget)
        history_zero.write_result(result_path, result)
        raise


if __name__ == "__main__":
    main()
