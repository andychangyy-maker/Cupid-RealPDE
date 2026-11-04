#!/usr/bin/env python3
"""Train-side no-slip audit for the frozen C208 point predictor.

The candidate is fixed before scoring: for each sample, a spatial pixel is
solid iff both raw history velocity channels are exactly zero in all 20
observed frames.  The candidate sets the two predicted velocity channels to
zero at that same pixel for all 20 future frames.  Pressure remains zero.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_c208_streamwise_transport_oracle as transport  # noqa: E402
import train_c208_phase_residual_probe as phase  # noqa: E402
import train_c208_real_sim_rehearsal as c208  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402


EXPERIMENT = "c208_history_zero_projection"
FORMAL_C208_SHA256 = transport.FORMAL_C208_SHA256
FORMAL_SPLIT_MANIFEST_SHA256 = transport.FORMAL_SPLIT_MANIFEST_SHA256
FORMAL_REAL_STATS_SHA256 = transport.FORMAL_REAL_STATS_SHA256
FORMAL_COUNTS = dict(transport.FORMAL_COUNTS)
AOA_VALUES = transport.AOA_VALUES
FORMAL_EVAL_BATCH_SIZE = transport.FORMAL_EVAL_BATCH_SIZE
FORMAL_NUM_WORKERS = transport.FORMAL_NUM_WORKERS
WALL_BUDGET_SECONDS = 6.0 * 60.0
GPU_BUDGET_SECONDS = 5.0 * 60.0
POINT_DELTA_MIN = 0.10
POINT_SHADOW_DELTA_MIN = 0.06
AOA_POSITIVE_MIN = 4
WORST_AOA_SHADOW_MIN = -0.01

EXPERIMENT_CONTRACT = {
    "hypothesis": (
        "Frozen C208 predicts non-zero velocity inside pixels that the current "
        "sample's complete raw 20-frame velocity history identifies as no-slip "
        "support; exact history-only zero projection may recover point score."),
    "evidence_class": "Real 2576 train-side diagnostic; target used only for scoring",
    "exact_baseline": transport.EXPERIMENT_CONTRACT["exact_baseline"],
    "data_boundary": {
        "train_windows": FORMAL_COUNTS["train_windows"],
        "tail_windows": FORMAL_COUNTS["tail_windows"],
        "tail_permission": "zero construction, indexing, target reads, and scoring",
        "cache": "reuse the frozen C208 2576 prediction/history/truth cache path",
        "ordered_window_mask_digest": (
            transport.FORMAL_ORDERED_WINDOW_MASK_DIGEST),
    },
    "candidate": {
        "mask": (
            "per sample, history[...,u:v] == 0 exactly for both channels in "
            "every one of the 20 observed frames"),
        "transform": (
            "set future prediction u/v to exactly zero at the same spatial "
            "mask for all 20 forecast frames"),
        "pressure": "remains exactly zero",
        "target_used_for_mask_or_transform": False,
        "forbidden": [
            "threshold", "dilation", "erosion", "morphology", "soft blend",
            "trajectory aggregation", "another sample or window", "AoA/Re",
            "target", "parameter fitting", "checkpoint", "ZIP", "tail access",
        ],
    },
    "gate": {
        "pooled_point_delta_min": POINT_DELTA_MIN,
        "pooled_point_shadow_delta_min": POINT_SHADOW_DELTA_MIN,
        "pooled_rel_l2_delta_min": 0.0,
        "pooled_tke_delta_min": 0.0,
        "pooled_mvpe_delta_min": 0.0,
        "positive_aoa_shadow_min_count": AOA_POSITIVE_MIN,
        "aoa_count": len(AOA_VALUES),
        "worst_aoa_shadow_delta_min": WORST_AOA_SHADOW_MIN,
    },
    "budget": {
        "wall_seconds": WALL_BUDGET_SECONDS,
        "gpu_seconds": GPU_BUDGET_SECONDS,
    },
    "decision": {
        "pass": "GO_HISTORY_ZERO_PROJECTION_HEADROOM",
        "fail": "STOP_HISTORY_ZERO_PROJECTION_NO_HEADROOM",
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
            raise RuntimeError("GPU budget was started twice")
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
            raise BudgetExpired(f"wall budget expired during {stage}")
        if elapsed["gpu_seconds"] > GPU_BUDGET_SECONDS:
            raise BudgetExpired(f"GPU budget expired during {stage}")


def history_zero_projection(
        baseline: torch.Tensor, history: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float | bool]]:
    """Apply the one preregistered transform without mutating either input."""
    if (baseline.ndim != 5 or baseline.shape[1] != transport.OUTPUT_STEPS
            or baseline.shape[-1] != 3):
        raise ValueError("baseline must have shape (N,20,H,W,3)")
    if history.shape != baseline.shape:
        raise ValueError("history and baseline shapes must match")
    if (baseline.dtype != torch.float32 or history.dtype != torch.float32
            or baseline.device.type != "cpu" or history.device.type != "cpu"):
        raise TypeError("projection requires contiguous CPU FP32 caches")
    if not baseline.is_contiguous() or not history.is_contiguous():
        raise TypeError("projection requires contiguous CPU FP32 caches")
    if bool(torch.count_nonzero(baseline[..., 2:])):
        raise ValueError("baseline pressure must already be exactly zero")

    solid_mask = transport.observation_solid_mask(history)
    candidate = baseline.clone()
    candidate[..., :2].masked_fill_(solid_mask[:, None, :, :, None], 0.0)
    nonzero_before = 0
    nonzero_after = 0
    outside_bitwise_equal = True
    candidate_finite = True
    pressure_zero = True
    for start in range(0, len(baseline), FORMAL_EVAL_BATCH_SIZE):
        stop = min(start + FORMAL_EVAL_BATCH_SIZE, len(baseline))
        batch_mask = solid_mask[start:stop]
        base_uv = baseline[start:stop, ..., :2].permute(0, 2, 3, 1, 4)
        candidate_uv = candidate[start:stop, ..., :2].permute(0, 2, 3, 1, 4)
        masked_before = base_uv[batch_mask]
        masked_after = candidate_uv[batch_mask]
        outside = ~batch_mask
        base_outside = base_uv[outside].contiguous()
        candidate_outside = candidate_uv[outside].contiguous()
        outside_bitwise_equal = bool(
            outside_bitwise_equal and torch.equal(
                base_outside.view(torch.int32),
                candidate_outside.view(torch.int32)))
        nonzero_before += int(torch.count_nonzero(masked_before).item())
        nonzero_after += int(torch.count_nonzero(masked_after).item())
        candidate_finite = bool(
            candidate_finite
            and torch.isfinite(candidate[start:stop]).all())
        pressure_zero = bool(
            pressure_zero
            and torch.count_nonzero(candidate[start:stop, ..., 2:]).item() == 0)
    projected_pixels = int(torch.count_nonzero(solid_mask).item())
    projected_elements = projected_pixels * transport.OUTPUT_STEPS * 2
    report: dict[str, int | float | bool] = {
        "masked_sample_pixels": projected_pixels,
        "total_sample_pixels": int(solid_mask.numel()),
        "masked_sample_pixel_fraction": float(
            projected_pixels / solid_mask.numel()),
        "future_uv_elements_projected": projected_elements,
        "nonzero_future_uv_elements_before": nonzero_before,
        "nonzero_future_uv_elements_after": nonzero_after,
        "future_uv_zero_on_mask": nonzero_after == 0,
        "outside_mask_uv_bitwise_equal": outside_bitwise_equal,
        "candidate_finite": candidate_finite,
        "pressure_zero": pressure_zero,
    }
    return candidate, solid_mask, report


def gate_result(pooled: dict, by_aoa: dict,
                elapsed: dict[str, float]) -> dict[str, object]:
    if set(by_aoa) != {str(aoa) for aoa in AOA_VALUES}:
        raise ValueError("AoA diagnostics differ from the frozen five groups")
    delta = pooled["delta"]
    aoa_shadows = {
        key: float(row["point_shadow_delta"]) for key, row in by_aoa.items()}
    positive_count = sum(value > 0.0 for value in aoa_shadows.values())
    worst = min(aoa_shadows.values())
    checks = {
        "pooled_point_delta_ge_0_10": delta["point_score"] >= POINT_DELTA_MIN,
        "pooled_point_shadow_delta_ge_0_06": (
            pooled["point_shadow_delta"] >= POINT_SHADOW_DELTA_MIN),
        "pooled_rel_l2_delta_nonnegative": delta["rel_l2_score"] >= 0.0,
        "pooled_tke_delta_nonnegative": delta["tke_rel_l2_score"] >= 0.0,
        "pooled_mvpe_delta_nonnegative": delta["mvpe_rel_l2_score"] >= 0.0,
        "positive_aoa_shadow_ge_4_of_5": positive_count >= AOA_POSITIVE_MIN,
        "worst_aoa_shadow_ge_minus_0_01": worst >= WORST_AOA_SHADOW_MIN,
        "gpu_within_5_minutes": elapsed["gpu_seconds"] <= GPU_BUDGET_SECONDS,
        "wall_within_6_minutes": elapsed["wall_seconds"] <= WALL_BUDGET_SECONDS,
    }
    passed = all(checks.values())
    return {
        "pass": passed,
        "decision": (
            "GO_HISTORY_ZERO_PROJECTION_HEADROOM" if passed
            else "STOP_HISTORY_ZERO_PROJECTION_NO_HEADROOM"),
        "checks": checks,
        "aoa_point_shadow_delta": aoa_shadows,
        "positive_aoa_shadow_count": positive_count,
        "worst_aoa_point_shadow_delta": worst,
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
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def write_result(path: Path, result: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def _failure_result(decision: str, exc: Exception, contract_sha: str,
                    identities: dict, counts: dict, raw_frames_preloaded: bool,
                    tail: transport.TailAccessCounters,
                    budget: Budget) -> dict:
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
        write_result(result_path, {
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
        identities = transport.validate_formal_inputs(args)
        budget.check("provenance validation")
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal history-zero audit requires CUDA")
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
            raise RuntimeError(f"fixed split counts changed: {counts}")
        runtime_manifest_sha = transport.canonical_sha256(runtime_manifest)
        if runtime_manifest_sha != FORMAL_SPLIT_MANIFEST_SHA256:
            raise RuntimeError("runtime split canonical SHA changed")
        supplied_manifest = identities.pop("split_manifest")
        if runtime_manifest != supplied_manifest:
            raise RuntimeError("supplied split differs from runtime split")
        train_key_digest = phase.ordered_window_key_digest(train_windows, metadata)
        run_manifest = {
            "experiment": EXPERIMENT,
            "contract_canonical_sha256": contract_sha,
            "split_manifest_canonical_sha256": runtime_manifest_sha,
            "ordered_train_window_key_digest": train_key_digest,
            "counts": counts,
            "checkpoint_sha256": identities["point_checkpoint_sha256"],
            "real_stats_sha256": identities["real_stats_sha256"],
        }

        train_dataset = common.repro.WindowDataset(trajectories, train_windows)
        aoas = phase.window_aoas(train_windows, metadata)
        normalizer = common.repro.Normalizer(args.real_stats, device)
        point_payload = torch.load(
            args.point_checkpoint, map_location="cpu", weights_only=False)
        if (point_payload.get("phase") != "real"
                or int(point_payload.get("iteration", -1)) != 4000):
            raise ValueError("checkpoint is not frozen C208 real iteration 4000")
        config = point_payload.get("model_config", {})
        mismatch = {
            key: {"actual": config.get(key), "expected": value}
            for key, value in c208.EXPECTED_MODEL_CONFIG.items()
            if config.get(key) != value}
        if mismatch:
            raise ValueError(f"C208 model config changed: {mismatch}")
        point_model = c208.load_model(point_payload, device)
        common.set_model_stats(point_model, normalizer)
        point_model.requires_grad_(False)
        point_model.eval()

        baseline, history, target = transport.build_full_cache(
            point_model, train_dataset, normalizer, device, budget,
            args.eval_batch_size)
        cache_hash_before = {
            "baseline": transport.tensor_sha256(baseline),
            "history": transport.tensor_sha256(history),
            "target": transport.tensor_sha256(target),
        }
        candidate, solid_mask, projection = history_zero_projection(
            baseline, history)
        mask_diagnostics = transport.per_sample_mask_diagnostics(
            solid_mask, train_windows)
        mask_contract_matches = all(
            mask_diagnostics[key] == expected
            for key, expected in transport.FORMAL_MASK_DIAGNOSTIC.items())
        if not mask_contract_matches:
            raise RuntimeError(
                "per-sample observation mask fingerprint changed: "
                f"{mask_diagnostics}")
        window_mask_digest = transport.ordered_window_mask_digest(
            train_key_digest, solid_mask)
        if window_mask_digest != transport.FORMAL_ORDERED_WINDOW_MASK_DIGEST:
            raise RuntimeError(
                f"ordered window/mask digest changed: {window_mask_digest}")
        run_manifest["ordered_window_mask_digest"] = window_mask_digest
        mask_hash = transport.tensor_sha256(solid_mask.contiguous())
        cache_hash_before["solid_mask"] = mask_hash

        del point_model, point_payload, train_dataset, trajectories
        gc.collect()
        torch.cuda.empty_cache()

        baseline_array = baseline.numpy()
        candidate_array = candidate.numpy()
        target_array = target.numpy()
        pooled = transport.metric_comparison(
            baseline_array, candidate_array, target_array)
        by_aoa = transport.aoa_diagnostics(
            baseline_array, candidate_array, target_array, aoas)
        torch.cuda.synchronize(device)
        budget.check("official pooled and AoA scoring")
        cache_hash_after = {
            "baseline": transport.tensor_sha256(baseline),
            "history": transport.tensor_sha256(history),
            "target": transport.tensor_sha256(target),
            "solid_mask": transport.tensor_sha256(solid_mask.contiguous()),
        }
        input_unchanged = cache_hash_before == cache_hash_after
        elapsed = budget.elapsed()
        guards = {
            "checkpoint_hash": (
                identities["point_checkpoint_sha256"] == FORMAL_C208_SHA256),
            "real_stats_hash": (
                identities["real_stats_sha256"] == FORMAL_REAL_STATS_SHA256),
            "split_manifest_hash": (
                runtime_manifest_sha == FORMAL_SPLIT_MANIFEST_SHA256),
            "tail_access_zero": not tail.report()["tail_accessed"],
            "per_sample_mask_diagnostic_fingerprint": mask_contract_matches,
            "ordered_window_mask_digest": (
                window_mask_digest == transport.FORMAL_ORDERED_WINDOW_MASK_DIGEST),
            "future_uv_zero_on_mask": bool(projection["future_uv_zero_on_mask"]),
            "outside_mask_uv_bitwise_equal": bool(
                projection["outside_mask_uv_bitwise_equal"]),
            "candidate_finite": bool(projection["candidate_finite"]),
            "pressure_zero": bool(projection["pressure_zero"]),
            "input_hashes_unchanged": input_unchanged,
        }
        if not all(guards.values()):
            raise RuntimeError(f"history-zero hard guard failed: {guards}")
        gate = gate_result(pooled, by_aoa, elapsed)
        result = {
            "experiment": EXPERIMENT,
            "decision": gate["decision"],
            "contract": EXPERIMENT_CONTRACT,
            "contract_canonical_sha256": contract_sha,
            "provenance": identities,
            "run_manifest": run_manifest,
            "run_manifest_canonical_sha256": transport.canonical_sha256(
                run_manifest),
            "counts": counts,
            "raw_real_frames_preloaded_before_window_gate": raw_frames_preloaded,
            "tail_windows_constructed": False,
            "tail_targets_used": False,
            **tail.report(),
            "cache": {
                "cached_once": True,
                "shape": list(baseline.shape),
                "dtype": str(baseline.dtype).removeprefix("torch."),
                "hash_before": cache_hash_before,
                "hash_after": cache_hash_after,
                "input_hashes_unchanged": input_unchanged,
                "solid_mask_semantics": (
                    "per_sample_observation_zero_support"),
                "solid_mask_diagnostics": mask_diagnostics,
                "ordered_window_mask_digest": window_mask_digest,
            },
            "projection": projection,
            "official_exact_pooled": pooled,
            "official_exact_by_aoa": by_aoa,
            "guards": guards,
            "gate": gate,
            "elapsed": elapsed,
            "checkpoint_written": False,
            "zip_written": False,
        }
        write_result(result_path, result)
        print(json.dumps({
            "event": "complete", "decision": result["decision"],
            "point_delta": pooled["delta"]["point_score"],
            "point_shadow_delta": pooled["point_shadow_delta"],
        }), flush=True)
    except BudgetExpired as exc:
        result = _failure_result(
            "STOP_HISTORY_ZERO_PROJECTION_BUDGET", exc, contract_sha,
            identities, counts, raw_frames_preloaded, tail, budget)
        write_result(result_path, result)
        print(json.dumps({
            "event": "complete", "decision": result["decision"]}), flush=True)
    except Exception as exc:
        result = _failure_result(
            "STOP_HISTORY_ZERO_PROJECTION_ERROR", exc, contract_sha,
            identities, counts, raw_frames_preloaded, tail, budget)
        write_result(result_path, result)
        raise


if __name__ == "__main__":
    main()
