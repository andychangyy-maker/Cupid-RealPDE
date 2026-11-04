#!/usr/bin/env python3
"""Evaluate a fixed dual-seed temporal-mean teacher that preserves TKE."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for source in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import evaluate_b210_fixed_function_ensemble as function_eval  # noqa: E402
import evaluate_b210_scratch_point_pair_by_aoa as point_eval  # noqa: E402
import train_b210_np_functional_real_ft12_loao as ft12  # noqa: E402
import train_b210_scratch_real_dense_aug as scratch  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402


SCHEMA = "b210_tke_invariant_mean_teacher_v1"
MEAN_WEIGHT_SEED0 = 0.50
FLUCTUATION_TOLERANCE = 2e-6
TKE_SCORE_TOLERANCE = 1e-5
GATE = {
    "rel_l2_score": 0.02,
    "mvpe_rel_l2_score": -0.005,
    "point_score": 0.015,
    "nonnegative_aoas": 4,
    "worst_aoa_point": -0.03,
}


def temporal_mean_teacher(seed0: torch.Tensor,
                          seed1: torch.Tensor) -> torch.Tensor:
    if seed0.shape != seed1.shape or seed0.dtype != seed1.dtype:
        raise ValueError("teacher endpoints differ in shape or dtype")
    if seed0.ndim != 5 or seed0.shape[-1] < 2:
        raise ValueError("teacher endpoints must be aligned 5D fields")
    first = seed0.float()
    base = seed1.float()
    return base + MEAN_WEIGHT_SEED0 * (
        first.mean(dim=1, keepdim=True) - base.mean(dim=1, keepdim=True))


def zero_mean_fluctuation(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean(dim=1, keepdim=True)


def teacher_gate(pooled_delta: dict[str, float],
                 by_aoa_delta: dict[str, dict[str, float]],
                 fluctuation_max_abs_delta: float) -> dict:
    point_deltas = [float(row["point_score"]) for row in by_aoa_delta.values()]
    checks = {
        "finite": all(math.isfinite(float(value)) for value in (
            *pooled_delta.values(), *point_deltas,
            fluctuation_max_abs_delta)),
        "fluctuation_invariant": (
            fluctuation_max_abs_delta <= FLUCTUATION_TOLERANCE),
        "tke_invariant": (
            abs(float(pooled_delta["tke_rel_l2_score"]))
            <= TKE_SCORE_TOLERANCE),
        "rel_signal": (
            float(pooled_delta["rel_l2_score"]) >= GATE["rel_l2_score"]),
        "mvpe_safe": (
            float(pooled_delta["mvpe_rel_l2_score"])
            >= GATE["mvpe_rel_l2_score"]),
        "point_signal": (
            float(pooled_delta["point_score"]) >= GATE["point_score"]),
        "aoa_count": (
            sum(value >= 0.0 for value in point_deltas)
            >= GATE["nonnegative_aoas"]),
        "aoa_floor": min(point_deltas) >= GATE["worst_aoa_point"],
    }
    return {
        "pass": all(checks.values()),
        "decision": (
            "PASS_B210_MEAN_TEACHER_FOR_SINGLE_MODEL_DISTILL"
            if all(checks.values()) else "STOP_B210_MEAN_TEACHER_GATE"),
        "checks": checks,
        "thresholds": {
            **GATE,
            "fluctuation_max_abs_delta": FLUCTUATION_TOLERANCE,
            "tke_score_abs_delta": TKE_SCORE_TOLERANCE,
        },
        "nonnegative_aoas": sum(value >= 0.0 for value in point_deltas),
        "worst_aoa_point_delta": min(point_deltas),
    }


@torch.inference_mode()
def evaluate_teacher(models, dataset, indices, normalizer,
                     device: torch.device, deadline: float):
    loader = DataLoader(
        dataset, batch_sampler=[
            list(indices[start:start + smu1.EVAL_BATCH_SIZE])
            for start in range(0, len(indices), smu1.EVAL_BATCH_SIZE)],
        num_workers=0, pin_memory=True)
    state = smu1.new_metric_state()
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    fluctuation_max = 0.0
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for model in models:
        model.eval()
    for history, target in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("mean-teacher evaluation exceeded wall budget")
        target_cpu = target.contiguous()
        scored_cpu = target_cpu[..., :2].ne(0.0).to(torch.uint8).contiguous()
        target_hasher.update(str(tuple(target_cpu.shape)).encode("ascii"))
        target_hasher.update(target_cpu.numpy().tobytes(order="C"))
        mask_hasher.update(str(tuple(scored_cpu.shape)).encode("ascii"))
        mask_hasher.update(scored_cpu.numpy().tobytes(order="C"))
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, _target_norm = normalizer.preprocess(history, target)
        seed0_norm = models[0](history_norm)
        seed1_norm, lower_norm, upper_norm = models[1].forward_distribution(
            history_norm)
        teacher_norm = temporal_mean_teacher(seed0_norm, seed1_norm)
        fluctuation_max = max(fluctuation_max, float((
            zero_mean_fluctuation(teacher_norm[..., :2])
            - zero_mean_fluctuation(seed1_norm.float()[..., :2])
        ).abs().max()))
        prediction = normalizer.postprocess_prediction(teacher_norm)
        smu1.update_metric_state(
            state, prediction.float(), target.float(),
            lower_norm.float() * scale, upper_norm.float() * scale)
    if int(state["samples"]) != len(indices):
        raise RuntimeError("mean teacher did not consume complete partition")
    return smu1.finalize_metric_state(state), state, {
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
        "samples": int(state["samples"]),
        "scored_elements": int(state["sps_scored"]),
        "normalized_fluctuation_max_abs_delta": fluctuation_max,
    }


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    source0 = function_eval.load_pair(args.pair_result_seed0)
    source1 = function_eval.load_pair(args.pair_result_seed1)
    evaluation0 = source0["evaluation_manifest"]
    evaluation1 = source1["evaluation_manifest"]
    if (evaluation0["sha256"] != evaluation1["sha256"]
            or int(evaluation0["tail_windows"])
            != int(evaluation1["tail_windows"])):
        raise RuntimeError("seed endpoints use different clean-tail evaluations")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean prefix manifest byte identity changed")
    rows, _manifest = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_path = Path(str(evaluation0["path"]))
    if dr1.sha256(evaluation_path) != evaluation0["sha256"]:
        raise RuntimeError("fixed evaluation manifest changed")
    evaluation_rows = scratch.load_evaluation_rows(evaluation_path, rows)
    seals = {
        "control": function_eval.candidate_seal(source0),
        "candidate": function_eval.candidate_seal(source1),
    }
    audit = scratch.TailAudit(seals)
    trajectories, metadata, windows = point_eval.load_tail_rows(
        args.real_dir, evaluation_rows, audit)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    indices_by_aoa = scratch.group_window_indices_by_aoa(windows, metadata)

    stats0 = torch.load(
        source0["normalizer"]["path"], map_location="cpu", weights_only=False)
    stats1 = torch.load(
        source1["normalizer"]["path"], map_location="cpu", weights_only=False)
    digest0 = smu1.statistics_digest(stats0)
    digest1 = smu1.statistics_digest(stats1)
    if (digest0 != source0["normalizer"]["statistics_sha256"]
            or digest1 != source1["normalizer"]["statistics_sha256"]
            or digest0 != digest1):
        raise RuntimeError("seed endpoint normalizers differ or changed")
    device = torch.device(args.device)
    normalizer = smu1.FoldNormalizer(stats0, device)
    models = [
        point_eval.load_primary_endpoint_mmap(seal, normalizer, device)
        for seal in seals.values()
    ]
    deadline = time.perf_counter() + args.wall_budget_seconds
    by_aoa = {}
    states = []
    fluctuation_max = 0.0
    for aoa, indices in indices_by_aoa.items():
        metrics, state, target_audit = evaluate_teacher(
            models, dataset, indices, normalizer, device, deadline)
        fluctuation_max = max(
            fluctuation_max,
            float(target_audit["normalized_fluctuation_max_abs_delta"]))
        by_aoa[str(aoa)] = {
            "windows": len(indices), "metrics": metrics,
            "target_audit": target_audit,
        }
        states.append(state)
    pooled = smu1.finalize_metric_state(smu1.merge_metric_states(states))
    seed1_pooled = source1["arms"]["candidate"]["metrics"]
    pooled_comparison = scratch.compare(pooled, seed1_pooled)
    by_aoa_delta = {
        str(aoa): scratch.compare(
            by_aoa[str(aoa)]["metrics"],
            source1["arms"]["candidate"]["metrics_by_aoa"][str(aoa)]
        )["delta"] for aoa in dr1.AOA_VALUES
    }
    gate = teacher_gate(
        pooled_comparison["delta"], by_aoa_delta, fluctuation_max)
    result = {
        "schema": SCHEMA,
        "mean_weight_seed0": MEAN_WEIGHT_SEED0,
        "composition": (
            "seed1 + 0.5 * (mean_t(seed0) - mean_t(seed1))"),
        "pair_results": {
            "seed0": {"path": str(args.pair_result_seed0.resolve()),
                      "sha256": dr1.sha256(args.pair_result_seed0)},
            "seed1": {"path": str(args.pair_result_seed1.resolve()),
                      "sha256": dr1.sha256(args.pair_result_seed1)},
        },
        "seed1_pooled": seed1_pooled,
        "teacher": {"pooled": pooled, "by_aoa": by_aoa},
        "delta_from_seed1": pooled_comparison,
        "delta_by_aoa": by_aoa_delta,
        "normalized_fluctuation_max_abs_delta": fluctuation_max,
        "gate": gate,
        "selection": "mean weight fixed before this clean716 evaluation",
        "submission_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    del models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-result-seed0", type=Path, required=True)
    parser.add_argument("--pair-result-seed1", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3600)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "event": "b210_tke_invariant_mean_teacher_complete",
        "teacher": result["teacher"]["pooled"],
        "delta": result["delta_from_seed1"]["delta"],
        "gate": result["gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
