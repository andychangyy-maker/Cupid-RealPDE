#!/usr/bin/env python3
"""Evaluate a fixed Dense4k-mean / Dense12k-fluctuation role teacher."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for source in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import evaluate_b210_scratch_point_pair_by_aoa as point_eval  # noqa: E402
import train_b210_np_functional_real_ft12_loao as ft12  # noqa: E402
import train_b210_scratch_real_dense_aug as scratch  # noqa: E402
import train_b210_realclosure_dense12k as dense12k  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d  # noqa: E402


SCHEMA = "b210_dense4k_mean_dense12k_fluctuation_v1"
EXPECTED_REFIT_SHA256 = (
    "12829b233d6cc358f496a91c4c78e6e05cf96d3a1d97f476656fbae5d4b2e67b")
FLUCTUATION_TOLERANCE = 2e-6
TKE_SCORE_TOLERANCE = 1e-5
GATE = {
    "rel_l2_score": 0.02,
    "mvpe_rel_l2_score": -0.03,
    "point_score": 0.01,
    "native_sps_score": -0.03,
    "nonnegative_aoas": 4,
    "worst_aoa_point": -0.03,
}


def role_teacher(dense4k: torch.Tensor,
                 dense12k: torch.Tensor) -> torch.Tensor:
    if dense4k.shape != dense12k.shape or dense4k.dtype != dense12k.dtype:
        raise ValueError("role endpoints differ in shape or dtype")
    if dense4k.ndim != 5 or dense4k.shape[-1] < 2:
        raise ValueError("role endpoints must be aligned 5D fields")
    early = dense4k.float()
    late = dense12k.float()
    return late + early.mean(dim=1, keepdim=True) - late.mean(
        dim=1, keepdim=True)


def zero_mean_fluctuation(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean(dim=1, keepdim=True)


def role_gate(delta: Mapping[str, float],
              by_aoa_delta: Mapping[str, Mapping[str, float]],
              fluctuation_max: float) -> dict:
    point_deltas = [float(row["point_score"])
                    for row in by_aoa_delta.values()]
    checks = {
        "finite": all(math.isfinite(float(value)) for value in (
            *delta.values(), *point_deltas, fluctuation_max)),
        "fluctuation_invariant": fluctuation_max <= FLUCTUATION_TOLERANCE,
        "tke_invariant": (
            abs(float(delta["tke_rel_l2_score"]))
            <= TKE_SCORE_TOLERANCE),
        "rel_signal": float(delta["rel_l2_score"]) >= GATE["rel_l2_score"],
        "mvpe_safe": (
            float(delta["mvpe_rel_l2_score"])
            >= GATE["mvpe_rel_l2_score"]),
        "point_signal": float(delta["point_score"]) >= GATE["point_score"],
        "sps_safe": (
            float(delta["native_sps_score"])
            >= GATE["native_sps_score"]),
        "aoa_count": (
            sum(value >= 0.0 for value in point_deltas)
            >= GATE["nonnegative_aoas"]),
        "aoa_floor": min(point_deltas) >= GATE["worst_aoa_point"],
    }
    passed = all(checks.values())
    return {
        "pass": passed,
        "decision": (
            "PASS_B210_ROLE_TEACHER_FOR_SCRATCH_RETRAIN"
            if passed else "STOP_B210_ROLE_TEACHER_GATE"),
        "checks": checks,
        "thresholds": {
            **GATE,
            "fluctuation_max_abs_delta": FLUCTUATION_TOLERANCE,
            "tke_score_abs_delta": TKE_SCORE_TOLERANCE,
        },
        "nonnegative_aoas": sum(value >= 0.0 for value in point_deltas),
        "worst_aoa_point_delta": min(point_deltas),
    }


def load_checkpoint(path: Path, expected_sha: str, normalizer,
                    device: torch.device) -> EnsembleUNet2d:
    if dr1.sha256(path) != expected_sha:
        raise RuntimeError(f"checkpoint SHA changed: {path}")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint has no model state")
    config = ft12.validate_b210_model_config(payload["model_config"])
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device).eval()


@torch.inference_mode()
def evaluate_partition(models, dataset, indices, normalizer,
                       device: torch.device, deadline: float):
    loader = DataLoader(
        dataset, batch_sampler=[
            list(indices[start:start + smu1.EVAL_BATCH_SIZE])
            for start in range(0, len(indices), smu1.EVAL_BATCH_SIZE)],
        num_workers=0, pin_memory=True)
    states = {"control": smu1.new_metric_state(),
              "teacher": smu1.new_metric_state()}
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    fluctuation_max = 0.0
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for history, target in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("role-teacher evaluation exceeded wall budget")
        target_cpu = target.contiguous()
        scored_cpu = target_cpu[..., :2].ne(0.0).to(torch.uint8).contiguous()
        target_hasher.update(str(tuple(target_cpu.shape)).encode("ascii"))
        target_hasher.update(target_cpu.numpy().tobytes(order="C"))
        mask_hasher.update(str(tuple(scored_cpu.shape)).encode("ascii"))
        mask_hasher.update(scored_cpu.numpy().tobytes(order="C"))
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, _target_norm = normalizer.preprocess(history, target)
        early_norm = models[0](history_norm)
        late_norm, lower_norm, upper_norm = models[1].forward_distribution(
            history_norm)
        teacher_norm = role_teacher(early_norm, late_norm)
        fluctuation_max = max(fluctuation_max, float((
            zero_mean_fluctuation(teacher_norm[..., :2])
            - zero_mean_fluctuation(late_norm.float()[..., :2])
        ).abs().max()))
        extents = (lower_norm.float() * scale, upper_norm.float() * scale)
        smu1.update_metric_state(
            states["control"],
            normalizer.postprocess_prediction(late_norm.float()),
            target.float(), *extents)
        smu1.update_metric_state(
            states["teacher"],
            normalizer.postprocess_prediction(teacher_norm),
            target.float(), *extents)
    if any(int(state["samples"]) != len(indices) for state in states.values()):
        raise RuntimeError("role teacher did not consume complete partition")
    audit = {
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
        "samples": len(indices),
        "scored_elements": int(states["teacher"]["sps_scored"]),
        "normalized_fluctuation_max_abs_delta": fluctuation_max,
    }
    return ({key: smu1.finalize_metric_state(value)
             for key, value in states.items()}, states, audit)


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    early = json.loads(args.dense4k_result.read_text(encoding="utf-8"))
    late = json.loads(args.dense12k_result.read_text(encoding="utf-8"))
    if early.get("schema") != scratch.SCHEMA:
        raise RuntimeError("Dense4k result schema changed")
    if late.get("schema") != dense12k.SCHEMA:
        raise RuntimeError("Dense12k result schema changed")
    if (early["start"]["checkpoint_sha256"]
            != late["start"]["checkpoint_sha256"]):
        raise RuntimeError("role endpoints do not share the Sim start")
    evaluation = late["evaluation_manifest"]
    if (early["evaluation_manifest"]["sha256"] != evaluation["sha256"]
            or dr1.sha256(Path(evaluation["path"])) != evaluation["sha256"]):
        raise RuntimeError("role endpoint evaluation manifests differ")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean prefix manifest changed")
    rows, _manifest = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_rows = scratch.load_evaluation_rows(Path(evaluation["path"]), rows)
    early_seal = early["arms"]["candidate"]["seal"]
    pseudo_seals = {
        "control": early_seal,
        "candidate": {"path": str(args.dense12k_refit.resolve()),
                      "file_sha256": EXPECTED_REFIT_SHA256},
    }
    audit = scratch.TailAudit(pseudo_seals)
    trajectories, metadata, windows = point_eval.load_tail_rows(
        args.real_dir, evaluation_rows, audit)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    indices_by_aoa = scratch.group_window_indices_by_aoa(windows, metadata)

    stats = torch.load(
        late["normalizer"]["path"], map_location="cpu", weights_only=False)
    if (smu1.statistics_digest(stats)
            != late["normalizer"]["statistics_sha256"]
            or late["normalizer"]["statistics_sha256"]
            != early["normalizer"]["statistics_sha256"]):
        raise RuntimeError("role endpoint normalizers differ")
    device = torch.device(args.device)
    normalizer = smu1.FoldNormalizer(stats, device)
    models = [
        load_checkpoint(
            Path(early_seal["path"]), early_seal["file_sha256"],
            normalizer, device),
        load_checkpoint(
            args.dense12k_refit, EXPECTED_REFIT_SHA256,
            normalizer, device),
    ]
    deadline = time.perf_counter() + args.wall_budget_seconds
    by_aoa = {}
    pooled_states = {"control": [], "teacher": []}
    fluctuation_max = 0.0
    for aoa, indices in indices_by_aoa.items():
        metrics, states, target_audit = evaluate_partition(
            models, dataset, indices, normalizer, device, deadline)
        fluctuation_max = max(
            fluctuation_max,
            float(target_audit["normalized_fluctuation_max_abs_delta"]))
        by_aoa[str(aoa)] = {
            "windows": len(indices), "metrics": metrics,
            "delta": scratch.compare(
                metrics["teacher"], metrics["control"])["delta"],
            "target_audit": target_audit,
        }
        for key in pooled_states:
            pooled_states[key].append(states[key])
    pooled = {
        key: smu1.finalize_metric_state(smu1.merge_metric_states(values))
        for key, values in pooled_states.items()
    }
    comparison = scratch.compare(pooled["teacher"], pooled["control"])
    by_aoa_delta = {key: row["delta"] for key, row in by_aoa.items()}
    gate = role_gate(comparison["delta"], by_aoa_delta, fluctuation_max)
    result = {
        "schema": SCHEMA,
        "composition": (
            "Dense12k + mean_t(Dense4k) - mean_t(Dense12k)"),
        "mean_weight_dense4k": 1.0,
        "assets": {
            "dense4k_result": {"path": str(args.dense4k_result.resolve()),
                               "sha256": dr1.sha256(args.dense4k_result)},
            "dense12k_result": {"path": str(args.dense12k_result.resolve()),
                                "sha256": dr1.sha256(args.dense12k_result)},
            "dense12k_refit": {"path": str(args.dense12k_refit.resolve()),
                               "sha256": EXPECTED_REFIT_SHA256},
        },
        "pooled": pooled,
        "comparison": comparison,
        "by_aoa": by_aoa,
        "normalized_fluctuation_max_abs_delta": fluctuation_max,
        "gate": gate,
        "selection": "full temporal-mean replacement fixed before evaluation",
        "submission_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    del models
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense4k-result", type=Path, required=True)
    parser.add_argument("--dense12k-result", type=Path, required=True)
    parser.add_argument("--dense12k-refit", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3600)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "event": "b210_dense4k_mean_dense12k_fluctuation_complete",
        "control": result["pooled"]["control"],
        "teacher": result["pooled"]["teacher"],
        "delta": result["comparison"]["delta"],
        "gate": result["gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
