#!/usr/bin/env python3
"""Evaluate a fixed 50/50 function ensemble of two scratch B210 endpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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

import evaluate_b210_scratch_point_pair_by_aoa as point_eval  # noqa: E402
import train_b210_np_functional_real_ft12_loao as ft12  # noqa: E402
import train_b210_scratch_real_dense_aug as scratch  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402


SCHEMA = "b210_fixed_function_ensemble_v1"
ALPHA_B = 0.50


def load_pair(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != scratch.SCHEMA:
        raise ValueError(f"{path} is not a scratch Real pair result")
    return payload


def candidate_seal(payload: dict) -> dict:
    seal = payload.get("arms", {}).get("candidate", {}).get("seal")
    if not isinstance(seal, dict):
        raise ValueError("pair result has no sealed candidate endpoint")
    return seal


def fixed_blend(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape or first.dtype != second.dtype:
        raise ValueError("function-ensemble tensors differ in shape or dtype")
    return torch.lerp(first.float(), second.float(), ALPHA_B)


@torch.inference_mode()
def evaluate_fixed(models, dataset, indices, normalizer,
                   device: torch.device, deadline: float):
    loader = DataLoader(
        dataset, batch_sampler=[
            list(indices[start:start + smu1.EVAL_BATCH_SIZE])
            for start in range(0, len(indices), smu1.EVAL_BATCH_SIZE)],
        num_workers=0, pin_memory=True)
    state = smu1.new_metric_state()
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    disagreement_square_sum = 0.0
    disagreement_elements = 0
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for model in models:
        model.eval()
    for history, target in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("fixed B210 ensemble evaluation exceeded budget")
        target_cpu = target.contiguous()
        scored_cpu = target_cpu[..., :2].ne(0.0).to(torch.uint8).contiguous()
        target_hasher.update(str(tuple(target_cpu.shape)).encode("ascii"))
        target_hasher.update(target_cpu.numpy().tobytes(order="C"))
        mask_hasher.update(str(tuple(scored_cpu.shape)).encode("ascii"))
        mask_hasher.update(scored_cpu.numpy().tobytes(order="C"))
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, _target_norm = normalizer.preprocess(history, target)
        outputs = [model.forward_distribution(history_norm) for model in models]
        prediction_norm = fixed_blend(outputs[0][0], outputs[1][0])
        lower_norm = fixed_blend(outputs[0][1], outputs[1][1])
        upper_norm = fixed_blend(outputs[0][2], outputs[1][2])
        prediction = normalizer.postprocess_prediction(prediction_norm)
        first_prediction = normalizer.postprocess_prediction(outputs[0][0].float())
        second_prediction = normalizer.postprocess_prediction(outputs[1][0].float())
        delta = first_prediction[..., :2] - second_prediction[..., :2]
        disagreement_square_sum += float(delta.square().sum(dtype=torch.float64))
        disagreement_elements += delta.numel()
        smu1.update_metric_state(
            state, prediction.float(), target.float(),
            lower_norm.float() * scale, upper_norm.float() * scale)
    if int(state["samples"]) != len(indices):
        raise RuntimeError("fixed ensemble did not consume complete partition")
    return smu1.finalize_metric_state(state), state, {
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
        "samples": int(state["samples"]),
        "scored_elements": int(state["sps_scored"]),
        "endpoint_point_disagreement_rms": (
            disagreement_square_sum / disagreement_elements) ** 0.5,
    }


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    source_a = load_pair(args.pair_result_a)
    source_b = load_pair(args.pair_result_b)
    evaluation_a = source_a["evaluation_manifest"]
    evaluation_b = source_b["evaluation_manifest"]
    if (evaluation_a["sha256"] != evaluation_b["sha256"]
            or int(evaluation_a["tail_windows"])
            != int(evaluation_b["tail_windows"])):
        raise RuntimeError("pair results use different clean-tail evaluations")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean prefix manifest byte identity changed")
    rows, _manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_path = Path(str(evaluation_a["path"]))
    if dr1.sha256(evaluation_path) != evaluation_a["sha256"]:
        raise RuntimeError("fixed evaluation manifest changed")
    evaluation_rows = scratch.load_evaluation_rows(evaluation_path, rows)
    seals = {
        "control": candidate_seal(source_a),
        "candidate": candidate_seal(source_b),
    }
    audit = scratch.TailAudit(seals)
    trajectories, metadata, windows = point_eval.load_tail_rows(
        args.real_dir, evaluation_rows, audit)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    indices_by_aoa = scratch.group_window_indices_by_aoa(windows, metadata)

    stats_a = torch.load(
        source_a["normalizer"]["path"], map_location="cpu", weights_only=False)
    stats_b = torch.load(
        source_b["normalizer"]["path"], map_location="cpu", weights_only=False)
    digest_a = smu1.statistics_digest(stats_a)
    digest_b = smu1.statistics_digest(stats_b)
    if (digest_a != source_a["normalizer"]["statistics_sha256"]
            or digest_b != source_b["normalizer"]["statistics_sha256"]
            or digest_a != digest_b):
        raise RuntimeError("endpoint normalizers differ or changed")
    device = torch.device(args.device)
    normalizer = smu1.FoldNormalizer(stats_a, device)
    models = [
        point_eval.load_primary_endpoint_mmap(seal, normalizer, device)
        for seal in seals.values()
    ]
    deadline = time.perf_counter() + args.wall_budget_seconds
    by_aoa = {}
    states = []
    audits = []
    for aoa, indices in indices_by_aoa.items():
        metrics, state, target_audit = evaluate_fixed(
            models, dataset, indices, normalizer, device, deadline)
        by_aoa[str(aoa)] = {
            "windows": len(indices), "metrics": metrics,
            "target_audit": target_audit,
        }
        states.append(state)
        audits.append(target_audit)
    pooled = smu1.finalize_metric_state(smu1.merge_metric_states(states))
    endpoint_metrics = {
        "seed0": source_a["arms"]["candidate"]["metrics"],
        "seed1": source_b["arms"]["candidate"]["metrics"],
    }
    result = {
        "schema": SCHEMA,
        "alpha_b": ALPHA_B,
        "pair_results": {
            "seed0": {"path": str(args.pair_result_a.resolve()),
                      "sha256": dr1.sha256(args.pair_result_a)},
            "seed1": {"path": str(args.pair_result_b.resolve()),
                      "sha256": dr1.sha256(args.pair_result_b)},
        },
        "endpoint_metrics": endpoint_metrics,
        "fixed_ensemble": {"pooled": pooled, "by_aoa": by_aoa},
        "delta_from_seed0": scratch.compare(pooled, endpoint_metrics["seed0"]),
        "delta_from_seed1": scratch.compare(pooled, endpoint_metrics["seed1"]),
        "target_audits": audits,
        "selection": "alpha fixed before opening clean716",
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
    parser.add_argument("--pair-result-a", type=Path, required=True)
    parser.add_argument("--pair-result-b", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3600)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "event": "b210_fixed_function_ensemble_complete",
        "pooled": result["fixed_ensemble"]["pooled"],
        "delta_from_seed0": result["delta_from_seed0"]["delta"],
    }), flush=True)


if __name__ == "__main__":
    main()
