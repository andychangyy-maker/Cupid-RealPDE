#!/usr/bin/env python3
"""Build the exact fold-held P0 absolute-endpoint cache for distillation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

WEIGHT_B = 0.625
WEIGHT_C = 0.25
FULL_BOUNDS_SHA256 = {
    "lower": "d14aa4304e3e26829a5deede05f74315fbcc801f2eb169b30789a6eb363c8671",
    "upper": "b2926bec51dac7d3aff02f2523dc1dc204379bf1f5bf07160fe979087f672e45",
}
CACHE_FORMAT = "p0_oof_absolute_endpoints_uv_float32_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint-a", type=Path, required=True)
    parser.add_argument("--oof-dir-a", type=Path, required=True)
    parser.add_argument("--point-checkpoint-b", type=Path, required=True)
    parser.add_argument("--oof-dir-b", type=Path, required=True)
    parser.add_argument("--mean-expert-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_reconstruction_module():
    import evaluate_p1_observation_gated_uq_blend as p1_uq
    return p1_uq


def source_identities(args: argparse.Namespace, p1_uq) -> dict[str, dict]:
    paths = {
        "point_checkpoint_a": args.point_checkpoint_a,
        "point_checkpoint_b": args.point_checkpoint_b,
        "mean_expert_checkpoint": args.mean_expert_checkpoint,
        "oof_summary_a": args.oof_dir_a / "summary.json",
        "oof_summary_b": args.oof_dir_b / "summary.json",
        "split_manifest": args.split_manifest,
        "real_stats": args.real_stats,
    }
    for fold in range(5):
        paths[f"oof_a_fold_{fold:02d}"] = (
            args.oof_dir_a / f"fold_{fold:02d}_best.pth")
        paths[f"oof_b_fold_{fold:02d}"] = (
            args.oof_dir_b / f"fold_{fold:02d}_best.pth")
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing cache sources: {missing}")
    return {
        name: p1_uq.point_gate.s1.file_identity(path)
        for name, path in paths.items()
    }


def reconstruction_order(sample_folds: list[int]) -> np.ndarray:
    folds = np.asarray(sample_folds, dtype=np.int64)
    if folds.shape != (725,) or set(folds.tolist()) != set(range(5)):
        raise ValueError("expected 725 samples spanning condition folds 0..4")
    order = np.concatenate([
        np.flatnonzero(folds == fold) for fold in range(5)])
    if not np.array_equal(np.sort(order), np.arange(len(folds))):
        raise RuntimeError("condition-fold reconstruction order is not a permutation")
    return order


def validate_reconstruction_alignment(
        dataset, target: np.ndarray, folds: np.ndarray, aoas: np.ndarray,
        sample_folds: list[int], raw_aoas: np.ndarray,
        dataset_index: np.ndarray) -> None:
    if (target.shape != (725, 20, 32, 64, 3)
            or folds.shape != (725,) or aoas.shape != (725,)):
        raise ValueError("reconstructed OOF arrays have unexpected shapes")
    if not np.array_equal(folds, np.asarray(sample_folds)[dataset_index]):
        raise RuntimeError("reconstructed folds do not align with dataset indices")
    if not np.array_equal(aoas, raw_aoas[dataset_index]):
        raise RuntimeError("reconstructed AoAs do not align with dataset indices")
    for row, original_index in enumerate(dataset_index.tolist()):
        _history, expected_target = dataset[original_index]
        if not np.array_equal(target[row], expected_target.numpy()):
            raise RuntimeError(
                f"reconstructed target differs at cache row {row}, "
                f"dataset index {original_index}")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite cache: {args.output}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    p1_uq = load_reconstruction_module()
    identities = source_identities(args, p1_uq)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    trajectories, metadata = p1_uq.point_gate.common.repro.load_trajectories(
        args.real_dir, 0)
    point_windows, tail_windows, runtime_manifest = (
        p1_uq.point_gate.common.repro.make_split(metadata, 20, 0.2))
    if len(point_windows) != 2576 or len(tail_windows) != 725:
        raise RuntimeError("frozen 2576/725 split size changed")
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    sample_folds, fold_manifest = p1_uq.point_gate.trajectory_folds(
        runtime_manifest, 5)
    p1_uq.point_gate.s1.validate_s1_split(
        runtime_manifest, frozen_manifest, len(tail_windows), sample_folds)
    dataset = p1_uq.point_gate.common.repro.WindowDataset(
        trajectories, tail_windows)
    raw_aoas = np.asarray([
        int(metadata[window.trajectory]["aoa"])
        for window in tail_windows
    ], dtype=np.int64)
    if not set(raw_aoas.tolist()).issubset(set(p1_uq.AOA_VALUES)):
        raise ValueError("tail split contains an unexpected AoA")

    reconstructed = p1_uq.reconstruct_components(
        args, dataset, sample_folds, fold_manifest,
        np.zeros((len(dataset), 1), dtype=np.float64), raw_aoas, device)
    dataset_index = reconstruction_order(sample_folds)
    target = reconstructed["target"]
    folds = reconstructed["folds"]
    aoas = reconstructed["aoas"]
    validate_reconstruction_alignment(
        dataset, target, folds, aoas, sample_folds, raw_aoas, dataset_index)

    weights_b = np.full(len(dataset), WEIGHT_B, dtype=np.float32)
    weights_c = np.full(len(dataset), WEIGHT_C, dtype=np.float32)
    lower_full, upper_full = p1_uq.compose_sample_bounds(
        reconstructed["components"], weights_b, weights_c)
    full_identities = {
        "lower": p1_uq.point_gate.s1.array_identity(lower_full),
        "upper": p1_uq.point_gate.s1.array_identity(upper_full),
    }
    for side, expected in FULL_BOUNDS_SHA256.items():
        if full_identities[side]["sha256"] != expected:
            raise RuntimeError(
                f"P0 {side} float32 identity differs from frozen contract")

    lower_uv = np.ascontiguousarray(lower_full[..., :2], dtype=np.float32)
    upper_uv = np.ascontiguousarray(upper_full[..., :2], dtype=np.float32)
    target_uv_fp16 = np.ascontiguousarray(target[..., :2], dtype=np.float16)
    uv_identities = {
        "lower": p1_uq.point_gate.s1.array_identity(lower_uv),
        "upper": p1_uq.point_gate.s1.array_identity(upper_uv),
    }
    target_float32_identity = p1_uq.point_gate.s1.array_identity(target)
    target_storage_identity = p1_uq.point_gate.s1.array_identity(
        target_uv_fp16)
    metadata_payload = {
        "format": CACHE_FORMAT,
        "evidence_class": "fixed725_fold_held_p0_absolute_endpoints",
        "counts": {
            "point_windows_unused": len(point_windows),
            "tail_windows": len(dataset),
            "folds": [sample_folds.count(fold) for fold in range(5)],
        },
        "composition": {"weight_b": WEIGHT_B, "weight_c": WEIGHT_C},
        "ordering": (
            "condition fold 0..4; stable original WindowDataset order "
            "within each fold; dataset_index maps each cache row back"),
        "fold_manifest": fold_manifest,
        "full_float32_bounds_identity": full_identities,
        "stored_uv_float32_identity": uv_identities,
        "target_float32_identity": target_float32_identity,
        "stored_target_uv_fp16_identity": target_storage_identity,
        "source_identities": identities,
        "oof_summaries": reconstructed["oof_summaries"],
        "mean_expert": reconstructed["mean_expert"],
    }
    payload = {
        "lower": torch.from_numpy(lower_uv),
        "upper": torch.from_numpy(upper_uv),
        "target": torch.from_numpy(target_uv_fp16),
        "fold": torch.from_numpy(folds.astype(np.int8, copy=False)),
        "true_aoa": torch.from_numpy(aoas.astype(np.int8, copy=False)),
        "dataset_index": torch.from_numpy(
            dataset_index.astype(np.int16, copy=False)),
        "metadata": metadata_payload,
    }
    expected_bytes = sum(
        value.numel() * value.element_size()
        for value in payload.values() if isinstance(value, torch.Tensor))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output.parent).free
    if free_bytes < expected_bytes + 256 * 1024**2:
        raise OSError(
            f"insufficient free disk for cache: need at least "
            f"{(expected_bytes + 256 * 1024**2) / 1024**3:.2f} GiB")
    temporary = args.output.with_suffix(args.output.suffix + ".building")
    if temporary.exists():
        raise FileExistsError(f"stale temporary cache exists: {temporary}")
    del reconstructed, lower_full, upper_full, trajectories, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        torch.save(payload, temporary)
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    output_identity = p1_uq.point_gate.s1.file_identity(args.output)
    print(json.dumps({
        "event": "p0_absolute_endpoint_cache_complete",
        "cache": output_identity,
        "stored_tensor_bytes": expected_bytes,
        "full_bounds_identity": full_identities,
        "stored_uv_identity": uv_identities,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
