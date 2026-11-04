#!/usr/bin/env python3
"""Build the frozen fold-aware C208-RSM1 role-teacher cache."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch

import build_c208_rtd1_teacher_cache as rtd1_cache
import train_c208_real_sim_rehearsal as rehearsal
import train_hybrid_fno2d as common
import train_p1_direct_native_endpoint_teacher as endpoint_teacher
import train_p1_direct_native_quantile_teacher as quantile_teacher
from train_uq_budget_reallocator import trajectory_folds


SCHEMA_VERSION = 1
CACHE_METHOD = "c208_rsm1_real2576_foldaware_roles_physical_v1"
EVIDENCE_CLASS = (
    "fixed725_checkpoint_selection_contaminated_teacher_diagnostic")
FLUCTUATION_WEIGHTS = {"a": 0.5, "b": 0.5}
MEAN_WEIGHTS = {"a": 0.25, "b": 0.25, "c": 0.5}
EXPECTED_FLUCTUATION_SHAPE = (2576, 20, 32, 64, 2)
EXPECTED_MEAN_SHAPE = (2576, 1, 32, 64, 2)


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


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite teacher cache: {args.output}")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary teacher cache exists: {temporary}")


@torch.inference_mode()
def collect_distribution_points_float32(
        model, loader, normalizer, device: torch.device) -> torch.Tensor:
    """Collect physical fold-model point predictions without FP16 transit."""
    rows = []
    dtype = next(model.parameters()).dtype
    model.eval()
    for history, _unused_target in loader:
        history = history.to(device, non_blocking=True)
        history_norm = (
            history - normalizer.mean_input) / normalizer.std_input
        prediction_norm, _lower, _upper = model.forward_distribution(
            history_norm.to(dtype=dtype))
        prediction = common.to_physical_target(
            prediction_norm.float(), normalizer)[..., :2]
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("fold model produced a non-finite point")
        rows.append(prediction.cpu())
    return torch.cat(rows).contiguous()


def compose_rsm1_roles(
        prediction_a: torch.Tensor, prediction_b: torch.Tensor,
        prediction_c: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose formal P1 into A/B fluctuations and its temporal mean."""
    if not all(value.dtype == torch.float32 for value in (
            prediction_a, prediction_b, prediction_c)):
        raise TypeError("RSM1 roles must be composed in float32")
    mean_a = prediction_a.mean(dim=1, keepdim=True)
    mean_b = prediction_b.mean(dim=1, keepdim=True)
    mean_c = prediction_c.mean(dim=1, keepdim=True)
    fluctuation_a = prediction_a - mean_a
    fluctuation_b = prediction_b - mean_b
    p1_mean = (
        MEAN_WEIGHTS["a"] * mean_a
        + MEAN_WEIGHTS["b"] * mean_b
        + MEAN_WEIGHTS["c"] * mean_c)
    reconstructed_p1 = (
        FLUCTUATION_WEIGHTS["a"] * fluctuation_a
        + FLUCTUATION_WEIGHTS["b"] * fluctuation_b
        + p1_mean)
    base = prediction_a.lerp(prediction_b, 0.5)
    formal_p1 = base + 0.5 * (
        mean_c - base.mean(dim=1, keepdim=True))
    if not torch.allclose(reconstructed_p1, formal_p1, atol=2e-6, rtol=1e-6):
        raise RuntimeError("RSM1 decomposition differs from the formal P1 formula")
    if (not bool(torch.isfinite(reconstructed_p1).all())
            or not bool(torch.allclose(
                fluctuation_a.mean(dim=1),
                torch.zeros_like(mean_a[:, 0]), atol=2e-6, rtol=0.0))
            or not bool(torch.allclose(
                fluctuation_b.mean(dim=1),
                torch.zeros_like(mean_b[:, 0]), atol=2e-6, rtol=0.0))):
        raise FloatingPointError("composed RSM1 roles are invalid")
    return fluctuation_a, fluctuation_b, p1_mean


def scatter_role_rows(
        destination_a: torch.Tensor, destination_b: torch.Tensor,
        destination_mean: torch.Tensor, indices: list[int],
        rows_a: torch.Tensor, rows_b: torch.Tensor,
        rows_mean: torch.Tensor) -> None:
    """Write fold rows at their natural fixed2576 dataset indices."""
    if len(indices) != len(set(indices)) or not (
            len(indices) == len(rows_a) == len(rows_b) == len(rows_mean)):
        raise ValueError("RSM1 fold rows do not map one-to-one to natural indices")
    output_rows = torch.as_tensor(indices, dtype=torch.long)
    destination_a[output_rows] = rows_a.to(torch.float16)
    destination_b[output_rows] = rows_b.to(torch.float16)
    destination_mean[output_rows] = rows_mean.to(torch.float16)


def build_cache(args: argparse.Namespace, dataset, point_folds: list[int],
                fold_manifest: dict, normalizer, device: torch.device):
    if (len(dataset) != EXPECTED_FLUCTUATION_SHAPE[0]
            or len(point_folds) != len(dataset)):
        raise RuntimeError("RSM1 cache requires the exact Real2576 train split")
    if set(point_folds) != set(range(5)):
        raise RuntimeError("RSM1 cache requires all five condition folds")
    sources = endpoint_teacher.validate_formal_source_files(args)
    contracts = endpoint_teacher.validate_formal_oof_contracts(
        args, fold_manifest)
    payload_a = torch.load(
        args.point_checkpoint_a, map_location="cpu", weights_only=False)
    payload_b = torch.load(
        args.point_checkpoint_b, map_location="cpu", weights_only=False)
    fluctuation_a = torch.empty(
        EXPECTED_FLUCTUATION_SHAPE, dtype=torch.float16)
    fluctuation_b = torch.empty_like(fluctuation_a)
    p1_mean = torch.empty(EXPECTED_MEAN_SHAPE, dtype=torch.float16)

    model_c, metadata_c = endpoint_teacher.point_gate.load_model(
        args.mean_expert_checkpoint, normalizer, device)
    all_indices = list(range(len(dataset)))
    c_loader = quantile_teacher.make_loader(
        dataset, all_indices, args.batch_size, args.num_workers)
    prediction_c = rtd1_cache.collect_point_predictions_float32(
        model_c, c_loader, normalizer, device)
    del model_c
    gc.collect()
    torch.cuda.empty_cache()

    fold_rows = []
    for fold in range(5):
        indices = [index for index, value in enumerate(point_folds)
                   if value == fold]
        loader = quantile_teacher.make_loader(
            dataset, indices, args.batch_size, args.num_workers)
        model_a, metadata_a = endpoint_teacher.s1.load_fold_model(
            payload_a, args.oof_dir_a / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_a, normalizer, device)
        prediction_a = collect_distribution_points_float32(
            model_a, loader, normalizer, device)
        del model_a
        gc.collect()
        torch.cuda.empty_cache()

        model_b, metadata_b = endpoint_teacher.s1.load_fold_model(
            payload_b, args.oof_dir_b / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_b, normalizer, device)
        prediction_b = collect_distribution_points_float32(
            model_b, loader, normalizer, device)
        del model_b
        c_rows = prediction_c[indices]
        a_rows, b_rows, mean_rows = compose_rsm1_roles(
            prediction_a, prediction_b, c_rows)
        scatter_role_rows(
            fluctuation_a, fluctuation_b, p1_mean, indices,
            a_rows, b_rows, mean_rows)
        fold_rows.append({
            "fold": fold,
            "samples": len(indices),
            "model_a": metadata_a,
            "model_b": metadata_b,
        })
        print(json.dumps({
            "event": "c208_rsm1_cache_fold",
            "fold": fold,
            "samples": len(indices),
        }), flush=True)
        del prediction_a, prediction_b, c_rows, a_rows, b_rows, mean_rows
        gc.collect()
        torch.cuda.empty_cache()
    del payload_a, payload_b, prediction_c
    if (not bool(torch.isfinite(fluctuation_a).all())
            or not bool(torch.isfinite(fluctuation_b).all())
            or not bool(torch.isfinite(p1_mean).all())):
        raise FloatingPointError("completed RSM1 cache is invalid")
    return (
        fluctuation_a.contiguous(), fluctuation_b.contiguous(),
        p1_mean.contiguous(), {
            "source_contracts": sources,
            "oof_contracts": contracts,
            "reconstruction": {
                "mean_expert": metadata_c,
                "folds": fold_rows,
            },
        })


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal RSM1 teacher reconstruction requires CUDA")
    quantile_teacher.seed_everything(0)
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    train_windows, validation_windows, runtime_manifest = (
        common.repro.make_split(metadata, 20, 0.2))
    counts = {
        "train_windows": len(train_windows),
        "val_windows": len(validation_windows),
        "trajectories": len(trajectories),
    }
    if counts != rehearsal.FORMAL_REAL_COUNTS:
        raise RuntimeError(f"frozen Real split counts changed: {counts}")
    frozen_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8"))
    if runtime_manifest != frozen_manifest:
        raise RuntimeError("runtime split differs from the frozen manifest")
    split_sha = rehearsal.canonical_sha256(runtime_manifest)
    stats_sha = rehearsal.sha256(args.real_stats)
    if split_sha != rehearsal.FORMAL_REAL_SPLIT_SHA256:
        raise RuntimeError("Real split canonical SHA-256 mismatch")
    if stats_sha != rehearsal.FORMAL_REAL_STATS_SHA256:
        raise RuntimeError("Real normalizer SHA-256 mismatch")
    _tail_folds, fold_manifest = trajectory_folds(runtime_manifest, 5)
    point_folds = quantile_teacher.partition_fold_labels(
        runtime_manifest, fold_manifest, "train_starts")
    dataset = common.repro.WindowDataset(trajectories, train_windows)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    a_fluct, b_fluct, p1_mean, provenance = build_cache(
        args, dataset, point_folds, fold_manifest, normalizer, device)
    tensors = {
        "a_fluctuation_uv_physical": a_fluct,
        "b_fluctuation_uv_physical": b_fluct,
        "p1_temporal_mean_uv_physical": p1_mean,
    }
    identities = {
        name: rtd1_cache.tensor_sha256(value)
        for name, value in tensors.items()
    }
    metadata_payload = {
        "evidence_class": EVIDENCE_CLASS,
        "split_manifest_sha256": split_sha,
        "real_stats_sha256": stats_sha,
        "train_keys": rtd1_cache.train_keys(metadata, train_windows),
        "fluctuation_weights": FLUCTUATION_WEIGHTS,
        "mean_weights": MEAN_WEIGHTS,
        "tensor_sha256": identities,
        "tensor_storage": {name: "float16" for name in tensors},
        "units": {name: "physical_velocity" for name in tensors},
        "fold_manifest": fold_manifest,
        "composition_precision": "float32_before_final_float16_storage",
        **provenance,
        "validation_windows_inferred": 0,
        "wall_seconds_before_save": time.perf_counter() - started,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": CACHE_METHOD,
        "metadata": metadata_payload,
        **tensors,
        "fold": torch.as_tensor(point_folds, dtype=torch.int8),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(json.dumps({
        "event": "c208_rsm1_teacher_cache_complete",
        "output": str(args.output),
        "samples": len(a_fluct),
        "tensor_sha256": identities,
        "wall_seconds": time.perf_counter() - started,
    }), flush=True)


if __name__ == "__main__":
    main()
