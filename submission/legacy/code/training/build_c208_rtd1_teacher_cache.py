#!/usr/bin/env python3
"""Build the single frozen C208-RTD1 Real2576 teacher cache."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch

import train_c208_real_sim_rehearsal as rehearsal
import train_hybrid_fno2d as common
import train_p1_direct_native_endpoint_teacher as endpoint_teacher
import train_p1_direct_native_quantile_teacher as quantile_teacher
from train_uq_budget_reallocator import trajectory_folds


SCHEMA_VERSION = 1
CACHE_METHOD = "c208_rtd1_real2576_foldaware_physical_v1"
EVIDENCE_CLASS = (
    "fixed725_checkpoint_selection_contaminated_teacher_diagnostic")
POINT_WEIGHT_B = 0.5
POINT_WEIGHT_C = 0.5
BOUNDS_WEIGHT_B = 0.625
BOUNDS_WEIGHT_C = 0.25
EXPECTED_SHAPE = (2576, 20, 32, 64, 2)


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


def tensor_sha256(value: torch.Tensor, chunk_rows: int = 16) -> str:
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("tensor identity requires a contiguous CPU tensor")
    digest = hashlib.sha256()
    for start in range(0, value.shape[0], chunk_rows):
        block = value[start:start + chunk_rows].numpy()
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def train_keys(metadata: list[dict], windows) -> list[list[object]]:
    return [[metadata[window.trajectory]["file"], int(window.start)]
            for window in windows]


@torch.inference_mode()
def collect_absolute_endpoints_float32(model, loader, normalizer,
                                       device: torch.device):
    """Collect physical A/B point and absolute bounds without FP16 transit."""
    predictions, lowers, uppers = [], [], []
    dtype = next(model.parameters()).dtype
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    model.eval()
    for history, _unused_target in loader:
        history = history.to(device, non_blocking=True)
        normalized = (
            history - normalizer.mean_input) / normalizer.std_input
        prediction_norm, lower_extent, upper_extent = (
            model.forward_distribution(normalized.to(dtype=dtype)))
        prediction = common.to_physical_target(
            prediction_norm.float(), normalizer)[..., :2]
        lower = prediction - lower_extent.float() * scale
        upper = prediction + upper_extent.float() * scale
        if (not bool(torch.isfinite(lower).all())
                or not bool(torch.isfinite(upper).all())
                or bool(torch.any(lower > upper))):
            raise FloatingPointError("fold model produced invalid endpoints")
        predictions.append(prediction.cpu())
        lowers.append(lower.cpu())
        uppers.append(upper.cpu())
    return (torch.cat(predictions).contiguous(),
            torch.cat(lowers).contiguous(),
            torch.cat(uppers).contiguous())


@torch.inference_mode()
def collect_point_predictions_float32(model, loader, normalizer,
                                      device: torch.device) -> torch.Tensor:
    """Collect physical point forecasts without an FP16 intermediate."""
    rows = []
    dtype = next(model.parameters()).dtype
    model.eval()
    for history, _unused_target in loader:
        history = history.to(device, non_blocking=True)
        history_norm = (
            history - normalizer.mean_input) / normalizer.std_input
        prediction_norm = model(history_norm.to(dtype=dtype))
        prediction = common.to_physical_target(
            prediction_norm.float(), normalizer)[..., :2]
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("mean expert produced a non-finite point")
        rows.append(prediction.cpu())
    return torch.cat(rows).contiguous()


def compose_rtd1_teacher(
        prediction_a: torch.Tensor, lower_a: torch.Tensor,
        upper_a: torch.Tensor, prediction_b: torch.Tensor,
        lower_b: torch.Tensor, upper_b: torch.Tensor,
        prediction_c: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose current-P1 point and fixed-P0 absolute bounds."""
    base_point = prediction_a.lerp(prediction_b, POINT_WEIGHT_B)
    point_correction = POINT_WEIGHT_C * (
        prediction_c.mean(dim=1, keepdim=True)
        - base_point.mean(dim=1, keepdim=True))
    point = base_point + point_correction
    lower, upper = endpoint_teacher.compose_p0_absolute_endpoints(
        prediction_a, lower_a, upper_a,
        prediction_b, lower_b, upper_b, prediction_c,
        weight_b=BOUNDS_WEIGHT_B, weight_c=BOUNDS_WEIGHT_C)
    if (not bool(torch.isfinite(point).all())
            or bool(torch.any(lower > upper))):
        raise FloatingPointError("composed RTD1 teacher is invalid")
    return point, lower, upper


def build_cache(args: argparse.Namespace, dataset, point_folds: list[int],
                fold_manifest: dict, normalizer, device: torch.device):
    if len(dataset) != EXPECTED_SHAPE[0] or len(point_folds) != len(dataset):
        raise RuntimeError("RTD1 cache requires the exact Real2576 train split")
    if set(point_folds) != set(range(5)):
        raise RuntimeError("RTD1 cache requires all five condition folds")
    sources = endpoint_teacher.validate_formal_source_files(args)
    contracts = endpoint_teacher.validate_formal_oof_contracts(
        args, fold_manifest)
    payload_a = torch.load(
        args.point_checkpoint_a, map_location="cpu", weights_only=False)
    payload_b = torch.load(
        args.point_checkpoint_b, map_location="cpu", weights_only=False)
    point = torch.empty(EXPECTED_SHAPE, dtype=torch.float16)
    lower = torch.empty(EXPECTED_SHAPE, dtype=torch.float32)
    upper = torch.empty(EXPECTED_SHAPE, dtype=torch.float32)

    model_c, metadata_c = endpoint_teacher.point_gate.load_model(
        args.mean_expert_checkpoint, normalizer, device)
    all_indices = list(range(len(dataset)))
    c_loader = quantile_teacher.make_loader(
        dataset, all_indices, args.batch_size, args.num_workers)
    prediction_c = collect_point_predictions_float32(
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
        prediction_a, lower_a, upper_a = collect_absolute_endpoints_float32(
            model_a, loader, normalizer, device)
        del model_a
        gc.collect()
        torch.cuda.empty_cache()

        model_b, metadata_b = endpoint_teacher.s1.load_fold_model(
            payload_b, args.oof_dir_b / f"fold_{fold:02d}_best.pth",
            fold, args.point_checkpoint_b, normalizer, device)
        prediction_b, lower_b, upper_b = collect_absolute_endpoints_float32(
            model_b, loader, normalizer, device)
        del model_b
        c_rows = prediction_c[indices]
        point_rows, lower_rows, upper_rows = compose_rtd1_teacher(
            prediction_a, lower_a, upper_a,
            prediction_b, lower_b, upper_b, c_rows)
        output_rows = torch.as_tensor(indices, dtype=torch.long)
        point[output_rows] = point_rows.to(torch.float16)
        lower[output_rows] = lower_rows
        upper[output_rows] = upper_rows
        fold_rows.append({
            "fold": fold,
            "samples": len(indices),
            "model_a": metadata_a,
            "model_b": metadata_b,
        })
        print(json.dumps({
            "event": "c208_rtd1_cache_fold",
            "fold": fold,
            "samples": len(indices),
        }), flush=True)
        del (prediction_a, lower_a, upper_a, prediction_b, lower_b, upper_b,
             c_rows, point_rows, lower_rows, upper_rows, output_rows)
        gc.collect()
        torch.cuda.empty_cache()
    del payload_a, payload_b, prediction_c
    if (not bool(torch.isfinite(point).all())
            or not bool(torch.isfinite(lower).all())
            or not bool(torch.isfinite(upper).all())
            or bool(torch.any(lower > upper))):
        raise FloatingPointError("completed RTD1 cache is invalid")
    return point.contiguous(), lower.contiguous(), upper.contiguous(), {
        "source_contracts": sources,
        "oof_contracts": contracts,
        "reconstruction": {
            "mean_expert": metadata_c,
            "folds": fold_rows,
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal RTD1 teacher reconstruction requires CUDA")
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
    point, lower, upper, provenance = build_cache(
        args, dataset, point_folds, fold_manifest, normalizer, device)
    identities = {
        "point_uv_physical": tensor_sha256(point),
        "lower_uv_physical": tensor_sha256(lower),
        "upper_uv_physical": tensor_sha256(upper),
    }
    metadata_payload = {
        "evidence_class": EVIDENCE_CLASS,
        "split_manifest_sha256": split_sha,
        "real_stats_sha256": stats_sha,
        "train_keys": train_keys(metadata, train_windows),
        "point_weights": {"b": POINT_WEIGHT_B, "c": POINT_WEIGHT_C},
        "bounds_weights": {"b": BOUNDS_WEIGHT_B, "c": BOUNDS_WEIGHT_C},
        "tensor_sha256": identities,
        "tensor_storage": {
            "point_uv_physical": "float16",
            "lower_uv_physical": "float32",
            "upper_uv_physical": "float32",
        },
        "units": {
            "point_uv_physical": "physical_velocity",
            "lower_uv_physical": "physical_velocity",
            "upper_uv_physical": "physical_velocity",
        },
        "fold_manifest": fold_manifest,
        **provenance,
        "validation_windows_inferred": 0,
        "wall_seconds_before_save": time.perf_counter() - started,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": CACHE_METHOD,
        "metadata": metadata_payload,
        "point_uv_physical": point,
        "lower_uv_physical": lower,
        "upper_uv_physical": upper,
        "fold": torch.as_tensor(point_folds, dtype=torch.int8),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(json.dumps({
        "event": "c208_rtd1_teacher_cache_complete",
        "output": str(args.output),
        "samples": len(point),
        "tensor_sha256": identities,
        "wall_seconds": time.perf_counter() - started,
    }), flush=True)


if __name__ == "__main__":
    main()
