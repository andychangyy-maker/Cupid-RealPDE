#!/usr/bin/env python3
"""Cache formal-P1 A/B fluctuation and temporal-mean roles on Sim windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import build_sim_p1_point_teacher_cache as point_cache
import train_hybrid_fno2d as common


CACHE_SCHEMA_VERSION = 1
CACHE_METHOD = "sim_formal_p1_role_teacher_physical_v1"
FLUCTUATION_WEIGHTS = {"a": 0.5, "b": 0.5}
MEAN_WEIGHTS = {"a": 0.25, "b": 0.25, "c": 0.5}
TENSOR_KEYS = (
    "a_fluctuation_uv_physical",
    "b_fluctuation_uv_physical",
    "p1_temporal_mean_uv_physical",
)
TENSOR_STORAGE = {name: "float16" for name in TENSOR_KEYS}
TENSOR_UNITS = {name: "physical_velocity" for name in TENSOR_KEYS}
DECOMPOSITION_TOLERANCE = 1e-6
COMPOSITION = {
    "fluctuation_weights": FLUCTUATION_WEIGHTS,
    "mean_weights": MEAN_WEIGHTS,
    "formula": (
        "A_fluct=A-mean_t(A); B_fluct=B-mean_t(B); "
        "P1_mean=0.25*mean_t(A)+0.25*mean_t(B)+0.5*mean_t(C)"
    ),
    "formal_p1": (
        "base=A.lerp(B,0.5); "
        "point=base+0.5*(mean_t(C)-mean_t(base))"
    ),
    "source_prediction_dtype": "float16",
    "decomposition_precision": "float32_before_final_float16_storage",
    "validation_tolerance": DECOMPOSITION_TOLERANCE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", type=Path, nargs=3, required=True,
        metavar=("A", "B", "C"))
    parser.add_argument(
        "--expected-checkpoint-sha256", nargs=3, required=True,
        metavar=("A_SHA256", "B_SHA256", "C_SHA256"))
    parser.add_argument("--sim-dir", type=Path, default=Path("data/sim"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def temporary_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".tmp")


def validate_args(args: argparse.Namespace) -> None:
    point_cache.validate_args(args)
    temporary = temporary_path(args.output)
    if temporary.exists():
        raise FileExistsError(
            f"stale temporary teacher cache exists: {temporary}")


def tensor_sha256(value: torch.Tensor, chunk_rows: int = 16) -> str:
    """Hash the exact bytes of a contiguous CPU tensor."""
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("tensor identity requires a contiguous CPU tensor")
    if chunk_rows <= 0:
        raise ValueError("tensor hash chunk size must be positive")
    digest = hashlib.sha256()
    for start in range(0, value.shape[0], chunk_rows):
        block = value[start:start + chunk_rows].numpy()
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def expected_tensor_shapes(dataset) -> dict[str, list[int]]:
    prediction_shape = point_cache.expected_prediction_shape(dataset)
    if len(prediction_shape) != 5:
        raise ValueError("Sim prediction must have shape (N,T,H,W,2)")
    mean_shape = (
        prediction_shape[0], 1, *prediction_shape[2:])
    return {
        TENSOR_KEYS[0]: list(prediction_shape),
        TENSOR_KEYS[1]: list(prediction_shape),
        TENSOR_KEYS[2]: list(mean_shape),
    }


def expected_teacher_sha_map(values) -> dict[str, str]:
    if isinstance(values, dict):
        if set(values) != {"a", "b", "c"}:
            raise ValueError("expected teacher SHA mapping must contain a, b, and c")
        result = {name: str(values[name]).lower() for name in ("a", "b", "c")}
    else:
        if len(values) != 3:
            raise ValueError("exactly three expected teacher SHA values are required")
        result = {
            name: str(digest).lower()
            for name, digest in zip(("a", "b", "c"), values)
        }
    for name, digest in result.items():
        if (len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"expected teacher {name} SHA-256 is invalid")
    return result


def compose_p1_roles_float32(
        prediction_a: torch.Tensor, prediction_b: torch.Tensor,
        prediction_c: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Decompose source predictions and prove equivalence to formal P1."""
    predictions = (prediction_a, prediction_b, prediction_c)
    if not all(value.dtype == torch.float32 for value in predictions):
        raise TypeError("P1 roles must be decomposed in float32")
    if not all(value.shape == prediction_a.shape for value in predictions):
        raise ValueError("P1 source prediction shapes differ")
    if prediction_a.ndim != 5 or prediction_a.shape[1] <= 0:
        raise ValueError("P1 source predictions must have shape (N,T,H,W,2)")
    if prediction_a.shape[-1] != 2:
        raise ValueError("P1 source predictions must contain exactly u and v")
    if not all(bool(torch.isfinite(value).all()) for value in predictions):
        raise FloatingPointError("P1 source predictions contain non-finite values")

    mean_a = prediction_a.mean(dim=1, keepdim=True)
    mean_b = prediction_b.mean(dim=1, keepdim=True)
    mean_c = prediction_c.mean(dim=1, keepdim=True)
    fluctuation_a = prediction_a - mean_a
    fluctuation_b = prediction_b - mean_b
    p1_mean = (
        MEAN_WEIGHTS["a"] * mean_a
        + MEAN_WEIGHTS["b"] * mean_b
        + MEAN_WEIGHTS["c"] * mean_c)

    reconstructed = (
        FLUCTUATION_WEIGHTS["a"] * fluctuation_a
        + FLUCTUATION_WEIGHTS["b"] * fluctuation_b
        + p1_mean)
    base = prediction_a.lerp(prediction_b, 0.5)
    formal_p1 = base + 0.5 * (
        mean_c - base.mean(dim=1, keepdim=True))
    metrics = {
        "reconstruction_max_abs": float(
            (reconstructed - formal_p1).abs().max()),
        "a_fluctuation_mean_max_abs": float(
            fluctuation_a.mean(dim=1).abs().max()),
        "b_fluctuation_mean_max_abs": float(
            fluctuation_b.mean(dim=1).abs().max()),
    }
    if any(value > DECOMPOSITION_TOLERANCE for value in metrics.values()):
        raise RuntimeError(
            f"formal P1 role decomposition exceeded tolerance: {metrics}")
    return fluctuation_a, fluctuation_b, p1_mean, metrics


def compose_roles_in_place(
        prediction_a: torch.Tensor, prediction_b: torch.Tensor,
        prediction_c: torch.Tensor, chunk_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Replace half-precision A/B sources with roles using bounded FP32 buffers."""
    point_cache.validate_prediction_pair(prediction_a, prediction_b)
    point_cache.validate_prediction_pair(prediction_a, prediction_c)
    if chunk_size <= 0:
        raise ValueError("composition chunk size must be positive")
    mean_shape = (len(prediction_a), 1, *prediction_a.shape[2:])
    p1_mean = torch.empty(mean_shape, dtype=torch.float16)
    validation = {
        "reconstruction_max_abs": 0.0,
        "a_fluctuation_mean_max_abs": 0.0,
        "b_fluctuation_mean_max_abs": 0.0,
    }
    for start in range(0, len(prediction_a), chunk_size):
        stop = min(start + chunk_size, len(prediction_a))
        roles = compose_p1_roles_float32(
            prediction_a[start:stop].float(),
            prediction_b[start:stop].float(),
            prediction_c[start:stop].float())
        fluctuation_a, fluctuation_b, mean_rows, metrics = roles
        prediction_a[start:stop].copy_(fluctuation_a)
        prediction_b[start:stop].copy_(fluctuation_b)
        p1_mean[start:stop].copy_(mean_rows)
        for name, value in metrics.items():
            validation[name] = max(validation[name], value)
    validation["tolerance"] = DECOMPOSITION_TOLERANCE
    return (
        prediction_a.contiguous(), prediction_b.contiguous(),
        p1_mean.contiguous(), validation)


def validate_cache_payload(
        payload: dict, manifest: list[dict], dataset,
        expected_checkpoint_sha256,
        ) -> tuple[dict[str, torch.Tensor], dict]:
    """Validate cache identity against teachers and the ordered Sim dataset."""
    if not isinstance(payload, dict):
        raise TypeError("role teacher cache payload must be a mapping")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("role teacher cache schema version mismatch")
    if payload.get("method") != CACHE_METHOD:
        raise ValueError("role teacher cache method mismatch")
    if payload.get("composition") != COMPOSITION:
        raise ValueError("role teacher cache composition mismatch")

    manifest_hash = point_cache.canonical_sha256(manifest)
    if payload.get("split_manifest_sha256") != manifest_hash:
        raise ValueError("role teacher cache split manifest hash mismatch")
    if payload.get("split_manifest") != manifest:
        raise ValueError("role teacher cache split manifest mismatch")
    if not hasattr(dataset, "windows") or len(dataset.windows) != len(dataset):
        raise ValueError("Sim dataset windows do not map one-to-one to samples")
    keys = point_cache.ordered_window_keys(dataset.windows, manifest)
    if payload.get("window_keys") != keys:
        raise ValueError("role teacher cache window order mismatch")
    if payload.get("window_order_sha256") != point_cache.canonical_sha256(keys):
        raise ValueError("role teacher cache window-order hash mismatch")
    if payload.get("windows") != len(keys):
        raise ValueError("role teacher cache window count mismatch")

    expected_shas = expected_teacher_sha_map(expected_checkpoint_sha256)
    identities = payload.get("teacher_checkpoints")
    if not isinstance(identities, dict) or set(identities) != {"a", "b", "c"}:
        raise ValueError("role teacher checkpoint identities are incomplete")
    for name, expected_sha in expected_shas.items():
        identity = identities[name]
        if not isinstance(identity, dict):
            raise TypeError(f"role teacher checkpoint {name} identity is invalid")
        actual = identity.get("actual_sha256")
        expected = identity.get("expected_sha256")
        if (actual != expected_sha or expected != expected_sha
                or identity.get("sha256_match") is not True):
            raise ValueError(f"role teacher checkpoint {name} SHA-256 mismatch")
        if identity.get("phase") != "sim":
            raise ValueError(f"role teacher checkpoint {name} is not Sim phase")
        if not isinstance(identity.get("size_bytes"), int) or (
                identity["size_bytes"] <= 0):
            raise ValueError(f"role teacher checkpoint {name} size is invalid")
        if not isinstance(identity.get("path"), str) or not identity["path"]:
            raise ValueError(f"role teacher checkpoint {name} path is invalid")

    shapes = expected_tensor_shapes(dataset)
    if payload.get("tensor_shapes") != shapes:
        raise ValueError("role teacher cache recorded tensor shapes mismatch")
    if payload.get("tensor_storage") != TENSOR_STORAGE:
        raise ValueError("role teacher cache recorded tensor storage mismatch")
    if payload.get("units") != TENSOR_UNITS:
        raise ValueError("role teacher cache units mismatch")
    recorded_hashes = payload.get("tensor_sha256")
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != set(
            TENSOR_KEYS):
        raise ValueError("role teacher cache tensor identities are incomplete")

    tensors = {}
    for name in TENSOR_KEYS:
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"role teacher cache {name} must be a tensor")
        if list(value.shape) != shapes[name]:
            raise ValueError(f"role teacher cache {name} shape mismatch")
        if value.dtype != torch.float16:
            raise TypeError(f"role teacher cache {name} must be float16")
        if value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError(f"role teacher cache {name} must be contiguous CPU data")
        if not point_cache.tensor_is_finite(value):
            raise ValueError(f"role teacher cache {name} contains non-finite values")
        digest = tensor_sha256(value)
        if recorded_hashes.get(name) != digest:
            raise ValueError(f"role teacher cache {name} SHA-256 mismatch")
        tensors[name] = value

    validation = payload.get("decomposition_validation")
    validation_keys = {
        "reconstruction_max_abs",
        "a_fluctuation_mean_max_abs",
        "b_fluctuation_mean_max_abs",
        "tolerance",
    }
    if not isinstance(validation, dict) or set(validation) != validation_keys:
        raise ValueError("role teacher decomposition validation is incomplete")
    if validation.get("tolerance") != DECOMPOSITION_TOLERANCE:
        raise ValueError("role teacher decomposition tolerance mismatch")
    for name in validation_keys - {"tolerance"}:
        value = validation[name]
        if (not isinstance(value, (int, float)) or value < 0
                or value > DECOMPOSITION_TOLERANCE):
            raise ValueError(f"role teacher decomposition metric {name} is invalid")

    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "method": CACHE_METHOD,
        "composition": COMPOSITION,
        "teacher_checkpoints": identities,
        "split_manifest_sha256": manifest_hash,
        "window_order_sha256": point_cache.canonical_sha256(keys),
        "windows": len(keys),
        "tensor_shapes": shapes,
        "tensor_storage": TENSOR_STORAGE,
        "tensor_sha256": recorded_hashes,
        "units": TENSOR_UNITS,
        "decomposition_validation": validation,
    }
    return tensors, metadata


def load_and_validate_cache(
        path: Path, manifest: list[dict], dataset,
        expected_checkpoint_sha256,
        ) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensors, metadata = validate_cache_payload(
        payload, manifest, dataset, expected_checkpoint_sha256)
    metadata = {
        **metadata,
        "path": str(path.resolve()),
        "sha256": point_cache.sha256(path),
    }
    return tensors, metadata


def save_cache_atomically(
        payload: dict, output: Path, manifest: list[dict], dataset,
        expected_checkpoint_sha256,
        ) -> dict:
    """Validate, serialize, revalidate, then atomically publish a new cache."""
    temporary = temporary_path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite teacher cache: {output}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary teacher cache exists: {temporary}")
    validate_cache_payload(
        payload, manifest, dataset, expected_checkpoint_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    _tensors, metadata = load_and_validate_cache(
        temporary, manifest, dataset, expected_checkpoint_sha256)
    del _tensors
    if output.exists():
        raise FileExistsError(f"refusing to overwrite teacher cache: {output}")
    os.replace(temporary, output)
    return {**metadata, "path": str(output.resolve())}


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    expected_shas = expected_teacher_sha_map(
        args.expected_checkpoint_sha256)
    identities = {
        name: point_cache.checkpoint_identity(path, expected_shas[name])
        for name, path in zip(("a", "b", "c"), args.checkpoints)
    }
    for name, path in zip(("a", "b", "c"), args.checkpoints):
        identities[name]["phase"] = point_cache.validate_checkpoint_phase(path)

    trajectories, windows, manifest = common.scratch.load_simulations(
        args.sim_dir, args.max_files, args.window_stride, True)
    dataset = common.scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    keys = point_cache.ordered_window_keys(windows, manifest)
    prediction_shape = point_cache.expected_prediction_shape(dataset)

    print(json.dumps({
        "event": "role_cache_contract",
        "windows": len(windows),
        "prediction_shape": list(prediction_shape),
        "split_manifest_sha256": point_cache.canonical_sha256(manifest),
        "window_order_sha256": point_cache.canonical_sha256(keys),
        "teacher_checkpoints": identities,
        "inference_dtype": str(dtype).removeprefix("torch."),
    }), flush=True)

    prediction_a = point_cache.collect_prediction(
        args.checkpoints[0], loader, device, dtype, prediction_shape)
    prediction_b = point_cache.collect_prediction(
        args.checkpoints[1], loader, device, dtype, prediction_shape)
    prediction_c = point_cache.collect_prediction(
        args.checkpoints[2], loader, device, dtype, prediction_shape)
    a_fluct, b_fluct, p1_mean, validation = compose_roles_in_place(
        prediction_a, prediction_b, prediction_c, args.batch_size)
    del prediction_c
    tensors = {
        TENSOR_KEYS[0]: a_fluct,
        TENSOR_KEYS[1]: b_fluct,
        TENSOR_KEYS[2]: p1_mean,
    }
    identities_by_tensor = {
        name: tensor_sha256(value) for name, value in tensors.items()
    }
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "method": CACHE_METHOD,
        "composition": COMPOSITION,
        "teacher_checkpoints": identities,
        "sim_dir": str(args.sim_dir.resolve()),
        "window_stride": args.window_stride,
        "max_files": args.max_files,
        "exclude_final_window": True,
        "inference_dtype": str(dtype).removeprefix("torch."),
        "split_manifest": manifest,
        "split_manifest_sha256": point_cache.canonical_sha256(manifest),
        "window_keys": keys,
        "window_order_sha256": point_cache.canonical_sha256(keys),
        "windows": len(keys),
        "tensor_shapes": expected_tensor_shapes(dataset),
        "tensor_storage": TENSOR_STORAGE,
        "tensor_sha256": identities_by_tensor,
        "units": TENSOR_UNITS,
        "decomposition_validation": validation,
        **tensors,
    }
    metadata = save_cache_atomically(
        payload, args.output, manifest, dataset, expected_shas)
    print(json.dumps({
        "event": "role_teacher_cache_complete",
        **metadata,
    }), flush=True)


if __name__ == "__main__":
    main()
