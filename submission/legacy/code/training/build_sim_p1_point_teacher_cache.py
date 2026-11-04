#!/usr/bin/env python3
"""Cache the current P1 A/B/C point composition on deterministic Sim windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for path in (CODE_ROOT, Path(__file__).parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_hybrid_fno2d as common  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


CACHE_SCHEMA_VERSION = 1
CACHE_METHOD = "sim_current_p1_point_teacher"
POINT_WEIGHT_B = 0.5
TEMPORAL_MEAN_WEIGHT_C = 0.5
COMPOSITION = {
    "point_weight_b": POINT_WEIGHT_B,
    "point_temporal_mean_weight_c": TEMPORAL_MEAN_WEIGHT_C,
    "formula": (
        "base=A.lerp(B,0.5); "
        "point=base+0.5*(mean_t(C)-mean_t(base))"),
    "precision": (
        "float16 checkpoint predictions and stage buffers, "
        "float32 per-chunk arithmetic, float16 cache"),
}
NORMALIZER_KEYS = (
    "mean_input", "mean_target", "std_input", "std_target")


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


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.window_stride <= 0:
        raise ValueError("batch size and window stride must be positive")
    if args.max_files < 0 or args.num_workers < 0:
        raise ValueError("max files and workers must be non-negative")
    if len(set(path.resolve() for path in args.checkpoints)) != 3:
        raise ValueError("A, B, and C checkpoints must be distinct files")
    missing = [str(path) for path in args.checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing teacher checkpoints: {missing}")
    for name, digest in zip(
            ("A", "B", "C"), args.expected_checkpoint_sha256):
        normalized = str(digest).lower()
        if (len(normalized) != 64
                or any(character not in "0123456789abcdef"
                       for character in normalized)):
            raise ValueError(f"expected {name} SHA-256 is invalid")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite teacher cache: {args.output}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_window_keys(windows, manifest: list[dict]) -> list[list[object]]:
    keys = []
    for window in windows:
        trajectory = int(window.trajectory)
        if not 0 <= trajectory < len(manifest):
            raise ValueError("Sim window has an invalid trajectory index")
        start = int(window.start)
        row = manifest[trajectory]
        if start not in row.get("starts", []):
            raise ValueError("Sim window start is absent from the manifest")
        keys.append([str(row["file"]), start])
    return keys


def tensor_is_finite(value: torch.Tensor, chunk_size: int = 32) -> bool:
    return all(
        bool(torch.isfinite(value[start:start + chunk_size]).all())
        for start in range(0, value.shape[0], chunk_size))


def expected_prediction_shape(dataset) -> tuple[int, ...]:
    if not hasattr(dataset, "windows"):
        raise TypeError("Sim dataset must expose its ordered windows")
    if not len(dataset):
        raise ValueError("Sim dataset contains no windows")
    _history, target = dataset[0]
    if target.ndim != 4 or target.shape[-1] < 2:
        raise ValueError("Sim target must have shape (T,H,W,C>=2)")
    return (len(dataset), *target.shape[:-1], 2)


def validate_cache_payload(
        payload: dict, manifest: list[dict], dataset,
        ) -> tuple[torch.Tensor, dict]:
    """Validate cache identity against the current ordered Sim dataset."""
    if not isinstance(payload, dict):
        raise TypeError("teacher cache payload must be a mapping")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("teacher cache schema version mismatch")
    if payload.get("method") != CACHE_METHOD:
        raise ValueError("teacher cache method mismatch")
    if payload.get("composition") != COMPOSITION:
        raise ValueError("teacher cache composition mismatch")

    manifest_hash = canonical_sha256(manifest)
    if payload.get("split_manifest_sha256") != manifest_hash:
        raise ValueError("teacher cache split manifest hash mismatch")
    if payload.get("split_manifest") != manifest:
        raise ValueError("teacher cache split manifest mismatch")

    keys = ordered_window_keys(dataset.windows, manifest)
    if payload.get("window_keys") != keys:
        raise ValueError("teacher cache window order mismatch")
    if payload.get("window_order_sha256") != canonical_sha256(keys):
        raise ValueError("teacher cache window-order hash mismatch")

    identities = payload.get("teacher_checkpoints")
    if not isinstance(identities, dict) or set(identities) != {"a", "b", "c"}:
        raise ValueError("teacher cache checkpoint identities are incomplete")
    for name, identity in identities.items():
        actual = (
            identity.get("actual_sha256")
            if isinstance(identity, dict) else None)
        expected = (
            identity.get("expected_sha256")
            if isinstance(identity, dict) else None)
        digests = (actual, expected)
        if any(
                not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)
                for digest in digests):
            raise ValueError(f"teacher cache checkpoint {name} has invalid SHA-256")
        if actual != expected or identity.get("sha256_match") is not True:
            raise ValueError(f"teacher cache checkpoint {name} SHA-256 mismatch")
        if identity.get("phase") != "sim":
            raise ValueError(f"teacher cache checkpoint {name} is not Sim phase")

    expected_shape = expected_prediction_shape(dataset)
    if payload.get("prediction_shape") != list(expected_shape):
        raise ValueError("teacher cache recorded prediction shape mismatch")
    if payload.get("prediction_dtype") != "float16":
        raise ValueError("teacher cache recorded dtype must be float16")
    prediction = payload.get("prediction_uv_physical")
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("teacher cache prediction must be a tensor")
    if tuple(prediction.shape) != expected_shape:
        raise ValueError("teacher cache prediction tensor shape mismatch")
    if prediction.dtype != torch.float16:
        raise TypeError("teacher cache prediction tensor must be float16")
    if not tensor_is_finite(prediction):
        raise ValueError("teacher cache prediction contains non-finite values")

    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "method": CACHE_METHOD,
        "composition": COMPOSITION,
        "teacher_checkpoints": identities,
        "split_manifest_sha256": manifest_hash,
        "window_order_sha256": canonical_sha256(keys),
        "windows": len(keys),
        "prediction_shape": list(expected_shape),
        "prediction_dtype": "float16",
    }
    return prediction, metadata


def load_and_validate_cache(
        path: Path, manifest: list[dict], dataset,
        ) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prediction, metadata = validate_cache_payload(payload, manifest, dataset)
    metadata = {**metadata, "path": str(path.resolve()), "sha256": sha256(path)}
    return prediction, metadata


def checkpoint_identity(path: Path, expected_sha256: str) -> dict:
    expected = expected_sha256.lower()
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"teacher checkpoint SHA-256 mismatch for {path}: "
            f"actual={actual}, expected={expected}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "actual_sha256": actual,
        "expected_sha256": expected,
        "sha256_match": True,
    }


def validate_checkpoint_phase(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    phase = payload.get("phase")
    del payload
    if phase != "sim":
        raise ValueError(f"teacher checkpoint {path} must have phase='sim'")
    return phase


def load_teacher(path: Path, device: torch.device, dtype: torch.dtype):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("phase") != "sim":
        raise ValueError(f"teacher checkpoint {path} must have phase='sim'")
    config_dict = dict(payload.get("model_config", {}))
    architecture = config_dict.pop("architecture", None)
    if architecture != "ensemble_unet":
        raise ValueError(f"teacher checkpoint {path} is not ensemble_unet")
    stats = payload.get("normalizer")
    if not isinstance(stats, dict) or any(key not in stats for key in NORMALIZER_KEYS):
        raise ValueError(f"teacher checkpoint {path} has no complete normalizer")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**config_dict))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device=device, dtype=dtype).eval()
    tensors = {
        key: torch.as_tensor(stats[key], device=device, dtype=dtype)
        for key in NORMALIZER_KEYS
    }
    if any(value.shape != (3,) for value in tensors.values()):
        raise ValueError(f"teacher checkpoint {path} has invalid normalizer shape")
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError(f"teacher checkpoint {path} has non-finite normalizer")
    if bool((tensors["std_input"] <= 0).any()) or bool(
            (tensors["std_target"] <= 0).any()):
        raise ValueError(f"teacher checkpoint {path} has non-positive std")
    model.set_normalizer_statistics(
        tensors["mean_input"], tensors["mean_target"],
        tensors["std_input"], tensors["std_target"])
    return model, tensors


@torch.no_grad()
def collect_prediction(
        checkpoint: Path, loader: DataLoader,
        device: torch.device, dtype: torch.dtype,
        expected_shape: tuple[int, ...],
        ) -> torch.Tensor:
    model, stats = load_teacher(checkpoint, device, dtype)
    result = torch.empty(expected_shape, dtype=torch.float16)
    offset = 0
    for history, _target in loader:
        history = history.to(device=device, dtype=dtype, non_blocking=True)
        normalized = (
            (history - stats["mean_input"]) / stats["std_input"])
        prediction = model(normalized)
        physical = (
            prediction * stats["std_target"] + stats["mean_target"])
        batch = physical[..., :2].to(torch.float16).cpu()
        stop = offset + batch.shape[0]
        if tuple(batch.shape[1:]) != expected_shape[1:] or stop > len(result):
            raise ValueError(
                f"teacher checkpoint {checkpoint} produced an invalid shape")
        result[offset:stop].copy_(batch)
        offset = stop
    del model, stats
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if offset != len(result):
        raise ValueError(
            f"teacher checkpoint {checkpoint} produced {offset} windows, "
            f"expected {len(result)}")
    if not tensor_is_finite(result):
        raise ValueError(f"teacher checkpoint {checkpoint} produced non-finite values")
    return result


def validate_prediction_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.dtype != torch.float16 or right.dtype != torch.float16:
        raise TypeError("P1 source predictions must be float16")
    if left.shape != right.shape:
        raise ValueError("P1 source prediction shapes differ")


def blend_point_base_in_place(
        prediction_a: torch.Tensor, prediction_b: torch.Tensor,
        chunk_size: int = 16,
        ) -> torch.Tensor:
    """Replace A with current-P1 A.lerp(B, 0.5) using bounded buffers."""
    validate_prediction_pair(prediction_a, prediction_b)
    if chunk_size <= 0:
        raise ValueError("composition chunk size must be positive")
    for start in range(0, len(prediction_a), chunk_size):
        stop = min(start + chunk_size, len(prediction_a))
        base = prediction_a[start:stop].float()
        base.lerp_(prediction_b[start:stop].float(), POINT_WEIGHT_B)
        prediction_a[start:stop].copy_(base)
    return prediction_a


def add_p1_temporal_mean_correction_in_place(
        base_prediction: torch.Tensor, prediction_c: torch.Tensor,
        chunk_size: int = 16,
        ) -> torch.Tensor:
    """Add current P1's 0.5-weight C temporal-mean correction."""
    validate_prediction_pair(base_prediction, prediction_c)
    if chunk_size <= 0:
        raise ValueError("composition chunk size must be positive")
    for start in range(0, len(base_prediction), chunk_size):
        stop = min(start + chunk_size, len(base_prediction))
        base = base_prediction[start:stop].float()
        expert = prediction_c[start:stop].float()
        base.add_(
            expert.mean(dim=1, keepdim=True)
            - base.mean(dim=1, keepdim=True),
            alpha=TEMPORAL_MEAN_WEIGHT_C)
        base_prediction[start:stop].copy_(base)
    return base_prediction


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # Hash every source before any checkpoint payload is deserialized.
    identities = {
        name: checkpoint_identity(path, expected)
        for name, path, expected in zip(
            ("a", "b", "c"), args.checkpoints,
            args.expected_checkpoint_sha256)
    }
    # Reject any non-Sim source before data loading or teacher inference.
    for name, path in zip(("a", "b", "c"), args.checkpoints):
        identities[name]["phase"] = validate_checkpoint_phase(path)

    trajectories, windows, manifest = common.scratch.load_simulations(
        args.sim_dir, args.max_files, args.window_stride, True)
    dataset = common.scratch.SimWindowDataset(
        trajectories, windows, mask_probability=0.0, noise_scale=0.0)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    keys = ordered_window_keys(windows, manifest)
    expected_shape = expected_prediction_shape(dataset)

    print(json.dumps({
        "event": "cache_contract",
        "windows": len(windows),
        "prediction_shape": list(expected_shape),
        "split_manifest_sha256": canonical_sha256(manifest),
        "window_order_sha256": canonical_sha256(keys),
        "teacher_checkpoints": identities,
        "inference_dtype": str(dtype).removeprefix("torch."),
    }), flush=True)

    prediction_a = collect_prediction(
        args.checkpoints[0], loader, device, dtype, expected_shape)
    prediction_b = collect_prediction(
        args.checkpoints[1], loader, device, dtype, expected_shape)
    blend_point_base_in_place(
        prediction_a, prediction_b, chunk_size=args.batch_size)
    del prediction_b
    prediction_c = collect_prediction(
        args.checkpoints[2], loader, device, dtype, expected_shape)
    teacher = add_p1_temporal_mean_correction_in_place(
        prediction_a, prediction_c, chunk_size=args.batch_size).contiguous()
    del prediction_c
    if tuple(teacher.shape) != expected_shape or not tensor_is_finite(teacher):
        raise ValueError("composed teacher prediction violates the cache contract")

    args.output.parent.mkdir(parents=True, exist_ok=True)
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
        "split_manifest_sha256": canonical_sha256(manifest),
        "window_keys": keys,
        "window_order_sha256": canonical_sha256(keys),
        "prediction_shape": list(expected_shape),
        "prediction_dtype": "float16",
        "prediction_uv_physical": teacher,
    }
    # Validate the exact in-memory payload before it becomes an artifact.
    validate_cache_payload(payload, manifest, dataset)
    torch.save(payload, args.output)
    del payload, teacher
    # Validate the serialized artifact as well; this also rejects silent writes.
    _prediction, metadata = load_and_validate_cache(args.output, manifest, dataset)
    del _prediction
    print(json.dumps({"event": "complete", **metadata}), flush=True)


if __name__ == "__main__":
    main()
