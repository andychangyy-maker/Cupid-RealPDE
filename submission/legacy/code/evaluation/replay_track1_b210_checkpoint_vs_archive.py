#!/usr/bin/env python3
"""Replay one dense B210 checkpoint against its packaged INT8/FP16 ZIP.

This is a deployment-fidelity check, not a model-selection score.  It runs the
dense source and the freshly extracted submission on one identical input
stream and records hashes plus numerical deltas caused by FP16/INT8 storage.
No pass threshold is imposed on those deltas; finite outputs, ordered bounds,
input immutability, pressure-zero behavior, and deterministic ZIP replay are
hard requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Mapping
import zipfile

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_ROOT = CODE_ROOT / "submissions"
for path in (CODE_ROOT, SUBMISSIONS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)
import build_track1_b210_full_oof_native_uq_package as package_builder  # noqa: E402


SCHEMA = "track1_b210_checkpoint_vs_archive_replay_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-npy", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=210)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        if tuple(handle.namelist()) != package_builder.EXPECTED_MEMBERS:
            raise RuntimeError("archive member layout changed")
        if handle.testzip() is not None:
            raise RuntimeError("archive is corrupt")
        for name in handle.namelist():
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe archive member: {name}")
        handle.extractall(destination)


def _purge_packaged_modules() -> None:
    for name in tuple(sys.modules):
        if name == "models" or name.startswith("models.") or name == "b210_zip":
            sys.modules.pop(name, None)


def _load_submission(package_dir: Path):
    _purge_packaged_modules()
    sys.path.insert(0, str(package_dir))
    spec = importlib.util.spec_from_file_location(
        "b210_zip", package_dir / "submission.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load extracted submission.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["b210_zip"] = module
    spec.loader.exec_module(module)
    return module


def _input(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    if args.input_npy is not None:
        value = np.load(args.input_npy, allow_pickle=False)
        source = {
            "kind": "npy",
            "path": str(args.input_npy.resolve()),
            "sha256": package_builder.sha256(args.input_npy),
        }
    else:
        if args.samples <= 0:
            raise ValueError("samples must be positive")
        generator = np.random.default_rng(args.seed)
        value = generator.normal(
            loc=0.0, scale=0.05,
            size=(args.samples, 20, 32, 64, 3),
        ).astype(np.float32)
        value[..., 0] += 0.15
        value[..., 2] = 0.0
        source = {"kind": "deterministic_synthetic", "seed": args.seed}
    value = np.asarray(value)
    if (value.dtype != np.float32 or value.ndim != 5
            or value.shape[1:] != (20, 32, 64, 3)
            or not np.isfinite(value).all()
            or np.count_nonzero(value[..., 2])):
        raise ValueError("replay input must be finite float32 (N,20,32,64,3) with p=0")
    return np.ascontiguousarray(value), {
        **source,
        "shape": list(value.shape),
        "array_sha256": _hash_array(value),
    }


def _validate_outputs(result: Mapping[str, object], shape: tuple[int, ...]) -> dict:
    if not isinstance(result, Mapping) or set(result) != {
            "prediction", "lower", "upper"}:
        raise TypeError("prediction result must contain point/lower/upper")
    arrays = {
        key: np.asarray(result[key])
        for key in ("prediction", "lower", "upper")
    }
    if any(array.shape != shape or array.dtype != np.float32
           or not np.isfinite(array).all() for array in arrays.values()):
        raise FloatingPointError("prediction output shape/dtype/finiteness changed")
    if np.any(arrays["lower"] > arrays["upper"]):
        raise FloatingPointError("prediction bounds are unordered")
    if any(np.count_nonzero(array[..., 2]) for array in arrays.values()):
        raise RuntimeError("prediction output pressure is not exactly zero")
    return arrays


def _source_predict(
    model: EnsembleUNet2d, raw_input: np.ndarray,
    stats: Mapping[str, torch.Tensor], device: torch.device,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    predictions, lowers, uppers = [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(raw_input), 16):
            raw = torch.from_numpy(np.ascontiguousarray(
                raw_input[start:start + 16])).to(device=device, dtype=dtype)
            normalized = (raw - stats["mean_input"]) / stats["std_input"]
            point_norm, lower_norm, upper_norm = model.forward_distribution(
                normalized)
            prediction = (
                point_norm * stats["std_target"] + stats["mean_target"])
            prediction[..., 2] = 0.0
            lower = prediction.clone()
            upper = prediction.clone()
            lower[..., :2] = (
                prediction[..., :2]
                - lower_norm * stats["std_target"][:2])
            upper[..., :2] = (
                prediction[..., :2]
                + upper_norm * stats["std_target"][:2])
            bundle = torch.stack((prediction, lower, upper)).float().cpu().numpy()
            predictions.append(bundle[0])
            lowers.append(bundle[1])
            uppers.append(bundle[2])
    return {
        "prediction": np.concatenate(predictions).astype(np.float32, copy=False),
        "lower": np.concatenate(lowers).astype(np.float32, copy=False),
        "upper": np.concatenate(uppers).astype(np.float32, copy=False),
    }


def _delta(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = left.astype(np.float64) - right.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(np.square(difference)))),
    }


def run(args: argparse.Namespace) -> dict:
    checkpoint_sha = package_builder.normalized_sha256(
        args.checkpoint_sha256, "checkpoint SHA-256")
    archive_sha = package_builder.normalized_sha256(
        args.archive_sha256, "archive SHA-256")
    if (not args.checkpoint.is_file()
            or package_builder.sha256(args.checkpoint) != checkpoint_sha):
        raise RuntimeError("dense checkpoint SHA-256 changed")
    if (not args.archive.is_file()
            or package_builder.sha256(args.archive) != archive_sha):
        raise RuntimeError("archive SHA-256 changed")
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA replay requested but CUDA is unavailable")
    raw_input, input_report = _input(args)
    before = raw_input.copy()

    dense = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = dense.get("model_state_dict")
    config_values = dict(dense["model_config"])
    if config_values.pop("architecture", "ensemble_unet") != "ensemble_unet":
        raise ValueError("dense checkpoint is not ensemble_unet")
    if not isinstance(state, Mapping):
        raise TypeError("dense checkpoint has no tensor state")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    source_model = EnsembleUNet2d(
        EnsembleUNet2dConfig(**config_values)).to(device=device, dtype=dtype)
    source_model.load_state_dict(state, strict=True)
    stats = {
        key: torch.as_tensor(dense["normalizer"][key], device=device, dtype=dtype)
        for key in ("mean_input", "mean_target", "std_input", "std_target")
    }
    source = _validate_outputs(
        _source_predict(source_model, raw_input, stats, device, dtype),
        tuple(raw_input.shape))
    if not np.array_equal(raw_input, before):
        raise RuntimeError("dense source replay mutated input")

    with tempfile.TemporaryDirectory(prefix="b210-source-zip-replay-") as temporary:
        package_dir = Path(temporary)
        _safe_extract(args.archive, package_dir)
        package_path = str(package_dir)
        try:
            submission = _load_submission(package_dir)
            packaged_device = submission._get_model()["device"]
            if torch.device(packaged_device).type != device.type:
                raise RuntimeError(
                    f"package selected {packaged_device}, requested {device}")
            first = _validate_outputs(
                submission.predict(raw_input, metadata={}), tuple(raw_input.shape))
            second = _validate_outputs(
                submission.predict(raw_input, metadata=None), tuple(raw_input.shape))
            deterministic = all(
                np.array_equal(first[key], second[key]) for key in first)
            if not deterministic:
                raise RuntimeError("packaged replay is not deterministic")
        finally:
            while package_path in sys.path:
                sys.path.remove(package_path)
            _purge_packaged_modules()
    if not np.array_equal(raw_input, before):
        raise RuntimeError("packaged replay mutated input")

    result = {
        "schema": SCHEMA,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "dense_model_state_sha256": package_builder.model_state_sha256(state),
        },
        "archive": {
            "path": str(args.archive.resolve()),
            "sha256": archive_sha,
            "bytes": args.archive.stat().st_size,
        },
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        "runtime_dtype": str(dtype),
        "batch_size": 16,
        "input": input_report,
        "input_bitwise_unchanged": True,
        "zip_deterministic": True,
        "outputs": {
            key: {
                "dense_sha256": _hash_array(source[key]),
                "zip_sha256": _hash_array(first[key]),
                "dense_vs_zip": _delta(source[key], first[key]),
            }
            for key in ("prediction", "lower", "upper")
        },
        "bounds_ordered": True,
        "pressure_exactly_zero": True,
        "interpretation": (
            "Numerical deltas are descriptive INT8-storage/FP16 effects, not "
            "a model-selection gate."
        ),
        "official_submission_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    temporary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_output.replace(args.output)
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
