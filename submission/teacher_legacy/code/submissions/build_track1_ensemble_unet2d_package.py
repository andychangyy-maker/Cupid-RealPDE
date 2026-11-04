#!/usr/bin/env python3
"""Build a compact Track 1 ensemble U-Net archive."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import torch

from build_track1_blended_ensemble_unet2d_package import compact_state_dict


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--wrapper", type=Path,
        default=Path("code/submissions/track1_ensemble_unet2d.py"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "runs/submissions/track1_v9_ensemble_unet2d_fp16.zip"))
    parser.add_argument(
        "--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--storage", choices=("dense", "int8"), default="dense")
    parser.add_argument("--quantize-min-elements", type=int, default=1024)
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_checkpoint(
        source: Path, target: Path, dtype_name: str,
        storage: str = "dense", min_elements: int = 1024,
        ) -> dict[str, int]:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    if config.get("architecture") != "ensemble_unet":
        raise ValueError("checkpoint architecture must be ensemble_unet")
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    if storage == "int8":
        state, statistics = compact_state_dict(
            payload["model_state_dict"], min_elements)
        state_payload = {"quantized_state_dict": state}
    else:
        state = {
            key: value.to(dtype=dtype)
            if torch.is_floating_point(value) else value
            for key, value in payload["model_state_dict"].items()
        }
        state_payload = {"model_state_dict": state}
        statistics = {
            "quantized_tensors": 0,
            "quantized_elements": 0,
        }
    compact = {
        "phase": payload.get("phase", "real"),
        "model_config": config,
        **state_payload,
        "normalizer": payload.get("normalizer", {}),
        "inference_dtype": dtype_name,
    }
    if storage == "int8":
        compact["storage_format"] = "symmetric_per_dim0_int8_v1"
    torch.save(compact, target)
    return statistics


def main() -> None:
    args = parse_args()
    if args.quantize_min_elements <= 0:
        raise ValueError("quantize-min-elements must be positive")
    checkpoint = args.checkpoint.resolve()
    wrapper = args.wrapper.resolve()
    model_init = REPO_ROOT / "code/submissions/ensemble_models_init.py"
    model_base = REPO_ROOT / "code/models/unet2d.py"
    model_ensemble = REPO_ROOT / "code/models/ensemble_unet2d.py"
    output = args.output.resolve()
    inputs = (checkpoint, wrapper, model_init, model_base, model_ensemble)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    compact = output.with_suffix(output.suffix + ".checkpoint")
    if temporary.exists():
        temporary.unlink()
    if compact.exists():
        compact.unlink()
    try:
        quantization = compact_checkpoint(
            checkpoint, compact, args.dtype, args.storage,
            args.quantize_min_elements)
        entries = (
            (wrapper, "submission.py"),
            (model_init, "models/__init__.py"),
            (model_base, "models/unet2d.py"),
            (model_ensemble, "models/ensemble_unet2d.py"),
            (compact, "ensemble_unet2d.pth"),
        )
        with zipfile.ZipFile(
                temporary, "w", zipfile.ZIP_DEFLATED,
                compresslevel=6, allowZip64=True) as archive:
            for source, name in entries:
                archive.write(source, name)
        size = temporary.stat().st_size
        if size > args.max_size_mib * 1024**2:
            raise RuntimeError(
                f"archive is {size / 1024**2:.2f} MiB, above limit")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
        if compact.exists():
            compact.unlink()

    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"corrupt archive member: {bad_member}")
    expected = [
        "submission.py",
        "models/__init__.py",
        "models/unet2d.py",
        "models/ensemble_unet2d.py",
        "ensemble_unet2d.pth",
    ]
    if members != expected:
        raise RuntimeError(f"unexpected archive layout: {members}")
    print({
        "archive": str(output),
        "dtype": args.dtype,
        "storage": args.storage,
        "size_mib": output.stat().st_size / 1024**2,
        "sha256": sha256(output),
        "source_checkpoint_sha256": sha256(checkpoint),
        "quantization": quantization,
        "members": members,
    })


if __name__ == "__main__":
    main()
