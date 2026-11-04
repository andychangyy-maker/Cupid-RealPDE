#!/usr/bin/env python3
"""Build a compact FP16/FP32 Track 1 U-Net submission archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--wrapper", type=Path,
        default=Path("code/submissions/track1_unet2d_native.py"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/submissions/track1_v9_unet2d_native_fp16.zip"))
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--calibration-json", type=Path,
        help="Optional audit_unet_native_intervals.py summary.")
    parser.add_argument(
        "--calibration-scale", type=float,
        help="Residual-scale key to take from --calibration-json.")
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibration(path: Path | None, scale: float | None) -> dict | None:
    if (path is None) != (scale is None):
        raise ValueError(
            "--calibration-json and --calibration-scale must be supplied together")
    if path is None:
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    key = f"{float(scale):.2f}"
    if key not in report.get("calibrations", {}):
        raise KeyError(f"calibration report has no residual-scale key {key}")
    values = torch.as_tensor(report["calibrations"][key], dtype=torch.float32)
    if values.shape != (20, 2, 2):
        raise ValueError(f"unexpected calibration shape {tuple(values.shape)}")
    if not torch.isfinite(values).all() or torch.any(values <= 0):
        raise ValueError("calibration values must be finite and positive")
    return {
        "method": "horizon_channel_residual_stress",
        "residual_scale": float(scale),
        "lower_multiplier_uv": values[..., 0].tolist(),
        "upper_multiplier_uv": values[..., 1].tolist(),
    }


def compact_checkpoint(source: Path, target: Path, dtype_name: str,
                       calibration: dict | None = None) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    state = {
        key: value.to(dtype=dtype) if torch.is_floating_point(value) else value
        for key, value in payload["model_state_dict"].items()
    }
    compact = {
        "phase": payload.get("phase", "real"),
        "model_config": payload["model_config"],
        "model_state_dict": state,
        "inference_dtype": dtype_name,
    }
    if calibration is not None:
        compact["interval_calibration"] = calibration
    torch.save(compact, target)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    wrapper = args.wrapper.resolve()
    model_source = REPO_ROOT / "code/models/unet2d.py"
    output = args.output.resolve()
    missing = [str(path) for path in (checkpoint, wrapper, model_source)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")
    if args.calibration_json is not None and not args.calibration_json.is_file():
        raise FileNotFoundError(args.calibration_json)
    calibration = load_calibration(
        args.calibration_json, args.calibration_scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    compact = output.with_suffix(output.suffix + ".checkpoint")
    if temporary.exists():
        temporary.unlink()
    if compact.exists():
        compact.unlink()
    try:
        compact_checkpoint(checkpoint, compact, args.dtype, calibration)
        entries = (
            (wrapper, "submission.py"),
            (model_source, "unet2d.py"),
            (compact, "unet2d.pth"),
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
    expected = ["submission.py", "unet2d.py", "unet2d.pth"]
    if members != expected:
        raise RuntimeError(f"unexpected archive layout: {members}")
    print({
        "archive": str(output),
        "dtype": args.dtype,
        "size_mib": output.stat().st_size / 1024**2,
        "sha256": sha256(output),
        "source_checkpoint_sha256": sha256(checkpoint),
        "interval_calibration": calibration,
        "members": members,
    })


if __name__ == "__main__":
    main()
