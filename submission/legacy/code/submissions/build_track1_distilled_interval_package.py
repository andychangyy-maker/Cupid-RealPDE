#!/usr/bin/env python3
"""Build a compact Track 1 package that bypasses the quantile UQ decoder."""

from __future__ import annotations

import argparse
import hashlib
import math
import zipfile
from pathlib import Path

import torch

from build_track1_ensemble_unet2d_package import (
    compact_checkpoint as compact_base_checkpoint,
)
from build_track1_blended_ensemble_unet2d_package import compact_state_dict
from build_track1_unet2d_package import (
    compact_checkpoint as compact_unet_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
UNET_CONFIG_FIELDS = {
    "in_steps", "out_steps", "in_channels", "out_channels",
    "base", "depth", "groups", "difference_steps",
    "use_coordinates", "use_regime_film", "multi_scale_regime_film",
    "mean_fluctuation", "predict_uncertainty",
    "uncertainty_base_widths", "learn_temporal_spread",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--interval-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--fuse-point-heads", action=argparse.BooleanOptionalAction,
        default=False,
        help=("replace the linear ensemble mean with one exactly averaged "
              "UNet point head and drop all unused UQ weights"))
    parser.add_argument(
        "--int8-storage", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument("--quantize-min-elements", type=int, default=1024)
    parser.add_argument(
        "--interval-scale-uv", type=float, nargs=2, default=(1.0, 1.0),
        metavar=("U", "V"),
        help="positive deployment scale around the point prediction")
    parser.add_argument(
        "--wrapper", type=Path,
        default=Path(
            "code/submissions/track1_distilled_interval_head.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_head(
        source: Path, target: Path,
        interval_scale_uv: tuple[float, float]) -> dict:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload["model_config"].get(
            "architecture") != "distilled_interval_head":
        raise ValueError("checkpoint is not a distilled interval head")
    state = {
        key: value.to(torch.float16)
        if torch.is_floating_point(value) else value
        for key, value in payload["model_state_dict"].items()
    }
    torch.save({
        "phase": payload.get(
            "phase", "real_interval_distillation"),
        "iteration": payload.get("iteration"),
        "model_config": payload["model_config"],
        "model_state_dict": state,
        "teacher": payload.get("teacher", {}),
        "inference_dtype": "float16",
        "interval_scale_uv": list(interval_scale_uv),
    }, target)
    return {
        "parameters": sum(value.numel() for value in state.values()),
        "iteration": payload.get("iteration"),
        "interval_scale_uv": list(interval_scale_uv),
    }


def compact_fused_point_checkpoint(
        source: Path, target: Path, int8_storage: bool,
        quantize_min_elements: int) -> dict:
    """Convert an adapter-free ensemble into its exact single-head mean."""
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = dict(payload["model_config"])
    if config.get("architecture") != "ensemble_unet":
        raise ValueError("point-head fusion requires an ensemble_unet")
    if config.get("member_adapter_channels", 0):
        raise ValueError("cannot fuse point heads with member adapters")
    if config.get("mean_profile_channels", 0):
        raise ValueError("mean-profile refiner is not supported by fusion")
    members = int(config["ensemble_members"])
    base_fields = UNET_CONFIG_FIELDS
    fused_config = {
        "architecture": "unet",
        **{
            key: value for key, value in config.items()
            if key in base_fields
        },
        "predict_uncertainty": False,
    }
    source_state = payload["model_state_dict"]
    fused_state = {}
    dropped_prefixes = (
        "member_point_heads.", "member_feature_adapters.",
        "mean_profile_refiner.", "uq_projections.", "uq_fusion.",
        "quantile_head.")
    for key, value in source_state.items():
        if key.startswith(dropped_prefixes):
            continue
        fused_state[key] = value
    decoder_levels = int(config["depth"]) - 1
    for level in range(decoder_levels):
        for suffix in ("weight", "bias"):
            values = [
                source_state[
                    f"member_point_heads.{member}.{level}.{suffix}"]
                for member in range(members)
            ]
            fused_state[f"point_heads.{level}.{suffix}"] = torch.stack(
                [value.float() for value in values]).mean(dim=0)
    parameters = sum(value.numel() for value in fused_state.values())
    if int8_storage:
        stored_state, quantization = compact_state_dict(
            fused_state, quantize_min_elements)
        state_payload = {"quantized_state_dict": stored_state}
        storage_format = "symmetric_per_dim0_int8_v1"
    else:
        stored_state = {
            key: value.to(torch.float16)
            if torch.is_floating_point(value) else value
            for key, value in fused_state.items()
        }
        quantization = {
            "quantized_tensors": 0, "quantized_elements": 0}
        state_payload = {"model_state_dict": stored_state}
        storage_format = "float16"
    torch.save({
        "phase": payload.get("phase"),
        "iteration": payload.get("iteration"),
        "model_config": fused_config,
        **state_payload,
        "normalizer": payload.get("normalizer", {}),
        "inference_dtype": "float16",
        "storage_format": storage_format,
        "point_head_fusion": {
            "source_architecture": "ensemble_unet",
            "members": members,
            "operation": "arithmetic mean of linear point-head weights",
        },
    }, target)
    return {
        "parameters": parameters,
        "members_fused": members,
        "storage_format": storage_format,
        **quantization,
    }



def main() -> None:
    args = parse_args()
    base = args.base_checkpoint.resolve()
    interval_scale_uv = tuple(float(value) for value in args.interval_scale_uv)
    if any(not math.isfinite(value) or value <= 0
           for value in interval_scale_uv):
        raise ValueError("interval scales must be finite and positive")
    if args.quantize_min_elements <= 0:
        raise ValueError("quantize-min-elements must be positive")
    if args.int8_storage and not args.fuse_point_heads:
        raise ValueError(
            "int8 storage currently requires --fuse-point-heads")
    interval = args.interval_checkpoint.resolve()
    wrapper = args.wrapper.resolve()
    model_init = REPO_ROOT / "code/submissions/ensemble_models_init.py"
    model_base = REPO_ROOT / "code/models/unet2d.py"
    model_ensemble = REPO_ROOT / "code/models/ensemble_unet2d.py"
    model_corrector = (
        REPO_ROOT / "code/models/distilled_flow_corrector.py")
    model_interval = (
        REPO_ROOT / "code/models/distilled_interval_head.py")
    output = args.output.resolve()
    inputs = (
        base, interval, wrapper, model_init, model_base,
        model_ensemble, model_corrector, model_interval)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    compact_base = output.with_suffix(output.suffix + ".base")
    compact_interval = output.with_suffix(output.suffix + ".interval")
    for path in (temporary, compact_base, compact_interval):
        if path.exists():
            path.unlink()
    try:
        if args.fuse_point_heads:
            base_metadata = compact_fused_point_checkpoint(
                base, compact_base, args.int8_storage,
                args.quantize_min_elements)
        else:
            base_payload = torch.load(
                base, map_location="cpu", weights_only=False)
            architecture = base_payload["model_config"]["architecture"]
            if architecture == "ensemble_unet":
                compact_base_checkpoint(base, compact_base, "float16")
            elif architecture == "unet":
                compact_unet_checkpoint(base, compact_base, "float16")
            else:
                raise ValueError(
                    f"unsupported base architecture: {architecture}")
            base_metadata = {"architecture": architecture}
        head_metadata = compact_head(
            interval, compact_interval, interval_scale_uv)
        entries = (
            (wrapper, "submission.py"),
            (model_init, "models/__init__.py"),
            (model_base, "models/unet2d.py"),
            (model_ensemble, "models/ensemble_unet2d.py"),
            (model_corrector, "models/distilled_flow_corrector.py"),
            (model_interval, "models/distilled_interval_head.py"),
            (compact_base, "base_model.pth"),
            (compact_interval, "interval_head.pth"),
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
        for path in (temporary, compact_base, compact_interval):
            if path.exists():
                path.unlink()

    expected = [
        "submission.py",
        "models/__init__.py",
        "models/unet2d.py",
        "models/ensemble_unet2d.py",
        "models/distilled_flow_corrector.py",
        "models/distilled_interval_head.py",
        "base_model.pth",
        "interval_head.pth",
    ]
    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"corrupt archive member: {bad_member}")
    if members != expected:
        raise RuntimeError(f"unexpected archive layout: {members}")
    print({
        "archive": str(output),
        "size_mib": output.stat().st_size / 1024**2,
        "sha256": sha256(output),
        "base_sha256": sha256(base),
        "interval_sha256": sha256(interval),
        "interval_head": head_metadata,
        "base": base_metadata,
        "members": members,
    })


if __name__ == "__main__":
    main()
