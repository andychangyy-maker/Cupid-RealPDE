#!/usr/bin/env python3
"""Build a sub-256 MiB Track 1 archive containing two blended U-Nets.

Weights are stored with symmetric per-output-channel int8 quantization, then
dequantized to FP16 on CUDA when the submission process loads.  This is a
storage optimization only: convolution inference remains ordinary PyTorch
FP16 rather than integer arithmetic.
"""

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
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--alpha-b", type=float, default=0.5)
    parser.add_argument(
        "--wrapper", type=Path,
        default=Path(
            "code/submissions/track1_blended_ensemble_unet2d.py"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "runs/submissions/track1_v9_blended_ensemble_unet2d_int8.zip"))
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    parser.add_argument("--quantize-min-elements", type=int, default=1024)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize_tensor(value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Symmetrically quantize one tensor along its first dimension."""
    value_float = value.detach().cpu().float()
    reduction_dims = tuple(range(1, value_float.ndim))
    scale = value_float.abs().amax(dim=reduction_dims, keepdim=True) / 127.0
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    quantized = torch.round(value_float / scale).clamp(-127, 127).to(torch.int8)
    return {"quantized": quantized, "scale": scale}


def compact_state_dict(
        state: dict[str, torch.Tensor], min_elements: int
        ) -> tuple[dict[str, object], dict[str, int]]:
    compact: dict[str, object] = {}
    quantized_tensors = 0
    quantized_elements = 0
    for key, value in state.items():
        value = value.detach().cpu()
        if (torch.is_floating_point(value) and value.ndim >= 2
                and value.numel() >= min_elements):
            compact[key] = quantize_tensor(value)
            quantized_tensors += 1
            quantized_elements += value.numel()
        elif torch.is_floating_point(value):
            compact[key] = value.to(torch.float16)
        else:
            compact[key] = value
    return compact, {
        "quantized_tensors": quantized_tensors,
        "quantized_elements": quantized_elements,
    }


def compact_checkpoint(
        source: Path, target: Path, min_elements: int) -> dict[str, int]:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    if config.get("architecture") != "ensemble_unet":
        raise ValueError(f"{source} is not an ensemble_unet checkpoint")
    state, statistics = compact_state_dict(
        payload["model_state_dict"], min_elements)
    compact = {
        "phase": payload.get("phase", "real"),
        "model_config": config,
        "quantized_state_dict": state,
        "normalizer": payload.get("normalizer", {}),
        "inference_dtype": "float16",
        "storage_format": "symmetric_per_dim0_int8_v1",
    }
    torch.save(compact, target)
    return statistics


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha_b <= 1.0:
        raise ValueError("alpha-b must lie in [0,1]")
    if args.quantize_min_elements <= 0:
        raise ValueError("quantize-min-elements must be positive")

    checkpoint_a = args.checkpoint_a.resolve()
    checkpoint_b = args.checkpoint_b.resolve()
    wrapper = args.wrapper.resolve()
    model_init = REPO_ROOT / "code/submissions/ensemble_models_init.py"
    model_base = REPO_ROOT / "code/models/unet2d.py"
    model_ensemble = REPO_ROOT / "code/models/ensemble_unet2d.py"
    output = args.output.resolve()
    inputs = (
        checkpoint_a, checkpoint_b, wrapper, model_init,
        model_base, model_ensemble)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    compact_a = output.with_suffix(output.suffix + ".model_a")
    compact_b = output.with_suffix(output.suffix + ".model_b")
    blend_config = output.with_suffix(output.suffix + ".blend.json")
    for path in (temporary, compact_a, compact_b, blend_config):
        if path.exists():
            path.unlink()
    try:
        statistics_a = compact_checkpoint(
            checkpoint_a, compact_a, args.quantize_min_elements)
        statistics_b = compact_checkpoint(
            checkpoint_b, compact_b, args.quantize_min_elements)
        blend_config.write_text(json.dumps({
            "alpha_b": args.alpha_b,
            "storage": "int8 weights dequantized to FP16 on CUDA",
        }, indent=2) + "\n", encoding="utf-8")
        entries = (
            (wrapper, "submission.py"),
            (model_init, "models/__init__.py"),
            (model_base, "models/unet2d.py"),
            (model_ensemble, "models/ensemble_unet2d.py"),
            (compact_a, "model_a.pth"),
            (compact_b, "model_b.pth"),
            (blend_config, "blend_config.json"),
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
        for path in (temporary, compact_a, compact_b, blend_config):
            if path.exists():
                path.unlink()

    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"corrupt archive member: {bad_member}")
    expected = [
        "submission.py", "models/__init__.py", "models/unet2d.py",
        "models/ensemble_unet2d.py", "model_a.pth", "model_b.pth",
        "blend_config.json",
    ]
    if members != expected:
        raise RuntimeError(f"unexpected archive layout: {members}")
    print({
        "archive": str(output),
        "alpha_b": args.alpha_b,
        "size_mib": output.stat().st_size / 1024**2,
        "sha256": sha256(output),
        "source_a_sha256": sha256(checkpoint_a),
        "source_b_sha256": sha256(checkpoint_b),
        "quantization_a": statistics_a,
        "quantization_b": statistics_b,
        "members": members,
    })


if __name__ == "__main__":
    main()
