#!/usr/bin/env python3
"""Build a sub-256 MiB Track 1 archive containing three blended U-Nets."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

from build_track1_blended_ensemble_unet2d_package import (
    compact_checkpoint,
    sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", type=Path, nargs=3, required=True,
        metavar=("A", "B", "C"))
    parser.add_argument(
        "--weights", type=float, nargs=3, required=True,
        metavar=("A", "B", "C"))
    parser.add_argument(
        "--wrapper", type=Path,
        default=Path("code/submissions/track1_triple_ensemble_unet2d.py"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "runs/submissions/track1_v9_triple_ensemble_unet2d_int8.zip"))
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    parser.add_argument("--quantize-min-elements", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = tuple(float(value) for value in args.weights)
    if (any(not math.isfinite(value) or value < 0.0 for value in weights)
            or not math.isclose(sum(weights), 1.0, abs_tol=1e-8)):
        raise ValueError("weights must be finite/non-negative and sum to one")
    if args.quantize_min_elements <= 0:
        raise ValueError("quantize-min-elements must be positive")

    checkpoints = tuple(path.resolve() for path in args.checkpoints)
    wrapper = args.wrapper.resolve()
    model_init = REPO_ROOT / "code/submissions/ensemble_models_init.py"
    model_base = REPO_ROOT / "code/models/unet2d.py"
    model_ensemble = REPO_ROOT / "code/models/ensemble_unet2d.py"
    output = args.output.resolve()
    inputs = (*checkpoints, wrapper, model_init, model_base, model_ensemble)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    compact_paths = tuple(
        output.with_suffix(output.suffix + f".model_{index}")
        for index in range(3))
    blend_config = output.with_suffix(output.suffix + ".blend.json")
    for path in (temporary, *compact_paths, blend_config):
        if path.exists():
            path.unlink()
    try:
        quantization = [
            compact_checkpoint(source, target, args.quantize_min_elements)
            for source, target in zip(checkpoints, compact_paths)
        ]
        blend_config.write_text(json.dumps({
            "weights": weights,
            "checkpoints": ["model_a.pth", "model_b.pth", "model_c.pth"],
            "storage": "int8 weights dequantized to FP16 on CUDA",
        }, indent=2) + "\n", encoding="utf-8")
        entries = (
            (wrapper, "submission.py"),
            (model_init, "models/__init__.py"),
            (model_base, "models/unet2d.py"),
            (model_ensemble, "models/ensemble_unet2d.py"),
            (compact_paths[0], "model_a.pth"),
            (compact_paths[1], "model_b.pth"),
            (compact_paths[2], "model_c.pth"),
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
        for path in (temporary, *compact_paths, blend_config):
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
        "model_c.pth", "blend_config.json",
    ]
    if members != expected:
        raise RuntimeError(f"unexpected archive layout: {members}")
    print({
        "archive": str(output),
        "weights": weights,
        "size_mib": output.stat().st_size / 1024**2,
        "sha256": sha256(output),
        "source_sha256": [sha256(path) for path in checkpoints],
        "quantization": quantization,
        "members": members,
    })


if __name__ == "__main__":
    main()
