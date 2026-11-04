#!/usr/bin/env python3
"""Build a compact one-backbone Track 1 distilled-corrector archive."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import torch

from build_track1_ensemble_unet2d_package import (
    compact_checkpoint as compact_base_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--corrector-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--wrapper", type=Path,
        default=Path(
            "code/submissions/track1_distilled_flow_corrector.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_corrector(source: Path, target: Path) -> dict:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    if config.get("architecture") != "distilled_flow_corrector":
        raise ValueError("corrector checkpoint has an invalid architecture")
    state = {
        key: value.to(torch.float16)
        if torch.is_floating_point(value) else value
        for key, value in payload["model_state_dict"].items()
    }
    compact = {
        "phase": payload.get("phase", "real_distillation"),
        "iteration": payload.get("iteration"),
        "model_config": config,
        "model_state_dict": state,
        "teacher": payload.get("teacher", {}),
        "inference_dtype": "float16",
    }
    torch.save(compact, target)
    return {
        "parameters": sum(value.numel() for value in state.values()),
        "iteration": payload.get("iteration"),
    }


def main() -> None:
    args = parse_args()
    base = args.base_checkpoint.resolve()
    corrector = args.corrector_checkpoint.resolve()
    wrapper = args.wrapper.resolve()
    model_init = REPO_ROOT / "code/submissions/ensemble_models_init.py"
    model_base = REPO_ROOT / "code/models/unet2d.py"
    model_ensemble = REPO_ROOT / "code/models/ensemble_unet2d.py"
    model_corrector = (
        REPO_ROOT / "code/models/distilled_flow_corrector.py")
    output = args.output.resolve()
    inputs = (
        base, corrector, wrapper, model_init,
        model_base, model_ensemble, model_corrector)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    compact_base = output.with_suffix(output.suffix + ".base")
    compact_head = output.with_suffix(output.suffix + ".corrector")
    for path in (temporary, compact_base, compact_head):
        if path.exists():
            path.unlink()
    try:
        compact_base_checkpoint(base, compact_base, "float16")
        head_metadata = compact_corrector(
            corrector, compact_head)
        entries = (
            (wrapper, "submission.py"),
            (model_init, "models/__init__.py"),
            (model_base, "models/unet2d.py"),
            (model_ensemble, "models/ensemble_unet2d.py"),
            (model_corrector, "models/distilled_flow_corrector.py"),
            (compact_base, "base_model.pth"),
            (compact_head, "corrector.pth"),
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
        for path in (temporary, compact_base, compact_head):
            if path.exists():
                path.unlink()

    expected = [
        "submission.py",
        "models/__init__.py",
        "models/unet2d.py",
        "models/ensemble_unet2d.py",
        "models/distilled_flow_corrector.py",
        "base_model.pth",
        "corrector.pth",
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
        "corrector_sha256": sha256(corrector),
        "corrector": head_metadata,
        "members": members,
    })


if __name__ == "__main__":
    main()
