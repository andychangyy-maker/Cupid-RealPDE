#!/usr/bin/env python3
"""Build and integrity-check a compact Track 1 HybridFNO2d archive."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--wrapper", type=Path,
        default=Path("code/submissions/track1_hybrid_fno2d_native.py"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/submissions/track1_v9_hybrid_fno2d_native.zip"))
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    wrapper = args.wrapper.resolve()
    model_source = REPO_ROOT / "code/models/hybrid_fno2d.py"
    output = args.output.resolve()
    missing = [str(path) for path in (checkpoint, wrapper, model_source)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    entries = (
        (wrapper, "submission.py"),
        (model_source, "hybrid_fno2d.py"),
        (checkpoint, "hybrid_fno2d.pth"),
    )
    try:
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
    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"corrupt archive member: {bad_member}")
    expected = [name for _source, name in entries]
    if members != expected:
        raise RuntimeError(f"unexpected archive layout: {members}")
    print({
        "archive": str(output),
        "size_mib": output.stat().st_size / 1024**2,
        "sha256": sha256(output),
        "checkpoint_sha256": sha256(checkpoint),
        "members": members,
    })


if __name__ == "__main__":
    main()
