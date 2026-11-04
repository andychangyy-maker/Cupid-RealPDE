#!/usr/bin/env python3
"""Build the frozen zero-training S2b fused-mean FTQ latency probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_H96_SHA256 = (
    "45e0dd383852244c3c1cf6a3b817908dca449f79101794d5ab3a4e7b045d347d")
H96_MEMBERS = {
    "models/__init__.py": (
        "ca44e67d82b6d00c74b8043fd918b91fb17e186b133e45361155e3a949ca3187"),
    "models/unet2d.py": (
        "1dfd903ebba16b887993ca8867dadebc8d3276354747ff5de636406fee42017e"),
    "models/ensemble_unet2d.py": (
        "54a00c8e48dacf3010c8f5c729239bf1d149eb067819f4df8bb56b2175530b82"),
    "ensemble_unet2d.pth": (
        "f5d9f10ad9c764cc3f181a4f0b788b53ed15ba436f1d190bddda7152dbedd28b"),
}
EXPECTED_H96_LAYOUT = [
    "submission.py",
    "models/__init__.py",
    "models/unet2d.py",
    "models/ensemble_unet2d.py",
    "ensemble_unet2d.pth",
]
OUTPUT_LAYOUT = [
    "submission.py",
    "models/__init__.py",
    "models/unet2d.py",
    "models/ensemble_unet2d.py",
    "models/fast_temporal_quantile_head.py",
    "models/distilled_flow_corrector.py",
    "ensemble_unet2d.pth",
]
FIXED_TIMESTAMP = (2026, 8, 23, 0, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h96-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIMESTAMP)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def copy_member(source: zipfile.ZipFile, target: zipfile.ZipFile,
                name: str) -> None:
    with source.open(name) as input_handle:
        with target.open(zip_info(name), "w", force_zip64=True) as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)


def copy_file(source: Path, target: zipfile.ZipFile, name: str) -> None:
    with source.open("rb") as input_handle:
        with target.open(zip_info(name), "w", force_zip64=True) as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)


def main() -> None:
    args = parse_args()
    h96 = args.h96_archive.resolve()
    output = args.output.resolve()
    wrapper = REPO_ROOT / "code/evaluation/s2_ftq_latency_submission.py"
    ftq = REPO_ROOT / "code/models/fast_temporal_quantile_head.py"
    flow = REPO_ROOT / "code/models/distilled_flow_corrector.py"
    inputs = (h96, wrapper, ftq, flow)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if file_sha256(h96) != EXPECTED_H96_SHA256:
        raise ValueError("H96 archive SHA-256 differs from the frozen contract")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(h96) as source:
            if source.namelist() != EXPECTED_H96_LAYOUT:
                raise ValueError("H96 archive layout differs from the contract")
            for name, expected in H96_MEMBERS.items():
                if member_sha256(source, name) != expected:
                    raise ValueError(f"H96 member differs from contract: {name}")
            with zipfile.ZipFile(
                    temporary, "w", compression=zipfile.ZIP_DEFLATED,
                    compresslevel=6, allowZip64=True) as target:
                copy_file(wrapper, target, "submission.py")
                for name in EXPECTED_H96_LAYOUT[1:4]:
                    copy_member(source, target, name)
                copy_file(ftq, target, "models/fast_temporal_quantile_head.py")
                copy_file(flow, target, "models/distilled_flow_corrector.py")
                copy_member(source, target, "ensemble_unet2d.pth")
        if temporary.stat().st_size > args.max_size_mib * 1024**2:
            raise RuntimeError("S2b latency probe exceeds the package size limit")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
        identities = {
            name: {
                "bytes": archive.getinfo(name).file_size,
                "sha256": member_sha256(archive, name),
            }
            for name in members
        }
    if bad_member is not None:
        raise RuntimeError(f"corrupt archive member: {bad_member}")
    if members != OUTPUT_LAYOUT:
        raise RuntimeError(f"unexpected archive layout: {members}")
    print(json.dumps({
        "archive": str(output),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "source_h96_sha256": EXPECTED_H96_SHA256,
        "members": identities,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
