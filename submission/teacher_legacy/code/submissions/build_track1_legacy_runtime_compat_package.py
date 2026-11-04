#!/usr/bin/env python3
"""Put compatible new B210 weights behind a previously verified runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping
import zipfile

import torch


EXPECTED_MEMBERS = (
    "submission.py",
    "models/__init__.py",
    "models/unet2d.py",
    "models/ensemble_unet2d.py",
    "ensemble_unet2d.pth",
)
CHECKPOINT_MEMBER = "ensemble_unet2d.pth"
LEGACY_DEFAULTS = {
    "acceleration_steps": 0,
    "phase_delta_hidden_channels": 0,
    "phase_delta_steps": 5,
}
MAX_BYTES = 256 * 1024**2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path, expected_sha256: str) -> None:
    if sha256(path) != expected_sha256.lower():
        raise RuntimeError(f"archive SHA-256 differs: {path}")
    with zipfile.ZipFile(path) as archive:
        if tuple(archive.namelist()) != EXPECTED_MEMBERS:
            raise RuntimeError(f"archive layout differs: {path}")
        if archive.testzip() is not None:
            raise RuntimeError(f"archive CRC validation failed: {path}")


def legacy_config(config: Mapping[str, object]) -> dict[str, object]:
    value = dict(config)
    observed = {
        key: value.pop(key) for key in LEGACY_DEFAULTS if key in value
    }
    if observed != LEGACY_DEFAULTS:
        raise RuntimeError(
            f"candidate legacy-default fields differ: {observed}"
        )
    return value


def stored_layout(value: object) -> tuple:
    if isinstance(value, Mapping) and set(value) == {"quantized", "scale"}:
        quantized = value["quantized"]
        scale = value["scale"]
        if not isinstance(quantized, torch.Tensor) or not isinstance(scale, torch.Tensor):
            raise TypeError("quantized state entry is not tensor-backed")
        return (
            "quantized", tuple(quantized.shape), str(quantized.dtype),
            tuple(scale.shape), str(scale.dtype),
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError("plain state entry is not a tensor")
    return ("plain", tuple(value.shape), str(value.dtype))


def validate_compatible_payloads(anchor: dict, candidate: dict) -> dict:
    required = {
        "phase", "model_config", "quantized_state_dict", "normalizer",
        "inference_dtype", "storage_format",
    }
    if set(anchor) != required or set(candidate) != required:
        raise RuntimeError("checkpoint payload schema differs")
    if legacy_config(candidate["model_config"]) != anchor["model_config"]:
        raise RuntimeError("candidate architecture is not legacy-compatible")
    anchor_state = anchor["quantized_state_dict"]
    candidate_state = candidate["quantized_state_dict"]
    if not isinstance(anchor_state, Mapping) or not isinstance(candidate_state, Mapping):
        raise TypeError("quantized state dictionary is missing")
    if set(anchor_state) != set(candidate_state):
        raise RuntimeError("candidate state keys differ from legacy runtime")
    mismatches = [
        key for key in anchor_state
        if stored_layout(anchor_state[key]) != stored_layout(candidate_state[key])
    ]
    if mismatches:
        raise RuntimeError(f"candidate state layouts differ: {mismatches}")
    for key in ("phase", "inference_dtype", "storage_format"):
        if candidate[key] != anchor[key]:
            raise RuntimeError(f"candidate {key} differs from runtime anchor")
    return {
        "state_entries": len(candidate_state),
        "removed_legacy_defaults": dict(LEGACY_DEFAULTS),
        "shared_config_exact": True,
        "state_layout_exact": True,
    }


def tensor_payload_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            tensor_payload_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    return left == right


def build(args: argparse.Namespace) -> dict:
    anchor_path = args.runtime_anchor.resolve()
    candidate_path = args.candidate.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    validate_archive(anchor_path, args.runtime_anchor_sha256)
    validate_archive(candidate_path, args.candidate_sha256)

    with zipfile.ZipFile(anchor_path) as archive:
        anchor_infos = {info.filename: copy.copy(info) for info in archive.infolist()}
        anchor_members = {name: archive.read(name) for name in EXPECTED_MEMBERS}
    with zipfile.ZipFile(candidate_path) as archive:
        candidate_members = {name: archive.read(name) for name in EXPECTED_MEMBERS}
    if candidate_members["submission.py"] != anchor_members["submission.py"]:
        raise RuntimeError("candidate and runtime anchor wrappers differ")

    anchor_payload = torch.load(
        io.BytesIO(anchor_members[CHECKPOINT_MEMBER]),
        map_location="cpu", weights_only=False,
    )
    candidate_payload = torch.load(
        io.BytesIO(candidate_members[CHECKPOINT_MEMBER]),
        map_location="cpu", weights_only=False,
    )
    compatibility = validate_compatible_payloads(anchor_payload, candidate_payload)
    converted = dict(candidate_payload)
    converted["model_config"] = legacy_config(candidate_payload["model_config"])
    checkpoint_buffer = io.BytesIO()
    torch.save(converted, checkpoint_buffer)
    converted_bytes = checkpoint_buffer.getvalue()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED,
            compresslevel=6, allowZip64=True,
        ) as target:
            for name in EXPECTED_MEMBERS:
                payload = (
                    converted_bytes if name == CHECKPOINT_MEMBER
                    else anchor_members[name]
                )
                target.writestr(anchor_infos[name], payload)
        if temporary.stat().st_size > MAX_BYTES:
            raise RuntimeError("legacy-runtime candidate exceeds 256 MiB")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(output) as archive:
        if tuple(archive.namelist()) != EXPECTED_MEMBERS or archive.testzip() is not None:
            raise RuntimeError("built archive failed layout/CRC validation")
        restored = torch.load(
            io.BytesIO(archive.read(CHECKPOINT_MEMBER)),
            map_location="cpu", weights_only=False,
        )
        code_equal = {
            name: archive.read(name) == anchor_members[name]
            for name in EXPECTED_MEMBERS if name != CHECKPOINT_MEMBER
        }
    if not all(code_equal.values()):
        raise RuntimeError("built runtime code differs from anchor")
    if not tensor_payload_equal(
        restored["quantized_state_dict"], candidate_payload["quantized_state_dict"]
    ):
        raise RuntimeError("built archive changed candidate tensor payload")
    if restored["normalizer"] != candidate_payload["normalizer"]:
        raise RuntimeError("built archive changed candidate normalizer")

    return {
        "schema": "b210_legacy_runtime_compat_package_v1",
        "runtime_anchor": {
            "path": str(anchor_path),
            "sha256": args.runtime_anchor_sha256.lower(),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": args.candidate_sha256.lower(),
        },
        "output": {
            "path": str(output),
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
        },
        "compatibility": compatibility,
        "runtime_code_bitwise_anchor": code_equal,
        "candidate_tensor_payload_bitwise": True,
        "submission_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-anchor", type=Path, required=True)
    parser.add_argument("--runtime-anchor-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.result.exists():
        raise FileExistsError(args.result)
    result = build(args)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "legacy_runtime_package_complete", **result["output"]}), flush=True)


if __name__ == "__main__":
    main()
