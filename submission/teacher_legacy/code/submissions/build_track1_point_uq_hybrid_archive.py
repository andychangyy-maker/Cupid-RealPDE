#!/usr/bin/env python3
"""Build an exact point/UQ hybrid from two compatible Track 1 archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

import torch


CHECKPOINT_MEMBER = "ensemble_unet2d.pth"
EXPECTED_MEMBERS = (
    "submission.py",
    "models/__init__.py",
    "models/unet2d.py",
    "models/ensemble_unet2d.py",
    CHECKPOINT_MEMBER,
)
UQ_PREFIXES = (
    "uq_projections.",
    "uq_fusion.",
    "quantile_head.",
    "utility_refinement_head.",
    "temporal_uq_refiner.",
    "coverage_calibration_head.",
    "residual_mixture_head.",
    "output_space_uq_allocator.",
    "output_space_uq_budgeter.",
)
COMPATIBLE_EXPLICIT_DEFAULTS = {
    "acceleration_steps": 0,
    "phase_delta_hidden_channels": 0,
    "phase_delta_steps": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-archive", type=Path, required=True)
    parser.add_argument("--uq-archive", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_uq_key(key: str) -> bool:
    return key.startswith(UQ_PREFIXES)


def normalized_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    for key, value in COMPATIBLE_EXPLICIT_DEFAULTS.items():
        normalized.setdefault(key, value)
    return normalized


def _load_archive(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        members = tuple(archive.namelist())
        if members != EXPECTED_MEMBERS:
            raise ValueError(f"unexpected archive members for {path}: {members}")
        files = {name: archive.read(name) for name in members}
    payload = torch.load(
        io.BytesIO(files[CHECKPOINT_MEMBER]),
        map_location="cpu",
        weights_only=False,
    )
    if "quantized_state_dict" not in payload:
        raise ValueError(f"archive is not an INT8-storage package: {path}")
    return files, payload


def _tensor_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            tuple(left) == tuple(right)
            and all(_tensor_equal(left[key], right[key]) for key in left)
        )
    return left == right


def _update_signature(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, Mapping):
        for key in sorted(value):
            digest.update(key.encode())
            _update_signature(digest, value[key])
        return
    digest.update(repr(value).encode())


def state_signature(state: Mapping[str, Any], *, uq: bool) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if is_uq_key(key) != uq:
            continue
        digest.update(key.encode())
        _update_signature(digest, state[key])
    return digest.hexdigest()


def build_hybrid_payload(
    point_payload: Mapping[str, Any], uq_payload: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    for label, payload in (("point", point_payload), ("uq", uq_payload)):
        if payload.get("storage_format") != "symmetric_per_dim0_int8_v1":
            raise ValueError(f"{label} payload has unsupported storage format")
        if payload.get("model_config", {}).get("architecture") != "ensemble_unet":
            raise ValueError(f"{label} payload is not ensemble_unet")
    point_config = normalized_model_config(point_payload["model_config"])
    uq_config = normalized_model_config(uq_payload["model_config"])
    if point_config != uq_config:
        raise ValueError("point and UQ model configurations differ")
    if not _tensor_equal(point_payload["normalizer"], uq_payload["normalizer"]):
        raise ValueError("point and UQ normalizers differ")
    point_state = point_payload["quantized_state_dict"]
    uq_state = uq_payload["quantized_state_dict"]
    if tuple(point_state) != tuple(uq_state):
        raise ValueError("point and UQ state-key order differs")

    hybrid_state = {
        key: uq_state[key] if is_uq_key(key) else point_state[key]
        for key in point_state
    }
    hybrid = dict(point_payload)
    hybrid["model_config"] = point_config
    hybrid["quantized_state_dict"] = hybrid_state
    audit = {
        "state_keys": len(hybrid_state),
        "point_keys": sum(not is_uq_key(key) for key in hybrid_state),
        "uq_keys": sum(is_uq_key(key) for key in hybrid_state),
        "point_state_sha256": state_signature(hybrid_state, uq=False),
        "uq_state_sha256": state_signature(hybrid_state, uq=True),
        "point_source_exact": all(
            _tensor_equal(hybrid_state[key], point_state[key])
            for key in hybrid_state
            if not is_uq_key(key)
        ),
        "uq_source_exact": all(
            _tensor_equal(hybrid_state[key], uq_state[key])
            for key in hybrid_state
            if is_uq_key(key)
        ),
    }
    return hybrid, audit


def build_archive(
    *,
    point_archive: Path,
    uq_archive: Path,
    runtime_archive: Path,
    output: Path,
    report: Path,
    max_size_mib: float,
) -> dict[str, Any]:
    point_files, point_payload = _load_archive(point_archive)
    uq_files, uq_payload = _load_archive(uq_archive)
    runtime_files, runtime_payload = _load_archive(runtime_archive)
    del point_files, uq_files
    if normalized_model_config(runtime_payload["model_config"]) != normalized_model_config(
        point_payload["model_config"]
    ):
        raise ValueError("runtime model configuration differs from hybrid sources")

    hybrid, audit = build_hybrid_payload(point_payload, uq_payload)
    buffer = io.BytesIO()
    torch.save(hybrid, buffer)
    runtime_files[CHECKPOINT_MEMBER] = buffer.getvalue()

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for name in EXPECTED_MEMBERS:
                archive.writestr(name, runtime_files[name])
        if temporary.stat().st_size > max_size_mib * 1024**2:
            raise RuntimeError("hybrid archive exceeds size limit")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(output) as archive:
        if tuple(archive.namelist()) != EXPECTED_MEMBERS or archive.testzip() is not None:
            raise RuntimeError("hybrid archive integrity check failed")

    result = {
        "schema": "b210_point_uq_hybrid_archive_v1",
        "point_archive": {
            "path": str(point_archive.resolve()),
            "sha256": sha256(point_archive),
        },
        "uq_archive": {
            "path": str(uq_archive.resolve()),
            "sha256": sha256(uq_archive),
        },
        "runtime_archive": {
            "path": str(runtime_archive.resolve()),
            "sha256": sha256(runtime_archive),
        },
        "output": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
        "audit": audit,
    }
    report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    result = build_archive(
        point_archive=args.point_archive,
        uq_archive=args.uq_archive,
        runtime_archive=args.runtime_archive,
        output=args.output,
        report=args.report,
        max_size_mib=args.max_size_mib,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
