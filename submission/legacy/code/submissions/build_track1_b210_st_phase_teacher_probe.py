#!/usr/bin/env python3
"""Build the fixed B210/ST isoenergetic teacher-transfer probe.

This is deliberately a post-hoc clean716 transfer probe, not an unbiased
model-selection result.  The builder keeps the B210 U3200 runtime and UQ
checkpoint intact, losslessly collapses each duplicated ST temporal bank,
forms one fixed 50/50 FP32 seed soup, and stores only that ST soup as INT8.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import zipfile

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.spatiotemporal_b210 import (  # noqa: E402
    SpatiotemporalB210,
    SpatiotemporalB210Config,
)


SCHEMA = "track1_b210_st_phase_teacher_probe_v1"
STORAGE_FORMAT = "symmetric_per_dim0_int8_v1"
BASE_CHECKPOINT = "ensemble_unet2d.pth"
ST_CHECKPOINT = "spatiotemporal_b210.pth"
BASE_MEMBERS = (
    "submission.py",
    "models/__init__.py",
    "models/unet2d.py",
    "models/ensemble_unet2d.py",
    BASE_CHECKPOINT,
)
OUTPUT_MEMBERS = (
    "submission.py",
    "models/__init__.py",
    "models/unet2d.py",
    "models/ensemble_unet2d.py",
    "models/pyramid_causal3d_operator.py",
    "models/spatiotemporal_b210.py",
    BASE_CHECKPOINT,
    ST_CHECKPOINT,
)
EXPECTED_ENDPOINT_SCHEMA = "st_b210_source_anchored_twin_endpoint_v1"
EXPECTED_ENDPOINT_ARCHITECTURE = "source_anchored_twin_spatiotemporal_b210"
EXPECTED_ADAPTATION_MODE = "duplicated_trainable_control"
EXPECTED_UPDATE = 12000
FIXED_SEED_WEIGHT = 0.5
QUANTIZE_MIN_ELEMENTS = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--seed4-checkpoint", type=Path, required=True)
    parser.add_argument("--seed17-checkpoint", type=Path, required=True)
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


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _nested_tensor_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            tuple(left) == tuple(right)
            and all(_nested_tensor_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_tensor_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _read_base_archive(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = tuple(archive.namelist())
        if names != BASE_MEMBERS:
            raise ValueError(f"unexpected B210 archive members: {names}")
        if archive.testzip() is not None:
            raise ValueError("B210 archive is corrupt")
        files = {name: archive.read(name) for name in names}
    payload = torch.load(
        io.BytesIO(files[BASE_CHECKPOINT]),
        map_location="cpu",
        weights_only=False,
    )
    if (
        payload.get("storage_format") != STORAGE_FORMAT
        or payload.get("model_config", {}).get("architecture") != "ensemble_unet"
        or "quantized_state_dict" not in payload
        or payload.get("inference_dtype") != "float16"
    ):
        raise ValueError("base archive is not the expected FP16/INT8 B210 package")
    return files, payload


def _load_endpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("model_config", {})
    if (
        payload.get("schema") != EXPECTED_ENDPOINT_SCHEMA
        or payload.get("arm") != "dual_adapt_control"
        or payload.get("adaptation_mode") != EXPECTED_ADAPTATION_MODE
        or payload.get("endpoint") != "ema"
        or int(payload.get("update", -1)) != EXPECTED_UPDATE
        or config.get("architecture") != EXPECTED_ENDPOINT_ARCHITECTURE
        or config.get("adaptation_mode") != EXPECTED_ADAPTATION_MODE
        or payload.get("uq") is not False
    ):
        raise ValueError(f"unexpected ST endpoint contract: {path}")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"ST endpoint has no model state: {path}")
    return payload


def collapse_identical_temporal_banks(
    twin_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Losslessly map a twin state to one ordinary ST-B210 state.

    Every source-bank tensor must have a bitwise-identical real-bank partner.
    A mismatch is a hard error: averaging inside a checkpoint is not permitted.
    """
    collapsed: dict[str, torch.Tensor] = {}
    matched = 0
    for key, value in twin_state.items():
        if not torch.is_tensor(value):
            raise TypeError(f"state value is not a tensor: {key}")
        if ".real_bank." in key:
            source_key = key.replace(".real_bank.", ".source_bank.")
            if source_key not in twin_state:
                raise ValueError(f"orphan real-bank tensor: {key}")
            continue
        if ".source_bank." in key:
            real_key = key.replace(".source_bank.", ".real_bank.")
            real = twin_state.get(real_key)
            if not torch.is_tensor(real):
                raise ValueError(f"missing real-bank partner: {real_key}")
            if value.dtype != real.dtype or value.shape != real.shape:
                raise ValueError(f"temporal-bank tensor contract differs: {key}")
            if not torch.equal(value, real):
                maximum = float((value.float() - real.float()).abs().max())
                raise ValueError(
                    f"temporal banks are not bitwise equal at {key}; max={maximum}"
                )
            output_key = key.replace(".source_bank.", ".")
            matched += 1
        else:
            output_key = key
        if output_key in collapsed:
            raise ValueError(f"collapse produced duplicate key: {output_key}")
        collapsed[output_key] = value.detach().cpu().clone()
    if matched == 0:
        raise ValueError("checkpoint contains no duplicated temporal banks")
    return collapsed, {
        "matched_bank_tensors": matched,
        "twin_tensors": len(twin_state),
        "collapsed_tensors": len(collapsed),
        "bitwise_equal": True,
        "collapsed_state_sha256": tensor_state_sha256(collapsed),
    }


def _single_config(twin_config: Mapping[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(SpatiotemporalB210Config)}
    config_values = {name: twin_config[name] for name in names}
    config = SpatiotemporalB210Config(**config_values)
    config.validate()
    return {
        "architecture": "spatiotemporal_b210",
        "interaction_mode": "causal_temporal_candidate",
        "flattened_history_bypass": False,
        **{field.name: getattr(config, field.name) for field in fields(config)},
    }


def validate_single_state(
    config_dict: Mapping[str, Any], state: Mapping[str, torch.Tensor]
) -> None:
    values = {
        field.name: config_dict[field.name]
        for field in fields(SpatiotemporalB210Config)
    }
    model = SpatiotemporalB210(
        SpatiotemporalB210Config(**values),
        interaction_mode=str(config_dict["interaction_mode"]),
    )
    model.load_state_dict(state, strict=True)


def equal_weight_fp32_soup(
    seed4: Mapping[str, torch.Tensor], seed17: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if tuple(seed4) != tuple(seed17):
        raise ValueError("seed states have different key order")
    soup: dict[str, torch.Tensor] = {}
    for key in seed4:
        left, right = seed4[key], seed17[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"seed tensor contract differs: {key}")
        if left.is_floating_point():
            soup[key] = (left.float() + right.float()).mul_(FIXED_SEED_WEIGHT)
        else:
            if not torch.equal(left, right):
                raise ValueError(f"non-floating seed tensors differ: {key}")
            soup[key] = left.clone()
    return soup


def _quantize_tensor(value: torch.Tensor) -> dict[str, torch.Tensor]:
    value = value.detach().cpu().float()
    dimensions = tuple(range(1, value.ndim))
    scale = (value.abs().amax(dim=dimensions, keepdim=True) / 127.0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    quantized = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    return {"quantized": quantized, "scale": scale}


def compact_state_dict(
    state: Mapping[str, torch.Tensor], min_elements: int = QUANTIZE_MIN_ELEMENTS
) -> tuple[dict[str, Any], dict[str, int]]:
    stored: dict[str, Any] = {}
    tensors = elements = 0
    for key, value in state.items():
        if value.is_floating_point() and value.ndim >= 2 and value.numel() >= min_elements:
            stored[key] = _quantize_tensor(value)
            tensors += 1
            elements += value.numel()
        elif value.is_floating_point():
            stored[key] = value.detach().cpu().to(torch.float16)
        else:
            stored[key] = value.detach().cpu().clone()
    return stored, {
        "quantized_tensors": tensors,
        "quantized_elements": elements,
        "total_tensors": len(stored),
    }


def validate_archive_size_bytes(size: int, max_size_mib: float) -> None:
    """Require a strict byte-level margin below the configured archive cap."""
    if max_size_mib <= 0:
        raise ValueError("max-size-mib must be positive")
    if size >= max_size_mib * 1024**2:
        raise RuntimeError(
            f"archive is {size / 1024**2:.3f} MiB, not below "
            f"{max_size_mib:.3f} MiB"
        )


def build_archive(
    *,
    base_archive: Path,
    seed4_checkpoint: Path,
    seed17_checkpoint: Path,
    output: Path,
    report: Path,
    max_size_mib: float,
) -> dict[str, Any]:
    if max_size_mib <= 0:
        raise ValueError("max-size-mib must be positive")
    base_files, base_payload = _read_base_archive(base_archive)
    endpoints = {
        "seed4": _load_endpoint(seed4_checkpoint),
        "seed17": _load_endpoint(seed17_checkpoint),
    }
    if not _nested_tensor_equal(
        endpoints["seed4"]["normalizer"], endpoints["seed17"]["normalizer"]
    ):
        raise ValueError("ST seed normalizers differ")
    if endpoints["seed4"]["model_config"] != endpoints["seed17"]["model_config"]:
        raise ValueError("ST seed model configurations differ")

    collapsed: dict[str, dict[str, torch.Tensor]] = {}
    collapse_audit: dict[str, Any] = {}
    for seed, payload in endpoints.items():
        collapsed[seed], collapse_audit[seed] = collapse_identical_temporal_banks(
            payload["model_state_dict"]
        )
    single_config = _single_config(endpoints["seed4"]["model_config"])
    for state in collapsed.values():
        validate_single_state(single_config, state)
    soup = equal_weight_fp32_soup(collapsed["seed4"], collapsed["seed17"])
    validate_single_state(single_config, soup)
    soup_sha = tensor_state_sha256(soup)
    stored_state, quantization = compact_state_dict(soup)
    st_payload = {
        "schema": SCHEMA,
        "model_config": single_config,
        "quantized_state_dict": stored_state,
        "normalizer": endpoints["seed4"]["normalizer"],
        "inference_dtype": "float16",
        "storage_format": STORAGE_FORMAT,
        "fixed_seed_weights": {"seed4": 0.5, "seed17": 0.5},
        "unquantized_soup_state_sha256": soup_sha,
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(st_payload, checkpoint_buffer)

    runtime = REPO_ROOT / "code/submissions/track1_b210_st_phase_teacher_probe.py"
    st_model = REPO_ROOT / "code/models/spatiotemporal_b210.py"
    st_dependency = REPO_ROOT / "code/models/pyramid_causal3d_operator.py"
    required = (
        base_archive,
        seed4_checkpoint,
        seed17_checkpoint,
        runtime,
        st_model,
        st_dependency,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")

    package_files = {
        "submission.py": runtime.read_bytes(),
        "models/__init__.py": base_files["models/__init__.py"],
        "models/unet2d.py": base_files["models/unet2d.py"],
        "models/ensemble_unet2d.py": base_files["models/ensemble_unet2d.py"],
        "models/pyramid_causal3d_operator.py": st_dependency.read_bytes(),
        "models/spatiotemporal_b210.py": st_model.read_bytes(),
        BASE_CHECKPOINT: base_files[BASE_CHECKPOINT],
        ST_CHECKPOINT: checkpoint_buffer.getvalue(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for name in OUTPUT_MEMBERS:
                archive.writestr(name, package_files[name])
        size = temporary.stat().st_size
        validate_archive_size_bytes(size, max_size_mib)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(output) as archive:
        if tuple(archive.namelist()) != OUTPUT_MEMBERS:
            raise RuntimeError("final archive member order changed")
        if archive.testzip() is not None:
            raise RuntimeError("final archive failed integrity check")
        base_checkpoint_preserved = (
            archive.read(BASE_CHECKPOINT) == base_files[BASE_CHECKPOINT]
        )
    if not base_checkpoint_preserved:
        raise RuntimeError("B210 checkpoint bytes changed")

    result = {
        "schema": SCHEMA,
        "evidence_class": "posthoc_clean716_transfer_probe_not_unbiased_selection",
        "submission_authorized": False,
        "fixed_method": {
            "seed_weights": {"seed4": 0.5, "seed17": 0.5},
            "point_fusion": "restore_temporal_mean_rms(0.5*B210+0.5*ST_soup,B210)",
            "projection_scope": "per_sample_pixel_uv_across_20_horizons",
            "pressure": "exact_zero",
            "bounds": "B210_lower_upper_elementwise_passthrough_no_recenter",
        },
        "sources": {
            "base_archive": {
                "path": str(base_archive.resolve()),
                "sha256": sha256(base_archive),
                "checkpoint_bytes_preserved": base_checkpoint_preserved,
                "model_config": base_payload["model_config"],
            },
            "seed4": {
                "path": str(seed4_checkpoint.resolve()),
                "sha256": sha256(seed4_checkpoint),
                "model_state_sha256": endpoints["seed4"]["model_state_sha256"],
            },
            "seed17": {
                "path": str(seed17_checkpoint.resolve()),
                "sha256": sha256(seed17_checkpoint),
                "model_state_sha256": endpoints["seed17"]["model_state_sha256"],
            },
        },
        "collapse": collapse_audit,
        "soup": {
            "arithmetic": "FP32_equal_weight_mean",
            "state_sha256": soup_sha,
            "state_tensors": len(soup),
        },
        "storage": {
            "format": STORAGE_FORMAT,
            "inference_dtype": "float16_on_cuda_float32_on_cpu",
            "quantize_min_elements": QUANTIZE_MIN_ELEMENTS,
            **quantization,
        },
        "output": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "mib": output.stat().st_size / 1024**2,
            "sha256": sha256(output),
            "members": list(OUTPUT_MEMBERS),
            "below_256_mib": output.stat().st_size < 256 * 1024**2,
        },
    }
    report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    result = build_archive(
        base_archive=args.base_archive.resolve(),
        seed4_checkpoint=args.seed4_checkpoint.resolve(),
        seed17_checkpoint=args.seed17_checkpoint.resolve(),
        output=args.output.resolve(),
        report=args.report.resolve(),
        max_size_mib=args.max_size_mib,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
