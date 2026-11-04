#!/usr/bin/env python3
"""Build the sealed EMA-u4000 causal phase adapter submission archive.

The known B210 FP16/INT8 archive is used as a byte-preserved base.  The large
dense checkpoint is read only to bind provenance and metadata; it is never
copied into the output.  Only the 41,667-parameter EMA adapter is added.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
from typing import Any, Mapping
import zipfile

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
)


SCHEMA = "track1_b210_causal_phase_direction_package_v1"
ENDPOINT_SCHEMA = "b210_causal_phase_direction_adapter_endpoint_v1"
EXPERIMENT_SCHEMA = "b210_causal_phase_direction_real4k_matched_v1"
STORAGE_FORMAT = "symmetric_per_dim0_int8_v1"
FP16_STORAGE_FORMAT = "dense_fp16_v1"
ADAPTER_STORAGE_CHOICES = ("fp16", "int8")
EXPECTED_ARM = "causal_temporal"
EXPECTED_UPDATE = 4_000
EXPECTED_PARAMETERS = 41_667
QUANTIZE_MIN_ELEMENTS = 1_024
BASE_CHECKPOINT = "ensemble_unet2d.pth"
ADAPTER_CHECKPOINT = "causal_phase_direction.pth"
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
    "models/temporal_transport.py",
    "models/causal_phase_direction.py",
    BASE_CHECKPOINT,
    ADAPTER_CHECKPOINT,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--base-archive-sha256", required=True)
    parser.add_argument("--base-dense-checkpoint", type=Path, required=True)
    parser.add_argument("--base-dense-checkpoint-sha256", required=True)
    parser.add_argument("--base-model-state-sha256", required=True)
    parser.add_argument("--adapter-endpoint", type=Path, required=True)
    parser.add_argument("--adapter-endpoint-sha256", required=True)
    parser.add_argument(
        "--adapter-storage", choices=ADAPTER_STORAGE_CHOICES, default="fp16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    expected = str(expected).strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"{label} SHA-256 must be 64 lowercase hex digits")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} changed: {actual} != {expected}")
    return actual


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Match the training seal's container-independent tensor-state hash."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise TypeError(f"non-tensor adapter state: {key}")
        tensor = value.detach().cpu().contiguous()
        name_bytes = key.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack("<Q", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<Q", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<q", int(dimension)))
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _normalizer_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    names = ("mean_input", "mean_target", "std_input", "std_target")
    if set(left) != set(names) or set(right) != set(names):
        return False
    return all(torch.equal(
        torch.as_tensor(left[name]).float(),
        torch.as_tensor(right[name]).float()) for name in names)


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
        map_location="cpu", weights_only=False)
    if (payload.get("storage_format") != STORAGE_FORMAT
            or payload.get("inference_dtype") != "float16"
            or payload.get("model_config", {}).get("architecture")
            != "ensemble_unet"
            or not isinstance(payload.get("quantized_state_dict"), Mapping)):
        raise ValueError("base archive is not the expected FP16/INT8 B210")
    return files, payload


def _read_dense_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True)
    if (payload.get("model_config", {}).get("architecture") != "ensemble_unet"
            or not isinstance(payload.get("model_state_dict"), Mapping)):
        raise ValueError("dense source is not an EnsembleUNet2d checkpoint")
    return {
        "model_config": payload["model_config"],
        "model_state_dict": payload["model_state_dict"],
        "normalizer": payload["normalizer"],
        "phase": payload.get("phase"),
        "iteration": payload.get("iteration"),
    }


def validate_compact_base_lineage(
        dense_state: Mapping[str, torch.Tensor],
        stored_state: Mapping[str, Any],
        min_elements: int = QUANTIZE_MIN_ELEMENTS) -> dict[str, int]:
    """Prove the packaged INT8 state is the deterministic dense-state image."""
    if tuple(dense_state) != tuple(stored_state):
        raise ValueError("packaged and dense B210 state keys/order differ")
    quantized = dense = elements = 0
    for key, source in dense_state.items():
        stored = stored_state[key]
        if (source.is_floating_point() and source.ndim >= 2
                and source.numel() >= min_elements):
            expected = _quantize_tensor(source)
            if (not isinstance(stored, Mapping)
                    or set(stored) != {"quantized", "scale"}
                    or not torch.equal(stored["quantized"], expected["quantized"])
                    or not torch.equal(stored["scale"], expected["scale"])):
                raise ValueError(f"packaged B210 INT8 lineage differs at {key}")
            quantized += 1
            elements += source.numel()
        elif source.is_floating_point():
            expected = source.detach().cpu().to(torch.float16)
            if not torch.equal(stored, expected):
                raise ValueError(f"packaged B210 FP16 lineage differs at {key}")
            dense += 1
        else:
            if not torch.equal(stored, source.detach().cpu()):
                raise ValueError(f"packaged B210 tensor lineage differs at {key}")
            dense += 1
    return {
        "quantized_tensors": quantized,
        "quantized_elements": elements,
        "non_quantized_tensors": dense,
        "all_tensors_exact": len(dense_state),
    }


def _adapter_config(payload: Mapping[str, Any]) -> CausalPhaseDirectionConfig:
    config = payload.get("adapter_config")
    if not isinstance(config, Mapping):
        raise ValueError("adapter endpoint has no adapter_config")
    names = {field.name for field in fields(CausalPhaseDirectionConfig)}
    values = {name: config[name] for name in names}
    result = CausalPhaseDirectionConfig(**values)
    result.validate()
    if (config.get("architecture") != "causal_phase_direction_adapter"
            or result.mixer_mode != EXPECTED_ARM):
        raise ValueError("endpoint is not the causal temporal phase adapter")
    return result


def _load_ema_endpoint(
        path: Path, dense_sha: str) -> tuple[
            dict[str, Any], CausalPhaseDirectionConfig,
            Mapping[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != ENDPOINT_SCHEMA
            or payload.get("experiment_schema") != EXPERIMENT_SCHEMA
            or payload.get("arm") != EXPECTED_ARM
            or int(payload.get("update", -1)) != EXPECTED_UPDATE
            or int(payload.get("adapter_parameters", -1)) != EXPECTED_PARAMETERS):
        raise ValueError("adapter endpoint contract is not sealed causal u4000")
    base = payload.get("base_checkpoint")
    if not isinstance(base, Mapping) or base.get("sha256") != dense_sha:
        raise ValueError("adapter endpoint is not bound to the requested B210")
    mechanics = payload.get("mechanics", {})
    for name in (
            "update0_bitwise_identity", "direction_head_gradient_u1",
            "history_encoder_gradient_u2", "direction_gate_gradient_u2"):
        if mechanics.get(name) is not True:
            raise ValueError(f"adapter mechanics seal failed: {name}")
    state = payload.get("ema_adapter_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("adapter endpoint has no EMA state")
    state_sha = tensor_state_sha256(state)
    if state_sha != payload.get("ema_adapter_state_sha256"):
        raise ValueError("adapter EMA state SHA-256 is inconsistent")
    config = _adapter_config(payload)
    model = CausalPhaseDirectionAdapter(config)
    model.load_state_dict(state, strict=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETERS:
        raise ValueError(f"adapter parameter count changed: {count}")
    return dict(payload), config, state


def _quantize_tensor(value: torch.Tensor) -> dict[str, torch.Tensor]:
    value = value.detach().cpu().float()
    dimensions = tuple(range(1, value.ndim))
    scale = (value.abs().amax(dim=dimensions, keepdim=True) / 127.0).clamp_min(
        torch.finfo(torch.float32).tiny)
    quantized = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    return {"quantized": quantized, "scale": scale}


def compact_state_dict(
        state: Mapping[str, torch.Tensor],
        min_elements: int = QUANTIZE_MIN_ELEMENTS,
        ) -> tuple[dict[str, Any], dict[str, int]]:
    stored: dict[str, Any] = {}
    quantized_tensors = quantized_elements = 0
    for key, value in state.items():
        if (value.is_floating_point() and value.ndim >= 2
                and value.numel() >= min_elements):
            stored[key] = _quantize_tensor(value)
            quantized_tensors += 1
            quantized_elements += value.numel()
        elif value.is_floating_point():
            stored[key] = value.detach().cpu().to(torch.float16)
        else:
            stored[key] = value.detach().cpu().clone()
    return stored, {
        "quantized_tensors": quantized_tensors,
        "quantized_elements": quantized_elements,
        "total_tensors": len(stored),
    }


def fp16_state_dict(
        state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: (value.detach().cpu().to(torch.float16)
              if value.is_floating_point() else value.detach().cpu().clone())
        for key, value in state.items()
    }


def restore_state_dict(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    stored_state = payload.get("quantized_state_dict")
    if stored_state is None:
        stored_state = payload["model_state_dict"]
    restored = {}
    for key, value in stored_state.items():
        if isinstance(value, Mapping) and "quantized" in value:
            restored[key] = value["quantized"].float() * value["scale"].float()
        elif torch.is_floating_point(value):
            restored[key] = value.float()
        else:
            restored[key] = value
    return restored


def state_recovery_error(
        source: Mapping[str, torch.Tensor],
        restored: Mapping[str, torch.Tensor]) -> dict[str, float | int]:
    if tuple(source) != tuple(restored):
        raise ValueError("source/restored adapter state keys differ")
    squared_error = squared_reference = 0.0
    maximum = 0.0
    elements = 0
    for key, value in source.items():
        candidate = restored[key]
        if value.is_floating_point():
            reference = value.detach().cpu().float()
            difference = candidate.detach().cpu().float() - reference
            squared_error += float(difference.double().square().sum())
            squared_reference += float(reference.double().square().sum())
            maximum = max(maximum, float(difference.abs().max()))
            elements += reference.numel()
        elif not torch.equal(candidate, value.detach().cpu()):
            raise ValueError(f"non-floating adapter state changed: {key}")
    rms = (squared_error / max(elements, 1)) ** 0.5
    reference_rms = (squared_reference / max(elements, 1)) ** 0.5
    return {
        "floating_elements": elements,
        "rms_error": rms,
        "max_abs_error": maximum,
        "relative_rms_error": rms / max(reference_rms, 1.0e-30),
    }


def _assert_no_private_path_leak(
        package_files: Mapping[str, bytes], source_paths: tuple[Path, ...]) -> None:
    needles = {str(path.resolve()).encode("utf-8") for path in source_paths}
    needles.update((b"/root/autodl-tmp/", b"file:///root/"))
    for name, data in package_files.items():
        for needle in needles:
            if needle and needle in data:
                raise RuntimeError(f"private source path leaked into {name}")


def validate_archive_size_bytes(size: int, max_size_mib: float) -> None:
    if max_size_mib <= 0:
        raise ValueError("max-size-mib must be positive")
    if size >= max_size_mib * 1024**2:
        raise RuntimeError(
            f"archive is {size / 1024**2:.3f} MiB, not below "
            f"{max_size_mib:.3f} MiB")


def build_archive(
        *, base_archive: Path, base_archive_sha256: str,
        base_dense_checkpoint: Path, base_dense_checkpoint_sha256: str,
        base_model_state_sha256: str,
        adapter_endpoint: Path, adapter_endpoint_sha256: str,
        output: Path, report: Path, max_size_mib: float,
        adapter_storage: str = "fp16",
        ) -> dict[str, Any]:
    if adapter_storage not in ADAPTER_STORAGE_CHOICES:
        raise ValueError(
            f"adapter_storage must be one of {ADAPTER_STORAGE_CHOICES}")
    required = (
        base_archive, base_dense_checkpoint, adapter_endpoint,
        REPO_ROOT / "code/submissions/track1_b210_causal_phase_direction.py",
        REPO_ROOT / "code/models/temporal_transport.py",
        REPO_ROOT / "code/models/causal_phase_direction.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package inputs: {missing}")

    archive_sha = require_sha(
        base_archive, base_archive_sha256, "base archive")
    dense_sha = require_sha(
        base_dense_checkpoint, base_dense_checkpoint_sha256,
        "base dense checkpoint")
    endpoint_sha = require_sha(
        adapter_endpoint, adapter_endpoint_sha256, "adapter endpoint")
    base_files, base_payload = _read_base_archive(base_archive)
    dense = _read_dense_metadata(base_dense_checkpoint)
    expected_state_sha = str(base_model_state_sha256).strip().lower()
    if (len(expected_state_sha) != 64
            or any(c not in "0123456789abcdef" for c in expected_state_sha)):
        raise ValueError("base model-state SHA-256 must be 64 lowercase hex digits")
    actual_state_sha = tensor_state_sha256(dense["model_state_dict"])
    if actual_state_sha != expected_state_sha:
        raise RuntimeError(
            "dense B210 model state changed: "
            f"{actual_state_sha} != {expected_state_sha}")
    if base_payload["model_config"] != dense["model_config"]:
        raise ValueError("base archive and dense checkpoint model configs differ")
    if not _normalizer_equal(
            base_payload["normalizer"], dense["normalizer"]):
        raise ValueError("base archive and dense checkpoint normalizers differ")
    base_lineage = validate_compact_base_lineage(
        dense["model_state_dict"], base_payload["quantized_state_dict"])

    endpoint, config, ema_state = _load_ema_endpoint(
        adapter_endpoint, dense_sha)
    if endpoint["base_checkpoint"].get("model_state_sha256") != actual_state_sha:
        raise ValueError("adapter endpoint B210 model-state lineage differs")
    if not _normalizer_equal(
            base_payload["normalizer"], endpoint["normalizer"]):
        raise ValueError("adapter and packaged B210 normalizers differ")
    int8_state, int8_statistics = compact_state_dict(ema_state)
    fp16_state = fp16_state_dict(ema_state)
    recovery = {
        "fp16": state_recovery_error(ema_state, fp16_state),
        "int8": state_recovery_error(
            ema_state, restore_state_dict({"quantized_state_dict": int8_state})),
    }
    if recovery["fp16"]["rms_error"] > recovery["int8"]["rms_error"]:
        raise RuntimeError("FP16 adapter storage is unexpectedly less faithful")
    if adapter_storage == "fp16":
        state_payload = {"model_state_dict": fp16_state}
        adapter_storage_format = FP16_STORAGE_FORMAT
        selected_statistics = {
            "quantized_tensors": 0,
            "quantized_elements": 0,
            "total_tensors": len(fp16_state),
        }
    else:
        state_payload = {"quantized_state_dict": int8_state}
        adapter_storage_format = STORAGE_FORMAT
        selected_statistics = int8_statistics
    model_config = {
        "architecture": "causal_phase_direction_adapter",
        **asdict(config),
    }
    adapter_payload = {
        "schema": SCHEMA,
        "deployment_state": "ema_adapter_state_u4000",
        "model_config": model_config,
        **state_payload,
        "inference_dtype": "float16",
        "storage_format": adapter_storage_format,
        "adapter_parameters": EXPECTED_PARAMETERS,
        "source_endpoint_sha256": endpoint_sha,
        "source_ema_state_sha256": endpoint["ema_adapter_state_sha256"],
        "base_dense_checkpoint_sha256": dense_sha,
    }
    restored = restore_state_dict(adapter_payload)
    CausalPhaseDirectionAdapter(config).load_state_dict(restored, strict=True)
    adapter_buffer = io.BytesIO()
    torch.save(adapter_payload, adapter_buffer)

    runtime = REPO_ROOT / "code/submissions/track1_b210_causal_phase_direction.py"
    temporal_model = REPO_ROOT / "code/models/temporal_transport.py"
    phase_model = REPO_ROOT / "code/models/causal_phase_direction.py"
    package_files = {
        "submission.py": runtime.read_bytes(),
        "models/__init__.py": base_files["models/__init__.py"],
        "models/unet2d.py": base_files["models/unet2d.py"],
        "models/ensemble_unet2d.py": base_files["models/ensemble_unet2d.py"],
        "models/temporal_transport.py": temporal_model.read_bytes(),
        "models/causal_phase_direction.py": phase_model.read_bytes(),
        BASE_CHECKPOINT: base_files[BASE_CHECKPOINT],
        ADAPTER_CHECKPOINT: adapter_buffer.getvalue(),
    }
    _assert_no_private_path_leak(
        package_files,
        (base_archive, base_dense_checkpoint, adapter_endpoint, output, report))

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED,
                compresslevel=6, allowZip64=True) as archive:
            for name in OUTPUT_MEMBERS:
                archive.writestr(name, package_files[name])
        validate_archive_size_bytes(temporary.stat().st_size, max_size_mib)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(output) as archive:
        if tuple(archive.namelist()) != OUTPUT_MEMBERS:
            raise RuntimeError("final archive member order changed")
        if archive.testzip() is not None:
            raise RuntimeError("final archive failed integrity check")
        base_preserved = archive.read(BASE_CHECKPOINT) == base_files[BASE_CHECKPOINT]
    if not base_preserved:
        raise RuntimeError("B210 compact checkpoint bytes changed")

    result = {
        "schema": SCHEMA,
        "deployment": {
            "adapter_state": "ema_u4000",
            "adapter_parameters": EXPECTED_PARAMETERS,
            "point": "B210_mean_rms_plus_causal_temporal_direction",
            "pressure": "B210_passthrough_then_exact_zero",
            "bounds": (
                "bitwise_B210_absolute_endpoints_equivalent_to_"
                "normalized_point_delta_compensation"),
        },
        "sources": {
            "base_archive": {
                "filename": base_archive.name,
                "sha256": archive_sha,
                "checkpoint_bytes_preserved": base_preserved,
            },
            "base_dense_checkpoint": {
                "filename": base_dense_checkpoint.name,
                "sha256": dense_sha,
                "phase": dense["phase"],
                "iteration": dense["iteration"],
                "model_state_sha256": actual_state_sha,
            },
            "adapter_endpoint": {
                "filename": adapter_endpoint.name,
                "sha256": endpoint_sha,
                "ema_state_sha256": endpoint["ema_adapter_state_sha256"],
            },
        },
        "storage": {
            "base": "byte_preserved_symmetric_per_dim0_int8_v1",
            "adapter": adapter_storage_format,
            "inference": "fp16_on_cuda_fp32_on_cpu",
            "quantize_min_elements": QUANTIZE_MIN_ELEMENTS,
            **selected_statistics,
        },
        "adapter_storage_recovery": recovery,
        "base_point_state_lineage": {
            "matched": True,
            **base_lineage,
        },
        "output": {
            "filename": output.name,
            "bytes": output.stat().st_size,
            "mib": output.stat().st_size / 1024**2,
            "sha256": sha256(output),
            "members": list(OUTPUT_MEMBERS),
            "below_256_mib": output.stat().st_size < 256 * 1024**2,
        },
        "private_source_paths_embedded": False,
        "official_submission_performed": False,
    }
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = build_archive(
        base_archive=args.base_archive.resolve(),
        base_archive_sha256=args.base_archive_sha256,
        base_dense_checkpoint=args.base_dense_checkpoint.resolve(),
        base_dense_checkpoint_sha256=args.base_dense_checkpoint_sha256,
        base_model_state_sha256=args.base_model_state_sha256,
        adapter_endpoint=args.adapter_endpoint.resolve(),
        adapter_endpoint_sha256=args.adapter_endpoint_sha256,
        output=args.output.resolve(), report=args.report.resolve(),
        max_size_mib=args.max_size_mib,
        adapter_storage=args.adapter_storage)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
