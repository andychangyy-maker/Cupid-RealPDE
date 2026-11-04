#!/usr/bin/env python3
"""Build the full-Real point + fresh whole-AoA OOF Native-UQ candidate.

The builder deliberately takes three independently SHA-bound inputs:

* the all-81/full-trajectory point-stage result;
* the fixed-u3200 fresh Native-UQ refit result; and
* a previously timed B210 runtime archive.

It reconstructs the dense state with non-UQ tensors from the point checkpoint
and UQ tensors from the refit endpoint, verifies that this is bitwise equal to
the refit's complete endpoint, and only then applies the established INT8
storage / FP16 inference transform.  Runtime source files are copied byte for
byte from the timed archive, so packaging cannot silently change batch size or
import-time CUDA prewarming.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
from typing import Any, Mapping
import zipfile

import torch

from build_track1_blended_ensemble_unet2d_package import compact_state_dict


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


CHECKPOINT_MEMBER = "ensemble_unet2d.pth"
EXPECTED_MEMBERS = (
    "submission.py",
    "models/__init__.py",
    "models/unet2d.py",
    "models/ensemble_unet2d.py",
    CHECKPOINT_MEMBER,
)
RUNTIME_MEMBERS = EXPECTED_MEMBERS[:-1]
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

POINT_RESULT_SCHEMA = "b210_full_released_real_final_point_v1"
POINT_ENDPOINT_SCHEMA = "b210_full_released_real_final_endpoint_v1"
UQ_RESULT_SCHEMA = "b210_full_real_oof_native_uq_run_v1"
UQ_ENDPOINT_SCHEMA = "b210_full_real_oof_native_uq_refit_v1"
UQ_ENDPOINT = "fixed_full_oof_u3200"
STORAGE_FORMAT = "symmetric_per_dim0_int8_v1"
SCHEMA = "b210_full_real_oof_native_uq_package_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor state independently of the torch.save container."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
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


def normalized_sha256(value: str, label: str) -> str:
    result = value.strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef"
                                for character in result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def is_uq_key(key: str) -> bool:
    return key.startswith(UQ_PREFIXES)


def _json_native(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_json_native(item) for item in value]
    if isinstance(value, list):
        return [_json_native(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    return value


def _tensor_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return torch.equal(torch.as_tensor(left), torch.as_tensor(right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _tensor_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _tensor_equal(a, b) for a, b in zip(left, right))
    return left == right


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> dict:
    expected = normalized_sha256(expected_sha256, f"{label} SHA-256")
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"{label} file SHA-256 changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} is not a JSON object")
    return value


def _verify_json_self_seal(payload: Mapping[str, object]) -> str:
    normalized = json.loads(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    seal = normalized.pop("json_self_seal", None)
    if (not isinstance(seal, Mapping)
            or seal.get("algorithm") != "sha256"
            or seal.get("computed_after_json_roundtrip") is not True):
        raise RuntimeError("fresh-UQ JSON round-trip self-seal is missing")
    digest = canonical_sha256(normalized)
    if digest != seal.get("payload_without_self_seal_sha256"):
        raise RuntimeError("fresh-UQ JSON round-trip self-seal changed")
    return digest


def _load_point_source(
    result_path: Path, result_sha256: str,
) -> tuple[dict, dict, Path, dict]:
    result = _load_bound_json(result_path, result_sha256, "point result")
    endpoints = result.get("endpoints")
    row = endpoints.get("ema_u4000") if isinstance(endpoints, Mapping) else None
    seal = row.get("seal") if isinstance(row, Mapping) else None
    if (result.get("schema") != POINT_RESULT_SCHEMA
            or result.get("primary_endpoint") != "ema_u4000"
            or result.get("fresh_native_uq_crossfit_refit_required") is not True
            or result.get("submission_authorized") is not False
            or not isinstance(seal, Mapping)):
        raise RuntimeError("point-stage result contract changed")
    checkpoint = Path(str(seal.get("path", ""))).resolve()
    if (not checkpoint.is_file()
            or sha256(checkpoint) != seal.get("file_sha256")):
        raise RuntimeError("point EMA-u4000 checkpoint SHA-256 changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != POINT_ENDPOINT_SCHEMA
            or payload.get("endpoint") != "ema_u4000"
            or payload.get("primary") is not True
            or payload.get("training_scope") != "all_clean81_full13202"
            or int(payload.get("iteration", -1)) != 4000
            or int(payload.get("optimizer_windows", -1)) != 13_202
            or int(payload.get("optimizer_stride", -1)) != 5
            or int(payload.get("normalizer_windows", -1)) != 2_544
            or int(payload.get("normalizer_stride", -1)) != 20
            or payload.get("fresh_native_uq_crossfit_refit_required") is not True
            or payload.get("submission_authorized") is not False
            or not isinstance(state, Mapping)):
        raise RuntimeError("point EMA-u4000 checkpoint contract changed")
    state_digest = model_state_sha256(state)
    if (state_digest != seal.get("model_state_sha256")
            or state_digest != payload.get("model_state_sha256")):
        raise RuntimeError("point EMA-u4000 tensor-state SHA-256 changed")
    if canonical_sha256(payload["model_config"]) != seal.get(
            "model_config_sha256"):
        raise RuntimeError("point model-config SHA-256 changed")
    if canonical_sha256(_json_native(payload["normalizer"])) != seal.get(
            "normalizer_sha256"):
        raise RuntimeError("point normalizer SHA-256 changed")
    return result, payload, checkpoint, dict(seal)


def _load_uq_source(
    result_path: Path, result_sha256: str, point_result_sha256: str,
) -> tuple[dict, dict, Path, dict]:
    result = _load_bound_json(result_path, result_sha256, "fresh-UQ result")
    _verify_json_self_seal(result)
    seal = result.get("endpoint")
    point = result.get("point")
    verification = result.get("endpoint_verification")
    if (result.get("schema") != UQ_RESULT_SCHEMA
            or result.get("decision")
            != "COMPLETE_FIXED_FULL_OOF_NATIVE_UQ_U3200"
            or result.get("primary_endpoint") != UQ_ENDPOINT
            or int(result.get("fixed_cycles", -1)) != 4
            or int(result.get("updates_per_cycle", -1)) != 800
            or int(result.get("total_updates", -1)) != 3200
            or result.get("point_state_bitwise") is not True
            or result.get("checkpoint_selection") is not False
            or result.get("clean716_manifest_read") is not False
            or result.get("evaluation_manifest_read") is not False
            or result.get("submission_authorized") is not False
            or not isinstance(seal, Mapping)
            or not isinstance(point, Mapping)
            or not isinstance(verification, Mapping)
            or point.get("result_sha256") != point_result_sha256
            or verification.get("exact") is not True
            or verification.get("file_sha256") != seal.get("file_sha256")):
        raise RuntimeError("fresh-UQ refit result contract changed")
    endpoint = Path(str(seal.get("path", ""))).resolve()
    if not endpoint.is_file() or sha256(endpoint) != seal.get("file_sha256"):
        raise RuntimeError("fresh-UQ endpoint SHA-256 changed")
    payload = torch.load(endpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    point_report = payload.get("point_checkpoint")
    uq_initialization = payload.get("uq_initialization")
    if (payload.get("schema") != UQ_ENDPOINT_SCHEMA
            or payload.get("endpoint") != UQ_ENDPOINT
            or payload.get("primary") is not True
            or int(payload.get("iteration", -1)) != 3200
            or payload.get("point_state_bitwise") is not True
            or payload.get("checkpoint_selection") is not False
            or payload.get("clean716_manifest_read") is not False
            or payload.get("evaluation_manifest_read") is not False
            or payload.get("submission_authorized") is not False
            or not isinstance(state, Mapping)
            or not isinstance(point_report, Mapping)
            or not isinstance(uq_initialization, Mapping)
            or point_report.get("result_sha256") != point_result_sha256
            or uq_initialization.get("full_real_uq_state_inherited") is not False):
        raise RuntimeError("fresh-UQ endpoint contract changed")
    state_digest = model_state_sha256(state)
    if (state_digest != seal.get("model_state_sha256")
            or state_digest != payload.get("model_state_sha256")):
        raise RuntimeError("fresh-UQ endpoint tensor-state SHA-256 changed")
    return result, payload, endpoint, dict(seal)


def _normalized_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    # Importing the dataclass has negligible cost and avoids allocating B210.
    from models.ensemble_unet2d import EnsembleUNet2dConfig

    values = dict(config)
    architecture = values.pop("architecture", "ensemble_unet")
    if architecture != "ensemble_unet":
        raise ValueError("model architecture is not ensemble_unet")
    return {
        "architecture": architecture,
        **_json_native(asdict(EnsembleUNet2dConfig(**values))),
    }


def _load_runtime_archive(
    path: Path, expected_sha256: str, normalized_source_config: Mapping[str, Any],
    source_normalizer: Mapping[str, Any], source_keys: set[str],
) -> tuple[dict[str, bytes], dict, dict]:
    expected = normalized_sha256(expected_sha256, "runtime archive SHA-256")
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError("runtime archive SHA-256 changed")
    with zipfile.ZipFile(path) as archive:
        if tuple(archive.namelist()) != EXPECTED_MEMBERS:
            raise RuntimeError("runtime archive member layout changed")
        if archive.testzip() is not None:
            raise RuntimeError("runtime archive is corrupt")
        files = {name: archive.read(name) for name in EXPECTED_MEMBERS}
    payload = torch.load(
        io.BytesIO(files[CHECKPOINT_MEMBER]),
        map_location="cpu",
        weights_only=False,
    )
    stored = payload.get("quantized_state_dict")
    if (payload.get("storage_format") != STORAGE_FORMAT
            or payload.get("inference_dtype") != "float16"
            or not isinstance(stored, Mapping)
            or set(stored) != source_keys
            or _normalized_model_config(payload["model_config"])
            != dict(normalized_source_config)
            or not _tensor_equal(payload.get("normalizer"), source_normalizer)):
        raise RuntimeError("runtime archive is incompatible with source model")
    wrapper = files["submission.py"].decode("utf-8")
    if ("_BATCH_SIZE = 16" not in wrapper
            or "def _prewarm_cuda_runtime():" not in wrapper
            or "_prewarm_cuda_runtime()" not in wrapper
            or "dummy = np.zeros((_BATCH_SIZE, 20, 32, 64, 3)" not in wrapper):
        raise RuntimeError("runtime archive lost B16 import-prewarm contract")
    runtime_report = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": expected,
        "members": {
            name: {"bytes": len(files[name]), "sha256": bytes_sha256(files[name])}
            for name in EXPECTED_MEMBERS
        },
        "batch_size": 16,
        "cuda_import_prewarm": True,
        "inference_dtype": "float16",
        "storage_format": STORAGE_FORMAT,
        "normalized_config_sha256": canonical_sha256(
            normalized_source_config),
    }
    return files, payload, runtime_report


def _audit_compaction(
    source: Mapping[str, torch.Tensor], compact: Mapping[str, object],
) -> dict[str, object]:
    if set(source) != set(compact):
        raise RuntimeError("compact tensor-state keys changed")
    maximum_error = 0.0
    quantized_tensors = 0
    plain_fp16_tensors = 0
    for key, original in source.items():
        stored = compact[key]
        if isinstance(stored, Mapping) and "quantized" in stored:
            restored = stored["quantized"].float() * stored["scale"].float()
            error = (restored - original.detach().cpu().float()).abs()
            bound = stored["scale"].float() / 2.0 + 1.0e-6
            if not bool(torch.all(error <= bound)):
                raise RuntimeError(f"INT8 error bound failed at {key}")
            maximum_error = max(maximum_error, float(error.max()))
            quantized_tensors += 1
        elif torch.is_floating_point(original):
            if not isinstance(stored, torch.Tensor) or stored.dtype != torch.float16:
                raise RuntimeError(f"plain floating tensor is not FP16 at {key}")
            if not torch.equal(stored, original.detach().cpu().to(torch.float16)):
                raise RuntimeError(f"plain FP16 tensor changed at {key}")
            plain_fp16_tensors += 1
        elif not torch.equal(stored, original.detach().cpu()):
            raise RuntimeError(f"non-floating tensor changed at {key}")
    return {
        "per_dim0_int8_error_bound": True,
        "maximum_int8_abs_error": maximum_error,
        "quantized_tensors_rechecked": quantized_tensors,
        "plain_fp16_tensors_rechecked": plain_fp16_tensors,
    }


def build_candidate(
    *,
    point_result: Path,
    point_result_sha256: str,
    uq_result: Path,
    uq_result_sha256: str,
    runtime_archive: Path,
    runtime_archive_sha256: str,
    output: Path | None,
    report: Path | None,
    quantize_min_elements: int = 1024,
    max_size_mib: float = 256.0,
    dry_run: bool = False,
) -> dict[str, object]:
    if quantize_min_elements <= 0:
        raise ValueError("quantize-min-elements must be positive")
    if not dry_run and (output is None or report is None):
        raise ValueError("output and report are required outside dry-run")
    point_result_sha256 = normalized_sha256(
        point_result_sha256, "point result SHA-256")
    uq_result_sha256 = normalized_sha256(
        uq_result_sha256, "fresh-UQ result SHA-256")
    point_json, point_payload, point_checkpoint, point_seal = _load_point_source(
        point_result.resolve(), point_result_sha256)
    uq_json, uq_payload, uq_checkpoint, uq_seal = _load_uq_source(
        uq_result.resolve(), uq_result_sha256, point_result_sha256)
    del point_json, uq_json

    point_state = point_payload["model_state_dict"]
    uq_state = uq_payload["model_state_dict"]
    if set(point_state) != set(uq_state):
        raise RuntimeError("point and fresh-UQ endpoint state keys differ")
    uq_keys = sorted(key for key in uq_state if is_uq_key(key))
    point_keys = sorted(key for key in uq_state if not is_uq_key(key))
    if not uq_keys or not point_keys:
        raise RuntimeError("point/UQ tensor partition is empty")
    point_partition = {key: point_state[key] for key in point_keys}
    endpoint_point_partition = {key: uq_state[key] for key in point_keys}
    if any(not torch.equal(point_partition[key], endpoint_point_partition[key])
           for key in point_keys):
        raise RuntimeError("fresh-UQ endpoint changed the point partition")
    point_partition_sha = model_state_sha256(point_partition)
    if (point_partition_sha != uq_payload.get("point_partition_sha256")
            or point_partition_sha != uq_seal.get("point_partition_sha256")):
        raise RuntimeError("fresh-UQ point-partition seal changed")
    if (not _tensor_equal(point_payload["normalizer"], uq_payload["normalizer"])
            or _normalized_model_config(point_payload["model_config"])
            != _normalized_model_config(uq_payload["model_config"])):
        raise RuntimeError("point and fresh-UQ endpoint metadata differ")

    # Reconstruct explicitly instead of trusting the endpoint's complete state.
    composed = {
        key: uq_state[key] if is_uq_key(key) else point_state[key]
        for key in uq_state
    }
    if any(not torch.equal(composed[key], uq_state[key]) for key in composed):
        raise RuntimeError("explicit point/UQ composition differs from endpoint")
    composed_sha = model_state_sha256(composed)
    if composed_sha != uq_seal.get("model_state_sha256"):
        raise RuntimeError("explicit composition tensor-state SHA-256 changed")

    normalized_config = _normalized_model_config(uq_payload["model_config"])
    runtime_files, runtime_payload, runtime_report = _load_runtime_archive(
        runtime_archive.resolve(), runtime_archive_sha256,
        normalized_config, point_payload["normalizer"], set(composed))
    compact_state, quantization = compact_state_dict(
        composed, quantize_min_elements)
    quantization_audit = _audit_compaction(composed, compact_state)
    compact_payload = {
        "phase": "real",
        # Keep the already-timed runtime's compact config representation.  It
        # was checked above after expansion through the current dataclass.
        "model_config": runtime_payload["model_config"],
        "quantized_state_dict": compact_state,
        "normalizer": point_payload["normalizer"],
        "inference_dtype": "float16",
        "storage_format": STORAGE_FORMAT,
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(compact_payload, checkpoint_buffer)
    checkpoint_bytes = checkpoint_buffer.getvalue()

    result: dict[str, object] = {
        "schema": SCHEMA,
        "dry_run": bool(dry_run),
        "point_source": {
            "result": str(point_result.resolve()),
            "result_sha256": point_result_sha256,
            "checkpoint": str(point_checkpoint),
            "checkpoint_sha256": point_seal["file_sha256"],
            "model_state_sha256": point_seal["model_state_sha256"],
            "point_partition_sha256": point_partition_sha,
        },
        "uq_source": {
            "result": str(uq_result.resolve()),
            "result_sha256": uq_result_sha256,
            "checkpoint": str(uq_checkpoint),
            "checkpoint_sha256": uq_seal["file_sha256"],
            "model_state_sha256": uq_seal["model_state_sha256"],
            "uq_partition_sha256": model_state_sha256({
                key: uq_state[key] for key in uq_keys}),
        },
        "composition": {
            "state_keys": len(composed),
            "point_keys": len(point_keys),
            "uq_keys": len(uq_keys),
            "point_source_exact_before_compaction": True,
            "uq_source_exact_before_compaction": True,
            "equals_complete_refit_endpoint_bitwise": True,
            "dense_model_state_sha256": composed_sha,
            "normalized_model_config_sha256": canonical_sha256(
                normalized_config),
            "normalizer_sha256": canonical_sha256(
                _json_native(point_payload["normalizer"])),
        },
        "runtime_template": runtime_report,
        "storage": {
            "format": STORAGE_FORMAT,
            "inference_dtype": "float16",
            "quantize_min_elements": quantize_min_elements,
            "checkpoint_bytes": len(checkpoint_bytes),
            "checkpoint_sha256": bytes_sha256(checkpoint_bytes),
            "quantization": quantization,
            "audit": quantization_audit,
        },
        "archive": None,
        "runtime_members_bitwise_identical": {
            name: True for name in RUNTIME_MEMBERS
        },
        "only_checkpoint_member_changed": True,
        "source_vs_zip_prediction_replay_required": not dry_run,
        "a800_abba_latency_required": not dry_run,
        "official_submission_performed": False,
        "submission_authorized": False,
    }
    if dry_run:
        return result

    assert output is not None and report is not None
    output = output.resolve()
    report = report.resolve()
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite package output or report")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED,
            compresslevel=6, allowZip64=True,
        ) as archive:
            for name in EXPECTED_MEMBERS:
                value = (checkpoint_bytes if name == CHECKPOINT_MEMBER
                         else runtime_files[name])
                archive.writestr(name, value)
        if temporary.stat().st_size > max_size_mib * 1024**2:
            raise RuntimeError("candidate archive exceeds size limit")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    with zipfile.ZipFile(output) as archive:
        if (tuple(archive.namelist()) != EXPECTED_MEMBERS
                or archive.testzip() is not None
                or archive.read(CHECKPOINT_MEMBER) != checkpoint_bytes
                or any(archive.read(name) != runtime_files[name]
                       for name in RUNTIME_MEMBERS)):
            raise RuntimeError("candidate archive round-trip verification failed")
    result["archive"] = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "under_256_mib": output.stat().st_size <= 256 * 1024**2,
        "members": list(EXPECTED_MEMBERS),
    }
    temporary_report = report.with_name(report.name + ".tmp")
    temporary_report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_report.replace(report)
    return result


def contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "point_result_schema": POINT_RESULT_SCHEMA,
        "point_endpoint_schema": POINT_ENDPOINT_SCHEMA,
        "uq_result_schema": UQ_RESULT_SCHEMA,
        "uq_endpoint_schema": UQ_ENDPOINT_SCHEMA,
        "uq_endpoint": UQ_ENDPOINT,
        "uq_prefixes": list(UQ_PREFIXES),
        "storage_format": STORAGE_FORMAT,
        "inference_dtype": "float16",
        "runtime_batch_size": 16,
        "runtime_cuda_import_prewarm": True,
        "runtime_source": "byte-preserved SHA-bound timed B210 archive",
        "official_submission_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--point-result", type=Path)
    parser.add_argument("--point-result-sha256")
    parser.add_argument("--uq-result", type=Path)
    parser.add_argument("--uq-result-sha256")
    parser.add_argument("--runtime-archive", type=Path)
    parser.add_argument("--runtime-archive-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quantize-min-elements", type=int, default=1024)
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.contract_only:
        execution_values = (
            args.point_result, args.point_result_sha256,
            args.uq_result, args.uq_result_sha256,
            args.runtime_archive, args.runtime_archive_sha256,
            args.output, args.report,
        )
        if args.dry_run or any(value is not None for value in execution_values):
            raise ValueError("contract-only mode received execution arguments")
        print(json.dumps(contract(), indent=2, sort_keys=True))
        return
    required = (
        args.point_result, args.point_result_sha256,
        args.uq_result, args.uq_result_sha256,
        args.runtime_archive, args.runtime_archive_sha256,
    )
    if any(value is None for value in required):
        raise ValueError("package build is missing a required SHA-bound input")
    result = build_candidate(
        point_result=args.point_result,
        point_result_sha256=args.point_result_sha256,
        uq_result=args.uq_result,
        uq_result_sha256=args.uq_result_sha256,
        runtime_archive=args.runtime_archive,
        runtime_archive_sha256=args.runtime_archive_sha256,
        output=args.output,
        report=args.report,
        quantize_min_elements=args.quantize_min_elements,
        max_size_mib=args.max_size_mib,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
