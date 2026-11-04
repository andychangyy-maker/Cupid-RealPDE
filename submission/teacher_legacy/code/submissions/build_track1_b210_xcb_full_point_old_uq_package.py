#!/usr/bin/env python3
"""Package the final full-Real XCB point with the frozen old XCB UQ state.

This is intentionally XCB-specific.  It accepts one sealed dense FP32 XCB
point endpoint and the exact existing XCB Native-UQ800 archive.  Non-UQ state
is compacted from the new endpoint; all 77 UQ compact entries and all runtime
source files are copied bit for bit from the old archive.  It never opens any
training or evaluation data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import zipfile

import torch


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import build_track1_b210_full_oof_native_uq_package as full_package
import build_track1_b210_strict_point_uq_factorial as strict_factorial
from build_track1_blended_ensemble_unet2d_package import compact_state_dict
import build_track1_point_uq_hybrid_archive as hybrid


SCHEMA = "b210_xcb_full_point_old_uq_package_v1"
POINT_ENDPOINT_SCHEMA = "b210_xcb_full_released_real_final_endpoint_v1"
POINT_EXPERIMENT = "b210_xcb_full_released_real_final_point"
START_XCB_SIM_SHA256 = (
    "06ade419811d722e58235acdcb34e480dd6ae88555dd52037ebcb6bbeba2ce99"
)
OLD_XCB_ARCHIVE_SHA256 = (
    "b77cc5e00f3137cb9861f55e2662da31286b071b2c8dee79499f0332a86f400c"
)
EXPECTED_STATE_KEYS = 396
EXPECTED_POINT_KEYS = 319
EXPECTED_UQ_KEYS = 77
EXPECTED_POINT_NUMEL = 176_872_958
EXPECTED_UQ_NUMEL = 3_680_422
RUNTIME_MEMBERS = hybrid.EXPECTED_MEMBERS[:-1]


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compose_compact_state(
    dense_state: Mapping[str, torch.Tensor],
    old_compact_state: Mapping[str, Any],
    quantize_min_elements: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(dense_state) != set(old_compact_state):
        raise ValueError("dense XCB and old archive state keys differ")
    point_dense = {
        key: dense_state[key]
        for key in old_compact_state
        if not hybrid.is_uq_key(key)
    }
    compact_point, quantization = compact_state_dict(
        point_dense, quantize_min_elements
    )
    composed = {
        key: (
            old_compact_state[key]
            if hybrid.is_uq_key(key)
            else compact_point[key]
        )
        for key in old_compact_state
    }
    point_source_exact = all(
        hybrid._tensor_equal(composed[key], compact_point[key])
        for key in composed
        if not hybrid.is_uq_key(key)
    )
    uq_source_exact = all(
        hybrid._tensor_equal(composed[key], old_compact_state[key])
        for key in composed
        if hybrid.is_uq_key(key)
    )
    if not point_source_exact or not uq_source_exact:
        raise RuntimeError("XCB point/UQ compact composition changed a source")
    audit = {
        "point_source_exact_after_compaction": point_source_exact,
        "old_uq_source_bitwise_exact": uq_source_exact,
        "point_state_sha256": hybrid.state_signature(composed, uq=False),
        "uq_state_sha256": hybrid.state_signature(composed, uq=True),
        "point_compaction": quantization,
        "point_compaction_audit": full_package._audit_compaction(
            point_dense, compact_point
        ),
    }
    return composed, audit


def _load_point_endpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = full_package.normalized_sha256(
        expected_sha256, "XCB point endpoint SHA-256"
    )
    if not path.is_file() or hybrid.sha256(path) != expected:
        raise RuntimeError("XCB full-Real point endpoint SHA-256 changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    required = {
        "schema": POINT_ENDPOINT_SCHEMA,
        "experiment": POINT_EXPERIMENT,
        "phase": "real",
        "training_scope": "all_clean81_full13202",
        "iteration": 4000,
        "endpoint": "ema_u4000",
        "primary": True,
        "held_aoa": None,
        "optimizer_windows": 13_202,
        "optimizer_stride": 5,
        "normalizer_windows": 2_544,
        "normalizer_stride": 20,
        "optimizer_state_saved": False,
        "endpoint_selection_allowed": False,
        "fresh_native_uq_crossfit_refit_required": True,
        "submission_authorized": False,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise RuntimeError("XCB full-Real point endpoint contract changed")
    if not isinstance(state, Mapping) or len(state) != EXPECTED_STATE_KEYS:
        raise RuntimeError("XCB full-Real point tensor state is invalid")
    if any(
        tensor.is_floating_point() and tensor.dtype != torch.float32
        for tensor in state.values()
        if torch.is_tensor(tensor)
    ):
        raise TypeError("XCB full-Real point endpoint is not strict FP32")
    state_sha = full_package.model_state_sha256(state)
    if state_sha != payload.get("model_state_sha256"):
        raise RuntimeError("XCB point model-state SHA-256 changed")
    if payload.get("start", {}).get("checkpoint_sha256") != START_XCB_SIM_SHA256:
        raise RuntimeError("XCB point endpoint has the wrong Sim start")
    return payload


def _validate_xcb_compatibility(
    point_payload: Mapping[str, Any], old_payload: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_config = full_package._normalized_model_config(
        old_payload["model_config"]
    )
    point_config = full_package._normalized_model_config(
        point_payload["model_config"]
    )
    if point_config != old_config:
        raise ValueError("new point and old XCB runtime configs differ")
    if (
        point_config.get("architecture") != "ensemble_unet"
        or point_config.get("cross_conditioned_backbone") is not True
        or int(point_config.get("member_adapter_channels", -1)) != 64
    ):
        raise ValueError("sources are not the frozen B210 XCB/member architecture")
    if not hybrid._tensor_equal(
        point_payload.get("normalizer"), old_payload.get("normalizer")
    ):
        raise ValueError("new point and old XCB normalizers differ")
    state = point_payload["model_state_dict"]
    old_state = old_payload["quantized_state_dict"]
    if set(state) != set(old_state) or len(old_state) != EXPECTED_STATE_KEYS:
        raise ValueError("new point and old XCB state layouts differ")
    point_keys = [key for key in old_state if not hybrid.is_uq_key(key)]
    uq_keys = [key for key in old_state if hybrid.is_uq_key(key)]
    point_numel = sum(int(state[key].numel()) for key in point_keys)
    uq_numel = sum(int(state[key].numel()) for key in uq_keys)
    if (
        len(point_keys) != EXPECTED_POINT_KEYS
        or len(uq_keys) != EXPECTED_UQ_KEYS
        or point_numel != EXPECTED_POINT_NUMEL
        or uq_numel != EXPECTED_UQ_NUMEL
    ):
        raise RuntimeError("XCB point/UQ responsibility boundary changed")
    return point_config, {
        "state_keys": len(old_state),
        "point_keys": len(point_keys),
        "uq_keys": len(uq_keys),
        "point_numel": point_numel,
        "uq_numel": uq_numel,
        "uq_prefixes": list(hybrid.UQ_PREFIXES),
    }


def build_candidate(
    *, point_checkpoint: Path, point_checkpoint_sha256: str,
    old_xcb_archive: Path, old_xcb_archive_sha256: str,
    output: Path, report: Path, quantize_min_elements: int = 1024,
    max_size_mib: float = 256.0,
) -> dict[str, Any]:
    if quantize_min_elements <= 0:
        raise ValueError("quantize-min-elements must be positive")
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite XCB output or report")
    old_expected = full_package.normalized_sha256(
        old_xcb_archive_sha256, "old XCB archive SHA-256"
    )
    if old_expected != OLD_XCB_ARCHIVE_SHA256:
        raise RuntimeError("old XCB archive is not the frozen source identity")
    if (
        not old_xcb_archive.is_file()
        or hybrid.sha256(old_xcb_archive) != old_expected
    ):
        raise RuntimeError("old XCB archive SHA-256 changed")
    runtime_files, old_payload = hybrid._load_archive(old_xcb_archive)
    point_payload = _load_point_endpoint(
        point_checkpoint, point_checkpoint_sha256
    )
    normalized_config, partition = _validate_xcb_compatibility(
        point_payload, old_payload
    )
    old_state = old_payload["quantized_state_dict"]
    composed, composition = compose_compact_state(
        point_payload["model_state_dict"], old_state,
        quantize_min_elements,
    )
    old_vs_candidate = strict_factorial.compare_states(
        old_state, composed
    )
    if old_vs_candidate["summary"]["uq"]["changed_keys"] != 0:
        raise RuntimeError("old XCB UQ compact state was not preserved")
    payload = {
        "phase": "real",
        # Preserve the exact config representation already proven with the
        # runtime source; compatibility was checked after dataclass expansion.
        "model_config": old_payload["model_config"],
        "quantized_state_dict": composed,
        "normalizer": old_payload["normalizer"],
        "inference_dtype": "float16",
        "storage_format": "symmetric_per_dim0_int8_v1",
    }
    archive = strict_factorial._write_archive(
        output, runtime_files, payload, max_size_mib
    )
    with zipfile.ZipFile(output) as built:
        roundtrip = torch.load(
            io.BytesIO(built.read(hybrid.CHECKPOINT_MEMBER)),
            map_location="cpu", weights_only=False,
        )
        for name in RUNTIME_MEMBERS:
            if built.read(name) != runtime_files[name]:
                raise RuntimeError(f"XCB runtime member changed: {name}")
    if not all(
        hybrid._tensor_equal(
            roundtrip["quantized_state_dict"][key], composed[key]
        )
        for key in composed
    ):
        raise RuntimeError("XCB diagnostic archive state changed on round-trip")

    result = {
        "schema": SCHEMA,
        "evaluation_data_read": False,
        "clean716_read": False,
        "point_source": {
            "path": str(point_checkpoint.resolve()),
            "sha256": hybrid.sha256(point_checkpoint),
            "model_state_sha256": point_payload["model_state_sha256"],
            "training_scope": point_payload["training_scope"],
            "optimizer_windows": point_payload["optimizer_windows"],
            "normalizer_windows": point_payload["normalizer_windows"],
            "start_checkpoint_sha256": START_XCB_SIM_SHA256,
        },
        "old_uq_runtime_source": {
            "path": str(old_xcb_archive.resolve()),
            "sha256": old_expected,
            "compact_uq_state_sha256": hybrid.state_signature(
                old_state, uq=True
            ),
            "runtime_member_sha256": {
                name: bytes_sha256(runtime_files[name])
                for name in RUNTIME_MEMBERS
            },
        },
        "architecture": {
            "normalized_config_sha256": full_package.canonical_sha256(
                normalized_config
            ),
            "cross_conditioned_backbone": True,
            "member_adapter_channels": 64,
            **partition,
        },
        "composition": composition,
        "old_archive_vs_candidate_state": old_vs_candidate,
        "archive": archive,
        "runtime_members_bitwise_identical": {
            name: True for name in RUNTIME_MEMBERS
        },
        "only_checkpoint_member_changed": True,
        "official_submission_performed": False,
        "submission_authorized": False,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_name(report.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--point-checkpoint-sha256", required=True)
    parser.add_argument("--old-xcb-archive", type=Path, required=True)
    parser.add_argument("--old-xcb-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--quantize-min-elements", type=int, default=1024)
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_candidate(
        point_checkpoint=args.point_checkpoint.resolve(),
        point_checkpoint_sha256=args.point_checkpoint_sha256,
        old_xcb_archive=args.old_xcb_archive.resolve(),
        old_xcb_archive_sha256=args.old_xcb_archive_sha256,
        output=args.output.resolve(),
        report=args.report.resolve(),
        quantize_min_elements=args.quantize_min_elements,
        max_size_mib=args.max_size_mib,
    )
    print(json.dumps({
        "schema": result["schema"],
        "point_source": result["point_source"],
        "old_uq_runtime_source": result["old_uq_runtime_source"],
        "architecture": result["architecture"],
        "composition": result["composition"],
        "archive": result["archive"],
        "report": str(args.report.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
