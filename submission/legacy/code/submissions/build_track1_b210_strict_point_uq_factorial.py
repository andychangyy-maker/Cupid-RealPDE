#!/usr/bin/env python3
"""Build and attest a strict B210 point x UQ archive factorial.

The two input archives each contain both responsibilities.  This builder makes
the four actual tensor combinations below while copying one runtime template
byte for byte:

* AA: old point + old UQ
* DA: new point + old UQ
* AD: old point + fresh UQ
* DD: new point + fresh UQ

No evaluation data are opened by this program.  The resulting report contains
per-tensor compact-state and dequantized-value differences, so a later scorer
can distinguish a genuine factorial from four renamed whole archives.
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

import build_track1_point_uq_hybrid_archive as hybrid


SCHEMA = "b210_strict_point_uq_factorial_archive_v1"
ARM_SOURCES = {
    "AA": ("old", "old"),
    "DA": ("new", "old"),
    "AD": ("old", "fresh"),
    "DD": ("new", "fresh"),
}
RUNTIME_MEMBERS = hybrid.EXPECTED_MEMBERS[:-1]


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compact_value_equal(left: Any, right: Any) -> bool:
    return hybrid._tensor_equal(left, right)


def compact_value_signature(value: Any) -> str:
    digest = hashlib.sha256()
    hybrid._update_signature(digest, value)
    return digest.hexdigest()


def materialize_compact_value(value: Any) -> torch.Tensor:
    if isinstance(value, Mapping):
        if set(value) != {"quantized", "scale"}:
            raise ValueError("unsupported compact tensor mapping")
        quantized = value["quantized"]
        scale = value["scale"]
        if not torch.is_tensor(quantized) or not torch.is_tensor(scale):
            raise TypeError("invalid quantized compact tensor")
        return quantized.detach().cpu().float() * scale.detach().cpu().float()
    if not torch.is_tensor(value):
        raise TypeError("compact state entry is not a tensor")
    return value.detach().cpu().float()


def compact_value_description(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        tensor = materialize_compact_value(value)
        return {
            "storage": "per_dim0_int8",
            "shape": list(tensor.shape),
            "quantized_dtype": str(value["quantized"].dtype),
            "scale_dtype": str(value["scale"].dtype),
            "numel": tensor.numel(),
        }
    if not torch.is_tensor(value):
        raise TypeError("compact state entry is not a tensor")
    return {
        "storage": "plain_tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": value.numel(),
    }


def compare_states(
    old_state: Mapping[str, Any], new_state: Mapping[str, Any]
) -> dict[str, Any]:
    if tuple(old_state) != tuple(new_state):
        raise ValueError("old/new compact state-key order differs")
    rows: list[dict[str, Any]] = []
    aggregate = {
        "point": {"keys": 0, "changed_keys": 0, "numel": 0,
                  "abs_sum": 0.0, "square_sum": 0.0, "max_abs": 0.0},
        "uq": {"keys": 0, "changed_keys": 0, "numel": 0,
               "abs_sum": 0.0, "square_sum": 0.0, "max_abs": 0.0},
    }
    for key in old_state:
        partition = "uq" if hybrid.is_uq_key(key) else "point"
        old_value = old_state[key]
        new_value = new_state[key]
        old_description = compact_value_description(old_value)
        new_description = compact_value_description(new_value)
        if (old_description["shape"] != new_description["shape"]
                or old_description["numel"] != new_description["numel"]):
            raise ValueError(f"old/new tensor shape differs at {key}")
        exact = compact_value_equal(old_value, new_value)
        old_tensor = materialize_compact_value(old_value)
        new_tensor = materialize_compact_value(new_value)
        difference = new_tensor.sub(old_tensor)
        absolute = difference.abs()
        square_sum = float(torch.sum(difference.double().square()))
        row = {
            "key": key,
            "partition": partition,
            "compact_exact": exact,
            "old_compact_sha256": compact_value_signature(old_value),
            "new_compact_sha256": compact_value_signature(new_value),
            "old": old_description,
            "new": new_description,
            "dequantized_difference": {
                "mean_abs": float(absolute.double().mean()),
                "rms": (square_sum / max(difference.numel(), 1)) ** 0.5,
                "max_abs": float(absolute.max()) if absolute.numel() else 0.0,
            },
        }
        rows.append(row)
        stats = aggregate[partition]
        stats["keys"] += 1
        stats["changed_keys"] += int(not exact)
        stats["numel"] += difference.numel()
        stats["abs_sum"] += float(torch.sum(absolute.double()))
        stats["square_sum"] += square_sum
        stats["max_abs"] = max(stats["max_abs"], row["dequantized_difference"]["max_abs"])
        del old_tensor, new_tensor, difference, absolute

    summaries = {}
    for partition, raw in aggregate.items():
        count = max(int(raw["numel"]), 1)
        summaries[partition] = {
            "keys": raw["keys"],
            "changed_keys": raw["changed_keys"],
            "unchanged_keys": raw["keys"] - raw["changed_keys"],
            "numel": raw["numel"],
            "mean_abs": raw["abs_sum"] / count,
            "rms": (raw["square_sum"] / count) ** 0.5,
            "max_abs": raw["max_abs"],
        }
    return {"summary": summaries, "tensors": rows}


def _validate_compatible_sources(
    old_payload: Mapping[str, Any], new_payload: Mapping[str, Any],
    runtime_payload: Mapping[str, Any],
) -> None:
    for label, payload in (
        ("old", old_payload), ("new", new_payload),
        ("runtime", runtime_payload),
    ):
        if payload.get("storage_format") != "symmetric_per_dim0_int8_v1":
            raise ValueError(f"{label} archive has unsupported storage format")
        if payload.get("inference_dtype") != "float16":
            raise ValueError(f"{label} archive is not FP16 inference")
    old_config = hybrid.normalized_model_config(old_payload["model_config"])
    new_config = hybrid.normalized_model_config(new_payload["model_config"])
    runtime_config = hybrid.normalized_model_config(runtime_payload["model_config"])
    if not old_config == new_config == runtime_config:
        raise ValueError("old/new/runtime model configurations differ")
    if not (
        compact_value_equal(old_payload["normalizer"], new_payload["normalizer"])
        and compact_value_equal(old_payload["normalizer"], runtime_payload["normalizer"])
    ):
        raise ValueError("old/new/runtime normalizers differ")
    states = [
        payload["quantized_state_dict"]
        for payload in (old_payload, new_payload, runtime_payload)
    ]
    if not all(tuple(state) == tuple(states[0]) for state in states[1:]):
        raise ValueError("old/new/runtime compact state-key order differs")


def _write_archive(
    path: Path, runtime_files: Mapping[str, bytes], payload: Mapping[str, Any],
    max_size_mib: float,
) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    checkpoint = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".building")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6,
            allowZip64=True,
        ) as archive:
            for name in hybrid.EXPECTED_MEMBERS:
                archive.writestr(
                    name,
                    checkpoint if name == hybrid.CHECKPOINT_MEMBER
                    else runtime_files[name],
                )
        if temporary.stat().st_size > max_size_mib * 1024**2:
            raise RuntimeError(f"factorial archive exceeds size limit: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    with zipfile.ZipFile(path) as archive:
        if (tuple(archive.namelist()) != hybrid.EXPECTED_MEMBERS
                or archive.testzip() is not None):
            raise RuntimeError(f"factorial archive integrity failed: {path}")
        for name in RUNTIME_MEMBERS:
            if archive.read(name) != runtime_files[name]:
                raise RuntimeError(f"runtime member changed in {path}: {name}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hybrid.sha256(path),
        "checkpoint_member_sha256": bytes_sha256(checkpoint),
        "under_size_limit": path.stat().st_size <= max_size_mib * 1024**2,
    }


def audit_factorial_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(payloads) != set(ARM_SOURCES):
        raise ValueError(f"factorial payloads require {tuple(ARM_SOURCES)}")
    states = {
        arm: payloads[arm]["quantized_state_dict"] for arm in ARM_SOURCES
    }
    if not all(tuple(state) == tuple(states["AA"]) for state in states.values()):
        raise ValueError("factorial arm state-key order differs")

    def partition_equal(left: str, right: str, uq: bool) -> bool:
        return all(
            compact_value_equal(states[left][key], states[right][key])
            for key in states[left]
            if hybrid.is_uq_key(key) == uq
        )

    identities = {
        "AA_point_equals_AD": partition_equal("AA", "AD", False),
        "DA_point_equals_DD": partition_equal("DA", "DD", False),
        "AA_uq_equals_DA": partition_equal("AA", "DA", True),
        "AD_uq_equals_DD": partition_equal("AD", "DD", True),
    }
    contrasts = {
        "old_point_differs_from_new": not partition_equal("AA", "DA", False),
        "old_uq_differs_from_fresh": not partition_equal("AA", "AD", True),
    }
    valid = all(identities.values()) and all(contrasts.values())
    return {
        "identities": identities,
        "contrasts": contrasts,
        "strict_factorial": valid,
    }


def audit_factorial_archives(archives: Mapping[str, Path]) -> dict[str, Any]:
    payloads = {}
    runtime_hashes = {}
    for arm in ARM_SOURCES:
        files, payload = hybrid._load_archive(archives[arm])
        payloads[arm] = payload
        runtime_hashes[arm] = {
            name: bytes_sha256(files[name]) for name in RUNTIME_MEMBERS
        }
    audit = audit_factorial_payloads(payloads)
    audit["runtime_members_identical"] = len(
        {json.dumps(value, sort_keys=True) for value in runtime_hashes.values()}
    ) == 1
    audit["runtime_member_sha256"] = runtime_hashes
    audit["strict_factorial"] = (
        audit["strict_factorial"] and audit["runtime_members_identical"]
    )
    return audit


def build_factorial(
    *, old_archive: Path, new_archive: Path, runtime_archive: Path,
    output_dir: Path, report: Path, max_size_mib: float = 256.0,
) -> dict[str, Any]:
    old_files, old_payload = hybrid._load_archive(old_archive)
    new_files, new_payload = hybrid._load_archive(new_archive)
    runtime_files, runtime_payload = hybrid._load_archive(runtime_archive)
    del old_files, new_files
    _validate_compatible_sources(old_payload, new_payload, runtime_payload)
    source_payloads = {"old": old_payload, "new": new_payload,
                       "fresh": new_payload}
    source_states = {
        label: payload["quantized_state_dict"]
        for label, payload in source_payloads.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        raise FileExistsError(f"refusing to overwrite {report}")

    arms: dict[str, Any] = {}
    built_payloads: dict[str, Mapping[str, Any]] = {}
    for arm, (point_source, uq_source) in ARM_SOURCES.items():
        point_payload = source_payloads[point_source]
        uq_payload = source_payloads[uq_source]
        payload, composition = hybrid.build_hybrid_payload(
            point_payload, uq_payload
        )
        path = output_dir / f"{arm}.zip"
        archive = _write_archive(
            path, runtime_files, payload, max_size_mib
        )
        built_payloads[arm] = payload
        arms[arm] = {
            "point_source": point_source,
            "uq_source": uq_source,
            "archive": archive,
            "composition": composition,
            "expected_point_state_sha256": hybrid.state_signature(
                source_states[point_source], uq=False
            ),
            "expected_uq_state_sha256": hybrid.state_signature(
                source_states[uq_source], uq=True
            ),
        }

    factorial_audit = audit_factorial_payloads(built_payloads)
    if not factorial_audit["strict_factorial"]:
        raise RuntimeError("built archives did not form a strict factorial")
    archive_audit = audit_factorial_archives(
        {arm: Path(arms[arm]["archive"]["path"]) for arm in arms}
    )
    if not archive_audit["strict_factorial"]:
        raise RuntimeError("serialized archives did not form a strict factorial")
    runtime_sha = {
        name: bytes_sha256(runtime_files[name]) for name in RUNTIME_MEMBERS
    }
    result = {
        "schema": SCHEMA,
        "evaluation_data_read": False,
        "clean716_read": False,
        "sources": {
            "old": {
                "path": str(old_archive.resolve()),
                "sha256": hybrid.sha256(old_archive),
                "point_state_sha256": hybrid.state_signature(
                    source_states["old"], uq=False),
                "uq_state_sha256": hybrid.state_signature(
                    source_states["old"], uq=True),
            },
            "new": {
                "path": str(new_archive.resolve()),
                "sha256": hybrid.sha256(new_archive),
                "point_state_sha256": hybrid.state_signature(
                    source_states["new"], uq=False),
                "uq_state_sha256": hybrid.state_signature(
                    source_states["new"], uq=True),
            },
            "runtime": {
                "path": str(runtime_archive.resolve()),
                "sha256": hybrid.sha256(runtime_archive),
                "member_sha256": runtime_sha,
            },
        },
        "partition": {
            "uq_prefixes": list(hybrid.UQ_PREFIXES),
            "point_keys": sum(
                not hybrid.is_uq_key(key) for key in source_states["old"]),
            "uq_keys": sum(
                hybrid.is_uq_key(key) for key in source_states["old"]),
        },
        "old_vs_new_state": compare_states(
            source_states["old"], source_states["new"]
        ),
        "arms": arms,
        "factorial_audit": factorial_audit,
        "serialized_archive_audit": archive_audit,
        "official_submission_performed": False,
        "submission_authorized": False,
    }
    temporary = report.with_name(report.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-archive", type=Path, required=True)
    parser.add_argument("--new-archive", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-size-mib", type=float, default=256.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_factorial(
        old_archive=args.old_archive.resolve(),
        new_archive=args.new_archive.resolve(),
        runtime_archive=args.runtime_archive.resolve(),
        output_dir=args.output_dir.resolve(),
        report=args.report.resolve(),
        max_size_mib=args.max_size_mib,
    )
    print(json.dumps({
        "schema": result["schema"],
        "arms": {
            arm: row["archive"] for arm, row in result["arms"].items()
        },
        "state_summary": result["old_vs_new_state"]["summary"],
        "factorial_audit": result["serialized_archive_audit"],
        "report": str(args.report.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
