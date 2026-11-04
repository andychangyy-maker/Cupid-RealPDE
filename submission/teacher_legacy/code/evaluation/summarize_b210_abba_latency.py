#!/usr/bin/env python3
"""Summarize a frozen C208-B210-B210-C208 deployment latency sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_PEAK_RESERVED_BYTES = 14 * 1024 * 1024 * 1024
MAX_BLOCK_SPREAD = 0.10
MAX_POOLED_RATIO = 1.05
MAX_ADVERSE_RATIO = 1.10
MAX_STARTUP_RATIO = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c208-a", type=Path, required=True)
    parser.add_argument("--b210-a", type=Path, required=True)
    parser.add_argument("--b210-b", type=Path, required=True)
    parser.add_argument("--c208-b", type=Path, required=True)
    parser.add_argument("--expected-c208-archive-sha256", required=True)
    parser.add_argument("--expected-b210-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def load_block(path: Path, expected_archive_sha: str,
               expected_arm: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evidence_class") != "s2_deployment_archive_latency":
        raise ValueError(f"{path} is not an archive latency result")
    archive = payload.get("archive", {})
    if archive.get("sha256") != expected_archive_sha.lower():
        raise ValueError(f"{path} {expected_arm} archive SHA-256 differs")
    contract = payload.get("execution_contract", {})
    required_contract = {
        "execution_mode": "eager",
        "torch_compile_requested": False,
        "channels_last_requested": False,
        "cudnn_benchmark": False,
        "input_c_contiguous": True,
        "input_pressure_zero": True,
        "input_bitwise_unchanged": True,
    }
    if any(contract.get(key) != value
           for key, value in required_contract.items()):
        raise ValueError(f"{path} violates the frozen execution contract")
    if not payload.get("outputs", {}).get("bounds_ordered", False):
        raise ValueError(f"{path} has unordered interval bounds")
    output_rows = payload.get("outputs", {})
    tensor_rows = [value for value in output_rows.values()
                   if isinstance(value, dict) and "finite" in value]
    if not tensor_rows or not all(row.get("finite") is True
                                  for row in tensor_rows):
        raise ValueError(f"{path} has missing or non-finite outputs")
    if archive.get("bytes", MAX_ARCHIVE_BYTES + 1) > MAX_ARCHIVE_BYTES:
        raise ValueError(f"{path} archive exceeds 256 MiB")
    return payload


def relative_spread(values: list[float]) -> float:
    return (max(values) / min(values)) - 1.0


def summarize(paths: list[Path], c208_sha: str,
              b210_sha: str) -> dict[str, object]:
    arm_names = ["c208", "b210", "b210", "c208"]
    expected_shas = [c208_sha, b210_sha, b210_sha, c208_sha]
    blocks = [
        load_block(path, sha, arm)
        for path, sha, arm in zip(paths, expected_shas, arm_names)
    ]
    frozen_keys = (
        "batch_size", "warmup_repeats", "repeats", "device", "gpu",
        "torch_version",
    )
    for key in frozen_keys:
        values = [block.get(key) for block in blocks]
        if len(set(map(str, values))) != 1:
            raise ValueError(f"ABBA blocks disagree on {key}")
    expected_runtime = {
        "batch_size": 16,
        "warmup_repeats": 0,
        "repeats": 50,
        "device": "cuda:0",
    }
    if any(block.get(key) != value
           for block in blocks
           for key, value in expected_runtime.items()):
        raise ValueError("ABBA blocks violate the frozen runtime contract")
    if any("4080 SUPER" not in str(block.get("gpu")) for block in blocks):
        raise ValueError("ABBA must run on the frozen RTX 4080 SUPER")
    benchmark_shas = [
        block.get("benchmark_source", {}).get("sha256") for block in blocks]
    if None in benchmark_shas or len(set(benchmark_shas)) != 1:
        raise ValueError("ABBA blocks disagree on benchmark source")
    example_shas = [block.get("example_h5", {}).get("sha256")
                    for block in blocks]
    if len(set(example_shas)) != 1:
        raise ValueError("ABBA blocks disagree on example input")

    c208_blocks = [blocks[0], blocks[3]]
    b210_blocks = [blocks[1], blocks[2]]
    c208_steady = [float(block["end_to_end"]["median_ms"])
                   for block in c208_blocks]
    b210_steady = [float(block["end_to_end"]["median_ms"])
                   for block in b210_blocks]
    c208_startup = [float(block["startup_to_first_output_seconds"])
                    for block in c208_blocks]
    b210_startup = [float(block["startup_to_first_output_seconds"])
                    for block in b210_blocks]
    c208_spread = relative_spread(c208_steady)
    b210_spread = relative_spread(b210_steady)
    pooled_ratio = (
        statistics.median(b210_steady) / statistics.median(c208_steady))
    adverse_ratio = max(b210_steady) / min(c208_steady)
    startup_ratio = (
        statistics.median(b210_startup) / statistics.median(c208_startup))
    peak_reserved = max(int(block["peak_cuda_reserved_bytes"])
                        for block in b210_blocks)
    gates = {
        "c208_two_block_spread_le_10pct": c208_spread <= MAX_BLOCK_SPREAD,
        "b210_two_block_spread_le_10pct": b210_spread <= MAX_BLOCK_SPREAD,
        "pooled_b210_over_c208_le_1p05": pooled_ratio <= MAX_POOLED_RATIO,
        "adverse_b210_over_c208_le_1p10": adverse_ratio <= MAX_ADVERSE_RATIO,
        "startup_b210_over_c208_le_1p25": startup_ratio <= MAX_STARTUP_RATIO,
        "b210_peak_reserved_lt_14gib": peak_reserved < MAX_PEAK_RESERVED_BYTES,
    }
    return {
        "evidence_class": "s2_deployment_archive_latency_abba",
        "sequence": ["c208", "b210", "b210", "c208"],
        "blocks": [file_identity(path) for path in paths],
        "archive_sha256": {
            "c208": c208_sha.lower(),
            "b210": b210_sha.lower(),
        },
        "steady_median_ms": {
            "c208_blocks": c208_steady,
            "b210_blocks": b210_steady,
            "c208_pooled": statistics.median(c208_steady),
            "b210_pooled": statistics.median(b210_steady),
        },
        "startup_to_first_output_seconds": {
            "c208_blocks": c208_startup,
            "b210_blocks": b210_startup,
        },
        "ratios": {
            "c208_two_block_spread": c208_spread,
            "b210_two_block_spread": b210_spread,
            "pooled_b210_over_c208": pooled_ratio,
            "adverse_b210_over_c208": adverse_ratio,
            "startup_b210_over_c208": startup_ratio,
        },
        "b210_peak_reserved_bytes": peak_reserved,
        "gates": gates,
        "decision": (
            "PASS_B210_ABBA_LATENCY"
            if all(gates.values()) else "STOP_B210_ABBA_LATENCY"),
    }


def main() -> None:
    args = parse_args()
    paths = [args.c208_a, args.b210_a, args.b210_b, args.c208_b]
    result = summarize(
        paths,
        args.expected_c208_archive_sha256,
        args.expected_b210_archive_sha256,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "b210_abba_latency_complete",
                      "decision": result["decision"],
                      "output": file_identity(args.output)}), flush=True)


if __name__ == "__main__":
    main()
