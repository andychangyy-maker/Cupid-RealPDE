#!/usr/bin/env python3
"""Apply the preregistered Sim10k gate to temporal amplitude factorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_sim as sim  # noqa: E402


SCHEMA = "b210_temporal_amplitude_factorization_sim_gate_v1"
REFERENCE = {
    "rel_l2_score": 95.6958677800766,
    "tke_rel_l2_score": 89.98593937890698,
    "mvpe_rel_l2_score": 96.7543645563888,
    "point_score": 94.19458861498815,
    "native_sps_score": 28.566965554752848,
}
THRESHOLDS = {
    "rel_l2_score": 0.02,
    "tke_rel_l2_score": 0.15,
    "mvpe_rel_l2_score": 0.0,
    "point_score": 0.08,
    "native_sps_score": 0.05,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def evaluate(summary: Mapping[str, object]) -> dict:
    candidate = summary.get("paired_candidate", {}).get("terminal_anchor")
    audit = summary.get("paired_candidate", {}).get("controller_audit")
    config = summary.get("model_config")
    checkpoint = summary.get("checkpoint")
    if (not isinstance(candidate, Mapping)
            or not isinstance(audit, Mapping)
            or not isinstance(config, Mapping)):
        raise ValueError("temporal-amplitude summary lacks anchor/audit/config")
    delta = {
        key: float(candidate[key]) - value
        for key, value in REFERENCE.items()
    }
    mechanics = (
        "temporal_output_updated", "temporal_body_updated",
        "temporal_mean_gradient_seen",
        "temporal_fluctuation_gradient_seen",
        "temporal_amplitude_gradient_seen",
        "temporal_projection_gradient_seen",
        "temporal_body_gradient_seen",
    )
    checks = {
        "finite": all(math.isfinite(value) for value in delta.values()),
        "completed_sim10k": int(summary.get("completed_updates", -1)) == 10_000,
        "method": summary.get("method")
        == sim.TEMPORAL_AMPLITUDE_FACTORIZATION_METHOD,
        "parameters": int(summary.get("parameters", -1))
        == sim.EXPECTED_TEMPORAL_AMPLITUDE_FACTORIZATION_PARAMETERS,
        "temporal_point_decoder": config.get("temporal_point_decoder") is True,
        "amplitude_factorization": config.get(
            "temporal_point_amplitude_factorization") is True,
        "temporal_point_channels": int(config.get(
            "temporal_point_channels", -1)) == sim.TEMPORAL_POINT_CHANNELS,
        "temporal_point_blocks": int(config.get(
            "temporal_point_blocks", -1)) == sim.TEMPORAL_POINT_BLOCKS,
        "temporal_point_kernel": int(config.get(
            "temporal_point_kernel", -1)) == sim.TEMPORAL_POINT_KERNEL,
        "amplitude_limit": float(config.get(
            "temporal_point_amplitude_limit", -1.0))
        == sim.TEMPORAL_POINT_AMPLITUDE_LIMIT,
        "legacy_point_heads_removed": (
            int(summary.get("parameters", -1)) < sim.EXPECTED_PARAMETERS),
        "shared_energy_controller": (
            config.get("physical_energy_controller") is True
            and config.get("physical_energy_multiscale") is False
            and config.get("physical_energy_horizon_local") is False
            and config.get("physical_energy_history_conditioned") is False),
        "mechanics": all(audit.get(key) is True for key in mechanics),
        "checkpoint_declared": (
            isinstance(checkpoint, str)
            and isinstance(summary.get("checkpoint_sha256"), str)),
        **{
            key: delta[key] >= threshold
            for key, threshold in THRESHOLDS.items()
        },
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "pass": passed,
        "decision": (
            "PASS_B210_TEMPORAL_AMPLITUDE_SIM10K_TO_PREREGISTER_REAL"
            if passed else "STOP_B210_TEMPORAL_AMPLITUDE_SIM10K"),
        "reference": dict(REFERENCE),
        "candidate": dict(candidate),
        "delta": delta,
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "mechanics": {key: audit.get(key) for key in mechanics},
        "checkpoint": checkpoint,
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = evaluate(summary)
    checkpoint = Path(str(result["checkpoint"]))
    if (not checkpoint.is_file()
            or sha256(checkpoint) != result["checkpoint_sha256"]):
        raise RuntimeError(
            "temporal-amplitude checkpoint SHA does not match summary")
    result["summary"] = str(args.summary.resolve())
    result["summary_sha256"] = sha256(args.summary)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "b210_temporal_amplitude_sim_gate_complete",
        "decision": result["decision"],
        "delta": result["delta"],
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
