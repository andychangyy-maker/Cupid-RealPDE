#!/usr/bin/env python3
"""Apply the preregistered Sim10k gate to the history-energy endpoint."""

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


SCHEMA = "b210_history_energy_sim_gate_v1"
REFERENCE = {
    "rel_l2_score": 95.6958677800766,
    "tke_rel_l2_score": 89.98593937890698,
    "mvpe_rel_l2_score": 96.7543645563888,
    "point_score": 94.19458861498815,
    "native_sps_score": 28.566965554752848,
}
THRESHOLDS = {
    "rel_l2_score": -0.05,
    "tke_rel_l2_score": 0.20,
    "mvpe_rel_l2_score": -0.03,
    "point_score": 0.08,
    "native_sps_score": -0.10,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def evaluate(summary: Mapping[str, object]) -> dict:
    candidate = summary.get("paired_candidate", {}).get("terminal_anchor")
    config = summary.get("model_config")
    checkpoint = summary.get("checkpoint")
    if not isinstance(candidate, Mapping) or not isinstance(config, Mapping):
        raise ValueError("history-energy summary lacks terminal anchor/config")
    delta = {
        key: float(candidate[key]) - value
        for key, value in REFERENCE.items()
    }
    checks = {
        "finite": all(math.isfinite(value) for value in delta.values()),
        "completed_sim10k": int(summary.get("completed_updates", -1)) == 10_000,
        "method": summary.get("method") == sim.HISTORY_ENERGY_METHOD,
        "parameters": int(summary.get("parameters", -1))
        == sim.EXPECTED_HISTORY_ENERGY_PARAMETERS,
        "history_controller": config.get(
            "physical_energy_history_conditioned") is True,
        "history_steps": int(config.get("physical_energy_history_steps", -1)) == 5,
        "separate_arm": (
            config.get("physical_energy_multiscale") is False
            and config.get("physical_energy_horizon_local") is False),
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
            "PASS_B210_HISTORY_ENERGY_SIM10K_TO_FROZEN_REAL4K"
            if passed else "STOP_B210_HISTORY_ENERGY_SIM10K"),
        "reference": dict(REFERENCE),
        "candidate": dict(candidate),
        "delta": delta,
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
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
        raise RuntimeError("history-energy checkpoint SHA does not match summary")
    result["summary"] = str(args.summary.resolve())
    result["summary_sha256"] = sha256(args.summary)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "b210_history_energy_sim_gate_complete",
        "decision": result["decision"],
        "delta": result["delta"],
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
