#!/usr/bin/env python3
"""Retain the composite Sim24k energy function during dense Real4k."""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
import sys
from typing import Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training", PRIVATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import b210_controller_functional_rehearsal as functional  # noqa: E402
import train_b210_multiscale_energy_real4k_rehearsal as base  # noqa: E402
import train_b210_self_describing_real_dense_aug as self_desc  # noqa: E402


EXPERIMENT = "b210_xcb_member_multiscale_real4k_controller_functional"
SCHEMA = "b210_xcb_member_multiscale_real4k_controller_functional_v1"
REFERENCE_EXPERIMENT = (
    "b210_xcb_member_multiscale_standard24k_real_dense_aug")
REFERENCE_SCHEMA = (
    "b210_xcb_member_multiscale_standard24k_real_dense_aug_v1")
EXPECTED_ITERATION = 4_000
EXPECTED_STATUS = "PASS_B210_XCB_MEMBER_MULTISCALE_STANDARD24K_SIM"
THRESHOLDS = {
    "official_point_delta": 0.005,
    "official_non_time_delta": 0.015,
    "native_sps_delta": -0.02,
    "positive_aoas": 4,
    "worst_aoa_official_point": -0.04,
}


def architecture_contract(
        horizon_local_energy_controller: bool,
        multiscale_energy_controller: bool = False) -> tuple:
    # The reused runner historically requests (False, True); the serialized
    # checkpoint remains the sole architecture authority.
    if horizon_local_energy_controller or not multiscale_energy_controller:
        raise ValueError("functional composite architecture request changed")
    payload = torch.load(
        self_desc.checkpoint_path(), map_location="cpu", weights_only=False)
    return self_desc.architecture_from_payload(payload)


def bind_validator(validator):
    return partial(
        validator,
        expected_iteration=EXPECTED_ITERATION,
        expected_status=EXPECTED_STATUS)


def load_reference(path: Path, expected_sha256: str,
                   evaluation_manifest: Path, start_sha256: str) -> dict:
    actual = base.dr1.sha256(path)
    if actual != base.dr1.normalized_sha256(
            expected_sha256, "reference SHA"):
        raise RuntimeError("functional reference result changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema") != REFERENCE_SCHEMA
            or payload.get("experiment") != REFERENCE_EXPERIMENT
            or payload.get("start", {}).get("checkpoint_sha256")
            != start_sha256):
        raise RuntimeError("functional reference identity changed")
    evaluation = payload.get("evaluation_manifest", {})
    if (evaluation.get("sha256") != base.dr1.sha256(evaluation_manifest)
            or int(evaluation.get("tail_windows", -1)) != base.dense4k.TAIL_WINDOWS):
        raise RuntimeError("functional reference tail identity changed")
    candidate = payload.get("arms", {}).get("candidate", {})
    if (not isinstance(candidate.get("metrics"), Mapping)
            or not isinstance(candidate.get("metrics_by_aoa"), Mapping)
            or not isinstance(candidate.get("target_audit"), Mapping)):
        raise RuntimeError("functional reference candidate is incomplete")
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "metrics": dict(candidate["metrics"]),
        "metrics_by_aoa": {
            key: dict(value)
            for key, value in candidate["metrics_by_aoa"].items()},
        "target_audit": dict(candidate["target_audit"]),
        "normalizer": dict(payload.get("normalizer", {})),
        "seal": dict(candidate.get("seal", {})),
    }


def official_deltas(delta: Mapping[str, object]) -> tuple[float, float]:
    point = (
        0.46 * float(delta["rel_l2_score"])
        + 0.10 * float(delta["tke_rel_l2_score"])
        + 0.09 * float(delta["mvpe_rel_l2_score"]))
    return point, point + 0.25 * float(delta["native_sps_score"])


def release_gate(reference_comparison: Mapping[str, object],
                 _old_comparison: Mapping[str, object]) -> dict:
    delta = reference_comparison["delta"]
    point, non_time = official_deltas(delta)
    aoa = []
    for key, row in reference_comparison["by_aoa"].items():
        row_point, row_non_time = official_deltas(row["delta"])
        aoa.append((key, row_point, row_non_time))
    positive_aoas = sum(row[2] > 0.0 for row in aoa)
    worst_point = min(row[1] for row in aoa)
    checks = {
        "finite": all(math.isfinite(value) for value in (
            point, non_time, worst_point,
            float(delta["native_sps_score"]))),
        "official_point": point >= THRESHOLDS["official_point_delta"],
        "official_non_time": (
            non_time >= THRESHOLDS["official_non_time_delta"]),
        "native_sps": (
            float(delta["native_sps_score"])
            >= THRESHOLDS["native_sps_delta"]),
        "aoa_count": positive_aoas >= THRESHOLDS["positive_aoas"],
        "aoa_floor": (
            worst_point >= THRESHOLDS["worst_aoa_official_point"]),
    }
    passed = all(checks.values())
    return {
        "pass": passed,
        "decision": (
            "PASS_B210_CONTROLLER_FUNCTIONAL_TO_CONDITION_OOF"
            if passed else "STOP_B210_CONTROLLER_FUNCTIONAL"),
        "checks": checks,
        "thresholds": dict(THRESHOLDS),
        "official_point_delta": point,
        "official_non_time_delta": non_time,
        "positive_aoas": positive_aoas,
        "worst_aoa_official_point_delta": worst_point,
        "by_aoa_official": {
            key: {"point_delta": row_point, "non_time_delta": row_non_time}
            for key, row_point, row_non_time in aoa},
    }


def configure_runner() -> None:
    base.EXPERIMENT = EXPERIMENT
    base.SCHEMA = SCHEMA
    base.REFERENCE_SCHEMA = REFERENCE_SCHEMA
    base.REFERENCE_EXPERIMENT = REFERENCE_EXPERIMENT
    base.dense4k.architecture_contract = architecture_contract
    base.ft12.validate_start_checkpoint = bind_validator(
        base.ft12.validate_start_checkpoint)
    base.prepare_sim = functional.prepare_sim
    base.SimRehearsalBackward = functional.ControllerFunctionalBackward
    base.load_reference = load_reference
    base.release_gate = release_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--start-sim-sha256", required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--reference-result-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 3600)
    return parser.parse_args()


def main() -> None:
    configure_runner()
    args = parse_args()
    result = base.run(args)
    print(json.dumps({
        "event": "b210_controller_functional_real4k_complete",
        "candidate": result["candidate"]["metrics"],
        "reference_delta": result["reference_comparison"]["delta"],
        "gate": result["gate"],
        "result": str(args.output_dir / "result.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
