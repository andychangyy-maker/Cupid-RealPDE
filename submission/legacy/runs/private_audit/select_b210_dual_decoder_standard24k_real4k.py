#!/usr/bin/env python3
"""Compare matched Real4k endpoints from shared and separate role routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping


CONTRACTS = {
    "control": {
        "experiment": "b210_dualdecoder_sharedfused_standard24k_real_dense_aug",
        "schema": "b210_dualdecoder_sharedfused_standard24k_real_dense_aug_v1",
        "parameters": 196_316_169,
    },
    "candidate": {
        "experiment": (
            "b210_dualdecoder_separatepointuqgate_standard24k_real_dense_aug"),
        "schema": (
            "b210_dualdecoder_separatepointuqgate_standard24k_real_dense_aug_v1"),
        "parameters": 196_319_109,
    },
}
THRESHOLDS = {
    "point_delta": -0.01,
    "component_delta": -0.04,
    "native_sps_delta": 0.10,
    "non_time_delta": 0.02,
    "positive_aoas": 3,
    "worst_aoa_point_delta": -0.05,
}
METRIC_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
    "native_sps_score", "native_sps_coverage", "native_sps_mean_width",
)
SELECTION_SCHEMA = "b210_dual_decoder_standard24k_real4k_selection_v1"
SELECTION_EVENT = "b210_dual_decoder_standard24k_real4k_selection"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_start_sha(path: Path) -> str:
    summary = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(summary["checkpoint"])
    expected = str(summary["checkpoint_sha256"])
    if not checkpoint.is_file() or sha256(checkpoint) != expected:
        raise RuntimeError("Sim24k start checkpoint identity changed")
    return expected


def load_arm(path: Path, arm: str, start_sha: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    contract = CONTRACTS[arm]
    if (result.get("experiment") != contract["experiment"]
            or result.get("schema") != contract["schema"]
            or int(result.get("parameters", -1)) != contract["parameters"]
            or result.get("start", {}).get("checkpoint_sha256") != start_sha):
        raise RuntimeError(f"{arm} Real4k result contract changed")
    candidate = result.get("arms", {}).get("candidate", {})
    metrics = candidate.get("metrics")
    by_aoa = candidate.get("metrics_by_aoa")
    seal = candidate.get("seal", {})
    if (not isinstance(metrics, Mapping)
            or any(key not in metrics for key in METRIC_KEYS)
            or not isinstance(by_aoa, Mapping)
            or set(by_aoa) != {"0", "5", "10", "15", "20"}
            or int(seal.get("iteration", -1)) != 4_000
            or seal.get("endpoint") != "ema_u4000"):
        raise RuntimeError(f"{arm} Real4k candidate endpoint is incomplete")
    checkpoint = Path(str(seal["path"]))
    if (not checkpoint.is_file()
            or sha256(checkpoint) != seal.get("file_sha256")):
        raise RuntimeError(f"{arm} Real4k checkpoint identity changed")
    return {
        "result": str(path.resolve()),
        "result_sha256": sha256(path),
        "start_checkpoint_sha256": start_sha,
        "endpoint": str(checkpoint.resolve()),
        "endpoint_sha256": seal["file_sha256"],
        "metrics": {key: float(metrics[key]) for key in METRIC_KEYS},
        "metrics_by_aoa": {
            key: {metric: float(values[metric]) for metric in METRIC_KEYS}
            for key, values in by_aoa.items()},
        "target_audit": candidate.get("target_audit"),
        "evaluation_manifest": result.get("evaluation_manifest"),
        "normalizer_statistics_sha256": result.get(
            "normalizer", {}).get("statistics_sha256"),
        "access_audit": result.get("access_audit"),
        "internal_dense_minus_sparse": result.get("comparison"),
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }


def metric_delta(candidate: Mapping[str, float],
                 control: Mapping[str, float]) -> dict[str, float]:
    return {key: float(candidate[key]) - float(control[key])
            for key in METRIC_KEYS}


def official_deltas(delta: Mapping[str, float]) -> tuple[float, float]:
    point = (
        0.46 * float(delta["rel_l2_score"])
        + 0.10 * float(delta["tke_rel_l2_score"])
        + 0.09 * float(delta["mvpe_rel_l2_score"]))
    return point, point + 0.25 * float(delta["native_sps_score"])


def select(control: dict, candidate: dict) -> dict:
    identity_checks = {
        "evaluation_manifest": (
            control["evaluation_manifest"] == candidate["evaluation_manifest"]),
        "normalizer": (
            control["normalizer_statistics_sha256"]
            == candidate["normalizer_statistics_sha256"]),
        "target_audit": control["target_audit"] == candidate["target_audit"],
        "access_audit": control["access_audit"] == candidate["access_audit"],
    }
    delta = metric_delta(candidate["metrics"], control["metrics"])
    point, non_time = official_deltas(delta)
    by_aoa = {}
    for aoa in sorted(candidate["metrics_by_aoa"], key=int):
        row_delta = metric_delta(
            candidate["metrics_by_aoa"][aoa],
            control["metrics_by_aoa"][aoa])
        row_point, row_non_time = official_deltas(row_delta)
        by_aoa[aoa] = {
            "delta": row_delta,
            "official_point_delta": row_point,
            "official_non_time_delta": row_non_time,
        }
    positive_aoas = sum(
        row["official_non_time_delta"] > 0.0 for row in by_aoa.values())
    worst_aoa_point = min(
        row["official_point_delta"] for row in by_aoa.values())
    components = [
        delta[key] for key in (
            "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")]
    checks = {
        "identical_research_contract": all(identity_checks.values()),
        "finite": all(math.isfinite(value) for value in (
            *delta.values(), point, non_time, worst_aoa_point)),
        "point": point >= THRESHOLDS["point_delta"],
        "components": min(components) >= THRESHOLDS["component_delta"],
        "native_sps": delta["native_sps_score"] >= THRESHOLDS["native_sps_delta"],
        "non_time": non_time >= THRESHOLDS["non_time_delta"],
        "positive_aoas": positive_aoas >= THRESHOLDS["positive_aoas"],
        "worst_aoa_point": (
            worst_aoa_point >= THRESHOLDS["worst_aoa_point_delta"]),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "GO_SEPARATE_POINT_UQ_GATE_TO_UQ_AND_DEPLOYMENT"
            if passed else "STOP_SEPARATE_POINT_UQ_GATE_AT_REAL4K"),
        "pass": passed,
        "checks": checks,
        "identity_checks": identity_checks,
        "thresholds": dict(THRESHOLDS),
        "delta": delta,
        "official_point_delta": point,
        "official_non_time_delta": non_time,
        "positive_aoas": positive_aoas,
        "worst_aoa_point_delta": worst_aoa_point,
        "by_aoa": by_aoa,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--control-sim-summary", type=Path, required=True)
    parser.add_argument("--candidate-sim-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    starts = {
        "control": load_start_sha(args.control_sim_summary),
        "candidate": load_start_sha(args.candidate_sim_summary),
    }
    arms = {
        "control": load_arm(args.control_result, "control", starts["control"]),
        "candidate": load_arm(
            args.candidate_result, "candidate", starts["candidate"]),
    }
    result = {
        "schema": SELECTION_SCHEMA,
        "arms": arms,
        "selection": select(arms["control"], arms["candidate"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "event": SELECTION_EVENT,
        **result["selection"],
        "output": str(args.output.resolve()),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
