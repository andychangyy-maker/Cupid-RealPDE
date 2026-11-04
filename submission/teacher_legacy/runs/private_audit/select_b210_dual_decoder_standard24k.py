#!/usr/bin/env python3
"""Mechanically gate the matched shared/separate dual-decoder Sim24k pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import torch


CONTRACTS = {
    "control": {
        "method": "b210_dualdecoder_sharedfused_standard24k_v1",
        "status": "PASS_B210_DUALDECODER_SHAREDFUSED_STANDARD24K_SIM",
        "mode": "shared_fused",
        "parameters": 196_316_169,
    },
    "candidate": {
        "method": "b210_dualdecoder_separatepointuqgate_standard24k_v1",
        "status": "PASS_B210_DUALDECODER_SEPARATEPOINTUQGATE_STANDARD24K_SIM",
        "mode": "separate_point_uq_fluctuation_gates",
        "parameters": 196_319_109,
    },
}
THRESHOLDS = {
    "point_delta": -0.01,
    "component_delta": -0.04,
    "native_sps_delta": 0.10,
    "non_time_delta": 0.02,
    "gate_max_abs_tanh": 0.95,
}
METRIC_KEYS = (
    "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
    "native_sps_score", "native_sps_coverage", "native_sps_mean_width",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arm(path: Path, arm: str) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    contract = CONTRACTS[arm]
    if (summary.get("method") != contract["method"]
            or summary.get("status") != contract["status"]
            or int(summary.get("completed_updates", -1)) != 4_000
            or int(summary.get("cumulative_supervised_updates", -1)) != 24_000
            or int(summary.get("parameters", -1)) != contract["parameters"]
            or summary.get("model_config", {}).get("dual_decoder_mode")
            != contract["mode"]):
        raise RuntimeError(f"{arm} Sim24k summary contract changed")
    checkpoint = Path(summary["checkpoint"])
    if (not checkpoint.is_file()
            or sha256(checkpoint) != summary.get("checkpoint_sha256")):
        raise RuntimeError(f"{arm} checkpoint identity changed")
    metrics = summary.get("paired_candidate", {}).get("terminal_anchor")
    if (not isinstance(metrics, Mapping)
            or any(key not in metrics for key in METRIC_KEYS)):
        raise RuntimeError(f"{arm} terminal anchor is incomplete")
    return {
        "summary": str(path.resolve()),
        "summary_sha256": sha256(path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "metrics": {key: float(metrics[key]) for key in METRIC_KEYS},
        "wall_seconds": float(summary["wall_seconds"]),
    }


def official_deltas(delta: Mapping[str, float]) -> tuple[float, float]:
    point = (
        0.46 * float(delta["rel_l2_score"])
        + 0.10 * float(delta["tke_rel_l2_score"])
        + 0.09 * float(delta["mvpe_rel_l2_score"]))
    return point, point + 0.25 * float(delta["native_sps_score"])


def gate_statistics(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", {})
    result = {}
    for role in ("fluctuation_role_gates", "uq_fluctuation_role_gates"):
        values = [
            tensor.detach().float().reshape(-1)
            for name, tensor in state.items()
            if name.startswith(f"{role}.")]
        if not values:
            raise RuntimeError(f"candidate checkpoint lacks {role}")
        raw = torch.cat(values)
        alpha = raw.tanh()
        result[role] = {
            "count": int(alpha.numel()),
            "raw_finite": bool(torch.isfinite(raw).all()),
            "mean": float(alpha.mean()),
            "mean_abs": float(alpha.abs().mean()),
            "median_abs": float(alpha.abs().median()),
            "max_abs": float(alpha.abs().max()),
            "fraction_abs_gt_0p95": float((alpha.abs() > 0.95).float().mean()),
        }
    point = torch.cat([
        value.detach().float().reshape(-1).tanh()
        for name, value in state.items()
        if name.startswith("fluctuation_role_gates.")])
    uq = torch.cat([
        value.detach().float().reshape(-1).tanh()
        for name, value in state.items()
        if name.startswith("uq_fluctuation_role_gates.")])
    result["role_difference"] = {
        "rms": float((point - uq).square().mean().sqrt()),
        "mae": float((point - uq).abs().mean()),
        "same_sign_fraction": float(((point * uq) >= 0).float().mean()),
        "cosine": float(torch.nn.functional.cosine_similarity(
            point, uq, dim=0)),
    }
    return result


def select(control: dict, candidate: dict, gate_stats: dict) -> dict:
    delta = {
        key: candidate["metrics"][key] - control["metrics"][key]
        for key in METRIC_KEYS}
    point, non_time = official_deltas(delta)
    components = {
        key: delta[key]
        for key in (
            "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")}
    role_rows = (
        gate_stats["fluctuation_role_gates"],
        gate_stats["uq_fluctuation_role_gates"],
    )
    checks = {
        "finite": all(math.isfinite(value) for value in (
            *delta.values(), point, non_time)),
        "point": point >= THRESHOLDS["point_delta"],
        "components": min(components.values()) >= THRESHOLDS["component_delta"],
        "native_sps": delta["native_sps_score"] >= THRESHOLDS["native_sps_delta"],
        "non_time": non_time >= THRESHOLDS["non_time_delta"],
        "gates_finite": all(row["raw_finite"] for row in role_rows),
        "gates_not_saturated": all(
            row["max_abs"] < THRESHOLDS["gate_max_abs_tanh"]
            for row in role_rows),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "GO_SEPARATE_POINT_UQ_GATE_TO_MATCHED_REAL4K"
            if passed else "STOP_SEPARATE_POINT_UQ_GATE_AT_SIM24K"),
        "pass": passed,
        "checks": checks,
        "thresholds": dict(THRESHOLDS),
        "delta": delta,
        "component_delta": components,
        "official_point_delta": point,
        "official_non_time_delta": non_time,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    arms = {
        "control": load_arm(args.control_summary, "control"),
        "candidate": load_arm(args.candidate_summary, "candidate"),
    }
    gate_stats = gate_statistics(Path(arms["candidate"]["checkpoint"]))
    result = {
        "schema": "b210_dual_decoder_standard24k_selection_v1",
        "arms": arms,
        "gate_statistics": gate_stats,
        "selection": select(arms["control"], arms["candidate"], gate_stats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "event": "b210_dual_decoder_standard24k_selection",
        **result["selection"],
        "output": str(args.output.resolve()),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
