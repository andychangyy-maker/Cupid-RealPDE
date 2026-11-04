#!/usr/bin/env python3
"""Apply the preregistered Sim10k gate to isoenergetic time transport."""

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


SCHEMA = "b210_isoenergetic_temporal_transport_sim_gate_v4"
REFERENCE = {
    "rel_l2_score": 95.6958677800766,
    "tke_rel_l2_score": 89.98593937890698,
    "mvpe_rel_l2_score": 96.7543645563888,
    "point_score": 94.19458861498815,
    "native_sps_score": 28.566965554752848,
}
TKE_FLOOR_REFERENCE = {
    "rel_l2_score": 95.71349170444508,
    "tke_rel_l2_score": 90.0550501455309,
    "mvpe_rel_l2_score": 96.77014482189533,
    "point_score": 94.22728986026087,
    "native_sps_score": 28.409255000126873,
}
OFFICIAL_FINAL_WEIGHTS = {
    "rel_l2_score": 46.0,
    "tke_rel_l2_score": 10.0,
    "mvpe_rel_l2_score": 9.0,
    "time_score": 10.0,
    "native_sps_score": 25.0,
}
OFFICIAL_POINT_WEIGHT_SUM = 65.0


def official_point_proxy(metrics: Mapping[str, object]) -> float:
    return sum(
        OFFICIAL_FINAL_WEIGHTS[key] * float(metrics[key])
        for key in (
            "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score")
    ) / OFFICIAL_POINT_WEIGHT_SUM


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
    training_contract = summary.get("training_contract")
    checkpoint = summary.get("checkpoint")
    if (not isinstance(candidate, Mapping)
            or not isinstance(audit, Mapping)
            or not isinstance(config, Mapping)):
        raise ValueError("temporal-transport summary lacks anchor/audit/config")
    invariant = audit.get("transport_invariant_audit")
    if not isinstance(invariant, Mapping):
        raise ValueError("temporal-transport summary lacks invariant audit")
    method = summary.get("method")
    tke_floor_objective = method == sim.OFFICIAL_POINT_TKE_FLOOR_METHOD
    reference = TKE_FLOOR_REFERENCE if tke_floor_objective else REFERENCE
    delta = {
        key: float(candidate[key]) - value
        for key, value in reference.items()
    }
    reference_official_point = official_point_proxy(reference)
    candidate_official_point = official_point_proxy(candidate)
    official_point_delta = (
        candidate_official_point - reference_official_point)
    mechanics = (
        "transport_output_updated", "transport_body_updated",
        "transport_output_gradient_seen", "transport_body_gradient_seen",
    )
    official_objective = method in {
        sim.OFFICIAL_POINT_TEMPORAL_TRANSPORT_METHOD,
        sim.OFFICIAL_POINT_TKE_FLOOR_METHOD,
    }
    objective_contract = (
        isinstance(training_contract, Mapping)
        and training_contract.get("official_point_objective") is True
        and tuple(training_contract.get(
            "accuracy_weights_rel_tke_mvpe", ()))
        == sim.OFFICIAL_POINT_ACCURACY_WEIGHTS)
    tke_floor_contract = (
        not tke_floor_objective
        or (
            isinstance(training_contract, Mapping)
            and training_contract.get(
                "official_point_controller_tke_floor") is True
            and float(training_contract.get(
                "tke_aux_persistent_floor", -1.0))
            == sim.TKE_AUX_PERSISTENT_FLOOR_WEIGHT
            and int(training_contract.get("tke_aux_updates", -1))
            == sim.TERMINAL_UPDATE))
    checks = {
        "finite": (
            all(math.isfinite(value) for value in delta.values())
            and math.isfinite(candidate_official_point)
            and math.isfinite(official_point_delta)),
        "completed_sim10k": int(summary.get("completed_updates", -1)) == 10_000,
        "method": method in {
            sim.ISOENERGETIC_TEMPORAL_TRANSPORT_METHOD,
            sim.OFFICIAL_POINT_TEMPORAL_TRANSPORT_METHOD,
            sim.OFFICIAL_POINT_TKE_FLOOR_METHOD},
        "accuracy_objective_contract": (
            objective_contract if official_objective else True),
        "controller_tke_floor_contract": tke_floor_contract,
        "parameters": int(summary.get("parameters", -1))
        == sim.EXPECTED_TEMPORAL_TRANSPORT_PARAMETERS,
        "transport_enabled": config.get(
            "isoenergetic_temporal_transport") is True,
        "legacy_point_heads_retained": (
            config.get("temporal_point_decoder") is False
            and int(summary.get("parameters", -1))
            > sim.EXPECTED_PARAMETERS),
        "transport_channels": int(config.get(
            "temporal_transport_channels", -1))
        == sim.TEMPORAL_TRANSPORT_CHANNELS,
        "transport_blocks": int(config.get(
            "temporal_transport_blocks", -1))
        == sim.TEMPORAL_TRANSPORT_BLOCKS,
        "transport_kernel": int(config.get(
            "temporal_transport_kernel", -1))
        == sim.TEMPORAL_TRANSPORT_KERNEL,
        "transport_max_displacement": float(config.get(
            "temporal_transport_max_displacement", -1.0))
        == sim.TEMPORAL_TRANSPORT_MAX_DISPLACEMENT,
        "shared_energy_controller": (
            config.get("physical_energy_controller") is True
            and config.get("physical_energy_multiscale") is False
            and config.get("physical_energy_horizon_local") is False
            and config.get("physical_energy_history_conditioned") is False),
        "mechanics": all(audit.get(key) is True for key in mechanics),
        "transport_invariants": (
            invariant.get("pass") is True
            and invariant.get("pressure_bitwise_exact") is True
            and float(invariant.get("mean_max_abs_error", math.inf)) <= 2e-5
            and float(invariant.get("energy_max_abs_error", math.inf)) <= 2e-5),
        "checkpoint_declared": (
            isinstance(checkpoint, str)
            and isinstance(summary.get("checkpoint_sha256"), str)),
        # The competition weights supplied during this run are 46/10/9 for
        # Rel/TKE/MVPE.  Individual component gains and the historical
        # 50/30/20 local point_score are diagnostics, not hard gates.
        "official_point_proxy_improved": official_point_delta > 0.0,
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "pass": passed,
        "decision": (
            "PASS_B210_TEMPORAL_TRANSPORT_SIM10K_TO_PREREGISTER_REAL"
            if passed else "STOP_B210_TEMPORAL_TRANSPORT_SIM10K"),
        "reference": dict(reference),
        "reference_role": (
            "previous_passed_isoenergetic_transport"
            if tke_floor_objective else "frozen_sim_anchor"),
        "candidate": dict(candidate),
        "delta": delta,
        "official_final_weights": dict(OFFICIAL_FINAL_WEIGHTS),
        "reference_official_point_proxy": reference_official_point,
        "candidate_official_point_proxy": candidate_official_point,
        "official_point_proxy_delta": official_point_delta,
        "official_point_objective": official_objective,
        "official_point_controller_tke_floor": tke_floor_objective,
        "component_and_legacy_point_deltas_are_diagnostic_only": True,
        "checks": checks,
        "mechanics": {key: audit.get(key) for key in mechanics},
        "transport_invariant_audit": dict(invariant),
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
            "temporal-transport checkpoint SHA does not match summary")
    result["summary"] = str(args.summary.resolve())
    result["summary_sha256"] = sha256(args.summary)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "b210_temporal_transport_sim_gate_complete",
        "decision": result["decision"],
        "delta": result["delta"],
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
