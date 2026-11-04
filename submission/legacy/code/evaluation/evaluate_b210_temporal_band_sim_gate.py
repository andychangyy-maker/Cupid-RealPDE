#!/usr/bin/env python3
"""Apply the frozen Sim10k gate for the XCB temporal-band mixer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_sim as sim  # noqa: E402


REFERENCE = {
    "rel_l2_score": 95.75649115634144,
    "tke_rel_l2_score": 90.16171413632367,
    "mvpe_rel_l2_score": 96.80810428449287,
    "native_sps_score": 28.780897857464726,
}
POINT_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_rel_l2_score": 0.10,
    "mvpe_rel_l2_score": 0.09,
}
POINT_DELTA_MIN = 0.030
COMPONENT_DELTA_MIN = -0.030
SPS_DELTA_MIN = -0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate(summary: dict) -> dict:
    candidate = summary.get("paired_candidate", {}).get("terminal_anchor", {})
    audit = summary.get("paired_candidate", {}).get("controller_audit", {})
    invariant = audit.get("transport_invariant_audit", {})
    config = summary.get("model_config", {})
    training = summary.get("training_contract", {})
    component_delta = {
        key: float(candidate.get(key, float("nan"))) - reference
        for key, reference in REFERENCE.items()
    }
    point_delta = sum(
        POINT_WEIGHTS[key] * component_delta[key]
        for key in POINT_WEIGHTS)
    mechanics = all(bool(audit.get(key)) for key in (
        "transport_output_updated", "transport_body_updated",
        "transport_output_gradient_seen", "transport_body_gradient_seen"))
    checks = {
        "method": summary.get("method") == sim.TEMPORAL_BAND_MIXER_METHOD,
        "completed_updates": int(summary.get("completed_updates", -1)) == 10_000,
        "parameter_contract": int(summary.get("parameters", -1))
        == sim.EXPECTED_XCB_MEMBER_TEMPORAL_BAND_PARAMETERS,
        "checkpoint_sealed": bool(
            summary.get("checkpoint") and summary.get("checkpoint_sha256")),
        "architecture_contract": bool(
            config.get("temporal_band_mixer")
            and not config.get("isoenergetic_temporal_transport")
            and config.get("cross_conditioned_backbone")
            and int(config.get("member_adapter_channels", -1))
            == sim.MEMBER_FEATURE_ADAPTER_CHANNELS
            and int(config.get("temporal_band_channels", -1))
            == sim.TEMPORAL_BAND_CHANNELS
            and int(config.get("temporal_band_count", -1))
            == sim.TEMPORAL_BAND_COUNT
            and math.isclose(float(config.get(
                "temporal_band_log_gain_limit", float("nan"))),
                sim.TEMPORAL_BAND_LOG_GAIN_LIMIT)
            and math.isclose(float(config.get(
                "temporal_band_phase_limit", float("nan"))),
                sim.TEMPORAL_BAND_PHASE_LIMIT)),
        "standard_objective": (
            training.get("accuracy_weights_rel_tke_mvpe") == [1.0, 4.0, 1.0]
            and bool(training.get(
                "standard_objective_member_feature_adapters"))),
        "transport_mechanics": mechanics,
        "transport_mean_pressure_invariants": bool(
            invariant.get("pass") and invariant.get("finite")
            and float(invariant.get("mean_max_abs_error", float("inf")))
            <= 2.0e-5
            and invariant.get("pressure_bitwise_exact")
            and invariant.get("energy_is_invariant") is False),
        "finite_metrics": all(math.isfinite(value)
                              for value in component_delta.values()),
        "point_delta": point_delta + 1.0e-12 >= POINT_DELTA_MIN,
        "component_floors": all(
            component_delta[key] + 1.0e-12 >= COMPONENT_DELTA_MIN
            for key in POINT_WEIGHTS),
        "sps_floor": (
            component_delta["native_sps_score"] + 1.0e-12 >= SPS_DELTA_MIN),
    }
    return {
        "method": sim.TEMPORAL_BAND_MIXER_METHOD,
        "reference": REFERENCE,
        "candidate": candidate,
        "component_delta": component_delta,
        "point65_delta": point_delta,
        "thresholds": {
            "point65_delta_min": POINT_DELTA_MIN,
            "component_delta_min": COMPONENT_DELTA_MIN,
            "native_sps_delta_min": SPS_DELTA_MIN,
        },
        "checks": checks,
        "pass": all(checks.values()),
        "decision": (
            "PASS_TO_STANDARD24K_REAL4K"
            if all(checks.values()) else "STOP_AFTER_SIM10K"),
    }


def main() -> None:
    args = parse_args()
    result = evaluate(json.loads(args.summary.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
