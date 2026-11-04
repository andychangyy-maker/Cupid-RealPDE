from __future__ import annotations

from pathlib import Path
import sys


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = EVALUATION_ROOT.parent
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_b210_temporal_band_sim_gate as subject  # noqa: E402


def summary(*, point_delta: float = 0.04, sps_delta: float = 0.0) -> dict:
    candidate = dict(subject.REFERENCE)
    candidate["rel_l2_score"] += point_delta / 0.46
    candidate["native_sps_score"] += sps_delta
    return {
        "method": subject.sim.TEMPORAL_BAND_MIXER_METHOD,
        "completed_updates": 10_000,
        "parameters": subject.sim.EXPECTED_XCB_MEMBER_TEMPORAL_BAND_PARAMETERS,
        "checkpoint": "/sealed/model_10000.pth",
        "checkpoint_sha256": "a" * 64,
        "model_config": {
            "temporal_band_mixer": True,
            "isoenergetic_temporal_transport": False,
            "cross_conditioned_backbone": True,
            "member_adapter_channels": subject.sim.MEMBER_FEATURE_ADAPTER_CHANNELS,
            "temporal_band_channels": subject.sim.TEMPORAL_BAND_CHANNELS,
            "temporal_band_count": subject.sim.TEMPORAL_BAND_COUNT,
            "temporal_band_log_gain_limit": (
                subject.sim.TEMPORAL_BAND_LOG_GAIN_LIMIT),
            "temporal_band_phase_limit": subject.sim.TEMPORAL_BAND_PHASE_LIMIT,
        },
        "paired_candidate": {
            "terminal_anchor": candidate,
            "controller_audit": {
                "transport_output_updated": True,
                "transport_body_updated": True,
                "transport_output_gradient_seen": True,
                "transport_body_gradient_seen": True,
                "transport_invariant_audit": {
                    "pass": True,
                    "finite": True,
                    "mean_max_abs_error": 1.0e-7,
                    "energy_max_abs_error": 0.1,
                    "energy_is_invariant": False,
                    "pressure_bitwise_exact": True,
                },
            },
        },
        "training_contract": {
            "accuracy_weights_rel_tke_mvpe": [1.0, 4.0, 1.0],
            "standard_objective_member_feature_adapters": True,
        },
    }


def test_gate_passes_exact_threshold_contract():
    result = subject.evaluate(summary(point_delta=0.030))
    assert result["pass"] is True
    assert result["decision"] == "PASS_TO_STANDARD24K_REAL4K"


def test_gate_rejects_point_sps_and_component_failures():
    assert subject.evaluate(summary(point_delta=0.029))["pass"] is False
    assert subject.evaluate(summary(sps_delta=-0.1001))["pass"] is False
    payload = summary(point_delta=0.04)
    payload["paired_candidate"]["terminal_anchor"][
        "tke_rel_l2_score"] -= 0.031
    assert subject.evaluate(payload)["checks"]["component_floors"] is False


def test_gate_does_not_require_energy_invariance_but_requires_mean_and_pressure():
    payload = summary()
    audit = payload["paired_candidate"]["controller_audit"][
        "transport_invariant_audit"]
    audit["energy_max_abs_error"] = 10.0
    assert subject.evaluate(payload)["pass"] is True
    audit["mean_max_abs_error"] = 3.0e-5
    assert subject.evaluate(payload)["pass"] is False
