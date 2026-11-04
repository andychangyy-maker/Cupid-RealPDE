from __future__ import annotations

from pathlib import Path
import sys


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = EVALUATION_ROOT.parent
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_b210_temporal_transport_sim_gate as subject  # noqa: E402


def summary_with_delta(*, official_objective=False, tke_floor=False,
                       **overrides):
    reference = subject.TKE_FLOOR_REFERENCE if tke_floor else subject.REFERENCE
    candidate = dict(reference)
    candidate["rel_l2_score"] += 0.10
    candidate.update(overrides)
    mechanics = {
        "transport_output_updated": True,
        "transport_body_updated": True,
        "transport_output_gradient_seen": True,
        "transport_body_gradient_seen": True,
        "transport_invariant_audit": {
            "pass": True,
            "finite": True,
            "mean_max_abs_error": 1e-7,
            "energy_max_abs_error": 1e-7,
            "pressure_bitwise_exact": True,
        },
    }
    return {
        "method": (
            subject.sim.OFFICIAL_POINT_TKE_FLOOR_METHOD
            if tke_floor else (
            subject.sim.OFFICIAL_POINT_TEMPORAL_TRANSPORT_METHOD
            if official_objective
            else subject.sim.ISOENERGETIC_TEMPORAL_TRANSPORT_METHOD)),
        "completed_updates": 10_000,
        "parameters": subject.sim.EXPECTED_TEMPORAL_TRANSPORT_PARAMETERS,
        "checkpoint": "/sealed/model_10000.pth",
        "checkpoint_sha256": "a" * 64,
        "model_config": {
            "isoenergetic_temporal_transport": True,
            "temporal_point_decoder": False,
            "temporal_transport_channels": (
                subject.sim.TEMPORAL_TRANSPORT_CHANNELS),
            "temporal_transport_blocks": subject.sim.TEMPORAL_TRANSPORT_BLOCKS,
            "temporal_transport_kernel": subject.sim.TEMPORAL_TRANSPORT_KERNEL,
            "temporal_transport_max_displacement": (
                subject.sim.TEMPORAL_TRANSPORT_MAX_DISPLACEMENT),
            "physical_energy_controller": True,
            "physical_energy_multiscale": False,
            "physical_energy_horizon_local": False,
            "physical_energy_history_conditioned": False,
        },
        "paired_candidate": {
            "terminal_anchor": candidate,
            "controller_audit": mechanics,
        },
        "training_contract": ({
            "official_point_objective": True,
            "accuracy_weights_rel_tke_mvpe": [46.0, 10.0, 9.0],
            "official_point_controller_tke_floor": tke_floor,
            "tke_aux_persistent_floor": (
                subject.sim.TKE_AUX_PERSISTENT_FLOOR_WEIGHT
                if tke_floor else 0.0),
            "tke_aux_updates": (
                subject.sim.TERMINAL_UPDATE if tke_floor else 1500),
        } if official_objective or tke_floor else {}),
    }


def test_gate_uses_official_point_weights_not_individual_components():
    result = subject.evaluate(summary_with_delta())
    assert result["pass"] is True
    assert result["official_point_proxy_delta"] > 0.0

    # TKE may fall when the 46%-weighted Rel gain more than compensates it.
    result = subject.evaluate(summary_with_delta(
        rel_l2_score=subject.REFERENCE["rel_l2_score"] + 0.10,
        tke_rel_l2_score=subject.REFERENCE["tke_rel_l2_score"] - 0.20))
    assert result["pass"] is True

    result = subject.evaluate(summary_with_delta(
        rel_l2_score=subject.REFERENCE["rel_l2_score"] - 0.10,
        tke_rel_l2_score=subject.REFERENCE["tke_rel_l2_score"] + 0.20))
    assert result["pass"] is False
    assert result["checks"]["official_point_proxy_improved"] is False


def test_gate_rejects_broken_energy_invariant():
    payload = summary_with_delta()
    payload["paired_candidate"]["controller_audit"][
        "transport_invariant_audit"]["energy_max_abs_error"] = 3e-5
    result = subject.evaluate(payload)
    assert result["pass"] is False
    assert result["checks"]["transport_invariants"] is False


def test_gate_attests_official_objective_weights():
    result = subject.evaluate(summary_with_delta(official_objective=True))
    assert result["pass"] is True
    assert result["official_point_objective"] is True
    payload = summary_with_delta(official_objective=True)
    payload["training_contract"][
        "accuracy_weights_rel_tke_mvpe"] = [1.0, 4.0, 1.0]
    result = subject.evaluate(payload)
    assert result["pass"] is False
    assert result["checks"]["accuracy_objective_contract"] is False


def test_gate_attests_persistent_controller_tke_floor_and_stronger_reference():
    result = subject.evaluate(summary_with_delta(tke_floor=True))
    assert result["pass"] is True
    assert result["official_point_controller_tke_floor"] is True
    assert result["reference_role"] == "previous_passed_isoenergetic_transport"

    payload = summary_with_delta(tke_floor=True)
    payload["training_contract"]["tke_aux_persistent_floor"] = 0.003
    result = subject.evaluate(payload)
    assert result["pass"] is False
    assert result["checks"]["controller_tke_floor_contract"] is False
