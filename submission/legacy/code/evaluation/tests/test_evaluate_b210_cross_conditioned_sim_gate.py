from __future__ import annotations

from pathlib import Path
import sys


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = EVALUATION_ROOT.parent
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_b210_cross_conditioned_sim_gate as subject  # noqa: E402


def summary_with_delta(**overrides):
    candidate = {
        key: value + subject.THRESHOLDS[key] + 0.01
        for key, value in subject.REFERENCE.items()
    }
    candidate.update(overrides)
    mechanics = {
        "cross_projection_updated": True,
        "cross_body_updated": True,
        "cross_projection_gradient_seen": True,
        "cross_body_gradient_seen": True,
    }
    return {
        "method": subject.sim.CROSS_CONDITIONED_BACKBONE_METHOD,
        "completed_updates": 10_000,
        "parameters": (
            subject.sim.EXPECTED_CROSS_CONDITIONED_BACKBONE_PARAMETERS),
        "checkpoint": "/sealed/model_10000.pth",
        "checkpoint_sha256": "a" * 64,
        "model_config": {
            "cross_conditioned_backbone": True,
            "cross_conditioned_steps": subject.sim.CROSS_CONDITIONED_STEPS,
            "cross_conditioned_channels": subject.sim.CROSS_CONDITIONED_CHANNELS,
            "physical_energy_controller": True,
            "physical_energy_multiscale": False,
            "physical_energy_horizon_local": False,
            "physical_energy_history_conditioned": False,
        },
        "paired_candidate": {
            "terminal_anchor": candidate,
            "controller_audit": mechanics,
        },
    }


def test_gate_passes_only_when_every_metric_passes():
    result = subject.evaluate(summary_with_delta())
    assert result["pass"] is True
    weak = subject.REFERENCE["mvpe_rel_l2_score"] - 0.001
    result = subject.evaluate(summary_with_delta(mvpe_rel_l2_score=weak))
    assert result["pass"] is False
    assert result["checks"]["mvpe_rel_l2_score"] is False


def test_gate_rejects_untrained_cross_conditioner():
    payload = summary_with_delta()
    payload["paired_candidate"]["controller_audit"][
        "cross_body_updated"] = False
    result = subject.evaluate(payload)
    assert result["pass"] is False
    assert result["checks"]["mechanics"] is False
