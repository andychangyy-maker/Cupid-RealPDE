from __future__ import annotations

from pathlib import Path
import sys


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = EVALUATION_ROOT.parent
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_b210_temporal_point_sim_gate as subject  # noqa: E402


def summary_with_delta(**overrides):
    candidate = {
        key: value + subject.THRESHOLDS[key] + 0.01
        for key, value in subject.REFERENCE.items()
    }
    candidate.update(overrides)
    mechanics = {
        "temporal_output_updated": True,
        "temporal_body_updated": True,
        "temporal_mean_gradient_seen": True,
        "temporal_fluctuation_gradient_seen": True,
        "temporal_projection_gradient_seen": True,
        "temporal_body_gradient_seen": True,
    }
    return {
        "method": subject.sim.TEMPORAL_POINT_DECODER_METHOD,
        "completed_updates": 10_000,
        "parameters": subject.sim.EXPECTED_TEMPORAL_POINT_DECODER_PARAMETERS,
        "checkpoint": "/sealed/model_10000.pth",
        "checkpoint_sha256": "a" * 64,
        "model_config": {
            "temporal_point_decoder": True,
            "temporal_point_channels": subject.sim.TEMPORAL_POINT_CHANNELS,
            "temporal_point_blocks": subject.sim.TEMPORAL_POINT_BLOCKS,
            "temporal_point_kernel": subject.sim.TEMPORAL_POINT_KERNEL,
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
    weak = subject.REFERENCE["tke_rel_l2_score"] + 0.149
    result = subject.evaluate(summary_with_delta(tke_rel_l2_score=weak))
    assert result["pass"] is False
    assert result["checks"]["tke_rel_l2_score"] is False


def test_gate_rejects_untrained_temporal_body():
    payload = summary_with_delta()
    payload["paired_candidate"]["controller_audit"][
        "temporal_body_gradient_seen"] = False
    result = subject.evaluate(payload)
    assert result["pass"] is False
    assert result["checks"]["mechanics"] is False
