from __future__ import annotations

from pathlib import Path
import sys


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = EVALUATION_ROOT.parent
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_b210_history_energy_sim_gate as subject  # noqa: E402


def summary_with_delta(**overrides):
    candidate = {
        key: value + subject.THRESHOLDS[key] + 0.01
        for key, value in subject.REFERENCE.items()
    }
    candidate.update(overrides)
    return {
        "method": subject.sim.HISTORY_ENERGY_METHOD,
        "completed_updates": 10_000,
        "parameters": subject.sim.EXPECTED_HISTORY_ENERGY_PARAMETERS,
        "checkpoint": "/sealed/model_10000.pth",
        "checkpoint_sha256": "a" * 64,
        "model_config": {
            "physical_energy_history_conditioned": True,
            "physical_energy_history_steps": 5,
            "physical_energy_multiscale": False,
            "physical_energy_horizon_local": False,
        },
        "paired_candidate": {"terminal_anchor": candidate},
    }


def test_gate_passes_only_when_all_frozen_thresholds_pass():
    result = subject.evaluate(summary_with_delta())
    assert result["pass"] is True
    assert result["decision"].startswith("PASS_")

    weak = subject.REFERENCE["tke_rel_l2_score"] + 0.199
    result = subject.evaluate(summary_with_delta(tke_rel_l2_score=weak))
    assert result["pass"] is False
    assert result["checks"]["tke_rel_l2_score"] is False


def test_gate_rejects_wrong_architecture_even_with_good_metrics():
    payload = summary_with_delta()
    payload["model_config"]["physical_energy_multiscale"] = True
    result = subject.evaluate(payload)
    assert result["pass"] is False
    assert result["checks"]["separate_arm"] is False
