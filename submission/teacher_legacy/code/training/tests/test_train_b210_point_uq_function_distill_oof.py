from __future__ import annotations

from pathlib import Path
import sys


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_point_uq_function_distill_oof as subject  # noqa: E402


def metric_state(sps: float) -> dict:
    return {
        "samples": 1,
        "rel_error_sum": 0.0,
        "tke_error_sum": 0.0,
        "mvpe_error_sum": 0.0,
        "sps_numerator": sps,
        "sps_scored": 1,
        "sps_covered": 1,
        "sps_width_sum": 0.0,
    }


def fold(control: float, gains: dict[float, float]) -> dict:
    arms = {
        "control": {
            "metric_state": metric_state(control),
            "metrics": {"native_sps_score": 100.0 * control},
        }
    }
    for weight, gain in gains.items():
        arms[subject.arm_label(weight)] = {
            "metric_state": metric_state(control + gain),
            "metrics": {"native_sps_score": 100.0 * (control + gain)},
        }
    return {"arms": arms}


def test_selection_uses_best_passing_preregistered_weight() -> None:
    gains = {0.5: 0.0004, 1.5: 0.0020, 4.0: -0.0010}
    rows = [fold(0.5, gains) for _ in range(5)]
    selected = subject.select_weight(rows)
    assert selected["pass"] is True
    assert selected["selected"] == "distill_1.500"
    assert selected["selected_weight"] == 1.5


def test_objective_changes_only_extent_distillation_weight() -> None:
    control = vars(subject.objective(0.0))
    candidate = vars(subject.objective(1.5))
    differing = {key for key in control if control[key] != candidate[key]}
    assert differing == {"extent_distill_weight"}
