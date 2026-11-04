from __future__ import annotations

from pathlib import Path
import sys

import pytest


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_legacy_tke_floor_real_paired as subject  # noqa: E402


def _comparison(delta, by_aoa_delta=None):
    by_aoa_delta = delta if by_aoa_delta is None else by_aoa_delta
    return {
        "delta": delta,
        "by_aoa": {
            str(aoa): {"delta": dict(by_aoa_delta)}
            for aoa in subject.dr1.AOA_VALUES
        },
    }


def test_arm_contract_changes_only_point_objective_and_floor():
    control = subject._arm_contract("control")
    candidate = subject._arm_contract("candidate")
    assert control["point_accuracy_weights"] == (5.0, 3.0, 2.0)
    assert control["legacy_tke_gradient_floor"] is False
    assert candidate["point_accuracy_weights"] == (46.0, 10.0, 9.0)
    assert candidate["legacy_tke_gradient_floor"] is True
    assert candidate["legacy_point_accuracy_weights"] == (5.0, 3.0, 2.0)


def test_official_point_contribution_uses_published_weights():
    delta = {
        "rel_l2_score": 1.0,
        "tke_rel_l2_score": 1.0,
        "mvpe_rel_l2_score": 1.0,
    }
    assert subject.official_point_contribution(delta) == pytest.approx(0.65)


def test_promotion_gate_requires_joint_gain_and_aoa_breadth():
    passing_delta = {
        "rel_l2_score": 0.08,
        "tke_rel_l2_score": 0.03,
        "mvpe_rel_l2_score": 0.02,
        "native_sps_score": 0.0,
    }
    passing = subject.promotion_gate(_comparison(passing_delta))
    assert passing["pass"] is True
    failing = dict(passing_delta, rel_l2_score=-0.031)
    assert subject.promotion_gate(_comparison(failing))["pass"] is False
