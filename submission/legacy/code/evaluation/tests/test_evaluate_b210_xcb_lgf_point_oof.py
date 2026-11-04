from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / (
    "evaluate_b210_xcb_lgf_point_oof.py")
SPEC = importlib.util.spec_from_file_location("xcb_lgf_oof", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def test_published_point_and_non_time_delta():
    baseline = {
        "rel_l2_score": 90.0, "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 70.0, "native_sps_score": 40.0,
    }
    candidate = {
        "rel_l2_score": 91.0, "tke_rel_l2_score": 82.0,
        "mvpe_rel_l2_score": 73.0, "native_sps_score": 44.0,
    }
    result = subject.comparison(candidate, baseline)
    assert result["official_point_contribution_delta"] == pytest.approx(0.93)
    assert result["non_time_contribution_delta"] == pytest.approx(1.93)


def test_contract_rejects_missing_member_floor():
    contract = {
        "point_accuracy_weights": [46.0, 10.0, 9.0],
        "member_accuracy_weights": [46.0, 10.0, 9.0],
        "legacy_tke_gradient_floor": True,
        "legacy_tke_gradient_floor_on_members": False,
        "legacy_point_accuracy_weights": [5.0, 3.0, 2.0],
        "legacy_member_accuracy_weights": [5.0, 3.0, 2.0],
    }
    with pytest.raises(ValueError, match="frozen"):
        subject.validate_candidate_contract(contract)


def test_fold_parser_is_unique_and_bounded():
    assert subject.parse_folds("0,2,4") == [0, 2, 4]
    with pytest.raises(ValueError, match="unique"):
        subject.parse_folds("0,0")
