from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / (
    "evaluate_b210_xcb_sim_anchor_point_oof.py")
SPEC = importlib.util.spec_from_file_location("sim_anchor_oof", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def test_parameter_scope_excludes_uq_and_separates_xcb():
    assert subject.parameter_in_scope("encoder.0.weight", "all_point")
    assert not subject.parameter_in_scope("encoder.0.weight", "xcb")
    assert subject.parameter_in_scope("member_feature_adapters.0.weight", "xcb")
    assert not subject.parameter_in_scope("quantile_head.weight", "all_point")


def test_parse_alphas_has_strict_open_interval():
    assert subject.parse_alphas(".1,.05,.1") == [0.05, 0.1]
    with pytest.raises(ValueError, match="inside"):
        subject.parse_alphas("0,.1")


def test_official_point_contribution_uses_published_weights():
    metrics = {
        "rel_l2_score": 10.0,
        "tke_rel_l2_score": 20.0,
        "mvpe_rel_l2_score": 30.0,
    }
    assert subject.point_contribution(metrics) == pytest.approx(9.3)


def test_reproduction_tolerance_is_below_the_admission_delta():
    assert subject.REPRODUCTION_ABS_TOLERANCE == pytest.approx(5.0e-5)
    assert subject.REPRODUCTION_ABS_TOLERANCE < 0.005 / 10
