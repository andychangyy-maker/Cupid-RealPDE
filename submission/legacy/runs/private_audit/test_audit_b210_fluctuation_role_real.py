from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).with_name("audit_b210_fluctuation_role_real.py")
SPEC = importlib.util.spec_from_file_location("fluctuation_role_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def test_component_state_is_exact_for_perfect_prediction():
    target = torch.randn(3, 20, 4, 5, 3)
    state = audit.new_component_state()
    audit.update_component_state(state, target.clone(), target)
    values = audit.finalize_component_state(state)
    assert values["temporal_mean_relative_l2"] == 0.0
    assert values["fluctuation_relative_l2"] == 0.0
    assert values["fluctuation_cosine"] == pytest.approx(1.0)
    assert values["prediction_to_target_fluctuation_rms_ratio"] == pytest.approx(1.0)


def test_delta_uses_published_track1_weights():
    keys = (
        "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
        "point_score", "native_sps_score", "native_sps_coverage",
        "native_sps_mean_width",
    )
    control = {key: 10.0 for key in keys}
    candidate = {key: 11.0 for key in keys}
    values = audit.delta(candidate, control)
    assert values["official_point_contribution"] == pytest.approx(0.65)
    assert values["official_non_time_contribution"] == pytest.approx(0.90)
