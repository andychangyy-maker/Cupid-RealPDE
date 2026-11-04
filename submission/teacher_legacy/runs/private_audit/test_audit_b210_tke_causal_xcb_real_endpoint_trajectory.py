from __future__ import annotations

import pytest

import audit_b210_tke_causal_xcb_real_endpoint_trajectory as subject


def test_trajectory_updates_are_fixed_and_terminal_is_not_snapshotted():
    assert subject.SNAPSHOT_UPDATES == (500, 1000, 1500, 2000, 3000)
    assert subject.TERMINAL_UPDATE == 4000
    assert max(subject.SNAPSHOT_UPDATES) < subject.TERMINAL_UPDATE
    assert subject.ARMS == ("standard", "causal")


def test_official_delta_uses_published_quality_weights():
    control = {
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 85.0,
        "native_sps_score": 40.0,
    }
    causal = {
        "rel_l2_score": 91.0,
        "tke_rel_l2_score": 82.0,
        "mvpe_rel_l2_score": 88.0,
        "native_sps_score": 44.0,
    }
    value = subject.official_delta(causal, control)
    assert value["official_point65_delta"] == pytest.approx(
        0.46 + 0.20 + 0.27)
    assert value["official_non_time_delta"] == pytest.approx(1.93)


def test_historical_terminal_state_hashes_are_distinct_and_well_formed():
    hashes = {
        value["terminal_model_state_sha256"]
        for value in subject.EXPECTED.values()
    }
    assert len(hashes) == 2
    for value in hashes:
        assert len(value) == 64
        int(value, 16)
    assert subject.TERMINAL_REPRODUCTION_TOLERANCE == {
        "rel_l2_score": 0.02,
        "tke_rel_l2_score": 0.02,
        "mvpe_rel_l2_score": 0.02,
        "native_sps_score": 0.05,
    }
