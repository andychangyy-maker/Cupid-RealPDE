from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import compare_b210_np_to_frozen_c208 as subject  # noqa: E402


def _state(samples: int, offset: float) -> dict:
    state = subject.b210.smu1.new_metric_state()
    state.update({
        "samples": samples,
        "rel_error_sum": samples * (0.10 + offset),
        "tke_error_sum": samples * (0.20 + offset),
        "mvpe_error_sum": samples * (0.30 + offset),
        "sps_numerator": samples * (20.0 + offset),
        "sps_scored": samples * 100,
        "sps_covered": samples * 80,
        "sps_width_sum": samples * 5.0,
    })
    return state


def _audit(samples: int, token: str) -> dict:
    return {
        "ordered_raw_target_sha256": token * 64,
        "ordered_scored_mask_sha256": token.upper() * 64,
        "samples": samples,
        "scored_elements": samples * 100,
    }


def _rows():
    candidates = []
    anchors = []
    for index, (held, samples) in enumerate(((10, 1), (0, 3))):
        candidate_state = _state(samples, -0.01 * (index + 1))
        anchor_state = _state(samples, 0.0)
        audit = _audit(samples, str(index + 1))
        common = {
            "held_aoa": held,
            "samples": samples,
            "manifest_file_sha256": "a" * 64,
            "manifest": {
                "semantic_sha256": "b" * 64,
                "windows": 2544,
                "excluded": "7575_0.h5",
            },
            "normalizer": {"sha256": "c" * 64},
            "held_ordered_key_sha256": str(index + 3) * 64,
        }
        candidates.append({
            **common,
            "endpoints": {
                subject.CANDIDATE_ENDPOINT: {
                    "metric_state": candidate_state,
                    "metrics": subject.b210.smu1.finalize_metric_state(
                        candidate_state),
                    "held_target_audit": audit,
                },
            },
        })
        anchors.append({
            **common,
            "endpoints": {
                subject.ANCHOR_ENDPOINT: {
                    "metric_state": anchor_state,
                    "metrics": subject.b210.smu1.finalize_metric_state(
                        anchor_state),
                    "held_target_audit": copy.deepcopy(audit),
                },
            },
        })
    return candidates, anchors


@pytest.fixture(autouse=True)
def _skip_upstream_deep_validation(monkeypatch):
    monkeypatch.setattr(subject.b210, "validate_fold_rows", lambda _rows: None)
    monkeypatch.setattr(subject.mef2, "validate_fold_rows", lambda _rows: None)


def test_comparison_uses_fixed_endpoints_and_pools_sufficient_states():
    candidates, anchors = _rows()
    result = subject.compare_rows(candidates, anchors)
    expected_candidate_state = subject.b210.smu1.merge_metric_states([
        row["endpoints"][subject.CANDIDATE_ENDPOINT]["metric_state"]
        for row in candidates
    ])
    expected_anchor_state = subject.b210.smu1.merge_metric_states([
        row["endpoints"][subject.ANCHOR_ENDPOINT]["metric_state"]
        for row in anchors
    ])
    assert result["pooled_candidate_metric_state"] == expected_candidate_state
    assert result["pooled_anchor_metric_state"] == expected_anchor_state
    assert result["pooled_candidate_metrics"] == (
        subject.b210.smu1.finalize_metric_state(expected_candidate_state))
    fold_score_average = sum(
        row["endpoints"][subject.CANDIDATE_ENDPOINT]["metrics"][
            "rel_l2_score"]
        for row in candidates
    ) / 2
    assert result["pooled_candidate_metrics"]["rel_l2_score"] != (
        pytest.approx(fold_score_average))
    assert result["fixed_endpoints"] == {
        "candidate": "ema_u12000",
        "anchor": "control_ema",
        "endpoint_selection_allowed": False,
        "raw_endpoint_may_replace_primary": False,
    }
    assert result["gate"]["primary_endpoint"] == "ema_u12000"
    assert result["official_evidence"] is False
    assert result["submission_authorized"] is False


@pytest.mark.parametrize("field", [
    "ordered_raw_target_sha256",
    "ordered_scored_mask_sha256",
])
def test_comparison_rejects_target_or_mask_mismatch(field):
    candidates, anchors = _rows()
    anchors[0]["endpoints"][subject.ANCHOR_ENDPOINT][
        "held_target_audit"][field] = "f" * 64
    with pytest.raises(RuntimeError, match="target/mask SHA audits differ"):
        subject.compare_rows(candidates, anchors)


def test_comparison_rejects_held_key_mismatch():
    candidates, anchors = _rows()
    anchors[1]["held_ordered_key_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="ordered keys differ"):
        subject.compare_rows(candidates, anchors)


def test_comparison_rejects_metric_state_sample_mismatch():
    candidates, anchors = _rows()
    anchors[0]["endpoints"][subject.ANCHOR_ENDPOINT][
        "metric_state"]["samples"] += 1
    with pytest.raises(RuntimeError, match="metric_state samples changed"):
        subject.compare_rows(candidates, anchors)


def test_comparison_rejects_any_raw_endpoint_substitution():
    candidates, anchors = _rows()
    candidates[0]["endpoints"]["raw_u12000"] = copy.deepcopy(
        candidates[0]["endpoints"][subject.CANDIDATE_ENDPOINT])
    del candidates[0]["endpoints"][subject.CANDIDATE_ENDPOINT]
    with pytest.raises(RuntimeError, match="fixed endpoint is missing"):
        subject.compare_rows(candidates, anchors)
