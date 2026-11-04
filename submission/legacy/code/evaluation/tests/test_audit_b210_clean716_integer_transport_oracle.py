from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "audit_b210_clean716_integer_transport_oracle.py")
SPEC = importlib.util.spec_from_file_location(
    "b210_clean716_integer_transport_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_contract_is_explicit_target_visible_finite_integer_oracle():
    assert subject.EVIDENCE_CLASS == "oracle"
    assert subject.HORIZON_BLOCKS == ((0, 5), (5, 10), (10, 15), (15, 20))
    assert len(subject.CANDIDATE_SHIFTS) == 15
    assert set(subject.CANDIDATE_SHIFTS) == {
        (dy, dx) for dy in range(-1, 2) for dx in range(-2, 3)}
    assert subject.CANDIDATE_SHIFTS[0] == (0, 0)
    assert subject.WAKE_X_SLICE == (13, 64)


def test_integer_translation_direction_border_and_identity_are_frozen():
    value = torch.zeros(1, 20, 4, 6, 2, dtype=torch.float32)
    value[..., 0] = torch.arange(6, dtype=torch.float32)
    value[..., 1] = torch.arange(4, dtype=torch.float32)[None, None, :, None]

    identity = subject.integer_translate(value, 0, 0)
    right = subject.integer_translate(value, 0, 1)
    down = subject.integer_translate(value, 1, 0)

    assert torch.equal(identity, value)
    assert identity.data_ptr() != value.data_ptr()
    assert right[0, 0, 2, 3, 0] == value[0, 0, 2, 2, 0]
    assert right[0, 0, 2, 0, 0] == value[0, 0, 2, 0, 0]
    assert down[0, 0, 2, 3, 1] == value[0, 0, 1, 3, 1]
    assert down[0, 0, 0, 3, 1] == value[0, 0, 0, 3, 1]


def test_oracle_recovers_known_per_block_shifts_and_keeps_identity_ties():
    generator = torch.Generator().manual_seed(17)
    prediction = torch.randn(
        2, 20, 6, 8, 2, generator=generator, dtype=torch.float32)
    target = prediction.clone()
    expected = [(1, 2), (0, -1), (-1, 1), (1, -2)]
    for block, (start, stop) in enumerate(subject.HORIZON_BLOCKS):
        target[0, start:stop] = subject.integer_translate(
            prediction[0:1, start:stop], *expected[block])[0]

    oracle, indices = subject.target_visible_block_oracle(prediction, target)
    selected = [
        subject.CANDIDATE_SHIFTS[int(index)] for index in indices[0].tolist()]

    assert selected == expected
    assert indices[1].tolist() == [subject.IDENTITY_INDEX] * 4
    assert torch.equal(oracle, target)


def test_official_metrics_and_46_10_9_point_are_exact():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(3, 20, 32, 64, 2)).astype(np.float32)
    prediction = target + rng.normal(
        scale=0.05, size=target.shape).astype(np.float32)
    errors = subject.official_error_triples(prediction, target)
    metrics = subject.metrics_from_errors(errors)

    expected = {
        "rel_l2_score": subject.OFFICIAL.score_error(float(np.mean(
            subject.OFFICIAL.rel_l2_per_sample(prediction, target, 2)))),
        "tke_score": subject.OFFICIAL.score_error(float(np.mean(
            subject.OFFICIAL.tke_rel_l2_per_sample(prediction, target, 2)))),
        "mvpe_score": subject.OFFICIAL.score_error(float(np.mean(
            subject.OFFICIAL.mvpe_rel_l2_per_sample(prediction, target)))),
    }
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value)
    contribution = 0.46 * expected["rel_l2_score"] + (
        0.10 * expected["tke_score"]) + 0.09 * expected["mvpe_score"]
    assert metrics["point_final_contribution_46_10_9"] == pytest.approx(
        contribution)
    assert metrics["point_score_normalized_46_10_9"] == pytest.approx(
        contribution / 0.65)


def test_sse_and_selection_reports_preserve_reduction_and_counts():
    baseline = np.asarray([4.0, 6.0])
    oracle = np.asarray([1.0, 5.0])
    report = subject.sse_comparison(baseline, oracle)
    assert report["baseline_uv_sse"] == 10.0
    assert report["oracle_uv_sse"] == 6.0
    assert report["uv_sse_reduction_fraction"] == pytest.approx(0.4)

    indices = np.full((2, 4), subject.IDENTITY_INDEX, dtype=np.int16)
    indices[0, 0] = subject.CANDIDATE_SHIFTS.index((1, 2))
    distribution = subject.selection_distribution(indices)
    assert distribution["all_blocks"]["choices"] == 8
    assert distribution["all_blocks"]["identity_choices"] == 7
    assert distribution["all_blocks"]["non_identity_fraction"] == 0.125
