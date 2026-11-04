from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = EVALUATION_ROOT / "evaluate_c208_loao_global_spatial_bias.py"
SPEC = importlib.util.spec_from_file_location("c208_global_spatial_bias", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def synthetic_groups():
    shape = (5, 20, 2, 3, 2)
    point = np.zeros(shape, dtype=np.float32)
    target = np.empty(shape, dtype=np.float32)
    aoas = np.asarray(subject.AOA_VALUES)
    for index in range(5):
        target[index] = np.float32(index + 1)
    return point, target, aoas


def test_literal_dense_bias_uses_other_four_aoas_and_broadcasts_time():
    point, target, aoas = synthetic_groups()

    candidate, biases, counts = subject.build_loao_candidate(
        point, target, aoas)

    assert biases[0].shape == (1, 1, 2, 3, 2)
    assert biases[0].dtype == np.float32
    assert np.allclose(biases[0], 3.5)
    assert np.allclose(candidate[aoas == 0], 3.5)
    assert counts == {held: 4 for held in subject.AOA_VALUES}
    assert subject.e0.centered_fluctuation_max(point, candidate) == 0.0


def test_mutating_held_targets_cannot_change_held_bias_or_candidate():
    point, target, aoas = synthetic_groups()
    first_candidate, first_biases, _ = subject.build_loao_candidate(
        point, target, aoas)
    mutated = target.copy()
    mutated[aoas == 0] = np.float32(999.0)

    second_candidate, second_biases, _ = subject.build_loao_candidate(
        point, mutated, aoas)

    assert np.array_equal(first_biases[0], second_biases[0])
    assert np.array_equal(
        first_candidate[aoas == 0], second_candidate[aoas == 0])


def test_gate_uses_only_loao_pooled_and_held_rows():
    pooled_delta = {
        "rel_l2_score": 0.04,
        "tke_score": 0.0,
        "mvpe_score": 0.05,
        "sps_score": 0.12,
        "point_score": 0.03,
        "coverage": 0.0,
        "mean_width": 0.0,
    }
    rows = [
        {"held_aoa": held, "non_time_shadow_delta": value}
        for held, value in zip(
            subject.AOA_VALUES, (0.08, 0.07, 0.06, 0.05, -0.005))
    ]

    checks, decision = subject.gate_decision(
        pooled_delta, rows, {"synthetic_guard": True},
        elapsed=10.0, gpu_seconds=5.0)
    assert all(checks.values())
    assert decision == subject.PASS_DECISION

    rows[-1]["non_time_shadow_delta"] = -0.02
    checks, decision = subject.gate_decision(
        pooled_delta, rows, {"synthetic_guard": True},
        elapsed=10.0, gpu_seconds=5.0)
    assert not checks["loao_worst_aoa_ge_minus_0_01"]
    assert decision == subject.STOP_DECISION

    try:
        subject.gate_decision(
            pooled_delta, rows[:-1], {"synthetic_guard": True},
            elapsed=10.0, gpu_seconds=5.0)
    except ValueError as error:
        assert "exactly the five" in str(error)
    else:
        raise AssertionError("gate accepted fewer than five held-AoA rows")


def test_operator_freezes_bounds_tke_centered_fluctuation_and_pressure():
    rng = np.random.default_rng(17)
    shape = (10, 20, 25, 38, 2)
    target = rng.normal(0.4, 0.08, size=shape).astype(np.float32)
    point = target + rng.normal(0.01, 0.02, size=shape).astype(np.float32)
    aoas = np.repeat(np.asarray(subject.AOA_VALUES), 2)
    lower = point - np.float32(0.06)
    upper = point + np.float32(0.06)
    lower_hash = subject.e0.array_sha256(lower)
    upper_hash = subject.e0.array_sha256(upper)

    candidate, _biases, _counts = subject.build_loao_candidate(
        point, target, aoas)
    baseline = subject.e0.score_bundle(point, target, lower, upper)
    shifted = subject.e0.score_bundle(candidate, target, lower, upper)
    point_full = subject.e0.pad_zero_pressure(point)
    candidate_full = subject.e0.pad_zero_pressure(candidate)

    assert subject.e0.array_sha256(lower) == lower_hash
    assert subject.e0.array_sha256(upper) == upper_hash
    assert shifted["coverage"] == baseline["coverage"]
    assert shifted["covered_elements"] == baseline["covered_elements"]
    assert shifted["scored_elements"] == baseline["scored_elements"]
    assert shifted["mean_width"] == baseline["mean_width"]
    assert abs(shifted["tke_score"] - baseline["tke_score"]) <= 1e-5
    assert subject.e0.centered_fluctuation_max(point, candidate) <= 1e-6
    assert np.array_equal(point_full[..., :2], point)
    assert np.array_equal(candidate_full[..., :2], candidate)
    assert np.count_nonzero(point_full[..., 2]) == 0
    assert np.count_nonzero(candidate_full[..., 2]) == 0
