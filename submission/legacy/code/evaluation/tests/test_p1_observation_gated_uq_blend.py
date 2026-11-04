from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

import evaluate_p1_observation_gated_uq_blend as subject  # noqa: E402


def fake_components(n: int = 2, height: int = 3,
                    width: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    shape = (n, 20, height, width, 3)
    a_prediction = rng.normal(0.0, 0.2, shape).astype(np.float32)
    b_prediction = rng.normal(0.0, 0.2, shape).astype(np.float32)
    c_prediction = rng.normal(0.0, 0.2, shape).astype(np.float32)
    a_extent_lower = rng.uniform(0.01, 0.08, shape).astype(np.float32)
    a_extent_upper = rng.uniform(0.01, 0.08, shape).astype(np.float32)
    b_extent_lower = rng.uniform(0.01, 0.08, shape).astype(np.float32)
    b_extent_upper = rng.uniform(0.01, 0.08, shape).astype(np.float32)
    for value in (a_prediction, b_prediction, c_prediction):
        value[..., 2] = 0.0
    return {
        "a_prediction": a_prediction,
        "a_lower": a_prediction - a_extent_lower,
        "a_upper": a_prediction + a_extent_upper,
        "b_prediction": b_prediction,
        "b_lower": b_prediction - b_extent_lower,
        "b_upper": b_prediction + b_extent_upper,
        "c_prediction": c_prediction,
    }


def test_frozen_cache_array_identities_match_exact_artifact():
    assert subject.POINT_ARRAY_SHA256 == (
        "9b2d1f11e6017f6e505d52a62ec2158aa67c2bb86d02bdb10a66823dfbb46c52")
    assert subject.TARGET_ARRAY_SHA256 == (
        "88d334a29e9adc540471ab3cba859c8f8d07cf2bed4f3377a18eccd1ca8b2cc9")


def test_point_cache_passes_plain_string_to_torch_mmap(monkeypatch, tmp_path):
    class LoadedAsString(RuntimeError):
        pass

    monkeypatch.setattr(
        subject.point_gate.s1, "file_identity",
        lambda _path: {"sha256": subject.POINT_CACHE_SHA256})

    def fake_load(filename, **kwargs):
        assert isinstance(filename, str)
        assert kwargs["mmap"] is True
        raise LoadedAsString

    monkeypatch.setattr(subject.torch, "load", fake_load)
    with pytest.raises(LoadedAsString):
        subject.load_point_cache(tmp_path / "point.pt")


def test_constant_p0_weights_match_existing_compose_bitwise_and_freeze_point():
    components = fake_components()
    arrays_a = {
        "prediction": components["a_prediction"],
        "lower": components["a_lower"],
        "upper": components["a_upper"],
    }
    arrays_b = {
        "prediction": components["b_prediction"],
        "lower": components["b_lower"],
        "upper": components["b_upper"],
    }
    reference = subject.point_gate.s1.blend_candidate(
        arrays_a, arrays_b, components["c_prediction"],
        weight_b=subject.BASELINE_WEIGHT_B,
        mean_weight=subject.BASELINE_WEIGHT_C)
    exact_point = np.arange(
        np.prod(components["a_prediction"].shape), dtype=np.float32).reshape(
            components["a_prediction"].shape)
    point_before = exact_point.copy()
    lower, upper = subject.compose_sample_bounds(
        components,
        np.full(2, subject.BASELINE_WEIGHT_B),
        np.full(2, subject.BASELINE_WEIGHT_C))

    assert np.array_equal(lower, reference["lower"])
    assert np.array_equal(upper, reference["upper"])
    assert np.array_equal(exact_point, point_before)


def test_stratified_sample_is_deterministic_bounded_and_oracle_monotone():
    components = fake_components(n=1, height=4, width=5)
    target = np.ones((1, 20, 4, 5, 3), dtype=np.float32) * 0.05
    target[..., 2] = 0.0
    first = subject.stratified_sample(
        components, target, 0, sample_weight=0.8, max_elements=97)
    second = subject.stratified_sample(
        components, target, 0, sample_weight=0.8, max_elements=97)

    assert len(first["target"]) <= 97
    for key in (
            "target", "a_lower", "lower_delta", "a_upper",
            "upper_delta", "correction_index"):
        assert np.array_equal(first[key], second[key])
    oracle = subject.coordinate_oracle(first)
    assert subject.WEIGHT_B_BOUNDS[0] <= oracle["weight_b"] <= (
        subject.WEIGHT_B_BOUNDS[1])
    assert subject.WEIGHT_C_BOUNDS[0] <= oracle["weight_c"] <= (
        subject.WEIGHT_C_BOUNDS[1])
    assert oracle["selected_objective"] >= oracle["identity_objective"]


def test_coordinate_tie_prefers_zero_then_small_absolute_delta():
    trials = [
        {"objective": 1.0, "delta": 0.10},
        {"objective": 1.0, "delta": -0.025},
        {"objective": 1.0, "delta": 0.0},
    ]
    assert subject.choose_trial(trials)["delta"] == 0.0
    trials.pop()
    assert subject.choose_trial(trials)["delta"] == -0.025


def test_single_alpha_ridge_uses_train_zscore_and_clips_both_outputs():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(50, len(subject.FEATURE_NAMES)))
    labels = np.column_stack((
        0.625 + 0.04 * features[:, 0],
        0.25 + 0.03 * features[:, 1],
    ))
    model = subject.fit_ridge(features, labels)
    assert np.allclose(model["mean"], features.mean(axis=0))
    assert np.allclose(model["scale"], features.std(axis=0))
    extreme = np.full((2, len(subject.FEATURE_NAMES)), 1e6)
    predicted = subject.predict_ridge(model, extreme)
    assert np.all(predicted[:, 0] >= subject.WEIGHT_B_BOUNDS[0])
    assert np.all(predicted[:, 0] <= subject.WEIGHT_B_BOUNDS[1])
    assert np.all(predicted[:, 1] >= subject.WEIGHT_C_BOUNDS[0])
    assert np.all(predicted[:, 1] <= subject.WEIGHT_C_BOUNDS[1])

    groups = np.arange(len(features)) % 5
    crossfit, records = subject.crossfit_ridge(
        features, labels, groups, tuple(range(5)))
    assert crossfit.shape == labels.shape
    assert np.isfinite(crossfit).all()
    assert len(records) == 5
    assert all(record["test_samples"] == 10 for record in records)


def test_gate_contract_is_exact_and_independent():
    pooled = {
        "sps_delta": 0.08,
        "coverage_delta": -0.0025,
        "width_ratio": 1.02,
    }
    condition = [{"sps_delta": value} for value in (
        0.1, 0.1, 0.1, 0.1, -0.03)]
    loao = [{"sps_delta": value} for value in (
        0.1, 0.1, 0.1, 0.1, -0.05)]
    checks = subject.gate_checks(0.20, pooled, condition, loao)
    assert all(checks.values())

    failing = dict(pooled)
    failing["coverage_delta"] = -0.0025001
    checks = subject.gate_checks(0.20, failing, condition, loao)
    assert not checks["deployable_coverage_drop_le_0_0025"]
    assert checks["deployable_pooled_sps_gain_ge_0_08"]


def test_weight_ranges_reject_out_of_contract_values():
    components = fake_components(n=1)
    with pytest.raises(ValueError, match="outside contract"):
        subject.compose_sample_bounds(
            components, np.asarray([1.01]), np.asarray([0.25]))
