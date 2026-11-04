from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "audit_b210_clean716_modal_headroom.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_b210_clean716_modal_headroom", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_contract_is_loao_whole_future_target_visible_oracle():
    assert subject.EVIDENCE_CLASS == "oracle"
    assert subject.RANKS == (8, 16, 32)
    assert subject.AOA_VALUES == (0, 5, 10, 15, 20)
    assert subject.EXPERIMENT_CONTRACT["target_visible"] is True
    assert subject.EXPERIMENT_CONTRACT["deployable"] is False
    assert "complete zero-time-mean" in (
        subject.EXPERIMENT_CONTRACT["pod_sample"])


def test_loao_fit_indices_exclude_every_held_row():
    aoa = np.repeat(np.asarray(subject.AOA_VALUES), 3)
    for held in subject.AOA_VALUES:
        fit = subject.fit_indices_for_held_aoa(aoa, held)
        assert len(fit) == 12
        assert np.all(aoa[fit] != held)
        assert set(aoa[fit].tolist()) == set(subject.AOA_VALUES) - {held}
    with pytest.raises(ValueError, match="five-group"):
        subject.fit_indices_for_held_aoa(aoa[:-3], 0)


def test_zero_time_mean_future_is_exact_enough_and_non_mutating():
    rng = np.random.default_rng(11)
    values = rng.normal(size=(4, 5, 3, 4, 2)).astype(np.float32)
    before = values.copy()
    fluctuation = subject.zero_time_mean_future(values)
    np.testing.assert_array_equal(values, before)
    np.testing.assert_allclose(
        fluctuation.mean(axis=1), 0.0, rtol=0.0, atol=2.0e-7)


def _rank_two_futures(samples: int = 10) -> np.ndarray:
    first = np.zeros((4, 3, 5, 2), dtype=np.float32)
    second = np.zeros_like(first)
    first[:, :, :, 0] = np.asarray([-2, -1, 1, 2], dtype=np.float32)[
        :, None, None]
    second[:, :, :, 1] = np.asarray([1, -1, -1, 1], dtype=np.float32)[
        :, None, None]
    coefficient = np.arange(1, samples + 1, dtype=np.float32)
    values = (
        coefficient[:, None, None, None, None] * first
        + (coefficient[::-1] ** 2)[:, None, None, None, None] * second)
    return values.astype(np.float32)


def test_dual_pod_and_projection_reconstruct_known_rank_two_futures():
    values = _rank_two_futures()
    matrix = subject.materialize_fluctuation_matrix(
        values, np.arange(len(values)))
    basis, report = subject.dual_pod_basis(
        matrix, max_rank=2, device=torch.device("cpu"))
    assert basis.shape == (2, matrix.shape[1])
    torch.testing.assert_close(
        basis @ basis.T, torch.eye(2), rtol=2.0e-5, atol=2.0e-5)
    assert report["basis_orthogonality_max_abs_error"] < 2.0e-5

    projected, projection = subject.project_held_fluctuations(
        values, np.arange(len(values)), basis, ranks=(1, 2),
        device=torch.device("cpu"), batch_size=3)
    assert projected[1].shape == values.shape
    np.testing.assert_allclose(projected[2], values, rtol=2.0e-5, atol=2.0e-5)
    assert projection["by_rank"]["2"][
        "held_fluctuation_sse_removed_fraction"] == pytest.approx(
            1.0, abs=2.0e-6)
    assert projection["by_rank"]["2"][
        "projected_temporal_mean_max_abs"] < 1.0e-5


def test_official_metric_summary_uses_published_point_weights():
    rng = np.random.default_rng(29)
    target = rng.normal(size=(3, 20, 32, 64, 2)).astype(np.float32)
    prediction = target + rng.normal(
        scale=0.05, size=target.shape).astype(np.float32)
    errors = subject.official_error_triples(prediction, target)
    metrics = subject.metrics_from_errors(errors)
    contribution = (
        0.46 * metrics["rel_l2_score"]
        + 0.10 * metrics["tke_score"]
        + 0.09 * metrics["mvpe_score"])
    assert metrics["point_final_contribution_46_10_9"] == pytest.approx(
        contribution)
    assert metrics["point_score_normalized_46_10_9"] == pytest.approx(
        contribution / 0.65)
