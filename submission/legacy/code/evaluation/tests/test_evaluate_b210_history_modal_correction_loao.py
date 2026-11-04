from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "evaluate_b210_history_modal_correction_loao.py")
SPEC = importlib.util.spec_from_file_location(
    "evaluate_b210_history_modal_correction_loao", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def test_contract_limits_claim_and_freezes_precision_and_weights():
    assert subject.CONTRACT["end_to_end_condition_holdout"] is False
    assert subject.CONTRACT["baseline_training_scope"] == (
        "full_AoA_frozen_baseline")
    assert subject.CONTRACT["pod_and_projection_dtype"] == "float32"
    assert subject.CONTRACT["linear_controller_dtype"] == "float64"
    assert subject.POINT_WEIGHTS == {
        "rel_l2_score": 0.46,
        "tke_score": 0.10,
        "mvpe_score": 0.09,
    }


def test_loao_fit_indices_exclude_every_held_row():
    aoa = np.repeat(np.asarray(subject.EXPECTED_AOAS), 4)
    for held in subject.EXPECTED_AOAS:
        fit, test = subject.fit_indices_for_held_aoa(aoa, held)
        assert np.all(aoa[fit] != held)
        assert np.all(aoa[test] == held)
        assert not np.intersect1d(fit, test).size


def _synthetic_flows(samples: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    history = rng.normal(size=(samples, 4, 2, 3, 2)).astype(np.float32)
    baseline = (0.7 * history + 0.1 * np.roll(history, 1, axis=1)).astype(
        np.float32)
    target = (baseline + 0.08 * history - 0.08 * history.mean(
        axis=1, keepdims=True)).astype(np.float32)
    return history, baseline, target


def test_fit_api_is_fit_sliced_and_predict_api_cannot_accept_target():
    assert "held" not in inspect.signature(
        subject.fit_modal_controller).parameters
    assert "target" not in inspect.signature(
        subject.predict_modal_coefficients).parameters
    history, baseline, target = _synthetic_flows(24, 7)
    controller = subject.fit_modal_controller(
        fit_history=history[:18], fit_prediction=baseline[:18],
        fit_target=target[:18], rank=2, ridge_lambda=1.0,
        device=torch.device("cpu"), projection_batch_size=5)
    poisoned_held_target = np.full_like(target[18:], np.nan)
    coefficients, report = subject.predict_modal_coefficients(
        controller, history=history[18:], prediction=baseline[18:],
        device=torch.device("cpu"), projection_batch_size=3)
    assert np.all(np.isfinite(coefficients))
    assert np.all(np.isnan(poisoned_held_target))
    assert report["samples"] == 6
    assert controller.report["held_target_access_counter"] == {
        "basis": 0, "feature_scaler": 0, "target_scaler": 0,
        "ridge": 0, "clipping": 0,
    }


def test_reconstructed_correction_has_zero_temporal_mean():
    history, _, target = _synthetic_flows(20, 11)
    basis, _ = subject.dual_pod_basis(
        target, rank=3, device=torch.device("cpu"))
    coefficients = subject.project_fluctuation(
        history[:5], basis, torch.device("cpu"), batch_size=2)
    correction = subject.reconstruct_correction(
        coefficients, basis, target.shape[1:], torch.device("cpu"), 2)
    np.testing.assert_allclose(
        correction.mean(axis=1), 0.0, rtol=0.0, atol=2.0e-7)


def test_metric_summary_pools_errors_before_score_mapping():
    rng = np.random.default_rng(19)
    target = rng.normal(size=(3, 20, 32, 64, 2)).astype(np.float32)
    prediction = target + rng.normal(
        scale=0.1, size=target.shape).astype(np.float32)
    errors = subject.official_error_triples(prediction, target)
    metrics = subject.metrics_from_errors(errors)
    means = errors.mean(axis=0)
    assert metrics["rel_l2_score"] == pytest.approx(
        subject.OFFICIAL.score_error(float(means[0])))
    assert metrics["tke_score"] == pytest.approx(
        subject.OFFICIAL.score_error(float(means[1])))
    assert metrics["mvpe_score"] == pytest.approx(
        subject.OFFICIAL.score_error(float(means[2])))
    expected = (
        0.46 * metrics["rel_l2_score"]
        + 0.10 * metrics["tke_score"]
        + 0.09 * metrics["mvpe_score"])
    assert metrics["point_final_contribution_46_10_9"] == pytest.approx(
        expected)


def test_row_summary_keeps_internal_oracle_name_in_public_oracle_section():
    vectors = subject._empty_diagnostic_vectors(3)
    for branch, scale in (
            ("baseline", 1.0), ("history_controller", 0.9),
            ("matched_oracle", 0.5)):
        vectors["official_errors"][branch][:] = scale
        vectors["overall_sse"][branch][:] = scale
        vectors["wake_sse"][branch][:] = scale
        vectors["horizon_sse"][branch][:] = scale
        vectors["wake_horizon_sse"][branch][:] = scale
    vectors["coefficient_target"][:] = 1.0
    vectors["coefficient_prediction"][:] = 0.8
    summary = subject.summarize_rows(
        vectors, np.arange(3), np.asarray([0, 0, 0]))
    assert set(summary) >= {
        "history_controller", "matched_target_visible_oracle"}
    assert len(summary["history_controller"]["horizon_blocks"]) == 4
    assert len(summary["matched_target_visible_oracle"][
        "horizon_blocks"]) == 4


def test_raw_h5_alignment_checks_every_window_and_detects_wrong_order(
        tmp_path: Path):
    manifest = []
    histories = []
    targets = []
    labels = []
    for index, name in enumerate(("a.h5", "b.h5")):
        u = (np.arange(44 * 4 * 6, dtype=np.float32).reshape(44, 4, 6)
             + 10000 * index)
        v = -u
        with h5py.File(tmp_path / name, "w") as handle:
            handle["u"] = u
            handle["v"] = v
            handle["re"] = 100 + index
            handle["aoa"] = 5 * index
        decoded = np.stack((u[:, ::2, ::2], v[:, ::2, ::2]), axis=-1)
        histories.append(decoded[:20])
        targets.append(decoded[20:40])
        labels.append(5 * index)
        manifest.append({
            "file": name, "frames": 44, "re": 100 + index,
            "aoa": 5 * index, "val_starts": [0],
        })
    history = np.stack(histories).astype(np.float32)
    target = np.stack(targets).astype(np.float32)
    aoa = np.asarray(labels)
    report = subject.validate_raw_h5_alignment(
        real_dir=tmp_path, manifest=manifest, history=history,
        target=target, aoa=aoa)
    assert report["passed"] is True
    assert report["windows"] == 2
    assert report["maximum_history_absolute_difference"] == 0.0
    with pytest.raises(ValueError, match="history row 0"):
        subject.validate_raw_h5_alignment(
            real_dir=tmp_path, manifest=manifest,
            history=history[::-1], target=target, aoa=aoa)


def test_cached_fold_rows_are_bound_to_manifest_window_order():
    manifest = [
        {"file": "a.h5", "re": 100, "aoa": 0, "val_starts": [4, 8]},
        {"file": "b.h5", "re": 200, "aoa": 5, "val_starts": [12]},
    ]
    metadata = {"fold_assignments": [
        {"file": "a.h5", "re": 100, "aoa": 0, "fold": 3, "windows": 2},
        {"file": "b.h5", "re": 200, "aoa": 5, "fold": 1, "windows": 1},
    ]}
    report = subject.validate_cached_labels(
        manifest=manifest, metadata=metadata,
        aoa=np.asarray([0, 0, 5]), fold=np.asarray([3, 3, 1]))
    assert report["passed"] is True
    with pytest.raises(ValueError, match="cached fold rows"):
        subject.validate_cached_labels(
            manifest=manifest, metadata=metadata,
            aoa=np.asarray([0, 0, 5]), fold=np.asarray([3, 1, 3]))


def _gate_summary(delta_point: float, delta_rel: float = 0.1,
                  delta_tke: float = 0.1) -> dict[str, object]:
    return {
        "history_controller": {
            "official_point_metrics": {
                "delta_history_controller_minus_baseline": {
                    "rel_l2_score": delta_rel,
                    "tke_score": delta_tke,
                    "mvpe_score": 0.0,
                    "point_final_contribution_46_10_9": delta_point,
                }
            }
        }
    }


def test_preregistered_gate_distinguishes_go_inconclusive_and_stop():
    positive = {str(a): _gate_summary(0.04) for a in subject.EXPECTED_AOAS}
    assert subject.evaluate_gate(
        _gate_summary(0.04), positive)["decision"] == "GO_MODAL_CONTROLLER"
    one_bad = dict(positive)
    one_bad["20"] = _gate_summary(-0.02)
    assert subject.evaluate_gate(
        _gate_summary(0.02), one_bad)["decision"] == (
            "INCONCLUSIVE_MODAL_CONTROLLER")
    assert subject.evaluate_gate(
        _gate_summary(-0.001), positive)["decision"] == (
            "STOP_LINEAR_MODAL_CONTROLLER")
