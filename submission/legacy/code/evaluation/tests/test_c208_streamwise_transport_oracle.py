from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "evaluate_c208_streamwise_transport_oracle.py")
SPEC = importlib.util.spec_from_file_location("c208_transport_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def _flow(samples: int = 2, height: int = 32,
          width: int = 64, seed: int = 11) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(
        samples, 20, height, width, 3, generator=generator)
    value[..., 2] = 0.0
    return value


def _upper(shadow: float = 0.30) -> dict:
    return {
        "pooled": {"point_shadow_delta": shadow},
        "by_aoa": {
            str(aoa): {"point_shadow_delta": 0.06}
            for aoa in subject.AOA_VALUES},
    }


def _pooled(component: float = 0.30, shadow: float = 0.30) -> dict:
    return {
        "delta": {
            "rel_l2_score": component,
            "tke_rel_l2_score": component,
            "mvpe_rel_l2_score": component,
            "point_score": component,
        },
        "point_shadow_delta": shadow,
    }


def _by_aoa(shadow: float = 0.06) -> dict:
    return {
        str(aoa): {"point_shadow_delta": shadow}
        for aoa in subject.AOA_VALUES}


def test_contract_freezes_625_lexicographic_family_boundary_and_budget():
    assert subject.HORIZON_BLOCKS == ((0, 5), (5, 10), (10, 15), (15, 20))
    assert subject.DX_GRID == (-1.0, -0.5, 0.0, 0.5, 1.0)
    assert subject.DX_COMBINATIONS.shape == (625, 4)
    assert np.unique(subject.DX_COMBINATIONS, axis=0).shape == (625, 4)
    np.testing.assert_array_equal(subject.DX_COMBINATIONS[0], [-1] * 4)
    np.testing.assert_array_equal(subject.DX_COMBINATIONS[624], [1] * 4)
    np.testing.assert_array_equal(subject.DX_COMBINATIONS[312], [0] * 4)
    np.testing.assert_array_equal(
        subject.DX_COMBINATIONS[44], [-1, -0.5, 0.5, 1])
    np.testing.assert_array_equal(
        subject.DX_COMBINATIONS[580], [1, 0.5, -0.5, -1])
    assert subject.IDENTITY_COMBINATION_INDEX == 312
    assert subject.SAMPLE_COORDINATE_ROUNDS == 2
    assert subject.WALL_BUDGET_SECONDS == 1200.0
    assert subject.GPU_BUDGET_SECONDS == 720.0
    assert subject.EXPERIMENT_CONTRACT["data_boundary"]["train_windows"] == 2576
    assert subject.EXPERIMENT_CONTRACT["warp"]["align_corners"] is False


def test_zero_rows_bypass_grid_sample_bitwise_and_fp32_is_required(monkeypatch):
    prediction = _flow(samples=2, height=5, width=7)
    history = torch.ones_like(prediction)
    history[..., 2] = 0.0
    solid = subject.observation_solid_mask(history)
    before = prediction.clone()
    calls = []
    original = subject.F.grid_sample

    def recording_grid_sample(value, *args, **kwargs):
        calls.append(value.shape[0])
        return original(value, *args, **kwargs)

    monkeypatch.setattr(subject.F, "grid_sample", recording_grid_sample)
    shifts = torch.zeros(2, 4)
    actual = subject.streamwise_warp(prediction, shifts, solid)
    assert calls == []
    assert torch.equal(actual, prediction)

    shifts[1, 0] = 0.5
    mixed = subject.streamwise_warp(prediction, shifts, solid)
    assert calls == [5]  # only one non-zero row times five horizons
    assert torch.equal(mixed[0], prediction[0])
    assert torch.equal(prediction, before)
    with pytest.raises(TypeError, match="FP32"):
        subject.streamwise_warp(prediction.half(), shifts, solid)


def test_per_sample_masks_remain_distinct_and_bind_to_ordered_windows():
    history = torch.ones(3, 20, 4, 7, 3)
    history[..., 2] = 0.0
    history[0, :, 1, 2, :2] = 0.0
    history[1, :, 2, 3, :2] = 0.0
    history[2, :, 1, 2, :2] = 0.0
    windows = [
        SimpleNamespace(trajectory=0, start=0),
        SimpleNamespace(trajectory=0, start=20),
        SimpleNamespace(trajectory=1, start=0),
    ]

    masks = subject.observation_solid_mask(history)
    report = subject.per_sample_mask_diagnostics(masks, windows)

    assert torch.equal(masks[0], masks[2])
    assert not torch.equal(masks[0], masks[1])
    assert report["semantics"] == "per_sample_observation_zero_support"
    assert report["target_used_for_mask"] is False
    assert report["unique_masks"] == 2
    assert report["trajectories_with_varying_masks"] == 1
    assert report["samples_different_from_trajectory_first"] == 1
    assert report["trajectory_aggregation_used_for_warp"] is False
    digest = subject.ordered_window_mask_digest("0" * 64, masks)
    assert len(digest) == 64
    assert digest != subject.ordered_window_mask_digest("1" * 64, masks)
    assert digest != subject.ordered_window_mask_digest(
        "0" * 64, masks[[1, 0, 2]].contiguous())


def test_mixed_batch_uses_each_samples_own_source_footprint_mask():
    prediction = torch.zeros(2, 20, 3, 8, 3, dtype=torch.float32)
    prediction[..., 0] = torch.arange(8, dtype=torch.float32)
    solid = torch.zeros(2, 3, 8, dtype=torch.bool)
    solid[0, 1, 2] = True
    solid[1, 1, 5] = True
    shifted = subject._streamwise_warp_steps(
        prediction[:, :5], torch.tensor([0.5, -0.5]), solid)

    # Sample 0: +0.5 at x=3 touches its own x=2 solid pixel.
    assert torch.equal(
        shifted[0, :, 1, 3, :2], prediction[0, :5, 1, 3, :2])
    # Sample 1: -0.5 at x=4 touches its own x=5 solid pixel.
    assert torch.equal(
        shifted[1, :, 1, 4, :2], prediction[1, :5, 1, 4, :2])
    # The other sample's mask must not trigger fallback here.
    assert not torch.equal(
        shifted[0, :, 1, 4, :2], prediction[0, :5, 1, 4, :2])
    assert not torch.equal(
        shifted[1, :, 1, 3, :2], prediction[1, :5, 1, 3, :2])


def test_positive_moves_right_and_solid_source_footprint_falls_back_no_wrap():
    prediction = torch.zeros(1, 20, 3, 8, 3, dtype=torch.float32)
    prediction[..., 0] = torch.arange(8, dtype=torch.float32)
    empty_solid = torch.zeros(1, 3, 8, dtype=torch.bool)
    shifted = subject._streamwise_warp_steps(
        prediction[:, :5], torch.tensor([1.0]), empty_solid)
    assert shifted[0, 0, 0, 3, 0] == prediction[0, 0, 0, 2, 0]
    assert shifted[0, 0, 0, 0, 0] == prediction[0, 0, 0, 0, 0]
    assert shifted[0, 0, 0, 0, 0] != prediction[0, 0, 0, 7, 0]

    solid = empty_solid.clone()
    solid[:, 1, 2] = True
    positive_half = subject._streamwise_warp_steps(
        prediction[:, :5], torch.tensor([0.5]), solid)
    negative_half = subject._streamwise_warp_steps(
        prediction[:, :5], torch.tensor([-0.5]), solid)
    # +0.5 at x=3 samples x=2/3; -0.5 at x=1 samples x=1/2.
    assert torch.equal(
        positive_half[:, :, 1, 3, :2], prediction[:, :5, 1, 3, :2])
    assert torch.equal(
        negative_half[:, :, 1, 1, :2], prediction[:, :5, 1, 1, :2])
    assert torch.equal(positive_half[..., 2], torch.zeros_like(
        positive_half[..., 2]))


def test_all_625_sufficient_stat_errors_match_direct_official_scorer():
    baseline = _flow(samples=1)
    target = _flow(samples=1, seed=17)
    solid = torch.zeros(1, 32, 64, dtype=torch.bool)
    solid[:, 16, 20] = True
    statistics = subject.block_sufficient_statistics(baseline, target, solid)
    triples = subject.exhaustive_error_triples_from_statistics(
        statistics, target)[0].numpy()
    maximum = 0.0
    for start in range(0, subject.COMBINATION_COUNT, 25):
        stop = min(start + 25, subject.COMBINATION_COUNT)
        count = stop - start
        prediction = baseline.expand(count, -1, -1, -1, -1).clone()
        truth = target.expand(count, -1, -1, -1, -1)
        masks = solid.expand(count, -1, -1)
        shifts = torch.from_numpy(
            subject.DX_COMBINATIONS[start:stop].copy())
        candidate = subject.streamwise_warp(prediction, shifts, masks)
        direct = subject.direct_official_error_triple(candidate, truth)
        maximum = max(maximum, float(np.max(
            np.abs(direct - triples[start:stop]))))
    assert maximum <= subject.SANITY_ERROR_TOLERANCE


def test_generous_upper_combines_independent_unachievable_metric_minima():
    errors = np.full((5, 625, 3), 3.0, dtype=np.float32)
    errors[:, subject.IDENTITY_COMBINATION_INDEX] = 2.0
    errors[:, 0, 0] = 0.1
    errors[:, 1, 1] = 0.2
    errors[:, 2, 2] = 0.3
    aoas = np.asarray(subject.AOA_VALUES)

    upper = subject.generous_upper_bound(errors, aoas)

    expected = np.asarray([[0.1, 0.2, 0.3]] * 5)
    expected_comparison = subject.error_comparison(
        errors[:, subject.IDENTITY_COMBINATION_INDEX], expected)
    assert upper["simultaneously_achievable"] is False
    assert upper["pooled"]["point_shadow_delta"] == pytest.approx(
        expected_comparison["point_shadow_delta"])


def test_achievable_search_is_global_two_round_exact_pooled_and_l1_tied():
    errors = np.full((3, 625, 3), 4.0, dtype=np.float32)
    errors[:, subject.IDENTITY_COMBINATION_INDEX] = 2.0
    # Identical best triples: index 313 has L1=.5 and beats index 0 L1=4.
    errors[:, 0] = 1.0
    errors[:, 313] = 1.0

    selection, report = subject.achievable_sample_coordinate_search(errors)

    np.testing.assert_array_equal(selection, np.full(3, 313))
    assert report["rounds"] == 2
    assert report["sample_order"] == "0..N-1 in both rounds"
    assert report["accepted_updates_by_round"] == [3, 0]
    selected = errors[np.arange(3), selection]
    expected = subject.pooled_shadow_from_error_sums(
        np.sum(selected, axis=0, dtype=np.float64), 3)
    assert report["final_exact_pooled_shadow"] == pytest.approx(expected)
    assert report["objective_gain"] > 0.0


def test_gate_uses_three_decisions_and_saturation_over_nonzero_choices():
    shifts = np.zeros((100, 4), dtype=np.float32)
    shifts[:10, 0] = 0.5
    go = subject.gate_result(
        _upper(), _pooled(), _by_aoa(), shifts)
    assert go["decision"] == "GO_WARP_ORACLE_WORTH_LOAO"
    assert go["nonzero_block_choices"] == 10
    assert go["saturated_over_nonzero_block_choice_ratio"] == 0.0

    saturated = shifts.copy()
    saturated[:5, 0] = 1.0
    boundary = subject.gate_result(
        _upper(), _pooled(), _by_aoa(), saturated)
    assert boundary["saturated_block_choices"] == 5
    assert boundary["saturated_over_nonzero_block_choice_ratio"] == 0.5
    assert boundary["pass"] is True

    weak_upper = _upper(shadow=0.1)
    stop = subject.gate_result(
        weak_upper, _pooled(component=-0.1, shadow=0.1),
        _by_aoa(shadow=0.01), shifts)
    assert stop["decision"] == "STOP_WARP_NO_HEADROOM"

    inconclusive = subject.gate_result(
        _upper(), _pooled(component=-1e-9, shadow=0.30),
        _by_aoa(), shifts)
    assert inconclusive["decision"] == "INCONCLUSIVE_WARP_HEADROOM"


def test_tail_counters_report_actual_nonzero_state_for_failure_payloads():
    counters = subject.TailAccessCounters()
    assert counters.report()["tail_accessed"] is False
    counters.target_read_attempts = 1
    report = counters.report()
    assert report["tail_accessed"] is True
    assert report["tail_target_read_attempts"] == 1
    with pytest.raises(ValueError, match="cannot be negative"):
        counters.target_reads_completed = -1
        counters.report()


def test_contract_only_writes_result_without_reading_inputs(tmp_path, monkeypatch):
    output = tmp_path / "contract"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT),
        "--point-checkpoint", str(tmp_path / "missing.pth"),
        "--split-manifest", str(tmp_path / "missing.json"),
        "--output-dir", str(output),
        "--contract-only",
    ])

    subject.main()

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["decision"] == "CONTRACT_ONLY_NO_DATA_NO_GPU"
    assert result["tail_accessed"] is False
    assert result["contract"]["oracle"]["combinations_per_sample"] == 625
