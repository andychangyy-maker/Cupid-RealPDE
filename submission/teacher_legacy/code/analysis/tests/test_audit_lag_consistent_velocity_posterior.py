from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "audit_lag_consistent_velocity_posterior.py"
SPEC = importlib.util.spec_from_file_location("lag_velocity_posterior", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def test_synthetic_integer_translation_locks_sign_and_two_to_one_scale():
    report = subject.synthetic_integer_translation_check(torch.device("cpu"))
    assert report["pass"] is True
    assert report["feature_x_minus_one_maps_to_field_x_minus_two"] is True
    assert report["native_max_abs_error"] <= subject.WARP_SANITY_TOLERANCE
    assert report["opposite_mse"] > 1.0e-3
    assert report["matcher_expected_backtrace_offset"] == [-1, 0]
    assert report["matcher_correct_interior_fraction"] >= .999


def test_feature_contract_and_posterior_use_only_legal_history_indices():
    history = torch.randn(2, 20, 32, 64, 2)
    feature = subject.history_features(history)
    assert feature.shape == (2, 20, 6, 16, 32)
    mask = subject.history_feature_mask(history)
    posterior = subject.posterior_from_features(feature, 18, feature_mask=mask)
    assert posterior["probability"].shape == (2, 4, 81, 16, 32)
    assert posterior["velocity_per_lag"].shape == (2, 4, 2, 16, 32)
    assert posterior["velocity_effective"].shape == (2, 2, 16, 32)
    assert torch.isfinite(posterior["trace_covariance"]).all()
    assert float(posterior["velocity_effective"].abs().max()) <= (
        subject.ONE_STEP_CAP_FEATURE_PIXELS + 1.e-7)
    with pytest.raises(ValueError, match="legal internal"):
        subject.posterior_from_features(feature, 19)


def test_fit_windows_excludes_aoa10_and_uses_re_ordinal_mod4():
    manifest = []
    for aoa in (0, 5, 10, 15, 20):
        for index in range(4):
            manifest.append({"file": f"{aoa}_{index}.h5", "frames": 100,
                             "re": 10000 + index, "aoa": aoa,
                             "val_starts": [20, 40]})
    rows = subject.fit_windows(manifest)
    assert {row.aoa for row in rows} == set(subject.FIT_AOAS)
    assert all(row.aoa != 10 for row in rows)
    assert {row.fold for row in rows if row.aoa == 0} == {0, 1, 2, 3}
    assert max(row.start + subject.HISTORY_STEPS for row in rows) <= 60


def test_donor_preserves_same_aoa_and_re_ordinal_fold():
    rows = []
    for file_index in range(3):
        for ordinal, start in enumerate((0, 20)):
            rows.append(subject.Window(f"f{file_index}.h5", start, 0, 10000 + file_index,
                                       file_index, 2, ordinal))
    donor, report = subject.donor_indices(rows)
    assert report["same_aoa_and_ordinal_mod4_fold"] is True
    for index, donor_index in enumerate(donor):
        assert rows[index].aoa == rows[int(donor_index)].aoa
        assert rows[index].fold == rows[int(donor_index)].fold
        assert rows[index].file != rows[int(donor_index)].file


def _score_group(reduction=.021, spatial_gain=.006):
    leaf = {"native_vs_identity": {"reduction_fraction": reduction},
            "native_minus_spatial_null_fraction_of_identity": spatial_gain}
    return {"overall": dict(leaf),
            "by_fold": {str(index): dict(leaf) for index in range(4)},
            "by_aoa": {str(aoa): dict(leaf) for aoa in subject.FIT_AOAS},
            "trajectory_summary": {"median_reduction_fraction": reduction,
                                   "positive_fraction": 1.0}}


def test_gate_uses_fixed_multistep_cluster_contract():
    common = _score_group()
    report = {
        "primary_multistep": {
            "mask_variants": {"common_valid": common,
                              "all_pixels": _score_group(),
                              "eroded_common_valid": _score_group()},
            "by_horizon": {str(horizon): {"mask_variants": {
                "common_valid": _score_group()}}
                for horizon in subject.PRIMARY_HORIZONS}},
        "repeat_one_step": {str(terminal): {
            "common_valid": {"reduction_fraction": .01}}
            for terminal in subject.REPEAT_ONE_STEP_TERMINALS},
        "lag_consistency": {"by_fold": {str(index): {
            "native_reduction_vs_time_reversed": .11,
            "native_reduction_vs_lag_donor": .11} for index in range(4)}}}
    assert subject.gate(report)[0] is True
    report["primary_multistep"]["by_horizon"]["2"]["mask_variants"][
        "common_valid"]["overall"]["native_vs_identity"]["reduction_fraction"] = -.01
    assert subject.gate(report)[0] is False


def test_crossfit_statistics_exclude_held_fold():
    histories = np.stack([
        np.full((20, 32, 64, 2), float(index + 1), dtype=np.float32)
        for index in range(4)], axis=0)
    windows = [subject.Window(f"f{index}.h5", 0, 0, 10000 + index,
                              index, index, 0) for index in range(4)]
    stats = subject.crossfit_feature_statistics(
        histories, windows, torch.device("cpu"), chunk_size=2)
    assert torch.allclose(stats[0]["mean"].reshape(-1)[:2], torch.tensor([3., 3.]))
    assert torch.allclose(stats[3]["mean"].reshape(-1)[:2], torch.tensor([2., 2.]))


def test_masked_warp_never_fills_zero_centre():
    source = torch.ones(1, 2, 32, 64)
    source[:, :, 10, 10] = 0.
    velocity = torch.zeros_like(source)
    velocity[:, 0] = .4
    warped = subject.warp_field(source, velocity)
    assert torch.equal(warped[:, :, 10, 10], torch.zeros(1, 2))


def test_feature_mask_covers_forward_difference_stencil():
    history = torch.ones(1, 20, 32, 64, 2)
    history[:, :, 2, 2] = 0.
    mask = subject.history_feature_mask(history)
    # Raw (2,2) belongs to the 3x3 dependency footprints of feature cells
    # starting at raw (0,0) and (2,2).
    assert mask[0, 0, 0, 0, 0].item() is False
    assert mask[0, 0, 0, 1, 1].item() is False
