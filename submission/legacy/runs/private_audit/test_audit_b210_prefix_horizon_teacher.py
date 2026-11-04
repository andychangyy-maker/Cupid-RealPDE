from __future__ import annotations

import sys
from pathlib import Path

import torch


PRIVATE = Path(__file__).resolve().parent
if str(PRIVATE) not in sys.path:
    sys.path.insert(0, str(PRIVATE))

import audit_b210_prefix_horizon_teacher as audit  # noqa: E402


def test_build_windows_requires_45_frames_and_tracks_aoa():
    trajectories = [torch.zeros(50, 2, 3, 2), torch.zeros(44, 2, 3, 2)]
    metadata = [{"aoa": 10}, {"aoa": 20}]
    windows = audit.build_windows(trajectories, metadata)
    assert [(row.trajectory, row.start, row.aoa) for row in windows] == [
        (0, 0, 10), (0, 5, 10)]


def test_error_accumulator_detects_teacher_direction():
    truth = torch.ones(2, 15, 1, 1, 2)
    long = torch.zeros_like(truth)
    short = torch.full_like(truth, 0.75)
    accumulator = audit.ErrorAccumulator()
    accumulator.update(long, short, truth)
    report = accumulator.report()
    assert report["short_relative_mse_reduction"] == 0.9375
    assert report["short_better_sample_fraction"] == 1.0
    assert report["correction_positive_sample_fraction"] == 1.0
    assert abs(report["correction_oracle_global_cosine"] - 1.0) < 1e-12
    assert abs(report["least_squares_correction_alpha"] - 4 / 3) < 1e-12


def test_uvp_appends_exact_zero_pressure():
    values = torch.randn(2, 20, 4, 5, 2)
    result = audit.uvp(values)
    assert result.shape == (2, 20, 4, 5, 3)
    assert torch.count_nonzero(result[..., 2]) == 0
