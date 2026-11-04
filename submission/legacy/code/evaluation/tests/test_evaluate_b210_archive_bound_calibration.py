from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

import evaluate_b210_archive_bound_calibration as subject  # noqa: E402


def test_condition_labels_use_frozen_condition_formula():
    manifest = [
        {"file": "a.h5", "re": 100, "aoa": 0, "val_starts": [0, 20]},
        {"file": "b.h5", "re": 200, "aoa": 5, "val_starts": [0]},
    ]

    labels = subject.condition_labels(manifest)

    assert labels["fold"].tolist() == [0, 0, 3]
    assert labels["aoa"].tolist() == [0, 0, 5]
    assert labels["trajectory"].tolist() == [0, 0, 1]


def test_coordinate_search_never_lowers_sample_objective():
    sample = {
        "point": np.zeros(1000, dtype=np.float32),
        "target": np.full(1000, 0.01, dtype=np.float32),
        "lower": np.full(1000, -0.03, dtype=np.float32),
        "upper": np.full(1000, 0.03, dtype=np.float32),
        "weight": np.ones(1000, dtype=np.float64),
    }

    result = subject.coordinate_search(sample)

    assert result["selected_objective"] >= result["identity_objective"]
    assert result["sample_points"] == 1000
    assert set(result["parameters"]) == set(subject.PARAMETER_NAMES)


def test_apply_groups_changes_only_bounds():
    shape = (2, 20, 1, 1, 2)
    arrays = {
        "prediction": np.zeros(shape, dtype=np.float32),
        "target": np.ones(shape, dtype=np.float32),
        "lower": np.full(shape, -1.0, dtype=np.float32),
        "upper": np.full(shape, 1.0, dtype=np.float32),
    }
    groups = []
    for block, (start, stop) in enumerate(subject.HORIZON_BLOCKS):
        for channel in subject.MEASURED_CHANNELS:
            groups.append({
                "block": block,
                "horizon_start": start,
                "horizon_stop": stop,
                "channel": channel,
                "parameters": {
                    "lower_scale": 0.9,
                    "lower_offset": 0.0,
                    "upper_scale": 0.8,
                    "upper_offset": 0.0,
                },
            })
    point_before = arrays["prediction"].copy()
    output = subject.allocate_bounds(arrays)

    report = subject.apply_groups(
        arrays, groups, np.ones(2, dtype=bool), output)

    assert np.array_equal(arrays["prediction"], point_before)
    assert np.all(output["lower"] == -0.9)
    assert np.all(output["upper"] == 0.8)
    assert report["reversed_before_projection"] == 0
