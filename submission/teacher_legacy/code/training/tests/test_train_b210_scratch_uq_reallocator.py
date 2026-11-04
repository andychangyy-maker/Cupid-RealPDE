from __future__ import annotations

import sys
from pathlib import Path

import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_b210_scratch_uq_reallocator as subject  # noqa: E402


def _metrics(score: float, elements: int = 100) -> dict[str, float]:
    return {
        "sps_score": score,
        "coverage": 0.90,
        "mean_width": 0.016,
        "scored_elements": elements,
    }


def test_native_auxiliary_has_expected_local_channels() -> None:
    members = torch.zeros(2, 4, 20, 3, 5, 3)
    members[:, 1, ..., 0] = 0.1
    members[:, 2, ..., 1] = -0.2
    lower = torch.full((2, 20, 3, 5, 2), 0.01)
    upper = torch.full((2, 20, 3, 5, 2), 0.02)
    auxiliary = subject.native_auxiliary(
        members, lower, upper, torch.tensor([0.2, 0.1, 1.0]))
    assert auxiliary.shape == (2, 160, 3, 5)
    assert torch.isfinite(auxiliary).all()
    assert (auxiliary >= 0).all()
    assert torch.count_nonzero(auxiliary) > 0


def test_native_auxiliary_rejects_misaligned_extents() -> None:
    members = torch.zeros(1, 4, 20, 3, 5, 3)
    lower = torch.zeros(1, 19, 3, 5, 2)
    upper = torch.zeros_like(lower)
    try:
        subject.native_auxiliary(
            members, lower, upper, torch.ones(3))
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("misaligned extents were accepted")


def test_choose_arm_requires_fold_and_aoa_stability() -> None:
    fold_rows = []
    for fold in range(5):
        fold_rows.append({
            "fold": fold,
            "baseline": _metrics(53.0),
            "dynamic": _metrics(53.22),
            "native_aware": _metrics(53.31),
        })
    by_aoa = {
        str(aoa): {
            "baseline": _metrics(53.0),
            "dynamic": _metrics(53.20),
            "native_aware": _metrics(53.28),
        }
        for aoa in (0, 5, 10, 15, 20)
    }
    summary = {
        "folds": fold_rows,
        "oof": {
            "baseline": _metrics(53.0),
            "dynamic": _metrics(53.22),
            "native_aware": _metrics(53.31),
        },
        "by_aoa": by_aoa,
    }
    selected, gates = subject.choose_arm(summary)
    assert selected == "native_aware"
    assert gates["dynamic"]["pass"]
    assert gates["native_aware"]["pass"]

    summary["folds"][0]["native_aware"] = _metrics(52.90)
    selected, gates = subject.choose_arm(summary)
    assert selected == "dynamic"
    assert not gates["native_aware"]["pass"]


def test_sample_attributes_preserve_manifest_window_order() -> None:
    manifest = [
        {"file": "a.h5", "re": 100, "aoa": 0, "val_starts": [1, 2]},
        {"file": "b.h5", "re": 200, "aoa": 5, "val_starts": [3]},
    ]
    folds, aoas = subject.sample_attributes(manifest)
    assert len(folds) == 3
    assert aoas == [0, 0, 5]
