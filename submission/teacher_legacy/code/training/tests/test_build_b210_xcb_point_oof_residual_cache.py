from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / (
    "build_b210_xcb_point_oof_residual_cache.py")
SPEC = importlib.util.spec_from_file_location("point_oof_cache", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def rows():
    return [
        {
            "file": f"{re}_{aoa}.h5", "frames": 80,
            "re": re, "aoa": aoa, "val_starts": [40],
        }
        for re in (10, 20, 30, 40, 50)
        for aoa in (0, 5, 10, 15, 20)
    ]


def test_manifest_and_assignments_match_point_oof_contract():
    manifest = subject.validate_manifest(rows())
    assignments = subject.condition_fold_assignments(manifest, 5)
    assert set(assignments.values()) == set(range(5))
    assert assignments["10_0.h5"] == 0
    assert assignments["20_0.h5"] == 1
    assert assignments["10_5.h5"] == 2


def test_manifest_rejects_duplicate_window():
    manifest = rows()
    manifest[0]["val_starts"] = [40, 40]
    with pytest.raises(ValueError, match="duplicate"):
        subject.validate_manifest(manifest)


def test_manifest_rejects_path_escape():
    manifest = rows()
    manifest[0]["file"] = "../escape.h5"
    with pytest.raises(ValueError, match="unsafe"):
        subject.validate_manifest(manifest)


def test_uvp_windows_have_official_shapes():
    import torch

    trajectory = torch.randn(80, 32, 64, 2)
    history, target = subject.uvp_windows(trajectory, [0, 40])
    assert history.shape == (2, 20, 32, 64, 3)
    assert target.shape == (2, 20, 32, 64, 3)
    assert torch.count_nonzero(history[..., 2]) == 0
