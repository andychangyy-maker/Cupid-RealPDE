from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
SCRIPT = TRAINING_ROOT / "train_ensemble_unet2d.py"
SPEC = importlib.util.spec_from_file_location("ensemble_unet_rs16", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def synthetic_baseline_manifest() -> tuple[list[dict], list[dict]]:
    """Build a valid stride-20 split with the formal aggregate counts."""
    # (baseline train windows, validation windows, trajectory count)
    groups = ((28, 8, 10), (31, 8, 3), (31, 9, 5), (32, 9, 64))
    metadata = []
    manifest = []
    trajectory = 0
    for train_count, val_count, count in groups:
        for _ in range(count):
            val_boundary = 20 * (train_count + 1)
            val_starts = [
                val_boundary + 20 * offset for offset in range(val_count)]
            item = {
                "file": f"trajectory_{trajectory:02d}.h5",
                "frames": val_starts[-1] + 40,
                "re": 6000 + 100 * trajectory,
                "aoa": 5 * (trajectory % 5),
            }
            train_starts = list(range(0, val_boundary - 40 + 1, 20))
            metadata.append(item)
            manifest.append({
                **item,
                "train_starts": train_starts,
                "val_starts": val_starts,
                "val_boundary": val_boundary,
            })
            trajectory += 1
    return metadata, manifest


def test_stride20_exactly_replays_frozen_baseline_split():
    metadata, frozen = synthetic_baseline_manifest()

    train, val, rebuilt = subject.rebuild_real_split_from_frozen_manifest(
        metadata, frozen, 20)

    assert len(train) == 2576
    assert len(val) == 725
    assert rebuilt == frozen


def test_stride16_rebuild_has_expected_train_count_and_same_fixed725():
    metadata, frozen = synthetic_baseline_manifest()

    train, val, rebuilt = subject.rebuild_real_split_from_frozen_manifest(
        metadata, frozen, 16)

    assert len(train) == 3140
    assert len(val) == 725
    assert [entry["val_starts"] for entry in rebuilt] == [
        entry["val_starts"] for entry in frozen]
    assert [entry["val_boundary"] for entry in rebuilt] == [
        entry["val_boundary"] for entry in frozen]
    assert not ({(window.trajectory, window.start) for window in train}
                & {(window.trajectory, window.start) for window in val})


def test_rebuild_rejects_optimizer_interval_overlap_with_frozen_validation():
    metadata = [{"file": "one.h5", "frames": 160, "re": 6000, "aoa": 0}]
    frozen = [{
        **metadata[0],
        "train_starts": [0, 20, 40],
        "val_starts": [80, 100],
        "val_boundary": 100,
    }]

    with pytest.raises(ValueError, match="interval overlaps"):
        subject.rebuild_real_split_from_frozen_manifest(
            metadata, frozen, 16)


def test_rebuild_rejects_metadata_mismatch_and_exact_key_intersection():
    metadata = [{"file": "one.h5", "frames": 160, "re": 6000, "aoa": 0}]
    mismatched = [{
        **metadata[0], "re": 7000, "train_starts": [0],
        "val_starts": [100], "val_boundary": 100,
    }]
    with pytest.raises(ValueError, match="metadata mismatch"):
        subject.rebuild_real_split_from_frozen_manifest(
            metadata, mismatched, 20)

    intersecting = [{
        **metadata[0], "train_starts": [0],
        "val_starts": [48], "val_boundary": 100,
    }]
    with pytest.raises(ValueError, match="key intersection"):
        subject.rebuild_real_split_from_frozen_manifest(
            metadata, intersecting, 16)
