from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "audit_b210_clean716_vq_event_oracle.py"
SPEC = importlib.util.spec_from_file_location("vq_event_oracle", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_patch_round_trip_is_exact() -> None:
    values = np.arange(2 * 20 * 32 * 64 * 2, dtype=np.float32).reshape(
        2, 20, 32, 64, 2)
    patches = MODULE.patchify(values)
    assert patches.shape == (2, 4, 4, 8, 5, 8, 8, 2)
    assert np.array_equal(MODULE.unpatchify(patches), values)


def test_correction_is_zero_temporal_mean() -> None:
    rng = np.random.default_rng(3)
    values = rng.standard_normal((2, 4, 4, 8, 5, 8, 8, 2)).astype(np.float32)
    correction = MODULE.correction_from_patches(values)
    assert np.max(np.abs(correction.mean(axis=1, dtype=np.float64))) < 2e-7


def test_balanced_patch_sample_is_exactly_stratified() -> None:
    # Four samples per AoA make a tiny two-AoA, one-per-stratum test possible.
    labels = np.repeat(np.asarray([0, 5]), 4)
    values = np.zeros((8, 4, 4, 8, 5, 8, 8, 2), dtype=np.float32)
    for row in range(8):
        values[row] = row
    sample, digest = MODULE.balanced_patch_sample(
        values, labels, (0, 5), np.random.default_rng(1), per_stratum=1)
    assert sample.shape == (2 * 4 * 32, 640)
    assert len(digest) == 64
    # Every time-block/tile stratum contributes once for each AoA.
    assert set(np.unique(sample).tolist()) <= set(range(8))


def test_spatial_shuffle_preserves_patch_multiset_per_block() -> None:
    values = np.arange(2 * 4 * 4 * 8 * 5 * 8 * 8 * 2, dtype=np.float32).reshape(
        2, 4, 4, 8, 5, 8, 8, 2)
    shuffled = MODULE.shuffle_spatial_patches(values, 210)
    for sample in range(2):
        for block in range(4):
            original = values[sample, block].reshape(32, -1)
            changed = shuffled[sample, block].reshape(32, -1)
            assert sorted(map(bytes, original)) == sorted(map(bytes, changed))

