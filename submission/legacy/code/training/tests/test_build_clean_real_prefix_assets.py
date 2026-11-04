from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "build_clean_real_prefix_assets.py"
SPEC = importlib.util.spec_from_file_location("clean_prefix_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def write_trajectory(path: Path, frames: int = 100) -> None:
    u = np.full((frames, 4, 6), 2.0, dtype=np.float32)
    v = np.full((frames, 4, 6), -3.0, dtype=np.float32)
    u[60:] = 1000.0
    v[60:] = -1000.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=u)
        handle.create_dataset("v", data=v)
        handle.create_dataset("re", data=6300)
        handle.create_dataset("aoa", data=10)


def test_split_row_freezes_training_prefix_without_tail_starts(tmp_path):
    path = tmp_path / "6300_10.h5"
    write_trajectory(path)

    row = subject.split_row(path, stride=20, val_fraction=0.2)

    assert row == {
        "file": "6300_10.h5",
        "frames": 100,
        "re": 6300,
        "aoa": 10,
        "train_starts": [0, 20],
        "val_boundary": 60,
    }
    assert "val_starts" not in row


def test_prefix_moments_do_not_include_tail_sentinel(tmp_path):
    path = tmp_path / "6300_10.h5"
    write_trajectory(path)

    sums, squares, count = subject.prefix_channel_moments(path, boundary=60)

    assert count == 60 * 2 * 3
    assert np.allclose(sums / count, [2.0, -3.0])
    assert np.allclose(squares / count, [4.0, 9.0])
