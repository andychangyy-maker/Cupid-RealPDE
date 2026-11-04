#!/usr/bin/env python3
"""XCB lineage adapter for the full-trajectory same-AoA residual cache."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import build_b210_same_aoa_point_oof_residual_cache as base
import train_b210_xcb_same_aoa_point_oof_fold as point_fold


SCHEMA = "b210_xcb_same_aoa_trajectory_point_oof_residual_cache_v1"
COMPLETION_EVENT = "b210_xcb_same_aoa_point_oof_residual_cache_complete"
EXPECTED_SAMPLES = base.EXPECTED_SAMPLES
fold_plan = base.fold_plan
dr1 = base.dr1
ft12 = base.ft12
read_trajectory = base.read_trajectory
uvp_windows = base.uvp_windows
predict_fold = base.predict_fold
full_evaluation_rows = base.full_evaluation_rows


@contextmanager
def configured_base() -> Iterator[None]:
    replacements = {
        "point_fold": point_fold,
        "SCHEMA": SCHEMA,
        "COMPLETION_EVENT": COMPLETION_EVENT,
    }
    original = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def load_cache(path: Path, expected_sha256: str) -> tuple[dict, dict]:
    with configured_base():
        return base.load_cache(path, expected_sha256)


def main() -> None:
    with configured_base():
        base.main()


if __name__ == "__main__":
    main()
