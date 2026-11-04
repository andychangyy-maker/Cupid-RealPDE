#!/usr/bin/env python3
"""XCB lineage adapter for full-trajectory same-AoA point cross-fit.

Only lineage identity, architecture validation and checkpoint schemas differ
from ``train_b210_same_aoa_point_oof_fold``.  Fold assignment, fit-only H5
access, prefix-only normalization, full stride-5 Real4k and fixed EMA-u4000
selection remain implemented by the shared runner.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

import train_b210_same_aoa_point_oof_fold as base
import train_b210_xcb_full_released_real_final_point as xcb_final


EXPERIMENT = "b210_xcb_same_aoa_trajectory_point_oof"
SCHEMA = "b210_xcb_same_aoa_trajectory_point_oof_fold_v1"
ENDPOINT_SCHEMA = "b210_xcb_same_aoa_trajectory_point_oof_endpoint_v1"
ACKNOWLEDGEMENT = "RUN_B210_XCB_SAME_AOA_TRAJECTORY_POINT_OOF_V1"
COMPLETION_EVENT = "b210_xcb_same_aoa_point_oof_fold_complete"
START_SHA256 = xcb_final.FORMAL_START_SHA256
START_EXPECTED_ITERATION = xcb_final.START_EXPECTED_ITERATION
START_EXPECTED_STATUS = xcb_final.START_EXPECTED_STATUS
MODEL_LINEAGE = xcb_final.MODEL_LINEAGE
LINEAGE_FIELD_REQUIRED = True
UQ_SOURCE_LABEL = "xcb_sim24k"
FINAL_POINT_RESULT_SCHEMA = xcb_final.SCHEMA
FINAL_POINT_ENDPOINT_SCHEMA = xcb_final.ENDPOINT_SCHEMA
FORMAL_FINAL_POINT_RESULT_SHA256 = (
    "f3287c057aff9e37d18db7d5827fdc8568ab75fe304724ec8066ffa0ed239264")
FORMAL_FINAL_POINT_RESULT_PATH = Path(
    "/root/autodl-tmp/b210_xcb_full_released_real_final_point_20260905_v3/"
    "point_stage_result.json")

# Downstream cache/UQ adapters intentionally consume these shared interfaces.
fold_plan = base.fold_plan
whole_aoa = base.whole_aoa
support = base.support
dr1 = base.dr1
ft12 = base.ft12
smu1 = base.smu1
small4k = base.small4k
sim_tke = base.sim_tke
FOLDS = base.FOLDS
ENDPOINT = base.ENDPOINT
UPDATES = base.UPDATES


def architecture_contract() -> tuple:
    return xcb_final.architecture_contract()


@contextmanager
def configured_base() -> Iterator[None]:
    """Bind XCB identities into the shared runner and restore them exactly."""
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "SCHEMA": SCHEMA,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "ACKNOWLEDGEMENT": ACKNOWLEDGEMENT,
        "COMPLETION_EVENT": COMPLETION_EVENT,
        "START_SHA256": START_SHA256,
        "START_EXPECTED_ITERATION": START_EXPECTED_ITERATION,
        "START_EXPECTED_STATUS": START_EXPECTED_STATUS,
        "MODEL_LINEAGE": MODEL_LINEAGE,
        "LINEAGE_FIELD_REQUIRED": LINEAGE_FIELD_REQUIRED,
        "UQ_SOURCE_LABEL": UQ_SOURCE_LABEL,
        "FINAL_POINT_RESULT_SCHEMA": FINAL_POINT_RESULT_SCHEMA,
        "FINAL_POINT_ENDPOINT_SCHEMA": FINAL_POINT_ENDPOINT_SCHEMA,
        "FORMAL_FINAL_POINT_RESULT_SHA256": FORMAL_FINAL_POINT_RESULT_SHA256,
        "architecture_contract": architecture_contract,
    }
    original = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def validate_endpoint(
        path: Path, seal: Mapping[str, object], *, held_fold: int) -> dict:
    with configured_base():
        return base.validate_endpoint(path, seal, held_fold=held_fold)


def main() -> None:
    with configured_base():
        base.main()


if __name__ == "__main__":
    main()
