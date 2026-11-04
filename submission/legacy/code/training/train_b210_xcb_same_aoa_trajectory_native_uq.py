#!/usr/bin/env python3
"""XCB lineage adapter for same-AoA trajectory-crossfit Native-UQ.

The shared runner owns the folds, objective, fixed u800 evidence, pooled
sufficient-state aggregation and fixed all3341 u3200 refit.  This adapter only
binds XCB point/cache schemas, the exact Sim24k UQ source and the sealed XCB
all81/full-Real point result.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import build_b210_xcb_same_aoa_point_oof_residual_cache as cache_builder
import train_b210_same_aoa_trajectory_native_uq as base
import train_b210_xcb_same_aoa_point_oof_fold as point_fold


EXPERIMENT = "b210_xcb_same_aoa_trajectory_native_uq"
FOLD_SCHEMA = "b210_xcb_same_aoa_trajectory_native_uq_fold_v1"
AGGREGATE_SCHEMA = "b210_xcb_same_aoa_trajectory_native_uq_aggregate_v1"
FOLD_ENDPOINT_SCHEMA = "b210_xcb_same_aoa_trajectory_native_uq_endpoint_v1"
REFIT_SCHEMA = "b210_xcb_same_aoa_trajectory_native_uq_refit_v1"
REFIT_RESULT_SCHEMA = "b210_xcb_same_aoa_trajectory_native_uq_run_v1"
REFIT_CYCLE_SCHEMA = "b210_xcb_same_aoa_trajectory_native_uq_cycle_v1"
ACKNOWLEDGEMENT = "RUN_B210_XCB_SAME_AOA_TRAJECTORY_NATIVE_UQ_V1"
FOLD_COMPLETION_EVENT = "b210_xcb_same_aoa_native_uq_fold_complete"
AGGREGATE_COMPLETION_EVENT = "b210_xcb_same_aoa_native_uq_aggregate_complete"
REFIT_COMPLETION_EVENT = "b210_xcb_same_aoa_native_uq_refit_complete"
FINAL_ENDPOINT = "fixed_xcb_same_aoa_oof_u3200"

fold_plan = base.fold_plan
whole_refit = base.whole_refit
whole_loao = base.whole_loao
dr1 = base.dr1
ft12 = base.ft12
smu1 = base.smu1
uq_release = base.uq_release
uq_train = base.uq_train
EXPECTED_SAMPLES = base.EXPECTED_SAMPLES
UPDATES_PER_FOLD = base.UPDATES_PER_FOLD
CYCLES = base.CYCLES
TOTAL_UPDATES = base.TOTAL_UPDATES
FORMAL_FINAL_POINT_RESULT_PATH = point_fold.FORMAL_FINAL_POINT_RESULT_PATH
FORMAL_FINAL_POINT_RESULT_SHA256 = point_fold.FORMAL_FINAL_POINT_RESULT_SHA256


@contextmanager
def configured_base() -> Iterator[None]:
    """Bind only lineage-specific interfaces, restoring shared globals."""
    replacements = {
        "point_fold": point_fold,
        "cache_builder": cache_builder,
        "EXPERIMENT": EXPERIMENT,
        "FOLD_SCHEMA": FOLD_SCHEMA,
        "AGGREGATE_SCHEMA": AGGREGATE_SCHEMA,
        "FOLD_ENDPOINT_SCHEMA": FOLD_ENDPOINT_SCHEMA,
        "REFIT_SCHEMA": REFIT_SCHEMA,
        "REFIT_RESULT_SCHEMA": REFIT_RESULT_SCHEMA,
        "REFIT_CYCLE_SCHEMA": REFIT_CYCLE_SCHEMA,
        "ACKNOWLEDGEMENT": ACKNOWLEDGEMENT,
        "FOLD_COMPLETION_EVENT": FOLD_COMPLETION_EVENT,
        "AGGREGATE_COMPLETION_EVENT": AGGREGATE_COMPLETION_EVENT,
        "REFIT_COMPLETION_EVENT": REFIT_COMPLETION_EVENT,
        "FINAL_ENDPOINT": FINAL_ENDPOINT,
    }
    original = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def load_final_point_artifact(
        result_path: Path, expected_result_sha256: str, device):
    with configured_base():
        return base.load_final_point_artifact(
            result_path, expected_result_sha256, device)


def reset_uq_from_lineage_sim(
        model, path: Path, expected_sha256: str, expected_model_config):
    with configured_base():
        return base.reset_uq_from_lineage_sim(
            model, path, expected_sha256, expected_model_config)


def main() -> None:
    with configured_base():
        base.main()


if __name__ == "__main__":
    main()
