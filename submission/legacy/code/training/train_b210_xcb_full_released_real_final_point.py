#!/usr/bin/env python3
"""Run the final full-released-Real point stage from XCB/member Sim24k.

This module is only a lineage adapter.  The clean81 loader, sparse-prefix
normalizer, full stride-5 schedule, mature Real4k optimizer loop, endpoint
seals, and CLI remain implemented by
``train_b210_full_released_real_final_point``.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Iterator

import train_b210_full_released_real_final_point as base


EXPERIMENT = "b210_xcb_full_released_real_final_point"
SCHEMA = "b210_xcb_full_released_real_final_point_v1"
ENDPOINT_SCHEMA = "b210_xcb_full_released_real_final_endpoint_v1"
FORMAL_BRANCH = "codex/b210-xcb-full-released-real-final-point-v2"
ACKNOWLEDGEMENT = "RUN_B210_XCB_FULL_RELEASED_REAL_FINAL_POINT_V1"
FORMAL_START_SHA256 = (
    "06ade419811d722e58235acdcb34e480dd6ae88555dd52037ebcb6bbeba2ce99")
MODEL_LINEAGE = "b210_xcb_member_standard24k_sim"
START_EXPECTED_ITERATION = 4_000
START_EXPECTED_STATUS = "PASS_B210_XCB_MEMBER_STANDARD24K_SIM"
EXPECTED_PARAMETERS = 180_553_371
EXPECTED_CONTROLLER_PARAMETERS = 7_169
COMPLETION_EVENT = "b210_xcb_full_released_real_final_point_complete"


def architecture_contract() -> tuple:
    config = base.sim_tke.b210_config(
        "candidate", cross_conditioned_backbone=True,
        member_feature_adapters=True)
    return config, EXPECTED_PARAMETERS, EXPECTED_CONTROLLER_PARAMETERS


CONTRACT = copy.deepcopy(base.CONTRACT)
CONTRACT["question"] = (
    "Transfer the already frozen full released-Real support policy to one "
    "higher-potential XCB/member lineage without changing the Real4k recipe.")
CONTRACT["method_evidence"].update({
    "support_evidence_lineage": "B210 pure-Sim10k",
    "candidate_lineage": MODEL_LINEAGE,
    "lineage_transfer_is_a_new_candidate": True,
    "xcb_matched_full_support_claimed": False,
})
CONTRACT["start"] = {
    "kind": (
        "exact XCB/member Standard24k Sim endpoint; never an old Real or UQ "
        "checkpoint"),
    "checkpoint_sha256": FORMAL_START_SHA256,
    "phase": "sim",
    "local_iteration": START_EXPECTED_ITERATION,
    "cumulative_supervised_updates": 24_000,
    "required_status": START_EXPECTED_STATUS,
    "architecture": (
        "B210 depth4/member4 + cross-conditioned backbone + multiscale "
        "member adapters"),
    "parameters": EXPECTED_PARAMETERS,
    "controller_parameters": EXPECTED_CONTROLLER_PARAMETERS,
    "provenance_caveat": (
        "The from-random Sim24k lifecycle used only released competition "
        "assets, including its existing Real-closure augmentation and P1 "
        "distillation; this stage nevertheless starts before any prior "
        "Real4k/UQ checkpoint."),
}
CONTRACT["outputs"]["lineage"] = MODEL_LINEAGE
CONTRACT_SHA256 = base.dr1.canonical_sha256(CONTRACT)


@contextmanager
def configured_base() -> Iterator[None]:
    """Bind only the lineage-specific identities, restoring them on exit."""
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "SCHEMA": SCHEMA,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "FORMAL_BRANCH": FORMAL_BRANCH,
        "ACKNOWLEDGEMENT": ACKNOWLEDGEMENT,
        "FORMAL_START_SHA256": FORMAL_START_SHA256,
        "MODEL_LINEAGE": MODEL_LINEAGE,
        "START_EXPECTED_ITERATION": START_EXPECTED_ITERATION,
        "START_EXPECTED_STATUS": START_EXPECTED_STATUS,
        "START_MATCHES_SUPPORT_LINEAGE": False,
        "COMPLETION_EVENT": COMPLETION_EVENT,
        "CONTRACT": CONTRACT,
        "CONTRACT_SHA256": CONTRACT_SHA256,
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


def main() -> None:
    with configured_base():
        base.main()


if __name__ == "__main__":
    main()
