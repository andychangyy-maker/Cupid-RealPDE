#!/usr/bin/env python3
"""Run the B210-v7 budgeted-disagreement Real-ft12 protocol.

The data split, endpoint sealing, EMA policy, optimizer, and held-label access
rules are inherited unchanged from the audited B210 Real-ft12 runner.  This
entry point changes only the exact Sim10k start architecture and gives the
result a separate schema/contract identity.
"""

from __future__ import annotations

from copy import deepcopy

import train_b210_np_functional_real_ft12_loao as core


EXPERIMENT = "b210_budgeted_disagreement_real_ft12_loao"
SCHEMA = "b210_budgeted_disagreement_real_ft12_fold_v1"
AGGREGATE_SCHEMA = "b210_budgeted_disagreement_real_ft12_aggregate_v1"
ENDPOINT_SCHEMA = "b210_budgeted_disagreement_real_ft12_endpoint_v1"
FORMAL_BRANCH = "codex/b210-legacy-tke-gradient-floor"
FORMAL_ACKNOWLEDGEMENT = "B210_BUDGETED_DISAGREEMENT_REAL_FT12"
EXPECTED_PARAMETERS = core.b210_sim.EXPECTED_BUDGETED_DISAGREEMENT_PARAMETERS


def expected_config():
    """Return the exact passed v7 Sim architecture."""
    return core.b210_sim.b210_config(
        "candidate",
        member_feature_adapters=True,
        mean_preserving_member_adapters=True,
        detach_member_adapter_context=True,
        detach_uq_context=True,
        budgeted_disagreement_allocation=True,
    )


def configure_core() -> dict:
    """Bind the audited generic runner to the separately identified v7 arm."""
    contract = deepcopy(core.CONTRACT)
    contract["hypothesis"] = (
        "The Sim-validated budget-preserving disagreement allocator retains "
        "its SPS gain through the unchanged prefix-only Real-ft12 protocol, "
        "without sacrificing the fixed EMA-u12000 point endpoint.")
    contract["start"].update({
        "architecture": (
            "B210 v7 isolated distribution tower plus 80-parameter "
            "budgeted disagreement allocator"),
        "parameters": EXPECTED_PARAMETERS,
        "sim_method": core.b210_sim.BUDGETED_DISAGREEMENT_ALLOCATION_METHOD,
        "member_feature_adapters": True,
        "mean_preserving_member_adapters": True,
        "detach_member_adapter_context": True,
        "detach_uq_context": True,
        "budgeted_disagreement_allocation": True,
    })
    contract["training"]["architecture_delta"] = (
        "none after loading v7; all 180,355,823 parameters follow the frozen "
        "Real-ft12 optimizer and EMA schedule")
    contract["training"]["distribution_semantics"] = (
        "base UQ sees collapsed members; detached real disagreement only "
        "redistributes spatial interval width budget")

    core.EXPERIMENT = EXPERIMENT
    core.SCHEMA = SCHEMA
    core.AGGREGATE_SCHEMA = AGGREGATE_SCHEMA
    core.ENDPOINT_SCHEMA = ENDPOINT_SCHEMA
    core.FORMAL_BRANCH = FORMAL_BRANCH
    core.FORMAL_ACKNOWLEDGEMENT = FORMAL_ACKNOWLEDGEMENT
    core.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
    core.expected_b210_config = expected_config
    core.CONTRACT = contract
    core.CONTRACT_SHA256 = core.dr1.canonical_sha256(contract)
    return contract


def main() -> None:
    configure_core()
    core.main()


if __name__ == "__main__":
    main()
