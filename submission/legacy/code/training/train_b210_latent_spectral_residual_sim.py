#!/usr/bin/env python3
"""Paired Sim10k test of one low-rank global block in the B210 bottleneck.

Both arms instantiate the identical expanded graph so their common RNG and
update-zero predictions match.  The control freezes the zero-output spectral
branch; the candidate trains it jointly with the otherwise unchanged B210
recipe.  This isolates global latent interaction from generic extra compute or
parameter initialization.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import torch

import train_b210_np_functional_sim as sim


METHOD = "b210_latent_spectral_residual_r64_m2x4_seed1_v1"
SCHEMA = "b210_latent_spectral_residual_sim_v1"
RANK = 64
MODES_Y = 2
MODES_X = 4
RESIDUAL_LIMIT = 0.50
EXPECTED_TOTAL_PARAMETERS = 178_991_255
EXPECTED_MODULE_PARAMETERS = 285_680
EXPECTED_CONTROLLER_PARAMETERS = 7_169
ORIGINAL_CONFIG = sim.b210_config
ORIGINAL_BUILD = sim.build_arm_model
ORIGINAL_SIM_UPDATE = sim.sim_update
ORIGINAL_RUN_EXPERIMENT = sim.run_experiment
INITIAL_STATES: dict[str, OrderedDict[str, torch.Tensor]] = {}
AUDIT = {
    f"{arm}_{group}_{suffix}": False
    for arm in ("control", "candidate")
    for group in ("projection", "spectral", "input")
    for suffix in ("gradient_seen", "updated_at_u200")
}


def spectral_config(*args, **kwargs):
    return replace(
        ORIGINAL_CONFIG(*args, **kwargs),
        latent_spectral_rank=RANK,
        latent_spectral_modes_y=MODES_Y,
        latent_spectral_modes_x=MODES_X,
        latent_spectral_residual_limit=RESIDUAL_LIMIT,
    )


def state_group(name: str) -> str:
    if name.startswith("expand."):
        return "projection"
    if name.startswith(("weight_positive", "weight_negative")):
        return "spectral"
    if name.startswith(("norm.", "reduce.")):
        return "input"
    raise ValueError(f"unknown latent spectral parameter: {name}")


def module_state(model) -> OrderedDict[str, torch.Tensor]:
    module = model.latent_spectral_residual
    if module is None:
        raise RuntimeError("latent spectral residual is missing")
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in module.state_dict().items())


def group_state(state, group: str) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in state.items()
        if state_group(name) == group)


def group_has_gradient(module, group: str) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for name, parameter in module.named_parameters()
        if state_group(name) == group)


def configure_trainable(model, arm: str) -> tuple[torch.nn.Parameter, ...]:
    if arm not in {"control", "candidate"}:
        raise ValueError(f"unknown B210 Sim arm {arm}")
    for name, parameter in model.named_parameters():
        candidate_only = (
            name.startswith("energy_controller.")
            or name.startswith("latent_spectral_residual."))
        parameter.requires_grad_(arm == "candidate" or not candidate_only)
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    controller = sum(
        parameter.numel() for parameter in model.energy_controller.parameters())
    spectral = sum(
        parameter.numel()
        for parameter in model.latent_spectral_residual.parameters())
    expected = total if arm == "candidate" else total - controller - spectral
    if sum(parameter.numel() for parameter in trainable) != expected:
        raise RuntimeError("latent spectral trainability contract changed")
    return trainable


def build_with_contract(arm: str, *args, **kwargs):
    model = ORIGINAL_BUILD(arm, *args, **kwargs)
    module = model.latent_spectral_residual
    if module is None:
        raise RuntimeError("latent spectral module was not constructed")
    total = sum(parameter.numel() for parameter in model.parameters())
    spectral = sum(parameter.numel() for parameter in module.parameters())
    controller = sum(
        parameter.numel() for parameter in model.energy_controller.parameters())
    if (total != EXPECTED_TOTAL_PARAMETERS
            or spectral != EXPECTED_MODULE_PARAMETERS
            or controller != EXPECTED_CONTROLLER_PARAMETERS):
        raise RuntimeError(
            "latent spectral parameter contract changed: "
            f"total={total}, spectral={spectral}, controller={controller}")
    if (torch.count_nonzero(module.expand.weight)
            or torch.count_nonzero(module.expand.bias)):
        raise RuntimeError("latent spectral projection is not exact zero")
    INITIAL_STATES[arm] = module_state(model)
    return model


def sim_update_with_audit(*args, **kwargs):
    values, audit, key, flags = ORIGINAL_SIM_UPDATE(*args, **kwargs)
    model = args[0]
    iteration = int(args[8])
    arm = str(args[11])
    module = model.latent_spectral_residual
    for group in ("projection", "spectral", "input"):
        AUDIT[f"{arm}_{group}_gradient_seen"] |= group_has_gradient(
            module, group)
    if iteration == sim.CONTROL_UPDATES:
        current = module_state(model)
        initial = INITIAL_STATES[arm]
        for group in ("projection", "spectral", "input"):
            AUDIT[f"{arm}_{group}_updated_at_u200"] = not (
                sim.tensor_state_equal(
                    group_state(initial, group), group_state(current, group)))
    values.update({
        f"latent_spectral_{group}_gradient_nonzero": float(
            group_has_gradient(module, group))
        for group in ("projection", "spectral", "input")
    })
    return values, audit, key, flags


def validate_mechanical_audit() -> None:
    failures = []
    for group in ("projection", "spectral", "input"):
        if AUDIT[f"control_{group}_gradient_seen"]:
            failures.append(f"control_{group}_gradient_seen")
        if AUDIT[f"control_{group}_updated_at_u200"]:
            failures.append(f"control_{group}_updated")
        if not AUDIT[f"candidate_{group}_gradient_seen"]:
            failures.append(f"candidate_{group}_gradient_missing")
        if not AUDIT[f"candidate_{group}_updated_at_u200"]:
            failures.append(f"candidate_{group}_update_missing")
    if failures:
        raise RuntimeError(
            f"latent spectral mechanical audit failed: {failures}; {AUDIT}")


def run_with_contract(*args, **kwargs):
    args[1].method = METHOD
    summary = ORIGINAL_RUN_EXPERIMENT(*args, **kwargs)
    validate_mechanical_audit()
    summary["schema"] = SCHEMA
    summary["method"] = METHOD
    summary["parameters"] = EXPECTED_TOTAL_PARAMETERS
    summary["training_contract"]["latent_spectral_residual"] = {
        "role": "single bounded global residual after local bottleneck",
        "rank": RANK,
        "modes_y": MODES_Y,
        "modes_x": MODES_X,
        "residual_limit": RESIDUAL_LIMIT,
        "module_parameters": EXPECTED_MODULE_PARAMETERS,
        "control": "identical graph, branch frozen at exact-zero output",
        "candidate": "branch and complete native B210 graph jointly trained",
        "inference": "one extra low-rank rFFT2/irFFT2 at 4x8 latent grid",
    }
    summary["paired_candidate"]["latent_spectral_audit"] = dict(AUDIT)
    return summary


def main() -> None:
    sim.EXPECTED_PARAMETERS = EXPECTED_TOTAL_PARAMETERS
    sim.b210_config = spectral_config
    sim.build_arm_model = build_with_contract
    sim.configure_trainable = configure_trainable
    sim.sim_update = sim_update_with_audit
    sim.run_experiment = run_with_contract
    sim.REAL_CLOSURE_SEED1_METHOD = METHOD
    sim.main()


if __name__ == "__main__":
    main()
