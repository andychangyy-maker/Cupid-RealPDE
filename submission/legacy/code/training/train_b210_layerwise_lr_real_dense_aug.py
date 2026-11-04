#!/usr/bin/env python3
"""Paired Real4k test of a fixed slow encoder learning-rate group.

Both arms use the matched-compute-selected Sim endpoint and identical dense
Real data/augmentation.  The control keeps the legacy single learning rate.
The candidate updates every parameter but uses 0.1x learning rate for the stem
and encoder, preserving source representations without freezing them.
"""

from __future__ import annotations

from typing import Iterable

import torch

import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_scratch_real_dense_aug as paired


EXPERIMENT = "b210_layerwise_lr_real_dense_aug"
SCHEMA = "b210_layerwise_lr_real_dense_aug_v1"
SLOW_PREFIXES = ("stem.", "encoder.")
ENCODER_LR_MULTIPLIER = 0.10
ORIGINAL_TRAIN = small4k.train_candidate
ORIGINAL_ADAMW = torch.optim.AdamW
_CURRENT_MODEL: torch.nn.Module | None = None
_CURRENT_ARM: str | None = None


def partition_optimizer_parameters(
        model: torch.nn.Module,
        parameters: Iterable[torch.nn.Parameter],
        ) -> tuple[tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
    supplied = tuple(parameters)
    if not supplied or len({id(value) for value in supplied}) != len(supplied):
        raise ValueError("layerwise optimizer parameters are empty or duplicated")
    slow_ids = {
        id(value) for name, value in model.named_parameters()
        if name.startswith(SLOW_PREFIXES)
    }
    if not slow_ids:
        raise RuntimeError("layerwise optimizer found no stem/encoder parameters")
    supplied_ids = {id(value) for value in supplied}
    if not slow_ids.issubset(supplied_ids):
        raise RuntimeError("layerwise optimizer did not receive the full encoder")
    slow = tuple(value for value in supplied if id(value) in slow_ids)
    fast = tuple(value for value in supplied if id(value) not in slow_ids)
    if (not slow or not fast or len(slow) + len(fast) != len(supplied)
            or {id(value) for value in slow}.intersection(
                id(value) for value in fast)):
        raise RuntimeError("layerwise optimizer partition is incomplete")
    return slow, fast


def layerwise_adamw(parameters, *args, **kwargs):
    if _CURRENT_MODEL is None or _CURRENT_ARM != "candidate":
        raise RuntimeError("layerwise AdamW requested outside candidate arm")
    supplied = tuple(parameters)
    slow, fast = partition_optimizer_parameters(_CURRENT_MODEL, supplied)
    if "lr" in kwargs:
        learning_rate = float(kwargs["lr"])
    elif args:
        learning_rate = float(args[0])
    else:
        learning_rate = 1.0e-3
    groups = (
        {"params": slow, "lr": learning_rate * ENCODER_LR_MULTIPLIER,
         "layerwise_role": "stem_encoder"},
        {"params": fast, "lr": learning_rate,
         "layerwise_role": "decoder_heads_uq_controller"},
    )
    return ORIGINAL_ADAMW(groups, *args, **kwargs)


def train_arm(*args, **kwargs):
    global _CURRENT_ARM, _CURRENT_MODEL
    model = kwargs.get("model")
    if model is None and args:
        model = args[0]
    if not isinstance(model, torch.nn.Module):
        raise TypeError("layerwise runner has no model")
    access = kwargs.get("access")
    if access is None and len(args) >= 9:
        access = args[8]
    arm = getattr(access, "arm", None)
    if arm not in {"control", "candidate"}:
        raise ValueError("layerwise runner cannot identify paired arm")
    if _CURRENT_ARM is not None or _CURRENT_MODEL is not None:
        raise RuntimeError("layerwise paired arm state is already active")
    _CURRENT_ARM = arm
    _CURRENT_MODEL = model
    original_adamw = small4k.torch.optim.AdamW
    if arm == "candidate":
        small4k.torch.optim.AdamW = layerwise_adamw
    try:
        training, seal = ORIGINAL_TRAIN(*args, **kwargs)
    finally:
        small4k.torch.optim.AdamW = original_adamw
        _CURRENT_ARM = None
        _CURRENT_MODEL = None
    slow, fast = partition_optimizer_parameters(
        model, tuple(model.parameters()))
    training["real_transfer_strategy"] = (
        "legacy_full_finetune" if arm == "control"
        else "stem_encoder_lr_multiplier_0p10")
    training["layerwise_lr"] = {
        "enabled": arm == "candidate",
        "slow_prefixes": list(SLOW_PREFIXES),
        "slow_parameters": sum(value.numel() for value in slow),
        "fast_parameters": sum(value.numel() for value in fast),
        "slow_lr_multiplier": (
            ENCODER_LR_MULTIPLIER if arm == "candidate" else 1.0),
        "all_parameters_trainable": all(
            value.requires_grad for value in model.parameters()),
    }
    return training, seal


def identical_dense_augmentation() -> dict[str, dict[str, float]]:
    return {
        "control": dict(paired.AUGMENTATION),
        "candidate": dict(paired.AUGMENTATION),
    }


def identical_dense_windows(trajectories, _manifest_windows) -> dict[str, list]:
    control = paired.dense_windows(trajectories, 5)
    candidate = paired.dense_windows(trajectories, 5)
    if [(row.trajectory, row.start) for row in control] != [
            (row.trajectory, row.start) for row in candidate]:
        raise RuntimeError("layerwise paired dense windows diverged")
    return {"control": control, "candidate": candidate}


def install_contract() -> None:
    if getattr(paired, "_layerwise_lr_contract_installed", False):
        raise RuntimeError("layerwise Real contract installed twice")
    paired.EXPERIMENT = EXPERIMENT
    paired.SCHEMA = SCHEMA
    paired.CONTROL_STRIDE = 5
    paired.CANDIDATE_STRIDE = 5
    paired.arm_window_contract = identical_dense_windows
    paired.arm_augmentation_contract = identical_dense_augmentation
    paired.small4k.train_candidate = train_arm
    paired._layerwise_lr_contract_installed = True


def main() -> None:
    install_contract()
    paired.main()


if __name__ == "__main__":
    main()
