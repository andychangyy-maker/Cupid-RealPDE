#!/usr/bin/env python3
"""Paired Real4k test of preserving a selected Sim encoder during adaptation.

Both arms use identical stride-5 Real windows, identical input augmentation,
the same sample/bootstrap stream and the terminal Sim checkpoint selected by
the matched-compute audit.
The control updates every parameter.  The candidate freezes only ``stem`` and
``encoder``; bottleneck, decoder, regime modulation, point/UQ heads and the
physical-energy controller remain trainable.  The clean716 tail is still
opened only after both fixed EMA-u4000 endpoints have been sealed by the reused
runner.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

import train_b210_np_functional_real_ft12_loao as ft12
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_scratch_real_dense_aug as paired


EXPERIMENT = "b210_encoder_preserving_real_dense_aug"
SCHEMA = "b210_encoder_preserving_real_dense_aug_v1"
FROZEN_PREFIXES = ("stem.", "encoder.")
EXPECTED_FROZEN_PARAMETERS = 83_505_870
EXPECTED_TRAINABLE_PARAMETERS = 95_199_705
ORIGINAL_TRAIN_CANDIDATE = small4k.train_candidate


def frozen_parameter_sha256(model: torch.nn.Module) -> str:
    state = OrderedDict(
        (name, value)
        for name, value in model.named_parameters()
        if name.startswith(FROZEN_PREFIXES))
    if not state:
        raise RuntimeError("encoder-preserving frozen state is empty")
    return small4k.ema_lib.tensor_state_sha256(state)


def configure_encoder_preserving_trainable(
        model: torch.nn.Module,
        expected_parameters: int | None = None,
        ) -> tuple[torch.nn.Parameter, ...]:
    expected_total = (
        ft12.EXPECTED_PARAMETERS
        if expected_parameters is None else int(expected_parameters))
    named = tuple(model.named_parameters())
    if sum(value.numel() for _name, value in named) != expected_total:
        raise RuntimeError("encoder-preserving total parameter contract changed")
    for name, parameter in named:
        parameter.requires_grad_(not name.startswith(FROZEN_PREFIXES))
    frozen = tuple(
        value for _name, value in named if not value.requires_grad)
    trainable = tuple(
        value for _name, value in named if value.requires_grad)
    frozen_count = sum(value.numel() for value in frozen)
    trainable_count = sum(value.numel() for value in trainable)
    if (frozen_count != EXPECTED_FROZEN_PARAMETERS
            or trainable_count != EXPECTED_TRAINABLE_PARAMETERS
            or frozen_count + trainable_count != expected_total):
        raise RuntimeError(
            "encoder-preserving partition changed: "
            f"frozen={frozen_count}, trainable={trainable_count}")
    if not all(
            value.requires_grad
            for value in model.energy_controller.parameters()):
        raise RuntimeError("physical-energy controller must remain trainable")
    return trainable


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
        raise RuntimeError("identical dense window construction diverged")
    return {"control": control, "candidate": candidate}


def train_arm(*args, **kwargs):
    model = kwargs.get("model")
    if model is None and args:
        model = args[0]
    if not isinstance(model, torch.nn.Module):
        raise TypeError("encoder-preserving runner has no model")
    access = kwargs.get("access")
    if access is None and len(args) >= 9:
        access = args[8]
    arm = getattr(access, "arm", None)
    if arm not in {"control", "candidate"}:
        raise ValueError("encoder-preserving runner cannot identify paired arm")
    original_configure = ft12.configure_trainable
    frozen_before = (
        frozen_parameter_sha256(model) if arm == "candidate" else None)
    if arm == "candidate":
        ft12.configure_trainable = configure_encoder_preserving_trainable
    try:
        training, seal = ORIGINAL_TRAIN_CANDIDATE(*args, **kwargs)
    finally:
        ft12.configure_trainable = original_configure
    frozen_after = (
        frozen_parameter_sha256(model) if arm == "candidate" else None)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen stem/encoder changed during Real4k")
    training["real_transfer_strategy"] = (
        "freeze_stem_encoder" if arm == "candidate" else "full_finetune")
    training["frozen_parameter_prefixes"] = (
        list(FROZEN_PREFIXES) if arm == "candidate" else [])
    training["frozen_parameters"] = (
        EXPECTED_FROZEN_PARAMETERS if arm == "candidate" else 0)
    training["trainable_parameters"] = (
        EXPECTED_TRAINABLE_PARAMETERS
        if arm == "candidate" else ft12.EXPECTED_PARAMETERS)
    training["frozen_parameter_state_sha256_before"] = frozen_before
    training["frozen_parameter_state_sha256_after"] = frozen_after
    training["frozen_parameter_state_unchanged"] = (
        frozen_before == frozen_after if arm == "candidate" else None)
    return training, seal


def install_contract() -> None:
    if getattr(paired, "_encoder_preserving_contract_installed", False):
        raise RuntimeError("encoder-preserving contract installed twice")
    paired.EXPERIMENT = EXPERIMENT
    paired.SCHEMA = SCHEMA
    paired.CONTROL_STRIDE = 5
    paired.CANDIDATE_STRIDE = 5
    paired.arm_window_contract = identical_dense_windows
    paired.arm_augmentation_contract = identical_dense_augmentation
    paired.small4k.train_candidate = train_arm
    paired._encoder_preserving_contract_installed = True


def main() -> None:
    install_contract()
    paired.main()


if __name__ == "__main__":
    main()
