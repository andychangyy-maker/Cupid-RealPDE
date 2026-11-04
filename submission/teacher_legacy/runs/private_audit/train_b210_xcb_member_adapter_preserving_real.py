#!/usr/bin/env python3
"""Paired Real4k test that preserves the Sim-trained member adapters.

Both arms start from the same XCB/member Standard24k endpoint and use the
identical dense Real windows, augmentation, sample stream and objective.  The
control updates every parameter.  The candidate freezes only the multiscale
member feature adapters while leaving the backbone, cross-conditioner, member
readouts, energy controller and UQ tower trainable.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (REPO_ROOT, PRIVATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_xcb_member_standard24k_real_dense_aug as xcb  # noqa: E402


EXPERIMENT = "b210_xcb_member_adapter_preserving_real"
SCHEMA = "b210_xcb_member_adapter_preserving_real_v1"
ADAPTER_PREFIX = "member_feature_adapters."
_ORIGINAL_TRAIN = xcb.self_describing.real.small4k.train_candidate
_ORIGINAL_CONFIGURE = xcb.self_describing.real.ft12.configure_trainable
_FREEZE_ADAPTERS = False


def identical_dense_windows(trajectories, _manifest_windows):
    real = xcb.self_describing.real
    windows = real.dense_windows(trajectories, real.CANDIDATE_STRIDE)
    return {"control": list(windows), "candidate": list(windows)}


def identical_augmentation():
    augmentation = dict(xcb.self_describing.real.AUGMENTATION)
    return {"control": dict(augmentation), "candidate": dict(augmentation)}


def selected_state(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    state = OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.state_dict().items()
        if name.startswith(ADAPTER_PREFIX))
    if not state:
        raise RuntimeError("member-adapter preservation found no adapter state")
    return state


def configure_trainable(model, expected_parameters=None):
    if not _FREEZE_ADAPTERS:
        return _ORIGINAL_CONFIGURE(model, expected_parameters)
    total = sum(value.numel() for value in model.parameters())
    if expected_parameters is not None and total != expected_parameters:
        raise RuntimeError("member-adapter model parameter contract changed")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith(ADAPTER_PREFIX))
    frozen = tuple(
        parameter for name, parameter in model.named_parameters()
        if name.startswith(ADAPTER_PREFIX))
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad)
    if not frozen or not trainable:
        raise RuntimeError("member-adapter parameter partition is empty")
    if any(value.requires_grad for value in frozen):
        raise RuntimeError("a protected member adapter remains trainable")
    if not all(
            value.requires_grad
            for value in model.energy_controller.parameters()):
        raise RuntimeError("energy controller must remain trainable")
    if not all(
            value.requires_grad
            for value in model.cross_conditioner.parameters()):
        raise RuntimeError("cross-conditioner must remain trainable")
    return trainable


def train_arm(model, *args, **kwargs):
    global _FREEZE_ADAPTERS
    access = kwargs.get("access")
    if access is None and len(args) >= 8:
        access = args[7]
    arm = getattr(access, "arm", None)
    if arm not in {"control", "candidate"}:
        raise ValueError("cannot identify member-adapter paired arm")
    _FREEZE_ADAPTERS = arm == "candidate"
    before = selected_state(model)
    try:
        training, seal = _ORIGINAL_TRAIN(model, *args, **kwargs)
    finally:
        _FREEZE_ADAPTERS = False
    after = selected_state(model)
    unchanged = (
        before.keys() == after.keys()
        and all(torch.equal(before[name], after[name]) for name in before))
    if arm == "candidate" and not unchanged:
        raise RuntimeError("protected member-adapter state changed")
    if arm == "control" and unchanged:
        raise RuntimeError("control member adapters did not update")
    frozen_parameters = sum(
        value.numel() for name, value in model.named_parameters()
        if name.startswith(ADAPTER_PREFIX))
    training["member_adapter_preservation"] = {
        "enabled": arm == "candidate",
        "prefix": ADAPTER_PREFIX,
        "state_tensors": len(before),
        "parameters": frozen_parameters,
        "raw_state_bitwise_unchanged": unchanged,
        "cross_conditioner_trainable": all(
            value.requires_grad
            for value in model.cross_conditioner.parameters()),
        "inference_graph_changed": False,
    }
    return training, seal


def main() -> None:
    real = xcb.self_describing.real
    real.EXPERIMENT = EXPERIMENT
    real.SCHEMA = SCHEMA
    real.architecture_contract = (
        xcb.self_describing.self_describing_architecture_contract)
    real.ft12.validate_start_checkpoint = xcb.bind_validator(
        real.ft12.validate_start_checkpoint)
    real.arm_window_contract = identical_dense_windows
    real.arm_augmentation_contract = identical_augmentation
    real.ft12.configure_trainable = configure_trainable
    real.small4k.train_candidate = train_arm
    real.main()


if __name__ == "__main__":
    main()
