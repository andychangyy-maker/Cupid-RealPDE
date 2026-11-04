#!/usr/bin/env python3
"""Paired Real4k test of a CEOD-style conditional distribution objective."""

from __future__ import annotations

import math

import torch

from conditional_embedding_objective import conditional_embedding_discrepancy
import train_b210_np_functional_real_small4k_confirm_held0 as small4k
import train_b210_scratch_real_dense_aug as paired


EXPERIMENT = "b210_conditional_embedding_real_dense_aug"
SCHEMA = "b210_conditional_embedding_real_dense_aug_v1"
CEOD_WEIGHT = 0.05
CEOD_REGULARIZATION = 0.10
ORIGINAL_TRAIN = small4k.train_candidate


def conditional_embedding_auxiliary(
        history_input: torch.Tensor, prediction: torch.Tensor,
        target: torch.Tensor, _target_norm: torch.Tensor, normalizer,
        ) -> tuple[torch.Tensor, dict[str, float]]:
    physical_prediction = normalizer.postprocess_prediction(prediction)
    loss, metrics = conditional_embedding_discrepancy(
        history_input, physical_prediction, target,
        regularization=CEOD_REGULARIZATION)
    gradient = torch.autograd.grad(
        CEOD_WEIGHT * loss, prediction, retain_graph=True,
        allow_unused=False)[0]
    gradient_norm = float(gradient.detach().float().square().sum().sqrt())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise FloatingPointError(
            "conditional embedding prediction gradient is invalid")
    return loss, {
        **{f"ceod_{key}": value for key, value in metrics.items()},
        "ceod_weighted_prediction_gradient_norm": gradient_norm,
        "ceod_uses_target_free_history_condition": 1.0,
        "ceod_uses_physical_uv_summaries": 1.0,
    }


def train_arm(*args, **kwargs):
    access = kwargs.get("access")
    if access is None and len(args) >= 9:
        access = args[8]
    arm = getattr(access, "arm", None)
    if arm not in {"control", "candidate"}:
        raise ValueError("conditional embedding runner cannot identify arm")
    if (kwargs.get("region_auxiliary") is not None
            or float(kwargs.get("region_auxiliary_weight", 0.0)) != 0.0):
        raise RuntimeError("conditional embedding region slot is already used")
    if arm == "candidate":
        kwargs["region_auxiliary"] = conditional_embedding_auxiliary
        kwargs["region_auxiliary_weight"] = CEOD_WEIGHT
    training, seal = ORIGINAL_TRAIN(*args, **kwargs)
    training["real_transfer_strategy"] = (
        "legacy_full_finetune" if arm == "control"
        else "full_finetune_plus_conditional_embedding")
    training["conditional_embedding"] = {
        "enabled": arm == "candidate",
        "weight": CEOD_WEIGHT if arm == "candidate" else 0.0,
        "regularization": CEOD_REGULARIZATION,
        "condition": "target-free physical history summaries",
        "outputs": "physical prediction/target uv summaries",
        "same_batch_conditional_operator": True,
        "inference_change": False,
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
        raise RuntimeError("conditional embedding dense windows diverged")
    return {"control": control, "candidate": candidate}


def install_contract() -> None:
    if getattr(paired, "_conditional_embedding_contract_installed", False):
        raise RuntimeError("conditional embedding contract installed twice")
    paired.EXPERIMENT = EXPERIMENT
    paired.SCHEMA = SCHEMA
    paired.CONTROL_STRIDE = 5
    paired.CANDIDATE_STRIDE = 5
    paired.arm_window_contract = identical_dense_windows
    paired.arm_augmentation_contract = identical_dense_augmentation
    paired.small4k.train_candidate = train_arm
    paired._conditional_embedding_contract_installed = True


def main() -> None:
    install_contract()
    paired.main()


if __name__ == "__main__":
    main()
