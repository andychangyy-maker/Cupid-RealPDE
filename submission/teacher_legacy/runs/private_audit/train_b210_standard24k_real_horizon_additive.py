#!/usr/bin/env python3
"""Compare additive Rel supervision with equal-value horizon structure."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (REPO_ROOT, PRIVATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_standard24k_real_dense_aug as standard24  # noqa: E402
import train_b210_standard24k_real_horizon_consistency as horizon  # noqa: E402
import train_b210_standard24k_real_horizon_structure as structure  # noqa: E402


ADDITIVE_COEFFICIENT = 0.20
BUDGET_MIXER_KIND = "post_warmup_additive_rel_vs_structure_020"


def additive_rel_vs_structure_mixer(
        direct_loss: torch.Tensor, consistency_loss: torch.Tensor,
        prediction: torch.Tensor, target: torch.Tensor, normalizer,
        iteration: int, fraction: float) -> tuple[
            torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if fraction not in {0.0, horizon.CONSISTENCY_BUDGET_FRACTION}:
        raise ValueError("unexpected horizon arm fraction")
    real = standard24.self_describing.real
    loss_args = real.ft12.npt1.joint_real.real_loss_args()
    rel_loss, _details = (
        real.ft12.npt1.ensemble_train.common.accuracy_objective(
            prediction, target, normalizer, (1.0, 0.0, 0.0)))
    scale = rel_loss.detach() / consistency_loss.detach().clamp_min(1e-12)
    normalized_consistency = consistency_loss * scale
    active = float(iteration > int(loss_args.mse_warmup_updates))
    coefficient = active * ADDITIVE_COEFFICIENT
    auxiliary = (
        rel_loss if fraction == 0.0 else normalized_consistency)
    mixed = direct_loss + coefficient * auxiliary
    return mixed, scale, {
        "rel_component_loss": rel_loss,
        "additive_coefficient": direct_loss.new_tensor(coefficient),
        "post_warmup_active": direct_loss.new_tensor(active),
        "structure_arm": direct_loss.new_tensor(float(fraction > 0.0)),
    }


def main() -> None:
    real = standard24.self_describing.real
    real.EXPERIMENT = "b210_standard24k_real_horizon_additive"
    real.SCHEMA = "b210_standard24k_real_horizon_additive_v1"
    real.architecture_contract = (
        standard24.self_describing.self_describing_architecture_contract)
    real.ft12.validate_start_checkpoint = standard24.bind_standard24_validator(
        real.ft12.validate_start_checkpoint)
    real.arm_window_contract = horizon.identical_dense_windows
    real.arm_augmentation_contract = horizon.identical_augmentation
    real.small4k.train_candidate = horizon.bind_cross_horizon_trainer(
        real.small4k.train_candidate,
        consistency_objective=(
            structure.physical_fluctuation_cosine_consistency),
        consistency_kind=structure.CONSISTENCY_KIND,
        budget_mixer=additive_rel_vs_structure_mixer,
        budget_mixer_kind=BUDGET_MIXER_KIND)
    real.main()


if __name__ == "__main__":
    main()
