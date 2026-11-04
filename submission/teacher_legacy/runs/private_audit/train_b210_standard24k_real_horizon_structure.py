#!/usr/bin/env python3
"""Paired Real4k test of energy-orthogonal horizon structure matching."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (REPO_ROOT, PRIVATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_standard24k_real_dense_aug as standard24  # noqa: E402
import train_b210_standard24k_real_horizon_consistency as horizon  # noqa: E402


CONSISTENCY_KIND = "physical_uv_fluctuation_cosine"


def physical_fluctuation(value: torch.Tensor, normalizer) -> torch.Tensor:
    physical = (
        value * normalizer.std_target + normalizer.mean_target
    )[..., :2]
    return physical - physical.mean(dim=(2, 3), keepdim=True)


def physical_fluctuation_cosine_consistency(
        prediction: torch.Tensor, teacher: torch.Tensor,
        normalizer) -> torch.Tensor:
    student = physical_fluctuation(
        prediction[:, horizon.SHIFT_STEPS:
                   horizon.SHIFT_STEPS + horizon.CONSISTENCY_STEPS],
        normalizer).flatten(start_dim=2)
    target = physical_fluctuation(
        teacher[:, :horizon.CONSISTENCY_STEPS],
        normalizer).detach().flatten(start_dim=2)
    cosine = F.cosine_similarity(student, target, dim=-1, eps=1e-12)
    return (1.0 - cosine).mean()


def main() -> None:
    real = standard24.self_describing.real
    real.EXPERIMENT = "b210_standard24k_real_horizon_structure"
    real.SCHEMA = "b210_standard24k_real_horizon_structure_v1"
    real.architecture_contract = (
        standard24.self_describing.self_describing_architecture_contract)
    real.ft12.validate_start_checkpoint = standard24.bind_standard24_validator(
        real.ft12.validate_start_checkpoint)
    real.arm_window_contract = horizon.identical_dense_windows
    real.arm_augmentation_contract = horizon.identical_augmentation
    real.small4k.train_candidate = horizon.bind_cross_horizon_trainer(
        real.small4k.train_candidate,
        consistency_objective=physical_fluctuation_cosine_consistency,
        consistency_kind=CONSISTENCY_KIND)
    real.main()


if __name__ == "__main__":
    main()
