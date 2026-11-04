#!/usr/bin/env python3
"""Compare additive Rel supervision with true-future structure geometry."""

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
import train_b210_standard24k_real_horizon_additive as additive  # noqa: E402
import train_b210_standard24k_real_horizon_consistency as horizon  # noqa: E402
import train_b210_standard24k_real_horizon_structure as structure  # noqa: E402


CONSISTENCY_KIND = "physical_uv_full_future_fluctuation_cosine_to_target"
BUDGET_MIXER_KIND = "post_warmup_additive_rel_vs_target_structure_020"


def physical_target_fluctuation_cosine(
        prediction: torch.Tensor, target: torch.Tensor,
        normalizer) -> torch.Tensor:
    """Match spatial fluctuation direction without imposing its magnitude."""
    student = structure.physical_fluctuation(
        prediction, normalizer).flatten(start_dim=2)
    truth = structure.physical_fluctuation(
        target, normalizer).detach().flatten(start_dim=2)
    cosine = F.cosine_similarity(student, truth, dim=-1, eps=1e-12)
    return (1.0 - cosine).mean()


def main() -> None:
    real = standard24.self_describing.real
    real.EXPERIMENT = "b210_standard24k_real_target_structure_additive"
    real.SCHEMA = "b210_standard24k_real_target_structure_additive_v1"
    real.architecture_contract = (
        standard24.self_describing.self_describing_architecture_contract)
    real.ft12.validate_start_checkpoint = standard24.bind_standard24_validator(
        real.ft12.validate_start_checkpoint)
    real.arm_window_contract = horizon.identical_dense_windows
    real.arm_augmentation_contract = horizon.identical_augmentation
    real.small4k.train_candidate = horizon.bind_cross_horizon_trainer(
        real.small4k.train_candidate,
        consistency_objective=physical_target_fluctuation_cosine,
        consistency_kind=CONSISTENCY_KIND,
        budget_mixer=additive.additive_rel_vs_structure_mixer,
        budget_mixer_kind=BUDGET_MIXER_KIND,
        reference_source="target")
    real.main()


if __name__ == "__main__":
    main()
