#!/usr/bin/env python3
"""Compare additive Rel with target-energy-weighted temporal geometry."""

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


CONSISTENCY_KIND = (
    "physical_uv_target_energy_weighted_per_pixel_temporal_cosine")
BUDGET_MIXER_KIND = (
    "post_warmup_additive_rel_vs_target_temporal_structure_020")


def physical_temporal_fluctuation(
        value: torch.Tensor, normalizer) -> torch.Tensor:
    physical_uv = (
        value * normalizer.std_target + normalizer.mean_target
    )[..., :2]
    return physical_uv - physical_uv.mean(dim=1, keepdim=True)


def target_energy_weighted_temporal_cosine(
        prediction: torch.Tensor, target: torch.Tensor,
        normalizer) -> torch.Tensor:
    """Match local 20-frame waveform phase without imposing its amplitude."""
    student = physical_temporal_fluctuation(prediction, normalizer)
    truth = physical_temporal_fluctuation(
        target, normalizer).detach()
    student_vector = student.permute(0, 2, 3, 1, 4).flatten(start_dim=3)
    truth_vector = truth.permute(0, 2, 3, 1, 4).flatten(start_dim=3)
    cosine = F.cosine_similarity(
        student_vector, truth_vector, dim=-1, eps=1e-12)

    target_energy = truth.square().mean(dim=(1, 4))
    weights = target_energy / target_energy.sum(
        dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return ((1.0 - cosine) * weights).sum(dim=(1, 2)).mean()


def main() -> None:
    real = standard24.self_describing.real
    real.EXPERIMENT = "b210_standard24k_real_target_temporal_structure_additive"
    real.SCHEMA = (
        "b210_standard24k_real_target_temporal_structure_additive_v1")
    real.architecture_contract = (
        standard24.self_describing.self_describing_architecture_contract)
    real.ft12.validate_start_checkpoint = standard24.bind_standard24_validator(
        real.ft12.validate_start_checkpoint)
    real.arm_window_contract = horizon.identical_dense_windows
    real.arm_augmentation_contract = horizon.identical_augmentation
    real.small4k.train_candidate = horizon.bind_cross_horizon_trainer(
        real.small4k.train_candidate,
        consistency_objective=target_energy_weighted_temporal_cosine,
        consistency_kind=CONSISTENCY_KIND,
        budget_mixer=additive.additive_rel_vs_structure_mixer,
        budget_mixer_kind=BUDGET_MIXER_KIND,
        reference_source="target")
    real.main()


if __name__ == "__main__":
    main()
