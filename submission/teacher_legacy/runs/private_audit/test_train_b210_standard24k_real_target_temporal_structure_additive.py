from __future__ import annotations

import sys
from pathlib import Path

import torch


PRIVATE = Path(__file__).resolve().parent
if str(PRIVATE) not in sys.path:
    sys.path.insert(0, str(PRIVATE))

import train_b210_standard24k_real_target_temporal_structure_additive as temporal  # noqa: E402


class Normalizer:
    mean_target = torch.tensor([0.2, -0.1, 0.0])
    std_target = torch.tensor([2.0, 0.5, 1.0])


def test_temporal_fluctuation_removes_local_time_mean():
    value = torch.randn(2, 20, 4, 5, 3)
    result = temporal.physical_temporal_fluctuation(value, Normalizer())
    assert result.shape == (2, 20, 4, 5, 2)
    assert torch.allclose(
        result.mean(dim=1), torch.zeros(2, 4, 5, 2), atol=2e-6)


def test_temporal_cosine_is_per_pixel_scale_and_offset_invariant():
    prediction = torch.randn(2, 20, 4, 5, 3)
    target = prediction.clone()
    scale = 0.5 + 2.0 * torch.rand(2, 1, 4, 5, 1)
    offset = torch.randn(2, 1, 4, 5, 2)
    target[..., :2] = scale * prediction[..., :2] + offset
    loss = temporal.target_energy_weighted_temporal_cosine(
        prediction, target, Normalizer())
    assert loss < 2e-6


def test_temporal_cosine_detects_phase_reversal():
    prediction = torch.randn(2, 20, 4, 5, 3)
    target = prediction.clone()
    target[..., :2] = torch.flip(prediction[..., :2], dims=(1,))
    loss = temporal.target_energy_weighted_temporal_cosine(
        prediction, target, Normalizer())
    assert loss > 0.25


def test_temporal_cosine_gradient_is_locally_energy_orthogonal():
    prediction = torch.randn(2, 20, 4, 5, 3, requires_grad=True)
    target = torch.randn_like(prediction)
    loss = temporal.target_energy_weighted_temporal_cosine(
        prediction, target, Normalizer())
    loss.backward()
    student = temporal.physical_temporal_fluctuation(
        prediction, Normalizer()).detach()
    gradient_physical = prediction.grad[..., :2] / Normalizer.std_target[:2]
    gradient_physical = gradient_physical - gradient_physical.mean(
        dim=1, keepdim=True)
    radial_dot = (student * gradient_physical).sum(dim=(1, 4))
    scale = (
        torch.linalg.vector_norm(student, dim=(1, 4))
        * torch.linalg.vector_norm(gradient_physical, dim=(1, 4))
    ).clamp_min(1e-12)
    assert torch.max(torch.abs(radial_dot) / scale) < 2e-5


def test_temporal_structure_contract_is_frozen():
    assert temporal.additive.ADDITIVE_COEFFICIENT == 0.20
    assert "energy_weighted_per_pixel" in temporal.CONSISTENCY_KIND
