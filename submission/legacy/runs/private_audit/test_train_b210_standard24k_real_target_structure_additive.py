from __future__ import annotations

import sys
from pathlib import Path

import torch


PRIVATE = Path(__file__).resolve().parent
if str(PRIVATE) not in sys.path:
    sys.path.insert(0, str(PRIVATE))

import train_b210_standard24k_real_target_structure_additive as target_structure  # noqa: E402


class Normalizer:
    mean_target = torch.tensor([0.2, -0.1, 0.0])
    std_target = torch.tensor([2.0, 0.5, 1.0])


def test_target_structure_is_scale_and_offset_invariant():
    prediction = torch.randn(2, 20, 4, 5, 3)
    target = prediction.clone()
    target[..., :2] = (
        3.0 * prediction[..., :2] + torch.tensor([4.0, -7.0]))
    loss = target_structure.physical_target_fluctuation_cosine(
        prediction, target, Normalizer())
    assert loss < 1e-6


def test_target_structure_uses_all_future_horizons():
    prediction = torch.randn(2, 20, 4, 5, 3)
    target = prediction.clone()
    target[:, 0, ..., :2] *= -1.0
    loss = target_structure.physical_target_fluctuation_cosine(
        prediction, target, Normalizer())
    assert loss > 0.01


def test_target_structure_gradient_is_first_order_energy_orthogonal():
    prediction = torch.randn(2, 20, 4, 5, 3, requires_grad=True)
    target = torch.randn_like(prediction)
    loss = target_structure.physical_target_fluctuation_cosine(
        prediction, target, Normalizer())
    loss.backward()
    student = target_structure.structure.physical_fluctuation(
        prediction, Normalizer()).detach()
    gradient_physical = prediction.grad[..., :2] / Normalizer.std_target[:2]
    gradient_physical = gradient_physical - gradient_physical.mean(
        dim=(2, 3), keepdim=True)
    radial_dot = (student * gradient_physical).sum(dim=(2, 3, 4))
    scale = (
        torch.linalg.vector_norm(student, dim=(2, 3, 4))
        * torch.linalg.vector_norm(gradient_physical, dim=(2, 3, 4))
    ).clamp_min(1e-12)
    assert torch.max(torch.abs(radial_dot) / scale) < 2e-5


def test_target_structure_contract_is_frozen():
    assert target_structure.additive.ADDITIVE_COEFFICIENT == 0.20
    assert target_structure.CONSISTENCY_KIND.endswith("_to_target")
