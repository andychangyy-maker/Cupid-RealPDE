from __future__ import annotations

import sys
from pathlib import Path

import torch


PRIVATE = Path(__file__).resolve().parent
if str(PRIVATE) not in sys.path:
    sys.path.insert(0, str(PRIVATE))

import train_b210_standard24k_real_horizon_consistency as horizon  # noqa: E402
import train_b210_standard24k_real_horizon_structure as structure  # noqa: E402


class Normalizer:
    mean_target = torch.tensor([0.2, -0.1, 0.0])
    std_target = torch.tensor([2.0, 0.5, 1.0])


def test_physical_fluctuation_removes_mean_and_uses_uv():
    value = torch.randn(2, 9, 4, 5, 3)
    result = structure.physical_fluctuation(value, Normalizer())
    assert result.shape == (2, 9, 4, 5, 2)
    assert torch.allclose(
        result.mean(dim=(2, 3)), torch.zeros(2, 9, 2), atol=2e-6)


def test_cosine_consistency_is_scale_and_offset_invariant():
    prediction = torch.randn(2, 20, 4, 5, 3)
    teacher = prediction.clone()
    teacher[:, :horizon.CONSISTENCY_STEPS, ..., :2] = (
        3.0 * prediction[
            :, horizon.SHIFT_STEPS:
               horizon.SHIFT_STEPS + horizon.CONSISTENCY_STEPS, ..., :2]
        + torch.tensor([4.0, -7.0]))
    loss = structure.physical_fluctuation_cosine_consistency(
        prediction, teacher, Normalizer())
    assert loss < 1e-6


def test_cosine_gradient_is_first_order_energy_orthogonal():
    prediction = torch.randn(2, 20, 4, 5, 3, requires_grad=True)
    teacher = torch.randn_like(prediction)
    loss = structure.physical_fluctuation_cosine_consistency(
        prediction, teacher, Normalizer())
    loss.backward()
    student = structure.physical_fluctuation(
        prediction[:, horizon.SHIFT_STEPS:
                   horizon.SHIFT_STEPS + horizon.CONSISTENCY_STEPS],
        Normalizer()).detach()
    gradient_physical = prediction.grad[
        :, horizon.SHIFT_STEPS:
           horizon.SHIFT_STEPS + horizon.CONSISTENCY_STEPS, ..., :2
    ] / Normalizer.std_target[:2]
    gradient_physical = gradient_physical - gradient_physical.mean(
        dim=(2, 3), keepdim=True)
    radial_dot = (student * gradient_physical).sum(dim=(2, 3, 4))
    scale = (
        torch.linalg.vector_norm(student, dim=(2, 3, 4))
        * torch.linalg.vector_norm(gradient_physical, dim=(2, 3, 4))
    ).clamp_min(1e-12)
    assert torch.max(torch.abs(radial_dot) / scale) < 2e-5
