from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from physics.weak_dynamics import (  # noqa: E402
    block_integral_moments,
    signed_dynamics_fields,
    weak_dynamics_loss_physical,
)


class WeakDynamicsTests(unittest.TestCase):
    def test_signed_fields_have_expected_shapes_and_signs(self):
        yy, xx = torch.meshgrid(
            torch.arange(8, dtype=torch.float32),
            torch.arange(12, dtype=torch.float32), indexing="ij")
        velocity = torch.zeros(2, 9, 8, 12, 2)
        velocity[..., 0] = xx
        velocity[..., 1] = yy
        fields = signed_dynamics_fields(velocity)
        self.assertEqual(tuple(fields["temporal"].shape), (2, 8, 8, 12, 2))
        self.assertEqual(tuple(fields["divergence"].shape), (2, 9, 7, 11, 1))
        self.assertEqual(tuple(fields["vorticity"].shape), (2, 9, 7, 11, 1))
        torch.testing.assert_close(
            fields["divergence"], torch.full_like(fields["divergence"], 2.0))
        self.assertEqual(int(torch.count_nonzero(fields["vorticity"])), 0)

    def test_block_integral_cancels_signed_checkerboard(self):
        field = torch.ones(1, 8, 8, 8, 1)
        field[:, :, 1::2] *= -1.0
        moment = block_integral_moments(field, 4, 4)
        self.assertEqual(tuple(moment.shape), (1, 1, 2, 2, 2))
        self.assertEqual(int(torch.count_nonzero(moment)), 0)

    def test_loss_is_exact_for_equal_fields_and_constant_offset(self):
        target = torch.randn(2, 20, 16, 32, 2)
        equal = weak_dynamics_loss_physical(target, target)
        self.assertEqual(float(equal), 0.0)
        offset = target + target.new_tensor([3.0, -2.0])
        shifted = weak_dynamics_loss_physical(offset, target)
        torch.testing.assert_close(shifted, torch.zeros_like(shifted))

    def test_loss_has_finite_nonzero_gradient_for_wrong_dynamics(self):
        target = torch.randn(2, 20, 16, 32, 2)
        prediction = target.detach().clone()
        time = torch.linspace(0.0, 1.0, 20).view(1, 20, 1, 1, 1)
        prediction = (prediction + 0.2 * time).requires_grad_()
        loss, components = weak_dynamics_loss_physical(
            prediction, target, return_components=True)
        self.assertEqual(set(components), {
            "temporal_b4", "temporal_b8", "divergence_b4",
            "divergence_b8", "vorticity_b4", "vorticity_b8"})
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
