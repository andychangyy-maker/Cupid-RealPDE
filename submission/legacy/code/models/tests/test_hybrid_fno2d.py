import unittest
from pathlib import Path
import sys

import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.hybrid_fno2d import HybridFNO2d, HybridFNO2dConfig


class HybridFNO2dTests(unittest.TestCase):
    def small_model(self, **overrides):
        values = dict(width=8, layers=2, modes_y=3, modes_x=4,
                      groups=2, padding=1)
        values.update(overrides)
        return HybridFNO2d(HybridFNO2dConfig(**values))

    def test_shape_finite_and_backward(self):
        model = self.small_model()
        x = torch.randn(2, 20, 8, 12, 3, requires_grad=True)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 20, 8, 12, 3))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(model.blocks[0].spectral.weight_positive.grad)

    def test_normalized_persistence_transform(self):
        model = self.small_model(layers=1)
        with torch.no_grad():
            model.decoder[-1].weight.zero_()
            model.decoder[-1].bias.zero_()
        model.set_normalizer_statistics(
            torch.tensor([2.0, 4.0, 0.0]),
            torch.tensor([1.0, 2.0, 0.0]),
            torch.tensor([3.0, 6.0, 0.0]),
            torch.tensor([2.0, 4.0, 0.0]))
        x = torch.randn(1, 20, 8, 12, 3)
        output = model(x)
        expected = x[:, -1] * torch.tensor([1.5, 1.5, 0.0])
        expected = expected + torch.tensor([0.5, 0.5, 0.0])
        self.assertTrue(torch.allclose(output[:, 0], expected, atol=1e-6))
        self.assertTrue(torch.allclose(output[:, -1], expected, atol=1e-6))

    def test_state_round_trip_preserves_output(self):
        model = self.small_model()
        x = torch.randn(1, 20, 8, 12, 3)
        expected = model(x)
        restored = self.small_model()
        restored.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(expected, restored(x)))

    def test_native_uncertainty_is_pixelwise_positive_and_scaled(self):
        model = self.small_model(predict_uncertainty=True, uncertainty_hidden=6)
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3), torch.ones(3),
            torch.tensor([2.0, 4.0, 1.0]))
        x = torch.randn(2, 20, 8, 12, 3, requires_grad=True)
        prediction, lower, upper = model.forward_distribution(x)
        self.assertEqual(tuple(prediction.shape), (2, 20, 8, 12, 3))
        self.assertEqual(tuple(lower.shape), (2, 20, 8, 12, 2))
        self.assertEqual(tuple(upper.shape), (2, 20, 8, 12, 2))
        self.assertTrue(torch.all(lower > 0))
        self.assertTrue(torch.all(upper > 0))
        physical = lower * torch.tensor([2.0, 4.0])
        expected = torch.tensor([0.01075, 0.00550])
        self.assertTrue(torch.allclose(physical, expected, atol=1e-7))
        (prediction.square().mean() + lower.mean() + upper.mean()).backward()
        self.assertIsNotNone(model.uncertainty_decoder[-1].weight.grad)

    def test_temporal_decoder_shares_horizon_dynamics(self):
        model = self.small_model(
            predict_uncertainty=True, uncertainty_hidden=6,
            temporal_decoder_width=4, temporal_decoder_layers=2,
            temporal_decoder_groups=2)
        x = torch.randn(2, 20, 8, 12, 3, requires_grad=True)
        prediction, lower, upper = model.forward_distribution(x)
        self.assertEqual(tuple(prediction.shape), (2, 20, 8, 12, 3))
        self.assertEqual(tuple(lower.shape), (2, 20, 8, 12, 2))
        loss = prediction.square().mean() + lower.mean() + upper.mean()
        loss.backward()
        self.assertIsNotNone(model.horizon_embedding.grad)
        self.assertIsNotNone(model.temporal_blocks[0].operator[0].weight.grad)

    def test_energy_gain_scales_fluctuation_not_temporal_mean(self):
        model = self.small_model(predict_energy_gain=True, energy_gain_hidden=6)
        x = torch.randn(2, 20, 8, 12, 3, requires_grad=True)
        identity_prediction = model(x)
        identity_velocity = identity_prediction[..., :2]
        identity_mean = identity_velocity.mean(dim=1)
        identity_fluctuation = identity_velocity - identity_mean.unsqueeze(1)
        with torch.no_grad():
            model.energy_gain_decoder[-1].bias.fill_(torch.log(torch.tensor(2.0)))
        scaled_prediction = model(x)
        scaled_velocity = scaled_prediction[..., :2]
        scaled_mean = scaled_velocity.mean(dim=1)
        scaled_fluctuation = scaled_velocity - scaled_mean.unsqueeze(1)
        self.assertTrue(torch.allclose(identity_mean, scaled_mean, atol=1e-6))
        self.assertTrue(torch.allclose(
            scaled_fluctuation, 2.0 * identity_fluctuation, atol=2e-6))
        scaled_prediction.square().mean().backward()
        self.assertIsNotNone(model.energy_gain_decoder[-1].weight.grad)

    def test_rejects_overlapping_modes(self):
        model = self.small_model(modes_y=6, padding=0)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            model(torch.randn(1, 20, 8, 12, 3))


if __name__ == "__main__":
    unittest.main()
