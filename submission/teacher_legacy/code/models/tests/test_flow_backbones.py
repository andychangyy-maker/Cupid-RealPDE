from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.dit2d import DiT2d, DiT2dConfig
from models.unet2d import UNet2d, UNet2dConfig, build_history_features


class FlowBackboneTests(unittest.TestCase):
    def test_acceleration_features_are_recent_second_differences(self):
        x = torch.zeros(1, 6, 2, 3, 3)
        time = torch.arange(6, dtype=torch.float32)
        x[..., 0] = time.square().view(1, 6, 1, 1)
        x[..., 1] = (3.0 * time.square()).view(1, 6, 1, 1)
        value = build_history_features(
            x, difference_steps=0, use_coordinates=False,
            acceleration_steps=2)
        self.assertEqual(tuple(value.shape), (1, 22, 2, 3))
        expected = torch.tensor([2.0, 6.0, 2.0, 6.0]).view(1, 4, 1, 1)
        torch.testing.assert_close(
            value[:, -4:], expected.expand(-1, -1, 2, 3))

    def test_acceleration_feature_config_roundtrip(self):
        model = UNet2d(UNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            acceleration_steps=4))
        config = model.architecture_config()
        config.pop("architecture")
        rebuilt = UNet2d(UNet2dConfig(**config))
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def models(self):
        yield UNet2d(UNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2))
        yield DiT2d(DiT2dConfig(
            image_height=16, image_width=32, patch_size=4,
            dimension=64, depth=2, heads=4, difference_steps=2))

    @staticmethod
    def set_stats(model):
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))

    def test_distribution_shapes_and_order(self):
        x = torch.randn(2, 20, 16, 32, 3)
        for model in self.models():
            self.set_stats(model)
            prediction, lower, upper = model.forward_distribution(x)
            self.assertEqual(tuple(prediction.shape), (2, 20, 16, 32, 3))
            self.assertEqual(tuple(lower.shape), (2, 20, 16, 32, 2))
            self.assertEqual(tuple(upper.shape), (2, 20, 16, 32, 2))
            self.assertTrue(torch.all(lower > 0))
            self.assertTrue(torch.all(upper > 0))
            self.assertTrue(torch.isfinite(prediction).all())

    def test_zero_initialized_point_is_persistence(self):
        x = torch.randn(1, 20, 16, 32, 3)
        for model in self.models():
            self.set_stats(model)
            prediction = model(x)
            expected = x[:, -1:].expand(-1, 20, -1, -1, -1)
            torch.testing.assert_close(prediction, expected)

    def test_point_and_uncertainty_heads_receive_gradient(self):
        x = torch.randn(1, 20, 16, 32, 3)
        for model in self.models():
            self.set_stats(model)
            prediction, lower, upper = model.forward_distribution(x)
            loss = prediction.square().mean() + lower.mean() + upper.mean()
            loss.backward()
            if isinstance(model, UNet2d):
                point = model.point_heads[-1].weight.grad
                interval = model.uncertainty_heads[-1].weight.grad
            else:
                point = model.point_head.weight.grad
                interval = model.uncertainty_head.weight.grad
            self.assertIsNotNone(point)
            self.assertIsNotNone(interval)
            self.assertGreater(float(point.abs().sum()), 0.0)
            self.assertGreater(float(interval.abs().sum()), 0.0)

    def test_config_roundtrip(self):
        for model in self.models():
            config = model.architecture_config()
            architecture = config.pop("architecture")
            rebuilt = (
                UNet2d(UNet2dConfig(**config)) if architecture == "unet"
                else DiT2d(DiT2dConfig(**config)))
            rebuilt.load_state_dict(model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
