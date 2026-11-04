from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.uq_reallocator import UQBudgetReallocator, UQReallocatorConfig


class UQReallocatorTests(unittest.TestCase):
    @staticmethod
    def model(**overrides) -> UQBudgetReallocator:
        return UQBudgetReallocator(UQReallocatorConfig(
            hidden_channels=16, global_hidden=32, groups=4,
            template_height=8, template_width=12, **overrides))

    def inputs(self):
        torch.manual_seed(3)
        history = torch.randn(2, 20, 8, 12, 3)
        prediction = torch.randn(2, 20, 8, 12, 3)
        lower = torch.rand(2, 20, 8, 12, 2) * 0.02 + 0.002
        upper = torch.rand(2, 20, 8, 12, 2) * 0.02 + 0.002
        mask = torch.ones(2, 8, 12, dtype=torch.bool)
        mask[:, 2:4, 4:7] = False
        return history, prediction, lower, upper, mask

    def test_zero_initialized_model_is_exact_identity(self):
        model = self.model()
        history, prediction, lower, upper, mask = self.inputs()
        result_lower, result_upper = model(
            history, prediction, lower, upper, mask)
        torch.testing.assert_close(result_lower, lower)
        torch.testing.assert_close(result_upper, upper)

    def test_local_reallocation_preserves_masked_native_budget(self):
        model = self.model(use_global_budget=False)
        history, prediction, lower, upper, mask = self.inputs()
        with torch.no_grad():
            model.spatial_template.normal_(mean=0.0, std=0.5)
        result_lower, result_upper = model(
            history, prediction, lower, upper, mask)
        native = torch.cat((lower, upper), dim=-1)
        result = torch.cat((result_lower, result_upper), dim=-1)
        expanded = mask[:, None, :, :, None]
        native_sum = (native * expanded).sum(dim=(2, 3))
        result_sum = (result * expanded).sum(dim=(2, 3))
        torch.testing.assert_close(result_sum, native_sum, rtol=1e-5, atol=1e-6)

    def test_all_branches_receive_gradient(self):
        model = self.model()
        history, prediction, lower, upper, mask = self.inputs()
        result_lower, result_upper = model(
            history, prediction, lower, upper, mask)
        loss = result_lower.square().mean() + result_upper.mean()
        loss.backward()
        self.assertGreater(float(model.local_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.spatial_template.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.global_budget[-1].weight.grad.abs().sum()), 0.0)

    def test_regime_film_is_identity_initialized_and_trainable(self):
        model = self.model(
            use_static_template=False, use_global_budget=False,
            use_regime_film=True)
        history, prediction, lower, upper, mask = self.inputs()
        identity_lower, identity_upper = model(
            history, prediction, lower, upper, mask)
        torch.testing.assert_close(identity_lower, lower)
        torch.testing.assert_close(identity_upper, upper)
        with torch.no_grad():
            model.local_head.weight.normal_(mean=0.0, std=0.01)
        result_lower, result_upper = model(
            history, prediction, lower, upper, mask)
        (result_lower.square().mean() + result_upper.mean()).backward()
        self.assertGreater(float(model.regime_film[-1].weight.grad.abs().sum()), 0.0)

    def test_auxiliary_features_are_validated_and_identity_initialized(self):
        model = self.model(auxiliary_channels=3)
        history, prediction, lower, upper, mask = self.inputs()
        auxiliary = torch.randn(2, 3, 8, 12)
        result_lower, result_upper = model(
            history, prediction, lower, upper, mask, auxiliary=auxiliary)
        torch.testing.assert_close(result_lower, lower)
        torch.testing.assert_close(result_upper, upper)
        with self.assertRaisesRegex(ValueError, "expected"):
            model(history, prediction, lower, upper, mask)

    def test_config_roundtrip(self):
        model = self.model()
        config = model.architecture_config()
        self.assertEqual(config.pop("architecture"), "uq_budget_reallocator")
        rebuilt = UQBudgetReallocator(UQReallocatorConfig(**config))
        rebuilt.load_state_dict(model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
