from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch.nn import functional as F


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.ensemble_unet2d import (
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
    MatchedLateAxisMixer,
    RegimeGatedConservativeClosure,
    conservative_history_features,
    hard_region_conditioning,
)
from models.unet2d import ResidualConvBlock
from train_ensemble_unet2d import (
    common,
    mean_profile_objective,
    soft_native_sps_objective,
    trajectory_member_mask,
)


class EnsembleUNet2dTests(unittest.TestCase):
    @staticmethod
    def model() -> EnsembleUNet2d:
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            disagreement_scale_fraction=0.01))
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        return model

    def test_shapes_monotonicity_and_positive_intervals(self):
        model = self.model()
        x = torch.randn(2, 20, 16, 32, 3)
        prediction, members, quantiles, lower, upper = model.forward_quantiles(x)
        self.assertEqual(tuple(prediction.shape), (2, 20, 16, 32, 3))
        self.assertEqual(tuple(members.shape), (2, 3, 20, 16, 32, 3))
        self.assertEqual(tuple(quantiles.shape), (2, 20, 16, 32, 2, 7))
        self.assertEqual(tuple(lower.shape), (2, 20, 16, 32, 2))
        self.assertEqual(tuple(upper.shape), (2, 20, 16, 32, 2))
        self.assertTrue(torch.all(quantiles.diff(dim=-1) > 0.0))
        self.assertTrue(torch.all(lower > 0.0))
        self.assertTrue(torch.all(upper > 0.0))

    def test_member_feature_adapter_readout_fusion_preserves_inference(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            member_adapter_channels=4)).eval()
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        with torch.no_grad():
            for heads in model.member_point_heads:
                for head in heads:
                    head.weight.normal_(std=0.02)
                    head.bias.normal_(std=0.02)
            for adapters in model.member_feature_adapters:
                for adapter in adapters:
                    adapter.body[-1].weight.normal_(std=0.02)
                    adapter.body[-1].bias.normal_(std=0.02)
        x = torch.randn(2, 20, 16, 32, 3)
        before = model.forward_distribution(x)

        self.assertTrue(model.fuse_member_feature_adapters_for_inference())
        self.assertIsNone(model.member_feature_adapters)
        self.assertIsNotNone(model.fused_member_feature_adapter_readouts)
        self.assertFalse(model.fuse_member_feature_adapters_for_inference())
        after = model.forward_distribution(x)

        for observed, expected in zip(after, before):
            torch.testing.assert_close(
                observed, expected, rtol=2e-5, atol=2e-6)

    def test_member_feature_adapter_fusion_is_noop_when_disabled(self):
        model = self.model()
        self.assertFalse(model.fuse_member_feature_adapters_for_inference())

    def test_late_axis_pair_has_exact_state_compute_and_causality(self):
        torch.manual_seed(19)
        causal = MatchedLateAxisMixer(4, 5, 2, "causal_temporal")
        torch.manual_seed(19)
        spatial = MatchedLateAxisMixer(
            4, 5, 2, "framewise_spatial_control")
        self.assertEqual(causal.state_dict().keys(), spatial.state_dict().keys())
        for name, value in causal.state_dict().items():
            torch.testing.assert_close(
                value, spatial.state_dict()[name], rtol=0.0, atol=0.0)
        self.assertEqual(causal.left_pad, spatial.left_pad)
        self.assertEqual(causal.right_pad, spatial.right_pad)
        self.assertEqual(causal.kernel, spatial.kernel)

        trajectory = torch.randn(2, 20, 4, 8, 16)
        changed_future = trajectory.clone()
        changed_future[:, 10:] += torch.randn_like(
            changed_future[:, 10:]) * 10.0
        torch.testing.assert_close(
            causal(trajectory)[:, :10], causal(changed_future)[:, :10],
            rtol=0.0, atol=0.0)

        changed_frames = trajectory.clone()
        changed_frames[:, 1:] += torch.randn_like(
            changed_frames[:, 1:]) * 10.0
        torch.testing.assert_close(
            spatial(trajectory)[:, :1], spatial(changed_frames)[:, :1],
            rtol=0.0, atol=0.0)

    def test_late_multiscale_sidecar_is_paired_identity_and_roundtrips(self):
        common = dict(
            base=8, depth=4, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            late_multiscale_sidecar_channels=(4, 8),
            late_multiscale_sidecar_blocks=1,
            late_multiscale_sidecar_kernel=5)
        torch.manual_seed(211)
        causal = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common, late_multiscale_sidecar_mode="causal_temporal"))
        torch.manual_seed(211)
        control = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common,
            late_multiscale_sidecar_mode="framewise_spatial_control"))
        self.assertEqual(causal.state_dict().keys(), control.state_dict().keys())
        for name, value in causal.state_dict().items():
            torch.testing.assert_close(
                value, control.state_dict()[name], rtol=0.0, atol=0.0)
        self.assertEqual(
            sum(value.numel() for value in causal.parameters()),
            sum(value.numel() for value in control.parameters()))

        torch.manual_seed(211)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**{
            key: value for key, value in common.items()
            if not key.startswith("late_multiscale_sidecar")
        }))
        for model in (baseline, causal, control):
            model.set_normalizer_statistics(
                torch.zeros(3), torch.zeros(3),
                torch.ones(3), torch.ones(3))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(
                value, causal.state_dict()[name], rtol=0.0, atol=0.0)
        x = torch.randn(2, 20, 16, 32, 3)
        torch.testing.assert_close(
            causal(x), baseline(x), rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            control(x), baseline(x), rtol=0.0, atol=0.0)
        for candidate_value, baseline_value in zip(
                causal.forward_quantiles(x), baseline.forward_quantiles(x)):
            torch.testing.assert_close(
                candidate_value, baseline_value, rtol=0.0, atol=0.0)

        stored = causal.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(causal.state_dict(), strict=True)

    def test_late_multiscale_sidecar_learns_beyond_zero_fusion(self):
        config = EnsembleUNet2dConfig(
            base=8, depth=4, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            late_multiscale_sidecar_mode="causal_temporal",
            late_multiscale_sidecar_channels=(4, 8))
        model = EnsembleUNet2d(config)
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        with torch.no_grad():
            for heads in model.member_point_heads:
                for head in heads:
                    head.weight.normal_(std=0.02)
                    head.bias.normal_(std=0.02)
        sidecar = model.late_multiscale_sidecar
        self.assertIsNotNone(sidecar)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
        x = torch.randn(2, 20, 16, 32, 3)
        target = torch.randn_like(x)

        optimizer.zero_grad(set_to_none=True)
        (model(x) - target).square().mean().backward()
        fusion_gradient = sum(
            float(projection.weight.grad.abs().sum())
            for projection in (sidecar.half_fusion, sidecar.quarter_fusion))
        upstream_before = float(sidecar.half_input.weight.grad.abs().sum())
        self.assertGreater(fusion_gradient, 0.0)
        self.assertEqual(upstream_before, 0.0)
        optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        (model(x) - target).square().mean().backward()
        upstream_after = float(sidecar.half_input.weight.grad.abs().sum())
        future_gradient = float(
            sidecar.observed_to_future_logits.grad.abs().sum())
        axis_gradient = float(
            sidecar.half_axis[0].depthwise.weight.grad.abs().sum())
        observed_time_gradient = float(
            sidecar.observed_time_half.grad.abs().sum())
        horizon_gradient = float(sidecar.horizon_half.grad.abs().sum())
        self.assertGreater(upstream_after, 0.0)
        self.assertGreater(future_gradient, 0.0)
        self.assertGreater(axis_gradient, 0.0)
        self.assertGreater(observed_time_gradient, 0.0)
        self.assertGreater(horizon_gradient, 0.0)

    def test_late_multiscale_sidecar_axis_is_the_only_functional_difference(self):
        common = dict(
            base=8, depth=4, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            late_multiscale_sidecar_channels=(4, 8))
        torch.manual_seed(31)
        causal = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common, late_multiscale_sidecar_mode="causal_temporal"))
        torch.manual_seed(31)
        control = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common,
            late_multiscale_sidecar_mode="framewise_spatial_control"))
        control.load_state_dict(causal.state_dict(), strict=True)
        for model in (causal, control):
            model.set_normalizer_statistics(
                torch.zeros(3), torch.zeros(3),
                torch.ones(3), torch.ones(3))
            with torch.no_grad():
                for heads in model.member_point_heads:
                    for head in heads:
                        head.weight.normal_(std=0.02)
                    # The sidecar must be observable for this semantic test.
                model.late_multiscale_sidecar.half_fusion.weight.normal_(
                    std=0.02)
                model.late_multiscale_sidecar.quarter_fusion.weight.normal_(
                    std=0.02)
        # Restore all weights after the identical mutations; only the mode is
        # intentionally outside state_dict.
        control.load_state_dict(causal.state_dict(), strict=True)
        x = torch.randn(2, 20, 16, 32, 3)
        self.assertFalse(torch.equal(causal(x), control(x)))

    def test_fluctuation_role_tower_is_exact_identity_and_role_isolated(self):
        common = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            mean_fluctuation=True, member_adapter_channels=4)
        torch.manual_seed(73)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common))
        torch.manual_seed(73)
        separated = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common, fluctuation_role_channels=4))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(
                value, separated.state_dict()[name], rtol=0.0, atol=0.0)
        for model in (baseline, separated):
            model.set_normalizer_statistics(
                torch.zeros(3), torch.zeros(3),
                torch.ones(3), torch.ones(3))
            with torch.no_grad():
                for heads in model.member_point_heads:
                    for head in heads:
                        head.weight.normal_(std=0.02)
                        head.bias.normal_(std=0.02)
        # Restore identical point readouts after the independent mutation
        # above, then verify the zero role projections are an exact no-op.
        separated.member_point_heads.load_state_dict(
            baseline.member_point_heads.state_dict())
        x = torch.randn(2, 20, 16, 32, 3)
        torch.testing.assert_close(
            separated(x), baseline(x), rtol=0.0, atol=0.0)
        self.assertFalse(separated.fuse_member_feature_adapters_for_inference())

        prediction = separated(x)
        temporal_mean = prediction.mean(dim=1)
        temporal_mean.square().mean().backward()
        self.assertLess(sum(
            0.0 if parameter.grad is None else float(parameter.grad.abs().sum())
            for parameter in separated.fluctuation_role_adapters.parameters()),
            1.0e-8)

        separated.zero_grad(set_to_none=True)
        prediction = separated(x)
        fluctuation = prediction - prediction.mean(dim=1, keepdim=True)
        fluctuation.square().mean().backward()
        output_gradient = sum(
            float(adapter.body[-1].weight.grad.abs().sum())
            for adapter in separated.fluctuation_role_adapters)
        self.assertGreater(output_gradient, 0.0)

        optimizer = torch.optim.SGD(
            separated.fluctuation_role_adapters.parameters(), lr=1.0e-3)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        prediction = separated(x)
        fluctuation = prediction - prediction.mean(dim=1, keepdim=True)
        fluctuation.square().mean().backward()
        body_gradient = sum(
            float(adapter.body[0].weight.grad.abs().sum())
            for adapter in separated.fluctuation_role_adapters)
        self.assertGreater(body_gradient, 0.0)

        stored = separated.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(separated.state_dict(), strict=True)

    def test_fluctuation_role_tower_requires_mean_fluctuation(self):
        with self.assertRaisesRegex(ValueError, "mean_fluctuation"):
            EnsembleUNet2dConfig(
                base=8, depth=3, groups=4,
                mean_fluctuation=False,
                fluctuation_role_channels=4).validate()

    def test_dual_decoder_has_equal_capacity_control_and_role_gradients(self):
        common = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            mean_fluctuation=True, member_adapter_channels=4,
            cross_conditioned_backbone=True,
            cross_conditioned_channels=4)
        torch.manual_seed(117)
        split = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common, dual_decoder_mode="mean_fluctuation")).eval()
        torch.manual_seed(117)
        fused = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common, dual_decoder_mode="shared_fused")).eval()
        self.assertEqual(
            sum(value.numel() for value in split.parameters()),
            sum(value.numel() for value in fused.parameters()))
        self.assertEqual(split.state_dict().keys(), fused.state_dict().keys())
        for name, value in split.state_dict().items():
            torch.testing.assert_close(
                value, fused.state_dict()[name], rtol=0.0, atol=0.0)

        for model in (split, fused):
            model.set_normalizer_statistics(
                torch.zeros(3), torch.zeros(3),
                torch.ones(3), torch.ones(3))
            with torch.no_grad():
                for heads in model.member_point_heads:
                    for head in heads:
                        head.weight.normal_(std=0.02)
                        head.bias.normal_(std=0.02)
                for adapters in model.member_feature_adapters:
                    for adapter in adapters:
                        adapter.body[-1].weight.normal_(std=0.02)
                        adapter.body[-1].bias.normal_(std=0.02)
                model.quantile_head.weight.normal_(std=0.02)
        # Restore an exact common parameter state after the mutations above;
        # routing is the only candidate/control difference.
        fused.load_state_dict(split.state_dict(), strict=True)
        x = torch.randn(2, 20, 16, 32, 3)

        prediction = split(x)
        prediction.mean(dim=1).square().mean().backward()
        mean_to_dynamics = sum(
            0.0 if parameter.grad is None
            else float(parameter.grad.abs().sum())
            for parameter in split.fluctuation_decoder.parameters())
        self.assertLess(mean_to_dynamics, 1.0e-8)

        split.zero_grad(set_to_none=True)
        prediction = split(x)
        fluctuation = prediction - prediction.mean(dim=1, keepdim=True)
        fluctuation.square().mean().backward()
        fluctuation_to_dynamics = sum(
            0.0 if parameter.grad is None
            else float(parameter.grad.abs().sum())
            for parameter in split.fluctuation_decoder.parameters())
        fluctuation_to_members = sum(
            0.0 if parameter.grad is None
            else float(parameter.grad.abs().sum())
            for parameter in split.member_feature_adapters.parameters())
        self.assertGreater(fluctuation_to_dynamics, 0.0)
        self.assertGreater(fluctuation_to_members, 0.0)

        split.zero_grad(set_to_none=True)
        _prediction, lower, upper = split.forward_distribution(x)
        (lower.mean() + upper.mean()).backward()
        uq_to_semantic_dynamics = sum(
            0.0 if parameter.grad is None
            else float(parameter.grad.abs().sum())
            for parameter in split.fluctuation_uq_projections.parameters())
        self.assertGreater(uq_to_semantic_dynamics, 0.0)

        fused.zero_grad(set_to_none=True)
        fused(x).mean(dim=1).square().mean().backward()
        fused_mean_to_second_decoder = sum(
            0.0 if parameter.grad is None
            else float(parameter.grad.abs().sum())
            for parameter in fused.fluctuation_decoder.parameters())
        self.assertGreater(fused_mean_to_second_decoder, 0.0)
        self.assertFalse(split.fuse_member_feature_adapters_for_inference())

        stored = split.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(split.state_dict(), strict=True)

    def test_dual_decoder_rejects_conflicting_point_paths(self):
        with self.assertRaisesRegex(ValueError, "mean_fluctuation"):
            EnsembleUNet2dConfig(
                base=8, depth=3, groups=4, mean_fluctuation=False,
                dual_decoder_mode="mean_fluctuation").validate()
        with self.assertRaisesRegex(ValueError, "unspecialized"):
            EnsembleUNet2dConfig(
                base=8, depth=3, groups=4,
                dual_decoder_mode="mean_fluctuation",
                fluctuation_role_channels=4).validate()

    def test_asymmetric_dual_decoder_fuses_mean_but_splits_fluctuation(self):
        config = EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            mean_fluctuation=True,
            dual_decoder_mode="fused_mean_fluctuation")
        model = EnsembleUNet2d(config).eval()
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        with torch.no_grad():
            for heads in model.member_point_heads:
                for head in heads:
                    head.weight.normal_(std=0.02)
                    head.bias.normal_(std=0.02)
        x = torch.randn(2, 20, 16, 32, 3)

        prediction = model(x)
        prediction.mean(dim=1).square().mean().backward()
        mean_gradient = sum(
            0.0 if parameter.grad is None
            else float(parameter.grad.abs().sum())
            for parameter in model.fluctuation_decoder.parameters())
        self.assertGreater(mean_gradient, 0.0)

        model.zero_grad(set_to_none=True)
        prediction = model(x)
        fluctuation = prediction - prediction.mean(dim=1, keepdim=True)
        fluctuation.square().mean().backward()
        fluctuation_gradient = sum(
            0.0 if parameter.grad is None
            else float(parameter.grad.abs().sum())
            for parameter in model.fluctuation_decoder.parameters())
        main_fluctuation_gradient = sum(
            0.0 if parameter.grad is None
            else float(parameter.grad.abs().sum())
            for parameter in model.decoder.parameters())
        self.assertGreater(fluctuation_gradient, 0.0)
        self.assertLess(main_fluctuation_gradient, 1.0e-8)

    def test_learned_fluctuation_gate_starts_at_fused_and_gets_gradient(self):
        common = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            mean_fluctuation=True)
        torch.manual_seed(131)
        shared = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common, dual_decoder_mode="shared_fused")).eval()
        torch.manual_seed(131)
        gated = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common, dual_decoder_mode="learned_fluctuation_gate")).eval()
        missing, unexpected = gated.load_state_dict(
            shared.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertEqual(
            set(missing),
            {f"fluctuation_role_gates.{index}"
             for index in range(common["depth"] - 1)})
        with torch.no_grad():
            for heads in gated.member_point_heads:
                for head in heads:
                    head.weight.normal_(std=0.02)
                    head.bias.normal_(std=0.02)
            gated.quantile_head.weight.normal_(std=0.02)
        shared.load_state_dict({
            name: value for name, value in gated.state_dict().items()
            if not name.startswith("fluctuation_role_gates.")
        }, strict=True)
        for model in (shared, gated):
            model.set_normalizer_statistics(
                torch.zeros(3), torch.zeros(3),
                torch.ones(3), torch.ones(3))
        x = torch.randn(2, 20, 16, 32, 3)
        torch.testing.assert_close(
            gated(x), shared(x), rtol=0.0, atol=0.0)
        shared_distribution = shared.forward_distribution(x)
        gated_distribution = gated.forward_distribution(x)
        for gated_value, shared_value in zip(
                gated_distribution, shared_distribution):
            torch.testing.assert_close(
                gated_value, shared_value, rtol=0.0, atol=0.0)

        prediction = gated(x)
        fluctuation = prediction - prediction.mean(dim=1, keepdim=True)
        fluctuation.square().mean().backward()
        gate_gradient = sum(
            0.0 if gate.grad is None else float(gate.grad.abs().sum())
            for gate in gated.fluctuation_role_gates)
        self.assertGreater(gate_gradient, 0.0)
        self.assertEqual(
            sum(gate.numel() for gate in gated.fluctuation_role_gates),
            sum(8 * 2 ** level for level in range(common["depth"] - 1)))

        stored = gated.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(gated.state_dict(), strict=True)

    def test_separate_point_uq_gates_are_identity_and_role_isolated(self):
        common = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            mean_fluctuation=True)
        torch.manual_seed(137)
        shared = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common, dual_decoder_mode="shared_fused")).eval()
        torch.manual_seed(137)
        separated = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common,
            dual_decoder_mode="separate_point_uq_fluctuation_gates")).eval()
        missing, unexpected = separated.load_state_dict(
            shared.state_dict(), strict=False)
        self.assertFalse(unexpected)
        expected_missing = {
            f"{prefix}.{index}"
            for prefix in (
                "fluctuation_role_gates", "uq_fluctuation_role_gates")
            for index in range(common["depth"] - 1)}
        self.assertEqual(set(missing), expected_missing)
        with torch.no_grad():
            for heads in separated.member_point_heads:
                for head in heads:
                    head.weight.normal_(std=0.02)
                    head.bias.normal_(std=0.02)
            separated.quantile_head.weight.normal_(std=0.02)
        shared.load_state_dict({
            name: value for name, value in separated.state_dict().items()
            if not (name.startswith("fluctuation_role_gates.")
                    or name.startswith("uq_fluctuation_role_gates."))
        }, strict=True)
        for model in (shared, separated):
            model.set_normalizer_statistics(
                torch.zeros(3), torch.zeros(3),
                torch.ones(3), torch.ones(3))
        x = torch.randn(2, 20, 16, 32, 3)
        shared_values = shared.forward_distribution(x)
        separated_values = separated.forward_distribution(x)
        for candidate, control in zip(separated_values, shared_values):
            torch.testing.assert_close(
                candidate, control, rtol=0.0, atol=0.0)

        prediction = separated(x)
        fluctuation = prediction - prediction.mean(dim=1, keepdim=True)
        fluctuation.square().mean().backward()
        point_gate_gradient = sum(
            0.0 if gate.grad is None else float(gate.grad.abs().sum())
            for gate in separated.fluctuation_role_gates)
        uq_gate_gradient = sum(
            0.0 if gate.grad is None else float(gate.grad.abs().sum())
            for gate in separated.uq_fluctuation_role_gates)
        self.assertGreater(point_gate_gradient, 0.0)
        self.assertEqual(uq_gate_gradient, 0.0)

        separated.zero_grad(set_to_none=True)
        _prediction, lower, upper = separated.forward_distribution(x)
        (lower.mean() + upper.mean()).backward()
        uq_gate_gradient = sum(
            0.0 if gate.grad is None else float(gate.grad.abs().sum())
            for gate in separated.uq_fluctuation_role_gates)
        self.assertGreater(uq_gate_gradient, 0.0)

        point_before = separated(x).detach()
        with torch.no_grad():
            for gate in separated.uq_fluctuation_role_gates:
                gate.fill_(0.5)
        torch.testing.assert_close(
            separated(x), point_before, rtol=0.0, atol=0.0)

        stored = separated.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(separated.state_dict(), strict=True)

    def test_laplacian_point_pyramid_is_identity_at_update_zero_and_trainable(self):
        common_config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        torch.manual_seed(37)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common_config))
        torch.manual_seed(37)
        pyramid = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, laplacian_point_pyramid=True))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(
                value, pyramid.state_dict()[name], rtol=0.0, atol=0.0)
        for model in (baseline, pyramid):
            model.set_normalizer_statistics(
                torch.zeros(3), torch.zeros(3),
                torch.ones(3), torch.ones(3))
        x = torch.randn(2, 20, 16, 32, 3)
        torch.testing.assert_close(
            pyramid(x), baseline(x), rtol=0.0, atol=0.0)

        with torch.no_grad():
            for heads in pyramid.member_point_heads:
                for head in heads:
                    head.weight.normal_(std=0.02)
                    head.bias.normal_(std=0.02)
        output = pyramid(x)
        self.assertFalse(torch.equal(output, baseline(x)))
        output.square().mean().backward()
        for heads in pyramid.member_point_heads:
            for head in heads:
                self.assertIsNotNone(head.weight.grad)
                self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)

        stored = pyramid.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(pyramid.state_dict(), strict=True)

    def test_laplacian_point_pyramid_removes_redundant_low_frequencies(self):
        features = [
            torch.full((1, 1, 2, 4), 3.0),
            torch.full((1, 1, 4, 8), 5.0),
            torch.full((1, 1, 8, 16), 7.0),
        ]
        heads = torch.nn.ModuleList([torch.nn.Identity() for _ in features])
        fused = EnsembleUNet2d._laplacian_multi_scale(
            features, heads, (8, 16))
        # Spatial constants in the middle/fine heads lie entirely in the
        # preceding low-pass space, so only the coarse head survives.
        torch.testing.assert_close(
            fused, torch.ones_like(fused), rtol=0.0, atol=0.0)

    def test_low_rank_point_readout_adapters_are_zero_identity_and_trainable(self):
        common_config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            laplacian_point_pyramid=True)
        torch.manual_seed(41)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common_config))
        torch.manual_seed(41)
        adapted = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, point_readout_adapter_rank=4))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(
                value, adapted.state_dict()[name], rtol=0.0, atol=0.0)
        for model in (baseline, adapted):
            model.set_normalizer_statistics(
                torch.zeros(3), torch.zeros(3),
                torch.ones(3), torch.ones(3))
        x = torch.randn(2, 20, 16, 32, 3)
        torch.testing.assert_close(
            adapted(x), baseline(x), rtol=0.0, atol=0.0)

        target = torch.randn_like(adapted(x))
        (adapted(x) - target).square().mean().backward()
        for member in adapted.member_point_head_adapters:
            for adapter in member:
                self.assertGreater(float(adapter.up.weight.grad.abs().sum()), 0)
                self.assertEqual(float(adapter.down.weight.grad.abs().sum()), 0)
        optimizer = torch.optim.SGD(
            adapted.member_point_head_adapters.parameters(), lr=1.0e-3)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        (adapted(x) - target).square().mean().backward()
        self.assertGreater(sum(
            float(adapter.down.weight.grad.abs().sum())
            for member in adapted.member_point_head_adapters
            for adapter in member), 0.0)

        stored = adapted.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(adapted.state_dict(), strict=True)

        with self.assertRaisesRegex(ValueError, "require legacy"):
            EnsembleUNet2dConfig(
                base=8, depth=3, groups=4,
                temporal_point_decoder=True,
                point_readout_adapter_rank=4).validate()

    def test_phase_delta_is_exact_zero_at_initialization_and_trainable(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            acceleration_steps=2, ensemble_members=3,
            uq_hidden_channels=8, phase_delta_hidden_channels=8,
            phase_delta_steps=3))
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        x = torch.randn(2, 20, 16, 32, 3)
        delta = model.phase_delta(x)
        self.assertEqual(tuple(delta.shape), (2, 20, 16, 32, 2))
        self.assertEqual(int(torch.count_nonzero(delta)), 0)
        prediction = model(x)
        expected = x[:, -1:].expand(-1, 20, -1, -1, -1)
        torch.testing.assert_close(prediction, expected, rtol=0.0, atol=3e-7)
        delta.sum().backward()
        self.assertGreater(
            float(model.phase_delta.output.weight.grad.abs().sum()), 0.0)

    def test_phase_delta_config_roundtrip(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            acceleration_steps=2, ensemble_members=3,
            uq_hidden_channels=8, phase_delta_hidden_channels=8,
            phase_delta_steps=3))
        config = model.architecture_config()
        config.pop("architecture")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**config))
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def test_hard_region_expert_is_exact_identity_and_trainable(self):
        common_config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        torch.manual_seed(11)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common_config))
        torch.manual_seed(11)
        expert = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, hard_region_expert_channels=8,
            hard_region_expert_blocks=1))
        baseline_state = baseline.state_dict()
        expert_state = expert.state_dict()
        for name, value in baseline_state.items():
            torch.testing.assert_close(
                expert_state[name], value, rtol=0.0, atol=0.0)
        expert.load_state_dict(baseline_state, strict=False)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        expert.set_normalizer_statistics(*statistics)
        baseline.eval()
        expert.eval()
        x = torch.randn(2, 20, 16, 32, 3)
        expected = baseline(x)
        actual = expert(x)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        loss = (actual - torch.randn_like(actual)).square().mean()
        loss.backward()
        self.assertGreater(
            float(expert.hard_region_expert.output.weight.grad.abs().sum()),
            0.0)
        self.assertEqual(
            int(torch.count_nonzero(
                expert.hard_region_expert.body[0].weight.grad)), 0)

        stored = expert.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(expert.state_dict(), strict=True)

    def test_hard_region_gate_is_soft_and_suppresses_inlet(self):
        history = torch.randn(3, 20, 16, 32, 3)
        maps, gate = hard_region_conditioning(history)
        self.assertEqual(tuple(maps.shape), (3, 6, 16, 32))
        self.assertEqual(tuple(gate.shape), (3, 1, 16, 32))
        self.assertTrue(bool(torch.isfinite(maps).all()))
        self.assertTrue(bool(torch.isfinite(gate).all()))
        self.assertTrue(bool(((gate >= 0.0) & (gate <= 1.0)).all()))
        self.assertLess(
            float(gate[..., :4].mean()),
            float(gate[:, :, 4:12, 20:].mean()))

    def test_conservative_closure_is_identity_routed_and_trainable(self):
        common_config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        torch.manual_seed(23)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common_config))
        torch.manual_seed(23)
        candidate = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, conservative_closure_experts=3,
            conservative_closure_channels=8,
            conservative_closure_blocks=1))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(
                candidate.state_dict()[name], value, rtol=0.0, atol=0.0)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        candidate.set_normalizer_statistics(*statistics)
        baseline.eval()
        candidate.eval()
        history = torch.randn(2, 20, 16, 32, 3)
        expected = baseline(history)
        actual = candidate(history)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        weights = candidate.conservative_closure.route(history)
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(weights.shape[0]),
            rtol=1.0e-6, atol=1.0e-6)
        self.assertTrue(bool((weights > 0.0).all()))
        self.assertGreater(float((weights[0] - weights[1]).abs().max()), 0.0)

        (actual - torch.randn_like(actual)).square().mean().backward()
        closure = candidate.conservative_closure
        self.assertGreater(
            float(closure.flux_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(closure.inflow_head.weight.grad.abs().sum()), 0.0)
        self.assertEqual(
            int(torch.count_nonzero(closure.body[0].weight.grad)), 0)
        self.assertEqual(
            int(torch.count_nonzero(closure.router[-1].weight.grad)), 0)

        stored = candidate.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(candidate.state_dict(), strict=True)

    def test_conservative_closure_features_and_flux_telescope(self):
        history = torch.randn(2, 20, 8, 12, 3)
        features = conservative_history_features(history)
        self.assertEqual(tuple(features.shape), (2, 20, 8, 12))
        self.assertTrue(bool(torch.isfinite(features).all()))

        flux = torch.randn(2, 5, 4, 8, 12)
        inflow = torch.randn_like(flux)
        residual = RegimeGatedConservativeClosure.flux_divergence(
            flux, inflow)
        self.assertEqual(tuple(residual.shape), (2, 5, 2, 8, 12))
        expected_sum = -(
            flux[:, :, :2, :, -1].sum(dim=-1)
            - inflow[:, :, :2, :, 0].sum(dim=-1)
            + flux[:, :, 2:, -1, :].sum(dim=-1)
            - inflow[:, :, 2:, 0, :].sum(dim=-1))
        torch.testing.assert_close(
            residual.sum(dim=(-2, -1)), expected_sum,
            rtol=1.0e-5, atol=1.0e-5)

    def test_recurrent_latent_rollout_is_identity_bounded_and_trainable(self):
        common_config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        torch.manual_seed(29)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common_config))
        torch.manual_seed(29)
        rollout = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, recurrent_latent_rollout_channels=8,
            recurrent_latent_rollout_limit=0.2,
            recurrent_latent_rollout_downsample=4))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(
                rollout.state_dict()[name], value, rtol=0.0, atol=0.0)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        rollout.set_normalizer_statistics(*statistics)
        baseline.eval()
        rollout.eval()
        x = torch.randn(1, 20, 16, 32, 3)
        baseline_values = baseline.forward_quantiles(x)
        rollout_values = rollout.forward_quantiles(x)
        self.assertTrue(all(torch.equal(first, second) for first, second in zip(
            baseline_values, rollout_values)))

        rollout.train()
        prediction = rollout(x)
        target = torch.randn_like(prediction)
        (prediction - target).square().mean().backward()
        self.assertGreater(float(
            rollout.recurrent_latent_rollout.output.weight.grad.abs().sum()),
            0.0)
        body_parameters = tuple(
            parameter for name, parameter in
            rollout.recurrent_latent_rollout.named_parameters()
            if not name.startswith("output."))
        self.assertEqual(sum(
            int(torch.count_nonzero(parameter.grad))
            for parameter in body_parameters if parameter.grad is not None), 0)

        optimizer = torch.optim.SGD(
            rollout.recurrent_latent_rollout.parameters(), lr=1.0e-3)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        prediction = rollout(x)
        (prediction - target).square().mean().backward()
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in body_parameters if parameter.grad is not None),
            0.0)
        with torch.no_grad():
            features = rollout._decode(x)
            correction = rollout.recurrent_latent_rollout(features[-1])
        torch.testing.assert_close(
            correction.mean(dim=1), torch.zeros_like(correction[:, 0]),
            rtol=0.0, atol=2.0e-7)
        self.assertLessEqual(float(correction.abs().max()), 0.200001)

        stored = rollout.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(rollout.state_dict(), strict=True)

    def test_recurrent_latent_spectral_split_assigns_orthogonal_bands(self):
        config = EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            recurrent_latent_rollout_channels=8,
            recurrent_latent_rollout_limit=0.2,
            recurrent_latent_rollout_downsample=4,
            recurrent_latent_rollout_lowpass_modes=4)
        model = EnsembleUNet2d(config)
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        with torch.no_grad():
            for heads in model.member_point_heads:
                for head in heads:
                    head.weight.normal_(std=0.01)
        x = torch.randn(1, 20, 16, 32, 3)
        features = model._decode(x)
        raw_members = model._members_from_features(x, features)
        prediction, members, *_rest = model.forward_uq_training(x)
        flat_velocity = raw_members[..., :2].reshape(
            3, 20, 16, 32, 2)
        expected_velocity = model.recurrent_latent_rollout.temporal_lowpass(
            flat_velocity).reshape(1, 3, 20, 16, 32, 2)
        torch.testing.assert_close(
            members[..., :2], expected_velocity, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            members[..., 2:], raw_members[..., 2:], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            prediction, members.mean(dim=1), rtol=0.0, atol=0.0)

        with torch.no_grad():
            model.recurrent_latent_rollout.output.weight.normal_(std=0.1)
            correction = model.recurrent_latent_rollout(features[-1])
            leaked_lowpass = model.recurrent_latent_rollout.temporal_lowpass(
                correction)
        self.assertLess(float(leaked_lowpass.abs().max()), 2.0e-6)
        self.assertLessEqual(float(correction.abs().max()), 0.200001)
        torch.testing.assert_close(
            correction.mean(dim=1), torch.zeros_like(correction[:, 0]),
            rtol=0.0, atol=2.0e-6)

        with self.assertRaisesRegex(ValueError, "requires its rollout"):
            EnsembleUNet2dConfig(
                base=8, depth=3, groups=4,
                recurrent_latent_rollout_lowpass_modes=4).validate()

    def test_latent_ode_is_mandatory_persistence_then_trains_dynamics(self):
        config = EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            latent_ode_point_decoder=True, latent_ode_channels=8,
            latent_ode_solver_substeps=1, latent_ode_step_size=0.25,
            latent_ode_derivative_limit=1.0,
            latent_ode_residual_limit=0.3)
        model = EnsembleUNet2d(config)
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        self.assertEqual(len(model.member_point_heads), 0)
        self.assertIsNotNone(model.latent_ode_point_decoder)
        x = torch.randn(1, 20, 16, 32, 3)
        expected = x[:, -1].unsqueeze(1).expand(-1, 20, -1, -1, -1)
        point, members, *_rest = model.forward_uq_training(x)
        torch.testing.assert_close(point, expected, rtol=0.0, atol=3.0e-7)
        torch.testing.assert_close(
            members,
            expected.unsqueeze(1).expand(-1, 3, -1, -1, -1, -1),
            rtol=0.0, atol=0.0)

        with torch.no_grad():
            features = model._decode(x)
            trajectory = model.latent_ode_point_decoder.latent_trajectory(
                features)
        self.assertGreater(
            float((trajectory[:, 1:] - trajectory[:, :-1]).abs().sum()),
            0.0)

        target = torch.randn_like(point)
        (point - target).square().mean().backward()
        decoder = model.latent_ode_point_decoder
        self.assertGreater(sum(
            float(head.weight.grad.abs().sum())
            for head in decoder.member_heads), 0.0)
        self.assertEqual(sum(
            int(torch.count_nonzero(parameter.grad))
            for parameter in decoder.vector_field.parameters()
            if parameter.grad is not None), 0)

        optimizer = torch.optim.SGD(decoder.parameters(), lr=1.0e-3)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        point = model(x)
        (point - target).square().mean().backward()
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in decoder.vector_field.parameters()
            if parameter.grad is not None), 0.0)
        self.assertLessEqual(
            float((point - expected).abs().max()), 0.300001)

        stored = model.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def test_latent_ode_validation_rejects_conflicting_or_invalid_configs(self):
        with self.assertRaisesRegex(ValueError, "separate arms"):
            EnsembleUNet2dConfig(
                base=8, groups=4, temporal_point_decoder=True,
                latent_ode_point_decoder=True).validate()
        with self.assertRaisesRegex(ValueError, "divisible"):
            EnsembleUNet2dConfig(
                base=8, groups=4, latent_ode_point_decoder=True,
                latent_ode_channels=6).validate()
        with self.assertRaisesRegex(ValueError, "legacy multi-scale"):
            EnsembleUNet2dConfig(
                base=8, groups=4, latent_ode_point_decoder=True,
                point_readout_adapter_rank=2).validate()
        with self.assertRaisesRegex(ValueError, "pyramids/adapters"):
            EnsembleUNet2dConfig(
                base=8, groups=4, latent_ode_point_decoder=True,
                member_adapter_channels=4).validate()

    def test_latent_ude_band_decoder_centers_closure_and_trains_all_paths(self):
        config = EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            latent_ude_band_decoder=True, latent_ode_channels=8,
            latent_ode_solver_substeps=1, latent_ode_step_size=0.25,
            latent_ode_derivative_limit=1.0,
            latent_ude_closure_limit=0.1,
            latent_ode_residual_limit=0.3)
        model = EnsembleUNet2d(config)
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        self.assertEqual(len(model.member_point_heads), 0)
        decoder = model.latent_ude_band_point_decoder
        self.assertIsNotNone(decoder)
        x = torch.randn(1, 20, 16, 32, 3)
        expected = x[:, -1].unsqueeze(1).expand(-1, 20, -1, -1, -1)
        point, members, *_rest = model.forward_uq_training(x)
        torch.testing.assert_close(point, expected, rtol=0.0, atol=3.0e-7)
        torch.testing.assert_close(
            members,
            expected.unsqueeze(1).expand(-1, 3, -1, -1, -1, -1),
            rtol=0.0, atol=0.0)

        features = model._decode(x)
        context = decoder.encoder(features[0])
        state = context.unsqueeze(1).expand(-1, 3, -1, -1, -1).clone()
        _derivative, closure = decoder.vector_field.derivative(
            state, context, 0)
        torch.testing.assert_close(
            closure.mean(dim=1), torch.zeros_like(closure[:, 0]),
            rtol=0.0, atol=2.0e-7)
        self.assertGreater(
            float((closure[:, 0] - closure[:, 1]).abs().sum()), 0.0)

        native = torch.randn(2, 20, 3, 16, 32)
        high = decoder._spatial_highpass(native, (8, 16))
        pooled = F.interpolate(
            high.reshape(40, 3, 16, 32),
            size=(8, 16), mode="area")
        self.assertLess(float(pooled.abs().max()), 3.0e-7)

        target = torch.randn_like(point)
        (point - target).square().mean().backward()
        coarse_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in decoder.member_coarse_heads.parameters()
            if parameter.grad is not None)
        band_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in decoder.member_band_heads.parameters()
            if parameter.grad is not None)
        dynamics_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in decoder.vector_field.parameters()
            if parameter.grad is not None)
        self.assertGreater(coarse_gradient, 0.0)
        self.assertGreater(band_gradient, 0.0)
        self.assertEqual(dynamics_gradient, 0.0)

        optimizer = torch.optim.SGD(decoder.parameters(), lr=1.0e-3)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        point = model(x)
        (point - target).square().mean().backward()
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in decoder.vector_field.drift.parameters()
            if parameter.grad is not None), 0.0)
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in decoder.vector_field.closure.parameters()
            if parameter.grad is not None), 0.0)
        self.assertLessEqual(
            float(decoder.coarse_member_residual(
                decoder.latent_trajectory(features), 0).abs().max()),
            0.300001)

        stored = model.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def test_latent_ude_validation_rejects_conflicting_configs(self):
        with self.assertRaisesRegex(ValueError, "separate arms"):
            EnsembleUNet2dConfig(
                base=8, groups=4, latent_ode_point_decoder=True,
                latent_ude_band_decoder=True).validate()
        with self.assertRaisesRegex(ValueError, "owns its point bands"):
            EnsembleUNet2dConfig(
                base=8, groups=4, latent_ude_band_decoder=True,
                laplacian_point_pyramid=True).validate()
        with self.assertRaisesRegex(ValueError, "closure limit"):
            EnsembleUNet2dConfig(
                base=8, groups=4, latent_ude_closure_limit=0.0).validate()

    def test_latent_ude_coarse_residual_has_fixed_responsibilities(self):
        common = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        torch.manual_seed(31)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common))
        torch.manual_seed(31)
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common, latent_ude_coarse_residual=True,
            latent_ode_channels=8, latent_ode_solver_substeps=1,
            latent_ude_coarse_residual_limit=0.3))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(
                model.state_dict()[name], value, rtol=0.0, atol=0.0)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        baseline.set_normalizer_statistics(*statistics)
        model.set_normalizer_statistics(*statistics)
        x = torch.randn(1, 20, 16, 32, 3)
        torch.testing.assert_close(
            model(x), baseline(x), rtol=0.0, atol=0.0)

        module = model.latent_ude_coarse_residual
        self.assertIsNotNone(module)
        with torch.no_grad():
            module.member_heads[0].weight.normal_(std=0.01)
        features = model._decode(x)
        correction = module(features[0], (16, 32))
        torch.testing.assert_close(
            correction.mean(dim=2),
            torch.zeros_like(correction[:, :, 0]),
            rtol=0.0, atol=3.0e-7)
        torch.testing.assert_close(
            correction.mean(dim=(3, 4)),
            torch.zeros_like(correction[:, :, :, 0, 0]),
            rtol=0.0, atol=3.0e-7)

        target = torch.randn_like(model(x))
        (model(x) - target).square().mean().backward()
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in module.member_heads.parameters()
            if parameter.grad is not None), 0.0)
        stored = model.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def test_latent_ude_coarse_residual_rejects_decoder_conflict(self):
        with self.assertRaisesRegex(ValueError, "legacy direct path"):
            EnsembleUNet2dConfig(
                base=8, groups=4, latent_ude_coarse_residual=True,
                latent_ude_band_decoder=True).validate()
        with self.assertRaisesRegex(ValueError, "coarse residual limit"):
            EnsembleUNet2dConfig(
                base=8, groups=4,
                latent_ude_coarse_residual_limit=0.0).validate()
        with self.assertRaisesRegex(ValueError, "velocity-only"):
            EnsembleUNet2dConfig(
                base=8, groups=4,
                latent_ude_coarse_velocity_only=True).validate()

    def test_velocity_only_coarse_ude_never_changes_pressure(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            latent_ude_coarse_residual=True,
            latent_ude_coarse_velocity_only=True,
            latent_ode_channels=8))
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        module = model.latent_ude_coarse_residual
        self.assertEqual(module.residual_channels, 2)
        with torch.no_grad():
            for head in module.member_heads:
                head.weight.normal_(std=0.01)
        x = torch.randn(1, 20, 16, 32, 3)
        features = model._decode(x)
        correction = module(features[0], (16, 32))
        torch.testing.assert_close(
            correction[..., 2], torch.zeros_like(correction[..., 2]),
            rtol=0.0, atol=0.0)

    def test_exclusive_coarse_ude_owns_low_frequency_uv_fluctuation(self):
        config = EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            latent_ude_exclusive_coarse_fluctuation=True,
            latent_ude_coarse_velocity_only=True,
            latent_ode_channels=8)
        model = EnsembleUNet2d(config)
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        x = torch.randn(1, 20, 16, 32, 3)
        features = model._decode(x)
        direct_config = replace(
            config, latent_ude_exclusive_coarse_fluctuation=False,
            latent_ude_coarse_velocity_only=False)
        direct = EnsembleUNet2d(direct_config)
        direct.load_state_dict({
            name: value for name, value in model.state_dict().items()
            if not name.startswith("latent_ude_coarse_residual.")},
            strict=False)
        direct.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        direct_members = direct._members_from_features(x, features)
        members = model._members_from_features(x, features)
        torch.testing.assert_close(
            members[..., :2].mean(dim=2),
            direct_members[..., :2].mean(dim=2),
            rtol=0.0, atol=3.0e-7)
        torch.testing.assert_close(
            members[..., 2], direct_members[..., 2],
            rtol=0.0, atol=0.0)
        fluctuation = members[..., :2] - members[..., :2].mean(
            dim=2, keepdim=True)
        pooled = F.interpolate(
            fluctuation.permute(0, 1, 2, 5, 3, 4).reshape(
                3 * 20, 2, 16, 32),
            size=features[0].shape[-2:], mode="area")
        self.assertLess(float(pooled.abs().max()), 5.0e-7)

        with self.assertRaisesRegex(ValueError, "separate arms"):
            replace(
                config, latent_ude_coarse_residual=True).validate()

    def test_residual_mixture_preserves_legacy_state_and_is_well_formed(self):
        common_config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            residual_quantile_intervals=True)
        torch.manual_seed(17)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common_config))
        torch.manual_seed(17)
        mixture_model = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, residual_mixture_components=3,
            residual_mixture_probability_shrinkage=0.25))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(
                mixture_model.state_dict()[name], value, rtol=0.0, atol=0.0)
        mixture_model.load_state_dict(baseline.state_dict(), strict=False)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        mixture_model.set_normalizer_statistics(*statistics)
        x = torch.randn(2, 20, 16, 32, 3)
        expected_prediction = baseline.forward_uq_training(x)[0]
        values = mixture_model.forward_uq_training(x, return_mixture=True)
        prediction, _members, _quantiles, lower, upper, *_rest, mixture = values
        torch.testing.assert_close(
            prediction, expected_prediction, rtol=0.0, atol=0.0)
        self.assertIsNotNone(mixture)
        logits, means, scales = mixture
        expected_shape = (2, 20, 16, 32, 2, 3)
        self.assertEqual(tuple(logits.shape), expected_shape)
        self.assertEqual(tuple(means.shape), expected_shape)
        self.assertEqual(tuple(scales.shape), expected_shape)
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertTrue(bool(torch.isfinite(means).all()))
        self.assertTrue(bool((scales > 0.0).all()))
        self.assertTrue(bool(torch.isfinite(lower).all()))
        self.assertTrue(bool(torch.isfinite(upper).all()))
        self.assertTrue(bool((lower + upper >= 0.0).all()))
        mixture_quantiles = mixture_model._residual_mixture_quantiles(mixture)
        self.assertEqual(tuple(mixture_quantiles.shape), (
            2, 20, 16, 32, 2, 7))
        self.assertTrue(bool(torch.isfinite(mixture_quantiles).all()))
        self.assertTrue(bool((mixture_quantiles.diff(dim=-1) > 0.0).all()))

        loss = logits.square().mean() + means.square().mean() + scales.mean()
        loss.backward()
        self.assertGreater(
            float(mixture_model.residual_mixture_head.weight.grad.abs().sum()),
            0.0)

    def test_backbone_residual_dropout_is_eval_exact_and_state_compatible(self):
        common_config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            physical_energy_controller=True,
            physical_energy_member_sync=True)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common_config))
        dropout = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, backbone_residual_dropout=0.2))
        dropout.load_state_dict(baseline.state_dict(), strict=True)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        dropout.set_normalizer_statistics(*statistics)
        baseline.eval()
        dropout.eval()
        x = torch.randn(1, 20, 16, 32, 3)
        for expected, actual in zip(
                baseline.forward_quantiles(x), dropout.forward_quantiles(x)):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(
            dropout.architecture_config()["backbone_residual_dropout"], 0.2)
        stored = dropout.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        self.assertEqual(EnsembleUNet2dConfig(**stored), dropout.config)

    def test_backbone_dropout_only_masks_residual_branch_during_training(self):
        block = ResidualConvBlock(4, 4, 2, residual_dropout=0.2)
        x = torch.randn(2, 4, 8, 8)
        block.train()
        with mock.patch(
                "models.unet2d.F.dropout2d",
                side_effect=lambda value, **_kwargs: torch.zeros_like(value),
                ) as dropout:
            torch.testing.assert_close(block(x), x, rtol=0.0, atol=0.0)
            dropout.assert_called_once()
        block.eval()
        with mock.patch("models.unet2d.F.dropout2d") as dropout:
            block(x)
            dropout.assert_not_called()

    def test_decoupled_member_residual_is_exact_identity_and_zero_mean(self):
        common_config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common_config))
        decoupled = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, decoupled_member_residual_channels=8,
            decoupled_member_residual_blocks=1))
        decoupled.load_state_dict(
            {name: value for name, value in baseline.state_dict().items()
             if name in decoupled.state_dict()}, strict=False)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        decoupled.set_normalizer_statistics(*statistics)
        baseline.eval()
        decoupled.eval()
        x = torch.randn(2, 20, 16, 32, 3)
        baseline_values = baseline.forward_quantiles(x)
        decoupled_values = decoupled.forward_quantiles(x)
        self.assertTrue(torch.equal(
            decoupled_values[0], baseline_values[0]))
        self.assertTrue(torch.equal(
            decoupled_values[1], baseline_values[1]))
        self.assertEqual(
            int(torch.count_nonzero(
                decoupled.decoupled_member_residual.output.weight)), 0)

        torch.manual_seed(123)
        paired_baseline = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config))
        torch.manual_seed(123)
        paired_decoupled = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, decoupled_member_residual_channels=8,
            decoupled_member_residual_blocks=1))
        common_state = {
            name: value for name, value in paired_decoupled.state_dict().items()
            if not name.startswith("decoupled_member_residual.")}
        self.assertTrue(all(torch.equal(
            value, paired_baseline.state_dict()[name])
            for name, value in common_state.items()))

        with torch.no_grad():
            decoupled.decoupled_member_residual.output.weight.normal_()
        _prediction, members, *_rest = decoupled.forward_uq_training(x)
        raw_features = decoupled._decode(x)
        raw_members = decoupled._members_from_features(x, raw_features)
        delta = members - raw_members.detach()
        torch.testing.assert_close(
            delta.mean(dim=1), torch.zeros_like(delta[:, 0]),
            rtol=0.0, atol=2e-6)

    def test_member_feature_adapters_preserve_identity_rng_and_gradients(self):
        common_config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        torch.manual_seed(321)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common_config))
        torch.manual_seed(321)
        adapted = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, member_adapter_channels=8))
        common_state = {
            name: value for name, value in adapted.state_dict().items()
            if not name.startswith("member_feature_adapters.")}
        self.assertTrue(all(torch.equal(
            value, baseline.state_dict()[name])
            for name, value in common_state.items()))
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        adapted.set_normalizer_statistics(*statistics)
        baseline.eval()
        adapted.eval()
        x = torch.randn(2, 20, 16, 32, 3)
        baseline_values = baseline.forward_quantiles(x)
        adapted_values = adapted.forward_quantiles(x)
        self.assertTrue(all(torch.equal(first, second) for first, second in zip(
            baseline_values, adapted_values)))

        # Point projections are also zero at construction, so adapters only
        # receive a live gradient once the legacy heads have left update zero.
        with torch.no_grad():
            for heads in adapted.member_point_heads:
                for head in heads:
                    head.weight.normal_(std=0.01)
        adapted.train()
        _prediction, members, *_rest = adapted.forward_uq_training(x)
        members.square().mean().backward()
        adapter_output_grad = sum(
            float(adapter.body[-1].weight.grad.abs().sum())
            for member in adapted.member_feature_adapters
            for adapter in member)
        point_head_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in adapted.member_point_heads.parameters()
            if parameter.grad is not None)
        self.assertGreater(adapter_output_grad, 0.0)
        self.assertGreater(point_head_grad, 0.0)

    def test_mean_preserving_adapters_isolate_point_and_keep_member_path(self):
        config = EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            member_adapter_channels=8,
            mean_preserving_member_adapters=True)
        model = EnsembleUNet2d(config)
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        with torch.no_grad():
            for heads in model.member_point_heads:
                for head in heads:
                    head.weight.normal_(std=0.01)
            for member in model.member_feature_adapters:
                for adapter in member:
                    adapter.body[-1].weight.normal_(std=0.01)
        model.train()
        x = torch.randn(2, 20, 16, 32, 3)
        prediction, members, *_rest = model.forward_uq_training(x)
        features = model._decode(x)
        base_members = model._members_from_features(
            x, features, use_member_adapters=False)
        torch.testing.assert_close(
            prediction, base_members.mean(dim=1), rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            members.mean(dim=1), prediction, rtol=0.0, atol=2e-6)

        prediction.square().mean().backward()
        adapter_point_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.member_feature_adapters.parameters()
            if parameter.grad is not None)
        self.assertEqual(adapter_point_grad, 0.0)

        model.zero_grad(set_to_none=True)
        _prediction, members, *_rest = model.forward_uq_training(x)
        members.square().mean().backward()
        adapter_member_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.member_feature_adapters.parameters()
            if parameter.grad is not None)
        backbone_member_grad = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if name.startswith(("stem.", "encoder.", "decoder."))
            and parameter.grad is not None)
        point_head_member_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.member_point_heads.parameters()
            if parameter.grad is not None)
        self.assertGreater(adapter_member_grad, 0.0)
        self.assertGreater(backbone_member_grad, 0.0)
        self.assertGreater(point_head_member_grad, 0.0)

    def test_detached_adapter_context_keeps_only_legacy_feature_gradient(self):
        with self.assertRaisesRegex(ValueError, "requires mean-preserving"):
            EnsembleUNet2dConfig(
                base=8, depth=3, groups=4, difference_steps=2,
                ensemble_members=3, uq_hidden_channels=8,
                member_adapter_channels=8,
                detach_member_adapter_context=True).validate()
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            member_adapter_channels=8,
            mean_preserving_member_adapters=True,
            detach_member_adapter_context=True))
        with torch.no_grad():
            for heads in model.member_point_heads:
                for head in heads:
                    head.weight.normal_(std=0.01)
            for member in model.member_feature_adapters:
                for adapter in member:
                    adapter.body[-1].weight.normal_(std=0.01)
        x = torch.randn(2, 20, 16, 32, 3)
        features = model._decode(x)
        base_members = model._members_from_features(
            x, features, use_member_adapters=False)
        members = model._members_from_features(
            x, features, base_members=base_members)
        stochastic_residual = members - base_members
        context_gradients = torch.autograd.grad(
            stochastic_residual.square().mean(), features,
            allow_unused=True, retain_graph=True)
        context_gradient_sum = sum(
            float(value.abs().sum())
            for value in context_gradients if value is not None)
        self.assertLess(context_gradient_sum, 1e-6)

        legacy_gradients = torch.autograd.grad(
            members.square().mean(), features, allow_unused=False)
        self.assertGreater(
            sum(float(value.abs().sum()) for value in legacy_gradients), 0.0)

    def test_detached_uq_context_isolates_distribution_tower_gradients(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            member_adapter_channels=8,
            mean_preserving_member_adapters=True,
            detach_member_adapter_context=True,
            detach_uq_context=True,
            temporal_uq_refinement=True,
            temporal_uq_channels=8,
            temporal_uq_blocks=1))
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        with torch.no_grad():
            for heads in model.member_point_heads:
                for head in heads:
                    head.weight.normal_(std=0.01)
            for member in model.member_feature_adapters:
                for adapter in member:
                    adapter.body[-1].weight.normal_(std=0.01)
            model.quantile_head.weight.normal_(std=0.01)
        x = torch.randn(2, 20, 16, 32, 3)
        _prediction, _members, quantiles, lower, upper, *_rest = (
            model.forward_uq_training(x))
        (quantiles.square().mean()
         + lower.square().mean()
         + upper.square().mean()).backward()
        isolated_parameters = tuple(
            parameter for name, parameter in model.named_parameters()
            if name.startswith((
                "stem.", "encoder.", "decoder.",
                "member_point_heads.", "member_feature_adapters.")))
        self.assertTrue(all(
            parameter.grad is None for parameter in isolated_parameters))
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in model.uq_projections.parameters()
            if parameter.grad is not None), 0.0)
        self.assertGreater(
            float(model.quantile_head.weight.grad.abs().sum()), 0.0)

        model.zero_grad(set_to_none=True)
        features = model._decode(x)
        base_members = model._members_from_features(
            x, features, use_member_adapters=False)
        base_members.square().mean().backward()
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if name.startswith(("stem.", "encoder.", "decoder."))
            and parameter.grad is not None), 0.0)
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in model.member_point_heads.parameters()
            if parameter.grad is not None), 0.0)
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.member_feature_adapters.parameters()))

    def test_output_space_uq_allocator_is_identity_and_preserves_width_budget(self):
        common_config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            uq_flow_context=True)
        torch.manual_seed(83)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common_config))
        torch.manual_seed(83)
        candidate = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, output_space_uq_channels=8,
            output_space_uq_blocks=1))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(
                value, candidate.state_dict()[name], rtol=0.0, atol=0.0)
        for model in (baseline, candidate):
            model.set_normalizer_statistics(
                torch.zeros(3), torch.zeros(3),
                torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        x = torch.randn(2, 20, 16, 32, 3)
        baseline_values = baseline.forward_distribution(x)
        candidate_values = candidate.forward_distribution(x)
        for observed, expected in zip(candidate_values, baseline_values):
            torch.testing.assert_close(
                observed, expected, rtol=0.0, atol=0.0)

        with torch.no_grad():
            candidate.output_space_uq_allocator[-1].weight.normal_(std=0.02)
            candidate.output_space_uq_allocator[-1].bias.normal_(std=0.02)
        prediction, lower, upper = candidate.forward_distribution(x)
        base_prediction, base_lower, base_upper = baseline_values
        torch.testing.assert_close(
            prediction, base_prediction, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(lower, base_lower))
        self.assertFalse(torch.equal(upper, base_upper))
        torch.testing.assert_close(
            (lower + upper).mean(dim=(2, 3)),
            (base_lower + base_upper).mean(dim=(2, 3)),
            rtol=2e-6, atol=2e-7)

        (lower.square().mean() + upper.square().mean()).backward()
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in candidate.output_space_uq_allocator.parameters()
            if parameter.grad is not None), 0.0)

    def test_output_space_uq_reorders_signed_extents_after_allocation(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            uq_flow_context=True, residual_quantile_intervals=True,
            output_space_uq_channels=8, output_space_uq_blocks=1))
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        with torch.no_grad():
            model.output_space_uq_allocator[-1].weight.zero_()
            bias = model.output_space_uq_allocator[-1].bias.reshape(20, 2, 2)
            bias[..., 0].fill_(2.0)
            bias[..., 1].fill_(-2.0)
        lower = torch.full((1, 20, 8, 8, 2), -0.25)
        upper = torch.full_like(lower, 0.30)
        history = torch.randn(1, 20, 8, 8, 3)
        prediction = torch.randn_like(history)
        allocated = model._allocate_output_space_interval(
            lower, upper, history, prediction)
        self.assertTrue(bool(((allocated[0] + allocated[1]) < 0.0).any()))
        ordered = model._ordered_interval_extents(*allocated)
        self.assertTrue(bool(((ordered[0] + ordered[1]) >= 0.0).all()))

    def test_output_space_budgeter_is_identity_and_changes_only_intervals(self):
        common_config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            uq_flow_context=True)
        torch.manual_seed(97)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common_config))
        torch.manual_seed(97)
        candidate = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common_config, output_space_uq_budget_channels=8,
            output_space_uq_budget_blocks=1))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(
                value, candidate.state_dict()[name], rtol=0.0, atol=0.0)
        for model in (baseline, candidate):
            model.set_normalizer_statistics(
                torch.zeros(3), torch.zeros(3),
                torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        x = torch.randn(2, 20, 16, 32, 3)
        baseline_values = baseline.forward_distribution(x)
        candidate_values = candidate.forward_distribution(x)
        for observed, expected in zip(candidate_values, baseline_values):
            torch.testing.assert_close(
                observed, expected, rtol=0.0, atol=0.0)

        with torch.no_grad():
            candidate.output_space_uq_budgeter[-1].bias.fill_(0.25)
        prediction, lower, upper = candidate.forward_distribution(x)
        base_prediction, base_lower, base_upper = baseline_values
        torch.testing.assert_close(
            prediction, base_prediction, rtol=0.0, atol=0.0)
        self.assertGreater(
            float((lower + upper).mean()),
            float((base_lower + base_upper).mean()))
        (lower.square().mean() + upper.square().mean()).backward()
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in candidate.output_space_uq_budgeter.parameters()
            if parameter.grad is not None), 0.0)

    def test_budgeted_disagreement_only_reallocates_spatial_width(self):
        with self.assertRaisesRegex(ValueError, "requires mean-preserving"):
            EnsembleUNet2dConfig(
                base=8, depth=3, groups=4, difference_steps=2,
                ensemble_members=3, uq_hidden_channels=8,
                member_adapter_channels=8,
                budgeted_disagreement_allocation=True).validate()
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            member_adapter_channels=8,
            mean_preserving_member_adapters=True,
            detach_member_adapter_context=True,
            detach_uq_context=True,
            budgeted_disagreement_allocation=True))
        lower = torch.rand(2, 20, 16, 32, 2) + 0.01
        upper = torch.rand(2, 20, 16, 32, 2) + 0.01
        members = torch.randn(2, 3, 20, 16, 32, 3)
        initial_lower, initial_upper = model._budgeted_disagreement_interval(
            lower, upper, members)
        self.assertTrue(torch.equal(initial_lower, lower))
        self.assertTrue(torch.equal(initial_upper, upper))

        with torch.no_grad():
            model.disagreement_allocation_strength.copy_(
                torch.linspace(-0.5, 0.5, 40).reshape(20, 2))
            model.disagreement_allocation_sharpness.fill_(0.25)
        allocated_lower, allocated_upper = (
            model._budgeted_disagreement_interval(lower, upper, members))
        base_budget = (lower + upper).mean(dim=(2, 3))
        allocated_budget = (
            allocated_lower + allocated_upper).mean(dim=(2, 3))
        torch.testing.assert_close(
            allocated_budget, base_budget, rtol=2e-6, atol=2e-6)
        self.assertGreater(float(
            (allocated_lower - lower).abs().mean()), 1e-4)

        model.zero_grad(set_to_none=True)
        allocated_lower.square().mean().backward()
        self.assertIsNotNone(model.disagreement_allocation_strength.grad)
        self.assertGreater(float(
            model.disagreement_allocation_strength.grad.abs().sum()), 0.0)
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.member_feature_adapters.parameters()))

    def test_decoupled_member_loss_has_no_point_or_backbone_gradient(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            decoupled_member_residual_channels=8,
            decoupled_member_residual_blocks=1))
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        model.train()
        x = torch.randn(2, 20, 16, 32, 3)
        _prediction, members, *_rest = model.forward_uq_training(x)
        target = torch.randn_like(members)
        (members - target).square().mean().backward()
        self.assertGreater(float(
            model.decoupled_member_residual.output.weight.grad.abs().sum()),
            0.0)
        point_parameters = tuple(
            parameter for name, parameter in model.named_parameters()
            if (name.startswith("member_point_heads.")
                or name.startswith("encoder.")
                or name.startswith("decoder.")))
        self.assertTrue(all(
            parameter.grad is None for parameter in point_parameters))

    def test_backbone_residual_dropout_validation(self):
        self.assertEqual(
            EnsembleUNet2dConfig(base=8, groups=4).backbone_residual_dropout,
            0.0)
        with self.assertRaisesRegex(ValueError, "dropout"):
            EnsembleUNet2dConfig(
                base=8, groups=4, backbone_residual_dropout=1.0).validate()

    def test_latent_spectral_residual_is_zero_identity_then_trainable(self):
        common = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        torch.manual_seed(771)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**common))
        torch.manual_seed(771)
        spectral = EnsembleUNet2d(EnsembleUNet2dConfig(
            **common, latent_spectral_rank=4,
            latent_spectral_modes_y=2, latent_spectral_modes_x=3,
            latent_spectral_residual_limit=0.5))
        x = torch.randn(2, 20, 16, 32, 3)

        baseline_features = baseline._decode(x)
        spectral_features = spectral._decode(x)
        self.assertTrue(all(
            torch.equal(left, right)
            for left, right in zip(baseline_features, spectral_features)))
        common_state = {
            key: value for key, value in spectral.state_dict().items()
            if not key.startswith("latent_spectral_residual.")}
        self.assertEqual(set(common_state), set(baseline.state_dict()))
        self.assertTrue(all(
            torch.equal(value, baseline.state_dict()[key])
            for key, value in common_state.items()))

        branch = spectral.latent_spectral_residual
        self.assertIsNotNone(branch)
        self.assertEqual(int(torch.count_nonzero(branch.expand.weight)), 0)
        loss = spectral._decode(x)[0].square().mean()
        loss.backward()
        self.assertGreater(float(branch.expand.weight.grad.abs().sum()), 0.0)
        optimizer = torch.optim.SGD(branch.parameters(), lr=0.1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        spectral._decode(x)[0].square().mean().backward()
        self.assertGreater(
            float(branch.weight_positive.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(branch.weight_negative.grad.abs().sum()), 0.0)

        stored = spectral.architecture_config()
        self.assertEqual(stored["latent_spectral_rank"], 4)
        stored.pop("architecture")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(spectral.state_dict(), strict=True)

    def test_latent_spectral_validation_rejects_invalid_contract(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            EnsembleUNet2dConfig(
                base=8, groups=4, latent_spectral_rank=-1).validate()
        with self.assertRaisesRegex(ValueError, "modes"):
            EnsembleUNet2dConfig(
                base=8, groups=4, latent_spectral_modes_x=0).validate()

    def test_b210_preserves_head_groups_and_uses_deterministic_fallback(self):
        config = EnsembleUNet2dConfig(
            base=210, depth=4, groups=8, difference_steps=5,
            use_coordinates=True, use_regime_film=True,
            multi_scale_regime_film=True, mean_fluctuation=True,
            ensemble_members=4, uq_hidden_channels=160,
            member_adapter_channels=0, mean_profile_channels=104,
            uq_flow_context=True, residual_quantile_intervals=True,
            utility_refinement=True, utility_refinement_kernel=3,
            utility_refinement_blocks=0, temporal_uq_refinement=True,
            temporal_uq_channels=24, temporal_uq_blocks=2,
            temporal_uq_kernel=3, coverage_calibration=False,
            disagreement_scale_fraction=0.01,
            physical_energy_controller=True,
            physical_energy_member_sync=True,
            physical_energy_hidden_channels=32,
            backbone_residual_dropout=0.2)
        model = EnsembleUNet2d(config)
        controller_parameters = sum(
            value.numel() for value in model.energy_controller.parameters())
        self.assertEqual(controller_parameters, 7_169)
        self.assertEqual(
            sum(value.numel() for value in model.parameters()), 178_705_575)
        backbone_blocks = (
            [stage[1] for stage in model.encoder]
            + [model.bottleneck] + list(model.decoder))
        backbone_groups = []
        for block in backbone_blocks:
            backbone_groups.append([
                module.num_groups for module in block.body
                if isinstance(module, torch.nn.GroupNorm)])
        self.assertEqual(backbone_groups, [
            [7, 7], [8, 8], [8, 8], [8, 8],
            [8, 8], [8, 7], [7, 7],
        ])
        all_groups = {
            module.num_groups for module in model.modules()
            if isinstance(module, torch.nn.GroupNorm)}
        self.assertIn(8, all_groups)

    def test_multiscale_energy_controller_is_zero_identity_then_uses_all_scales(self):
        config = EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            physical_energy_controller=True,
            physical_energy_member_sync=True,
            physical_energy_multiscale=True,
            physical_energy_hidden_channels=8)
        model = EnsembleUNet2d(config)
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        x = torch.randn(1, 20, 16, 32, 3)
        features = model._decode(x)
        raw = model._physical_energy_delta(features)
        self.assertEqual(tuple(raw.shape), (1, 1, 16, 32))
        self.assertEqual(int(torch.count_nonzero(raw)), 0)
        with torch.no_grad():
            model.energy_controller.body[-1].weight.fill_(0.1)
        features = [value.detach().requires_grad_() for value in features]
        model._physical_energy_delta(features).square().mean().backward()
        self.assertTrue(all(
            value.grad is not None and torch.count_nonzero(value.grad)
            for value in features))
        stored = model.architecture_config()
        self.assertTrue(stored["physical_energy_multiscale"])
        stored.pop("architecture")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def test_multiscale_energy_requires_enabled_controller(self):
        with self.assertRaisesRegex(ValueError, "multiscale physical energy"):
            EnsembleUNet2dConfig(
                base=8, groups=4,
                physical_energy_multiscale=True).validate()

    def test_horizon_local_energy_is_exact_then_preserves_means(self):
        config = EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            physical_energy_controller=True,
            physical_energy_member_sync=True,
            physical_energy_horizon_local=True,
            physical_energy_hidden_channels=8)
        model = EnsembleUNet2d(config)
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        features = model._decode(torch.randn(2, 20, 16, 32, 3))
        prediction = torch.randn(2, 20, 16, 32, 3)
        members = torch.randn(2, 3, 20, 16, 32, 3)
        initial_point = model._energy_controlled_prediction(
            prediction, features)
        initial_members = model._energy_controlled_members(members, features)
        self.assertTrue(torch.equal(initial_point, prediction))
        self.assertTrue(torch.equal(initial_members, members))
        self.assertEqual(
            tuple(model._physical_energy_delta(features).shape),
            (2, 20, 16, 32))
        with torch.no_grad():
            model.energy_controller.body[-1].weight.normal_(std=0.05)
            model.energy_controller.body[-1].bias.normal_(std=0.01)
        point = model._energy_controlled_prediction(prediction, features)
        changed_members = model._energy_controlled_members(members, features)
        torch.testing.assert_close(
            point[..., :2].mean(dim=1),
            prediction[..., :2].mean(dim=1), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            changed_members[..., :2].mean(dim=2),
            members[..., :2].mean(dim=2), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(point[..., 2:], prediction[..., 2:]))
        self.assertTrue(torch.equal(
            changed_members[..., 2:], members[..., 2:]))
        point[..., :2].square().mean().backward()
        self.assertGreater(float(
            model.energy_controller.body[-1].weight.grad.abs().sum()), 0.0)
        stored = model.architecture_config()
        self.assertTrue(stored["physical_energy_horizon_local"])

    def test_horizon_local_energy_validation(self):
        with self.assertRaisesRegex(ValueError, "horizon-local physical"):
            EnsembleUNet2dConfig(
                base=8, groups=4,
                physical_energy_horizon_local=True).validate()
        with self.assertRaisesRegex(ValueError, "separate arms"):
            EnsembleUNet2dConfig(
                base=8, groups=4, physical_energy_controller=True,
                physical_energy_horizon_local=True,
                physical_energy_multiscale=True).validate()

    def test_history_energy_is_observation_conditioned_and_zero_identity(self):
        config = EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            physical_energy_controller=True,
            physical_energy_member_sync=True,
            physical_energy_history_conditioned=True,
            physical_energy_history_steps=5,
            physical_energy_hidden_channels=8)
        model = EnsembleUNet2d(config)
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        history = torch.randn(2, 20, 16, 32, 3)
        features = model._decode(history)
        prediction = torch.randn(2, 20, 16, 32, 3)
        initial = model._energy_controlled_prediction(
            prediction, features, history)
        self.assertTrue(torch.equal(initial, prediction))
        raw = model._physical_energy_delta(features, history)
        self.assertEqual(tuple(raw.shape), (2, 1, 16, 32))
        self.assertEqual(int(torch.count_nonzero(raw)), 0)
        with torch.no_grad():
            model.energy_controller.body[-1].weight.fill_(0.1)
        detached_features = [value.detach() for value in features]
        history.requires_grad_()
        changed = model._energy_controlled_prediction(
            prediction, detached_features, history)
        changed[..., :2].square().mean().backward()
        self.assertIsNotNone(history.grad)
        self.assertGreater(float(history.grad.abs().sum()), 0.0)
        self.assertTrue(torch.equal(changed[..., 2:], prediction[..., 2:]))
        self.assertTrue(all(value.grad_fn is None for value in detached_features))
        stored = model.architecture_config()
        self.assertTrue(stored["physical_energy_history_conditioned"])
        stored.pop("architecture")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def test_history_energy_requires_controller_and_fuses_with_multiscale(self):
        with self.assertRaisesRegex(ValueError, "history-conditioned"):
            EnsembleUNet2dConfig(
                base=8, groups=4,
                physical_energy_history_conditioned=True).validate()
        config = EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            physical_energy_controller=True,
            physical_energy_member_sync=True,
            physical_energy_history_conditioned=True,
            physical_energy_multiscale=True,
            physical_energy_history_steps=5,
            physical_energy_hidden_channels=8)
        model = EnsembleUNet2d(config)
        history = torch.randn(2, 20, 16, 32, 3)
        features = model._decode(history)
        raw = model._physical_energy_delta(features, history)
        self.assertEqual(tuple(raw.shape), (2, 1, 16, 32))
        self.assertEqual(int(torch.count_nonzero(raw)), 0)
        with torch.no_grad():
            model.energy_controller.body[-1].weight.fill_(0.1)
            model.energy_controller.history.body[-1].weight.fill_(0.1)
        features = [value.detach().requires_grad_() for value in features]
        history.requires_grad_()
        model._physical_energy_delta(features, history).square().mean().backward()
        self.assertTrue(all(
            value.grad is not None and torch.count_nonzero(value.grad)
            for value in features))
        self.assertIsNotNone(history.grad)
        self.assertGreater(float(history.grad.abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "separate arms"):
            EnsembleUNet2dConfig(
                base=8, groups=4, physical_energy_controller=True,
                physical_energy_history_conditioned=True,
                physical_energy_horizon_local=True).validate()

    def test_mean_profile_objective_is_differentiable(self):
        class IdentityNormalizer:
            mean_target = torch.zeros(3)
            std_target = torch.ones(3)

        prediction = torch.randn(
            2, 20, 32, 64, 3, requires_grad=True)
        target = torch.randn_like(prediction)
        loss = mean_profile_objective(
            prediction, target, IdentityNormalizer())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)
        # MVPE averages over time, so every horizon receives a gradient at probes.
        self.assertTrue(torch.all(
            prediction.grad[..., :2].abs().sum(dim=(0, 2, 3, 4)) > 0.0))

    def test_soft_native_sps_only_updates_interval_path(self):
        class IdentityNormalizer:
            mean_target = torch.zeros(3)
            std_target = torch.ones(3)

        prediction = torch.randn(
            2, 20, 32, 64, 3, requires_grad=True)
        target = torch.randn_like(prediction).detach() + 0.1
        lower = torch.full(
            (2, 20, 32, 64, 2), 0.02, requires_grad=True)
        upper = torch.full(
            (2, 20, 32, 64, 2), 0.02, requires_grad=True)
        loss, score = soft_native_sps_objective(
            prediction, target, lower, upper,
            IdentityNormalizer(), tau_fraction=0.05)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(score))
        loss.backward()
        self.assertIsNone(prediction.grad)
        self.assertGreater(float(lower.grad.abs().sum()), 0.0)
        self.assertGreater(float(upper.grad.abs().sum()), 0.0)

    def test_soft_sps_proxy_scale_keeps_dataset_mask_and_factor(self):
        class MetricNormalizer:
            mean_target = torch.zeros(3)
            std_target = torch.ones(3)

        class ProxyNormalizer:
            mean_target = torch.tensor([10.0, -7.0, 0.0])
            std_target = torch.tensor([2.0, 3.0, 1.0])

        target = torch.ones(1, 20, 20, 64, 3)
        target[..., :32, :2] = 0.0
        prediction = target.clone()
        prediction[..., 32:, :2] -= 0.1
        lower = torch.full((1, 20, 20, 64, 2), 0.05)
        upper = torch.full_like(lower, 0.05)
        loss, score = soft_native_sps_objective(
            prediction, target, lower, upper, ProxyNormalizer(),
            tau_fraction=0.05, metric_normalizer=MetricNormalizer())

        proxy_residual = (
            target[..., :2] - prediction[..., :2]
        ) * ProxyNormalizer.std_target[:2]
        lower_physical = lower * ProxyNormalizer.std_target[:2]
        upper_physical = upper * ProxyNormalizer.std_target[:2]
        tau = lower.new_tensor([0.01075, 0.00550]).view(
            1, 1, 1, 1, 2) * 0.05
        efficiency = (
            torch.sigmoid((proxy_residual + lower_physical) / tau)
            * torch.sigmoid((upper_physical - proxy_residual) / tau)
            * torch.exp(
                -(lower_physical + upper_physical) / common.SPS_SIGMA))
        scored = target[..., :2] != 0.0
        factor = common.sample_sps_factor(
            prediction[..., :2], target[..., :2])
        expected = (factor * efficiency[scored].mean()).mean()
        torch.testing.assert_close(score, expected)
        torch.testing.assert_close(loss, -expected)

    def test_zero_initialized_members_are_persistence(self):
        model = self.model()
        x = torch.randn(1, 20, 16, 32, 3)
        prediction, members, _quantiles, _lower, _upper = model.forward_quantiles(x)
        expected = model._persistence(x).unsqueeze(1).expand(-1, 20, -1, -1, -1)
        torch.testing.assert_close(prediction, expected)
        for member in members.unbind(dim=1):
            torch.testing.assert_close(member, expected)
        torch.testing.assert_close(
            members.std(dim=1, unbiased=False), torch.zeros_like(prediction))

    def test_every_member_and_quantile_head_receive_gradient(self):
        model = self.model()
        x = torch.randn(1, 20, 16, 32, 3)
        _prediction, members, quantiles, lower, upper = model.forward_quantiles(x)
        loss = quantiles.square().mean() + lower.mean() + upper.mean()
        for index, member in enumerate(members.unbind(dim=1), start=1):
            loss = loss + index * member.square().mean()
        loss.backward()
        for heads in model.member_point_heads:
            gradient = heads[-1].weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertIsNotNone(model.quantile_head.weight.grad)
        self.assertGreater(float(model.quantile_head.weight.grad.abs().sum()), 0.0)

    def test_member_specific_spatial_adapters_receive_gradient(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            member_adapter_channels=4,
            disagreement_scale_fraction=0.01))
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        for heads in model.member_point_heads:
            for head in heads:
                torch.nn.init.normal_(head.weight, std=0.01)
        x = torch.randn(1, 20, 16, 32, 3)
        _prediction, members, _quantiles, _lower, _upper = (
            model.forward_quantiles(x))
        sum(member.square().mean() for member in members.unbind(1)).backward()
        for adapters in model.member_feature_adapters:
            for adapter in adapters:
                gradient = adapter.body[-1].weight.grad
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_flow_context_adds_seventy_direct_motion_channels(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            uq_flow_context=True,
            disagreement_scale_fraction=0.01))
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        x = torch.randn(2, 20, 16, 32, 3)
        prediction, members, quantiles, lower, upper = (
            model.forward_quantiles(x))
        context = model._flow_context(x, prediction)
        self.assertEqual(tuple(context.shape), (2, 70, 16, 32))
        self.assertEqual(tuple(members.shape), (2, 3, 20, 16, 32, 3))
        self.assertEqual(tuple(quantiles.shape), (2, 20, 16, 32, 2, 7))
        self.assertTrue(torch.all(lower > 0.0))
        self.assertTrue(torch.all(upper > 0.0))

    def test_residual_quantile_intervals_can_shift_away_from_point(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            residual_quantile_intervals=True))
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        quantiles = torch.arange(1.0, 8.0).view(
            1, 1, 1, 1, 1, 7).expand(1, 2, 3, 4, 2, 7)
        lower_extent, upper_extent = model._utility_interval(quantiles)
        self.assertTrue(torch.all(lower_extent < 0.0))
        self.assertTrue(torch.all(upper_extent > 0.0))
        self.assertTrue(torch.all(lower_extent + upper_extent > 0.0))

    def test_utility_refinement_starts_exact_and_receives_gradient(self):
        config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**config))
        refined = EnsembleUNet2d(EnsembleUNet2dConfig(
            **config, utility_refinement=True,
            utility_refinement_kernel=3,
            utility_refinement_blocks=1))
        self._copy_shared_state(baseline, refined)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        refined.set_normalizer_statistics(*statistics)
        x = torch.randn(1, 20, 16, 32, 3)
        expected = baseline.forward_quantiles(x)
        actual = refined.forward_quantiles(x)
        for expected_value, actual_value in zip(expected, actual):
            torch.testing.assert_close(
                actual_value, expected_value, rtol=0.0, atol=0.0)
        (actual[3].mean() + actual[4].mean()).backward()
        gradient = refined.utility_refinement_head[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_interval_endpoint_ordering_repairs_only_invalid_elements(self):
        lower = torch.tensor([[[[[0.2, -0.7], [-0.6, 0.1]]]]])
        upper = torch.tensor([[[[[0.4, 0.3], [0.2, -0.3]]]]])
        repaired_lower, repaired_upper = (
            EnsembleUNet2d._ordered_interval_extents(lower, upper))
        expected_lower = torch.tensor(
            [[[[[0.2, -0.3], [-0.2, 0.3]]]]])
        expected_upper = torch.tensor(
            [[[[[0.4, 0.7], [0.6, -0.1]]]]])
        torch.testing.assert_close(repaired_lower, expected_lower)
        torch.testing.assert_close(repaired_upper, expected_upper)
        self.assertTrue(torch.all(repaired_lower + repaired_upper >= 0.0))

    def test_temporal_uq_refinement_starts_exact_and_receives_gradient(self):
        config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**config))
        refined = EnsembleUNet2d(EnsembleUNet2dConfig(
            **config, temporal_uq_refinement=True,
            temporal_uq_channels=8, temporal_uq_blocks=1,
            temporal_uq_kernel=3))
        self._copy_shared_state(baseline, refined)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        refined.set_normalizer_statistics(*statistics)
        x = torch.randn(1, 20, 16, 32, 3)
        expected = baseline.forward_quantiles(x)
        actual = refined.forward_quantiles(x)
        for expected_value, actual_value in zip(expected, actual):
            torch.testing.assert_close(
                actual_value, expected_value, rtol=0.0, atol=0.0)
        actual[2].square().mean().backward()
        gradient = refined.temporal_uq_refiner.output.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertEqual(
            tuple(refined.temporal_uq_refiner.horizon_embedding.shape),
            (1, 8, 20, 1, 1))

    def test_coverage_calibration_starts_at_nominal_mass(self):
        config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**config))
        calibrated = EnsembleUNet2d(EnsembleUNet2dConfig(
            **config, coverage_calibration=True,
            coverage_calibration_kernel=3,
            coverage_calibration_blocks=1))
        self._copy_shared_state(baseline, calibrated)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        calibrated.set_normalizer_statistics(*statistics)
        x = torch.randn(1, 20, 16, 32, 3)
        expected = baseline.forward_quantiles(x)
        actual = calibrated.forward_uq_training(x)
        for expected_value, actual_value in zip(expected, actual[:5]):
            torch.testing.assert_close(
                actual_value, expected_value, rtol=0.0, atol=0.0)
        logits = actual[-1]
        self.assertIsNotNone(logits)
        levels = torch.tensor(calibrated.config.quantile_levels)
        masses = torch.stack([
            levels[upper] - levels[lower]
            for lower, upper in calibrated.interval_candidate_pairs])
        probabilities = torch.sigmoid(logits)
        torch.testing.assert_close(
            probabilities,
            masses.view(1, 1, 1, 1, 1, -1).expand_as(probabilities))
        logits.mean().backward()
        gradient = calibrated.coverage_calibration_head[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        temperature = EnsembleUNet2d(EnsembleUNet2dConfig(
            **config, coverage_calibration=True,
            coverage_calibration_mode="temperature",
            coverage_calibration_kernel=1,
            coverage_calibration_blocks=0))
        self._copy_shared_state(baseline, temperature)
        temperature.set_normalizer_statistics(*statistics)
        temperature_values = temperature.forward_uq_training(x)
        for expected_value, actual_value in zip(
                expected, temperature_values[:5]):
            torch.testing.assert_close(
                actual_value, expected_value, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            torch.sigmoid(temperature_values[-1]),
            masses.view(1, 1, 1, 1, 1, -1).expand_as(
                temperature_values[-1]))

    @staticmethod
    def _copy_shared_state(source, destination):
        state = destination.state_dict()
        state.update(source.state_dict())
        destination.load_state_dict(state, strict=True)

    def test_cross_scale_film_starts_exact_and_receives_gradient(self):
        config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**config))
        for heads in baseline.member_point_heads:
            for head in heads:
                torch.nn.init.normal_(head.weight, std=0.01)
        conditioned = EnsembleUNet2d(EnsembleUNet2dConfig(
            **config, multi_scale_regime_film=True))
        self._copy_shared_state(baseline, conditioned)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        conditioned.set_normalizer_statistics(*statistics)
        x = torch.randn(1, 20, 16, 32, 3)
        expected = baseline(x)
        actual = conditioned(x)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        actual[..., :2].square().mean().backward()
        gradient = conditioned.decoder_regime_film[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_local_dynamics_cross_conditioner_is_exact_then_trains_body(self):
        config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**config))
        for heads in baseline.member_point_heads:
            for head in heads:
                torch.nn.init.normal_(head.weight, std=0.01)
        conditioned = EnsembleUNet2d(EnsembleUNet2dConfig(
            **config, cross_conditioned_backbone=True,
            cross_conditioned_steps=3, cross_conditioned_channels=8))
        self._copy_shared_state(baseline, conditioned)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        conditioned.set_normalizer_statistics(*statistics)
        x = torch.randn(1, 20, 16, 32, 3)
        baseline.eval()
        conditioned.eval()
        expected = baseline.forward_quantiles(x)
        actual = conditioned.forward_quantiles(x)
        for expected_value, actual_value in zip(expected, actual):
            torch.testing.assert_close(
                actual_value, expected_value, rtol=0.0, atol=0.0)

        conditioned.train()
        optimizer = torch.optim.SGD(conditioned.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        conditioned(x)[..., :2].square().mean().backward()
        projection_gradient = (
            conditioned.cross_conditioner.projections[-1].weight.grad)
        self.assertIsNotNone(projection_gradient)
        self.assertGreater(float(projection_gradient.abs().sum()), 0.0)
        optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        conditioned(x)[..., :2].square().mean().backward()
        body_gradient = conditioned.cross_conditioner.body[0].weight.grad
        self.assertIsNotNone(body_gradient)
        self.assertGreater(float(body_gradient.abs().sum()), 0.0)
        stored = conditioned.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(conditioned.state_dict(), strict=True)

    def test_cross_conditioner_and_member_adapters_are_joint_exact_noop(self):
        config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**config))
        for heads in baseline.member_point_heads:
            for head in heads:
                torch.nn.init.normal_(head.weight, std=0.01)
        joint = EnsembleUNet2d(EnsembleUNet2dConfig(
            **config, cross_conditioned_backbone=True,
            cross_conditioned_steps=3, cross_conditioned_channels=8,
            member_adapter_channels=8))
        self._copy_shared_state(baseline, joint)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        joint.set_normalizer_statistics(*statistics)
        baseline.eval()
        joint.eval()
        x = torch.randn(2, 20, 16, 32, 3)
        expected = baseline.forward_quantiles(x)
        actual = joint.forward_quantiles(x)
        self.assertTrue(all(torch.equal(first, second)
                            for first, second in zip(expected, actual)))

        joint.train()
        _prediction, members, *_rest = joint.forward_uq_training(x)
        members.square().mean().backward()
        self.assertGreater(float(
            joint.cross_conditioner.projections[-1].weight.grad.abs().sum()),
            0.0)
        adapter_output_gradient = sum(
            float(adapter.body[-1].weight.grad.abs().sum())
            for member in joint.member_feature_adapters
            for adapter in member)
        self.assertGreater(adapter_output_gradient, 0.0)

    def test_mean_profile_refiner_is_shared_over_members_and_time(self):
        config = dict(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8)
        baseline = EnsembleUNet2d(EnsembleUNet2dConfig(**config))
        for heads in baseline.member_point_heads:
            for head in heads:
                torch.nn.init.normal_(head.weight, std=0.01)
        refined = EnsembleUNet2d(EnsembleUNet2dConfig(
            **config, mean_profile_channels=4))
        self._copy_shared_state(baseline, refined)
        statistics = (
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.tensor([0.2, 0.1, 0.3]))
        baseline.set_normalizer_statistics(*statistics)
        refined.set_normalizer_statistics(*statistics)
        x = torch.randn(1, 20, 16, 32, 3)
        baseline_features = baseline._decode(x)
        refined_features = refined._decode(x)
        baseline_members = baseline._members_from_features(
            x, baseline_features)
        initial_members = refined._members_from_features(
            x, refined_features)
        torch.testing.assert_close(
            initial_members, baseline_members, rtol=0.0, atol=0.0)
        initial_members[..., :2].square().mean().backward()
        gradient = refined.mean_profile_refiner[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        with torch.no_grad():
            torch.nn.init.normal_(
                refined.mean_profile_refiner[-1].weight, std=0.01)
        changed = refined._members_from_features(
            x, refined._decode(x))
        delta = changed - baseline_members
        torch.testing.assert_close(
            delta, delta[:, :1].expand_as(delta))
        torch.testing.assert_close(
            delta, delta[:, :, :1].expand_as(delta))
        torch.testing.assert_close(
            delta[..., 2], torch.zeros_like(delta[..., 2]))

    def test_config_roundtrip(self):
        model = self.model()
        config = model.architecture_config()
        self.assertEqual(config.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**config))
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def test_temporal_point_decoder_is_persistence_then_trains_through_body(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            temporal_point_decoder=True, temporal_point_channels=4,
            temporal_point_blocks=2, temporal_point_kernel=5))
        statistics = (
            torch.tensor([1.0, -2.0, 0.5]),
            torch.tensor([0.5, 1.0, -0.5]),
            torch.tensor([2.0, 3.0, 4.0]),
            torch.tensor([1.0, 1.5, 2.0]))
        model.set_normalizer_statistics(*statistics)
        x = torch.randn(2, 20, 16, 32, 3)
        features = model._decode(x)
        members = model._members_from_features(x, features)
        prediction = members.mean(dim=1)
        expected = model._persistence(x).unsqueeze(1).expand_as(prediction)
        torch.testing.assert_close(
            members, expected.unsqueeze(1).expand_as(members),
            rtol=0.0, atol=0.0)
        torch.testing.assert_close(prediction, expected, rtol=0.0, atol=5e-7)
        self.assertEqual(tuple(prediction.shape), (2, 20, 16, 32, 3))
        self.assertEqual(len(model.member_point_heads), 0)

        target = torch.randn_like(prediction)
        (prediction - target).square().mean().backward()
        mean_gradient = (
            model.temporal_point_decoder.member_mean_heads[0][0].weight.grad)
        fluctuation_gradient = (
            model.temporal_point_decoder
            .member_fluctuation_heads[0][0].weight.grad)
        self.assertIsNotNone(mean_gradient)
        self.assertIsNotNone(fluctuation_gradient)
        self.assertGreater(float(mean_gradient.abs().sum()), 0.0)
        self.assertGreater(float(fluctuation_gradient.abs().sum()), 0.0)

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for heads in (
                    model.temporal_point_decoder.member_fluctuation_heads):
                for head in heads:
                    torch.nn.init.normal_(head.weight, std=0.05)
        model(x).square().mean().backward()
        projection_gradient = (
            model.temporal_point_decoder.projections[0][0].weight.grad)
        body_gradient = (
            model.temporal_point_decoder.scale_blocks[0][0]
            .body[0].weight.grad)
        horizon_gradient = (
            model.temporal_point_decoder.horizon_embedding.grad)
        for gradient in (
                projection_gradient, body_gradient, horizon_gradient):
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

        stored = model.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def test_temporal_point_decoder_validation(self):
        with self.assertRaisesRegex(ValueError, "temporal point kernel"):
            EnsembleUNet2dConfig(
                base=8, groups=4, temporal_point_kernel=4).validate()
        with self.assertRaisesRegex(ValueError, "factorization requires"):
            EnsembleUNet2dConfig(
                base=8, groups=4,
                temporal_point_amplitude_factorization=True).validate()

    def test_temporal_amplitude_factorization_is_bounded_and_trainable(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            temporal_point_decoder=True, temporal_point_channels=4,
            temporal_point_blocks=2, temporal_point_kernel=5,
            temporal_point_amplitude_factorization=True,
            temporal_point_amplitude_limit=3.0))
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        x = torch.randn(2, 20, 16, 32, 3)
        features = model._decode(x)
        members = model._members_from_features(x, features)
        expected = model._persistence(x).unsqueeze(1).unsqueeze(1)
        torch.testing.assert_close(
            members, expected.expand_as(members), rtol=0.0, atol=0.0)
        self.assertTrue(any(
            torch.count_nonzero(head.weight)
            for heads in model.temporal_point_decoder.member_fluctuation_heads
            for head in heads))
        self.assertTrue(all(
            torch.count_nonzero(head.weight) == 0
            for heads in model.temporal_point_decoder.member_amplitude_heads
            for head in heads))

        target = torch.randn_like(members.mean(dim=1))
        (model(x) - target).square().mean().backward()
        amplitude_gradient = (
            model.temporal_point_decoder.member_amplitude_heads[0][0]
            .weight.grad)
        self.assertIsNotNone(amplitude_gradient)
        self.assertGreater(float(amplitude_gradient.abs().sum()), 0.0)

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for heads in model.temporal_point_decoder.member_amplitude_heads:
                for head in heads:
                    head.weight.fill_(100.0)
        members = model._members_from_features(x, model._decode(x))
        fluctuation = members - members.mean(dim=2, keepdim=True)
        rms = fluctuation.square().mean(dim=2).sqrt()
        self.assertLessEqual(float(rms.max()), 3.0 + 1e-4)
        members.square().mean().backward()
        gradients = (
            model.temporal_point_decoder
            .member_fluctuation_heads[0][0].weight.grad,
            model.temporal_point_decoder.projections[0][0].weight.grad,
            model.temporal_point_decoder.scale_blocks[0][0]
            .body[0].weight.grad,
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

        stored = model.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def test_isoenergetic_temporal_transport_is_identity_and_invariant(self):
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=3, groups=4, difference_steps=2,
            ensemble_members=3, uq_hidden_channels=8,
            isoenergetic_temporal_transport=True,
            temporal_transport_channels=8,
            temporal_transport_blocks=2,
            temporal_transport_kernel=5,
            temporal_transport_max_displacement=1.5))
        model.set_normalizer_statistics(
            torch.zeros(3), torch.zeros(3),
            torch.ones(3), torch.ones(3))
        prediction = torch.randn(2, 20, 16, 32, 3)
        feature = torch.randn(2, 8, 16, 32)
        transported = model.temporal_transport(prediction, feature)
        self.assertTrue(torch.equal(transported, prediction))

        zero_energy = torch.randn(2, 1, 16, 32, 3).expand(
            -1, 20, -1, -1, -1).clone().requires_grad_(True)
        zero_output = model.temporal_transport(zero_energy, feature)
        (zero_output - torch.randn_like(zero_output)).square().mean().backward()
        self.assertTrue(torch.isfinite(zero_energy.grad).all())
        for parameter in model.temporal_transport.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        model.zero_grad(set_to_none=True)

        with torch.no_grad():
            model.temporal_transport.output.weight.normal_(std=0.02)
            model.temporal_transport.output.bias.fill_(0.1)
        transported = model.temporal_transport(prediction, feature)
        source_uv = prediction[..., :2]
        transported_uv = transported[..., :2]
        torch.testing.assert_close(
            transported_uv.mean(dim=1), source_uv.mean(dim=1),
            rtol=1e-5, atol=2e-6)
        source_energy = (
            source_uv - source_uv.mean(dim=1, keepdim=True)
        ).square().mean(dim=1)
        transported_energy = (
            transported_uv - transported_uv.mean(dim=1, keepdim=True)
        ).square().mean(dim=1)
        torch.testing.assert_close(
            transported_energy, source_energy, rtol=1e-5, atol=2e-6)
        self.assertTrue(torch.equal(transported[..., 2], prediction[..., 2]))

        target = torch.randn_like(transported)
        (transported - target).square().mean().backward()
        gradients = (
            model.temporal_transport.output.weight.grad,
            model.temporal_transport.blocks[0].body[0].weight.grad,
            model.temporal_transport.feature_projection.weight.grad,
            model.temporal_transport.trajectory_projection.weight.grad,
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

        stored = model.architecture_config()
        self.assertEqual(stored.pop("architecture"), "ensemble_unet")
        rebuilt = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
        rebuilt.load_state_dict(model.state_dict(), strict=True)

    def test_isoenergetic_temporal_transport_validation(self):
        with self.assertRaisesRegex(ValueError, "transport kernel"):
            EnsembleUNet2dConfig(
                base=8, groups=4, temporal_transport_kernel=4).validate()
        with self.assertRaisesRegex(ValueError, "displacement"):
            EnsembleUNet2dConfig(
                base=8, groups=4,
                temporal_transport_max_displacement=0.0).validate()

    def test_trajectory_bootstrap_is_fixed_and_nonempty(self):
        trajectory_ids = torch.tensor([3, 3, 17, 17, 29, 41])
        mask = trajectory_member_mask(
            trajectory_ids, probability=0.50, seed=7, member_count=4)
        self.assertEqual(tuple(mask.shape), (6, 4))
        self.assertTrue(bool(mask.any(dim=1).all()))
        torch.testing.assert_close(mask[0], mask[1])
        torch.testing.assert_close(mask[2], mask[3])
        torch.testing.assert_close(
            mask,
            trajectory_member_mask(
                trajectory_ids, probability=0.50, seed=7, member_count=4))
        self.assertGreater(torch.unique(mask, dim=0).shape[0], 1)


if __name__ == "__main__":
    unittest.main()
