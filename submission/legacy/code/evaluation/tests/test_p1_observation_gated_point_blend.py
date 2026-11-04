from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

import evaluate_mean_expert_oof_blend as s1  # noqa: E402
import evaluate_p1_observation_gated_point_blend as subject  # noqa: E402


class ObservationFeaturesTest(unittest.TestCase):
    def test_exact_eight_feature_definition(self) -> None:
        history = np.zeros((6, 2, 2, 3), dtype=np.float32)
        history[:, 0, 0, 0] = np.arange(1.0, 7.0)
        history[:, 0, 0, 1] = 2.0 * np.arange(1.0, 7.0)

        actual = subject.observation_features(history)

        expected = np.asarray([
            6.0,
            12.0,
            0.0,
            0.0,
            np.std(np.arange(1.0, 7.0)),
            np.std(2.0 * np.arange(1.0, 7.0)),
            1.0,
            2.0,
        ])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        self.assertEqual(actual.shape, (len(subject.FEATURE_NAMES),))

    def test_empty_measured_mask_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no measured"):
            subject.observation_features(np.zeros((6, 2, 2, 3)))


class BlendAndOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(7)
        shape = (3, 6, 16, 24, 3)
        self.a = generator.normal(size=shape).astype(np.float32)
        self.b = generator.normal(size=shape).astype(np.float32)
        self.c = generator.normal(size=shape).astype(np.float32)
        for value in (self.a, self.b, self.c):
            value[..., 2] = 0.0

    def test_constant_baseline_is_bitwise_s1_formula(self) -> None:
        weights = np.full(self.a.shape[0], 0.5)
        actual = subject.blend_sample_weights(
            self.a, self.b, self.c, weights, weights)
        expected = s1.blend_point_candidate(
            self.a, self.b, self.c, weight_b=0.5, mean_weight=0.5)
        self.assertTrue(np.array_equal(actual, expected))

    def test_mse_initialization_improves_exact_bounded_target(self) -> None:
        expected_b = np.asarray([0.3125, 0.5, 0.6875])
        expected_c = np.asarray([0.6875, 0.3125, 0.5])
        target = subject.blend_sample_weights(
            self.a, self.b, self.c, expected_b, expected_c)
        actual_b, actual_c = subject.mse_oracle_initialization(
            self.a, self.b, self.c, target)
        candidate = subject.blend_sample_weights(
            self.a, self.b, self.c, actual_b, actual_c)
        baseline = subject.blend_sample_weights(
            self.a, self.b, self.c,
            np.full(3, 0.5), np.full(3, 0.5))

        self.assertLess(
            float(np.mean((candidate[..., :2] - target[..., :2]) ** 2)),
            float(np.mean((baseline[..., :2] - target[..., :2]) ** 2)))
        np.testing.assert_allclose(actual_b, expected_b, atol=2e-5, rtol=0.0)
        np.testing.assert_allclose(actual_c, expected_c, atol=0.0, rtol=0.0)


class ABHeadroomTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(19)
        shape = (3, 20, 2, 3, 3)
        self.a = generator.normal(size=shape).astype(np.float32)
        self.b = generator.normal(size=shape).astype(np.float32)
        self.c = generator.normal(size=shape).astype(np.float32)
        for value in (self.a, self.b, self.c):
            value[..., 2] = 0.0
        baseline_weights = np.full(shape[0], subject.BASE_WEIGHT)
        self.baseline = subject.blend_sample_weights(
            self.a, self.b, self.c, baseline_weights, baseline_weights)

    def test_scalar_and_four_block_candidates_keep_fixed_temporal_mean(
            self) -> None:
        scalar = subject.ab_headroom_prediction(
            self.a, self.b, self.baseline,
            np.asarray([0.25, 0.5, 0.75]))
        blocks = subject.ab_headroom_prediction(
            self.a, self.b, self.baseline,
            np.asarray([
                [0.25, 0.50, 0.75, 0.25],
                [0.75, 0.25, 0.50, 0.75],
                [0.50, 0.50, 0.50, 0.50],
            ]))
        fixed_mean = self.baseline.mean(axis=1, keepdims=True)

        np.testing.assert_allclose(
            scalar.mean(axis=1, keepdims=True), fixed_mean,
            rtol=0.0, atol=subject.AB_HEADROOM_FIXED_MEAN_TOLERANCE)
        np.testing.assert_allclose(
            blocks.mean(axis=1, keepdims=True), fixed_mean,
            rtol=0.0, atol=subject.AB_HEADROOM_FIXED_MEAN_TOLERANCE)
        self.assertTrue(np.array_equal(scalar[1], self.baseline[1]))
        self.assertTrue(np.array_equal(blocks[2], self.baseline[2]))

    def test_four_contiguous_five_step_blocks_are_applied_exactly(self) -> None:
        weights = np.asarray([
            [0.25, 0.375, 0.625, 0.75],
            [0.75, 0.625, 0.375, 0.25],
            [0.25, 0.50, 0.75, 0.50],
        ], dtype=np.float32)
        actual = subject.ab_headroom_prediction(
            self.a, self.b, self.baseline, weights)
        a_fluctuation = self.a - self.a.mean(axis=1, keepdims=True)
        b_fluctuation = self.b - self.b.mean(axis=1, keepdims=True)
        baseline_mean = self.baseline.mean(axis=1, keepdims=True)
        expected_fluctuation = self.baseline - baseline_mean
        for block, (start, stop) in enumerate(
                subject.AB_HEADROOM_HORIZON_BLOCKS):
            expected_fluctuation[:, start:stop] += (
                (weights[:, block, None, None, None, None] - 0.5)
                * (b_fluctuation[:, start:stop]
                   - a_fluctuation[:, start:stop]))
        expected_fluctuation -= expected_fluctuation.mean(
            axis=1, keepdims=True)
        expected = baseline_mean + expected_fluctuation

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_block_oracle_falls_back_to_exact_baseline(self) -> None:
        times = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
        baseline = np.zeros((2, 20, 1, 1, 3), dtype=np.float32)
        baseline[..., 0] = 2.0 + np.sin(times)[None, :, None, None]
        baseline[..., 1] = 3.0 + np.cos(times)[None, :, None, None]
        lower = baseline - 0.5
        upper = baseline + 0.5
        target = baseline.copy()
        initial = np.full((2, 4), subject.BASE_WEIGHT)

        weights, optimizer = subject.bounded_ab_target_oracle(
            baseline, baseline, baseline, lower, upper, target,
            horizon_blocks=4, initial_weights=initial)
        candidate = subject.ab_headroom_prediction(
            baseline, baseline, baseline, weights)

        self.assertTrue(np.array_equal(weights, initial))
        self.assertTrue(np.array_equal(candidate, baseline))
        self.assertEqual(optimizer["samples_using_baseline_fallback"], 2)

    def test_decision_contract_boundaries(self) -> None:
        self.assertEqual(
            subject.ab_headroom_decision(0.149999, 0.0),
            "STOP_AB_ROUTING_NO_HEADROOM")
        self.assertEqual(
            subject.ab_headroom_decision(0.15, 0.0),
            "STOP_AB_ROUTING_INSUFFICIENT_FOR_PGR")
        self.assertEqual(
            subject.ab_headroom_decision(0.199999, 0.0),
            "STOP_AB_ROUTING_INSUFFICIENT_FOR_PGR")
        self.assertEqual(
            subject.ab_headroom_decision(0.20, -0.030001), "STOP_TKE")
        self.assertEqual(
            subject.ab_headroom_decision(0.20, -0.03),
            "GO_AB_ROUTER_CROSSFIT")
        self.assertEqual(
            subject.ab_headroom_decision(
                0.25, 0.0, within_budget=False),
            "STOP_AB_ROUTING_TIME_BUDGET")

    def test_source_matched_reconstruction_gate_precedes_oracles(self) -> None:
        bad_metrics = dict(subject.P1_SOURCE_MATCHED_OOF_REFERENCE)
        bad_metrics["sps_score"] += 2.0 * subject.P1_POINT_TOLERANCE
        lower = np.full_like(self.baseline, -10.0)
        upper = np.full_like(self.baseline, 10.0)
        target = self.baseline.copy()

        with mock.patch.object(subject, "score", return_value=bad_metrics), \
                mock.patch.object(
                    subject, "bounded_ab_target_oracle",
                    side_effect=AssertionError("oracle must not run")):
            result = subject.evaluate_ab_headroom(
                self.a, self.b, self.c, lower, upper, target)

        self.assertEqual(
            result["decision"], "STOP_P1_SOURCE_MATCHED_RECONSTRUCTION")
        self.assertFalse(result["p1_reconstruction_control"]["pass"])


class RidgeAndGateTest(unittest.TestCase):
    def test_shadow_mapping_and_official_time_sps_pair(self) -> None:
        expected = {
            "rel_l2_score": 0.47201126,
            "tke_score": 0.08586562,
            "mvpe_score": 0.09410214,
            "time_score": 0.10102825,
            "sps_score": 0.24699273,
        }
        self.assertEqual(subject.FINAL_SHADOW_DERIVATIVES, expected)
        self.assertAlmostEqual(sum(expected.values()), 1.0, places=12)

        sharp2 = {
            "rel_l2_score": 94.482163,
            "tke_score": 78.396313,
            "mvpe_score": 93.662972,
            "time_score": 87.333962,
            "sps_score": 41.663108,
        }
        utility_refinement = {
            "rel_l2_score": 94.482163,
            "tke_score": 78.396313,
            "mvpe_score": 93.662972,
            "time_score": 87.313190,
            "sps_score": 41.716403,
        }
        predicted_delta = sum(
            expected[key] * (utility_refinement[key] - sharp2[key])
            for key in expected)
        official_delta = 80.538866 - 80.527779
        self.assertLess(abs(predicted_delta - official_delta), 5e-4)

    def test_crossfit_is_finite_and_bounded(self) -> None:
        generator = np.random.default_rng(3)
        features = generator.normal(size=(25, len(subject.FEATURE_NAMES)))
        groups = np.repeat(np.arange(5), 5)
        normalized = np.column_stack((
            0.2 * features[:, 0] - 0.1 * features[:, 2],
            -0.15 * features[:, 1] + 0.05 * features[:, 3],
        ))
        labels = 0.5 + 0.25 * np.clip(normalized, -1.0, 1.0)

        prediction, models = subject.crossfit_weights(
            features, labels, groups, tuple(range(5)))

        self.assertEqual(prediction.shape, labels.shape)
        self.assertEqual(len(models), 5)
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all(prediction >= subject.MIN_WEIGHT))
        self.assertTrue(np.all(prediction <= subject.MAX_WEIGHT))

    def test_shadow_uses_only_frozen_four_metric_derivatives(self) -> None:
        baseline = {key: 10.0 for key in subject.SHADOW_DERIVATIVES}
        candidate = {key: 11.0 for key in subject.SHADOW_DERIVATIVES}

        actual = subject.shadow_delta(baseline, candidate)

        self.assertAlmostEqual(
            actual["shadow_final_gain"],
            sum(subject.SHADOW_DERIVATIVES.values()))
        self.assertEqual(actual["time_score_delta_assumed"], 0.0)

    def test_reconstruction_gate_uses_source_matched_oof_and_reports_drift(
            self) -> None:
        actual = dict(subject.P1_SOURCE_MATCHED_OOF_REFERENCE)
        actual["point_score"] += 0.9 * subject.P1_POINT_TOLERANCE

        control = subject.p1_reconstruction_control(actual)

        self.assertTrue(control["pass"])
        self.assertEqual(
            control["source_matched_result_sha256"],
            subject.P1_SOURCE_MATCHED_OOF_RESULT_SHA256)
        self.assertFalse(
            control["deployment_reference_drift_only"]["used_as_gate"])
        self.assertIn(
            "sps_score", control["source_matched_absolute_differences"])

        actual["point_score"] += 0.2 * subject.P1_POINT_TOLERANCE
        self.assertFalse(subject.p1_reconstruction_control(actual)["pass"])

    def test_bounds_risk_reports_new_crossing_without_clipping(self) -> None:
        target = np.ones((1, 6, 2, 2, 3), dtype=np.float32)
        lower = np.zeros_like(target)
        upper = np.full_like(target, 2.0)
        baseline = np.ones_like(target)
        candidate = baseline.copy()
        candidate[0, 0, 0, 0, 0] = 3.0

        risk = subject.bounds_risk(
            baseline, candidate, lower, upper, target)

        self.assertTrue(risk["baseline_all_elements_within_bounds"])
        self.assertFalse(risk["candidate_all_elements_within_bounds"])
        self.assertFalse(risk["conditional_within_bounds_contract_pass"])
        self.assertGreater(risk["new_scored_outside_rate"], 0.0)
        self.assertFalse(risk["point_clipping_used"])


class ArchiveContractTest(unittest.TestCase):
    def _write_archive(self, path: Path, config: dict) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("blend_config.json", json.dumps(config))
            for member in ("model_a.pth", "model_b.pth", "model_c.pth"):
                archive.writestr(member, f"same-{member}".encode())

    def test_decoupled_p1_fields_and_checkpoint_members_are_accepted(self) -> None:
        config = {
            "point_weight_b": 0.5,
            "point_mean_expert_weight": 0.5,
            "bounds_weight_b": 0.625,
            "bounds_mean_expert_weight": 0.25,
            "source_archive_sha256": subject.P0_ARCHIVE_SHA256,
            "checkpoints": ["model_a.pth", "model_b.pth", "model_c.pth"],
            "old_extra_key": "allowed because the whole ZIP SHA is pinned",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p1.zip"
            self._write_archive(path, config)
            expected_sha = s1.file_identity(path)["sha256"]

            identity = subject.validate_archive_blend(
                path, expected_sha, 0.5, 0.5, decoupled_p1=True)

        self.assertEqual(identity["blend_config"], config)
        self.assertEqual(
            set(identity["checkpoint_members"]),
            {"model_a.pth", "model_b.pth", "model_c.pth"})

    def test_decoupled_p1_wrong_bounds_are_rejected(self) -> None:
        config = {
            "point_weight_b": 0.5,
            "point_mean_expert_weight": 0.5,
            "bounds_weight_b": 0.5,
            "bounds_mean_expert_weight": 0.25,
            "source_archive_sha256": subject.P0_ARCHIVE_SHA256,
            "checkpoints": ["model_a.pth", "model_b.pth", "model_c.pth"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.zip"
            self._write_archive(path, config)
            expected_sha = s1.file_identity(path)["sha256"]
            with self.assertRaisesRegex(ValueError, "blend_config"):
                subject.validate_archive_blend(
                    path, expected_sha, 0.5, 0.5, decoupled_p1=True)


if __name__ == "__main__":
    unittest.main()
