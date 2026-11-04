from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from evaluate_s1_zip_point_bounds import (  # noqa: E402
    EVIDENCE_CLASS,
    P0_EXPECTED_METRICS,
    assert_p0_matches_submission,
    bounds_reuse_evidence,
    compose_packaged_candidate,
    compose_packaged_point,
    finalize_stream_accumulator,
    finalize_stream_identity,
    locked_identity,
    new_stream_accumulator,
    new_stream_identity,
    point_outside_bounds_rate,
    score_candidate,
    update_stream_accumulator,
    update_stream_identity,
    validate_archive_config,
    validate_packaged_batch,
    validate_p0_metrics,
)
from evaluate_mean_expert_oof_blend import array_identity  # noqa: E402


class S1ZipPointBoundsTests(unittest.TestCase):
    def test_evidence_class_is_explicitly_post_selection(self):
        self.assertEqual(
            EVIDENCE_CLASS,
            "deployment_zip_fixed_tail_post_selection_diagnostic")

    def test_packaged_candidate_matches_deployment_operation_order(self):
        prediction_a = torch.tensor([[[[[1.0]]], [[[3.0]]]]], dtype=torch.float16)
        prediction_b = torch.tensor([[[[[5.0]]], [[[7.0]]]]], dtype=torch.float16)
        prediction_c = torch.tensor([[[[[9.0]]], [[[13.0]]]]], dtype=torch.float16)
        distribution_a = (
            prediction_a, prediction_a - 2.0, prediction_a + 2.0)
        distribution_b = (
            prediction_b, prediction_b - 2.0, prediction_b + 2.0)

        actual = compose_packaged_candidate(
            distribution_a, distribution_b, prediction_c, 0.625, 0.25)
        base = tuple(
            left.lerp(right, 0.625)
            for left, right in zip(distribution_a, distribution_b))
        correction = (
            prediction_c.mean(dim=1, keepdim=True)
            - base[0].mean(dim=1, keepdim=True)).mul(0.25)
        expected = tuple(value + correction for value in base)
        for left, right in zip(actual, expected):
            self.assertTrue(torch.equal(left, right))
        point_only = compose_packaged_point(
            prediction_a, prediction_b, prediction_c, 0.625, 0.25)
        self.assertTrue(torch.equal(point_only, actual[0]))

    def test_stream_identity_matches_concatenated_array(self):
        batches = [
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(6, 12, dtype=np.float32).reshape(2, 3),
        ]
        state = new_stream_identity()
        for batch in batches:
            update_stream_identity(state, batch)
        self.assertEqual(
            finalize_stream_identity(state),
            array_identity(np.concatenate(batches)))

    def test_stream_metrics_match_full_official_scoring(self):
        rng = np.random.default_rng(7)
        shape = (10, 20, 20, 40, 3)
        target = rng.normal(0.1, 0.03, size=shape).astype(np.float32)
        target[..., 2] = 0.0
        prediction = target + rng.normal(
            0.0, 0.01, size=shape).astype(np.float32)
        prediction[..., 2] = 0.0
        lower = prediction - np.float32(0.025)
        upper = prediction + np.float32(0.025)
        folds = np.arange(len(target)) % 5
        reference = score_candidate(
            prediction, lower, upper, target, folds)

        accumulator = new_stream_accumulator()
        for start, stop in ((0, 4), (4, 8), (8, 10)):
            update_stream_accumulator(
                accumulator, prediction[start:stop], lower[start:stop],
                upper[start:stop], target[start:stop], folds[start:stop])
        streamed = finalize_stream_accumulator(accumulator)
        keys = (
            "rel_l2_score", "tke_score", "mvpe_score", "sps_score",
            "coverage", "mean_width", "point_score",
            "point_outside_bounds_rate",
        )
        for key in keys:
            self.assertAlmostEqual(streamed[key], reference[key], delta=2e-6)
        for streamed_fold, reference_fold in zip(
                streamed["folds"], reference["folds"]):
            for key in keys:
                self.assertAlmostEqual(
                    streamed_fold[key], reference_fold[key], delta=2e-6)

    def test_p0_must_match_submission_bitwise(self):
        packaged = tuple(
            np.asarray([value], dtype=np.float32) for value in (1.0, 0.0, 2.0))
        assert_p0_matches_submission(packaged, packaged)
        changed = list(packaged)
        changed[0] = np.asarray([1.0001], dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "differs bitwise"):
            assert_p0_matches_submission(tuple(changed), packaged)

    def test_p0_frozen_metric_gate(self):
        validate_p0_metrics(dict(P0_EXPECTED_METRICS))
        failed = dict(P0_EXPECTED_METRICS)
        failed["sps_score"] += 0.001
        with self.assertRaisesRegex(RuntimeError, "metric gate failed"):
            validate_p0_metrics(failed)

    def test_archive_config_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "weight_b": 0.625,
                "mean_expert_weight": 0.25,
                "checkpoints": ["model_a.pth", "model_b.pth", "model_c.pth"],
            }
            (root / "blend_config.json").write_text(
                __import__("json").dumps(config), encoding="utf-8")
            self.assertEqual(validate_archive_config(root), config)
            config["weight_b"] = 0.5
            (root / "blend_config.json").write_text(
                __import__("json").dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not P0"):
                validate_archive_config(root)

    def test_bounds_reuse_requires_shared_read_only_arrays(self):
        lower = np.asarray([0.0], dtype=np.float32)
        upper = np.asarray([1.0], dtype=np.float32)
        lower.setflags(write=False)
        upper.setflags(write=False)
        shared = bounds_reuse_evidence(lower, upper, lower, upper)
        self.assertTrue(all(shared.values()))
        copied = bounds_reuse_evidence(lower, upper, lower.copy(), upper.copy())
        self.assertFalse(copied["same_objects_as_control"])
        self.assertTrue(copied["bitwise_equal_to_control"])

    def test_point_outside_bounds_rate_uses_official_nonzero_mask(self):
        target = np.array([[[[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]]],
                          dtype=np.float32)
        lower = np.zeros_like(target)
        upper = np.ones_like(target)
        prediction = np.array(
            [[[[[2.0, 9.0, 0.0], [0.5, -1.0, 0.0]]]]], dtype=np.float32)
        self.assertEqual(
            point_outside_bounds_rate(prediction, lower, upper, target),
            2.0 / 3.0)

    def test_packaged_batch_validation_is_fail_closed(self):
        shape = (1, 20, 2, 2, 3)
        target = np.ones(shape, dtype=np.float32)
        prediction = np.ones(shape, dtype=np.float32)
        lower = np.zeros(shape, dtype=np.float32)
        upper = np.full(shape, 2.0, dtype=np.float32)
        validate_packaged_batch({"P0": prediction}, lower, upper, target)
        invalid = prediction.copy()
        invalid.flat[0] = np.inf
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            validate_packaged_batch({"P1": invalid}, lower, upper, target)
        reversed_upper = upper.copy()
        reversed_upper.flat[0] = -1.0
        with self.assertRaisesRegex(FloatingPointError, "reversed"):
            validate_packaged_batch(
                {"P0": prediction}, lower, reversed_upper, target)

    def test_locked_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"frozen")
            expected = (
                "ffb304816a1090313e833215c08dae3d209cfad1ffd1f674f0909a2ae99e1394")
            self.assertEqual(
                locked_identity(path, expected, "artifact")["sha256"],
                expected)
            with self.assertRaisesRegex(ValueError, "differs from the contract"):
                locked_identity(path, "0" * 64, "artifact")


if __name__ == "__main__":
    unittest.main()
