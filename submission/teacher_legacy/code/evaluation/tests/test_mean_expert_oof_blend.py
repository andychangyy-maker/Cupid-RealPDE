from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from evaluate_mean_expert_oof_blend import (  # noqa: E402
    S1_BASELINE_ZIP_SHA256,
    S1_FOLD_COUNTS,
    S1_SPLIT_MANIFEST_SHA256,
    artifact_identities,
    array_identity,
    blend_candidate,
    blend_point_candidate,
    point_bounds_diagnostic,
    resolve_candidates,
    validate_s1_contract,
    validate_s1_split,
    verify_artifact_manifest,
    write_artifact_manifest,
)


class MeanExpertOOFBlendTests(unittest.TestCase):
    def test_resolve_preregistered_paired_candidates(self):
        args = Namespace(
            point_candidate=["P0:0.625:0.25", "P1:0.5:0.5"],
            weights_b=None,
            mean_expert_weights=None,
            fixed_bounds_candidate="P0",
        )
        candidates = resolve_candidates(args)
        self.assertEqual([candidate["name"] for candidate in candidates], [
            "P0", "P1",
        ])
        self.assertEqual(candidates[0]["weight_b"], 0.625)
        self.assertEqual(candidates[1]["mean_expert_weight_c"], 0.5)

    def test_fixed_bounds_contract_is_exact(self):
        args = Namespace(
            point_candidate=[
                "P0:0.625:0.25", "P1:0.5:0.5", "P2:0.25:0.5"],
            weights_b=None,
            mean_expert_weights=None,
            fixed_bounds_candidate="P0",
            folds=5,
            window_stride=20,
            val_fraction=0.2,
            max_files=0,
            device="cuda:0",
            baseline_zip=Path("baseline.zip"),
            expected_baseline_zip_sha256=S1_BASELINE_ZIP_SHA256,
            split_manifest=Path("split_manifest.json"),
            expected_split_manifest_sha256=S1_SPLIT_MANIFEST_SHA256,
            artifact_manifest=Path("artifacts.json"),
            freeze_artifact_manifest=None,
        )
        candidates = resolve_candidates(args)
        validate_s1_contract(args, candidates)
        candidates[-1]["weight_b"] = 0.375
        with self.assertRaisesRegex(ValueError, "exact candidates"):
            validate_s1_contract(args, candidates)

    def test_fixed_split_requires_725_and_exact_fold_counts(self):
        sample_folds = [
            fold for fold, count in enumerate(S1_FOLD_COUNTS)
            for _ in range(count)
        ]
        manifest = [{"file": "f.h5"}]
        validate_s1_split(manifest, list(manifest), 725, sample_folds)
        with self.assertRaisesRegex(RuntimeError, "tail/fold counts"):
            validate_s1_split(manifest, list(manifest), 724, sample_folds)

    def test_artifact_manifest_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "checkpoint.pth"
            artifact.write_bytes(b"frozen")
            identities = artifact_identities({"checkpoint": artifact})
            manifest = root / "artifacts.json"
            write_artifact_manifest(manifest, identities)
            self.assertEqual(
                len(verify_artifact_manifest(manifest, identities)["sha256"]),
                64)
            artifact.write_bytes(b"changed")
            changed = artifact_identities({"checkpoint": artifact})
            with self.assertRaisesRegex(ValueError, "differ"):
                verify_artifact_manifest(manifest, changed)

    def test_blend_candidate_applies_same_mean_correction_to_endpoints(self):
        shape = (2, 3, 1, 1, 1)
        arrays_a = {
            "prediction": np.zeros(shape, dtype=np.float32),
            "lower": np.full(shape, -2.0, dtype=np.float32),
            "upper": np.full(shape, 2.0, dtype=np.float32),
        }
        arrays_b = {
            "prediction": np.full(shape, 2.0, dtype=np.float32),
            "lower": np.zeros(shape, dtype=np.float32),
            "upper": np.full(shape, 4.0, dtype=np.float32),
        }
        prediction_c = np.full(shape, 4.0, dtype=np.float32)
        result = blend_candidate(
            arrays_a, arrays_b, prediction_c, weight_b=0.5,
            mean_weight=0.25)
        np.testing.assert_array_equal(result["prediction"], 1.75)
        np.testing.assert_array_equal(result["lower"], -0.25)
        np.testing.assert_array_equal(result["upper"], 3.75)
        point_only = blend_point_candidate(
            arrays_a["prediction"], arrays_b["prediction"], prediction_c,
            weight_b=0.5, mean_weight=0.25)
        np.testing.assert_array_equal(point_only, result["prediction"])
        identity = array_identity(result["lower"])
        self.assertEqual(identity["shape"], list(shape))
        self.assertEqual(identity["dtype"], "float32")
        self.assertEqual(len(identity["sha256"]), 64)

    @staticmethod
    def result(point, sps, fold_deltas, *, coverage=0.9, width=0.02):
        return {
            "point_score": point,
            "sps_score": sps,
            "coverage": coverage,
            "mean_width": width,
            "bounds_equal_to_control": True,
            "folds": [
                {"point_score": 10.0 + delta}
                for delta in fold_deltas
            ],
        }

    def test_diagnostic_uses_all_gates_and_preregistered_tie_break(self):
        results = {
            "P0": self.result(90.0, 53.0, [0.0] * 5),
            "P1": self.result(
                90.0108, 53.005, [0.01, 0.01, 0.01, 0.01, -0.01]),
            "P2": self.result(
                90.0100, 53.020, [0.01, 0.01, 0.01, 0.01, -0.01]),
        }
        diagnostic = point_bounds_diagnostic(results, "P0")
        self.assertTrue(diagnostic["candidates"]["P1"]["pass"])
        self.assertTrue(diagnostic["candidates"]["P2"]["pass"])
        self.assertEqual(diagnostic["selected_point_teacher"], "P2")
        self.assertEqual(diagnostic["decision"], "PASS")

        results["P1"] = self.result(90.01, 53.02, [0.01] * 5)
        results["P2"] = self.result(90.01, 53.02, [0.01] * 5)
        exact_tie = point_bounds_diagnostic(results, "P0")
        self.assertEqual(exact_tie["selected_point_teacher"], "P1")

    def test_diagnostic_stops_when_fixed_bounds_invariants_change(self):
        results = {
            "P0": self.result(90.0, 53.0, [0.0] * 5),
            "P1": self.result(
                90.02, 53.02, [0.01] * 5, coverage=0.91),
        }
        diagnostic = point_bounds_diagnostic(results, "P0")
        self.assertFalse(diagnostic["candidates"]["P1"]["pass"])
        self.assertEqual(diagnostic["decision"], "STOP_KEEP_CONTROL")


if __name__ == "__main__":
    unittest.main()
