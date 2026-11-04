from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "seal_b210_xcb_pw12_candidate.py"
SPEC = importlib.util.spec_from_file_location("xcb_pw12_seal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
seal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seal)

BUILDER_SCRIPT = (
    Path(__file__).resolve().parents[2] / "submissions/build_track1_xcb_pw12.py"
)
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "xcb_pw12_builder", BUILDER_SCRIPT
)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
builder = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(builder)


class SealMathTests(unittest.TestCase):
    def test_tracked_wrapper_patches_to_exact_candidate_identity(self):
        wrapper = (
            Path(__file__).resolve().parents[2]
            / "submissions/track1_ensemble_unet2d.py"
        )
        source = wrapper.read_bytes()
        self.assertEqual(builder.sha256_bytes(source), builder.SOURCE_WRAPPER_SHA256)
        patched = builder.patch_wrapper(source)
        self.assertEqual(
            builder.sha256_bytes(patched), builder.CANDIDATE_WRAPPER_SHA256
        )
        self.assertNotIn(b"torch.backends.cudnn.benchmark = True", patched)

    def test_generalized_final_reproduces_official_baseline(self):
        self.assertAlmostEqual(
            seal.generalized_final(seal.OFFICIAL_BASELINE),
            seal.OFFICIAL_BASELINE["final_score"],
            delta=2e-10,
        )

    def test_a800_ratios_preserve_goal_but_not_time_only_gate(self):
        deltas = dict(seal.EXPECTED_LOCAL_DELTAS)
        full = seal.projected_scenario(deltas, 0.9754634855588676)
        conservative = seal.projected_scenario(deltas, 1.1194985712065402)
        self.assertAlmostEqual(full["inferred_final"], 81.5445617543, delta=1e-10)
        self.assertAlmostEqual(full["inferred_gain"], 0.1913651203, delta=1e-10)
        self.assertAlmostEqual(
            conservative["inferred_final"], 81.4875180291, delta=1e-10
        )
        self.assertAlmostEqual(
            conservative["inferred_gain"], 0.1343213951, delta=1e-10
        )
        self.assertGreater(1.1194985712065402, 1.0)
        self.assertGreaterEqual(
            conservative["inferred_gain"], seal.MIN_PROJECTED_GAIN
        )

    def test_pair_spread_gate(self):
        self.assertAlmostEqual(
            seal.pair_spread([2.89486267976463, 2.91786848381162]),
            0.00794711410935009,
        )
        self.assertAlmostEqual(
            seal.pair_spread([2.82560952939093, 2.84449747204781]),
            0.00668455512356161,
        )
        with self.assertRaises(ValueError):
            seal.pair_spread([1.0])


if __name__ == "__main__":
    unittest.main()
