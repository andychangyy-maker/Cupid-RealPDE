from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_b210_p1_late_release import p1_weight, wrap_update


class P1InterventionTest(unittest.TestCase):
    def test_shared_prefix_and_fixed_terminal_interventions(self):
        for step in range(1, 5001):
            self.assertEqual(p1_weight("keep", step), p1_weight("off", step))
            self.assertEqual(p1_weight("keep", step), p1_weight("anneal", step))
        self.assertEqual(p1_weight("keep", 10000), .005)
        self.assertEqual(p1_weight("off", 5001), 0.)
        self.assertEqual(p1_weight("anneal", 7500), .0025)
        self.assertEqual(p1_weight("anneal", 10000), 0.)

    def test_coefficient_reaches_update_without_mutating_shared_config(self):
        config = SimpleNamespace(distill_weight=.005)
        seen = []

        def original(args, iteration, marker):
            seen.append((args.distill_weight, iteration, marker))
            return ({}, None, "key", {})

        wrapped = wrap_update(original, "off")
        result = wrapped(config, 5001, marker="same_batch")
        self.assertEqual(seen, [(0., 5001, "same_batch")])
        self.assertEqual(result[0]["effective_p1_distill_weight"], 0.)
        self.assertEqual(config.distill_weight, .005)

    def test_failed_update_restores_original_config(self):
        config = SimpleNamespace(distill_weight=.005)

        def original(args, iteration):
            self.assertEqual(args.distill_weight, 0.)
            raise RuntimeError("training failure")

        with self.assertRaisesRegex(RuntimeError, "training failure"):
            wrap_update(original, "off")(config, 8000)
        self.assertEqual(config.distill_weight, .005)


if __name__ == "__main__":
    unittest.main()
