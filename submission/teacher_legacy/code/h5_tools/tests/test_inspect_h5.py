import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "code" / "h5_tools" / "inspect_h5.py"
BUNDLED_TEST_DATA = (
    ROOT
    / "kits"
    / "track2"
    / "realpde_t2_starting_kit_v6"
    / "example_data"
    / "test_real"
)
SAMPLE_PATH = BUNDLED_TEST_DATA / "2968_0.h5"
SECOND_SAMPLE_PATH = BUNDLED_TEST_DATA / "5025_5.h5"


def load_module():
    spec = importlib.util.spec_from_file_location("inspect_h5", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InspectH5Tests(unittest.TestCase):
    def test_describe_h5_lists_velocity_shape_and_render_frame_writes_png(self):
        module = load_module()
        summary = module.describe_h5(SAMPLE_PATH)
        self.assertIn("u", summary)
        self.assertEqual(summary["u"]["shape"][1:], (64, 128))

        output = ROOT / "runs" / "figures" / "test_real_3750_0_u0.png"
        if output.exists():
            output.unlink()
        rendered = module.render_frame(SAMPLE_PATH, "u", 0, output)
        self.assertEqual(rendered, output)
        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 0)

    def test_real_sim_triptych_writes_a_six_panel_png(self):
        module = load_module()
        output = ROOT / "runs" / "figures" / "test_real_sim_triptych.png"
        if output.exists():
            output.unlink()

        rendered = module.render_real_sim_triptych(
            SAMPLE_PATH, SECOND_SAMPLE_PATH, "u", output
        )
        self.assertEqual(rendered, output)
        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 0)

    def test_real_sim_triptych_supports_per_source_scaling(self):
        module = load_module()
        output = ROOT / "runs" / "figures" / "test_real_sim_triptych_per_source.png"
        if output.exists():
            output.unlink()

        rendered = module.render_real_sim_triptych(
            SAMPLE_PATH, SECOND_SAMPLE_PATH, "u", output, scale_mode="per_source"
        )
        self.assertEqual(rendered, output)
        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
