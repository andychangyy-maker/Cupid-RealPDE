import copy
import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "code" / "tools" / "render_project_status.py"
STATUS_PATH = ROOT / "docs" / "status.json"


def load_module():
    spec = importlib.util.spec_from_file_location("render_project_status", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProjectStatusValidationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.valid = json.loads(STATUS_PATH.read_text(encoding="utf-8"))

    def test_repository_status_is_valid(self):
        loaded = self.module.load_status(STATUS_PATH)
        self.module.validate_status(loaded)
        self.assertEqual(
            loaded["tracks"]["track1"]["final_score_rule"],
            "leaderboard_unpublished",
        )
        self.assertAlmostEqual(
            loaded["tracks"]["track1"]["final_score"], 81.0070113816, places=6
        )

    def test_rejects_final_score_that_is_not_metric_mean(self):
        broken = copy.deepcopy(self.valid)
        broken["tracks"]["track2"]["final_score"] = 75.0
        with self.assertRaisesRegex(self.module.StatusError, "final_score"):
            self.module.validate_status(broken)

    def test_unpublished_final_rule_allows_non_mean_but_rejects_unknown_rule(self):
        allowed = copy.deepcopy(self.valid)
        allowed["tracks"]["track1"]["final_score"] = 1.0
        self.module.validate_status(allowed)

        broken = copy.deepcopy(self.valid)
        broken["tracks"]["track1"]["final_score_rule"] = "guess"
        with self.assertRaisesRegex(self.module.StatusError, "final_score_rule"):
            self.module.validate_status(broken)

    def test_rejects_bad_sha_and_out_of_range_metric(self):
        broken_sha = copy.deepcopy(self.valid)
        broken_sha["tracks"]["track1"]["artifact"]["sha256"] = "bad"
        with self.assertRaisesRegex(self.module.StatusError, "sha256"):
            self.module.validate_status(broken_sha)

        broken_score = copy.deepcopy(self.valid)
        broken_score["tracks"]["track1"]["metrics"]["sps"] = 101.0
        with self.assertRaisesRegex(self.module.StatusError, "0..100"):
            self.module.validate_status(broken_score)

    def test_rejects_missing_required_track(self):
        broken = copy.deepcopy(self.valid)
        del broken["tracks"]["track2"]
        with self.assertRaisesRegex(self.module.StatusError, "track2"):
            self.module.validate_status(broken)

    def test_rejects_nonstandard_json_numeric_constants(self):
        original = STATUS_PATH.read_text(encoding="utf-8")
        current_score = f'{self.valid["tracks"]["track1"]["final_score"]:.6f}'
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "status.json"
            for constant in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(constant=constant):
                    status_path.write_text(
                        original.replace(current_score, constant, 1),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(self.module.StatusError, "JSON constant"):
                        self.module.load_status(status_path)

    def test_rejects_nonfinite_and_ill_typed_scores_as_status_errors(self):
        invalid_values = (float("nan"), float("inf"), "NaN", "Infinity", "bad", True)
        for field in ("final_score", "metrics"):
            for value in invalid_values:
                with self.subTest(field=field, value=repr(value)):
                    broken = copy.deepcopy(self.valid)
                    if field == "final_score":
                        broken["tracks"]["track1"][field] = value
                    else:
                        broken["tracks"]["track1"][field]["rel_l2"] = value
                    with self.assertRaises(self.module.StatusError):
                        self.module.validate_status(broken)

    def test_rejects_wrong_container_types_as_status_errors(self):
        invalid_data = []
        with self.assertRaises(self.module.StatusError):
            self.module.validate_status(invalid_data)

        for location, value in (
            (("tracks",), []),
            (("tracks", "track1"), []),
            (("tracks", "track1", "metrics"), []),
            (("tracks", "track1", "artifact"), []),
        ):
            with self.subTest(location=location):
                broken = copy.deepcopy(self.valid)
                target = broken
                for key in location[:-1]:
                    target = target[key]
                target[location[-1]] = value
                with self.assertRaises(self.module.StatusError):
                    self.module.validate_status(broken)

    def test_requires_a_nonempty_milestone_optimization_status(self):
        for value in (None, "", [], True):
            with self.subTest(value=repr(value)):
                broken = copy.deepcopy(self.valid)
                broken["milestone"]["optimization_status"] = value
                with self.assertRaisesRegex(
                    self.module.StatusError, "optimization_status"
                ):
                    self.module.validate_status(broken)


class ProjectStatusMarkdownTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.status = self.module.load_status(STATUS_PATH)

    def test_replace_marked_section_preserves_human_text(self):
        source = (
            "# Header\n\n"
            "<!-- PROJECT_STATUS:START -->\nold\n"
            "<!-- PROJECT_STATUS:END -->\n\n"
            "Human notes.\n"
        )
        rendered = self.module.replace_marked_section(
            source,
            "<!-- PROJECT_STATUS:START -->",
            "<!-- PROJECT_STATUS:END -->",
            "new",
        )
        self.assertIn("\nnew\n", rendered)
        self.assertIn("Human notes.", rendered)
        self.assertNotIn("\nold\n", rendered)

    def test_missing_markers_are_rejected(self):
        with self.assertRaisesRegex(self.module.StatusError, "marker"):
            self.module.replace_marked_section(
                "# no markers\n",
                "<!-- PROJECT_STATUS:START -->",
                "<!-- PROJECT_STATUS:END -->",
                "new",
            )

    def test_readme_status_contains_scores_and_context(self):
        rendered = self.module.render_readme_status(self.status)
        for track in self.status["tracks"].values():
            self.assertIn(f'{track["final_score"]:.6f}', rendered)
            self.assertIn(str(track["submission_id"]), rendered)
        self.assertIn(self.status["milestone"]["note"], rendered)
        self.assertIn("不能直接横向比较", rendered)

    def test_results_summary_contains_all_five_metrics(self):
        rendered = self.module.render_results_summary(self.status)
        for label in ("Rel-L2", "TKE", "MVPE", "Time", "SPS"):
            self.assertIn(label, rendered)
        self.assertIn("Main Development", rendered)
        self.assertIn("CodaBench", rendered)

    def test_rendered_facts_follow_mutated_status_data(self):
        changed = copy.deepcopy(self.status)
        changed["milestone"].update(
            {
                "id": "M1",
                "name": "Adversarial milestone name",
                "note": "Adversarial milestone note",
                "date": "2030-01-02",
                "optimization_status": "Adversarial optimization status",
            }
        )
        changed["autodl"]["gpu"] = "Adversarial GPU"
        changed["tracks"]["track1"].update(
            {"method": "Adversarial Track 1 method", "phase": "Track 1 phase"}
        )
        changed["tracks"]["track2"].update(
            {"method": "Adversarial Track 2 method", "phase": "Track 2 phase"}
        )
        self.module.validate_status(changed)

        readme = self.module.render_readme_status(changed)
        results = self.module.render_results_summary(changed)
        svg = self.module.render_scorecard_svg(changed)

        for expected in (
            "M1",
            "Adversarial milestone name",
            "Adversarial milestone note",
            "2030-01-02",
            "Adversarial GPU",
            "Adversarial Track 1 method",
            "Adversarial Track 2 method",
            "Track 1 phase",
            "Track 2 phase",
            "Adversarial optimization status",
        ):
            self.assertIn(expected, readme)
        self.assertIn(
            "![RealPDE M1 Adversarial milestone name 成绩卡](docs/scorecard.svg)",
            readme,
        )

        for expected in (
            "Adversarial Track 1 method",
            "Adversarial Track 2 method",
            "Track 1 phase",
            "Track 2 phase",
        ):
            self.assertIn(expected, results)

        for expected in (
            "M1",
            "Adversarial milestone name",
            "Adversarial milestone note",
            "2030-01-02",
            "Adversarial GPU",
            "Adversarial Track 1 method",
            "Adversarial Track 2 method",
            "Track 1 phase",
            "Track 2 phase",
        ):
            self.assertIn(expected, svg)
        self.assertIn(
            '<title id="scorecard-title">RealPDE M1 Adversarial milestone name scores</title>',
            svg,
        )
        self.assertIn(
            "Adversarial milestone note · Scores are not directly comparable across tracks",
            svg,
        )

        for old_value in (
            "M0",
            self.status["milestone"]["name"],
            self.status["milestone"]["note"],
            "团队优化尚未开始",
            "![RealPDE双赛道首个官方baseline成绩卡]",
        ):
            self.assertNotIn(old_value, readme)
        self.assertNotIn("first official baseline", svg.lower())
        self.assertNotIn("M0", svg)
        self.assertNotIn("Not final", svg)
        self.assertNotIn(self.status["milestone"]["note"], svg)
        for old_value in (
            self.status["tracks"]["track1"]["method"],
            self.status["tracks"]["track2"]["method"],
            self.status["tracks"]["track1"]["phase"],
            self.status["tracks"]["track2"]["phase"],
        ):
            self.assertNotIn(old_value, results)


class ProjectStatusOutputTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.status = self.module.load_status(STATUS_PATH)

    def test_svg_is_valid_and_contains_scores_and_metric_labels(self):
        svg = self.module.render_scorecard_svg(self.status)
        root = ET.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        for text in (
            f'{self.status["tracks"]["track1"]["final_score"]:.6f}',
            f'{self.status["tracks"]["track2"]["final_score"]:.6f}',
            "Rel-L2",
            "TKE",
            "MVPE",
            "Time",
            "SPS",
            self.status["milestone"]["note"],
        ):
            self.assertIn(text, svg)
        self.assertIn('role="img"', svg)
        self.assertIn("#2563eb", svg)
        self.assertIn("#7c3aed", svg)

    def test_svg_description_uses_scores_from_the_status_data(self):
        changed = copy.deepcopy(self.status)
        for track_name, score in (("track1", 80.0), ("track2", 70.0)):
            changed["tracks"][track_name]["final_score"] = score
            for key in self.module.METRIC_KEYS:
                changed["tracks"][track_name]["metrics"][key] = score
        self.module.validate_status(changed)

        svg = self.module.render_scorecard_svg(changed)
        desc = svg[svg.index("<desc"):svg.index("</desc>") + len("</desc>")]

        self.assertIn("80.000000", desc)
        self.assertIn("70.000000", desc)
        self.assertNotIn("78.533004", desc)
        self.assertNotIn("74.237630", desc)

    def test_write_then_check_is_clean_and_check_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "README.md").write_text(
                "# Test\n\n<!-- PROJECT_STATUS:START -->\nold\n"
                "<!-- PROJECT_STATUS:END -->\n",
                encoding="utf-8",
            )
            (root / "docs" / "RESULTS.md").write_text(
                "# Results\n\n<!-- CURRENT_RESULTS:START -->\nold\n"
                "<!-- CURRENT_RESULTS:END -->\n",
                encoding="utf-8",
            )
            outputs = self.module.build_outputs(root, self.status)
            self.assertEqual(len(self.module.stale_outputs(outputs)), 3)
            self.module.write_outputs(outputs)
            self.assertEqual(self.module.stale_outputs(outputs), [])
            self.assertEqual(
                [path for path in root.rglob("*") if path.suffix in {".tmp", ".bak"}],
                [],
            )
            readme_path = root / "README.md"
            current_score = (
                f'{self.status["tracks"]["track1"]["final_score"]:.6f}'
            )
            changed = readme_path.read_text(encoding="utf-8").replace(
                current_score, "80.000000", 1
            )
            readme_path.write_text(changed, encoding="utf-8")
            refreshed = self.module.build_outputs(root, self.status)
            self.assertIn(readme_path, self.module.stale_outputs(refreshed))

    def test_check_mode_returns_one_for_stale_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "status.json").write_text(
                json.dumps(self.status, ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "<!-- PROJECT_STATUS:START -->\nold\n"
                "<!-- PROJECT_STATUS:END -->\n",
                encoding="utf-8",
            )
            (root / "docs" / "RESULTS.md").write_text(
                "<!-- CURRENT_RESULTS:START -->\nold\n"
                "<!-- CURRENT_RESULTS:END -->\n",
                encoding="utf-8",
            )
            self.assertEqual(self.module.main(["--root", str(root), "--check"]), 1)
            self.assertEqual(self.module.main(["--root", str(root)]), 0)
            before_check = {
                path: path.read_bytes()
                for path in (root / "README.md", root / "docs" / "RESULTS.md", root / "docs" / "scorecard.svg")
            }
            self.assertEqual(self.module.main(["--root", str(root), "--check"]), 0)
            self.assertEqual(
                {path: path.read_bytes() for path in before_check}, before_check
            )

    def test_second_replacement_failure_restores_all_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "README.md", root / "docs" / "RESULTS.md", root / "docs" / "scorecard.svg"]
            originals = {}
            outputs = {}
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                original = f"original-{index}\n"
                path.write_text(original, encoding="utf-8")
                originals[path] = original
                outputs[path] = f"replacement-{index}\n"

            replace = Path.replace
            calls = 0

            def fail_second_replace(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected second replacement failure")
                return replace(source, target)

            with mock.patch.object(Path, "replace", fail_second_replace):
                with self.assertRaisesRegex(OSError, "injected second"):
                    self.module.write_outputs(outputs)

            self.assertEqual(
                {path: path.read_text(encoding="utf-8") for path in paths}, originals
            )
            self.assertEqual(
                [path for path in root.rglob("*") if path.name not in {"README.md", "RESULTS.md", "scorecard.svg", "docs"}],
                [],
            )


if __name__ == "__main__":
    unittest.main()
