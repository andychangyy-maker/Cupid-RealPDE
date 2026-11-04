from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_ROOT.parent
SUBMISSIONS_ROOT = CODE_ROOT / "submissions"
for path in (CODE_ROOT, SUBMISSIONS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)
from build_track1_ensemble_unet2d_package import (  # noqa: E402
    compact_checkpoint,
)
from track1_ensemble_unet2d import (  # noqa: E402
    _BATCH_SIZE,
    _restore_state_dict,
)


class EnsembleUNet2dInt8PackageTests(unittest.TestCase):
    def test_release_wrapper_keeps_frozen_batch16_contract(self):
        self.assertEqual(_BATCH_SIZE, 16)

    @staticmethod
    def checkpoint(path: Path) -> dict[str, torch.Tensor]:
        model = EnsembleUNet2d(EnsembleUNet2dConfig(
            base=8, depth=2, groups=4, ensemble_members=2,
            uq_hidden_channels=8, difference_steps=2))
        state = model.state_dict()
        torch.save({
            "phase": "real",
            "model_config": model.architecture_config(),
            "model_state_dict": state,
            "normalizer": {
                "mean_input": torch.zeros(3),
                "mean_target": torch.zeros(3),
                "std_input": torch.ones(3),
                "std_target": torch.ones(3),
            },
        }, path)
        return state

    def test_int8_round_trip_obeys_per_dim0_error_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            compact = root / "compact.pth"
            original = self.checkpoint(source)
            statistics = compact_checkpoint(
                source, compact, "float16", "int8", 16)
            payload = torch.load(
                compact, map_location="cpu", weights_only=False)
            restored = _restore_state_dict(payload, torch.float32)

            self.assertEqual(
                payload["storage_format"],
                "symmetric_per_dim0_int8_v1")
            self.assertNotIn("model_state_dict", payload)
            self.assertGreater(statistics["quantized_tensors"], 0)
            for key, stored in payload["quantized_state_dict"].items():
                if isinstance(stored, dict) and "quantized" in stored:
                    error = (restored[key] - original[key].float()).abs()
                    bound = stored["scale"].float() / 2.0 + 1e-6
                    self.assertTrue(torch.all(error <= bound))

    def test_b210_functional_contract_survives_int8_archive_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            archive_path = root / "submission.zip"
            extracted = root / "extracted"
            config = EnsembleUNet2dConfig(
                base=7, depth=2, groups=8, ensemble_members=2,
                uq_hidden_channels=7, difference_steps=2,
                physical_energy_controller=True,
                physical_energy_member_sync=True,
                physical_energy_hidden_channels=7,
                backbone_residual_dropout=0.2)
            model = EnsembleUNet2d(config)
            with torch.no_grad():
                model.energy_controller.body[-1].weight.fill_(0.125)
                model.energy_controller.body[-1].bias.fill_(-0.0625)
            expected_config = model.architecture_config()
            torch.save({
                "phase": "real",
                "model_config": expected_config,
                "model_state_dict": model.state_dict(),
                "normalizer": {
                    "mean_input": torch.tensor([0.25, -0.5, 2.0]),
                    "mean_target": torch.tensor([1.25, -1.5, 3.0]),
                    "std_input": torch.tensor([2.0, 3.0, 4.0]),
                    "std_target": torch.tensor([5.0, 6.0, 7.0]),
                },
            }, source)

            subprocess.run([
                sys.executable,
                str(SUBMISSIONS_ROOT
                    / "build_track1_ensemble_unet2d_package.py"),
                "--checkpoint", str(source),
                "--output", str(archive_path),
                "--storage", "int8",
                "--quantize-min-elements", "2",
            ], cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            with zipfile.ZipFile(archive_path) as archive:
                payload = torch.load(
                    io.BytesIO(archive.read("ensemble_unet2d.pth")),
                    map_location="cpu", weights_only=False)
                self.assertEqual(payload["model_config"], expected_config)
                self.assertEqual(set(payload["normalizer"]), {
                    "mean_input", "mean_target", "std_input", "std_target",
                })
                self.assertNotIn("values", payload["normalizer"])
                archive.extractall(extracted)

            script = """
import importlib.util
import numpy as np
import torch
from models.unet2d import _group_norm

spec = importlib.util.spec_from_file_location('submission', 'submission.py')
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)
state = submission._get_model()
model = state['model']
assert model.config.physical_energy_controller is True
assert model.config.physical_energy_member_sync is True
assert model.config.backbone_residual_dropout == 0.2
assert model.energy_controller is not None
assert [_group_norm(8, width).num_groups for width in (210, 420, 840, 1680)] == [7, 7, 8, 8]

expected_stats = {
    'mean_input': [0.25, -0.5, 2.0],
    'mean_target': [1.25, -1.5, 3.0],
    'std_input': [2.0, 3.0, 4.0],
    'std_target': [5.0, 6.0, 7.0],
}
for key, expected in expected_stats.items():
    assert torch.equal(state[key].cpu(), torch.tensor(expected))
    assert tuple(expected) != submission._FALLBACK_STATS[key]

payload = torch.load(
    'ensemble_unet2d.pth', map_location='cpu', weights_only=False)
restored = submission._restore_state_dict(payload, state['dtype'])
loaded = model.state_dict()
controller_keys = [
    key for key in restored if key.startswith('energy_controller.')]
assert controller_keys
for key in controller_keys:
    assert torch.equal(loaded[key].cpu(), restored[key].cpu())

rng = np.random.default_rng(20260826)
for batch in (1, 16, 33):
    x = rng.standard_normal((batch, 20, 4, 4, 3), dtype=np.float32)
    before = x.copy()
    first = submission.predict(x, metadata={'batch': batch})
    second = submission.predict(x, metadata=None)
    assert np.array_equal(x, before)
    for key in ('prediction', 'lower', 'upper'):
        assert first[key].shape == (batch, 20, 4, 4, 3)
        assert first[key].dtype == np.float32
        assert np.isfinite(first[key]).all()
        assert np.array_equal(first[key], second[key])
        assert np.count_nonzero(first[key][..., 2]) == 0
    assert np.all(first['lower'] <= first['upper'])
"""
            subprocess.run(
                [sys.executable, "-c", script], cwd=extracted,
                check=True, capture_output=True, text=True)

    def test_dense_compaction_remains_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            compact = root / "compact.pth"
            self.checkpoint(source)
            statistics = compact_checkpoint(
                source, compact, "float16", "dense", 16)
            payload = torch.load(
                compact, map_location="cpu", weights_only=False)

            self.assertIn("model_state_dict", payload)
            self.assertNotIn("quantized_state_dict", payload)
            self.assertEqual(statistics["quantized_tensors"], 0)
            self.assertTrue(all(
                not torch.is_floating_point(value)
                or value.dtype == torch.float16
                for value in payload["model_state_dict"].values()))

    def test_archive_imports_and_predicts_from_extracted_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            archive_path = root / "submission.zip"
            extracted = root / "extracted"
            self.checkpoint(source)
            subprocess.run([
                sys.executable,
                str(SUBMISSIONS_ROOT
                    / "build_track1_ensemble_unet2d_package.py"),
                "--checkpoint", str(source),
                "--output", str(archive_path),
                "--storage", "int8",
                "--quantize-min-elements", "16",
            ], cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            expected = [
                "submission.py", "models/__init__.py", "models/unet2d.py",
                "models/ensemble_unet2d.py", "ensemble_unet2d.pth",
            ]
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.namelist(), expected)
                self.assertIsNone(archive.testzip())
                payload = torch.load(
                    io.BytesIO(archive.read("ensemble_unet2d.pth")),
                    map_location="cpu", weights_only=False)
                self.assertIn("quantized_state_dict", payload)
                archive.extractall(extracted)

            script = """
import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('submission', 'submission.py')
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)
x = np.zeros((1, 20, 8, 8, 3), dtype=np.float32)
first = submission.predict(x, metadata={})
second = submission.predict(x, metadata=None)
for key in ('prediction', 'lower', 'upper'):
    assert first[key].shape == (1, 20, 8, 8, 3)
    assert first[key].dtype == np.float32
    assert np.isfinite(first[key]).all()
    assert np.array_equal(first[key], second[key])
assert np.all(first['lower'] <= first['upper'])
"""
            subprocess.run(
                [sys.executable, "-c", script], cwd=extracted,
                check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
