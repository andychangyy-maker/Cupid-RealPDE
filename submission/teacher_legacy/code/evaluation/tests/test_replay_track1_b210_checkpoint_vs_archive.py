from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS_ROOT = CODE_ROOT / "submissions"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, SUBMISSIONS_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)
import build_track1_b210_full_oof_native_uq_package as package_builder  # noqa: E402
from build_track1_ensemble_unet2d_package import compact_checkpoint  # noqa: E402
import replay_track1_b210_checkpoint_vs_archive as replay  # noqa: E402


class CheckpointArchiveReplayTests(unittest.TestCase):
    def test_cpu_replay_records_dense_vs_package_quantization_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            compact = root / "compact.pth"
            archive = root / "candidate.zip"
            output = root / "replay.json"
            model = EnsembleUNet2d(EnsembleUNet2dConfig(
                base=8,
                depth=2,
                groups=4,
                ensemble_members=2,
                uq_hidden_channels=8,
                difference_steps=2,
                mean_profile_channels=8,
                temporal_uq_channels=8,
            ))
            torch.save({
                "phase": "real",
                "model_config": model.architecture_config(),
                "model_state_dict": model.state_dict(),
                "normalizer": {
                    "mean_input": [0.15, 0.0, 0.0],
                    "mean_target": [0.15, 0.0, 0.0],
                    "std_input": [0.1, 0.02, 1.0],
                    "std_target": [0.1, 0.02, 1.0],
                },
            }, source)
            compact_checkpoint(source, compact, "float16", "int8", 16)
            entries = (
                (SUBMISSIONS_ROOT / "track1_ensemble_unet2d.py", "submission.py"),
                (SUBMISSIONS_ROOT / "ensemble_models_init.py", "models/__init__.py"),
                (CODE_ROOT / "models/unet2d.py", "models/unet2d.py"),
                (CODE_ROOT / "models/ensemble_unet2d.py", "models/ensemble_unet2d.py"),
                (compact, package_builder.CHECKPOINT_MEMBER),
            )
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
                for path, name in entries:
                    handle.write(path, name)
            args = argparse.Namespace(
                checkpoint=source,
                checkpoint_sha256=package_builder.sha256(source),
                archive=archive,
                archive_sha256=package_builder.sha256(archive),
                output=output,
                input_npy=None,
                samples=1,
                seed=210,
                device="cpu",
            )
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                result = replay.run(args)
            self.assertTrue(output.is_file())
            self.assertTrue(result["zip_deterministic"])
            self.assertTrue(result["input_bitwise_unchanged"])
            self.assertTrue(result["bounds_ordered"])
            self.assertTrue(result["pressure_exactly_zero"])
            for key in ("prediction", "lower", "upper"):
                self.assertGreaterEqual(
                    result["outputs"][key]["dense_vs_zip"]["max_abs"], 0.0)


if __name__ == "__main__":
    unittest.main()
