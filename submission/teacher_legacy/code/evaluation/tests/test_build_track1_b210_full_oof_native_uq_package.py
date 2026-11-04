from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS_ROOT = CODE_ROOT / "submissions"
for path in (CODE_ROOT, SUBMISSIONS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_track1_b210_full_oof_native_uq_package as builder  # noqa: E402
from build_track1_blended_ensemble_unet2d_package import (  # noqa: E402
    compact_state_dict,
)


def write_json(path: Path, value: object) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return builder.sha256(path)


def self_seal(value: dict) -> dict:
    normalized = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    normalized["json_self_seal"] = {
        "algorithm": "sha256",
        "payload_without_self_seal_sha256": builder.canonical_sha256(normalized),
        "computed_after_json_roundtrip": True,
    }
    return normalized


class FullOofNativeUqPackageTests(unittest.TestCase):
    def fixture(self, root: Path) -> dict:
        config = {
            "architecture": "ensemble_unet",
            "base": 210,
            "depth": 4,
            "groups": 8,
            "ensemble_members": 4,
            "uq_hidden_channels": 160,
        }
        normalizer = {
            "mean_input": [0.15, 0.0, 0.0],
            "mean_target": [0.15, 0.0, 0.0],
            "std_input": [0.1, 0.02, 1.0],
            "std_target": [0.1, 0.02, 1.0],
        }
        point_state = {
            "stem.weight": torch.tensor([[1.0, -2.0], [0.5, 0.25]]),
            "stem.bias": torch.tensor([0.125, -0.25]),
            "uq_projections.0.weight": torch.full((2, 2), 3.0),
            "quantile_head.bias": torch.tensor([0.5, -0.5]),
        }
        uq_state = {
            **point_state,
            "uq_projections.0.weight": torch.full((2, 2), 7.0),
            "quantile_head.bias": torch.tensor([0.75, -0.75]),
        }
        point_checkpoint = root / "point.pth"
        point_state_sha = builder.model_state_sha256(point_state)
        point_payload = {
            "schema": builder.POINT_ENDPOINT_SCHEMA,
            "phase": "real",
            "training_scope": "all_clean81_full13202",
            "iteration": 4000,
            "endpoint": "ema_u4000",
            "primary": True,
            "model_state_dict": point_state,
            "model_state_sha256": point_state_sha,
            "model_config": config,
            "normalizer": normalizer,
            "optimizer_windows": 13_202,
            "optimizer_stride": 5,
            "normalizer_windows": 2_544,
            "normalizer_stride": 20,
            "fresh_native_uq_crossfit_refit_required": True,
            "submission_authorized": False,
        }
        torch.save(point_payload, point_checkpoint)
        point_seal = {
            "path": str(point_checkpoint),
            "file_sha256": builder.sha256(point_checkpoint),
            "model_state_sha256": point_state_sha,
            "model_config_sha256": builder.canonical_sha256(config),
            "normalizer_sha256": builder.canonical_sha256(normalizer),
        }
        point_result = root / "point.json"
        point_result_sha = write_json(point_result, {
            "schema": builder.POINT_RESULT_SCHEMA,
            "primary_endpoint": "ema_u4000",
            "fresh_native_uq_crossfit_refit_required": True,
            "submission_authorized": False,
            "endpoints": {"ema_u4000": {"seal": point_seal}},
        })

        point_partition = {
            key: value for key, value in point_state.items()
            if not builder.is_uq_key(key)}
        point_partition_sha = builder.model_state_sha256(point_partition)
        uq_checkpoint = root / "uq.pth"
        uq_state_sha = builder.model_state_sha256(uq_state)
        uq_payload = {
            "schema": builder.UQ_ENDPOINT_SCHEMA,
            "phase": "real",
            "iteration": 3200,
            "endpoint": builder.UQ_ENDPOINT,
            "primary": True,
            "model_state_dict": uq_state,
            "model_state_sha256": uq_state_sha,
            "model_config": config,
            "normalizer": normalizer,
            "point_checkpoint": {"result_sha256": point_result_sha},
            "point_partition_sha256": point_partition_sha,
            "point_state_bitwise": True,
            "uq_initialization": {"full_real_uq_state_inherited": False},
            "checkpoint_selection": False,
            "clean716_manifest_read": False,
            "evaluation_manifest_read": False,
            "submission_authorized": False,
        }
        torch.save(uq_payload, uq_checkpoint)
        uq_seal = {
            "path": str(uq_checkpoint),
            "file_sha256": builder.sha256(uq_checkpoint),
            "model_state_sha256": uq_state_sha,
            "point_partition_sha256": point_partition_sha,
        }
        uq_result = root / "uq.json"
        uq_result_payload = self_seal({
            "schema": builder.UQ_RESULT_SCHEMA,
            "decision": "COMPLETE_FIXED_FULL_OOF_NATIVE_UQ_U3200",
            "primary_endpoint": builder.UQ_ENDPOINT,
            "fixed_cycles": 4,
            "updates_per_cycle": 800,
            "total_updates": 3200,
            "point": {"result_sha256": point_result_sha},
            "point_state_bitwise": True,
            "checkpoint_selection": False,
            "clean716_manifest_read": False,
            "evaluation_manifest_read": False,
            "submission_authorized": False,
            "endpoint": uq_seal,
            "endpoint_verification": {
                "exact": True,
                "file_sha256": uq_seal["file_sha256"],
            },
        })
        uq_result_sha = write_json(uq_result, uq_result_payload)

        runtime = root / "runtime.zip"
        runtime_state, _ = compact_state_dict(point_state, 2)
        runtime_payload = {
            "phase": "real",
            "model_config": config,
            "quantized_state_dict": runtime_state,
            "normalizer": normalizer,
            "inference_dtype": "float16",
            "storage_format": builder.STORAGE_FORMAT,
        }
        buffer = io.BytesIO()
        torch.save(runtime_payload, buffer)
        wrapper = b"""_BATCH_SIZE = 16
def _prewarm_cuda_runtime():
    dummy = np.zeros((_BATCH_SIZE, 20, 32, 64, 3), dtype=np.float32)
_prewarm_cuda_runtime()
"""
        runtime_files = {
            "submission.py": wrapper,
            "models/__init__.py": b"runtime-init\n",
            "models/unet2d.py": b"runtime-unet\n",
            "models/ensemble_unet2d.py": b"runtime-ensemble\n",
            builder.CHECKPOINT_MEMBER: buffer.getvalue(),
        }
        with zipfile.ZipFile(runtime, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in builder.EXPECTED_MEMBERS:
                archive.writestr(name, runtime_files[name])
        return {
            "point_result": point_result,
            "point_result_sha": point_result_sha,
            "uq_result": uq_result,
            "uq_result_sha": uq_result_sha,
            "uq_checkpoint": uq_checkpoint,
            "runtime": runtime,
            "runtime_sha": builder.sha256(runtime),
            "runtime_files": runtime_files,
            "point_state": point_state,
            "uq_state": uq_state,
        }

    def test_build_preserves_timed_runtime_and_explicit_state_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)
            output = root / "candidate.zip"
            report = root / "build.json"
            result = builder.build_candidate(
                point_result=fixture["point_result"],
                point_result_sha256=fixture["point_result_sha"],
                uq_result=fixture["uq_result"],
                uq_result_sha256=fixture["uq_result_sha"],
                runtime_archive=fixture["runtime"],
                runtime_archive_sha256=fixture["runtime_sha"],
                output=output,
                report=report,
                quantize_min_elements=2,
            )
            self.assertTrue(output.is_file())
            self.assertTrue(report.is_file())
            self.assertTrue(result["only_checkpoint_member_changed"])
            self.assertTrue(result["composition"][
                "equals_complete_refit_endpoint_bitwise"])
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(tuple(archive.namelist()), builder.EXPECTED_MEMBERS)
                for name in builder.RUNTIME_MEMBERS:
                    self.assertEqual(
                        archive.read(name), fixture["runtime_files"][name])
                payload = torch.load(
                    io.BytesIO(archive.read(builder.CHECKPOINT_MEMBER)),
                    map_location="cpu", weights_only=False)
            point = payload["quantized_state_dict"]["stem.weight"]
            uq = payload["quantized_state_dict"]["uq_projections.0.weight"]
            restored_point = point["quantized"].float() * point["scale"].float()
            restored_uq = uq["quantized"].float() * uq["scale"].float()
            self.assertTrue(torch.allclose(
                restored_point, fixture["point_state"]["stem.weight"], atol=0.02))
            self.assertTrue(torch.allclose(
                restored_uq, fixture["uq_state"]["uq_projections.0.weight"],
                atol=0.06))

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            result = builder.build_candidate(
                point_result=fixture["point_result"],
                point_result_sha256=fixture["point_result_sha"],
                uq_result=fixture["uq_result"],
                uq_result_sha256=fixture["uq_result_sha"],
                runtime_archive=fixture["runtime"],
                runtime_archive_sha256=fixture["runtime_sha"],
                output=None,
                report=None,
                quantize_min_elements=2,
                dry_run=True,
            )
            self.assertTrue(result["dry_run"])
            self.assertIsNone(result["archive"])
            self.assertFalse(result["source_vs_zip_prediction_replay_required"])

    def test_rejects_refit_point_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)
            payload = torch.load(
                fixture["uq_checkpoint"], map_location="cpu", weights_only=False)
            payload["model_state_dict"]["stem.bias"][0] += 1.0
            payload["model_state_sha256"] = builder.model_state_sha256(
                payload["model_state_dict"])
            torch.save(payload, fixture["uq_checkpoint"])
            result = json.loads(fixture["uq_result"].read_text())
            result.pop("json_self_seal")
            result["endpoint"]["file_sha256"] = builder.sha256(
                fixture["uq_checkpoint"])
            result["endpoint"]["model_state_sha256"] = payload[
                "model_state_sha256"]
            result["endpoint_verification"]["file_sha256"] = result[
                "endpoint"]["file_sha256"]
            result = self_seal(result)
            fixture["uq_result_sha"] = write_json(fixture["uq_result"], result)
            with self.assertRaisesRegex(RuntimeError, "changed the point partition"):
                builder.build_candidate(
                    point_result=fixture["point_result"],
                    point_result_sha256=fixture["point_result_sha"],
                    uq_result=fixture["uq_result"],
                    uq_result_sha256=fixture["uq_result_sha"],
                    runtime_archive=fixture["runtime"],
                    runtime_archive_sha256=fixture["runtime_sha"],
                    output=None, report=None, quantize_min_elements=2,
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
