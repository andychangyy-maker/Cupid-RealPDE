from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_ROOT.parent
SUBMISSION_ROOT = CODE_ROOT / "submissions"
for path in (CODE_ROOT, SUBMISSION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
)
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)
import build_track1_b210_causal_phase_direction_package as builder  # noqa: E402


def _load_runtime_without_prewarm():
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        spec = importlib.util.spec_from_file_location(
            "phase_runtime_unit",
            SUBMISSION_ROOT / "track1_b210_causal_phase_direction.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        torch.cuda.is_available = original


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_phase_config() -> CausalPhaseDirectionConfig:
    return CausalPhaseDirectionConfig(
        in_steps=5, out_steps=5, hidden_channels=8,
        spatial_blocks=1, temporal_blocks=2, temporal_kernel=3, groups=4)


def test_update_zero_shape_and_pressure_passthrough_are_mechanical() -> None:
    torch.manual_seed(31)
    adapter = CausalPhaseDirectionAdapter(_tiny_phase_config()).eval()
    history = torch.randn(2, 5, 8, 10, 3)
    base = torch.randn(2, 5, 8, 10, 3)
    output = adapter(history, base)
    assert output.shape == base.shape
    assert torch.equal(output, base)
    assert torch.equal(output[..., 2], base[..., 2])


def test_normalized_extent_compensation_reconstructs_base_endpoints() -> None:
    runtime = _load_runtime_without_prewarm()
    torch.manual_seed(37)
    config = _tiny_phase_config()
    adapter = CausalPhaseDirectionAdapter(config).eval()
    with torch.no_grad():
        adapter.direction_head.weight.normal_(std=0.06)
        adapter.direction_head.bias.normal_(std=0.02)
    history = torch.randn(2, 5, 8, 10, 3)
    base = torch.randn(2, 5, 8, 10, 3)
    lower = torch.rand(2, 5, 8, 10, 2) + 0.2
    upper = torch.rand(2, 5, 8, 10, 2) + 0.3
    prediction, moved_lower, moved_upper = runtime._phase_distribution(
        adapter, history, base, lower, upper)
    assert not torch.equal(prediction[..., :2], base[..., :2])
    torch.testing.assert_close(
        prediction[..., :2] - moved_lower,
        base[..., :2] - lower, atol=5e-7, rtol=5e-7)
    torch.testing.assert_close(
        prediction[..., :2] + moved_upper,
        base[..., :2] + upper, atol=5e-7, rtol=5e-7)
    assert torch.equal(prediction[..., 2], base[..., 2])


def _write_fixture(root: Path) -> dict[str, Path | str]:
    normalizer = {
        "mean_input": torch.tensor([0.15, -0.001, 0.0]),
        "mean_target": torch.tensor([0.16, -0.002, 0.0]),
        "std_input": torch.tensor([0.11, 0.016, 1.0]),
        "std_target": torch.tensor([0.12, 0.017, 1.0]),
    }
    base = EnsembleUNet2d(EnsembleUNet2dConfig(
        base=8, depth=2, groups=4, difference_steps=2,
        ensemble_members=2, uq_hidden_channels=8))
    dense = root / "dense.pth"
    dense_payload = {
        "phase": "real", "iteration": 3_200,
        "model_config": base.architecture_config(),
        "model_state_dict": base.state_dict(),
        "normalizer": normalizer,
    }
    torch.save(dense_payload, dense)
    stored, _audit = builder.compact_state_dict(
        dense_payload["model_state_dict"],
        min_elements=builder.QUANTIZE_MIN_ELEMENTS)
    base_payload = {
        "phase": "real",
        "model_config": dense_payload["model_config"],
        "quantized_state_dict": stored,
        "normalizer": {name: value.tolist() for name, value in normalizer.items()},
        "inference_dtype": "float16",
        "storage_format": builder.STORAGE_FORMAT,
    }
    checkpoint = io.BytesIO()
    torch.save(base_payload, checkpoint)
    base_archive = root / "base.zip"
    base_files = {
        "submission.py": (
            SUBMISSION_ROOT / "track1_ensemble_unet2d.py").read_bytes(),
        "models/__init__.py": (
            SUBMISSION_ROOT / "ensemble_models_init.py").read_bytes(),
        "models/unet2d.py": (CODE_ROOT / "models/unet2d.py").read_bytes(),
        "models/ensemble_unet2d.py": (
            CODE_ROOT / "models/ensemble_unet2d.py").read_bytes(),
        builder.BASE_CHECKPOINT: checkpoint.getvalue(),
    }
    with zipfile.ZipFile(base_archive, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in builder.BASE_MEMBERS:
            archive.writestr(name, base_files[name])

    adapter = CausalPhaseDirectionAdapter(CausalPhaseDirectionConfig())
    with torch.no_grad():
        adapter.direction_head.weight.normal_(std=0.04)
        adapter.direction_head.bias.normal_(std=0.01)
    ema_state = adapter.state_dict()
    endpoint = root / "adapter.pt"
    torch.save({
        "schema": builder.ENDPOINT_SCHEMA,
        "experiment_schema": builder.EXPERIMENT_SCHEMA,
        "arm": builder.EXPECTED_ARM,
        "update": builder.EXPECTED_UPDATE,
        "base_checkpoint": {
            "sha256": _sha(dense),
            "model_state_sha256": builder.tensor_state_sha256(
                dense_payload["model_state_dict"]),
            "path": "/private/source",
        },
        "adapter_config": adapter.architecture_config(),
        "adapter_parameters": builder.EXPECTED_PARAMETERS,
        "ema_adapter_state": ema_state,
        "ema_adapter_state_sha256": builder.tensor_state_sha256(ema_state),
        "normalizer": normalizer,
        "mechanics": {
            "update0_bitwise_identity": True,
            "direction_head_gradient_u1": True,
            "history_encoder_gradient_u2": True,
            "direction_gate_gradient_u2": True,
        },
    }, endpoint)
    return {
        "base": base_archive,
        "base_sha": _sha(base_archive),
        "dense": dense,
        "dense_sha": _sha(dense),
        "endpoint": endpoint,
        "endpoint_sha": _sha(endpoint),
    }


def test_builder_strict_sha_rejects_changed_input() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "value.bin"
        path.write_bytes(b"sealed")
        with pytest.raises(RuntimeError, match="changed"):
            builder.require_sha(path, "0" * 64, "fixture")
        assert builder.require_sha(path, _sha(path), "fixture") == _sha(path)


def test_compact_point_lineage_fails_on_one_changed_int8_value() -> None:
    state = {
        "weight": torch.linspace(-1.0, 1.0, 2048).reshape(32, 64),
        "bias": torch.linspace(-0.5, 0.5, 32),
    }
    stored, _audit = builder.compact_state_dict(state)
    builder.validate_compact_base_lineage(state, stored)
    changed = {
        key: ({part: tensor.clone() for part, tensor in value.items()}
              if isinstance(value, dict) else value.clone())
        for key, value in stored.items()
    }
    changed["weight"]["quantized"][0, 0] += 1
    with pytest.raises(ValueError, match="INT8 lineage differs"):
        builder.validate_compact_base_lineage(state, changed)


def test_fp16_adapter_recovery_is_more_faithful_than_int8() -> None:
    torch.manual_seed(409)
    adapter = CausalPhaseDirectionAdapter(_tiny_phase_config())
    with torch.no_grad():
        adapter.direction_head.weight.normal_(std=0.04)
        adapter.direction_head.bias.normal_(std=0.01)
    source = adapter.state_dict()
    fp16 = builder.fp16_state_dict(source)
    int8, _audit = builder.compact_state_dict(source, min_elements=16)
    fp16_error = builder.state_recovery_error(source, fp16)
    int8_error = builder.state_recovery_error(
        source, builder.restore_state_dict({"quantized_state_dict": int8}))
    assert fp16_error["rms_error"] < int8_error["rms_error"]
    assert fp16_error["relative_rms_error"] < int8_error["relative_rms_error"]


def test_archive_is_int8_bound_path_free_and_v9_reentrant() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _write_fixture(root)
        output = root / "phase.zip"
        report = root / "report.json"
        result = builder.build_archive(
            base_archive=fixture["base"],
            base_archive_sha256=fixture["base_sha"],
            base_dense_checkpoint=fixture["dense"],
            base_dense_checkpoint_sha256=fixture["dense_sha"],
            base_model_state_sha256=builder.tensor_state_sha256(
                torch.load(fixture["dense"], map_location="cpu",
                           weights_only=False)["model_state_dict"]),
            adapter_endpoint=fixture["endpoint"],
            adapter_endpoint_sha256=fixture["endpoint_sha"],
            output=output, report=report, max_size_mib=256.0,
            adapter_storage="fp16")
        assert result["output"]["below_256_mib"] is True
        assert result["private_source_paths_embedded"] is False
        assert result["official_submission_performed"] is False
        assert (result["adapter_storage_recovery"]["fp16"]["rms_error"]
                < result["adapter_storage_recovery"]["int8"]["rms_error"])
        assert "/" not in json.loads(report.read_text())["output"]["filename"]

        extracted = root / "extracted"
        with zipfile.ZipFile(output) as archive:
            assert tuple(archive.namelist()) == builder.OUTPUT_MEMBERS
            assert archive.testzip() is None
            assert archive.read(builder.BASE_CHECKPOINT) == zipfile.ZipFile(
                fixture["base"]).read(builder.BASE_CHECKPOINT)
            for name in archive.namelist():
                data = archive.read(name)
                assert b"/root/autodl-tmp/" not in data
                assert b"/private/source" not in data
            adapter_payload = torch.load(
                io.BytesIO(archive.read(builder.ADAPTER_CHECKPOINT)),
                map_location="cpu", weights_only=False)
            assert adapter_payload["deployment_state"] == (
                "ema_adapter_state_u4000")
            assert adapter_payload["storage_format"] == (
                builder.FP16_STORAGE_FORMAT)
            assert "quantized_state_dict" not in adapter_payload
            assert all(not value.is_floating_point()
                       or value.dtype == torch.float16
                       for value in adapter_payload["model_state_dict"].values())
            archive.extractall(extracted)

        script = r"""
import importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location('phase_submission', 'submission.py')
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)
rng = np.random.default_rng(20260905)
x = rng.standard_normal((1, 20, 8, 8, 3), dtype=np.float32)
before = x.copy()
state = submission._get_models()
raw = torch.from_numpy(x).to(dtype=state['dtype'])
normalized = (raw - state['mean_input']) / state['std_input']
with torch.inference_mode():
    base_norm, lower_norm, upper_norm = state['base_model'].forward_distribution(
        normalized)
    base_point = base_norm * state['std_target'] + state['mean_target']
    base_point[..., 2] = 0.0
    base_lower = base_point.clone()
    base_upper = base_point.clone()
    base_lower[..., :2] = base_point[..., :2] - lower_norm * state['std_target'][:2]
    base_upper[..., :2] = base_point[..., :2] + upper_norm * state['std_target'][:2]
first = submission.predict(x, metadata={})
second = submission.predict(x, metadata=None)
assert np.array_equal(x, before)
for key in ('prediction', 'lower', 'upper'):
    assert first[key].shape == (1, 20, 8, 8, 3)
    assert first[key].dtype == np.float32
    assert np.isfinite(first[key]).all()
    assert np.array_equal(first[key], second[key])
    assert np.count_nonzero(first[key][..., 2]) == 0
assert not np.array_equal(first['prediction'][..., :2], base_point[..., :2].numpy())
assert np.array_equal(first['lower'][..., :2], base_lower[..., :2].numpy())
assert np.array_equal(first['upper'][..., :2], base_upper[..., :2].numpy())
assert np.all(first['lower'] <= first['upper'])
"""
        subprocess.run(
            [sys.executable, "-c", script], cwd=extracted,
            check=True, capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
