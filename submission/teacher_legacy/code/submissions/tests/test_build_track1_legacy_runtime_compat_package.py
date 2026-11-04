import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "build_track1_legacy_runtime_compat_package.py"
)
SPEC = importlib.util.spec_from_file_location("legacy_runtime_builder", SCRIPT)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def state(shape=(2, 3)):
    return {
        "weight": {
            "quantized": torch.zeros(shape, dtype=torch.int8),
            "scale": torch.ones((shape[0], 1), dtype=torch.float32),
        },
        "bias": torch.zeros(shape[0], dtype=torch.float16),
    }


def payload(config=None, stored=None):
    return {
        "phase": "real",
        "model_config": {"base": 210, **(config or {})},
        "quantized_state_dict": stored or state(),
        "normalizer": {"mean_input": [0.0, 0.0, 0.0]},
        "inference_dtype": "float16",
        "storage_format": "symmetric_per_dim0_int8_v1",
    }


def test_legacy_config_removes_only_frozen_defaults():
    config = {"base": 210, **SUBJECT.LEGACY_DEFAULTS}
    assert SUBJECT.legacy_config(config) == {"base": 210}
    bad = dict(config, acceleration_steps=1)
    with pytest.raises(RuntimeError):
        SUBJECT.legacy_config(bad)


def test_compatible_payload_requires_same_runtime_layout():
    anchor = payload()
    candidate = payload(SUBJECT.LEGACY_DEFAULTS)
    report = SUBJECT.validate_compatible_payloads(anchor, candidate)
    assert report["state_layout_exact"] is True
    candidate["quantized_state_dict"] = state((3, 3))
    with pytest.raises(RuntimeError):
        SUBJECT.validate_compatible_payloads(anchor, candidate)


def test_nested_tensor_payload_equality_is_bitwise():
    left = state()
    right = state()
    assert SUBJECT.tensor_payload_equal(left, right)
    right["weight"]["quantized"][0, 0] = 1
    assert not SUBJECT.tensor_payload_equal(left, right)
