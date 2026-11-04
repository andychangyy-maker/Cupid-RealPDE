from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
SCRIPT = TRAINING_ROOT / "build_c208_rtd1_teacher_cache.py"
SPEC = importlib.util.spec_from_file_location("c208_rtd1_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_point_and_bounds_use_distinct_frozen_p1_p0_weights():
    prediction_a = torch.zeros(1, 2, 1, 1, 2)
    prediction_b = torch.full_like(prediction_a, 2.0)
    prediction_c = torch.stack((
        torch.full((1, 1, 1, 2), 4.0),
        torch.full((1, 1, 1, 2), 6.0)), dim=1)
    lower_a, upper_a = prediction_a - 1.0, prediction_a + 1.0
    lower_b, upper_b = prediction_b - 1.0, prediction_b + 1.0
    point, lower, upper = subject.compose_rtd1_teacher(
        prediction_a, lower_a, upper_a,
        prediction_b, lower_b, upper_b, prediction_c)
    assert torch.allclose(point, torch.full_like(point, 3.0))
    assert torch.allclose(lower, torch.full_like(lower, 1.1875))
    assert torch.allclose(upper, torch.full_like(upper, 3.1875))
    assert subject.POINT_WEIGHT_B == subject.POINT_WEIGHT_C == 0.5
    assert subject.BOUNDS_WEIGHT_B == 0.625
    assert subject.BOUNDS_WEIGHT_C == 0.25


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, history):
        shape = (history.shape[0], 20, history.shape[2], history.shape[3], 3)
        return torch.full(shape, 0.1234567, device=history.device)

    def forward_distribution(self, history):
        prediction = self.forward(history)
        extent = torch.full(
            (*prediction.shape[:-1], 2), 0.00012345,
            device=history.device)
        return prediction, extent, extent


def test_collectors_keep_float32_until_final_point_storage():
    model = DummyModel()
    history = torch.zeros(1, 10, 2, 3, 3)
    loader = [(history, torch.empty(0))]
    normalizer = argparse.Namespace(
        mean_input=torch.zeros(3), std_input=torch.ones(3),
        mean_target=torch.zeros(3), std_target=torch.ones(3))
    point = subject.collect_point_predictions_float32(
        model, loader, normalizer, torch.device("cpu"))
    prediction, lower, upper = subject.collect_absolute_endpoints_float32(
        model, loader, normalizer, torch.device("cpu"))
    assert point.dtype == prediction.dtype == lower.dtype == upper.dtype \
        == torch.float32
    expected = torch.tensor(0.1234567, dtype=torch.float32)
    assert point[0, 0, 0, 0, 0] == expected
    assert point[0, 0, 0, 0, 0] != expected.to(torch.float16).to(torch.float32)
    assert lower[0, 0, 0, 0, 0] == pytest.approx(
        float(expected - torch.tensor(0.00012345)), abs=1e-8)
    assert upper[0, 0, 0, 0, 0] == pytest.approx(
        float(expected + torch.tensor(0.00012345)), abs=1e-8)


def test_tensor_identity_is_deterministic_and_rejects_noncontiguous():
    value = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    assert subject.tensor_sha256(value, 1) == subject.tensor_sha256(value, 16)
    with pytest.raises(ValueError, match="contiguous CPU"):
        subject.tensor_sha256(value.transpose(0, 1))


def test_output_is_never_overwritten(tmp_path):
    output = tmp_path / "teacher.pt"
    args = argparse.Namespace(batch_size=16, num_workers=0, output=output)
    subject.validate_args(args)
    output.write_bytes(b"occupied")
    with pytest.raises(FileExistsError, match="overwrite"):
        subject.validate_args(args)


def test_schema_declares_physical_contaminated_diagnostic_contract():
    assert subject.SCHEMA_VERSION == 1
    assert subject.CACHE_METHOD == (
        "c208_rtd1_real2576_foldaware_physical_v1")
    assert "checkpoint_selection_contaminated" in subject.EVIDENCE_CLASS
    assert subject.EXPECTED_SHAPE == (2576, 20, 32, 64, 2)
