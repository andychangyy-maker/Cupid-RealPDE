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
SCRIPT = TRAINING_ROOT / "build_c208_rsm1_teacher_cache.py"
SPEC = importlib.util.spec_from_file_location("c208_rsm1_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_role_composition_exactly_reconstructs_formal_p1_in_float32():
    prediction_a = torch.tensor(
        [[[[[1.1, 2.2]]], [[[3.3, 4.4]]]]], dtype=torch.float32)
    prediction_b = torch.tensor(
        [[[[[5.5, 6.6]]], [[[9.9, 10.1]]]]], dtype=torch.float32)
    prediction_c = torch.tensor(
        [[[[[7.7, 8.8]]], [[[11.2, 12.3]]]]], dtype=torch.float32)
    fluctuation_a, fluctuation_b, p1_mean = subject.compose_rsm1_roles(
        prediction_a, prediction_b, prediction_c)
    reconstructed = 0.5 * fluctuation_a + 0.5 * fluctuation_b + p1_mean
    base = prediction_a.lerp(prediction_b, 0.5)
    formal = base + 0.5 * (
        prediction_c.mean(dim=1, keepdim=True)
        - base.mean(dim=1, keepdim=True))
    assert torch.allclose(fluctuation_a.mean(dim=1), torch.zeros_like(
        fluctuation_a[:, 0]), atol=3e-7)
    assert torch.allclose(fluctuation_b.mean(dim=1), torch.zeros_like(
        fluctuation_b[:, 0]), atol=3e-7)
    assert torch.allclose(reconstructed, formal, atol=1e-7)
    assert torch.allclose(
        p1_mean,
        0.25 * prediction_a.mean(dim=1, keepdim=True)
        + 0.25 * prediction_b.mean(dim=1, keepdim=True)
        + 0.5 * prediction_c.mean(dim=1, keepdim=True))
    with pytest.raises(TypeError, match="float32"):
        subject.compose_rsm1_roles(
            prediction_a.half(), prediction_b.half(), prediction_c.half())


class DummyDistributionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(0.0))

    def forward_distribution(self, history):
        shape = (history.shape[0], 2, history.shape[2], history.shape[3], 3)
        prediction = torch.full(shape, 0.1234567, device=history.device)
        extent = torch.ones(*shape[:-1], 2, device=history.device)
        return prediction, extent, extent


def test_fold_point_collector_keeps_float32_before_role_composition():
    model = DummyDistributionModel()
    history = torch.zeros(1, 1, 2, 3, 3)
    normalizer = argparse.Namespace(
        mean_input=torch.zeros(3), std_input=torch.ones(3),
        mean_target=torch.zeros(3), std_target=torch.ones(3))
    result = subject.collect_distribution_points_float32(
        model, [(history, torch.empty(0))], normalizer,
        torch.device("cpu"))
    expected = torch.tensor(0.1234567, dtype=torch.float32)
    assert result.dtype == torch.float32
    assert result[0, 0, 0, 0, 0] == expected
    assert result[0, 0, 0, 0, 0] != expected.half().float()


def test_fold_rows_scatter_to_natural_dataset_indices_only():
    destination_a = torch.full((5, 2, 1), -1.0, dtype=torch.float16)
    destination_b = torch.full_like(destination_a, -2.0)
    destination_mean = torch.full((5, 1, 1), -3.0, dtype=torch.float16)
    rows_a = torch.tensor([[[20.0], [21.0]], [[40.0], [41.0]]])
    rows_b = rows_a + 100
    rows_mean = torch.tensor([[[220.0]], [[440.0]]])
    subject.scatter_role_rows(
        destination_a, destination_b, destination_mean, [2, 4],
        rows_a, rows_b, rows_mean)
    assert destination_a[:, 0, 0].tolist() == [-1.0, -1.0, 20.0, -1.0, 40.0]
    assert destination_b[:, 0, 0].tolist() == [-2.0, -2.0, 120.0, -2.0, 140.0]
    assert destination_mean[:, 0, 0].tolist() == [-3.0, -3.0, 220.0, -3.0, 440.0]
    with pytest.raises(ValueError, match="one-to-one"):
        subject.scatter_role_rows(
            destination_a, destination_b, destination_mean, [2, 2],
            rows_a, rows_b, rows_mean)


def test_schema_and_non_overwrite_contract(tmp_path):
    assert subject.SCHEMA_VERSION == 1
    assert subject.CACHE_METHOD == (
        "c208_rsm1_real2576_foldaware_roles_physical_v1")
    assert subject.FLUCTUATION_WEIGHTS == {"a": 0.5, "b": 0.5}
    assert subject.MEAN_WEIGHTS == {"a": 0.25, "b": 0.25, "c": 0.5}
    assert "checkpoint_selection_contaminated" in subject.EVIDENCE_CLASS
    output = tmp_path / "teacher.pt"
    args = argparse.Namespace(batch_size=16, num_workers=0, output=output)
    subject.validate_args(args)
    output.write_bytes(b"occupied")
    with pytest.raises(FileExistsError, match="overwrite"):
        subject.validate_args(args)
