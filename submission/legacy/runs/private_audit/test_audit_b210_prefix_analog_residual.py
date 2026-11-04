from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("audit_b210_prefix_analog_residual.py")
SPEC = importlib.util.spec_from_file_location("analog_audit", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_target_free_feature_has_fixed_shape_and_ignores_pressure():
    history = torch.randn(3, 20, 32, 64, 3)
    prediction = torch.randn(3, 20, 32, 64, 3)
    first = MODULE.target_free_features(history, prediction)
    history[..., 2] = 1000.0
    prediction[..., 2] = -1000.0
    second = MODULE.target_free_features(history, prediction)
    assert first.shape == (3, 544)
    assert torch.equal(first, second)


def test_masked_coarse_residual_ignores_unscored_cells():
    prediction = torch.zeros(1, 20, 32, 64, 3)
    target = torch.zeros_like(prediction)
    target[..., :16, :32, 0] = 2.0
    coarse = MODULE.masked_coarse_residual(prediction, target, 2, 2)
    assert coarse.shape == (1, 20, 2, 2, 2)
    assert torch.allclose(coarse[:, :, 0, 0, 0], torch.full((1, 20), 2.0))
    assert torch.count_nonzero(coarse[:, :, 1:]) == 0


def test_retrieval_excludes_the_query_trajectory():
    embedding = torch.tensor([[0.0], [0.1], [10.0]])
    residual = torch.tensor([[[[[100.0]]]], [[[[2.0]]]], [[[[4.0]]]]])
    memory_trajectory = torch.tensor([0, 0, 1])
    query_embedding = torch.tensor([[0.0]])
    query_trajectory = torch.tensor([0])
    retrieved, _ = MODULE.retrieve_coarse_residual(
        query_embedding, embedding, residual, query_trajectory,
        memory_trajectory, k=1)
    assert torch.equal(retrieved, residual[2:3])


def test_quantized_shrinkage_is_bounded():
    assert MODULE.quantized_shrinkage(-1.0) == 0.0
    assert MODULE.quantized_shrinkage(0.061) == 0.05
    assert MODULE.quantized_shrinkage(10.0) == MODULE.MAX_SHRINKAGE


def test_upsample_correction_respects_observable_mask():
    coarse = torch.ones(2, 20, 2, 2, 4)
    valid = torch.ones(2, 32, 64, dtype=torch.bool)
    valid[:, :, :10] = False
    full = MODULE.upsample_correction(coarse, valid)
    assert full.shape == (2, 20, 32, 64, 2)
    assert torch.count_nonzero(full[:, :, :, :10]) == 0
    assert torch.all(full[:, :, :, 10:] == 1.0)
