from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


CODE_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = CODE_ROOT / "evaluation"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, EVALUATION_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_msclosure_dct_planned_forcing_decomposition as subject  # noqa: E402
from models.msclosure_dct_planned_forcing import (  # noqa: E402
    CausalDCTPlanHead,
    DCTPlannedForcingMultipleShootingClosure,
    DCTPlannedForcingMultipleShootingConfig,
)


def _tiny_model():
    return DCTPlannedForcingMultipleShootingClosure(
        DCTPlannedForcingMultipleShootingConfig(
            e0_channels=8, e1_channels=16, e2_channels=24,
            e0_blocks=1, e1_blocks=1, e2_blocks=1,
            context_channels=8, cell_channels=16, cell_blocks=1))


def test_hard_replacement_exactly_assigns_free_k4_to_plan():
    torch.manual_seed(61)
    model = _tiny_model()
    batch, height, width = 2, 7, 9
    free = torch.ones(batch, height, width, 1, dtype=torch.bool)
    free[:, (0, -1), :, :] = False
    free[:, :, (0, -1), :] = False
    free_time = free.unsqueeze(1)
    plan = model.plan_head.reconstruct(
        torch.randn(batch, 2, 4, height, width))
    plan = plan * free_time
    dynamic = model.plan_head.reconstruct(
        torch.randn(batch, 2, 4, height, width))
    dynamic = dynamic + torch.randn(batch, 1, height, width, 2)
    fixed = torch.randn(batch, 1, height, width, 2).expand(-1, 20, -1, -1, -1)
    current_uv = torch.where(free_time, plan + dynamic, fixed)
    pressure = torch.randn(batch, 20, height, width, 1)
    prediction = torch.cat((current_uv, pressure), dim=-1)

    result = subject.decompose_and_hard_replace(
        model, prediction, plan, free)
    replacement = result["replacement_prediction"]
    replacement_coefficients = model.plan_head.temporal_coefficients(
        replacement[..., :2])
    plan_coefficients = model.plan_head.temporal_coefficients(plan)
    mask = subject._coefficient_mask(free, plan_coefficients)
    torch.testing.assert_close(
        replacement_coefficients * mask, plan_coefficients * mask,
        atol=2e-6, rtol=2e-6)
    assert torch.equal(replacement[..., 2:], pressure)
    fixed_expanded = (~free_time).expand_as(current_uv)
    assert torch.equal(
        replacement[..., :2][fixed_expanded], current_uv[fixed_expanded])
    torch.testing.assert_close(
        replacement[..., :2].mean(dim=1), current_uv.mean(dim=1),
        atol=2e-6, rtol=2e-6)
    assert result["invariants"]["pressure_bitwise_equal"]
    assert result["invariants"]["fixed_geometry_bitwise_equal"]


class _ToyDataset:
    def __init__(self, samples: int):
        self.samples = samples
        time = torch.linspace(-1.0, 1.0, 20)[:, None, None, None]
        self.target = torch.ones(20, 32, 64, 3) + time
        self.history = torch.zeros(20, 32, 64, 3)

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return self.history + 0.01 * index, self.target + 0.01 * index


class _CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.plan_head = CausalDCTPlanHead(8, 20)
        self.calls = 0

    def forward_components(self, history):
        self.calls += 1
        batch, _time, height, width, _channels = history.shape
        plan_coefficients = history.new_zeros(batch, 2, 4, height, width)
        plan_coefficients[:, :, 0] = 0.08
        dynamic_coefficients = history.new_zeros(
            batch, 2, 4, height, width)
        dynamic_coefficients[:, :, 1] = 0.04
        plan = self.plan_head.reconstruct(plan_coefficients)
        dynamic = self.plan_head.reconstruct(dynamic_coefficients)
        mean = history[:, -1, ..., :2].unsqueeze(1)
        uv = mean + plan + dynamic
        pressure = history.new_zeros(batch, 20, height, width, 1)
        free = torch.ones(batch, height, width, 1, dtype=torch.bool,
                          device=history.device)
        return {
            "prediction": torch.cat((uv, pressure), dim=-1),
            "applied_plan_normalized": plan,
            "free_mask": free,
        }


def test_paired_evaluator_calls_model_once_per_batch_for_both_arms():
    model = _CountingModel()
    dataset = _ToyDataset(5)
    labels = ["a.h5", "a.h5", "b.h5", "b.h5", "b.h5"]
    normalizer = subject.base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3),
        torch.device("cpu"))
    result = subject.evaluate_dataset_once(
        model, dataset, labels, normalizer, torch.device("cpu"),
        batch_size=2)
    assert model.calls == 3
    assert result["forward_calls"] == 3
    assert result["single_forward_supplies_both_arms"]
    assert set(result["aggregate"]) == set(subject.ARMS)
    assert list(result["per_prefix_horizon"]) == [
        str(index) for index in range(1, 21)]
    assert set(result["per_trajectory"]) == {"a.h5", "b.h5"}
    assert result["windows_per_trajectory"] == 48
    assert result["positive_trajectories_definition"].endswith("> 0.0")
    assert result["invariants"]["validated"]
    assert np.isfinite(list(result["decomposition"].values())).all()
    json.dumps(result, allow_nan=False)


def test_checkpoint_loader_roundtrip(tmp_path):
    model = _tiny_model()
    statistics = {
        "mean_input": [0.1, 0.2, 0.3],
        "std_input": [1.1, 1.2, 1.3],
        "mean_target": [0.4, 0.5, 0.6],
        "std_target": [1.4, 1.5, 1.6],
    }
    model.set_normalizer_statistics(
        *(torch.tensor(statistics[name]) for name in (
            "mean_input", "mean_target", "std_input", "std_target")))
    checkpoint = tmp_path / "endpoint.pt"
    torch.save({
        "schema": "test",
        "rollout_mode": "carry_detached",
        "model_config": model.config.__dict__,
        "normalizer": statistics,
        "model_state": model.state_dict(),
    }, checkpoint)
    loaded, normalizer, payload = subject.load_checkpoint(
        checkpoint, torch.device("cpu"))
    assert payload["schema"] == "test"
    assert loaded.architecture_config() == model.architecture_config()
    assert torch.equal(loaded.mean_target, normalizer.mean_target)
