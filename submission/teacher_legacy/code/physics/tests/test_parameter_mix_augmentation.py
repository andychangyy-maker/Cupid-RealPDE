from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from physics.parameter_mix_augmentation import (
    ParameterMixContract,
    PhaseAlignedParameterMixAugmentor,
)


@dataclass(frozen=True)
class Window:
    trajectory: int
    start: int


class Dataset:
    def __init__(self):
        base = torch.arange(80, dtype=torch.float32).view(80, 1, 1, 1)
        self.trajectories = [
            base.expand(-1, 8, 8, 3).clone(),
            (base + 10).expand(-1, 8, 8, 3).clone(),
            (base + 20).expand(-1, 8, 8, 3).clone(),
        ]
        self.windows = [Window(index, 20) for index in range(3)]

    def __len__(self):
        return len(self.windows)


class IdentityClosure:
    source_contract = {"test": True}

    def to(self, _device):
        return self

    def augment(self, history, target, _trajectory_ids, **_kwargs):
        selected = torch.zeros(len(history), dtype=torch.bool, device=history.device)
        return history, target, selected

    def augment_future(self, _history, target, _trajectory_ids, **_kwargs):
        selected = torch.zeros(len(target), dtype=torch.bool, device=target.device)
        return target, selected


def metadata():
    return [
        {"re": 100, "aoa": 0},
        {"re": 200, "aoa": 0},
        {"re": 300, "aoa": 0},
    ]


def test_grid_contract_adds_adjacent_aoa_neighbors():
    dataset = Dataset()
    rows = [
        {"re": 100, "aoa": 0},
        {"re": 100, "aoa": 5},
        {"re": 100, "aoa": 10},
    ]
    augmentor = PhaseAlignedParameterMixAugmentor(
        IdentityClosure(), dataset, rows,
        ParameterMixContract(
            include_aoa_neighbors=True, maximum_phase_shift=0))
    assert augmentor._partners == [(1,), (0, 2), (1,)]


def test_mix_is_target_consistent_and_does_not_consume_global_rng():
    dataset = Dataset()
    contract = ParameterMixContract(
        probability=1.0, lambda_low=0.25, lambda_high=0.25,
        maximum_phase_shift=0, signature_spatial_stride=2)
    augmentor = PhaseAlignedParameterMixAugmentor(
        IdentityClosure(), dataset, metadata(), contract)
    indices = torch.tensor([0, 2])
    trajectories = torch.tensor([0, 2])
    sequence = torch.stack([
        dataset.trajectories[0][20:60],
        dataset.trajectories[2][20:60],
    ])
    before = torch.random.get_rng_state().clone()
    augmentor.set_batch_context(indices, iteration=7)
    history, target, selected = augmentor.augment(
        sequence[:, :20], sequence[:, 20:], trajectories,
        probability=0.0, amplitude=0.0)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert selected.tolist() == [True, True]
    expected0 = sequence[0] + 0.25 * (
        dataset.trajectories[1][20:60] - sequence[0])
    expected1 = sequence[1] + 0.25 * (
        dataset.trajectories[1][20:60] - sequence[1])
    assert torch.equal(torch.cat((history[0], target[0])), expected0)
    assert torch.equal(torch.cat((history[1], target[1])), expected1)
    assert augmentor.last_audit["parameter_mix_selected_fraction"] == 1.0
    assert augmentor.source_contract == {"test": True}


def test_context_and_trajectory_identity_are_hard_guards():
    dataset = Dataset()
    augmentor = PhaseAlignedParameterMixAugmentor(
        IdentityClosure(), dataset, metadata(),
        ParameterMixContract(maximum_phase_shift=0))
    sequence = dataset.trajectories[0][20:60].unsqueeze(0)
    with pytest.raises(RuntimeError, match="context was not set"):
        augmentor.augment(
            sequence[:, :20], sequence[:, 20:], torch.tensor([0]),
            probability=0.0, amplitude=0.0)
    augmentor.set_batch_context(torch.tensor([0]), iteration=1)
    with pytest.raises(RuntimeError, match="trajectory IDs"):
        augmentor.augment(
            sequence[:, :20], sequence[:, 20:], torch.tensor([1]),
            probability=0.0, amplitude=0.0)


@pytest.mark.parametrize("field,value", [
    ("probability", 1.1), ("lambda_low", -0.1),
    ("lambda_high", 0.8), ("phase_shift_stride", 0),
])
def test_contract_rejects_invalid_values(field, value):
    values = vars(ParameterMixContract()).copy()
    values[field] = value
    with pytest.raises(ValueError):
        ParameterMixContract(**values).validate()
