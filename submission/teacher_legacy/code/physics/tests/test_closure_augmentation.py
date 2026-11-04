from __future__ import annotations

from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from physics.closure_augmentation import ClosureResidualAugmentor


def test_clean_closure_augmentation_is_continuous_masked_and_uv_only():
    residual = torch.zeros(80, 4, 6, 2)
    residual[:, 1:3, 1:5, 0] = torch.linspace(0, 1, 80)[:, None, None]
    residual[:, 1:3, 1:5, 1] = torch.linspace(1, 0, 80)[:, None, None]
    augmentor = ClosureResidualAugmentor(
        [{"aoa": 0, "re": 10000, "residual": residual}],
        [{"aoa": 0, "re": 10000}], candidate_count=1,
        source_contract={"manifest_sha256": "a" * 64})
    history = torch.zeros(1, 20, 4, 6, 3)
    target = torch.zeros_like(history)
    history[..., 1:3, 1:5, :2] = 1.0
    target[..., 1:3, 1:5, :2] = 1.0
    original_sequence = torch.cat((history, target), dim=1)

    torch.manual_seed(4)
    aug_history, aug_target, selected = augmentor.augment(
        history, target, torch.tensor([0]), probability=1.0,
        amplitude=0.05)
    augmented = torch.cat((aug_history, aug_target), dim=1)
    delta = augmented - original_sequence
    assert selected.tolist() == [True]
    assert torch.equal(delta[:, 0], torch.zeros_like(delta[:, 0]))
    assert torch.count_nonzero(delta[..., :2]) > 0
    assert torch.equal(delta[..., 2], torch.zeros_like(delta[..., 2]))
    assert torch.equal(delta[..., :1, :, :2], torch.zeros_like(delta[..., :1, :, :2]))
    assert torch.equal(delta[..., -1:, :, :2], torch.zeros_like(delta[..., -1:, :, :2]))


def test_disabled_closure_returns_inputs_by_identity():
    augmentor = ClosureResidualAugmentor(
        [{"aoa": 0, "re": 1, "residual": torch.zeros(41, 2, 2, 2)}],
        [{"aoa": 0, "re": 1}])
    history = torch.randn(1, 20, 2, 2, 3)
    target = torch.randn_like(history)
    out_history, out_target, selected = augmentor.augment(
        history, target, torch.tensor([0]), probability=0.0, amplitude=0.05)
    assert out_history is history
    assert out_target is target
    assert not selected.any()


def test_future_branch_preserves_observation_and_changes_only_future_uv():
    innovation = torch.zeros(80, 4, 6, 2)
    innovation[:, 1:3, 1:5, 0] = 0.1
    innovation[:, 1:3, 1:5, 1] = -0.05
    augmentor = ClosureResidualAugmentor(
        [{"aoa": 0, "re": 10000, "innovation": innovation}],
        [{"aoa": 0, "re": 10000}], candidate_count=1)
    history = torch.zeros(2, 20, 4, 6, 3)
    target = torch.zeros_like(history)
    history[..., 1:3, 1:5, :2] = 1.0
    target[..., 1:3, 1:5, :2] = 1.0
    history_before = history.clone()
    target_before = target.clone()

    torch.manual_seed(9)
    branched, selected = augmentor.augment_future(
        history, target, torch.tensor([0, 0]), probability=1.0,
        amplitude=0.05)

    assert selected.tolist() == [True, True]
    assert torch.equal(history, history_before)
    assert torch.equal(target, target_before)
    assert torch.count_nonzero(branched[..., :2] - target[..., :2]) > 0
    assert torch.equal(branched[..., 2], target[..., 2])
    assert torch.equal(
        branched[..., :1, :, :2], target[..., :1, :, :2])
    assert torch.equal(
        branched[..., -1:, :, :2], target[..., -1:, :, :2])


def test_disabled_future_branch_returns_target_by_identity():
    augmentor = ClosureResidualAugmentor(
        [{"aoa": 0, "re": 1, "innovation": torch.zeros(20, 2, 2, 2)}],
        [{"aoa": 0, "re": 1}])
    history = torch.randn(1, 20, 2, 2, 3)
    target = torch.randn_like(history)
    output, selected = augmentor.augment_future(
        history, target, torch.tensor([0]), probability=0.0,
        amplitude=0.05)
    assert output is target
    assert not selected.any()
