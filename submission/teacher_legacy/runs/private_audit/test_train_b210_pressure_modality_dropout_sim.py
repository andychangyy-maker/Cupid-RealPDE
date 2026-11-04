from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_pressure_modality_dropout_sim.py"
SPEC = importlib.util.spec_from_file_location("pressure_dropout", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_mask_is_deterministic_iteration_local_and_near_half():
    indices = torch.arange(4800)
    first = MODULE.pressure_dropout_mask(indices, 17)
    second = MODULE.pressure_dropout_mask(indices, 17)
    later = MODULE.pressure_dropout_mask(indices, 18)
    assert torch.equal(first, second)
    assert not torch.equal(first, later)
    assert 0.48 <= float(first.float().mean()) <= 0.52


def test_wrapper_changes_only_selected_history_pressure():
    history = torch.randn(8, 20, 2, 3, 3)
    target = torch.randn_like(history)
    trajectory_ids = torch.arange(8)
    sample_indices = torch.arange(8)
    original = history.clone()
    captured = {}

    class Normalizer:
        mean_input = torch.tensor([1.0, 2.0, 3.0])

    def fake_update(*args, **kwargs):
        captured["batch"] = args[1]
        return {}, None, "key", {}

    base = MODULE._ORIGINAL_SIM_UPDATE
    MODULE._ORIGINAL_SIM_UPDATE = fake_update
    positional = [None, (history, target, trajectory_ids, sample_indices), Normalizer()]
    positional.extend([None] * 5)
    positional.extend([7, None, None, "candidate"])
    try:
        values, _, _, _ = MODULE.pressure_dropout_sim_update(*positional)
    finally:
        MODULE._ORIGINAL_SIM_UPDATE = base

    changed, out_target, out_trajectories, out_indices = captured["batch"]
    selected = MODULE.pressure_dropout_mask(sample_indices, 7)
    assert torch.equal(changed[..., :2], original[..., :2])
    assert torch.equal(changed[~selected, ..., 2], original[~selected, ..., 2])
    assert torch.equal(
        changed[selected, ..., 2],
        torch.full_like(changed[selected, ..., 2], 3.0))
    assert out_target is target
    assert out_trajectories is trajectory_ids
    assert out_indices is sample_indices
    assert values["pressure_modality_dropout_fraction"] == float(
        selected.float().mean())
