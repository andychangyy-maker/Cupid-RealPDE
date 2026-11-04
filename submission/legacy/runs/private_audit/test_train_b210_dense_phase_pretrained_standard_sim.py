from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_dense_phase_pretrained_standard_sim.py"
SPEC = importlib.util.spec_from_file_location("dense_phase_transfer", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_private_arguments_are_removed_from_base_cli():
    argv, checkpoint, digest = MODULE.pop_private_args([
        "runner.py", "--output-dir", "out",
        "--pretrain-checkpoint", "dense.pth",
        "--pretrain-checkpoint-sha256", "a" * 64,
        "--sim-dir", "sim",
    ])
    assert argv == ["runner.py", "--output-dir", "out", "--sim-dir", "sim"]
    assert checkpoint == Path("dense.pth")
    assert digest == "a" * 64


def test_pretrain_payload_contract():
    MODULE.validate_pretrain_payload({
        "iteration": 10_000,
        "model_state_dict": {},
        "training_config": {
            "window_stride": 10,
            "method": "b210_sim_phase_dense_stride10_v1",
        },
    })
    with pytest.raises(RuntimeError, match="stride 10"):
        MODULE.validate_pretrain_payload({
            "iteration": 10_000,
            "model_state_dict": {},
            "training_config": {
                "window_stride": 20,
                "method": "b210_sim_phase_dense_stride10_v1",
            },
        })


def test_loaded_state_replaces_random_initialization():
    original = MODULE._ORIGINAL_BUILD_ARM_MODEL
    state = torch.nn.Linear(3, 2).state_dict()
    MODULE._PRETRAIN_PAYLOAD = {"model_state_dict": state}
    MODULE._ORIGINAL_BUILD_ARM_MODEL = lambda *args, **kwargs: torch.nn.Linear(3, 2)
    try:
        first = MODULE.build_pretrained_arm()
        second = MODULE.build_pretrained_arm()
    finally:
        MODULE._ORIGINAL_BUILD_ARM_MODEL = original
        MODULE._PRETRAIN_PAYLOAD = None
    assert all(torch.equal(first.state_dict()[key], value) for key, value in state.items())
    assert all(torch.equal(second.state_dict()[key], value) for key, value in state.items())
