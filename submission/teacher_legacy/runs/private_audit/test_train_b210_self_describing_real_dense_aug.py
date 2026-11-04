from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_self_describing_real_dense_aug.py"
SPEC = importlib.util.spec_from_file_location("self_describing_real", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def payload(*, base: int = 16) -> dict:
    config = MODULE.EnsembleUNet2dConfig(
        base=base, depth=4, groups=8, ensemble_members=4,
        physical_energy_controller=True)
    raw = config.__dict__.copy()
    raw["architecture"] = "ensemble_unet"
    model = MODULE.EnsembleUNet2d(config)
    return {
        "model_config": raw,
        "model_state_dict": model.state_dict(),
    }


def test_reads_exact_serialized_contract():
    config, total, controller = MODULE.architecture_from_payload(payload())
    assert config.base == 16
    model = MODULE.EnsembleUNet2d(config)
    assert total == sum(value.numel() for value in model.parameters())
    assert controller == sum(
        value.numel() for value in model.energy_controller.parameters())
    assert sum(
        value.numel() for value in model.state_dict().values()) > total


def test_accepts_base_width_not_divisible_by_requested_group_count():
    config, total, _controller = MODULE.architecture_from_payload(
        payload(base=18))
    assert config.base == 18
    assert total > 0


def test_rejects_wrong_architecture_and_missing_controller():
    wrong = payload()
    wrong["model_config"]["architecture"] = "fno"
    with pytest.raises(ValueError, match="ensemble U-Net"):
        MODULE.architecture_from_payload(wrong)
    incomplete = payload()
    incomplete["model_state_dict"] = {"stem.weight": torch.zeros(1)}
    with pytest.raises(RuntimeError, match="Missing key"):
        MODULE.architecture_from_payload(incomplete)


def test_cli_requires_start_checkpoint():
    assert MODULE.checkpoint_path(["runner", "--start-sim", "model.pth"]) == Path(
        "model.pth")
    with pytest.raises(ValueError, match="--start-sim"):
        MODULE.checkpoint_path(["runner"])
