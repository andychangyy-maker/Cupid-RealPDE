from __future__ import annotations

from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_latent_spectral_residual_sim as subject
import train_b210_latent_spectral_residual_continuation_sim as continuation
from models.ensemble_unet2d import EnsembleUNet2d


class TinyPairedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Linear(3, 5)
        self.energy_controller = torch.nn.Linear(5, 2)
        self.latent_spectral_residual = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.Linear(4, 5))


def test_exact_b210_spectral_parameter_contract_on_meta_device():
    config = subject.spectral_config("candidate")
    with torch.device("meta"):
        model = EnsembleUNet2d(config)
    assert sum(value.numel() for value in model.parameters()) == (
        subject.EXPECTED_TOTAL_PARAMETERS)
    assert sum(
        value.numel()
        for value in model.latent_spectral_residual.parameters()) == (
            subject.EXPECTED_MODULE_PARAMETERS)
    assert sum(
        value.numel() for value in model.energy_controller.parameters()) == (
            subject.EXPECTED_CONTROLLER_PARAMETERS)


def test_control_freezes_only_controller_and_spectral_branch():
    model = TinyPairedModel()
    trainable = subject.configure_trainable(model, "control")
    assert tuple(trainable) == tuple(model.stem.parameters())
    assert all(value.requires_grad for value in model.stem.parameters())
    assert not any(
        value.requires_grad
        for value in model.energy_controller.parameters())
    assert not any(
        value.requires_grad
        for value in model.latent_spectral_residual.parameters())


def test_candidate_trains_complete_expanded_graph():
    model = TinyPairedModel()
    trainable = subject.configure_trainable(model, "candidate")
    assert tuple(trainable) == tuple(model.parameters())
    assert all(value.requires_grad for value in model.parameters())


def test_spectral_state_groups_are_disjoint_and_complete():
    names = {
        "expand.weight", "expand.bias", "weight_positive", "weight_negative",
        "norm.weight", "norm.bias", "reduce.weight", "reduce.bias",
    }
    grouped = {
        group: {name for name in names if subject.state_group(name) == group}
        for group in ("projection", "spectral", "input")
    }
    assert set().union(*grouped.values()) == names
    assert not grouped["projection"].intersection(grouped["spectral"])
    assert not grouped["projection"].intersection(grouped["input"])
    assert not grouped["spectral"].intersection(grouped["input"])


def test_continuation_private_args_are_removed_without_reordering():
    sha = "a" * 64
    remaining, checkpoint, expected = continuation.pop_pretrain_args([
        "runner.py", "--output-dir", "run", "--pretrain-checkpoint", "m.pth",
        "--num-workers", "0", "--pretrain-checkpoint-sha256", sha,
    ])
    assert remaining == [
        "runner.py", "--output-dir", "run", "--num-workers", "0"]
    assert checkpoint == Path("m.pth")
    assert expected == sha


def test_continuation_accepts_only_exact_stage1_contract():
    payload = {
        "iteration": 10_000,
        "model_state_dict": {"weight": torch.zeros(1)},
        "training_config": {
            "window_stride": 20,
            "method": continuation.STAGE1_METHOD,
        },
        "model_config": {
            "architecture": "ensemble_unet",
            "base": 210,
            "depth": 4,
            "latent_spectral_rank": subject.RANK,
            "latent_spectral_modes_y": subject.MODES_Y,
            "latent_spectral_modes_x": subject.MODES_X,
            "latent_spectral_residual_limit": subject.RESIDUAL_LIMIT,
        },
    }
    continuation.validate_pretrain_payload(payload)
    payload["model_config"]["latent_spectral_rank"] = 32
    try:
        continuation.validate_pretrain_payload(payload)
    except RuntimeError as error:
        assert "model_config" in str(error)
    else:
        raise AssertionError("wrong spectral rank was accepted")
