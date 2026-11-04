from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


TRAINING = Path(__file__).resolve().parents[1]
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import train_b210_temporal_band_residual_stage as subject  # noqa: E402


@dataclass
class Window:
    trajectory: int


class Dataset:
    def __init__(self):
        self.windows = [
            Window(trajectory) for trajectory in range(100)
            for _ in range(48)
        ]

    def __len__(self):
        return len(self.windows)


def manifest():
    re_values = [10_125, 25_425] + list(range(2, 20))
    result = []
    for value in re_values:
        for aoa in (0, 5, 10, 15, 20):
            result.append({"re": value, "aoa": aoa})
    return result


def metrics(rel=10.0, tke=20.0, mvpe=30.0, sps=40.0):
    return {
        "rel_l2_score": rel,
        "tke_rel_l2_score": tke,
        "mvpe_rel_l2_score": mvpe,
        "native_sps_score": sps,
    }


def test_split_excludes_two_complete_reynolds_conditions():
    train, held, by_re = subject.split_indices(Dataset(), manifest())
    assert len(train) == 4_320
    assert len(held) == 480
    assert set(by_re) == {10_125, 25_425}
    assert all(len(value) == 240 for value in by_re.values())


def test_gate_requires_pooled_and_each_reynolds_direction():
    reference = metrics()
    candidate = metrics(rel=10.10, tke=20.01, mvpe=30.01, sps=40.0)
    mechanics = {"identity": True, "gradient": True}
    passed = subject.gate(
        reference, candidate,
        {"10125": reference, "25425": reference},
        {"10125": candidate, "25425": candidate}, mechanics)
    assert passed["pass"]
    assert passed["decision"] == "PASS_TO_REAL_RESPONSIBILITY_ISOLATION"

    reversed_re = subject.gate(
        reference, candidate,
        {"10125": reference, "25425": reference},
        {"10125": candidate, "25425": metrics(rel=9.0)}, mechanics)
    assert not reversed_re["pass"]
    assert not reversed_re["checks"]["both_reynolds_nonnegative"]


def test_gate_rejects_sps_and_mechanics_failures():
    reference = metrics()
    candidate = metrics(rel=10.10, tke=20.01, mvpe=30.01, sps=39.89)
    result = subject.gate(
        reference, candidate,
        {"10125": reference, "25425": reference},
        {"10125": candidate, "25425": candidate},
        {"identity": True, "gradient": False})
    assert not result["pass"]
    assert not result["checks"]["sps_floor"]
    assert not result["checks"]["mechanics"]


def test_freeze_contract_allows_only_temporal_parameters(monkeypatch):
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.temporal_transport = nn.Linear(2, 2)

    toy = Toy()
    for parameter in toy.parameters():
        parameter.requires_grad_(False)
    for parameter in toy.temporal_transport.parameters():
        parameter.requires_grad_(True)
    monkeypatch.setattr(
        subject.sim, "EXPECTED_TEMPORAL_BAND_MODULE_PARAMETERS",
        sum(parameter.numel() for parameter in toy.temporal_transport.parameters()))
    report = subject.freeze_contract(toy)
    assert report["trainable_parameters"] == 6
    assert all(
        name.startswith("temporal_transport.")
        for name in report["trainable_tensor_names"])


def test_validate_base_rejects_wrong_training_origin():
    payload = {
        "model_state_dict": {"weight": torch.ones(1)},
        "model_config": {
            "architecture": "ensemble_unet",
            "cross_conditioned_backbone": True,
            "member_adapter_channels": 64,
        },
        "training_config": {"method": "other", "window_stride": 20},
    }
    with pytest.raises(ValueError, match="Standard24k"):
        subject.validate_base_payload(payload)
