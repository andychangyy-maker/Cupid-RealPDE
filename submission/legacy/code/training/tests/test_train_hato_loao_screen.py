from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_hato_loao_screen as subject


@dataclass(frozen=True)
class ToyConfig:
    e0_channels: int = 96
    e1_channels: int = 192
    e2_channels: int = 256


class ToyHATO(nn.Module):
    def __init__(self, config: ToyConfig, *, interaction_mode: str):
        super().__init__()
        self.config = config
        self.interaction_mode = interaction_mode
        self.scale = nn.Parameter(torch.ones(()))
        self.attention = nn.Linear(1, 1)
        self.e1_observed_time = nn.Parameter(torch.ones(1))
        self.mean_head = nn.Linear(1, 1)
        self.fluctuation_head = nn.Linear(1, 1)
        self.register_buffer("persistence_scale", torch.ones(3))
        self.register_buffer("persistence_bias", torch.zeros(3))
        self.register_buffer("target_std", torch.ones(3))

    def set_normalizer_statistics(self, mean_input, mean_target,
                                  std_input, std_target):
        self.persistence_scale.copy_((std_input / std_target).detach())
        self.persistence_bias.copy_(((mean_input - mean_target) / std_target).detach())
        self.target_std.copy_(std_target.detach())

    def architecture_config(self):
        return self.config

    def forward_components(self, history):
        return {"history": history, "interaction_mode": self.interaction_mode}

    def forward(self, history):
        one = history.new_ones(1, 1)
        scale = (self.scale + self.e1_observed_time.sum()
                 + self.attention(one).sum() + self.mean_head(one).sum()
                 + self.fluctuation_head(one).sum())
        return history * scale


@pytest.fixture(autouse=True)
def toy_hato(monkeypatch):
    monkeypatch.setattr(subject, "_model_classes", lambda: (ToyConfig, ToyHATO))


def test_contract_schedule_and_tiny_config_are_frozen() -> None:
    assert (subject.SEED, subject.SCHEDULE_SEED) == (210, 220)
    assert (subject.UPDATES, subject.BATCH_SIZE, subject.HELD_AOA) == (2000, 8, 10)
    assert subject.FIT_AOAS == (0, 5, 15, 20)
    assert subject.ACKNOWLEDGEMENT == "RUN_M15_HATO_SIM2K"
    assert subject.GATE == {
        "delta_point_min": .08,
        "component_floor": -.05,
        "positive_trajectories_min": 15,
        "worst_trajectory_delta_point_min": -.05,
    }
    tiny = subject.model_config(tiny=True)
    assert (tiny.e0_channels, tiny.e1_channels, tiny.e2_channels) == (8, 24, 32)
    first = np.random.default_rng(subject.SCHEDULE_SEED).integers(
        0, 3840, size=(subject.UPDATES, subject.BATCH_SIZE), dtype=np.int64)
    second = np.random.default_rng(subject.SCHEDULE_SEED).integers(
        0, 3840, size=(subject.UPDATES, subject.BATCH_SIZE), dtype=np.int64)
    assert np.array_equal(first, second)


def test_pair_identity_and_component_contract() -> None:
    models, identity = subject.construct_pair(torch.device("cpu"), tiny=True)
    assert identity["bitwise_identical"]
    assert identity["only_interaction_difference"] == "interaction_mode"
    assert len(set(identity["parameter_counts"].values())) == 1
    left, right = (models[arm].state_dict() for arm in subject.ARMS)
    assert left.keys() == right.keys()
    assert all(torch.equal(value, right[key]) for key, value in left.items())
    component = subject._component_contract(
        models[subject.ARMS[0]], torch.zeros(1, 20, 8, 8, 3))
    assert component["forward_components_mapping"]
    assert component["component_keys"] == ["history", "interaction_mode"]
    normalizer = subject.base.ChannelNormalizer(
        torch.zeros(3), torch.ones(3), torch.full((3,), 2.0),
        torch.full((3,), 4.0), torch.device("cpu"))
    record = subject.set_pair_normalizer_statistics(models, normalizer)
    assert all(record["normalizer_buffers_bitwise_equal"].values())


def test_held_load_seal_guard() -> None:
    with pytest.raises(RuntimeError, match="endpoint state hash missing"):
        subject.require_endpoint_seals({
            subject.ARMS[0]: {"endpoint_state_sha256": "a"},
            subject.ARMS[1]: {},
        })
    subject.require_endpoint_seals({
        arm: {"endpoint_state_sha256": arm} for arm in subject.ARMS})


def test_gradient_audit_requires_time_attention_and_both_output_roles() -> None:
    model = subject._model_classes()[1](
        subject.model_config(tiny=True), interaction_mode="temporal_candidate")
    output = model(torch.randn(1, 20, 16, 16, 3))
    output.square().mean().backward()
    audit = subject._gradient_audit(model)
    assert audit["all_seen_gradients_finite"]
    assert audit["all_required_groups_live"]
    assert audit["trainable_tensor_coverage"] > .5


def test_actual_hato_attention_layout_matches_across_modes(monkeypatch) -> None:
    from models.hierarchical_axial_temporal_operator import (  # noqa: PLC0415
        HATOConfig, HierarchicalAxialTemporalOperator)
    monkeypatch.setattr(
        subject, "_model_classes", lambda: (HATOConfig, HierarchicalAxialTemporalOperator))
    models, _identity = subject.construct_pair(torch.device("cpu"), tiny=True)
    record = subject.attention_execution_contract(models, torch.device("cpu"), formal=False)
    assert record["pass"]
    assert record["levels"]["e1"]["sequence_shape"][subject.ARMS[0]] == [560, 20, 24]
    assert record["levels"]["e2"]["sequence_shape"][subject.ARMS[1]] == [160, 20, 32]


def test_protocol_alias_parses_to_same_destination(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hato", "--acknowledge-protocol", "ok"])
    assert subject.parse_args().acknowledge == "ok"
    monkeypatch.setattr(sys, "argv", ["hato", "--acknowledge", "again"])
    assert subject.parse_args().acknowledge == "again"


def test_evaluation_reports_cumulative_prefixes_not_single_frame_horizons(monkeypatch) -> None:
    class Panel:
        def __init__(self, *_args):
            self.history = torch.ones(20, 16, 16, 3)
            self.target = torch.full((20, 16, 16, 3), 2.0)

        def __len__(self):
            return 20 * 48

        def __getitem__(self, _index):
            return self.history, self.target

    monkeypatch.setattr(subject.base, "SimWindowDataset", Panel)
    models, _identity = subject.construct_pair(torch.device("cpu"), tiny=True)
    normalizer = subject.base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3),
        torch.device("cpu"))
    labels = [f"trajectory_{index:02d}.h5" for index in range(20) for _ in range(48)]
    result = subject.evaluate_pair(models, None, None, labels, normalizer, torch.device("cpu"))
    assert "per_horizon" not in result
    assert list(result["per_prefix_horizon"]) == [str(index) for index in range(1, 21)]
    assert result["per_prefix_horizon"]["1"]["tke_degenerate_single_frame"]
    assert not result["per_prefix_horizon"]["2"]["tke_degenerate_single_frame"]


def test_gate_is_structural_only_and_ignores_resources() -> None:
    delta = {key: 0.0 for key in (*subject.POINT_WEIGHTS,
                                 "point_contribution_46_10_9")}
    delta["point_contribution_46_10_9"] = .08
    evaluation = {
        "delta_candidate_minus_control": delta,
        "positive_trajectories": 15,
        "worst_trajectory_delta_point": -.05,
        "trajectory_count": 20,
        "windows_per_trajectory": 48,
    }
    identity = {"bitwise_identical": True,
                "only_interaction_difference": "interaction_mode"}
    attention = {"pass": True}
    assert subject.decide_gate(evaluation, identity, attention)["pass"]
    bad = dict(evaluation)
    bad["positive_trajectories"] = 14
    assert not subject.decide_gate(bad, identity, attention)["pass"]
    bad = dict(evaluation)
    bad["delta_candidate_minus_control"] = {
        **delta, "tke_rel_l2_score": -.051}
    assert not subject.decide_gate(bad, identity, attention)["pass"]
    assert not subject.decide_gate(evaluation, identity, {"pass": False})["pass"]


def test_formal_rejects_missing_execute_or_acknowledgement(tmp_path: Path) -> None:
    args = SimpleNamespace(execute=False, acknowledge="", held_aoa=10,
                           updates=2000, batch_size=8, output_dir=tmp_path,
                           sim_dir=tmp_path)
    with pytest.raises(PermissionError, match="exact acknowledgement"):
        subject.run(args, torch.device("cpu"))
