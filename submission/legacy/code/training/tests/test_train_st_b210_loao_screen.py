from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_st_b210_loao_screen as subject


@dataclass(frozen=True)
class ToyConfig:
    e0_channels: int = 8
    e1_channels: int = 16
    e2_channels: int = 24
    e3_channels: int = 32
    e0_blocks: int = 1
    e1_blocks: int = 1
    e2_blocks: int = 1
    e3_blocks: int = 1
    groups: int = 4
    max_displacement: float = 4.0


class ToySTB210(nn.Module):
    def __init__(self, config: ToyConfig, interaction_mode: str):
        super().__init__()
        self.config, self.interaction_mode = config, interaction_mode
        self.axis_mixer = nn.Parameter(torch.tensor(.1))
        self.observed_to_future_logits = nn.Parameter(torch.tensor(.1))
        self.decoder = nn.Parameter(torch.tensor(.1))
        self.output_head = nn.Parameter(torch.tensor(.1))
        self.register_buffer("persistence_scale", torch.ones(3))
        self.register_buffer("persistence_bias", torch.zeros(3))
        self.register_buffer("target_std", torch.ones(3))

    def architecture_config(self):
        return {"architecture": "toy_st_b210", "interaction_mode": self.interaction_mode,
                "flattened_history_bypass": False}

    def set_normalizer_statistics(self, mean_input, mean_target, std_input, std_target):
        self.persistence_scale.copy_(std_input / std_target)
        self.persistence_bias.copy_((mean_input - mean_target) / std_target)
        self.target_std.copy_(std_target)

    def axis_execution_contract(self):
        cross_time = self.interaction_mode == "causal_temporal_candidate"
        folded = not cross_time
        return [{"stage": stage, "block": 0, "kernel_size": 5, "left_pad": 4, "right_pad": 0,
                 "canonical_shape": [1, 20, 8, 16, 32],
                 "packed_shape": [1, 8, 20, 16, 32] if cross_time else [20, 8, 32, 16, 1],
                 "output_elements": 81920, "macs": 81920 * 21,
                 "cross_time": cross_time, "control_time_folded": folded}
                for stage in range(4)]

    def forward(self, history):
        # Each named group participates, allowing the runner's gradient audit.
        update = (self.axis_mixer + self.observed_to_future_logits + self.decoder
                  + self.output_head) * .001
        return history + update

    def forward_components(self, history):
        prediction = self(history)
        return {"prediction": prediction,
                "displacement": torch.zeros(*prediction.shape[:-1], 2, device=prediction.device),
                "fluctuation": torch.zeros_like(prediction)}


@pytest.fixture(autouse=True)
def toy_model(monkeypatch):
    monkeypatch.setattr(subject, "_model_classes", lambda: (ToyConfig, ToySTB210))


def test_frozen_protocol() -> None:
    assert subject.ARMS == ("spatial_axis_control", "temporal_candidate")
    assert subject.ARM_MODE["temporal_candidate"] == "causal_temporal_candidate"
    assert (subject.SEED, subject.SCHEDULE_SEED, subject.UPDATES, subject.BATCH_SIZE) == (210, 220, 2000, 8)
    assert subject.FIT_AOAS == (0, 5, 15, 20)
    assert subject.GATE["delta_point_min"] == .10
    assert subject.GATE["candidate_absolute_point_min"] == 60.225899
    assert subject.EXPECTED_PARAMETERS == 29_187_608
    assert subject.EXPECTED_STATE_TENSORS == 238
    assert subject.MAX_TRAIN_PEAK_BYTES == 21 * 1024 ** 3


def test_pair_is_bitwise_identical_except_for_mode() -> None:
    models, identity = subject.construct_pair(torch.device("cpu"), tiny=True)
    assert identity["bitwise_identical"]
    assert identity["differing_architecture_keys"] == ["interaction_mode"]
    left, right = (models[arm].state_dict() for arm in subject.ARMS)
    assert left.keys() == right.keys()
    assert all(torch.equal(value, right[key]) for key, value in left.items())


def test_normalizer_and_axis_contract() -> None:
    models, _identity = subject.construct_pair(torch.device("cpu"), tiny=True)
    normalizer = subject.base.ChannelNormalizer(torch.zeros(3), torch.zeros(3),
                                                torch.ones(3), torch.ones(3), torch.device("cpu"))
    buffers = subject.set_pair_normalizer_statistics(models, normalizer)
    assert all(buffers["normalizer_buffers_bitwise_equal"].values())
    contract = subject._axis_contract(models, torch.device("cpu"), formal=False)
    assert contract["pass"]
    assert contract["checks"]["output_elements_and_macs_equal"]
    assert contract["checks"]["axis_semantics"]


def test_train_audits_all_required_groups() -> None:
    model = ToySTB210(ToyConfig(), "causal_temporal_candidate")
    normalizer = subject.base.ChannelNormalizer(torch.zeros(3), torch.zeros(3),
                                                torch.ones(3), torch.ones(3), torch.device("cpu"))
    trajectory = torch.randn(40, 16, 16, 3)
    dataset = subject.base.SimWindowDataset([trajectory], [subject.base.SimWindow(0, 0)])
    endpoint = subject.train_arm("toy", model, dataset, np.zeros((2, 1), dtype=np.int64),
                                 normalizer, torch.device("cpu"), 2)
    assert endpoint["gradient_audit"]["all_required_groups_seen_over_training"]
    assert not endpoint["gradient_audit"]["nonfinite_gradient_tensors_over_training"]


def test_schedule_and_held_guard() -> None:
    schedule = np.random.default_rng(220).integers(0, 3840, size=(2000, 8), dtype=np.int64)
    assert (subject.validate_sample_stream(schedule, 3840)
            == subject.EXPECTED_SAMPLE_STREAM_SHA256)
    calls = []
    endpoints = {arm: {"endpoint_state_sha256": arm} for arm in subject.ARMS}
    assert subject.load_held_fields_after_guards(
        Path("/tmp"), ["held"], endpoints, {"pass": True},
        loader=lambda directory, rows: calls.append((directory, rows)) or "loaded") == "loaded"
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="missing endpoint"):
        subject.load_held_fields_after_guards(Path("/tmp"), ["held"],
                                              {subject.ARMS[0]: endpoints[subject.ARMS[0]]},
                                              {"pass": True}, loader=lambda *_: "bad")
    with pytest.raises(RuntimeError, match="mechanical health"):
        subject.load_held_fields_after_guards(Path("/tmp"), ["held"], endpoints,
                                              {"pass": False}, loader=lambda *_: "bad")


def test_gate_needs_complete_paired_panel() -> None:
    evaluation = {
        "delta_candidate_minus_control": {"rel_l2_score": .2, "tke_rel_l2_score": .1,
                                            "mvpe_rel_l2_score": .2, "point_contribution_46_10_9": .1},
        "positive_trajectories": 20, "worst_trajectory_delta_point": .0,
        "trajectory_count": 20, "windows_per_trajectory": 48,
        "aggregate": {"temporal_candidate": {"point_contribution_46_10_9": 60.3}},
    }
    identity = {"bitwise_identical": True}
    decision = subject.decide_gate(evaluation, identity, {"pass": True}, {"pass": True})
    assert decision["pass"]
    evaluation["trajectory_count"] = 19
    assert not subject.decide_gate(evaluation, identity, {"pass": True}, {"pass": True})["pass"]
