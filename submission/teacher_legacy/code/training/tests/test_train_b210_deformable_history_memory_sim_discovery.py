from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_b210_deformable_history_memory_sim_discovery as subject


def test_m14_contract_and_pair_are_frozen() -> None:
    assert (subject.SEED, subject.SCHEDULE_SEED) == (210, 220)
    assert (subject.UPDATES, subject.BATCH_SIZE, subject.HELD_AOA) == (
        2000, 8, 10)
    assert subject.GATE == {
        "delta_point_min": .05,
        "component_floor": -.03,
        "positive_trajectories_min": 15,
        "worst_trajectory_delta_point_min": -.05,
        "aligned_minus_zero_point_min": .01,
        "latency_ratio_max": 1.35,
        "train_peak_vram_ratio_max": 1.25,
        "parameter_fraction_max": .001,
    }
    control = subject.model_config("spatially_permuted", tiny=True)
    candidate = subject.model_config("aligned", tiny=True)
    changed = {
        key for key in control.__dataclass_fields__
        if getattr(control, key) != getattr(candidate, key)
    }
    assert changed == {"deformable_history_memory_mode"}
    assert control.depth == 4
    assert control.history_dynamics_state_mode == "none"
    assert control.deformable_history_memory_anchors == (0, 6, 13, 19)


def test_tiny_pair_is_bitwise_identical_and_trainable() -> None:
    models, identity = subject.construct_pair(
        torch.device("cpu"), tiny=True)
    assert identity["bitwise_identical"]
    assert identity["only_config_difference"] == (
        "deformable_history_memory_mode")
    assert len(set(identity["parameter_counts"].values())) == 1
    history = torch.randn(1, 20, 16, 16, 3)
    with torch.inference_mode():
        control = models["control"](history)
        candidate = models["candidate"](history)
    assert torch.equal(control, candidate)


def test_gate_requires_structure_quality_mechanism_and_resources() -> None:
    point_keys = (*subject.base.POINT_WEIGHTS,
                  "point_contribution_46_10_9")
    evaluation = {
        "delta_candidate_minus_control": {
            key: (.05 if key == "point_contribution_46_10_9" else 0.0)
            for key in point_keys},
        "delta_candidate_minus_zero_offset": {
            key: (.01 if key == "point_contribution_46_10_9" else 0.0)
            for key in point_keys},
        "positive_trajectories_candidate_minus_control": 15,
        "worst_trajectory_delta_candidate_minus_control": -.05,
        "trajectory_count": 20,
        "windows_per_trajectory": 48,
        "candidate_state_unchanged_by_zero_ablation": True,
    }
    resources = {
        "latency": {
            "parameter_increment": subject.EXPECTED_PARAMETER_INCREMENT,
            "parameter_fraction": .0009,
            "active_over_legacy_ratio": 1.35,
        },
        "train_peak_vram": {"active_over_legacy_ratio": 1.25},
    }
    identity = {
        "bitwise_identical": True,
        "only_config_difference": "deformable_history_memory_mode",
    }
    endpoints = {
        arm: {"memory_gradient_seen_nonzero": {
            "frame_encoder": True,
            "query_heads": True,
            "value_projections": True,
        }} for arm in subject.ARMS
    }
    assert subject.decide_gate(
        evaluation, resources, identity, endpoints)["pass"]
    bad = dict(evaluation)
    bad["positive_trajectories_candidate_minus_control"] = 14
    assert not subject.decide_gate(
        bad, resources, identity, endpoints)["pass"]
    bad = dict(evaluation)
    bad["delta_candidate_minus_zero_offset"] = {
        **evaluation["delta_candidate_minus_zero_offset"],
        "point_contribution_46_10_9": .0099,
    }
    assert not subject.decide_gate(
        bad, resources, identity, endpoints)["pass"]


def test_formal_run_rejects_missing_acknowledgement(tmp_path: Path) -> None:
    args = SimpleNamespace(
        execute=False,
        acknowledge="",
        sim_dir=None,
        output_dir=tmp_path,
        held_aoa=10,
        updates=2000,
        batch_size=8,
    )
    with pytest.raises(PermissionError, match="acknowledgement"):
        subject.run(args, torch.device("cpu"))


def test_variant_evaluation_uses_candidate_control_and_zero_scores(
        monkeypatch) -> None:
    class Candidate:
        def __init__(self):
            self.zero = False

        def set_deformable_history_memory_zero_offset(self, enabled: bool):
            self.zero = enabled

        def state_dict(self):
            return {"weight": torch.tensor([1.0])}

    candidate = Candidate()
    control = object()
    keys = (*subject.base.POINT_WEIGHTS,
            "point_contribution_46_10_9")

    def fake_evaluate(model, *_args, **_kwargs):
        value = 10.0 if model is control else (20.0 if candidate.zero else 30.0)
        score = {key: value for key in keys}
        return {
            "aggregate": score,
            "per_trajectory": {"trajectory.h5": score},
            "prediction_sha256": str(value),
        }

    monkeypatch.setattr(subject, "evaluate_model", fake_evaluate)
    result = subject.evaluate_variants(
        {"control": control, "candidate": candidate},
        None, None, [], None, torch.device("cpu"))

    assert result["delta_candidate_minus_control"][
        "point_contribution_46_10_9"] == 20.0
    assert result["delta_candidate_minus_zero_offset"][
        "point_contribution_46_10_9"] == 10.0
    assert not candidate.zero
    assert result["candidate_state_unchanged_by_zero_ablation"]
