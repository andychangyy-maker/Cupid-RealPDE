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

import train_pyramid_causal3d_loao_screen as subject


@dataclass(frozen=True)
class ToyConfig:
    e0_channels: int = 16
    e1_channels: int = 24
    e2_channels: int = 32
    e0_blocks: int = 2
    e1_blocks: int = 2
    e2_blocks: int = 2
    max_displacement: float = 1.5


class ToyAxisMixer(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.interaction_mode = mode
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, values):
        return values * self.weight


class ToyOperator(nn.Module):
    def __init__(self, config: ToyConfig, interaction_mode: str):
        super().__init__()
        self.config = config
        self.interaction_mode = interaction_mode
        self.axis_mixer = ToyAxisMixer(interaction_mode)
        self.observed_to_future_logits = nn.Parameter(torch.ones(1))
        self.horizon = nn.Parameter(torch.ones(1))
        self.coarse_spatial = nn.Linear(1, 1)
        self.mean_head = nn.Linear(1, 1)
        self.fluctuation_head = nn.Linear(1, 1)
        self.displacement_head = nn.Linear(1, 1)
        self.register_buffer("persistence_scale", torch.ones(3))
        self.register_buffer("persistence_bias", torch.zeros(3))
        self.register_buffer("target_std", torch.ones(3))
        self._zero_displacement_override = False

    def architecture_config(self):
        return self.config

    def set_normalizer_statistics(self, mean_input, mean_target,
                                  std_input, std_target):
        self.persistence_scale.copy_(std_input / std_target)
        self.persistence_bias.copy_((mean_input - mean_target) / std_target)
        self.target_std.copy_(std_target)

    def forward_components(self, history):
        values = self.axis_mixer(history)
        one = history.new_ones(1, 1)
        residual = (self.observed_to_future_logits.sum() + self.horizon.sum()
                    + self.coarse_spatial(one).sum() + self.mean_head(one).sum()
                    + self.fluctuation_head(one).sum()
                    + .01 * self.displacement_head(one).sum())
        displacement = self.config.max_displacement * torch.tanh(
            self.displacement_head(one)).reshape(1)
        return {"prediction": values + residual,
                "bounded_displacement": displacement}

    def forward(self, history):
        return self.forward_components(history)["prediction"]

    def set_zero_displacement_override(self, enabled: bool):
        self._zero_displacement_override = bool(enabled)


@pytest.fixture(autouse=True)
def toy_model(monkeypatch):
    monkeypatch.setattr(subject, "_model_classes", lambda: (ToyConfig, ToyOperator))


def test_protocol_and_training_contract_are_frozen() -> None:
    assert (subject.SEED, subject.SCHEDULE_SEED) == (210, 220)
    assert (subject.UPDATES, subject.BATCH_SIZE, subject.HELD_AOA) == (2000, 8, 10)
    assert subject.FIT_AOAS == (0, 5, 15, 20)
    assert subject.ACKNOWLEDGEMENT == "RUN_M16_PYRAMID_CAUSAL3D_SIM2K"
    assert subject.ARM_MODE == {
        "spatial_axis_control": "framewise_spatial_control",
        "temporal_candidate": "causal_temporal_candidate",
    }
    assert subject.GATE == {
        "delta_point_min": .08, "component_floor": -.05,
        "positive_trajectories_min": 15,
        "worst_trajectory_delta_point_min": -.05,
        "candidate_absolute_point_min": 59.85,
    }
    first = np.random.default_rng(220).integers(0, 3840, size=(2000, 8), dtype=np.int64)
    second = np.random.default_rng(220).integers(0, 3840, size=(2000, 8), dtype=np.int64)
    assert np.array_equal(first, second)


def test_held20_confirmation_protocol_is_independent_and_default(monkeypatch) -> None:
    protocol = subject.protocol_for(20)
    assert protocol["name"] == "held20_independent_confirmation"
    assert protocol["fit_aoas"] == (0, 5, 10, 15)
    assert protocol["schema"] == "m16_pyramid_causal3d_held20_confirmation_v1"
    assert protocol["acknowledgement"] == (
        "RUN_M16_PYRAMID_CAUSAL3D_HELD20_CONFIRMATION")
    assert protocol["gate"]["candidate_absolute_point_min"] == 59.18
    assert protocol["fit_inventory_sha256"] == (
        "9b5763d8a7ea6ebc870c90031733cd4049bc76d6a556da4e3c1cb1a36573cd93")
    assert protocol["held_inventory_sha256"] == (
        "d5a68f5c3c9ff093099b942aff9630f8f178bf7127171ed730c4608bf2907f9b")
    assert protocol["threshold_provenance"]["reference_point"] == 59.126205
    assert protocol["go_decision"] == "PASS_M16_CAUSAL3D_HELD20_CONFIRMATION"
    assert protocol["stop_decision"] == "FAIL_M16_CAUSAL3D_HELD20_CONFIRMATION"
    monkeypatch.setattr(sys, "argv", ["m16"])
    args = subject.parse_args()
    assert (args.mode, args.held_aoa, args.execute) == ("contract", 20, False)


def test_fixed_sample_index_sha_contract() -> None:
    schedule = np.random.default_rng(subject.SCHEDULE_SEED).integers(
        0, 3840, size=(subject.UPDATES, subject.BATCH_SIZE), dtype=np.int64)
    assert subject.validate_sample_index_schedule(schedule) == subject.SAMPLE_INDEX_SHA256
    changed = schedule.copy()
    changed[0, 0] = (changed[0, 0] + 1) % 3840
    with pytest.raises(RuntimeError, match="sample-index SHA mismatch"):
        subject.validate_sample_index_schedule(changed)


def test_inventory_metadata_seals_are_protocol_bound(monkeypatch) -> None:
    protocol = subject.protocol_for(20)
    fit_rows, held_rows = ["fit"], ["held"]
    monkeypatch.setattr(subject.base, "canonical_sha", lambda rows: (
        protocol["fit_inventory_sha256"] if rows is fit_rows
        else protocol["held_inventory_sha256"]))
    assert subject.validate_inventory_seals(protocol, fit_rows, held_rows) == {
        "fit_inventory_sha256": protocol["fit_inventory_sha256"],
        "held_inventory_sha256": protocol["held_inventory_sha256"],
    }
    monkeypatch.setattr(subject.base, "canonical_sha", lambda _rows: "bad")
    with pytest.raises(RuntimeError, match="inventory metadata seal mismatch"):
        subject.validate_inventory_seals(protocol, fit_rows, held_rows)


def test_pair_is_bitwise_identical_and_normalizer_buffers_match() -> None:
    models, identity = subject.construct_pair(torch.device("cpu"), tiny=True)
    assert identity["bitwise_identical"]
    assert identity["only_interaction_difference"] == "interaction_mode"
    assert len(set(identity["parameter_counts"].values())) == 1
    left, right = (models[arm].state_dict() for arm in subject.ARMS)
    assert left.keys() == right.keys()
    assert all(torch.equal(value, right[key]) for key, value in left.items())
    normalizer = subject.base.ChannelNormalizer(
        torch.zeros(3), torch.ones(3), torch.full((3,), 2.0),
        torch.full((3,), 4.0), torch.device("cpu"))
    record = subject.set_pair_normalizer_statistics(models, normalizer)
    assert all(record["normalizer_buffers_bitwise_equal"].values())


def test_axis_execution_and_displacement_contract(monkeypatch) -> None:
    from models.pyramid_causal3d_operator import (  # noqa: PLC0415
        PyramidCausal3DOperatorConfig, PyramidCausal3DOperator)
    monkeypatch.setattr(subject, "_model_classes", lambda: (
        PyramidCausal3DOperatorConfig, PyramidCausal3DOperator))
    models, _identity = subject.construct_pair(torch.device("cpu"), tiny=True)
    record = subject.axis_execution_contract(models, torch.device("cpu"), formal=True)
    assert record["pass"]
    assert record["paired_block_io_shapes_equal"]
    assert record["packed_shapes_independently_expected"]
    first = record["derived_per_block"]["stage0_block0"]
    assert first["expected_packed_shape"]["temporal_candidate"] == [1, 8, 20, 32, 64]
    assert first["expected_packed_shape"]["spatial_axis_control"] == [20, 8, 64, 32, 1]
    for arm in subject.ARMS:
        displacement = record["arms"][arm]["bounded_displacement"]
        assert displacement["finite"] and displacement["within_declared_bound"]
    normalizer = subject.base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3),
        torch.device("cpu"))
    subject.set_pair_normalizer_statistics(models, normalizer)
    dataset = subject.base.SimWindowDataset(
        [torch.randn(40, 16, 16, 3)], [subject.base.SimWindow(0, 0)])
    assert subject.endpoint_component_health(
        models, dataset, normalizer, torch.device("cpu"))["pass"]
    candidate = models["temporal_candidate"].train()
    candidate(torch.randn(1, 20, 16, 16, 3)).square().mean().backward()
    gradient = subject._gradient_audit(candidate)
    assert gradient["required_tensor_count"] == 16
    assert gradient["all_required_tensors_nonzero_finite"]


def test_gradient_audit_has_all_key_groups() -> None:
    model = ToyOperator(ToyConfig(), "causal_temporal_candidate")
    model(torch.randn(1, 20, 8, 8, 3)).square().mean().backward()
    audit = subject._gradient_audit(model)
    assert audit["all_seen_gradients_finite"]
    assert audit["all_required_groups_live"]
    assert audit["trainable_tensor_coverage"] > .5


def test_endpoint_seal_guard() -> None:
    with pytest.raises(RuntimeError, match="missing endpoint"):
        subject.require_endpoint_seals({subject.ARMS[0]: {"endpoint_state_sha256": "x"}})
    subject.require_endpoint_seals({
        arm: {"endpoint_state_sha256": arm} for arm in subject.ARMS})


def test_mechanical_health_is_required_before_held_load() -> None:
    identity = {"bitwise_identical": True,
                "parameter_counts": {arm: 10 for arm in subject.ARMS}}
    normalizer = {"normalizer_buffers_bitwise_equal": {
        name: True for name in subject.NORMALIZER_BUFFER_NAMES}}
    resources = {"parameter_counts": {arm: 10 for arm in subject.ARMS},
                 "health": {"status": "PASS"}}
    endpoints = {arm: {"gradient_audit": {
        "all_required_groups_seen_over_training": True,
        "all_required_tensors_seen_over_training": True,
        "required_tensors_nonfinite_over_training": [],
        "all_seen_gradients_finite": True,
        "nonfinite_gradient_tensors_over_training": [],
    }} for arm in subject.ARMS}
    report = subject.mechanical_health(
        identity, normalizer, {"pass": True}, resources, endpoints,
        {"pass": True})
    assert report["pass"]
    subject.require_mechanical_health(report)
    bad = subject.mechanical_health(
        identity, normalizer, {"pass": False}, resources, endpoints,
        {"pass": True})
    with pytest.raises(RuntimeError, match="before held load"):
        subject.require_mechanical_health(bad)


def test_held_loader_is_unreachable_with_bad_seals_or_health(tmp_path: Path) -> None:
    calls = []
    def loader(sim_dir, rows):
        calls.append((sim_dir, rows))
        return "loaded"
    endpoints = {arm: {"endpoint_state_sha256": arm} for arm in subject.ARMS}
    healthy = {"pass": True, "checks": {"all": True}}
    assert subject.load_held_fields_after_guards(
        tmp_path, ["held"], endpoints, healthy, loader=loader) == "loaded"
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="missing endpoint"):
        subject.load_held_fields_after_guards(
            tmp_path, ["held"], {subject.ARMS[0]: endpoints[subject.ARMS[0]]},
            healthy, loader=loader)
    assert len(calls) == 1
    unhealthy = {"pass": False, "checks": {"axis": False}}
    with pytest.raises(RuntimeError, match="before held load"):
        subject.load_held_fields_after_guards(
            tmp_path, ["held"], endpoints, unhealthy, loader=loader)
    assert len(calls) == 1


def test_dependency_probe_is_same_checkpoint_and_not_a_go_claim(monkeypatch) -> None:
    model = ToyOperator(ToyConfig(), "causal_temporal_candidate")
    def metric(point):
        return {"rel_l2_score": point, "tke_rel_l2_score": point,
                "mvpe_rel_l2_score": point,
                "point_contribution_46_10_9": point,
                "mean_physical_errors": {"rel_l2_error": 1.0,
                                         "tke_rel_l2_error": 1.0,
                                         "mvpe_rel_l2_error": 1.0}}
    monkeypatch.setattr(subject, "_evaluate_same_model", lambda candidate, *_args: {
        "aggregate": metric(59.0), "per_prefix_horizon": {},
        "per_trajectory": {}, "prediction_sha256": "zero"})
    native = {"aggregate": {"temporal_candidate": metric(60.0)}}
    result = subject.endpoint_dependency_diagnostics(
        model, native, None, None, [], None, torch.device("cpu"))
    assert not result["axis_bypass"]["available"]
    assert result["zero_warp"]["available"]
    assert result["zero_warp"]["same_checkpoint_state_unchanged"]
    assert not result["zero_warp"]["causal_go_claim"]


def test_gate_enforces_point_components_panel_and_axis_contract() -> None:
    delta = {key: 0.0 for key in (*subject.POINT_WEIGHTS,
                                 "point_contribution_46_10_9")}
    delta["point_contribution_46_10_9"] = .08
    evaluation = {"delta_candidate_minus_control": delta,
                  "aggregate": {"temporal_candidate": {
                      "point_contribution_46_10_9": 59.85}},
                  "positive_trajectories": 15,
                  "worst_trajectory_delta_point": -.05,
                  "trajectory_count": 20, "windows_per_trajectory": 48}
    identity = {"bitwise_identical": True,
                "only_interaction_difference": "interaction_mode"}
    mechanical = {"pass": True}
    assert subject.decide_gate(evaluation, identity, {"pass": True}, mechanical)["pass"]
    assert not subject.decide_gate(evaluation, identity, {"pass": False}, mechanical)["pass"]
    bad = {**evaluation, "positive_trajectories": 14}
    assert not subject.decide_gate(bad, identity, {"pass": True}, mechanical)["pass"]
    low = {**evaluation, "aggregate": {"temporal_candidate": {
        "point_contribution_46_10_9": 59.849}}}
    gate = subject.decide_gate(low, identity, {"pass": True}, mechanical)
    assert gate["causal_attribution_pass"]
    assert not gate["investment_pass"] and not gate["pass"]
    confirmation_gate = subject.protocol_for(20)["gate"]
    confirmation = {**evaluation, "aggregate": {"temporal_candidate": {
        "point_contribution_46_10_9": 59.18}}}
    assert subject.decide_gate(
        confirmation, identity, {"pass": True}, mechanical,
        confirmation_gate)["pass"]
    confirmation["aggregate"]["temporal_candidate"][
        "point_contribution_46_10_9"] = 59.179
    assert not subject.decide_gate(
        confirmation, identity, {"pass": True}, mechanical,
        confirmation_gate)["pass"]


def test_held20_decision_is_tri_state_and_held10_is_unchanged() -> None:
    confirmation = subject.protocol_for(20)
    assert subject.decision_for(confirmation, {
        "causal_attribution_pass": False, "investment_pass": True,
        "pass": False}) == "FAIL_M16_CAUSAL3D_HELD20_CONFIRMATION"
    assert subject.decision_for(confirmation, {
        "causal_attribution_pass": True, "investment_pass": False,
        "pass": False}) == "MECHANISM_ONLY_STOP_M16_INVESTMENT"
    assert subject.decision_for(confirmation, {
        "causal_attribution_pass": True, "investment_pass": True,
        "pass": True}) == "PASS_M16_CAUSAL3D_HELD20_CONFIRMATION"
    discovery = subject.protocol_for(10)
    assert subject.decision_for(discovery, {
        "causal_attribution_pass": True, "investment_pass": False,
        "pass": False}) == "STOP_M16_CAUSAL3D"
    assert subject.decision_for(discovery, {
        "causal_attribution_pass": True, "investment_pass": True,
        "pass": True}) == "GO_M16_CAUSAL3D_CONFIRMATORY_SIM"


def test_protocol_alias_and_formal_ack_guard(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["m16", "--acknowledge-protocol", "ok"])
    assert subject.parse_args().acknowledge == "ok"
    args = SimpleNamespace(execute=False, acknowledge="", held_aoa=10,
                           updates=2000, batch_size=8, output_dir=tmp_path,
                           sim_dir=tmp_path)
    with pytest.raises(PermissionError, match="exact protocol acknowledgement"):
        subject.run(args, torch.device("cpu"))


def test_protocol_acknowledgements_cannot_be_crossed(tmp_path: Path) -> None:
    common = dict(execute=True, updates=2000, batch_size=8,
                  output_dir=tmp_path, sim_dir=tmp_path)
    with pytest.raises(PermissionError, match="held20_independent_confirmation"):
        subject.run(SimpleNamespace(
            **common, held_aoa=20, acknowledge=subject.ACKNOWLEDGEMENT),
            torch.device("cpu"))
    with pytest.raises(PermissionError, match="held10_discovery"):
        subject.run(SimpleNamespace(
            **common, held_aoa=10,
            acknowledge=subject.CONFIRMATION_ACKNOWLEDGEMENT),
            torch.device("cpu"))
    # Exact acknowledgements pass the protocol gate and reach the independent
    # formal-CUDA guard before touching an output directory or data.
    with pytest.raises(RuntimeError, match="requires CUDA"):
        subject.run(SimpleNamespace(
            **common, held_aoa=20,
            acknowledge=subject.CONFIRMATION_ACKNOWLEDGEMENT),
            torch.device("cpu"))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        subject.run(SimpleNamespace(
            **common, held_aoa=10, acknowledge=subject.ACKNOWLEDGEMENT),
            torch.device("cpu"))
