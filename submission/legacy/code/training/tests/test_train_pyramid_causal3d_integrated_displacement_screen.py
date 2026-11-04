from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_pyramid_causal3d_integrated_displacement_screen as subject


def _metric(point: float) -> dict[str, float]:
    return {
        "rel_l2_score": point,
        "tke_rel_l2_score": point,
        "mvpe_rel_l2_score": point,
        "point_contribution_46_10_9": point,
    }


def _evaluation(*, control: float = 59.932626120502356,
                candidate: float = 60.01262612050236,
                delta_point: float = .08,
                positive: int = 15,
                worst: float = -.05) -> dict[str, object]:
    delta = _metric(0.0)
    delta["point_contribution_46_10_9"] = delta_point
    return {
        "aggregate": {
            "absolute_control": _metric(control),
            "integrated_candidate": _metric(candidate),
        },
        "delta_candidate_minus_control": delta,
        "positive_trajectories": positive,
        "worst_trajectory_delta_point": worst,
        "trajectory_count": 20,
        "windows_per_trajectory": 48,
    }


def _identity() -> dict[str, object]:
    return {
        "bitwise_identical": True,
        "only_architecture_difference": "displacement_semantics",
    }


def test_protocols_and_frozen_training_contract():
    assert subject.ARMS == ("absolute_control", "integrated_candidate")
    assert subject.ARM_SEMANTICS == {
        "absolute_control": "absolute",
        "integrated_candidate": "horizon_integrated",
    }
    assert (subject.SEED, subject.SCHEDULE_SEED) == (210, 220)
    assert (subject.UPDATES, subject.BATCH_SIZE) == (2000, 8)
    assert (subject.EXPECTED_PARAMETERS, subject.EXPECTED_STATE_TENSORS) == (
        8_929_552, 184)
    discovery = subject.protocol_for(10)
    assert discovery["fit_aoas"] == (0, 5, 15, 20)
    assert discovery["acknowledgement"] == (
        "RUN_PC3D_INTEGRATED_DISPLACEMENT_HELD10")
    assert discovery["gate"]["absolute_control_reference"] == pytest.approx(
        59.932626120502356)
    confirmation = subject.protocol_for(20)
    assert confirmation["fit_aoas"] == (0, 5, 10, 15)
    assert confirmation["acknowledgement"] == (
        "RUN_PC3D_INTEGRATED_DISPLACEMENT_HELD20")
    assert confirmation["gate"]["candidate_absolute_point_min"] == 59.18
    assert confirmation["gate"]["absolute_control_reference"] == pytest.approx(
        59.27145154391036)


def test_pair_is_bitwise_identical_and_only_semantics_differs():
    models, identity = subject.construct_pair(torch.device("cpu"), tiny=True)
    assert identity["bitwise_identical"]
    assert identity["only_architecture_difference"] == "displacement_semantics"
    assert identity["differing_architecture_keys"] == ["displacement_semantics"]
    assert identity["common_interaction_mode"] == "causal_temporal_candidate"
    assert len(set(identity["parameter_counts"].values())) == 1
    left, right = (models[arm].state_dict() for arm in subject.ARMS)
    assert left.keys() == right.keys()
    assert all(torch.equal(value, right[key]) for key, value in left.items())
    assert models["absolute_control"].displacement_semantics == "absolute"
    assert models["integrated_candidate"].displacement_semantics == (
        "horizon_integrated")


def test_displacement_execution_contract_and_normalizer():
    models, _identity = subject.construct_pair(torch.device("cpu"), tiny=True)
    normalizer = subject.base.ChannelNormalizer(
        torch.zeros(3), torch.ones(3), torch.full((3,), 2.0),
        torch.full((3,), 4.0), torch.device("cpu"))
    buffers = subject.set_pair_normalizer_statistics(models, normalizer)
    assert all(buffers["normalizer_buffers_bitwise_equal"].values())
    contract = subject.displacement_execution_contract(
        models, torch.device("cpu"), formal=False)
    assert contract["pass"]
    assert all(contract["checks"].values())
    assert contract["pure_basis"] == {
        "absolute_exact_identity": True,
        "integrated_exact_cumsum": True,
        "horizon_one_bitwise_equal": True,
        "future_impulse_causality": True,
    }


def test_fixed_schedule_hash_is_reused_exactly():
    schedule = np.random.default_rng(subject.SCHEDULE_SEED).integers(
        0, 3840, size=(subject.UPDATES, subject.BATCH_SIZE), dtype=np.int64)
    assert subject.validate_sample_index_schedule(schedule) == (
        subject.SAMPLE_INDEX_SHA256)
    schedule[0, 0] = (schedule[0, 0] + 1) % 3840
    with pytest.raises(RuntimeError, match="sample-index SHA mismatch"):
        subject.validate_sample_index_schedule(schedule)


def test_evaluator_arm_rename_is_recursive():
    raw = {
        "aggregate": {
            "spatial_window_control": {"x": 1},
            "temporal_candidate": {"x": 2},
        },
        "prediction_sha256": {
            "spatial_window_control": "a",
            "temporal_candidate": "b",
        },
    }
    renamed = subject._rename_evaluator_arms(raw)
    assert set(renamed["aggregate"]) == {
        "absolute_control", "integrated_candidate"}
    assert set(renamed["prediction_sha256"]) == {
        "absolute_control", "integrated_candidate"}


def test_gate_enforces_science_and_absolute_control_fidelity():
    protocol = subject.protocol_for(10)
    good = subject.decide_gate(
        _evaluation(), _identity(), {"pass": True}, {"pass": True},
        protocol["gate"])
    assert good["absolute_control_fidelity_pass"]
    assert good["science_pass"] and good["pass"]
    assert subject.decision_for(protocol, good) == (
        "GO_INTEGRATED_DISPLACEMENT_HELD20")

    weak = subject.decide_gate(
        _evaluation(delta_point=.079), _identity(), {"pass": True},
        {"pass": True}, protocol["gate"])
    assert not weak["science_pass"] and not weak["pass"]
    assert subject.decision_for(protocol, weak) == (
        "STOP_INTEGRATED_DISPLACEMENT")

    drifted = subject.decide_gate(
        _evaluation(control=59.95 + .02), _identity(), {"pass": True},
        {"pass": True}, protocol["gate"])
    assert not drifted["absolute_control_fidelity_pass"]
    assert subject.decision_for(protocol, drifted) == (
        "INCONCLUSIVE_ABSOLUTE_CONTROL_FIDELITY")


def test_mechanical_and_endpoint_guards_block_held_loader(tmp_path: Path):
    with pytest.raises(RuntimeError, match="missing endpoint"):
        subject.require_endpoint_seals({
            "absolute_control": {"endpoint_state_sha256": "x"}})
    endpoints = {arm: {"endpoint_state_sha256": arm} for arm in subject.ARMS}
    calls = []
    def loader(sim_dir, rows):
        calls.append((sim_dir, rows))
        return "loaded"
    healthy = {"pass": True, "checks": {"all": True}}
    assert subject.load_held_fields_after_guards(
        tmp_path, ["held"], endpoints, healthy, loader=loader) == "loaded"
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="before held load"):
        subject.load_held_fields_after_guards(
            tmp_path, ["held"], endpoints,
            {"pass": False, "checks": {"basis": False}}, loader=loader)
    assert len(calls) == 1


def test_acknowledgements_cannot_cross_and_cuda_precedes_data(tmp_path: Path):
    common = dict(execute=True, updates=2000, batch_size=8,
                  output_dir=tmp_path, sim_dir=tmp_path,
                  discovery_result=None)
    with pytest.raises(PermissionError, match="held10_discovery"):
        subject.run(SimpleNamespace(
            **common, held_aoa=10,
            acknowledge=subject.CONFIRMATION_ACKNOWLEDGEMENT),
            torch.device("cpu"))
    with pytest.raises(PermissionError, match="held20_independent_confirmation"):
        subject.run(SimpleNamespace(
            **common, held_aoa=20, acknowledge=subject.ACKNOWLEDGEMENT),
            torch.device("cpu"))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        subject.run(SimpleNamespace(
            **common, held_aoa=10, acknowledge=subject.ACKNOWLEDGEMENT),
            torch.device("cpu"))


def test_contract_cli_defaults_to_held10(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["integrated"])
    args = subject.parse_args()
    assert (args.mode, args.held_aoa, args.execute) == ("contract", 10, False)


def test_held20_discovery_prerequisite_is_fail_closed(tmp_path: Path):
    git = {"commit": "abc", "branch": "branch"}
    with pytest.raises(RuntimeError, match="held20 requires"):
        subject.validate_discovery_prerequisite(None, git)
    path = tmp_path / "held10.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="prerequisite failed"):
        subject.validate_discovery_prerequisite(path, git)
    path.write_text(__import__("json").dumps({
        "schema": subject.SCHEMA,
        "held_aoa": 10,
        "decision": "GO_INTEGRATED_DISPLACEMENT_HELD20",
        "gate": {"pass": True},
        "git": git,
    }), encoding="utf-8")
    record = subject.validate_discovery_prerequisite(path, git)
    assert all(record["checks"].values())
    assert len(record["sha256"]) == 64
