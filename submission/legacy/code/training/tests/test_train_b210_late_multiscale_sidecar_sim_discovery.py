from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "code" / "training"
    / "train_b210_late_multiscale_sidecar_sim_discovery.py")
SPEC = importlib.util.spec_from_file_location("late_sidecar_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_config_replaces_failed_unified_state_with_late_sidecar():
    config = MODULE.model_config("causal_temporal", tiny=True)
    assert config.depth == 4
    assert config.history_dynamics_state_mode == "none"
    assert config.late_multiscale_sidecar_mode == "causal_temporal"
    assert config.late_multiscale_sidecar_channels == (4, 8)
    config.validate()


def test_paired_tiny_triplet_has_exact_common_state_and_matched_sidecars():
    MODULE.configure_harness()
    models, identity = MODULE.paired.construct_pair(
        torch.device("cpu"), tiny=True)
    assert identity["sidecar_pair_bitwise_identical"]
    assert identity["common_state_bitwise_identical_to_no_sidecar"]
    assert identity["only_sidecar_pair_config_difference"] == (
        "late_multiscale_sidecar_mode")
    assert identity["parameter_counts"]["spatial_control"] == (
        identity["parameter_counts"]["causal_temporal"])
    assert identity["parameter_counts"]["no_sidecar"] < (
        identity["parameter_counts"]["causal_temporal"])
    assert models["causal_temporal"].late_multiscale_sidecar.mode == (
        "causal_temporal")
    assert models["spatial_control"].late_multiscale_sidecar.mode == (
        "framewise_spatial_control")


def test_harness_metadata_and_decisions_are_candidate_specific():
    MODULE.configure_harness()
    assert MODULE.paired.SCHEMA == MODULE.SCHEMA
    assert MODULE.paired.ACKNOWLEDGEMENT == MODULE.ACKNOWLEDGEMENT
    assert MODULE.paired.GO_DECISION == MODULE.GO_DECISION
    assert MODULE.paired.STOP_DECISION == MODULE.STOP_DECISION
    assert MODULE.paired.SAMPLE_STREAM_FIELD.endswith("all_three_arms")
    assert "all_three" in MODULE.paired.HELD_SEAL_FIELD
    assert MODULE.paired.model_metadata()["history_dynamics_state"] is None


def test_triplet_gate_uses_strongest_reference_delta_only():
    evaluation = {
        "delta_temporal_minus_strongest_reference": {
            "point_contribution_46_10_9": .041,
        },
        "trajectory_count": MODULE.paired.EXPECTED_FILES_PER_AOA,
        "windows_per_trajectory": (
            MODULE.paired.EXPECTED_WINDOWS_PER_TRAJECTORY),
        "positive_trajectories_vs_strongest_local_reference": 11,
        "worst_trajectory_delta_point_vs_strongest_local_reference": -.2,
    }
    decision = MODULE.decide_triplet(evaluation)
    assert decision["pass"]
    evaluation["delta_temporal_minus_strongest_reference"][
        "point_contribution_46_10_9"] = .039
    assert not MODULE.decide_triplet(evaluation)["pass"]
