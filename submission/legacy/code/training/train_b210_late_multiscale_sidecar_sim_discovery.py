#!/usr/bin/env python3
"""Paired whole-AoA Sim screen for a late multiscale history sidecar.

Three arms retain the same mature B210 XCB/member/energy/UQ topology.  A
no-sidecar reference shares every common initial tensor with two matched
sidecar arms; those two differ only in whether preserved history is mixed
along causal time or spatial x with time folded into the batch.  Point
training and the held-field seal are inherited from the audited harness.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_b210_unified_temporal_state_sim_discovery as paired  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2dConfig  # noqa: E402


SCHEMA = "b210_late_multiscale_sidecar_whole_aoa_sim2k_v1"
ACKNOWLEDGEMENT = "RUN_B210_LATE_MULTISCALE_SIDECAR_SIM2K"
GO_DECISION = "GO_B210_LATE_SIDECAR_CONFIRMATION"
STOP_DECISION = "STOP_B210_LATE_MULTISCALE_SIDECAR"
PAIR_DIFFERENCE = "late_multiscale_sidecar_mode"
ARMS = ("no_sidecar", "spatial_control", "causal_temporal")
ARM_MODE = {
    "no_sidecar": "none",
    "spatial_control": "framewise_spatial_control",
    "causal_temporal": "causal_temporal",
}
SIDECAR_CHANNELS = (32, 48)
SIDECAR_BLOCKS = 1
SIDECAR_KERNEL = 5

_LEGACY_MODEL_CONFIG = paired.model_config


def model_config(
        mode: str, *, tiny: bool = False) -> EnsembleUNet2dConfig:
    if mode not in ARM_MODE.values():
        raise ValueError(f"unknown late sidecar mode {mode!r}")
    # Start from the exact mature B210 topology used by the previous paired
    # screen, remove its failed unified-state arm, and add only the late
    # two-scale sidecar.  A depth-four tiny model keeps the same contracts.
    legacy_mode = (
        "causal_temporal" if mode == "none" else mode)
    config = _LEGACY_MODEL_CONFIG(legacy_mode, tiny=False)
    overrides = dict(
        history_dynamics_state_mode="none",
        late_multiscale_sidecar_mode=mode,
        late_multiscale_sidecar_channels=SIDECAR_CHANNELS,
        late_multiscale_sidecar_blocks=SIDECAR_BLOCKS,
        late_multiscale_sidecar_kernel=SIDECAR_KERNEL,
    )
    if tiny:
        overrides.update(
            base=8, depth=4, groups=4,
            cross_conditioned_channels=8,
            uq_hidden_channels=8, member_adapter_channels=8,
            mean_profile_channels=8, temporal_uq_channels=8,
            physical_energy_hidden_channels=8,
            late_multiscale_sidecar_channels=(4, 8),
        )
    return replace(config, **overrides)


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def construct_triplet(
        device: torch.device, *, tiny: bool = False):
    models, starts = {}, {}
    for arm in ARMS:
        torch.manual_seed(paired.SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(paired.SEED)
        model = paired.EnsembleUNet2d(
            model_config(ARM_MODE[arm], tiny=tiny)).float().to(device)
        models[arm] = model
        starts[arm] = _snapshot(model)

    spatial = starts["spatial_control"]
    causal = starts["causal_temporal"]
    sidecar_pair_equal = (
        spatial.keys() == causal.keys()
        and all(torch.equal(value, causal[key])
                for key, value in spatial.items()))
    if not sidecar_pair_equal:
        raise RuntimeError("late sidecar paired arms are not bitwise identical")
    baseline = starts["no_sidecar"]
    common_equal = (
        baseline.keys() <= causal.keys()
        and all(torch.equal(value, causal[key])
                for key, value in baseline.items()))
    extras = sorted(set(causal) - set(baseline))
    if (not common_equal or not extras
            or not all(name.startswith("late_multiscale_sidecar.")
                       for name in extras)):
        raise RuntimeError(
            "no-sidecar and sidecar common-state identity changed")
    counts = {
        arm: sum(parameter.numel() for parameter in model.parameters())
        for arm, model in models.items()
    }
    if counts["spatial_control"] != counts["causal_temporal"]:
        raise RuntimeError("late sidecar paired parameter counts differ")
    return models, {
        "sidecar_pair_state_sha256": paired.state_sha(spatial),
        "no_sidecar_state_sha256": paired.state_sha(baseline),
        "sidecar_pair_bitwise_identical": sidecar_pair_equal,
        "common_state_bitwise_identical_to_no_sidecar": common_equal,
        "sidecar_extra_state_keys": extras,
        "parameter_counts": counts,
        "only_sidecar_pair_config_difference": PAIR_DIFFERENCE,
    }


def set_triplet_normalizer_statistics(models, normalizer):
    values = (
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target,
    )
    for model in models.values():
        model.set_normalizer_statistics(*values)
    names = ("persistence_scale", "persistence_bias", "target_std")
    reference = models[ARMS[0]]
    equality = {
        name: all(torch.equal(
            getattr(reference, name), getattr(models[arm], name))
            for arm in ARMS[1:])
        for name in names
    }
    if not all(equality.values()):
        raise RuntimeError(
            f"triplet normalizer-dependent buffers differ: {equality}")
    return {
        "normalizer_buffers_bitwise_equal_all_three_arms": equality,
        "normalizer_buffer_sha256": paired.state_sha({
            name: getattr(reference, name) for name in names}),
    }


def _comparison(candidate: Mapping[str, float | int],
                reference: Mapping[str, float | int]) -> dict[str, float]:
    keys = (*paired.POINT_WEIGHTS, "point_contribution_46_10_9")
    return {
        key: float(candidate[key]) - float(reference[key])
        for key in keys
    }


def evaluate_triplet(models, trajectories, windows, window_files,
                     normalizer, device: torch.device) -> dict[str, object]:
    file_counts = {
        name: window_files.count(name)
        for name in sorted(set(window_files))
    }
    if (len(file_counts) != paired.EXPECTED_FILES_PER_AOA
            or set(file_counts.values())
            != {paired.EXPECTED_WINDOWS_PER_TRAJECTORY}):
        raise RuntimeError("held evaluation panel changed")
    dataset = paired.SimWindowDataset(trajectories, windows)
    errors = {arm: [] for arm in ARMS}
    labels: list[str] = []
    for start in range(0, len(dataset), 16):
        indices = range(start, min(start + 16, len(dataset)))
        samples = [dataset[index] for index in indices]
        history = torch.stack([item[0] for item in samples]).to(device)
        target = torch.stack([item[1] for item in samples]).numpy()
        history = (
            history - normalizer.mean_input) / normalizer.std_input
        for arm in ARMS:
            models[arm].eval()
            with torch.inference_mode():
                prediction = models[arm](history)
                prediction = (
                    prediction * normalizer.std_target
                    + normalizer.mean_target)
            errors[arm].append(paired.error_rows(
                prediction.float().cpu().numpy(), target))
        labels.extend(window_files[index] for index in indices)
    joined = {
        arm: np.concatenate(parts) for arm, parts in errors.items()
    }
    aggregate = {arm: paired.score(joined[arm]) for arm in ARMS}
    temporal = aggregate["causal_temporal"]
    reference_arm = max(
        ("no_sidecar", "spatial_control"),
        key=lambda arm: float(
            aggregate[arm]["point_contribution_46_10_9"]))
    per_trajectory = {}
    label_array = np.asarray(labels)
    robust_gains = []
    for file_name in sorted(set(labels)):
        selected = label_array == file_name
        rows = {arm: paired.score(joined[arm][selected]) for arm in ARMS}
        local_reference = max(
            ("no_sidecar", "spatial_control"),
            key=lambda arm: float(
                rows[arm]["point_contribution_46_10_9"]))
        robust_delta = _comparison(
            rows["causal_temporal"], rows[local_reference])
        robust_gains.append(robust_delta["point_contribution_46_10_9"])
        per_trajectory[file_name] = {
            **rows,
            "strongest_reference_arm": local_reference,
            "delta_temporal_minus_strongest_reference": robust_delta,
        }
    return {
        "aggregate": aggregate,
        "strongest_reference_arm": reference_arm,
        "delta_temporal_minus_spatial": _comparison(
            temporal, aggregate["spatial_control"]),
        "delta_temporal_minus_no_sidecar": _comparison(
            temporal, aggregate["no_sidecar"]),
        "delta_temporal_minus_strongest_reference": _comparison(
            temporal, aggregate[reference_arm]),
        "per_trajectory": per_trajectory,
        "trajectory_count": len(per_trajectory),
        "windows_per_trajectory": paired.EXPECTED_WINDOWS_PER_TRAJECTORY,
        "positive_trajectories_vs_strongest_local_reference": sum(
            value > 0.0 for value in robust_gains),
        "worst_trajectory_delta_point_vs_strongest_local_reference": min(
            robust_gains),
    }


def decide_triplet(evaluation: Mapping[str, object]) -> dict[str, object]:
    delta = evaluation["delta_temporal_minus_strongest_reference"]
    if not isinstance(delta, Mapping):
        raise ValueError("triplet evaluation has no robust aggregate delta")
    complete = (
        int(evaluation.get("trajectory_count", -1))
        == paired.EXPECTED_FILES_PER_AOA
        and int(evaluation.get("windows_per_trajectory", -1))
        == paired.EXPECTED_WINDOWS_PER_TRAJECTORY)
    threshold = .04
    passed = (
        complete
        and float(delta["point_contribution_46_10_9"]) >= threshold)
    return {
        "primary_metric": (
            "temporal point minus max(spatial-control,no-sidecar) point"),
        "delta_point_min": threshold,
        "complete_panel": complete,
        "diagnostic_positive_trajectories": evaluation[
            "positive_trajectories_vs_strongest_local_reference"],
        "diagnostic_worst_trajectory_delta_point": evaluation[
            "worst_trajectory_delta_point_vs_strongest_local_reference"],
        "pass": passed,
    }


def model_metadata() -> dict[str, object]:
    return {
        "base": 210,
        "depth": 4,
        "difference_steps": 5,
        "backbone_residual_dropout": .20,
        "multi_scale_regime_film": True,
        "cross_conditioned": [64, 5],
        "member_adapter_channels": 64,
        "history_dynamics_state": None,
        "late_multiscale_sidecar": {
            "scales": ["half", "quarter"],
            "channels": list(SIDECAR_CHANNELS),
            "axis_blocks_per_scale": SIDECAR_BLOCKS,
            "axis_kernel": SIDECAR_KERNEL,
            "shared_observed_to_future": [20, 20],
            "fusion": "exact_zero_residual_into_decoder_scales_1_and_0",
        },
    }


def configure_harness() -> None:
    paired.SCHEMA = SCHEMA
    paired.ACKNOWLEDGEMENT = ACKNOWLEDGEMENT
    paired.GO_DECISION = GO_DECISION
    paired.STOP_DECISION = STOP_DECISION
    paired.EVIDENCE_CLASS = (
        "whole_AoA_held_Sim_late_decoder_causal_vs_spatial")
    paired.PAIR_DIFFERENCE = PAIR_DIFFERENCE
    paired.SAMPLE_STREAM_FIELD = "sample_stream_sha256_all_three_arms"
    paired.HELD_SEAL_FIELD = (
        "held_field_values_loaded_before_all_three_endpoint_seals")
    paired.ARMS = ARMS
    paired.ARM_MODE = ARM_MODE
    paired.model_config = model_config
    paired.model_metadata = model_metadata
    paired.construct_pair = construct_triplet
    paired.set_pair_normalizer_statistics = (
        set_triplet_normalizer_statistics)
    paired.evaluate_pair = evaluate_triplet
    paired.decide_gate = decide_triplet


def main() -> None:
    configure_harness()
    paired.main()


if __name__ == "__main__":
    main()
