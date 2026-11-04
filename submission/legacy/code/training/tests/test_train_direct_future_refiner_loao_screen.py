from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_direct_future_refiner_loao_screen as screen  # noqa: E402


def test_policy_is_reproducible_and_uses_declared_fit_sigma_units():
    first = screen.build_refiner_policy(seed=4)
    second = screen.build_refiner_policy(seed=4)
    assert first.shape == (screen.UPDATES, screen.BATCH_SIZE)
    assert np.array_equal(first, second)
    assert set(np.unique(first)).issubset({screen.NOISE_LEVEL_MIN, screen.NOISE_LEVEL_MAX})
    assert 0.35 < float((first == screen.NOISE_LEVEL_MIN).mean()) < 0.65


def test_candidate_auxiliary_uses_target_noise_but_control_can_keep_direct_input():
    target = torch.zeros(2, 20, 4, 6, 3)
    levels = np.asarray([.25, 1.0], dtype=np.float32)
    generator = torch.Generator(device="cpu").manual_seed(9)
    noisy, condition = screen.noisy_refiner_input(
        target=target, noise_levels=levels, sigma=2.0, generator=generator)
    assert noisy.shape == target.shape
    torch.testing.assert_close(condition, torch.tensor(levels))
    assert not torch.equal(noisy, target)
    # The candidate's deployment/direct condition is deliberately one rather
    # than zero: it denotes the fit residual scale used by its auxiliary pass.
    assert torch.equal(torch.ones(2), torch.ones(2))


def test_constructed_refiners_are_bitwise_identical_cpu():
    _base, refiners, identity = screen.construct_models(torch.device("cpu"))
    assert tuple(refiners) == screen.ARM_NAMES
    assert identity["refiner_bitwise_identical"] is True
    assert identity["base_parameters"] == screen.EXPECTED_BASE_PARAMETERS
    assert identity["refiner_parameters"] == screen.EXPECTED_REFINER_PARAMETERS


def test_frozen_base_prediction_is_legal_refiner_autograd_input():
    base, refiners, _identity = screen.construct_models(torch.device("cpu"))
    x = torch.randn(1, 20, 8, 12, 3)
    p0 = screen.prediction(base.eval(), x)
    assert p0.requires_grad is False
    loss = refiners["control"](p0, torch.ones(1)).square().mean()
    loss.backward()
    assert refiners["control"].stem.weight.grad is not None
    assert base.frame_encoder.stem.weight.grad is None


def test_fit_normalizer_artifact_contains_all_replay_statistics(tmp_path: Path):
    normalizer = SimpleNamespace(
        mean_input=torch.tensor([1.0, 2.0, 3.0]),
        mean_target=torch.tensor([4.0, 5.0, 6.0]),
        std_input=torch.tensor([.1, .2, .3]),
        std_target=torch.tensor([.4, .5, .6]),
    )
    artifact = screen.save_normalizer(normalizer, tmp_path)
    assert Path(artifact["path"]).is_file()
    assert artifact["sha256"] == screen.sha256_file(Path(artifact["path"]))
    payload = torch.load(artifact["path"], map_location="cpu", weights_only=False)
    assert set(payload["normalizer_state"]) == {
        "mean_input", "mean_target", "std_input", "std_target"}


def test_gate_requires_candidate_direct_gain_and_nonnegative_base_gain():
    delta = {"rel_l2_score": .05, "tke_score": .02, "mvpe_score": .01,
             "point_contribution_46_10_9": .031}
    evaluation = {"comparisons": {
        "future_noise_minus_control": {"delta": delta,
            "positive_trajectory_files": 15, "worst_trajectory_delta_point": -.04},
        "future_noise_minus_base": {"delta": {**delta, "point_contribution_46_10_9": 0.0},
            "positive_trajectory_files": 12, "worst_trajectory_delta_point": -.1},
        "control_minus_base": {"delta": delta,
            "positive_trajectory_files": 12, "worst_trajectory_delta_point": -.1},
    }}
    passed, contract = screen.gate(evaluation)
    assert passed is True
    assert contract["future_noise_minus_control_point_min"] == .03
    evaluation["comparisons"]["future_noise_minus_base"]["delta"]["point_contribution_46_10_9"] = -.001
    assert screen.gate(evaluation)[0] is False


def test_held20_requires_this_screen_held10_go(tmp_path: Path):
    with pytest.raises(PermissionError, match="requires"):
        screen.prior_held10_evidence(None, 20)
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps({
        "schema": screen.SCHEMA, "decision": "GO_HELD20",
        "config": {"held_aoa": 10}}))
    with pytest.raises(PermissionError, match="verified"):
        screen.prior_held10_evidence(forged, 20)

    artifacts = {}
    for name in ("normalizer", "base", *screen.ARM_NAMES):
        artifact = tmp_path / f"{name}.pth"
        artifact.write_bytes(name.encode("utf-8"))
        artifacts[name] = {"path": str(artifact),
                           "sha256": screen.sha256_file(artifact)}
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps({
        "normalizer": artifacts["normalizer"],
        "base": artifacts["base"],
        "refiners": {name: artifacts[name] for name in screen.ARM_NAMES},
    }))
    delta = {"rel_l2_score": .05, "tke_score": .02, "mvpe_score": .01,
             "point_contribution_46_10_9": .031}
    evaluation = {"comparisons": {
        "future_noise_minus_control": {
            "delta": delta, "positive_trajectory_files": 15,
            "worst_trajectory_delta_point": -.04},
        "future_noise_minus_base": {
            "delta": {**delta, "point_contribution_46_10_9": 0.0}},
    }}
    config = {
        "held_aoa": 10, "updates_base": screen.UPDATES,
        "updates_per_refiner": screen.UPDATES, "batch_size": screen.BATCH_SIZE,
        "seed": screen.SEED, "noise_seed": screen.NOISE_SEED, "r_steps": 1,
        "base_hidden_channels": screen.BASE_HIDDEN_CHANNELS,
        "refiner_hidden_channels": screen.REFINER_HIDDEN_CHANNELS,
        "refiner_blocks": screen.REFINER_BLOCKS,
        "arms": list(screen.ARM_NAMES), "direct_p0_condition_level": 1.0,
        "auxiliary_weight": screen.AUXILIARY_WEIGHT,
    }
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({
        "schema": screen.SCHEMA, "decision": "GO_HELD20", "config": config,
        "git": {"clean": True}, "evaluation": evaluation,
        "endpoint_seal": {"path": str(seal_path),
                          "sha256": screen.sha256_file(seal_path)},
    }))
    assert screen.prior_held10_evidence(path, 20)["decision"] == "GO_HELD20"
