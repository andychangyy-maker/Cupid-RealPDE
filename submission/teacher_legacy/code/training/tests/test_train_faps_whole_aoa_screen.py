from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_faps_whole_aoa_screen as screen  # noqa: E402


def test_pair_has_one_identical_initial_state_and_only_lambda_differs():
    arms, identity = screen.construct_pair(torch.device("cpu"))
    assert tuple(arms) == screen.ARM_NAMES
    assert identity["bitwise_identical"] is True
    assert identity["training_parameters"] == screen.EXPECTED_TRAINING_PARAMETERS
    assert identity["deployment_parameters"] == screen.EXPECTED_DEPLOYMENT_PARAMETERS
    assert identity["lambda_align"] == {"control": 0.0, "candidate": 1.0e-4}


def test_schedule_is_paired_and_contains_no_held_data_identity():
    schedule = screen.make_schedule([object()] * 48, 20, 8, np.random.default_rng(3))
    assert schedule.shape == (20, 8)
    assert int(schedule.min()) >= 0 and int(schedule.max()) < 48


def test_gate_requires_specified_quality_and_integrity_contracts():
    evaluation = {"candidate_minus_control": {"delta": {
        "rel_l2_score": .08, "tke_score": .06, "mvpe_score": .05,
        "point_contribution_46_10_9": .051},
        "positive_trajectory_files": 15, "worst_trajectory_delta_point": -.05},
        "latent_audits": {
            "control": {"history_future_matched_to_shuffled_ratio": .90},
            "candidate": {"z_history_batch_std": .01,
                "z_future_batch_std": .01, "z_history_effective_rank": 5.0,
                "z_future_effective_rank": 5.0,
                "decoder_fixed_permutation_delta": .001,
                "history_future_matched_to_shuffled_ratio": .80,
                "future_reconstruction_to_best_baseline_ratio": .80}}}
    passed, contract = screen.gate(evaluation)
    assert passed is True
    assert contract["candidate_minus_control_point_min"] == .05
    evaluation["candidate_minus_control"]["delta"]["tke_score"] = -.031
    assert screen.gate(evaluation)[0] is False


def test_loss_uses_stopgrad_future_alignment():
    arms, _ = screen.construct_pair(torch.device("cpu"))
    model = arms["candidate"]
    x = torch.randn(2, 20, 16, 48, 3)
    y = torch.randn(2, 20, 16, 48, 3)
    class UnitNormalizer:
        mean_input = torch.zeros(1, 1, 1, 1, 3)
        std_input = torch.ones(1, 1, 1, 1, 3)
        mean_target = torch.zeros(1, 1, 1, 1, 3)
        std_target = torch.ones(1, 1, 1, 1, 3)
    # A direct model-level check is more robust than depending on score losses.
    forecast = model.forecast_components(x)
    analysis = model.analysis_components(y, forecast["base"])
    alignment = (forecast["z_history"] - analysis["z_future"].detach()).square().mean()
    alignment.backward()
    assert any(parameter.grad is not None for parameter in model.forecaster.history_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.future_encoder.parameters())


def test_fit_only_gradient_audit_is_finite_and_uses_weighted_ratio():
    arms, _ = screen.construct_pair(torch.device("cpu"))
    model = arms["control"]
    x = torch.randn(2, 20, 16, 48, 3)
    y = torch.randn(2, 20, 16, 48, 3)

    class UnitNormalizer:
        mean_input = torch.zeros(1, 1, 1, 1, 3)
        std_input = torch.ones(1, 1, 1, 1, 3)
        mean_target = torch.zeros(1, 1, 1, 1, 3)
        std_target = torch.ones(1, 1, 1, 1, 3)

    audit = screen.initial_gradient_audit(model, x, y, UnitNormalizer())
    expected = (screen.LAMBDA_ALIGN["candidate"]
                * audit["unweighted_alignment_history_encoder_grad_norm"]
                / audit["forecast_history_encoder_grad_norm"])
    assert np.isfinite(list(audit.values())).all()
    assert audit["weighted_alignment_to_forecast_grad_ratio"] == pytest.approx(expected)


def test_latent_audit_compares_matched_shuffled_and_zero_states():
    arms, _ = screen.construct_pair(torch.device("cpu"))
    x = torch.randn(4, 20, 8, 12, 3)
    y = torch.randn(4, 20, 8, 12, 3)
    audit = screen.latent_audit(arms["candidate"], x, y)
    assert audit["z_history_effective_rank"] >= 1.0
    assert audit["z_future_effective_rank"] >= 1.0
    assert audit["history_future_shuffled_mse"] > 0.0
    assert audit["shuffled_future_reconstruction_mse"] > 0.0
    assert audit["zero_future_reconstruction_mse"] > 0.0


def test_held20_rejects_forged_decision_without_valid_gate(tmp_path: Path):
    forged = tmp_path / "forged.json"
    forged.write_text('{"schema":"faps_whole_aoa_sim_screen_v1",'
                      '"config":{"held_aoa":10},"decision":"GO_HELD20"}')
    with pytest.raises(PermissionError, match="verified held10 GO"):
        screen.prior_held10_evidence(forged, 20)
