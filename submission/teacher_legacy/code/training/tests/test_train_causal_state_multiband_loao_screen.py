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

import train_causal_state_multiband_loao_screen as screen  # noqa: E402


def test_arms_are_bitwise_matched_with_only_rollout_semantics_different():
    arms, identity = screen.construct_arms(torch.device("cpu"))
    assert tuple(arms) == screen.ARM_NAMES
    assert identity["bitwise_identical"] is True
    assert len(set(identity["parameters"].values())) == 1
    assert identity["only_variable"] == "future_state_accumulation"


def test_schedule_is_paired_and_fit_only():
    schedule = screen.make_schedule([object()] * 31)
    assert schedule.shape == (screen.UPDATES, screen.BATCH_SIZE)
    assert int(schedule.min()) >= 0 and int(schedule.max()) < 31


class UnitNormalizer:
    mean_input = torch.zeros(1, 1, 1, 1, 3)
    std_input = torch.ones(1, 1, 1, 1, 3)
    mean_target = torch.zeros(1, 1, 1, 1, 3)
    std_target = torch.ones(1, 1, 1, 1, 3)


def test_loss_is_finite_and_transition_gets_first_backward_gradient():
    model = screen.construct_arms(torch.device("cpu"))[0]["cumulative_candidate"]
    loss, terms = screen.loss_terms(model, torch.randn(2, 20, 32, 64, 3),
                                    torch.randn(2, 20, 32, 64, 3), UnitNormalizer())
    loss.backward()
    assert np.isfinite([float(value.detach()) for value in terms.values()]).all()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
               for parameter in model.transition.parameters())


def valid_evaluation() -> dict[str, object]:
    delta = {"rel_l2_score": .08, "tke_score": .06, "mvpe_score": .05,
             "point_contribution_46_10_9": .051}
    return {"cumulative_candidate_minus_independent_control": {
                "delta": delta, "positive_trajectory_files": 15,
                "worst_trajectory_delta_point": -.05},
            "same_checkpoint_state_ablations": {
                "native_minus_independent": {"point_contribution_46_10_9": .011},
                "native_minus_frozen": {"point_contribution_46_10_9": .002}},
            "mechanical": {"native_vs_independent_state_rms": .01,
                           "native_vs_independent_prediction_rms": .001}}


def test_gate_requires_quality_state_causality_and_all_bands():
    evaluation = valid_evaluation()
    assert screen.gate(evaluation)[0] is True
    evaluation["same_checkpoint_state_ablations"]["native_minus_independent"]["point_contribution_46_10_9"] = .009
    assert screen.gate(evaluation)[0] is False
    evaluation = valid_evaluation()
    evaluation["mechanical"]["native_vs_independent_prediction_rms"] = 0.
    assert screen.gate(evaluation)[0] is False
    # Band activity is diagnostic only: a shared scaffold must not become a
    # fabricated hard responsibility condition.
    assert screen.gate(valid_evaluation())[0] is True


def test_held20_rejects_forged_go(tmp_path: Path):
    path = tmp_path / "fake.json"
    path.write_text('{"schema":"causal_state_multiband_whole_aoa_sim_screen_v1",'
                    '"decision":"GO_HELD20"}')
    with pytest.raises(PermissionError, match="verified held10 GO"):
        screen.prior_held10_evidence(path, 20)


def test_prior_rejects_contract_tampering_before_artifact_read(tmp_path: Path):
    path = tmp_path / "tampered.json"
    path.write_text('{"schema":"causal_state_multiband_whole_aoa_sim_screen_v1",'
                    '"decision":"GO_HELD20","git":{"clean":true},'
                    '"config":{"schema":"wrong"},"evaluation":{}}')
    with pytest.raises(PermissionError, match="verified held10 GO"):
        screen.prior_held10_evidence(path, 20)
