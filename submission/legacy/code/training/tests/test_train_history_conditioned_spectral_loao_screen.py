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

import train_history_conditioned_spectral_loao_screen as screen  # noqa: E402


def test_three_arms_are_bitwise_identical_and_parameter_matched():
    arms, identity = screen.construct_arms(torch.device("cpu"))
    assert tuple(arms) == screen.ARM_NAMES
    assert identity["bitwise_identical"] is True
    assert len(set(identity["parameters"].values())) == 1
    assert identity["primary_comparison"] == "conditional_rank_minus_static_rank"


def test_paired_schedule_has_only_fit_indexes():
    schedule = screen.make_schedule([object()] * 27)
    assert schedule.shape == (screen.UPDATES, screen.BATCH_SIZE)
    assert int(schedule.min()) >= 0 and int(schedule.max()) < 27


def valid_evaluation() -> dict:
    delta = {"rel_l2_score": .08, "tke_score": .06, "mvpe_score": .05,
             "point_contribution_46_10_9": .051}
    return {"conditional_rank_minus_static_rank": {
                "delta": delta, "positive_trajectory_files": 15,
                "worst_trajectory_delta_point": -.05},
            "conditional_rank_minus_static_base": {
                "delta": {**delta, "point_contribution_46_10_9": .031}},
            "conditional_gate": {"actual_gate_per_rank_std_mean": 1.1e-5},
            "conditional_checkpoint_ablations": {
                "native_minus_condition_disabled": {
                    "point_contribution_46_10_9": .011},
                "native_minus_cross_trajectory_shuffled": {
                    "point_contribution_46_10_9": .006}}}


def test_gate_requires_primary_auxiliary_and_noncollapsed_condition():
    evaluation = valid_evaluation()
    assert screen.gate(evaluation)[0] is True
    evaluation["conditional_rank_minus_static_rank"]["delta"]["tke_score"] = -.031
    assert screen.gate(evaluation)[0] is False
    evaluation = valid_evaluation()
    evaluation["conditional_gate"]["actual_gate_per_rank_std_mean"] = 0.
    assert screen.gate(evaluation)[0] is False
    evaluation = valid_evaluation()
    evaluation["conditional_checkpoint_ablations"][
        "native_minus_cross_trajectory_shuffled"][
            "point_contribution_46_10_9"] = .0049
    assert screen.gate(evaluation)[0] is False


def test_gate_shuffle_keeps_window_ordinal_and_crosses_trajectory():
    files = [file for file in ("a", "b", "c") for _ in range(4)]
    mapping = screen.cross_trajectory_shuffle_index(files)
    assert mapping.tolist() == [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3]
    assert all(files[index] != files[int(source)]
               for index, source in enumerate(mapping))


def test_loss_and_conditioner_gradient_are_finite():
    model = screen.construct_arms(torch.device("cpu"))[0]["conditional_rank"]
    x = torch.randn(2, 20, 32, 64, 3)
    y = torch.randn(2, 20, 32, 64, 3)

    class UnitNormalizer:
        mean_input = torch.zeros(1, 1, 1, 1, 3)
        std_input = torch.ones(1, 1, 1, 1, 3)
        mean_target = torch.zeros(1, 1, 1, 1, 3)
        std_target = torch.ones(1, 1, 1, 1, 3)

    loss, terms = screen.loss_terms(model, x, y, UnitNormalizer())
    loss.backward()
    assert np.isfinite([float(value.detach()) for value in terms.values()]).all()
    assert any(parameter.grad is not None for parameter in model.conditioner.parameters())


def test_held20_rejects_forged_go(tmp_path: Path):
    path = tmp_path / "fake.json"
    path.write_text('{"schema":"history_conditioned_spectral_whole_aoa_sim_screen_v1",'
                    '"config":{"held_aoa":10},"decision":"GO_HELD20"}')
    with pytest.raises(PermissionError, match="verified held10 GO"):
        screen.prior_held10_evidence(path, 20)
