from __future__ import annotations

import json
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

import train_flow_backbone as backbone  # noqa: E402
import train_gap_conditioned_multilead_loao_screen as screen  # noqa: E402


def test_schedule_keeps_48_standard_windows_and_all_gaps_valid():
    assert screen.GAP_CHOICES == (0, 10, 20)
    windows = [backbone.common.scratch.SimWindow(0, start)
               for start in range(0, 1_000 - 40, 20)]
    assert len(windows) == 48
    schedule = screen.build_schedule(
        windows, updates=20, batch_size=8, rng=np.random.default_rng(4))
    assert schedule.shape == (20, 8, 3)
    assert set(schedule[..., 2].reshape(-1)).issubset(set(screen.GAP_CHOICES))
    # s=940, lead=20 predicts [980,1000), which is legal and must not be
    # accidentally excluded by the variable-lead screen.
    assert max(int(row[1]) + 40 + int(row[2])
               for row in schedule.reshape(-1, 3)) <= 1_000


def test_materialized_blind_and_conditioned_share_exact_examples():
    trajectory = torch.arange(1_000 * 2 * 3 * 3, dtype=torch.float32).reshape(
        1_000, 2, 3, 3)
    rows = np.asarray([[0, 940, 20], [0, 100, 5]], dtype=np.int64)
    blind = screen.materialize_batch([trajectory], rows, random_lead=True)
    conditioned = screen.materialize_batch([trajectory], rows, random_lead=True)
    fixed = screen.materialize_batch([trajectory], rows, random_lead=False)
    for left, right in zip(blind, conditioned, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    assert torch.equal(blind[2], torch.tensor([20, 5]))
    assert torch.equal(fixed[2], torch.zeros(2, dtype=torch.long))
    assert not torch.equal(blind[1], fixed[1])


def test_arm_policy_exposes_gap_only_in_conditioned_arm():
    assert screen.arm_policy("fixed_lead0") == (False, False)
    assert screen.arm_policy("random_lead_blind") == (True, False)
    assert screen.arm_policy("random_lead_conditioned") == (True, True)


def test_held20_requires_a_valid_held10_go(tmp_path: Path):
    with pytest.raises(PermissionError, match="requires"):
        screen.prior_held10_evidence(None, 20)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema": screen.SCHEMA, "decision": "STOP_MULTILEAD",
                                   "config": {"held_aoa": 10}}))
    with pytest.raises(PermissionError, match="not a held10 GO"):
        screen.prior_held10_evidence(invalid, 20)
    valid = tmp_path / "held10.json"
    valid.write_text(json.dumps({"schema": screen.SCHEMA, "decision": "GO_HELD20",
                                 "config": {"held_aoa": 10}}))
    evidence = screen.prior_held10_evidence(valid, 20)
    assert evidence is not None and evidence["decision"] == "GO_HELD20"


def test_constructed_arms_have_identical_augmented_initial_state_cpu():
    arms, identity = screen.construct_arms(torch.device("cpu"))
    assert tuple(arms) == screen.ARM_NAMES
    assert identity["bitwise_identical"] is True
    # The new screen keeps a single 3x128 bias-free projection (384 weights),
    # not an unrecorded no-parameter deterministic condition.
    assert identity["parameters"] == 1_343_496


def test_gate_uses_conditioned_vs_direct_as_primary_comparison():
    component_delta = {
        "rel_l2_score": .08, "tke_score": .06, "mvpe_score": .05,
        "point_contribution_46_10_9": .0473,
    }
    direct_delta = dict(component_delta)
    direct_delta["point_contribution_46_10_9"] = .051
    evaluation = {"conditioned_comparisons": {
        "conditioned_minus_fixed_lead0": {
            "delta": direct_delta, "positive_trajectory_files": 16,
            "worst_trajectory_delta_point": -.02,
        },
        # This comparison is explanatory only.  It must not turn an actual
        # improvement over the competition-task control into a failed gate.
        "conditioned_minus_random_lead_blind": {
            "delta": component_delta, "positive_trajectory_files": 14,
            "worst_trajectory_delta_point": -.06,
        },
    }}
    passed, contract = screen.gate(evaluation)
    assert passed is True
    assert contract["conditioned_minus_blind_is_diagnostic_only"] is True
