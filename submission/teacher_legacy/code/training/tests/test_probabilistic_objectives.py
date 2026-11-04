from __future__ import annotations

from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from training.probabilistic_objectives import (
    conditional_distribution_targets, functional_energy_score,
    pointwise_ensemble_crps)


def test_conditional_distribution_target_routing_preserves_identity():
    base = torch.zeros(1, 2, 2, 2, 2)
    future = torch.ones_like(base)
    member, uq = conditional_distribution_targets(
        base, future, members_only=True)
    assert member is future
    assert uq is base
    member, uq = conditional_distribution_targets(
        base, future, members_only=False)
    assert member is future
    assert uq is future


def test_functional_energy_score_matches_two_member_definition():
    target = torch.zeros(1, 1, 1, 1, 2)
    members = torch.tensor([[[[[[1.0, 0.0]]]], [[[[3.0, 0.0]]]]]])
    score, attraction, repulsion = functional_energy_score(members, target)
    expected_attraction = (
        torch.tensor(0.5).sqrt() + torch.tensor(4.5).sqrt()) / 2
    expected_repulsion = torch.tensor(2.0).sqrt()
    assert torch.allclose(attraction, expected_attraction)
    assert torch.allclose(repulsion, expected_repulsion)
    assert torch.allclose(score, expected_attraction - 0.5 * expected_repulsion)


def test_functional_energy_score_rewards_noncollapsed_covering_members():
    target = torch.ones(2, 2, 2, 2, 2)
    collapsed = torch.zeros(2, 4, 2, 2, 2, 2, requires_grad=True)
    covering = collapsed.detach().clone()
    covering[:, 0] = 0.5
    covering[:, 1] = 1.0
    covering[:, 2] = 1.0
    covering[:, 3] = 1.5
    collapsed_score = functional_energy_score(collapsed, target)[0]
    covering_score = functional_energy_score(covering, target)[0]
    assert covering_score < collapsed_score
    collapsed_score.backward()
    assert torch.isfinite(collapsed.grad).all()


def test_pointwise_crps_matches_empirical_two_member_definition():
    target = torch.zeros(1, 1, 1, 1, 2)
    members = torch.tensor([[[[[[1.0, -1.0]]]], [[[[3.0, 1.0]]]]]])
    score, attraction, repulsion = pointwise_ensemble_crps(members, target)
    # Four absolute member/target scalar errors average to 1.5.  The ordered
    # 2x2 member-pair matrix averages to 1.0 per scalar element.
    assert torch.allclose(attraction, torch.tensor(1.5))
    assert torch.allclose(repulsion, torch.tensor(1.0))
    assert torch.allclose(score, torch.tensor(1.0))


def test_pointwise_crps_has_finite_local_member_gradients():
    target = torch.randn(2, 3, 4, 5, 2)
    members = torch.randn(2, 4, 3, 4, 5, 2, requires_grad=True)
    score = pointwise_ensemble_crps(members, target)[0]
    score.backward()
    assert torch.isfinite(members.grad).all()
    assert torch.count_nonzero(members.grad) > 0
