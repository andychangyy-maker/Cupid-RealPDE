from __future__ import annotations

import pytest
import torch

import train_b210_aoa_risk_held_fold as subject


def test_first_group_observation_sets_data_scaled_weights():
    ema, q = subject.update_group_weights(
        torch.zeros(4), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(ema, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(q, torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_later_group_observation_uses_frozen_ema_beta():
    ema, q = subject.update_group_weights(
        torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.full((4,), 2.0))
    expected = torch.tensor([1.05, 2.0, 2.95, 3.90])
    assert torch.allclose(ema, expected)
    assert torch.allclose(q, expected / expected.sum())


def test_partition_requires_two_samples_from_every_fit_aoa():
    metadata = [{"aoa": value} for value in (0, 5, 10, 15)]
    groups = subject.partition_batch(
        torch.tensor([0, 1, 2, 3, 3, 2, 1, 0]), metadata,
        (0, 5, 10, 15))
    assert {key: value.tolist() for key, value in groups.items()} == {
        0: [0, 7], 5: [1, 6], 10: [2, 5], 15: [3, 4]}
    with pytest.raises(RuntimeError, match="exactly 2"):
        subject.partition_batch(
            torch.tensor([0, 0, 0, 1, 2, 2, 3, 3]), metadata,
            (0, 5, 10, 15))


def test_discovery_gate_uses_public_non_time_weights():
    control = {
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 90.0,
        "native_sps_score": 40.0,
    }
    candidate = {key: value + 0.1 for key, value in control.items()}
    result = subject.discovery_comparison(candidate, control)
    assert result["point_final_contribution_delta"] == pytest.approx(0.065)
    assert result["non_time_final_contribution_delta"] == pytest.approx(0.09)
    assert result["decision"] == "PASS_AOA_RISK_DISCOVERY"

