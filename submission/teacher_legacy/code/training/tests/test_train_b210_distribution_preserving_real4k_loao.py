from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for value in (CODE_ROOT, TRAINING_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import train_b210_distribution_preserving_real4k_loao as subject  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig  # noqa: E402


def tiny_config() -> EnsembleUNet2dConfig:
    return EnsembleUNet2dConfig(
        in_steps=20, out_steps=20, in_channels=3, out_channels=3,
        base=8, depth=2, groups=1, difference_steps=5,
        use_coordinates=True, mean_fluctuation=True,
        predict_uncertainty=True, ensemble_members=4,
        uq_hidden_channels=8, member_adapter_channels=4,
        mean_preserving_member_adapters=True,
        detach_member_adapter_context=True, detach_uq_context=True,
        budgeted_disagreement_allocation=True,
        physical_energy_controller=True, physical_energy_member_sync=True,
        physical_energy_hidden_channels=4,
        residual_quantile_intervals=True,
    )


def test_forward_substitution_exposes_base_but_caches_adapted_members():
    model = EnsembleUNet2d(tiny_config()).eval()
    with torch.no_grad():
        model.member_point_heads[0][0].weight.normal_(mean=0.0, std=0.1)
        for adapter in model.member_feature_adapters[0]:
            adapter.body[-1].weight.normal_(mean=0.0, std=0.1)
            adapter.body[-1].bias.fill_(0.1)
    value = torch.randn(1, 20, 16, 16, 3)
    with subject.base_member_supervision_forward(model) as cache:
        returned = model.forward_uq_training(value)[1]
    assert torch.equal(returned, cache["base_members"])
    assert not torch.equal(cache["adapted_members"], cache["base_members"])
    assert torch.allclose(
        cache["adapted_members"].mean(dim=1),
        cache["base_members"].mean(dim=1), atol=2e-6, rtol=0.0)


def test_gate_requires_quality_and_noncollapsed_distribution():
    comparison = {
        "point_final_contribution_delta": 0.0,
        "non_time_final_contribution_delta": 0.10,
        "delta": {"native_sps_score": 0.40},
    }
    diagnostic = {
        "native_minus_collapsed_sps": 0.50,
        "member_error_pairwise_correlation_mean": 0.70,
        "element_spread_log_spearman": 0.60,
    }
    assert subject.decision_gate(comparison, diagnostic)["pass"] is True
    diagnostic["member_error_pairwise_correlation_mean"] = 0.90
    result = subject.decision_gate(comparison, diagnostic)
    assert result["pass"] is False
    assert result["conditions"]["members_not_collapsed"] is False


def test_exact_v7_config_and_parameter_contract():
    config = subject.expected_config()
    assert config.mean_preserving_member_adapters is True
    assert config.detach_member_adapter_context is True
    assert config.detach_uq_context is True
    assert config.budgeted_disagreement_allocation is True
    assert subject.EXPECTED_PARAMETERS == 180_355_823
