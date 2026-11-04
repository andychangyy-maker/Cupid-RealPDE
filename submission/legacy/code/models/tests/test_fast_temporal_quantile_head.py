from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.fast_temporal_quantile_head import (  # noqa: E402
    FactorizedSpatioTemporalBlock,
    FastTemporalQuantileHead,
    FastTemporalQuantileHeadConfig,
    fused_mean_point_heads,
    fused_mean_prediction,
)
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


def small_config(**overrides) -> FastTemporalQuantileHeadConfig:
    values = {
        "out_steps": 4,
        "feature_channels": (16, 8, 4),
        "projection_channels": 4,
        "flow_channels": 4,
        "hidden_channels": 8,
        "blocks": 2,
        "groups": 4,
        "quantile_levels": (0.05, 0.25, 0.5, 0.75, 0.95),
    }
    values.update(overrides)
    return FastTemporalQuantileHeadConfig(**values)


def test_config_rejects_even_temporal_kernel() -> None:
    with pytest.raises(ValueError, match="kernels must be odd"):
        small_config(temporal_kernel=2).validate()


def test_factorized_blocks_use_depthwise_space_and_time() -> None:
    block = FactorizedSpatioTemporalBlock(8, 3, 3, 4)
    assert block.spatial[0].groups == 8
    assert block.spatial[0].kernel_size == (1, 3, 3)
    assert block.temporal[0].groups == 8
    assert block.temporal[0].kernel_size == (3, 1, 1)


def test_fused_mean_point_heads_average_linear_members() -> None:
    class PointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.member_feature_adapters = None
            self.member_point_heads = torch.nn.ModuleList([
                torch.nn.ModuleList([torch.nn.Conv2d(2, 3, 1)]),
                torch.nn.ModuleList([torch.nn.Conv2d(2, 3, 1)]),
            ])

    model = PointModel()
    with torch.no_grad():
        model.member_point_heads[0][0].weight.fill_(1.0)
        model.member_point_heads[0][0].bias.fill_(2.0)
        model.member_point_heads[1][0].weight.fill_(3.0)
        model.member_point_heads[1][0].bias.fill_(4.0)
    fused = fused_mean_point_heads(model)
    torch.testing.assert_close(
        fused[0].weight, torch.full_like(fused[0].weight, 2.0))
    torch.testing.assert_close(
        fused[0].bias, torch.full_like(fused[0].bias, 3.0))
    value = torch.randn(2, 2, 4, 4)
    expected = torch.stack([
        heads[0](value) for heads in model.member_point_heads]).mean(dim=0)
    torch.testing.assert_close(fused[0](value), expected)


def test_fused_mean_point_heads_reject_member_adapters() -> None:
    class PointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.member_feature_adapters = torch.nn.ModuleList([
                torch.nn.ModuleList([torch.nn.Identity()])])
            self.member_point_heads = torch.nn.ModuleList([
                torch.nn.ModuleList([torch.nn.Conv2d(2, 3, 1)])])

    with pytest.raises(ValueError, match="no member feature adapters"):
        fused_mean_point_heads(PointModel())


def test_fused_prediction_matches_member_mean_in_float32() -> None:
    torch.manual_seed(2)
    model = EnsembleUNet2d(EnsembleUNet2dConfig(
        base=8, depth=3, groups=4, difference_steps=2,
        ensemble_members=3, uq_hidden_channels=8,
        mean_fluctuation=True)).eval()
    model.set_normalizer_statistics(
        torch.zeros(3), torch.zeros(3),
        torch.ones(3), torch.tensor([0.2, 0.1, 1.0]))
    value = torch.randn(1, 20, 16, 32, 3)
    with torch.inference_mode():
        features = model._decode(value)
        expected = model._members_from_features(value, features).mean(dim=1)
        fused = fused_mean_point_heads(model)
        actual = fused_mean_prediction(model, value, features, fused)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_head_outputs_monotone_quantiles_and_ordered_bounds() -> None:
    torch.manual_seed(0)
    head = FastTemporalQuantileHead(small_config()).eval()
    batch, steps, height, width = 2, 4, 8, 8
    x = torch.randn(batch, 6, height, width, 3)
    prediction = torch.randn(batch, steps, height, width, 3)
    features = [
        torch.randn(batch, 16, 2, 2),
        torch.randn(batch, 8, 4, 4),
        torch.randn(batch, 4, height, width),
    ]
    target_std = torch.tensor([0.3, 0.2, 1.0])

    with torch.inference_mode():
        quantiles, lower, upper = head(
            x, prediction, features, target_std)

    assert quantiles.shape == (batch, steps, height, width, 2, 5)
    assert lower.shape == prediction[..., :2].shape
    assert upper.shape == prediction[..., :2].shape
    assert torch.isfinite(quantiles).all()
    assert torch.all(quantiles[..., 1:] > quantiles[..., :-1])
    assert torch.all(lower + upper >= 0.0)


def test_endpoint_refinement_is_identity_at_initialization() -> None:
    head = FastTemporalQuantileHead(small_config()).eval()
    lower = torch.randn(2, 4, 8, 8, 2)
    upper = torch.randn(2, 4, 8, 8, 2)
    with torch.inference_mode():
        refined_lower, refined_upper = head.endpoint_refinement(lower, upper)
    torch.testing.assert_close(refined_lower, lower, rtol=0.0, atol=0.0)
    torch.testing.assert_close(refined_upper, upper, rtol=0.0, atol=0.0)


def test_short_axis_accumulation_matches_cumsum_reference() -> None:
    torch.manual_seed(1)
    head = FastTemporalQuantileHead(small_config()).eval()
    hidden = torch.randn(2, 8, 4, 8, 8)
    with torch.inference_mode():
        actual = head._monotone_quantiles(hidden)
        raw = head.output(hidden).reshape(
            2, 2, 5, 4, 8, 8).permute(0, 3, 4, 5, 1, 2)
        center = raw[..., 2:3]
        lower_gaps = torch.nn.functional.softplus(raw[..., :2])
        lower = center - torch.cumsum(
            lower_gaps.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))
        upper = center + torch.cumsum(
            torch.nn.functional.softplus(raw[..., 3:]), dim=-1)
        expected = torch.cat((lower, center, upper), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_default_head_is_lightweight() -> None:
    head = FastTemporalQuantileHead(FastTemporalQuantileHeadConfig())
    parameters = sum(parameter.numel() for parameter in head.parameters())
    assert parameters == 10_474


def test_member_free_stem_uses_only_features_and_flow() -> None:
    config = small_config()
    head = FastTemporalQuantileHead(config)
    expected = len(config.feature_channels) * config.projection_channels
    expected += config.flow_channels
    assert head.stem[0].in_channels == expected
    assert len(head.interval_candidate_pairs) == 8


def test_default_head_retains_fifteen_utility_pairs() -> None:
    head = FastTemporalQuantileHead(FastTemporalQuantileHeadConfig())
    assert len(head.interval_candidate_pairs) == 15
    assert head.architecture_config()["architecture"] == (
        "fast_temporal_quantile_head_fused_mean_member_free")


def test_default_track1_shape_contract() -> None:
    head = FastTemporalQuantileHead(
        FastTemporalQuantileHeadConfig()).eval()
    x = torch.zeros(1, 20, 32, 64, 3)
    prediction = torch.zeros(1, 20, 32, 64, 3)
    features = [
        torch.zeros(1, 512, 8, 16),
        torch.zeros(1, 256, 16, 32),
        torch.zeros(1, 128, 32, 64),
    ]
    with torch.inference_mode():
        quantiles, lower, upper = head(
            x, prediction, features,
            torch.tensor([0.3, 0.2, 1.0]))
    assert quantiles.shape == (1, 20, 32, 64, 2, 7)
    assert lower.shape == (1, 20, 32, 64, 2)
    assert upper.shape == (1, 20, 32, 64, 2)
