import torch

from models.direct_unbounded_absolute_interval_head import (
    DirectUnboundedAbsoluteIntervalHead,
    DirectUnboundedAbsoluteIntervalHeadConfig,
)


def small_head() -> DirectUnboundedAbsoluteIntervalHead:
    return DirectUnboundedAbsoluteIntervalHead(
        DirectUnboundedAbsoluteIntervalHeadConfig(
            feature_channels=(32, 16), projection_channels=4,
            hidden_channels=16, blocks=1, groups=4))


def test_default_unbounded_head_has_frozen_capacity_and_no_limits() -> None:
    config = DirectUnboundedAbsoluteIntervalHeadConfig()
    head = DirectUnboundedAbsoluteIntervalHead(config)
    assert sum(value.numel() for value in head.parameters()) == 45_568
    assert head.architecture_config()["architecture"] == (
        "direct_unbounded_absolute_interval_head")
    assert not hasattr(config, "max_log_width_ratio")
    assert not hasattr(config, "max_center_fraction")


def test_zero_output_initializes_reference_interval_and_gradients() -> None:
    head = small_head()
    history = torch.randn(2, 20, 16, 24, 3)
    prediction = torch.randn(2, 20, 16, 24, 3)
    features = [
        torch.randn(2, 32, 8, 12),
        torch.randn(2, 16, 16, 24),
    ]
    target_std = torch.tensor([0.1, 0.02, 1.0])
    lower, upper = head(history, prediction, features, target_std)
    expected = torch.tensor([0.0215 / 0.1 / 2.0, 0.011 / 0.02 / 2.0])
    torch.testing.assert_close(
        lower.mean(dim=(0, 1, 2, 3)), expected)
    torch.testing.assert_close(
        upper.mean(dim=(0, 1, 2, 3)), expected)
    (lower.square().mean() + upper.square().mean()).backward()
    assert head.output.weight.grad is not None


def test_raw_geometry_is_unbounded_and_strictly_ordered() -> None:
    head = small_head()
    raw = torch.zeros(1, 20, 2, 3, 4)
    raw[..., :2] = 3.0
    raw[..., 2:] = torch.tensor([4.0, -5.0])
    target_std = torch.tensor([0.1, 0.02, 1.0])
    lower, upper = head.extents_from_raw(raw, target_std)
    scale = torch.tensor([0.0215 / 0.1, 0.011 / 0.02])
    torch.testing.assert_close(
        (lower + upper)[0, 0, 0, 0], scale * torch.exp(torch.tensor(3.0)))
    torch.testing.assert_close(
        (upper - lower)[0, 0, 0, 0] / 2.0,
        scale * torch.tensor([4.0, -5.0]))
    assert torch.all(lower + upper > 0.0)
