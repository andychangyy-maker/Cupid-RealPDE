import torch

from models.distilled_interval_head import (
    DistilledIntervalHead,
    DistilledIntervalHeadConfig,
)


def test_direct_interval_head_shapes_order_and_gradients():
    config = DistilledIntervalHeadConfig(
        feature_channels=(32, 16), projection_channels=4,
        hidden_channels=16, blocks=1, groups=4)
    model = DistilledIntervalHead(config)
    history = torch.randn(2, 20, 16, 24, 3)
    prediction = torch.randn(2, 20, 16, 24, 3)
    features = [
        torch.randn(2, 32, 8, 12),
        torch.randn(2, 16, 16, 24),
    ]
    target_std = torch.tensor([0.1, 0.02, 1.0])
    lower, upper = model(
        history, prediction, features, target_std)
    assert lower.shape == (2, 20, 16, 24, 2)
    assert upper.shape == lower.shape
    assert torch.all(lower + upper > 0)
    expected = torch.tensor(
        [0.0215 / 0.1 / 2.0, 0.011 / 0.02 / 2.0])
    torch.testing.assert_close(
        lower.mean(dim=(0, 1, 2, 3)), expected)
    torch.testing.assert_close(
        upper.mean(dim=(0, 1, 2, 3)), expected)
    (lower.square().mean() + upper.square().mean()).backward()
    assert model.output.weight.grad is not None
