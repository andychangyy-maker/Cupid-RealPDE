import torch

from models.distilled_flow_corrector import (
    DistilledFlowCorrector,
    DistilledFlowCorrectorConfig,
)


def test_zero_initialized_corrector_is_identity():
    model = DistilledFlowCorrector(DistilledFlowCorrectorConfig(
        hidden_channels=16, blocks=1, groups=4))
    history = torch.randn(2, 20, 16, 24, 3)
    prediction = torch.randn(2, 20, 16, 24, 3)
    lower = torch.rand(2, 20, 16, 24, 2) + 0.1
    upper = torch.rand(2, 20, 16, 24, 2) + 0.1
    corrected, new_lower, new_upper = model.correct_distribution(
        history, prediction, lower, upper)
    torch.testing.assert_close(corrected, prediction)
    torch.testing.assert_close(new_lower, lower)
    torch.testing.assert_close(new_upper, upper)


def test_corrector_shapes_order_and_gradients():
    model = DistilledFlowCorrector(DistilledFlowCorrectorConfig(
        hidden_channels=16, blocks=1, groups=4))
    with torch.no_grad():
        model.point_head.bias.fill_(0.05)
        model.interval_head.bias.fill_(0.1)
    history = torch.randn(2, 20, 16, 24, 3)
    prediction = torch.randn(2, 20, 16, 24, 3)
    lower = torch.rand(2, 20, 16, 24, 2) + 0.1
    upper = torch.rand(2, 20, 16, 24, 2) + 0.1
    corrected, new_lower, new_upper = model.correct_distribution(
        history, prediction, lower, upper)
    assert corrected.shape == prediction.shape
    assert new_lower.shape == lower.shape
    assert new_upper.shape == upper.shape
    assert torch.all(new_lower + new_upper > 0)
    fluctuation_delta = (
        corrected[..., :2] - prediction[..., :2])
    fluctuation_delta = (
        fluctuation_delta - fluctuation_delta.mean(dim=1, keepdim=True))
    torch.testing.assert_close(
        fluctuation_delta.mean(dim=1),
        torch.zeros_like(fluctuation_delta[:, 0]), atol=1e-6, rtol=0)
    (corrected.square().mean()
     + new_lower.square().mean()
     + new_upper.square().mean()).backward()
    assert model.point_head.weight.grad is not None
    assert model.interval_head.weight.grad is not None

def test_corrector_can_consume_base_decoder_features():
    model = DistilledFlowCorrector(DistilledFlowCorrectorConfig(
        hidden_channels=16, blocks=1, groups=4,
        base_feature_channels=32))
    history = torch.randn(2, 20, 16, 24, 3)
    prediction = torch.randn(2, 20, 16, 24, 3)
    features = torch.randn(2, 32, 16, 24)
    lower = torch.rand(2, 20, 16, 24, 2) + 0.1
    upper = torch.rand(2, 20, 16, 24, 2) + 0.1
    corrected, new_lower, new_upper = model.correct_distribution(
        history, prediction, lower, upper, features)
    assert corrected.shape == prediction.shape
    assert new_lower.shape == lower.shape
    assert new_upper.shape == upper.shape
    torch.testing.assert_close(corrected, prediction)
    torch.testing.assert_close(new_lower, lower)
    torch.testing.assert_close(new_upper, upper)
