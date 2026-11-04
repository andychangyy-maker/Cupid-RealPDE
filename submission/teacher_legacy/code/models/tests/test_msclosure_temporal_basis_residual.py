import torch

from models.msclosure_temporal_basis_residual import (
    FrozenMSClosureTemporalBasisResidual,
)
from models.multiple_shooting_closure import (
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
)


def _tiny_wrapper() -> FrozenMSClosureTemporalBasisResidual:
    config = MultipleShootingClosureConfig(
        e0_channels=8, e1_channels=16, e2_channels=24,
        e0_blocks=1, e1_blocks=1, e2_blocks=1,
        context_channels=8, cell_channels=16, cell_blocks=1,
    )
    return FrozenMSClosureTemporalBasisResidual(
        MultipleShootingClosure(config))


def test_dct_basis_is_zero_mean_orthogonal_and_unit_rms():
    controller = _tiny_wrapper().controller
    gram = controller.basis @ controller.basis.T / controller.out_steps
    assert torch.allclose(
        controller.basis.mean(dim=1), torch.zeros(controller.config.modes),
        atol=1e-7, rtol=0)
    assert torch.allclose(
        gram, torch.eye(controller.config.modes), atol=1e-6, rtol=0)


def test_zero_head_is_exact_identity_and_has_live_gradient():
    torch.manual_seed(6)
    wrapper = _tiny_wrapper()
    history = torch.randn(2, 20, 16, 16, 3)
    with torch.no_grad():
        baseline = wrapper.base(history)
    prediction = wrapper(history)
    assert torch.equal(prediction, baseline)
    prediction[..., :2].square().mean().backward()
    assert wrapper.controller.output.weight.grad is not None
    assert torch.count_nonzero(wrapper.controller.output.weight.grad) > 0
    assert all(parameter.grad is None for parameter in wrapper.base.parameters())


def test_nonzero_coefficients_change_waveform_but_not_mean_or_geometry():
    torch.manual_seed(7)
    wrapper = _tiny_wrapper()
    with torch.no_grad():
        wrapper.controller.output.bias.fill_(0.1)
    history = torch.randn(2, 20, 16, 16, 3)
    values = wrapper.forward_with_aux(history)
    delta = values["normalized_delta"]
    free = values["free_mask"].unsqueeze(1).expand_as(delta[..., :1])
    assert torch.allclose(
        delta.mean(dim=1), torch.zeros_like(delta[:, 0]),
        atol=1e-6, rtol=0)
    assert torch.count_nonzero(delta[..., 1:-1, 1:-1, :]) > 0
    assert torch.count_nonzero(torch.where(free, torch.zeros_like(delta), delta)) == 0
    assert torch.equal(
        values["prediction"][..., 2:],
        values["base_prediction"][..., 2:])
    base_mean = values["base_prediction"][..., :2].mean(dim=1)
    candidate_mean = values["prediction"][..., :2].mean(dim=1)
    assert torch.allclose(candidate_mean, base_mean, atol=1e-6, rtol=0)
