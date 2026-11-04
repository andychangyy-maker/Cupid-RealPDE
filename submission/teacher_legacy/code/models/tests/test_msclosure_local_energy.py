import torch

from models.msclosure_local_energy import FrozenMSClosureLocalEnergy
from models.multiple_shooting_closure import (
    MultipleShootingClosure,
    MultipleShootingClosureConfig,
)


def _tiny_wrapper() -> FrozenMSClosureLocalEnergy:
    config = MultipleShootingClosureConfig(
        e0_channels=8, e1_channels=16, e2_channels=24,
        e0_blocks=1, e1_blocks=1, e2_blocks=1,
        context_channels=8, cell_channels=16, cell_blocks=1,
    )
    return FrozenMSClosureLocalEnergy(MultipleShootingClosure(config))


def test_zero_controller_is_exact_identity_and_only_head_gets_gradient():
    torch.manual_seed(4)
    wrapper = _tiny_wrapper()
    history = torch.randn(2, 20, 16, 16, 3)
    with torch.no_grad():
        baseline = wrapper.base(history)
    prediction = wrapper(history)
    assert torch.equal(prediction, baseline)
    prediction.square().mean().backward()
    assert any(parameter.grad is not None
               for parameter in wrapper.controller.parameters())
    assert all(parameter.grad is None for parameter in wrapper.base.parameters())


def test_nonzero_controller_preserves_physical_mean_pressure_and_boundaries():
    torch.manual_seed(5)
    wrapper = _tiny_wrapper()
    with torch.no_grad():
        wrapper.controller.body[-1].bias.fill_(0.4)
    history = torch.randn(2, 20, 16, 16, 3)
    values = wrapper.forward_with_aux(history)
    base_uv = values["base_physical_uv"]
    corrected_uv = values["corrected_physical_uv"]
    assert torch.allclose(
        corrected_uv.mean(dim=1), base_uv.mean(dim=1), atol=1e-6, rtol=0)
    assert torch.equal(
        values["prediction"][..., 2:],
        values["base_prediction"][..., 2:])
    delta = values["delta_logk"]
    assert torch.count_nonzero(delta[:, :, 1:-1, 1:-1]) > 0
    assert torch.count_nonzero(delta[:, :, 0]) == 0
    assert torch.count_nonzero(delta[:, :, -1]) == 0
    assert torch.count_nonzero(delta[:, :, :, 0]) == 0
    assert torch.count_nonzero(delta[:, :, :, -1]) == 0
