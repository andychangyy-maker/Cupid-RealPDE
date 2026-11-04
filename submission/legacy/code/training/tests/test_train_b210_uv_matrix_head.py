"""CPU checks of the physical head and its isolated gradient path, not score evidence."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.uv_fluctuation_head import UVFluctuationHead, frozen_point_and_feature


def test_physical_uv_units_learnable_identity_mean_and_frozen_base():
    class FrozenBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.energy_controller = torch.nn.Conv2d(3, 1, 1)
            self.register_buffer("target_std", torch.tensor([2., .5, 3.]))
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            correction = self.energy_controller(inputs[:, 0].permute(0, 3, 1, 2))
            return inputs + correction.permute(0, 2, 3, 1).unsqueeze(1)

    torch.manual_seed(4)
    base = FrozenBase().double().requires_grad_(False).eval()
    before = {key: value.clone() for key, value in base.state_dict().items()}
    inputs = torch.randn(2, 5, 2, 3, 3, dtype=torch.float64, requires_grad=True)
    point, feature = frozen_point_and_feature(base, inputs)
    assert base.calls == 1 and not point.requires_grad and not feature.requires_grad
    heads = {name: UVFluctuationHead(name == "matrix", input_channels=3).double() for name in ("scalar", "matrix")}
    for head in heads.values():
        assert torch.equal(head(point, feature, base.target_std), point)
    matrix = heads["matrix"]
    with torch.no_grad():
        matrix.projection.bias[1] = torch.atanh(torch.tensor(.4, dtype=torch.float64))
    output = matrix(point, feature, base.target_std)
    expected_u = point[..., 0] + .1 * (point[..., 1] - point[..., 1].mean(dim=1, keepdim=True)) * .5 / 2.
    assert torch.allclose(output[..., 0], expected_u, atol=1e-12, rtol=0)
    assert torch.equal(output[..., 1:], point[..., 1:])
    assert torch.allclose(output[..., :2].mean(dim=1), point[..., :2].mean(dim=1), atol=1e-12, rtol=0)
    with torch.no_grad():
        matrix.projection.bias.zero_()
    target = point[..., :2] + .2 * (point[..., :2] - point[..., :2].mean(dim=1, keepdim=True))
    for head in heads.values():
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
        loss = (head(point, feature, base.target_std)[..., :2] - target).square().mean()
        loss.backward()
        assert all(parameter.grad is not None for parameter in head.parameters())
        assert sum(float(parameter.grad.square().sum()) for parameter in head.parameters()) > 0
        optimizer.step()
        changed = head(point, feature, base.target_std)
        assert not torch.equal(changed[..., :2], point[..., :2])
        assert torch.equal(changed[..., 2:], point[..., 2:])
        assert torch.allclose(changed[..., :2].mean(dim=1), point[..., :2].mean(dim=1), atol=1e-12, rtol=0)
    assert inputs.grad is None and all(parameter.grad is None for parameter in base.parameters())
    assert all(torch.equal(value, base.state_dict()[key]) for key, value in before.items())
    assert [sum(p.numel() for p in UVFluctuationHead(matrix).parameters()) for matrix in (False, True)] == [211, 844]
