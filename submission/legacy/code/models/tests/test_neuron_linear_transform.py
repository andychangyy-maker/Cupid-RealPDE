from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn.utils import parametrize


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.neuron_linear_transform import (
    install_neuron_linear_transforms,
    merge_neuron_linear_transforms,
    neuron_linear_parameter_count,
    neuron_linear_parameters,
)


class TinyTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(4, 6, 3, padding=1, groups=2)
        self.up = nn.ConvTranspose2d(6, 4, 2, stride=2, groups=2)
        self.readout = nn.Conv2d(4, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.readout(self.up(self.encoder(value)))


def test_installation_is_exact_identity_and_selective():
    torch.manual_seed(7)
    model = TinyTransferModel().float().eval()
    value = torch.randn(2, 4, 8, 10)
    before = model(value).detach().clone()
    readout_before = model.readout.weight.detach().clone()

    installed = install_neuron_linear_transforms(
        model, select=lambda name, _module: name != "readout")
    after = model(value).detach()

    assert [row.name for row in installed] == ["encoder", "up"]
    assert torch.equal(before, after)
    assert torch.equal(model.readout.weight, readout_before)
    assert not parametrize.is_parametrized(model.readout, "weight")
    assert all(not row.source_weight.requires_grad for row in installed)
    assert all(
        parameter.requires_grad
        for parameter in neuron_linear_parameters(installed))
    expected = 2 * (6 * 2 + 6 * 2)
    assert neuron_linear_parameter_count(installed) == expected


def test_only_domain_transform_receives_kernel_gradient():
    torch.manual_seed(11)
    model = TinyTransferModel().float().train()
    installed = install_neuron_linear_transforms(
        model, select=lambda name, _module: name == "encoder")
    model(torch.randn(2, 4, 8, 8)).square().mean().backward()

    row = installed[0]
    assert row.source_weight.grad is None
    assert row.transform.factor.grad is not None
    assert row.transform.offset.grad is not None
    assert torch.isfinite(row.transform.factor.grad).all()
    assert torch.isfinite(row.transform.offset.grad).all()
    assert bool(torch.count_nonzero(row.transform.factor.grad))
    assert bool(torch.count_nonzero(row.transform.offset.grad))


def test_merge_preserves_adapted_conv_and_removes_runtime_branch():
    torch.manual_seed(13)
    model = TinyTransferModel().float().eval()
    installed = install_neuron_linear_transforms(model)
    with torch.no_grad():
        for index, row in enumerate(installed, start=1):
            row.transform.factor.add_(0.01 * index)
            row.transform.offset.add_(0.001 * index)
    value = torch.randn(1, 4, 5, 7)
    adapted = model(value).detach().clone()

    merge_neuron_linear_transforms(installed)
    merged = model(value).detach()

    assert torch.equal(adapted, merged)
    assert all(
        not parametrize.is_parametrized(module, "weight")
        for module in (model.encoder, model.up, model.readout))
    assert all(
        "parametrizations.weight" not in name
        for name in model.state_dict())
