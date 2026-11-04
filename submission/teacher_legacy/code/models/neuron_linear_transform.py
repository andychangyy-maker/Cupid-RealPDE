"""Mergeable neuron-linear transformations for convolutional transfer.

The source convolution remains available as an ordinary PyTorch parameter,
while a target-domain factor and offset transform each input/output channel
pair of its kernel::

    W_target = W_source * factor + offset

Both transformation tensors are constant across the spatial kernel.  Their
identity initialization makes installation output preserving.  PyTorch's
parametrization API also lets a validated transformation be merged back into
the source convolution without an inference-time branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn
from torch.nn.utils import parametrize


SUPPORTED_CONVOLUTIONS = (nn.Conv2d, nn.ConvTranspose2d)


class NeuronLinearKernelTransform(nn.Module):
    """Affine kernel transform shared over spatial kernel positions."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        if weight.ndim != 4:
            raise ValueError("NLT currently requires a two-dimensional kernel")
        shape = (*weight.shape[:2], 1, 1)
        self.factor = nn.Parameter(torch.ones(
            shape, device=weight.device, dtype=weight.dtype))
        self.offset = nn.Parameter(torch.zeros(
            shape, device=weight.device, dtype=weight.dtype))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source.ndim != 4 or source.shape[:2] != self.factor.shape[:2]:
            raise ValueError("NLT source-kernel shape changed")
        return source * self.factor + self.offset


@dataclass(frozen=True)
class InstalledNeuronLinearTransform:
    """One installed transform and its fully qualified module name."""

    name: str
    module: nn.Module
    transform: NeuronLinearKernelTransform

    @property
    def source_weight(self) -> nn.Parameter:
        return self.module.parametrizations.weight.original


def install_neuron_linear_transforms(
        model: nn.Module,
        *,
        select: Callable[[str, nn.Module], bool] | None = None,
        freeze_source_weight: bool = True,
        freeze_source_bias: bool = True,
        ) -> tuple[InstalledNeuronLinearTransform, ...]:
    """Install identity-initialized NLTs on selected 2-D convolutions.

    ``select`` receives the name and module before any parametrization is
    installed.  Gathering candidates first avoids mutating the module tree
    while it is being traversed.
    """
    candidates = tuple(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, SUPPORTED_CONVOLUTIONS)
        and (select is None or select(name, module)))
    installed = []
    for name, module in candidates:
        if parametrize.is_parametrized(module, "weight"):
            raise ValueError(f"convolution {name!r} is already parametrized")
        transform = NeuronLinearKernelTransform(module.weight)
        parametrize.register_parametrization(module, "weight", transform)
        module.parametrizations.weight.original.requires_grad_(
            not freeze_source_weight)
        if module.bias is not None and freeze_source_bias:
            module.bias.requires_grad_(False)
        installed.append(InstalledNeuronLinearTransform(
            name=name, module=module, transform=transform))
    if not installed:
        raise ValueError("NLT selection did not match any convolution")
    return tuple(installed)


def neuron_linear_parameters(
        installed: Iterable[InstalledNeuronLinearTransform],
        ) -> tuple[nn.Parameter, ...]:
    """Return only target-domain factor/offset parameters."""
    result = []
    for row in installed:
        result.extend((row.transform.factor, row.transform.offset))
    return tuple(result)


def merge_neuron_linear_transforms(
        installed: Iterable[InstalledNeuronLinearTransform],
        ) -> None:
    """Merge target kernels into ordinary convolutions in place."""
    rows = tuple(installed)
    for row in rows:
        if not parametrize.is_parametrized(row.module, "weight"):
            raise ValueError(f"convolution {row.name!r} has no installed NLT")
        parametrize.remove_parametrizations(
            row.module, "weight", leave_parametrized=True)


def neuron_linear_parameter_count(
        installed: Iterable[InstalledNeuronLinearTransform],
        ) -> int:
    return sum(
        parameter.numel()
        for parameter in neuron_linear_parameters(installed))
