"""Small future-block denoising refiner for a frozen direct-20 forecaster.

The competition forecaster predicts a whole twenty-frame future block in one
pass.  This module takes such a block ``z`` and predicts a bounded additive
correction.  It deliberately does *not* read the observed history separately:
the frozen base prediction is its compact, task-specific history summary.
The discovery experiment can therefore isolate whether a future-block
denoising objective changes a direct forecaster's error geometry, rather than
quietly adding another history model.

``noise_level=0`` is an exact conditioning no-op.  The discovery deployment
path calls the module once with the frozen base prediction at level one, which
denotes one fit-residual RMS rather than an extra iteration.  During the
candidate's training, noisy true future blocks are additionally supplied with
their known normalized noise level, in the style of a one-step denoiser.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DirectFutureRefinerConfig:
    """Architecture contract for :class:`DirectFutureRefiner`."""

    steps: int = 20
    channels: int = 3
    hidden_channels: int = 32
    blocks: int = 3
    groups: int = 8
    residual_limit: float = 4.0
    output_initialization_std: float = 1.0e-3

    def validate(self) -> None:
        if any(value <= 0 for value in (
                self.steps, self.channels, self.hidden_channels,
                self.blocks, self.groups)):
            raise ValueError("refiner dimensions must be positive")
        if self.hidden_channels % self.groups:
            raise ValueError("hidden_channels must be divisible by groups")
        if self.residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")
        if self.output_initialization_std < 0.0:
            raise ValueError("output_initialization_std must be non-negative")


class _FiLMResidual3d(nn.Module):
    """Depthwise space-time filtering plus pointwise channel mixing."""

    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(
            channels, channels, 3, padding=1, groups=channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 1)
        # bias=False means a zero step leaves the non-conditioned path exact.
        self.step = nn.Linear(1, 2 * channels, bias=False)

    def forward(self, value: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        scale, shift = self.step(noise_level[:, None]).chunk(2, dim=1)
        conditioned = self.norm1(value)
        conditioned = conditioned * (1.0 + scale[:, :, None, None, None])
        conditioned = conditioned + shift[:, :, None, None, None]
        update = self.conv1(F.silu(conditioned))
        update = self.conv2(F.silu(self.norm2(update)))
        return value + update


class DirectFutureRefiner(nn.Module):
    """Refine a predicted future block through a small 3-D convolutional G."""

    def __init__(self, config: DirectFutureRefinerConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.stem = nn.Conv3d(config.channels, config.hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            _FiLMResidual3d(config.hidden_channels, config.groups)
            for _ in range(config.blocks))
        self.output = nn.Conv3d(config.hidden_channels, config.channels, 1)
        if config.output_initialization_std:
            nn.init.normal_(self.output.weight, std=config.output_initialization_std)
        else:
            nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def architecture_config(self) -> dict:
        return {"architecture": "direct_future_refiner", **asdict(self.config)}

    def _validate(self, future: torch.Tensor, noise_level: torch.Tensor | None) -> torch.Tensor:
        if future.ndim != 5 or future.shape[1] != self.config.steps or future.shape[-1] != self.config.channels:
            raise ValueError(
                "expected future [B,"
                f"{self.config.steps},H,W,{self.config.channels}], got {tuple(future.shape)}")
        batch = future.shape[0]
        if noise_level is None:
            return future.new_zeros(batch)
        noise = torch.as_tensor(noise_level, device=future.device, dtype=future.dtype)
        if noise.ndim == 0:
            noise = noise.expand(batch)
        if noise.shape != (batch,):
            raise ValueError("noise_level must be scalar or shape [B]")
        if not torch.isfinite(noise).all() or torch.any(noise < 0.0):
            raise ValueError("noise_level must be finite and non-negative")
        return noise

    def correction(self, future: torch.Tensor, noise_level: torch.Tensor | None = None) -> torch.Tensor:
        """Return the bounded additive correction in normalized target units."""
        noise = self._validate(future, noise_level)
        value = future.permute(0, 4, 1, 2, 3)
        value = self.stem(value)
        for block in self.blocks:
            value = block(value, noise)
        correction = self.output(F.silu(value)).permute(0, 2, 3, 4, 1)
        return self.config.residual_limit * torch.tanh(
            correction / self.config.residual_limit)

    def forward(self, future: torch.Tensor, noise_level: torch.Tensor | None = None) -> torch.Tensor:
        return future + self.correction(future, noise_level)
