"""Direct interval head with an unbounded ordered absolute geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.distilled_flow_corrector import (
    DepthwiseResidualBlock,
    DistilledFlowCorrector,
)
from models.unet2d import _group_norm


@dataclass(frozen=True)
class DirectUnboundedAbsoluteIntervalHeadConfig:
    out_steps: int = 20
    feature_channels: tuple[int, ...] = (512, 256, 128)
    projection_channels: int = 16
    hidden_channels: int = 64
    blocks: int = 3
    groups: int = 8
    kernel_size: int = 5
    reference_total_widths: tuple[float, float] = (0.02150, 0.01100)

    def validate(self) -> None:
        positive = (
            self.out_steps,
            self.projection_channels,
            self.hidden_channels,
            self.blocks,
            self.groups,
            self.kernel_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("unbounded interval dimensions must be positive")
        if not self.feature_channels or any(
                value <= 0 for value in self.feature_channels):
            raise ValueError("feature channels must be non-empty and positive")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel size must be odd")
        if len(self.reference_total_widths) != 2 or any(
                value <= 0 for value in self.reference_total_widths):
            raise ValueError("reference widths must contain positive u/v values")

    @property
    def context_channels(self) -> int:
        return 70 + len(self.feature_channels) * self.projection_channels


class DirectUnboundedAbsoluteIntervalHead(nn.Module):
    """Map one frozen Direct forecast onto every finite ordered UV interval."""

    def __init__(self, config: DirectUnboundedAbsoluteIntervalHeadConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, config.projection_channels, 1)
            for channels in config.feature_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(config.context_channels, config.hidden_channels, 1),
            _group_norm(config.groups, config.hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*(
            DepthwiseResidualBlock(
                config.hidden_channels, config.groups, config.kernel_size)
            for _ in range(config.blocks)
        ))
        self.output = nn.Conv2d(
            config.hidden_channels, 4 * config.out_steps, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def architecture_config(self) -> dict:
        return {
            "architecture": "direct_unbounded_absolute_interval_head",
            **asdict(self.config),
        }

    def extents_from_raw(
            self, raw: torch.Tensor, target_std: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode unbounded center units and log widths into signed extents."""
        if raw.shape[-1] != 4:
            raise ValueError("unbounded interval raw output must have 4 channels")
        if target_std.numel() < 2:
            raise ValueError("target std must contain u/v values")
        reference = raw.new_tensor(
            self.config.reference_total_widths).view(1, 1, 1, 1, 2)
        scale = reference / target_std[:2].to(
            device=raw.device, dtype=raw.dtype).view(1, 1, 1, 1, 2)
        width = scale * torch.exp(raw[..., :2])
        center = scale * raw[..., 2:]
        return 0.5 * width - center, 0.5 * width + center

    def forward(
            self, x: torch.Tensor, prediction: torch.Tensor,
            features: list[torch.Tensor], target_std: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError("decoder feature count does not match interval head")
        height, width = prediction.shape[2:4]
        projected = []
        for feature, projection, expected in zip(
                features, self.projections, self.config.feature_channels):
            if feature.shape[1] != expected:
                raise ValueError(
                    "decoder feature channel count does not match interval head")
            value = projection(feature.detach())
            if value.shape[-2:] != (height, width):
                value = F.interpolate(
                    value, size=(height, width), mode="bilinear",
                    align_corners=False)
            projected.append(value)
        context = DistilledFlowCorrector.flow_context(x, prediction)
        hidden = self.blocks(self.stem(torch.cat(
            (*projected, context), dim=1)))
        batch = prediction.shape[0]
        raw = self.output(hidden).reshape(
            batch, self.config.out_steps, 4, height, width
        ).permute(0, 1, 3, 4, 2)
        return self.extents_from_raw(raw, target_std)
