"""Fast multi-scale interval head distilled from the residual-quantile decoder."""

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
class DistilledIntervalHeadConfig:
    out_steps: int = 20
    feature_channels: tuple[int, ...] = (512, 256, 128)
    projection_channels: int = 16
    hidden_channels: int = 64
    blocks: int = 3
    groups: int = 8
    kernel_size: int = 5
    max_log_width_ratio: float = 2.0
    max_center_fraction: float = 1.5
    reference_total_widths: tuple[float, float] = (0.02150, 0.01100)

    def validate(self) -> None:
        positive = (
            self.out_steps, self.projection_channels,
            self.hidden_channels, self.blocks, self.groups,
            self.kernel_size, self.max_log_width_ratio,
            self.max_center_fraction,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("direct interval dimensions and limits must be positive")
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


class DistilledIntervalHead(nn.Module):
    """Predict direct normalized extents without running the quantile decoder."""

    def __init__(self, config: DistilledIntervalHeadConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, config.projection_channels, 1)
            for channels in config.feature_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(
                config.context_channels, config.hidden_channels, 1),
            _group_norm(config.groups, config.hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*(
            DepthwiseResidualBlock(
                config.hidden_channels, config.groups,
                config.kernel_size)
            for _ in range(config.blocks)
        ))
        self.output = nn.Conv2d(
            config.hidden_channels, 4 * config.out_steps, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def architecture_config(self) -> dict:
        return {
            "architecture": "distilled_interval_head",
            **asdict(self.config),
        }

    def forward(
            self, x: torch.Tensor, prediction: torch.Tensor,
            features: list[torch.Tensor], target_std: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError("decoder feature count does not match interval head")
        height, width = prediction.shape[2:4]
        projected = []
        for feature, projection, expected in zip(
                features, self.projections,
                self.config.feature_channels):
            if feature.shape[1] != expected:
                raise ValueError(
                    "decoder feature channel count does not match interval head")
            value = projection(feature.detach())
            if value.shape[-2:] != (height, width):
                value = F.interpolate(
                    value, size=(height, width),
                    mode="bilinear", align_corners=False)
            projected.append(value)
        context = DistilledFlowCorrector.flow_context(
            x, prediction)
        hidden = self.blocks(self.stem(torch.cat(
            (*projected, context), dim=1)))
        batch = prediction.shape[0]
        raw = self.output(hidden).reshape(
            batch, self.config.out_steps, 4, height, width
        ).permute(0, 1, 3, 4, 2)
        log_ratio = (
            self.config.max_log_width_ratio
            * torch.tanh(raw[..., :2]))
        reference = raw.new_tensor(
            self.config.reference_total_widths).view(
                1, 1, 1, 1, 2)
        normalized_reference = reference / target_std[:2].view(
            1, 1, 1, 1, 2).clamp_min(1e-8)
        total_width = normalized_reference * torch.exp(log_ratio)
        center = (
            self.config.max_center_fraction
            * torch.tanh(raw[..., 2:]) * total_width)
        lower_extent = 0.5 * total_width - center
        upper_extent = 0.5 * total_width + center
        return lower_extent, upper_extent
