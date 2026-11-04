"""Lightweight correction head distilled from a multi-backbone flow ensemble."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.unet2d import _group_norm


def _flatten_forecast(value: torch.Tensor) -> torch.Tensor:
    batch, steps, height, width, channels = value.shape
    return value.permute(0, 1, 4, 2, 3).reshape(
        batch, steps * channels, height, width)


@dataclass(frozen=True)
class DistilledFlowCorrectorConfig:
    in_steps: int = 20
    out_steps: int = 20
    hidden_channels: int = 64
    blocks: int = 3
    groups: int = 8
    kernel_size: int = 5
    base_feature_channels: int = 0
    max_point_delta: float = 0.75
    max_log_width_ratio: float = 0.50
    max_center_fraction: float = 0.50
    correct_fluctuation: bool = True
    correct_intervals: bool = True

    def validate(self) -> None:
        positive = (
            self.in_steps, self.out_steps, self.hidden_channels,
            self.blocks, self.groups, self.kernel_size,
            self.max_point_delta, self.max_log_width_ratio,
            self.max_center_fraction,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("corrector dimensions and limits must be positive")
        if self.in_steps < 6:
            raise ValueError("at least six history steps are required")
        if self.kernel_size % 2 == 0:
            raise ValueError("corrector kernel size must be odd")
        if self.base_feature_channels < 0:
            raise ValueError(
                "base feature channels must be non-negative")

    @property
    def context_channels(self) -> int:
        # recent history, recent differences, history moments, prediction,
        # prediction moments, and two coordinates.
        return (10 + 10 + 4 + 2 * self.out_steps + 4 + 2
                + self.base_feature_channels)


class DepthwiseResidualBlock(nn.Module):
    """Low-cost spatial block whose zero overhead is small versus a U-Net."""

    def __init__(self, channels: int, groups: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels, channels, kernel_size, padding=padding,
                groups=channels),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


class DistilledFlowCorrector(nn.Module):
    """Predict point and interval deltas from one frozen base forecast."""

    def __init__(self, config: DistilledFlowCorrectorConfig):
        super().__init__()
        config.validate()
        self.config = config
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
        point_channels = 2 + (
            2 * config.out_steps if config.correct_fluctuation else 0)
        self.point_head = nn.Conv2d(
            config.hidden_channels, point_channels, 1)
        self.interval_head = (
            nn.Conv2d(
                config.hidden_channels, 4 * config.out_steps, 1)
            if config.correct_intervals else None)
        nn.init.zeros_(self.point_head.weight)
        nn.init.zeros_(self.point_head.bias)
        if self.interval_head is not None:
            nn.init.zeros_(self.interval_head.weight)
            nn.init.zeros_(self.interval_head.bias)

    def architecture_config(self) -> dict:
        return {
            "architecture": "distilled_flow_corrector",
            **asdict(self.config),
        }

    @staticmethod
    def flow_context(
            x: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        history = x[..., :2].detach()
        prediction = prediction[..., :2].detach()
        recent = history[:, -5:]
        differences = history[:, -6:][:, 1:] - history[:, -6:][:, :-1]
        history_stats = torch.cat((
            history.mean(dim=1),
            history.std(dim=1, unbiased=False)), dim=-1
        ).permute(0, 3, 1, 2)
        prediction_stats = torch.cat((
            prediction.mean(dim=1),
            prediction.std(dim=1, unbiased=False)), dim=-1
        ).permute(0, 3, 1, 2)
        batch, _, height, width, _ = history.shape
        yy = torch.linspace(
            -1.0, 1.0, height, device=x.device, dtype=x.dtype)
        xx = torch.linspace(
            -1.0, 1.0, width, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coordinates = torch.stack((grid_y, grid_x)).unsqueeze(0).expand(
            batch, -1, -1, -1)
        return torch.cat((
            _flatten_forecast(recent),
            _flatten_forecast(differences),
            history_stats,
            _flatten_forecast(prediction),
            prediction_stats,
            coordinates,
        ), dim=1)

    def _features(
            self, x: torch.Tensor, prediction: torch.Tensor,
            base_features: torch.Tensor | None = None) -> torch.Tensor:
        context = self.flow_context(x, prediction)
        if self.config.base_feature_channels:
            if base_features is None:
                raise ValueError(
                    "configured corrector requires base decoder features")
            if base_features.shape[1] != self.config.base_feature_channels:
                raise ValueError(
                    "base decoder feature channel count does not match config")
            if base_features.shape[-2:] != context.shape[-2:]:
                base_features = F.interpolate(
                    base_features, size=context.shape[-2:],
                    mode="bilinear", align_corners=False)
            context = torch.cat((
                context, base_features.detach()), dim=1)
        return self.blocks(self.stem(context))

    def point_delta_from_features(
            self, features: torch.Tensor,
            reference: torch.Tensor) -> torch.Tensor:
        raw = self.point_head(features)
        mean = raw[:, :2].permute(0, 2, 3, 1).unsqueeze(1)
        if self.config.correct_fluctuation:
            batch, _, height, width = raw.shape
            fluctuation = raw[:, 2:].reshape(
                batch, self.config.out_steps, 2, height, width
            ).permute(0, 1, 3, 4, 2)
            fluctuation = fluctuation - fluctuation.mean(dim=1, keepdim=True)
            delta_uv = mean + fluctuation
        else:
            delta_uv = mean.expand(
                -1, self.config.out_steps, -1, -1, -1)
        delta_uv = self.config.max_point_delta * torch.tanh(delta_uv)
        if reference.shape[-1] == 2:
            return delta_uv
        return F.pad(delta_uv, (0, reference.shape[-1] - 2))

    def correct_point(
            self, x: torch.Tensor, prediction: torch.Tensor,
            base_features: torch.Tensor | None = None) -> torch.Tensor:
        features = self._features(
            x, prediction, base_features)
        return prediction + self.point_delta_from_features(
            features, prediction)

    def correct_distribution(
            self, x: torch.Tensor, prediction: torch.Tensor,
            lower_extent: torch.Tensor, upper_extent: torch.Tensor,
            base_features: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features(
            x, prediction, base_features)
        corrected_prediction = prediction + self.point_delta_from_features(
            features, prediction)
        if self.interval_head is None:
            return corrected_prediction, lower_extent, upper_extent

        batch, _, height, width = features.shape
        raw = self.interval_head(features).reshape(
            batch, self.config.out_steps, 4, height, width
        ).permute(0, 1, 3, 4, 2)
        log_ratio = (
            self.config.max_log_width_ratio * torch.tanh(raw[..., :2]))
        center_fraction = (
            self.config.max_center_fraction * torch.tanh(raw[..., 2:]))
        base_width = (lower_extent + upper_extent).clamp_min(1e-6)
        base_center = 0.5 * (upper_extent - lower_extent)
        width = base_width * torch.exp(log_ratio)
        center = base_center + center_fraction * base_width
        corrected_lower = 0.5 * width - center
        corrected_upper = 0.5 * width + center
        return corrected_prediction, corrected_lower, corrected_upper
