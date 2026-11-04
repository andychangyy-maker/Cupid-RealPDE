"""Spatial uncertainty-budget reallocation for a frozen flow forecaster.

The module preserves the native interval budget at initialization. It learns
where to move that budget using a nonlinear local branch and a static spatial
template, while a separate sample-conditioned branch may make smaller global
budget corrections. All inputs are available inside one scored sample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UQReallocatorConfig:
    in_steps: int = 20
    out_steps: int = 20
    recent_steps: int = 5
    hidden_channels: int = 48
    global_hidden: int = 160
    groups: int = 8
    template_height: int = 32
    template_width: int = 64
    use_dynamic_local: bool = True
    use_static_template: bool = True
    use_global_budget: bool = True
    use_regime_film: bool = False
    auxiliary_channels: int = 0
    log_allocation_limit: float = 3.0
    log_budget_limit: float = 1.25

    def validate(self) -> None:
        positive = (
            self.in_steps, self.out_steps, self.recent_steps,
            self.hidden_channels, self.global_hidden, self.groups,
            self.template_height, self.template_width,
            self.log_allocation_limit, self.log_budget_limit,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("UQ reallocator dimensions and limits must be positive")
        if self.auxiliary_channels < 0:
            raise ValueError("auxiliary_channels must be non-negative")
        if self.recent_steps >= self.in_steps:
            raise ValueError("recent_steps must be smaller than in_steps")

    @property
    def feature_channels(self) -> int:
        # recent u/v + recent differences + history mean/std + full point
        # forecast + point mean/std + coordinates.
        return (
            4 * self.recent_steps + 2 * self.out_steps + 10
            + self.auxiliary_channels)

    @property
    def width_channels(self) -> int:
        return 4 * self.out_steps


def _group_norm(groups: int, channels: int) -> nn.GroupNorm:
    while channels % groups and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int, dilation: int):
        super().__init__()
        self.body = nn.Sequential(
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels, channels, 3, padding=dilation,
                dilation=dilation),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


def _flatten_time_channels(value: torch.Tensor) -> torch.Tensor:
    batch, steps, height, width, channels = value.shape
    return value.permute(0, 1, 4, 2, 3).reshape(
        batch, steps * channels, height, width)


def build_reallocation_features(
        history: torch.Tensor, prediction: torch.Tensor,
        recent_steps: int,
        auxiliary: torch.Tensor | None = None) -> torch.Tensor:
    """Build local dynamics and forecast features in normalized units."""
    if history.ndim != 5 or prediction.ndim != 5:
        raise ValueError("history and prediction must be rank-five tensors")
    if history.shape[0] != prediction.shape[0] or history.shape[2:4] != prediction.shape[2:4]:
        raise ValueError("history and prediction batch/spatial shapes differ")
    velocity = history[..., :2]
    point = prediction[..., :2]
    recent = velocity[:, -recent_steps:]
    differences = velocity[:, 1:] - velocity[:, :-1]
    differences = differences[:, -recent_steps:]
    history_stats = torch.cat((
        velocity.mean(dim=1),
        velocity.std(dim=1, unbiased=False),
    ), dim=-1).permute(0, 3, 1, 2)
    point_stats = torch.cat((
        point.mean(dim=1),
        point.std(dim=1, unbiased=False),
    ), dim=-1).permute(0, 3, 1, 2)
    batch, _, height, width, _ = history.shape
    yy = torch.linspace(-1.0, 1.0, height, device=history.device,
                        dtype=history.dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=history.device,
                        dtype=history.dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    coordinates = torch.stack((grid_y, grid_x)).unsqueeze(0).expand(
        batch, -1, -1, -1)
    parts = [
        _flatten_time_channels(recent),
        _flatten_time_channels(differences),
        history_stats,
        _flatten_time_channels(point),
        point_stats,
        coordinates,
    ]
    if auxiliary is not None:
        if auxiliary.ndim != 4:
            raise ValueError("auxiliary features must have shape [B,C,H,W]")
        if (auxiliary.shape[0] != batch
                or auxiliary.shape[2:] != (height, width)):
            raise ValueError("auxiliary feature batch/spatial shapes differ")
        parts.append(auxiliary)
    return torch.cat(parts, dim=1)


class UQBudgetReallocator(nn.Module):
    """Reallocate native lower/upper widths without moving point predictions."""

    def __init__(self, config: UQReallocatorConfig):
        super().__init__()
        config.validate()
        self.config = config
        if config.use_dynamic_local:
            self.local_trunk = nn.Sequential(
                nn.Conv2d(
                    config.feature_channels, config.hidden_channels,
                    3, padding=1),
                DilatedResidualBlock(
                    config.hidden_channels, config.groups, dilation=1),
                DilatedResidualBlock(
                    config.hidden_channels, config.groups, dilation=2),
                DilatedResidualBlock(
                    config.hidden_channels, config.groups, dilation=4),
            )
            self.local_head = nn.Conv2d(
                config.hidden_channels, config.width_channels, 1)
            nn.init.zeros_(self.local_head.weight)
            nn.init.zeros_(self.local_head.bias)
            if config.use_regime_film:
                self.regime_film = nn.Sequential(
                    nn.Linear(2 * config.feature_channels, config.global_hidden),
                    nn.SiLU(),
                    nn.Linear(config.global_hidden, 2 * config.hidden_channels),
                )
                nn.init.zeros_(self.regime_film[-1].weight)
                nn.init.zeros_(self.regime_film[-1].bias)
            else:
                self.regime_film = None
        else:
            self.local_trunk = None
            self.local_head = None
            self.regime_film = None

        if config.use_static_template:
            self.spatial_template = nn.Parameter(torch.zeros(
                1, config.width_channels,
                config.template_height, config.template_width))
        else:
            self.spatial_template = None

        if config.use_global_budget:
            pooled_channels = 2 * config.feature_channels
            self.global_budget = nn.Sequential(
                nn.Linear(pooled_channels, config.global_hidden),
                nn.SiLU(),
                nn.Linear(config.global_hidden, config.width_channels),
            )
            nn.init.zeros_(self.global_budget[-1].weight)
            nn.init.zeros_(self.global_budget[-1].bias)
        else:
            self.global_budget = None

    def architecture_config(self) -> dict:
        return {"architecture": "uq_budget_reallocator", **asdict(self.config)}

    def _local_logits(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        logits = features.new_zeros(
            batch, self.config.width_channels, height, width)
        if self.local_head is not None and self.local_trunk is not None:
            hidden = self.local_trunk(features)
            if self.regime_film is not None:
                pooled = torch.cat((
                    features.mean(dim=(2, 3)),
                    features.std(dim=(2, 3), unbiased=False),
                ), dim=1)
                scale, shift = self.regime_film(pooled).chunk(2, dim=1)
                hidden = hidden * (1.0 + scale[..., None, None])
                hidden = hidden + shift[..., None, None]
            logits = logits + self.local_head(hidden)
        if self.spatial_template is not None:
            template = self.spatial_template
            if template.shape[-2:] != (height, width):
                template = F.interpolate(
                    template, size=(height, width), mode="bilinear",
                    align_corners=False)
            logits = logits + template
        return logits.reshape(
            batch, self.config.out_steps, 4, height, width
        ).permute(0, 1, 3, 4, 2)

    def _budget_logits(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        if self.global_budget is None:
            return features.new_zeros(
                batch, self.config.out_steps, 1, 1, 4)
        pooled = torch.cat((
            features.mean(dim=(2, 3)),
            features.std(dim=(2, 3), unbiased=False),
        ), dim=1)
        return self.global_budget(pooled).reshape(
            batch, self.config.out_steps, 1, 1, 4)

    def forward(
            self, history: torch.Tensor, prediction: torch.Tensor,
            native_lower: torch.Tensor, native_upper: torch.Tensor,
            valid_mask: torch.Tensor | None = None,
            auxiliary: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        if native_lower.shape != native_upper.shape:
            raise ValueError("native lower/upper shapes differ")
        expected = (
            history.shape[0], self.config.out_steps,
            history.shape[2], history.shape[3], 2)
        if tuple(native_lower.shape) != expected:
            raise ValueError(
                f"native widths have {tuple(native_lower.shape)}, expected {expected}")
        features = build_reallocation_features(
            history, prediction, self.config.recent_steps, auxiliary)
        if features.shape[1] != self.config.feature_channels:
            raise ValueError(
                f"reallocation features have {features.shape[1]} channels, "
                f"expected {self.config.feature_channels}")
        local_logits = self._local_logits(features).clamp(
            -self.config.log_allocation_limit,
            self.config.log_allocation_limit)
        allocation = torch.exp(local_logits)
        native = torch.cat((native_lower, native_upper), dim=-1)

        if valid_mask is None:
            mask = torch.ones_like(native[..., :1])
        else:
            if valid_mask.ndim == 3:
                valid_mask = valid_mask[:, None, :, :, None]
            elif valid_mask.ndim == 4:
                valid_mask = valid_mask[:, None, :, :, :]
            if valid_mask.ndim != 5:
                raise ValueError("valid_mask must have shape [B,H,W] or broadcastable rank five")
            mask = valid_mask.to(device=native.device, dtype=native.dtype)
        weighted_native = native * mask
        denominator = weighted_native.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        correction_mean = (
            native * allocation * mask
        ).sum(dim=(2, 3), keepdim=True) / denominator
        allocation = allocation / correction_mean.clamp_min(1e-8)

        budget_logits = self._budget_logits(features).clamp(
            -self.config.log_budget_limit, self.config.log_budget_limit)
        widths = native * allocation * torch.exp(budget_logits)
        return widths[..., :2], widths[..., 2:]


def build_uq_budget_reallocator(**overrides) -> UQBudgetReallocator:
    return UQBudgetReallocator(UQReallocatorConfig(**overrides))
