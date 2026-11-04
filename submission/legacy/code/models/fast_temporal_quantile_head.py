"""Lightweight explicit-time residual-quantile head for the S2 preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.distilled_flow_corrector import DistilledFlowCorrector
from models.unet2d import _group_norm


SPS_SIGMA = 0.0563870259


def fused_mean_point_heads(model: nn.Module) -> nn.ModuleList:
    """Fuse linear member point heads into their arithmetic mean."""
    if model.member_feature_adapters is not None:
        raise ValueError("fused point heads require no member feature adapters")
    if not model.member_point_heads:
        raise ValueError("model has no member point heads")
    scales = len(model.member_point_heads[0])
    if any(len(heads) != scales for heads in model.member_point_heads):
        raise ValueError("member point heads have inconsistent scale counts")

    fused = nn.ModuleList()
    for scale in range(scales):
        sources = [heads[scale] for heads in model.member_point_heads]
        first = sources[0]
        if any((source.bias is None) != (first.bias is None)
               for source in sources):
            raise ValueError("member point heads have inconsistent biases")
        head = nn.Conv2d(
            first.in_channels, first.out_channels, first.kernel_size,
            stride=first.stride, padding=first.padding,
            dilation=first.dilation, groups=first.groups,
            bias=first.bias is not None,
            device=first.weight.device, dtype=first.weight.dtype)
        with torch.no_grad():
            head.weight.copy_(torch.stack([
                source.weight.float() for source in sources]).mean(dim=0))
            if head.bias is not None:
                head.bias.copy_(torch.stack([
                    source.bias.float() for source in sources]).mean(dim=0))
        fused.append(head)
    return fused.eval()


def fused_mean_prediction(
        model: nn.Module, x: torch.Tensor, features: list[torch.Tensor],
        fused_heads: nn.ModuleList) -> torch.Tensor:
    """Run the fused point path, including any shared mean correction."""
    prediction = model._member_prediction(x, features, fused_heads)
    if model.mean_profile_refiner is not None:
        correction = model.mean_profile_refiner(
            features[-1]).permute(0, 2, 3, 1)
        if model.config.out_channels > 2:
            correction = F.pad(
                correction, (0, model.config.out_channels - 2))
        prediction = prediction + correction[:, None]
    return prediction


@dataclass(frozen=True)
class FastTemporalQuantileHeadConfig:
    out_steps: int = 20
    feature_channels: tuple[int, ...] = (512, 256, 128)
    projection_channels: int = 8
    flow_channels: int = 8
    hidden_channels: int = 16
    blocks: int = 2
    groups: int = 8
    spatial_kernel: int = 3
    temporal_kernel: int = 3
    quantile_levels: tuple[float, ...] = (
        0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95)
    quantile_gap_bias: float = -4.0
    endpoint_log_limit: float = 0.75

    def validate(self) -> None:
        positive = (
            self.out_steps, self.projection_channels, self.flow_channels,
            self.hidden_channels, self.blocks, self.groups,
            self.spatial_kernel, self.temporal_kernel,
            self.endpoint_log_limit,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("fast temporal quantile dimensions must be positive")
        if not self.feature_channels or any(
                value <= 0 for value in self.feature_channels):
            raise ValueError("feature channels must be non-empty and positive")
        if self.spatial_kernel % 2 == 0 or self.temporal_kernel % 2 == 0:
            raise ValueError("spatial and temporal kernels must be odd")
        levels = tuple(float(value) for value in self.quantile_levels)
        if len(levels) < 5 or len(levels) % 2 == 0:
            raise ValueError("quantile levels must contain an odd count >= 5")
        if levels != tuple(sorted(set(levels))):
            raise ValueError("quantile levels must be strictly increasing")
        if any(not 0.0 < value < 1.0 for value in levels):
            raise ValueError("quantile levels must lie in (0,1)")
        if levels[len(levels) // 2] != 0.5:
            raise ValueError("the middle quantile must be exactly 0.5")


class FactorizedSpatioTemporalBlock(nn.Module):
    """Depthwise spatial then temporal filtering with one residual path."""

    def __init__(self, channels: int, spatial_kernel: int,
                 temporal_kernel: int, groups: int):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(
                channels, channels, (1, spatial_kernel, spatial_kernel),
                padding=(0, spatial_kernel // 2, spatial_kernel // 2),
                groups=channels, bias=False),
            nn.Conv3d(channels, channels, 1, bias=False),
            _group_norm(groups, channels),
            nn.SiLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(
                channels, channels, (temporal_kernel, 1, 1),
                padding=(temporal_kernel // 2, 0, 0),
                groups=channels, bias=False),
            nn.Conv3d(channels, channels, 1, bias=False),
            _group_norm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.temporal(self.spatial(value)))


class EndpointRefinement(nn.Module):
    """Zero-initialized 3x3 spatial correction of selected endpoints."""

    def __init__(self, log_limit: float):
        super().__init__()
        self.log_limit = log_limit
        self.spatial = nn.Conv3d(
            4, 4, (1, 3, 3), padding=(0, 1, 1), groups=4)
        self.output = nn.Conv3d(4, 4, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, lower: torch.Tensor,
                upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.cat((lower, upper), dim=-1).permute(0, 4, 1, 2, 3)
        correction = self.log_limit * torch.tanh(
            self.output(F.silu(self.spatial(value))))
        correction = correction.permute(0, 2, 3, 4, 1)
        lower_scale, upper_scale = correction.chunk(2, dim=-1)
        return lower * torch.exp(lower_scale), upper * torch.exp(upper_scale)


class FastTemporalQuantileHead(nn.Module):
    """Predict seven residual quantiles with an explicit forecast axis."""

    def __init__(self, config: FastTemporalQuantileHeadConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, config.projection_channels, 1)
            for channels in config.feature_channels)
        self.flow_projection = nn.Conv2d(
            30 + 2 * config.out_steps, config.flow_channels, 1)
        input_channels = (
            len(config.feature_channels) * config.projection_channels
            + config.flow_channels)
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, config.hidden_channels, 1),
            _group_norm(config.groups, config.hidden_channels),
            nn.SiLU(),
        )
        self.horizon_embedding = nn.Parameter(torch.zeros(
            1, config.hidden_channels, config.out_steps, 1, 1))
        self.blocks = nn.Sequential(*(
            FactorizedSpatioTemporalBlock(
                config.hidden_channels, config.spatial_kernel,
                config.temporal_kernel, config.groups)
            for _ in range(config.blocks)))
        quantiles = len(config.quantile_levels)
        self.output = nn.Conv3d(config.hidden_channels, 2 * quantiles, 1)
        nn.init.zeros_(self.output.weight)
        bias = torch.full((2, quantiles), float(config.quantile_gap_bias))
        bias[:, quantiles // 2] = 0.0
        with torch.no_grad():
            self.output.bias.copy_(bias.reshape(-1))
        middle = quantiles // 2
        self.interval_candidate_pairs = tuple(
            (lower, upper)
            for lower in range(middle + 1)
            for upper in range(max(middle, lower + 1), quantiles))
        self.endpoint_refinement = EndpointRefinement(
            config.endpoint_log_limit)

    def architecture_config(self) -> dict:
        return {
            "architecture": "fast_temporal_quantile_head_fused_mean_member_free",
            **asdict(self.config),
        }

    def _project_context(
            self, x: torch.Tensor, prediction: torch.Tensor,
            features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError("decoder feature count does not match FTQ head")
        batch, steps, height, width, _ = prediction.shape
        if steps != self.config.out_steps:
            raise ValueError("prediction horizon does not match FTQ head")
        projected = []
        for feature, projection, expected in zip(
                features, self.projections, self.config.feature_channels):
            if feature.shape[1] != expected:
                raise ValueError(
                    "decoder feature channel count does not match FTQ head")
            value = projection(feature.detach())
            if value.shape[-2:] != (height, width):
                value = F.interpolate(
                    value, size=(height, width), mode="bilinear",
                    align_corners=False)
            projected.append(value)
        shared = torch.cat(projected, dim=1).unsqueeze(2).expand(
            -1, -1, steps, -1, -1)
        flow = self.flow_projection(
            DistilledFlowCorrector.flow_context(
                x, prediction.detach())).unsqueeze(2).expand(
                    -1, -1, steps, -1, -1)
        return torch.cat((shared, flow), dim=1)

    def _monotone_quantiles(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, _, steps, height, width = hidden.shape
        count = len(self.config.quantile_levels)
        raw = self.output(hidden).reshape(
            batch, 2, count, steps, height, width
        ).permute(0, 3, 4, 5, 1, 2)
        middle = count // 2
        center = raw[..., middle:middle + 1]
        lower_gaps = F.softplus(raw[..., :middle])
        running = lower_gaps[..., -1:]
        lower_distances = [running]
        for index in range(middle - 2, -1, -1):
            running = lower_gaps[..., index:index + 1] + running
            lower_distances.append(running)
        lower = center - torch.cat(lower_distances[::-1], dim=-1)
        upper_gaps = F.softplus(raw[..., middle + 1:])
        running = upper_gaps[..., :1]
        upper_distances = [running]
        for index in range(1, middle):
            running = running + upper_gaps[..., index:index + 1]
            upper_distances.append(running)
        upper = center + torch.cat(upper_distances, dim=-1)
        return torch.cat((lower, center, upper), dim=-1)

    def _utility_interval(
            self, quantiles: torch.Tensor,
            target_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        levels = quantiles.new_tensor(self.config.quantile_levels)
        lowers = []
        uppers = []
        masses = []
        for lower_index, upper_index in self.interval_candidate_pairs:
            lowers.append(-quantiles[..., lower_index])
            uppers.append(quantiles[..., upper_index])
            masses.append(levels[upper_index] - levels[lower_index])
        lower_stack = torch.stack(lowers, dim=-1)
        upper_stack = torch.stack(uppers, dim=-1)
        target_std = target_std[:2].to(
            device=quantiles.device, dtype=quantiles.dtype)
        physical_width = (
            (lower_stack + upper_stack)
            * target_std.view(1, 1, 1, 1, 2, 1))
        probability = torch.stack(masses).view(1, 1, 1, 1, 1, -1)
        utility = probability * torch.exp(-physical_width / SPS_SIGMA)
        selected = utility.argmax(dim=-1, keepdim=True)
        lower = lower_stack.gather(-1, selected).squeeze(-1)
        upper = upper_stack.gather(-1, selected).squeeze(-1)
        return lower, upper

    @staticmethod
    def _ordered_extents(
            lower: torch.Tensor,
            upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        invalid = lower + upper < 0.0
        return (
            torch.where(invalid, -upper, lower),
            torch.where(invalid, -lower, upper),
        )

    def forward(
            self, x: torch.Tensor, prediction: torch.Tensor,
            features: list[torch.Tensor], target_std: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self._project_context(x, prediction, features)
        hidden = self.blocks(
            self.stem(context) + self.horizon_embedding.to(context.dtype))
        quantiles = self._monotone_quantiles(hidden)
        lower, upper = self._utility_interval(quantiles, target_std)
        lower, upper = self.endpoint_refinement(lower, upper)
        lower, upper = self._ordered_extents(lower, upper)
        return quantiles, lower, upper
