"""Multi-scale, phase-conditioned U-Net for RealPDE Track 1.

The observed trajectory is folded into 2D channels, augmented with recent
velocity differences and coordinates, and decoded in one pass. The point
decoder separates temporal mean and zero-mean fluctuations; a native
multi-scale head predicts asymmetric u/v interval widths for every future
time, pixel, and channel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UNet2dConfig:
    in_steps: int = 20
    out_steps: int = 20
    in_channels: int = 3
    out_channels: int = 3
    base: int = 64
    depth: int = 4
    groups: int = 8
    difference_steps: int = 5
    acceleration_steps: int = 0
    use_coordinates: bool = True
    use_regime_film: bool = True
    multi_scale_regime_film: bool = False
    cross_conditioned_backbone: bool = False
    cross_conditioned_steps: int = 5
    cross_conditioned_channels: int = 64
    mean_fluctuation: bool = True
    predict_uncertainty: bool = True
    uncertainty_base_widths: tuple[float, float] = (0.01075, 0.00550)
    learn_temporal_spread: bool = True
    backbone_residual_dropout: float = 0.0

    def validate(self) -> None:
        positive = (
            self.in_steps, self.out_steps, self.in_channels,
            self.out_channels, self.base, self.depth, self.groups,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("architecture dimensions must be positive")
        if self.depth < 2:
            raise ValueError("U-Net depth must be at least two")
        if self.multi_scale_regime_film and not self.use_regime_film:
            raise ValueError(
                "multi-scale regime FiLM requires use_regime_film")
        if not 2 <= self.cross_conditioned_steps < self.in_steps:
            raise ValueError(
                "cross-conditioned steps must be in [2, in_steps)")
        if self.cross_conditioned_channels <= 0:
            raise ValueError("cross-conditioned channels must be positive")
        if not 0 <= self.difference_steps < self.in_steps:
            raise ValueError("difference_steps must be in [0, in_steps)")
        if (self.acceleration_steps < 0
                or self.acceleration_steps > max(0, self.in_steps - 2)):
            raise ValueError(
                "acceleration_steps must be in [0, in_steps - 2]")
        if self.out_channels < 2:
            raise ValueError("at least u/v output channels are required")
        if len(self.uncertainty_base_widths) != 2:
            raise ValueError("uncertainty base widths must contain u/v")
        if any(value <= 0 for value in self.uncertainty_base_widths):
            raise ValueError("uncertainty base widths must be positive")
        if not 0.0 <= self.backbone_residual_dropout < 1.0:
            raise ValueError("backbone residual dropout must be in [0, 1)")

    @property
    def feature_channels(self) -> int:
        result = self.in_steps * self.in_channels
        result += 2 * self.difference_steps
        result += 2 * self.acceleration_steps
        return result + (2 if self.use_coordinates else 0)


def _group_norm(groups: int, channels: int) -> nn.GroupNorm:
    while channels % groups and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def build_history_features(
        x: torch.Tensor, difference_steps: int,
        use_coordinates: bool, acceleration_steps: int = 0,
        ) -> torch.Tensor:
    """Convert [B,T,H,W,C] history to an augmented 2D feature map."""
    if x.ndim != 5:
        raise ValueError(f"expected rank-five history, got {tuple(x.shape)}")
    batch, steps, height, width, channels = x.shape
    flattened = x.permute(0, 1, 4, 2, 3).reshape(
        batch, steps * channels, height, width)
    features = [flattened]
    if difference_steps:
        velocity = x[:, -(difference_steps + 1):, ..., :2]
        difference = velocity[:, 1:] - velocity[:, :-1]
        features.append(difference.permute(0, 1, 4, 2, 3).reshape(
            batch, 2 * difference_steps, height, width))
    if acceleration_steps:
        velocity = x[:, -(acceleration_steps + 2):, ..., :2]
        difference = velocity[:, 1:] - velocity[:, :-1]
        acceleration = difference[:, 1:] - difference[:, :-1]
        features.append(acceleration.permute(0, 1, 4, 2, 3).reshape(
            batch, 2 * acceleration_steps, height, width))
    if use_coordinates:
        y = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        z = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, z, indexing="ij")
        features.append(torch.stack((yy, xx)).unsqueeze(0).expand(
            batch, -1, -1, -1))
    return torch.cat(features, dim=1)


def history_regime_summary(x: torch.Tensor) -> torch.Tensor:
    """Sample-local history descriptor without metadata or batch state."""
    velocity = x[..., :2]
    mean = velocity.mean(dim=(2, 3))
    variance = velocity.var(dim=(2, 3), unbiased=False).clamp_min(1e-8)
    return torch.cat((mean, variance.sqrt()), dim=-1).flatten(1)


class ForecastDistributionMixin:
    """Normalizer-aware persistence and positive physical interval widths."""

    def _register_forecast_buffers(self) -> None:
        channels = self.config.out_channels
        self.register_buffer("persistence_scale", torch.ones(channels))
        self.register_buffer("persistence_bias", torch.zeros(channels))
        self.register_buffer("target_std", torch.ones(channels))

    @torch.no_grad()
    def set_normalizer_statistics(
            self, mean_input: torch.Tensor, mean_target: torch.Tensor,
            std_input: torch.Tensor, std_target: torch.Tensor) -> None:
        channels = self.config.out_channels
        device = self.persistence_scale.device
        mean_input = torch.as_tensor(mean_input, device=device)
        mean_target = torch.as_tensor(mean_target, device=device)
        std_input = torch.as_tensor(std_input, device=device)
        std_target = torch.as_tensor(std_target, device=device)
        safe_target = std_target[:channels].clamp_min(1e-8)
        self.persistence_scale.copy_(
            (std_input[:channels] / safe_target).to(self.persistence_scale))
        self.persistence_bias.copy_(
            ((mean_input[:channels] - mean_target[:channels]) / safe_target
             ).to(self.persistence_bias))
        self.target_std.copy_(safe_target.to(self.target_std))

    def _persistence(self, x: torch.Tensor) -> torch.Tensor:
        channels = self.config.out_channels
        return (
            x[:, -1, ..., :channels]
            * self.persistence_scale.view(1, 1, 1, channels)
            + self.persistence_bias.view(1, 1, 1, channels))

    def _positive_widths(
            self, log_width: torch.Tensor,
            spread_power: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        if spread_power is not None:
            horizon = torch.arange(
                1, self.config.out_steps + 1,
                device=log_width.device, dtype=log_width.dtype)
            horizon = torch.log(horizon / self.config.out_steps).view(
                1, self.config.out_steps, 1, 1, 1)
            log_width = log_width + horizon * spread_power.view(1, 1, 1, 1, 4)
        log_width = log_width.clamp(-4.0, 4.0)
        physical_base = log_width.new_tensor(
            self.config.uncertainty_base_widths).view(1, 1, 1, 1, 2)
        normalized_base = physical_base / self.target_std[:2].clamp_min(
            1e-8).view(1, 1, 1, 1, 2)
        lower = normalized_base * torch.exp(log_width[..., :2])
        upper = normalized_base * torch.exp(log_width[..., 2:])
        return lower, upper


class ResidualConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, groups: int,
                 residual_dropout: float = 0.0):
        super().__init__()
        if not 0.0 <= residual_dropout < 1.0:
            raise ValueError("residual dropout must be in [0, 1)")
        self.residual_dropout = float(residual_dropout)
        self.body = nn.Sequential(
            _group_norm(groups, input_channels), nn.SiLU(),
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            _group_norm(groups, output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        if self.training and self.residual_dropout > 0.0:
            residual = F.dropout2d(
                residual, p=self.residual_dropout, training=True)
        return self.skip(x) + residual


class LocalDynamicsCrossConditioner(nn.Module):
    """Spatially condition every decoder scale from observed dynamics.

    This is deliberately a separate stream from the flattened-history stem.
    It retains recent velocity, acceleration, and local regime statistics at
    full resolution, then produces a spatial FiLM field for each decoder
    scale.  Zero-initialized projections make the complete branch an exact
    identity at update zero while leaving the projection gradient live.
    """

    def __init__(
            self, recent_steps: int, hidden_channels: int,
            decoder_channels: list[int], groups: int):
        super().__init__()
        self.recent_steps = int(recent_steps)
        # recent du/dt + recent d2u/dt2 + local recent mean/std, all u/v
        input_channels = 2 * recent_steps + 2 * (recent_steps - 1) + 4
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(hidden_channels, 2 * width, 1)
            for width in decoder_channels)
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        if (history.ndim != 5 or history.shape[1] <= self.recent_steps
                or history.shape[-1] < 2):
            raise ValueError(
                "cross-conditioned input must be [B,T,H,W,C>=2]")
        velocity = history[:, -(self.recent_steps + 1):, ..., :2]
        differences = velocity[:, 1:] - velocity[:, :-1]
        accelerations = differences[:, 1:] - differences[:, :-1]
        batch, steps, height, width, channels = differences.shape
        first = differences.permute(0, 1, 4, 2, 3).reshape(
            batch, steps * channels, height, width)
        second = accelerations.permute(0, 1, 4, 2, 3).reshape(
            batch, (steps - 1) * channels, height, width)
        mean = velocity.mean(dim=1).permute(0, 3, 1, 2)
        spread = velocity.std(dim=1, unbiased=False).permute(0, 3, 1, 2)
        return self.body(torch.cat((first, second, mean, spread), dim=1))

    def modulate(
            self, value: torch.Tensor, condition: torch.Tensor,
            level: int) -> torch.Tensor:
        if not 0 <= level < len(self.projections):
            raise ValueError("cross-conditioned decoder level changed")
        if condition.shape[-2:] != value.shape[-2:]:
            condition = F.interpolate(
                condition, size=value.shape[-2:], mode="bilinear",
                align_corners=False)
        scale, shift = self.projections[level](condition).chunk(2, dim=1)
        # Bounded modulation prevents a small branch from replacing the
        # backbone during later Sim-to-Real adaptation.
        return (
            value * (1.0 + 0.25 * torch.tanh(scale))
            + 0.25 * torch.tanh(shift))


class UNet2d(nn.Module, ForecastDistributionMixin):
    """Direct multi-horizon U-Net with mean/fluctuation and native UQ heads."""

    def __init__(self, config: UNet2dConfig):
        super().__init__()
        config.validate()
        self.config = config
        channels = [config.base * 2 ** level for level in range(config.depth)]
        self.stem = nn.Conv2d(config.feature_channels, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList()
        for level in range(config.depth - 1):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(channels[level], channels[level + 1], 3,
                          stride=2, padding=1),
                ResidualConvBlock(
                    channels[level + 1], channels[level + 1], config.groups,
                    config.backbone_residual_dropout),
            ))
        self.bottleneck = ResidualConvBlock(
            channels[-1], channels[-1], config.groups,
            config.backbone_residual_dropout)
        if config.use_regime_film:
            self.regime_film = nn.Sequential(
                nn.Linear(config.in_steps * 4, channels[-1]), nn.SiLU(),
                nn.Linear(channels[-1], 2 * channels[-1]),
            )
            nn.init.zeros_(self.regime_film[-1].weight)
            nn.init.zeros_(self.regime_film[-1].bias)
        else:
            self.regime_film = None

        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        decoder_channels = []
        for level in range(config.depth - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(
                channels[level], channels[level - 1], 2, stride=2))
            self.decoder.append(ResidualConvBlock(
                2 * channels[level - 1], channels[level - 1], config.groups,
                config.backbone_residual_dropout))
            decoder_channels.append(channels[level - 1])
        self.decoder_regime_film = (
            nn.ModuleList(
                nn.Linear(config.in_steps * 4, 2 * width)
                for width in decoder_channels)
            if config.multi_scale_regime_film else None)
        if self.decoder_regime_film is not None:
            for film in self.decoder_regime_film:
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)
        self.cross_conditioner = (
            LocalDynamicsCrossConditioner(
                config.cross_conditioned_steps,
                config.cross_conditioned_channels,
                decoder_channels, config.groups)
            if config.cross_conditioned_backbone else None)

        point_channels = config.out_channels * (
            config.out_steps + 1 if config.mean_fluctuation else config.out_steps)
        self.point_heads = nn.ModuleList(
            nn.Conv2d(width, point_channels, 1) for width in decoder_channels)
        for head in self.point_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if config.predict_uncertainty:
            self.uncertainty_heads = nn.ModuleList(
                nn.Conv2d(width, config.out_steps * 4, 1)
                for width in decoder_channels)
            for head in self.uncertainty_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            self.spread_power = (
                nn.Parameter(torch.zeros(4))
                if config.learn_temporal_spread else None)
        else:
            self.uncertainty_heads = None
            self.spread_power = None
        self._register_forecast_buffers()

    def architecture_config(self) -> dict:
        return {"architecture": "unet", **asdict(self.config)}

    @staticmethod
    def _multi_scale(features: list[torch.Tensor], heads: nn.ModuleList,
                     size: tuple[int, int]) -> torch.Tensor:
        values = []
        for feature, head in zip(features, heads):
            value = head(feature)
            if value.shape[-2:] != size:
                value = F.interpolate(
                    value, size=size, mode="bilinear", align_corners=False)
            values.append(value)
        return torch.stack(values).mean(dim=0)

    def _decode(self, x: torch.Tensor) -> list[torch.Tensor]:
        encoded = self.stem(build_history_features(
            x, self.config.difference_steps, self.config.use_coordinates,
            self.config.acceleration_steps))
        skips = [encoded]
        for block in self.encoder:
            encoded = block(encoded)
            skips.append(encoded)
        encoded = self.bottleneck(encoded)
        regime = (
            history_regime_summary(x)
            if (self.regime_film is not None
                or self.decoder_regime_film is not None)
            else None)
        if self.regime_film is not None:
            scale, shift = self.regime_film(regime).chunk(2, dim=-1)
            encoded = encoded * (1.0 + scale[..., None, None])
            encoded = encoded + shift[..., None, None]
        local_condition = (
            self.cross_conditioner.encode(x)
            if self.cross_conditioner is not None else None)
        features = []
        for index, (up, block) in enumerate(zip(self.up, self.decoder)):
            encoded = up(encoded)
            encoded = block(torch.cat((encoded, skips[-(index + 2)]), dim=1))
            if self.decoder_regime_film is not None:
                scale, shift = self.decoder_regime_film[index](
                    regime).chunk(2, dim=-1)
                encoded = encoded * (1.0 + scale[..., None, None])
                encoded = encoded + shift[..., None, None]
            if self.cross_conditioner is not None:
                encoded = self.cross_conditioner.modulate(
                    encoded, local_condition, index)
            features.append(encoded)
        return features

    def _point_prediction(self, x: torch.Tensor,
                          features: list[torch.Tensor]) -> torch.Tensor:
        batch, _, height, width, _ = x.shape
        raw = self._multi_scale(features, self.point_heads, (height, width))
        if self.config.mean_fluctuation:
            channels = self.config.out_channels
            mean_delta = raw[:, :channels].permute(0, 2, 3, 1).unsqueeze(1)
            fluctuation = raw[:, channels:].reshape(
                batch, self.config.out_steps, channels,
                height, width).permute(0, 1, 3, 4, 2)
            fluctuation = fluctuation - fluctuation.mean(dim=1, keepdim=True)
            residual = mean_delta + fluctuation
        else:
            residual = raw.reshape(
                batch, self.config.out_steps, self.config.out_channels,
                height, width).permute(0, 1, 3, 4, 2)
        return self._persistence(x).unsqueeze(1) + residual

    def forward_distribution(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.uncertainty_heads is None:
            raise RuntimeError("uncertainty decoder is disabled")
        features = self._decode(x)
        prediction = self._point_prediction(x, features)
        batch, _, height, width, _ = x.shape
        log_width = self._multi_scale(
            features, self.uncertainty_heads, (height, width)).reshape(
                batch, self.config.out_steps, 4, height, width
            ).permute(0, 1, 3, 4, 2)
        lower, upper = self._positive_widths(log_width, self.spread_power)
        return prediction, lower, upper

    def forward_with_features(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the point forecast and decoder features in one pass."""
        features = self._decode(x)
        return self._point_prediction(x, features), features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, _features = self.forward_with_features(x)
        return prediction


def build_unet2d(**overrides) -> UNet2d:
    return UNet2d(UNet2dConfig(**overrides))
