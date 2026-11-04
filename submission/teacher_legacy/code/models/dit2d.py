"""A single-pass, regime-conditioned DiT backbone for RealPDE Track 1.

This uses the useful DiT ingredients (patch tokens and AdaLN-Zero conditioned
transformer blocks) as a deterministic forecaster. It intentionally does not
run an iterative diffusion sampler, which would be incompatible with the
competition's inference-time score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .unet2d import (
    ForecastDistributionMixin,
    build_history_features,
    history_regime_summary,
)


@dataclass(frozen=True)
class DiT2dConfig:
    in_steps: int = 20
    out_steps: int = 20
    in_channels: int = 3
    out_channels: int = 3
    image_height: int = 32
    image_width: int = 64
    patch_size: int = 2
    dimension: int = 384
    depth: int = 8
    heads: int = 6
    mlp_ratio: float = 4.0
    difference_steps: int = 5
    use_coordinates: bool = True
    mean_fluctuation: bool = True
    predict_uncertainty: bool = True
    uncertainty_base_widths: tuple[float, float] = (0.01075, 0.00550)
    learn_temporal_spread: bool = True

    def validate(self) -> None:
        positive = (
            self.in_steps, self.out_steps, self.in_channels,
            self.out_channels, self.image_height, self.image_width,
            self.patch_size, self.dimension, self.depth, self.heads,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("architecture dimensions must be positive")
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("image dimensions must be divisible by patch size")
        if self.dimension % self.heads or self.dimension % 4:
            raise ValueError("dimension must be divisible by heads and four")
        if not 0 <= self.difference_steps < self.in_steps:
            raise ValueError("difference_steps must be in [0, in_steps)")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp ratio must be positive")
        if len(self.uncertainty_base_widths) != 2:
            raise ValueError("uncertainty base widths must contain u/v")

    @property
    def feature_channels(self) -> int:
        result = self.in_steps * self.in_channels + 2 * self.difference_steps
        return result + (2 if self.use_coordinates else 0)


def sincos_position_embedding(
        dimension: int, height: int, width: int) -> torch.Tensor:
    quarter = dimension // 4
    omega = torch.arange(quarter, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter, 1)))
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32), indexing="ij")
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat((y.sin(), y.cos(), x.sin(), x.cos()), dim=-1)


def modulation(x: torch.Tensor, shift: torch.Tensor,
               scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZeroBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dimension, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            dimension, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dimension, elementwise_affine=False)
        hidden = int(dimension * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(),
            nn.Linear(hidden, dimension))
        self.condition = nn.Sequential(
            nn.SiLU(), nn.Linear(dimension, 6 * dimension))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, tokens: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.condition(
            condition).chunk(6, dim=-1)
        normalized = modulation(
            self.norm_attention(tokens), shift_a, scale_a)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False)
        tokens = tokens + gate_a.unsqueeze(1) * attended
        tokens = tokens + gate_m.unsqueeze(1) * self.mlp(modulation(
            self.norm_mlp(tokens), shift_m, scale_m))
        return tokens


class DiT2d(nn.Module, ForecastDistributionMixin):
    def __init__(self, config: DiT2dConfig):
        super().__init__()
        config.validate()
        self.config = config
        patch = config.patch_size
        self.patch_embedding = nn.Conv2d(
            config.feature_channels, config.dimension,
            kernel_size=patch, stride=patch)
        token_height = config.image_height // patch
        token_width = config.image_width // patch
        self.register_buffer(
            "position_embedding",
            sincos_position_embedding(
                config.dimension, token_height, token_width).unsqueeze(0),
            persistent=False)
        self.regime_embedding = nn.Sequential(
            nn.Linear(config.in_steps * 4, config.dimension),
            nn.SiLU(), nn.Linear(config.dimension, config.dimension))
        self.blocks = nn.ModuleList(
            AdaLNZeroBlock(config.dimension, config.heads, config.mlp_ratio)
            for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.dimension)
        point_channels = config.out_channels * (
            config.out_steps + 1 if config.mean_fluctuation
            else config.out_steps)
        self.point_head = nn.Linear(
            config.dimension, point_channels * patch * patch)
        nn.init.zeros_(self.point_head.weight)
        nn.init.zeros_(self.point_head.bias)
        if config.predict_uncertainty:
            self.uncertainty_head = nn.Linear(
                config.dimension, config.out_steps * 4 * patch * patch)
            nn.init.zeros_(self.uncertainty_head.weight)
            nn.init.zeros_(self.uncertainty_head.bias)
            self.spread_power = (
                nn.Parameter(torch.zeros(4))
                if config.learn_temporal_spread else None)
        else:
            self.uncertainty_head = None
            self.spread_power = None
        self._register_forecast_buffers()

    def architecture_config(self) -> dict:
        return {"architecture": "dit", **asdict(self.config)}

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2:4] != (
                self.config.image_height, self.config.image_width):
            raise ValueError(
                "DiT input resolution does not match config: "
                f"{tuple(x.shape[2:4])}")
        features = build_history_features(
            x, self.config.difference_steps, self.config.use_coordinates)
        tokens = self.patch_embedding(features).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding.to(tokens)
        condition = self.regime_embedding(history_regime_summary(x))
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.final_norm(tokens)

    def _unpatchify(self, values: torch.Tensor,
                    output_channels: int) -> torch.Tensor:
        batch = values.shape[0]
        patch = self.config.patch_size
        height = self.config.image_height // patch
        width = self.config.image_width // patch
        values = values.reshape(
            batch, height, width, output_channels, patch, patch)
        return values.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, output_channels, height * patch, width * patch)

    def _point_prediction(
            self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch, _, height, width, _ = x.shape
        output_channels = self.config.out_channels * (
            self.config.out_steps + 1 if self.config.mean_fluctuation
            else self.config.out_steps)
        raw = self._unpatchify(self.point_head(tokens), output_channels)
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
        if self.uncertainty_head is None:
            raise RuntimeError("uncertainty decoder is disabled")
        tokens = self._tokens(x)
        prediction = self._point_prediction(x, tokens)
        batch = x.shape[0]
        log_width = self._unpatchify(
            self.uncertainty_head(tokens), self.config.out_steps * 4)
        log_width = log_width.reshape(
            batch, self.config.out_steps, 4,
            self.config.image_height, self.config.image_width
        ).permute(0, 1, 3, 4, 2)
        lower, upper = self._positive_widths(log_width, self.spread_power)
        return prediction, lower, upper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(x)
        return self._point_prediction(x, tokens)


def build_dit2d(**overrides) -> DiT2d:
    return DiT2d(DiT2dConfig(**overrides))
