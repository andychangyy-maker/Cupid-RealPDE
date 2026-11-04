"""Minimal future-aligned predictive-state (FAPS) forecaster.

The model is intentionally a *screening* architecture, not a production
submission model.  It makes one narrow, falsifiable claim: a compact state
read from the observed history may be more useful when it is aligned to a
state that can reconstruct the released future residual.

There are deliberately no shortcuts around this claim.  Both the history and
future encoders emit a normalized spatial latent map.  A single decoder is called twice
during training, once from the history state and once from the future state.
It may read only that latent, a fixed last-frame persistence field, coordinates
and a fixed horizon code.  In particular, it has no history feature skip,
trainable warp/base path, or future-feature skip.  The future encoder is not
part of :class:`FAPSForecaster`, the object returned for deployment.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .unet2d import ResidualConvBlock


@dataclass(frozen=True)
class FAPSConfig:
    in_steps: int = 20
    out_steps: int = 20
    channels: int = 3
    hidden_channels: int = 64
    latent_channels: int = 32
    groups: int = 8
    encoder_blocks: int = 2
    decoder_blocks: int = 2

    def validate(self) -> None:
        if min(self.in_steps, self.out_steps, self.channels,
               self.hidden_channels, self.latent_channels,
               self.groups, self.encoder_blocks, self.decoder_blocks) <= 0:
            raise ValueError("FAPS dimensions must be positive")
        if self.hidden_channels % self.groups:
            raise ValueError("hidden_channels must be divisible by groups")
        if self.latent_channels % self.groups:
            raise ValueError("latent_channels must be divisible by groups")


def _coordinates(reference: torch.Tensor) -> torch.Tensor:
    """Fixed normalized y/x coordinates for ``[B,*,H,W,*]`` references."""
    batch, height, width = reference.shape[0], reference.shape[-3], reference.shape[-2]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=reference.device,
                       dtype=reference.dtype),
        torch.linspace(-1.0, 1.0, width, device=reference.device,
                       dtype=reference.dtype), indexing="ij")
    return torch.stack((yy, xx), dim=0).unsqueeze(0).expand(
        batch, -1, -1, -1)


def _spatial_coordinates(batch: int, height: int, width: int, *,
                         device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij")
    return torch.stack((yy, xx), dim=0).unsqueeze(0).expand(
        batch, -1, -1, -1)


def normalize_latent(value: torch.Tensor, groups: int) -> torch.Tensor:
    """Affine-free group normalization shared by analysis and deployment.

    This is intentionally spatial: a state carries ``D x H/4 x W/4`` rather
    than collapsing a wake/phase field into one global vector.  There is no
    learned affine term which could silently restore a scale-specific bypass.
    """
    if value.ndim != 4:
        raise ValueError("latent must be [B,D,H,W]")
    if value.shape[1] % groups:
        raise ValueError("latent channel count must be divisible by groups")
    return F.group_norm(value, groups, weight=None, bias=None, eps=1.0e-6)


class SequenceLatentEncoder(nn.Module):
    """A small spatial encoder from a 20-frame field sequence to one state."""

    def __init__(self, config: FAPSConfig):
        super().__init__()
        self.steps = config.in_steps
        self.channels = config.channels
        self.latent_groups = config.groups
        input_channels = self.steps * self.channels + 2
        self.stem = nn.Conv2d(input_channels, config.hidden_channels, 3, stride=2, padding=1)
        self.blocks = nn.Sequential(*(
            ResidualConvBlock(config.hidden_channels, config.hidden_channels,
                              config.groups)
            for _ in range(config.encoder_blocks)))
        # One additional stride gives the stated small spatial state map
        # [B,latent,H/4,W/4].  No pooling/global bottleneck is present.
        self.projection = nn.Conv2d(config.hidden_channels, config.latent_channels,
                                    3, stride=2, padding=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5 or sequence.shape[1] != self.steps or sequence.shape[-1] != self.channels:
            raise ValueError(
                f"encoder expects [B,{self.steps},H,W,{self.channels}], got "
                f"{tuple(sequence.shape)}")
        batch, steps, height, width, channels = sequence.shape
        field = sequence.permute(0, 1, 4, 2, 3).reshape(
            batch, steps * channels, height, width)
        features = torch.cat((field, _coordinates(sequence)), dim=1)
        features = self.blocks(self.stem(features))
        return normalize_latent(self.projection(features), self.latent_groups)


class SharedResidualDecoder(nn.Module):
    """Decode a normalized state with only fixed permitted conditioning.

    The persistence field is supplied by a fixed normalizer mapping.  It is
    not predicted or warped, and the decoder produces a residual rather than
    an alternative trainable base forecast.
    """

    def __init__(self, config: FAPSConfig):
        super().__init__()
        self.steps = config.out_steps
        self.channels = config.channels
        self.latent_channels = config.latent_channels
        self.latent_groups = config.groups
        # Fixed horizon Fourier terms are buffers so horizon information cannot
        # become a latent history/future bypass.
        horizon = torch.arange(1, self.steps + 1, dtype=torch.float32)
        horizon = horizon / float(self.steps)
        self.register_buffer("horizon_features", torch.stack((
            torch.sin(torch.pi * horizon), torch.cos(torch.pi * horizon)), dim=1))
        in_channels = config.latent_channels + config.channels + 2 + 2
        self.stem = nn.Conv2d(in_channels, config.hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(*(
            ResidualConvBlock(config.hidden_channels, config.hidden_channels,
                              config.groups)
            for _ in range(config.decoder_blocks)))
        self.output = nn.Conv2d(config.hidden_channels, config.channels, 1)
        nn.init.normal_(self.output.weight, std=1.0e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent: torch.Tensor, persistence: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1] != self.latent_channels:
            raise ValueError("decoder latent shape mismatch")
        if persistence.ndim != 4 or persistence.shape[-1] != self.channels:
            raise ValueError("persistence must be [B,H,W,C]")
        batch, height, width, _channels = persistence.shape
        latent_height, latent_width = latent.shape[-2:]
        state = normalize_latent(latent, self.latent_groups)
        state = state.unsqueeze(1).expand(-1, self.steps, -1, -1, -1)
        # Keep the expensive decoder entirely on the spatial latent grid.
        # Only the final three-channel residual is resized to full output
        # resolution, so the two mandatory shared-decoder calls stay cheap.
        base_low = F.interpolate(persistence.permute(0, 3, 1, 2),
                                 size=(latent_height, latent_width),
                                 mode="bilinear", align_corners=False)
        base = base_low.unsqueeze(1).expand(
            -1, self.steps, -1, -1, -1)
        coordinates = _spatial_coordinates(
            batch, latent_height, latent_width, device=persistence.device,
            dtype=persistence.dtype).unsqueeze(1).expand(
            -1, self.steps, -1, -1, -1)
        horizon = self.horizon_features.to(persistence).view(
            1, self.steps, 2, 1, 1).expand(batch, -1, -1, latent_height, latent_width)
        fields = torch.cat((state, base, coordinates, horizon), dim=2).reshape(
            batch * self.steps, -1, latent_height, latent_width)
        residual_low = self.output(self.blocks(self.stem(fields)))
        residual = F.interpolate(residual_low, size=(height, width), mode="bilinear",
                                 align_corners=False)
        return residual.reshape(batch, self.steps, self.channels, height, width).permute(
            0, 1, 3, 4, 2)


class FAPSForecaster(nn.Module):
    """Deployment object: history encoder, fixed persistence and shared decoder."""

    def __init__(self, config: FAPSConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.history_encoder = SequenceLatentEncoder(config)
        self.decoder = SharedResidualDecoder(config)
        self.register_buffer("persistence_scale", torch.ones(config.channels))
        self.register_buffer("persistence_bias", torch.zeros(config.channels))

    @torch.no_grad()
    def set_normalizer_statistics(
            self, mean_input: torch.Tensor, mean_target: torch.Tensor,
            std_input: torch.Tensor, std_target: torch.Tensor) -> None:
        channels = self.config.channels
        device = self.persistence_scale.device
        mean_input = torch.as_tensor(mean_input, device=device)
        mean_target = torch.as_tensor(mean_target, device=device)
        std_input = torch.as_tensor(std_input, device=device)
        std_target = torch.as_tensor(std_target, device=device)
        safe_target = std_target[:channels].clamp_min(1.0e-8)
        self.persistence_scale.copy_(
            (std_input[:channels] / safe_target).to(self.persistence_scale))
        self.persistence_bias.copy_(((mean_input[:channels] - mean_target[:channels]) /
                                    safe_target).to(self.persistence_bias))

    def persistence(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or history.shape[1] != self.config.in_steps or history.shape[-1] != self.config.channels:
            raise ValueError("history shape mismatch")
        return (history[:, -1] * self.persistence_scale.view(1, 1, 1, -1)
                + self.persistence_bias.view(1, 1, 1, -1))

    def forecast_components(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.history_encoder(history)
        base = self.persistence(history)
        residual = self.decoder(state, base)
        return {"prediction": base.unsqueeze(1) + residual, "base": base,
                "z_history": state, "forecast_residual": residual}

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forecast_components(history)["prediction"]


class FutureAlignedPredictiveState(nn.Module):
    """Training-only future-analysis encoder around a deployable forecaster."""

    def __init__(self, config: FAPSConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.forecaster = FAPSForecaster(config)
        self.future_encoder = SequenceLatentEncoder(config)

    def architecture_config(self) -> dict:
        return {"architecture": "future_aligned_predictive_state", **asdict(self.config)}

    @torch.no_grad()
    def set_normalizer_statistics(self, *statistics: torch.Tensor) -> None:
        self.forecaster.set_normalizer_statistics(*statistics)

    def forecast_components(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forecaster.forecast_components(history)

    def analysis_components(self, target: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
        if target.ndim != 5 or target.shape[1] != self.config.out_steps:
            raise ValueError("target shape mismatch")
        residual = target - base.unsqueeze(1)
        future_state = self.future_encoder(residual)
        reconstruction = self.forecaster.decoder(future_state, base)
        return {"future_residual": residual, "z_future": future_state,
                "reconstructed_residual": reconstruction}

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forecaster(history)

    def deployment(self) -> FAPSForecaster:
        """Return a future-free copy with exactly the normal forecast path."""
        return copy.deepcopy(self.forecaster)
