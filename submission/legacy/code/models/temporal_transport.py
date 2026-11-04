"""Causal temporal-transport operator for RealPDE Track 1.

Unlike the legacy 2-D backbones, this model never folds the observed time
axis into channels.  A spatial encoder is shared by every observed frame, a
strictly causal mixer processes the resulting trajectory, and learned horizon
queries decode every requested future frame in one pass.  The primary output
path warps the latest observed field; separate heads predict a time-constant
mean correction and an exactly zero-time-mean fluctuation correction.

The module intentionally contains only a point forecaster.  It is a clean
control for testing whether explicit temporal transport raises the point/TKE
ceiling before attaching a structured predictive distribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, sqrt

import torch
from torch import nn
from torch.nn import functional as F

from .unet2d import ResidualConvBlock, _group_norm


MIXER_MODES = (
    "causal_temporal",
    "framewise_spatial_control",
)


@dataclass(frozen=True)
class TemporalTransportConfig:
    """Architecture contract for :class:`TemporalTransportOperator`."""

    in_steps: int = 20
    out_steps: int = 20
    in_channels: int = 3
    out_channels: int = 3
    hidden_channels: int = 64
    spatial_blocks: int = 2
    spatial_downsample: int = 2
    temporal_blocks: int = 3
    temporal_kernel: int = 5
    mixer_mode: str = "causal_temporal"
    groups: int = 8
    use_coordinates: bool = True
    max_displacement: float = 4.0
    residual_limit: float = 4.0
    output_initialization_std: float = 1.0e-3
    # Kept in the common backbone contract even though this first-stage model
    # intentionally has no interval head.  The shared trainer uses this flag
    # to choose between ``forward`` and ``forward_distribution``.
    predict_uncertainty: bool = False

    def validate(self) -> None:
        dimensions = (
            self.in_steps, self.out_steps, self.in_channels,
            self.out_channels, self.hidden_channels, self.spatial_blocks,
            self.spatial_downsample, self.temporal_blocks,
            self.temporal_kernel, self.groups,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("architecture dimensions must be positive")
        if self.out_channels > self.in_channels:
            raise ValueError(
                "persistence transport requires out_channels <= in_channels")
        if self.hidden_channels % self.groups:
            raise ValueError("hidden_channels must be divisible by groups")
        if self.temporal_kernel < 2:
            raise ValueError("temporal_kernel must be at least two")
        if self.mixer_mode not in MIXER_MODES:
            raise ValueError(
                f"mixer_mode must be one of {MIXER_MODES}, got "
                f"{self.mixer_mode!r}")
        if self.max_displacement <= 0.0:
            raise ValueError("max_displacement must be positive")
        if self.residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")
        if self.output_initialization_std < 0.0:
            raise ValueError("output_initialization_std must be non-negative")
        if self.predict_uncertainty:
            raise ValueError(
                "temporal transport is a point-only architecture")


@dataclass
class TemporalTransportOutput:
    """Named decomposition used by physics losses and diagnostic audits."""

    prediction: torch.Tensor
    transported: torch.Tensor
    mean_delta: torch.Tensor
    fluctuation: torch.Tensor
    displacement: torch.Tensor
    attention: torch.Tensor


def _coordinate_features(reference: torch.Tensor) -> torch.Tensor:
    """Return normalized y/x fields for an ``[N,C,H,W]`` tensor."""
    batch, _channels, height, width = reference.shape
    y = torch.linspace(
        -1.0, 1.0, height, device=reference.device,
        dtype=reference.dtype)
    x = torch.linspace(
        -1.0, 1.0, width, device=reference.device,
        dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx), dim=0).unsqueeze(0).expand(
        batch, -1, -1, -1)


class SharedFrameEncoder(nn.Module):
    """Apply exactly the same spatial encoder to every observed frame."""

    def __init__(self, config: TemporalTransportConfig):
        super().__init__()
        input_channels = config.in_channels + (
            2 if config.use_coordinates else 0)
        self.use_coordinates = bool(config.use_coordinates)
        self.in_channels = int(config.in_channels)
        self.hidden_channels = int(config.hidden_channels)
        self.stem = nn.Conv2d(
            input_channels, config.hidden_channels, 3,
            stride=config.spatial_downsample, padding=1)
        self.blocks = nn.Sequential(*(
            ResidualConvBlock(
                config.hidden_channels, config.hidden_channels,
                config.groups)
            for _ in range(config.spatial_blocks)
        ))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or history.shape[-1] != self.in_channels:
            raise ValueError(
                "frame encoder requires history [B,T,H,W,in_channels]")
        batch, steps, height, width, channels = history.shape
        frames = history.permute(0, 1, 4, 2, 3).reshape(
            batch * steps, channels, height, width)
        if self.use_coordinates:
            frames = torch.cat((frames, _coordinate_features(frames)), dim=1)
        encoded = self.blocks(self.stem(frames))
        latent_height, latent_width = encoded.shape[-2:]
        return encoded.reshape(
            batch, steps, self.hidden_channels,
            latent_height, latent_width)


class MatchedTrajectoryBlock(nn.Module):
    """One active mixer block with a parameter/MAC-matched control.

    ``causal_temporal`` applies the depthwise kernel along time with left-only
    padding.  ``framewise_spatial_control`` merges batch and time, rotates the
    x axis into the Conv3d depth axis, and applies the *same-shaped* depthwise
    kernel independently to every frame.  Thus the two modes have identical
    parameter tensor shapes and per-pixel multiply-accumulate counts; only the
    axis carrying the active depthwise interaction differs.

    Group normalization is evaluated on ``B*T`` independent frames in both
    modes.  The control therefore contains no hidden cross-frame path.
    """

    def __init__(
            self, channels: int, kernel: int, groups: int,
            mode: str = "causal_temporal"):
        super().__init__()
        if mode not in MIXER_MODES:
            raise ValueError(f"unsupported mixer mode {mode!r}")
        self.kernel = int(kernel)
        self.mode = str(mode)
        self.norm = _group_norm(groups, channels)
        # Keep this tensor shape identical across the candidate and control.
        # In control mode x is explicitly moved into the Conv3d depth axis.
        self.depthwise = nn.Conv3d(
            channels, channels, (kernel, 1, 1), groups=channels, bias=False)
        self.projection = nn.Conv3d(channels, 2 * channels, 1)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 5:
            raise ValueError(
                "temporal trajectory must be [B,T,C,H,W]")
        batch, steps, channels, height, width = trajectory.shape
        normalized = self.norm(trajectory.reshape(
            batch * steps, channels, height, width))
        if self.mode == "causal_temporal":
            normalized = normalized.reshape(
                batch, steps, channels, height, width
            ).permute(0, 2, 1, 3, 4)
            # torch pad order for [B,C,T,H,W] is W, H, T.
            normalized = F.pad(
                normalized, (0, 0, 0, 0, self.kernel - 1, 0))
            filtered = self.depthwise(normalized)
            value, gate = self.projection(filtered).chunk(2, dim=1)
            update = (value * F.silu(gate)).permute(0, 2, 1, 3, 4)
        else:
            # [B*T,C,H,W] -> [B*T,C,D=x,H=y,W=1].  Folding time into
            # the batch dimension makes cross-frame interaction impossible.
            normalized = normalized.permute(0, 1, 3, 2).unsqueeze(-1)
            left = (self.kernel - 1) // 2
            right = self.kernel - 1 - left
            normalized = F.pad(normalized, (0, 0, 0, 0, left, right))
            filtered = self.depthwise(normalized)
            value, gate = self.projection(filtered).chunk(2, dim=1)
            update = value * F.silu(gate)
            update = update.squeeze(-1).permute(0, 1, 3, 2).reshape(
                batch, steps, channels, height, width)
        return trajectory + update


class CausalTemporalMixer(nn.Module):
    """Stack matched candidate/control blocks over observed latent states."""

    def __init__(self, config: TemporalTransportConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            MatchedTrajectoryBlock(
                config.hidden_channels, config.temporal_kernel, config.groups,
                config.mixer_mode)
            for _ in range(config.temporal_blocks))

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            trajectory = block(trajectory)
        return trajectory


def _sinusoidal_horizon_queries(
        steps: int, channels: int) -> torch.Tensor:
    """Deterministic non-DC initialization for learned horizon queries."""
    horizon = torch.arange(1, steps + 1, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, channels, 2, dtype=torch.float32)
        * (-log(10000.0) / max(channels, 2)))
    result = torch.zeros(steps, channels, dtype=torch.float32)
    result[:, 0::2] = torch.sin(horizon * frequencies)
    if channels > 1:
        result[:, 1::2] = torch.cos(
            horizon * frequencies[:result[:, 1::2].shape[1]])
    return result


class HorizonQueryDecoder(nn.Module):
    """Decode all future horizons with local attention over observed states."""

    def __init__(self, config: TemporalTransportConfig):
        super().__init__()
        channels = config.hidden_channels
        self.out_steps = int(config.out_steps)
        self.scale = sqrt(float(channels))
        self.query = nn.Parameter(_sinusoidal_horizon_queries(
            config.out_steps, channels))
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.query_projection = nn.Linear(channels, channels, bias=False)
        self.decoder = ResidualConvBlock(
            channels, channels, config.groups)

    @staticmethod
    def _project_frames(
            projection: nn.Module, trajectory: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = trajectory.shape
        projected = projection(trajectory.reshape(
            batch * steps, channels, height, width))
        return projected.reshape(batch, steps, channels, height, width)

    def forward(
            self, trajectory: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        if trajectory.ndim != 5:
            raise ValueError("query decoder requires [B,T,C,H,W]")
        batch, _steps, channels, height, width = trajectory.shape
        keys = self._project_frames(self.key, trajectory)
        values = self._project_frames(self.value, trajectory)
        queries = self.query_projection(self.query).to(trajectory)
        scores = torch.einsum(
            "btdhw,od->bothw", keys, queries) / self.scale
        # Accumulate the softmax in fp32 so long histories remain stable under
        # fp16 deployment.  The result is returned in the latent dtype.
        attention = scores.float().softmax(dim=2).to(values)
        context = torch.einsum(
            "bothw,btdhw->bodhw", attention, values)
        horizon = queries.view(1, self.out_steps, channels, 1, 1)
        decoded = context + trajectory[:, -1:].expand(
            -1, self.out_steps, -1, -1, -1) + horizon
        decoded = self.decoder(decoded.reshape(
            batch * self.out_steps, channels, height, width))
        return decoded.reshape(
            batch, self.out_steps, channels, height, width), attention


class TemporalTransportOperator(nn.Module):
    """Single-pass causal temporal forecaster with learned spatial transport."""

    def __init__(self, config: TemporalTransportConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.frame_encoder = SharedFrameEncoder(config)
        self.temporal_mixer = CausalTemporalMixer(config)
        self.horizon_decoder = HorizonQueryDecoder(config)
        channels = config.hidden_channels
        self.mean_decoder = ResidualConvBlock(
            channels, channels, config.groups)
        self.mean_head = nn.Conv2d(channels, config.out_channels, 1)
        self.fluctuation_head = nn.Conv2d(
            channels, config.out_channels, 1)
        self.transport_head = nn.Conv2d(channels, 2, 1)
        self._initialize_output_heads()
        self.register_buffer(
            "persistence_scale", torch.ones(config.out_channels))
        self.register_buffer(
            "persistence_bias", torch.zeros(config.out_channels))

    def _initialize_output_heads(self) -> None:
        std = self.config.output_initialization_std
        for head in (
                self.mean_head, self.fluctuation_head,
                self.transport_head):
            if std:
                nn.init.normal_(head.weight, std=std)
            else:
                nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def architecture_config(self) -> dict:
        return {"architecture": "temporal_transport", **asdict(self.config)}

    @torch.no_grad()
    def set_normalizer_statistics(
            self, mean_input: torch.Tensor, mean_target: torch.Tensor,
            std_input: torch.Tensor, std_target: torch.Tensor) -> None:
        """Map the last normalized input into normalized target space."""
        channels = self.config.out_channels
        device = self.persistence_scale.device
        mean_input = torch.as_tensor(mean_input, device=device)
        mean_target = torch.as_tensor(mean_target, device=device)
        std_input = torch.as_tensor(std_input, device=device)
        std_target = torch.as_tensor(std_target, device=device)
        safe_target = std_target[:channels].clamp_min(1.0e-8)
        self.persistence_scale.copy_(
            (std_input[:channels] / safe_target).to(self.persistence_scale))
        self.persistence_bias.copy_((
            (mean_input[:channels] - mean_target[:channels]) / safe_target
        ).to(self.persistence_bias))

    def _validate_history(self, history: torch.Tensor) -> None:
        expected = (
            self.config.in_steps, self.config.in_channels)
        if (history.ndim != 5
                or history.shape[1] != expected[0]
                or history.shape[-1] != expected[1]):
            raise ValueError(
                "expected history [B,"
                f"{expected[0]},H,W,{expected[1]}], got "
                f"{tuple(history.shape)}")

    def _persistence(self, history: torch.Tensor) -> torch.Tensor:
        channels = self.config.out_channels
        return (
            history[:, -1, ..., :channels]
            * self.persistence_scale.view(1, 1, 1, channels)
            + self.persistence_bias.view(1, 1, 1, channels))

    def encode_frames(self, history: torch.Tensor) -> torch.Tensor:
        """Return per-frame spatial features before temporal interaction."""
        self._validate_history(history)
        return self.frame_encoder(history)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        """Return causal states for every observed frame."""
        return self.temporal_mixer(self.encode_frames(history))

    @staticmethod
    def warp(
            source: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        """Sample one source field with per-horizon pixel displacements.

        ``source`` is ``[B,H,W,C]`` and ``displacement`` is
        ``[B,T,H,W,2]`` in ``(x, y)`` pixel units.  Positive x displacement
        samples from the pixel to the right.  Border padding avoids injecting
        artificial zeros at the PIV field boundary.
        """
        if source.ndim != 4:
            raise ValueError("transport source must be [B,H,W,C]")
        if (displacement.ndim != 5 or displacement.shape[0] != source.shape[0]
                or displacement.shape[2:4] != source.shape[1:3]
                or displacement.shape[-1] != 2):
            raise ValueError(
                "displacement must be [B,T,H,W,2] aligned with source")
        batch, height, width, channels = source.shape
        steps = displacement.shape[1]
        dtype = displacement.dtype
        y = torch.linspace(
            -1.0, 1.0, height, device=source.device, dtype=dtype)
        x = torch.linspace(
            -1.0, 1.0, width, device=source.device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base_grid = torch.stack((xx, yy), dim=-1).view(
            1, 1, height, width, 2)
        pixel_scale = displacement.new_tensor((
            2.0 / max(width - 1, 1),
            2.0 / max(height - 1, 1),
        ))
        grid = base_grid + displacement * pixel_scale
        values = source.permute(0, 3, 1, 2).unsqueeze(1).expand(
            -1, steps, -1, -1, -1).reshape(
                batch * steps, channels, height, width)
        sampled = F.grid_sample(
            values.float(),
            grid.reshape(batch * steps, height, width, 2).float(),
            mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.reshape(
            batch, steps, channels, height, width
        ).permute(0, 1, 3, 4, 2).to(source)

    @staticmethod
    def _resize_horizon_field(
            value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        batch, steps, channels, height, width = value.shape
        if (height, width) != size:
            value = F.interpolate(
                value.reshape(batch * steps, channels, height, width),
                size=size, mode="bilinear", align_corners=False
            ).reshape(batch, steps, channels, *size)
        return value.permute(0, 1, 3, 4, 2)

    def forward_components(
            self, history: torch.Tensor) -> TemporalTransportOutput:
        """Forecast and expose the fixed-responsibility decomposition."""
        self._validate_history(history)
        target_size = tuple(history.shape[2:4])
        trajectory = self.encode_history(history)
        decoded, attention = self.horizon_decoder(trajectory)
        batch, steps, channels, latent_height, latent_width = decoded.shape

        flat = decoded.reshape(
            batch * steps, channels, latent_height, latent_width)
        raw_displacement = self.transport_head(flat).reshape(
            batch, steps, 2, latent_height, latent_width)
        displacement = self._resize_horizon_field(
            raw_displacement, target_size)
        displacement = self.config.max_displacement * torch.tanh(
            displacement)
        transported = self.warp(self._persistence(history), displacement)

        raw_fluctuation = self.fluctuation_head(flat).reshape(
            batch, steps, self.config.out_channels,
            latent_height, latent_width)
        fluctuation = self._resize_horizon_field(
            raw_fluctuation, target_size)
        fluctuation = self.config.residual_limit * torch.tanh(
            fluctuation / self.config.residual_limit)
        fluctuation = fluctuation - fluctuation.mean(dim=1, keepdim=True)

        mean_feature = self.mean_decoder(trajectory[:, -1])
        mean_delta = self.mean_head(mean_feature)
        if mean_delta.shape[-2:] != target_size:
            mean_delta = F.interpolate(
                mean_delta, size=target_size, mode="bilinear",
                align_corners=False)
        mean_delta = mean_delta.permute(0, 2, 3, 1)
        mean_delta = self.config.residual_limit * torch.tanh(
            mean_delta / self.config.residual_limit)

        prediction = transported + mean_delta.unsqueeze(1) + fluctuation
        return TemporalTransportOutput(
            prediction=prediction,
            transported=transported,
            mean_delta=mean_delta,
            fluctuation=fluctuation,
            displacement=displacement,
            attention=attention,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_components(history).prediction


def build_temporal_transport(**overrides) -> TemporalTransportOperator:
    return TemporalTransportOperator(TemporalTransportConfig(**overrides))
