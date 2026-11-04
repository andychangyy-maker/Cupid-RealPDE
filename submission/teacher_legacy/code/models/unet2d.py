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


HISTORY_AXIS_ADAPTER_MODES = (
    "none",
    "causal_temporal",
    "framewise_spatial_control",
)
HISTORY_DYNAMICS_STATE_MODES = HISTORY_AXIS_ADAPTER_MODES
DEFORMABLE_HISTORY_MEMORY_MODES = (
    "none",
    "aligned",
    "spatially_permuted",
)
DEFORMABLE_HISTORY_MEMORY_ANCHORS = (0, 6, 13, 19)


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
    laplacian_point_pyramid: bool = False
    predict_uncertainty: bool = True
    uncertainty_base_widths: tuple[float, float] = (0.01075, 0.00550)
    learn_temporal_spread: bool = True
    backbone_residual_dropout: float = 0.0
    latent_spectral_rank: int = 0
    latent_spectral_modes_y: int = 2
    latent_spectral_modes_x: int = 4
    latent_spectral_residual_limit: float = 0.50
    history_axis_adapter_mode: str = "none"
    history_axis_adapter_channels: int = 32
    history_axis_adapter_blocks: int = 2
    history_axis_adapter_kernel: int = 5
    history_axis_adapter_residual_limit: float = 0.25
    history_dynamics_state_mode: str = "none"
    history_dynamics_state_channels: int = 32
    history_dynamics_state_blocks: int = 2
    history_dynamics_state_kernel: int = 5
    deformable_history_memory_mode: str = "none"
    deformable_history_memory_channels: int = 32
    deformable_history_memory_anchors: tuple[int, ...] = (
        DEFORMABLE_HISTORY_MEMORY_ANCHORS)
    deformable_history_memory_max_offset: float = 1.5

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
        if self.latent_spectral_rank < 0:
            raise ValueError("latent_spectral_rank must be non-negative")
        if self.latent_spectral_modes_y <= 0 or self.latent_spectral_modes_x <= 0:
            raise ValueError("latent spectral modes must be positive")
        if not 0.0 < self.latent_spectral_residual_limit < float("inf"):
            raise ValueError("latent spectral residual limit must be positive")
        if self.history_axis_adapter_mode not in HISTORY_AXIS_ADAPTER_MODES:
            raise ValueError(
                "history axis adapter mode must be one of "
                f"{HISTORY_AXIS_ADAPTER_MODES}")
        if self.history_axis_adapter_channels <= 0:
            raise ValueError("history axis adapter channels must be positive")
        if self.history_axis_adapter_blocks <= 0:
            raise ValueError("history axis adapter blocks must be positive")
        if self.history_axis_adapter_kernel < 2:
            raise ValueError("history axis adapter kernel must be at least two")
        if not 0.0 < self.history_axis_adapter_residual_limit < float("inf"):
            raise ValueError(
                "history axis adapter residual limit must be positive")
        if self.history_dynamics_state_mode not in HISTORY_DYNAMICS_STATE_MODES:
            raise ValueError(
                "history dynamics state mode must be one of "
                f"{HISTORY_DYNAMICS_STATE_MODES}")
        if self.history_dynamics_state_channels <= 0:
            raise ValueError("history dynamics state channels must be positive")
        if self.history_dynamics_state_blocks <= 0:
            raise ValueError("history dynamics state blocks must be positive")
        if self.history_dynamics_state_kernel < 2:
            raise ValueError(
                "history dynamics state kernel must be at least two")
        if self.deformable_history_memory_mode not in (
                DEFORMABLE_HISTORY_MEMORY_MODES):
            raise ValueError(
                "deformable history memory mode must be one of "
                f"{DEFORMABLE_HISTORY_MEMORY_MODES}")
        if self.deformable_history_memory_channels <= 0:
            raise ValueError(
                "deformable history memory channels must be positive")
        if (tuple(self.deformable_history_memory_anchors)
                != DEFORMABLE_HISTORY_MEMORY_ANCHORS):
            raise ValueError(
                "deformable history memory anchors are frozen to "
                f"{DEFORMABLE_HISTORY_MEMORY_ANCHORS}")
        if (self.deformable_history_memory_mode != "none"
                and self.in_steps <= max(DEFORMABLE_HISTORY_MEMORY_ANCHORS)):
            raise ValueError(
                "deformable history memory requires all frozen anchors")
        if (self.deformable_history_memory_mode != "none"
                and self.depth != 4):
            raise ValueError(
                "deformable history memory requires the depth-four B210 "
                "pyramid")
        if (self.deformable_history_memory_mode != "none"
                and getattr(self, "dual_decoder_mode", "none") != "none"):
            raise ValueError(
                "deformable history memory is screened only with the "
                "single-decoder B210 path")
        if not 0.0 < self.deformable_history_memory_max_offset < float("inf"):
            raise ValueError(
                "deformable history memory max offset must be positive")

    @property
    def feature_channels(self) -> int:
        result = self.in_steps * self.in_channels
        result += 2 * self.difference_steps
        result += 2 * self.acceleration_steps
        return result + (2 if self.use_coordinates else 0)

    @property
    def dynamics_state_channels(self) -> int:
        """Channels in the optional spatial history-dynamics state."""
        if self.history_dynamics_state_mode == "none":
            return 0
        return 3 * self.history_dynamics_state_channels

    @property
    def regime_summary_channels(self) -> int:
        """History-regime width, including optional dynamics-state moments."""
        return self.in_steps * 4 + 2 * self.dynamics_state_channels


def _group_norm(groups: int, channels: int) -> nn.GroupNorm:
    while channels % groups and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class HistoryAxisMixerBlock(nn.Module):
    """Matched feature mixer acting across time or within each frame.

    Both modes use identical parameter tensors and per-output-element MACs.
    The temporal arm pads only the past side of the observed trajectory.  The
    control folds time into the batch and applies the same-shaped kernel along
    spatial x, so it has no route for cross-frame communication.
    """

    def __init__(
            self, channels: int, kernel: int, groups: int, mode: str):
        super().__init__()
        if mode not in HISTORY_AXIS_ADAPTER_MODES[1:]:
            raise ValueError(f"unsupported history axis mixer mode {mode!r}")
        self.kernel = int(kernel)
        self.mode = str(mode)
        self.norm = _group_norm(groups, channels)
        self.depthwise = nn.Conv3d(
            channels, channels, (kernel, 1, 1),
            groups=channels, bias=False)
        self.projection = nn.Conv3d(channels, 2 * channels, 1)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 5:
            raise ValueError("history mixer requires [B,T,C,H,W]")
        batch, steps, channels, height, width = trajectory.shape
        normalized = self.norm(trajectory.reshape(
            batch * steps, channels, height, width))
        if self.mode == "causal_temporal":
            normalized = normalized.reshape(
                batch, steps, channels, height, width
            ).permute(0, 2, 1, 3, 4)
            normalized = F.pad(
                normalized, (0, 0, 0, 0, self.kernel - 1, 0))
            filtered = self.depthwise(normalized)
            value, gate = self.projection(filtered).chunk(2, dim=1)
            update = (value * F.silu(gate)).permute(0, 2, 1, 3, 4)
        else:
            normalized = normalized.permute(0, 1, 3, 2).unsqueeze(-1)
            left = (self.kernel - 1) // 2
            right = self.kernel - 1 - left
            normalized = F.pad(normalized, (0, 0, 0, 0, left, right))
            filtered = self.depthwise(normalized)
            value, gate = self.projection(filtered).chunk(2, dim=1)
            update = (value * F.silu(gate)).squeeze(-1)
            update = update.permute(0, 1, 3, 2).reshape(
                batch, steps, channels, height, width)
        return trajectory + update


class HistoryAxisAdapter(nn.Module):
    """Residual pre-flatten adapter preserving the observed time axis."""

    def __init__(self, config: UNet2dConfig):
        super().__init__()
        if config.history_axis_adapter_mode == "none":
            raise ValueError("history axis adapter requires an active mode")
        channels = config.history_axis_adapter_channels
        self.in_channels = int(config.in_channels)
        self.hidden_channels = int(channels)
        self.residual_limit = float(
            config.history_axis_adapter_residual_limit)
        self.input_projection = nn.Conv2d(
            config.in_channels, channels, 1)
        self.blocks = nn.ModuleList(
            HistoryAxisMixerBlock(
                channels, config.history_axis_adapter_kernel,
                config.groups, config.history_axis_adapter_mode)
            for _ in range(config.history_axis_adapter_blocks))
        self.output_projection = nn.Conv2d(
            channels, config.in_channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or history.shape[-1] != self.in_channels:
            raise ValueError(
                "history axis adapter requires [B,T,H,W,in_channels]")
        batch, steps, height, width, channels = history.shape
        frames = history.permute(0, 1, 4, 2, 3).reshape(
            batch * steps, channels, height, width)
        encoded = F.silu(self.input_projection(frames)).reshape(
            batch, steps, self.hidden_channels, height, width)
        for block in self.blocks:
            encoded = block(encoded)
        residual = self.output_projection(encoded.reshape(
            batch * steps, self.hidden_channels, height, width))
        residual = residual.reshape(
            batch, steps, self.in_channels, height, width
        ).permute(0, 1, 3, 4, 2)
        return history + self.residual_limit * torch.tanh(
            residual / self.residual_limit)


class HistoryDynamicsState(nn.Module):
    """Encode observed frames into one shared spatial dynamics state.

    The encoded last frame, temporal mean, and total observed change carry
    complementary recent, persistent, and trend information.  Unlike the
    residual history adapter, this state is an additive input to all three
    learned history consumers and never changes the raw history itself.
    """

    def __init__(self, config: UNet2dConfig):
        super().__init__()
        if config.history_dynamics_state_mode == "none":
            raise ValueError("history dynamics state requires an active mode")
        self.in_channels = int(config.in_channels)
        self.hidden_channels = int(config.history_dynamics_state_channels)
        self.input_projection = nn.Conv2d(
            self.in_channels, self.hidden_channels, 1)
        self.blocks = nn.ModuleList(
            HistoryAxisMixerBlock(
                self.hidden_channels, config.history_dynamics_state_kernel,
                config.groups, config.history_dynamics_state_mode)
            for _ in range(config.history_dynamics_state_blocks))

    def encode_frames(self, history: torch.Tensor) -> torch.Tensor:
        """Return per-frame features as ``[B,T,C,H,W]`` for inspection/use."""
        if history.ndim != 5 or history.shape[-1] != self.in_channels:
            raise ValueError(
                "history dynamics state requires [B,T,H,W,in_channels]")
        batch, steps, height, width, channels = history.shape
        frames = history.permute(0, 1, 4, 2, 3).reshape(
            batch * steps, channels, height, width)
        encoded = F.silu(self.input_projection(frames)).reshape(
            batch, steps, self.hidden_channels, height, width)
        for block in self.blocks:
            encoded = block(encoded)
        return encoded

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_frames(history)
        return torch.cat((
            encoded[:, -1], encoded.mean(dim=1), encoded[:, -1] - encoded[:, 0],
        ), dim=1)


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


class LowRankLatentSpectralResidual(nn.Module):
    """One bounded global interaction at the coarsest U-Net resolution.

    The local U-Net remains the primary path.  A low-rank Fourier operator
    mixes only a fixed set of coarse spatial modes and returns a zero-init
    residual, so enabling the branch preserves the complete legacy forward at
    update zero.  The final projection is deliberately real-valued: after it
    learns to admit a global correction, gradients reach the spectral kernel
    without making the Fourier branch own the decoder.
    """

    def __init__(
            self, channels: int, rank: int, modes_y: int, modes_x: int,
            groups: int, residual_limit: float):
        super().__init__()
        if min(channels, rank, modes_y, modes_x) <= 0:
            raise ValueError("latent spectral dimensions must be positive")
        self.rank = int(rank)
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)
        self.residual_limit = float(residual_limit)
        self.norm = _group_norm(groups, channels)
        self.reduce = nn.Conv2d(channels, rank, 1)
        scale = rank ** -0.5
        shape = (rank, rank, modes_y, modes_x)
        self.weight_positive = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_negative = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat))
        self.expand = nn.Conv2d(rank, channels, 1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    @staticmethod
    def _multiply(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcyx,coyx->boyx", values, weights)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        reduced = F.silu(self.reduce(self.norm(value)))
        height, width = reduced.shape[-2:]
        if 2 * self.modes_y > height:
            raise ValueError(
                f"latent modes_y={self.modes_y} overlap at height={height}")
        if self.modes_x > width // 2 + 1:
            raise ValueError(
                f"latent modes_x={self.modes_x} exceeds rFFT width={width}")
        transformed = torch.fft.rfft2(reduced.float(), dim=(-2, -1))
        spectral = torch.zeros(
            reduced.shape[0], self.rank, height, width // 2 + 1,
            device=reduced.device, dtype=transformed.dtype)
        positive = self.weight_positive.to(dtype=transformed.dtype)
        negative = self.weight_negative.to(dtype=transformed.dtype)
        spectral[:, :, :self.modes_y, :self.modes_x] = self._multiply(
            transformed[:, :, :self.modes_y, :self.modes_x], positive)
        spectral[:, :, -self.modes_y:, :self.modes_x] = self._multiply(
            transformed[:, :, -self.modes_y:, :self.modes_x], negative)
        global_update = torch.fft.irfft2(
            spectral, s=(height, width), dim=(-2, -1)).to(reduced.dtype)
        correction = self.expand(F.silu(global_update))
        return value + self.residual_limit * torch.tanh(correction)


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
            decoder_channels: list[int], groups: int,
            dynamics_state_channels: int = 0):
        super().__init__()
        self.recent_steps = int(recent_steps)
        # recent du/dt + recent d2u/dt2 + local recent mean/std, all u/v
        self.dynamics_state_channels = int(dynamics_state_channels)
        input_channels = (
            2 * recent_steps + 2 * (recent_steps - 1) + 4
            + self.dynamics_state_channels)
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

    def encode(
            self, history: torch.Tensor,
            dynamics_state: torch.Tensor | None = None) -> torch.Tensor:
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
        inputs = [first, second, mean, spread]
        if self.dynamics_state_channels:
            if dynamics_state is None:
                raise ValueError("cross-conditioned dynamics state is missing")
            if (dynamics_state.ndim != 4
                    or dynamics_state.shape[1] != self.dynamics_state_channels
                    or dynamics_state.shape[0] != batch
                    or dynamics_state.shape[-2:] != (height, width)):
                raise ValueError("cross-conditioned dynamics state shape changed")
            inputs.append(dynamics_state)
        elif dynamics_state is not None:
            raise ValueError("unexpected cross-conditioned dynamics state")
        return self.body(torch.cat(inputs, dim=1))

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


class DeformableHistoryMemory(nn.Module):
    """Read a fixed set of observed frames at locally displaced locations.

    This is deliberately a feature-space side branch, not a future-state
    decoder or an output warp.  It preserves the direct U-Net point heads and
    only supplies one residual context field at the B210 bottleneck and first
    decoder scale.  The two experiment modes instantiate the same tensors and
    execute the same sampling path; they differ only in whether a location
    uses its own displacement field or a fixed half-grid permutation of it.
    """

    def __init__(
            self, config: UNet2dConfig, bottleneck_channels: int,
            first_decoder_channels: int):
        super().__init__()
        if config.deformable_history_memory_mode == "none":
            raise ValueError("deformable history memory requires an active mode")
        self.mode = str(config.deformable_history_memory_mode)
        self.memory_channels = int(config.deformable_history_memory_channels)
        self.anchors = tuple(int(value) for value in (
            config.deformable_history_memory_anchors))
        self.max_offset = float(config.deformable_history_memory_max_offset)
        self.frame_encoder = nn.Conv2d(
            config.in_channels, self.memory_channels, 3, padding=1)
        channels = (int(bottleneck_channels), int(first_decoder_channels))
        query_channels = 3 * len(self.anchors)
        self.query_heads = nn.ModuleList(
            nn.Conv2d(width, query_channels, 1) for width in channels)
        self.value_projections = nn.ModuleList(
            nn.Conv2d(self.memory_channels, width, 1) for width in channels)
        # The branch is an exact identity at update zero.  Query fields are
        # initialized normally so the value projection gets a live gradient
        # on the first optimization step.
        for projection in self.value_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def prepare(
            self, history: torch.Tensor,
            sizes: tuple[tuple[int, int], tuple[int, int]],
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode each observed frame once and average-pool two memory banks."""
        if history.ndim != 5 or history.shape[-1] != self.frame_encoder.in_channels:
            raise ValueError(
                "deformable history memory requires [B,T,H,W,in_channels]")
        batch, steps, height, width, channels = history.shape
        if steps <= max(self.anchors):
            raise ValueError("deformable history memory anchor exceeds history")
        frames = history.permute(0, 1, 4, 2, 3).reshape(
            batch * steps, channels, height, width)
        frames = F.silu(self.frame_encoder(frames))
        banks = []
        for size in sizes:
            pooled = F.adaptive_avg_pool2d(frames, size)
            banks.append(pooled.reshape(
                batch, steps, self.memory_channels, *size))
        return tuple(banks)  # type: ignore[return-value]

    @staticmethod
    def _base_grid(value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        y = torch.linspace(-1.0, 1.0, height, device=value.device,
                           dtype=value.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=value.device,
                           dtype=value.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).view(
            1, 1, height, width, 2)

    def apply(
            self, feature: torch.Tensor, bank: torch.Tensor, scale: int,
            *, zero_offset: bool = False) -> torch.Tensor:
        """Sample one bank and add its zero-initialized residual context."""
        if not 0 <= scale < len(self.query_heads):
            raise ValueError("deformable history memory scale changed")
        if feature.ndim != 4 or bank.ndim != 5:
            raise ValueError("deformable history memory feature rank changed")
        batch, channels, height, width = feature.shape
        if bank.shape != (
                batch, bank.shape[1], self.memory_channels, height, width):
            raise ValueError("deformable history memory bank shape changed")
        if bank.shape[1] <= max(self.anchors):
            raise ValueError("deformable history memory bank loses an anchor")
        raw = self.query_heads[scale](feature)
        anchor_count = len(self.anchors)
        logits, offsets = raw[:, :anchor_count], raw[:, anchor_count:]
        offsets = self.max_offset * torch.tanh(offsets.reshape(
            batch, anchor_count, 2, height, width))
        # Always construct both fields.  This keeps candidate/control graph
        # cost and parameter usage matched instead of giving the null a cheap
        # zero-offset shortcut.
        rolled_offsets = torch.roll(
            offsets, shifts=(height // 2, width // 2), dims=(-2, -1))
        # Keep both direct and rolled offset graphs connected in both arms.
        # The zero coefficient leaves the selected sampling coordinates exact,
        # while avoiding a candidate/control backward graph asymmetry.
        selected_offsets = (
            offsets + 0.0 * rolled_offsets
            if self.mode == "aligned" else
            rolled_offsets + 0.0 * offsets)
        if zero_offset:
            selected_offsets = torch.zeros_like(selected_offsets)
        scale_xy = offsets.new_tensor((
            2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)))
        grid = self._base_grid(feature) + selected_offsets.permute(
            0, 1, 3, 4, 2) * scale_xy
        values = bank[:, self.anchors].reshape(
            batch * anchor_count, self.memory_channels, height, width)
        sampled = F.grid_sample(
            values.float(), grid.reshape(
                batch * anchor_count, height, width, 2).float(),
            mode="bilinear", padding_mode="border", align_corners=True,
        ).to(feature.dtype).reshape(
            batch, anchor_count, self.memory_channels, height, width)
        weights = torch.softmax(logits, dim=1).unsqueeze(2)
        memory = (sampled * weights).sum(dim=1)
        return feature + self.value_projections[scale](memory)


class UNet2d(nn.Module, ForecastDistributionMixin):
    """Direct multi-horizon U-Net with mean/fluctuation and native UQ heads."""

    def __init__(self, config: UNet2dConfig):
        super().__init__()
        config.validate()
        self.config = config
        channels = [config.base * 2 ** level for level in range(config.depth)]
        self.stem = nn.Conv2d(
            config.feature_channels + config.dynamics_state_channels,
            channels[0], 3, padding=1)
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
        # Keep this optional architecture arm outside the legacy RNG stream.
        # Its zero output projection makes the complete model bitwise
        # identical to the local-only U-Net at update zero.
        with torch.random.fork_rng(devices=[]):
            self.latent_spectral_residual = (
                LowRankLatentSpectralResidual(
                    channels[-1], config.latent_spectral_rank,
                    config.latent_spectral_modes_y,
                    config.latent_spectral_modes_x, config.groups,
                    config.latent_spectral_residual_limit)
                if config.latent_spectral_rank else None)
        # This research arm is isolated from the legacy RNG stream.  The
        # zero output projection makes its update-zero mapping exactly the
        # identity while retaining a live first-step gradient.
        with torch.random.fork_rng(devices=[]):
            self.history_axis_adapter = (
                HistoryAxisAdapter(config)
                if config.history_axis_adapter_mode != "none" else None)
        # This is intentionally separate from the residual adapter: it emits
        # one shared state rather than modifying the observed trajectory.
        with torch.random.fork_rng(devices=[]):
            self.history_dynamics_state = (
                HistoryDynamicsState(config)
                if config.history_dynamics_state_mode != "none" else None)
        if config.use_regime_film:
            self.regime_film = nn.Sequential(
                nn.Linear(config.regime_summary_channels, channels[-1]), nn.SiLU(),
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
                nn.Linear(config.regime_summary_channels, 2 * width)
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
                decoder_channels, config.groups,
                config.dynamics_state_channels)
            if config.cross_conditioned_backbone else None)
        # Construct the optional memory inside an isolated RNG scope.  The
        # disabled path has no module/state and the active branch begins as an
        # exact residual identity, so legacy checkpoints and update-zero
        # candidate/control states remain comparable.
        with torch.random.fork_rng(devices=[]):
            self.deformable_history_memory = (
                DeformableHistoryMemory(
                    config, channels[-1], decoder_channels[0])
                if config.deformable_history_memory_mode != "none" else None)
        self._deformable_history_memory_zero_offset = False

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
        return UNet2d._multi_scale_values(
            [head(feature) for feature, head in zip(features, heads)], size)

    @staticmethod
    def _multi_scale_values(
            native: list[torch.Tensor],
            size: tuple[int, int]) -> torch.Tensor:
        if not native:
            raise ValueError("multi-scale point values changed")
        values = []
        for value in native:
            if value.shape[-2:] != size:
                value = F.interpolate(
                    value, size=size, mode="bilinear", align_corners=False)
            values.append(value)
        return torch.stack(values).mean(dim=0)

    @staticmethod
    def _laplacian_multi_scale(
            features: list[torch.Tensor], heads: nn.ModuleList,
            size: tuple[int, int]) -> torch.Tensor:
        """Assign each point head a non-overlapping approximate spatial band.

        Decoder features are ordered from coarse to fine.  The first head
        retains its complete field; every later head has the component visible
        at the preceding resolution removed before all bands are brought to
        the output grid.  Averaging preserves the legacy per-head gradient
        scale at the exact-zero point-head initialization.
        """
        if len(features) != len(heads) or not features:
            raise ValueError("Laplacian point features/heads changed")
        native = [head(feature) for feature, head in zip(features, heads)]
        return UNet2d._laplacian_multi_scale_values(native, size)

    @staticmethod
    def _laplacian_multi_scale_values(
            native: list[torch.Tensor],
            size: tuple[int, int]) -> torch.Tensor:
        if not native:
            raise ValueError("Laplacian point values changed")
        bands = [native[0]]
        for index in range(1, len(native)):
            coarse_size = native[index - 1].shape[-2:]
            low = F.interpolate(native[index], size=coarse_size, mode="area")
            low = F.interpolate(
                low, size=native[index].shape[-2:], mode="bilinear",
                align_corners=False)
            bands.append(native[index] - low)
        values = []
        for value in bands:
            if value.shape[-2:] != size:
                value = F.interpolate(
                    value, size=size, mode="bilinear", align_corners=False)
            values.append(value)
        return torch.stack(values).mean(dim=0)

    def _encode_history(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None,
                       torch.Tensor | None]:
        """Encode one history while retaining skips and regime context.

        Keeping this boundary explicit lets research models attach genuinely
        different decoders after one shared bottleneck.  The legacy path calls
        the same operations in the same order through :meth:`_decode`.
        """
        stem_history = (
            self.history_axis_adapter(x)
            if self.history_axis_adapter is not None else x)
        dynamics_state = (
            self.history_dynamics_state(x)
            if self.history_dynamics_state is not None else None)
        stem_features = build_history_features(
            stem_history, self.config.difference_steps,
            self.config.use_coordinates,
            self.config.acceleration_steps)
        # Keep the inactive path's exact feature construction and stem call.
        encoded = self.stem(
            torch.cat((stem_features, dynamics_state), dim=1)
            if dynamics_state is not None else stem_features)
        skips = [encoded]
        for block in self.encoder:
            encoded = block(encoded)
            skips.append(encoded)
        encoded = self.bottleneck(encoded)
        if self.latent_spectral_residual is not None:
            encoded = self.latent_spectral_residual(encoded)
        if self.regime_film is not None or self.decoder_regime_film is not None:
            regime = history_regime_summary(x)
            if dynamics_state is not None:
                spatial_mean = dynamics_state.mean(dim=(2, 3))
                spatial_std = dynamics_state.std(dim=(2, 3), unbiased=False)
                regime = torch.cat((regime, spatial_mean, spatial_std), dim=-1)
        else:
            regime = None
        if self.regime_film is not None:
            scale, shift = self.regime_film(regime).chunk(2, dim=-1)
            encoded = encoded * (1.0 + scale[..., None, None])
            encoded = encoded + shift[..., None, None]
        return encoded, skips, regime, dynamics_state

    @staticmethod
    def _decode_from_encoded(
            encoded: torch.Tensor, skips: list[torch.Tensor],
            regime: torch.Tensor | None, dynamics_state: torch.Tensor | None,
            x: torch.Tensor,
            up: nn.ModuleList, decoder: nn.ModuleList,
            decoder_regime_film: nn.ModuleList | None,
            cross_conditioner: LocalDynamicsCrossConditioner | None,
            deformable_history_memory: DeformableHistoryMemory | None = None,
            deformable_memory_bank: torch.Tensor | None = None,
            deformable_zero_offset: bool = False,
            ) -> list[torch.Tensor]:
        """Decode one bottleneck with an explicitly supplied operator path."""
        local_condition = (
            cross_conditioner.encode(x, dynamics_state)
            if cross_conditioner is not None else None)
        features = []
        for index, (upsample, block) in enumerate(zip(up, decoder)):
            encoded = upsample(encoded)
            encoded = block(torch.cat((encoded, skips[-(index + 2)]), dim=1))
            if decoder_regime_film is not None:
                if regime is None:
                    raise ValueError("decoder regime context is missing")
                scale, shift = decoder_regime_film[index](
                    regime).chunk(2, dim=-1)
                encoded = encoded * (1.0 + scale[..., None, None])
                encoded = encoded + shift[..., None, None]
            if cross_conditioner is not None:
                encoded = cross_conditioner.modulate(
                    encoded, local_condition, index)
            if deformable_history_memory is not None and index == 0:
                if deformable_memory_bank is None:
                    raise ValueError("deformable history memory bank is missing")
                encoded = deformable_history_memory.apply(
                    encoded, deformable_memory_bank, 1,
                    zero_offset=deformable_zero_offset)
            features.append(encoded)
        return features

    def set_deformable_history_memory_zero_offset(
            self, enabled: bool) -> None:
        """Select the zero-spatial-offset mechanism ablation at runtime.

        This is deliberately not serialized in the checkpoint.  It permits a
        same-weight aligned-versus-zero-offset inference check without adding
        a second model state or changing the screened candidate architecture.
        """
        if self.deformable_history_memory is None:
            raise RuntimeError("deformable history memory is disabled")
        self._deformable_history_memory_zero_offset = bool(enabled)

    def _decode(self, x: torch.Tensor) -> list[torch.Tensor]:
        encoded, skips, regime, dynamics_state = self._encode_history(x)
        if self.deformable_history_memory is None:
            return self._decode_from_encoded(
                encoded, skips, regime, dynamics_state, x, self.up,
                self.decoder, self.decoder_regime_film, self.cross_conditioner)
        banks = self.deformable_history_memory.prepare(
            x, (encoded.shape[-2:], skips[-2].shape[-2:]))
        encoded = self.deformable_history_memory.apply(
            encoded, banks[0], 0,
            zero_offset=self._deformable_history_memory_zero_offset)
        return self._decode_from_encoded(
            encoded, skips, regime, dynamics_state, x, self.up, self.decoder,
            self.decoder_regime_film, self.cross_conditioner,
            self.deformable_history_memory, banks[1],
            self._deformable_history_memory_zero_offset)

    def _point_prediction(self, x: torch.Tensor,
                          features: list[torch.Tensor]) -> torch.Tensor:
        batch, _, height, width, _ = x.shape
        point_fusion = (
            self._laplacian_multi_scale
            if self.config.laplacian_point_pyramid else self._multi_scale)
        raw = point_fusion(features, self.point_heads, (height, width))
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
