"""A high-frequency-aware 2D neural operator for 20-to-20 flow forecasting.

The released FNO treats time and the two spatial axes as one 3D Fourier
volume.  Its retained temporal/spatial modes are deliberately small, while
the non-spectral branch is only a pointwise convolution.  This model instead
encodes the complete observed time history into channels and applies 2D
operator blocks over the physical plane.  Every block combines:

* a high-mode global Fourier branch;
* a depthwise local convolution branch for unresolved/high-frequency flow;
* a pointwise channel mixer;
* sample-local GroupNorm instead of running-statistics BatchNorm.

The decoder predicts an increment around the last observed frame.  This is a
new point-model backbone, not a correction head attached to a released FNO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HybridFNO2dConfig:
    in_steps: int = 20
    out_steps: int = 20
    in_channels: int = 3
    out_channels: int = 3
    width: int = 48
    layers: int = 4
    modes_y: int = 14
    modes_x: int = 24
    local_kernel: int = 3
    groups: int = 8
    expansion: int = 2
    padding: int = 4
    use_local_branch: bool = True
    use_velocity_differences: bool = True
    predict_uncertainty: bool = False
    uncertainty_hidden: int = 24
    uncertainty_base_widths: tuple[float, float] = (0.01075, 0.00550)
    temporal_decoder_width: int = 0
    temporal_decoder_layers: int = 3
    temporal_decoder_groups: int = 4
    predict_energy_gain: bool = False
    energy_gain_hidden: int = 16

    def validate(self) -> None:
        integer_fields = (
            self.in_steps, self.out_steps, self.in_channels, self.out_channels,
            self.width, self.layers, self.modes_y, self.modes_x,
            self.local_kernel, self.groups, self.expansion,
            self.uncertainty_hidden, self.energy_gain_hidden,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("architecture dimensions must be positive")
        if self.temporal_decoder_width < 0:
            raise ValueError("temporal decoder width must be non-negative")
        if self.temporal_decoder_layers <= 0 or self.temporal_decoder_groups <= 0:
            raise ValueError("temporal decoder dimensions must be positive")
        if (self.temporal_decoder_width
                and self.temporal_decoder_width % self.temporal_decoder_groups):
            raise ValueError("temporal decoder width must be divisible by groups")
        if self.width % self.groups:
            raise ValueError("width must be divisible by groups")
        if self.local_kernel % 2 != 1:
            raise ValueError("local_kernel must be odd")
        if self.padding < 0:
            raise ValueError("padding must be non-negative")
        if len(self.uncertainty_base_widths) != 2:
            raise ValueError("uncertainty_base_widths must contain u/v widths")
        if any(value <= 0 for value in self.uncertainty_base_widths):
            raise ValueError("uncertainty base widths must be positive")

    def to_dict(self) -> dict[str, int | bool | tuple[float, float]]:
        return asdict(self)


class SpectralConv2d(nn.Module):
    """Global 2D Fourier convolution with independent +/- y modes."""

    def __init__(self, width: int, modes_y: int, modes_x: int):
        super().__init__()
        self.width = int(width)
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)
        scale = 1.0 / max(width, 1) ** 0.5
        shape = (width, width, modes_y, modes_x)
        self.weight_positive = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_negative = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _multiply(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcyx,coyx->boyx", values, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        if self.modes_y * 2 > height:
            raise ValueError(
                f"modes_y={self.modes_y} overlaps for padded height={height}")
        if self.modes_x > width // 2 + 1:
            raise ValueError(
                f"modes_x={self.modes_x} exceeds rFFT width={width // 2 + 1}")
        transformed = torch.fft.rfft2(x, dim=(-2, -1))
        output = torch.zeros(
            x.shape[0], self.width, height, width // 2 + 1,
            device=x.device, dtype=transformed.dtype)
        output[:, :, :self.modes_y, :self.modes_x] = self._multiply(
            transformed[:, :, :self.modes_y, :self.modes_x],
            self.weight_positive)
        output[:, :, -self.modes_y:, :self.modes_x] = self._multiply(
            transformed[:, :, -self.modes_y:, :self.modes_x],
            self.weight_negative)
        return torch.fft.irfft2(output, s=(height, width), dim=(-2, -1))


class HybridOperatorBlock(nn.Module):
    def __init__(self, config: HybridFNO2dConfig):
        super().__init__()
        width = config.width
        self.norm_operator = nn.GroupNorm(config.groups, width)
        self.spectral = SpectralConv2d(width, config.modes_y, config.modes_x)
        self.pointwise = nn.Conv2d(width, width, 1)
        if config.use_local_branch:
            padding = config.local_kernel // 2
            self.local = nn.Sequential(
                nn.Conv2d(
                    width, width, config.local_kernel, padding=padding,
                    groups=width, bias=False),
                nn.Conv2d(width, width, 1),
            )
        else:
            self.local = None
        self.operator_gain = nn.Parameter(torch.full((1, width, 1, 1), 0.5))
        hidden = width * config.expansion
        self.norm_mlp = nn.GroupNorm(config.groups, width)
        self.mlp = nn.Sequential(
            nn.Conv2d(width, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, width, 1),
        )
        self.mlp_gain = nn.Parameter(torch.full((1, width, 1, 1), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_operator(x)
        update = self.spectral(normalized) + self.pointwise(normalized)
        if self.local is not None:
            update = update + self.local(normalized)
        x = x + self.operator_gain * F.gelu(update)
        return x + self.mlp_gain * self.mlp(self.norm_mlp(x))


class TemporalDecoderBlock(nn.Module):
    """Residual local mixing over future time and the physical plane."""

    def __init__(self, width: int, groups: int):
        super().__init__()
        self.norm_operator = nn.GroupNorm(groups, width)
        self.operator = nn.Sequential(
            nn.Conv3d(
                width, width, kernel_size=3, padding=1,
                groups=width, bias=False),
            nn.Conv3d(width, width, kernel_size=1),
        )
        self.operator_gain = nn.Parameter(torch.full((1, width, 1, 1, 1), 0.5))
        self.norm_mlp = nn.GroupNorm(groups, width)
        self.mlp = nn.Sequential(
            nn.Conv3d(width, 2 * width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(2 * width, width, kernel_size=1),
        )
        self.mlp_gain = nn.Parameter(torch.full((1, width, 1, 1, 1), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.gelu(self.operator(self.norm_operator(x)))
        x = x + self.operator_gain * update
        return x + self.mlp_gain * self.mlp(self.norm_mlp(x))


class HybridFNO2d(nn.Module):
    """Direct multi-horizon flow operator with a normalized persistence skip."""

    def __init__(self, config: HybridFNO2dConfig):
        super().__init__()
        config.validate()
        self.config = config
        history_channels = config.in_steps * config.in_channels
        if config.use_velocity_differences:
            history_channels += (config.in_steps - 1) * min(2, config.in_channels)
        history_channels += 2  # normalized physical-plane coordinates
        self.lift = nn.Sequential(
            nn.Conv2d(history_channels, config.width, 1),
            nn.GELU(),
            nn.Conv2d(config.width, config.width, 3, padding=1),
        )
        self.blocks = nn.ModuleList(
            HybridOperatorBlock(config) for _ in range(config.layers))
        self.final_norm = nn.GroupNorm(config.groups, config.width)
        if config.temporal_decoder_width:
            temporal_width = config.temporal_decoder_width
            self.decoder = None
            self.temporal_seed = nn.Conv2d(config.width, temporal_width, 1)
            self.horizon_embedding = nn.Parameter(torch.empty(
                1, temporal_width, config.out_steps, 1, 1))
            nn.init.normal_(self.horizon_embedding, std=0.02)
            self.temporal_blocks = nn.ModuleList(
                TemporalDecoderBlock(
                    temporal_width, config.temporal_decoder_groups)
                for _ in range(config.temporal_decoder_layers))
            self.temporal_mean = nn.Conv3d(
                temporal_width, config.out_channels, kernel_size=1)
            nn.init.normal_(self.temporal_mean.weight, std=1e-3)
            nn.init.zeros_(self.temporal_mean.bias)
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(config.width, config.width * 2, 1),
                nn.GELU(),
                nn.Conv2d(
                    config.width * 2,
                    config.out_steps * config.out_channels,
                    1),
            )
            nn.init.normal_(self.decoder[-1].weight, std=1e-3)
            nn.init.zeros_(self.decoder[-1].bias)
            self.temporal_seed = None
            self.horizon_embedding = None
            self.temporal_blocks = None
            self.temporal_mean = None
        if config.predict_energy_gain:
            energy_source_width = (
                config.temporal_decoder_width or config.width)
            self.energy_gain_decoder = nn.Sequential(
                nn.Conv2d(
                    energy_source_width, config.energy_gain_hidden,
                    kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(config.energy_gain_hidden, 2, kernel_size=1),
            )
            # A zero log-gain makes the new branch exactly identity at
            # initialization.  TKE gradients can then change fluctuation
            # amplitude without directly moving the temporal mean flow.
            nn.init.zeros_(self.energy_gain_decoder[-1].weight)
            nn.init.zeros_(self.energy_gain_decoder[-1].bias)
        else:
            self.energy_gain_decoder = None
        if config.predict_uncertainty:
            # A native heteroscedastic decoder.  Output channels are grouped as
            # (lower_u, lower_v, upper_u, upper_v) for every future step.
            if config.temporal_decoder_width:
                self.uncertainty_decoder = nn.Sequential(
                    nn.Conv3d(
                        config.temporal_decoder_width,
                        config.uncertainty_hidden, 3, padding=1),
                    nn.GELU(),
                    nn.Conv3d(config.uncertainty_hidden, 4, 1),
                )
            else:
                self.uncertainty_decoder = nn.Sequential(
                    nn.Conv2d(
                        config.width, config.uncertainty_hidden, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(
                        config.uncertainty_hidden,
                        config.out_steps * 4,
                        1),
                )
            nn.init.zeros_(self.uncertainty_decoder[-1].weight)
            nn.init.zeros_(self.uncertainty_decoder[-1].bias)
        else:
            self.uncertainty_decoder = None

        # Transform the last normalized input frame into target-normalized
        # units.  Training scripts update these buffers whenever the domain
        # normalizer changes from sim to real.
        self.register_buffer(
            "persistence_scale", torch.ones(config.out_channels), persistent=True)
        self.register_buffer(
            "persistence_bias", torch.zeros(config.out_channels), persistent=True)
        self.register_buffer(
            "target_std", torch.ones(config.out_channels), persistent=True)

    def architecture_config(self) -> dict[str, int | bool]:
        return self.config.to_dict()

    @torch.no_grad()
    def set_normalizer_statistics(
            self, mean_input: torch.Tensor, mean_target: torch.Tensor,
            std_input: torch.Tensor, std_target: torch.Tensor) -> None:
        channels = self.config.out_channels
        mean_input = torch.as_tensor(mean_input, device=self.persistence_scale.device)
        mean_target = torch.as_tensor(mean_target, device=self.persistence_scale.device)
        std_input = torch.as_tensor(std_input, device=self.persistence_scale.device)
        std_target = torch.as_tensor(std_target, device=self.persistence_scale.device)
        safe_target = std_target[:channels].clamp_min(1e-8)
        scale = std_input[:channels] / safe_target
        bias = (mean_input[:channels] - mean_target[:channels]) / safe_target
        zero_target = std_target[:channels].abs() < 1e-8
        scale = torch.where(zero_target, torch.zeros_like(scale), scale)
        bias = torch.where(zero_target, torch.zeros_like(bias), bias)
        self.persistence_scale.copy_(scale.to(self.persistence_scale.dtype))
        self.persistence_bias.copy_(bias.to(self.persistence_bias.dtype))
        self.target_std.copy_(
            safe_target.to(device=self.target_std.device,
                           dtype=self.target_std.dtype))

    @staticmethod
    def _grid(batch: int, height: int, width: int, device: torch.device,
              dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def _latent_and_persistence(
            self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.config.in_steps, self.config.in_channels)
        if x.ndim != 5 or (x.shape[1], x.shape[-1]) != expected:
            raise ValueError(
                "expected [B, in_steps, H, W, in_channels] with "
                f"steps/channels={expected}, got {tuple(x.shape)}")
        batch, _, height, width, _ = x.shape
        raw = x.permute(0, 1, 4, 2, 3).reshape(batch, -1, height, width)
        features = [raw]
        if self.config.use_velocity_differences:
            velocity = x[..., :min(2, self.config.in_channels)]
            difference = velocity[:, 1:] - velocity[:, :-1]
            difference = difference.permute(0, 1, 4, 2, 3).reshape(
                batch, -1, height, width)
            features.append(difference)
        features.append(self._grid(batch, height, width, x.device, x.dtype))
        latent = self.lift(torch.cat(features, dim=1))
        padding = self.config.padding
        if padding:
            latent = F.pad(latent, (padding, padding, padding, padding))
        for block in self.blocks:
            latent = block(latent)
        if padding:
            latent = latent[..., padding:-padding, padding:-padding]
        persistence = (
            x[:, -1, ..., :self.config.out_channels]
            * self.persistence_scale.view(1, 1, 1, -1)
            + self.persistence_bias.view(1, 1, 1, -1))
        return self.final_norm(latent), persistence

    def _decode_latent(
            self, latent: torch.Tensor,
            persistence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = latent.shape
        if self.temporal_seed is not None:
            temporal = self.temporal_seed(latent).unsqueeze(2)
            temporal = temporal.expand(-1, -1, self.config.out_steps, -1, -1)
            temporal = temporal + self.horizon_embedding
            for block in self.temporal_blocks:
                temporal = block(temporal)
            residual = self.temporal_mean(temporal).permute(0, 2, 3, 4, 1)
            source = temporal
        else:
            residual = self.decoder(latent).reshape(
                batch, self.config.out_steps, self.config.out_channels,
                height, width).permute(0, 1, 3, 4, 2)
            source = latent
        prediction = persistence.unsqueeze(1) + residual
        if self.energy_gain_decoder is not None:
            energy_source = (
                source.mean(dim=2) if source.ndim == 5 else source)
            log_gain = self.energy_gain_decoder(energy_source).permute(0, 2, 3, 1)
            gain = torch.exp(log_gain.clamp(-1.5, 1.5)).unsqueeze(1)
            velocity = prediction[..., :2]
            velocity_mean = velocity.mean(dim=1, keepdim=True)
            velocity = velocity_mean + gain * (velocity - velocity_mean)
            prediction = torch.cat((velocity, prediction[..., 2:]), dim=-1)
        return prediction, source

    def forward_distribution(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized mean and positive normalized u/v half-widths."""
        if self.uncertainty_decoder is None:
            raise RuntimeError("uncertainty decoder is disabled in this model")
        latent, persistence = self._latent_and_persistence(x)
        prediction, decode_source = self._decode_latent(latent, persistence)
        if self.temporal_seed is not None:
            log_width = self.uncertainty_decoder(
                decode_source).permute(0, 2, 3, 4, 1)
        else:
            batch, _, height, width = latent.shape
            log_width = self.uncertainty_decoder(decode_source).reshape(
                batch, self.config.out_steps, 4, height, width
            ).permute(0, 1, 3, 4, 2)
        log_width = log_width.clamp(-4.0, 4.0)
        physical_base = log_width.new_tensor(
            self.config.uncertainty_base_widths).view(1, 1, 1, 1, 2)
        normalized_base = physical_base / self.target_std[:2].clamp_min(
            1e-8).view(1, 1, 1, 1, 2)
        lower = normalized_base * torch.exp(log_width[..., :2])
        upper = normalized_base * torch.exp(log_width[..., 2:])
        return prediction, lower, upper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, persistence = self._latent_and_persistence(x)
        prediction, _decode_source = self._decode_latent(latent, persistence)
        return prediction
