"""Shared-backbone U-Net ensemble with a conditional residual distribution.

The model keeps inference to one encoder/decoder pass. Several lightweight
point heads provide sample-local epistemic disagreement, while a nonlinear
multi-scale decoder predicts monotone residual quantiles. Prediction intervals
are selected from those quantiles by the official SPS width utility instead of
using a fixed coverage target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from models.unet2d import (
    ResidualConvBlock,
    UNet2d,
    UNet2dConfig,
    _group_norm,
    history_regime_summary,
)


SPS_SIGMA = 0.0563870259


def _short_gap_cumsum(values: torch.Tensor) -> torch.Tensor:
    # Match CUDA's Sklansky scan, including each FP16 rounding operation.
    # These three nonnegative gaps are the two halves of seven quantiles.
    if values.is_cuda and values.dtype == torch.float16 and values.shape[-1] == 3:
        first, second, third = values.unbind(dim=-1)
        pair01 = second + first
        return torch.stack((first, pair01, third + pair01), dim=-1)
    return torch.cumsum(values, dim=-1)


@dataclass(frozen=True)
class EnsembleUNet2dConfig(UNet2dConfig):
    ensemble_members: int = 4
    uq_hidden_channels: int = 64
    member_adapter_channels: int = 0
    mean_profile_channels: int = 0
    uq_flow_context: bool = False
    residual_quantile_intervals: bool = False
    utility_refinement: bool = False
    utility_refinement_log_limit: float = 0.75
    utility_refinement_kernel: int = 1
    utility_refinement_blocks: int = 0
    temporal_uq_refinement: bool = False
    temporal_uq_channels: int = 24
    temporal_uq_blocks: int = 2
    temporal_uq_kernel: int = 3
    temporal_uq_logit_limit: float = 1.5
    coverage_calibration: bool = False
    coverage_calibration_kernel: int = 3
    coverage_calibration_blocks: int = 1
    coverage_calibration_logit_limit: float = 2.0
    coverage_calibration_mode: str = "independent"
    coverage_calibration_temperature_limit: float = 0.75
    quantile_levels: tuple[float, ...] = (
        0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95)
    quantile_gap_bias: float = -4.0
    disagreement_scale_fraction: float = 0.0
    shared_member_uv_mean: bool = False
    physical_energy_controller: bool = False
    physical_energy_member_sync: bool = False
    physical_energy_multiscale: bool = False
    physical_energy_horizon_local: bool = False
    physical_energy_history_conditioned: bool = False
    physical_energy_history_steps: int = 5
    physical_energy_hidden_channels: int = 32
    physical_energy_log_limit: float = 1.0
    phase_delta_hidden_channels: int = 0
    phase_delta_steps: int = 5
    temporal_point_decoder: bool = False
    temporal_point_channels: int = 24
    temporal_point_blocks: int = 2
    temporal_point_kernel: int = 5
    temporal_point_amplitude_factorization: bool = False
    temporal_point_amplitude_limit: float = 3.0
    isoenergetic_temporal_transport: bool = False
    temporal_transport_channels: int = 24
    temporal_transport_blocks: int = 2
    temporal_transport_kernel: int = 5
    temporal_transport_max_displacement: float = 1.5

    def validate(self) -> None:
        super().validate()
        if self.ensemble_members < 2:
            raise ValueError("ensemble_members must be at least two")
        if self.uq_hidden_channels <= 0:
            raise ValueError("uq_hidden_channels must be positive")
        if self.member_adapter_channels < 0:
            raise ValueError("member_adapter_channels must be non-negative")
        if self.mean_profile_channels < 0:
            raise ValueError("mean_profile_channels must be non-negative")
        if self.shared_member_uv_mean and not self.mean_fluctuation:
            raise ValueError(
                "shared member u/v mean requires mean_fluctuation=True")
        if self.physical_energy_controller and not self.mean_fluctuation:
            raise ValueError(
                "physical energy controller requires mean_fluctuation=True")
        if (self.physical_energy_member_sync
                and not self.physical_energy_controller):
            raise ValueError(
                "member-synchronous energy requires the physical energy "
                "controller")
        if (self.physical_energy_multiscale
                and not self.physical_energy_controller):
            raise ValueError(
                "multiscale physical energy requires the controller")
        if (self.physical_energy_horizon_local
                and not self.physical_energy_controller):
            raise ValueError(
                "horizon-local physical energy requires the controller")
        if (self.physical_energy_history_conditioned
                and not self.physical_energy_controller):
            raise ValueError(
                "history-conditioned physical energy requires the controller")
        if (self.physical_energy_horizon_local
                and (self.physical_energy_multiscale
                     or self.physical_energy_history_conditioned)):
            raise ValueError(
                "physical energy controller variants are separate arms")
        if not 2 <= self.physical_energy_history_steps < self.in_steps:
            raise ValueError(
                "physical energy history steps must be in [2, in_steps)")
        if self.physical_energy_hidden_channels <= 0:
            raise ValueError(
                "physical energy hidden channels must be positive")
        if not 0.0 < self.physical_energy_log_limit < float("inf"):
            raise ValueError("physical energy log limit must be positive")
        if self.phase_delta_hidden_channels < 0:
            raise ValueError("phase delta hidden channels must be non-negative")
        if not 2 <= self.phase_delta_steps < self.in_steps:
            raise ValueError("phase delta steps must be in [2, in_steps)")
        if self.temporal_point_channels <= 0:
            raise ValueError("temporal point channels must be positive")
        if self.temporal_point_blocks <= 0:
            raise ValueError("temporal point blocks must be positive")
        if (self.temporal_point_kernel <= 0
                or self.temporal_point_kernel % 2 == 0):
            raise ValueError(
                "temporal point kernel must be positive and odd")
        if (self.temporal_point_amplitude_factorization
                and not self.temporal_point_decoder):
            raise ValueError(
                "temporal amplitude factorization requires its decoder")
        if not 0.0 < self.temporal_point_amplitude_limit < float("inf"):
            raise ValueError("temporal point amplitude limit must be positive")
        if self.temporal_transport_channels <= 0:
            raise ValueError("temporal transport channels must be positive")
        if self.temporal_transport_blocks <= 0:
            raise ValueError("temporal transport blocks must be positive")
        if (self.temporal_transport_kernel <= 0
                or self.temporal_transport_kernel % 2 == 0):
            raise ValueError(
                "temporal transport kernel must be positive and odd")
        if not 0.0 < self.temporal_transport_max_displacement < float("inf"):
            raise ValueError(
                "temporal transport displacement must be positive")
        if self.uq_flow_context and self.in_steps < 6:
            raise ValueError("UQ flow context requires at least six input steps")
        levels = tuple(float(value) for value in self.quantile_levels)
        if len(levels) < 5 or len(levels) % 2 == 0:
            raise ValueError("quantile_levels must contain an odd count >= 5")
        if levels != tuple(sorted(set(levels))):
            raise ValueError("quantile_levels must be strictly increasing")
        if any(not 0.0 < value < 1.0 for value in levels):
            raise ValueError("quantile levels must lie in (0,1)")
        if levels[len(levels) // 2] != 0.5:
            raise ValueError("the middle quantile must be exactly 0.5")
        if not self.predict_uncertainty:
            raise ValueError("ensemble_unet requires predict_uncertainty=True")
        if not 0.0 <= self.disagreement_scale_fraction < float("inf"):
            raise ValueError("disagreement scale must be finite/non-negative")
        if not 0.0 < self.utility_refinement_log_limit < float("inf"):
            raise ValueError("utility refinement log limit must be positive")
        if (self.utility_refinement_kernel <= 0
                or self.utility_refinement_kernel % 2 == 0):
            raise ValueError("utility refinement kernel must be positive and odd")
        if self.utility_refinement_blocks < 0:
            raise ValueError("utility refinement blocks must be non-negative")
        if self.temporal_uq_channels <= 0:
            raise ValueError("temporal UQ channels must be positive")
        if self.temporal_uq_blocks <= 0:
            raise ValueError("temporal UQ blocks must be positive")
        if (self.temporal_uq_kernel <= 0
                or self.temporal_uq_kernel % 2 == 0):
            raise ValueError("temporal UQ kernel must be positive and odd")
        if not 0.0 < self.temporal_uq_logit_limit < float("inf"):
            raise ValueError("temporal UQ logit limit must be positive")
        if (self.coverage_calibration_kernel <= 0
                or self.coverage_calibration_kernel % 2 == 0):
            raise ValueError(
                "coverage calibration kernel must be positive and odd")
        if self.coverage_calibration_blocks < 0:
            raise ValueError(
                "coverage calibration blocks must be non-negative")
        if not 0.0 < self.coverage_calibration_logit_limit < float("inf"):
            raise ValueError(
                "coverage calibration logit limit must be positive")
        if self.coverage_calibration_mode not in (
                "independent", "temperature"):
            raise ValueError(
                "coverage calibration mode must be independent/temperature")
        if not 0.0 < self.coverage_calibration_temperature_limit < float("inf"):
            raise ValueError(
                "coverage calibration temperature limit must be positive")


def _flatten_forecast(value: torch.Tensor) -> torch.Tensor:
    batch, steps, height, width, channels = value.shape
    return value.permute(0, 1, 4, 2, 3).reshape(
        batch, steps * channels, height, width)


class _ExactIdentityAddSTE(torch.autograd.Function):
    """Keep an exact zero-controller forward while retaining its gradient."""

    @staticmethod
    def forward(ctx, legacy: torch.Tensor, delta: torch.Tensor,
                bypass: torch.Tensor) -> torch.Tensor:
        return torch.where(bypass, legacy, legacy + delta)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, grad_output, None


class MemberFeatureAdapter(nn.Module):
    """Zero-initialized spatial residual branch for one member and scale."""

    def __init__(self, width: int, bottleneck: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, bottleneck, 1),
            _group_norm(groups, bottleneck),
            nn.SiLU(),
            ResidualConvBlock(bottleneck, bottleneck, groups),
            nn.Conv2d(bottleneck, width, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


class PhysicalEnergyController(nn.Module):
    """Predict one target-free spatial log-energy correction per sample."""

    def __init__(self, input_channels: int, hidden_channels: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.body(feature)


class HorizonLocalPhysicalEnergyController(nn.Module):
    """Predict a separate spatial log-amplitude correction per horizon."""

    def __init__(
            self, input_channels: int, hidden_channels: int,
            out_steps: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_steps, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.body(feature)


class MultiScalePhysicalEnergyController(nn.Module):
    """Fuse every decoder scale into the deployed log-energy correction."""

    def __init__(
            self, input_channels: list[int], hidden_channels: int,
            groups: int):
        super().__init__()
        if not input_channels:
            raise ValueError("multiscale energy controller needs features")
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, hidden_channels, 1),
                _group_norm(groups, hidden_channels),
                nn.SiLU(),
            )
            for width in input_channels)
        # Keep the final projection at body.5 so all existing zero-init and
        # controller-gradient audits retain the same semantic key.
        self.body = nn.Sequential(
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError("multiscale energy feature contract changed")
        size = features[-1].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            value = projection(feature)
            if value.shape[-2:] != size:
                value = F.interpolate(
                    value, size=size, mode="bilinear", align_corners=False)
            projected.append(value)
        return self.body(torch.stack(projected).mean(dim=0))


class HistoryConditionedPhysicalEnergyController(nn.Module):
    """Predict energy correction from observed dynamics, not decoder state.

    The controller sees recent normalized u/v first differences plus their
    temporal mean and standard deviation.  Its input therefore stays attached
    to observations when a later Sim-to-Real stage adapts the point backbone.
    """

    def __init__(
            self, recent_steps: int, hidden_channels: int, groups: int):
        super().__init__()
        if recent_steps < 2:
            raise ValueError("history energy controller needs two recent steps")
        self.recent_steps = int(recent_steps)
        input_channels = 2 * recent_steps + 4
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if (history.ndim != 5 or history.shape[1] <= self.recent_steps
                or history.shape[-1] < 2):
            raise ValueError("history energy input must be [B,T,H,W,C>=2]")
        velocity = history[:, -(self.recent_steps + 1):, ..., :2]
        differences = velocity[:, 1:] - velocity[:, :-1]
        batch, steps, height, width, channels = differences.shape
        local = differences.permute(0, 1, 4, 2, 3).reshape(
            batch, steps * channels, height, width)
        mean = velocity.mean(dim=1).permute(0, 3, 1, 2)
        spread = velocity.std(dim=1, unbiased=False).permute(0, 3, 1, 2)
        return self.body(torch.cat((local, mean, spread), dim=1))


class HybridMultiScaleHistoryEnergyController(nn.Module):
    """Add decoder-multiscale and observation-history energy corrections."""

    def __init__(
            self, decoder_channels: list[int], recent_steps: int,
            hidden_channels: int, groups: int):
        super().__init__()
        decoder = MultiScalePhysicalEnergyController(
            decoder_channels, hidden_channels, groups)
        # Preserve top-level body.5/projections names used by all controller
        # gradient, update, checkpoint, and packaging audits.
        self.projections = decoder.projections
        self.body = decoder.body
        self.history = HistoryConditionedPhysicalEnergyController(
            recent_steps, hidden_channels, groups)

    def forward(
            self, features: list[torch.Tensor], history: torch.Tensor,
            ) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError("hybrid energy feature contract changed")
        size = features[-1].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            value = projection(feature)
            if value.shape[-2:] != size:
                value = F.interpolate(
                    value, size=size, mode="bilinear", align_corners=False)
            projected.append(value)
        decoder_delta = self.body(torch.stack(projected).mean(dim=0))
        history_delta = self.history(history)
        if history_delta.shape[-2:] != size:
            history_delta = F.interpolate(
                history_delta, size=size, mode="bilinear",
                align_corners=False)
        return decoder_delta + history_delta


class PhaseConditionedLocalDelta(nn.Module):
    """Predict a shared high-resolution forecast delta from recent dynamics."""

    def __init__(
            self, in_steps: int, out_steps: int, hidden_channels: int,
            recent_steps: int, groups: int):
        super().__init__()
        self.out_steps = int(out_steps)
        self.recent_steps = int(recent_steps)
        # recent first differences plus their recent second differences
        input_channels = 2 * (2 * recent_steps - 1)
        self.local = nn.Sequential(
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
        self.condition = nn.Sequential(
            nn.Linear(in_steps * 4, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 2 * hidden_channels),
        )
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        self.output = nn.Conv2d(hidden_channels, out_steps * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        velocity = history[:, -(self.recent_steps + 1):, ..., :2]
        differences = velocity[:, 1:] - velocity[:, :-1]
        accelerations = differences[:, 1:] - differences[:, :-1]
        batch, _steps, height, width, _channels = differences.shape
        local = torch.cat((
            differences.reshape(batch, self.recent_steps, height, width, 2),
            accelerations.reshape(
                batch, self.recent_steps - 1, height, width, 2),
        ), dim=1).permute(0, 1, 4, 2, 3).reshape(
            batch, 2 * (2 * self.recent_steps - 1), height, width)
        feature = self.local(local)
        scale, shift = self.condition(
            history_regime_summary(history)).chunk(2, dim=-1)
        feature = (
            feature * (1.0 + 0.25 * torch.tanh(scale)[..., None, None])
            + 0.25 * shift[..., None, None])
        raw = self.output(feature).reshape(
            batch, self.out_steps, 2, height, width)
        return raw.permute(0, 1, 3, 4, 2)


class SpatioTemporalResidualBlock(nn.Module):
    """Residual 3-D convolution over forecast horizon and image space."""

    def __init__(self, channels: int, kernel: int, groups: int):
        super().__init__()
        padding = kernel // 2
        self.body = nn.Sequential(
            nn.Conv3d(
                channels, channels, kernel,
                padding=padding, bias=False),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(
                channels, channels, kernel,
                padding=padding, bias=False),
            _group_norm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.body(value))


class FactorizedTemporalPointBlock(nn.Module):
    """Cheap explicit forecast-time mixing followed by spatial mixing."""

    def __init__(self, channels: int, temporal_kernel: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(
                channels, channels, (temporal_kernel, 1, 1),
                padding=(temporal_kernel // 2, 0, 0),
                groups=channels, bias=False),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(
                channels, channels, (1, 3, 3), padding=(0, 1, 1),
                groups=channels, bias=False),
            _group_norm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 1, bias=False),
            _group_norm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.body(value))


class TemporalPointDecoder(nn.Module):
    """Decode explicit horizon-local point trajectories at every U-Net scale.

    Legacy point heads flatten all forecast times into ordinary 2-D output
    channels.  This decoder keeps the time axis explicit, shares factorized
    temporal/spatial dynamics across ensemble members, and retains lightweight
    member-specific mean and fluctuation projections.  The direct variant
    zero-initializes its fluctuation outputs; the stable variant instead
    zero-initializes separate amplitude outputs.  Both make update zero exact
    persistence while the stable shape path starts non-degenerate.
    """

    def __init__(
            self, input_channels: list[int], hidden_channels: int,
            out_steps: int, out_channels: int, members: int,
            blocks: int, temporal_kernel: int, groups: int,
            amplitude_factorization: bool = False,
            amplitude_limit: float = 3.0):
        super().__init__()
        if not input_channels:
            raise ValueError("temporal point decoder needs decoder features")
        self.out_steps = int(out_steps)
        self.out_channels = int(out_channels)
        self.amplitude_factorization = bool(amplitude_factorization)
        self.amplitude_limit = float(amplitude_limit)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, hidden_channels, 1, bias=False),
                _group_norm(groups, hidden_channels),
                nn.SiLU(),
            )
            for width in input_channels)
        position = torch.arange(out_steps, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, hidden_channels, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10_000.0)) / hidden_channels))
        embedding = torch.zeros(out_steps, hidden_channels)
        embedding[:, 0::2] = torch.sin(position * frequency)
        embedding[:, 1::2] = torch.cos(
            position * frequency[:embedding[:, 1::2].shape[1]])
        self.horizon_embedding = nn.Parameter(
            embedding.transpose(0, 1).reshape(
                1, hidden_channels, out_steps, 1, 1))
        self.scale_blocks = nn.ModuleList(
            nn.Sequential(*(
                FactorizedTemporalPointBlock(
                    hidden_channels, temporal_kernel, groups)
                for _ in range(blocks)))
            for _ in input_channels)
        self.member_mean_heads = nn.ModuleList(
            nn.ModuleList(
                nn.Conv2d(width, out_channels, 1)
                for width in input_channels)
            for _ in range(members))
        self.member_fluctuation_heads = nn.ModuleList(
            nn.ModuleList(
                nn.Conv3d(hidden_channels, out_channels, 1)
                for _ in input_channels)
            for _ in range(members))
        self.member_amplitude_heads = (
            nn.ModuleList(
                nn.ModuleList(
                    nn.Conv2d(width, out_channels, 1)
                    for width in input_channels)
                for _ in range(members))
            if self.amplitude_factorization else None)
        zero_heads = list(self.member_mean_heads)
        if self.member_amplitude_heads is not None:
            zero_heads.extend(self.member_amplitude_heads)
        else:
            zero_heads.extend(self.member_fluctuation_heads)
        for member_heads in zero_heads:
            for head in member_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def temporal_features(
            self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError("temporal point feature contract changed")
        return [
            blocks(
                projection(feature).unsqueeze(2)
                + self.horizon_embedding)
            for feature, projection, blocks in zip(
                features, self.projections, self.scale_blocks)
        ]

    def member_prediction(
            self, persistence: torch.Tensor, features: list[torch.Tensor],
            temporal_features: list[torch.Tensor], member: int,
            ) -> torch.Tensor:
        if (len(features) != len(self.projections)
                or len(temporal_features) != len(self.projections)):
            raise ValueError("temporal point scale contract changed")
        batch, height, width, channels = persistence.shape
        if channels != self.out_channels:
            raise ValueError("temporal point persistence channels changed")
        means = []
        fluctuations = []
        for feature, temporal, mean_head, fluctuation_head in zip(
                features, temporal_features,
                self.member_mean_heads[member],
                self.member_fluctuation_heads[member]):
            mean = mean_head(feature)
            if mean.shape[-2:] != (height, width):
                mean = F.interpolate(
                    mean, size=(height, width), mode="bilinear",
                    align_corners=False)
            means.append(mean)
            fluctuation = fluctuation_head(temporal)
            if fluctuation.shape[-2:] != (height, width):
                fluctuation = F.interpolate(
                    fluctuation,
                    size=(self.out_steps, height, width),
                    mode="trilinear", align_corners=False)
            if self.member_amplitude_heads is not None:
                fluctuation = fluctuation - fluctuation.mean(
                    dim=2, keepdim=True)
                rms = fluctuation.float().square().mean(
                    dim=2, keepdim=True).add(1e-6).sqrt().to(fluctuation)
                temporal_shape = fluctuation / rms
                amplitude = self.member_amplitude_heads[member][
                    len(fluctuations)](feature)
                if amplitude.shape[-2:] != (height, width):
                    amplitude = F.interpolate(
                        amplitude, size=(height, width), mode="bilinear",
                        align_corners=False)
                amplitude = self.amplitude_limit * torch.tanh(
                    amplitude / self.amplitude_limit)
                fluctuation = temporal_shape * amplitude.unsqueeze(2)
            fluctuations.append(fluctuation)
        mean_delta = torch.stack(means).mean(dim=0).permute(
            0, 2, 3, 1).unsqueeze(1)
        fluctuation = torch.stack(fluctuations).mean(dim=0)
        fluctuation = fluctuation - fluctuation.mean(dim=2, keepdim=True)
        fluctuation = fluctuation.permute(0, 2, 3, 4, 1)
        residual = mean_delta + fluctuation
        return persistence.unsqueeze(1) + residual


class IsoenergeticTemporalTransport(nn.Module):
    """Redistribute a point forecast in time without changing its energy.

    The strong flattened point heads remain the primary predictor.  This
    module predicts a small, spatially varying displacement on the explicit
    forecast-time axis, linearly samples the existing u/v trajectory, and
    restores its original temporal mean and RMS at every pixel/channel.  Its
    output projection is exactly zero at construction, so the untrained
    transport is a bitwise identity while the straight-through branch keeps
    the displacement head trainable.
    """

    def __init__(
            self, feature_channels: int, hidden_channels: int,
            out_steps: int, blocks: int, temporal_kernel: int, groups: int,
            max_displacement: float):
        super().__init__()
        self.out_steps = int(out_steps)
        self.max_displacement = float(max_displacement)
        self.feature_projection = nn.Conv2d(
            feature_channels, hidden_channels, 1)
        self.trajectory_projection = nn.Conv3d(2, hidden_channels, 1)
        position = torch.arange(out_steps, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, hidden_channels, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10_000.0)) / hidden_channels))
        embedding = torch.zeros(out_steps, hidden_channels)
        embedding[:, 0::2] = torch.sin(position * frequency)
        embedding[:, 1::2] = torch.cos(
            position * frequency[:embedding[:, 1::2].shape[1]])
        self.horizon_embedding = nn.Parameter(
            embedding.transpose(0, 1).reshape(
                1, hidden_channels, out_steps, 1, 1))
        self.blocks = nn.Sequential(*(
            FactorizedTemporalPointBlock(
                hidden_channels, temporal_kernel, groups)
            for _ in range(blocks)))
        self.output = nn.Conv3d(hidden_channels, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _linear_time_warp(
            fluctuation: torch.Tensor,
            displacement: torch.Tensor) -> torch.Tensor:
        """Sample ``fluctuation`` at t + displacement with edge clamping."""
        if fluctuation.shape != displacement.shape:
            raise ValueError("temporal transport tensors must be aligned")
        if fluctuation.ndim != 5 or fluctuation.shape[-1] != 2:
            raise ValueError("temporal transport requires [B,T,H,W,2]")
        steps = fluctuation.shape[1]
        values = fluctuation.permute(0, 2, 3, 4, 1)
        delta = displacement.permute(0, 2, 3, 4, 1)
        timeline = torch.arange(
            steps, dtype=delta.dtype, device=delta.device).view(
                1, 1, 1, 1, steps)
        coordinate = (timeline + delta).clamp(0.0, float(steps - 1))
        lower = coordinate.floor()
        upper = (lower + 1.0).clamp_max(float(steps - 1))
        weight = coordinate - lower
        low = torch.gather(values, -1, lower.long())
        high = torch.gather(values, -1, upper.long())
        warped = low + (high - low) * weight
        return warped.permute(0, 4, 1, 2, 3)

    @staticmethod
    def _restore_mean_and_rms(
            warped: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Restore source temporal mean and RMS in float32."""
        source32 = source.float()
        warped32 = warped.float()
        source_mean = source32.mean(dim=1, keepdim=True)
        source_fluctuation = source32 - source_mean
        candidate = warped32 - warped32.mean(dim=1, keepdim=True)
        source_energy = source_fluctuation.square().mean(dim=1, keepdim=True)
        candidate_energy = candidate.square().mean(dim=1, keepdim=True)
        # Zero fluctuation is the exact update-0 state of the legacy heads.
        # Mask both operands *before* division: wrapping a computed 0/0-ish
        # ratio in torch.where still leaves a non-finite inactive derivative.
        epsilon = 1e-12
        active = (
            (source_energy > epsilon) & (candidate_energy > epsilon))
        safe_source = torch.where(
            active, source_energy, torch.ones_like(source_energy))
        safe_candidate = torch.where(
            active, candidate_energy, torch.ones_like(candidate_energy))
        scale = torch.sqrt(safe_source / safe_candidate)
        restored_fluctuation = torch.where(
            active, candidate * scale, source_fluctuation)
        return (source_mean + restored_fluctuation).to(source)

    def forward(
            self, prediction: torch.Tensor,
            finest_feature: torch.Tensor) -> torch.Tensor:
        if (prediction.ndim != 5 or prediction.shape[1] != self.out_steps
                or prediction.shape[-1] < 2):
            raise ValueError(
                "temporal transport prediction contract changed")
        uv = prediction[..., :2]
        fluctuation = uv - uv.mean(dim=1, keepdim=True)
        context = fluctuation.detach().permute(0, 4, 1, 2, 3)
        feature = self.feature_projection(finest_feature)
        if feature.shape[-2:] != prediction.shape[2:4]:
            feature = F.interpolate(
                feature, size=prediction.shape[2:4], mode="bilinear",
                align_corners=False)
        value = (
            feature.unsqueeze(2)
            + self.trajectory_projection(context)
            + self.horizon_embedding)
        raw = self.output(self.blocks(F.silu(value))).permute(0, 2, 3, 4, 1)
        displacement = self.max_displacement * torch.tanh(raw)
        warped = self._linear_time_warp(fluctuation, displacement)
        candidate_uv = self._restore_mean_and_rms(warped, uv)
        delta = candidate_uv - uv
        bypass = raw.eq(0).flatten(1).all(dim=1).view(-1, 1, 1, 1, 1)
        transported_uv = _ExactIdentityAddSTE.apply(uv, delta, bypass)
        if prediction.shape[-1] == 2:
            return transported_uv
        return torch.cat((transported_uv, prediction[..., 2:]), dim=-1)


class SpatioTemporalQuantileRefiner(nn.Module):
    """Refine quantile logits with explicit horizon-local flow context.

    The existing UQ decoder compresses all forecast horizons into ordinary
    2-D channels. This branch restores the time axis, combines a shared
    multi-scale feature with horizon-specific point/disagreement fields, and
    convolves jointly over time and space. Its output projection is exactly
    zero at construction so old checkpoints retain bitwise-identical outputs
    until the branch is trained.
    """

    def __init__(
            self, fused_channels: int, hidden_channels: int,
            out_steps: int, quantiles: int, kernel: int,
            blocks: int, groups: int):
        super().__init__()
        self.out_steps = out_steps
        self.quantiles = quantiles
        self.shared_projection = nn.Conv2d(
            fused_channels, hidden_channels, 1)
        # Per horizon: point mean u/v, member std u/v, member range u/v.
        self.member_projection = nn.Conv3d(6, hidden_channels, 1)
        self.horizon_embedding = nn.Parameter(torch.zeros(
            1, hidden_channels, out_steps, 1, 1))
        self.blocks = nn.Sequential(*(
            SpatioTemporalResidualBlock(
                hidden_channels, kernel, groups)
            for _ in range(blocks)))
        self.output = nn.Conv3d(
            hidden_channels, 2 * quantiles, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
            self, fused: torch.Tensor,
            members: torch.Tensor) -> torch.Tensor:
        detached = members[..., :2].detach()
        mean = detached.mean(dim=1)
        std = detached.std(dim=1, unbiased=False)
        extent = detached.amax(dim=1) - detached.amin(dim=1)
        member_context = torch.cat((mean, std, extent), dim=-1).permute(
            0, 4, 1, 2, 3)
        value = (
            self.shared_projection(fused).unsqueeze(2)
            + self.member_projection(member_context)
            + self.horizon_embedding)
        value = self.blocks(F.silu(value))
        raw = self.output(value).reshape(
            members.shape[0], 2, self.quantiles,
            self.out_steps, members.shape[3], members.shape[4])
        return raw.permute(0, 3, 4, 5, 1, 2)


class EnsembleUNet2d(UNet2d):
    """One U-Net backbone, bootstrap point heads, and a quantile UQ decoder."""

    def __init__(self, config: EnsembleUNet2dConfig):
        config.validate()
        # Build the shared U-Net without its legacy interval heads, then restore
        # the full ensemble config for inherited persistence helpers.
        base_config = UNet2dConfig(**{
            key: getattr(config, key)
            for key in UNet2dConfig.__dataclass_fields__
        })
        super().__init__(replace(base_config, predict_uncertainty=False))
        self.config = config
        self.register_buffer('_conditional_feature', torch.zeros(1, config.base, 32, 64), persistent=False)
        decoder_channels = [
            config.base * 2 ** level
            for level in range(config.depth - 2, -1, -1)
        ]

        first_member = self.point_heads
        del self.point_heads
        self.temporal_point_decoder = (
            TemporalPointDecoder(
                decoder_channels, config.temporal_point_channels,
                config.out_steps, config.out_channels,
                config.ensemble_members, config.temporal_point_blocks,
                config.temporal_point_kernel, config.groups,
                config.temporal_point_amplitude_factorization,
                config.temporal_point_amplitude_limit)
            if config.temporal_point_decoder else None)
        if self.temporal_point_decoder is None:
            self.member_point_heads = nn.ModuleList([first_member])
            point_channels = config.out_channels * (
                config.out_steps + 1
                if config.mean_fluctuation else config.out_steps)
            for _ in range(1, config.ensemble_members):
                heads = nn.ModuleList(
                    nn.Conv2d(width, point_channels, 1)
                    for width in decoder_channels)
                for head in heads:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
                self.member_point_heads.append(heads)
        else:
            # Keep the public module name for code that enumerates point heads;
            # temporal-point checkpoints store their projections under the
            # explicit temporal_point_decoder namespace instead.
            self.member_point_heads = nn.ModuleList()

        self.member_feature_adapters = (
            nn.ModuleList(
                nn.ModuleList(
                    MemberFeatureAdapter(
                        width, config.member_adapter_channels, config.groups)
                    for width in decoder_channels)
                for _ in range(config.ensemble_members))
            if config.member_adapter_channels else None)
        self.mean_profile_refiner = (
            nn.Sequential(
                nn.Conv2d(
                    config.base, config.mean_profile_channels, 1),
                _group_norm(config.groups, config.mean_profile_channels),
                nn.SiLU(),
                nn.Conv2d(
                    config.mean_profile_channels,
                    config.mean_profile_channels, (1, 9),
                    padding=(0, 4), groups=config.mean_profile_channels),
                _group_norm(config.groups, config.mean_profile_channels),
                nn.SiLU(),
                nn.Conv2d(
                    config.mean_profile_channels,
                    config.mean_profile_channels, (9, 1),
                    padding=(4, 0), groups=config.mean_profile_channels),
                _group_norm(config.groups, config.mean_profile_channels),
                nn.SiLU(),
                nn.Conv2d(config.mean_profile_channels, 2, 1),
            )
            if config.mean_profile_channels else None)
        if self.mean_profile_refiner is not None:
            nn.init.zeros_(self.mean_profile_refiner[-1].weight)
            nn.init.zeros_(self.mean_profile_refiner[-1].bias)
        self.energy_controller = (
            (HybridMultiScaleHistoryEnergyController(
                decoder_channels, config.physical_energy_history_steps,
                config.physical_energy_hidden_channels, config.groups)
             if (config.physical_energy_history_conditioned
                 and config.physical_energy_multiscale) else
             (HistoryConditionedPhysicalEnergyController(
                config.physical_energy_history_steps,
                config.physical_energy_hidden_channels, config.groups)
             if config.physical_energy_history_conditioned else
             (MultiScalePhysicalEnergyController(
                decoder_channels, config.physical_energy_hidden_channels,
                config.groups)
             if config.physical_energy_multiscale else
             (HorizonLocalPhysicalEnergyController(
                 config.base, config.physical_energy_hidden_channels,
                 config.out_steps, config.groups)
              if config.physical_energy_horizon_local else
              PhysicalEnergyController(
                  config.base, config.physical_energy_hidden_channels,
                  config.groups)))))
            if config.physical_energy_controller else None)
        self.phase_delta = (
            PhaseConditionedLocalDelta(
                config.in_steps, config.out_steps,
                config.phase_delta_hidden_channels,
                config.phase_delta_steps, config.groups)
            if config.phase_delta_hidden_channels else None)
        self.temporal_transport = (
            IsoenergeticTemporalTransport(
                config.base, config.temporal_transport_channels,
                config.out_steps, config.temporal_transport_blocks,
                config.temporal_transport_kernel, config.groups,
                config.temporal_transport_max_displacement)
            if config.isoenergetic_temporal_transport else None)

        self.uq_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, config.uq_hidden_channels, 1),
                _group_norm(config.groups, config.uq_hidden_channels),
                nn.SiLU(),
                ResidualConvBlock(
                    config.uq_hidden_channels,
                    config.uq_hidden_channels,
                    config.groups),
            )
            for width in decoder_channels)
        disagreement_channels = 4 * config.out_steps
        fusion_channels = (
            len(decoder_channels) * config.uq_hidden_channels
            + disagreement_channels
            + (70 if config.uq_flow_context else 0))
        self.uq_fusion = nn.Sequential(
            nn.Conv2d(
                fusion_channels, config.uq_hidden_channels, 3, padding=1),
            ResidualConvBlock(
                config.uq_hidden_channels,
                config.uq_hidden_channels,
                config.groups),
            ResidualConvBlock(
                config.uq_hidden_channels,
                config.uq_hidden_channels,
                config.groups),
        )
        quantile_channels = (
            config.out_steps * 2 * len(config.quantile_levels))
        self.quantile_head = nn.Conv2d(
            config.uq_hidden_channels, quantile_channels, 1)
        nn.init.zeros_(self.quantile_head.weight)
        bias = torch.full(
            (config.out_steps, 2, len(config.quantile_levels)),
            float(config.quantile_gap_bias))
        bias[..., len(config.quantile_levels) // 2] = 0.0
        with torch.no_grad():
            self.quantile_head.bias.copy_(bias.reshape(-1))
        middle = len(config.quantile_levels) // 2
        self.interval_candidate_pairs = tuple(
            (lower_index, upper_index)
            for lower_index in range(middle + 1)
            for upper_index in range(
                max(middle, lower_index + 1),
                len(config.quantile_levels)))
        self.coverage_calibration_head = None
        if config.coverage_calibration:
            calibration_outputs = (
                len(self.interval_candidate_pairs)
                if config.coverage_calibration_mode == "independent"
                else 2)
            output = nn.Conv2d(
                config.uq_hidden_channels,
                config.out_steps * 2 * calibration_outputs,
                config.coverage_calibration_kernel,
                padding=config.coverage_calibration_kernel // 2)
            self.coverage_calibration_head = (
                nn.Sequential(
                    *(ResidualConvBlock(
                        config.uq_hidden_channels,
                        config.uq_hidden_channels, config.groups)
                      for _ in range(config.coverage_calibration_blocks)),
                    output)
                if config.coverage_calibration_blocks else output)
            final = (
                self.coverage_calibration_head[-1]
                if config.coverage_calibration_blocks
                else self.coverage_calibration_head)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.utility_refinement_head = None
        if config.utility_refinement:
            output = nn.Conv2d(
                config.uq_hidden_channels, config.out_steps * 2 * 2,
                config.utility_refinement_kernel,
                padding=config.utility_refinement_kernel // 2)
            self.utility_refinement_head = (
                nn.Sequential(
                    *(ResidualConvBlock(
                        config.uq_hidden_channels,
                        config.uq_hidden_channels, config.groups)
                      for _ in range(config.utility_refinement_blocks)),
                    output)
                if config.utility_refinement_blocks else output)
        if self.utility_refinement_head is not None:
            output = (
                self.utility_refinement_head[-1]
                if config.utility_refinement_blocks
                else self.utility_refinement_head)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        self.temporal_uq_refiner = (
            SpatioTemporalQuantileRefiner(
                config.uq_hidden_channels,
                config.temporal_uq_channels,
                config.out_steps,
                len(config.quantile_levels),
                config.temporal_uq_kernel,
                config.temporal_uq_blocks,
                config.groups)
            if config.temporal_uq_refinement else None)

    def architecture_config(self) -> dict:
        return {"architecture": "ensemble_unet", **asdict(self.config)}

    def _member_prediction(
            self, x: torch.Tensor, features: list[torch.Tensor],
            heads: nn.ModuleList) -> torch.Tensor:
        batch, _, height, width, _ = x.shape
        raw = self._multi_scale(features, heads, (height, width))
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

    def _members_from_features(
            self, x: torch.Tensor,
            features: list[torch.Tensor]) -> torch.Tensor:
        members = []
        if self.temporal_point_decoder is not None:
            persistence = self._persistence(x)
            shared_temporal_features = (
                self.temporal_point_decoder.temporal_features(features)
                if self.member_feature_adapters is None else None)
            for index in range(self.config.ensemble_members):
                member_features = features
                if self.member_feature_adapters is not None:
                    member_features = [
                        adapter(feature) for adapter, feature in zip(
                            self.member_feature_adapters[index], features)]
                temporal_features = (
                    shared_temporal_features
                    if shared_temporal_features is not None
                    else self.temporal_point_decoder.temporal_features(
                        member_features))
                members.append(
                    self.temporal_point_decoder.member_prediction(
                        persistence, member_features,
                        temporal_features, index))
        else:
            for index, heads in enumerate(self.member_point_heads):
                member_features = features
                if self.member_feature_adapters is not None:
                    member_features = [
                        adapter(feature) for adapter, feature in zip(
                            self.member_feature_adapters[index], features)]
                members.append(self._member_prediction(
                    x, member_features, heads))
        stacked = torch.stack(members, dim=1)
        if self.mean_profile_refiner is not None:
            mean_correction = self.mean_profile_refiner(
                features[-1]).permute(0, 2, 3, 1)
            if self.config.out_channels > 2:
                mean_correction = F.pad(
                    mean_correction,
                    (0, self.config.out_channels - 2))
            stacked = (
                stacked
                + mean_correction[:, None, None])
        if self.phase_delta is not None:
            phase_delta = self.phase_delta(x)
            if self.config.out_channels > 2:
                phase_delta = F.pad(
                    phase_delta, (0, self.config.out_channels - 2))
            stacked = stacked + phase_delta[:, None]
        return stacked

    def _energy_controlled_prediction(
            self, prediction: torch.Tensor,
            features: list[torch.Tensor],
            history: torch.Tensor | None = None) -> torch.Tensor:
        """Scale only physical u/v time fluctuations by exp(delta_logK/2).

        ``delta_logK`` is sample- and pixel-conditional but constant across the
        forecast horizon.  It is predicted only from the finest decoder
        feature.  The residual ``expm1`` form makes a zero controller an exact
        no-op while preserving temporal means mathematically.  Pressure and
        any later output channels bypass the controller bitwise.
        """
        if self.energy_controller is None:
            return prediction
        if prediction.ndim != 5 or prediction.shape[-1] < 2:
            raise ValueError(
                "prediction must have shape (B,T,H,W,C) with C >= 2")
        raw_delta = self._physical_energy_delta(features, history)
        if raw_delta.shape[-2:] != prediction.shape[2:4]:
            raw_delta = F.interpolate(
                raw_delta, size=prediction.shape[2:4], mode="bilinear",
                align_corners=False)
        delta_logk = (
            self.config.physical_energy_log_limit * torch.tanh(raw_delta))
        uv = prediction[..., :2]
        uv_mean = uv.mean(dim=1, keepdim=True)
        fluctuation = uv - uv_mean
        if self.config.physical_energy_horizon_local:
            if raw_delta.shape[1] != prediction.shape[1]:
                raise ValueError("horizon-local energy length changed")
            relative_amplitude = torch.expm1(
                0.5 * delta_logk).permute(0, 1, 2, 3).unsqueeze(-1)
            candidate = fluctuation * (1.0 + relative_amplitude)
            candidate = candidate - candidate.mean(dim=1, keepdim=True)
            delta = candidate - fluctuation
            bypass = raw_delta.eq(0).unsqueeze(-1)
            controlled_uv = _ExactIdentityAddSTE.apply(uv, delta, bypass)
        else:
            relative_amplitude = torch.expm1(
                0.5 * delta_logk).permute(0, 2, 3, 1).unsqueeze(1)
            controlled_uv = uv + fluctuation * relative_amplitude
        if prediction.shape[-1] == 2:
            return controlled_uv
        return torch.cat((controlled_uv, prediction[..., 2:]), dim=-1)

    def _energy_controlled_members(
            self, members: torch.Tensor,
            features: list[torch.Tensor],
            history: torch.Tensor | None = None) -> torch.Tensor:
        """Apply one spatial energy state consistently to every member.

        Each member keeps its own temporal mean.  A zero controller takes an
        exact identity branch (including signed zero and other IEEE payloads),
        while the straight-through backward still trains the zero-initialized
        controller projection.
        """
        if self.energy_controller is None:
            return members
        if members.ndim != 6 or members.shape[-1] < 2:
            raise ValueError(
                "members must have shape (B,M,T,H,W,C) with C >= 2")
        raw_delta = self._physical_energy_delta(features, history)
        if raw_delta.shape[-2:] != members.shape[3:5]:
            raw_delta = F.interpolate(
                raw_delta, size=members.shape[3:5], mode="bilinear",
                align_corners=False)
        delta_logk = (
            self.config.physical_energy_log_limit * torch.tanh(raw_delta))
        uv = members[..., :2]
        uv_mean = uv.mean(dim=2, keepdim=True)
        fluctuation = uv - uv_mean
        if self.config.physical_energy_horizon_local:
            if raw_delta.shape[1] != members.shape[2]:
                raise ValueError("horizon-local member energy length changed")
            relative_amplitude = torch.expm1(
                0.5 * delta_logk)[:, None, ..., None]
            candidate = fluctuation * (1.0 + relative_amplitude)
            candidate = candidate - candidate.mean(dim=2, keepdim=True)
            delta = candidate - fluctuation
            bypass = raw_delta.eq(0)[:, None, ..., None]
        else:
            relative_amplitude = torch.expm1(0.5 * delta_logk).permute(
                0, 2, 3, 1)[:, None, None]
            delta = fluctuation * relative_amplitude
            bypass = raw_delta.eq(0).permute(0, 2, 3, 1)[:, None, None]
        controlled_uv = _ExactIdentityAddSTE.apply(uv, delta, bypass)
        if members.shape[-1] == 2:
            return controlled_uv
        return torch.cat((controlled_uv, members[..., 2:]), dim=-1)

    def _physical_energy_delta(
            self, features: list[torch.Tensor],
            history: torch.Tensor | None = None) -> torch.Tensor:
        if self.energy_controller is None:
            raise RuntimeError("physical energy controller is disabled")
        if self.config.physical_energy_history_conditioned:
            if history is None:
                raise ValueError("history-conditioned energy requires history")
            if self.config.physical_energy_multiscale:
                return self.energy_controller(features, history)
            return self.energy_controller(history)
        return self.energy_controller(
            features if self.config.physical_energy_multiscale
            else features[-1])

    @staticmethod
    def _members_with_shared_uv_mean(
            members: torch.Tensor) -> torch.Tensor:
        """Share each member's u/v temporal mean but preserve its fluctuation.

        Pressure and any later channels are returned by identity.  The
        transformation has no parameters and keeps gradients through the
        cross-member mean, so member losses jointly train the shared u/v mean
        while each member retains its own zero-time-mean role.
        """
        if members.ndim != 6 or members.shape[-1] < 2:
            raise ValueError(
                "members must have shape (B,M,T,H,W,C) with C >= 2")
        uv = members[..., :2]
        member_mean = uv.mean(dim=2, keepdim=True)
        shared_mean = member_mean.mean(dim=1, keepdim=True)
        shared_uv = uv - member_mean + shared_mean
        if members.shape[-1] == 2:
            return shared_uv
        return torch.cat((shared_uv, members[..., 2:]), dim=-1)

    @staticmethod
    def _flow_context(
            x: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Direct high-resolution motion cues for residual localization."""
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

    def _fused_uq_features(
            self, features: list[torch.Tensor],
            members: torch.Tensor, x: torch.Tensor,
            prediction: torch.Tensor | None = None, *,
            detach_members: bool = True) -> torch.Tensor:
        height, width = members.shape[3:5]
        projected = []
        for feature, projection in zip(features, self.uq_projections):
            value = projection(feature)
            if value.shape[-2:] != (height, width):
                value = F.interpolate(
                    value, size=(height, width), mode="bilinear",
                    align_corners=False)
            projected.append(value)

        member_uv = members[..., :2]
        if detach_members:
            member_uv = member_uv.detach()
        member_std = member_uv.std(dim=1, unbiased=False)
        member_range = member_uv.amax(dim=1) - member_uv.amin(dim=1)
        if self.config.disagreement_scale_fraction:
            target_std = self.target_std[:2].view(1, 1, 1, 1, 2)
            reference = self._t2_base_widths.to(member_uv).view(
                    1, 1, 1, 1, 2)
            reference = (
                reference * self.config.disagreement_scale_fraction)
            member_std = torch.log1p(
                member_std * target_std / reference)
            member_range = torch.log1p(
                member_range * target_std / reference)
        disagreement = torch.cat((member_std, member_range), dim=-1)
        fusion_inputs = [*projected, _flatten_forecast(disagreement)]
        if self.config.uq_flow_context:
            fusion_inputs.append(self._flow_context(
                x, (member_uv.mean(dim=1)
                    if prediction is None else prediction)))
        return self.uq_fusion(torch.cat(fusion_inputs, dim=1))

    def _quantiles_from_fused(
            self, fused: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
        batch = members.shape[0]
        height, width = members.shape[3:5]
        raw = self.quantile_head(fused).reshape(
            batch, self.config.out_steps, 2,
            len(self.config.quantile_levels), height, width
        ).permute(0, 1, 4, 5, 2, 3)
        if self.temporal_uq_refiner is not None:
            correction = self.temporal_uq_refiner(fused, members)
            raw = raw + self.config.temporal_uq_logit_limit * torch.tanh(
                correction)

        middle = len(self.config.quantile_levels) // 2
        center = raw[..., middle:middle + 1]
        lower_gaps = F.softplus(raw[..., :middle])
        lower_distance = _short_gap_cumsum(
            lower_gaps.flip(dims=(-1,))).flip(dims=(-1,))
        lower = center - lower_distance
        upper_gaps = F.softplus(raw[..., middle + 1:])
        upper = center + _short_gap_cumsum(upper_gaps)
        return torch.cat((lower, center, upper), dim=-1)

    def _quantiles_from_features(
            self, features: list[torch.Tensor],
            members: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        fused = self._fused_uq_features(features, members, x)
        return self._quantiles_from_fused(fused, members)

    def _refine_utility_interval(
            self, lower: torch.Tensor, upper: torch.Tensor,
            fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.utility_refinement_head is None:
            return lower, upper
        batch, steps, height, width, channels = lower.shape
        raw = self.utility_refinement_head(fused).reshape(
            batch, steps, channels, 2, height, width
        ).permute(0, 1, 4, 5, 2, 3)
        log_scale = (
            self.config.utility_refinement_log_limit * torch.tanh(raw))
        return (
            lower * torch.exp(log_scale[..., 0]),
            upper * torch.exp(log_scale[..., 1]),
        )

    @staticmethod
    def _ordered_interval_extents(
            lower: torch.Tensor,
            upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Order residual endpoints without discarding interval asymmetry.

        Residual quantiles intentionally allow signed extents. Independent
        positive rescaling of both sides can therefore very rarely move the
        residual lower endpoint ``-lower`` above the upper endpoint
        ``upper``. Swapping those endpoints maps ``(lower, upper)`` to
        ``(-upper, -lower)`` and guarantees a non-negative total width while
        leaving every already-valid element exactly unchanged.
        """
        invalid = lower + upper < 0.0
        return (
            torch.where(invalid, -upper, lower),
            torch.where(invalid, -lower, upper),
        )

    def _interval_candidates(
            self, quantiles: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor]:
        levels = self._t2_quantile_levels.to(quantiles)
        lower_candidates = []
        upper_candidates = []
        valid_candidates = []
        masses = []
        for lower_index, upper_index in self.interval_candidate_pairs:
            lower_quantile = quantiles[..., lower_index]
            upper_quantile = quantiles[..., upper_index]
            if self.config.residual_quantile_intervals:
                # Extents may be signed: prediction - lower_extent is the
                # learned lower residual quantile, and prediction +
                # upper_extent is the learned upper residual quantile.
                # Their sum remains the strictly positive interval width.
                lower_width = -lower_quantile
                upper_width = upper_quantile
                valid = torch.ones_like(lower_width, dtype=torch.bool)
            else:
                lower_width = (-lower_quantile).clamp_min(1e-6)
                upper_width = upper_quantile.clamp_min(1e-6)
                valid = (
                    (lower_quantile <= 0.0) & (upper_quantile >= 0.0))
            lower_candidates.append(lower_width)
            upper_candidates.append(upper_width)
            valid_candidates.append(valid)
            masses.append(levels[upper_index] - levels[lower_index])
        return (
            torch.stack(lower_candidates, dim=-1),
            torch.stack(upper_candidates, dim=-1),
            torch.stack(valid_candidates, dim=-1),
            torch.stack(masses))

    def _coverage_logits(self, fused: torch.Tensor) -> torch.Tensor | None:
        if self.coverage_calibration_head is None:
            return None
        batch, _, height, width = fused.shape
        raw = self.coverage_calibration_head(fused)
        levels = self._t2_quantile_levels.to(raw)
        masses = torch.stack([
            levels[upper] - levels[lower]
            for lower, upper in self.interval_candidate_pairs])
        base_logits = torch.logit(masses.clamp(1e-5, 1.0 - 1e-5))
        if self.config.coverage_calibration_mode == "temperature":
            correction = raw.reshape(
                batch, self.config.out_steps, 2, 2, height, width
            ).permute(0, 1, 4, 5, 2, 3)
            bias = (
                self.config.coverage_calibration_logit_limit
                * torch.tanh(correction[..., 0:1]))
            slope = torch.exp(
                self.config.coverage_calibration_temperature_limit
                * torch.tanh(correction[..., 1:2]))
            return base_logits * slope + bias
        correction = raw.reshape(
            batch, self.config.out_steps, 2,
            len(self.interval_candidate_pairs), height, width
        ).permute(0, 1, 4, 5, 2, 3)
        return (
            base_logits
            + self.config.coverage_calibration_logit_limit
            * torch.tanh(correction))

    def _utility_interval(
            self, quantiles: torch.Tensor,
            coverage_logits: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        target_std = self.target_std[:2].view(1, 1, 1, 1, 2, 1)
        lower_stack, upper_stack, valid_stack, masses = (
            self._interval_candidates(quantiles))
        probability = (
            masses.view(1, 1, 1, 1, 1, -1)
            if coverage_logits is None
            else torch.sigmoid(coverage_logits))
        physical_width = (lower_stack + upper_stack) * target_std
        utility_stack = probability * torch.exp(
            -physical_width / SPS_SIGMA)
        utility_stack = torch.where(
            valid_stack, utility_stack, utility_stack.new_full((), -1.0))
        selected = utility_stack.argmax(dim=-1, keepdim=True)
        lower = lower_stack.gather(-1, selected).squeeze(-1)
        upper = upper_stack.gather(-1, selected).squeeze(-1)
        return lower, upper

    def forward_uq_training(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor, torch.Tensor, list[torch.Tensor],
                       torch.Tensor | None]:
        features = self._decode(x)
        raw_members = self._members_from_features(x, features)
        if self.config.physical_energy_member_sync:
            raw_members = self._energy_controlled_members(
                raw_members, features, x)
            prediction = raw_members.mean(dim=1)
        else:
            prediction = self._energy_controlled_prediction(
                raw_members.mean(dim=1), features, x)
        if self.temporal_transport is not None:
            prediction = self.temporal_transport(prediction, features[-1])
        members = (
            self._members_with_shared_uv_mean(raw_members)
            if self.config.shared_member_uv_mean else raw_members)
        fused = self._fused_uq_features(
            features, members, x,
            prediction=(
                prediction if self.config.shared_member_uv_mean else None))
        quantiles = self._quantiles_from_fused(fused, members)
        coverage_logits = self._coverage_logits(fused)
        lower, upper = self._utility_interval(
            quantiles, coverage_logits)
        lower, upper = self._refine_utility_interval(
            lower, upper, fused)
        lower, upper = self._ordered_interval_extents(lower, upper)
        return (prediction, members, quantiles, lower, upper, features,
                coverage_logits)

    def forward_quantiles_with_features(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        return self.forward_uq_training(x)[:6]

    def forward_quantiles(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor, torch.Tensor]:
        values = self.forward_quantiles_with_features(x)
        return values[:5]

    def forward_distribution_with_features(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       list[torch.Tensor]]:
        prediction, _members, _quantiles, lower, upper, features = (
            self.forward_quantiles_with_features(x))
        return prediction, lower, upper, features

    def forward_distribution(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction, lower, upper, _features = (
            self.forward_distribution_with_features(x))
        self._conditional_feature.copy_(_features[-1])
        return prediction, lower, upper

    def forward_with_features(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self._decode(x)
        members = self._members_from_features(x, features)
        if self.config.physical_energy_member_sync:
            members = self._energy_controlled_members(members, features, x)
            prediction = members.mean(dim=1)
        else:
            prediction = self._energy_controlled_prediction(
                members.mean(dim=1), features, x)
        if self.temporal_transport is not None:
            prediction = self.temporal_transport(prediction, features[-1])
        return prediction, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, _features = self.forward_with_features(x)
        return prediction


def build_ensemble_unet2d(**overrides) -> EnsembleUNet2d:
    return EnsembleUNet2d(EnsembleUNet2dConfig(**overrides))
