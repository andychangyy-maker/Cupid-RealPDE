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
)


SPS_SIGMA = 0.0563870259


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
        decoder_channels = [
            config.base * 2 ** level
            for level in range(config.depth - 2, -1, -1)
        ]

        first_member = self.point_heads
        del self.point_heads
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
        return stacked

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
            members: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        height, width = members.shape[3:5]
        projected = []
        for feature, projection in zip(features, self.uq_projections):
            value = projection(feature)
            if value.shape[-2:] != (height, width):
                value = F.interpolate(
                    value, size=(height, width), mode="bilinear",
                    align_corners=False)
            projected.append(value)

        detached_uv = members[..., :2].detach()
        member_std = detached_uv.std(dim=1, unbiased=False)
        member_range = detached_uv.amax(dim=1) - detached_uv.amin(dim=1)
        if self.config.disagreement_scale_fraction:
            target_std = self.target_std[:2].view(1, 1, 1, 1, 2)
            reference = detached_uv.new_tensor(
                self.config.uncertainty_base_widths).view(
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
                x, detached_uv.mean(dim=1)))
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
        lower_distance = torch.cumsum(
            lower_gaps.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))
        lower = center - lower_distance
        upper_gaps = F.softplus(raw[..., middle + 1:])
        upper = center + torch.cumsum(upper_gaps, dim=-1)
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
        levels = quantiles.new_tensor(self.config.quantile_levels)
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
        levels = raw.new_tensor(self.config.quantile_levels)
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
        members = self._members_from_features(x, features)
        prediction = members.mean(dim=1)
        fused = self._fused_uq_features(features, members, x)
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
        return prediction, lower, upper

    def forward_with_features(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self._decode(x)
        prediction = self._members_from_features(
            x, features).mean(dim=1)
        return prediction, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, _features = self.forward_with_features(x)
        return prediction


def build_ensemble_unet2d(**overrides) -> EnsembleUNet2d:
    return EnsembleUNet2d(EnsembleUNet2dConfig(**overrides))
