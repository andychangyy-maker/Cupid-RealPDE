"""Matched native-feature isoenergetic phase adapters for a frozen B210.

Both arms reuse the exact 7,778-parameter trunk of
``IsoenergeticTemporalTransport``.  They see the same detached B210 finest
decoder feature and point forecast and execute both deterministic output
maps.  The only experimental variable is which map supplies the correction:

``legacy_time_warp_control``
    Interpret the two raw channels as a bounded displacement on forecast
    time and linearly resample the B210 fluctuation.

``additive_direction_candidate``
    Interpret the same raw channels as a bounded update to the normalized
    zero-mean forecast direction.

Both maps restore the B210 temporal mean and RMS per pixel and velocity
channel.  The output projection is inherited from the legacy module and is
exactly zero initialized, so both arms are bitwise B210 identities at update
zero while retaining a live straight-through gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .ensemble_unet2d import (
    EnsembleUNet2d,
    IsoenergeticTemporalTransport,
    _ExactIdentityAddSTE,
)


LEGACY_TIME_WARP_CONTROL = "legacy_time_warp_control"
ADDITIVE_DIRECTION_CANDIDATE = "additive_direction_candidate"
MODES = (LEGACY_TIME_WARP_CONTROL, ADDITIVE_DIRECTION_CANDIDATE)


@dataclass(frozen=True)
class NativeFeaturePhaseConfig:
    feature_channels: int = 210
    hidden_channels: int = 24
    out_steps: int = 20
    blocks: int = 2
    temporal_kernel: int = 5
    groups: int = 4
    update_limit: float = 1.5
    projection_epsilon: float = 1.0e-12

    def validate(self) -> None:
        dimensions = (
            self.feature_channels, self.hidden_channels, self.out_steps,
            self.blocks, self.temporal_kernel, self.groups)
        if any(value <= 0 for value in dimensions):
            raise ValueError("native-feature phase dimensions must be positive")
        if self.hidden_channels % self.groups:
            raise ValueError("hidden channels must be divisible by groups")
        if self.temporal_kernel % 2 == 0:
            raise ValueError("temporal kernel must be odd")
        if self.update_limit <= 0.0 or self.projection_epsilon <= 0.0:
            raise ValueError("phase limits must be positive")


@dataclass
class NativeFeaturePhaseFields:
    prediction: torch.Tensor
    base_prediction: torch.Tensor
    raw: torch.Tensor
    bounded_field: torch.Tensor
    legacy_warp_uv: torch.Tensor
    additive_direction_uv: torch.Tensor
    source_temporal_mean: torch.Tensor
    source_temporal_rms: torch.Tensor


class NativeFeatureIsoenergeticPhase(IsoenergeticTemporalTransport):
    """The legacy transport trunk with two matched phase coordinates."""

    def __init__(
            self, config: NativeFeaturePhaseConfig | None = None, *,
            mode: str = ADDITIVE_DIRECTION_CANDIDATE) -> None:
        self.phase_config = (
            NativeFeaturePhaseConfig() if config is None else config)
        self.phase_config.validate()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        cfg = self.phase_config
        super().__init__(
            cfg.feature_channels, cfg.hidden_channels, cfg.out_steps,
            cfg.blocks, cfg.temporal_kernel, cfg.groups, cfg.update_limit)

    def architecture_config(self) -> dict[str, object]:
        cfg = self.phase_config
        return {
            "architecture": "native_feature_isoenergetic_phase",
            "mode": self.mode,
            "feature_channels": cfg.feature_channels,
            "hidden_channels": cfg.hidden_channels,
            "out_steps": cfg.out_steps,
            "blocks": cfg.blocks,
            "temporal_kernel": cfg.temporal_kernel,
            "groups": cfg.groups,
            "update_limit": cfg.update_limit,
            "projection_epsilon": cfg.projection_epsilon,
            "trunk": "IsoenergeticTemporalTransport",
        }

    def execution_contract(self) -> dict[str, object]:
        return {
            "base_feature": "detached finest B210 decoder feature",
            "base_forecast": "detached B210 point forecast",
            "shared_tensors": True,
            "shared_trunk_conv_macs": True,
            "both_output_maps_executed": True,
            "control": "bounded linear forecast-time warp",
            "candidate": "bounded additive normalized-direction update",
            "temporal_mean_restored": True,
            "temporal_rms_restored": True,
            "pressure_passthrough": "bitwise",
            "update_zero": "bitwise frozen-B210 identity",
        }

    def _raw_field(
            self, prediction: torch.Tensor,
            finest_feature: torch.Tensor) -> torch.Tensor:
        if (prediction.ndim != 5
                or prediction.shape[1] != self.phase_config.out_steps
                or prediction.shape[-1] < 2):
            raise ValueError("prediction must be [B,20,H,W,C>=2]")
        if finest_feature.ndim != 4:
            raise ValueError("finest feature must be [B,C,H,W]")
        uv = prediction[..., :2]
        fluctuation = uv - uv.mean(dim=1, keepdim=True)
        context = fluctuation.detach().permute(0, 4, 1, 2, 3)
        feature = self.feature_projection(finest_feature.detach())
        if feature.shape[-2:] != prediction.shape[2:4]:
            feature = F.interpolate(
                feature, size=prediction.shape[2:4], mode="bilinear",
                align_corners=False)
        value = (
            feature.unsqueeze(2)
            + self.trajectory_projection(context)
            + self.horizon_embedding)
        return self.output(self.blocks(F.silu(value))).permute(0, 2, 3, 4, 1)

    def _source_geometry(
            self, source_uv: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = source_uv.float()
        mean = source.mean(dim=1, keepdim=True)
        fluctuation = source - mean
        energy = fluctuation.square().mean(dim=1, keepdim=True)
        active = energy > self.phase_config.projection_epsilon
        safe_energy = torch.where(active, energy, torch.ones_like(energy))
        rms = torch.where(active, safe_energy.sqrt(), torch.zeros_like(energy))
        safe_rms = torch.where(active, rms, torch.ones_like(rms))
        direction = torch.where(
            active, fluctuation / safe_rms, torch.zeros_like(fluctuation))
        return mean, rms, direction

    def forward_with_fields(
            self, prediction: torch.Tensor,
            finest_feature: torch.Tensor) -> NativeFeaturePhaseFields:
        raw = self._raw_field(prediction, finest_feature)
        bounded = self.phase_config.update_limit * torch.tanh(raw)
        source_uv = prediction[..., :2]
        source_mean, source_rms, source_direction = self._source_geometry(
            source_uv)

        # Execute both deterministic interpretations in both arms.  The
        # learned tensors, convolution MACs, feature inputs and eager operator
        # calls therefore match; ``mode`` selects only the returned map.
        warped = self._linear_time_warp(
            source_uv - source_uv.mean(dim=1, keepdim=True), bounded)
        legacy_uv = self._restore_mean_and_rms(warped, source_uv)

        direction_update = bounded.float()
        direction_update = direction_update - direction_update.mean(
            dim=1, keepdim=True)
        proposed_direction = source_direction + direction_update
        additive_uv = self._restore_mean_and_rms(
            proposed_direction, source_uv)

        selected = (
            legacy_uv if self.mode == LEGACY_TIME_WARP_CONTROL
            else additive_uv)
        delta = selected - source_uv
        bypass = raw.eq(0.0).flatten(1).all(dim=1).view(-1, 1, 1, 1, 1)
        corrected_uv = _ExactIdentityAddSTE.apply(source_uv, delta, bypass)
        corrected = (
            corrected_uv if prediction.shape[-1] == 2 else
            torch.cat((corrected_uv, prediction[..., 2:]), dim=-1))
        return NativeFeaturePhaseFields(
            prediction=corrected,
            base_prediction=prediction,
            raw=raw,
            bounded_field=bounded,
            legacy_warp_uv=legacy_uv,
            additive_direction_uv=additive_uv,
            source_temporal_mean=source_mean.to(source_uv),
            source_temporal_rms=source_rms.to(source_uv),
        )

    def forward(
            self, prediction: torch.Tensor,
            finest_feature: torch.Tensor) -> torch.Tensor:
        return self.forward_with_fields(prediction, finest_feature).prediction


class FrozenB210NativeFeaturePhase(nn.Module):
    """Run one frozen B210 pass and preserve its absolute interval endpoints."""

    def __init__(
            self, base_model: EnsembleUNet2d,
            phase: NativeFeatureIsoenergeticPhase) -> None:
        super().__init__()
        self.base_model = base_model.requires_grad_(False).eval()
        self.phase = phase

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        self.phase.train(mode)
        return self

    def forward_components(
            self, history: torch.Tensor,
            ) -> tuple[NativeFeaturePhaseFields, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base, lower, upper, features = (
                self.base_model.forward_distribution_with_features(history))
        fields = self.phase.forward_with_fields(
            base.detach(), features[-1].detach())
        point_delta = fields.prediction[..., :2] - base[..., :2]
        # EnsembleUNet2d returns normalized extents.  Adjusting them by the
        # point delta reconstructs the exact frozen absolute endpoints.
        return (
            fields,
            lower.detach() + point_delta,
            upper.detach() - point_delta,
        )

    def forward_distribution(
            self, history: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fields, lower, upper = self.forward_components(history)
        return fields.prediction, lower, upper

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_components(history)[0].prediction
