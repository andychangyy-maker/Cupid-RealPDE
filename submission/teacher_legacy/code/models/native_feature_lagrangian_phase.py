"""Low-resolution semi-Lagrangian phase correction for a frozen B210.

The controller predicts a spatial backtrace displacement for every modifiable
forecast horizon.  It reads only the current sample's observed history, the
frozen B210 point forecast, and its finest native decoder feature.  The flow is
predicted on a quarter-resolution grid and bilinearly expanded before
``grid_sample`` transports the B210 u/v field.

The transport coordinate is deliberately narrower than a free residual:

* the formal default modifies all twenty horizons; an optional two-step
  anchor remains available for controlled diagnostics;
* pressure is copied bitwise from B210;
Two output modes use the exact same learned trunk and state dictionary.
``raw_transport`` returns the spatial warp without temporal reprojection;
``isoenergetic_transport`` restores the source temporal mean and per-channel
centered RMS at every output pixel.

The zero-initialized flow head is an exact B210 identity through the same
straight-through branch used by the other phase adapters, while gradients
still traverse the differentiable spatial sampler on the first update.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .early_anchor_isoenergetic_phase import restore_temporal_mean_rms
from .ensemble_unet2d import EnsembleUNet2d, _ExactIdentityAddSTE


RAW_TRANSPORT = "raw_transport"
ISOENERGETIC_TRANSPORT = "isoenergetic_transport"
MODES = (RAW_TRANSPORT, ISOENERGETIC_TRANSPORT)


def _group_count(requested: int, channels: int) -> int:
    value = min(int(requested), int(channels))
    while channels % value:
        value -= 1
    return value


class DepthwiseSpatialResidual(nn.Module):
    """A moderate-capacity spatial block that is cheap on the 1/4 grid."""

    def __init__(
            self, channels: int, expansion: int, kernel: int,
            groups: int) -> None:
        super().__init__()
        expanded = channels * expansion
        self.norm = nn.GroupNorm(_group_count(groups, channels), channels)
        self.expand = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(
            expanded, expanded, kernel, padding=kernel // 2,
            groups=expanded)
        self.depthwise_norm = nn.GroupNorm(
            _group_count(groups, expanded), expanded)
        self.project = nn.Conv2d(expanded, channels, 1)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1.0e-2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.expand(F.silu(self.norm(value)))
        residual = self.depthwise(residual)
        residual = F.silu(self.depthwise_norm(residual))
        residual = self.project(residual)
        return value + residual * self.layer_scale.view(1, -1, 1, 1)


@dataclass(frozen=True)
class LagrangianPhaseConfig:
    feature_channels: int = 210
    history_steps: int = 6
    out_steps: int = 20
    anchor_steps: int = 0
    feature_width: int = 64
    history_width: int = 32
    phase_width: int = 32
    geometry_width: int = 16
    model_width: int = 96
    coarse_width: int = 128
    native_blocks: int = 1
    coarse_blocks: int = 3
    expansion: int = 2
    spatial_kernel: int = 5
    groups: int = 8
    control_downsample: int = 4
    max_displacement_pixels: float = 1.5
    projection_epsilon: float = 1.0e-12

    def validate(self) -> None:
        positive = (
            self.feature_channels, self.history_steps, self.out_steps,
            self.feature_width, self.history_width, self.phase_width,
            self.geometry_width, self.model_width, self.coarse_width,
            self.native_blocks, self.coarse_blocks, self.expansion,
            self.spatial_kernel, self.groups, self.control_downsample)
        if any(int(value) <= 0 for value in positive):
            raise ValueError("Lagrangian phase dimensions must be positive")
        if self.history_steps < 2:
            raise ValueError("history_steps must provide at least one difference")
        if not 0 <= self.anchor_steps <= self.out_steps - 2:
            raise ValueError("anchor_steps must leave at least two tail frames")
        if self.spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel must be odd")
        if self.control_downsample != 4:
            raise ValueError("the current trunk has a fixed 1/4 control grid")
        if (not math.isfinite(self.max_displacement_pixels)
                or self.max_displacement_pixels <= 0.0):
            raise ValueError("max_displacement_pixels must be positive")
        if (not math.isfinite(self.projection_epsilon)
                or self.projection_epsilon <= 0.0):
            raise ValueError("projection_epsilon must be positive")


@dataclass
class LagrangianPhaseFields:
    prediction: torch.Tensor
    base_prediction: torch.Tensor
    raw_control: torch.Tensor
    control_displacement: torch.Tensor
    displacement: torch.Tensor
    raw_transport_uv: torch.Tensor
    isoenergetic_transport_uv: torch.Tensor
    source_tail_mean: torch.Tensor
    source_tail_rms: torch.Tensor


def spatial_backtrace_warp(
        source: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``source`` at ``(x+dx,y+dy)`` in pixel units.

    ``source`` and ``displacement`` are ``[B,T,H,W,2]``.  Positive ``dx``
    samples from the right.  The operation is evaluated in FP32 with border
    padding and ``align_corners=True`` so one displacement unit is one cell.
    """
    if source.ndim != 5 or source.shape[-1] != 2:
        raise ValueError("source must be u/v [B,T,H,W,2]")
    if displacement.shape != source.shape:
        raise ValueError("displacement must align with source")
    batch, steps, height, width, channels = source.shape
    if height < 2 or width < 2:
        raise ValueError("spatial backtrace requires H,W >= 2")
    work = source.float()
    flow = displacement.float()
    y = torch.linspace(-1.0, 1.0, height, device=source.device)
    x = torch.linspace(-1.0, 1.0, width, device=source.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    identity_grid = torch.stack((grid_x, grid_y), dim=-1).view(
        1, 1, height, width, 2)
    pixel_scale = flow.new_tensor((
        2.0 / float(width - 1), 2.0 / float(height - 1)))
    sampling_grid = identity_grid + flow * pixel_scale
    images = work.permute(0, 1, 4, 2, 3).reshape(
        batch * steps, channels, height, width)
    sampling_grid = sampling_grid.reshape(
        batch * steps, height, width, 2)
    warped = F.grid_sample(
        images, sampling_grid, mode="bilinear", padding_mode="border",
        align_corners=True)
    return warped.reshape(
        batch, steps, channels, height, width).permute(0, 1, 3, 4, 2)


class NativeFeatureLagrangianPhase(nn.Module):
    """Predict and apply a smooth per-horizon spatial transport coordinate."""

    def __init__(
            self, config: LagrangianPhaseConfig | None = None, *,
            mode: str = RAW_TRANSPORT) -> None:
        super().__init__()
        self.phase_config = (
            LagrangianPhaseConfig() if config is None else config)
        self.phase_config.validate()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        cfg = self.phase_config
        tail_steps = cfg.out_steps - cfg.anchor_steps
        history_channels = 2 * (cfg.history_steps - 1) + 4

        self.feature_projection = nn.Conv2d(
            cfg.feature_channels, cfg.feature_width, 1)
        self.history_projection = nn.Conv2d(
            history_channels, cfg.history_width, 1)
        self.phase_projection = nn.Conv2d(
            2 * tail_steps, cfg.phase_width, 1)
        self.geometry_projection = nn.Conv2d(4, cfg.geometry_width, 1)
        fused_channels = (
            cfg.feature_width + cfg.history_width + cfg.phase_width
            + cfg.geometry_width)
        self.input_fusion = nn.Conv2d(
            fused_channels, cfg.model_width, 1)
        self.native = nn.Sequential(*(
            DepthwiseSpatialResidual(
                cfg.model_width, cfg.expansion, cfg.spatial_kernel, cfg.groups)
            for _ in range(cfg.native_blocks)))
        self.downsample = nn.Sequential(
            nn.Conv2d(
                cfg.model_width, cfg.model_width, 3, stride=2, padding=1,
                groups=cfg.model_width),
            nn.Conv2d(cfg.model_width, cfg.coarse_width, 1),
            nn.GroupNorm(
                _group_count(cfg.groups, cfg.coarse_width), cfg.coarse_width),
            nn.SiLU(),
            nn.Conv2d(
                cfg.coarse_width, cfg.coarse_width, 3, stride=2, padding=1,
                groups=cfg.coarse_width),
            nn.GroupNorm(
                _group_count(cfg.groups, cfg.coarse_width), cfg.coarse_width),
            nn.SiLU(),
        )
        self.coarse = nn.Sequential(*(
            DepthwiseSpatialResidual(
                cfg.coarse_width, cfg.expansion, cfg.spatial_kernel,
                cfg.groups)
            for _ in range(cfg.coarse_blocks)))
        self.flow_head = nn.Conv2d(cfg.coarse_width, 2 * tail_steps, 1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": "native_feature_lagrangian_phase",
            "mode": self.mode,
            **vars(self.phase_config),
            "flow_grid": "two_stride2_layers_then_bilinear_upsample",
            "transport": "FP32_border_grid_sample_of_uv_field",
        }

    def execution_contract(self) -> dict[str, object]:
        cfg = self.phase_config
        return {
            "inputs": (
                "current history, detached frozen-B210 finest feature, and "
                "detached frozen-B210 point forecast"),
            "future_target_or_metadata": False,
            "cross_sample_state": False,
            "low_resolution_flow": True,
            "flow_per_tail_horizon": True,
            "both_output_maps_executed": True,
            "selected_mode": self.mode,
            "max_displacement_pixels_per_component": (
                cfg.max_displacement_pixels),
            "anchor_steps": cfg.anchor_steps,
            "anchor_passthrough": (
                "bitwise" if cfg.anchor_steps else "disabled_full20"),
            "pressure_passthrough": "bitwise",
            "raw_transport_temporal_reprojection": False,
            "isoenergetic_temporal_mean_preserved": True,
            "isoenergetic_per_channel_temporal_rms_preserved": True,
            "isoenergetic_full_horizon_tke_preserved": True,
            "update_zero": "bitwise frozen-B210 identity with live flow gradient",
        }

    def _history_features(self, history: torch.Tensor) -> torch.Tensor:
        cfg = self.phase_config
        if (history.ndim != 5 or history.shape[1] < cfg.history_steps
                or history.shape[-1] < 2):
            raise ValueError(
                "history must be [B,T>=history_steps,H,W,C>=2]")
        velocity = history[:, -cfg.history_steps:, ..., :2].float()
        difference = velocity[:, 1:] - velocity[:, :-1]
        mean = velocity.mean(dim=1)
        spread = velocity.std(dim=1, unbiased=False)
        return torch.cat((
            difference.permute(0, 1, 4, 2, 3).flatten(1, 2),
            mean.permute(0, 3, 1, 2),
            spread.permute(0, 3, 1, 2),
        ), dim=1)

    def _source_geometry(
            self, source_tail: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = source_tail.float()
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

    @staticmethod
    def _resize_to(
            value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if value.shape[-2:] == size:
            return value
        return F.interpolate(
            value, size=size, mode="bilinear", align_corners=False)

    def _raw_controls(
            self, prediction: torch.Tensor, finest_feature: torch.Tensor,
            history: torch.Tensor, tail_mean: torch.Tensor,
            tail_rms: torch.Tensor, tail_direction: torch.Tensor,
            ) -> torch.Tensor:
        cfg = self.phase_config
        if (prediction.ndim != 5
                or prediction.shape[1] != cfg.out_steps
                or prediction.shape[-1] < 2):
            raise ValueError("prediction must be [B,out_steps,H,W,C>=2]")
        if (finest_feature.ndim != 4
                or finest_feature.shape[0] != prediction.shape[0]
                or finest_feature.shape[1] != cfg.feature_channels):
            raise ValueError(
                "finest_feature must be [B,feature_channels,Hf,Wf]")
        if (history.shape[0] != prediction.shape[0]
                or history.shape[2:4] != prediction.shape[2:4]):
            raise ValueError("history and prediction batches/grids must align")
        target_size = prediction.shape[2:4]
        dtype = self.input_fusion.weight.dtype
        feature = self._resize_to(
            self.feature_projection(finest_feature.detach().to(dtype)),
            target_size)
        observed = self.history_projection(
            self._history_features(history.detach()).to(dtype))
        observed = self._resize_to(observed, target_size)
        batch = prediction.shape[0]
        tail_steps = cfg.out_steps - cfg.anchor_steps
        phase = tail_direction.detach().permute(0, 4, 1, 2, 3).reshape(
            batch, 2 * tail_steps, *target_size).to(dtype)
        phase = self.phase_projection(phase)
        geometry = torch.cat((
            tail_mean.detach().squeeze(1),
            torch.log1p(tail_rms.detach()).squeeze(1),
        ), dim=-1).permute(0, 3, 1, 2).to(dtype)
        geometry = self.geometry_projection(geometry)
        hidden = self.input_fusion(torch.cat(
            (feature, observed, phase, geometry), dim=1))
        coarse = self.coarse(self.downsample(self.native(hidden)))
        raw = self.flow_head(coarse).reshape(
            batch, 2, tail_steps, *coarse.shape[-2:])
        return raw.permute(0, 2, 3, 4, 1)

    def _displacement(
            self, raw_control: torch.Tensor,
            output_size: tuple[int, int],
            ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.phase_config
        control = cfg.max_displacement_pixels * torch.tanh(raw_control)
        batch, steps, grid_h, grid_w, _ = control.shape
        full = F.interpolate(
            control.permute(0, 1, 4, 2, 3).reshape(
                batch * steps, 2, grid_h, grid_w),
            size=output_size, mode="bilinear", align_corners=False)
        full = full.reshape(
            batch, steps, 2, *output_size).permute(0, 1, 3, 4, 2)
        return control, full

    def forward_with_fields(
            self, prediction: torch.Tensor, finest_feature: torch.Tensor,
            history: torch.Tensor) -> LagrangianPhaseFields:
        cfg = self.phase_config
        # Validate the public tensors before slicing the source tail.
        if (prediction.ndim != 5
                or prediction.shape[1] != cfg.out_steps
                or prediction.shape[-1] < 2):
            raise ValueError("prediction must be [B,out_steps,H,W,C>=2]")
        source_uv = prediction[..., :2]
        source_tail = source_uv[:, cfg.anchor_steps:]
        tail_mean, tail_rms, tail_direction = self._source_geometry(
            source_tail)
        raw = self._raw_controls(
            prediction, finest_feature, history,
            tail_mean, tail_rms, tail_direction)
        control, displacement = self._displacement(
            raw, prediction.shape[2:4])

        source32 = source_tail.float()
        warped = spatial_backtrace_warp(source32, displacement)
        identity_sample = spatial_backtrace_warp(
            source32, torch.zeros_like(displacement))
        # This is a fully raw warp of the complete u/v field.  Subtracting the
        # identity sampler only cancels its floating-point residue; there is
        # no spatial-mean or temporal-moment restoration in this branch.
        raw_transport = (source32 + warped - identity_sample).to(source_tail)
        isoenergetic_transport = restore_temporal_mean_rms(
            raw_transport, source_tail,
            epsilon=cfg.projection_epsilon)

        selected_tail = (
            raw_transport if self.mode == RAW_TRANSPORT
            else isoenergetic_transport)
        tail_delta = selected_tail - source_tail
        bypass = raw.eq(0.0).flatten(1).all(dim=1).view(
            -1, 1, 1, 1, 1)
        corrected_tail = _ExactIdentityAddSTE.apply(
            source_tail, tail_delta, bypass)
        # The anchor bypass is a direct tensor path even for non-zero flow;
        # it never relies on adding a numerically zero residual.
        corrected_uv = torch.cat((
            source_uv[:, :cfg.anchor_steps], corrected_tail), dim=1)
        corrected = (
            corrected_uv if prediction.shape[-1] == 2 else
            torch.cat((corrected_uv, prediction[..., 2:]), dim=-1))
        return LagrangianPhaseFields(
            prediction=corrected,
            base_prediction=prediction,
            raw_control=raw,
            control_displacement=control,
            displacement=displacement,
            raw_transport_uv=raw_transport,
            isoenergetic_transport_uv=isoenergetic_transport,
            source_tail_mean=tail_mean.to(source_uv),
            source_tail_rms=tail_rms.to(source_uv),
        )

    def forward(
            self, prediction: torch.Tensor, finest_feature: torch.Tensor,
            history: torch.Tensor) -> torch.Tensor:
        return self.forward_with_fields(
            prediction, finest_feature, history).prediction


class FrozenB210LagrangianPhase(nn.Module):
    """Run frozen B210 once and keep its absolute interval endpoints fixed."""

    def __init__(
            self, base_model: EnsembleUNet2d,
            phase: NativeFeatureLagrangianPhase) -> None:
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
            ) -> tuple[LagrangianPhaseFields, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base, lower, upper, features = (
                self.base_model.forward_distribution_with_features(history))
        fields = self.phase.forward_with_fields(
            base.detach(), features[-1].detach(), history.detach())
        point_delta = fields.prediction[..., :2] - base[..., :2]
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
