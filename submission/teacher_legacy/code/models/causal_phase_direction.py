"""History-conditioned, isoenergetic phase correction for a frozen B210.

The mature B210 remains responsible for the temporal mean and temporal
fluctuation RMS at every pixel and velocity channel.  This adapter sees the
twenty observed frames on an explicit time axis and may change only the
zero-mean, unit-RMS direction of the future velocity trajectory.  A fixed
algebraic projection restores the B210 mean and RMS after every learned
update; pressure and any additional channels pass through bitwise unchanged.

This is intentionally different from ``IsoenergeticTemporalTransport`` in
``ensemble_unet2d.py``.  That legacy branch conditions on a flattened-B210
feature and the point forecast, while its optional ``history`` argument is
unused.  Here every observed frame passes through one shared spatial encoder;
the main arm then uses a strict left-causal mixer, while the explicit
``pooled_history_no_mixer`` ablation averages the frame states directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .temporal_transport import (
    CausalTemporalMixer,
    SharedFrameEncoder,
    TemporalTransportConfig,
)
from .unet2d import ResidualConvBlock


POOLED_HISTORY_MODE = "pooled_history_no_mixer"


PHASE_DIRECTION_MODES = (
    "causal_temporal",
    "framewise_spatial_control",
    POOLED_HISTORY_MODE,
)


class _ExactIdentityAddSTE(torch.autograd.Function):
    """Use the exact base value at update zero while retaining delta gradient."""

    @staticmethod
    def forward(
            ctx, base: torch.Tensor, delta: torch.Tensor,
            bypass: torch.Tensor) -> torch.Tensor:
        del ctx
        return torch.where(bypass, base, base + delta)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        del ctx
        return grad_output, grad_output, None


@dataclass(frozen=True)
class CausalPhaseDirectionConfig:
    """Small adapter dimensions; the defaults target the Track-1 20/20 task."""

    in_steps: int = 20
    out_steps: int = 20
    in_channels: int = 3
    out_channels: int = 3
    hidden_channels: int = 24
    spatial_blocks: int = 1
    spatial_downsample: int = 2
    temporal_blocks: int = 5
    temporal_kernel: int = 5
    groups: int = 4
    mixer_mode: str = "causal_temporal"
    use_coordinates: bool = True
    direction_limit: float = 1.0
    projection_epsilon: float = 1.0e-12

    def validate(self) -> None:
        dimensions = (
            self.in_steps,
            self.out_steps,
            self.in_channels,
            self.out_channels,
            self.hidden_channels,
            self.spatial_blocks,
            self.spatial_downsample,
            self.temporal_blocks,
            self.temporal_kernel,
            self.groups,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("phase-direction dimensions must be positive")
        if self.in_channels < 2 or self.out_channels < 2:
            raise ValueError("phase direction requires at least u/v channels")
        if self.hidden_channels % self.groups:
            raise ValueError("hidden_channels must be divisible by groups")
        if self.temporal_kernel < 2:
            raise ValueError("temporal_kernel must be at least two")
        if self.mixer_mode not in PHASE_DIRECTION_MODES:
            raise ValueError(
                f"mixer_mode must be one of {PHASE_DIRECTION_MODES}")
        if self.direction_limit <= 0.0:
            raise ValueError("direction_limit must be positive")
        if self.projection_epsilon <= 0.0:
            raise ValueError("projection_epsilon must be positive")

    def history_encoder_config(self) -> TemporalTransportConfig:
        """Build the already-audited explicit-time encoder/mixer contract."""
        return TemporalTransportConfig(
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            in_channels=self.in_channels,
            out_channels=min(self.in_channels, self.out_channels),
            hidden_channels=self.hidden_channels,
            spatial_blocks=self.spatial_blocks,
            spatial_downsample=self.spatial_downsample,
            temporal_blocks=self.temporal_blocks,
            temporal_kernel=self.temporal_kernel,
            # ``pooled_history_no_mixer`` deliberately has no mixer.  This
            # value is consumed only when constructing its shared frame
            # encoder, whose execution is independent of the mixer mode.
            mixer_mode=(
                "causal_temporal"
                if self.mixer_mode == POOLED_HISTORY_MODE
                else self.mixer_mode),
            groups=self.groups,
            use_coordinates=self.use_coordinates,
        )


@dataclass
class CausalPhaseDirectionOutput:
    """Inspectable responsibility split returned by ``forward_components``."""

    prediction: torch.Tensor
    base_prediction: torch.Tensor
    projected_uv: torch.Tensor
    direction_update: torch.Tensor
    direction_gate: torch.Tensor
    source_direction: torch.Tensor
    candidate_direction: torch.Tensor
    source_temporal_mean: torch.Tensor
    source_temporal_rms: torch.Tensor


class SharedForecastEncoder(nn.Module):
    """Apply one spatial encoder to every B210 future u/v frame."""

    def __init__(self, config: CausalPhaseDirectionConfig):
        super().__init__()
        channels = config.hidden_channels
        self.stem = nn.Conv2d(
            2, channels, 3, stride=config.spatial_downsample, padding=1)
        self.body = nn.Sequential(*(
            ResidualConvBlock(channels, channels, config.groups)
            for _ in range(config.spatial_blocks)))

    def forward(self, direction: torch.Tensor) -> torch.Tensor:
        if direction.ndim != 5 or direction.shape[-1] != 2:
            raise ValueError("forecast direction must be [B,T,H,W,2]")
        batch, steps, height, width, channels = direction.shape
        frames = direction.permute(0, 1, 4, 2, 3).reshape(
            batch * steps, channels, height, width)
        encoded = self.body(self.stem(frames))
        return encoded.reshape(
            batch, steps, encoded.shape[1], encoded.shape[2],
            encoded.shape[3])


class CausalPhaseDirectionAdapter(nn.Module):
    """Predict a history-conditioned future direction and restore B210 energy."""

    def __init__(self, config: CausalPhaseDirectionConfig):
        super().__init__()
        config.validate()
        self.config = config
        component_config = config.history_encoder_config()
        self.frame_encoder = SharedFrameEncoder(component_config)
        # The pooled-history ablation keeps the exact same shared per-frame
        # spatial representation and fixed mean aggregation, but owns no
        # temporal/spatial mixer parameters or execution path.
        # Construct the full mixer first so every later common module consumes
        # exactly the same RNG stream as the two established arms.  The pooled
        # arm drops it only after all shared tensors have been initialized.
        self.temporal_mixer = CausalTemporalMixer(component_config)
        self.forecast_encoder = SharedForecastEncoder(config)
        channels = config.hidden_channels
        self.horizon_embedding = nn.Parameter(
            self._sinusoidal_embedding(config.out_steps, channels))
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            ResidualConvBlock(channels, channels, config.groups),
        )
        self.direction_head = nn.Conv2d(channels, 2, 1)
        self.gate_head = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.direction_head.weight)
        nn.init.zeros_(self.direction_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        if config.mixer_mode == POOLED_HISTORY_MODE:
            self.temporal_mixer = nn.Identity()

    @staticmethod
    def _sinusoidal_embedding(steps: int, channels: int) -> torch.Tensor:
        position = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
        index = torch.arange(0, channels, 2, dtype=torch.float32)
        frequency = torch.exp(
            index * (-torch.log(torch.tensor(10_000.0)) / channels))
        result = torch.zeros(steps, channels, dtype=torch.float32)
        result[:, 0::2] = torch.sin(position * frequency)
        if channels > 1:
            result[:, 1::2] = torch.cos(
                position * frequency[:result[:, 1::2].shape[1]])
        return result

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": "causal_phase_direction_adapter",
            "base_responsibility": "temporal_mean_and_fluctuation_rms",
            "adapter_responsibility": "zero_mean_unit_rms_uv_direction",
            "pressure_passthrough": True,
            "flattened_history_bypass": False,
            **asdict(self.config),
        }

    def encode_frames(self, history: torch.Tensor) -> torch.Tensor:
        """Expose pre-mixer states for the strict-causality test."""
        return self.frame_encoder(history)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        """Return one mixed, or explicitly unmixed, state per frame."""
        return self.temporal_mixer(self.encode_frames(history))

    def _validate_inputs(
            self, history: torch.Tensor,
            base_prediction: torch.Tensor) -> None:
        expected_history = (
            self.config.in_steps, self.config.in_channels)
        if (history.ndim != 5
                or history.shape[1] != expected_history[0]
                or history.shape[-1] != expected_history[1]):
            raise ValueError(
                "expected history [B,"
                f"{expected_history[0]},H,W,{expected_history[1]}]")
        if (base_prediction.ndim != 5
                or base_prediction.shape[0] != history.shape[0]
                or base_prediction.shape[1] != self.config.out_steps
                or base_prediction.shape[2:4] != history.shape[2:4]
                or base_prediction.shape[-1] != self.config.out_channels):
            raise ValueError(
                "base prediction must be [B,out_steps,H,W,out_channels] "
                "and spatially aligned with history")

    def _source_geometry(
            self, source_uv: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source = source_uv.float()
        mean = source.mean(dim=1, keepdim=True)
        fluctuation = source - mean
        energy = fluctuation.square().mean(dim=1, keepdim=True)
        active = energy > self.config.projection_epsilon
        # Mask before sqrt/division.  torch.where around an already computed
        # sqrt(0) would still leave an infinite inactive derivative.
        safe_energy = torch.where(active, energy, torch.ones_like(energy))
        rms = torch.where(active, safe_energy.sqrt(), torch.zeros_like(energy))
        safe_rms = torch.where(active, rms, torch.ones_like(rms))
        direction = torch.where(
            active, fluctuation / safe_rms, torch.zeros_like(fluctuation))
        return mean, rms, direction, active

    def project_direction(
            self, source_uv: torch.Tensor,
            direction_update: torch.Tensor,
            ) -> tuple[
                torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor]:
        """Restore source temporal mean/RMS after changing only direction."""
        if source_uv.shape != direction_update.shape:
            raise ValueError("source and direction update must be aligned")
        mean, rms, source_direction, source_active = (
            self._source_geometry(source_uv))
        update = direction_update.float()
        update = update - update.mean(dim=1, keepdim=True)
        candidate = source_direction + update
        candidate = candidate - candidate.mean(dim=1, keepdim=True)
        candidate_energy = candidate.square().mean(dim=1, keepdim=True)
        active = source_active & (
            candidate_energy > self.config.projection_epsilon)
        safe_candidate_rms = torch.where(
            active, candidate_energy, torch.ones_like(candidate_energy)).sqrt()
        unit_candidate = torch.where(
            active, candidate / safe_candidate_rms, source_direction)
        restored = mean + rms * unit_candidate
        restored = torch.where(source_active, restored, source_uv.float())
        return (
            restored.to(source_uv), source_direction.to(source_uv),
            unit_candidate.to(source_uv), mean.to(source_uv), rms.to(source_uv))

    def predict_direction_update(
            self, history: torch.Tensor,
            source_direction: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a bounded, exactly zero-time-mean u/v direction update."""
        history_states = self.encode_history(history)
        # Fixed mean pooling gives every observed frame a direct route to the
        # correction.  Candidate states contain strict-causal prefix
        # interactions, the matched control mixes only spatial x, and the
        # pooled ablation has no learned mixer at all.
        history_context = history_states.mean(dim=1, keepdim=True).expand(
            -1, self.config.out_steps, -1, -1, -1)
        forecast = self.forecast_encoder(source_direction)
        if forecast.shape[-2:] != history_context.shape[-2:]:
            raise RuntimeError("history and forecast encoder grids diverged")
        horizon = self.horizon_embedding.to(forecast).view(
            1, self.config.out_steps, self.config.hidden_channels, 1, 1)
        forecast = forecast + horizon
        batch, steps, channels, height, width = forecast.shape
        fused = torch.cat((history_context, forecast), dim=2).reshape(
            batch * steps, 2 * channels, height, width)
        fused = self.fusion(fused)
        raw = self.direction_head(fused).reshape(
            batch, steps, 2, height, width)
        gate = self.gate_head(fused).reshape(
            batch, steps, 1, height, width).sigmoid()
        if (height, width) != tuple(history.shape[2:4]):
            raw = F.interpolate(
                raw.reshape(batch * steps, 2, height, width),
                size=history.shape[2:4], mode="bilinear",
                align_corners=False).reshape(
                    batch, steps, 2, history.shape[2], history.shape[3])
            gate = F.interpolate(
                gate.reshape(batch * steps, 1, height, width),
                size=history.shape[2:4], mode="bilinear",
                align_corners=False).reshape(
                    batch, steps, 1, history.shape[2], history.shape[3])
        update = raw.permute(0, 1, 3, 4, 2)
        gate = gate.permute(0, 1, 3, 4, 2)
        update = (
            self.config.direction_limit * gate * torch.tanh(update))
        return update - update.mean(dim=1, keepdim=True), gate

    def forward_components(
            self, history: torch.Tensor,
            base_prediction: torch.Tensor) -> CausalPhaseDirectionOutput:
        self._validate_inputs(history, base_prediction)
        base_prediction = base_prediction.detach()
        source_uv = base_prediction[..., :2]
        (source_mean, source_rms, source_direction,
         _source_active) = self._source_geometry(source_uv)
        direction_update, direction_gate = self.predict_direction_update(
            history, source_direction.to(source_uv))
        (candidate_uv, source_direction, candidate_direction,
         source_mean, source_rms) = self.project_direction(
            source_uv, direction_update)
        delta = candidate_uv - source_uv
        bypass = direction_update.eq(0).flatten(1).all(dim=1).view(
            -1, 1, 1, 1, 1)
        projected_uv = _ExactIdentityAddSTE.apply(source_uv, delta, bypass)
        prediction = (
            projected_uv if base_prediction.shape[-1] == 2
            else torch.cat((
                projected_uv, base_prediction[..., 2:]), dim=-1))
        return CausalPhaseDirectionOutput(
            prediction=prediction,
            base_prediction=base_prediction,
            projected_uv=projected_uv,
            direction_update=direction_update,
            direction_gate=direction_gate,
            source_direction=source_direction,
            candidate_direction=candidate_direction,
            source_temporal_mean=source_mean,
            source_temporal_rms=source_rms,
        )

    def forward(
            self, history: torch.Tensor,
            base_prediction: torch.Tensor) -> torch.Tensor:
        return self.forward_components(history, base_prediction).prediction


class FrozenB210CausalPhase(nn.Module):
    """Keep a mature B210 frozen and train/deploy only its phase adapter."""

    def __init__(
            self, base_model: nn.Module,
            adapter: CausalPhaseDirectionAdapter):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.base_model.requires_grad_(False)
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        self.adapter.train(mode)
        return self

    def architecture_config(self) -> dict[str, object]:
        base_config = (
            self.base_model.architecture_config()
            if hasattr(self.base_model, "architecture_config") else None)
        return {
            "architecture": "frozen_b210_causal_phase",
            "base_model": base_config,
            "adapter": self.adapter.architecture_config(),
        }

    def _base_point(self, history: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.base_model(history).detach()

    def forward_components(
            self, history: torch.Tensor,
            adapter_history: torch.Tensor | None = None,
            ) -> CausalPhaseDirectionOutput:
        if adapter_history is None:
            adapter_history = history
        return self.adapter.forward_components(
            adapter_history, self._base_point(history))

    def forward(
            self, history: torch.Tensor,
            adapter_history: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_components(history, adapter_history).prediction

    def forward_distribution(
            self, history: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(self.base_model, "forward_distribution"):
            raise AttributeError("base B210 has no distribution interface")
        with torch.no_grad():
            base_prediction, lower_extent, upper_extent = (
                self.base_model.forward_distribution(history))
        prediction = self.adapter(history, base_prediction.detach())
        if (lower_extent.shape != prediction[..., :2].shape
                or upper_extent.shape != prediction[..., :2].shape):
            raise ValueError("B210 interval extents must align with future u/v")
        point_delta = prediction[..., :2] - base_prediction[..., :2]
        # EnsembleUNet2d returns normalized *extents*, while the submission
        # wrapper forms absolute endpoints as point-lower and point+upper.
        # Compensating by the point delta keeps those absolute endpoints,
        # coverage and width unchanged.  SPS itself may still move because its
        # accuracy multiplier is recomputed from the new point forecast.  An
        # extent may become signed if the new point crosses an old endpoint;
        # the reconstructed lower/upper remain the original ordered endpoints.
        lower_extent = lower_extent.detach() + point_delta
        upper_extent = upper_extent.detach() - point_delta
        return prediction, lower_extent, upper_extent

    def adapter_state_dict(self) -> Mapping[str, torch.Tensor]:
        """Return only the small trainable state; never duplicate B210 weights."""
        return self.adapter.state_dict()


def build_causal_phase_direction(
        **overrides: object) -> CausalPhaseDirectionAdapter:
    return CausalPhaseDirectionAdapter(
        CausalPhaseDirectionConfig(**overrides))
