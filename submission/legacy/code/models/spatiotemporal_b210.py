"""Spatiotemporal B210 point backbone for RealPDE Track 1.

Unlike the established B210 U-Net, this model never flattens the observed
history into input channels.  All twenty frames remain an explicit axis at
every encoder scale.  A shared learned map then turns the four observed
pyramid trajectories into four future trajectories, which feed a wide
B210-style skip decoder.

The two research arms have exactly the same parameters and common operator
graph.  They differ only inside :class:`MatchedCausalAxisMixer`: the candidate
applies its depthwise kernel left-causally along observed time, while the
control folds time into the batch and applies that same kernel left-causally
along spatial x.  This makes temporal interaction testable without changing
state keys, tensor shapes, initialization, or per-output-element axial MACs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .pyramid_causal3d_operator import (
    FramewiseSpatialResidual,
    MatchedCausalAxisMixer,
    PyramidCausal3DOperator,
    SeparableSpaceTimeBlock,
    _sinusoidal_embedding,
)
from .unet2d import ForecastDistributionMixin


SPATIOTEMPORAL_B210_MODES = (
    "causal_temporal_candidate",
    "framewise_spatial_control",
)


@dataclass(frozen=True)
class SpatiotemporalB210Config:
    """Frozen primary dimensions with overridable widths for tiny tests."""

    in_steps: int = 20
    out_steps: int = 20
    in_channels: int = 3
    out_channels: int = 3
    e0_channels: int = 64
    e1_channels: int = 128
    e2_channels: int = 256
    e3_channels: int = 512
    e0_blocks: int = 2
    e1_blocks: int = 2
    e2_blocks: int = 2
    e3_blocks: int = 2
    groups: int = 8
    axis_kernel: int = 5
    channel_mlp_ratio: float = 4.0
    decoder_blocks: int = 1
    max_displacement: float = 4.0
    residual_limit: float = 4.0
    output_initialization_std: float = 1.0e-3
    predict_uncertainty: bool = False

    @property
    def widths(self) -> tuple[int, int, int, int]:
        return (
            self.e0_channels,
            self.e1_channels,
            self.e2_channels,
            self.e3_channels,
        )

    @property
    def stage_blocks(self) -> tuple[int, int, int, int]:
        return (
            self.e0_blocks,
            self.e1_blocks,
            self.e2_blocks,
            self.e3_blocks,
        )

    def validate(self) -> None:
        dimensions = (
            self.in_steps,
            self.out_steps,
            self.in_channels,
            self.out_channels,
            *self.widths,
            *self.stage_blocks,
            self.groups,
            self.axis_kernel,
            self.decoder_blocks,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("ST-B210 architecture dimensions must be positive")
        if self.in_steps != 20 or self.out_steps != 20:
            raise ValueError(
                "ST-B210 is frozen to twenty observed and future steps")
        if self.out_channels > self.in_channels:
            raise ValueError(
                "transport source requires out_channels <= in_channels")
        if any(width % self.groups for width in self.widths):
            raise ValueError("all ST-B210 widths must divide the group count")
        if self.axis_kernel < 2:
            raise ValueError("axis_kernel must be at least two")
        if self.channel_mlp_ratio <= 0:
            raise ValueError("channel_mlp_ratio must be positive")
        if self.max_displacement <= 0:
            raise ValueError("max_displacement must be positive")
        if self.residual_limit <= 0:
            raise ValueError("residual_limit must be positive")
        if self.output_initialization_std <= 0:
            raise ValueError("output_initialization_std must be positive")
        if self.predict_uncertainty:
            raise ValueError("the ST-B210 discovery backbone is point-only")


def _spatial_stack(
        input_channels: int,
        output_channels: int,
        blocks: int,
        groups: int,
        ) -> nn.Sequential:
    """Build one B210-style spatial decoder stage on frame tensors."""
    layers: list[nn.Module] = [
        FramewiseSpatialResidual(input_channels, output_channels, groups),
    ]
    layers.extend(
        FramewiseSpatialResidual(output_channels, output_channels, groups)
        for _ in range(1, blocks))
    return nn.Sequential(*layers)


class SpatiotemporalB210(nn.Module, ForecastDistributionMixin):
    """Four-scale explicit-time forecaster with a wide future decoder."""

    def __init__(
            self,
            config: SpatiotemporalB210Config,
            interaction_mode: str = "causal_temporal_candidate",
            ) -> None:
        super().__init__()
        config.validate()
        if interaction_mode not in SPATIOTEMPORAL_B210_MODES:
            raise ValueError(
                "interaction_mode must be one of "
                f"{SPATIOTEMPORAL_B210_MODES}")
        self.config = config
        self.interaction_mode = str(interaction_mode)
        widths = config.widths

        # The stem preserves the native model grid.  It is shared independently
        # by every frame and therefore cannot hide a flattened-history route.
        self.stem = nn.Conv2d(
            config.in_channels + 2, widths[0], 3, padding=1)
        self.stage0 = nn.ModuleList(
            SeparableSpaceTimeBlock(widths[0], config, interaction_mode)
            for _ in range(config.e0_blocks))
        self.down1 = nn.Conv2d(widths[0], widths[1], 3, stride=2, padding=1)
        self.stage1 = nn.ModuleList(
            SeparableSpaceTimeBlock(widths[1], config, interaction_mode)
            for _ in range(config.e1_blocks))
        self.down2 = nn.Conv2d(widths[1], widths[2], 3, stride=2, padding=1)
        self.stage2 = nn.ModuleList(
            SeparableSpaceTimeBlock(widths[2], config, interaction_mode)
            for _ in range(config.e2_blocks))
        self.down3 = nn.Conv2d(widths[2], widths[3], 3, stride=2, padding=1)
        self.stage3 = nn.ModuleList(
            SeparableSpaceTimeBlock(widths[3], config, interaction_mode)
            for _ in range(config.e3_blocks))

        self.observed_time0 = nn.Parameter(
            _sinusoidal_embedding(config.in_steps, widths[0]))
        self.observed_time1 = nn.Parameter(
            _sinusoidal_embedding(config.in_steps, widths[1]))
        self.observed_time2 = nn.Parameter(
            _sinusoidal_embedding(config.in_steps, widths[2]))
        self.observed_time3 = nn.Parameter(
            _sinusoidal_embedding(config.in_steps, widths[3]))
        # A single learned observed-to-future coordinate map is common to all
        # scales and both arms.  This is not a flattened-history encoder.
        self.observed_to_future_logits = nn.Parameter(torch.empty(
            config.out_steps, config.in_steps))
        with torch.no_grad():
            observed = torch.arange(config.in_steps, dtype=torch.float32)
            self.observed_to_future_logits.copy_(
                -0.12 * (config.in_steps - 1 - observed).square().view(
                    1, config.in_steps).expand(config.out_steps, -1))
        self.horizon0 = nn.Parameter(
            _sinusoidal_embedding(config.out_steps, widths[0]))
        self.horizon1 = nn.Parameter(
            _sinusoidal_embedding(config.out_steps, widths[1]))
        self.horizon2 = nn.Parameter(
            _sinusoidal_embedding(config.out_steps, widths[2]))
        self.horizon3 = nn.Parameter(
            _sinusoidal_embedding(config.out_steps, widths[3]))

        # This decoder mirrors B210's coarse-to-fine skip topology, but every
        # tensor is a future trajectory rather than a flattened 2-D history.
        self.coarse_spatial = _spatial_stack(
            widths[3], widths[3], config.decoder_blocks, config.groups)
        self.up32 = nn.ConvTranspose2d(widths[3], widths[2], 2, stride=2)
        self.fuse2 = _spatial_stack(
            2 * widths[2], widths[2], config.decoder_blocks, config.groups)
        self.up21 = nn.ConvTranspose2d(widths[2], widths[1], 2, stride=2)
        self.fuse1 = _spatial_stack(
            2 * widths[1], widths[1], config.decoder_blocks, config.groups)
        self.up10 = nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2)
        self.fuse0 = _spatial_stack(
            2 * widths[0], widths[0], config.decoder_blocks, config.groups)
        self.output_spatial = _spatial_stack(
            widths[0], widths[0], config.decoder_blocks, config.groups)
        self.mean_spatial = _spatial_stack(
            widths[0], widths[0], config.decoder_blocks, config.groups)
        self.mean_head = nn.Conv2d(widths[0], config.out_channels, 1)
        self.fluctuation_head = nn.Conv2d(
            widths[0], config.out_channels, 1)
        self.transport_head = nn.Conv2d(widths[0], 2, 1)
        for head in (self.mean_head, self.fluctuation_head):
            nn.init.normal_(
                head.weight, std=config.output_initialization_std)
            nn.init.zeros_(head.bias)
        # Persistence is exact at update zero while both output and upstream
        # residual paths have live gradients on the first optimization step.
        nn.init.zeros_(self.transport_head.weight)
        nn.init.zeros_(self.transport_head.bias)
        self._zero_displacement_override = False
        self._register_forecast_buffers()

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": "spatiotemporal_b210",
            "interaction_mode": self.interaction_mode,
            "flattened_history_bypass": False,
            **asdict(self.config),
        }

    @staticmethod
    def _coordinates(history: torch.Tensor) -> torch.Tensor:
        batch, steps, height, width, _channels = history.shape
        y = torch.linspace(
            -1.0, 1.0, height, device=history.device, dtype=history.dtype)
        x = torch.linspace(
            -1.0, 1.0, width, device=history.device, dtype=history.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx)).view(
            1, 1, 2, height, width).expand(batch, steps, -1, -1, -1)

    @staticmethod
    def _frame_apply(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError("framewise modules require [B,T,C,H,W]")
        batch, steps, channels, height, width = value.shape
        output = module(value.reshape(batch * steps, channels, height, width))
        return output.reshape(
            batch, steps, output.shape[1], output.shape[2], output.shape[3])

    def _validate_history(self, history: torch.Tensor) -> None:
        if (history.ndim != 5
                or history.shape[1] != self.config.in_steps
                or history.shape[-1] != self.config.in_channels):
            raise ValueError(
                f"expected history [B,{self.config.in_steps},H,W,"
                f"{self.config.in_channels}], got {tuple(history.shape)}")
        if history.shape[2] % 8 or history.shape[3] % 8:
            raise ValueError("ST-B210 requires H/W divisible by eight")

    @staticmethod
    def _add_time(value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return value + embedding.to(value).view(
            1, value.shape[1], -1, 1, 1)

    def encode_history(
            self, history: torch.Tensor,
            ) -> tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return four explicit observed-time pyramid trajectories."""
        self._validate_history(history)
        frames = history.permute(0, 1, 4, 2, 3)
        stage0 = self._frame_apply(
            self.stem,
            torch.cat((frames, self._coordinates(history)), dim=2))
        stage0 = self._add_time(stage0, self.observed_time0)
        for block in self.stage0:
            stage0 = block(stage0)
        stage1 = self._add_time(
            self._frame_apply(self.down1, stage0), self.observed_time1)
        for block in self.stage1:
            stage1 = block(stage1)
        stage2 = self._add_time(
            self._frame_apply(self.down2, stage1), self.observed_time2)
        for block in self.stage2:
            stage2 = block(stage2)
        stage3 = self._add_time(
            self._frame_apply(self.down3, stage2), self.observed_time3)
        for block in self.stage3:
            stage3 = block(stage3)
        return stage0, stage1, stage2, stage3

    def _future_project(
            self, history: torch.Tensor,
            horizon: torch.Tensor,
            ) -> torch.Tensor:
        weights = self.observed_to_future_logits.float().softmax(
            dim=-1).to(history)
        future = torch.einsum("ft,btchw->bfchw", weights, history)
        return future + horizon.to(future).view(
            1, self.config.out_steps, -1, 1, 1)

    def project_future_pyramid(
            self,
            observed: tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ) -> tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project every encoder scale to a still-explicit future axis."""
        stage0, stage1, stage2, stage3 = observed
        return (
            self._future_project(stage0, self.horizon0),
            self._future_project(stage1, self.horizon1),
            self._future_project(stage2, self.horizon2),
            self._future_project(stage3, self.horizon3),
        )

    def decode_future(
            self,
            pyramid: tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ) -> torch.Tensor:
        """Decode four future feature scales without collapsing future time."""
        future0, future1, future2, future3 = pyramid
        decoded3 = self._frame_apply(self.coarse_spatial, future3)
        decoded2 = self._frame_apply(self.up32, decoded3)
        decoded2 = self._frame_apply(
            self.fuse2, torch.cat((decoded2, future2), dim=2))
        decoded1 = self._frame_apply(self.up21, decoded2)
        decoded1 = self._frame_apply(
            self.fuse1, torch.cat((decoded1, future1), dim=2))
        decoded0 = self._frame_apply(self.up10, decoded1)
        decoded0 = self._frame_apply(
            self.fuse0, torch.cat((decoded0, future0), dim=2))
        return self._frame_apply(self.output_spatial, decoded0)

    def axis_execution_contract(self) -> list[dict[str, object]]:
        """Return the last audited execution facts for all eight mixers."""
        result: list[dict[str, object]] = []
        stages = (self.stage0, self.stage1, self.stage2, self.stage3)
        for stage_index, stage in enumerate(stages):
            for block_index, block in enumerate(stage):
                contract = block.axis.last_axis_contract
                if contract is not None:
                    result.append({
                        "stage": stage_index,
                        "block": block_index,
                        **contract,
                    })
        return result

    def set_axis_bypass_override(self, enabled: bool) -> None:
        """Make every matched axial mixer an exact identity for diagnosis."""
        for stage in (self.stage0, self.stage1, self.stage2, self.stage3):
            for block in stage:
                block.axis.set_bypass_override(enabled)

    def set_zero_displacement_override(self, enabled: bool) -> None:
        """Select an exact same-checkpoint no-transport mechanism ablation."""
        self._zero_displacement_override = bool(enabled)

    # Reuse the already manually audited PC3D sampler rather than carrying a
    # subtly different coordinate convention in this closely related model.
    warp = staticmethod(PyramidCausal3DOperator.warp)

    def forward_components(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_history(history)
        decoded = self.decode_future(
            self.project_future_pyramid(self.encode_history(history)))

        raw_displacement = self._frame_apply(self.transport_head, decoded)
        displacement = self.config.max_displacement * torch.tanh(
            raw_displacement.permute(0, 1, 3, 4, 2))
        if self._zero_displacement_override:
            displacement = torch.zeros_like(displacement)
        persistence = self._persistence(history)
        transported = (
            persistence.unsqueeze(1).expand(
                -1, self.config.out_steps, -1, -1, -1)
            if self._zero_displacement_override
            else self.warp(persistence, displacement))

        mean_feature = self._frame_apply(
            self.mean_spatial, decoded.mean(dim=1, keepdim=True))
        mean_delta = self._frame_apply(
            self.mean_head, mean_feature)[:, 0].permute(0, 2, 3, 1)
        mean_delta = self.config.residual_limit * torch.tanh(
            mean_delta / self.config.residual_limit)

        fluctuation = self._frame_apply(
            self.fluctuation_head, decoded).permute(0, 1, 3, 4, 2)
        fluctuation = self.config.residual_limit * torch.tanh(
            fluctuation / self.config.residual_limit)
        fluctuation = fluctuation - fluctuation.mean(dim=1, keepdim=True)
        prediction = transported + mean_delta.unsqueeze(1) + fluctuation
        return {
            "prediction": prediction,
            "transported": transported,
            "persistence": persistence,
            "mean_delta": mean_delta,
            "fluctuation": fluctuation,
            "displacement": displacement,
            "decoded_future": decoded,
        }

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_components(history)["prediction"]


def build_spatiotemporal_b210(
        interaction_mode: str = "causal_temporal_candidate",
        **overrides: object,
        ) -> SpatiotemporalB210:
    return SpatiotemporalB210(
        SpatiotemporalB210Config(**overrides), interaction_mode)


# Compatibility aliases for the alternative CamelCase spelling used in early
# design notes.  They refer to the exact same classes and create no new state.
SpatioTemporalB210Config = SpatiotemporalB210Config
SpatioTemporalB210 = SpatiotemporalB210
