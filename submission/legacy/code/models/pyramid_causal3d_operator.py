"""Pyramid Causal-3D Operator for RealPDE Track 1.

The model is a new primary backbone: all observed frames remain explicit
through a shared-frame spatial pyramid and local separable space-time blocks.
For a controlled architecture test, every axial block owns one identical
depthwise Conv3d kernel.  The candidate applies it causally along observed
time; the control folds time into the batch and applies that exact kernel
causally along spatial x.  Canonical input/output elements and per-element
operations are therefore matched, while only the interacting axis differs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log

import torch
from torch import nn
from torch.nn import functional as F

from .unet2d import ForecastDistributionMixin, _group_norm


PYRAMID_CAUSAL3D_MODES = (
    "causal_temporal_candidate",
    "framewise_spatial_control",
)


@dataclass(frozen=True)
class PyramidCausal3DOperatorConfig:
    in_steps: int = 20
    out_steps: int = 20
    in_channels: int = 3
    out_channels: int = 3
    e0_channels: int = 96
    e1_channels: int = 160
    e2_channels: int = 256
    e0_blocks: int = 2
    e1_blocks: int = 2
    e2_blocks: int = 2
    groups: int = 8
    axis_kernel: int = 5
    channel_mlp_ratio: float = 4.0
    max_displacement: float = 4.0
    residual_limit: float = 4.0
    output_initialization_std: float = 1.0e-3
    predict_uncertainty: bool = False

    def validate(self) -> None:
        dimensions = (
            self.in_steps, self.out_steps, self.in_channels,
            self.out_channels, self.e0_channels, self.e1_channels,
            self.e2_channels, self.e0_blocks, self.e1_blocks,
            self.e2_blocks, self.groups, self.axis_kernel,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("PC3D architecture dimensions must be positive")
        if self.in_steps != 20 or self.out_steps != 20:
            raise ValueError("PC3D screen is frozen to twenty observed/future steps")
        if self.out_channels > self.in_channels:
            raise ValueError("transport source requires out_channels <= in_channels")
        if any(width % self.groups for width in (
                self.e0_channels, self.e1_channels, self.e2_channels)):
            raise ValueError("all pyramid widths must divide the group count")
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
            raise ValueError("PC3D discovery backbone is point-only")


class FramewiseSpatialResidual(nn.Module):
    """Two-dimensional local update, independent for every time slice."""

    def __init__(self, input_channels: int, output_channels: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            _group_norm(groups, input_channels), nn.SiLU(),
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            _group_norm(groups, output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels else nn.Identity())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.skip(value) + self.body(value)


class MatchedCausalAxisMixer(nn.Module):
    """Apply one shared kernel causally along time or independently along x."""

    def __init__(self, channels: int, kernel_size: int, groups: int,
                 interaction_mode: str):
        super().__init__()
        if interaction_mode not in PYRAMID_CAUSAL3D_MODES:
            raise ValueError(f"unsupported interaction_mode {interaction_mode!r}")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.left_pad = self.kernel_size - 1
        self.right_pad = 0
        self.interaction_mode = str(interaction_mode)
        self.norm = _group_norm(groups, channels)
        # This is the single learnable axial kernel tensor.  Both modes invoke
        # the same module; only T or x is placed on its depth dimension.
        self.depthwise = nn.Conv3d(
            channels, channels, (kernel_size, 1, 1),
            groups=channels, bias=False)
        self.projection = nn.Conv3d(channels, 2 * channels, 1)
        self.last_axis_contract: dict[str, object] | None = None
        self._bypass_override = False

    def set_bypass_override(self, enabled: bool) -> None:
        self._bypass_override = bool(enabled)

    def _contract(self, shape: tuple[int, ...],
                  packed_shape: tuple[int, ...]) -> dict[str, object]:
        batch, steps, channels, height, width = shape
        output_elements = batch * steps * channels * height * width
        control = self.interaction_mode == "framewise_spatial_control"
        return {
            "canonical_shape": shape,
            "packed_shape": packed_shape,
            "kernel_size": self.kernel_size,
            "left_pad": self.left_pad,
            "right_pad": self.right_pad,
            "output_elements": output_elements,
            "macs": output_elements * (self.kernel_size + 2 * channels),
            "control_time_folded": control,
            "cross_time": not control,
            "bypassed": self._bypass_override,
        }

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 5:
            raise ValueError("axis mixer requires [B,T,C,H,W]")
        batch, steps, channels, height, width = trajectory.shape
        if channels != self.channels:
            raise ValueError("axis mixer channel width changed")
        if self._bypass_override:
            packed_shape = (
                (batch, channels, steps, height, width)
                if self.interaction_mode == "causal_temporal_candidate" else
                (batch * steps, channels, width, height, 1))
            self.last_axis_contract = self._contract(
                tuple(trajectory.shape), packed_shape)
            return trajectory
        normalized = self.norm(trajectory.reshape(
            batch * steps, channels, height, width))
        if self.interaction_mode == "causal_temporal_candidate":
            packed = normalized.reshape(
                batch, steps, channels, height, width).permute(0, 2, 1, 3, 4)
            packed_shape = tuple(packed.shape)
            padded = F.pad(packed, (0, 0, 0, 0, self.left_pad, 0))
            filtered = self.depthwise(padded)
            value, gate = self.projection(filtered).chunk(2, dim=1)
            update = (value * F.silu(gate)).permute(0, 2, 1, 3, 4)
            control_time_folded = False
            cross_time = True
        else:
            # [B*T,C,H,W] -> [B*T,C,D=x,H=y,W=1].  T is irreversibly
            # folded into batch before the shared kernel, excluding any
            # cross-frame communication in the control.
            packed = normalized.permute(0, 1, 3, 2).unsqueeze(-1)
            packed_shape = tuple(packed.shape)
            padded = F.pad(packed, (0, 0, 0, 0, self.left_pad, 0))
            filtered = self.depthwise(padded)
            value, gate = self.projection(filtered).chunk(2, dim=1)
            update = (value * F.silu(gate)).squeeze(-1).permute(
                0, 1, 3, 2).reshape(batch, steps, channels, height, width)
            control_time_folded = True
            cross_time = False
        # Depthwise K MACs plus the shared C->2C point projection.  Common
        # norm/residual work is deliberately outside this axial accounting.
        self.last_axis_contract = self._contract(
            tuple(trajectory.shape), packed_shape)
        # Assert the routing facts rather than trusting stale metadata.
        self.last_axis_contract["control_time_folded"] = control_time_folded
        self.last_axis_contract["cross_time"] = cross_time
        return trajectory + update


class SeparableSpaceTimeBlock(nn.Module):
    """Matched axial interaction, channel MLP, then framewise spatial update."""

    def __init__(self, channels: int, config: PyramidCausal3DOperatorConfig,
                 interaction_mode: str):
        super().__init__()
        self.channels = int(channels)
        self.axis = MatchedCausalAxisMixer(
            channels, config.axis_kernel, config.groups, interaction_mode)
        self.channel_norm = nn.LayerNorm(channels)
        hidden = int(round(channels * config.channel_mlp_ratio))
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(),
            nn.Linear(hidden, channels))
        self.spatial = FramewiseSpatialResidual(channels, channels, config.groups)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        trajectory = self.axis(trajectory)
        channel_last = trajectory.permute(0, 1, 3, 4, 2)
        channel_last = channel_last + self.channel_mlp(
            self.channel_norm(channel_last))
        trajectory = channel_last.permute(0, 1, 4, 2, 3)
        batch, steps, channels, height, width = trajectory.shape
        spatial_input = trajectory.reshape(
            batch * steps, channels, height, width)
        # At a singleton spatial scale, the preceding permutation produces a
        # tensor that is simultaneously NCHW- and NHWC-contiguous.  PyTorch's
        # CPU GroupNorm backward can then classify X and dY differently and
        # reject the otherwise valid gradient.  Canonicalize only that CPU
        # autograd boundary; official CUDA shapes never enter this branch.
        if (torch.is_grad_enabled() and spatial_input.device.type == "cpu"
                and (height == 1 or width == 1)):
            spatial_input = spatial_input.clone(
                memory_format=torch.contiguous_format)
        spatial = self.spatial(spatial_input)
        return spatial.reshape(batch, steps, channels, height, width)


def _sinusoidal_embedding(steps: int, channels: int) -> torch.Tensor:
    position = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, channels, 2, dtype=torch.float32)
        * (-log(10000.0) / max(channels, 2)))
    result = torch.zeros(steps, channels, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * frequency)
    if channels > 1:
        result[:, 1::2] = torch.cos(
            position * frequency[:result[:, 1::2].shape[1]])
    return result


class PyramidCausal3DOperator(nn.Module, ForecastDistributionMixin):
    """Multiscale causal space-time forecaster with bounded source transport."""

    def __init__(self, config: PyramidCausal3DOperatorConfig,
                 interaction_mode: str = "causal_temporal_candidate"):
        super().__init__()
        config.validate()
        if interaction_mode not in PYRAMID_CAUSAL3D_MODES:
            raise ValueError(f"interaction_mode must be one of {PYRAMID_CAUSAL3D_MODES}")
        self.config = config
        self.interaction_mode = str(interaction_mode)
        widths = (config.e0_channels, config.e1_channels, config.e2_channels)
        self.stem = nn.Conv2d(
            config.in_channels + 2, widths[0], 3, stride=2, padding=1)
        self.stage0 = nn.ModuleList(
            SeparableSpaceTimeBlock(widths[0], config, self.interaction_mode)
            for _ in range(config.e0_blocks))
        self.down1 = nn.Conv2d(widths[0], widths[1], 3, stride=2, padding=1)
        self.stage1 = nn.ModuleList(
            SeparableSpaceTimeBlock(widths[1], config, self.interaction_mode)
            for _ in range(config.e1_blocks))
        self.down2 = nn.Conv2d(widths[1], widths[2], 3, stride=2, padding=1)
        self.stage2 = nn.ModuleList(
            SeparableSpaceTimeBlock(widths[2], config, self.interaction_mode)
            for _ in range(config.e2_blocks))

        self.observed_time0 = nn.Parameter(_sinusoidal_embedding(20, widths[0]))
        self.observed_time1 = nn.Parameter(_sinusoidal_embedding(20, widths[1]))
        self.observed_time2 = nn.Parameter(_sinusoidal_embedding(20, widths[2]))
        # One bounded observed->future map is reused at all spatial scales.
        self.observed_to_future_logits = nn.Parameter(torch.empty(20, 20))
        with torch.no_grad():
            observed = torch.arange(20, dtype=torch.float32)
            self.observed_to_future_logits.copy_(
                -0.12 * (19.0 - observed).square().view(1, 20).expand(20, -1))
        self.horizon0 = nn.Parameter(_sinusoidal_embedding(20, widths[0]))
        self.horizon1 = nn.Parameter(_sinusoidal_embedding(20, widths[1]))
        self.horizon2 = nn.Parameter(_sinusoidal_embedding(20, widths[2]))

        self.coarse_spatial = FramewiseSpatialResidual(
            widths[2], widths[2], config.groups)
        self.up21 = nn.ConvTranspose2d(widths[2], widths[1], 2, stride=2)
        self.fuse1 = FramewiseSpatialResidual(
            2 * widths[1], widths[1], config.groups)
        self.up10 = nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2)
        self.fuse0 = FramewiseSpatialResidual(
            2 * widths[0], widths[0], config.groups)
        self.output_spatial = FramewiseSpatialResidual(
            widths[0], widths[0], config.groups)
        self.mean_spatial = FramewiseSpatialResidual(
            widths[0], widths[0], config.groups)
        packed_channels = 4 * config.out_channels
        self.mean_head = nn.Conv2d(widths[0], packed_channels, 1)
        self.fluctuation_head = nn.Conv2d(widths[0], packed_channels, 1)
        self.transport_head = nn.Conv2d(widths[0], 8, 1)
        for head in (self.mean_head, self.fluctuation_head):
            nn.init.normal_(head.weight, std=config.output_initialization_std)
            nn.init.zeros_(head.bias)
        # Exact update-zero transport: its own gradient is live immediately;
        # common residual heads keep the complete upstream graph live too.
        nn.init.zeros_(self.transport_head.weight)
        nn.init.zeros_(self.transport_head.bias)
        self._zero_displacement_override = False
        self._register_forecast_buffers()

    def architecture_config(self) -> dict:
        return {
            "architecture": "pyramid_causal3d_operator",
            "interaction_mode": self.interaction_mode,
            **asdict(self.config),
        }

    @staticmethod
    def _coordinates(history: torch.Tensor) -> torch.Tensor:
        batch, steps, height, width, _channels = history.shape
        y = torch.linspace(-1.0, 1.0, height, device=history.device,
                           dtype=history.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=history.device,
                           dtype=history.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx)).view(1, 1, 2, height, width).expand(
            batch, steps, -1, -1, -1)

    @staticmethod
    def _frame_apply(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = value.shape
        output = module(value.reshape(batch * steps, channels, height, width))
        return output.reshape(batch, steps, output.shape[1],
                              output.shape[2], output.shape[3])

    def _validate_history(self, history: torch.Tensor) -> None:
        if (history.ndim != 5 or history.shape[1] != self.config.in_steps
                or history.shape[-1] != self.config.in_channels):
            raise ValueError(
                f"expected history [B,{self.config.in_steps},H,W,"
                f"{self.config.in_channels}], got {tuple(history.shape)}")
        if history.shape[2] % 8 or history.shape[3] % 8:
            raise ValueError("PC3D requires H/W divisible by 8")

    @staticmethod
    def _add_time(value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return value + embedding.to(value).view(1, value.shape[1], -1, 1, 1)

    def encode_history(
            self, history: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_history(history)
        frames = history.permute(0, 1, 4, 2, 3)
        stage0 = self._frame_apply(
            self.stem, torch.cat((frames, self._coordinates(history)), dim=2))
        stage0 = self._add_time(stage0, self.observed_time0)
        for block in self.stage0:
            stage0 = block(stage0)
        stage1 = self._frame_apply(self.down1, stage0)
        stage1 = self._add_time(stage1, self.observed_time1)
        for block in self.stage1:
            stage1 = block(stage1)
        stage2 = self._frame_apply(self.down2, stage1)
        stage2 = self._add_time(stage2, self.observed_time2)
        for block in self.stage2:
            stage2 = block(stage2)
        return stage0, stage1, stage2

    def _future_project(self, history: torch.Tensor,
                        horizon: torch.Tensor) -> torch.Tensor:
        weights = self.observed_to_future_logits.float().softmax(dim=-1).to(history)
        future = torch.einsum("ft,btchw->bfchw", weights, history)
        return future + horizon.to(future).view(1, 20, -1, 1, 1)

    def decode_future(
            self, pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ) -> torch.Tensor:
        stage0, stage1, stage2 = pyramid
        future0 = self._future_project(stage0, self.horizon0)
        future1 = self._future_project(stage1, self.horizon1)
        future2 = self._future_project(stage2, self.horizon2)
        future2 = self._frame_apply(self.coarse_spatial, future2)
        decoded1 = self._frame_apply(self.up21, future2)
        decoded1 = self._frame_apply(
            self.fuse1, torch.cat((decoded1, future1), dim=2))
        decoded0 = self._frame_apply(self.up10, decoded1)
        decoded0 = self._frame_apply(
            self.fuse0, torch.cat((decoded0, future0), dim=2))
        return self._frame_apply(self.output_spatial, decoded0)

    def axis_execution_contract(self) -> list[dict[str, object]]:
        """Return the last audited axial execution contract at every scale."""
        result = []
        for stage_index, stage in enumerate((self.stage0, self.stage1, self.stage2)):
            for block_index, block in enumerate(stage):
                contract = block.axis.last_axis_contract
                if contract is not None:
                    result.append({"stage": stage_index, "block": block_index,
                                   **contract})
        return result

    def set_zero_displacement_override(self, enabled: bool) -> None:
        """Select an exact same-checkpoint no-warp mechanism ablation."""
        self._zero_displacement_override = bool(enabled)

    def set_axis_bypass_override(self, enabled: bool) -> None:
        """Make every axial mixer an exact identity for endpoint diagnosis."""
        for stage in (self.stage0, self.stage1, self.stage2):
            for block in stage:
                block.axis.set_bypass_override(enabled)

    @staticmethod
    def warp(source: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        if source.ndim != 4 or displacement.ndim != 5:
            raise ValueError("warp requires source [B,H,W,C], displacement [B,T,H,W,2]")
        batch, height, width, channels = source.shape
        if (displacement.shape[0] != batch
                or displacement.shape[2:4] != (height, width)
                or displacement.shape[-1] != 2):
            raise ValueError("warp source/displacement shapes are not aligned")
        steps = displacement.shape[1]
        y = torch.linspace(-1.0, 1.0, height, device=source.device,
                           dtype=displacement.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=source.device,
                           dtype=displacement.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack((xx, yy), dim=-1).view(1, 1, height, width, 2)
        scale = displacement.new_tensor((
            2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)))
        grid = base + displacement * scale
        values = source.permute(0, 3, 1, 2).unsqueeze(1).expand(
            -1, steps, -1, -1, -1).reshape(
                batch * steps, channels, height, width)
        sampled = F.grid_sample(
            values.float(), grid.reshape(batch * steps, height, width, 2).float(),
            mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.reshape(batch, steps, channels, height, width).permute(
            0, 1, 3, 4, 2).to(source)

    @staticmethod
    def _pixel_shuffle(value: torch.Tensor, channels: int,
                       target_size: tuple[int, int]) -> torch.Tensor:
        batch, steps, packed, height, width = value.shape
        if packed != 4 * channels:
            raise ValueError("PixelShuffle channel contract changed")
        restored = F.pixel_shuffle(
            value.reshape(batch * steps, packed, height, width), 2)
        if restored.shape[-2:] != target_size:
            raise ValueError("PixelShuffle output resolution changed")
        return restored.reshape(batch, steps, channels, *target_size).permute(
            0, 1, 3, 4, 2)

    def forward_components(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_history(history)
        target_size = tuple(history.shape[2:4])
        decoded = self.decode_future(self.encode_history(history))

        raw_displacement = self._frame_apply(self.transport_head, decoded)
        displacement = self._pixel_shuffle(raw_displacement, 2, target_size)
        displacement = self.config.max_displacement * torch.tanh(displacement)
        if self._zero_displacement_override:
            displacement = torch.zeros_like(displacement)
        persistence = self._persistence(history)
        transported = (
            persistence.unsqueeze(1).expand(
                -1, self.config.out_steps, -1, -1, -1)
            if self._zero_displacement_override else
            self.warp(persistence, displacement))

        mean_feature = decoded.mean(dim=1, keepdim=True)
        mean_feature = self._frame_apply(self.mean_spatial, mean_feature)
        mean_delta = self._pixel_shuffle(
            self._frame_apply(self.mean_head, mean_feature),
            self.config.out_channels, target_size)[:, 0]
        mean_delta = self.config.residual_limit * torch.tanh(
            mean_delta / self.config.residual_limit)

        fluctuation = self._pixel_shuffle(
            self._frame_apply(self.fluctuation_head, decoded),
            self.config.out_channels, target_size)
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
        }

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_components(history)["prediction"]


def build_pyramid_causal3d_operator(
        interaction_mode: str = "causal_temporal_candidate",
        **overrides: object) -> PyramidCausal3DOperator:
    return PyramidCausal3DOperator(
        PyramidCausal3DOperatorConfig(**overrides), interaction_mode)
