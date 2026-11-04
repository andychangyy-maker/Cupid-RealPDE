"""Hierarchical Axial Temporal Operator (HATO) for direct forecasting.

HATO keeps the twenty observed snapshots as an explicit axis until the
future-query decoder.  The paired ``spatial_window_control`` has exactly the
same tensors as the temporal candidate, but replaces every length-twenty
history sequence with a 4x5 within-frame spatial sequence.  Both arms pad to
the same number of sequences before attention and crop afterwards, so their
attention modules see the same sequence length and batch size.

This is deliberately point-only.  Native uncertainty is a separate research
question and must not be silently coupled to this backbone screen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .unet2d import ForecastDistributionMixin, _group_norm


HATO_MODES = ("temporal_candidate", "spatial_window_control")


@dataclass(frozen=True)
class HATOConfig:
    """Static HATO dimensions.

    The released Track 1 interface is twenty history frames to twenty future
    frames.  Keeping that contract explicit prevents a control from gaining a
    different attention sequence length through an accidental configuration.
    """

    in_steps: int = 20
    out_steps: int = 20
    in_channels: int = 3
    out_channels: int = 3
    e0_channels: int = 96
    e1_channels: int = 192
    e2_channels: int = 256
    e1_blocks: int = 2
    e2_blocks: int = 2
    heads_e1: int = 6
    heads_e2: int = 8
    mlp_ratio: float = 4.0
    groups: int = 8
    def validate(self) -> None:
        if self.in_steps != 20 or self.out_steps != 20:
            raise ValueError("HATO is frozen to twenty history and future steps")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("channel counts must be positive")
        if min(self.e0_channels, self.e1_channels, self.e2_channels) <= 0:
            raise ValueError("pyramid widths must be positive")
        if self.e1_blocks != 2 or self.e2_blocks != 2:
            raise ValueError("HATO screen freezes two blocks at E1 and E2")
        if self.e1_channels % self.heads_e1 or self.e2_channels % self.heads_e2:
            raise ValueError("attention widths must divide their head counts")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.groups <= 0:
            raise ValueError("groups must be positive")


# Long-form alias retained for documents written before the trainer contract
# was frozen.  New callers should use HATOConfig.
HierarchicalAxialTemporalOperatorConfig = HATOConfig


class FramewiseSpatialResidual(nn.Module):
    """A local spatial residual applied independently to every frame."""

    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            _group_norm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            _group_norm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5:
            raise ValueError("framewise spatial residual requires [B,T,C,H,W]")
        batch, steps, channels, height, width = values.shape
        flat = values.reshape(batch * steps, channels, height, width)
        return (flat + self.body(flat)).reshape(
            batch, steps, channels, height, width)


class AxialHistoryBlock(nn.Module):
    """One matched length-20 attention, MLP, and local spatial update."""

    window_height = 4
    window_width = 5

    def __init__(self, channels: int, heads: int, mlp_ratio: float,
                 groups: int, mode: str):
        super().__init__()
        if mode not in HATO_MODES:
            raise ValueError(f"unsupported HATO mode {mode!r}")
        self.channels = int(channels)
        self.mode = str(mode)
        self.norm_attention = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(channels)
        hidden = int(round(channels * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels))
        self.spatial = FramewiseSpatialResidual(channels, groups)

    @classmethod
    def padded_width(cls, width: int) -> int:
        return ((int(width) + cls.window_width - 1) // cls.window_width
                * cls.window_width)

    @classmethod
    def padded_height(cls, height: int) -> int:
        return ((int(height) + cls.window_height - 1) // cls.window_height
                * cls.window_height)

    def _temporal_sequences(self, values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch, steps, channels, height, width = values.shape
        padded_height = self.padded_height(height)
        padded_width = self.padded_width(width)
        padded = F.pad(values, (0, padded_width - width, 0, padded_height - height))
        sequences = padded.permute(0, 3, 4, 1, 2).reshape(
            batch * padded_height * padded_width, steps, channels)
        return sequences, padded_height, padded_width

    def _spatial_sequences(self, values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch, steps, channels, height, width = values.shape
        padded_height = self.padded_height(height)
        padded_width = self.padded_width(width)
        padded = F.pad(values, (0, padded_width - width, 0, padded_height - height))
        # [B,T,C,H,W] -> [B,T,H/4,W/5,4*5,C].  The sequence count is
        # B*T*(H/4)*(Wpad/5) == B*H*Wpad because T=20 and 4*5=20.
        sequences = padded.reshape(
            batch, steps, channels, padded_height // 4, 4, padded_width // 5, 5
        ).permute(0, 1, 3, 5, 4, 6, 2).reshape(
            batch * steps * (padded_height // 4) * (padded_width // 5),
            self.window_height * self.window_width, channels)
        return sequences, padded_height, padded_width

    def _restore_temporal(self, sequences: torch.Tensor, batch: int,
                          height: int, width: int,
                          padded_height: int, padded_width: int) -> torch.Tensor:
        return sequences.reshape(batch, padded_height, padded_width, 20, self.channels).permute(
            0, 3, 4, 1, 2)[..., :height, :width]

    def _restore_spatial(self, sequences: torch.Tensor, batch: int,
                         height: int, width: int,
                         padded_height: int, padded_width: int) -> torch.Tensor:
        restored = sequences.reshape(
            batch, 20, padded_height // 4, padded_width // 5, 4, 5, self.channels
        ).permute(0, 1, 6, 2, 4, 3, 5).reshape(
            batch, 20, self.channels, padded_height, padded_width)
        return restored[..., :height, :width]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5 or values.shape[1] != 20:
            raise ValueError("HATO axial blocks require [B,20,C,H,W]")
        batch, _steps, channels, height, width = values.shape
        if channels != self.channels:
            raise ValueError("HATO axial channel width changed")
        if self.mode == "temporal_candidate":
            sequences, padded_height, padded_width = self._temporal_sequences(values)
        else:
            sequences, padded_height, padded_width = self._spatial_sequences(values)
        normalized = self.norm_attention(sequences)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     need_weights=False)
        sequences = sequences + attended
        sequences = sequences + self.mlp(self.norm_mlp(sequences))
        if self.mode == "temporal_candidate":
            updated = self._restore_temporal(
                sequences, batch, height, width, padded_height, padded_width)
        else:
            updated = self._restore_spatial(
                sequences, batch, height, width, padded_height, padded_width)
        return self.spatial(updated)


class FutureCrossAttention(nn.Module):
    """Twenty learned future queries cross-attending to all history frames."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.channels = int(channels)
        self.future_queries = nn.Parameter(torch.empty(20, channels))
        nn.init.normal_(self.future_queries, std=0.02)
        self.query_norm = nn.LayerNorm(channels)
        self.history_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(channels)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or history.shape[1] != 20:
            raise ValueError("future cross-attention requires [B,20,C,H,W]")
        batch, steps, channels, height, width = history.shape
        if channels != self.channels:
            raise ValueError("future cross-attention channel width changed")
        memory = history.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, steps, channels)
        query = self.future_queries.unsqueeze(0).expand(
            memory.shape[0], -1, -1)
        future, _ = self.attention(
            self.query_norm(query), self.history_norm(memory),
            self.history_norm(memory), need_weights=False)
        future = self.output_norm(future + query)
        return future.reshape(batch, height, width, 20, channels).permute(
            0, 3, 4, 1, 2)


class FutureAxisBlock(nn.Module):
    """A future-axis attention plus independent spatial residual mixer."""

    def __init__(self, channels: int, heads: int, mlp_ratio: float,
                 groups: int):
        super().__init__()
        self.channels = int(channels)
        self.norm_attention = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(channels)
        hidden = int(round(channels * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels))
        self.spatial = FramewiseSpatialResidual(channels, groups)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = values.shape
        if steps != 20 or channels != self.channels:
            raise ValueError("future-axis shape changed")
        sequences = values.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, steps, channels)
        normalized = self.norm_attention(sequences)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     need_weights=False)
        sequences = sequences + attended
        sequences = sequences + self.mlp(self.norm_mlp(sequences))
        updated = sequences.reshape(batch, height, width, steps, channels).permute(
            0, 3, 4, 1, 2)
        return self.spatial(updated)


class HierarchicalAxialTemporalOperator(nn.Module, ForecastDistributionMixin):
    """Direct future-query HATO point forecaster."""

    def __init__(self, config: HATOConfig,
                 interaction_mode: str = "temporal_candidate"):
        super().__init__()
        config.validate()
        if interaction_mode not in HATO_MODES:
            raise ValueError(f"interaction_mode must be one of {HATO_MODES}")
        self.config = config
        self.interaction_mode = str(interaction_mode)
        # E0 is intentionally /2, rather than a costly full-resolution
        # future tensor.  E1/E2 are /4 and /8 at the 64x128 Sim resolution.
        self.e0 = nn.Sequential(
            nn.Conv2d(config.in_channels + 2, config.e0_channels, 3, stride=2,
                      padding=1),
            nn.SiLU(),
            nn.Conv2d(config.e0_channels, config.e0_channels, 3, padding=1),
        )
        self.down1 = nn.Sequential(
            nn.SiLU(), nn.Conv2d(config.e0_channels, config.e1_channels,
                                 3, stride=2, padding=1))
        self.down2 = nn.Sequential(
            nn.SiLU(), nn.Conv2d(config.e1_channels, config.e2_channels,
                                 3, stride=2, padding=1))
        self.e1_blocks = nn.ModuleList(
            AxialHistoryBlock(config.e1_channels, config.heads_e1,
                              config.mlp_ratio, config.groups,
                              self.interaction_mode)
            for _ in range(config.e1_blocks))
        self.e2_blocks = nn.ModuleList(
            AxialHistoryBlock(config.e2_channels, config.heads_e2,
                              config.mlp_ratio, config.groups,
                              self.interaction_mode)
            for _ in range(config.e2_blocks))
        self.e1_future = FutureCrossAttention(config.e1_channels, config.heads_e1)
        self.e2_future = FutureCrossAttention(config.e2_channels, config.heads_e2)
        self.coarse_to_fine = nn.Conv2d(config.e2_channels, config.e1_channels,
                                        1)
        self.future_fuse = nn.Sequential(
            nn.Conv2d(2 * config.e1_channels, config.e1_channels, 3, padding=1),
            nn.SiLU(), nn.Conv2d(config.e1_channels, config.e1_channels, 3, padding=1),
        )
        self.future_axis = FutureAxisBlock(
            config.e1_channels, config.heads_e1, config.mlp_ratio, config.groups)
        self.up_to_e0 = nn.ConvTranspose2d(
            config.e1_channels, config.e0_channels, 2, stride=2)
        self.output_fuse = nn.Sequential(
            nn.Conv2d(2 * config.e0_channels, config.e0_channels, 3, padding=1),
            nn.SiLU(), nn.Conv2d(config.e0_channels, config.e0_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.output_spatial = FramewiseSpatialResidual(
            config.e0_channels, config.groups)
        # Separate heads give the mean shift and horizon-specific residual
        # different representations; centering the latter enforces its role.
        self.mean_head = nn.Conv2d(
            config.e0_channels, config.out_channels * 4, 1)
        self.fluctuation_head = nn.Conv2d(
            config.e0_channels, config.out_channels * 4, 1)
        # Start near persistence but do not zero the heads: every upstream
        # attention and time-embedding tensor receives a first-step gradient.
        for head in (self.mean_head, self.fluctuation_head):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)
        self.e1_observed_time = nn.Parameter(torch.empty(20, config.e1_channels))
        self.e2_observed_time = nn.Parameter(torch.empty(20, config.e2_channels))
        nn.init.normal_(self.e1_observed_time, std=0.02)
        nn.init.normal_(self.e2_observed_time, std=0.02)
        self._register_forecast_buffers()

    def architecture_config(self) -> dict:
        return {"architecture": "hierarchical_axial_temporal_operator",
                "interaction_mode": self.interaction_mode,
                **asdict(self.config)}

    @staticmethod
    def _apply_frame_module(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = values.shape
        output = module(values.reshape(batch * steps, channels, height, width))
        return output.reshape(batch, steps, output.shape[1], output.shape[2], output.shape[3])

    @staticmethod
    def _frame_coordinates(x: torch.Tensor) -> torch.Tensor:
        """Dynamic normalized y/x coordinates for each observed frame."""
        batch, steps, height, width, _channels = x.shape
        y = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        z = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, z, indexing="ij")
        coordinate = torch.stack((yy, xx), dim=0).view(1, 1, 2, height, width)
        return coordinate.expand(batch, steps, -1, -1, -1)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5 or x.shape[1] != 20 or x.shape[-1] != self.config.in_channels:
            raise ValueError("HATO requires [B,20,H,W,in_channels]")
        height, width = x.shape[2:4]
        if height % 8 or width % 8:
            raise ValueError("HATO requires H/W divisible by 8")

    @staticmethod
    def _pixel_shuffle_future(values: torch.Tensor, channels: int,
                              height: int, width: int) -> torch.Tensor:
        """Restore a [B,T,4C,H/2,W/2] head to [B,T,H,W,C]."""
        batch, steps, packed, low_height, low_width = values.shape
        if packed != 4 * channels:
            raise ValueError("HATO PixelShuffle channels changed")
        restored = F.pixel_shuffle(values.reshape(
            batch * steps, packed, low_height, low_width), 2)
        return restored.reshape(batch, steps, channels, height, width).permute(
            0, 1, 3, 4, 2)

    def forward_components(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return forecast decomposition for loss/debugging without UQ heads."""
        self._validate_input(x)
        frames = x.permute(0, 1, 4, 2, 3)
        e0 = self._apply_frame_module(
            self.e0, torch.cat((frames, self._frame_coordinates(x)), dim=2))
        e1 = self._apply_frame_module(self.down1, e0)
        # Observed-time embeddings are shared by the paired modes.  They let
        # temporal attention distinguish frame order while leaving the
        # spatial-window control's parameter tensors exactly identical.
        e1 = e1 + self.e1_observed_time.view(1, 20, -1, 1, 1)
        for block in self.e1_blocks:
            e1 = block(e1)
        # The coarse stage consumes the temporally updated E1 hierarchy,
        # rather than a stale pre-attention downsample.
        e2 = self._apply_frame_module(self.down2, e1)
        e2 = e2 + self.e2_observed_time.view(1, 20, -1, 1, 1)
        for block in self.e2_blocks:
            e2 = block(e2)
        fine_future = self.e1_future(e1)
        coarse_future = self.e2_future(e2)
        coarse_up = self._apply_frame_module(self.coarse_to_fine, coarse_future)
        batch, steps, channels, height, width = coarse_up.shape
        coarse_up = F.interpolate(coarse_up.reshape(batch * steps, channels, height, width),
                                  size=fine_future.shape[-2:], mode="bilinear",
                                  align_corners=False).reshape(
            batch, steps, channels, *fine_future.shape[-2:])
        fused = self._apply_frame_module(
            self.future_fuse, torch.cat((fine_future, coarse_up), dim=2))
        fused = self.future_axis(fused)
        up = self._apply_frame_module(self.up_to_e0, fused)
        latest_e0 = e0[:, -1:].expand(-1, 20, -1, -1, -1)
        decoded = self._apply_frame_module(
            self.output_fuse, torch.cat((up, latest_e0), dim=2))
        decoded = self.output_spatial(decoded)
        mean_feature = decoded.mean(dim=1)
        mean_low = self.mean_head(mean_feature)
        mean_delta = F.pixel_shuffle(mean_low, 2).permute(0, 2, 3, 1).unsqueeze(1)
        fluctuation = self._pixel_shuffle_future(
            self._apply_frame_module(self.fluctuation_head, decoded),
            self.config.out_channels, x.shape[2], x.shape[3])
        fluctuation = fluctuation - fluctuation.mean(dim=1, keepdim=True)
        raw = mean_delta + fluctuation
        persistence = self._persistence(x).unsqueeze(1)
        prediction = persistence + mean_delta + fluctuation
        return {
            "prediction": prediction,
            "persistence": persistence,
            "mean_delta": mean_delta,
            "fluctuation": fluctuation,
            "raw_residual": raw,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x)["prediction"]


def build_hierarchical_axial_temporal_operator(
        **overrides: object) -> HierarchicalAxialTemporalOperator:
    interaction_mode = str(overrides.pop("interaction_mode", "temporal_candidate"))
    return HierarchicalAxialTemporalOperator(
        HATOConfig(**overrides), interaction_mode)
