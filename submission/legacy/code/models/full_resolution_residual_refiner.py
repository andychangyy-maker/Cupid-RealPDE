"""A matched full-resolution versus phase-packed future residual refiner.

The two execution modes own the *same* convolutional graph.  ``phase_pack_2``
only moves the four spatial parity phases into the batch dimension before that
graph and reverses the permutation afterwards.  Consequently its feature
element count and convolutional MAC count are identical to a full-resolution
execution, while its local receptive field represents a deliberately different
two-pixel phase lattice.  There is no batch reduction or normalization in the
module, so the packing cannot create a path between original batch examples.

The output projection is zero initialized.  Thus update zero is an exact
identity for both modes, and the paired modes are bitwise equal before the
output projection has received an update.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


REFINER_EXECUTION_MODES = ("full_resolution", "phase_pack_2")


@dataclass(frozen=True)
class FullResolutionResidualRefinerConfig:
    """Fixed small residual-refiner architecture and its execution mode."""

    steps: int = 20
    channels: int = 3
    hidden_channels: int = 32
    blocks: int = 3
    residual_limit: float = 4.0
    execution_mode: str = "full_resolution"

    def validate(self) -> None:
        if any(value <= 0 for value in (
                self.steps, self.channels, self.hidden_channels, self.blocks)):
            raise ValueError("refiner dimensions must be positive")
        if self.residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")
        if self.execution_mode not in REFINER_EXECUTION_MODES:
            raise ValueError(
                "execution_mode must be one of "
                f"{REFINER_EXECUTION_MODES}")


def _validate_future(future: torch.Tensor, *, steps: int, channels: int) -> None:
    if (future.ndim != 5 or future.shape[1] != steps
            or future.shape[-1] != channels):
        raise ValueError(
            f"expected future [B,{steps},H,W,{channels}], got "
            f"{tuple(future.shape)}")


def phase_pack_2(future: torch.Tensor) -> torch.Tensor:
    """Move the four ``(y mod 2, x mod 2)`` phases into the batch dimension."""
    if future.ndim != 5:
        raise ValueError("phase_pack_2 requires future [B,T,H,W,C]")
    batch, steps, height, width, channels = future.shape
    if height % 2 or width % 2:
        raise ValueError("phase_pack_2 requires even H and W")
    return future.reshape(batch, steps, height // 2, 2, width // 2, 2, channels).permute(
        0, 3, 5, 1, 2, 4, 6).reshape(
            batch * 4, steps, height // 2, width // 2, channels)


def phase_unpack_2(packed: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    """Invert :func:`phase_pack_2` without mixing packed batch entries."""
    if packed.ndim != 5:
        raise ValueError("phase_unpack_2 requires packed [B*4,T,H,W,C]")
    packed_batch, steps, height, width, channels = packed.shape
    if batch_size <= 0 or packed_batch != 4 * batch_size:
        raise ValueError("packed batch must equal four times batch_size")
    return packed.reshape(batch_size, 2, 2, steps, height, width, channels).permute(
        0, 3, 4, 1, 5, 2, 6).reshape(
            batch_size, steps, height * 2, width * 2, channels)


class _PerPositionChannelLayerNorm(nn.Module):
    """LayerNorm over channels, separately at every batch/time/pixel position."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class _ResidualRefinerBlock(nn.Module):
    """Per-position channel normalization followed by local separable Conv3d."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = _PerPositionChannelLayerNorm(channels)
        self.depthwise = nn.Conv3d(
            channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv3d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(F.silu(self.norm(value)))
        update = self.pointwise(F.silu(update))
        return value + update


class FullResolutionResidualRefiner(nn.Module):
    """Bounded residual correction over a complete ``[B,T,H,W,3]`` future."""

    def __init__(self, config: FullResolutionResidualRefinerConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.stem = nn.Conv3d(
            config.channels, config.hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            _ResidualRefinerBlock(config.hidden_channels)
            for _ in range(config.blocks))
        self.output = nn.Conv3d(config.hidden_channels, config.channels, 1)
        # This is intentionally exact zero, rather than a small random head.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_execution_contract: dict[str, object] | None = None

    def architecture_config(self) -> dict[str, object]:
        return {"architecture": "full_resolution_residual_refiner", **asdict(self.config)}

    def execution_contract(self, future: torch.Tensor) -> dict[str, object]:
        """Return element and convolutional-MAC accounting for this execution."""
        _validate_future(
            future, steps=self.config.steps, channels=self.config.channels)
        batch, steps, height, width, channels = future.shape
        if self.config.execution_mode == "phase_pack_2":
            if height % 2 or width % 2:
                raise ValueError("phase_pack_2 requires even H and W")
            effective_shape = (batch * 4, steps, height // 2, width // 2, channels)
        else:
            effective_shape = tuple(future.shape)
        effective_elements = batch * steps * height * width
        hidden_elements = effective_elements * self.config.hidden_channels
        output_elements = effective_elements * channels
        # A Conv3d MAC is one multiply-accumulate for one input/output-channel
        # pair and one kernel position.  Biases, activations and LayerNorm are
        # deliberately excluded from this conventional convolutional budget.
        stem_macs = hidden_elements * channels * 27
        block_macs = hidden_elements * (27 + self.config.hidden_channels)
        output_macs = output_elements * self.config.hidden_channels
        return {
            "execution_mode": self.config.execution_mode,
            "canonical_shape": tuple(future.shape),
            "effective_shape": effective_shape,
            "feature_elements": {
                "input": output_elements,
                "stem_hidden": hidden_elements,
                "per_residual_block": hidden_elements,
                "output": output_elements,
            },
            "theoretical_macs": {
                "stem": stem_macs,
                "residual_blocks": self.config.blocks * block_macs,
                "output": output_macs,
                "total": stem_macs + self.config.blocks * block_macs + output_macs,
            },
        }

    def _raw_correction(self, future: torch.Tensor) -> torch.Tensor:
        value = self.stem(future.permute(0, 4, 1, 2, 3))
        for block in self.blocks:
            value = block(value)
        return self.output(F.silu(value)).permute(0, 2, 3, 4, 1)

    def correction(self, future: torch.Tensor) -> torch.Tensor:
        """Return the bounded additive correction in normalized target units."""
        contract = self.execution_contract(future)
        if self.config.execution_mode == "phase_pack_2":
            raw = phase_unpack_2(
                self._raw_correction(phase_pack_2(future)),
                batch_size=future.shape[0])
        else:
            raw = self._raw_correction(future)
        self.last_execution_contract = contract
        return self.config.residual_limit * torch.tanh(
            raw / self.config.residual_limit)

    def forward(self, future: torch.Tensor) -> torch.Tensor:
        return future + self.correction(future)

