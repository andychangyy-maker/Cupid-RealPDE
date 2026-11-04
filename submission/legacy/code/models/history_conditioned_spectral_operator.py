"""Small history-conditioned spectral neural operator for 20-to-20 flow.

The purpose of this module is deliberately narrow: distinguish a fixed
Fourier operator from an operator whose *spectral map* is continuously
conditioned on the observed trajectory.  It is not an output-side FiLM or a
mixture of separately decoded experts.  At each retained Fourier mode a
rank-four update to the channel-to-channel map is selected from a summary of
the input history.

All three discovery-screen arms instantiate exactly this topology:

``static_base``
    rank update is forward-zero (the static Fourier map only);
``static_rank``
    rank update has a learned, input-independent gate;
``conditional_rank``
    the same learned gate plus a history-dependent gate.

The inactive paths are still evaluated and multiplied by zero rather than
being removed or frozen, so all arms retain the same optimizer parameter
groups, weight decay, forward operators, and tensor shapes.  They do not use
a straight-through estimator: a control may not receive a ghost gradient from
a path absent in its own forward.  The scientific comparison is
conditional_rank minus static_rank; static_base only separates the benefit of
a rank correction from the benefit of conditioning it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


ARM_MODES = ("static_base", "static_rank", "conditional_rank")


@dataclass(frozen=True)
class HistoryConditionedSpectralConfig:
    in_steps: int = 20
    out_steps: int = 20
    channels: int = 3
    width: int = 32
    layers: int = 3
    modes_y: int = 8
    modes_x: int = 14
    rank: int = 4
    condition_hidden: int = 64
    groups: int = 8
    local_kernel: int = 3
    residual_limit: float = 4.0

    def validate(self) -> None:
        values = (self.in_steps, self.out_steps, self.channels, self.width,
                  self.layers, self.modes_y, self.modes_x, self.rank,
                  self.condition_hidden, self.groups, self.local_kernel)
        if any(value <= 0 for value in values):
            raise ValueError("architecture dimensions must be positive")
        if self.width % self.groups:
            raise ValueError("width must be divisible by groups")
        if self.local_kernel % 2 != 1:
            raise ValueError("local_kernel must be odd")
        if self.residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")


def _coordinates(reference: torch.Tensor) -> torch.Tensor:
    batch, _channels, height, width = reference.shape
    y = torch.linspace(-1.0, 1.0, height, device=reference.device,
                       dtype=reference.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=reference.device,
                       dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx)).unsqueeze(0).expand(batch, -1, -1, -1)


class ConditionGate(nn.Module):
    """Map a history feature map to one continuous rank gate per sample."""

    def __init__(self, width: int, hidden: int, rank: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, hidden)
        self.last = nn.Linear(hidden, rank)
        # A zero gate gives all arms exactly the same initial forward.  The
        # last bias still gets a first-step gradient, so this is not a frozen
        # conditioner or a permanent cold-start branch.
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        pooled = feature.mean(dim=(-2, -1))
        return self.last(F.silu(self.first(self.norm(pooled))))


class DynamicSpectralConv2d(nn.Module):
    """Static complex spectral convolution plus a gated rank update."""

    def __init__(self, width: int, modes_y: int, modes_x: int, rank: int):
        super().__init__()
        self.width, self.modes_y, self.modes_x, self.rank = (
            int(width), int(modes_y), int(modes_x), int(rank))
        scale = width ** -.5
        shape = (width, width, modes_y, modes_x)
        self.static_positive = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.static_negative = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        factor_shape = (rank, width, modes_y, modes_x)
        self.left_positive = nn.Parameter(scale * torch.randn(*factor_shape, dtype=torch.cfloat))
        self.right_positive = nn.Parameter(scale * torch.randn(*factor_shape, dtype=torch.cfloat))
        self.left_negative = nn.Parameter(scale * torch.randn(*factor_shape, dtype=torch.cfloat))
        self.right_negative = nn.Parameter(scale * torch.randn(*factor_shape, dtype=torch.cfloat))

    def _mix_modes(self, transformed: torch.Tensor, static: torch.Tensor,
                   left: torch.Tensor, right: torch.Tensor,
                   gate: torch.Tensor) -> torch.Tensor:
        static_out = torch.einsum("bcyx,ocyx->boyx", transformed, static)
        projected = torch.einsum("bcyx,rcyx->bryx", transformed, right)
        dynamic = torch.einsum("bryx,royx,br->boyx", projected, left,
                               gate.to(projected.dtype))
        return static_out + dynamic

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        if self.modes_y * 2 > height or self.modes_x > width // 2 + 1:
            raise ValueError("spectral modes exceed current grid")
        transformed = torch.fft.rfft2(x, dim=(-2, -1))
        output = torch.zeros_like(transformed)
        output[:, :, :self.modes_y, :self.modes_x] = self._mix_modes(
            transformed[:, :, :self.modes_y, :self.modes_x],
            self.static_positive, self.left_positive, self.right_positive, gate)
        output[:, :, -self.modes_y:, :self.modes_x] = self._mix_modes(
            transformed[:, :, -self.modes_y:, :self.modes_x],
            self.static_negative, self.left_negative, self.right_negative, gate)
        return torch.fft.irfft2(output, s=(height, width), dim=(-2, -1))


class ConditionedOperatorBlock(nn.Module):
    def __init__(self, config: HistoryConditionedSpectralConfig):
        super().__init__()
        width = config.width
        self.norm_operator = nn.GroupNorm(config.groups, width)
        self.spectral = DynamicSpectralConv2d(width, config.modes_y,
                                               config.modes_x, config.rank)
        self.local = nn.Sequential(
            nn.Conv2d(width, width, config.local_kernel,
                      padding=config.local_kernel // 2, groups=width,
                      bias=False),
            nn.Conv2d(width, width, 1),
        )
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm_mlp = nn.GroupNorm(config.groups, width)
        self.mlp = nn.Sequential(nn.Conv2d(width, 2 * width, 1), nn.GELU(),
                                 nn.Conv2d(2 * width, width, 1))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_operator(x)
        update = (self.spectral(normalized, gate) + self.local(normalized)
                  + self.pointwise(normalized))
        x = x + F.gelu(update)
        return x + self.mlp(self.norm_mlp(x))


class HistoryConditionedSpectralOperator(nn.Module):
    """Direct multi-horizon local+spectral neural operator.

    ``arm_mode`` changes only the gate expression.  All modules are present
    and executed in every mode, including the conditioner in the static arms.
    """

    def __init__(self, config: HistoryConditionedSpectralConfig,
                 arm_mode: str = "conditional_rank"):
        super().__init__()
        config.validate()
        if arm_mode not in ARM_MODES:
            raise ValueError(f"arm_mode must be one of {ARM_MODES}")
        self.config, self.arm_mode = config, arm_mode
        history_channels = config.in_steps * config.channels
        # Four recent u/v differences let the condition see an explicit local
        # velocity tendency without giving it target or metadata information.
        history_channels += 4 * min(2, config.channels) + 2
        self.lift = nn.Sequential(nn.Conv2d(history_channels, config.width, 1),
                                  nn.GELU(),
                                  nn.Conv2d(config.width, config.width, 3, padding=1))
        self.conditioner = ConditionGate(config.width, config.condition_hidden,
                                         config.rank)
        self.static_gate = nn.Parameter(torch.zeros(config.rank))
        self.blocks = nn.ModuleList(ConditionedOperatorBlock(config)
                                    for _ in range(config.layers))
        self.final_norm = nn.GroupNorm(config.groups, config.width)
        self.output = nn.Conv2d(config.width,
                                config.out_steps * config.channels, 1)
        nn.init.normal_(self.output.weight, std=1.0e-3)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("persistence_scale", torch.ones(config.channels))
        self.register_buffer("persistence_bias", torch.zeros(config.channels))

    def architecture_config(self) -> dict:
        return {"architecture": "history_conditioned_spectral_operator",
                "arm_mode": self.arm_mode, **asdict(self.config)}

    @torch.no_grad()
    def set_normalizer_statistics(self, mean_input: torch.Tensor,
                                  mean_target: torch.Tensor,
                                  std_input: torch.Tensor,
                                  std_target: torch.Tensor) -> None:
        safe_target = torch.as_tensor(std_target, device=self.persistence_scale.device).clamp_min(1e-8)
        input_std = torch.as_tensor(std_input, device=self.persistence_scale.device)
        input_mean = torch.as_tensor(mean_input, device=self.persistence_scale.device)
        target_mean = torch.as_tensor(mean_target, device=self.persistence_scale.device)
        self.persistence_scale.copy_(input_std[:self.config.channels] / safe_target[:self.config.channels])
        self.persistence_bias.copy_((input_mean[:self.config.channels] - target_mean[:self.config.channels]) /
                                    safe_target[:self.config.channels])

    def _features(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or history.shape[1] != self.config.in_steps or history.shape[-1] != self.config.channels:
            raise ValueError("expected history [B,20,H,W,channels]")
        batch, steps, height, width, channels = history.shape
        flattened = history.permute(0, 1, 4, 2, 3).reshape(batch, steps * channels, height, width)
        velocity = history[..., :min(2, channels)]
        differences = velocity[:, 1:] - velocity[:, :-1]
        recent = differences[:, -4:].permute(0, 1, 4, 2, 3).reshape(
            batch, 4 * min(2, channels), height, width)
        return torch.cat((flattened, recent, _coordinates(flattened)), dim=1)

    def _gate(self, lifted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition_gate = self.conditioner(lifted)
        static = self.static_gate.view(1, -1).expand_as(condition_gate)
        if self.arm_mode == "conditional_rank":
            raw_gate = static + condition_gate
        elif self.arm_mode == "static_rank":
            # The conditioner still executes, but neither it nor its input may
            # receive a ghost gradient from a path absent at deployment.
            raw_gate = static + 0.0 * condition_gate
        else:
            raw_gate = 0.0 * static + 0.0 * condition_gate
        return torch.tanh(raw_gate), condition_gate

    def forward_components(
            self, history: torch.Tensor,
            gate_override: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        lifted = self.lift(self._features(history))
        gate, condition_gate = self._gate(lifted)
        static_gate = torch.tanh(self.static_gate).view(1, -1).expand_as(gate)
        if gate_override is not None:
            if gate_override.shape != gate.shape:
                raise ValueError("gate_override shape mismatch")
            gate = gate_override.to(gate)
        state = lifted
        for block in self.blocks:
            state = block(state, gate)
        residual = self.output(self.final_norm(state))
        batch, _channels, height, width = residual.shape
        residual = residual.reshape(batch, self.config.out_steps, self.config.channels,
                                    height, width).permute(0, 1, 3, 4, 2)
        residual = self.config.residual_limit * torch.tanh(residual / self.config.residual_limit)
        persistence = (history[:, -1] * self.persistence_scale.view(1, 1, 1, -1)
                       + self.persistence_bias.view(1, 1, 1, -1))
        return {"prediction": persistence.unsqueeze(1) + residual,
                "residual": residual, "persistence": persistence,
                "gate": gate, "static_gate": static_gate,
                "condition_gate": condition_gate}

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_components(history)["prediction"]
