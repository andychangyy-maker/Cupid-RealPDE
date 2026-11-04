"""Causal predictive-state forecaster with an exact spatial-band decoder.

This is deliberately a small discovery model.  Its only experimental switch
is whether a future state is accumulated causally or recomputed from the
observed terminal state for every horizon.  Both modes execute the same
transition cell the same number of times and feed the same three-band decoder.

There is no direct flattened-history output head.  The observed state may
modulate each decoder only through FiLM; every predicted residual originates
from that horizon's latent state.  A same-checkpoint independent-rollout
ablation isolates state accumulation, while a frozen-state result is retained
only as an auxiliary diagnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


ROLLOUT_MODES = ("cumulative", "independent")
STATE_MODES = (*ROLLOUT_MODES, "frozen")


@dataclass(frozen=True)
class CausalStateMultibandConfig:
    in_steps: int = 20
    out_steps: int = 20
    channels: int = 3
    state_channels: int = 96
    decoder_channels: int = 64
    context_channels: int = 96
    groups: int = 8

    def validate(self) -> None:
        values = (self.in_steps, self.out_steps, self.channels,
                  self.state_channels, self.decoder_channels,
                  self.context_channels, self.groups)
        if any(value <= 0 for value in values):
            raise ValueError("architecture dimensions must be positive")
        if any(value % self.groups for value in (self.state_channels,
                                                  self.decoder_channels)):
            raise ValueError("state and decoder channels must divide groups")


def _coordinates(reference: torch.Tensor) -> torch.Tensor:
    batch, _channels, height, width = reference.shape
    y = torch.linspace(-1., 1., height, device=reference.device,
                       dtype=reference.dtype)
    x = torch.linspace(-1., 1., width, device=reference.device,
                       dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx)).unsqueeze(0).expand(batch, -1, -1, -1)


class FilmResidual(nn.Module):
    """A local residual block whose only auxiliary path is affine FiLM."""

    def __init__(self, channels: int, condition_channels: int, groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.film = nn.Conv2d(condition_channels, 2 * channels, 1)
        self.first = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.second = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, bias = self.film(condition).chunk(2, dim=1)
        normalized = self.norm(state) * (1. + scale) + bias
        update = self.second(F.silu(self.first(F.silu(normalized))))
        return state + update


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)
        self.candidate = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, previous: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((previous, observed), dim=1)).chunk(2, dim=1)
        reset, update = torch.sigmoid(reset), torch.sigmoid(update)
        proposal = torch.tanh(self.candidate(torch.cat((reset * previous, observed), dim=1)))
        return (1. - update) * previous + update * proposal


class StateTransition(nn.Module):
    """One explicit, history-conditioned discrete future state step."""

    def __init__(self, config: CausalStateMultibandConfig):
        super().__init__()
        channels = config.state_channels
        self.horizon = nn.Embedding(config.out_steps, config.context_channels)
        self.context = nn.Sequential(nn.Linear(channels, config.context_channels), nn.SiLU(),
                                     nn.Linear(config.context_channels, config.context_channels))
        self.first = FilmResidual(channels, config.context_channels, config.groups)
        self.second = FilmResidual(channels, config.context_channels, config.groups)

    def condition(self, terminal_state: torch.Tensor, horizon: int) -> torch.Tensor:
        pooled = terminal_state.mean(dim=(-2, -1))
        vector = self.context(pooled) + self.horizon.weight[horizon].view(1, -1)
        return vector.unsqueeze(-1).unsqueeze(-1)

    def forward(self, state: torch.Tensor, terminal_state: torch.Tensor,
                horizon: int) -> torch.Tensor:
        condition = self.condition(terminal_state, horizon)
        delta = self.second(self.first(state, condition), condition) - state
        return state + delta


class BandDecoder(nn.Module):
    """Decode a state into low/mid/high exact average-pool residual bands."""

    def __init__(self, config: CausalStateMultibandConfig):
        super().__init__()
        state, hidden = config.state_channels, config.decoder_channels
        self.low_in = nn.Conv2d(state, hidden, 1)
        self.mid_in = nn.Conv2d(state, hidden, 1)
        self.high_in = nn.Conv2d(state, hidden, 1)
        self.low = FilmResidual(hidden, state, config.groups)
        self.mid = FilmResidual(hidden, state, config.groups)
        self.high = FilmResidual(hidden, state, config.groups)
        self.low_out = nn.Conv2d(hidden, config.channels, 1)
        self.mid_out = nn.Conv2d(hidden, config.channels, 1)
        self.high_out = nn.Conv2d(hidden, config.channels, 1)
        # A small, shared random readout lets the transition receive gradients
        # at update 1.  Both rollout arms are still bitwise same-state; exact
        # persistence would silence every upstream path on that first step.
        for layer in (self.low_out, self.mid_out, self.high_out):
            nn.init.normal_(layer.weight, std=1.0e-3)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _up(value: torch.Tensor, scale: int) -> torch.Tensor:
        return F.interpolate(value, scale_factor=scale, mode="nearest")

    @classmethod
    def _mid_band(cls, native: torch.Tensor) -> torch.Tensor:
        # This is evaluated at the native 16x32 scale, *before* expansion to
        # 32x64.  Thus avg_pool4(full_mid)==0, rather than accidentally
        # cancelling the entire branch by pool-after-nearest-upsample.
        coarse = F.avg_pool2d(native, 2, stride=2)
        return native - cls._up(coarse, 2)

    @classmethod
    def _high_band(cls, native: torch.Tensor) -> torch.Tensor:
        coarse = F.avg_pool2d(native, 2, stride=2)
        return native - cls._up(coarse, 2)

    def forward(self, future_state: torch.Tensor,
                terminal_state: torch.Tensor) -> dict[str, torch.Tensor]:
        low_feature = self.low_in(future_state)
        low = self._up(self.low_out(self.low(low_feature, terminal_state)), 4)

        state16 = self._up(future_state, 2)
        condition16 = self._up(terminal_state, 2)
        mid16 = self.mid_out(self.mid(self.mid_in(state16), condition16))
        mid = self._up(self._mid_band(mid16), 2)

        state32 = self._up(future_state, 4)
        condition32 = self._up(terminal_state, 4)
        high_native = self.high_out(self.high(self.high_in(state32), condition32))
        high = self._high_band(high_native)
        return {"low": low, "mid": mid, "high": high,
                "residual": low + mid + high}


class CausalStateMultibandForecaster(nn.Module):
    """Matched cumulative-state and independent-state forecasting arms."""

    def __init__(self, config: CausalStateMultibandConfig,
                 rollout_mode: str = "cumulative"):
        super().__init__()
        config.validate()
        if rollout_mode not in ROLLOUT_MODES:
            raise ValueError(f"rollout_mode must be one of {ROLLOUT_MODES}")
        self.config, self.rollout_mode = config, rollout_mode
        self.frame_stem = nn.Sequential(
            nn.Conv2d(config.channels + 2, config.state_channels, 5, stride=4, padding=2),
            nn.SiLU(), nn.Conv2d(config.state_channels, config.state_channels, 3, padding=1))
        self.history_cell = ConvGRUCell(config.state_channels)
        self.transition = StateTransition(config)
        self.decoder = BandDecoder(config)
        self.register_buffer("persistence_scale", torch.ones(config.channels))
        self.register_buffer("persistence_bias", torch.zeros(config.channels))

    def architecture_config(self) -> dict[str, object]:
        return {"architecture": "causal_state_multiband", "rollout_mode": self.rollout_mode,
                **asdict(self.config)}

    @torch.no_grad()
    def set_normalizer_statistics(self, mean_input: torch.Tensor, mean_target: torch.Tensor,
                                  std_input: torch.Tensor, std_target: torch.Tensor) -> None:
        target_std = torch.as_tensor(std_target, device=self.persistence_scale.device).clamp_min(1e-8)
        input_std = torch.as_tensor(std_input, device=self.persistence_scale.device)
        input_mean = torch.as_tensor(mean_input, device=self.persistence_scale.device)
        target_mean = torch.as_tensor(mean_target, device=self.persistence_scale.device)
        self.persistence_scale.copy_(input_std[:self.config.channels] / target_std[:self.config.channels])
        self.persistence_bias.copy_((input_mean[:self.config.channels] - target_mean[:self.config.channels]) / target_std[:self.config.channels])

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or history.shape[1] != self.config.in_steps or history.shape[-1] != self.config.channels:
            raise ValueError("expected history [B,20,H,W,channels]")
        batch, steps, height, width, channels = history.shape
        frames = history.permute(0, 1, 4, 2, 3).reshape(batch * steps, channels, height, width)
        encoded = self.frame_stem(torch.cat((frames, _coordinates(frames)), dim=1))
        encoded = encoded.reshape(batch, steps, self.config.state_channels, height // 4, width // 4)
        state = torch.zeros_like(encoded[:, 0])
        for observed in encoded.unbind(dim=1):
            state = self.history_cell(state, observed)
        return state

    def future_states(self, terminal_state: torch.Tensor, state_mode: str | None = None) -> torch.Tensor:
        mode = self.rollout_mode if state_mode is None else state_mode
        if mode not in STATE_MODES:
            raise ValueError(f"state_mode must be one of {STATE_MODES}")
        state = terminal_state
        result = []
        for horizon in range(self.config.out_steps):
            if mode == "cumulative":
                state = self.transition(state, terminal_state, horizon)
            elif mode == "independent":
                state = self.transition(terminal_state, terminal_state, horizon)
            else:
                state = terminal_state
            result.append(state)
        return torch.stack(result, dim=1)

    def forward_components(self, history: torch.Tensor,
                           state_mode: str | None = None) -> dict[str, torch.Tensor]:
        terminal = self.encode_history(history)
        states = self.future_states(terminal, state_mode)
        decoded = [self.decoder(state, terminal) for state in states.unbind(dim=1)]
        # Internal decoder maps are BCHW; the public forecasting contract is
        # BTHWC, matching the starting-kit normalizer and scorer.
        components = {name: torch.stack([value[name] for value in decoded], dim=1).permute(0, 1, 3, 4, 2)
                      for name in ("low", "mid", "high", "residual")}
        persistence = (history[:, -1] * self.persistence_scale.view(1, 1, 1, -1)
                       + self.persistence_bias.view(1, 1, 1, -1))
        components.update({"prediction": persistence.unsqueeze(1) + components["residual"],
                           "persistence": persistence, "terminal_state": terminal,
                           "future_states": states})
        return components

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_components(history)["prediction"]
