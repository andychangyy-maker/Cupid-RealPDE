"""Direct local stable-oscillator forecaster with a matched real-pole control.

The two families share every learned tensor.  Their sole difference is the
fixed two-by-two temporal transform applied to each local modal pair: bounded
complex rotation versus two stable real poles.  Future states are synthesized
in closed form for every horizon, never recursively rolled out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, pi

import torch
from torch import nn

from .causal_state_multiband import (BandDecoder, CausalStateMultibandConfig,
                                     ConvGRUCell, _coordinates)


TEMPORAL_FAMILIES = ("complex", "real", "zero_frequency")


@dataclass(frozen=True)
class LocalStableOscillatorConfig:
    in_steps: int = 20
    out_steps: int = 20
    channels: int = 3
    state_channels: int = 96
    decoder_channels: int = 64
    groups: int = 8
    modes: int = 4
    pairs_per_mode: int = 12
    gamma_floor: float = 1.0e-4
    gamma_max: float = 0.1
    omega_max: float = 2.0 * pi / 24.0

    def validate(self) -> None:
        if (self.in_steps, self.out_steps, self.channels) != (20, 20, 3):
            raise ValueError("local stable oscillator is fixed to 20-to-20 three-field inputs")
        if self.modes <= 0 or self.pairs_per_mode <= 0:
            raise ValueError("modes and pairs_per_mode must be positive")
        if self.state_channels != 2 * self.modes * self.pairs_per_mode:
            raise ValueError("state channels must equal 2*modes*pairs_per_mode")
        if any(value <= 0 for value in (self.decoder_channels, self.groups)):
            raise ValueError("decoder dimensions must be positive")
        if any(value % self.groups for value in (self.state_channels, self.decoder_channels)):
            raise ValueError("state and decoder channels must divide groups")
        if not 0.0 < self.gamma_floor < self.gamma_max or self.omega_max <= 0.0:
            raise ValueError("rates require 0 < gamma_floor < gamma_max and omega_max > 0")


class TerminalResidualBlock(nn.Module):
    """Shared local terminal-state refinement, independent of temporal arm."""

    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.first = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.second = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.second(torch.nn.functional.silu(
            self.first(torch.nn.functional.silu(self.norm(state)))))


class LocalStableOscillatorForecaster(nn.Module):
    """20 direct future states parameterized by local stable modal pairs."""

    def __init__(self, config: LocalStableOscillatorConfig,
                 temporal_family: str = "complex"):
        super().__init__()
        config.validate()
        if temporal_family not in TEMPORAL_FAMILIES:
            raise ValueError(f"temporal_family must be one of {TEMPORAL_FAMILIES}")
        self.config, self.temporal_family = config, temporal_family
        channels = config.state_channels
        self.frame_stem = nn.Sequential(
            nn.Conv2d(config.channels + 2, channels, 5, stride=4, padding=2),
            nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))
        self.history_cell = ConvGRUCell(channels)
        self.terminal_refiner = nn.Sequential(
            TerminalResidualBlock(channels, config.groups),
            TerminalResidualBlock(channels, config.groups))
        self.slow_head = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.SiLU(),
                                       nn.Conv2d(channels, channels, 1))
        self.modal_head = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.SiLU(),
                                        nn.Conv2d(channels, channels, 1))
        # One gamma and one temporal coordinate per local mode.  The final
        # zero weight makes raw2 exactly zero at initialization, so complex
        # and real arms have bitwise same temporal states there.
        self.rate_head = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.SiLU(),
                                       nn.Conv2d(channels, 2 * config.modes, 1))
        final = self.rate_head[-1]
        nn.init.zeros_(final.weight)
        gamma_fraction = (.01 - config.gamma_floor) / (config.gamma_max - config.gamma_floor)
        gamma_raw = log(gamma_fraction / (1. - gamma_fraction))
        nn.init.constant_(final.bias[:config.modes], gamma_raw)
        nn.init.zeros_(final.bias[config.modes:])
        decoder_config = CausalStateMultibandConfig(
            state_channels=channels, decoder_channels=config.decoder_channels,
            context_channels=channels, groups=config.groups)
        self.decoder = BandDecoder(decoder_config)
        self.register_buffer("persistence_scale", torch.ones(config.channels))
        self.register_buffer("persistence_bias", torch.zeros(config.channels))

    def architecture_config(self) -> dict[str, object]:
        return {"architecture": "local_stable_oscillator", "temporal_family": self.temporal_family,
                **asdict(self.config)}

    @torch.no_grad()
    def set_normalizer_statistics(self, mean_input: torch.Tensor, mean_target: torch.Tensor,
                                  std_input: torch.Tensor, std_target: torch.Tensor) -> None:
        target_std = torch.as_tensor(std_target, device=self.persistence_scale.device).clamp_min(1e-8)
        input_std = torch.as_tensor(std_input, device=self.persistence_scale.device)
        input_mean = torch.as_tensor(mean_input, device=self.persistence_scale.device)
        target_mean = torch.as_tensor(mean_target, device=self.persistence_scale.device)
        self.persistence_scale.copy_(input_std[:self.config.channels] / target_std[:self.config.channels])
        self.persistence_bias.copy_((input_mean[:self.config.channels] - target_mean[:self.config.channels]) /
                                    target_std[:self.config.channels])

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5 or history.shape[1:] != (self.config.in_steps, 32, 64, self.config.channels):
            raise ValueError("expected history [B,20,32,64,3]")
        batch, steps, height, width, channels = history.shape
        frames = history.permute(0, 1, 4, 2, 3).reshape(batch * steps, channels, height, width)
        encoded = self.frame_stem(torch.cat((frames, _coordinates(frames)), dim=1))
        encoded = encoded.reshape(batch, steps, self.config.state_channels, height // 4, width // 4)
        state = torch.zeros_like(encoded[:, 0])
        for observed in encoded.unbind(1):
            state = self.history_cell(state, observed)
        return self.terminal_refiner(state)

    def rate_maps(self, terminal_state: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_gamma, raw_temporal2 = self.rate_head(terminal_state).chunk(2, dim=1)
        gamma = self.config.gamma_floor + (self.config.gamma_max - self.config.gamma_floor) * torch.sigmoid(raw_gamma)
        omega = self.config.omega_max * torch.tanh(raw_temporal2)
        # The real control uses the same bounded second coordinate as its
        # symmetric coupling.  Its generator is
        # [[-gamma-|b|, b], [b, -gamma-|b|]], whose eigenvalues are
        # -gamma and -gamma-2|b|.  Hence its maximum gain envelope exactly
        # matches the complex family's exp(-gamma*h), rather than receiving a
        # slower-decaying pole.  torch.where chooses a useful +1 subgradient
        # at the common b=0 initialization without altering the forward value.
        coupling = omega
        coupling_abs = torch.where(coupling >= 0., coupling, -coupling)
        real_log_rates = torch.stack((-gamma, -gamma - 2. * coupling_abs), dim=2)
        return {"raw_gamma": raw_gamma, "raw_temporal2": raw_temporal2,
                "gamma": gamma, "omega": omega, "real_coupling": coupling,
                "real_coupling_abs": coupling_abs,
                "real_log_rates": real_log_rates}

    def future_states(self, terminal_state: torch.Tensor, temporal_family: str | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        family = self.temporal_family if temporal_family is None else temporal_family
        if family not in TEMPORAL_FAMILIES:
            raise ValueError(f"temporal_family must be one of {TEMPORAL_FAMILIES}")
        slow, modal = self.slow_head(terminal_state), self.modal_head(terminal_state)
        rates = self.rate_maps(terminal_state)
        batch, _channels, height, width = modal.shape
        modal = modal.reshape(batch, self.config.modes, self.config.pairs_per_mode, 2, height, width)
        first, second = modal.unbind(3)
        h = torch.arange(1, self.config.out_steps + 1, device=modal.device, dtype=modal.dtype).view(1, -1, 1, 1, 1)
        gamma, raw2 = rates["gamma"].unsqueeze(1), rates["raw_temporal2"].unsqueeze(1)
        decay = torch.exp(-gamma * h)
        if family == "real":
            coupling = rates["real_coupling"].unsqueeze(1)
            coupling_abs = rates["real_coupling_abs"].unsqueeze(1)
            envelope = torch.exp(-(gamma + coupling_abs) * h)
            cosine, sine = torch.cosh(coupling * h), torch.sinh(coupling * h)
            left = envelope.unsqueeze(3) * (first.unsqueeze(1) * cosine.unsqueeze(3) + second.unsqueeze(1) * sine.unsqueeze(3))
            right = envelope.unsqueeze(3) * (first.unsqueeze(1) * sine.unsqueeze(3) + second.unsqueeze(1) * cosine.unsqueeze(3))
        else:
            omega = torch.zeros_like(raw2) if family == "zero_frequency" else rates["omega"].unsqueeze(1)
            cosine, sine = torch.cos(omega * h), torch.sin(omega * h)
            left = decay.unsqueeze(3) * (first.unsqueeze(1) * cosine.unsqueeze(3) - second.unsqueeze(1) * sine.unsqueeze(3))
            right = decay.unsqueeze(3) * (first.unsqueeze(1) * sine.unsqueeze(3) + second.unsqueeze(1) * cosine.unsqueeze(3))
        dynamic = torch.stack((left, right), dim=3).reshape(batch, self.config.out_steps,
                                                             self.config.state_channels, height, width)
        return slow.unsqueeze(1) + dynamic, {"slow_state": slow, "modal_coefficients": modal,
                                               "dynamic_states": dynamic, **rates}

    def forward_components(self, history: torch.Tensor,
                           temporal_family: str | None = None) -> dict[str, torch.Tensor]:
        terminal = self.encode_history(history)
        states, diagnostics = self.future_states(terminal, temporal_family)
        batch, steps, channels, height, width = states.shape
        flat_state = states.reshape(batch * steps, channels, height, width)
        slow = diagnostics["slow_state"]
        condition = slow.unsqueeze(1).expand(-1, steps, -1, -1, -1).reshape_as(flat_state)
        dynamic_decoded = self.decoder(flat_state, condition)
        slow_decoded = self.decoder(slow, slow)
        # Preserve the mean/responsibility split exactly: modal dynamics are
        # time-zero-mean, while slow state alone owns the residual mean.
        parts = {}
        for name in ("low", "mid", "high", "residual"):
            value = dynamic_decoded[name].reshape(batch, steps, self.config.channels, 32, 64)
            zero_mean = value - value.mean(1, keepdim=True)
            slow_value = slow_decoded[name].unsqueeze(1)
            parts[name] = (zero_mean + slow_value).permute(0, 1, 3, 4, 2)
        persistence = (history[:, -1] * self.persistence_scale.view(1, 1, 1, -1) +
                       self.persistence_bias.view(1, 1, 1, -1))
        parts.update({"prediction": persistence.unsqueeze(1) + parts["residual"],
                      "persistence": persistence, "terminal_state": terminal,
                      "future_states": states, "slow_mean_residual": slow_decoded["residual"].permute(0, 2, 3, 1),
                      **diagnostics})
        return parts

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_components(history)["prediction"]
