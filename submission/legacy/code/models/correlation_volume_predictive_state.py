"""Local correlation-volume predictive states for 20-to-20 flow forecasting.

The candidate consumes signed local correspondence bins; the matched control
radializes precisely the same projected cost volume.  Matching features are
field-only.  Coordinates enter only after correspondence construction, when a
direct (non-recurrent) future state is formed for every horizon.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .causal_state_multiband import BandDecoder, CausalStateMultibandConfig, _coordinates


ARM_MODES = ("signed_candidate", "radial_control")


@dataclass(frozen=True)
class CorrelationVolumePredictiveStateConfig:
    in_steps: int = 20
    out_steps: int = 20
    channels: int = 3
    reference_lag: int = 1
    motion_lags: tuple[int, ...] = (4, 8, 16)
    radius: int = 4
    feature_channels: int = 48
    state_channels: int = 96
    decoder_channels: int = 64
    groups: int = 8

    def validate(self) -> None:
        if self.in_steps != 20 or self.out_steps != 20 or self.channels != 3:
            raise ValueError("CVPS is fixed to the Track-1 20x20 three-field contract")
        lags = (self.reference_lag, *self.motion_lags)
        if self.reference_lag <= 0 or not self.motion_lags or any(lag <= 0 or lag >= self.in_steps for lag in lags):
            raise ValueError("reference and motion lags must be valid positive history offsets")
        if len(set(lags)) != len(lags):
            raise ValueError("reference and motion lags must be distinct")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if any(value % self.groups for value in (self.feature_channels, self.state_channels,
                                                  self.decoder_channels)):
            raise ValueError("feature/state/decoder channels must divide groups")


class ResidualStateBlock(nn.Module):
    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.first = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.second = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.second(F.silu(self.first(F.silu(self.norm(value)))))
        return value + update


class FieldOnlyFrameEncoder(nn.Module):
    """No coordinate input: correlation may only encode field correspondences."""

    def __init__(self, config: CorrelationVolumePredictiveStateConfig):
        super().__init__()
        channels = config.feature_channels
        self.network = nn.Sequential(
            # Matching deliberately remains at 16x32.  At 8x16, a radius of
            # four makes half the bins boundary bins; downsampling happens
            # only after correspondence has been formed.
            nn.Conv2d(config.channels, channels, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(config.groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(config.groups, channels), nn.SiLU())

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return self.network(field)


class LocalCosineCostVolume(nn.Module):
    """Cosine cost volumes for fixed lags and a fixed local radius."""

    def __init__(self, config: CorrelationVolumePredictiveStateConfig):
        super().__init__()
        self.lags, self.radius = (config.reference_lag, *config.motion_lags), config.radius

    def one(self, latest: torch.Tensor, lagged: torch.Tensor) -> torch.Tensor:
        latest = F.normalize(latest, dim=1, eps=1e-6)
        lagged = F.normalize(lagged, dim=1, eps=1e-6)
        batch, channels, height, width = latest.shape
        kernel = 2 * self.radius + 1
        # Replicate padding avoids a trivial zero-support mask.  It does not
        # make boundary offsets equivalent, so formal diagnostics separately
        # report padding-free interior and boundary angular energy.
        padded = F.pad(lagged, (self.radius,) * 4, mode="replicate")
        patches = F.unfold(padded, kernel_size=kernel, padding=0)
        patches = patches.reshape(batch, channels, kernel * kernel, height, width)
        return (latest.unsqueeze(2) * patches).sum(dim=1)

    def forward(self, latest: torch.Tensor, lagged: list[torch.Tensor]) -> torch.Tensor:
        if len(lagged) != len(self.lags):
            raise ValueError("lag feature count mismatch")
        return torch.cat([self.one(latest, value) for value in lagged], dim=1)


class CostBinProjection(nn.Module):
    """Matched fixed dense signed/radial K-by-K maps for delta volumes.

    There is intentionally no learned bin-mixing map between the raw cost and
    radialization: it could trivially hide angular bins in a radial subspace.
    The shared learned bin mixing begins in state_lift after this operation.
    The upstream field encoders are architecturally shared but are trained
    separately in the two end-to-end arms, so this remains an architecture
    A/B rather than an information-theoretic ablation of one frozen feature.
    """

    def __init__(self, lags: int, radius: int):
        super().__init__()
        self.lags, self.radius = lags, radius
        self.bins = (2 * radius + 1) ** 2
        offsets = [(dy, dx) for dy in range(-radius, radius + 1)
                   for dx in range(-radius, radius + 1)]
        groups: dict[int, list[int]] = {}
        for index, (dy, dx) in enumerate(offsets):
            groups.setdefault(dy * dy + dx * dx, []).append(index)
        radial = torch.zeros(self.bins, self.bins)
        for indexes in groups.values():
            for row in indexes:
                radial[row, indexes] = 1. / len(indexes)
        # Arm selection is deliberately buffer-only.  It must neither add a
        # learnable parameter nor change a checkpoint's learnable state.  The
        # signed matrix is an explicit identity so both arms execute the same
        # dense bin-mixing operation (and hence the same FLOPs).
        self.register_buffer("signed_matrix", torch.eye(self.bins), persistent=False)
        self.register_buffer("radial_matrix", radial, persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64), persistent=False)

    def forward(self, costs: torch.Tensor, radialize: bool) -> torch.Tensor:
        batch, channels, height, width = costs.shape
        if channels != self.lags * self.bins:
            raise ValueError("cost volume channel mismatch")
        value = costs.reshape(batch, self.lags, self.bins, height, width).permute(0, 1, 3, 4, 2)
        mode_matrix = self.radial_matrix if radialize else self.signed_matrix
        # Do not conditionalize the operation itself: the candidate's signed
        # identity and control's radial average are both dense K-by-K maps.
        value = torch.einsum("blhwr,rs->blhws", value, mode_matrix.to(value))
        # Radial averaging lowers variance.  Per-lag bin normalization makes
        # that a geometry comparison rather than an unintentional scale/lr
        # comparison; it is identically executed in both arms.
        value = (value - value.mean(dim=-1, keepdim=True)) / value.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return value.permute(0, 1, 4, 2, 3).reshape(batch, channels, height, width)

    def radialize_raw(self, costs: torch.Tensor) -> torch.Tensor:
        """Apply R to raw bins for the angular-residual shuffle intervention."""
        batch, channels, height, width = costs.shape
        if channels != self.lags * self.bins:
            raise ValueError("cost volume channel mismatch")
        value = costs.reshape(batch, self.lags, self.bins, height, width).permute(0, 1, 3, 4, 2)
        value = torch.einsum("blhwr,rs->blhws", value, self.radial_matrix.to(value))
        return value.permute(0, 1, 4, 2, 3).reshape_as(costs)


class CorrelationVolumePredictiveState(nn.Module):
    def __init__(self, config: CorrelationVolumePredictiveStateConfig,
                 arm_mode: str = "signed_candidate"):
        super().__init__()
        config.validate()
        if arm_mode not in ARM_MODES:
            raise ValueError(f"arm_mode must be one of {ARM_MODES}")
        self.config, self.arm_mode = config, arm_mode
        self.encoder = FieldOnlyFrameEncoder(config)
        self.cost = LocalCosineCostVolume(config)
        self.cost_projection = CostBinProjection(len(config.motion_lags), config.radius)
        bins = self.cost_projection.bins * len(config.motion_lags)
        # Raw-bin mean/std (one each per lag) are computed before I/R and fed
        # identically to both arms.  This lets fusion retain match confidence
        # without using the radial arm's lower post-projection variance.
        self.state_lift = nn.Conv2d(bins + config.feature_channels + 2 + 2 * len(config.motion_lags),
                                    config.state_channels, 3, stride=2, padding=1)
        self.state_blocks = nn.Sequential(*(ResidualStateBlock(config.state_channels, config.groups) for _ in range(4)))
        self.horizon = nn.Embedding(config.out_steps, config.state_channels)
        self.horizon_projection = nn.Linear(config.state_channels, config.state_channels, bias=False)
        band_config = CausalStateMultibandConfig(state_channels=config.state_channels,
                                                 decoder_channels=config.decoder_channels,
                                                 context_channels=config.state_channels,
                                                 groups=config.groups)
        self.decoder = BandDecoder(band_config)
        self.register_buffer("persistence_scale", torch.ones(config.channels))
        self.register_buffer("persistence_bias", torch.zeros(config.channels))

    def architecture_config(self) -> dict[str, object]:
        return {"architecture": "correlation_volume_predictive_state", "arm_mode": self.arm_mode,
                **asdict(self.config)}

    @torch.no_grad()
    def set_normalizer_statistics(self, mean_input: torch.Tensor, mean_target: torch.Tensor,
                                  std_input: torch.Tensor, std_target: torch.Tensor) -> None:
        target_std = torch.as_tensor(std_target, device=self.persistence_scale.device).clamp_min(1e-8)
        input_std, input_mean = torch.as_tensor(std_input, device=self.persistence_scale.device), torch.as_tensor(mean_input, device=self.persistence_scale.device)
        target_mean = torch.as_tensor(mean_target, device=self.persistence_scale.device)
        self.persistence_scale.copy_(input_std[:3] / target_std[:3])
        self.persistence_bias.copy_((input_mean[:3] - target_mean[:3]) / target_std[:3])

    def _raw_cost(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 5 or history.shape[1:] != (self.config.in_steps, 32, 64, self.config.channels):
            raise ValueError("expected history [B,20,32,64,3]")
        fields = [history[:, -1], history[:, -1 - self.config.reference_lag],
                  *(history[:, -1 - lag] for lag in self.config.motion_lags)]
        # One B*(1+lags) encoder call preserves exact sharing while avoiding
        # four launch-bound tiny convolutions on the correspondence path.
        batch = history.shape[0]
        encoded = self.encoder(torch.cat([field.permute(0, 3, 1, 2) for field in fields], dim=0))
        latest_feature, reference_feature, *motion_features = encoded.split(batch, dim=0)
        reference = self.cost.one(latest_feature, reference_feature)
        motion = [self.cost.one(latest_feature, value) - reference for value in motion_features]
        return latest_feature, torch.cat(motion, dim=1)

    def correlation_components(self, history: torch.Tensor, radial_override: bool | None = None,
                               angular_donor_history: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if history.ndim != 5 or history.shape[1:] != (self.config.in_steps, 32, 64, self.config.channels):
            raise ValueError("expected history [B,20,32,64,3]")
        latest_feature, native_raw = self._raw_cost(history)
        raw = native_raw
        if angular_donor_history is not None:
            if angular_donor_history.shape != history.shape:
                raise ValueError("angular donor history must match history shape")
            _donor_feature, donor_raw = self._raw_cost(angular_donor_history)
            # V = R(V) + A(V); donors provide A only, preserving the native
            # radial/context distribution while breaking direction identity.
            raw = self.cost_projection.radialize_raw(native_raw) + donor_raw - self.cost_projection.radialize_raw(donor_raw)
        radialize = self.arm_mode == "radial_control" if radial_override is None else bool(radial_override)
        projected = self.cost_projection(raw, radialize)
        batch, _channels, height, width = raw.shape
        native_values = native_raw.reshape(batch, len(self.config.motion_lags), self.cost_projection.bins, height, width)
        # Native-only confidence statistics remain distribution preserving
        # under the angular residual shuffle.
        raw_stats = torch.cat((native_values.mean(2), native_values.std(2, unbiased=False)), dim=1)
        return {"latest_feature": latest_feature, "raw_cost": raw, "raw_stats": raw_stats,
                "projected_cost": projected, "native_raw_cost": native_raw}

    def forward_components(self, history: torch.Tensor, radial_override: bool | None = None,
                           angular_donor_history: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        matching = self.correlation_components(history, radial_override, angular_donor_history)
        latest = matching["latest_feature"]
        base = self.state_blocks(self.state_lift(torch.cat((matching["projected_cost"], matching["raw_stats"], latest,
                                                            _coordinates(latest)), dim=1)))
        queries = self.horizon_projection(self.horizon.weight).view(1, self.config.out_steps,
                                                                      self.config.state_channels, 1, 1)
        states = base.unsqueeze(1) + queries
        batch, steps, channels, height, width = states.shape
        flat_state = states.reshape(batch * steps, channels, height, width)
        terminal = base.unsqueeze(1).expand(-1, steps, -1, -1, -1).reshape_as(flat_state)
        decoded = self.decoder(flat_state, terminal)
        components = {name: value.reshape(batch, steps, 3, 32, 64).permute(0, 1, 3, 4, 2)
                      for name, value in decoded.items() if name in ("low", "mid", "high", "residual")}
        persistence = history[:, -1] * self.persistence_scale.view(1, 1, 1, -1) + self.persistence_bias.view(1, 1, 1, -1)
        components.update({"prediction": persistence.unsqueeze(1) + components["residual"],
                           "persistence": persistence, "direct_states": states, **matching})
        return components

    def forward(self, history: torch.Tensor, angular_donor_history: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_components(history, angular_donor_history=angular_donor_history)["prediction"]
