"""Small deployed residual-TKE head for a frozen member-synchronous B210."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.ensemble_unet2d import EnsembleUNet2d
from models.unet2d import _group_norm


class _ExactBaselineSelectSTE(torch.autograd.Function):
    """Return exact baseline bytes at zero while differentiating candidate."""

    @staticmethod
    def forward(ctx, baseline, candidate, bypass):
        return torch.where(bypass, baseline, candidate)

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output, None


class DirectResidualTKEHead(nn.Module):
    """Predict one bounded spatial log-TKE correction from decoded features."""

    def __init__(
            self, feature_channels: int = 210, hidden_channels: int = 32,
            groups: int = 8, log_limit: float = 1.0,
            tke_epsilon: float = 1e-8) -> None:
        super().__init__()
        if feature_channels <= 0 or hidden_channels <= 0:
            raise ValueError("TKE-head channel counts must be positive")
        if groups <= 0 or hidden_channels % groups:
            raise ValueError("TKE-head groups must divide hidden channels")
        if log_limit <= 0.0 or tke_epsilon <= 0.0:
            raise ValueError("TKE-head limits must be positive")
        self.feature_channels = int(feature_channels)
        self.hidden_channels = int(hidden_channels)
        self.groups = int(groups)
        self.log_limit = float(log_limit)
        self.tke_epsilon = float(tke_epsilon)
        self.body = nn.Sequential(
            nn.Conv2d(feature_channels + 1, hidden_channels, 1),
            _group_norm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                      groups=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": "direct_residual_tke_head",
            "feature_channels": self.feature_channels,
            "hidden_channels": self.hidden_channels,
            "groups": self.groups,
            "log_limit": self.log_limit,
            "tke_epsilon": self.tke_epsilon,
        }

    def physical_tke_map(
            self, prediction: torch.Tensor,
            target_std: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 5 or prediction.shape[-1] < 2:
            raise ValueError("prediction must be (B,T,H,W,C), C >= 2")
        scale = target_std[:2].to(
            device=prediction.device, dtype=prediction.dtype).view(
                1, 1, 1, 1, 2)
        physical_uv = prediction[..., :2] * scale
        fluctuation = physical_uv - physical_uv.mean(dim=1, keepdim=True)
        return 0.5 * fluctuation.square().mean(dim=1).sum(dim=-1)

    def forward(
            self, finest_feature: torch.Tensor,
            prediction: torch.Tensor,
            target_std: torch.Tensor) -> torch.Tensor:
        if (finest_feature.ndim != 4
                or finest_feature.shape[1] != self.feature_channels):
            raise ValueError("TKE-head finest feature shape changed")
        tke = self.physical_tke_map(prediction, target_std)
        if finest_feature.shape[-2:] != tke.shape[-2:]:
            finest_feature = F.interpolate(
                finest_feature, size=tke.shape[-2:], mode="bilinear",
                align_corners=False)
        log_tke = torch.log(tke.clamp_min(self.tke_epsilon)).unsqueeze(1)
        raw = self.body(torch.cat((finest_feature, log_tke), dim=1))
        return self.log_limit * torch.tanh(raw)


class FrozenB210DirectTKE(nn.Module):
    """Attach a deployed TKE head without mutating the source checkpoint."""

    def __init__(
            self, base: EnsembleUNet2d,
            head: DirectResidualTKEHead | None = None) -> None:
        super().__init__()
        if not base.config.physical_energy_member_sync:
            raise ValueError("direct TKE head requires member-synchronous B210")
        self.base = base
        self.head = head or DirectResidualTKEHead(
            feature_channels=base.config.base, groups=base.config.groups)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    @staticmethod
    def _correct_temporal_fluctuation(
            value: torch.Tensor, delta_logk: torch.Tensor,
            time_dim: int) -> torch.Tensor:
        uv = value[..., :2]
        mean = uv.mean(dim=time_dim, keepdim=True)
        amplitude = torch.expm1(0.5 * delta_logk).permute(0, 2, 3, 1)
        if value.ndim == 6:
            amplitude = amplitude[:, None, None]
            bypass = delta_logk.eq(0).permute(0, 2, 3, 1)[:, None, None]
        else:
            amplitude = amplitude[:, None]
            bypass = delta_logk.eq(0).permute(0, 2, 3, 1)[:, None]
        candidate_uv = uv + (uv - mean) * amplitude
        controlled_uv = _ExactBaselineSelectSTE.apply(
            uv, candidate_uv, bypass)
        if value.shape[-1] == 2:
            return controlled_uv
        return torch.cat((controlled_uv, value[..., 2:]), dim=-1)

    @staticmethod
    def _select_baseline(
            baseline: torch.Tensor, candidate: torch.Tensor,
            delta_logk: torch.Tensor) -> torch.Tensor:
        if baseline.shape != candidate.shape:
            raise ValueError("TKE-head candidate shape changed")
        bypass = delta_logk.eq(0).permute(0, 2, 3, 1).unsqueeze(1)
        while bypass.ndim < baseline.ndim:
            bypass = bypass.unsqueeze(-1)
        return _ExactBaselineSelectSTE.apply(baseline, candidate, bypass)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.head.train(mode)
        return self

    def forward_uq_training(self, x: torch.Tensor):
        base_values = self.base.forward_uq_training(x)
        (base_prediction, base_members, base_quantiles, base_lower,
         base_upper, features, base_coverage) = base_values
        delta_logk = self.head(
            features[-1], base_prediction, self.base.target_std)
        prediction = self._correct_temporal_fluctuation(
            base_prediction, delta_logk, time_dim=1)
        members = self._correct_temporal_fluctuation(
            base_members, delta_logk, time_dim=2)
        fused = self.base._fused_uq_features(
            features, members, x,
            prediction=(prediction
                        if self.base.config.shared_member_uv_mean else None),
            detach_members=False)
        quantiles = self.base._quantiles_from_fused(fused, members)
        coverage = self.base._coverage_logits(fused)
        lower, upper = self.base._utility_interval(quantiles, coverage)
        lower, upper = self.base._refine_utility_interval(
            lower, upper, fused)
        lower, upper = self.base._ordered_interval_extents(lower, upper)
        quantiles = self._select_baseline(
            base_quantiles, quantiles, delta_logk)
        lower = self._select_baseline(base_lower, lower, delta_logk)
        upper = self._select_baseline(base_upper, upper, delta_logk)
        if coverage is not None and base_coverage is not None:
            coverage = self._select_baseline(
                base_coverage, coverage, delta_logk)
        return (prediction, members, quantiles, lower, upper, features,
                coverage)

    def forward_distribution(self, x: torch.Tensor):
        prediction, _members, _quantiles, lower, upper = (
            self.forward_uq_training(x)[:5])
        return prediction, lower, upper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_uq_training(x)[0]
