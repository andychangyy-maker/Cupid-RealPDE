"""Softly fuse a causal zero-time-mean specialist with a frozen B210.

The mature B210 remains the identity anchor.  The specialist proposes only a
future u/v residual whose temporal mean is exactly zero at every pixel and
channel.  Two independent, zero-initialized continuous gates assign that
proposal to point prediction and to the centre of the native B210 interval.
Consequently update zero is bitwise identical to B210, point and uncertainty
roles can move independently, interval width is inherited from B210, and a
hard replacement is not representable by the public interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn

from .causal_phase_direction import (
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
)


class _ExactIdentityAddSTE(torch.autograd.Function):
    """Return the exact anchor when disabled while retaining delta gradient."""

    @staticmethod
    def forward(
            ctx, anchor: torch.Tensor, delta: torch.Tensor,
            bypass: torch.Tensor) -> torch.Tensor:
        del ctx
        return torch.where(bypass, anchor, anchor + delta)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        del ctx
        return gradient, gradient, None


@dataclass(frozen=True)
class CausalResidualSoftGateConfig:
    """Configuration of the two-role output-space fusion."""

    mixer_mode: str = "causal_temporal"
    point_gate_limit: float = 1.0
    uq_gate_limit: float = 1.0
    specialist_seed: int = 4210
    specialist_head_std: float = 1.0e-3

    def validate(self) -> None:
        # Delegate mixer validation to the already-audited phase adapter.
        CausalPhaseDirectionConfig(mixer_mode=self.mixer_mode).validate()
        if self.point_gate_limit <= 0.0 or self.uq_gate_limit <= 0.0:
            raise ValueError("soft-gate limits must be positive")
        if self.specialist_head_std <= 0.0:
            raise ValueError("specialist head std must be positive")


@dataclass
class CausalResidualSoftGateOutput:
    """Inspectable point/UQ responsibility split."""

    prediction: torch.Tensor
    lower_extent: torch.Tensor
    upper_extent: torch.Tensor
    base_prediction: torch.Tensor
    base_lower_extent: torch.Tensor
    base_upper_extent: torch.Tensor
    specialist_prediction: torch.Tensor
    specialist_residual: torch.Tensor
    point_delta: torch.Tensor
    uq_center_delta: torch.Tensor
    point_gate: torch.Tensor
    uq_gate: torch.Tensor


class CausalResidualSoftGate(nn.Module):
    """Fuse one causal-history residual into point and native-UQ separately."""

    def __init__(self, config: CausalResidualSoftGateConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.specialist = CausalPhaseDirectionAdapter(
            CausalPhaseDirectionConfig(mixer_mode=config.mixer_mode))

        # The legacy phase adapter starts its direction projection at zero.
        # Here the *outer* gates own exact identity.  A small deterministic,
        # matched non-zero proposal lets both role gates receive gradient on
        # update one; the specialist body then receives gradient on update two.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.specialist_seed)
            nn.init.normal_(
                self.specialist.direction_head.weight,
                mean=0.0, std=config.specialist_head_std)
            nn.init.zeros_(self.specialist.direction_head.bias)

        # Time-invariant per-velocity-channel gates preserve the specialist's
        # exact per-pixel temporal-zero-mean constraint after fusion.
        self.point_gate_logits = nn.Parameter(torch.zeros(1, 1, 1, 1, 2))
        self.uq_gate_logits = nn.Parameter(torch.zeros(1, 1, 1, 1, 2))

    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture": "causal_residual_soft_gate",
            "anchor_responsibility": (
                "B210 point, pressure, native interval width"),
            "specialist_responsibility": (
                "strict zero-time-mean future uv residual"),
            "point_fusion": "tanh(point_gate) * specialist_residual",
            "uq_fusion": "tanh(uq_gate) * specialist_interval_center_shift",
            "hard_replacement_available": False,
            "pressure_passthrough": True,
            "native_interval_width_preserved": True,
            **asdict(self.config),
        }

    def gate_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.config.point_gate_limit * torch.tanh(self.point_gate_logits),
            self.config.uq_gate_limit * torch.tanh(self.uq_gate_logits),
        )

    def forward_components(
            self, history: torch.Tensor, base_prediction: torch.Tensor,
            base_lower_extent: torch.Tensor,
            base_upper_extent: torch.Tensor,
            ) -> CausalResidualSoftGateOutput:
        base_prediction = base_prediction.detach()
        base_lower_extent = base_lower_extent.detach()
        base_upper_extent = base_upper_extent.detach()
        base_uv = base_prediction[..., :2]
        if (base_lower_extent.shape != base_uv.shape
                or base_upper_extent.shape != base_uv.shape):
            raise ValueError("native B210 extents must align with future u/v")

        specialist = self.specialist.forward_components(
            history, base_prediction)
        residual = specialist.prediction[..., :2] - base_uv
        # Remove only floating projection drift.  This is the hard structural
        # constraint consumed by both role gates.
        residual = residual - residual.mean(dim=1, keepdim=True)
        point_gate, uq_gate = self.gate_values()
        point_delta = point_gate.to(residual) * residual
        uq_center_delta = uq_gate.to(residual) * residual

        point_bypass = self.point_gate_logits.eq(0).all().reshape(
            1, 1, 1, 1, 1)
        point_uv = _ExactIdentityAddSTE.apply(
            base_uv, point_delta, point_bypass)
        prediction = (
            point_uv if base_prediction.shape[-1] == 2
            else torch.cat((point_uv, base_prediction[..., 2:]), dim=-1))

        # Shift the two absolute interval endpoints together.  The point delta
        # is detached in this coordinate conversion so an interval-only loss
        # cannot leak into the point gate; the forward values are unchanged.
        lower_delta = point_delta.detach() - uq_center_delta
        upper_delta = uq_center_delta - point_delta.detach()
        extent_bypass = (
            self.point_gate_logits.eq(0).all()
            & self.uq_gate_logits.eq(0).all()).reshape(1, 1, 1, 1, 1)
        lower_extent = _ExactIdentityAddSTE.apply(
            base_lower_extent, lower_delta, extent_bypass)
        upper_extent = _ExactIdentityAddSTE.apply(
            base_upper_extent, upper_delta, extent_bypass)
        return CausalResidualSoftGateOutput(
            prediction=prediction,
            lower_extent=lower_extent,
            upper_extent=upper_extent,
            base_prediction=base_prediction,
            base_lower_extent=base_lower_extent,
            base_upper_extent=base_upper_extent,
            specialist_prediction=specialist.prediction,
            specialist_residual=residual,
            point_delta=point_delta,
            uq_center_delta=uq_center_delta,
            point_gate=point_gate,
            uq_gate=uq_gate,
        )

    def forward(
            self, history: torch.Tensor, base_prediction: torch.Tensor,
            base_lower_extent: torch.Tensor,
            base_upper_extent: torch.Tensor,
            ) -> torch.Tensor:
        return self.forward_components(
            history, base_prediction,
            base_lower_extent, base_upper_extent).prediction


class FrozenB210CausalResidualSoftGate(nn.Module):
    """Deployment/training wrapper around one immutable native-UQ B210."""

    def __init__(self, base_model: nn.Module,
                 fusion: CausalResidualSoftGate):
        super().__init__()
        self.base_model = base_model
        self.fusion = fusion
        self.base_model.requires_grad_(False)
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        self.fusion.train(mode)
        return self

    def architecture_config(self) -> dict[str, object]:
        base = (self.base_model.architecture_config()
                if hasattr(self.base_model, "architecture_config") else None)
        return {
            "architecture": "frozen_b210_causal_residual_soft_gate",
            "base_model": base,
            "fusion": self.fusion.architecture_config(),
        }

    def forward_components(
            self, history: torch.Tensor,
            adapter_history: torch.Tensor | None = None,
            ) -> CausalResidualSoftGateOutput:
        if not hasattr(self.base_model, "forward_distribution"):
            raise AttributeError("base B210 has no distribution interface")
        if adapter_history is None:
            adapter_history = history
        with torch.no_grad():
            prediction, lower, upper = self.base_model.forward_distribution(
                history)
        return self.fusion.forward_components(
            adapter_history, prediction, lower, upper)

    def forward(
            self, history: torch.Tensor,
            adapter_history: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_components(history, adapter_history).prediction

    def forward_distribution(
            self, history: torch.Tensor,
            adapter_history: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = self.forward_components(history, adapter_history)
        return values.prediction, values.lower_extent, values.upper_extent

    def fusion_state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.fusion.state_dict()


def build_causal_residual_soft_gate(
        mixer_mode: str = "causal_temporal",
        ) -> CausalResidualSoftGate:
    return CausalResidualSoftGate(
        CausalResidualSoftGateConfig(mixer_mode=mixer_mode))


__all__ = [
    "CausalResidualSoftGate",
    "CausalResidualSoftGateConfig",
    "CausalResidualSoftGateOutput",
    "FrozenB210CausalResidualSoftGate",
    "build_causal_residual_soft_gate",
]
