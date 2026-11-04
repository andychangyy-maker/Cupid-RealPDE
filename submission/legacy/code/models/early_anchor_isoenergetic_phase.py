"""Matched full-horizon and early-anchor isoenergetic phase adapters.

The trainable trunk is exactly :class:`NativeFeatureIsoenergeticPhase`.
Only the deterministic output geometry differs:

``full20``
    Apply the additive-direction projection over all forecast frames.

``anchor2_tail18``
    Copy the first two B210 frames exactly and project only the remaining
    eighteen frames.  The projection restores the B210 tail mean and centered
    RMS independently at every pixel and velocity channel.

Keeping the anchors fixed and the tail first and second moments fixed also
keeps the full-horizon mean and variance fixed.  Consequently the official
full-horizon TKE map is unchanged, without allowing energy to move between
the protected early frames and the tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .ensemble_unet2d import _ExactIdentityAddSTE
from .native_feature_isoenergetic_phase import (
    ADDITIVE_DIRECTION_CANDIDATE,
    NativeFeatureIsoenergeticPhase,
    NativeFeaturePhaseConfig,
)


FULL20 = "full20"
ANCHOR2_TAIL18 = "anchor2_tail18"
GEOMETRIES = (FULL20, ANCHOR2_TAIL18)


def restore_temporal_mean_rms(
        candidate: torch.Tensor, reference: torch.Tensor, *,
        epsilon: float = 1.0e-12) -> torch.Tensor:
    """Give ``candidate`` the temporal mean and centered RMS of ``reference``.

    Both tensors are ``[B,T,H,W,C]`` and must be aligned.  In a degenerate
    reference or candidate direction the reference is returned, which keeps
    the operation finite and prevents an arbitrary phase choice.
    """
    if candidate.shape != reference.shape or candidate.ndim != 5:
        raise ValueError("candidate/reference temporal fields must be aligned")
    if candidate.shape[1] < 2:
        raise ValueError("temporal projection requires at least two frames")
    if not 0.0 < float(epsilon) < float("inf"):
        raise ValueError("projection epsilon must be finite and positive")
    candidate32 = candidate.float()
    reference32 = reference.float()
    reference_mean = reference32.mean(dim=1, keepdim=True)
    reference_fluctuation = reference32 - reference_mean
    candidate_fluctuation = (
        candidate32 - candidate32.mean(dim=1, keepdim=True))
    reference_energy = reference_fluctuation.square().mean(
        dim=1, keepdim=True)
    candidate_energy = candidate_fluctuation.square().mean(
        dim=1, keepdim=True)
    active = (
        (reference_energy > epsilon) & (candidate_energy > epsilon))
    safe_reference = torch.where(
        active, reference_energy, torch.ones_like(reference_energy))
    safe_candidate = torch.where(
        active, candidate_energy, torch.ones_like(candidate_energy))
    scale = torch.sqrt(safe_reference / safe_candidate)
    restored_active = reference_mean + candidate_fluctuation * scale
    # Select the original tensor after reconstruction so an inactive element
    # is a bitwise fallback, not merely an algebraically equivalent one.
    restored = torch.where(active, restored_active, reference32)
    return restored.to(reference)


def anchor_tail_isoenergetic_projection(
        candidate_uv: torch.Tensor, base_uv: torch.Tensor, *,
        anchor_steps: int = 2, epsilon: float = 1.0e-12) -> torch.Tensor:
    """Protect early B210 frames and restore the B210 tail moments.

    This target-free mapping is also useful as a read-only diagnostic on an
    already trained full-horizon phase endpoint.
    """
    if candidate_uv.shape != base_uv.shape or candidate_uv.ndim != 5:
        raise ValueError("candidate/base velocity fields must be aligned")
    steps = int(candidate_uv.shape[1])
    if not 0 < int(anchor_steps) <= steps - 2:
        raise ValueError("anchor_steps must leave at least two tail frames")
    if candidate_uv.shape[-1] != 2:
        raise ValueError("phase projection requires exactly two velocity channels")
    tail = restore_temporal_mean_rms(
        candidate_uv[:, anchor_steps:], base_uv[:, anchor_steps:],
        epsilon=epsilon)
    return torch.cat((base_uv[:, :anchor_steps], tail), dim=1)


@dataclass
class EarlyAnchorPhaseFields:
    prediction: torch.Tensor
    base_prediction: torch.Tensor
    raw: torch.Tensor
    bounded_field: torch.Tensor
    full20_uv: torch.Tensor
    anchor_tail_uv: torch.Tensor
    anchor_steps: int


class EarlyAnchorIsoenergeticPhase(NativeFeatureIsoenergeticPhase):
    """One native-feature trunk with two matched output geometries."""

    def __init__(
            self, config: NativeFeaturePhaseConfig | None = None, *,
            geometry: str = FULL20, anchor_steps: int = 2) -> None:
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}")
        resolved = NativeFeaturePhaseConfig() if config is None else config
        if not 0 < int(anchor_steps) <= resolved.out_steps - 2:
            raise ValueError("anchor_steps must leave at least two tail frames")
        self.geometry = geometry
        self.anchor_steps = int(anchor_steps)
        super().__init__(resolved, mode=ADDITIVE_DIRECTION_CANDIDATE)

    def architecture_config(self) -> dict[str, object]:
        values = super().architecture_config()
        values.update({
            "architecture": "early_anchor_isoenergetic_phase",
            "geometry": self.geometry,
            "anchor_steps": self.anchor_steps,
            "native_mode": ADDITIVE_DIRECTION_CANDIDATE,
        })
        return values

    def execution_contract(self) -> dict[str, object]:
        return {
            "base_feature": "detached finest B210 decoder feature",
            "base_forecast": "detached B210 point forecast",
            "shared_trainable_trunk": True,
            "both_output_geometries_executed": True,
            "full20": "all-horizon additive-direction mean/RMS projection",
            "anchor": {
                "steps": self.anchor_steps,
                "prefix": "bitwise frozen B210",
                "tail": "restore frozen B210 tail mean and centered RMS",
            },
            "full_horizon_mean_preserved": True,
            "full_horizon_tke_preserved": True,
            "pressure_passthrough": "bitwise",
            "update_zero": "bitwise frozen-B210 identity",
        }

    def forward_with_fields(
            self, prediction: torch.Tensor,
            finest_feature: torch.Tensor) -> EarlyAnchorPhaseFields:
        # The parent executes the unchanged trunk and full20 geometry.
        native = super().forward_with_fields(prediction, finest_feature)
        base_uv = prediction[..., :2]
        full20_uv = native.additive_direction_uv
        anchor_uv = anchor_tail_isoenergetic_projection(
            full20_uv, base_uv, anchor_steps=self.anchor_steps,
            epsilon=self.phase_config.projection_epsilon)
        selected = full20_uv if self.geometry == FULL20 else anchor_uv
        delta = selected - base_uv
        bypass = native.raw.eq(0.0).flatten(1).all(dim=1).view(
            -1, 1, 1, 1, 1)
        corrected_uv = _ExactIdentityAddSTE.apply(base_uv, delta, bypass)
        corrected = (
            corrected_uv if prediction.shape[-1] == 2 else
            torch.cat((corrected_uv, prediction[..., 2:]), dim=-1))
        return EarlyAnchorPhaseFields(
            prediction=corrected,
            base_prediction=prediction,
            raw=native.raw,
            bounded_field=native.bounded_field,
            full20_uv=full20_uv,
            anchor_tail_uv=anchor_uv,
            anchor_steps=self.anchor_steps,
        )

    def forward(
            self, prediction: torch.Tensor,
            finest_feature: torch.Tensor) -> torch.Tensor:
        return self.forward_with_fields(prediction, finest_feature).prediction
