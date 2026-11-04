"""PC3D with a controlled absolute or horizon-integrated displacement basis.

The parent operator already predicts one raw transport field per forecast
horizon.  This experiment changes only the parameterisation seen by the
existing bounded displacement map::

    absolute:   D_h = max_displacement * tanh(r_h)
    integrated: D_h = max_displacement * tanh(sum_{k <= h} r_k)

PixelShuffle is a per-horizon permutation, so accumulating the packed raw
transport tensor here is exactly equivalent to accumulating it after
PixelShuffle.  Every module, parameter, buffer, source warp and residual path
remains inherited from :class:`PyramidCausal3DOperator`.
"""

from __future__ import annotations

import torch
from torch import nn

from .pyramid_causal3d_operator import (
    PyramidCausal3DOperator,
    PyramidCausal3DOperatorConfig,
)


DISPLACEMENT_SEMANTICS = (
    "absolute",
    "horizon_integrated",
)


class PyramidCausal3DIntegratedDisplacementOperator(PyramidCausal3DOperator):
    """Keep the causal PC3D graph fixed and select one displacement basis."""

    def __init__(
            self, config: PyramidCausal3DOperatorConfig,
            displacement_semantics: str = "horizon_integrated") -> None:
        if displacement_semantics not in DISPLACEMENT_SEMANTICS:
            raise ValueError(
                "displacement_semantics must be one of "
                f"{DISPLACEMENT_SEMANTICS}")
        super().__init__(config, interaction_mode="causal_temporal_candidate")
        self.displacement_semantics = str(displacement_semantics)

    def architecture_config(self) -> dict:
        result = super().architecture_config()
        result["architecture"] = (
            "pyramid_causal3d_integrated_displacement_screen")
        result["displacement_semantics"] = self.displacement_semantics
        return result

    def apply_displacement_basis(self, raw: torch.Tensor) -> torch.Tensor:
        """Map packed raw horizon fields without adding state or parameters."""
        if raw.ndim != 5 or raw.shape[1] != self.config.out_steps:
            raise ValueError(
                "raw displacement must be [B,out_steps,C,H,W]")
        if self.displacement_semantics == "absolute":
            return raw
        return torch.cumsum(raw, dim=1)

    def _frame_apply(self, module: nn.Module,
                     value: torch.Tensor) -> torch.Tensor:
        output = super()._frame_apply(module, value)
        if module is self.transport_head:
            output = self.apply_displacement_basis(output)
        return output


def build_pyramid_causal3d_integrated_displacement_operator(
        displacement_semantics: str = "horizon_integrated",
        **overrides: object,
        ) -> PyramidCausal3DIntegratedDisplacementOperator:
    return PyramidCausal3DIntegratedDisplacementOperator(
        PyramidCausal3DOperatorConfig(**overrides), displacement_semantics)
