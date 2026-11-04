"""Metadata-free residual correction for Track 1 FNO predictions."""

from __future__ import annotations

import torch
from torch import nn

try:
    from .structured_interval_head import StructuredIntervalHead
except ImportError:  # Submission ZIP imports modules from its archive root.
    from structured_interval_head import StructuredIntervalHead


class ConditionedResidualCorrector(StructuredIntervalHead):
    """Predict a bounded point correction with learned global flow context.

    The local branch uses the same causal features as the structured interval
    head.  A small 3-D encoder extracts an operating-condition embedding from
    the observed frames and FiLM-modulates the local branch.  Re/AoA are only
    auxiliary training targets; inference needs no metadata.
    """

    def __init__(self, hidden: int = 24, context_hidden: int = 16,
                 architecture: str = "deep",
                 max_correction_uv: tuple[float, float] = (0.020, 0.010)):
        super().__init__(hidden=hidden, architecture=architecture,
                         asymmetric=False, global_context=False)
        if context_hidden <= 0:
            raise ValueError("context_hidden must be positive")
        children = list(self.network.children())
        self.local_trunk = nn.Sequential(*children[:-1])
        del self.network
        self.context_hidden = int(context_hidden)
        self.context_encoder = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(8, context_hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.film = nn.Linear(context_hidden, 2 * hidden)
        self.condition_head = nn.Sequential(
            nn.Linear(context_hidden, context_hidden),
            nn.GELU(),
            nn.Linear(context_hidden, 2),
        )
        self.correction_head = nn.Conv3d(hidden, 2, kernel_size=1)
        self.register_buffer(
            "max_correction_uv",
            torch.tensor(max_correction_uv).view(1, 1, 1, 1, 2),
        )
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def _context(self, observation: torch.Tensor) -> torch.Tensor:
        obs_z = (observation[..., :2] - self.mean) / self.std
        return self.context_encoder(
            obs_z.permute(0, 4, 1, 2, 3).contiguous())

    def forward(self, prediction: torch.Tensor,
                observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return corrected UV prediction and normalized Re/AoA estimates."""
        local = self.local_trunk(self._features(prediction, observation))
        context = self._context(observation)
        scale, bias = self.film(context).chunk(2, dim=1)
        scale = scale.view(scale.shape[0], -1, 1, 1, 1)
        bias = bias.view(bias.shape[0], -1, 1, 1, 1)
        local = local * (1.0 + 0.1 * torch.tanh(scale)) + bias
        raw = self.correction_head(local)
        correction = torch.tanh(
            raw.permute(0, 2, 3, 4, 1).contiguous()) * self.max_correction_uv
        corrected = prediction[..., :2] + correction
        condition = torch.tanh(self.condition_head(context))
        return corrected, condition
