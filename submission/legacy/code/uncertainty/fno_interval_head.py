"""Input-dependent interval head for a frozen Track 1 FNO point predictor."""

from __future__ import annotations

import torch
from torch import nn


class FNOIntervalHead(nn.Module):
    """Predict symmetric physical-space u/v half-widths with a pointwise MLP."""

    def __init__(self, hidden: int = 32,
                 base_half_widths: tuple[float, float] = (0.01075, 0.00550)):
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.hidden = int(hidden)
        self.register_buffer(
            "mean", torch.tensor([0.154962569, -0.000517793698]).view(1, 1, 1, 1, 2))
        self.register_buffer(
            "std", torch.tensor([0.0968104079, 0.0159636438]).view(1, 1, 1, 1, 2))
        self.register_buffer(
            "base_half_widths", torch.tensor(base_half_widths).view(1, 2, 1, 1, 1))
        # Features: pred(u,v), last-observed(u,v), absolute jump(u,v), and t/y/x.
        self.network = nn.Sequential(
            nn.Conv3d(9, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, 2, kernel_size=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, prediction: torch.Tensor,
                last_observed: torch.Tensor) -> torch.Tensor:
        """Return half-widths shaped ``(B,T,H,W,2)`` in physical units."""
        if prediction.ndim != 5 or prediction.shape[-1] < 2:
            raise ValueError("prediction must have shape (B,T,H,W,C>=2)")
        if last_observed.ndim == 5:
            if last_observed.shape[1] != 1:
                raise ValueError("5D last_observed must have a singleton time axis")
            last_observed = last_observed[:, 0]
        if last_observed.ndim != 4 or last_observed.shape[-1] < 2:
            raise ValueError("last_observed must have shape (B,H,W,C>=2)")

        pred = prediction[..., :2]
        last = last_observed[..., :2].unsqueeze(1).expand(-1, pred.shape[1], -1, -1, -1)
        pred_z = (pred - self.mean) / self.std
        last_z = (last - self.mean) / self.std
        jump = torch.abs(pred_z - last_z)

        batch, timesteps, height, width, _ = pred.shape
        dtype = pred.dtype
        device = pred.device
        time_grid = torch.linspace(-1.0, 1.0, timesteps, dtype=dtype, device=device)
        y_grid = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
        x_grid = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
        time_grid = time_grid.view(1, timesteps, 1, 1, 1).expand(
            batch, -1, height, width, -1)
        y_grid = y_grid.view(1, 1, height, 1, 1).expand(
            batch, timesteps, -1, width, -1)
        x_grid = x_grid.view(1, 1, 1, width, 1).expand(
            batch, timesteps, height, -1, -1)
        features = torch.cat((pred_z, last_z, jump, time_grid, y_grid, x_grid), dim=-1)
        features = features.permute(0, 4, 1, 2, 3).contiguous()

        # The clamp bounds each learned multiplier to [exp(-2), exp(2)].
        log_multiplier = self.network(features).clamp(-2.0, 2.0)
        half_width = self.base_half_widths * torch.exp(log_multiplier)
        return half_width.permute(0, 2, 3, 4, 1).contiguous()
