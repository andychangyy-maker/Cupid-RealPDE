"""Spatial and asymmetric interval heads for Track 1 FNO predictions."""

from __future__ import annotations

import torch
from torch import nn


class StructuredIntervalHead(nn.Module):
    """Predict local lower/upper half-widths from flow and uncertainty features."""

    local_feature_channels = 15
    global_feature_channels = 8

    def __init__(self, hidden: int = 24, architecture: str = "spatial",
                 asymmetric: bool = True,
                 global_context: bool = False,
                 condition_context: bool = False,
                 base_half_widths: tuple[float, float] = (0.01075, 0.00550)):
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if architecture not in ("pointwise", "spatial", "dilated", "deep"):
            raise ValueError(
                "architecture must be pointwise, spatial, dilated, or deep")
        self.hidden = int(hidden)
        self.architecture = architecture
        self.asymmetric = bool(asymmetric)
        self.global_context = bool(global_context)
        self.condition_context = bool(condition_context)
        self.feature_channels = (
            self.local_feature_channels
            + (self.global_feature_channels if self.global_context else 0)
            + (2 if self.condition_context else 0)
        )
        self.register_buffer(
            "mean", torch.tensor([0.154962569, -0.000517793698]).view(1, 1, 1, 1, 2))
        self.register_buffer(
            "std", torch.tensor([0.0968104079, 0.0159636438]).view(1, 1, 1, 1, 2))
        self.register_buffer(
            "base_half_widths", torch.tensor(base_half_widths).view(1, 1, 1, 1, 2))

        output_channels = 4 if asymmetric else 2
        if architecture == "pointwise":
            convolutions = (
                nn.Conv3d(self.feature_channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(hidden, hidden, kernel_size=1),
                nn.GELU(),
            )
        elif architecture == "dilated":
            # A dense 3x3x3 layer followed by dilation 2 has a 7x7x7
            # receptive field without increasing the parameter count.
            convolutions = (
                nn.Conv3d(self.feature_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv3d(
                    hidden, hidden, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
            )
        else:
            convolutions = (
                nn.Conv3d(self.feature_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
            )
            if architecture == "deep":
                # Three dense layers also yield a 7x7x7 receptive field, while
                # retaining every intermediate spatial interaction.
                convolutions += (
                    nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
                    nn.GELU(),
                )
        self.network = nn.Sequential(
            *convolutions,
            nn.Conv3d(hidden, output_channels, kernel_size=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _gradient_magnitude(value: torch.Tensor) -> torch.Tensor:
        """Finite-difference gradient magnitude over y/x for channel-last fields."""
        dy = torch.zeros_like(value)
        dx = torch.zeros_like(value)
        dy[:, :, 1:-1] = 0.5 * (value[:, :, 2:] - value[:, :, :-2])
        dy[:, :, 0] = value[:, :, 1] - value[:, :, 0]
        dy[:, :, -1] = value[:, :, -1] - value[:, :, -2]
        dx[:, :, :, 1:-1] = 0.5 * (value[:, :, :, 2:] - value[:, :, :, :-2])
        dx[:, :, :, 0] = value[:, :, :, 1] - value[:, :, :, 0]
        dx[:, :, :, -1] = value[:, :, :, -1] - value[:, :, :, -2]
        return torch.sqrt(dy.square() + dx.square() + 1e-8)

    def _features(self, prediction: torch.Tensor,
                  observation: torch.Tensor,
                  condition: torch.Tensor | None = None) -> torch.Tensor:
        if prediction.ndim != 5 or prediction.shape[-1] < 2:
            raise ValueError("prediction must have shape (B,T,H,W,C>=2)")
        if observation.ndim != 5 or observation.shape[-1] < 2:
            raise ValueError("observation must have shape (B,T,H,W,C>=2)")
        if (prediction.shape[0] != observation.shape[0]
                or prediction.shape[2:4] != observation.shape[2:4]):
            raise ValueError("prediction and observation batch/spatial shapes must match")

        pred_z = (prediction[..., :2] - self.mean) / self.std
        obs_z = (observation[..., :2] - self.mean) / self.std
        timesteps = pred_z.shape[1]
        last_z = obs_z[:, -1:].expand(-1, timesteps, -1, -1, -1)
        jump = torch.abs(pred_z - last_z)
        obs_std = obs_z.std(dim=1, keepdim=True, unbiased=False).expand(
            -1, timesteps, -1, -1, -1)
        pred_std = pred_z.std(dim=1, keepdim=True, unbiased=False).expand(
            -1, timesteps, -1, -1, -1)
        gradient = self._gradient_magnitude(pred_z)

        batch, _, height, width, _ = pred_z.shape
        dtype = pred_z.dtype
        device = pred_z.device
        time_grid = torch.linspace(-1.0, 1.0, timesteps, dtype=dtype, device=device)
        y_grid = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
        x_grid = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
        time_grid = time_grid.view(1, timesteps, 1, 1, 1).expand(
            batch, -1, height, width, -1)
        y_grid = y_grid.view(1, 1, height, 1, 1).expand(
            batch, timesteps, -1, width, -1)
        x_grid = x_grid.view(1, 1, 1, width, 1).expand(
            batch, timesteps, height, -1, -1)
        features = torch.cat((
            pred_z, last_z, jump, obs_std, pred_std, gradient,
            time_grid, y_grid, x_grid,
        ), dim=-1)
        if self.global_context:
            # Frame-level prediction statistics expose horizon-dependent flow
            # intensity. Observation-level statistics summarize the operating
            # condition without relying on metadata, which is empty on scored
            # calls. All features are repeated spatially so either the 1x1x1 or
            # spatial head can consume them.
            pred_frame_mean = pred_z.mean(dim=(2, 3), keepdim=True).expand(
                -1, -1, height, width, -1)
            pred_frame_std = pred_z.std(
                dim=(2, 3), keepdim=True, unbiased=False).expand(
                    -1, -1, height, width, -1)
            obs_global_mean = obs_z.mean(
                dim=(1, 2, 3), keepdim=True).expand(
                    -1, timesteps, height, width, -1)
            obs_global_std = obs_z.std(
                dim=(1, 2, 3), keepdim=True, unbiased=False).expand(
                    -1, timesteps, height, width, -1)
            features = torch.cat((
                features, pred_frame_mean, pred_frame_std,
                obs_global_mean, obs_global_std,
            ), dim=-1)
        if self.condition_context:
            if (condition is None or condition.ndim != 2
                    or tuple(condition.shape) != (batch, 2)):
                raise ValueError(
                    "condition-context heads require condition shape (B, 2)")
            expanded_condition = condition.to(
                dtype=dtype, device=device).view(batch, 1, 1, 1, 2).expand(
                    -1, timesteps, height, width, -1)
            features = torch.cat((features, expanded_condition), dim=-1)
        return features.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, prediction: torch.Tensor,
                observation: torch.Tensor,
                condition: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positive lower and upper half-widths shaped ``(B,T,H,W,2)``."""
        output = self.network(self._features(prediction, observation, condition))
        output = output.permute(0, 2, 3, 4, 1).contiguous().clamp(-2.5, 2.5)
        if self.asymmetric:
            lower_log = output[..., :2]
            upper_log = output[..., 2:]
        else:
            lower_log = upper_log = output
        lower = self.base_half_widths * torch.exp(lower_log)
        upper = self.base_half_widths * torch.exp(upper_log)
        return lower, upper
