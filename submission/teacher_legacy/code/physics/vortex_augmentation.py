"""Physics-shaped counterfactual perturbations for released flow windows."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _curl_streamfunction(streamfunction: torch.Tensor) -> torch.Tensor:
    padded = F.pad(streamfunction, (1, 1, 1, 1), mode="replicate")
    d_column = 0.5 * (
        padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2])
    d_row = 0.5 * (
        padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1])
    # Physical y points opposite to the array-row direction.  With
    # (du,dv)=(d psi/dy,-d psi/dx), both components carry a minus sign here.
    return torch.cat((-d_row, -d_column), dim=1)


def _advect_perturbation(perturbation: torch.Tensor,
                         base_velocity: torch.Tensor,
                         mask: torch.Tensor,
                         advection_cells: float) -> torch.Tensor:
    batch, _channels, height, width = perturbation.shape
    row = torch.linspace(-1.0, 1.0, height, device=perturbation.device,
                         dtype=perturbation.dtype)
    column = torch.linspace(-1.0, 1.0, width, device=perturbation.device,
                            dtype=perturbation.dtype)
    yy, xx = torch.meshgrid(row, column, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(
        batch, -1, -1, -1).clone()
    grid[..., 0] -= (
        2.0 * advection_cells * base_velocity[:, 0] / (width - 1))
    grid[..., 1] += (
        2.0 * advection_cells * base_velocity[:, 1] / (height - 1))
    numerator = F.grid_sample(
        perturbation * mask, grid, mode="bilinear", padding_mode="border",
        align_corners=True)
    denominator = F.grid_sample(
        mask, grid, mode="bilinear", padding_mode="border",
        align_corners=True)
    result = numerator / denominator.clamp_min(1.0e-4)
    return torch.where(denominator > 0.05, result, torch.zeros_like(result))


def augment_vortex_sequence(
        history: torch.Tensor, target: torch.Tensor, *, probability: float,
        amplitude: float, advection_cells: float = 0.5,
        diffusion: float = 0.05,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add a small divergence-free perturbation evolved across all 40 frames.

    ``amplitude`` is the perturbation RMS as a fraction of each sample's
    released u/v RMS.  The same counterfactual vortex is advected continuously
    through input and target, unlike independent input noise.
    """
    if history.ndim != 5 or target.ndim != 5:
        raise ValueError("history and target must be rank-five flow windows")
    if history.shape[0] != target.shape[0] or history.shape[2:4] != target.shape[2:4]:
        raise ValueError("history and target batch/spatial shapes must match")
    if history.shape[-1] < 2 or target.shape[-1] < 2:
        raise ValueError("at least u/v channels are required")
    if not 0.0 <= probability <= 1.0 or amplitude < 0.0:
        raise ValueError("probability/amplitude are invalid")
    if advection_cells < 0.0 or not 0.0 <= diffusion <= 1.0:
        raise ValueError("advection cells/diffusion are invalid")
    if probability == 0.0 or amplitude == 0.0:
        selected = torch.zeros(
            history.shape[0], device=history.device, dtype=torch.bool)
        return history, target, selected

    sequence = torch.cat((history, target), dim=1)
    batch, steps, height, width, _channels = sequence.shape
    selected = torch.rand(batch, device=sequence.device) < probability
    coarse_height = max(2, height // 8)
    coarse_width = max(2, width // 8)
    streamfunction = torch.randn(
        batch, 1, coarse_height, coarse_width,
        device=sequence.device, dtype=sequence.dtype)
    streamfunction = F.interpolate(
        streamfunction, size=(height, width), mode="bicubic",
        align_corners=False)
    streamfunction = F.avg_pool2d(
        F.pad(streamfunction, (2, 2, 2, 2), mode="reflect"),
        5, stride=1)
    perturbation = _curl_streamfunction(streamfunction)

    fluid_any = sequence[..., :2].abs().sum(dim=-1).ne(0.0)
    initial_mask = fluid_any[:, 0].unsqueeze(1).to(sequence.dtype)
    velocity_rms = (
        (sequence[..., :2].square()
         * fluid_any.unsqueeze(-1)).sum(dim=(1, 2, 3, 4))
        / fluid_any.sum(dim=(1, 2, 3)).clamp_min(1).mul(2)
    ).sqrt()
    perturbation_rms = (
        (perturbation.square() * initial_mask).sum(dim=(1, 2, 3))
        / initial_mask.sum(dim=(1, 2, 3)).clamp_min(1).mul(2)
    ).sqrt().clamp_min(1.0e-8)
    scale = amplitude * velocity_rms / perturbation_rms
    scale = scale * selected.to(scale.dtype)
    perturbation = perturbation * scale[:, None, None, None] * initial_mask

    augmented = sequence.clone()
    for index in range(steps):
        mask = fluid_any[:, index].unsqueeze(1).to(sequence.dtype)
        augmented[:, index, ..., :2] += (
            perturbation * mask).permute(0, 2, 3, 1)
        if index + 1 == steps:
            continue
        base_velocity = sequence[:, index, ..., :2].permute(0, 3, 1, 2)
        perturbation = _advect_perturbation(
            perturbation, base_velocity, mask, advection_cells)
        if diffusion:
            smoothed = F.avg_pool2d(
                F.pad(perturbation, (1, 1, 1, 1), mode="replicate"),
                3, stride=1)
            perturbation = perturbation + diffusion * (
                smoothed - perturbation)
        next_mask = fluid_any[:, index + 1].unsqueeze(1).to(sequence.dtype)
        perturbation = perturbation * next_mask
    return augmented[:, :history.shape[1]], augmented[:, history.shape[1]:], selected

