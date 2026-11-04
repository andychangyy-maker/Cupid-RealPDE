"""Single shared nonlinear feature correction for the Track1 point branch."""
import torch
from torch import nn


class SharedPointFeatureRefiner(nn.Module):
    def __init__(self, width: int, hidden: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, width, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.body(feature)
