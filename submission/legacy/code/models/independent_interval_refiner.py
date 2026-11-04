"""A fixed small U-Net learning interval features independently of B210."""
import torch
from torch import nn
from torch.nn import functional as F

from .unet2d import ResidualConvBlock


class IndependentIntervalRefiner(nn.Module):
    def __init__(self, target_std):
        super().__init__()
        self.encoder0 = ResidualConvBlock(160, 32, 8)
        self.encoder1 = ResidualConvBlock(32, 64, 8)
        self.encoder2 = ResidualConvBlock(64, 128, 8)
        self.decoder1 = ResidualConvBlock(192, 64, 8)
        self.decoder0 = ResidualConvBlock(96, 32, 8)
        self.output = nn.Conv2d(32, 80, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        scale = torch.as_tensor(target_std).detach().float()[:2]
        self.register_buffer('reference', (scale.new_tensor([.01075, .00550]) / scale).reshape(1, 1, 1, 1, 2))

    def forward(self, history, point, lower, upper):
        batch, steps, height, width, _ = point.shape
        assert steps == 20 and history.shape[1] == 20
        joined = torch.cat([value[..., :2].permute(0, 1, 4, 2, 3).reshape(batch, 40, height, width)
                            for value in (history, point, lower, upper)], dim=1)
        level0 = self.encoder0(joined)
        level1 = self.encoder1(F.avg_pool2d(level0, 2))
        level2 = self.encoder2(F.avg_pool2d(level1, 2))
        value = self.decoder1(torch.cat((F.interpolate(level2, size=level1.shape[-2:], mode='bilinear', align_corners=False), level1), dim=1))
        value = self.decoder0(torch.cat((F.interpolate(value, size=level0.shape[-2:], mode='bilinear', align_corners=False), level0), dim=1))
        raw = self.output(value).reshape(batch, 20, 4, height, width).permute(0, 1, 3, 4, 2)
        shift = raw[..., :2] * self.reference
        width_update = .5 * (lower + upper) * torch.expm1(raw[..., 2:].clamp(-4., 4.))
        return lower + width_update - shift, upper + width_update + shift
