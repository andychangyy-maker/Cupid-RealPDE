"""Independent full-resolution temporal-mean forecast from physical history."""
import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1),
                         nn.GroupNorm(8, outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1),
                         nn.GroupNorm(8, outputs), nn.SiLU())


class PhysicalMeanUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = block(45, 32)
        self.down1 = block(32, 64)
        self.down2 = block(64, 128)
        self.dropout = nn.Dropout2d(.2)
        self.up1 = block(192, 64)
        self.up0 = block(96, 32)
        self.output = nn.Conv2d(32, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        y, x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 64), indexing='ij')
        self.register_buffer('coordinates', torch.stack((y, x))[None])

    @staticmethod
    def scale(history):
        uv = history[..., :2].float()
        mean = uv.mean((1, 2, 3), keepdim=True)
        scale = (uv-mean).square().sum(-1, keepdim=True).mean((1, 2, 3), keepdim=True).add(1e-8).sqrt()
        return mean, scale

    def forward(self, history):
        uv = history[..., :2].float()
        mean, scale = self.scale(history)
        canonical = ((uv-mean)/scale).permute(0, 1, 4, 2, 3).flatten(1, 2)
        descriptor = torch.cat((mean, torch.log1p(scale)), -1)[:, 0].permute(0, 3, 1, 2)
        context = descriptor.expand(-1, -1, uv.shape[2], uv.shape[3])
        x = torch.cat((canonical, context, self.coordinates.expand(uv.shape[0], -1, -1, -1)), 1)
        a = self.stem(x)
        b = self.down1(F.avg_pool2d(a, 2))
        c = self.dropout(self.down2(F.avg_pool2d(b, 2)))
        b = self.up1(torch.cat((F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b), 1))
        a = self.up0(torch.cat((F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False), a), 1))
        residual = self.output(a).permute(0, 2, 3, 1)
        return uv.mean(1) + scale[:, 0] * residual


def replace_mean(point, mean):
    uv = point[..., :2]
    replacement = uv - uv.mean(1, keepdim=True) + mean[:, None]
    return torch.cat((replacement, point[..., 2:]), -1)
