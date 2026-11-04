"""Input-normalized direct forecasting with separate mean, RMS and time shape."""
import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())


class PointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = block(46, 48); self.e1 = block(48, 96)
        self.e2 = block(96, 192); self.e3 = block(192, 384)
        self.d2 = block(576, 192); self.d1 = block(288, 96); self.d0 = block(144, 48)
        self.mean_head = nn.Conv2d(48, 2, 1)
        self.energy_head = nn.Conv2d(48, 2, 1)
        self.shape_head = nn.Conv2d(48, 40, 1)
        for layer in [self.mean_head, self.energy_head]:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, history, details=False):
        uv = history[..., :2].float()
        mean = uv.mean(1, keepdim=True)
        dynamic = uv-mean
        scale = dynamic.square().mean((1, 2, 3), keepdim=True).clamp_min(1e-8).sqrt()
        local_rms = dynamic.square().mean(1).clamp_min(1e-8).sqrt()
        b, _, h, w, _ = uv.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=uv.device),
                               torch.linspace(-1, 1, w, device=uv.device), indexing='ij')
        coordinates = torch.stack((yy, xx))[None].expand(b, -1, -1, -1)
        normalized = (dynamic/scale).permute(0, 1, 4, 2, 3).flatten(1, 2)
        magnitude = torch.log(scale[:, 0]/.01).permute(0, 3, 1, 2).expand(-1, -1, h, w)
        x = torch.cat((normalized, (mean[:, 0]/.1).permute(0, 3, 1, 2), magnitude, coordinates), 1)
        x = x.to(self.e0[0].weight.dtype)
        a = self.e0(x); b1 = self.e1(F.avg_pool2d(a, 2))
        c = self.e2(F.avg_pool2d(b1, 2)); d = self.e3(F.avg_pool2d(c, 2))
        c = self.d2(torch.cat((F.interpolate(d, size=c.shape[-2:], mode='bilinear', align_corners=False), c), 1))
        b1 = self.d1(torch.cat((F.interpolate(c, size=b1.shape[-2:], mode='bilinear', align_corners=False), b1), 1))
        a = self.d0(torch.cat((F.interpolate(b1, size=a.shape[-2:], mode='bilinear', align_corners=False), a), 1))
        future_mean = mean[:, 0]+self.mean_head(a).float().permute(0, 2, 3, 1)*scale[:, 0]
        future_rms = local_rms*torch.exp(.7*torch.tanh(self.energy_head(a).float().permute(0, 2, 3, 1)))
        shape = self.shape_head(a).float().reshape(len(uv), 20, 2, h, w).permute(0, 1, 3, 4, 2)
        shape = shape-shape.mean(1, keepdim=True)
        shape = shape/shape.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
        forecast = future_mean[:, None]+future_rms[:, None]*shape
        if details:
            return forecast, future_mean, future_rms
        return forecast


def combine(p, t):
    """Fixed historical 50/50 shape combination, P mean and average UV energy."""
    pu, tu = p[..., :2], t[..., :2]
    mean = pu.mean(1, keepdim=True)
    pd, td = pu-mean, tu-tu.mean(1, keepdim=True)
    dynamic = .5*(pd+td)
    target_energy = .5*(pd.square().sum(-1).mean(1)+td.square().sum(-1).mean(1))
    actual_energy = dynamic.square().sum(-1).mean(1)
    gain = (target_energy/actual_energy.clamp_min(1e-12)).sqrt().clamp_max(2)
    return mean+dynamic*gain[:, None, :, :, None]
