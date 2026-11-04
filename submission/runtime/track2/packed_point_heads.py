"""Pack shared-input linear member heads without changing stored learned weights."""
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F


def pack(head_groups):
    layers = nn.ModuleList()
    for level in range(len(head_groups[0])):
        heads = [group[level] for group in head_groups]
        first = heads[0]
        assert all(isinstance(h, nn.Conv2d) and h.kernel_size == (1, 1) and h.stride == (1, 1)
                   and h.padding == (0, 0) and h.groups == 1 for h in heads)
        layer = nn.Conv2d(first.in_channels, sum(h.out_channels for h in heads), 1)
        layer.to(device=first.weight.device, dtype=first.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(torch.cat([h.weight for h in heads], 0))
            layer.bias.copy_(torch.cat([h.bias for h in heads], 0))
        layer.requires_grad_(False).eval()
        layers.append(layer)
    return layers


def members_from_packed(self, x, features):
    batch, _, height, width, _ = x.shape
    raw_all = self._multi_scale(features, self._packed_point_heads, (height, width))
    members = []
    for raw in raw_all.split(self._packed_output_channels, dim=1):
        channels = self.config.out_channels
        if self.config.mean_fluctuation:
            mean_delta = raw[:, :channels].permute(0, 2, 3, 1).unsqueeze(1)
            fluctuation = raw[:, channels:].reshape(batch, self.config.out_steps, channels, height, width).permute(0, 1, 3, 4, 2)
            fluctuation = fluctuation - fluctuation.mean(dim=1, keepdim=True)
            residual = mean_delta + fluctuation
        else:
            residual = raw.reshape(batch, self.config.out_steps, channels, height, width).permute(0, 1, 3, 4, 2)
        members.append(self._persistence(x).unsqueeze(1) + residual)
    stacked = torch.stack(members, dim=1)
    if self.mean_profile_refiner is not None:
        correction = self.mean_profile_refiner(features[-1]).permute(0, 2, 3, 1)
        if self.config.out_channels > 2:
            correction = F.pad(correction, (0, self.config.out_channels - 2))
        stacked = stacked + correction[:, None, None]
    if self.phase_delta is not None:
        correction = self.phase_delta(x)
        if self.config.out_channels > 2:
            correction = F.pad(correction, (0, self.config.out_channels - 2))
        stacked = stacked + correction[:, None]
    return stacked


def install(model):
    assert model.temporal_point_decoder is None and model.member_feature_adapters is None
    assert len(model.member_point_heads) == model.config.ensemble_members
    model._packed_point_heads = pack(model.member_point_heads)
    for layer in model._packed_point_heads:
        layer.to(memory_format=torch.channels_last)
    model._packed_output_channels = model.member_point_heads[0][0].out_channels
    model._members_from_features = MethodType(members_from_packed, model)


def verify_algebra():
    torch.manual_seed(20260925)
    heads = nn.ModuleList([nn.ModuleList([nn.Conv2d(c, 9, 1).double() for c in (13, 7, 5)]) for _ in range(4)])
    features = [torch.randn(2, c, h, w, dtype=torch.float64) for c, h, w in ((13, 3, 5), (7, 6, 10), (5, 12, 20))]
    def multi(group):
        return torch.stack([F.interpolate(h(v), size=(12, 20), mode='bilinear', align_corners=False) for h, v in zip(group, features)]).mean(0)
    original = torch.cat([multi(group) for group in heads], 1)
    changed = multi(pack(heads))
    error = float((changed - original).abs().max())
    assert error < 1e-12, error
    return error
