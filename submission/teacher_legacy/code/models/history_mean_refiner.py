"""Direct physical-history input for the existing temporal-mean refiner."""
import torch
from torch import nn


class HistoryMeanRefiner(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        first = native[0]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(32012)
            self.history_projection = nn.Conv2d(43, first.out_channels, 1, bias=False)
            nn.init.zeros_(self.history_projection.weight)
        self.history_projection.to(device=first.weight.device, dtype=first.weight.dtype)
        self.history = None

    def forward(self, features):
        assert self.history is not None
        value = self.native[0](features) + self.history_projection(self.history)
        for layer in self.native[1:]:
            value = layer(value)
        return value


class HistoryMeanMixin:
    def _members_from_features(self, x, features, *args, **kwargs):
        physical = x[..., :2].float() * self.mean_history_std[:2] + self.mean_history_mean[:2]
        mean = physical.mean((1, 2, 3), keepdim=True)
        centered = physical - mean
        scale = centered.square().sum(-1, keepdim=True).mean((1, 2, 3), keepdim=True).add(1e-8).sqrt()
        canonical = (centered / scale).permute(0, 1, 4, 2, 3).flatten(1, 2)
        reference = self.mean_history_std[:2].square().sum().sqrt().clamp_min(1e-8)
        descriptor = torch.cat((mean / reference, torch.log1p(scale / reference)), -1)
        descriptor = descriptor[:, 0].permute(0, 3, 1, 2).expand(-1, -1, x.shape[2], x.shape[3])
        refiner = self.mean_profile_refiner
        assert refiner.history is None
        refiner.history = torch.cat((canonical, descriptor), 1).to(features[-1])
        try:
            return super()._members_from_features(x, features, *args, **kwargs)
        finally:
            refiner.history = None


def install(model, normalizer):
    assert model.config.out_steps == 20 and model.config.mean_fluctuation
    assert model.config.dual_decoder_mode == 'none'
    assert model.config.physical_energy_member_sync
    assert model.mean_profile_refiner is not None
    assert not isinstance(model, HistoryMeanMixin)
    device = next(model.parameters()).device
    model.register_buffer('mean_history_mean', torch.as_tensor(normalizer.mean_input, device=device).float().clone())
    model.register_buffer('mean_history_std', torch.as_tensor(normalizer.std_input, device=device).float().clone())
    model.mean_profile_refiner = HistoryMeanRefiner(model.mean_profile_refiner)
    model.__class__ = type('HistoryMeanEnsemble', (HistoryMeanMixin, type(model)), {})
    return model
