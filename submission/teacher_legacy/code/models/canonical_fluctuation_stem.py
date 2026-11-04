"""Add a sample-local normalized temporal fluctuation view to the existing stem."""
import torch
from torch import nn


class CanonicalFluctuationStem(nn.Conv2d):
    def forward(self, features):
        base = super().forward(features)
        batch, _, height, width = features.shape
        history = features[:, :60].reshape(batch, 20, 3, height, width)[:, :, :2].float()
        fluctuation = history - history.mean(dim=1, keepdim=True)
        rms = (fluctuation.square().mean(dim=(1, 3, 4), keepdim=True) + 1e-6).sqrt()
        shape = (fluctuation / rms).reshape(batch, 40, height, width)
        amplitude = rms.log1p().reshape(batch, 2, 1, 1).expand(-1, -1, height, width)
        view = torch.cat((shape, amplitude), dim=1).to(features.dtype)
        return base + self.fluctuation_projection(view)


def install(model):
    assert type(model.stem) is nn.Conv2d
    assert model.config.in_steps == 20 and model.config.in_channels == 3
    assert model.stem.out_channels == 210 and model.stem.in_channels >= 60
    assert sum(p.numel() for p in model.parameters()) == 178705575
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    old_parameters = dict(model.named_parameters())
    with torch.random.fork_rng(devices=[]):
        projection = nn.Conv2d(42, 210, 1)
        nn.init.zeros_(projection.weight)
        nn.init.zeros_(projection.bias)
    model.stem.__class__ = CanonicalFluctuationStem
    model.stem.fluctuation_projection = projection.to(model.stem.weight)
    assert sum(p.numel() for p in projection.parameters()) == 9030
    assert sum(p.numel() for p in model.parameters()) == 178714605
    assert all(dict(model.named_parameters())[name] is value for name, value in old_parameters.items())
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng:
        assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
