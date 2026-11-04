"""Per-window UV coordinates, with outputs restored to the original target units."""
import torch


class ReversibleHistoryMixin:
    def canonical_history(self, x):
        variance, mean = torch.var_mean(x[..., :2].float(), dim=(1, 2, 3), correction=0, keepdim=True)
        scale = torch.sqrt(variance + 1e-6)
        value = torch.cat((((x[..., :2].float() - mean) / scale).to(x.dtype), x[..., 2:]), dim=-1)
        ratio = self.history_coordinate_ratio.float()
        return value, mean * ratio + self.history_coordinate_offset.float(), scale * ratio

    @staticmethod
    def restore_point(value, shift, scale):
        return torch.cat((value[..., :2].float() * scale + shift, value[..., 2:].float()), dim=-1)

    def forward_with_features(self, x):
        value, shift, scale = self.canonical_history(x)
        prediction, features = super().forward_with_features(value)
        return self.restore_point(prediction, shift, scale), features

    def forward_uq_training(self, x, **kwargs):
        assert not kwargs.get('return_mixture', False)
        value, shift, scale = self.canonical_history(x)
        prediction, members, quantiles, lower, upper, features, coverage = super().forward_uq_training(value, **kwargs)
        return (self.restore_point(prediction, shift, scale),
                self.restore_point(members, shift[:, None], scale[:, None]),
                quantiles.float() * scale[..., None], lower.float() * scale, upper.float() * scale,
                features, coverage)

    def forward_distribution(self, x):
        value, shift, scale = self.canonical_history(x)
        prediction, _members, _quantiles, lower, upper, _features, _coverage = super().forward_uq_training(value)
        return self.restore_point(prediction, shift, scale), lower.float() * scale, upper.float() * scale


def install(model, normalizer):
    assert not isinstance(model, ReversibleHistoryMixin)
    assert model.config.in_steps == model.config.out_steps == 20
    assert model.config.in_channels == model.config.out_channels == 3
    assert model.config.physical_energy_member_sync
    device = next(model.parameters()).device
    values = {key: torch.as_tensor(getattr(normalizer, key), device=device, dtype=torch.float32)[:2]
              for key in ('mean_input', 'mean_target', 'std_input', 'std_target')}
    model.register_buffer('history_coordinate_ratio', values['std_input'] / values['std_target'])
    model.register_buffer('history_coordinate_offset', (values['mean_input'] - values['mean_target']) / values['std_target'])
    model.__class__ = type('ReversibleHistoryEnsemble', (ReversibleHistoryMixin, type(model)), {})
    return model
