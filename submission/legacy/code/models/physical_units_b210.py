"""History-only physical UV coordinates with the original magnitude observable.

One common UV RMS preserves the u/v amplitude ratio. Existing FiLM layers see
the physical history descriptor; no trainable parameters or extra backbone.
This representation must be trained consistently from Sim through Real/UQ.
"""
import torch

from models.unet2d import history_regime_summary


class PhysicalUnitsMixin:
    @torch.no_grad()
    def set_normalizer_statistics(self, mean_input, mean_target, std_input, std_target):
        super().set_normalizer_statistics(mean_input, mean_target, std_input, std_target)
        for name, value in zip(('mean_input', 'mean_target', 'std_input', 'std_target'),
                               (mean_input, mean_target, std_input, std_target)):
            buffer = getattr(self, 'units_' + name)
            buffer.copy_(torch.as_tensor(value, device=buffer.device, dtype=buffer.dtype))
        # Internal UV input/output coordinates are identical; pressure retains
        # its original global normalizer. This avoids applying persistence twice.
        self.persistence_scale[:2].fill_(1.)
        self.persistence_bias[:2].zero_()
        self.target_std[:2].fill_(1.)

    def physical_coordinates(self, x):
        physical = x[..., :2].float() * self.units_std_input[:2] + self.units_mean_input[:2]
        mean = physical.mean((1, 2, 3), keepdim=True)
        centered = physical - mean
        scale = centered.square().sum(-1, keepdim=True).mean((1, 2, 3), keepdim=True).add(1e-8).sqrt()
        canonical = torch.cat(((centered / scale).to(x.dtype), x[..., 2:]), -1)
        context = history_regime_summary(physical).to(x.dtype)
        return canonical, mean, scale, context

    def restore_units(self, value, mean, scale):
        uv = (value[..., :2].float() * scale + mean - self.units_mean_target[:2]) / self.units_std_target[:2]
        return torch.cat((uv, value[..., 2:].float()), -1)

    def forward_with_features(self, x):
        canonical, mean, scale, context = self.physical_coordinates(x)
        assert self._units_context is None
        self._units_context = context
        try:
            point, features = super().forward_with_features(canonical)
        finally:
            self._units_context = None
        return self.restore_units(point, mean, scale), features

    def forward_uq_training(self, x):
        canonical, mean, scale, context = self.physical_coordinates(x)
        assert self._units_context is None
        self._units_context = context
        self._units_uq_values = (x, mean, scale)
        original_statistics = (self.target_std, self.persistence_scale, self.persistence_bias)
        try:
            point, members, quantiles, lower, upper, features, coverage = super().forward_uq_training(canonical)
        finally:
            self._units_context = None
            self._units_uq_values = None
            self.target_std, self.persistence_scale, self.persistence_bias = original_statistics
        if self._units_uq_global:
            return (self.restore_units(point, mean, scale),
                    self.restore_units(members, mean[:, None], scale[:, None]),
                    quantiles.float(), lower.float(), upper.float(), features, coverage)
        extent_scale = scale / self.units_std_target[:2]
        return (self.restore_units(point, mean, scale),
                self.restore_units(members, mean[:, None], scale[:, None]),
                quantiles.float() * extent_scale[..., None], lower.float() * extent_scale,
                upper.float() * extent_scale, features, coverage)

    def _fused_uq_features(self, features, members, x, prediction=None, *,
                          detach_members=True, fluctuation_features=None):
        if self._units_uq_global:
            history, mean, scale = self._units_uq_values
            # The point has finished in canonical coordinates. Native UQ now
            # sees original target coordinates and its true physical widths.
            self.target_std = self.units_std_target
            self.persistence_scale = self.units_std_input / self.units_std_target
            self.persistence_bias = (self.units_mean_input - self.units_mean_target) / self.units_std_target
            members = self.restore_units(members, mean[:, None], scale[:, None])
            prediction = None if prediction is None else self.restore_units(prediction, mean, scale)
            x = history
        return super()._fused_uq_features(features, members, x, prediction,
                    detach_members=detach_members, fluctuation_features=fluctuation_features)

    def _quantiles_from_fused(self, fused, members):
        if self._units_uq_global:
            _, mean, scale = self._units_uq_values
            members = self.restore_units(members, mean[:, None], scale[:, None])
        return super()._quantiles_from_fused(fused, members)

    def forward_distribution(self, x):
        point, _, _, lower, upper, _, _ = self.forward_uq_training(x)
        return point, lower, upper


def install(model, normalizer, *, uq_global=False):
    assert not isinstance(model, PhysicalUnitsMixin)
    assert model.config.in_channels == model.config.out_channels == 3
    assert model.config.physical_energy_member_sync and model.config.use_regime_film
    assert model.config.dual_decoder_mode == 'none' and model.history_dynamics_state is None
    device = next(model.parameters()).device
    for name in ('mean_input', 'mean_target', 'std_input', 'std_target'):
        model.register_buffer('units_' + name, torch.as_tensor(getattr(normalizer, name), device=device, dtype=torch.float32).clone())
    model.__class__ = type('PhysicalUnitsEnsemble', (PhysicalUnitsMixin, type(model)), {})
    model._units_context = None
    model._units_uq_values = None
    model._units_uq_global = uq_global
    if uq_global:
        assert model.config.residual_mixture_components == 0
        assert model.config.output_space_uq_channels == model.config.output_space_uq_budget_channels == 0
        assert not model.config.budgeted_disagreement_allocation

    def physical_context(module, args):
        assert model._units_context is not None
        return (model._units_context,)

    films = [model.regime_film]
    if model.decoder_regime_film is not None:
        films.extend(model.decoder_regime_film)
    for film in films:
        film.register_forward_pre_hook(physical_context)
    model.set_normalizer_statistics(*(getattr(normalizer, name) for name in ('mean_input', 'mean_target', 'std_input', 'std_target')))
    return model
