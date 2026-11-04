"""Recenter native residual UQ on a fixed, physically reflected point average."""
import torch


def reflect_normalized(value, mean, std):
    reflected = value.flip(-3).clone()
    reflected[..., 1] = -reflected[..., 1] - 2 * mean[1] / std[1]
    return reflected


def attach_reflected_point(model, normalizer):
    original = model.forward_uq_training

    def forward(x, **kwargs):
        values = list(original(x, **kwargs))
        with torch.no_grad():
            reflected = reflect_normalized(x, normalizer.mean_input, normalizer.std_input)
            other = model(reflected)
            other = reflect_normalized(other, normalizer.mean_target, normalizer.std_target)
        values[0] = (values[0] + other) * .5
        return tuple(values)

    model.forward_uq_training = forward
