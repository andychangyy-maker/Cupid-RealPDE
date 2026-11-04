"""Double the native UQ channels while preserving its initial function."""
from dataclasses import replace

import torch


def widen_uq(model, prefixes):
    old_width = model.config.uq_hidden_channels
    config = replace(model.config, uq_hidden_channels=2 * old_width)
    with torch.random.fork_rng(devices=[]):
        wide = type(model)(config).to(dtype=next(model.parameters()).dtype)
    source, destination = model.state_dict(), wide.state_dict()
    generator = torch.Generator().manual_seed(31010)
    changed = []
    for name, target in destination.items():
        value = source[name]
        if value.shape == target.shape:
            destination[name] = value.clone()
            continue
        if not name.startswith(prefixes):
            raise ValueError('widening changed a point tensor: ' + name)
        if value.ndim == 1:
            assert target.shape[0] == 2 * value.shape[0] == 2 * old_width
            destination[name] = value.repeat_interleave(2, dim=0)
            changed.append(name)
            continue
        assert value.ndim >= 2 and value.shape[2:] == target.shape[2:]
        if target.shape[0] != value.shape[0]:
            assert target.shape[0] == 2 * value.shape[0] == 2 * old_width
            value = value.repeat_interleave(2, dim=0)
        if target.shape[1] != value.shape[1]:
            if name == 'uq_fusion.0.weight':
                expanded = len(model.uq_projections) * old_width
                if model.fluctuation_uq_projections is not None:
                    expanded *= 2
                assert target.shape[1] == value.shape[1] + expanded
            else:
                expanded = value.shape[1]
                assert target.shape[1] == 2 * expanded == 2 * old_width
            head, rest = value[:, :expanded], value[:, expanded:]
            fraction = .25 + .5 * torch.rand(head.shape, generator=generator, dtype=value.dtype)
            split = torch.stack((head * fraction, head * (1 - fraction)), dim=2)
            value = torch.cat((split.flatten(1, 2), rest), dim=1)
        assert value.shape == target.shape, name
        destination[name] = value
        changed.append(name)
    wide.load_state_dict(destination, strict=True)
    assert changed
    for name, value in source.items():
        if not name.startswith(prefixes):
            assert torch.equal(value, wide.state_dict()[name]), name
    return wide, {'old_width': old_width, 'new_width': 2 * old_width,
                  'split_seed': 31010, 'split_fraction': [.25, .75],
                  'changed_tensors': changed,
                  'uq_parameters': sum(p.numel() for n, p in wide.named_parameters() if n.startswith(prefixes))}


def check_initial_function(original, candidate, history, normalizer):
    """Use a fit input only; no held arrays or target-dependent scoring."""
    precision = torch.get_float32_matmul_precision()
    convolution_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        original.to(history.device).eval()
        candidate.eval()
        with torch.no_grad():
            old = original.forward_distribution(history)
            new = candidate.forward_distribution(history)
            assert torch.equal(old[0], new[0]), 'widening changed the initial point function'
            scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
            difference = max(float(((a - b) * scale).abs().max()) for a, b in zip(old[1:], new[1:]))
            assert difference <= 2e-6, ('initial interval function differs', difference)
        return {'fit_batch_size': len(history), 'point_bitwise': True,
                'physical_interval_max_abs': difference, 'tf32': False}
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = convolution_tf32
        original.cpu()
