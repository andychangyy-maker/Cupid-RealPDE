"""Insert zero-weight earlier-history channels into the existing UQ convolution."""
from dataclasses import replace

import torch


def extend_history_uq(model, prefixes):
    assert not model.config.uq_full_history_context
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        extended = type(model)(replace(model.config, uq_full_history_context=True)).float()
    assert torch.equal(before, torch.get_rng_state())
    source, destination = model.state_dict(), extended.state_dict()
    key = 'uq_fusion.0.weight'
    offset = source[key].shape[1]-70
    assert key.startswith(prefixes) and offset == 560
    for name, value in source.items():
        if name != key:
            destination[name] = value.clone()
        else:
            padded = torch.zeros_like(destination[name])
            padded[:, :offset] = value[:, :offset]
            padded[:, offset+30:] = value[:, offset:]
            destination[name] = padded
    extended.load_state_dict(destination, strict=True)
    assert all(torch.equal(value, destination[name]) for name, value in source.items() if name != key)
    assert torch.equal(source[key][:, :offset], destination[key][:, :offset])
    assert torch.equal(source[key][:, offset:], destination[key][:, offset+30:])
    assert sum(p.numel() for p in extended.parameters())-sum(p.numel() for p in model.parameters()) == 43200
    return extended, dict(version='fused_full_history_v2', input='normalized_history_all20_uv',
                          added_parameters=43200, inserted_channels=[offset, offset+30],
                          old_coefficients_bitwise=True, cpu_rng_unchanged=True)
