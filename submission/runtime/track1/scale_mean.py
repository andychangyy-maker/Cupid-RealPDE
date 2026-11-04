"""Immutable input-scale regional mean correction for an output bundle."""
from pathlib import Path
import numpy as np
import torch

_table = None


def correct(raw, bundle):
    global _table
    if _table is None:
        with np.load(Path(__file__).with_name('mean_calibration.npz'), allow_pickle=False) as data:
            region = ((np.arange(32)[:, None] // 16) * 4 + np.arange(64)[None, :] // 16) * 2
            region = region[..., None] + np.arange(2)
            coefficients = torch.as_tensor(data['coefficients'][region][None, None],
                                           dtype=torch.float32, device=bundle.device)
            _table = (coefficients, float(data['center']), float(data['spread']), float(data['shrink']))
    coefficients, center, spread, shrink = _table
    uv = raw[..., :2]
    scale = .5 * torch.log(uv.square().mean((1, 2, 3, 4)).clamp_min(1e-12))
    f = ((scale - center) / spread)[:, None, None, None, None]
    alpha = (coefficients[..., 0] + f * coefficients[..., 1]).clamp(0, .5) * shrink
    direction = uv.mean(1, keepdim=True) - bundle[0, ..., :2].mean(1, keepdim=True)
    shift = direction * alpha * (uv != 0).any(1, keepdim=True)
    bundle[..., :2] += shift[None]
    return bundle
