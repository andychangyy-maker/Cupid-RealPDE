"""RealPDE Track 1 FNO baseline with calibrated u/v prediction intervals.

This wrapper preserves the official FNO baseline point prediction and adds
constant physical-space uncertainty bounds fitted on the repository's held-out
real-data trajectories.  Package this file as ``submission.py`` together with
the checkpoint and the official loader dependencies.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from load_baseline import load_baseline  # noqa: E402

_CKPT_PATH = os.path.join(_HERE, "sim_real_fno_fp16.pth")

# GaussianNormalizer statistics fitted on train_real 32x64 windows, channel
# order [u, v, p].  The p channel is unmeasured in the real data.
_MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
_STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
_MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
_STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)

# Calibrated in physical units on all 725 held-out windows.  A trajectory-parity
# two-fold cross-check selected essentially the same widths, reducing the risk
# that they merely overfit individual, strongly-overlapping time windows.
_INTERVAL_HALF_WIDTH = np.array([0.00975, 0.00525, 0.0], dtype=np.float32)

_BATCH = 16

_state = {"model": None, "device": None}


def _get_model():
    if _state["model"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _meta = load_baseline(_CKPT_PATH, device=device)
        model = model.to(device)
        model.eval()
        _state["model"] = model
        _state["device"] = device
    return _state["model"], _state["device"]


def predict(input_array, metadata=None):
    """Return point prediction and calibrated lower/upper bounds."""
    x = np.asarray(input_array, dtype=np.float32)
    model, device = _get_model()

    mean_in = torch.from_numpy(_MEAN_IN).to(device)
    std_in = torch.from_numpy(_STD_IN).to(device)
    mean_tgt = torch.from_numpy(_MEAN_TGT).to(device)
    std_tgt = torch.from_numpy(_STD_TGT).to(device)

    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], _BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + _BATCH])).to(device)
            xb = (xb - mean_in) / std_in
            yb = model(xb)
            yb = yb * std_tgt + mean_tgt
            outs.append(yb.float().cpu().numpy())

    prediction = np.concatenate(outs, axis=0).astype(np.float32, copy=False)
    prediction[..., 2] = 0.0
    half_width = _INTERVAL_HALF_WIDTH.reshape((1,) * (prediction.ndim - 1) + (-1,))
    lower = (prediction - half_width).astype(np.float32, copy=False)
    upper = (prediction + half_width).astype(np.float32, copy=False)
    return {"prediction": prediction, "lower": lower, "upper": upper}
