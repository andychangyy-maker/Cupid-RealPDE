"""Track 1 v9 TKE4 FNO with input-dependent SPS prediction intervals."""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from fno_interval_head import FNOIntervalHead  # noqa: E402
from load_baseline import load_baseline  # noqa: E402

_CKPT_PATH = os.path.join(_HERE, "sim_real_fno_fp16.pth")
_HEAD_PATH = os.path.join(_HERE, "interval_head.pth")

_MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
_STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
_MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
_STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)

_BATCH = 16
_state = {"model": None, "head": None, "multiplier": None, "device": None}


def _get_models():
    if _state["model"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _meta = load_baseline(_CKPT_PATH, device=device)
        model = model.to(device)
        model.eval()

        payload = torch.load(_HEAD_PATH, map_location=device, weights_only=False)
        head = FNOIntervalHead(hidden=int(payload["head_hidden"])).to(device)
        head.load_state_dict(payload["head_state_dict"])
        head.eval()
        multiplier = torch.tensor(
            payload["robust_multiplier_uv"], dtype=torch.float32,
            device=device).reshape(1, 1, 1, 1, 2)

        _state.update({
            "model": model,
            "head": head,
            "multiplier": multiplier,
            "device": device,
        })
    return (_state["model"], _state["head"],
            _state["multiplier"], _state["device"])


def predict(input_array, metadata=None):
    """Return unchanged TKE4 point predictions and learned finite intervals."""
    x = np.asarray(input_array, dtype=np.float32)
    model, head, multiplier, device = _get_models()

    mean_in = torch.from_numpy(_MEAN_IN).to(device)
    std_in = torch.from_numpy(_STD_IN).to(device)
    mean_tgt = torch.from_numpy(_MEAN_TGT).to(device)
    std_tgt = torch.from_numpy(_STD_TGT).to(device)

    predictions = []
    lowers = []
    uppers = []
    with torch.inference_mode():
        for index in range(0, x.shape[0], _BATCH):
            raw = torch.from_numpy(
                np.ascontiguousarray(x[index:index + _BATCH])).to(device)
            prediction = model((raw - mean_in) / std_in)
            prediction = prediction * std_tgt + mean_tgt
            prediction[..., 2] = 0.0
            half_width_uv = head(
                prediction[..., :2], raw[:, -1, ..., :2]) * multiplier
            half_width = torch.zeros_like(prediction)
            half_width[..., :2] = half_width_uv
            predictions.append(prediction.float().cpu().numpy())
            lowers.append((prediction - half_width).float().cpu().numpy())
            uppers.append((prediction + half_width).float().cpu().numpy())

    prediction = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)
    lower = np.concatenate(lowers, axis=0).astype(np.float32, copy=False)
    upper = np.concatenate(uppers, axis=0).astype(np.float32, copy=False)
    return {"prediction": prediction, "lower": lower, "upper": upper}
