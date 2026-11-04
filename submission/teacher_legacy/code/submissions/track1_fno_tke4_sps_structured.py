"""Track 1 v9 TKE4 FNO with spatial asymmetric SPS intervals."""

from __future__ import annotations

import os
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from load_baseline import load_baseline  # noqa: E402
from structured_interval_head import StructuredIntervalHead  # noqa: E402


_CKPT_PATH = os.path.join(_HERE, "sim_real_fno_fp16.pth")
_HEAD_PATH = os.path.join(_HERE, "interval_head.pth")

_MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
_STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
_MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
_STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)

_BATCH = 16
_state = {
    "model": None,
    "head": None,
    "lower_multiplier": None,
    "upper_multiplier": None,
    "lower_power": None,
    "upper_power": None,
    "device": None,
}


def _as_broadcast_multiplier(values, device):
    """Accept legacy per-channel or horizon-by-channel calibration arrays."""
    multiplier = torch.tensor(values, dtype=torch.float32, device=device)
    if multiplier.ndim == 1 and tuple(multiplier.shape) == (2,):
        return multiplier.reshape(1, 1, 1, 1, 2)
    if multiplier.ndim == 2 and multiplier.shape[1] == 2:
        return multiplier.reshape(1, multiplier.shape[0], 1, 1, 2)
    raise ValueError(
        "Interval multipliers must have shape (2,) or (T_out, 2)")


def _get_models():
    if _state["model"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _meta = load_baseline(_CKPT_PATH, device=device)
        model = model.to(device)
        model.eval()

        payload = torch.load(_HEAD_PATH, map_location=device, weights_only=False)
        head = StructuredIntervalHead(
            hidden=int(payload["head_hidden"]),
            architecture=payload["architecture"],
            asymmetric=bool(payload["asymmetric"]),
            global_context=bool(payload.get("global_context", False)),
        ).to(device)
        head.load_state_dict(payload["head_state_dict"])
        head.eval()
        lower_multiplier = _as_broadcast_multiplier(
            payload["robust_lower_multiplier_uv"], device)
        upper_multiplier = _as_broadcast_multiplier(
            payload["robust_upper_multiplier_uv"], device)
        lower_power = _as_broadcast_multiplier(
            payload.get("lower_width_power_uv", [1.0, 1.0]), device)
        upper_power = _as_broadcast_multiplier(
            payload.get("upper_width_power_uv", [1.0, 1.0]), device)
        _state.update({
            "model": model,
            "head": head,
            "lower_multiplier": lower_multiplier,
            "upper_multiplier": upper_multiplier,
            "lower_power": lower_power,
            "upper_power": upper_power,
            "device": device,
        })
    return (
        _state["model"], _state["head"],
        _state["lower_multiplier"], _state["upper_multiplier"],
        _state["lower_power"], _state["upper_power"],
        _state["device"],
    )


def predict(input_array, metadata=None):
    """Return unchanged point predictions and finite per-element bounds."""
    x = np.asarray(input_array, dtype=np.float32)
    (model, head, lower_multiplier, upper_multiplier,
     lower_power, upper_power, device) = _get_models()

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
            lower_uv, upper_uv = head(prediction[..., :2], raw)
            base_width = head.base_half_widths
            lower_uv = base_width * (lower_uv / base_width).pow(lower_power)
            upper_uv = base_width * (upper_uv / base_width).pow(upper_power)
            lower_width = torch.zeros_like(prediction)
            upper_width = torch.zeros_like(prediction)
            lower_width[..., :2] = lower_uv * lower_multiplier
            upper_width[..., :2] = upper_uv * upper_multiplier
            predictions.append(prediction.float().cpu().numpy())
            lowers.append((prediction - lower_width).float().cpu().numpy())
            uppers.append((prediction + upper_width).float().cpu().numpy())

    prediction = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)
    lower = np.concatenate(lowers, axis=0).astype(np.float32, copy=False)
    upper = np.concatenate(uppers, axis=0).astype(np.float32, copy=False)
    return {"prediction": prediction, "lower": lower, "upper": upper}
