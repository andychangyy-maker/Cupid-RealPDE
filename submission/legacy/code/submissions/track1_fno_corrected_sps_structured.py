"""Track 1 FNO with conditioned point correction and structured intervals."""

from __future__ import annotations

import os
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from conditioned_residual_corrector import ConditionedResidualCorrector  # noqa: E402
from load_baseline import load_baseline  # noqa: E402
from structured_interval_head import StructuredIntervalHead  # noqa: E402


_CKPT_PATH = os.path.join(_HERE, "sim_real_fno_fp16.pth")
_CORRECTOR_PATH = os.path.join(_HERE, "correction_head.pth")
_INTERVAL_PATH = os.path.join(_HERE, "interval_head.pth")

_MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
_STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
_MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
_STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)

_BATCH = 16
_state = {
    "model": None,
    "corrector": None,
    "interval": None,
    "lower_multiplier": None,
    "upper_multiplier": None,
    "lower_power": None,
    "upper_power": None,
    "correction_scale": None,
    "continuity_weight": None,
    "continuity_power": None,
    "device": None,
}


def _as_broadcast_multiplier(values, device):
    multiplier = torch.tensor(values, dtype=torch.float32, device=device)
    if multiplier.ndim == 1 and tuple(multiplier.shape) == (2,):
        return multiplier.reshape(1, 1, 1, 1, 2)
    if multiplier.ndim == 2 and multiplier.shape[1] == 2:
        return multiplier.reshape(1, multiplier.shape[0], 1, 1, 2)
    raise ValueError("Interval calibration must have shape (2,) or (T_out, 2)")


def _get_models():
    if _state["model"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _meta = load_baseline(_CKPT_PATH, device=device)
        model = model.to(device)
        model.eval()

        correction_payload = torch.load(
            _CORRECTOR_PATH, map_location=device, weights_only=False)
        corrector = ConditionedResidualCorrector(
            hidden=int(correction_payload["hidden"]),
            context_hidden=int(correction_payload["context_hidden"]),
            architecture=correction_payload["architecture"],
        ).to(device)
        corrector.load_state_dict(correction_payload["corrector_state_dict"])
        corrector.eval()

        interval_payload = torch.load(
            _INTERVAL_PATH, map_location=device, weights_only=False)
        interval = StructuredIntervalHead(
            hidden=int(interval_payload["head_hidden"]),
            architecture=interval_payload["architecture"],
            asymmetric=bool(interval_payload["asymmetric"]),
            global_context=bool(interval_payload.get("global_context", False)),
            condition_context=bool(
                interval_payload.get("condition_context", False)),
        ).to(device)
        interval.load_state_dict(interval_payload["head_state_dict"])
        interval.eval()
        lower_multiplier = _as_broadcast_multiplier(
            interval_payload["robust_lower_multiplier_uv"], device)
        upper_multiplier = _as_broadcast_multiplier(
            interval_payload["robust_upper_multiplier_uv"], device)
        lower_power = _as_broadcast_multiplier(
            interval_payload.get("lower_width_power_uv", [1.0, 1.0]), device)
        upper_power = _as_broadcast_multiplier(
            interval_payload.get("upper_width_power_uv", [1.0, 1.0]), device)
        correction_scale = float(interval_payload.get("correction_scale", 1.0))
        continuity_weight = float(interval_payload.get("continuity_weight", 0.0))
        continuity_power = float(interval_payload.get("continuity_power", 4.0))
        _state.update({
            "model": model,
            "corrector": corrector,
            "interval": interval,
            "lower_multiplier": lower_multiplier,
            "upper_multiplier": upper_multiplier,
            "lower_power": lower_power,
            "upper_power": upper_power,
            "correction_scale": correction_scale,
            "continuity_weight": continuity_weight,
            "continuity_power": continuity_power,
            "device": device,
        })
    return tuple(_state[name] for name in (
        "model", "corrector", "interval", "lower_multiplier",
        "upper_multiplier", "lower_power", "upper_power",
        "correction_scale", "continuity_weight", "continuity_power",
        "device"))


def predict(input_array, metadata=None):
    """Return corrected points and finite per-element prediction intervals."""
    x = np.asarray(input_array, dtype=np.float32)
    (model, corrector, interval, lower_multiplier, upper_multiplier,
     lower_power, upper_power, correction_scale, continuity_weight,
     continuity_power, device) = _get_models()
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
            corrected_uv, condition = corrector(prediction[..., :2], raw)
            prediction[..., :2] += correction_scale * (
                corrected_uv - prediction[..., :2])
            if continuity_weight > 0.0:
                phase = torch.linspace(
                    1.0, 0.0, prediction.shape[1], dtype=prediction.dtype,
                    device=prediction.device).pow(continuity_power)
                ramp = phase.view(1, -1, 1, 1, 1)
                jump = raw[:, -1:, ..., :2] - prediction[:, :1, ..., :2]
                prediction[..., :2] += continuity_weight * ramp * jump
            # The interval head was trained against the deployed correction
            # scale.  This is equal to ``corrected_uv`` at scale 1, but using
            # the actual point keeps scale-0/partial-correction ablations exact.
            lower_uv, upper_uv = interval(
                prediction[..., :2], raw, condition)
            base_width = interval.base_half_widths
            lower_uv = base_width * (lower_uv / base_width).pow(lower_power)
            upper_uv = base_width * (upper_uv / base_width).pow(upper_power)
            lower_width = torch.zeros_like(prediction)
            upper_width = torch.zeros_like(prediction)
            lower_width[..., :2] = lower_uv * lower_multiplier
            upper_width[..., :2] = upper_uv * upper_multiplier
            predictions.append(prediction.float().cpu().numpy())
            lowers.append((prediction - lower_width).float().cpu().numpy())
            uppers.append((prediction + upper_width).float().cpu().numpy())

    prediction = np.concatenate(predictions).astype(np.float32, copy=False)
    lower = np.concatenate(lowers).astype(np.float32, copy=False)
    upper = np.concatenate(uppers).astype(np.float32, copy=False)
    return {"prediction": prediction, "lower": lower, "upper": upper}
