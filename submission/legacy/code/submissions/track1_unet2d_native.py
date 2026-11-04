"""Track 1 v9 U-Net forecaster with native asymmetric intervals."""

from __future__ import annotations

import os
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from unet2d import UNet2d, UNet2dConfig  # noqa: E402


_CHECKPOINT = os.path.join(_HERE, "unet2d.pth")
_MEAN_INPUT = (0.154960855842, -0.000513992854, 0.0)
_MEAN_TARGET = (0.154962569475, -0.000517793698, 0.0)
_STD_INPUT = (0.096805654466, 0.015960684046, 1.0)
_STD_TARGET = (0.096810407937, 0.015963643789, 1.0)
_BATCH_SIZE = 32
_state = {
    "model": None, "device": None, "dtype": None,
    "lower_multiplier": None, "upper_multiplier": None,
}


def _get_model():
    if _state["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(_CHECKPOINT, map_location="cpu", weights_only=False)
        config_dict = dict(payload["model_config"])
        architecture = config_dict.pop("architecture", "unet")
        if architecture != "unet":
            raise ValueError(f"packaged checkpoint is {architecture}, not unet")
        config = UNet2dConfig(**config_dict)
        if not config.predict_uncertainty:
            raise ValueError("packaged checkpoint has no native interval decoder")
        dtype_name = payload.get("inference_dtype", "float32")
        dtype = torch.float16 if dtype_name == "float16" else torch.float32
        model = UNet2d(config).to(device=device, dtype=dtype)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        model.set_normalizer_statistics(
            torch.tensor(_MEAN_INPUT, device=device),
            torch.tensor(_MEAN_TARGET, device=device),
            torch.tensor(_STD_INPUT, device=device),
            torch.tensor(_STD_TARGET, device=device),
        )
        calibration = payload.get("interval_calibration", {})
        lower_multiplier = torch.as_tensor(
            calibration.get("lower_multiplier_uv", [[1.0, 1.0]] * 20),
            device=device, dtype=dtype)
        upper_multiplier = torch.as_tensor(
            calibration.get("upper_multiplier_uv", [[1.0, 1.0]] * 20),
            device=device, dtype=dtype)
        if lower_multiplier.shape != (20, 2) or upper_multiplier.shape != (20, 2):
            raise ValueError("interval calibration must have shape (20, 2)")
        if (not torch.isfinite(lower_multiplier).all()
                or not torch.isfinite(upper_multiplier).all()
                or torch.any(lower_multiplier <= 0)
                or torch.any(upper_multiplier <= 0)):
            raise ValueError("interval calibration must be finite and positive")
        _state.update({
            "model": model, "device": device, "dtype": dtype,
            "lower_multiplier": lower_multiplier.view(1, 20, 1, 1, 2),
            "upper_multiplier": upper_multiplier.view(1, 20, 1, 1, 2),
        })
    return (_state["model"], _state["device"], _state["dtype"],
            _state["lower_multiplier"], _state["upper_multiplier"])


def predict(input_array, metadata=None):
    """Return float32 point forecasts and per-element lower/upper bounds."""
    raw_input = np.asarray(input_array, dtype=np.float32)
    if raw_input.ndim != 5 or raw_input.shape[1] != 20 or raw_input.shape[-1] != 3:
        raise ValueError("input_array must have shape (N, 20, H, W, 3)")
    (model, device, dtype,
     lower_multiplier, upper_multiplier) = _get_model()
    mean_input = torch.tensor(_MEAN_INPUT, device=device, dtype=dtype)
    std_input = torch.tensor(_STD_INPUT, device=device, dtype=dtype)
    mean_target = torch.tensor(_MEAN_TARGET, device=device, dtype=dtype)
    std_target = torch.tensor(_STD_TARGET, device=device, dtype=dtype)

    predictions = []
    lower_bounds = []
    upper_bounds = []
    with torch.inference_mode():
        for start in range(0, raw_input.shape[0], _BATCH_SIZE):
            raw = torch.from_numpy(np.ascontiguousarray(
                raw_input[start:start + _BATCH_SIZE])).to(
                    device=device, dtype=dtype)
            normalized = (raw - mean_input) / std_input
            pred_norm, lower_norm, upper_norm = model.forward_distribution(normalized)
            prediction = pred_norm * std_target + mean_target
            prediction[..., 2] = 0.0
            lower_width = torch.zeros_like(prediction)
            upper_width = torch.zeros_like(prediction)
            lower_width[..., :2] = (
                lower_norm * std_target[:2] * lower_multiplier)
            upper_width[..., :2] = (
                upper_norm * std_target[:2] * upper_multiplier)
            predictions.append(prediction.float().cpu().numpy())
            lower_bounds.append((prediction - lower_width).float().cpu().numpy())
            upper_bounds.append((prediction + upper_width).float().cpu().numpy())

    prediction = np.concatenate(predictions).astype(np.float32, copy=False)
    lower = np.concatenate(lower_bounds).astype(np.float32, copy=False)
    upper = np.concatenate(upper_bounds).astype(np.float32, copy=False)
    return {"prediction": prediction, "lower": lower, "upper": upper}
