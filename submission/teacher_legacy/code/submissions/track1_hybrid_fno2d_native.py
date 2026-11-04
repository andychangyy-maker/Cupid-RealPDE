"""Track 1 v9 HybridFNO2d with its native heteroscedastic intervals."""

from __future__ import annotations

import os
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from hybrid_fno2d import HybridFNO2d, HybridFNO2dConfig  # noqa: E402


_CHECKPOINT = os.path.join(_HERE, "hybrid_fno2d.pth")
_MEAN_INPUT = (0.154960855842, -0.000513992854, 0.0)
_MEAN_TARGET = (0.154962569475, -0.000517793698, 0.0)
_STD_INPUT = (0.096805654466, 0.015960684046, 1.0)
_STD_TARGET = (0.096810407937, 0.015963643789, 1.0)
_LOWER_MULTIPLIER = (1.0, 1.0)
_UPPER_MULTIPLIER = (1.0, 1.0)
_BATCH_SIZE = 32
_state = {"model": None, "device": None}


def _get_model():
    if _state["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(_CHECKPOINT, map_location="cpu", weights_only=False)
        config = HybridFNO2dConfig(**payload["model_config"])
        if not config.predict_uncertainty:
            raise ValueError("packaged checkpoint has no native interval decoder")
        model = HybridFNO2d(config)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model = model.to(device).eval()
        model.set_normalizer_statistics(
            torch.tensor(_MEAN_INPUT, device=device),
            torch.tensor(_MEAN_TARGET, device=device),
            torch.tensor(_STD_INPUT, device=device),
            torch.tensor(_STD_TARGET, device=device),
        )
        _state.update({"model": model, "device": device})
    return _state["model"], _state["device"]


def predict(input_array, metadata=None):
    """Return point forecasts and finite per-element lower/upper bounds."""
    raw_input = np.asarray(input_array, dtype=np.float32)
    if raw_input.ndim != 5 or raw_input.shape[1] != 20 or raw_input.shape[-1] != 3:
        raise ValueError("input_array must have shape (N, 20, H, W, 3)")
    model, device = _get_model()
    mean_input = torch.tensor(_MEAN_INPUT, device=device)
    std_input = torch.tensor(_STD_INPUT, device=device)
    mean_target = torch.tensor(_MEAN_TARGET, device=device)
    std_target = torch.tensor(_STD_TARGET, device=device)
    lower_multiplier = torch.tensor(
        _LOWER_MULTIPLIER, device=device).view(1, 1, 1, 1, 2)
    upper_multiplier = torch.tensor(
        _UPPER_MULTIPLIER, device=device).view(1, 1, 1, 1, 2)

    predictions = []
    lower_bounds = []
    upper_bounds = []
    with torch.inference_mode():
        for start in range(0, raw_input.shape[0], _BATCH_SIZE):
            raw = torch.from_numpy(np.ascontiguousarray(
                raw_input[start:start + _BATCH_SIZE])).to(device)
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
