"""Track 1 v9 shared-backbone ensemble with residual-quantile bounds."""

from __future__ import annotations

import os
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


_CHECKPOINT = os.path.join(_HERE, "ensemble_unet2d.pth")
_BATCH_SIZE = 16
_VALIDATE_OUTPUTS = False
_FALLBACK_STATS = {
    "mean_input": (0.154960855842, -0.000513992854, 0.0),
    "mean_target": (0.154962569475, -0.000517793698, 0.0),
    "std_input": (0.096805654466, 0.015960684046, 1.0),
    "std_target": (0.096810407937, 0.015963643789, 1.0),
}
_state = {
    "model": None,
    "device": None,
    "dtype": None,
    "mean_input": None,
    "mean_target": None,
    "std_input": None,
    "std_target": None,
}


def _restore_state_dict(payload, dtype):
    stored_state = payload.get("quantized_state_dict")
    if stored_state is None:
        stored_state = payload["model_state_dict"]
    restored = {}
    for key, stored in stored_state.items():
        if isinstance(stored, dict) and "quantized" in stored:
            value = stored["quantized"].float() * stored["scale"].float()
            restored[key] = value.to(dtype=dtype)
        elif torch.is_floating_point(stored):
            restored[key] = stored.to(dtype=dtype)
        else:
            restored[key] = stored
    return restored


def _get_model():
    if _state["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(_CHECKPOINT, map_location="cpu", weights_only=False)
        config_dict = dict(payload["model_config"])
        architecture = config_dict.pop("architecture", "ensemble_unet")
        if architecture != "ensemble_unet":
            raise ValueError(
                f"packaged checkpoint is {architecture}, not ensemble_unet")
        config = EnsembleUNet2dConfig(**config_dict)
        dtype_name = payload.get("inference_dtype", "float32")
        dtype = (
            torch.float16
            if device.type == "cuda" and dtype_name == "float16"
            else torch.float32)
        model = EnsembleUNet2d(config).to(device=device, dtype=dtype)
        restored = _restore_state_dict(payload, dtype)
        model.load_state_dict(restored, strict=True)
        del restored
        model.eval()

        stored_stats = payload.get("normalizer", {})
        tensors = {
            key: torch.as_tensor(
                stored_stats.get(key, fallback),
                device=device, dtype=dtype)
            for key, fallback in _FALLBACK_STATS.items()
        }
        # Forecast-normalizer buffers are part of the strictly loaded state.
        # Recomputing them from FP16 statistics changes rounding, especially
        # persistence_bias, after the strict load.
        _state.update({
            "model": model,
            "device": device,
            "dtype": dtype,
            **tensors,
        })
    return _state


def predict(input_array, metadata=None):
    """Return float32 point forecasts and learned per-element bounds."""
    del metadata
    raw_input = np.asarray(input_array, dtype=np.float32)
    if (raw_input.ndim != 5 or raw_input.shape[1] != 20
            or raw_input.shape[-1] != 3):
        raise ValueError("input_array must have shape (N, 20, H, W, 3)")

    state = _get_model()
    predictions = []
    lower_bounds = []
    upper_bounds = []
    with torch.inference_mode():
        for start in range(0, raw_input.shape[0], _BATCH_SIZE):
            raw = torch.from_numpy(np.ascontiguousarray(
                raw_input[start:start + _BATCH_SIZE])).to(
                    device=state["device"], dtype=state["dtype"])
            normalized = (
                (raw - state["mean_input"]) / state["std_input"])
            pred_norm, lower_extent_norm, upper_extent_norm = (
                state["model"].forward_distribution(normalized))
            prediction = (
                pred_norm * state["std_target"] + state["mean_target"])
            prediction[..., 2] = 0.0

            lower = prediction.clone()
            upper = prediction.clone()
            lower[..., :2] = (
                prediction[..., :2]
                - lower_extent_norm * state["std_target"][:2])
            upper[..., :2] = (
                prediction[..., :2]
                + upper_extent_norm * state["std_target"][:2])
            if (_VALIDATE_OUTPUTS
                    and (not torch.isfinite(prediction).all()
                    or not torch.isfinite(lower).all()
                    or not torch.isfinite(upper).all()
                    or torch.any(lower > upper))):
                raise FloatingPointError(
                    "ensemble produced invalid prediction bounds")
            bundle = torch.stack(
                (prediction, lower, upper)).float().cpu().numpy()
            predictions.append(bundle[0])
            lower_bounds.append(bundle[1])
            upper_bounds.append(bundle[2])

    return {
        "prediction": np.concatenate(predictions).astype(
            np.float32, copy=False),
        "lower": np.concatenate(lower_bounds).astype(np.float32, copy=False),
        "upper": np.concatenate(upper_bounds).astype(np.float32, copy=False),
    }


def _prewarm_cuda_runtime():
    if not torch.cuda.is_available():
        return
    _get_model()
    dummy = np.zeros((_BATCH_SIZE, 20, 32, 64, 3), dtype=np.float32)
    predict(dummy, metadata=None)
    torch.cuda.synchronize()


_prewarm_cuda_runtime()
del _prewarm_cuda_runtime
