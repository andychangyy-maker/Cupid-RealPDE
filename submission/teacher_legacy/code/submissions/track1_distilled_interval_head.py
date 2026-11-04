"""Track 1 single C128 point model with a fast distilled interval head."""

from __future__ import annotations

import os
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from models.distilled_interval_head import (  # noqa: E402
    DistilledIntervalHead,
    DistilledIntervalHeadConfig,
)
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)
from models.unet2d import UNet2d, UNet2dConfig  # noqa: E402


_BASE_CHECKPOINT = os.path.join(_HERE, "base_model.pth")
_INTERVAL_CHECKPOINT = os.path.join(_HERE, "interval_head.pth")
_BATCH_SIZE = 32
_VALIDATE_OUTPUTS = False
_FALLBACK_STATS = {
    "mean_input": (0.154960855842, -0.000513992854, 0.0),
    "mean_target": (0.154962569475, -0.000517793698, 0.0),
    "std_input": (0.096805654466, 0.015960684046, 1.0),
    "std_target": (0.096810407937, 0.015963643789, 1.0),
}
_state = {
    "base": None,
    "head": None,
    "device": None,
    "dtype": None,
}


def _restore_state_dict(payload, dtype):
    stored_state = payload.get("quantized_state_dict")
    if stored_state is None:
        return payload["model_state_dict"]
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


def _load_state():
    if _state["base"] is not None:
        return _state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_payload = torch.load(
        _BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    base_config = dict(base_payload["model_config"])
    architecture = base_config.pop("architecture", "ensemble_unet")
    if architecture == "ensemble_unet":
        base = EnsembleUNet2d(EnsembleUNet2dConfig(**base_config))
    elif architecture == "unet":
        base = UNet2d(UNet2dConfig(**base_config))
    else:
        raise ValueError(f"unsupported packaged base: {architecture}")
    base = base.to(device=device, dtype=dtype)
    base_state = _restore_state_dict(base_payload, dtype)
    base.load_state_dict(base_state, strict=True)
    del base_state
    base.eval()

    head_payload = torch.load(
        _INTERVAL_CHECKPOINT, map_location="cpu", weights_only=False)
    head_config = dict(head_payload["model_config"])
    head_architecture = head_config.pop(
        "architecture", "distilled_interval_head")
    if head_architecture != "distilled_interval_head":
        raise ValueError("packaged head is not a distilled interval head")
    if "feature_channels" in head_config:
        head_config["feature_channels"] = tuple(
            head_config["feature_channels"])
    if "reference_total_widths" in head_config:
        head_config["reference_total_widths"] = tuple(
            head_config["reference_total_widths"])
    head = DistilledIntervalHead(
        DistilledIntervalHeadConfig(**head_config)).to(
            device=device, dtype=dtype)
    head.load_state_dict(
        head_payload["model_state_dict"], strict=True)
    head.eval()
    interval_scale = torch.as_tensor(
        head_payload.get("interval_scale_uv", (1.0, 1.0)),
        device=device, dtype=dtype)
    if (interval_scale.shape != (2,)
            or not torch.isfinite(interval_scale).all()
            or torch.any(interval_scale <= 0)):
        raise ValueError("packaged interval scales must be positive u/v values")
    interval_scale = interval_scale.view(1, 1, 1, 1, 2)

    stored_stats = base_payload.get("normalizer", {})
    tensors = {
        key: torch.as_tensor(
            stored_stats.get(key, fallback),
            device=device, dtype=dtype)
        for key, fallback in _FALLBACK_STATS.items()
    }
    base.set_normalizer_statistics(
        tensors["mean_input"], tensors["mean_target"],
        tensors["std_input"], tensors["std_target"])
    _state.update({
        "base": base,
        "head": head,
        "interval_scale": interval_scale,
        "device": device,
        "dtype": dtype,
        **tensors,
    })
    return _state


def predict(input_array, metadata=None):
    """Return point predictions and direct per-element interval bounds."""
    del metadata
    raw_input = np.asarray(input_array, dtype=np.float32)
    if (raw_input.ndim != 5 or raw_input.shape[1] != 20
            or raw_input.shape[-1] != 3):
        raise ValueError("input_array must have shape (N, 20, H, W, 3)")

    state = _load_state()
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
            prediction_norm, features = (
                state["base"].forward_with_features(normalized))
            lower_extent, upper_extent = state["head"](
                normalized, prediction_norm, features,
                state["base"].target_std)
            lower_extent = lower_extent * state["interval_scale"]
            upper_extent = upper_extent * state["interval_scale"]
            prediction = (
                prediction_norm * state["std_target"]
                + state["mean_target"])
            prediction[..., 2] = 0.0
            lower = prediction.clone()
            upper = prediction.clone()
            lower[..., :2] = (
                prediction[..., :2]
                - lower_extent * state["std_target"][:2])
            upper[..., :2] = (
                prediction[..., :2]
                + upper_extent * state["std_target"][:2])
            if (_VALIDATE_OUTPUTS
                    and (not torch.isfinite(prediction).all()
                    or not torch.isfinite(lower).all()
                    or not torch.isfinite(upper).all()
                    or torch.any(lower > upper))):
                raise FloatingPointError(
                    "distilled interval head produced invalid output")
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
