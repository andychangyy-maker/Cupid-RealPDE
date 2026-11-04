"""Track 1 single-backbone forecast with a distilled flow-context corrector."""

from __future__ import annotations

import os
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from models.distilled_flow_corrector import (  # noqa: E402
    DistilledFlowCorrector,
    DistilledFlowCorrectorConfig,
)
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


_BASE_CHECKPOINT = os.path.join(_HERE, "base_model.pth")
_CORRECTOR_CHECKPOINT = os.path.join(_HERE, "corrector.pth")
_BATCH_SIZE = 32
_FALLBACK_STATS = {
    "mean_input": (0.154960855842, -0.000513992854, 0.0),
    "mean_target": (0.154962569475, -0.000517793698, 0.0),
    "std_input": (0.096805654466, 0.015960684046, 1.0),
    "std_target": (0.096810407937, 0.015963643789, 1.0),
}
_state = {
    "base": None,
    "corrector": None,
    "device": None,
    "dtype": None,
}


def _load_state():
    if _state["base"] is not None:
        return _state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_payload = torch.load(
        _BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    base_config = dict(base_payload["model_config"])
    architecture = base_config.pop("architecture", "ensemble_unet")
    if architecture != "ensemble_unet":
        raise ValueError("packaged base is not an ensemble_unet")
    base = EnsembleUNet2d(EnsembleUNet2dConfig(**base_config)).to(
        device=device, dtype=dtype)
    base.load_state_dict(base_payload["model_state_dict"], strict=True)
    base.eval()

    corrector_payload = torch.load(
        _CORRECTOR_CHECKPOINT, map_location="cpu", weights_only=False)
    corrector_config = dict(corrector_payload["model_config"])
    corrector_architecture = corrector_config.pop(
        "architecture", "distilled_flow_corrector")
    if corrector_architecture != "distilled_flow_corrector":
        raise ValueError("packaged head is not a distilled corrector")
    corrector = DistilledFlowCorrector(
        DistilledFlowCorrectorConfig(**corrector_config)).to(
            device=device, dtype=dtype)
    corrector.load_state_dict(
        corrector_payload["model_state_dict"], strict=True)
    corrector.eval()

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
        "corrector": corrector,
        "device": device,
        "dtype": dtype,
        **tensors,
    })
    return _state


def predict(input_array, metadata=None):
    """Return one-backbone distilled point forecasts and learned intervals."""
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
            prediction_norm, lower_extent, upper_extent, features = (
                state["base"].forward_distribution_with_features(
                    normalized))
            prediction_norm, lower_extent, upper_extent = (
                state["corrector"].correct_distribution(
                    normalized, prediction_norm,
                    lower_extent, upper_extent, features[-1]))
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
            if (not torch.isfinite(prediction).all()
                    or not torch.isfinite(lower).all()
                    or not torch.isfinite(upper).all()
                    or torch.any(lower > upper)):
                raise FloatingPointError(
                    "distilled corrector produced invalid output")
            predictions.append(prediction.float().cpu().numpy())
            lower_bounds.append(lower.float().cpu().numpy())
            upper_bounds.append(upper.float().cpu().numpy())

    return {
        "prediction": np.concatenate(predictions).astype(
            np.float32, copy=False),
        "lower": np.concatenate(lower_bounds).astype(np.float32, copy=False),
        "upper": np.concatenate(upper_bounds).astype(np.float32, copy=False),
    }
