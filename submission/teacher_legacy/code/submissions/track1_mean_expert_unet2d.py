"""Track 1 A/B distribution blend with a TKE-invariant mean expert."""

from __future__ import annotations

import json
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


_BLEND_CONFIG = os.path.join(_HERE, "blend_config.json")
_BATCH_SIZE = 16
_FALLBACK_STATS = {
    "mean_input": (0.154960855842, -0.000513992854, 0.0),
    "mean_target": (0.154962569475, -0.000517793698, 0.0),
    "std_input": (0.096805654466, 0.015960684046, 1.0),
    "std_target": (0.096810407937, 0.015963643789, 1.0),
}
_state = {
    "models": None, "device": None, "dtype": None,
    "weight_b": None, "mean_expert_weight": None,
}


def _restore_state_dict(payload, dtype):
    restored = {}
    for key, stored in payload["quantized_state_dict"].items():
        if isinstance(stored, dict) and "quantized" in stored:
            value = stored["quantized"].float() * stored["scale"].float()
            restored[key] = value.to(dtype=dtype)
        elif torch.is_floating_point(stored):
            restored[key] = stored.to(dtype=dtype)
        else:
            restored[key] = stored
    return restored


def _load_one(checkpoint, device, dtype):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config_dict = dict(payload["model_config"])
    architecture = config_dict.pop("architecture", "ensemble_unet")
    if architecture != "ensemble_unet":
        raise ValueError(
            f"packaged checkpoint is {architecture}, not ensemble_unet")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**config_dict)).to(
        device=device, dtype=dtype)
    restored = _restore_state_dict(payload, dtype)
    model.load_state_dict(restored, strict=True)
    del restored
    model.eval()
    stored_stats = payload.get("normalizer", {})
    tensors = {
        key: torch.as_tensor(
            stored_stats.get(key, fallback), device=device, dtype=dtype)
        for key, fallback in _FALLBACK_STATS.items()
    }
    model.set_normalizer_statistics(
        tensors["mean_input"], tensors["mean_target"],
        tensors["std_input"], tensors["std_target"])
    return {"model": model, **tensors}


def _get_models():
    if _state["models"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        with open(_BLEND_CONFIG, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        checkpoints = [os.path.join(_HERE, name) for name in config["checkpoints"]]
        weight_b = float(config["weight_b"])
        mean_expert_weight = float(config["mean_expert_weight"])
        if (len(checkpoints) != 3
                or not 0.0 <= weight_b <= 1.0
                or not 0.0 <= mean_expert_weight <= 1.0):
            raise ValueError("invalid packaged mean-expert configuration")
        models = [
            _load_one(checkpoint, device, dtype)
            for checkpoint in checkpoints
        ]
        _state.update({
            "models": models,
            "device": device,
            "dtype": dtype,
            "weight_b": weight_b,
            "mean_expert_weight": mean_expert_weight,
        })
    return _state


def _normalize(model_state, raw):
    return (raw - model_state["mean_input"]) / model_state["std_input"]


def _physical_point(model_state, normalized):
    prediction_norm = model_state["model"](normalized)
    prediction = (
        prediction_norm * model_state["std_target"]
        + model_state["mean_target"])
    prediction[..., 2] = 0.0
    return prediction


def _physical_distribution(model_state, normalized):
    prediction_norm, lower_extent, upper_extent = (
        model_state["model"].forward_distribution(normalized))
    prediction = (
        prediction_norm * model_state["std_target"]
        + model_state["mean_target"])
    prediction[..., 2] = 0.0
    lower = prediction.clone()
    upper = prediction.clone()
    lower[..., :2] = (
        prediction[..., :2]
        - lower_extent * model_state["std_target"][:2])
    upper[..., :2] = (
        prediction[..., :2]
        + upper_extent * model_state["std_target"][:2])
    return prediction, lower, upper


def predict(input_array, metadata=None):
    """Blend A/B bounds and inject only C's temporal-mean correction."""
    del metadata
    raw_input = np.asarray(input_array, dtype=np.float32)
    if (raw_input.ndim != 5 or raw_input.shape[1] != 20
            or raw_input.shape[-1] != 3):
        raise ValueError("input_array must have shape (N, 20, H, W, 3)")

    state = _get_models()
    predictions = []
    lower_bounds = []
    upper_bounds = []
    with torch.inference_mode():
        for start in range(0, raw_input.shape[0], _BATCH_SIZE):
            raw = torch.from_numpy(np.ascontiguousarray(
                raw_input[start:start + _BATCH_SIZE])).to(
                    device=state["device"], dtype=state["dtype"])
            a = _physical_distribution(
                state["models"][0], _normalize(state["models"][0], raw))
            b = _physical_distribution(
                state["models"][1], _normalize(state["models"][1], raw))
            base = tuple(
                value_a.lerp(value_b, state["weight_b"])
                for value_a, value_b in zip(a, b))
            expert = _physical_point(
                state["models"][2], _normalize(state["models"][2], raw))
            mean_delta = (
                expert.mean(dim=1, keepdim=True)
                - base[0].mean(dim=1, keepdim=True))
            correction = mean_delta.mul(state["mean_expert_weight"])
            prediction, lower, upper = tuple(
                value + correction for value in base)
            if (not torch.isfinite(prediction).all()
                    or not torch.isfinite(lower).all()
                    or not torch.isfinite(upper).all()
                    or torch.any(lower > upper)):
                raise FloatingPointError(
                    "mean-expert ensemble produced invalid prediction bounds")
            predictions.append(prediction.float().cpu().numpy())
            lower_bounds.append(lower.float().cpu().numpy())
            upper_bounds.append(upper.float().cpu().numpy())

    return {
        "prediction": np.concatenate(predictions).astype(
            np.float32, copy=False),
        "lower": np.concatenate(lower_bounds).astype(np.float32, copy=False),
        "upper": np.concatenate(upper_bounds).astype(np.float32, copy=False),
    }
