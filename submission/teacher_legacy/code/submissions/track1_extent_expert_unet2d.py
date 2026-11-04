"""Track 1 point/interval-decoupled three-expert UNet ensemble."""

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
_state = {"models": None}


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
    model.load_state_dict(_restore_state_dict(payload, dtype), strict=True)
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


def _get_state():
    if _state["models"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        with open(_BLEND_CONFIG, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        checkpoints = [
            os.path.join(_HERE, name) for name in config["checkpoints"]]
        point_weight_b = float(config["point_weight_b"])
        mean_expert_weight = float(config["mean_expert_weight"])
        extent_weights = tuple(float(value) for value in config["extent_weights"])
        extent_scale = float(config["extent_scale"])
        extent_power = float(config.get("extent_power", 1.0))
        if (len(checkpoints) != 3 or len(extent_weights) != 3
                or any(value < 0.0 for value in extent_weights)
                or abs(sum(extent_weights) - 1.0) > 1e-5
                or not 0.0 <= point_weight_b <= 1.0
                or not 0.0 <= mean_expert_weight <= 1.0
                or extent_scale <= 0.0 or extent_power < 1.0):
            raise ValueError("invalid packaged extent-expert configuration")
        _state.update({
            "models": [
                _load_one(checkpoint, device, dtype)
                for checkpoint in checkpoints],
            "device": device,
            "dtype": dtype,
            "point_weight_b": point_weight_b,
            "mean_expert_weight": mean_expert_weight,
            "extent_weights": extent_weights,
            "extent_scale": extent_scale,
            "extent_power": extent_power,
        })
    return _state


def _normalize(model_state, raw):
    return (raw - model_state["mean_input"]) / model_state["std_input"]


def _physical_distribution(model_state, normalized):
    prediction_norm, lower_norm, upper_norm = (
        model_state["model"].forward_distribution(normalized))
    prediction = (
        prediction_norm * model_state["std_target"]
        + model_state["mean_target"])
    prediction[..., 2] = 0.0
    lower_extent = lower_norm * model_state["std_target"][:2]
    upper_extent = upper_norm * model_state["std_target"][:2]
    return prediction, lower_extent, upper_extent


def predict(input_array, metadata=None):
    """Keep the point ensemble fixed and blend only learned interval extents."""
    del metadata
    raw_input = np.asarray(input_array, dtype=np.float32)
    if (raw_input.ndim != 5 or raw_input.shape[1] != 20
            or raw_input.shape[-1] != 3):
        raise ValueError("input_array must have shape (N, 20, H, W, 3)")

    state = _get_state()
    predictions = []
    lower_bounds = []
    upper_bounds = []
    with torch.inference_mode():
        for start in range(0, raw_input.shape[0], _BATCH_SIZE):
            raw = torch.from_numpy(np.ascontiguousarray(
                raw_input[start:start + _BATCH_SIZE])).to(
                    device=state["device"], dtype=state["dtype"])
            distributions = [
                _physical_distribution(model, _normalize(model, raw))
                for model in state["models"]]
            point_a, point_b, point_c = (
                distribution[0] for distribution in distributions)
            base = point_a.lerp(point_b, state["point_weight_b"])
            correction = state["mean_expert_weight"] * (
                point_c.mean(dim=1, keepdim=True)
                - base.mean(dim=1, keepdim=True))
            prediction = base + correction
            if abs(state["extent_power"] - 1.0) < 1e-8:
                lower_extent = sum(
                    weight * distribution[1]
                    for weight, distribution in zip(
                        state["extent_weights"], distributions))
                upper_extent = sum(
                    weight * distribution[2]
                    for weight, distribution in zip(
                        state["extent_weights"], distributions))
            else:
                reference = prediction.new_tensor(0.01)
                power = state["extent_power"]
                lower_extent = reference * sum(
                    weight * torch.pow(distribution[1] / reference, power)
                    for weight, distribution in zip(
                        state["extent_weights"], distributions)
                ).pow(1.0 / power)
                upper_extent = reference * sum(
                    weight * torch.pow(distribution[2] / reference, power)
                    for weight, distribution in zip(
                        state["extent_weights"], distributions)
                ).pow(1.0 / power)
            lower_extent = state["extent_scale"] * lower_extent
            upper_extent = state["extent_scale"] * upper_extent
            lower = prediction.clone()
            upper = prediction.clone()
            lower[..., :2] -= lower_extent
            upper[..., :2] += upper_extent
            if (not torch.isfinite(prediction).all()
                    or not torch.isfinite(lower).all()
                    or not torch.isfinite(upper).all()
                    or torch.any(lower > upper)):
                raise FloatingPointError(
                    "extent-expert ensemble produced invalid bounds")
            predictions.append(prediction.float().cpu().numpy())
            lower_bounds.append(lower.float().cpu().numpy())
            upper_bounds.append(upper.float().cpu().numpy())

    return {
        "prediction": np.concatenate(predictions).astype(
            np.float32, copy=False),
        "lower": np.concatenate(lower_bounds).astype(np.float32, copy=False),
        "upper": np.concatenate(upper_bounds).astype(np.float32, copy=False),
    }
