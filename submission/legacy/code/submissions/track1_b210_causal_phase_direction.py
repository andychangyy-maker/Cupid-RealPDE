"""Track 1 B210 UQ with a compact causal phase-direction adapter.

The frozen B210 keeps its temporal mean/RMS, pressure and absolute interval
endpoints.  The adapter changes only the future u/v temporal direction.  A
normalized point-delta compensation would reconstruct the same endpoints;
the runtime retains the already-computed B210 endpoints directly to avoid an
otherwise unnecessary FP16 subtract/add round trip.
"""

from __future__ import annotations

from dataclasses import fields
import os
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
)
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


_BASE_CHECKPOINT = os.path.join(_HERE, "ensemble_unet2d.pth")
_ADAPTER_CHECKPOINT = os.path.join(_HERE, "causal_phase_direction.pth")
_BATCH_SIZE = 16
_state = {
    "base_model": None,
    "adapter": None,
    "device": None,
    "dtype": None,
    "mean_input": None,
    "mean_target": None,
    "std_input": None,
    "std_target": None,
}


def _restore_state_dict(payload, dtype):
    """Restore the same symmetric per-dim0 INT8 format as the base ZIP."""
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


def _adapter_config(payload):
    config = payload["model_config"]
    if config.get("architecture") != "causal_phase_direction_adapter":
        raise ValueError("adapter checkpoint has the wrong architecture")
    names = {field.name for field in fields(CausalPhaseDirectionConfig)}
    values = {name: config[name] for name in names}
    if values["mixer_mode"] != "causal_temporal":
        raise ValueError("submission requires the causal temporal adapter")
    return CausalPhaseDirectionConfig(**values)


def _get_models():
    if _state["base_model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_payload = torch.load(
            _BASE_CHECKPOINT, map_location="cpu", weights_only=False)
        adapter_payload = torch.load(
            _ADAPTER_CHECKPOINT, map_location="cpu", weights_only=False)
        if adapter_payload.get("deployment_state") != "ema_adapter_state_u4000":
            raise ValueError("adapter package is not the sealed EMA-u4000 state")
        dtype = (
            torch.float16
            if device.type == "cuda"
            and base_payload.get("inference_dtype") == "float16"
            and adapter_payload.get("inference_dtype") == "float16"
            else torch.float32)

        base_values = dict(base_payload["model_config"])
        architecture = base_values.pop("architecture", "ensemble_unet")
        if architecture != "ensemble_unet":
            raise ValueError("base checkpoint is not ensemble_unet")
        base_model = EnsembleUNet2d(
            EnsembleUNet2dConfig(**base_values)).to(
                device=device, dtype=dtype)
        base_model.load_state_dict(
            _restore_state_dict(base_payload, dtype), strict=True)
        base_model.eval()

        adapter = CausalPhaseDirectionAdapter(
            _adapter_config(adapter_payload)).to(device=device, dtype=dtype)
        adapter.load_state_dict(
            _restore_state_dict(adapter_payload, dtype), strict=True)
        adapter.eval()

        stored_stats = base_payload["normalizer"]
        statistics = {
            name: torch.as_tensor(
                stored_stats[name], device=device, dtype=dtype)
            for name in (
                "mean_input", "mean_target", "std_input", "std_target")
        }
        _state.update({
            "base_model": base_model,
            "adapter": adapter,
            "device": device,
            "dtype": dtype,
            **statistics,
        })
    return _state


def _phase_distribution(
        adapter, normalized_history, base_prediction,
        base_lower_extent, base_upper_extent):
    """Change point direction and compensate normalized B210 extents."""
    prediction = adapter(normalized_history, base_prediction.detach())
    point_delta = prediction[..., :2] - base_prediction[..., :2]
    lower_extent = base_lower_extent + point_delta
    upper_extent = base_upper_extent - point_delta
    return prediction, lower_extent, upper_extent


def predict(input_array, metadata=None):
    """Return float32 phase points and B210-inherited lower/upper bounds."""
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
            normalized = (
                (raw - state["mean_input"]) / state["std_input"])
            base_prediction, base_lower, base_upper = state[
                "base_model"].forward_distribution(normalized)
            # Preserve the exact operation order of the source B210 wrapper
            # before changing its point.  Returning these values directly is
            # algebraically identical to normalized extent compensation, but
            # does not introduce a second FP16 round trip.
            base_prediction_physical = (
                base_prediction * state["std_target"]
                + state["mean_target"])
            base_prediction_physical[..., 2] = 0.0
            lower = base_prediction_physical.clone()
            upper = base_prediction_physical.clone()
            lower[..., :2] = (
                base_prediction_physical[..., :2]
                - base_lower * state["std_target"][:2])
            upper[..., :2] = (
                base_prediction_physical[..., :2]
                + base_upper * state["std_target"][:2])

            prediction_norm = state["adapter"](
                normalized, base_prediction.detach())
            prediction = (
                prediction_norm * state["std_target"]
                + state["mean_target"])
            prediction[..., 2] = 0.0
            if (not torch.isfinite(prediction).all()
                    or not torch.isfinite(lower).all()
                    or not torch.isfinite(upper).all()
                    or torch.any(lower > upper)):
                raise FloatingPointError("causal phase output is invalid")
            bundle = torch.stack((prediction, lower, upper)).float().cpu().numpy()
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
    _get_models()
    dummy = np.zeros((_BATCH_SIZE, 20, 32, 64, 3), dtype=np.float32)
    predict(dummy, metadata={})
    torch.cuda.synchronize()


_prewarm_cuda_runtime()
del _prewarm_cuda_runtime
