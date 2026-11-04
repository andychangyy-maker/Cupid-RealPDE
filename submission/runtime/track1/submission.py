"""Track 1 v9 shared-backbone ensemble with residual-quantile bounds."""

from __future__ import annotations

import os
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402
from scale_mean import correct  # noqa: E402
from shape_model import PointModel, reconstruct, mix_distribution_shape
from fast_runtime import prepare, compile_bundle, _bundle
from pinned_io import CapturedIO

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
    "shape_model": None,
    "graphs": {},
    "pinned_graphs": {},
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
        shape_model = PointModel().to(device=device, dtype=dtype).eval()
        shape_payload = torch.load(
            os.path.join(_HERE, "shape_model.pth"), map_location="cpu",
            weights_only=False)
        shape_model.load_state_dict(shape_payload["state_dict"], strict=True)
        prepare(model, shape_model, _HERE, device, dtype)
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
            "shape_model": shape_model,
            "quantile_weights": torch.tensor(
                [.1, .125, .175, .2, .175, .125, .1],
                device=device, dtype=torch.float32),
            **tensors,
        })
    return _state


def _graph_batch(raw32, state):
    key = tuple(raw32.shape)
    if state["device"].type != "cuda" or key[0] not in (1, 16):
        return _bundle(raw32, state)
    operation = state["bundle_operation"]
    if key not in state["graphs"]:
        static = torch.zeros_like(raw32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = operation(static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = operation(static)
        state["graphs"][key] = (static, graph, output)
    static, graph, output = state["graphs"][key]
    static.copy_(raw32)
    graph.replay()
    return output


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
            value = raw_input[start:start + _BATCH_SIZE]
            if (state["device"].type == "cuda" and state.get("bundle_compiled", False)
                    and len(value) in (1, 16) and value.shape[1:] == (20, 32, 64, 3)):
                if len(value) not in state["pinned_graphs"]:
                    state["pinned_graphs"][len(value)] = CapturedIO(
                        state["bundle_operation"], len(value), state["device"])
                bundle = state["pinned_graphs"][len(value)](value)
            else:
                raw32 = torch.from_numpy(np.ascontiguousarray(value)).to(device=state["device"])
                bundle = _graph_batch(raw32, state).cpu().numpy()
            if (_VALIDATE_OUTPUTS
                    and (not np.isfinite(bundle).all()
                    or np.any(bundle[1] > bundle[2]))):
                raise FloatingPointError("ensemble produced invalid prediction bounds")
            predictions.append(bundle[0])
            lower_bounds.append(bundle[1])
            upper_bounds.append(bundle[2])

    if len(predictions) == 1:
        return {"prediction": predictions[0], "lower": lower_bounds[0], "upper": upper_bounds[0]}
    return {
        "prediction": np.concatenate(predictions).astype(
            np.float32, copy=False),
        "lower": np.concatenate(lower_bounds).astype(np.float32, copy=False),
        "upper": np.concatenate(upper_bounds).astype(np.float32, copy=False),
    }


def _prewarm_cuda_runtime():
    if not torch.cuda.is_available():
        return
    state = _get_model()
    state["bundle_operation"] = compile_bundle(_bundle, state, _HERE)
    for batch_size in (1, _BATCH_SIZE):
        dummy = np.zeros((batch_size, 20, 32, 64, 3), dtype=np.float32)
        predict(dummy, metadata=None)
    torch.cuda.synchronize()
    # Captured 1/16 batches retain their chosen kernels; do not autotune unseen tails.
    torch.backends.cudnn.benchmark = False


_prewarm_cuda_runtime()
del _prewarm_cuda_runtime
