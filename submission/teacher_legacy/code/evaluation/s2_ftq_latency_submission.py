"""Zero-training Track 1 FTQ submission used only for the S2 latency gate."""

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
from models.fast_temporal_quantile_head import (  # noqa: E402
    FastTemporalQuantileHead,
    FastTemporalQuantileHeadConfig,
    fused_mean_point_heads,
    fused_mean_prediction,
)


_CHECKPOINT = os.path.join(_HERE, "ensemble_unet2d.pth")
_BATCH_SIZE = 32
_FALLBACK_STATS = {
    "mean_input": (0.154960855842, -0.000513992854, 0.0),
    "mean_target": (0.154962569475, -0.000517793698, 0.0),
    "std_input": (0.096805654466, 0.015960684046, 1.0),
    "std_target": (0.096810407937, 0.015963643789, 1.0),
}
_state = {
    "model": None,
    "head": None,
    "fused_heads": None,
    "device": None,
    "dtype": None,
    "mean_input": None,
    "mean_target": None,
    "std_input": None,
    "std_target": None,
}


def _get_model():
    if _state["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(_CHECKPOINT, map_location="cpu", weights_only=False)
        config_dict = dict(payload["model_config"])
        architecture = config_dict.pop("architecture", "ensemble_unet")
        if architecture != "ensemble_unet":
            raise ValueError(
                f"packaged checkpoint is {architecture}, not ensemble_unet")
        dtype_name = payload.get("inference_dtype", "float32")
        dtype = (
            torch.float16
            if device.type == "cuda" and dtype_name == "float16"
            else torch.float32)
        model = EnsembleUNet2d(
            EnsembleUNet2dConfig(**config_dict)).to(
                device=device, dtype=dtype)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()

        stored_stats = payload.get("normalizer", {})
        tensors = {
            key: torch.as_tensor(
                stored_stats.get(key, fallback),
                device=device, dtype=dtype)
            for key, fallback in _FALLBACK_STATS.items()
        }
        model.set_normalizer_statistics(
            tensors["mean_input"], tensors["mean_target"],
            tensors["std_input"], tensors["std_target"])

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        head = FastTemporalQuantileHead(
            FastTemporalQuantileHeadConfig()).to(
                device=device, dtype=dtype).eval()
        fused_heads = fused_mean_point_heads(model)
        _state.update({
            "model": model,
            "head": head,
            "fused_heads": fused_heads,
            "device": device,
            "dtype": dtype,
            **tensors,
        })
    return _state


def predict(input_array, metadata=None):
    """Return the fused H96 mean point path with the member-free FTQ graph."""
    del metadata
    raw_input = np.asarray(input_array, dtype=np.float32)
    if (raw_input.ndim != 5 or raw_input.shape[1] != 20
            or raw_input.shape[-1] != 3):
        raise ValueError("input_array must have shape (N, 20, H, W, 3)")

    state = _get_model()
    bundles = []
    with torch.inference_mode():
        for start in range(0, raw_input.shape[0], _BATCH_SIZE):
            raw = torch.from_numpy(np.ascontiguousarray(
                raw_input[start:start + _BATCH_SIZE])).to(
                    device=state["device"], dtype=state["dtype"])
            normalized = (
                (raw - state["mean_input"]) / state["std_input"])
            features = state["model"]._decode(normalized)
            prediction_norm = fused_mean_prediction(
                state["model"], normalized, features,
                state["fused_heads"])
            _quantiles, lower_extent, upper_extent = state["head"](
                normalized, prediction_norm, features,
                state["model"].target_std)

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
            bundle = torch.stack(
                (prediction, lower, upper)).float().cpu().numpy()
            bundles.append(bundle)

    if len(bundles) == 1:
        bundle = bundles[0]
        return {
            "prediction": bundle[0],
            "lower": bundle[1],
            "upper": bundle[2],
        }
    return {
        "prediction": np.concatenate([
            bundle[0] for bundle in bundles]).astype(
            np.float32, copy=False),
        "lower": np.concatenate([
            bundle[1] for bundle in bundles]).astype(np.float32, copy=False),
        "upper": np.concatenate([
            bundle[2] for bundle in bundles]).astype(np.float32, copy=False),
    }
