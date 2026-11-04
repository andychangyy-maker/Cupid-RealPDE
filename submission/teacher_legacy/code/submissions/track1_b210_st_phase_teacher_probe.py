"""Track 1 B210 UQ plus a fixed isoenergetic causal-ST phase teacher.

The B210 package remains responsible for uncertainty bounds, temporal means,
and temporal fluctuation RMS.  The equal-weight seed-soup ST model contributes
only the direction of each u/v temporal fluctuation vector.  No prediction or
metadata is retained between calls.
"""

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
from models.spatiotemporal_b210 import (  # noqa: E402
    SpatiotemporalB210,
    SpatiotemporalB210Config,
)


_BASE_CHECKPOINT = os.path.join(_HERE, "ensemble_unet2d.pth")
_ST_CHECKPOINT = os.path.join(_HERE, "spatiotemporal_b210.pth")
_BATCH_SIZE = 16
_FIXED_BLEND_WEIGHT = 0.5
_state = {
    "base_model": None,
    "st_model": None,
    "device": None,
    "dtype": None,
    "base_mean_input": None,
    "base_mean_target": None,
    "base_std_input": None,
    "base_std_target": None,
    "st_mean_input": None,
    "st_mean_target": None,
    "st_std_input": None,
    "st_std_target": None,
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


def _stats(payload, prefix, device, dtype):
    stored = payload["normalizer"]
    return {
        f"{prefix}_mean_input": torch.as_tensor(
            stored["mean_input"], device=device, dtype=dtype
        ),
        f"{prefix}_mean_target": torch.as_tensor(
            stored["mean_target"], device=device, dtype=dtype
        ),
        f"{prefix}_std_input": torch.as_tensor(
            stored["std_input"], device=device, dtype=dtype
        ),
        f"{prefix}_std_target": torch.as_tensor(
            stored["std_target"], device=device, dtype=dtype
        ),
    }


def _get_models():
    if _state["base_model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_payload = torch.load(
            _BASE_CHECKPOINT, map_location="cpu", weights_only=False
        )
        st_payload = torch.load(
            _ST_CHECKPOINT, map_location="cpu", weights_only=False
        )
        dtype = (
            torch.float16
            if device.type == "cuda"
            and base_payload.get("inference_dtype") == "float16"
            and st_payload.get("inference_dtype") == "float16"
            else torch.float32
        )

        base_config_values = dict(base_payload["model_config"])
        architecture = base_config_values.pop("architecture", "ensemble_unet")
        if architecture != "ensemble_unet":
            raise ValueError("base checkpoint is not ensemble_unet")
        base_model = EnsembleUNet2d(
            EnsembleUNet2dConfig(**base_config_values)
        ).to(device=device, dtype=dtype)
        base_model.load_state_dict(
            _restore_state_dict(base_payload, dtype), strict=True
        )
        base_model.eval()

        st_config_values = dict(st_payload["model_config"])
        st_architecture = st_config_values.pop("architecture", None)
        interaction_mode = st_config_values.pop(
            "interaction_mode", "causal_temporal_candidate"
        )
        st_config_values.pop("flattened_history_bypass", None)
        if st_architecture != "spatiotemporal_b210":
            raise ValueError("teacher checkpoint is not spatiotemporal_b210")
        if interaction_mode != "causal_temporal_candidate":
            raise ValueError("teacher checkpoint is not causal temporal")
        st_model = SpatiotemporalB210(
            SpatiotemporalB210Config(**st_config_values),
            interaction_mode=interaction_mode,
        ).to(device=device, dtype=dtype)
        st_model.load_state_dict(_restore_state_dict(st_payload, dtype), strict=True)
        st_model.eval()

        _state.update(
            {
                "base_model": base_model,
                "st_model": st_model,
                "device": device,
                "dtype": dtype,
                **_stats(base_payload, "base", device, dtype),
                **_stats(st_payload, "st", device, dtype),
            }
        )
    return _state


def restore_temporal_mean_rms(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """Keep candidate temporal direction with reference mean and RMS.

    Tensors are [B,T,H,W,C].  Statistics are independent for every
    sample/pixel/channel and reduce only the twenty-horizon axis.
    """
    if candidate.shape != reference.shape or candidate.ndim != 5:
        raise ValueError("candidate/reference must have equal [B,T,H,W,C] shapes")
    if not candidate.is_floating_point() or not reference.is_floating_point():
        raise TypeError("temporal projection requires floating tensors")
    candidate = candidate.float()
    reference = reference.float()
    reference_mean = reference.mean(dim=1, keepdim=True)
    reference_fluctuation = reference - reference_mean
    candidate_fluctuation = candidate - candidate.mean(dim=1, keepdim=True)
    reference_rms = reference_fluctuation.square().mean(
        dim=1, keepdim=True
    ).sqrt()
    candidate_rms = candidate_fluctuation.square().mean(
        dim=1, keepdim=True
    ).sqrt()
    valid = candidate_rms > epsilon
    projected = reference_mean + candidate_fluctuation * (
        reference_rms / candidate_rms.clamp_min(epsilon)
    )
    return torch.where(valid, projected, reference)


def _fuse_point(base_prediction: torch.Tensor, st_prediction: torch.Tensor):
    if base_prediction.shape != st_prediction.shape:
        raise ValueError("B210 and ST predictions have different shapes")
    base_uv = base_prediction[..., :2].float()
    st_uv = st_prediction[..., :2].float()
    mixed_uv = (base_uv + st_uv).mul(_FIXED_BLEND_WEIGHT)
    fused = base_prediction.float().clone()
    fused[..., :2] = restore_temporal_mean_rms(mixed_uv, base_uv)
    fused[..., 2] = 0.0
    return fused


def predict(input_array, metadata=None):
    """Return fused float32 points and bitwise-passthrough B210 bounds."""
    del metadata
    raw_input = np.asarray(input_array, dtype=np.float32)
    if (
        raw_input.ndim != 5
        or raw_input.shape[1] != 20
        or raw_input.shape[-1] != 3
    ):
        raise ValueError("input_array must have shape (N, 20, H, W, 3)")

    state = _get_models()
    predictions = []
    lower_bounds = []
    upper_bounds = []
    with torch.inference_mode():
        for start in range(0, raw_input.shape[0], _BATCH_SIZE):
            raw = torch.from_numpy(
                np.ascontiguousarray(raw_input[start : start + _BATCH_SIZE])
            ).to(device=state["device"], dtype=state["dtype"])

            base_normalized = (
                (raw - state["base_mean_input"]) / state["base_std_input"]
            )
            base_norm, lower_extent_norm, upper_extent_norm = state[
                "base_model"
            ].forward_distribution(base_normalized)
            base_prediction = (
                base_norm * state["base_std_target"] + state["base_mean_target"]
            )
            base_prediction[..., 2] = 0.0
            # These operations deliberately match the source B210 wrapper.
            lower = base_prediction.clone()
            upper = base_prediction.clone()
            lower[..., :2] = (
                base_prediction[..., :2]
                - lower_extent_norm * state["base_std_target"][:2]
            )
            upper[..., :2] = (
                base_prediction[..., :2]
                + upper_extent_norm * state["base_std_target"][:2]
            )

            st_normalized = (
                (raw - state["st_mean_input"]) / state["st_std_input"]
            )
            st_norm = state["st_model"](st_normalized)
            st_prediction = (
                st_norm * state["st_std_target"] + state["st_mean_target"]
            )
            st_prediction[..., 2] = 0.0
            prediction = _fuse_point(base_prediction, st_prediction)

            if (
                not torch.isfinite(prediction).all()
                or not torch.isfinite(lower).all()
                or not torch.isfinite(upper).all()
                or torch.any(lower > upper)
            ):
                raise FloatingPointError("phase-teacher output is invalid")
            predictions.append(prediction.cpu().numpy())
            # Never recenter or rescale these B210-native bounds.
            lower_bounds.append(lower.float().cpu().numpy())
            upper_bounds.append(upper.float().cpu().numpy())

    return {
        "prediction": np.concatenate(predictions).astype(np.float32, copy=False),
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
