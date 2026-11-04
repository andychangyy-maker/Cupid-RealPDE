"""RealPDE Track 2 v6 wrapper for the official 918343 forecaster."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from typing import Any, Dict, Optional, Tuple

from scale_mean import correct

import torch

from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig


_CHECKPOINT_NAME = "ensemble_unet2d.pth"
_POLICY_NAME = "policy.json"
_EXPECTED_SHAPE = (1, 20, 32, 64, 3)
_OFFICIAL_STATS = {
    "mean_input": (0.154960855842, -0.000513992854, 0.0),
    "mean_target": (0.154962569475, -0.000517793698, 0.0),
    "std_input": (0.096805654466, 0.015960684046, 1.0),
    "std_target": (0.096810407937, 0.015963643789, 1.0),
}


def _restore_state_dict(payload: dict, dtype: torch.dtype) -> dict:
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


def _load_policy(submission_dir: str) -> dict:
    path = os.path.join(submission_dir, _POLICY_NAME)
    if not os.path.exists(path):
        return {
            "mode": "channel",
            "alpha": 0.0,
            "beta": 0.8,
            "update_clip": 0.25,
        }
    with open(path, "r", encoding="utf-8") as handle:
        policy = json.load(handle)
    mode = str(policy["mode"])
    alpha = float(policy["alpha"])
    beta = float(policy["beta"])
    update_clip = float(policy["update_clip"])
    if mode != "channel":
        raise ValueError(f"unsupported adaptation mode: {mode}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("policy alpha must lie in [0, 1]")
    if not 0.0 <= beta < 1.0:
        raise ValueError("policy beta must lie in [0, 1)")
    if update_clip <= 0.0:
        raise ValueError("policy update_clip must be positive")
    return {
        "mode": mode,
        "alpha": alpha,
        "beta": beta,
        "update_clip": update_clip,
        "cuda_graph": bool(policy.get("cuda_graph", False)),
    }


def _load_model(submission_dir: str, device: torch.device):
    checkpoint = os.path.join(submission_dir, _CHECKPOINT_NAME)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config_dict = dict(payload["model_config"])
    architecture = config_dict.pop("architecture", "ensemble_unet")
    if architecture != "ensemble_unet":
        raise ValueError(f"checkpoint architecture is {architecture}, not ensemble_unet")

    dtype_name = payload.get("inference_dtype", "float32")
    dtype = (
        torch.float16
        if device.type == "cuda" and dtype_name == "float16"
        else torch.float32
    )
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**config_dict)).to(
        device=device, dtype=dtype
    )
    restored = _restore_state_dict(payload, dtype)
    model.load_state_dict(restored, strict=True)
    from packed_point_heads import install
    install(model)
    if device.type == "cuda":
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                layer.to(memory_format=torch.channels_last)
    model._t2_base_widths = torch.tensor(
        model.config.uncertainty_base_widths, device=device, dtype=dtype)
    model._t2_quantile_levels = torch.tensor(
        model.config.quantile_levels, device=device, dtype=dtype)
    stored_stats = payload.get("normalizer")
    if not isinstance(stored_stats, dict):
        raise ValueError("checkpoint is missing its normalizer statistics")
    missing = sorted(set(_OFFICIAL_STATS) - set(stored_stats))
    if missing:
        raise ValueError(f"checkpoint normalizer is missing: {missing}")
    model_stats = {
        key: torch.as_tensor(stored_stats[key], device=device, dtype=torch.float32)
        for key in _OFFICIAL_STATS
    }
    official_stats = {
        key: torch.as_tensor(value, device=device, dtype=torch.float32)
        for key, value in _OFFICIAL_STATS.items()
    }
    del restored, payload
    model.eval()
    model.requires_grad_(False)
    model._t2_compiled = False
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        eager = model.forward_distribution
        try:
            # Triton builds small C launchers. The pinned evaluator image has no
            # system compiler; use the accompanying offline compiler consistently.
            # The evaluator may mount the submitted files read-only. Only the
            # private temporary copy and compiler caches need write permission.
            model._t2_runtime_dir = tempfile.TemporaryDirectory(prefix="realpde_t2_")
            runtime_root = model._t2_runtime_dir.name
            compiler_root = os.path.join(runtime_root, "tinycc")
            shutil.copytree(os.path.join(os.path.abspath(submission_dir), "vendor", "tinycc"),
                            compiler_root)
            for name in ("cc", os.path.join("bin", "tcc")):
                path = os.path.join(compiler_root, name)
                os.chmod(path, os.stat(path).st_mode | 0o111)
            os.environ["CC"] = os.path.join(compiler_root, "cc")
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(runtime_root, "inductor")
            os.environ["TRITON_CACHE_DIR"] = os.path.join(runtime_root, "triton")
            # The official participant worker is daemonic and cannot spawn a pool.
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
            import torch._inductor.config as inductor_config
            inductor_config.compile_threads = 1
            model.forward_distribution = torch.compile(eager, fullgraph=True, dynamic=False)
            with torch.inference_mode():
                model.forward_distribution(torch.zeros(_EXPECTED_SHAPE, device=device, dtype=dtype))
            torch.cuda.synchronize(device)
        except Exception as error:
            # Compilation is optional; the frozen eager model remains usable.
            model.forward_distribution = eager
            torch.backends.cudnn.benchmark = False
            warnings.warn("Track2 compiler unavailable; using eager forward: "
                          + type(error).__name__ + ": " + str(error))
        else:
            model._t2_compiled = True
            print("[track2] compiled forward ready; temporary compiler and caches", flush=True)
    return model, dtype, model_stats, official_stats


class B210StreamingModel:
    """Frozen forecaster plus a resettable causal regional mean correction."""

    def __init__(
        self,
        model: EnsembleUNet2d,
        device: torch.device,
        dtype: torch.dtype,
        model_stats: dict,
        official_stats: dict,
        alpha: float,
        beta: float,
        update_clip: float,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.model_stats = model_stats
        self.official_stats = official_stats
        self.alpha = alpha
        self.beta = beta
        self.update_clip = update_clip
        self._prev_base_prediction: Optional[torch.Tensor] = None
        self._bias: Optional[torch.Tensor] = None
        self._graph = None
        self._reset_history()

    def reset_ttt_state(self) -> None:
        self.model.eval()
        self._prev_base_prediction = None
        self._bias = None
        self._reset_history()

    def _reset_history(self) -> None:
        if not hasattr(self, '_history_sum'):
            self._history_sum = torch.zeros((1, 32, 64, 2), device=self.device)
            self._history_count = torch.zeros((), device=self.device)
            self._history_previous = (torch.zeros_like(self._history_sum),
                                      torch.zeros_like(self._history_sum))
            self._history_cross = torch.zeros((2, 4, 2), device=self.device, dtype=torch.float64)
            self._history_square = torch.zeros((2, 4, 2), device=self.device, dtype=torch.float64)
        for value in (self._history_sum, self._history_count, self._history_cross,
                      self._history_square, *self._history_previous):
            value.zero_()

    def _history_shift(self, value, previous_target, base, valid=None):
        stats = self.official_stats
        if valid is None:
            valid = float(previous_target is not None)
        # At the official stride of 20, this input is the previous revealed target.
        mean_input = value[..., :2].mean(dim=1) * stats['std_input'][:2] + stats['mean_input'][:2]
        previous_direction, previous_mean = self._history_previous
        residual = mean_input - previous_mean
        self._history_cross.add_((previous_direction * residual).reshape(1, 2, 16, 4, 16, 2)
                                .mean(dim=(0, 2, 4), dtype=torch.float64) * valid)
        self._history_square.add_(previous_direction.square().reshape(1, 2, 16, 4, 16, 2)
                                 .mean(dim=(0, 2, 4), dtype=torch.float64) * valid)
        self._history_sum.add_(mean_input)
        self._history_count.add_(1)
        mean_prediction = base[..., :2].mean(dim=1) * stats['std_target'][:2] + stats['mean_target'][:2]
        direction = self._history_sum / self._history_count - mean_prediction
        previous_direction.copy_(direction)
        previous_mean.copy_(mean_prediction)
        observations = self._history_count - 1
        alpha = (self._history_cross / self._history_square.clamp_min(1e-20)).clamp(0, .5)
        alpha = alpha * observations / (observations + 5) * (observations >= 3)
        correction = (alpha[None, :, None, :, None, :] * direction.reshape(1, 2, 16, 4, 16, 2))
        correction = correction.reshape(1, 32, 64, 2).float()
        return (correction / stats['std_target'][:2]).unsqueeze(1)

    def _forward_base(
        self, input_official_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        official = torch.as_tensor(
            input_official_norm, device=self.device, dtype=torch.float32
        )
        raw = (
            official * self.official_stats["std_input"]
            + self.official_stats["mean_input"]
        )
        raw_model_dtype = raw.to(dtype=self.dtype)
        model_input = (
            raw_model_dtype
            - self.model_stats["mean_input"].to(dtype=self.dtype)
        ) / self.model_stats["std_input"].to(dtype=self.dtype)

        prediction_model, lower_extent, upper_extent = (
            self.model.forward_distribution(model_input)
        )
        model_mean_target = self.model_stats["mean_target"].to(dtype=self.dtype)
        model_std_target = self.model_stats["std_target"].to(dtype=self.dtype)
        prediction_raw = prediction_model * model_std_target + model_mean_target
        lower_raw = prediction_raw.clone()
        upper_raw = prediction_raw.clone()
        lower_raw[..., :2] = (
            prediction_raw[..., :2]
            - lower_extent * model_std_target[:2]
        )
        upper_raw[..., :2] = (
            prediction_raw[..., :2]
            + upper_extent * model_std_target[:2]
        )
        prediction_raw[..., 2] = 0.0
        lower_raw[..., 2] = 0.0
        upper_raw[..., 2] = 0.0

        physical_bundle = torch.stack((prediction_raw, lower_raw, upper_raw)).float()
        current_mean = official[..., :2].mean(dim=1) * self.official_stats['std_input'][:2] + self.official_stats['mean_input'][:2]
        history_mean = ((self._history_sum + current_mean) / (self._history_count + 1)).unsqueeze(1)
        prediction_raw, lower_raw, upper_raw = correct(raw, physical_bundle, history_mean).unbind(0)

        official_mean_target = self.official_stats["mean_target"]
        official_std_target = self.official_stats["std_target"]
        prediction = (
            prediction_raw.float() - official_mean_target
        ) / official_std_target
        lower = (lower_raw.float() - official_mean_target) / official_std_target
        upper = (upper_raw.float() - official_mean_target) / official_std_target
        prediction[..., 2] = 0.0
        lower[..., 2] = 0.0
        upper[..., 2] = 0.0
        return prediction, lower, upper

    def _adapt(self, prev_target_norm) -> Optional[float]:
        if (
            self.alpha == 0.0
            or prev_target_norm is None
            or self._prev_base_prediction is None
        ):
            return None
        target = torch.as_tensor(
            prev_target_norm, device=self.device, dtype=torch.float32
        )
        if tuple(target.shape) != _EXPECTED_SHAPE:
            raise ValueError(
                f"prev_target_norm has shape {tuple(target.shape)}, "
                f"expected {_EXPECTED_SHAPE}"
            )
        residual = target[..., :2] - self._prev_base_prediction[..., :2]
        update = residual.mean(dim=(1, 2, 3), keepdim=True).clamp(
            -self.update_clip, self.update_clip
        )
        self._bias = (
            update
            if self._bias is None
            else self.beta * self._bias + (1.0 - self.beta) * update
        )
        return float(torch.mean(residual.square()).item())

    def ttt_step(
        self,
        input_norm,
        prev_target_norm=None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        adapt_loss = self._adapt(prev_target_norm)
        value = torch.as_tensor(
            input_norm, device=self.device, dtype=torch.float32
        )
        if tuple(value.shape) != _EXPECTED_SHAPE:
            raise ValueError(
                f"input_norm has shape {tuple(value.shape)}, expected {_EXPECTED_SHAPE}"
            )

        with torch.inference_mode():
            if self._graph is None:
                base, lower, upper = self._forward_base(value)
            else:
                self._graph_input.copy_(value)
                self._graph_previous_valid.fill_(prev_target_norm is not None)
                self._graph.replay()
                base, lower, upper = (v.clone() for v in self._graph_outputs)

            if self._bias is not None:
                correction = self.alpha * self._bias
                prediction = base.clone()
                prediction[..., :2] += correction
                lower[..., :2] += correction
                upper[..., :2] += correction
            else:
                prediction = base

            if self._graph is None:
                shift = self._history_shift(value, prev_target_norm, base)
                prediction = prediction.clone()
                prediction[..., :2] += shift
                lower[..., :2] += shift
                upper[..., :2] += shift

            prediction[..., 2] = 0.0
            lower[..., 2] = 0.0
            upper[..., 2] = 0.0
            self._prev_base_prediction = (base if self._graph is None
                                          else self._graph_base_prediction).detach()

        return prediction, {
            "adapt_loss": adapt_loss,
            "lower": lower,
            "upper": upper,
        }


def get_ttt_model(submission_dir: str, device: str):
    """Factory called once by the Track 2 evaluator before streaming starts."""
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
    torch_device = torch.device(device)
    model, dtype, model_stats, official_stats = _load_model(
        submission_dir, torch_device
    )
    policy = _load_policy(submission_dir)
    wrapped = B210StreamingModel(
        model=model,
        device=torch_device,
        dtype=dtype,
        model_stats=model_stats,
        official_stats=official_stats,
        alpha=policy["alpha"],
        beta=policy["beta"],
        update_clip=policy["update_clip"],
    )

    if torch_device.type == "cuda":
        dummy = torch.zeros(
            _EXPECTED_SHAPE, device=torch_device, dtype=torch.float32
        )
        with torch.inference_mode():
            wrapped._forward_base(dummy)
        torch.cuda.synchronize(torch_device)
        if policy.get("cuda_graph", False):
            wrapped._graph_input = dummy.clone()
            wrapped._graph_previous_valid = torch.zeros((), device=torch_device)
            capture_stream = torch.cuda.Stream(device=torch_device)
            capture_stream.wait_stream(torch.cuda.current_stream(torch_device))
            with torch.cuda.stream(capture_stream), torch.inference_mode():
                for _ in range(3):
                    wrapped._forward_base(wrapped._graph_input)
            torch.cuda.current_stream(torch_device).wait_stream(capture_stream)
            torch.cuda.synchronize(torch_device)
            wrapped._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(wrapped._graph), torch.inference_mode():
                base, lower, upper = wrapped._forward_base(wrapped._graph_input)
                wrapped._graph_base_prediction = base
                shift = wrapped._history_shift(wrapped._graph_input, None,
                                               base, wrapped._graph_previous_valid)
                prediction = base.clone()
                prediction[..., :2] += shift
                lower[..., :2] += shift
                upper[..., :2] += shift
                wrapped._graph_outputs = prediction, lower, upper
            wrapped.reset_ttt_state()
    return wrapped
