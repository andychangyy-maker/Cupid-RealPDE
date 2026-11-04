"""Deterministic FP32 EMA shadow state for named model parameters.

This module is intentionally independent of the C208 training entry points.
Callers own the optimizer and must call :meth:`update_after_optimizer_step`
after a successful optimizer step and before the scheduler step.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import struct
from typing import Mapping

import torch
from torch import nn


EMA_SEAL_SCHEMA = "fp32_named_parameter_ema_seal_v1"
UPDATE_ORDER_CONTRACT = "after_optimizer_step_before_scheduler_step"


def canonical_sha256(value) -> str:
    """Hash JSON-compatible evidence using a stable canonical encoding."""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor state independent of ``torch.save`` container details."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack("<Q", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<Q", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<q", int(dimension)))
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _parameter_manifest(model: nn.Module) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if not (parameter.is_floating_point()
                or parameter.dtype == torch.complex64):
            raise TypeError(
                "EMA parameter must be floating point or complex64: "
                f"{name} ({parameter.dtype})")
        manifest.append({
            "index": index,
            "name": name,
            "shape": list(parameter.shape),
            "source_dtype": str(parameter.dtype),
            "numel": parameter.numel(),
        })
    if not manifest:
        raise ValueError("EMA requires at least one named parameter")
    return manifest


def _cpu_clone_state(
        state_dict: Mapping[str, torch.Tensor],
        ) -> OrderedDict[str, torch.Tensor]:
    cloned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state entry is not a tensor: {name}")
        cloned[name] = value.detach().cpu().clone()
    return cloned


class FP32EMAShadow:
    """EMA state that is neither registered on the model nor optimizer-owned."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        decay = float(decay)
        if not math.isfinite(decay) or not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be finite and lie in (0, 1)")
        self.decay = decay
        self._manifest = _parameter_manifest(model)
        self._manifest_sha256 = canonical_sha256(self._manifest)
        # Complex64 stores FP32 real and imaginary components.  Retaining that
        # dtype is essential for spectral parameters: converting them to FP32
        # silently discards the imaginary component.  The real-only route is
        # intentionally byte-for-byte the historical implementation.
        self._shadow: OrderedDict[str, torch.Tensor] = OrderedDict(
            (name, parameter.detach().to(
                dtype=(torch.complex64 if parameter.is_complex()
                       else torch.float32)).clone())
            for name, parameter in model.named_parameters()
        )
        for shadow in self._shadow.values():
            shadow.requires_grad_(False)
        self._updates = 0

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def parameter_manifest_sha256(self) -> str:
        return self._manifest_sha256

    def parameter_manifest(self) -> list[dict[str, object]]:
        """Return a detached JSON-compatible copy of the frozen manifest."""
        return [
            {**entry, "shape": list(entry["shape"])}
            for entry in self._manifest
        ]

    def shadow_parameter_state(self) -> OrderedDict[str, torch.Tensor]:
        """Return CPU clones so callers cannot mutate the live shadow."""
        return _cpu_clone_state(self._shadow)

    def _validated_named_parameters(
            self, model: nn.Module,
            ) -> tuple[tuple[str, nn.Parameter], ...]:
        named = tuple(model.named_parameters())
        if len(named) != len(self._manifest):
            raise ValueError("named-parameter count changed after EMA creation")
        for entry, (name, parameter) in zip(self._manifest, named):
            if name != entry["name"]:
                raise ValueError("named-parameter order changed after EMA creation")
            if list(parameter.shape) != entry["shape"]:
                raise ValueError(f"parameter shape changed after EMA creation: {name}")
            if str(parameter.dtype) != entry["source_dtype"]:
                raise ValueError(f"parameter dtype changed after EMA creation: {name}")
            if parameter.device != self._shadow[name].device:
                raise ValueError(f"parameter device changed after EMA creation: {name}")
        return named

    @torch.no_grad()
    def update_after_optimizer_step(
            self, model: nn.Module, *, optimizer_step: int) -> None:
        """Update once from post-optimizer raw parameters.

        ``optimizer_step`` makes missing, duplicate, and reordered calls visible;
        it cannot introspect whether the caller actually ran the optimizer.
        """
        if isinstance(optimizer_step, bool) or not isinstance(optimizer_step, int):
            raise TypeError("optimizer_step must be an integer")
        expected = self._updates + 1
        if optimizer_step != expected:
            raise ValueError(
                f"EMA expected optimizer_step={expected}, got {optimizer_step}")
        named = self._validated_named_parameters(model)
        one_minus_decay = 1.0 - self.decay
        for name, parameter in named:
            source = parameter.detach()
            shadow_dtype = self._shadow[name].dtype
            if source.dtype != shadow_dtype:
                source = source.to(dtype=shadow_dtype)
            self._shadow[name].mul_(self.decay).add_(
                source, alpha=one_minus_decay)
        self._updates = optimizer_step

    def _validated_raw_state(
            self, model: nn.Module,
            ) -> tuple[OrderedDict[str, torch.Tensor], list[str]]:
        self._validated_named_parameters(model)
        raw_state = _cpu_clone_state(model.state_dict())
        parameter_names = [str(entry["name"]) for entry in self._manifest]
        missing_parameters = set(parameter_names).difference(raw_state)
        if missing_parameters:
            raise ValueError(
                "named parameters missing from model state: "
                f"{sorted(missing_parameters)}")
        persistent_buffers = [
            name for name, _buffer in model.named_buffers()
            if name in raw_state
        ]
        classified = set(parameter_names).union(persistent_buffers)
        unclassified = set(raw_state).difference(classified)
        if unclassified:
            raise ValueError(
                "model state contains entries outside named parameters and "
                f"persistent buffers: {sorted(unclassified)}")
        return raw_state, persistent_buffers

    def materialize_model_state(
            self, model: nn.Module,
            ) -> OrderedDict[str, torch.Tensor]:
        """Build a strict-loadable EMA state with exact current raw buffers."""
        raw_state, _buffer_names = self._validated_raw_state(model)
        materialized: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, raw_value in raw_state.items():
            if name in self._shadow:
                materialized[name] = self._shadow[name].detach().cpu().clone()
            else:
                materialized[name] = raw_value
        return materialized

    def seal(self, model: nn.Module) -> dict[str, object]:
        """Create a self-verifying, strict-loadable EMA endpoint payload."""
        raw_state, buffer_names = self._validated_raw_state(model)
        materialized: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, raw_value in raw_state.items():
            if name in self._shadow:
                materialized[name] = self._shadow[name].detach().cpu().clone()
            else:
                materialized[name] = raw_value
        parameter_state = OrderedDict(
            (name, materialized[name]) for name in self._shadow)
        buffer_state = OrderedDict(
            (name, materialized[name]) for name in buffer_names)
        seal: dict[str, object] = {
            "schema": EMA_SEAL_SCHEMA,
            "decay": self.decay,
            "updates": self._updates,
            "update_order": UPDATE_ORDER_CONTRACT,
            "shadow_dtype": str(torch.float32),
            "parameter_manifest": self.parameter_manifest(),
            "parameter_manifest_sha256": self._manifest_sha256,
            "buffer_names": buffer_names,
            "shadow_parameter_state_sha256": tensor_state_sha256(
                parameter_state),
            "buffer_state_sha256": tensor_state_sha256(buffer_state),
            "raw_model_state_sha256": tensor_state_sha256(raw_state),
            "materialized_model_state_sha256": tensor_state_sha256(
                materialized),
        }
        if any(value.is_complex() for value in self._shadow.values()):
            seal["complex_shadow_dtype"] = str(torch.complex64)
        seal["seal_metadata_sha256"] = canonical_sha256(seal)
        seal["model_state"] = materialized
        return seal

    @classmethod
    def from_seal(
            cls, model: nn.Module, seal: Mapping[str, object],
            ) -> "FP32EMAShadow":
        """Validate a seal and restore its shadow without mutating ``model``."""
        if seal.get("schema") != EMA_SEAL_SCHEMA:
            raise ValueError("unsupported EMA seal schema")
        if seal.get("update_order") != UPDATE_ORDER_CONTRACT:
            raise ValueError("EMA seal update-order contract mismatch")
        if seal.get("shadow_dtype") != str(torch.float32):
            raise ValueError("EMA seal shadow dtype must be torch.float32")
        updates = seal.get("updates")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("EMA seal updates must be a non-negative integer")
        decay = seal.get("decay")
        if not isinstance(decay, (int, float)):
            raise ValueError("EMA seal decay must be numeric")
        helper = cls(model, float(decay))
        expected_complex_dtype = (
            str(torch.complex64)
            if any(value.is_complex() for value in helper._shadow.values())
            else None)
        if seal.get("complex_shadow_dtype") != expected_complex_dtype:
            raise ValueError("EMA seal complex shadow dtype mismatch")
        manifest = seal.get("parameter_manifest")
        if manifest != helper._manifest:
            raise ValueError("EMA seal parameter manifest does not match model")
        if seal.get("parameter_manifest_sha256") != canonical_sha256(manifest):
            raise ValueError("EMA seal parameter manifest hash mismatch")
        if seal.get("parameter_manifest_sha256") != helper._manifest_sha256:
            raise ValueError("EMA seal parameter manifest identity mismatch")
        metadata = {
            key: value for key, value in seal.items()
            if key not in {"seal_metadata_sha256", "model_state"}
        }
        if seal.get("seal_metadata_sha256") != canonical_sha256(metadata):
            raise ValueError("EMA seal metadata hash mismatch")
        model_state = seal.get("model_state")
        if not isinstance(model_state, Mapping):
            raise ValueError("EMA seal model_state must be a mapping")
        live_state, live_buffer_names = helper._validated_raw_state(model)
        if list(model_state) != list(live_state):
            raise ValueError("EMA seal state keys/order do not match model")
        for name, live_value in live_state.items():
            sealed_value = model_state[name]
            if not isinstance(sealed_value, torch.Tensor):
                raise TypeError(f"EMA seal state entry is not a tensor: {name}")
            if sealed_value.shape != live_value.shape:
                raise ValueError(f"EMA seal tensor shape mismatch: {name}")
            if name in helper._shadow:
                expected_dtype = helper._shadow[name].dtype
                if sealed_value.dtype != expected_dtype:
                    raise ValueError(
                        f"EMA shadow tensor dtype mismatch: {name} "
                        f"({sealed_value.dtype} != {expected_dtype})")
            elif sealed_value.dtype != live_value.dtype:
                raise ValueError(f"EMA buffer dtype mismatch: {name}")
        if seal.get("buffer_names") != live_buffer_names:
            raise ValueError("EMA seal buffer names do not match model")
        parameter_state = OrderedDict(
            (name, model_state[name]) for name in helper._shadow)
        buffer_state = OrderedDict(
            (name, model_state[name]) for name in live_buffer_names)
        expected_hashes = {
            "shadow_parameter_state_sha256": tensor_state_sha256(
                parameter_state),
            "buffer_state_sha256": tensor_state_sha256(buffer_state),
            "materialized_model_state_sha256": tensor_state_sha256(model_state),
        }
        for field, expected_hash in expected_hashes.items():
            if seal.get(field) != expected_hash:
                raise ValueError(f"EMA seal hash mismatch: {field}")
        raw_hash = seal.get("raw_model_state_sha256")
        if (not isinstance(raw_hash, str) or len(raw_hash) != 64
                or any(character not in "0123456789abcdef" for character in raw_hash)):
            raise ValueError("EMA seal raw-model hash is malformed")
        for name, parameter in model.named_parameters():
            helper._shadow[name].copy_(
                model_state[name].detach().to(
                    device=parameter.device,
                    dtype=helper._shadow[name].dtype))
        helper._updates = updates
        return helper
