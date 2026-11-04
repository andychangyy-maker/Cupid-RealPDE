#!/usr/bin/env python3
"""Resume-safe dense Real adaptation for a causal PC3D Sim checkpoint.

Fit data are restricted to the attested clean81 prefixes.  The normalizer is
computed from the existing sparse stride-20 (2544 window) stream, while the
optimizer consumes the separate stride-5 dense stream (9933 windows for the
formal manifest).  Real PIV measures u/v only: its third, compatibility
pressure channel is *exactly zero*, is never augmented, and has an explicit
zero target in the mature all-channel normalized MSE.  A clean716 replay is deliberately reachable only
after an atomic EMA endpoint seal has been written and re-hashed.

The resume checkpoint contains model/optimizer/scheduler/EMA state, every RNG
state, the deterministic sample schedule and its cursor, and immutable source
and data seals.  It is therefore an exact continuation point, rather than a
weights-only warm start.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402
import train_c208_dr1_clean81_paired_loao as clean81  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_ensemble_unet2d as mature_real  # noqa: E402
import fp32_ema_shadow as ema_lib  # noqa: E402
import train_pyramid_causal3d_sim_long as sim_long  # noqa: E402
from models.pyramid_causal3d_operator import (  # noqa: E402
    PyramidCausal3DOperator, PyramidCausal3DOperatorConfig,
)


SCHEMA = "pc3d_real_dense_augmentation_v1"
RESUME_SCHEMA = "pc3d_real_dense_exact_resume_v1"
ENDPOINT_SCHEMA = "pc3d_real_dense_ema_endpoint_v1"
EXCLUDED_BASENAME = clean81.EXCLUDED_BASENAME
FORMAL_MANIFEST_SHA256 = clean81.FORMAL_MANIFEST_SHA256
NORMALIZER_STRIDE = 20
OPTIMIZER_STRIDE = 5
EXPECTED_NORMALIZER_WINDOWS = 2544
EXPECTED_OPTIMIZER_WINDOWS = 9933
RAW_NORMALIZER_STATS_SHA256 = "3f2605d84ad8da57fedddfaeff292050f1173ce37f75defa9dd0de6e4b4e9492"
UPDATES = 4_000
BATCH_SIZE = 8
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-5
EMA_DECAY = .999
GRAD_CLIP = 1.0
SCHEDULER_T_MAX = 12_000
SEED = 4
ACKNOWLEDGEMENT = "RUN_PC3D_REAL_DENSE_AUGMENTATION"
MSE_WARMUP_UPDATES = 1_500
ACCURACY_WEIGHTS = (5.0, 3.0, 2.0)
MSE_WEIGHT = .25
AUGMENTATION = {
    "gain_std": .01,
    "normalized_noise_std": .01,
    "mask_aware_blur_probability": .15,
    "mask_aware_blur_strength": .20,
}


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return base.state_sha({key: value.detach().cpu() for key, value in state.items()})


def real_training_contract(*, updates: int, batch_size: int,
                           scheduler_t_max: int) -> dict[str, object]:
    """Describe every choice that can change an exact Real continuation."""
    return {
        "updates": int(updates), "batch_size": int(batch_size), "seed": SEED,
        "sample_schedule": "seed4_epoch_randperm_with_short_final_batch",
        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR", "scheduler_t_max": int(scheduler_t_max),
        "grad_clip": GRAD_CLIP, "ema_decay": EMA_DECAY,
        "mse_warmup_updates": MSE_WARMUP_UPDATES,
        "accuracy_weights": list(ACCURACY_WEIGHTS), "mse_weight": MSE_WEIGHT,
        "loss": "u1_u1500_all3_mse_then_accuracy_5_3_2_plus_.25_all3_mse",
        "augmentation": dict(AUGMENTATION), "endpoint": "ema_u4000_only",
        "cuda_reproducibility": {
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False, "cudnn_deterministic": True,
        },
    }


def atomic_torch_save(path: Path, value: object) -> None:
    """Atomically publish a complete checkpoint; never expose a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


class Window:
    def __init__(self, trajectory: int, start: int):
        self.trajectory, self.start = int(trajectory), int(start)


class RealWindowDataset:
    """Promote measured [u,v] to the fixed [u,v,zero-pressure] API once."""
    def __init__(self, trajectories: Sequence[torch.Tensor], windows: Sequence[Window]):
        self.trajectories, self.windows = list(trajectories), list(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[int(index)]
        uv = self.trajectories[window.trajectory][window.start:window.start + 40]
        if tuple(uv.shape[-1:]) != (2,) or len(uv) != 40:
            raise ValueError("Real window must contain exactly 40 [u,v] frames")
        pressure = torch.zeros((*uv.shape[:-1], 1), dtype=uv.dtype)
        uvp = torch.cat((uv, pressure), dim=-1)
        return uvp[:20], uvp[20:]


class PrefixAccessAudit:
    """Allow only clean81 u/v prefix reads before endpoint sealing."""
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self._opened: dict[str, set[str]] = {"fit": set(), "clean716": set()}
        self.seal: dict[str, object] | None = None

    def record_endpoint_seal(self, seal: Mapping[str, object]) -> None:
        if self.seal is not None:
            raise RuntimeError("PC3D Real endpoint was sealed twice")
        self.seal = dict(seal)

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        if filename == EXCLUDED_BASENAME:
            raise RuntimeError("excluded trajectory reached PC3D H5 opener")
        if filename in self._opened[phase]:
            raise RuntimeError(f"PC3D H5 opened twice: {filename}")
        if phase == "fit":
            if self.seal is not None:
                raise RuntimeError("fit prefix cannot be opened after endpoint seal")
        elif phase == "clean716":
            if self.seal is None:
                raise RuntimeError("clean716 cannot be opened before endpoint seal")
            endpoint = Path(str(self.seal["path"]))
            if not endpoint.is_file() or file_sha256(endpoint) != self.seal["file_sha256"]:
                raise RuntimeError("sealed endpoint changed before clean716 evaluation")
        else:
            raise ValueError(f"unsupported access phase {phase!r}")

    def record_open(self, row: Mapping[str, object], phase: str, boundary: int) -> None:
        filename = str(row["file"])
        self._opened[phase].add(filename)
        self.events.append({
            "file": filename, "phase": phase,
            "channels_opened": ["u", "v"], "pressure_opened": False,
            "max_read_frame_exclusive": int(boundary),
            "endpoint_sealed_before_open": self.seal is not None,
        })


def load_clean_manifest(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows, report = clean81.load_manifest(path, FORMAL_MANIFEST_SHA256)
    if len(rows) != 81 or sum(len(row["train_starts"]) for row in rows) != EXPECTED_NORMALIZER_WINDOWS:
        raise RuntimeError("clean81 stride20/2544 manifest contract changed")
    if any(row["file"] == EXCLUDED_BASENAME for row in rows):
        raise RuntimeError("clean81 manifest includes excluded trajectory")
    return rows, report


def dense_windows(trajectories: Sequence[torch.Tensor], stride: int = OPTIMIZER_STRIDE) -> list[Window]:
    if stride <= 0:
        raise ValueError("window stride must be positive")
    return [Window(trajectory, start)
            for trajectory, values in enumerate(trajectories)
            for start in range(0, int(values.shape[0]) - 40 + 1, stride)]


def load_real_rows(real_dir: Path, rows: Sequence[Mapping[str, object]], *,
                   phase: str, audit: PrefixAccessAudit) -> tuple[list[torch.Tensor], list[Window]]:
    root = real_dir.resolve()
    trajectories: list[torch.Tensor] = []
    windows: list[Window] = []
    for trajectory, row in enumerate(rows):
        audit.before_open(row, phase)
        filename = str(row["file"])
        path = (root / filename).resolve()
        if path.parent != root or path.name != filename or not path.is_file():
            raise FileNotFoundError(f"unsafe or missing Real H5: {filename}")
        boundary = int(row["val_boundary"])
        with h5py.File(path, "r") as handle:
            if "u" not in handle or "v" not in handle:
                raise ValueError(f"{filename} lacks measured u/v")
            # Do not even touch a possible p key: Real pressure is unmeasured.
            if handle["u"].shape != handle["v"].shape:
                raise ValueError(f"u/v shapes differ in {filename}")
            if int(handle["u"].shape[0]) != int(row["frames"]):
                raise ValueError(f"frame count changed in {filename}")
            u = np.asarray(handle["u"][:boundary, ::2, ::2], dtype=np.float32)
            v = np.asarray(handle["v"][:boundary, ::2, ::2], dtype=np.float32)
        if u.shape != v.shape or int(u.shape[0]) != boundary:
            raise ValueError(f"prefix shape mismatch in {filename}")
        audit.record_open(row, phase, boundary)
        trajectories.append(torch.from_numpy(np.ascontiguousarray(np.stack((u, v), axis=-1))))
        windows.extend(Window(trajectory, int(start)) for start in row["train_starts"])
    return trajectories, windows


def normalizer_from_sparse_windows(trajectories: Sequence[torch.Tensor],
                                   windows: Sequence[Window], device: torch.device):
    """Use the audited stride-20 windows, and only then append zero pressure."""
    if len(windows) != EXPECTED_NORMALIZER_WINDOWS:
        raise RuntimeError("PC3D normalizer must use exactly the clean81 2544 windows")
    raw_statistics = smu1.fold_statistics(
        smu1.window_sufficient_statistics(trajectories, windows), range(len(windows)))
    raw_digest = smu1.statistics_digest(raw_statistics)
    if raw_digest != RAW_NORMALIZER_STATS_SHA256:
        raise RuntimeError("clean81 stride20 raw normalizer digest changed")
    # Reuse the mature FoldNormalizer rule: a structurally-zero Real pressure
    # standard deviation is replaced by one, not by a numerical epsilon.
    normalizer = smu1.FoldNormalizer(raw_statistics, device=device)
    return normalizer, {
        "kind": "clean81_prefix_stride20_2544_window_audit",
        "windows": len(windows), "stride": NORMALIZER_STRIDE,
        "raw_statistics_sha256": raw_digest,
        "mean_input": normalizer.mean_input.detach().cpu().tolist(),
        "mean_target": normalizer.mean_target.detach().cpu().tolist(),
        "std_input": normalizer.std_input.detach().cpu().tolist(),
        "std_target": normalizer.std_target.detach().cpu().tolist(),
        "pressure_mean_exact_zero": True,
        "pressure_std_source_exact_zero": True,
        "pressure_std_effective": float(normalizer.std_target[2].detach().cpu()),
    }


def real_channel_mask(batch: torch.Tensor) -> torch.Tensor:
    if batch.shape[-1] != 3:
        raise ValueError("PC3D expects compatibility [u,v,p] channels")
    # The measurement mask is semantic, not inferred from a potentially calm u/v frame.
    return torch.tensor((True, True, False), device=batch.device).view(1, 1, 1, 1, 3)


def augment_real_history(history_physical: torch.Tensor, normalizer) -> torch.Tensor:
    """Exact mature order: physical gain, normalize, masked blur, masked noise."""
    gained = mature_real.add_training_input_gain(history_physical, AUGMENTATION["gain_std"])
    normalized, _ = normalizer.preprocess(gained, history_physical)
    normalized = mature_real.masked_velocity_blur(
        normalized, history_physical, AUGMENTATION["mask_aware_blur_probability"],
        AUGMENTATION["mask_aware_blur_strength"])
    return mature_real.add_training_input_noise(
        normalized, history_physical, AUGMENTATION["normalized_noise_std"])


def real_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mature all-three-channel normalized MSE; p has an explicit zero target."""
    return (prediction - target).square().mean()


def zero_unmeasured_pressure(prediction: torch.Tensor) -> torch.Tensor:
    """Keep Real predictions in the v9 [u,v,p=0] compatibility convention."""
    if prediction.shape[-1] != 3:
        raise ValueError("PC3D Real prediction must expose [u,v,p]")
    result = prediction.clone()
    result[..., 2] = 0.0
    return result


def _config_dict(value: object) -> dict[str, object]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("PC3D checkpoint model_config must be a mapping/dataclass")


def load_pc3d_sim_checkpoint(path: Path, device: torch.device) -> tuple[PyramidCausal3DOperator, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("PC3D Sim checkpoint payload must be a mapping")
    required = {
        "checkpoint_version", "schema", "endpoint_kind", "ema_state",
        "model_config", "interaction_mode", "model_state", "model_state_sha256",
        "normalizer_tensors", "normalizer_sha256", "next_update", "sample_cursor",
        "schedule_sha256", "config_seal", "data_seal", "git_seal", "device_type",
    }
    missing = required.difference(payload)
    if missing:
        raise KeyError("Stage-S endpoint lacks required seals: " + ", ".join(sorted(missing)))
    if (payload["checkpoint_version"] != sim_long.CHECKPOINT_VERSION
            or payload["schema"] != sim_long.SCHEMA
            or payload["endpoint_kind"] != sim_long.ENDPOINT_KIND
            or payload["ema_state"] is not None
            or int(payload["next_update"]) != sim_long.UPDATES
            or int(payload["sample_cursor"]) != sim_long.UPDATES
            or payload["schedule_sha256"] != sim_long.SAMPLE_INDEX_SHA256
            or payload["device_type"] != "cuda"):
        raise RuntimeError("Real adaptation requires the sealed CUDA Stage-S raw-u10000 endpoint")
    sealed_model_config = _config_dict(payload["model_config"])
    expected_stage_s_config = {
        **sim_long.training_contract(), "architecture": dict(sealed_model_config),
        "device_type": "cuda",
    }
    if payload["config_seal"] != sim_long.canonical_sha(expected_stage_s_config):
        raise RuntimeError("PC3D Sim checkpoint configuration seal mismatch")
    config_data = dict(sealed_model_config)
    architecture = config_data.pop("architecture", "pyramid_causal3d_operator")
    interaction_mode = config_data.pop("interaction_mode", "causal_temporal_candidate")
    if architecture != "pyramid_causal3d_operator":
        raise ValueError("start checkpoint is not a PC3D checkpoint")
    if interaction_mode != "causal_temporal_candidate":
        raise ValueError("PC3D Real requires a causal-temporal Sim endpoint")
    if (payload.get("interaction_mode") is not None
            and payload["interaction_mode"] != interaction_mode):
        raise RuntimeError("PC3D Sim checkpoint interaction-mode seals disagree")
    config = PyramidCausal3DOperatorConfig(**config_data)
    if (config.in_channels, config.out_channels, config.in_steps, config.out_steps) != (3, 3, 20, 20):
        raise ValueError("PC3D Sim checkpoint violates [u,v,p] 20-to-20 contract")
    state = payload["model_state"]
    if not isinstance(state, Mapping):
        raise KeyError("PC3D Sim checkpoint lacks model_state_dict")
    model = PyramidCausal3DOperator(config, interaction_mode).float()
    if sum(parameter.numel() for parameter in model.parameters()) != sim_long.EXPECTED_PARAMETERS:
        raise RuntimeError("Stage-S PC3D parameter count changed")
    model.load_state_dict(state, strict=True)
    if str(payload["model_state_sha256"]) != tensor_state_sha(model.state_dict()):
        raise RuntimeError("PC3D Sim checkpoint model-state seal mismatch")
    report = {
        "path": str(path.resolve()), "file_sha256": file_sha256(path),
        "model_state_sha256": tensor_state_sha(model.state_dict()),
        "stage_s_schema": payload["schema"], "stage_s_update": int(payload["next_update"]),
        "stage_s_schedule_sha256": payload["schedule_sha256"],
        "stage_s_config_sha256": payload["config_seal"],
        "stage_s_data_seal": dict(payload["data_seal"]),
        "stage_s_git_seal": dict(payload["git_seal"]),
        "model_config": {"architecture": architecture, "interaction_mode": interaction_mode, **asdict(config)},
        "model_config_sha256": canonical_sha({"architecture": architecture, "interaction_mode": interaction_mode, **asdict(config)}),
    }
    tensors = payload["normalizer_tensors"]
    normalizer_fields = {"mean_input", "mean_target", "std_input", "std_target"}
    if not isinstance(tensors, Mapping) or set(tensors) != normalizer_fields:
        raise ValueError("PC3D Sim normalizer tensor inventory changed")
    normalized = {name: value.detach().cpu() for name, value in tensors.items()}
    report["sim_normalizer_sha256"] = tensor_state_sha(normalized)
    if payload["normalizer_sha256"] != report["sim_normalizer_sha256"]:
        raise RuntimeError("PC3D Sim normalizer seal mismatch")
    return model.to(device), report


def install_normalizer(model: PyramidCausal3DOperator, normalizer) -> None:
    model.set_normalizer_statistics(normalizer.mean_input, normalizer.mean_target,
                                   normalizer.std_input, normalizer.std_target)


def capture_rng(device: torch.device) -> dict[str, object]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_rng(state: Mapping[str, object], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_real_recipe(device: torch.device) -> None:
    """Stage R follows the mature clean81 Real4k seed, not Stage-S's seed."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)


def make_sample_schedule(samples: int, updates: int, batch_size: int) -> list[list[int]]:
    """Mature seed-4 epoch-randperm stream; final epoch batches may be short."""
    if samples <= 0 or updates <= 0 or batch_size <= 0:
        raise ValueError("sample count, updates, and batch size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    batches: list[list[int]] = []
    while len(batches) < updates:
        epoch = torch.randperm(samples, generator=generator).tolist()
        batches.extend([epoch[start:start + batch_size] for start in range(0, samples, batch_size)])
    return batches[:updates]


def schedule_sha(schedule: Sequence[Sequence[int]]) -> str:
    return canonical_sha([[int(index) for index in batch] for batch in schedule])


def _normalizer_payload(normalizer) -> dict[str, torch.Tensor]:
    return {name: getattr(normalizer, name).detach().cpu().clone()
            for name in ("mean_input", "mean_target", "std_input", "std_target")}


def _normalizer_sha(normalizer) -> str:
    return tensor_state_sha(_normalizer_payload(normalizer))


def build_resume_payload(*, model: nn.Module, optimizer, scheduler, ema: ema_lib.FP32EMAShadow,
                         update: int, schedule: Sequence[Sequence[int]], source: Mapping[str, object],
                         normalizer, device: torch.device,
                         trace: Sequence[Mapping[str, object]],
                         endpoint_seal: Mapping[str, object] | None = None) -> dict[str, object]:
    schedule_batches = [[int(index) for index in batch] for batch in schedule]
    batch_size = max(map(len, schedule_batches))
    contract = real_training_contract(
        updates=len(schedule_batches), batch_size=batch_size,
        scheduler_t_max=int(scheduler.state_dict()["T_max"]))
    return {
        "schema": RESUME_SCHEMA, "update": int(update), "sample_cursor": int(update),
        "schedule": schedule_batches, "schedule_sha256": schedule_sha(schedule),
        "source": dict(source), "normalizer": _normalizer_payload(normalizer),
        "normalizer_sha256": _normalizer_sha(normalizer),
        "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "model_state_sha256": tensor_state_sha(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "ema_seal": ema.seal(model), "rng": capture_rng(device), "trace": [dict(row) for row in trace],
        "endpoint_seal": None if endpoint_seal is None else dict(endpoint_seal),
        "training_contract": contract, "device_type": device.type,
    }


def restore_resume(path: Path, *, model: nn.Module, optimizer, scheduler,
                   expected_source: Mapping[str, object], normalizer,
                   device: torch.device, schedule: Sequence[Sequence[int]]) -> tuple[int, ema_lib.FP32EMAShadow, list[dict[str, object]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != RESUME_SCHEMA:
        raise ValueError("not a PC3D exact resume checkpoint")
    expected_contract = real_training_contract(
        updates=len(schedule), batch_size=max(map(len, schedule)),
        scheduler_t_max=int(scheduler.state_dict()["T_max"]))
    if payload.get("training_contract") != expected_contract:
        raise RuntimeError("resume training contract changed")
    if payload.get("device_type") != device.type:
        raise RuntimeError("resume device type changed")
    if payload["source"] != dict(expected_source) or payload["normalizer_sha256"] != _normalizer_sha(normalizer):
        raise RuntimeError("resume source or normalizer seal changed")
    if (payload["schedule"] != [[int(index) for index in batch] for batch in schedule]
            or payload["schedule_sha256"] != schedule_sha(schedule)):
        raise RuntimeError("resume sample schedule changed")
    cursor = int(payload["sample_cursor"])
    if cursor != int(payload["update"]) or not 0 <= cursor <= len(schedule):
        raise ValueError("resume sample cursor is invalid")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if payload.get("model_state_sha256") != tensor_state_sha(model.state_dict()):
        raise RuntimeError("resume raw-model state seal changed")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    ema = ema_lib.FP32EMAShadow.from_seal(model, payload["ema_seal"])
    restore_rng(payload["rng"], device)
    trace = payload.get("trace")
    if not isinstance(trace, list):
        raise ValueError("resume trace is malformed")
    return cursor, ema, [dict(row) for row in trace]


def seal_endpoint(path: Path, *, model: PyramidCausal3DOperator, ema: ema_lib.FP32EMAShadow,
                  source: Mapping[str, object], normalizer, update: int,
                  schedule: Sequence[Sequence[int]]) -> dict[str, object]:
    if update != UPDATES or ema.updates != UPDATES:
        raise RuntimeError("PC3D Real endpoint is not EMA-u4000")
    ema_seal = ema.seal(model)
    state = ema_seal["model_state"]
    payload = {"schema": ENDPOINT_SCHEMA, "update": int(update), "model_state_dict": state,
               "model_state_sha256": tensor_state_sha(state), "model_config": asdict(model.config),
               "interaction_mode": model.interaction_mode, "source": dict(source),
               "normalizer": _normalizer_payload(normalizer), "normalizer_sha256": _normalizer_sha(normalizer),
               "sample_schedule_sha256": schedule_sha(schedule), "ema": ema_seal, "ema_decay": EMA_DECAY,
               "real_pressure": "unmeasured_exact_zero_and_unscored"}
    atomic_torch_save(path, payload)
    return {"path": str(path.resolve()), "file_sha256": file_sha256(path),
            "model_state_sha256": payload["model_state_sha256"], "update": int(update),
            "schema": ENDPOINT_SCHEMA}


def load_ema_endpoint(seal: Mapping[str, object], device: torch.device) -> PyramidCausal3DOperator:
    path = Path(str(seal["path"]))
    if not path.is_file() or file_sha256(path) != seal["file_sha256"]:
        raise RuntimeError("PC3D EMA endpoint seal mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != ENDPOINT_SCHEMA:
        raise ValueError("invalid PC3D endpoint schema")
    config = PyramidCausal3DOperatorConfig(**payload["model_config"])
    model = PyramidCausal3DOperator(config, str(payload["interaction_mode"])).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if tensor_state_sha(model.state_dict()) != seal["model_state_sha256"]:
        raise RuntimeError("PC3D endpoint state seal mismatch")
    return model.to(device)


def _score_error_matrix(scoring, errors: np.ndarray) -> dict[str, float]:
    physical = errors.mean(axis=0, dtype=np.float64)
    rel, tke, mvpe = (scoring.score_error(float(value)) for value in physical)
    return {"rel_l2_score": rel, "tke_rel_l2_score": tke, "mvpe_rel_l2_score": mvpe,
            "public_point_contribution_46_10_9": .46 * rel + .10 * tke + .09 * mvpe}


def evaluate_clean716(model: PyramidCausal3DOperator, dataset: RealWindowDataset,
                      normalizer, aoas: Sequence[int], device: torch.device) -> dict[str, object]:
    scoring = base.SCORING
    errors: list[np.ndarray] = []
    labels: list[int] = []
    prediction_hasher = hashlib.sha256()
    target_hasher = hashlib.sha256()
    model.eval()
    for start in range(0, len(dataset), 8):
        history, target = zip(*(dataset[index] for index in range(start, min(start + 8, len(dataset)))))
        history = torch.stack(history).to(device)
        target_array = np.ascontiguousarray(torch.stack(target).numpy())
        history = (history - normalizer.mean_input) / normalizer.std_input
        with torch.inference_mode():
            prediction = zero_unmeasured_pressure(
                model(history) * normalizer.std_target + normalizer.mean_target)
        prediction_array = np.ascontiguousarray(prediction.cpu().float().numpy())
        if not np.isfinite(prediction_array).all() or not np.array_equal(target_array[..., 2], np.zeros_like(target_array[..., 2])):
            raise FloatingPointError("invalid clean716 PC3D fields")
        measured = scoring.measured_channels(target_array)
        if measured != 2:
            raise RuntimeError("official v9 measured-channel semantics changed for Real pressure")
        errors.append(np.stack((scoring.rel_l2_per_sample(prediction_array, target_array, measured),
                                scoring.tke_rel_l2_per_sample(prediction_array, target_array, measured),
                                scoring.mvpe_rel_l2_per_sample(prediction_array, target_array)), axis=1))
        labels.extend(int(aoas[index]) for index in range(start, min(start + 8, len(dataset))))
        prediction_hasher.update(prediction_array.tobytes())
        target_hasher.update(target_array.tobytes())
    joined = np.concatenate(errors)
    if len(aoas) != len(dataset) or set(labels) != set(clean81.AOA_VALUES):
        raise RuntimeError("clean716 must retain all five AoA groups")
    pooled = _score_error_matrix(scoring, joined)
    label_array = np.asarray(labels)
    by_aoa = {str(aoa): {"windows": int(np.count_nonzero(label_array == aoa)),
                         **_score_error_matrix(scoring, joined[label_array == aoa])}
              for aoa in clean81.AOA_VALUES}
    if any(value["windows"] == 0 for value in by_aoa.values()):
        raise RuntimeError("a clean716 AoA group is empty")
    return {"windows": len(dataset), "official_measured_channels": 2,
            "pooled": pooled, "by_aoa": by_aoa,
            "prediction_sha256": prediction_hasher.hexdigest(), "target_sha256": target_hasher.hexdigest()}


def load_clean716_rows(path: Path, fit_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_file = {str(row["file"]): row for row in fit_rows}
    if not isinstance(raw, list) or len(raw) != len(by_file):
        raise ValueError("clean716 manifest must cover exactly clean81")
    rows = []
    for value in raw:
        filename = value.get("file") if isinstance(value, Mapping) else None
        if filename not in by_file:
            raise ValueError("clean716 manifest membership changed")
        fit = by_file[str(filename)]
        starts = value.get("val_starts")
        if (not isinstance(starts, list) or not starts or any(not isinstance(item, int) for item in starts)
                or any(item < int(fit["val_boundary"]) or item + 40 > int(fit["frames"]) for item in starts)):
            raise ValueError("clean716 window is outside the frozen tail")
        rows.append({**fit, "val_boundary": int(fit["frames"]), "train_starts": list(starts)})
    if sum(len(row["train_starts"]) for row in rows) != 716:
        raise RuntimeError("clean716 window count changed")
    return rows


def train(model: PyramidCausal3DOperator, dataset: RealWindowDataset, normalizer, source: Mapping[str, object],
          output_dir: Path, *, updates: int = UPDATES, batch_size: int = BATCH_SIZE,
          resume: Path | None = None, checkpoint_every: int = 250, device: torch.device) -> tuple[dict[str, object], dict[str, object]]:
    if updates <= 0 or batch_size <= 0 or checkpoint_every <= 0:
        raise ValueError("updates, batch-size, and checkpoint interval must be positive")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    schedule = make_sample_schedule(len(dataset), updates, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX)
    ema = ema_lib.FP32EMAShadow(model, EMA_DECAY)
    cursor, trace = 0, []
    if resume is not None:
        cursor, ema, trace = restore_resume(resume, model=model, optimizer=optimizer, scheduler=scheduler,
                                expected_source=source, normalizer=normalizer,
                                device=device, schedule=schedule)
    else:
        seed_real_recipe(device)
    started = time.perf_counter()
    resume_path = output_dir / "resume.pt"
    for cursor in range(cursor, updates):
        indices = schedule[cursor]
        history, target = zip(*(dataset[index] for index in indices))
        history = torch.stack(history).to(device)
        target = torch.stack(target).to(device)
        history_physical = history
        history = augment_real_history(history_physical, normalizer)
        _unused_history, target = normalizer.preprocess(history_physical, target)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        loss_args = argparse.Namespace(point_loss_mode="official_accuracy", mse_weight=MSE_WEIGHT,
                                       mse_warmup_updates=MSE_WARMUP_UPDATES)
        loss, details = mature_real.point_loss(loss_args, prediction, target, normalizer,
                                               ACCURACY_WEIGHTS, update := cursor + 1)
        mse = real_mse(prediction, target)
        accuracy = loss if update <= MSE_WARMUP_UPDATES else loss - MSE_WEIGHT * mse
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite PC3D Real loss at update {cursor + 1}")
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(gradient)):
            raise FloatingPointError(f"non-finite PC3D Real gradient at update {cursor + 1}")
        optimizer.step(); ema.update_after_optimizer_step(model, optimizer_step=update); scheduler.step()
        if update == 1 or update % checkpoint_every == 0 or update == updates:
            row = {"update": update, "loss": float(loss.detach()), "mse_all3": float(mse.detach()),
                   "accuracy": float(accuracy.detach()), "grad_norm": float(gradient.detach()),
                   "learning_rate": float(optimizer.param_groups[0]["lr"]),
                   **{key: float(value) for key, value in details.items()}}
            trace.append(row)
            payload = build_resume_payload(model=model, optimizer=optimizer, scheduler=scheduler, ema=ema,
                                           update=update, schedule=schedule, source=source, normalizer=normalizer,
                                           device=device, trace=trace)
            atomic_torch_save(resume_path, payload)
            print(json.dumps({"event": "pc3d_real_dense_progress", **row}, sort_keys=True), flush=True)
    endpoint = seal_endpoint(output_dir / "ema_endpoint.pt", model=model, ema=ema, source=source,
                             normalizer=normalizer, update=updates, schedule=schedule)
    payload = build_resume_payload(model=model, optimizer=optimizer, scheduler=scheduler, ema=ema,
                                   update=updates, schedule=schedule, source=source, normalizer=normalizer,
                                   device=device, trace=trace, endpoint_seal=endpoint)
    atomic_torch_save(resume_path, payload)
    return {"updates": updates, "batch_size": batch_size, "sample_schedule_sha256": schedule_sha(schedule),
            "sample_cursor": updates, "trace": trace, "elapsed_seconds": time.perf_counter() - started,
            "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
            "scheduler": f"CosineAnnealingLR_Tmax_{SCHEDULER_T_MAX}", "ema_decay": EMA_DECAY,
            "seed": SEED, "mse_warmup_updates": MSE_WARMUP_UPDATES,
            "loss": "u1_u1500_normalized_all3_mse_then_accuracy_5_3_2_plus_.25_all3_mse"}, endpoint


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError("PC3D Real production requires --execute and the exact acknowledgement")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PC3D Real production requires CUDA")
    if args.updates != UPDATES or args.batch_size != BATCH_SIZE:
        raise ValueError("PC3D Real production freezes updates4000 and batch8")
    if args.clean716_manifest is None:
        raise ValueError("PC3D Real production requires the frozen clean716 manifest")
    if any(value is None for value in (args.real_dir, args.clean_manifest, args.start_sim, args.output_dir)):
        raise ValueError("PC3D Real production requires real-dir, clean-manifest, start-sim, and output-dir")
    git = base.git_report()
    if not git["clean"]:
        raise RuntimeError(f"PC3D Real production requires clean Git: {git['porcelain']}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError("refusing to overwrite an existing PC3D Real output directory")
    if args.output_dir.exists() and args.resume is not None and args.resume.parent != args.output_dir:
        raise ValueError("resume checkpoint must belong to output-dir")
    rows, manifest_report = load_clean_manifest(args.clean_manifest)
    audit = PrefixAccessAudit()
    trajectories, sparse_windows = load_real_rows(args.real_dir, rows, phase="fit", audit=audit)
    normalizer, normalizer_report = normalizer_from_sparse_windows(trajectories, sparse_windows, device)
    optimizer_windows = dense_windows(trajectories)
    if len(optimizer_windows) != EXPECTED_OPTIMIZER_WINDOWS:
        raise RuntimeError(f"PC3D dense stride5 window count changed: {len(optimizer_windows)}")
    dataset = RealWindowDataset(trajectories, optimizer_windows)
    model, start = load_pc3d_sim_checkpoint(args.start_sim, device)
    if start["stage_s_git_seal"] != git:
        raise RuntimeError("Stage S and Stage R must run from the same clean Git identity")
    install_normalizer(model, normalizer)
    source = {"start_sim": start, "clean81_manifest_sha256": manifest_report["semantic_sha256"],
              "clean81_manifest_file_sha256": file_sha256(args.clean_manifest),
              "normalizer_sha256": _normalizer_sha(normalizer), "optimizer_windows": len(optimizer_windows),
              "optimizer_stride": OPTIMIZER_STRIDE, "git": git}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training, endpoint = train(model, dataset, normalizer, source, args.output_dir, updates=args.updates,
                               batch_size=args.batch_size, resume=args.resume,
                               checkpoint_every=args.checkpoint_every, device=device)
    audit.record_endpoint_seal(endpoint)
    evaluation = None
    if args.clean716_manifest is not None:
        tail_rows = load_clean716_rows(args.clean716_manifest, rows)
        tail_trajectories, tail_windows = load_real_rows(args.real_dir, tail_rows, phase="clean716", audit=audit)
        endpoint_model = load_ema_endpoint(endpoint, device)
        tail_aoas = [int(tail_rows[window.trajectory]["aoa"]) for window in tail_windows]
        evaluation = evaluate_clean716(endpoint_model, RealWindowDataset(tail_trajectories, tail_windows),
                                       normalizer, tail_aoas, device)
    result = {"schema": SCHEMA, "git": git, "start": start, "manifest": manifest_report, "normalizer": normalizer_report,
              "optimization": {"stride": OPTIMIZER_STRIDE, "windows": len(optimizer_windows), "augmentation": AUGMENTATION,
                               "default_updates": UPDATES, "default_batch_size": BATCH_SIZE,
                               "training_contract": real_training_contract(
                                   updates=UPDATES, batch_size=BATCH_SIZE,
                                   scheduler_t_max=SCHEDULER_T_MAX)},
              "real_pressure": {"source": "unmeasured", "compatibility_channel": "exact_zero",
                                "augmented": False, "mse": "all3_normalized_with_explicit_zero_target",
                                "official_v9_measured_channels": 2},
              "training": training, "endpoint": endpoint, "access_audit": audit.events,
              "clean716_evaluation": evaluation, "clean716_opened_before_endpoint_seal": False,
              "submission_authorized": False}
    path = args.output_dir / "result.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "formal"), default="contract")
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path, help="new causal-temporal PC3D Sim checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--clean716-manifest", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", "--acknowledge-protocol", dest="acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        print(json.dumps({"schema": SCHEMA, "acknowledgement": ACKNOWLEDGEMENT,
                          "device": "CUDA", **real_training_contract(
                              updates=UPDATES, batch_size=BATCH_SIZE,
                              scheduler_t_max=SCHEDULER_T_MAX)}, sort_keys=True))
        return
    result = run(args)
    print(json.dumps({"event": "pc3d_real_dense_complete", "endpoint": result["endpoint"],
                      "clean716_evaluated": result["clean716_evaluation"] is not None}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
