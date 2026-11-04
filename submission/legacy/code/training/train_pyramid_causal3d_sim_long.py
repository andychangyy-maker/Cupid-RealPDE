#!/usr/bin/env python3
"""Stage S: fresh, single-arm PC3D Sim-all100 long training.

This deliberately trains only the causal-temporal PC3D candidate from its
seed-210 random initialisation.  All 100 Sim trajectories contribute their
48 stride-20, 20-to-20 windows (4,800 windows total).  The three fixed
fit-only diagnostics at u2000/u5000/u10000 are reporting only: the raw u10000
endpoint is always the endpoint and no diagnostic can select a checkpoint.

Checkpoints live in an ignored ``checkpoints/`` directory.  They are written
atomically and include every state needed to resume the next update exactly,
including Python/NumPy/Torch RNG state, the deterministic sample cursor, and
strict configuration/data/Git seals.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Mapping

# Must be present before a CUDA BLAS handle is created.  Combined with the
# cuDNN flags below, this made an interrupted full PC3D step bitwise identical
# to the uninterrupted step on the local RTX 4090 preflight.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_b210_unified_temporal_state_sim_discovery as base  # noqa: E402


SCHEMA = "pc3d_sim_all100_long_stage_s_v1"
ACKNOWLEDGEMENT = "RUN_PC3D_SIM_ALL100_LONG_STAGE_S"
SEED = 210
SCHEDULE_SEED = 220
UPDATES = 10_000
BATCH_SIZE = 8
LR = 3.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
CHECKPOINT_INTERVAL = 1_000
TRACE_INTERVAL = 250
DIAGNOSTIC_UPDATES = (2_000, 5_000, 10_000)
DIAGNOSTIC_BATCH_SIZE = 8
EXPECTED_TRAJECTORIES = 100
WINDOW_STRIDE = 20
EXPECTED_WINDOWS_PER_TRAJECTORY = 48
EXPECTED_WINDOWS = EXPECTED_TRAJECTORIES * EXPECTED_WINDOWS_PER_TRAJECTORY
EXPECTED_PARAMETERS = 8_929_552
SAMPLE_INDEX_SHA256 = "bd6f71937b09a57aed1e0276862867291ff8b457b61774d83cabdea9c4dbadbe"
LOSS_WEIGHTS = (1.0, 4.0, 1.0)
MSE_WEIGHT = .25
EMA_ENABLED = False
ENDPOINT_KIND = "raw_model_u10000"
CHECKPOINT_VERSION = 2


def _model_classes():
    try:
        from models.pyramid_causal3d_operator import (  # noqa: PLC0415
            PyramidCausal3DOperator, PyramidCausal3DOperatorConfig)
    except ImportError as error:
        raise RuntimeError("PC3D model unavailable") from error
    return PyramidCausal3DOperatorConfig, PyramidCausal3DOperator


def model_config(*, tiny: bool = False):
    """Return the frozen production architecture (or a test-only small one)."""
    config_type, _operator = _model_classes()
    if not tiny:
        return config_type()
    names = {field.name for field in fields(config_type)} if is_dataclass(config_type) else set()
    small = {"e0_channels": 8, "e1_channels": 24, "e2_channels": 32,
             "e0_blocks": 1, "e1_blocks": 1, "e2_blocks": 1}
    return config_type(**{key: value for key, value in small.items() if key in names})


def _config_value(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _config_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_config_value(item) for item in value]
    return value


def training_contract(*, updates: int = UPDATES,
                      batch_size: int = BATCH_SIZE) -> dict[str, object]:
    return {
        "schema": SCHEMA, "stage": "S", "arm": "temporal_candidate",
        "interaction_mode": "causal_temporal_candidate", "seed": SEED,
        "schedule_seed": SCHEDULE_SEED, "updates": int(updates),
        "batch_size": int(batch_size), "optimizer": "AdamW", "lr": LR,
        "weight_decay": WEIGHT_DECAY, "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": int(updates), "grad_clip": GRAD_CLIP,
        "accuracy_weights": list(LOSS_WEIGHTS), "mse_weight": MSE_WEIGHT,
        "augmentation": "none", "teacher": "none", "uq": "none",
        "ema_enabled": EMA_ENABLED, "endpoint_kind": ENDPOINT_KIND,
        "window": {"history_steps": 20, "future_steps": 20,
                   "stride": WINDOW_STRIDE, "windows": EXPECTED_WINDOWS},
        "expected_parameters": EXPECTED_PARAMETERS,
        "sample_index_sha256": SAMPLE_INDEX_SHA256,
        "cuda_reproducibility": {
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False, "cudnn_deterministic": True,
        },
        "fixed_diagnostic_updates": list(DIAGNOSTIC_UPDATES),
        "trace_interval": TRACE_INTERVAL,
        "diagnostics_select_checkpoint": False,
    }


def canonical_sha(value: object) -> str:
    return base.canonical_sha(_config_value(value))


def seed_everything(seed: int, device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def construct_model(device: torch.device, *, tiny: bool = False) -> nn.Module:
    """Construct exactly the fresh, single causal-temporal candidate."""
    _config, operator = _model_classes()
    seed_everything(SEED, device)
    return operator(model_config(tiny=tiny), "causal_temporal_candidate").float().to(device)


def state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_sha(model_or_state: nn.Module | Mapping[str, torch.Tensor]) -> str:
    state = (state_snapshot(model_or_state) if isinstance(model_or_state, nn.Module)
             else model_or_state)
    return base.state_sha(state)


def model_architecture(model: nn.Module) -> object:
    value = model.architecture_config() if hasattr(model, "architecture_config") else model_config()
    return _config_value(value)


def normalizer_tensors(normalizer) -> dict[str, torch.Tensor]:
    names = ("mean_input", "mean_target", "std_input", "std_target")
    values = {name: getattr(normalizer, name).detach().cpu().clone() for name in names}
    if any(value.shape != (3,) for value in values.values()):
        raise RuntimeError("Stage S normalizer must contain four [3] channel tensors")
    return values


def normalizer_sha256(normalizer) -> str:
    return base.state_sha(normalizer_tensors(normalizer))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def interaction_mode(model: nn.Module) -> str:
    return str(getattr(model, "interaction_mode", getattr(model, "mode", "")))


def make_schedule(dataset_length: int, *, updates: int = UPDATES,
                  batch_size: int = BATCH_SIZE) -> np.ndarray:
    if dataset_length != EXPECTED_WINDOWS and (updates == UPDATES and batch_size == BATCH_SIZE):
        raise ValueError(f"Stage S requires exactly {EXPECTED_WINDOWS} windows, got {dataset_length}")
    if dataset_length <= 0 or updates <= 0 or batch_size <= 0:
        raise ValueError("dataset length, updates, and batch size must be positive")
    return np.random.default_rng(SCHEDULE_SEED).integers(
        0, dataset_length, size=(updates, batch_size), dtype=np.int64)


def schedule_seal(schedule: np.ndarray) -> str:
    if schedule.ndim != 2 or schedule.dtype != np.int64:
        raise ValueError("sample schedule must be a rank-two int64 array")
    return hashlib.sha256(schedule.tobytes()).hexdigest()


def validate_formal_schedule(schedule: np.ndarray) -> str:
    if schedule.shape != (UPDATES, BATCH_SIZE):
        raise RuntimeError(f"Stage S formal schedule shape changed: {schedule.shape}")
    digest = schedule_seal(schedule)
    if digest != SAMPLE_INDEX_SHA256:
        raise RuntimeError(f"Stage S sample-index SHA mismatch: {digest} != {SAMPLE_INDEX_SHA256}")
    return digest


def make_diagnostic_indices(dataset_length: int) -> np.ndarray:
    if dataset_length <= 0:
        raise ValueError("diagnostic dataset is empty")
    return np.random.default_rng(SEED + 1).integers(
        0, dataset_length, size=DIAGNOSTIC_BATCH_SIZE, dtype=np.int64)


def batch_at(dataset, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    samples = [dataset[int(index)] for index in indices]
    return (torch.stack([sample[0] for sample in samples]),
            torch.stack([sample[1] for sample in samples]))


def validate_all100_dataset(trajectories, windows) -> dict[str, object]:
    if len(trajectories) != EXPECTED_TRAJECTORIES:
        raise RuntimeError(f"Stage S requires {EXPECTED_TRAJECTORIES} trajectories, got {len(trajectories)}")
    if len(windows) != EXPECTED_WINDOWS:
        raise RuntimeError(f"Stage S requires {EXPECTED_WINDOWS} windows, got {len(windows)}")
    counts = [0] * EXPECTED_TRAJECTORIES
    expected_starts = tuple(range(0, 960, WINDOW_STRIDE))
    starts: list[list[int]] = [[] for _ in trajectories]
    for window in windows:
        trajectory, start = int(window.trajectory), int(window.start)
        if not 0 <= trajectory < EXPECTED_TRAJECTORIES:
            raise RuntimeError("window references a non-Sim-all100 trajectory")
        counts[trajectory] += 1
        starts[trajectory].append(start)
    if any(count != EXPECTED_WINDOWS_PER_TRAJECTORY for count in counts):
        raise RuntimeError("Stage S requires 48 windows for every trajectory")
    if any(tuple(sorted(value)) != expected_starts for value in starts):
        raise RuntimeError("Stage S window starts must be 0..940 with stride 20")
    return {"trajectories": len(trajectories), "windows": len(windows),
            "windows_per_trajectory": EXPECTED_WINDOWS_PER_TRAJECTORY,
            "stride": WINDOW_STRIDE}


def data_seal(sim_dir: Path, inventory_rows, trajectories, windows) -> dict[str, object]:
    shape = validate_all100_dataset(trajectories, windows)
    inventory_sha256 = canonical_sha(inventory_rows)
    window_sha256 = canonical_sha([
        {"trajectory": int(window.trajectory), "start": int(window.start)}
        for window in windows])
    return {"seal_kind": "semantic_inventory_and_window_contract_not_h5_content_sha256",
            "sim_dir": str(sim_dir.resolve()), "inventory_sha256": inventory_sha256,
            "windows_sha256": window_sha256, **shape}


def capture_rng_state(device: torch.device) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(), "cuda": None,
    }
    if device.type == "cuda":
        state["cuda"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, object], device: torch.device) -> None:
    required = {"python", "numpy", "torch_cpu", "cuda"}
    if set(state) != required:
        raise RuntimeError("checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu())
    cuda_state = state["cuda"]
    if device.type == "cuda":
        if not isinstance(cuda_state, list):
            raise RuntimeError("CUDA resume requires CUDA RNG state")
        torch.cuda.set_rng_state_all([
            torch.as_tensor(value, dtype=torch.uint8).cpu() for value in cuda_state])
    elif cuda_state is not None:
        raise RuntimeError("refusing a CUDA checkpoint on a non-CUDA device")


def atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    """Save a complete checkpoint or leave the previous checkpoint untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_save(value: object, path: Path) -> None:
    """Publish the monitoring trace without exposing a partial JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def checkpoint_payload(*, model: nn.Module, optimizer, scheduler,
                       next_update: int, schedule_sha256: str,
                       config_seal: str, data: Mapping[str, object],
                       git: Mapping[str, object], diagnostics: list[dict[str, object]],
                       trace: list[dict[str, object]],
                       normalizer,
                       device: torch.device) -> dict[str, object]:
    if next_update < 0:
        raise ValueError("sample cursor cannot be negative")
    return {
        "checkpoint_version": CHECKPOINT_VERSION, "schema": SCHEMA,
        "endpoint_kind": ENDPOINT_KIND, "ema_state": None,
        "model_config": model_architecture(model),
        "interaction_mode": interaction_mode(model),
        "normalizer_tensors": normalizer_tensors(normalizer),
        "normalizer_sha256": normalizer_sha256(normalizer),
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "next_update": int(next_update),
        "sample_cursor": int(next_update), "schedule_sha256": schedule_sha256,
        "config_seal": config_seal, "data_seal": dict(data), "git_seal": dict(git),
        "model_state_sha256": state_sha(model), "rng_state": capture_rng_state(device),
        "diagnostics": list(diagnostics), "trace": list(trace),
        "device_type": device.type,
    }


def _load_checkpoint(path: Path, *, model: nn.Module, optimizer, scheduler,
                     expected_config_seal: str, expected_data: Mapping[str, object],
                     expected_git: Mapping[str, object], expected_schedule_sha256: str,
                     updates: int, normalizer, device: torch.device) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
    # Keep metadata/normalizer tensors on CPU for device-independent seal
    # comparison.  ``Optimizer.load_state_dict`` moves optimizer tensors to
    # the devices of their bound parameters.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"checkpoint_version", "schema", "endpoint_kind", "ema_state",
                "model_config", "interaction_mode", "normalizer_tensors", "normalizer_sha256",
                "model_state", "optimizer_state", "scheduler_state", "next_update",
                "sample_cursor", "schedule_sha256", "config_seal", "data_seal",
                "git_seal", "model_state_sha256", "rng_state", "diagnostics", "trace",
                "device_type"}
    missing = required.difference(payload)
    if missing:
        raise RuntimeError("checkpoint missing required state: " + ", ".join(sorted(missing)))
    if (payload["checkpoint_version"] != CHECKPOINT_VERSION or payload["schema"] != SCHEMA
            or payload["endpoint_kind"] != ENDPOINT_KIND or payload["ema_state"] is not None):
        raise RuntimeError("checkpoint is not a raw Stage S PC3D checkpoint")
    if payload["device_type"] != device.type:
        raise RuntimeError("checkpoint device type differs from requested device")
    if payload["model_config"] != model_architecture(model):
        raise RuntimeError("checkpoint model configuration differs from constructed candidate")
    if payload["interaction_mode"] != interaction_mode(model):
        raise RuntimeError("checkpoint interaction mode differs from causal-temporal candidate")
    expected_normalizer = normalizer_tensors(normalizer)
    if (payload["normalizer_sha256"] != normalizer_sha256(normalizer)
            or set(payload["normalizer_tensors"]) != set(expected_normalizer)
            or any(not torch.equal(payload["normalizer_tensors"][name], value)
                   for name, value in expected_normalizer.items())):
        raise RuntimeError("checkpoint normalizer state seal mismatch")
    if payload["config_seal"] != expected_config_seal:
        raise RuntimeError("checkpoint configuration seal mismatch")
    if payload["data_seal"] != dict(expected_data):
        raise RuntimeError("checkpoint data seal mismatch")
    if payload["git_seal"] != dict(expected_git):
        raise RuntimeError("checkpoint Git seal mismatch")
    if payload["schedule_sha256"] != expected_schedule_sha256:
        raise RuntimeError("checkpoint sample-stream seal mismatch")
    cursor = int(payload["next_update"])
    if cursor != int(payload["sample_cursor"]) or not 0 <= cursor <= updates:
        raise RuntimeError("checkpoint sample cursor is invalid")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    if state_sha(model) != payload["model_state_sha256"]:
        raise RuntimeError("checkpoint model-state seal mismatch")
    restore_rng_state(payload["rng_state"], device)
    diagnostics = payload["diagnostics"]
    if not isinstance(diagnostics, list):
        raise RuntimeError("checkpoint diagnostics are malformed")
    trace = payload["trace"]
    if not isinstance(trace, list):
        raise RuntimeError("checkpoint trace is malformed")
    return cursor, diagnostics, trace


@torch.inference_mode()
def fixed_diagnostic(model: nn.Module, dataset, normalizer, indices: np.ndarray,
                     device: torch.device, update: int) -> dict[str, object]:
    """A fixed fit-only measurement; it has no checkpoint-selection role."""
    was_training = model.training
    model.eval()
    history, target = batch_at(dataset, indices)
    history, target = normalizer.preprocess(history.to(device), target.to(device))
    prediction = model(history)
    accuracy, details = base.accuracy_objective(prediction, target, normalizer, LOSS_WEIGHTS)
    mse = (prediction - target).square().mean()
    if was_training:
        model.train()
    return {"update": int(update), "role": "fixed_fit_diagnostic_not_checkpoint_selection",
            "accuracy": float(accuracy), "mse": float(mse),
            "loss": float(accuracy + MSE_WEIGHT * mse),
            **{key: float(value) for key, value in details.items()}}


def _optimizer(model: nn.Module, updates: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)


def train(*, model: nn.Module, dataset, normalizer, device: torch.device,
          updates: int = UPDATES, batch_size: int = BATCH_SIZE,
          checkpoint_dir: Path | None = None, checkpoint_interval: int = CHECKPOINT_INTERVAL,
          resume: Path | None = None, config_seal: str = "test", data: Mapping[str, object] | None = None,
          git: Mapping[str, object] | None = None,
          expected_schedule_sha256: str | None = None) -> dict[str, object]:
    """Train and return only the raw terminal state; no best-checkpoint logic exists."""
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    schedule = make_schedule(len(dataset), updates=updates, batch_size=batch_size)
    stream_sha256 = schedule_seal(schedule)
    if expected_schedule_sha256 is not None and stream_sha256 != expected_schedule_sha256:
        raise RuntimeError("sample stream differs from the required formal schedule seal")
    diagnostic_indices = make_diagnostic_indices(len(dataset))
    data = {} if data is None else dict(data)
    git = {} if git is None else dict(git)
    optimizer, scheduler = _optimizer(model, updates)
    cursor, diagnostics, trace = 0, [], []
    final_checkpoint_path: Path | None = None
    if resume is not None:
        cursor, diagnostics, trace = _load_checkpoint(
            resume, model=model, optimizer=optimizer, scheduler=scheduler,
            expected_config_seal=config_seal, expected_data=data, expected_git=git,
            expected_schedule_sha256=stream_sha256, updates=updates,
            normalizer=normalizer, device=device)
        if cursor == updates:
            final_checkpoint_path = resume
    elif checkpoint_dir is not None:
        atomic_torch_save(checkpoint_payload(
            model=model, optimizer=optimizer, scheduler=scheduler, next_update=0,
            schedule_sha256=stream_sha256, config_seal=config_seal, data=data, git=git,
            diagnostics=diagnostics, trace=trace, normalizer=normalizer, device=device),
            checkpoint_dir / "checkpoint_u00000000.pt")
    started = time.perf_counter()
    for update_index in range(cursor, updates):
        update = update_index + 1
        model.train()
        history, target = batch_at(dataset, schedule[update_index])
        history, target = normalizer.preprocess(history.to(device), target.to(device))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        accuracy, _details = base.accuracy_objective(prediction, target, normalizer, LOSS_WEIGHTS)
        mse = (prediction - target).square().mean()
        loss = accuracy + MSE_WEIGHT * mse
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite PC3D Stage S loss at update {update}")
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite PC3D Stage S gradient at update {update}")
        optimizer.step()
        scheduler.step()
        if update in DIAGNOSTIC_UPDATES:
            diagnostics = [row for row in diagnostics if int(row["update"]) != update]
            diagnostics.append(fixed_diagnostic(
                model, dataset, normalizer, diagnostic_indices, device, update))
        if update % TRACE_INTERVAL == 0 or update == updates:
            row = {"update": update, "loss": float(loss.detach()),
                   "accuracy": float(accuracy.detach()), "mse": float(mse.detach()),
                   "grad_norm_before_clip": float(grad_norm.detach()),
                   "lr": float(optimizer.param_groups[0]["lr"])}
            trace = [item for item in trace if int(item["update"]) != update]
            trace.append(row)
            print(json.dumps({"event": "pc3d_stage_s_progress", **row}, sort_keys=True),
                  flush=True)
            if checkpoint_dir is not None:
                atomic_json_save(trace, checkpoint_dir / "trace.json")
        if checkpoint_dir is not None and (update % checkpoint_interval == 0 or update == updates):
            final_checkpoint_path = checkpoint_dir / f"checkpoint_u{update:08d}.pt"
            atomic_torch_save(checkpoint_payload(
                model=model, optimizer=optimizer, scheduler=scheduler, next_update=update,
                schedule_sha256=stream_sha256, config_seal=config_seal, data=data, git=git,
                diagnostics=diagnostics, trace=trace, normalizer=normalizer,
                device=device), final_checkpoint_path)
    final_checkpoint = (None if final_checkpoint_path is None else {
        "path": str(final_checkpoint_path.resolve()),
        "sha256": file_sha256(final_checkpoint_path)})
    return {"updates": updates, "sample_cursor": updates,
            "sample_stream_sha256": stream_sha256,
            "diagnostic_indices_sha256": hashlib.sha256(diagnostic_indices.tobytes()).hexdigest(),
            "diagnostics": diagnostics, "trace": trace,
            "endpoint_kind": ENDPOINT_KIND,
            "endpoint_state_sha256": state_sha(model), "ema_used": EMA_ENABLED,
            "checkpoint_selection": "none_raw_terminal_endpoint",
            "final_checkpoint": final_checkpoint,
            "wall_seconds": time.perf_counter() - started}


def run(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if not args.execute or args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError("Stage S requires --execute and the exact protocol acknowledgement")
    git = base.git_report()
    if not git["clean"]:
        raise RuntimeError(f"Stage S formal run requires clean Git: {git['porcelain']}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production Stage S requires CUDA")
    if args.updates != UPDATES or args.batch_size != BATCH_SIZE:
        raise ValueError("Stage S freezes updates10000 and batch8")
    validate_formal_schedule(make_schedule(EXPECTED_WINDOWS))
    if args.sim_dir is None or args.output_dir is None:
        raise ValueError("Stage S requires --sim-dir and --output-dir")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    if args.resume is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to start a fresh Stage S run in a non-empty output directory")
    inventory = base.inventory(args.sim_dir)
    trajectories, windows, _window_files = base.load_fields(args.sim_dir, inventory)
    seal = data_seal(args.sim_dir, inventory, trajectories, windows)
    dataset = base.SimWindowDataset(trajectories, windows)
    normalizer = base.ChannelNormalizer(*base.window_statistics(trajectories, windows), device=device)
    model = construct_model(device)
    if parameter_count(model) != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Stage S parameter count changed: {parameter_count(model)} != {EXPECTED_PARAMETERS}")
    model.set_normalizer_statistics(normalizer.mean_input, normalizer.mean_target,
                                    normalizer.std_input, normalizer.std_target)
    contract = {**training_contract(updates=args.updates, batch_size=args.batch_size),
                "architecture": model_architecture(model), "device_type": device.type}
    config_seal = canonical_sha(contract)
    checkpoint_dir = args.output_dir / "checkpoints"
    result = train(model=model, dataset=dataset, normalizer=normalizer, device=device,
                   updates=args.updates, batch_size=args.batch_size,
                   checkpoint_dir=checkpoint_dir, checkpoint_interval=args.checkpoint_interval,
                   resume=args.resume, config_seal=config_seal, data=seal, git=git,
                   expected_schedule_sha256=SAMPLE_INDEX_SHA256)
    if result["final_checkpoint"] is None:
        raise RuntimeError("Stage S did not produce a final checkpoint")
    result = {"schema": SCHEMA, "training": contract, "config_sha256": config_seal,
              "data_seal": seal, "git_seal": git,
              "checkpoints": {"directory": str(checkpoint_dir), "ignored": True,
                              "atomic": True, "interval": args.checkpoint_interval},
              **result}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save(result, args.output_dir / "result.json")
    return result


def smoke(device: torch.device) -> dict[str, object]:
    """CPU-safe model construction/training smoke; not a production protocol run."""
    model = construct_model(device, tiny=True)
    # The official MVPE probes require the production 32x64 spatial extent.
    trajectories = [torch.randn(40, 32, 64, 3)]
    dataset = base.SimWindowDataset(trajectories, [base.SimWindow(0, 0)])
    normalizer = base.ChannelNormalizer(torch.zeros(3), torch.zeros(3),
                                        torch.ones(3), torch.ones(3), device)
    model.set_normalizer_statistics(normalizer.mean_input, normalizer.mean_target,
                                    normalizer.std_input, normalizer.std_target)
    return train(model=model, dataset=dataset, normalizer=normalizer, device=device,
                 updates=2, batch_size=1, checkpoint_interval=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "smoke", "formal"), default="contract")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", "--acknowledge-protocol", dest="acknowledge", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "contract":
        print(json.dumps(training_contract(), sort_keys=True))
        return
    device = torch.device(args.device)
    if args.mode == "formal":
        result = run(args, device)
    else:
        result = smoke(device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
