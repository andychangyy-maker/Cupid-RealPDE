#!/usr/bin/env python3
"""Train one minimal pooled-history phase adapter on the mature Real4k stream.

This is the direct responsibility ablation implied by the sealed causal-phase
audit: retain the shared per-frame spatial encoder and its temporal mean, but
remove all five learned mixer blocks.  B210, forecast conditioning, fusion,
projection, data, loss, optimizer, schedule and EMA match the prior Real4k
experiment.  The entry has no clean716 or submission surface.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "code", Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
    POOLED_HISTORY_MODE,
)
import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_causal_phase_direction_real4k as phase  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
import train_pyramid_causal3d_real_dense_aug as real4k  # noqa: E402


SCHEMA = "b210_pooled_history_phase_real4k_v1"
ENDPOINT_SCHEMA = "b210_pooled_history_phase_adapter_endpoint_v1"
SEAL_SCHEMA = "b210_pooled_history_phase_file_seal_v1"
ACKNOWLEDGEMENT = "RUN_B210_POOLED_HISTORY_PHASE_REAL4K_V1"
ARM = POOLED_HISTORY_MODE

# These are aliases, followed by import-time equality assertions below, so a
# future change to the mature comparison stream fails closed instead of
# silently creating a different recipe.
UPDATES = phase.UPDATES
BATCH_SIZE = phase.BATCH_SIZE
LR = phase.LR
WEIGHT_DECAY = phase.WEIGHT_DECAY
GRAD_CLIP = phase.GRAD_CLIP
EMA_DECAY = phase.EMA_DECAY
SCHEDULER_T_MAX = phase.SCHEDULER_T_MAX
FULL_ADAPTER_PARAMETERS = phase.EXPECTED_ADAPTER_PARAMETERS
POOLED_ADAPTER_PARAMETERS = 34_827
REMOVED_MIXER_PARAMETERS = 6_840
FULL_CONV_MAC_PER_SAMPLE = 411_648_000
POOLED_CONV_MAC_PER_SAMPLE = 346_521_600
REMOVED_MIXER_CONV_MAC_PER_SAMPLE = 65_126_400
EXPECTED_SAMPLE_SCHEDULE_SHA256 = (
    "9381a97fc114e4234a889d686158533d0088701676b683d268d12d4203f1be43")

if (UPDATES, BATCH_SIZE, LR, WEIGHT_DECAY, GRAD_CLIP,
        EMA_DECAY, SCHEDULER_T_MAX, real4k.SEED,
        real4k.OPTIMIZER_STRIDE,
        real4k.EXPECTED_OPTIMIZER_WINDOWS) != (
        4_000, 8, 2.0e-4, 1.0e-5, 1.0, 0.999, 12_000, 4, 5, 9_933):
    raise RuntimeError("mature causal-phase Real4k recipe changed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("contract", "checkpoint-smoke", "target-free-dry-run", "formal"),
        default="contract")
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--base-checkpoint-sha256")
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args(argv)


def _git_report() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ("git", *args), cwd=REPO_ROOT, check=True,
            capture_output=True, text=True).stdout.strip()

    porcelain = run("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "clean": not bool(porcelain),
        "porcelain": porcelain.splitlines(),
    }


def _state_snapshot(
        model_or_state: nn.Module | Mapping[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
    state = (model_or_state.state_dict()
             if isinstance(model_or_state, nn.Module) else model_or_state)
    return {
        str(name): value.detach().cpu().clone()
        for name, value in state.items()
    }


def build_adapter(device: torch.device) -> CausalPhaseDirectionAdapter:
    """Build the no-mixer arm from the exact mature seed-4 RNG stream."""
    torch.manual_seed(real4k.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(real4k.SEED)
    adapter = CausalPhaseDirectionAdapter(CausalPhaseDirectionConfig(
        mixer_mode=POOLED_HISTORY_MODE)).float()
    count = sum(parameter.numel() for parameter in adapter.parameters())
    if count != POOLED_ADAPTER_PARAMETERS:
        raise RuntimeError(f"pooled adapter parameter count changed: {count}")
    if any(name.startswith("temporal_mixer.") for name in adapter.state_dict()):
        raise RuntimeError("pooled adapter unexpectedly retained mixer state")
    return adapter.to(device)


def common_initialization_report() -> dict[str, object]:
    """Prove deleting the mixer did not perturb any shared initial tensor."""
    torch.manual_seed(real4k.SEED)
    full = CausalPhaseDirectionAdapter(CausalPhaseDirectionConfig(
        mixer_mode="causal_temporal")).float()
    pooled = build_adapter(torch.device("cpu"))
    common = {
        name: value for name, value in full.state_dict().items()
        if not name.startswith("temporal_mixer.")}
    pooled_state = pooled.state_dict()
    if tuple(common) != tuple(pooled_state):
        raise RuntimeError("pooled/common initial state keys differ")
    if any(not torch.equal(common[name], pooled_state[name]) for name in common):
        raise RuntimeError("pooled/common initialization is not bitwise equal")
    mixer_parameters = sum(
        parameter.numel() for parameter in full.temporal_mixer.parameters())
    if mixer_parameters != REMOVED_MIXER_PARAMETERS:
        raise RuntimeError("removed mixer parameter count changed")
    return {
        "seed": real4k.SEED,
        "common_tensors": len(common),
        "common_state_sha256": ema_lib.tensor_state_sha256(common),
        "bitwise_equal_to_causal_initialization": True,
        "removed_mixer_state_tensors": len(full.state_dict()) - len(common),
        "removed_mixer_parameters": mixer_parameters,
    }


@torch.inference_mode()
def _module_compute_profile(
        adapter: CausalPhaseDirectionAdapter) -> dict[str, object]:
    """Count convolution MACs and module dispatches at the Track-1 shape."""
    rows: list[dict[str, object]] = []
    normalization_calls = 0
    handles = []

    def convolution_hook(module, _inputs, output):
        kernel = int(np.prod(module.kernel_size))
        per_output = kernel * module.in_channels // module.groups
        rows.append({
            "type": type(module).__name__,
            "output_elements": int(output.numel()),
            "mac": int(output.numel()) * per_output,
        })

    def normalization_hook(_module, _inputs, _output):
        nonlocal normalization_calls
        normalization_calls += 1

    for module in adapter.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, nn.GroupNorm):
            handles.append(module.register_forward_hook(normalization_hook))
    try:
        torch.manual_seed(17)
        history = torch.randn(1, 20, 32, 64, 3)
        base = torch.randn(1, 20, 32, 64, 3)
        adapter(history, base)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "state_tensors": len(adapter.state_dict()),
        "conv_mac_per_sample": sum(int(row["mac"]) for row in rows),
        "conv_module_invocations": len(rows),
        "conv2d_invocations": sum(row["type"] == "Conv2d" for row in rows),
        "conv3d_invocations": sum(row["type"] == "Conv3d" for row in rows),
        "groupnorm_module_invocations": normalization_calls,
    }


def compute_report() -> dict[str, object]:
    torch.manual_seed(real4k.SEED)
    full = CausalPhaseDirectionAdapter(CausalPhaseDirectionConfig(
        mixer_mode="causal_temporal")).float().eval()
    pooled = build_adapter(torch.device("cpu")).eval()
    full_profile = _module_compute_profile(full)
    pooled_profile = _module_compute_profile(pooled)
    if (full_profile["parameters"] != FULL_ADAPTER_PARAMETERS
            or pooled_profile["parameters"] != POOLED_ADAPTER_PARAMETERS
            or full_profile["conv_mac_per_sample"] != FULL_CONV_MAC_PER_SAMPLE
            or pooled_profile["conv_mac_per_sample"] != POOLED_CONV_MAC_PER_SAMPLE):
        raise RuntimeError("adapter compute contract changed")
    return {
        "reference_causal": full_profile,
        "pooled_history_no_mixer": pooled_profile,
        "removed": {
            "parameters": REMOVED_MIXER_PARAMETERS,
            "parameter_fraction": (
                REMOVED_MIXER_PARAMETERS / FULL_ADAPTER_PARAMETERS),
            "conv_mac_per_sample": REMOVED_MIXER_CONV_MAC_PER_SAMPLE,
            "conv_mac_fraction": (
                REMOVED_MIXER_CONV_MAC_PER_SAMPLE / FULL_CONV_MAC_PER_SAMPLE),
            "conv3d_module_invocations": (
                full_profile["conv3d_invocations"]
                - pooled_profile["conv3d_invocations"]),
            "groupnorm_module_invocations": (
                full_profile["groupnorm_module_invocations"]
                - pooled_profile["groupnorm_module_invocations"]),
        },
        "launch_note": (
            "Module invocations are exact eager module-path counts, not CUDA "
            "kernel counts. The removed path also contains padding, SiLU, "
            "multiply and residual-add kernels; A800 profiling is deferred."),
        "end_to_end_note": (
            "Frozen B210 is unchanged and dominates inference, so end-to-end "
            "latency reduction will be smaller than adapter-only MAC reduction."),
    }


def experiment_contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "question": (
            "Does shared per-frame spatial history plus fixed temporal mean "
            "retain the phase gain without any learned history mixer?"),
        "single_arm": ARM,
        "architecture": {
            "history": "shared_frame_encoder_then_exact_temporal_mean",
            "learned_history_mixer": None,
            "unchanged": [
                "B210 forecast direction encoder", "horizon embedding",
                "fusion", "direction head", "gate head",
                "exact B210 temporal mean/RMS projection"],
            "parameters": POOLED_ADAPTER_PARAMETERS,
            "removed_parameters": REMOVED_MIXER_PARAMETERS,
            "conv_mac_per_sample": POOLED_CONV_MAC_PER_SAMPLE,
            "removed_conv_mac_per_sample": REMOVED_MIXER_CONV_MAC_PER_SAMPLE,
        },
        "comparison": {
            "reference_schema": phase.SCHEMA,
            "base": "same explicit SHA-256 B210 checkpoint",
            "real_data": "same clean81 stride5 9933 windows",
            "sample_schedule": "same seed-4 schedule and SHA",
            "sample_schedule_sha256": EXPECTED_SAMPLE_SCHEDULE_SHA256,
            "augmentation": dict(real4k.AUGMENTATION),
            "loss": "same physical u/v official Rel-L2 mapping",
            "optimizer": "AdamW",
            "updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "scheduler_t_max": SCHEDULER_T_MAX,
            "ema_decay": EMA_DECAY,
            "initialization": (
                "same seed and bitwise-equal common tensors; mixer tensors absent"),
        },
        "evaluation": {
            "clean716_argument": False,
            "clean716_read": False,
            "checkpoint_selection": False,
            "score_gate": None,
            "sweep": None,
            "submission_authorized": False,
        },
    }


def _normalizer_from_base_payload(
        payload: Mapping[str, object], device: torch.device):
    values = payload.get("normalizer")
    if (not isinstance(values, Mapping)
            or set(values) != {
                "mean_input", "mean_target", "std_input", "std_target"}):
        raise RuntimeError("B210 checkpoint lacks the sealed Real normalizer")
    return smu1.FoldNormalizer(tuple(
        torch.as_tensor(values[name], dtype=torch.float32) for name in (
            "mean_input", "mean_target", "std_input", "std_target")),
        device=device)


def _normalizer_payload(normalizer) -> dict[str, torch.Tensor]:
    return {
        name: getattr(normalizer, name).detach().cpu().clone()
        for name in ("mean_input", "mean_target", "std_input", "std_target")
    }


def checkpoint_smoke(args: argparse.Namespace) -> dict[str, object]:
    if args.base_checkpoint is None or args.base_checkpoint_sha256 is None:
        raise ValueError("checkpoint-smoke requires B210 path and SHA-256")
    device = torch.device(args.device)
    if device.type != "cpu":
        raise RuntimeError("checkpoint-smoke is deliberately CPU-only")
    payload, config, state, source = phase.load_base_payload(
        args.base_checkpoint, args.base_checkpoint_sha256)
    normalizer = _normalizer_from_base_payload(payload, device)
    base = phase.build_base(config, state, normalizer, device)
    adapter = build_adapter(device).eval()
    clone = CausalPhaseDirectionAdapter(adapter.config).float().eval()
    clone.load_state_dict(adapter.state_dict(), strict=True)
    history = torch.zeros((1, 20, 32, 64, 3), dtype=torch.float32)
    with torch.inference_mode():
        base_prediction = base(history)
        prediction = clone(history, base_prediction)
    if not torch.equal(prediction, base_prediction):
        raise RuntimeError("pooled update-0 is not exact B210 identity")
    return {
        "schema": "b210_pooled_history_checkpoint_smoke_v1",
        "device": "cpu",
        "base": source,
        "prediction_shape": list(prediction.shape),
        "prediction_finite": bool(torch.isfinite(prediction).all()),
        "update0_bitwise_identity": True,
        "adapter_strict_roundtrip": True,
        "initialization": common_initialization_report(),
        "compute": compute_report(),
        "raw_h5_opened": False,
        "clean716_opened": False,
    }


def _dense_history_rows(
        rows: Sequence[Mapping[str, object]],
        ) -> list[tuple[int, int]]:
    return [
        (trajectory, start)
        for trajectory, row in enumerate(rows)
        for start in range(
            0, int(row["val_boundary"]) - 40 + 1,
            real4k.OPTIMIZER_STRIDE)
    ]


def _target_free_history_panel(
        real_dir: Path, rows: Sequence[Mapping[str, object]],
        selected_indices: Sequence[int],
        ) -> tuple[torch.Tensor, list[dict[str, object]]]:
    dense = _dense_history_rows(rows)
    if len(dense) != real4k.EXPECTED_OPTIMIZER_WINDOWS:
        raise RuntimeError("clean81 dense history count changed")
    samples: list[torch.Tensor] = []
    access: list[dict[str, object]] = []
    root = real_dir.resolve()
    for index in selected_indices:
        trajectory, start = dense[int(index)]
        row = rows[trajectory]
        filename = str(row["file"])
        path = (root / filename).resolve()
        if (path.parent != root or path.name != filename
                or not path.is_file()):
            raise FileNotFoundError(f"unsafe or missing Real H5: {filename}")
        stop = start + 20
        if stop > int(row["val_boundary"]) - 20:
            raise RuntimeError("target-free history crossed future-only tail")
        with h5py.File(path, "r") as handle:
            if handle["u"].shape != handle["v"].shape:
                raise RuntimeError(f"Real u/v shape mismatch: {filename}")
            u = np.asarray(
                handle["u"][start:stop, ::2, ::2], dtype=np.float32)
            v = np.asarray(
                handle["v"][start:stop, ::2, ::2], dtype=np.float32)
        if u.shape != v.shape or int(u.shape[0]) != 20:
            raise RuntimeError(f"incomplete target-free history: {filename}")
        uv = np.stack((u, v), axis=-1)
        pressure = np.zeros((*uv.shape[:-1], 1), dtype=np.float32)
        samples.append(torch.from_numpy(np.ascontiguousarray(
            np.concatenate((uv, pressure), axis=-1))))
        access.append({
            "dense_index": int(index), "file": filename,
            "frame_slice": [start, stop], "channels": ["u", "v"],
            "pressure_opened": False, "target_returned": False,
        })
    return torch.stack(samples), access


def target_free_dry_run(args: argparse.Namespace) -> dict[str, object]:
    required = (
        args.base_checkpoint, args.base_checkpoint_sha256,
        args.real_dir, args.clean_manifest)
    if any(value is None for value in required):
        raise ValueError(
            "target-free-dry-run requires B210, clean81 manifest and Real dir")
    device = torch.device(args.device)
    if device.type != "cpu":
        raise RuntimeError("target-free-dry-run is deliberately CPU-only")
    payload, config, state, source = phase.load_base_payload(
        args.base_checkpoint, args.base_checkpoint_sha256)
    normalizer = _normalizer_from_base_payload(payload, device)
    base = phase.build_base(config, state, normalizer, device)
    rows, manifest = real4k.load_clean_manifest(args.clean_manifest)
    selected = real4k.make_sample_schedule(
        real4k.EXPECTED_OPTIMIZER_WINDOWS, 1, BATCH_SIZE)[0]
    history_physical, access = _target_free_history_panel(
        args.real_dir, rows, selected)
    history, _unused = normalizer.preprocess(
        history_physical, history_physical)
    adapter = build_adapter(device).eval()
    with torch.inference_mode():
        base_prediction = base(history)
        prediction = adapter(history, base_prediction)
        context = adapter.encode_history(history).mean(dim=1)
        reversed_context = adapter.encode_history(history.flip(1)).mean(dim=1)
    if not torch.equal(prediction, base_prediction):
        raise RuntimeError("pooled dry-run lost exact update-0 identity")
    order_delta = float((context - reversed_context).abs().max())
    if order_delta > 2.0e-6:
        raise RuntimeError("pooled history is not time-permutation invariant")
    return {
        "schema": "b210_pooled_history_target_free_dry_run_v1",
        "device": "cpu",
        "base": source,
        "manifest": manifest,
        "histories": len(selected),
        "selected_dense_indices": [int(index) for index in selected],
        "history_sha256": hashlib.sha256(
            history_physical.numpy().tobytes()).hexdigest(),
        "access": access,
        "future_target_materialized": False,
        "clean716_opened": False,
        "update0_bitwise_identity": True,
        "reverse_time_context_max_abs_delta": order_delta,
        "time_permutation_invariant": True,
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _save_endpoint(
        path: Path, payload: Mapping[str, object],
        expected_base_sha: str, expected_schedule_sha: str,
        ) -> dict[str, object]:
    real4k.atomic_torch_save(path, dict(payload))
    file_sha = phase.sha256(path)
    loaded = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if (not isinstance(loaded, Mapping)
            or loaded.get("schema") != ENDPOINT_SCHEMA
            or loaded.get("experiment_schema") != SCHEMA
            or loaded.get("arm") != ARM
            or int(loaded.get("update", -1)) != UPDATES
            or int(loaded.get("adapter_parameters", -1))
            != POOLED_ADAPTER_PARAMETERS
            or loaded.get("sample_schedule_sha256")
            != expected_schedule_sha):
        raise RuntimeError("pooled endpoint identity changed after reload")
    expected_config = build_adapter(
        torch.device("cpu")).architecture_config()
    if loaded.get("adapter_config") != expected_config:
        raise RuntimeError("pooled endpoint adapter config changed")
    source = loaded.get("base_checkpoint")
    if (not isinstance(source, Mapping)
            or source.get("sha256") != expected_base_sha
            or not isinstance(source.get("model_state_sha256"), str)):
        raise RuntimeError("pooled endpoint references the wrong B210")
    normalizer = loaded.get("normalizer")
    if (not isinstance(normalizer, Mapping)
            or set(normalizer) != {
                "mean_input", "mean_target", "std_input", "std_target"}
            or any(tuple(torch.as_tensor(value).shape) != (3,)
                       for value in normalizer.values())):
        raise RuntimeError("pooled endpoint normalizer changed")
    mechanics = loaded.get("mechanics")
    required_mechanics = (
        "initial_common_state_sha256",
        "update0_bitwise_identity", "direction_head_gradient_u1",
        "frame_history_gradient_u2", "forecast_gradient_u2",
        "direction_gate_gradient_u2", "temporal_mixer_absent")
    if (not isinstance(mechanics, Mapping)
            or mechanics.get("initial_common_state_sha256")
            != common_initialization_report()["common_state_sha256"]
            or not all(mechanics.get(name) is True
                       for name in required_mechanics[1:])
            or mechanics.get("temporal_mixer_state_keys") != []):
        raise RuntimeError("pooled endpoint mechanics evidence is incomplete")
    adapter = build_adapter(torch.device("cpu"))
    state_shas = {}
    for name in ("raw_adapter_state", "ema_adapter_state"):
        state = loaded.get(name)
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(f"pooled endpoint lacks {name}")
        if any(key.startswith("temporal_mixer.") for key in state):
            raise RuntimeError("pooled endpoint contains mixer state")
        adapter.load_state_dict(state, strict=True)
        digest = ema_lib.tensor_state_sha256(state)
        if digest != loaded.get(f"{name}_sha256"):
            raise RuntimeError(f"pooled endpoint {name} SHA changed")
        if any(value.is_floating_point() and not bool(torch.isfinite(value).all())
               for value in state.values()):
            raise FloatingPointError(f"pooled endpoint {name} is non-finite")
        state_shas[name] = digest
    ema = loaded.get("ema")
    if (not isinstance(ema, Mapping)
            or int(ema.get("updates", -1)) != UPDATES
            or ema.get("raw_model_state_sha256") != state_shas["raw_adapter_state"]
            or ema.get("materialized_model_state_sha256")
            != state_shas["ema_adapter_state"]):
        raise RuntimeError("pooled endpoint EMA seal changed")
    ema_metadata = dict(ema)
    stored_ema_metadata_sha = ema_metadata.pop("seal_metadata_sha256", None)
    if stored_ema_metadata_sha != ema_lib.canonical_sha256(ema_metadata):
        raise RuntimeError("pooled endpoint EMA metadata seal changed")
    seal = {
        "schema": SEAL_SCHEMA,
        "endpoint_schema": ENDPOINT_SCHEMA,
        "experiment_schema": SCHEMA,
        "arm": ARM,
        "update": UPDATES,
        "path": str(path.resolve()),
        "file_sha256": file_sha,
        "state_sha256": state_shas,
        "base_checkpoint_sha256": expected_base_sha,
        "sample_schedule_sha256": expected_schedule_sha,
        "strict_load": True,
        "finite": True,
        "temporal_mixer_absent": True,
    }
    seal["seal_metadata_sha256"] = real4k.canonical_sha(seal)
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    _atomic_json(seal_path, seal)
    if phase.sha256(path) != file_sha:
        raise RuntimeError("pooled endpoint changed while publishing seal")
    return {
        **seal,
        "seal_path": str(seal_path.resolve()),
        "seal_file_sha256": phase.sha256(seal_path),
    }


def _train(
        *, base: nn.Module, adapter: CausalPhaseDirectionAdapter,
        dataset, normalizer, schedule: Sequence[Sequence[int]],
        source: Mapping[str, object], output_dir: Path,
        device: torch.device,
        ) -> dict[str, object]:
    if (len(dataset) != real4k.EXPECTED_OPTIMIZER_WINDOWS
            or len(schedule) != UPDATES):
        raise RuntimeError("pooled Real4k dataset/schedule length changed")
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX)
    ema = ema_lib.FP32EMAShadow(adapter, EMA_DECAY)
    initial_state = _state_snapshot(adapter)
    initial_state_sha = ema_lib.tensor_state_sha256(initial_state)
    if any(name.startswith("temporal_mixer.") for name in initial_state):
        raise RuntimeError("pooled initial state contains mixer tensors")
    base.eval().requires_grad_(False)
    adapter.train()
    real4k.seed_real_recipe(device)
    trace = []
    identity_u0 = False
    direction_gradient_u1 = False
    history_gradient_u2 = False
    forecast_gradient_u2 = False
    gate_gradient_u2 = False
    started = time.perf_counter()
    for cursor, indices in enumerate(schedule):
        update = cursor + 1
        history, target = zip(*(dataset[index] for index in indices))
        history_physical = torch.stack(history).to(device)
        target_physical = torch.stack(target).to(device)
        adapter_history = real4k.augment_real_history(
            history_physical, normalizer)
        base_history, target_normalized = normalizer.preprocess(
            history_physical, target_physical)
        with torch.no_grad():
            base_prediction = base(base_history).detach()
        optimizer.zero_grad(set_to_none=True)
        components = adapter.forward_components(
            adapter_history, base_prediction)
        if update == 1:
            identity_u0 = torch.equal(
                components.prediction, components.base_prediction)
            if not identity_u0:
                raise RuntimeError("pooled update-0 is not exact B210 identity")
        loss, details = phase.phase_objective(
            components.prediction, target_normalized, normalizer)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite pooled loss at u{update}")
        loss.backward()
        if update == 1:
            gradient = adapter.direction_head.weight.grad
            direction_gradient_u1 = bool(
                gradient is not None and torch.isfinite(gradient).all()
                and torch.count_nonzero(gradient))
            if not direction_gradient_u1:
                raise RuntimeError("pooled direction head lacks u1 gradient")
        if update == 2:
            def live(parameter: torch.Tensor) -> bool:
                gradient = parameter.grad
                return bool(
                    gradient is not None and torch.isfinite(gradient).all()
                    and torch.count_nonzero(gradient))

            history_gradient_u2 = live(adapter.frame_encoder.stem.weight)
            forecast_gradient_u2 = live(adapter.forecast_encoder.stem.weight)
            gate_gradient_u2 = live(adapter.gate_head.weight)
            if not all((history_gradient_u2, forecast_gradient_u2,
                        gate_gradient_u2)):
                raise RuntimeError("pooled common paths lack u2 gradients")
        gradient_norm = nn.utils.clip_grad_norm_(
            adapter.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite pooled gradient at u{update}")
        optimizer.step()
        ema.update_after_optimizer_step(adapter, optimizer_step=update)
        scheduler.step()

        if update in (1, 2, 500, 1_000, 2_000, 3_000, UPDATES):
            base_uv = components.base_prediction[..., :2].float()
            output_uv = components.prediction[..., :2].float()
            base_mean = base_uv.mean(dim=1, keepdim=True)
            output_mean = output_uv.mean(dim=1, keepdim=True)
            base_rms = (base_uv - base_mean).square().mean(
                dim=1, keepdim=True).sqrt()
            output_rms = (output_uv - output_mean).square().mean(
                dim=1, keepdim=True).sqrt()
            row = {
                "update": update,
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "max_temporal_mean_delta": float(
                    (output_mean - base_mean).abs().max()),
                "max_temporal_rms_delta": float(
                    (output_rms - base_rms).abs().max()),
                "direction_gate_mean": float(
                    components.direction_gate.detach().mean()),
                "direction_gate_min": float(
                    components.direction_gate.detach().amin()),
                "direction_gate_max": float(
                    components.direction_gate.detach().amax()),
                **{key: float(value) for key, value in details.items()},
            }
            trace.append(row)
            print(json.dumps({
                "event": "b210_pooled_history_real4k_progress", **row,
            }, sort_keys=True), flush=True)
    if ema.updates != UPDATES:
        raise RuntimeError("pooled EMA update count changed")

    raw_state = _state_snapshot(adapter)
    ema_state = ema.materialize_model_state(adapter)
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "experiment_schema": SCHEMA,
        "arm": ARM,
        "endpoint": "fixed_raw_and_ema_u4000",
        "update": UPDATES,
        "adapter_config": adapter.architecture_config(),
        "adapter_parameters": POOLED_ADAPTER_PARAMETERS,
        "raw_adapter_state": raw_state,
        "raw_adapter_state_sha256": ema_lib.tensor_state_sha256(raw_state),
        "ema_adapter_state": ema_state,
        "ema_adapter_state_sha256": ema_lib.tensor_state_sha256(ema_state),
        "ema": {key: value for key, value in ema.seal(adapter).items()
                if key != "model_state"},
        "normalizer": _normalizer_payload(normalizer),
        "base_checkpoint": dict(source),
        "sample_schedule_sha256": real4k.schedule_sha(schedule),
        "training": {
            "updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "seed": real4k.SEED,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": SCHEDULER_T_MAX,
            "ema_decay": EMA_DECAY,
            "objective": "official_projected_rel_score_loss",
            "augmentation": dict(real4k.AUGMENTATION),
            "trace": trace,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "mechanics": {
            "initial_common_state_sha256": initial_state_sha,
            "update0_bitwise_identity": identity_u0,
            "direction_head_gradient_u1": direction_gradient_u1,
            "frame_history_gradient_u2": history_gradient_u2,
            "forecast_gradient_u2": forecast_gradient_u2,
            "direction_gate_gradient_u2": gate_gradient_u2,
            "temporal_mixer_absent": True,
            "temporal_mixer_state_keys": [],
        },
        "clean716_opened": False,
        "checkpoint_selected": False,
        "score_gate": None,
        "submission_authorized": False,
    }
    path = output_dir / "pooled_history_no_mixer_adapter_raw_ema_u4000.pt"
    seal = _save_endpoint(
        path, payload, str(source["sha256"]),
        str(payload["sample_schedule_sha256"]))
    return {
        "endpoint": seal,
        "raw_adapter_state_sha256": payload["raw_adapter_state_sha256"],
        "ema_adapter_state_sha256": payload["ema_adapter_state_sha256"],
        "mechanics": payload["mechanics"],
        "trace": trace,
    }


def run_formal(args: argparse.Namespace) -> dict[str, object]:
    if (not args.execute
            or args.acknowledge_protocol != ACKNOWLEDGEMENT):
        raise PermissionError(
            "formal pooled Real4k requires --execute and exact acknowledgement")
    required = (
        args.base_checkpoint, args.base_checkpoint_sha256,
        args.real_dir, args.clean_manifest, args.output_dir)
    if any(value is None for value in required):
        raise ValueError("formal pooled Real4k requires all paths and B210 SHA")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal pooled Real4k requires CUDA")
    git = _git_report()
    if not git["clean"]:
        raise RuntimeError(f"formal pooled Real4k requires clean Git: {git['porcelain']}")

    payload, config, base_state, source = phase.load_base_payload(
        args.base_checkpoint, args.base_checkpoint_sha256)
    rows, manifest = real4k.load_clean_manifest(args.clean_manifest)
    access = real4k.PrefixAccessAudit()
    trajectories, sparse_windows = real4k.load_real_rows(
        args.real_dir, rows, phase="fit", audit=access)
    normalizer, normalizer_report = real4k.normalizer_from_sparse_windows(
        trajectories, sparse_windows, device)
    windows = real4k.dense_windows(trajectories)
    if len(windows) != real4k.EXPECTED_OPTIMIZER_WINDOWS:
        raise RuntimeError("pooled Real4k window count changed")
    dataset = real4k.RealWindowDataset(trajectories, windows)
    schedule = real4k.make_sample_schedule(len(dataset), UPDATES, BATCH_SIZE)
    schedule_sha = real4k.schedule_sha(schedule)
    if schedule_sha != EXPECTED_SAMPLE_SCHEDULE_SHA256:
        raise RuntimeError("pooled Real4k sample schedule changed")
    source.update({
        "clean_manifest_semantic_sha256": manifest["semantic_sha256"],
        "clean_manifest_file_sha256": phase.sha256(args.clean_manifest),
        "normalizer_sha256": real4k.tensor_state_sha(
            _normalizer_payload(normalizer)),
        "optimizer_windows": len(dataset),
        "optimizer_stride": real4k.OPTIMIZER_STRIDE,
    })
    base = phase.build_base(config, base_state, normalizer, device)
    del payload, base_state
    gc.collect()
    adapter = build_adapter(device)
    initialization = common_initialization_report()
    if (initialization["common_state_sha256"]
            != ema_lib.tensor_state_sha256(adapter.state_dict())):
        raise RuntimeError("formal pooled initialization differs from CPU seal")

    args.output_dir.mkdir(parents=True)
    arm = _train(
        base=base, adapter=adapter, dataset=dataset,
        normalizer=normalizer, schedule=schedule, source=source,
        output_dir=args.output_dir, device=device)
    result = {
        "schema": SCHEMA,
        "mode": "formal",
        "contract": experiment_contract(),
        "git": git,
        "base": source,
        "manifest": manifest,
        "normalizer": normalizer_report,
        "real_prefix_access": access.events,
        "sample_schedule_sha256": schedule_sha,
        "initialization": initialization,
        "compute": compute_report(),
        "arm": arm,
        "clean716_opened": False,
        "checkpoint_selection": False,
        "score_gate": None,
        "submission_authorized": False,
    }
    _atomic_json(args.output_dir / "result.json", result)
    del base, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "contract":
        return experiment_contract()
    if args.mode == "checkpoint-smoke":
        return checkpoint_smoke(args)
    if args.mode == "target-free-dry-run":
        return target_free_dry_run(args)
    return run_formal(args)


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
