#!/usr/bin/env python3
"""Two-stage target-free phase distillation followed by matched Real4k.

Stage 1 visits every released Sim4800 and clean81 dense-stride5 history once.
It never materializes a future ground-truth tensor: frozen B210 U3200 and the
fixed seed-4/17 causal-ST EMA weight soup provide the sole direction target.
Stage 2 compares a direct zero-start adapter with the Stage-1 initialization
on one identical Real4k stream.  This entry never exposes clean716 arguments,
does not select checkpoints, and cannot build a submission package.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
        REPO_ROOT / "code", REPO_ROOT / "code" / "submissions",
        Path(__file__).parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
)
from models.spatiotemporal_b210 import (  # noqa: E402
    SpatiotemporalB210,
    SpatiotemporalB210Config,
)
import build_track1_b210_st_phase_teacher_probe as teacher_build  # noqa: E402
import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_causal_phase_direction_real4k as direct_phase  # noqa: E402
import train_b210_unified_temporal_state_sim_discovery as sim_data  # noqa: E402
import train_pyramid_causal3d_real_dense_aug as real4k  # noqa: E402


SCHEMA = "b210_causal_phase_direction_two_stage_kd_v1"
STAGE1_ENDPOINT_SCHEMA = "b210_causal_phase_kd_raw_endpoint_v1"
STAGE2_ENDPOINT_SCHEMA = "b210_causal_phase_kd_real4k_endpoint_v1"
SEAL_SCHEMA = "b210_causal_phase_adapter_file_seal_v1"

STAGE1_ACKNOWLEDGEMENT = "RUN_B210_CAUSAL_PHASE_KD_STAGE1_U1842_V1"
STAGE2_ACKNOWLEDGEMENT = "RUN_B210_CAUSAL_PHASE_KD_REAL4K_PAIR_V1"

SIM_HISTORIES = 4_800
REAL_HISTORIES = 9_933
STAGE1_HISTORIES = SIM_HISTORIES + REAL_HISTORIES
BATCH_SIZE = 8
STAGE1_UPDATES = math.ceil(STAGE1_HISTORIES / BATCH_SIZE)
STAGE2_UPDATES = 4_000
SEED = 4
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
SCHEDULER_T_MAX = 12_000
EMA_DECAY = 0.999
TEACHER_WEIGHT = 0.5
EXPECTED_ADAPTER_PARAMETERS = direct_phase.EXPECTED_ADAPTER_PARAMETERS
STAGE1_PERMUTATION_SHA256 = (
    "05a048a2079e71c6dc93f26c0b7c99d456d53424c697f8b24ae95b0fba4fb183")
STAGE2_ARMS = ("direct_zero_start_control", "kd_init_candidate")
TRACE_UPDATES_STAGE1 = (1, 2, 500, 1_000, 1_500, STAGE1_UPDATES)
TRACE_UPDATES_STAGE2 = (1, 2, 500, 1_000, 2_000, 3_000, STAGE2_UPDATES)
AUGMENTATION_PROBE_UPDATES = (1, 2, STAGE2_UPDATES)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("contract", "inventory", "mechanics-smoke", "stage1", "stage2"),
        default="contract")
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--base-checkpoint-sha256")
    parser.add_argument("--teacher-seed4-checkpoint", type=Path)
    parser.add_argument("--teacher-seed4-checkpoint-sha256")
    parser.add_argument("--teacher-seed17-checkpoint", type=Path)
    parser.add_argument("--teacher-seed17-checkpoint-sha256")
    parser.add_argument("--sim-source-checkpoint", type=Path)
    parser.add_argument("--sim-source-checkpoint-sha256")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--stage1-endpoint", type=Path)
    parser.add_argument("--stage1-endpoint-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args(argv)


def experiment_contract() -> dict[str, object]:
    adapter = CausalPhaseDirectionConfig(mixer_mode="causal_temporal")
    return {
        "schema": SCHEMA,
        "adapter": {
            "architecture": "causal_phase_direction_adapter",
            "config": asdict(adapter),
            "parameters": EXPECTED_ADAPTER_PARAMETERS,
            "learned_responsibility": "future_uv_temporal_direction_only",
            "projection": "restore_B210_temporal_mean_and_rms",
            "pressure": "zero_for_all_Stage1_inputs; not_in_loss",
        },
        "teacher": {
            "base": "B210 full-prefix native-UQ terminal u3200, frozen FP32",
            "st": (
                "causal ST Real12k seed4/17 dual_adapt_control EMA; collapse "
                "identical banks then fixed 0.5/0.5 FP32 weight soup"),
            "target": "restore_mean_rms(0.5*B210 + 0.5*ST_soup, B210)",
            "normalizer": "one bitwise-equal Real normalizer for B210 and ST",
            "metadata_or_target_routing": False,
        },
        "stage1": {
            "data": {
                "sim": {
                    "histories": SIM_HISTORIES,
                    "stride": 20,
                    "transform": (
                        "Sim input z-score -> Real input mean/std; pressure=0"),
                },
                "real": {
                    "histories": REAL_HISTORIES,
                    "stride": 5,
                    "scope": "clean81 optimizer-visible prefixes",
                },
                "total_unique_histories": STAGE1_HISTORIES,
                "passes": 1,
                "future_ground_truth_materialized": False,
            },
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "updates": STAGE1_UPDATES,
            "last_batch_size": STAGE1_HISTORIES % BATCH_SIZE,
            "permutation_sha256": STAGE1_PERMUTATION_SHA256,
            "augmentation": None,
            "loss": "physical_uv_Rel-L2_then_official_1-1/(1+0.5*mean_error)",
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "scheduler": f"CosineAnnealingLR(T_max={SCHEDULER_T_MAX})",
            "ema": None,
            "endpoint": f"raw_u{STAGE1_UPDATES}",
        },
        "stage2": {
            "arms": {
                STAGE2_ARMS[0]: "fresh exact-zero-update causal adapter",
                STAGE2_ARMS[1]: f"strict-loaded Stage1 raw_u{STAGE1_UPDATES}",
            },
            "only_arm_difference": "adapter_initial_state",
            "data": "same clean81 stride5 9933-window Real stream",
            "augmentation": dict(real4k.AUGMENTATION),
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "updates_each_arm": STAGE2_UPDATES,
            "loss": "same physical_uv official Rel-L2 mapping as Stage1",
            "optimizer": "fresh AdamW for each arm",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "scheduler": f"CosineAnnealingLR(T_max={SCHEDULER_T_MAX})",
            "ema": {
                "decay": EMA_DECAY,
                "initialized_from": "each arm's own Real update-0 state",
            },
            "fixed_report": ["raw_u4000", "ema_u4000"],
        },
        "audit": {
            "stage1_target_free_access": True,
            "order_permutation_mechanics": True,
            "atomic_endpoint_then_file_and_tensor_hash_seal": True,
            "clean716_cli_argument": False,
            "clean716_read": False,
            "endpoint_selection": False,
            "score_gate": None,
            "sweep": None,
            "submission_authorized": False,
        },
    }


def _canonical_sha(value: object) -> str:
    return real4k.canonical_sha(value)


def _state_snapshot(
        model_or_state: nn.Module | Mapping[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
    state = (model_or_state.state_dict()
             if isinstance(model_or_state, nn.Module) else model_or_state)
    return {
        str(name): value.detach().cpu().clone()
        for name, value in state.items()
    }


def _normalizer_tensors(values: Any) -> dict[str, torch.Tensor]:
    names = ("mean_input", "mean_target", "std_input", "std_target")
    if isinstance(values, Mapping):
        result = {name: torch.as_tensor(values[name]).float().cpu().clone()
                  for name in names}
    else:
        result = {name: torch.as_tensor(getattr(values, name)).float().cpu().clone()
                  for name in names}
    if any(tuple(value.shape) != (3,) for value in result.values()):
        raise ValueError("normalizer must contain four float [3] tensors")
    if (torch.any(result["std_input"] <= 0)
            or torch.any(result["std_target"] <= 0)):
        raise ValueError("normalizer standard deviations must be positive")
    return result


def _normalizer_sha(values: Any) -> str:
    return sim_data.state_sha(_normalizer_tensors(values))


def _normalizers_equal(left: Any, right: Any) -> bool:
    a, b = _normalizer_tensors(left), _normalizer_tensors(right)
    return all(torch.equal(a[name], b[name]) for name in a)


def _normalizer_from_payload(
        values: Mapping[str, object], device: torch.device):
    tensors = _normalizer_tensors(values)
    return sim_data.ChannelNormalizer(
        tensors["mean_input"], tensors["mean_target"],
        tensors["std_input"], tensors["std_target"], device)


def _normalizer_report(values: Any) -> dict[str, object]:
    tensors = _normalizer_tensors(values)
    return {
        "sha256": _normalizer_sha(tensors),
        **{name: value.tolist() for name, value in tensors.items()},
    }


def _require_file_sha(path: Path, expected: str | None, role: str) -> str:
    if expected is None:
        raise ValueError(f"{role} requires an explicit SHA-256")
    expected = str(expected).strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"{role} SHA-256 must be 64 lowercase hex digits")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = direct_phase.sha256(path)
    if actual != expected:
        raise RuntimeError(f"{role} changed: {actual} != {expected}")
    return actual


def _to_physical_target(value: torch.Tensor, normalizer) -> torch.Tensor:
    return value * normalizer.std_target + normalizer.mean_target


def _normalize_input(value: torch.Tensor, normalizer) -> torch.Tensor:
    return (value - normalizer.mean_input) / normalizer.std_input


def restore_mean_rms(
        candidate: torch.Tensor, reference: torch.Tensor,
        epsilon: float = 1.0e-12) -> torch.Tensor:
    """Use candidate direction with reference temporal mean/RMS."""
    if (candidate.shape != reference.shape or candidate.ndim != 5
            or candidate.shape[-1] != 2):
        raise ValueError("projection needs aligned [B,T,H,W,2] tensors")
    candidate, reference = candidate.float(), reference.float()
    reference_mean = reference.mean(dim=1, keepdim=True)
    reference_fluctuation = reference - reference_mean
    candidate_fluctuation = candidate - candidate.mean(dim=1, keepdim=True)
    reference_rms = reference_fluctuation.square().mean(
        dim=1, keepdim=True).sqrt()
    candidate_rms = candidate_fluctuation.square().mean(
        dim=1, keepdim=True).sqrt()
    valid = candidate_rms > epsilon
    projected = reference_mean + candidate_fluctuation * (
        reference_rms / candidate_rms.clamp_min(epsilon))
    return torch.where(valid, projected, reference)


def teacher_target(
        base_prediction_normalized: torch.Tensor,
        st_prediction_normalized: torch.Tensor,
        normalizer) -> torch.Tensor:
    """Build the sole Stage-1 pseudo-label in physical u/v space."""
    if base_prediction_normalized.shape != st_prediction_normalized.shape:
        raise ValueError("B210/ST teacher predictions are not aligned")
    base = _to_physical_target(base_prediction_normalized, normalizer)[..., :2]
    st = _to_physical_target(st_prediction_normalized, normalizer)[..., :2]
    return restore_mean_rms(
        TEACHER_WEIGHT * base + TEACHER_WEIGHT * st, base)


def map_sim_to_real_input(
        history: torch.Tensor, domains: torch.Tensor,
        sim_normalizer, real_normalizer) -> torch.Tensor:
    """Map Sim u/v z-scores onto Real physical input moments; force p=0."""
    if history.ndim != 5 or history.shape[-1] != 3:
        raise ValueError("history must be [B,T,H,W,3]")
    domains = torch.as_tensor(domains, device=history.device)
    if tuple(domains.shape) != (history.shape[0],):
        raise ValueError("one domain label is required per history")
    if torch.any((domains != 0) & (domains != 1)):
        raise ValueError("domain labels must be 0=Sim or 1=Real")
    result = history.clone()
    sim_mask = domains == 0
    if bool(sim_mask.any()):
        sim_uv = history[sim_mask, ..., :2]
        z = ((sim_uv - sim_normalizer.mean_input[:2])
             / sim_normalizer.std_input[:2])
        result[sim_mask, ..., :2] = (
            z * real_normalizer.std_input[:2]
            + real_normalizer.mean_input[:2])
    result[..., 2] = 0.0
    return result


@dataclass(frozen=True)
class HistoryWindow:
    domain: str
    trajectory: int
    start: int


class TargetFreeAccessAudit:
    """Record only Stage-1 history reads and reject target-like access."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self._files: set[tuple[str, str]] = set()
        self.future_tensor_requests = 0

    def record_history_open(
            self, *, domain: str, filename: str, channels: Sequence[str],
            stop: int, allowed_history_stop: int,
            pressure_opened: bool) -> None:
        if domain not in ("sim", "real"):
            raise ValueError("target-free audit supports only Sim/Real histories")
        identity = (domain, filename)
        if identity in self._files:
            raise RuntimeError(f"Stage1 history H5 opened twice: {identity}")
        if stop != allowed_history_stop or stop <= 0:
            raise RuntimeError("history loader attempted an unapproved frame range")
        if domain == "real" and tuple(channels) != ("u", "v"):
            raise RuntimeError("Real Stage1 may open only measured u/v history")
        if domain == "sim" and tuple(channels) != ("u", "v", "p"):
            raise RuntimeError("Sim Stage1 history channel contract changed")
        self._files.add(identity)
        self.events.append({
            "domain": domain,
            "file": filename,
            "purpose": "history_input_only",
            "channels_opened": list(channels),
            "pressure_opened": bool(pressure_opened),
            "frame_slice": [0, int(stop)],
            "future_ground_truth_returned": False,
        })

    def record_future_tensor_request(self) -> None:
        self.future_tensor_requests += 1
        raise RuntimeError("Stage1 is target-free; future tensor access denied")

    def report(self) -> dict[str, object]:
        return {
            "schema": "b210_phase_kd_target_free_access_v1",
            "events": list(self.events),
            "opened_files": len(self._files),
            "future_tensor_requests": self.future_tensor_requests,
            "future_ground_truth_materialized": False,
            "clean716_path_argument_exposed": False,
            "clean716_opened": False,
        }


class TargetFreeHistoryDataset:
    """Return exactly one observed history and a domain bit, never a target."""

    def __init__(
            self, sim_trajectories: Sequence[torch.Tensor],
            real_trajectories: Sequence[torch.Tensor],
            windows: Sequence[HistoryWindow]):
        self.sim_trajectories = list(sim_trajectories)
        self.real_trajectories = list(real_trajectories)
        self.windows = list(windows)
        identities = {(w.domain, w.trajectory, w.start) for w in self.windows}
        if len(identities) != len(self.windows):
            raise ValueError("Stage1 history inventory contains duplicates")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        window = self.windows[int(index)]
        trajectories = (self.sim_trajectories if window.domain == "sim"
                        else self.real_trajectories)
        history = trajectories[window.trajectory][
            window.start:window.start + 20]
        if len(history) != 20:
            raise RuntimeError("history-only materialization is incomplete")
        if window.domain == "real":
            if history.shape[-1] != 2:
                raise RuntimeError("Real history must contain measured u/v only")
            history = torch.cat((
                history,
                torch.zeros((*history.shape[:-1], 1), dtype=history.dtype),
            ), dim=-1)
            domain = 1
        else:
            if history.shape[-1] != 3:
                raise RuntimeError("Sim history must contain u/v/p")
            domain = 0
        return history, domain

    def target(self, _index: int) -> torch.Tensor:
        raise RuntimeError("Stage1 target access is forbidden by contract")


def _safe_h5_path(root: Path, filename: str) -> Path:
    root = root.resolve()
    path = (root / filename).resolve()
    if path.parent != root or path.name != filename or not path.is_file():
        raise FileNotFoundError(f"unsafe or missing H5: {filename}")
    return path


def load_sim_history_prefixes(
        sim_dir: Path, audit: TargetFreeAccessAudit,
        ) -> tuple[list[torch.Tensor], list[HistoryWindow], dict[str, object]]:
    """Read the union of Sim frames used as histories, never future-only tail."""
    rows = sim_data.inventory(sim_dir)
    trajectories: list[torch.Tensor] = []
    windows: list[HistoryWindow] = []
    inventory_rows: list[dict[str, object]] = []
    for trajectory, row in enumerate(rows):
        filename = str(row["file"])
        frames = int(row["frames"])
        starts = tuple(range(0, frames - 40, 20))
        if starts != tuple(range(0, 960, 20)):
            raise RuntimeError(f"unexpected Sim history starts for {filename}")
        history_stop = starts[-1] + 20
        path = _safe_h5_path(sim_dir, filename)
        with h5py.File(path, "r") as handle:
            arrays = [
                np.asarray(handle[key][:history_stop, ::2, ::2], dtype=np.float32)
                for key in ("u", "v", "p")
            ]
        if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
            raise RuntimeError(f"Sim history channel shapes changed: {filename}")
        values = torch.from_numpy(np.ascontiguousarray(
            np.stack(arrays, axis=-1)))
        trajectories.append(values)
        windows.extend(
            HistoryWindow("sim", trajectory, start) for start in starts)
        audit.record_history_open(
            domain="sim", filename=filename, channels=("u", "v", "p"),
            stop=history_stop, allowed_history_stop=history_stop,
            pressure_opened=True)
        inventory_rows.append({
            **dict(row), "history_starts": list(starts),
            "max_history_frame_exclusive": history_stop,
        })
    if len(trajectories) != 100 or len(windows) != SIM_HISTORIES:
        raise RuntimeError("released Sim100/4800 history contract changed")
    return trajectories, windows, {
        "domain": "released_Sim100",
        "trajectories": len(trajectories),
        "histories": len(windows),
        "stride": 20,
        "history_steps": 20,
        "future_ground_truth_materialized": False,
        "inventory_sha256": _canonical_sha(inventory_rows),
        "window_sha256": _canonical_sha([
            {"trajectory": w.trajectory, "start": w.start} for w in windows]),
    }


def load_real_history_prefixes(
        real_dir: Path, rows: Sequence[Mapping[str, object]],
        audit: TargetFreeAccessAudit,
        ) -> tuple[list[torch.Tensor], list[HistoryWindow], dict[str, object]]:
    """Read only frames that occur in a clean81 stride-5 observed history."""
    trajectories: list[torch.Tensor] = []
    windows: list[HistoryWindow] = []
    access_rows: list[dict[str, object]] = []
    for trajectory, row in enumerate(rows):
        filename = str(row["file"])
        if filename == real4k.EXCLUDED_BASENAME:
            raise RuntimeError("excluded Real trajectory reached Stage1 loader")
        boundary = int(row["val_boundary"])
        starts = tuple(range(0, boundary - 40 + 1, real4k.OPTIMIZER_STRIDE))
        if not starts:
            raise RuntimeError(f"Real prefix has no dense histories: {filename}")
        history_stop = starts[-1] + 20
        if history_stop > boundary - 20:
            raise RuntimeError("Stage1 Real read crossed into future-only tail")
        path = _safe_h5_path(real_dir, filename)
        with h5py.File(path, "r") as handle:
            if tuple(handle["u"].shape) != tuple(handle["v"].shape):
                raise RuntimeError(f"Real u/v shape mismatch: {filename}")
            arrays = [
                np.asarray(handle[key][:history_stop, ::2, ::2], dtype=np.float32)
                for key in ("u", "v")
            ]
        if arrays[0].shape != arrays[1].shape:
            raise RuntimeError(f"Real history channel shapes changed: {filename}")
        trajectories.append(torch.from_numpy(np.ascontiguousarray(
            np.stack(arrays, axis=-1))))
        windows.extend(
            HistoryWindow("real", trajectory, start) for start in starts)
        audit.record_history_open(
            domain="real", filename=filename, channels=("u", "v"),
            stop=history_stop, allowed_history_stop=history_stop,
            pressure_opened=False)
        access_rows.append({
            "file": filename, "val_boundary": boundary,
            "histories": len(starts), "history_starts": list(starts),
            "max_history_frame_exclusive": history_stop,
        })
    if len(trajectories) != 81 or len(windows) != REAL_HISTORIES:
        raise RuntimeError("clean81 stride5/9933 history contract changed")
    return trajectories, windows, {
        "domain": "clean81_optimizer_visible_prefixes",
        "trajectories": len(trajectories),
        "histories": len(windows),
        "stride": real4k.OPTIMIZER_STRIDE,
        "history_steps": 20,
        "future_ground_truth_materialized": False,
        "access_sha256": _canonical_sha(access_rows),
        "window_sha256": _canonical_sha([
            {"trajectory": w.trajectory, "start": w.start} for w in windows]),
    }


def make_stage1_schedule(
        samples: int = STAGE1_HISTORIES,
        ) -> tuple[list[list[int]], dict[str, object]]:
    """One seed-4 permutation: every combined-domain history exactly once."""
    if samples <= 0:
        raise ValueError("Stage1 sample count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    permutation = torch.randperm(
        samples, generator=generator, dtype=torch.int64).numpy()
    digest = hashlib.sha256(
        np.ascontiguousarray(permutation).tobytes()).hexdigest()
    batches = [
        permutation[start:start + BATCH_SIZE].tolist()
        for start in range(0, samples, BATCH_SIZE)
    ]
    unique = np.unique(permutation)
    report = {
        "seed": SEED,
        "samples": samples,
        "updates": len(batches),
        "batch_size": BATCH_SIZE,
        "last_batch_size": len(batches[-1]),
        "permutation_sha256": digest,
        "bijection": (
            len(unique) == samples
            and int(unique[0]) == 0 and int(unique[-1]) == samples - 1),
        "duplicates": int(samples - len(unique)),
        "missing": int(samples - len(unique)),
        "identity_order": bool(np.array_equal(
            permutation, np.arange(samples, dtype=np.int64))),
        "inverse_roundtrip": bool(np.array_equal(
            permutation[np.argsort(permutation)],
            np.arange(samples, dtype=np.int64))),
    }
    if samples == STAGE1_HISTORIES:
        if digest != STAGE1_PERMUTATION_SHA256:
            raise RuntimeError("formal Stage1 seed-4 permutation changed")
        if len(batches) != STAGE1_UPDATES or len(batches[-1]) != 5:
            raise RuntimeError("formal Stage1 batch partition changed")
    if not report["bijection"] or report["identity_order"]:
        raise RuntimeError("Stage1 order is not a nontrivial permutation")
    return batches, report


def _tensor_sha(value: torch.Tensor) -> str:
    return ema_lib.tensor_state_sha256({"tensor": value.detach().cpu()})


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


def _adapter_from_config(values: Mapping[str, object]) -> CausalPhaseDirectionAdapter:
    raw = dict(values)
    if raw.pop("architecture", None) != "causal_phase_direction_adapter":
        raise ValueError("endpoint has the wrong adapter architecture")
    for key in (
            "base_responsibility", "adapter_responsibility",
            "pressure_passthrough", "flattened_history_bypass"):
        raw.pop(key, None)
    config = CausalPhaseDirectionConfig(**raw)
    if config.mixer_mode != "causal_temporal":
        raise ValueError("KD endpoint is not a causal temporal adapter")
    return CausalPhaseDirectionAdapter(config).float()


def write_endpoint_and_seal(
        path: Path, payload: Mapping[str, object],
        state_fields: Sequence[str]) -> dict[str, object]:
    """Atomic write, independent reload, strict-load, tensor/file sidecar."""
    real4k.atomic_torch_save(path, dict(payload))
    file_sha = direct_phase.sha256(path)
    loaded = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(loaded, Mapping):
        raise TypeError("adapter endpoint did not reload as a mapping")
    model = _adapter_from_config(loaded["adapter_config"])
    state_shas: dict[str, str] = {}
    for field in state_fields:
        state = loaded.get(field)
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(f"endpoint lacks {field}")
        model.load_state_dict(state, strict=True)
        if any(value.is_floating_point() and not bool(torch.isfinite(value).all())
               for value in state.values()):
            raise FloatingPointError(f"non-finite tensor in {field}")
        digest = ema_lib.tensor_state_sha256(state)
        declared = loaded.get(f"{field}_sha256")
        if digest != declared:
            raise RuntimeError(f"{field} tensor SHA mismatch")
        state_shas[field] = digest
    seal = {
        "schema": SEAL_SCHEMA,
        "endpoint_schema": loaded.get("schema"),
        "experiment_schema": loaded.get("experiment_schema"),
        "stage": loaded.get("stage"),
        "arm": loaded.get("arm"),
        "update": loaded.get("update"),
        "path": str(path.resolve()),
        "file_sha256": file_sha,
        "state_sha256": state_shas,
        "strict_load": True,
        "finite": True,
    }
    seal["seal_metadata_sha256"] = _canonical_sha(seal)
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    _atomic_json(seal_path, seal)
    if direct_phase.sha256(path) != file_sha:
        raise RuntimeError("endpoint changed while publishing its seal")
    return {**seal, "seal_path": str(seal_path.resolve()),
            "seal_file_sha256": direct_phase.sha256(seal_path)}


def verify_companion_seal(path: Path, expected_file_sha: str) -> dict[str, object]:
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    if not seal_path.is_file():
        raise FileNotFoundError(f"missing endpoint sidecar seal: {seal_path}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    metadata = {key: value for key, value in seal.items()
                if key != "seal_metadata_sha256"}
    if (seal.get("schema") != SEAL_SCHEMA
            or seal.get("seal_metadata_sha256") != _canonical_sha(metadata)
            or seal.get("file_sha256") != expected_file_sha
            or Path(str(seal.get("path"))).resolve() != path.resolve()
            or direct_phase.sha256(path) != expected_file_sha):
        raise RuntimeError("Stage1 endpoint companion seal is invalid")
    return {**seal, "seal_path": str(seal_path.resolve()),
            "seal_file_sha256": direct_phase.sha256(seal_path)}


def load_b210_u3200(
        path: Path, expected_sha: str, device: torch.device,
        ) -> tuple[nn.Module, object, dict[str, object]]:
    payload, config, state, source = direct_phase.load_base_payload(
        path, expected_sha)
    extension = payload.get("uq_terminal_extension")
    if (payload.get("schema") != "b210_full_prefix_native_uq_refit_v1"
            or payload.get("phase") != "real"
            or not isinstance(extension, Mapping)
            or int(extension.get("terminal", -1)) != 3_200
            or payload.get("point_state_bitwise") is not True
            or payload.get("submission_authorized") is not False):
        raise RuntimeError("base is not the sealed B210 full-prefix U3200 endpoint")
    if source["model_state_sha256"] != payload.get("model_state_sha256"):
        raise RuntimeError("B210 payload model-state seal changed")
    values = payload.get("normalizer")
    if not isinstance(values, Mapping):
        raise RuntimeError("B210 U3200 has no Real normalizer")
    normalizer = _normalizer_from_payload(values, device)
    model = direct_phase.build_base(config, state, normalizer, device)
    report = {
        **source,
        "schema": payload["schema"],
        "uq_terminal": 3_200,
        "point_state_bitwise": True,
        "normalizer": _normalizer_report(values),
    }
    del payload, state
    gc.collect()
    return model, normalizer, report


def load_sim_input_normalizer(
        path: Path, expected_sha: str,
        teacher_sources: Sequence[Mapping[str, object]],
        device: torch.device,
        ) -> tuple[object, dict[str, object]]:
    file_sha = _require_file_sha(path, expected_sha, "Sim source checkpoint")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if (not isinstance(payload, Mapping)
            or payload.get("schema") != "st_b210_lifecycle_endpoint_v1"
            or payload.get("arm") != "temporal_candidate"
            or payload.get("interaction_mode") != "causal_temporal_candidate"
            or payload.get("phase") != "sim"
            or payload.get("endpoint") != "raw_u10000"
            or int(payload.get("update", -1)) != 10_000
            or payload.get("uq") is not False
            or payload.get("checkpoint_selected") is not False):
        raise RuntimeError("Sim normalizer source is not the causal ST raw-u10000 endpoint")
    values = payload.get("normalizer")
    if not isinstance(values, Mapping):
        raise RuntimeError("Sim source checkpoint lacks a normalizer")
    normalizer_sha = _normalizer_sha(values)
    if normalizer_sha != payload.get("normalizer_sha256"):
        raise RuntimeError("Sim source normalizer tensor SHA changed")
    for source in teacher_sources:
        common = source.get("common_sim_endpoint")
        if (not isinstance(common, Mapping)
                or common.get("file_sha256") != file_sha
                or common.get("normalizer_sha256") != normalizer_sha):
            raise RuntimeError("teacher does not reference the supplied Sim source")
    normalizer = _normalizer_from_payload(values, device)
    report = {
        "path": str(path.resolve()),
        "file_sha256": file_sha,
        "schema": payload["schema"],
        "endpoint": payload["endpoint"],
        "update": payload["update"],
        "normalizer": _normalizer_report(values),
    }
    del payload
    return normalizer, report


def load_st_teacher_soup(
        seed4_path: Path, seed4_sha: str,
        seed17_path: Path, seed17_sha: str,
        real_normalizer, device: torch.device,
        ) -> tuple[SpatiotemporalB210, dict[str, object],
                   tuple[Mapping[str, object], Mapping[str, object]]]:
    file_shas = {
        "seed4": _require_file_sha(
            seed4_path, seed4_sha, "teacher seed4 checkpoint"),
        "seed17": _require_file_sha(
            seed17_path, seed17_sha, "teacher seed17 checkpoint"),
    }
    payloads = {
        "seed4": teacher_build._load_endpoint(seed4_path),
        "seed17": teacher_build._load_endpoint(seed17_path),
    }
    for seed, expected_seed in (("seed4", 4), ("seed17", 17)):
        source = payloads[seed].get("source")
        if (not isinstance(source, Mapping)
                or int(source.get("real_seed", -1)) != expected_seed
                or int(source.get("real_updates", -1)) != 12_000):
            raise RuntimeError(f"{seed} teacher provenance changed")
    if payloads["seed4"]["model_config"] != payloads["seed17"]["model_config"]:
        raise RuntimeError("teacher seed configurations differ")
    if not _normalizers_equal(
            payloads["seed4"]["normalizer"],
            payloads["seed17"]["normalizer"]):
        raise RuntimeError("teacher seed Real normalizers differ")
    if not _normalizers_equal(payloads["seed4"]["normalizer"], real_normalizer):
        raise RuntimeError("B210 and causal-ST Real normalizers differ")

    collapsed: dict[str, dict[str, torch.Tensor]] = {}
    collapse_reports: dict[str, object] = {}
    for seed, payload in payloads.items():
        collapsed[seed], collapse_reports[seed] = (
            teacher_build.collapse_identical_temporal_banks(
                payload["model_state_dict"]))
    config_values = teacher_build._single_config(
        payloads["seed4"]["model_config"])
    for state in collapsed.values():
        teacher_build.validate_single_state(config_values, state)
    soup = teacher_build.equal_weight_fp32_soup(
        collapsed["seed4"], collapsed["seed17"])
    teacher_build.validate_single_state(config_values, soup)
    soup_sha = teacher_build.tensor_state_sha256(soup)

    raw_config = dict(config_values)
    if raw_config.pop("architecture") != "spatiotemporal_b210":
        raise RuntimeError("collapsed teacher architecture changed")
    interaction_mode = raw_config.pop("interaction_mode")
    if interaction_mode != "causal_temporal_candidate":
        raise RuntimeError("teacher soup is not causal temporal")
    if raw_config.pop("flattened_history_bypass") is not False:
        raise RuntimeError("teacher gained a flattened-history bypass")
    model = SpatiotemporalB210(
        SpatiotemporalB210Config(**raw_config), interaction_mode).float()
    model.load_state_dict(soup, strict=True)
    before_stats = {
        name: getattr(model, name).detach().cpu().clone()
        for name in ("persistence_scale", "persistence_bias", "target_std")
    }
    model.set_normalizer_statistics(
        real_normalizer.mean_input, real_normalizer.mean_target,
        real_normalizer.std_input, real_normalizer.std_target)
    if any(not torch.equal(before_stats[name], getattr(model, name).cpu())
           for name in before_stats):
        raise RuntimeError("teacher soup embedded a different Real normalizer")
    model.requires_grad_(False)
    model = model.to(device).eval()
    sources = (
        dict(payloads["seed4"]["source"]),
        dict(payloads["seed17"]["source"]),
    )
    report = {
        "fixed_weights": {"seed4": TEACHER_WEIGHT, "seed17": TEACHER_WEIGHT},
        "fp32_weight_soup": True,
        "soup_state_sha256": soup_sha,
        "model_config": config_values,
        "normalizer_sha256": _normalizer_sha(real_normalizer),
        "seeds": {
            seed: {
                "path": str(path.resolve()),
                "file_sha256": file_shas[seed],
                "endpoint_model_state_sha256": payloads[seed][
                    "model_state_sha256"],
                "collapse": collapse_reports[seed],
                "real_seed": int(payloads[seed]["source"]["real_seed"]),
                "update": int(payloads[seed]["update"]),
                "endpoint": payloads[seed]["endpoint"],
            }
            for seed, path in (("seed4", seed4_path), ("seed17", seed17_path))
        },
    }
    del payloads, collapsed, soup
    gc.collect()
    return model, report, sources


def _teacher_forward(
        history_normalized: torch.Tensor, base: nn.Module,
        st_teacher: nn.Module, normalizer,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    # ``no_grad`` (not inference_mode): the frozen B210 prediction is then a
    # regular tensor that the trainable adapter forecast encoder may save for
    # its own parameter backward.
    with torch.no_grad():
        base_prediction = base(history_normalized).detach()
        st_prediction = st_teacher(history_normalized).detach()
        target = teacher_target(base_prediction, st_prediction, normalizer)
    return base_prediction, target


def order_permutation_diagnostic(
        history_normalized: torch.Tensor,
        base: nn.Module, st_teacher: nn.Module,
        adapter: CausalPhaseDirectionAdapter, normalizer,
        ) -> dict[str, object]:
    """Re-run reversed input and prove all teacher/student rows follow it."""
    if history_normalized.shape[0] < 2:
        raise ValueError("order diagnostic requires at least two histories")
    order = torch.arange(
        history_normalized.shape[0] - 1, -1, -1,
        device=history_normalized.device)
    inverse = torch.argsort(order)
    with torch.no_grad():
        base_a, target_a = _teacher_forward(
            history_normalized, base, st_teacher, normalizer)
        student_a = adapter(history_normalized, base_a)
        base_b, target_b = _teacher_forward(
            history_normalized[order], base, st_teacher, normalizer)
        student_b = adapter(history_normalized[order], base_b)
        student_a = _to_physical_target(student_a, normalizer)[..., :2]
        student_b = _to_physical_target(student_b, normalizer)[..., :2]
        deltas = {
            "teacher_target": float((target_a - target_b[inverse]).abs().max()),
            "base_prediction": float((base_a - base_b[inverse]).abs().max()),
            "student_prediction": float((student_a - student_b[inverse]).abs().max()),
        }
        loss_a, _ = direct_phase.physical_phase_objective(student_a, target_a)
        loss_b, _ = direct_phase.physical_phase_objective(student_b, target_b)
        loss_delta = float((loss_a - loss_b).abs())
    tolerance = 2.0e-6
    if max(*deltas.values(), loss_delta) > tolerance:
        raise RuntimeError("teacher/student is not sample-order permutation equivariant")
    return {
        "kind": "reverse_batch_reexecution_not_cached_reindex",
        "batch": int(history_normalized.shape[0]),
        "permutation_sha256": _tensor_sha(order.cpu()),
        "max_abs_after_inverse": deltas,
        "loss_abs_delta": loss_delta,
        "tolerance": tolerance,
        "passed": True,
    }


def _batch_histories(
        dataset: TargetFreeHistoryDataset, indices: Sequence[int],
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    samples = [dataset[int(index)] for index in indices]
    histories = torch.stack([sample[0] for sample in samples]).to(device)
    domains = torch.tensor(
        [sample[1] for sample in samples], dtype=torch.int64, device=device)
    return histories, domains


def _geometry_audit(
        prediction_normalized: torch.Tensor,
        base_normalized: torch.Tensor, normalizer) -> dict[str, float]:
    prediction = _to_physical_target(
        prediction_normalized, normalizer)[..., :2].float()
    base = _to_physical_target(base_normalized, normalizer)[..., :2].float()
    prediction_mean = prediction.mean(dim=1, keepdim=True)
    base_mean = base.mean(dim=1, keepdim=True)
    prediction_rms = (prediction - prediction_mean).square().mean(
        dim=1, keepdim=True).sqrt()
    base_rms = (base - base_mean).square().mean(
        dim=1, keepdim=True).sqrt()
    return {
        "max_physical_temporal_mean_delta": float(
            (prediction_mean - base_mean).abs().max()),
        "max_physical_temporal_rms_delta": float(
            (prediction_rms - base_rms).abs().max()),
    }


def _optimizer(adapter: nn.Module):
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX)
    return optimizer, scheduler


def train_stage1(
        *, dataset: TargetFreeHistoryDataset,
        schedule: Sequence[Sequence[int]], base: nn.Module,
        st_teacher: nn.Module, sim_normalizer, real_normalizer,
        source: Mapping[str, object], data_report: Mapping[str, object],
        access_report: Mapping[str, object], output_dir: Path,
        device: torch.device,
        ) -> dict[str, object]:
    if len(dataset) != STAGE1_HISTORIES or len(schedule) != STAGE1_UPDATES:
        raise RuntimeError("Stage1 dataset/schedule length changed")
    adapter = direct_phase.build_adapter("causal_temporal", device)
    initial_state_sha = ema_lib.tensor_state_sha256(adapter.state_dict())
    optimizer, scheduler = _optimizer(adapter)
    base.eval().requires_grad_(False)
    st_teacher.eval().requires_grad_(False)
    adapter.train()
    real4k.seed_real_recipe(device)
    trace: list[dict[str, object]] = []
    permutation_mechanics: dict[str, object] | None = None
    direction_gradient_u1 = False
    history_gradient_u2 = False
    started = time.perf_counter()
    for cursor, indices in enumerate(schedule):
        update = cursor + 1
        history, domains = _batch_histories(dataset, indices, device)
        history = map_sim_to_real_input(
            history, domains, sim_normalizer, real_normalizer)
        if torch.count_nonzero(history[..., 2]).item() != 0:
            raise RuntimeError("Stage1 pressure is not exactly zero")
        history_normalized = _normalize_input(history, real_normalizer)
        if update == 1:
            permutation_mechanics = order_permutation_diagnostic(
                history_normalized, base, st_teacher, adapter,
                real_normalizer)

        base_prediction, target_physical = _teacher_forward(
            history_normalized, base, st_teacher, real_normalizer)
        optimizer.zero_grad(set_to_none=True)
        components = adapter.forward_components(
            history_normalized, base_prediction)
        prediction_physical = _to_physical_target(
            components.prediction, real_normalizer)[..., :2]
        loss, details = direct_phase.physical_phase_objective(
            prediction_physical, target_physical)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Stage1 KD loss at u{update}")
        loss.backward()
        if update == 1:
            gradient = adapter.direction_head.weight.grad
            direction_gradient_u1 = bool(
                gradient is not None and torch.isfinite(gradient).all()
                and torch.count_nonzero(gradient))
            if not direction_gradient_u1:
                raise RuntimeError("Stage1 direction head lacks a finite u1 gradient")
        if update == 2:
            gradient = adapter.frame_encoder.stem.weight.grad
            history_gradient_u2 = bool(
                gradient is not None and torch.isfinite(gradient).all()
                and torch.count_nonzero(gradient))
            if not history_gradient_u2:
                raise RuntimeError("Stage1 history encoder lacks a finite u2 gradient")
        gradient_norm = nn.utils.clip_grad_norm_(adapter.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite Stage1 gradient at u{update}")
        optimizer.step()
        scheduler.step()

        if update in TRACE_UPDATES_STAGE1:
            teacher_mean = target_physical.mean(dim=1, keepdim=True)
            base_physical = _to_physical_target(
                base_prediction, real_normalizer)[..., :2]
            base_mean = base_physical.mean(dim=1, keepdim=True)
            teacher_rms = (target_physical - teacher_mean).square().mean(
                dim=1, keepdim=True).sqrt()
            base_rms = (base_physical - base_mean).square().mean(
                dim=1, keepdim=True).sqrt()
            row = {
                "update": update,
                "batch_size": len(indices),
                "sim_histories": int((domains == 0).sum()),
                "real_histories": int((domains == 1).sum()),
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "teacher_max_mean_delta_from_B210": float(
                    (teacher_mean - base_mean).abs().max()),
                "teacher_max_rms_delta_from_B210": float(
                    (teacher_rms - base_rms).abs().max()),
                **_geometry_audit(
                    components.prediction, base_prediction, real_normalizer),
                **{key: float(value) for key, value in details.items()},
            }
            trace.append(row)
            print(json.dumps({
                "event": "b210_phase_kd_stage1_progress", **row,
            }, sort_keys=True), flush=True)
    if not direction_gradient_u1 or not history_gradient_u2:
        raise RuntimeError("Stage1 endpoint lacks required gradient evidence")
    if permutation_mechanics is None:
        raise RuntimeError("Stage1 order-permutation diagnostic did not run")

    raw_state = _state_snapshot(adapter)
    payload = {
        "schema": STAGE1_ENDPOINT_SCHEMA,
        "experiment_schema": SCHEMA,
        "stage": "stage1_target_free_kd",
        "arm": "kd_pretrain",
        "endpoint": f"raw_u{STAGE1_UPDATES}",
        "update": STAGE1_UPDATES,
        "adapter_config": adapter.architecture_config(),
        "adapter_parameters": EXPECTED_ADAPTER_PARAMETERS,
        "raw_adapter_state": raw_state,
        "raw_adapter_state_sha256": ema_lib.tensor_state_sha256(raw_state),
        "ema": None,
        "normalizers": {
            "real": _normalizer_tensors(real_normalizer),
            "real_sha256": _normalizer_sha(real_normalizer),
            "sim": _normalizer_tensors(sim_normalizer),
            "sim_sha256": _normalizer_sha(sim_normalizer),
        },
        "source": dict(source),
        "data": dict(data_report),
        "target_free_access": dict(access_report),
        "training": {
            "unique_histories": STAGE1_HISTORIES,
            "updates": STAGE1_UPDATES,
            "batch_size": BATCH_SIZE,
            "last_batch_size": len(schedule[-1]),
            "seed": SEED,
            "sample_permutation_sha256": STAGE1_PERMUTATION_SHA256,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": SCHEDULER_T_MAX,
            "ema": None,
            "augmentation": None,
            "objective": "physical_uv_official_Rel-L2_mapping_only",
            "trace": trace,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "mechanics": {
            "initial_adapter_state_sha256": initial_state_sha,
            "direction_head_gradient_u1": direction_gradient_u1,
            "history_encoder_gradient_u2": history_gradient_u2,
            "order_permutation": permutation_mechanics,
        },
        "checkpoint_selected": False,
        "score_gate": None,
        "clean716_opened": False,
        "submission_authorized": False,
    }
    endpoint_path = output_dir / f"kd_pretrain_raw_u{STAGE1_UPDATES}.pt"
    seal = write_endpoint_and_seal(
        endpoint_path, payload, ("raw_adapter_state",))
    return {
        "stage": "stage1_target_free_kd",
        "endpoint": seal,
        "raw_adapter_state_sha256": payload["raw_adapter_state_sha256"],
        "trace": trace,
        "mechanics": payload["mechanics"],
        "target_free_access": dict(access_report),
        "checkpoint_selected": False,
    }


def load_stage1_endpoint(
        path: Path, expected_sha: str,
        real_normalizer, current_base: Mapping[str, object],
        clean_manifest_sha256: str,
        ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    file_sha = _require_file_sha(path, expected_sha, "Stage1 KD endpoint")
    companion = verify_companion_seal(path, file_sha)
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if (not isinstance(payload, Mapping)
            or payload.get("schema") != STAGE1_ENDPOINT_SCHEMA
            or payload.get("experiment_schema") != SCHEMA
            or payload.get("stage") != "stage1_target_free_kd"
            or payload.get("arm") != "kd_pretrain"
            or payload.get("endpoint") != f"raw_u{STAGE1_UPDATES}"
            or int(payload.get("update", -1)) != STAGE1_UPDATES
            or payload.get("ema") is not None
            or payload.get("checkpoint_selected") is not False
            or payload.get("score_gate") is not None
            or payload.get("clean716_opened") is not False):
        raise RuntimeError("candidate source is not the fixed Stage1 raw endpoint")
    expected_config = CausalPhaseDirectionAdapter(
        CausalPhaseDirectionConfig(mixer_mode="causal_temporal")
    ).architecture_config()
    if (payload.get("adapter_parameters") != EXPECTED_ADAPTER_PARAMETERS
            or payload.get("adapter_config") != expected_config):
        raise RuntimeError("Stage1 adapter topology changed")
    normalizers = payload.get("normalizers")
    if (not isinstance(normalizers, Mapping)
            or not _normalizers_equal(normalizers.get("real", {}), real_normalizer)
            or normalizers.get("real_sha256") != _normalizer_sha(real_normalizer)):
        raise RuntimeError("Stage1 and Stage2 Real normalizers differ")
    source = payload.get("source")
    stage1_base = source.get("base") if isinstance(source, Mapping) else None
    stage1_manifest = (
        source.get("clean_manifest") if isinstance(source, Mapping) else None)
    base_normalizer = (
        stage1_base.get("normalizer")
        if isinstance(stage1_base, Mapping) else None)
    if (not isinstance(stage1_base, Mapping)
            or stage1_base.get("sha256") != current_base.get("sha256")
            or stage1_base.get("model_state_sha256")
            != current_base.get("model_state_sha256")
            or not isinstance(base_normalizer, Mapping)
            or base_normalizer.get("sha256") != _normalizer_sha(real_normalizer)
            or not isinstance(stage1_manifest, Mapping)
            or stage1_manifest.get("file_sha256") != clean_manifest_sha256):
        raise RuntimeError("Stage1 source lineage differs from Stage2")
    training = payload.get("training")
    if (not isinstance(training, Mapping)
            or int(training.get("unique_histories", -1)) != STAGE1_HISTORIES
            or int(training.get("updates", -1)) != STAGE1_UPDATES
            or int(training.get("batch_size", -1)) != BATCH_SIZE
            or int(training.get("last_batch_size", -1)) != 5
            or int(training.get("seed", -1)) != SEED
            or training.get("sample_permutation_sha256")
            != STAGE1_PERMUTATION_SHA256
            or training.get("ema") is not None
            or training.get("augmentation") is not None
            or training.get("objective")
            != "physical_uv_official_Rel-L2_mapping_only"):
        raise RuntimeError("Stage1 one-pass KD training seal changed")
    target_free = payload.get("target_free_access")
    if (not isinstance(target_free, Mapping)
            or target_free.get("schema")
            != "b210_phase_kd_target_free_access_v1"
            or int(target_free.get("opened_files", -1)) != 181
            or int(target_free.get("future_tensor_requests", -1)) != 0
            or target_free.get("future_ground_truth_materialized") is not False
            or target_free.get("clean716_opened") is not False):
        raise RuntimeError("Stage1 target-free access seal changed")
    state = payload.get("raw_adapter_state")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("Stage1 endpoint lacks raw adapter state")
    state = _state_snapshot(state)
    state_sha = ema_lib.tensor_state_sha256(state)
    if state_sha != payload.get("raw_adapter_state_sha256"):
        raise RuntimeError("Stage1 raw adapter tensor SHA changed")
    adapter = _adapter_from_config(payload["adapter_config"])
    adapter.load_state_dict(state, strict=True)
    return state, {
        "path": str(path.resolve()),
        "file_sha256": file_sha,
        "raw_adapter_state_sha256": state_sha,
        "companion_seal": companion,
        "update": STAGE1_UPDATES,
        "endpoint": f"raw_u{STAGE1_UPDATES}",
    }


def _rng_state_sha(device: torch.device) -> str:
    values = {"cpu": torch.get_rng_state().cpu()}
    if device.type == "cuda":
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            values[f"cuda_{index}"] = state.cpu()
    return ema_lib.tensor_state_sha256(values)


def train_stage2_arm(
        *, arm: str, initial_state: Mapping[str, torch.Tensor],
        base: nn.Module, dataset, normalizer,
        schedule: Sequence[Sequence[int]], source: Mapping[str, object],
        output_dir: Path, device: torch.device,
        ) -> dict[str, object]:
    if arm not in STAGE2_ARMS:
        raise ValueError(f"unknown Stage2 arm {arm!r}")
    if len(dataset) != REAL_HISTORIES or len(schedule) != STAGE2_UPDATES:
        raise RuntimeError("Stage2 Real dataset/schedule changed")
    adapter = direct_phase.build_adapter("causal_temporal", device)
    adapter.load_state_dict(initial_state, strict=True)
    initial_state_sha = ema_lib.tensor_state_sha256(adapter.state_dict())
    optimizer, scheduler = _optimizer(adapter)
    # Required ordering: load this arm's u0 state first, then initialize EMA.
    ema = ema_lib.FP32EMAShadow(adapter, EMA_DECAY)
    ema_u0_sha = ema_lib.tensor_state_sha256(
        ema.materialize_model_state(adapter))
    if ema_u0_sha != initial_state_sha:
        raise RuntimeError("Stage2 EMA was not initialized from Real u0")
    base.eval().requires_grad_(False)
    adapter.train()
    real4k.seed_real_recipe(device)
    rng_u0_sha = _rng_state_sha(device)
    trace: list[dict[str, object]] = []
    augmentation_probes: dict[str, dict[str, str]] = {}
    direction_gradient_u1 = False
    history_gradient_u2 = False
    direct_identity_u0: bool | None = None
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
        if update in AUGMENTATION_PROBE_UPDATES:
            augmentation_probes[str(update)] = {
                "indices_sha256": _canonical_sha(
                    [int(index) for index in indices]),
                "augmented_history_sha256": _tensor_sha(adapter_history),
                "rng_after_augmentation_sha256": _rng_state_sha(device),
            }

        with torch.no_grad():
            base_prediction = base(base_history).detach()
        optimizer.zero_grad(set_to_none=True)
        components = adapter.forward_components(
            adapter_history, base_prediction)
        if update == 1:
            direct_identity_u0 = torch.equal(
                components.prediction, components.base_prediction)
            if arm == STAGE2_ARMS[0] and not direct_identity_u0:
                raise RuntimeError("direct control is not exact B210 at Real u0")
        loss, details = direct_phase.phase_objective(
            components.prediction, target_normalized, normalizer)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Stage2 loss at {arm} u{update}")
        loss.backward()
        if update == 1:
            gradient = adapter.direction_head.weight.grad
            direction_gradient_u1 = bool(
                gradient is not None and torch.isfinite(gradient).all()
                and torch.count_nonzero(gradient))
            if not direction_gradient_u1:
                raise RuntimeError(f"{arm} direction head lacks u1 gradient")
        if update == 2:
            gradient = adapter.frame_encoder.stem.weight.grad
            history_gradient_u2 = bool(
                gradient is not None and torch.isfinite(gradient).all()
                and torch.count_nonzero(gradient))
            if not history_gradient_u2:
                raise RuntimeError(f"{arm} history path lacks u2 gradient")
        gradient_norm = nn.utils.clip_grad_norm_(adapter.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite Stage2 gradient at {arm} u{update}")
        optimizer.step()
        ema.update_after_optimizer_step(adapter, optimizer_step=update)
        scheduler.step()

        if update in TRACE_UPDATES_STAGE2:
            row = {
                "update": update,
                "batch_size": len(indices),
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **_geometry_audit(
                    components.prediction, base_prediction, normalizer),
                **{key: float(value) for key, value in details.items()},
            }
            trace.append(row)
            print(json.dumps({
                "event": "b210_phase_kd_stage2_progress",
                "arm": arm, **row,
            }, sort_keys=True), flush=True)
    if not direction_gradient_u1 or not history_gradient_u2:
        raise RuntimeError(f"{arm} endpoint lacks gradient evidence")
    if ema.updates != STAGE2_UPDATES:
        raise RuntimeError(f"{arm} EMA update count changed")

    raw_state = _state_snapshot(adapter)
    ema_state = ema.materialize_model_state(adapter)
    payload = {
        "schema": STAGE2_ENDPOINT_SCHEMA,
        "experiment_schema": SCHEMA,
        "stage": "stage2_matched_real4k",
        "arm": arm,
        "endpoint": "fixed_raw_and_ema_u4000",
        "update": STAGE2_UPDATES,
        "adapter_config": adapter.architecture_config(),
        "adapter_parameters": EXPECTED_ADAPTER_PARAMETERS,
        "raw_adapter_state": raw_state,
        "raw_adapter_state_sha256": ema_lib.tensor_state_sha256(raw_state),
        "ema_adapter_state": ema_state,
        "ema_adapter_state_sha256": ema_lib.tensor_state_sha256(ema_state),
        "ema": {key: value for key, value in ema.seal(adapter).items()
                if key != "model_state"},
        "normalizer": _normalizer_tensors(normalizer),
        "normalizer_sha256": _normalizer_sha(normalizer),
        "source": dict(source),
        "training": {
            "updates": STAGE2_UPDATES,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "sample_schedule_sha256": real4k.schedule_sha(schedule),
            "augmentation": dict(real4k.AUGMENTATION),
            "objective": "physical_uv_official_Rel-L2_mapping_only",
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": SCHEDULER_T_MAX,
            "ema_decay": EMA_DECAY,
            "ema_initialized_from_real_u0": True,
            "trace": trace,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "mechanics": {
            "initial_adapter_state_sha256": initial_state_sha,
            "ema_u0_state_sha256": ema_u0_sha,
            "rng_u0_sha256": rng_u0_sha,
            "augmentation_probes": augmentation_probes,
            "prediction_bitwise_B210_at_real_u0": direct_identity_u0,
            "direction_head_gradient_u1": direction_gradient_u1,
            "history_encoder_gradient_u2": history_gradient_u2,
        },
        "reported_states": ["raw_u4000", "ema_u4000"],
        "checkpoint_selected": False,
        "score_gate": None,
        "clean716_opened": False,
        "submission_authorized": False,
    }
    path = output_dir / arm / "adapter_raw_ema_u4000.pt"
    seal = write_endpoint_and_seal(
        path, payload, ("raw_adapter_state", "ema_adapter_state"))
    result = {
        "arm": arm,
        "endpoint": seal,
        "initial_adapter_state_sha256": initial_state_sha,
        "ema_u0_state_sha256": ema_u0_sha,
        "raw_adapter_state_sha256": payload["raw_adapter_state_sha256"],
        "ema_adapter_state_sha256": payload["ema_adapter_state_sha256"],
        "augmentation_probes": augmentation_probes,
        "trace": trace,
        "checkpoint_selected": False,
    }
    del adapter, optimizer, scheduler, ema, payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def assert_matched_stage2(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if tuple(report["arm"] for report in reports) != STAGE2_ARMS:
        raise RuntimeError("Stage2 arms or execution order changed")
    left, right = reports
    if left["initial_adapter_state_sha256"] == right["initial_adapter_state_sha256"]:
        raise RuntimeError("KD candidate unexpectedly equals zero-start control")
    if left["augmentation_probes"] != right["augmentation_probes"]:
        raise RuntimeError("Stage2 arms did not receive identical augmentation probes")
    return {
        "only_initial_state_differs": True,
        "sample_schedule_shared": True,
        "augmentation_probe_updates": list(AUGMENTATION_PROBE_UPDATES),
        "augmentation_probes_bitwise_equal": True,
        "loss_optimizer_scheduler_equal": True,
        "endpoints_fixed_not_selected": True,
    }


def _git_report() -> dict[str, object]:
    return sim_data.git_report()


def _validate_formal_common(args: argparse.Namespace, acknowledgement: str) -> torch.device:
    if not args.execute or args.acknowledge_protocol != acknowledgement:
        raise PermissionError(
            "formal stage requires --execute and its exact acknowledgement")
    if args.output_dir is None:
        raise ValueError("formal stage requires --output-dir")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal two-stage phase training requires CUDA")
    git = _git_report()
    if not git["clean"]:
        raise RuntimeError(f"formal stage requires clean Git: {git['porcelain']}")
    return device


def run_stage1(args: argparse.Namespace) -> dict[str, object]:
    device = _validate_formal_common(args, STAGE1_ACKNOWLEDGEMENT)
    required = (
        args.base_checkpoint, args.base_checkpoint_sha256,
        args.teacher_seed4_checkpoint, args.teacher_seed4_checkpoint_sha256,
        args.teacher_seed17_checkpoint, args.teacher_seed17_checkpoint_sha256,
        args.sim_source_checkpoint, args.sim_source_checkpoint_sha256,
        args.sim_dir, args.real_dir, args.clean_manifest,
    )
    if any(value is None for value in required):
        raise ValueError("Stage1 requires all source paths and explicit SHA-256 values")
    base, real_normalizer, base_report = load_b210_u3200(
        args.base_checkpoint, args.base_checkpoint_sha256, device)
    st_teacher, teacher_report, teacher_sources = load_st_teacher_soup(
        args.teacher_seed4_checkpoint,
        args.teacher_seed4_checkpoint_sha256,
        args.teacher_seed17_checkpoint,
        args.teacher_seed17_checkpoint_sha256,
        real_normalizer, device)
    sim_normalizer, sim_source_report = load_sim_input_normalizer(
        args.sim_source_checkpoint, args.sim_source_checkpoint_sha256,
        teacher_sources, device)

    rows, manifest_report = real4k.load_clean_manifest(args.clean_manifest)
    audit = TargetFreeAccessAudit()
    sim_trajectories, sim_windows, sim_report = (
        load_sim_history_prefixes(args.sim_dir, audit))
    real_trajectories, real_windows, real_report = (
        load_real_history_prefixes(args.real_dir, rows, audit))
    windows = [*sim_windows, *real_windows]
    dataset = TargetFreeHistoryDataset(
        sim_trajectories, real_trajectories, windows)
    schedule, schedule_report = make_stage1_schedule(len(dataset))
    if sum(len(batch) for batch in schedule) != len(dataset):
        raise RuntimeError("Stage1 schedule does not visit every history once")
    access_report = audit.report()
    if access_report["future_tensor_requests"] != 0:
        raise RuntimeError("Stage1 future target access audit failed")
    source = {
        "base": base_report,
        "teacher": teacher_report,
        "sim_normalizer_source": sim_source_report,
        "clean_manifest": {
            **manifest_report,
            "path": str(args.clean_manifest.resolve()),
            "file_sha256": direct_phase.sha256(args.clean_manifest),
        },
        "git": _git_report(),
    }
    data_report = {
        "sim": sim_report,
        "real": real_report,
        "combined": {
            "ordering_before_permutation": "Sim4800_then_Real9933",
            "histories": len(dataset),
            "schedule": schedule_report,
        },
    }
    args.output_dir.mkdir(parents=True)
    stage = train_stage1(
        dataset=dataset, schedule=schedule, base=base,
        st_teacher=st_teacher, sim_normalizer=sim_normalizer,
        real_normalizer=real_normalizer, source=source,
        data_report=data_report, access_report=access_report,
        output_dir=args.output_dir, device=device)
    result = {
        "schema": SCHEMA,
        "mode": "stage1",
        "contract": experiment_contract(),
        "source": source,
        "data": data_report,
        "stage1": stage,
        "clean716_opened": False,
        "checkpoint_selection": False,
        "score_gate": None,
        "submission_authorized": False,
    }
    _atomic_json(args.output_dir / "result.json", result)
    return result


def run_stage2(args: argparse.Namespace) -> dict[str, object]:
    device = _validate_formal_common(args, STAGE2_ACKNOWLEDGEMENT)
    required = (
        args.base_checkpoint, args.base_checkpoint_sha256,
        args.stage1_endpoint, args.stage1_endpoint_sha256,
        args.real_dir, args.clean_manifest,
    )
    if any(value is None for value in required):
        raise ValueError("Stage2 requires B210, Stage1 seal, and clean81 paths")
    base, real_normalizer, base_report = load_b210_u3200(
        args.base_checkpoint, args.base_checkpoint_sha256, device)
    rows, manifest_report = real4k.load_clean_manifest(args.clean_manifest)
    manifest_file_sha = direct_phase.sha256(args.clean_manifest)
    candidate_state, stage1_report = load_stage1_endpoint(
        args.stage1_endpoint, args.stage1_endpoint_sha256, real_normalizer,
        base_report, manifest_file_sha)
    access = real4k.PrefixAccessAudit()
    trajectories, sparse_windows = real4k.load_real_rows(
        args.real_dir, rows, phase="fit", audit=access)
    derived_normalizer, derived_report = real4k.normalizer_from_sparse_windows(
        trajectories, sparse_windows, device)
    if not _normalizers_equal(derived_normalizer, real_normalizer):
        raise RuntimeError("clean81-derived and checkpoint Real normalizers differ")
    optimizer_windows = real4k.dense_windows(trajectories)
    if len(optimizer_windows) != REAL_HISTORIES:
        raise RuntimeError("Stage2 clean81 stride5 window count changed")
    dataset = real4k.RealWindowDataset(trajectories, optimizer_windows)
    schedule = real4k.make_sample_schedule(
        len(dataset), STAGE2_UPDATES, BATCH_SIZE)
    schedule_sha = real4k.schedule_sha(schedule)

    zero_adapter = direct_phase.build_adapter("causal_temporal", device)
    zero_state = _state_snapshot(zero_adapter)
    zero_state_sha = ema_lib.tensor_state_sha256(zero_state)
    del zero_adapter
    if zero_state_sha == stage1_report["raw_adapter_state_sha256"]:
        raise RuntimeError("Stage1 KD endpoint did not move from zero start")
    source = {
        "base": base_report,
        "stage1": stage1_report,
        "clean_manifest": {
            **manifest_report,
            "path": str(args.clean_manifest.resolve()),
            "file_sha256": manifest_file_sha,
        },
        "real_normalizer": derived_report,
        "real_prefix_access": access.events,
        "optimizer_windows": len(dataset),
        "optimizer_stride": real4k.OPTIMIZER_STRIDE,
        "sample_schedule_sha256": schedule_sha,
        "git": _git_report(),
    }
    initial_states = {
        STAGE2_ARMS[0]: zero_state,
        STAGE2_ARMS[1]: candidate_state,
    }
    args.output_dir.mkdir(parents=True)
    reports = [
        train_stage2_arm(
            arm=arm, initial_state=initial_states[arm], base=base,
            dataset=dataset, normalizer=real_normalizer,
            schedule=schedule, source=source, output_dir=args.output_dir,
            device=device)
        for arm in STAGE2_ARMS
    ]
    matched = assert_matched_stage2(reports)
    result = {
        "schema": SCHEMA,
        "mode": "stage2",
        "contract": experiment_contract(),
        "source": source,
        "arms": reports,
        "matched_mechanics": matched,
        "fixed_reported_states": [
            f"{arm}/{endpoint}"
            for arm in STAGE2_ARMS
            for endpoint in ("raw_u4000", "ema_u4000")
        ],
        "clean716_opened": False,
        "checkpoint_selection": False,
        "score_gate": None,
        "submission_authorized": False,
    }
    _atomic_json(args.output_dir / "result.json", result)
    return result


def _checkpoint_header(path: Path) -> dict[str, object]:
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    config = payload.get("model_config")
    return {
        key: payload.get(key)
        for key in (
            "schema", "experiment_schema", "lifecycle_schema", "arm",
            "adaptation_mode", "interaction_mode", "phase", "endpoint",
            "update", "iteration", "uq", "checkpoint_selected",
            "model_state_sha256", "normalizer_sha256")
        if key in payload
    } | {
        "architecture": (
            config.get("architecture") if isinstance(config, Mapping) else None),
        "state_tensors": (
            len(payload["model_state_dict"])
            if isinstance(payload.get("model_state_dict"), Mapping) else None),
    }


def _inventory_checkpoint(
        path: Path | None, expected_sha: str | None,
        role: str) -> dict[str, object]:
    if path is None:
        return {"role": role, "status": "missing_path_not_guessed"}
    result: dict[str, object] = {
        "role": role, "path": str(path.resolve()),
        "exists": path.is_file(),
    }
    if not path.is_file():
        result["status"] = "missing"
        return result
    actual = direct_phase.sha256(path)
    result.update({
        "status": "present",
        "bytes": path.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected_sha,
        "expected_sha256_matches": (
            None if expected_sha is None else actual == expected_sha),
        "header": _checkpoint_header(path),
    })
    return result


def inventory_report(args: argparse.Namespace) -> dict[str, object]:
    """Filesystem/checkpoint inventory only; never open an H5 field array."""
    checkpoints = [
        _inventory_checkpoint(
            args.base_checkpoint, args.base_checkpoint_sha256, "B210_U3200"),
        _inventory_checkpoint(
            args.teacher_seed4_checkpoint,
            args.teacher_seed4_checkpoint_sha256,
            "causal_ST_Real12k_seed4_EMA"),
        _inventory_checkpoint(
            args.teacher_seed17_checkpoint,
            args.teacher_seed17_checkpoint_sha256,
            "causal_ST_Real12k_seed17_EMA"),
        _inventory_checkpoint(
            args.sim_source_checkpoint,
            args.sim_source_checkpoint_sha256,
            "released_Sim_input_normalizer_source"),
    ]

    directories: dict[str, object] = {}
    for role, path in (("released_Sim", args.sim_dir),
                       ("released_Real_backing", args.real_dir)):
        if path is None:
            directories[role] = {"status": "missing_path_not_guessed"}
            continue
        files = sorted(path.glob("*.h5")) if path.is_dir() else []
        listing = [f"{item.name}\t{item.stat().st_size}" for item in files]
        directories[role] = {
            "path": str(path.resolve()),
            "exists": path.is_dir(),
            "h5_files": len(files),
            "bytes": sum(item.stat().st_size for item in files),
            "basename_size_inventory_sha256": hashlib.sha256(
                "\n".join(listing).encode("utf-8")).hexdigest(),
            "raw_h5_opened": False,
        }

    if args.clean_manifest is None:
        manifest: dict[str, object] = {"status": "missing_path_not_guessed"}
    elif not args.clean_manifest.is_file():
        manifest = {"path": str(args.clean_manifest.resolve()), "status": "missing"}
    else:
        rows, report = real4k.load_clean_manifest(args.clean_manifest)
        dense = sum(len(range(
            0, int(row["val_boundary"]) - 40 + 1,
            real4k.OPTIMIZER_STRIDE)) for row in rows)
        manifest = {
            "path": str(args.clean_manifest.resolve()),
            "status": "present",
            "bytes": args.clean_manifest.stat().st_size,
            "file_sha256": direct_phase.sha256(args.clean_manifest),
            "rows": len(rows),
            "normalizer_windows": sum(len(row["train_starts"]) for row in rows),
            "dense_stride5_histories": dense,
            **report,
        }
    missing = [item["role"] for item in checkpoints
               if item["status"] != "present"]
    missing.extend(role for role, item in directories.items()
                   if not item.get("exists", False))
    if manifest["status"] != "present":
        missing.append("clean81_manifest")
    return {
        "schema": "b210_phase_kd_read_only_inventory_v1",
        "checkpoints": checkpoints,
        "directories": directories,
        "clean81_manifest": manifest,
        "missing": missing,
        "raw_h5_opened": False,
        "clean716_manifest_opened": False,
        "clean716_h5_opened": False,
        "gpu_used": False,
    }


class _IdentityNormalizer:
    def __init__(self):
        self.mean_input = torch.zeros(3)
        self.mean_target = torch.zeros(3)
        self.std_input = torch.ones(3)
        self.std_target = torch.ones(3)


def mechanics_smoke() -> dict[str, object]:
    """Synthetic CPU-only contract smoke; never inspect private artifacts."""
    torch.manual_seed(SEED)
    normalizer = _IdentityNormalizer()
    sim_normalizer = _IdentityNormalizer()
    history = torch.randn(3, 20, 8, 8, 3)
    domains = torch.tensor((0, 1, 0))
    mapped = map_sim_to_real_input(
        history, domains, sim_normalizer, normalizer)
    if torch.count_nonzero(mapped[..., 2]).item() != 0:
        raise RuntimeError("mechanics smoke pressure mapping failed")
    base = torch.randn(3, 20, 8, 8, 3)
    st = torch.randn_like(base)
    target = teacher_target(base, st, normalizer)
    adapter = CausalPhaseDirectionAdapter(CausalPhaseDirectionConfig(
        hidden_channels=8, temporal_blocks=2, temporal_kernel=3,
        groups=4, mixer_mode="causal_temporal"))
    components = adapter.forward_components(mapped, base)
    prediction = components.prediction[..., :2]
    loss, _details = direct_phase.physical_phase_objective(prediction, target)
    loss.backward()
    schedule, schedule_report = make_stage1_schedule(17)
    ema = ema_lib.FP32EMAShadow(adapter, EMA_DECAY)
    ema_u0_equal = (
        ema_lib.tensor_state_sha256(ema.materialize_model_state(adapter))
        == ema_lib.tensor_state_sha256(adapter.state_dict()))
    return {
        "schema": "b210_phase_kd_mechanics_smoke_v1",
        "device": "cpu",
        "finite_loss": bool(torch.isfinite(loss)),
        "direction_head_gradient": bool(
            adapter.direction_head.weight.grad is not None
            and torch.count_nonzero(adapter.direction_head.weight.grad)),
        "pressure_exact_zero": True,
        "teacher_mean_rms_restored": {
            "max_mean_delta": float((
                target.mean(dim=1, keepdim=True)
                - base[..., :2].mean(dim=1, keepdim=True)).abs().max()),
            "max_rms_delta": float((
                (target - target.mean(dim=1, keepdim=True)).square().mean(
                    dim=1, keepdim=True).sqrt()
                - (base[..., :2] - base[..., :2].mean(
                    dim=1, keepdim=True)).square().mean(
                        dim=1, keepdim=True).sqrt()).abs().max()),
        },
        "one_pass_schedule_batches": schedule,
        "one_pass_schedule": schedule_report,
        "ema_initialized_from_u0": ema_u0_equal,
        "raw_h5_opened": False,
        "clean716_opened": False,
        "gpu_used": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "contract":
        return experiment_contract()
    if args.mode == "inventory":
        return inventory_report(args)
    if args.mode == "mechanics-smoke":
        return mechanics_smoke()
    if args.mode == "stage1":
        return run_stage1(args)
    return run_stage2(args)


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
