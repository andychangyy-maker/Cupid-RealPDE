#!/usr/bin/env python3
"""Frozen held-AoA=10 history-only coarse local-flow discovery experiment.

The normal entry point is deliberately single-use and executes the frozen
2,000-update contract.  ``--smoke`` is an in-memory architecture check only:
it neither opens the clean716 cache nor creates an experiment artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping

try:
    import h5py
except ModuleNotFoundError:  # permits the target-free --smoke in minimal environments
    h5py = None
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOOTH_PATH = REPO_ROOT / "code/evaluation/audit_b210_clean716_smooth_flow_oracle.py"
INTEGER_PATH = REPO_ROOT / "code/evaluation/audit_b210_clean716_integer_transport_oracle.py"
EXPECTED_SHAPE = (716, 20, 32, 64, 2)
HELD_AOA = 10
FIT_AOAS = (0, 5, 15, 20)
UPDATES = 2_000
BATCH_SIZE = 64
TEACHER_BATCH_SIZE = 16
SEED = 210
EXPECTED_PREDICTION_SHA256 = "4e373860669ebaa4368833cbf705bf07452fee7fae0d5682d2370d9ceab5c075"
EXPECTED_TARGET_SHA256 = "533d913297b969613db8ebb3e41a4cb519c207c4270b11fb6bc96be4ff8a622b"
EXPECTED_SMOOTH_SHA256 = "8a20a898773ece0b4d3f28f51489290ee62aeb976d20b6e22072625a8309db47"
EXPECTED_INTEGER_SHA256 = "edc156121a1ed2d6e017beb3b9a8547dad49342c40efb83eda3277d9cf3170e3"
EXPECTED_SCORING_SHA256 = "a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39"
EXPECTED_ARCHIVE_SHA256 = "b655aa406e9925786101e91cd49c08e1e60467f74d9262c65e6b6eaa76cb4e7c"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smooth = _load_module("b210_frozen_smooth_oracle", SMOOTH_PATH)
integer = _load_module("b210_frozen_integer_scoring", INTEGER_PATH)
SCHEMA = "b210_history_coarse_flow_held10_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def verify_frozen_sources(metadata: Mapping[str, Any]) -> dict[str, object]:
    """Fail before training if a frozen helper, scorer, or source moved."""
    expected = {
        "smooth_oracle_script": (SMOOTH_PATH, EXPECTED_SMOOTH_SHA256),
        "integer_scoring_script": (INTEGER_PATH, EXPECTED_INTEGER_SHA256),
        "official_scoring": (integer.SCORING_PATH, EXPECTED_SCORING_SHA256),
    }
    report: dict[str, object] = {}
    for name, (path, expected_sha) in expected.items():
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"frozen {name} SHA changed: {actual_sha} != {expected_sha}")
        report[name] = {"path": str(path.resolve()), "sha256": actual_sha}
    archive = metadata.get("archive")
    if not isinstance(archive, Mapping):
        raise ValueError("clean716 metadata has no source archive identity")
    archive_path = Path(str(archive.get("path", "")))
    if (str(archive.get("sha256")) != EXPECTED_ARCHIVE_SHA256
            or not archive_path.is_file()):
        raise RuntimeError("source archive metadata/path changed")
    archive_sha = sha256_file(archive_path)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("source archive bytes changed")
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    if tracked.strip():
        raise RuntimeError("tracked Git worktree must be clean before formal run")
    report["source_archive"] = {
        "path": str(archive_path.resolve()), "sha256": archive_sha,
        "bytes": archive_path.stat().st_size}
    report["git"] = {"commit": git_commit(), "tracked_worktree_clean": True}
    return report


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and any(value is None for value in
                              (args.cache_dir, args.real_dir, args.output_dir)):
        parser.error("--cache-dir, --real-dir, and --output-dir are required")
    return args


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def require_h5py() -> None:
    if h5py is None:
        raise ModuleNotFoundError("h5py is required for clean716 H5 reconstruction; install the frozen runtime")


def _identity_records(manifest: object) -> list[dict[str, int | str]]:
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("manifest must be a non-empty list")
    records: list[dict[str, int | str]] = []
    seen: set[tuple[str, int]] = set()
    for trajectory, entry in enumerate(manifest):
        if not isinstance(entry, Mapping):
            raise ValueError("manifest trajectory must be an object")
        for key in ("file", "frames", "re", "aoa", "val_starts"):
            if key not in entry:
                raise ValueError(f"manifest trajectory misses {key}")
        starts = entry["val_starts"]
        if not isinstance(starts, list) or not all(isinstance(v, int) for v in starts):
            raise ValueError("manifest val_starts must be integer list")
        for start in starts:
            identity = (str(entry["file"]), start)
            if identity in seen:
                raise ValueError("manifest contains duplicate sample identity")
            seen.add(identity)
            records.append({"row": len(records), "trajectory": trajectory,
                            "file": str(entry["file"]), "start": start,
                            "frames": int(entry["frames"]), "re": int(entry["re"]),
                            "aoa": int(entry["aoa"])})
    if len(records) != EXPECTED_SHAPE[0]:
        raise ValueError("manifest does not define exactly clean716 rows")
    return records


def reconstruct_histories(*, real_dir: Path, manifest: object, aoa: np.ndarray,
                          reynolds: np.ndarray, trajectory: np.ndarray, fold: np.ndarray) -> tuple[np.ndarray, list[dict[str, int | str]], dict[str, object]]:
    """Decode only the 20 historical frames in manifest order, bitwise checked."""
    require_h5py(); records = _identity_records(manifest)
    if not real_dir.is_dir():
        raise FileNotFoundError(real_dir)
    names = [str(item["file"]) for item in manifest]
    sorted_selected = [p.name for p in sorted(real_dir.glob("*.h5")) if p.name in set(names)]
    if sorted_selected != names:
        raise ValueError("manifest trajectory order differs from raw loader order")
    if any(np.shape(value) != (len(records),) for value in (aoa, reynolds, trajectory, fold)):
        raise ValueError("cache labels do not align with manifest rows")
    histories = np.empty(EXPECTED_SHAPE, dtype=np.float32)
    row = 0
    for trajectory_index, entry in enumerate(manifest):
        path = real_dir / str(entry["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            for key in ("u", "v", "re", "aoa"):
                if key not in handle:
                    raise ValueError(f"{path.name} misses {key}")
            if int(handle["u"].shape[0]) != int(entry["frames"]):
                raise ValueError(f"frame count differs for {path.name}")
            if int(np.asarray(handle["aoa"]).reshape(-1)[0]) != int(entry["aoa"]):
                raise ValueError(f"AoA differs for {path.name}")
            if int(np.asarray(handle["re"]).reshape(-1)[0]) != int(entry["re"]):
                raise ValueError(f"Re differs for {path.name}")
            for start in entry["val_starts"]:
                if start < 0 or start + 40 > int(handle["u"].shape[0]):
                    raise ValueError(f"window leaves bounds for {path.name}:{start}")
                if (int(aoa[row]) != int(entry["aoa"]) or
                        int(reynolds[row]) != int(entry["re"]) or int(fold[row]) < 0 or int(fold[row]) > 4 or
                        int(trajectory[row]) != trajectory_index):
                    raise ValueError(f"cache label/order differs at row {row}")
                u = np.asarray(handle["u"][start:start + 20, ::2, ::2], dtype=np.float32)
                v = np.asarray(handle["v"][start:start + 20, ::2, ::2], dtype=np.float32)
                decoded = np.ascontiguousarray(np.stack((u, v), axis=-1))
                if decoded.shape != EXPECTED_SHAPE[1:]:
                    raise ValueError(f"unexpected downsampled history shape at row {row}")
                histories[row] = decoded
                row += 1
    if row != len(records) or not np.isfinite(histories).all():
        raise FloatingPointError("history reconstruction failed finite/order check")
    return histories, records, {"passed": True, "windows": row,
        "trajectory_order": "manifest list equals sorted raw H5 intersection",
        "window_order": "manifest list then val_starts list",
        "decode": "float32(stack(u[start:start+20,::2,::2],v[...]),last_axis)",
        "history_raw_array_sha256": hashlib.sha256(histories.tobytes()).hexdigest()}


class TargetAccessor:
    """Fail-closed cache/H5 target access gate used by both experiment phases."""
    def __init__(self, *, target_path: Path, real_dir: Path,
                 identities: list[dict[str, int | str]], held: np.ndarray) -> None:
        self.target_path, self.real_dir, self.identities = target_path, real_dir, identities
        self.held = frozenset(int(row) for row in held)
        self.terminal_unlocked = False
        self.held_target_access_counter = 0
        self.held_target_access_attempts = 0

    def unlock_terminal(self) -> None:
        self.terminal_unlocked = True

    def _authorize(self, rows: np.ndarray) -> None:
        if any(int(row) in self.held for row in rows):
            if not self.terminal_unlocked:
                self.held_target_access_attempts += 1
                raise PermissionError("held target access is forbidden before head seal")
            self.held_target_access_counter += int(sum(int(row) in self.held for row in rows))

    def cache(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64); self._authorize(rows)
        target = np.load(self.target_path, mmap_mode="r")
        return np.array(target[rows], copy=True)

    def h5(self, rows: np.ndarray) -> np.ndarray:
        require_h5py(); rows = np.asarray(rows, dtype=np.int64); self._authorize(rows)
        decoded = np.empty((len(rows), *EXPECTED_SHAPE[1:]), dtype=np.float32)
        for output_row, row in enumerate(rows):
            identity = self.identities[int(row)]; path = self.real_dir / str(identity["file"])
            start = int(identity["start"])
            with h5py.File(path, "r") as handle:
                # This is intentionally the future half only, and only after authorization.
                u = np.asarray(handle["u"][start + 20:start + 40, ::2, ::2], dtype=np.float32)
                v = np.asarray(handle["v"][start + 20:start + 40, ::2, ::2], dtype=np.float32)
            decoded[output_row] = np.stack((u, v), axis=-1)
        return decoded

    def report(self) -> dict[str, int | bool]:
        return {"terminal_unlocked": self.terminal_unlocked,
                "held_target_access_counter": self.held_target_access_counter,
                "held_target_access_attempts": self.held_target_access_attempts}


def verify_target_alignment(accessor: TargetAccessor, rows: np.ndarray, label: str) -> dict[str, object]:
    h5_target, cache_target = accessor.h5(rows), accessor.cache(rows)
    if not np.array_equal(h5_target, cache_target):
        difference = float(np.max(np.abs(h5_target.astype(np.float64) - cache_target.astype(np.float64))))
        raise ValueError(f"{label} H5/cache target mismatch max_abs={difference}")
    return {"phase": label, "windows": int(len(rows)), "comparison": "bitwise_array_equal_per_window",
            "target_raw_array_sha256": hashlib.sha256(cache_target.tobytes()).hexdigest(), "passed": True}


def fit_held_indices(aoa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(aoa)
    if set(np.unique(labels).tolist()) != {0, 5, 10, 15, 20}:
        raise ValueError("clean716 AoA labels changed")
    fit, held = np.flatnonzero(labels != HELD_AOA), np.flatnonzero(labels == HELD_AOA)
    if len(fit) != 554 or len(held) != 162 or np.intersect1d(fit, held).size:
        raise RuntimeError("frozen held10 split is not 554/162 disjoint")
    if set(np.unique(labels[fit]).tolist()) != set(FIT_AOAS):
        raise RuntimeError("fit labels differ from frozen contract")
    return fit.astype(np.int64), held.astype(np.int64)


def validate_fold_assignments(metadata: Mapping[str, Any],
                              identities: list[dict[str, int | str]],
                              folds: np.ndarray) -> dict[str, object]:
    assignments = metadata.get("fold_assignments")
    if not isinstance(assignments, list):
        raise ValueError("metadata fold assignments are missing")
    by_file: dict[str, Mapping[str, Any]] = {}
    for item in assignments:
        if not isinstance(item, Mapping):
            raise ValueError("fold assignment must be an object")
        filename = str(item.get("file", ""))
        if not filename or filename in by_file:
            raise ValueError("fold assignment filenames must be unique")
        by_file[filename] = item
    values = np.asarray(folds)
    if values.shape != (len(identities),):
        raise ValueError("fold cache does not align with sample identities")
    for row, identity in enumerate(identities):
        item = by_file.get(str(identity["file"]))
        if item is None:
            raise ValueError("manifest file missing from fold assignments")
        if (int(values[row]) != int(item["fold"])
                or int(identity["aoa"]) != int(item["aoa"])
                or int(identity["re"]) != int(item["re"])):
            raise ValueError(f"fold assignment differs at sample row {row}")
    if set(np.unique(values).tolist()) != {0, 1, 2, 3, 4}:
        raise ValueError("fold cache must contain exactly folds 0--4")
    return {"passed": True, "samples": len(identities),
            "trajectories": len(by_file),
            "fold_counts": {str(value): int((values == value).sum())
                            for value in range(5)},
            "assignments_canonical_sha256": canonical_sha256(assignments)}


def make_features(history: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    if history.ndim != 5 or base.ndim != 5 or history.shape[1:] != (20, 32, 64, 2) or base.shape[1:] != (20, 32, 64, 2):
        raise ValueError("history/base must be B,20,32,64,2")
    last = history[:, -6:]
    delta = (last[:, 1:] - last[:, :-1]).permute(0, 1, 4, 2, 3).reshape(len(last), 10, 32, 64)
    mean = last.mean(1).permute(0, 3, 1, 2)
    std = last.std(1, unbiased=False).permute(0, 3, 1, 2)
    blocks = base.reshape(len(base), 4, 5, 32, 64, 2).mean(2).permute(0, 1, 4, 2, 3).reshape(len(base), 8, 32, 64)
    feature = torch.cat((delta, mean, std, blocks), dim=1)
    if feature.shape[1:] != (22, 32, 64):
        raise RuntimeError("frozen 22-channel feature construction failed")
    return feature


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, 32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, 32)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = F.silu(self.norm1(self.conv1(value)))
        return F.silu(value + self.norm2(self.conv2(residual)))


class CoarseFlowHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(22, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool2d((4, 8))
        self.blocks = nn.Sequential(ResidualBlock(), ResidualBlock())
        self.head = nn.Conv2d(32, 8, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        output = self.head(self.blocks(self.pool(self.stem(feature))))
        return output.reshape(len(output), 4, 2, 4, 8)


def optimize_raw_block_oracle(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact frozen smooth-oracle algorithm, returning raw controls or exact zero."""
    batch, _steps, height, width, channels = source.shape
    if source.shape != target.shape or channels != 2:
        raise ValueError("teacher source/target must align UV")
    raw = torch.zeros((batch, 2, 4, 8), device=source.device, dtype=source.dtype, requires_grad=True)
    identity_sse_raw = (source - target).square().sum((1, 2, 3, 4)).detach()
    optimizer = torch.optim.Adam((raw,), lr=smooth.ADAM_LR)
    for _ in range(smooth.ADAM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        warped = smooth.smooth_flow_warp(source, smooth.displacement_from_raw(raw, height, width))
        loss = ((warped - target).square().sum((1, 2, 3, 4)) /
                identity_sse_raw.clamp_min(1.0e-12) + smooth.TV_WEIGHT * smooth.control_tv(raw)).mean()
        loss.backward(); optimizer.step()
    with torch.no_grad():
        final_raw = raw.detach()
        warped = smooth.smooth_flow_warp(source, smooth.displacement_from_raw(final_raw, height, width))
        choose = (warped - target).square().sum((1, 2, 3, 4)) < identity_sse_raw
        return torch.where(choose[:, None, None, None], final_raw, torch.zeros_like(final_raw))


def build_fit_teacher_controls(*, base: np.ndarray, target: np.ndarray,
                               fit: np.ndarray, device: torch.device,
                               progress: Any = None) -> np.ndarray:
    controls = np.empty((len(fit), 4, 2, 4, 8), dtype=np.float32)
    for start in range(0, len(fit), TEACHER_BATCH_SIZE):
        stop = min(start + TEACHER_BATCH_SIZE, len(fit)); rows = fit[start:stop]
        source = torch.from_numpy(np.array(base[rows], copy=True)).to(device)
        # ``target`` is deliberately fit-local, so a global held row cannot be
        # accidentally addressed from inside the teacher loop.
        truth = torch.from_numpy(np.array(target[start:stop], copy=True)).to(device)
        if not (torch.isfinite(source).all() and torch.isfinite(truth).all()):
            raise FloatingPointError("fit base/target non-finite")
        for block, (left, right) in enumerate(integer.HORIZON_BLOCKS):
            controls[start:stop, block] = optimize_raw_block_oracle(source[:, left:right], truth[:, left:right]).cpu().numpy()
        if progress is not None and (start == 0 or stop == len(fit)
                                     or stop % 160 == 0):
            progress(stop, len(fit))
    if not np.isfinite(controls).all():
        raise FloatingPointError("teacher controls non-finite")
    return controls


def candidate_from_raw(base: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    # Required update-0 identity: the zero-initialized head takes the exact
    # bypass rather than a numerically-close grid_sample identity path.
    if not bool(torch.count_nonzero(raw.detach())):
        return base.clone()
    candidate = base.clone()
    for block, (left, right) in enumerate(integer.HORIZON_BLOCKS):
        candidate[:, left:right] = smooth.smooth_flow_warp(
            base[:, left:right], smooth.displacement_from_raw(raw[:, block], 32, 64))
    return candidate


def candidate_comparison(baseline_errors: np.ndarray, candidate_errors: np.ndarray) -> dict[str, object]:
    baseline, candidate = integer.metrics_from_errors(baseline_errors), integer.metrics_from_errors(candidate_errors)
    keys = ("rel_l2_score", "tke_score", "mvpe_score", "point_final_contribution_46_10_9", "point_score_normalized_46_10_9")
    return {"baseline": baseline, "candidate": candidate,
            "delta_candidate_minus_baseline": {key: float(candidate[key] - baseline[key]) for key in keys}}


def raw_flow_summary(raw: np.ndarray) -> dict[str, float | int]:
    """Deployment-flow diagnostics; no target is involved."""
    tensor = torch.from_numpy(np.asarray(raw, dtype=np.float32))
    displacement = smooth.displacement_from_raw(tensor.reshape(-1, 2, 4, 8), 32, 64).numpy()
    magnitude = np.linalg.norm(displacement.astype(np.float64), axis=-1)
    mean = displacement.mean(axis=(1, 2))
    return {"samples": int(len(raw)), "nonzero_control_fraction": float(np.any(raw != 0.0, axis=(1, 2, 3, 4)).mean()),
            "tanh_saturation_fraction_abs_raw_ge_3": float((np.abs(raw) >= 3.0).mean()),
            "mean_displacement_magnitude_pixels": float(magnitude.mean()),
            "max_displacement_magnitude_pixels": float(magnitude.max()),
            "spatial_mean_displacement_abs_mean_pixels": float(np.abs(mean).mean())}


def output_mean_drift(base: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if base.shape != candidate.shape or base.ndim != 5:
        raise ValueError("base/candidate output drift arrays must align")
    drift = (candidate.astype(np.float64) - base.astype(np.float64)).mean(
        axis=(2, 3))
    return {"absolute_mean": float(np.abs(drift).mean()),
            "absolute_max": float(np.abs(drift).max()),
            "signed_mean": float(drift.mean())}


def grouped_rollout(rows: np.ndarray, identities: list[dict[str, int | str]], baseline_errors: np.ndarray,
                    candidate_errors: np.ndarray) -> dict[str, dict[str, object]]:
    groups: dict[str, list[int]] = {}
    for local, row in enumerate(rows):
        identity = identities[int(row)]
        for label in (f"trajectory:{identity['file']}", f"re:{identity['re']}"):
            groups.setdefault(label, []).append(local)
    return {label: candidate_comparison(baseline_errors[np.asarray(local)], candidate_errors[np.asarray(local)])
            for label, local in groups.items()}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def seal_head(head: nn.Module, output_dir: Path) -> dict[str, object]:
    path = output_dir / "head_state.pt"
    torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "sealed_before_held_target_access": True}


def reload_sealed_head(seal: Mapping[str, object],
                       device: torch.device) -> CoarseFlowHead:
    path = Path(str(seal["path"]))
    if sha256_file(path) != str(seal["sha256"]):
        raise RuntimeError("sealed head hash changed before reload")
    state = torch.load(path, map_location="cpu", weights_only=True)
    head = CoarseFlowHead()
    head.load_state_dict(state, strict=True)
    return head.to(device).eval()


def _environment(device: torch.device) -> str:
    return "".join(f"{key}={value}\n" for key, value in {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.replace("\n", " "),
        "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "h5py": h5py.__version__ if h5py is not None else "unavailable", "device": str(device),
        "cuda_available": torch.cuda.is_available(), "pid": os.getpid()}.items())


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run.log"
    def log(event: str, **values: object) -> None:
        row = json.dumps({"event": event, **values}, ensure_ascii=False, sort_keys=True)
        print(row, flush=True)
        with log_path.open("a", encoding="utf-8") as handle: handle.write(row + "\n")
    paths, metadata, manifest = integer.validate_inputs(args.cache_dir)
    frozen_sources = verify_frozen_sources(metadata)
    extra = {name: args.cache_dir / f"{name}.npy" for name in ("re", "trajectory", "fold")}
    if any(not path.is_file() for path in extra.values()): raise FileNotFoundError(extra)
    device = resolve_device(args.device)
    prediction = np.load(paths["prediction"], mmap_mode="r")
    aoa, reynolds, trajectory, fold = (np.load((paths if name == "aoa" else extra)[name], mmap_mode="r") for name in ("aoa", "re", "trajectory", "fold"))
    if tuple(prediction.shape) != EXPECTED_SHAPE or prediction.dtype != np.float32: raise ValueError("base shape/dtype changed")
    if sha256_file(paths["prediction"]) != EXPECTED_PREDICTION_SHA256: raise ValueError("base prediction SHA changed")
    base_sha_before = sha256_file(paths["prediction"])
    histories, identities, history_report = reconstruct_histories(real_dir=args.real_dir, manifest=manifest, aoa=aoa, reynolds=reynolds, trajectory=trajectory, fold=fold)
    fold_report = validate_fold_assignments(metadata, identities, fold)
    fit, held = fit_held_indices(aoa)
    target_accessor = TargetAccessor(target_path=paths["target"], real_dir=args.real_dir,
                                     identities=identities, held=held)
    config = {"schema": SCHEMA, "held_aoa": HELD_AOA, "fit_aoas": list(FIT_AOAS), "shape": list(EXPECTED_SHAPE),
        "history": {"frames": 20, "last_frames_used": 6, "downsample": "::2", "channels": 22},
        "teacher": {"algorithm": "frozen smooth oracle helpers", "fit_only": True, "batch_size": TEACHER_BATCH_SIZE,
                    "raw_shape": [4, 2, 4, 8], "adam_steps": smooth.ADAM_STEPS, "adam_lr": smooth.ADAM_LR, "tv_weight": smooth.TV_WEIGHT},
        "student": "Conv3x3(22,32)+GN4+SiLU+AdaptiveAvgPool(4,8)+2x residual Conv3x3(32)+zero Conv1x1(32,8)",
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-5, "batch_size": BATCH_SIZE, "updates": UPDATES, "seed": SEED, "sampling": "uniform with replacement"},
        "evidence_scope": "controller-only AoA transfer on frozen full-AoA baseline; not end-to-end unseen-AoA generalization",
        "held_target_policy": "fail-closed accessor rejects held before head state SHA seal; terminal phase only", "warp": {"mode": "bilinear", "padding_mode": "border", "align_corners": True},
        "input_prediction_sha256": EXPECTED_PREDICTION_SHA256, "input_target_sha256": EXPECTED_TARGET_SHA256}
    write_json(args.output_dir / "config.json", config); (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n"); (args.output_dir / "environment.txt").write_text(_environment(device))
    log("frozen_sources_verified", **frozen_sources)
    log("history_reconstructed", **history_report)
    log("fold_assignments_verified", **fold_report)
    log("fit_held_split", fit_samples=len(fit), held_samples=len(held))
    # The accessor allows only fit rows in this phase.  It verifies every fit H5
    # future against the cache before the frozen teacher sees those labels.
    fit_target_alignment = verify_target_alignment(target_accessor, fit, "fit_pretraining")
    fit_target = target_accessor.cache(fit)
    controls = build_fit_teacher_controls(
        base=prediction, target=fit_target, fit=fit, device=device,
        progress=lambda complete, total: log(
            "fit_teacher_progress", samples_complete=complete,
            samples_total=total))
    np.save(args.output_dir / "fit_oracle_raw_controls.npy", controls)
    fit_identities = [identities[int(row)] for row in fit]
    write_json(args.output_dir / "fit_sample_identities.json", fit_identities)
    teacher_cache = {"controls": {"shape": list(controls.shape), "dtype": str(controls.dtype), "sha256": sha256_file(args.output_dir / "fit_oracle_raw_controls.npy")},
        "fit_identities_sha256": canonical_sha256(fit_identities), "teacher_config_sha256": canonical_sha256(config["teacher"]), "held_controls_written": False,
        "identity_choice_blocks": int(np.all(controls == 0.0, axis=(2, 3, 4)).sum()),
        "control_finite": bool(np.isfinite(controls).all()), "control_abs_max": float(np.abs(controls).max())}
    write_json(args.output_dir / "teacher_cache.json", teacher_cache); log("fit_teacher_cached", **teacher_cache)
    # No held H5 future or cache target has been requested; this assertion is a
    # hard phase barrier, not merely an audit field.
    if target_accessor.held_target_access_counter or target_accessor.held_target_access_attempts:
        raise RuntimeError("held target accessor was touched before training")
    preseal_target_accessor = target_accessor.report()
    torch.manual_seed(SEED); np_rng = np.random.default_rng(SEED)
    head = CoarseFlowHead().to(device); optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
    with torch.no_grad():
        zero_raw = head(make_features(torch.from_numpy(histories[fit[:2]]).to(device), torch.from_numpy(np.array(prediction[fit[:2]], copy=True)).to(device)))
        update0_identity = bool(torch.equal(candidate_from_raw(torch.from_numpy(np.array(prediction[fit[:2]], copy=True)).to(device), zero_raw), torch.from_numpy(np.array(prediction[fit[:2]], copy=True)).to(device)))
    if not update0_identity: raise RuntimeError("zero-init update0 is not exact base identity")
    losses: list[float] = []; stream_digest = hashlib.sha256()
    for update in range(UPDATES):
        local = np_rng.integers(0, len(fit), size=BATCH_SIZE, endpoint=False, dtype=np.int64); stream_digest.update(local.tobytes())
        rows = fit[local]
        feature = make_features(torch.from_numpy(histories[rows]).to(device), torch.from_numpy(np.array(prediction[rows], copy=True)).to(device))
        teacher = torch.from_numpy(controls[local]).to(device)
        optimizer.zero_grad(set_to_none=True); loss = F.mse_loss(head(feature), teacher); loss.backward(); optimizer.step()
        if not torch.isfinite(loss): raise FloatingPointError("non-finite student loss")
        losses.append(float(loss.detach().cpu()))
        if update == 0 or (update + 1) % 250 == 0 or update + 1 == UPDATES:
            log("student_progress", update=update + 1, updates=UPDATES,
                loss=losses[-1])
    history = {"updates": UPDATES, "losses": losses, "sample_stream_sha256": stream_digest.hexdigest(), "seed": SEED}
    write_json(args.output_dir / "training_history.json", history)
    head_seal = seal_head(head, args.output_dir); base_sha_seal = sha256_file(paths["prediction"])
    if base_sha_seal != base_sha_before: raise RuntimeError("base prediction mutated before seal")
    log("head_sealed", **head_seal)
    del optimizer, head
    eval_head = reload_sealed_head(head_seal, device)
    log("sealed_head_reloaded", sha256=head_seal["sha256"])
    # The independent terminal phase begins only after the state file + SHA exist.
    target_accessor.unlock_terminal()
    held_target_alignment = verify_target_alignment(target_accessor, held, "terminal_held")
    held_target = target_accessor.cache(held)
    held_base = np.array(prediction[held], copy=True); held_history = histories[held]
    fit_base = np.array(prediction[fit], copy=True); fit_history = histories[fit]
    with torch.inference_mode():
        raw = eval_head(make_features(torch.from_numpy(held_history).to(device), torch.from_numpy(held_base).to(device)))
        candidate = candidate_from_raw(torch.from_numpy(held_base).to(device), raw).cpu().numpy()
        fit_raw = eval_head(make_features(torch.from_numpy(fit_history).to(device), torch.from_numpy(fit_base).to(device)))
        fit_candidate = candidate_from_raw(torch.from_numpy(fit_base).to(device), fit_raw).cpu().numpy()
    finite = bool(np.isfinite(candidate).all() and np.isfinite(held_target).all())
    baseline_errors, candidate_errors = integer.official_error_triples(held_base, held_target), integer.official_error_triples(candidate, held_target)
    metrics = candidate_comparison(baseline_errors, candidate_errors); delta = metrics["delta_candidate_minus_baseline"]
    fit_baseline_errors = integer.official_error_triples(fit_base, fit_target)
    fit_candidate_errors = integer.official_error_triples(fit_candidate, fit_target)
    fit_raw_mse = float(np.mean((fit_raw.cpu().numpy().astype(np.float64) - controls.astype(np.float64)) ** 2))
    checks = {"update0_identity": update0_identity, "fit_held_disjoint": not bool(np.intersect1d(fit, held).size),
        "preseal_held_target_counter_zero": (preseal_target_accessor["held_target_access_counter"] == 0 and preseal_target_accessor["held_target_access_attempts"] == 0),
        "base_immutable": base_sha_before == base_sha_seal == sha256_file(paths["prediction"]), "finite": finite,
        "delta_point_ge_plus_0p03": float(delta["point_final_contribution_46_10_9"]) >= .03, "delta_rel_ge_0": float(delta["rel_l2_score"]) >= 0.0}
    archive_path = Path(str(metadata["archive"]["path"]))
    input_paths = {"prediction": paths["prediction"], "target": paths["target"], "aoa": paths["aoa"], "re": extra["re"], "fold": extra["fold"], "metadata": paths["metadata"],
                   "trajectory": extra["trajectory"], "manifest": paths["split_manifest"], "smooth_oracle_script": SMOOTH_PATH,
                   "integer_scoring_script": INTEGER_PATH, "official_scoring": integer.SCORING_PATH,
                   "source_archive": archive_path, "experiment_script": Path(__file__).resolve()}
    result = {"schema": SCHEMA, "evidence_scope": config["evidence_scope"], "terminal_held_aoa": HELD_AOA, "fit_samples": len(fit), "held_samples": len(held), "config_canonical_sha256": canonical_sha256(config),
        "git_commit": git_commit(), "frozen_sources": frozen_sources,
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in input_paths.items()},
        "history_reconstruction": history_report, "fold_assignments": fold_report,
        "fit_target_alignment": fit_target_alignment, "terminal_held_target_alignment": held_target_alignment,
        "preseal_target_accessor": preseal_target_accessor, "target_accessor": target_accessor.report(), "teacher_cache": teacher_cache, "head_seal": head_seal,
        "base_prediction_sha256_before_training": base_sha_before, "base_prediction_sha256_at_seal": base_sha_seal,
        "target_sha256_terminal": sha256_file(paths["target"]), "fit_raw_mse": fit_raw_mse,
        "fit_rollout": {"official_point_metrics": candidate_comparison(fit_baseline_errors, fit_candidate_errors),
                        "flow": raw_flow_summary(fit_raw.cpu().numpy()),
                        "output_spatial_mean_drift": output_mean_drift(fit_base, fit_candidate)},
        "terminal_held_score": {"official_point_metrics": metrics, "flow": raw_flow_summary(raw.cpu().numpy()),
                                 "output_spatial_mean_drift": output_mean_drift(held_base, candidate),
                                 "trajectory_and_re_aggregates": grouped_rollout(held, identities, baseline_errors, candidate_errors)},
        "checks": checks, "decision": "GO_FULL_LOAO_STRICT" if all(checks.values()) else "STOP_THIS_COARSE_FLOW_DISTILLATION"}
    if result["target_sha256_terminal"] != EXPECTED_TARGET_SHA256: raise RuntimeError("target SHA changed at terminal score")
    write_json(args.output_dir / "result.json", result); log("terminal_complete", decision=result["decision"], checks=checks)
    outputs = sorted(path for path in args.output_dir.iterdir() if path.name != "sha256.txt")
    lines = [f"{sha256_file(path)}  output:{path.name}" for path in outputs if path.is_file()]
    (args.output_dir / "sha256.txt").write_text("\n".join(lines) + "\n")
    return result


def smoke() -> None:
    torch.manual_seed(1); history = torch.randn(3, 20, 32, 64, 2); base = torch.randn_like(history)
    head = CoarseFlowHead(); feature = make_features(history, base); raw = head(feature)
    assert feature.shape == (3, 22, 32, 64) and raw.shape == (3, 4, 2, 4, 8)
    assert torch.equal(candidate_from_raw(base, raw), base)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
    optimizer.zero_grad(); F.mse_loss(head(feature), torch.ones_like(raw)).backward(); optimizer.step()
    print(json.dumps({"event": "b210_history_coarse_flow_smoke_complete", "target_access": False}))


def main() -> None:
    args = parse_args()
    if args.smoke: smoke()
    else: print(json.dumps({"event": "b210_history_coarse_flow_complete", "decision": run(args)["decision"]}))


if __name__ == "__main__": main()
