#!/usr/bin/env python3
"""Fit-side, history-only identifiability audit for lag-consistent velocity.

This is deliberately *not* a future-label experiment.  It opens only the
first 20 frames of windows belonging to AoA 0/5/15/20 and performs a primary
internal multi-step check ``X_16 -> X_(16+h)`` for h=1/2/3, with repeated
one-step checks at s=17/18.  In particular,
AoA 10 files are rejected before an H5 file is opened and no slice ending
after ``window_start + 20`` is legal.

The audit is the small preflight for a possible latent semi-Lagrangian model:
offsets from lags 1/4/8/16 are converted to a common velocity unit (feature
pixels per frame), fused as a confidence-weighted mixture posterior, and
used only to warp an observed history frame.  It has no trainable state.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np

# Required by torch deterministic CuBLAS paths; set before importing torch or
# creating a CUDA context.  This is a mechanical reproducibility setting and
# does not alter any registered scientific constant.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "b210_lag_consistent_velocity_posterior_history_only_v1"
FIT_AOAS = (0, 5, 15, 20)
FORBIDDEN_AOA = 10
HISTORY_STEPS = 20
PRIMARY_TERMINAL = 16
PRIMARY_HORIZONS = (1, 2, 3)
REPEAT_ONE_STEP_TERMINALS = (17, 18)
TERMINALS = (PRIMARY_TERMINAL, *REPEAT_ONE_STEP_TERMINALS)
LAGS = (1, 4, 8, 16)
RADIUS = 4
BETA = 6.0
FEATURE_DOWNSAMPLE = 2
FEATURE_SHAPE = (16, 32)
FIELD_SHAPE = (32, 64)
ALPHA_EPS = 1.0e-3
GAUSSIAN_3X3 = torch.tensor(((1., 2., 1.), (2., 4., 2.), (1., 2., 1.))) / 16.
# This is a prospective M8 geometry limit, not a fitted clipping parameter.
H20_CAP_FEATURE_PIXELS = 1.0
ONE_STEP_CAP_FEATURE_PIXELS = H20_CAP_FEATURE_PIXELS / 20.0
WARP_SANITY_TOLERANCE = 3.0e-5


@dataclass(frozen=True)
class Window:
    """A legal history window and its deterministic reporting fold."""

    file: str
    start: int
    aoa: int
    reynolds: int
    trajectory: int
    fold: int
    window_ordinal: int


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def git_report() -> dict[str, object]:
    def call(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", *arguments), cwd=REPO_ROOT, text=True).strip()
    porcelain = call("status", "--porcelain")
    return {"commit": call("rev-parse", "HEAD"),
            "branch": call("branch", "--show-current"),
            "clean": not bool(porcelain), "porcelain": porcelain.splitlines()}


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "h5py is required only for the real-history audit; use the project "
            "runtime or run --smoke") from exc
    return h5py


def load_manifest(path: Path) -> list[Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("manifest must be a non-empty list")
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ValueError("manifest entry must be an object")
        for key in ("file", "frames", "re", "aoa", "val_starts"):
            if key not in entry:
                raise ValueError(f"manifest entry lacks {key}")
    return value


def fit_windows(manifest: Iterable[Mapping[str, Any]]) -> list[Window]:
    """Build four reporting folds without opening any field file.

    The ordinal is calculated separately for every AoA from sorted Reynolds
    numbers.  The fold is a trajectory property, so its windows cannot leak
    into a different reporting fold.
    """
    selected: list[tuple[int, Mapping[str, Any]]] = []
    for trajectory, entry in enumerate(manifest):
        aoa = int(entry["aoa"])
        if aoa == FORBIDDEN_AOA:
            continue
        if aoa not in FIT_AOAS:
            raise ValueError(f"unexpected AoA {aoa}; audit admits only {FIT_AOAS}")
        selected.append((trajectory, entry))
    if not selected:
        raise ValueError("manifest has no fit-side trajectories")
    per_aoa: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for item in selected:
        per_aoa[int(item[1]["aoa"])].append(item)
    if set(per_aoa) != set(FIT_AOAS):
        raise ValueError("all four fit AoAs are required")
    trajectory_fold: dict[int, int] = {}
    for aoa, items in per_aoa.items():
        ordered = sorted(items, key=lambda item: (int(item[1]["re"]), str(item[1]["file"])))
        if len(ordered) < 4:
            raise ValueError(f"AoA {aoa} has fewer than four trajectories")
        for ordinal, (trajectory, _entry) in enumerate(ordered):
            trajectory_fold[trajectory] = ordinal % 4
    rows: list[Window] = []
    for trajectory, entry in selected:
        starts = entry["val_starts"]
        if not isinstance(starts, list) or not starts:
            raise ValueError("fit trajectory requires non-empty integer val_starts")
        frames = int(entry["frames"])
        for ordinal, start_raw in enumerate(starts):
            start = int(start_raw)
            if start < 0 or start + HISTORY_STEPS > frames:
                raise ValueError(f"history exceeds manifest bounds: {entry['file']}:{start}")
            rows.append(Window(str(entry["file"]), start, int(entry["aoa"]),
                               int(entry["re"]), trajectory,
                               trajectory_fold[trajectory], ordinal))
    if any(row.aoa == FORBIDDEN_AOA for row in rows):
        raise RuntimeError("AoA10 was admitted into the history audit")
    return rows


def load_histories(*, real_dir: Path, manifest: Iterable[Mapping[str, Any]],
                   windows: list[Window]) -> np.ndarray:
    """Read exactly [start,start+20) u/v samples for fit-side windows only."""
    h5py = _require_h5py()
    manifest_rows = list(manifest)
    by_file = {str(entry["file"]): entry for entry in manifest_rows}
    if len(by_file) != len(manifest_rows):
        # The iterator has normally been a list.  Keep this guard explicit for
        # callers passing a different iterable and malformed duplicate files.
        raise ValueError("manifest file names must be unique")
    histories = np.empty((len(windows), HISTORY_STEPS, *FIELD_SHAPE, 2), dtype=np.float32)
    for index, window in enumerate(windows):
        if window.aoa not in FIT_AOAS or window.aoa == FORBIDDEN_AOA:
            raise PermissionError("refusing to open non-fit/AoA10 field")
        entry = by_file.get(window.file)
        if entry is None or int(entry["aoa"]) != window.aoa:
            raise ValueError("window/manifest identity mismatch")
        path = real_dir / window.file
        if not path.is_file():
            raise FileNotFoundError(path)
        # The slice endpoint is deliberately stated in terms of HISTORY_STEPS;
        # do not refactor it into any future/label access path.
        stop = window.start + HISTORY_STEPS
        if stop > int(entry["frames"]):
            raise ValueError("requested frame lies outside the 20-frame history")
        with h5py.File(path, "r") as handle:
            if "u" not in handle or "v" not in handle:
                raise ValueError(f"{path} lacks u/v")
            u = np.asarray(handle["u"][window.start:stop, ::2, ::2], dtype=np.float32)
            v = np.asarray(handle["v"][window.start:stop, ::2, ::2], dtype=np.float32)
        if u.shape != (HISTORY_STEPS, *FIELD_SHAPE) or v.shape != u.shape:
            raise ValueError(
                f"{path.name}:{window.start} expected downsampled "
                f"{(HISTORY_STEPS, *FIELD_SHAPE)}, got {u.shape}/{v.shape}")
        histories[index] = np.stack((u, v), axis=-1)
    if not np.isfinite(histories).all():
        raise FloatingPointError("non-finite fit history")
    return histories


def history_features(history: torch.Tensor) -> torch.Tensor:
    """[u,v,dxu,dyu,dxv,dyv], then fixed 2x2 average to 16x32."""
    if history.ndim != 5 or tuple(history.shape[1:]) != (HISTORY_STEPS, *FIELD_SHAPE, 2):
        raise ValueError("history must be [B,20,32,64,2]")
    u, v = history[..., 0], history[..., 1]
    dxu = F.pad(u[..., 1:] - u[..., :-1], (0, 1, 0, 0))
    dyu = F.pad(u[..., 1:, :] - u[..., :-1, :], (0, 0, 0, 1))
    dxv = F.pad(v[..., 1:] - v[..., :-1], (0, 1, 0, 0))
    dyv = F.pad(v[..., 1:, :] - v[..., :-1, :], (0, 0, 0, 1))
    features = torch.stack((u, v, dxu, dyu, dxv, dyv), dim=2)
    batch, steps, channels, height, width = features.shape
    pooled = F.avg_pool2d(features.reshape(batch * steps, channels, height, width),
                          kernel_size=FEATURE_DOWNSAMPLE, stride=FEATURE_DOWNSAMPLE)
    return pooled.reshape(batch, steps, channels, *FEATURE_SHAPE)


def history_feature_mask(history: torch.Tensor) -> torch.Tensor:
    """Cells whose complete pooling/difference stencil is measured.

    ``history_features`` applies forward x/y differences before 2x2 pooling,
    so an interior output cell depends on a 3x3 raw-field footprint.  The
    right/bottom replicate only locks the outer boundary convention.
    """
    if history.ndim != 5 or tuple(history.shape[1:]) != (HISTORY_STEPS, *FIELD_SHAPE, 2):
        raise ValueError("history must be [B,20,32,64,2]")
    batch, steps = history.shape[:2]
    measured = history.abs().amax(dim=-1).gt(1.0e-12).float()
    measured = measured.reshape(batch * steps, 1, *FIELD_SHAPE)
    stencil = F.avg_pool2d(F.pad(measured, (0, 1, 0, 1), mode="replicate"), 3, 2)
    return stencil.reshape(batch, steps, 1, *FEATURE_SHAPE).eq(1.)


def crossfit_feature_statistics(histories: np.ndarray, windows: list[Window],
                                device: torch.device, chunk_size: int,
                                ) -> dict[int, dict[str, torch.Tensor]]:
    """Per-channel mean/std for each fold, fit on the other three folds."""
    if histories.shape != (len(windows), HISTORY_STEPS, *FIELD_SHAPE, 2):
        raise ValueError("history array/window metadata shape mismatch")
    sums = np.zeros((4, 6), dtype=np.float64)
    squares = np.zeros((4, 6), dtype=np.float64)
    counts = np.zeros(4, dtype=np.float64)
    with torch.inference_mode():
        for begin in range(0, len(windows), chunk_size):
            end = min(begin + chunk_size, len(windows))
            history = torch.from_numpy(np.ascontiguousarray(histories[begin:end])).to(device)
            feature = history_features(history).double()
            mask = history_feature_mask(history).double()
            for local, row in enumerate(windows[begin:end]):
                value, weight = feature[local], mask[local]
                sums[row.fold] += (value * weight).sum((0, 2, 3)).cpu().numpy()
                squares[row.fold] += (value.square() * weight).sum((0, 2, 3)).cpu().numpy()
                counts[row.fold] += float(weight.sum().cpu())
    total_sum, total_squares, total_count = sums.sum(0), squares.sum(0), counts.sum()
    result: dict[int, dict[str, torch.Tensor]] = {}
    for held_fold in range(4):
        count = total_count - counts[held_fold]
        if count <= 0:
            raise RuntimeError("cross-fit feature statistics have no fit support")
        mean = (total_sum - sums[held_fold]) / count
        variance = (total_squares - squares[held_fold]) / count - np.square(mean)
        std = np.sqrt(np.maximum(variance, 1.0e-12))
        result[held_fold] = {
            "mean": torch.tensor(mean, device=device, dtype=torch.float32).view(1, 1, 6, 1, 1),
            "std": torch.tensor(std, device=device, dtype=torch.float32).view(1, 1, 6, 1, 1),
            "fit_measured_cells": torch.tensor(count, dtype=torch.float64),
        }
    return result


def standardize_features(feature: torch.Tensor, mask: torch.Tensor,
                         statistics: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if feature.ndim != 5 or mask.shape != feature.shape[:2] + (1,) + feature.shape[3:]:
        raise ValueError("feature/mask shape mismatch")
    mean = statistics["mean"].to(device=feature.device, dtype=feature.dtype)
    std = statistics["std"].to(device=feature.device, dtype=feature.dtype)
    return ((feature - mean) / std.clamp_min(1.0e-6)) * mask.to(feature.dtype)


def offsets(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Backtrace displacement convention (x,y), matching grid_sample.
    values = [(dx, dy) for dy in range(-RADIUS, RADIUS + 1)
              for dx in range(-RADIUS, RADIUS + 1)]
    return torch.tensor(values, device=device, dtype=dtype)


def _patch_descriptor(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed 3x3 descriptor and an exact nonzero/PIV support mask.

    A single six-vector is too ambiguous in the wake and would assign an
    arbitrary direction to exact all-zero solid/PIV areas.  The descriptor is
    fixed (no learned matching stem): 3x3 neighbourhoods of the prescribed
    six features.  ``support`` is deliberately based on the unnormalised
    descriptor and is applied *after* smoothing so no neighbouring velocity
    leaks into an exact zero-support region.
    """
    if value.ndim != 4 or tuple(value.shape[1:]) != (6, *FEATURE_SHAPE):
        raise ValueError("descriptor input must be [B,6,16,32]")
    batch, channels, height, width = value.shape
    patch = F.unfold(F.pad(value, (1, 1, 1, 1), mode="replicate"), kernel_size=3)
    patch = patch.reshape(batch, channels * 9, height, width)
    support = patch.square().sum(dim=1, keepdim=True) > 1.0e-12
    return patch, support


def _descriptor_valid(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 4 or tuple(mask.shape[1:]) != (1, *FEATURE_SHAPE):
        raise ValueError("descriptor mask must be [B,1,16,32]")
    return F.avg_pool2d(F.pad(mask.float(), (1, 1, 1, 1), mode="replicate"),
                        3, 1).eq(1.)


def local_cosine_cost(current: torch.Tensor, past: torch.Tensor,
                      *, current_valid: torch.Tensor | None = None,
                      past_valid: torch.Tensor | None = None,
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-patch cosine cost [B,81,16,32] plus current support mask."""
    if current.shape != past.shape or tuple(current.shape[1:]) != (6, *FEATURE_SHAPE):
        raise ValueError("cost features must be aligned [B,6,16,32]")
    current, support = _patch_descriptor(current)
    past, _past_support = _patch_descriptor(past)
    if current_valid is not None:
        support = support & _descriptor_valid(current_valid)
    current = F.normalize(current, dim=1, eps=1.0e-6)
    past = F.normalize(past, dim=1, eps=1.0e-6)
    batch, channels, height, width = current.shape
    kernel = 2 * RADIUS + 1
    patches = F.unfold(F.pad(past, (RADIUS,) * 4, mode="replicate"), kernel_size=kernel)
    patches = patches.reshape(batch, channels, kernel * kernel, height, width)
    cost = (current.unsqueeze(2) * patches).sum(dim=1)
    if past_valid is not None:
        valid = _descriptor_valid(past_valid)
        valid = F.unfold(F.pad(valid.float(), (RADIUS,) * 4, mode="replicate"),
                         kernel_size=kernel)
        valid = valid.reshape(batch, kernel * kernel, height, width).eq(1.)
        cost = cost.masked_fill(~valid, -1.0e4)
    return cost, support


def gaussian_smooth(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 4 or tuple(vector.shape[1:]) != (2, *FEATURE_SHAPE):
        raise ValueError("velocity must be [B,2,16,32]")
    kernel = GAUSSIAN_3X3.to(device=vector.device, dtype=vector.dtype).view(1, 1, 3, 3)
    return F.conv2d(F.pad(vector, (1, 1, 1, 1), mode="replicate"),
                    kernel.expand(2, 1, 3, 3), groups=2)


def posterior_from_features(features: torch.Tensor, terminal: int,
                            *, feature_mask: torch.Tensor | None = None,
                            reverse_past: bool = False) -> dict[str, torch.Tensor]:
    """Fuse lag-conditioned offsets as atoms in common velocity space.

    ``reverse_past`` keeps the current frame fixed but reverses the available
    preceding history.  It is a diagnostic perturbation only and still cannot
    access a frame after ``terminal``.
    """
    if features.ndim != 5 or tuple(features.shape[1:]) != (HISTORY_STEPS, 6, *FEATURE_SHAPE):
        raise ValueError("features must be [B,20,6,16,32]")
    if terminal not in TERMINALS or terminal + 1 >= HISTORY_STEPS:
        raise ValueError("terminal must be a legal internal one-step index")
    if terminal - max(LAGS) < 0:
        raise ValueError("terminal lacks requested lag history")
    if (feature_mask is not None
            and feature_mask.shape != features.shape[:2] + (1,) + features.shape[3:]):
        raise ValueError("feature_mask must be [B,20,1,16,32]")
    current = features[:, terminal]
    per_lag_probability: list[torch.Tensor] = []
    per_lag_velocity: list[torch.Tensor] = []
    per_lag_entropy: list[torch.Tensor] = []
    support: torch.Tensor | None = None
    offset = offsets(features.device, features.dtype)  # [81,2] x/y
    for lag in LAGS:
        # Reverse indices remain within [0, terminal-1], never a future frame.
        past_index = lag - 1 if reverse_past else terminal - lag
        if past_index < 0 or past_index >= terminal:
            raise RuntimeError("history-only reverse index invariant failed")
        cost, current_support = local_cosine_cost(
            current, features[:, past_index],
            current_valid=(feature_mask[:, terminal] if feature_mask is not None else None),
            past_valid=(feature_mask[:, past_index] if feature_mask is not None else None))
        if support is None:
            support = current_support
        elif not torch.equal(support, current_support):
            raise RuntimeError("current descriptor support changed across lags")
        probability = torch.softmax(BETA * cost, dim=1)
        entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=1)
        velocity = torch.einsum("bkhw,kd->bdhw", probability, offset / float(lag))
        per_lag_probability.append(probability)
        per_lag_velocity.append(velocity)
        per_lag_entropy.append(entropy)
    probability = torch.stack(per_lag_probability, dim=1)  # B,L,K,H,W
    velocity = torch.stack(per_lag_velocity, dim=1)          # B,L,2,H,W
    entropy = torch.stack(per_lag_entropy, dim=1)             # B,L,H,W
    reliability = (1.0 - entropy / math.log((2 * RADIUS + 1) ** 2)).clamp(0., 1.)
    alpha = (reliability + ALPHA_EPS)
    alpha = alpha / alpha.sum(dim=1, keepdim=True)
    mean = (alpha.unsqueeze(2) * velocity).sum(dim=1)
    # Entropy already measures ambiguity inside each lag.  This covariance is
    # deliberately only across the four lag-conditioned velocity means: it
    # asks whether offset/lag estimates agree in a common physical coordinate,
    # without double-penalising a broad per-lag posterior.
    trace_covariance = (alpha *
                        (velocity - mean.unsqueeze(1)).square().sum(dim=2)).sum(dim=1)
    sigma = float(RADIUS) / 16.0
    gate = reliability.mean(dim=1) * torch.exp(-trace_covariance / (2. * sigma * sigma))
    if support is None:
        raise RuntimeError("lag posterior did not construct descriptor support")
    if feature_mask is not None:
        support = support & feature_mask[:, terminal]
    # Apply the centre mask again after smoothing: a measured neighbour must
    # never leak a displacement into an invalid centre cell.
    support_value = support.to(dtype=mean.dtype)
    velocity_pre_cap = support_value * gaussian_smooth(
        support_value * gate.unsqueeze(1) * mean)
    velocity_effective = ONE_STEP_CAP_FEATURE_PIXELS * torch.tanh(
        velocity_pre_cap / ONE_STEP_CAP_FEATURE_PIXELS)
    lag_dispersion = (alpha * (velocity - mean.unsqueeze(1)).square().sum(dim=2)).sum(dim=1)
    return {"probability": probability, "velocity_per_lag": velocity,
            "reliability": reliability, "alpha": alpha, "mean_velocity": mean,
            "trace_covariance": trace_covariance, "gate": gate,
            "velocity_pre_cap": velocity_pre_cap,
            "velocity_effective": velocity_effective, "lag_dispersion": lag_dispersion,
            "support": support}


def feature_velocity_to_field(velocity: torch.Tensor) -> torch.Tensor:
    """Convert feature-pixel/frame velocity (16x32) to 32x64 pixel units."""
    if velocity.ndim != 4 or tuple(velocity.shape[1:]) != (2, *FEATURE_SHAPE):
        raise ValueError("feature velocity must be [B,2,16,32]")
    return 2.0 * F.interpolate(velocity, size=FIELD_SHAPE, mode="bilinear", align_corners=True)


def warp_field(source: torch.Tensor, field_velocity: torch.Tensor) -> torch.Tensor:
    """Mask-normalised backtrace warp with fixed zero-support fallback."""
    if source.ndim != 4 or tuple(source.shape[1:]) != (2, *FIELD_SHAPE):
        raise ValueError("source must be [B,2,32,64]")
    if field_velocity.shape != source.shape:
        raise ValueError("field velocity must align source [B,2,32,64]")
    batch, _channels, height, width = source.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=source.device, dtype=source.dtype),
                            torch.arange(width, device=source.device, dtype=source.dtype), indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    grid = grid + field_velocity.permute(0, 2, 3, 1)
    grid[..., 0] = 2. * grid[..., 0] / float(width - 1) - 1.
    grid[..., 1] = 2. * grid[..., 1] / float(height - 1) - 1.
    source_valid = source.abs().amax(dim=1, keepdim=True).gt(1.0e-12)
    sampled_numerator = F.grid_sample(
        source * source_valid, grid, mode="bilinear", padding_mode="border",
        align_corners=True)
    sampled_denominator = F.grid_sample(
        source_valid.to(source.dtype), grid, mode="bilinear",
        padding_mode="border", align_corners=True)
    sampled = sampled_numerator / sampled_denominator.clamp_min(1.0e-6)
    sampled = torch.where(sampled_denominator > 1.0e-6, sampled, source)
    # No explicit HDF5 mask/fill-value exists.  Keep an observation-zero
    # centre exactly zero even if neighbouring measured cells are sampled.
    return torch.where(source_valid, sampled, source)


def synthetic_integer_translation_check(device: torch.device) -> dict[str, float | bool]:
    """Lock matcher sign and feature-pixel -> field-pixel conversion."""
    torch.manual_seed(17)
    source = torch.randn(1, 2, *FIELD_SHAPE, device=device)
    # A physical right shift by two field pixels means target[x]=source[x-2].
    target = torch.cat((source[..., :1].expand(-1, -1, -1, 2), source[..., :-2]), dim=-1)
    feature_velocity = torch.zeros(1, 2, *FEATURE_SHAPE, device=device)
    feature_velocity[:, 0] = -1.0
    native = warp_field(source, feature_velocity_to_field(feature_velocity))
    opposite = warp_field(source, feature_velocity_to_field(-feature_velocity))
    native_error = float((native - target).abs().max().detach().cpu())
    opposite_mse = float((opposite - target).square().mean().detach().cpu())
    past = torch.randn(1, 6, *FEATURE_SHAPE, device=device)
    current = torch.cat((past[..., :1], past[..., :-1]), dim=-1)
    cost, _support = local_cosine_cost(current, past)
    expected = torch.nonzero(
        (offsets(device, past.dtype) == past.new_tensor((-1., 0.))).all(dim=1),
        as_tuple=False).item()
    interior = cost[:, :, RADIUS + 2:-RADIUS - 2, RADIUS + 2:-RADIUS - 2]
    matcher_fraction = float((interior.argmax(dim=1) == expected).float().mean().cpu())
    passed = (native_error <= WARP_SANITY_TOLERANCE and opposite_mse > 1.0e-3
              and matcher_fraction >= .999)
    return {"pass": passed,
            "native_max_abs_error": native_error, "opposite_mse": opposite_mse,
            "native_max_abs_error_tolerance": WARP_SANITY_TOLERANCE,
            "matcher_expected_backtrace_offset": [-1, 0],
            "matcher_correct_interior_fraction": matcher_fraction,
            "feature_x_minus_one_maps_to_field_x_minus_two": True}


def donor_indices(windows: list[Window]) -> tuple[np.ndarray, dict[str, object]]:
    """Next trajectory within (AoA, ordinal-mod-4 fold), same start if present."""
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    by_file: dict[str, list[int]] = defaultdict(list)
    for index, window in enumerate(windows):
        groups[(window.aoa, window.fold)].append(index)
        by_file[window.file].append(index)
    mapping = np.empty(len(windows), dtype=np.int64)
    fallback = 0
    exact_start = 0
    for key, indexes in groups.items():
        files = sorted({windows[index].file for index in indexes},
                       key=lambda name: (windows[by_file[name][0]].reynolds, name))
        if len(files) < 2:
            raise ValueError(f"donor group {key} has fewer than two trajectories")
        starts = {name: {windows[index].start: index for index in by_file[name]}
                  for name in files}
        ordered = {name: by_file[name] for name in files}
        for index in indexes:
            source = windows[index]
            next_file = files[(files.index(source.file) + 1) % len(files)]
            donor = starts[next_file].get(source.start)
            if donor is None:
                donor = ordered[next_file][source.window_ordinal % len(ordered[next_file])]
                fallback += 1
            else:
                exact_start += 1
            mapping[index] = donor
    delta_re = np.asarray([abs(windows[index].reynolds - windows[int(donor)].reynolds)
                           for index, donor in enumerate(mapping)], dtype=np.float64)
    return mapping, {"fallback_window_ordinal_matches": fallback,
                     "exact_start_matches": exact_start,
                     "groups": len(groups), "same_aoa_and_ordinal_mod4_fold": True,
                     "delta_re_abs": {"min": float(delta_re.min()),
                                      "median": float(np.median(delta_re)),
                                      "mean": float(delta_re.mean()),
                                      "max": float(delta_re.max())}}


def _advance(mapping: np.ndarray, count: int) -> np.ndarray:
    value = np.arange(len(mapping), dtype=np.int64)
    for _ in range(count):
        value = mapping[value]
    return value


def _aggregate_sse(identity: np.ndarray, candidate: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    base = float(identity[rows].sum())
    value = float(candidate[rows].sum())
    return {"identity_uv_sse": base, "candidate_uv_sse": value,
            "reduction_fraction": float((base - value) / max(base, 1.0e-30))}


MASK_VARIANTS = ("all_pixels", "common_valid", "eroded_common_valid")


def evaluation_masks(source: torch.Tensor, truth: torch.Tensor) -> dict[str, torch.Tensor]:
    """Three frozen scoring supports; target is used only for evaluation."""
    if source.shape != truth.shape or tuple(source.shape[1:]) != (2, *FIELD_SHAPE):
        raise ValueError("source/truth must align [B,2,32,64]")
    source_valid = source.abs().amax(dim=1).gt(1.0e-12)
    truth_valid = truth.abs().amax(dim=1).gt(1.0e-12)
    common = source_valid & truth_valid
    eroded = F.avg_pool2d(common[:, None].float(), 3, 1, 1).eq(1.)[:, 0]
    return {"all_pixels": torch.ones_like(common), "common_valid": common,
            "eroded_common_valid": eroded}


def masked_sse(prediction: torch.Tensor, truth: torch.Tensor,
               masks: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    error = (prediction - truth).square().sum(dim=1)
    return {name: (error * mask).sum((1, 2)).cpu().numpy()
            for name, mask in masks.items()}


def _empty_metric(samples: int) -> dict[str, np.ndarray]:
    return {name: np.zeros(samples, dtype=np.float64) for name in MASK_VARIANTS}


def _score_groups(windows: list[Window], identity: np.ndarray, native: np.ndarray,
                  spatial_null: np.ndarray, donor: np.ndarray) -> dict[str, object]:
    def score(rows: np.ndarray) -> dict[str, object]:
        native_report = _aggregate_sse(identity, native, rows)
        spatial_report = _aggregate_sse(identity, spatial_null, rows)
        donor_report = _aggregate_sse(identity, donor, rows)
        denominator = max(native_report["identity_uv_sse"], 1.0e-30)
        return {"rows": int(len(rows)), "native_vs_identity": native_report,
                "spatial_null_vs_identity": spatial_report,
                "donor_vs_identity": donor_report,
                "native_minus_spatial_null_fraction_of_identity": float(
                    (spatial_report["candidate_uv_sse"] - native_report["candidate_uv_sse"])
                    / denominator),
                "native_minus_donor_fraction_of_identity": float(
                    (donor_report["candidate_uv_sse"] - native_report["candidate_uv_sse"])
                    / denominator)}

    all_rows = np.arange(len(windows), dtype=np.int64)
    folds = {str(value): score(np.asarray(
        [i for i, row in enumerate(windows) if row.fold == value], dtype=np.int64))
        for value in range(4)}
    aoas = {str(value): score(np.asarray(
        [i for i, row in enumerate(windows) if row.aoa == value], dtype=np.int64))
        for value in FIT_AOAS}
    by_file: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(windows):
        by_file[row.file].append(index)
    trajectories: dict[str, object] = {}
    trajectory_reductions: list[float] = []
    for name, indexes in sorted(by_file.items()):
        rows = np.asarray(indexes, dtype=np.int64)
        value = score(rows)
        reduction = float(value["native_vs_identity"]["reduction_fraction"])
        trajectory_reductions.append(reduction)
        trajectories[name] = {"aoa": windows[indexes[0]].aoa,
                              "re": windows[indexes[0]].reynolds,
                              "fold": windows[indexes[0]].fold, **value}
    reduction_array = np.asarray(trajectory_reductions, dtype=np.float64)
    return {"overall": score(all_rows), "by_fold": folds, "by_aoa": aoas,
            "by_trajectory": trajectories,
            "trajectory_summary": {
                "count": int(len(reduction_array)),
                "median_reduction_fraction": float(np.median(reduction_array)),
                "positive_fraction": float(np.mean(reduction_array > 0.)),
                "minimum_reduction_fraction": float(reduction_array.min()),
                "maximum_reduction_fraction": float(reduction_array.max())}}


def _dispersion_groups(windows: list[Window],
                       dispersion: Mapping[str, np.ndarray]) -> dict[str, object]:
    def summarize(rows: np.ndarray) -> dict[str, float]:
        values = {name: float(array[rows].mean()) for name, array in dispersion.items()}
        native = values["native"]
        values["native_reduction_vs_time_reversed"] = float(
            (values["time_reversed"] - native) / max(values["time_reversed"], 1.0e-30))
        values["native_reduction_vs_lag_donor"] = float(
            (values["lag_donor"] - native) / max(values["lag_donor"], 1.0e-30))
        return values
    return {
        "overall": summarize(np.arange(len(windows), dtype=np.int64)),
        "by_fold": {str(value): summarize(np.asarray(
            [i for i, row in enumerate(windows) if row.fold == value], dtype=np.int64))
            for value in range(4)},
        "by_aoa": {str(value): summarize(np.asarray(
            [i for i, row in enumerate(windows) if row.aoa == value], dtype=np.int64))
            for value in FIT_AOAS}}


def gate(report: Mapping[str, Any]) -> tuple[bool, dict[str, bool]]:
    primary = report["primary_multistep"]
    common = primary["mask_variants"]["common_valid"]
    all_pixels = primary["mask_variants"]["all_pixels"]
    eroded = primary["mask_variants"]["eroded_common_valid"]
    fold_values = [float(common["by_fold"][str(index)]["native_vs_identity"]
                         ["reduction_fraction"]) for index in range(4)]
    aoa_values = [float(common["by_aoa"][str(value)]["native_vs_identity"]
                        ["reduction_fraction"]) for value in FIT_AOAS]
    horizon_values = {str(horizon): float(primary["by_horizon"][str(horizon)]
        ["mask_variants"]["common_valid"]["overall"]["native_vs_identity"]
        ["reduction_fraction"]) for horizon in PRIMARY_HORIZONS}
    dispersion = report["lag_consistency"]
    reversed_values = [float(dispersion["by_fold"][str(index)]
                             ["native_reduction_vs_time_reversed"]) for index in range(4)]
    donor_values = [float(dispersion["by_fold"][str(index)]
                          ["native_reduction_vs_lag_donor"]) for index in range(4)]
    repeat_values = [float(report["repeat_one_step"][str(terminal)]
                           ["common_valid"]["reduction_fraction"])
                     for terminal in REPEAT_ONE_STEP_TERMINALS]
    checks = {
        "common_valid_pooled_ge_2pct": float(common["overall"]["native_vs_identity"]
                                                ["reduction_fraction"]) >= .02,
        "common_valid_h1_ge_1pct": horizon_values["1"] >= .01,
        "common_valid_h2_nonnegative": horizon_values["2"] >= 0.,
        "common_valid_h3_nonnegative": horizon_values["3"] >= 0.,
        "repeat_one_step_common_valid_both_nonnegative": min(repeat_values) >= 0.,
        "common_valid_all_aoa_nonnegative": min(aoa_values) >= 0.,
        "common_valid_fold_ge_0p5pct_3of4": sum(value >= .005 for value in fold_values) >= 3,
        "common_valid_trajectory_median_positive": float(common["trajectory_summary"]
            ["median_reduction_fraction"]) > 0.,
        "common_valid_trajectory_positive_ge_60pct": float(common["trajectory_summary"]
            ["positive_fraction"]) >= .60,
        "all_pixels_pooled_nonnegative": float(all_pixels["overall"]["native_vs_identity"]
                                                ["reduction_fraction"]) >= 0.,
        "eroded_common_valid_pooled_ge_1p5pct": float(eroded["overall"]["native_vs_identity"]
                                                       ["reduction_fraction"]) >= .015,
        "native_beats_spatial_null_ge_0p5pct": float(common["overall"]
            ["native_minus_spatial_null_fraction_of_identity"]) >= .005,
        "native_dispersion_vs_reversed_ge_10pct_3of4": sum(
            value >= .10 for value in reversed_values) >= 3,
        "native_dispersion_vs_lag_donor_ge_10pct_3of4": sum(
            value >= .10 for value in donor_values) >= 3,
    }
    return all(checks.values()), checks


def analyze_histories(histories: np.ndarray, windows: list[Window], device: torch.device,
                      chunk_size: int) -> dict[str, object]:
    if histories.shape != (len(windows), HISTORY_STEPS, *FIELD_SHAPE, 2):
        raise ValueError("history array/window metadata shape mismatch")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    samples = len(windows)
    count_terminal = len(TERMINALS)
    identity_sse = _empty_metric(samples)
    native_sse = _empty_metric(samples)
    spatial_null_sse = _empty_metric(samples)
    donor_sse = _empty_metric(samples)
    support_count = _empty_metric(samples)
    by_horizon_identity = {horizon: _empty_metric(samples) for horizon in PRIMARY_HORIZONS}
    by_horizon_native = {horizon: _empty_metric(samples) for horizon in PRIMARY_HORIZONS}
    by_horizon_spatial = {horizon: _empty_metric(samples) for horizon in PRIMARY_HORIZONS}
    by_horizon_donor = {horizon: _empty_metric(samples) for horizon in PRIMARY_HORIZONS}
    by_horizon_support = {horizon: _empty_metric(samples) for horizon in PRIMARY_HORIZONS}
    repeat_identity = {terminal: _empty_metric(samples)
                       for terminal in REPEAT_ONE_STEP_TERMINALS}
    repeat_native = {terminal: _empty_metric(samples)
                     for terminal in REPEAT_ONE_STEP_TERMINALS}
    velocities = np.empty((samples, count_terminal, 2, *FEATURE_SHAPE), dtype=np.float32)
    per_lag = np.empty((samples, count_terminal, len(LAGS), 2, *FEATURE_SHAPE), dtype=np.float32)
    alpha = np.empty((samples, count_terminal, len(LAGS), *FEATURE_SHAPE), dtype=np.float32)
    raw_h20_values = np.empty((samples, count_terminal, *FEATURE_SHAPE), dtype=np.float32)
    native_dispersion = np.empty(samples, dtype=np.float64)
    reversed_dispersion = np.empty(samples, dtype=np.float64)
    cap_touch = np.empty(samples, dtype=np.float64)
    zero_support = np.empty(samples, dtype=np.float64)
    feature_statistics = crossfit_feature_statistics(
        histories, windows, device, chunk_size)
    with torch.inference_mode():
        for fold in range(4):
            fold_rows = np.asarray(
                [index for index, row in enumerate(windows) if row.fold == fold],
                dtype=np.int64)
            for begin in range(0, len(fold_rows), chunk_size):
                index = fold_rows[begin:begin + chunk_size]
                history = torch.from_numpy(np.ascontiguousarray(histories[index])).to(device)
                mask = history_feature_mask(history)
                features = standardize_features(
                    history_features(history), mask, feature_statistics[fold])
                chunk_native_dispersion = torch.zeros(len(history), device=device)
                chunk_reversed_dispersion = torch.zeros(len(history), device=device)
                chunk_cap_touch = torch.zeros(len(history), device=device)
                chunk_zero_support = torch.zeros(len(history), device=device)
                for local_terminal, terminal in enumerate(TERMINALS):
                    posterior = posterior_from_features(features, terminal, feature_mask=mask)
                    reversed_posterior = posterior_from_features(
                        features, terminal, feature_mask=mask, reverse_past=True)
                    source = history[:, terminal].permute(0, 3, 1, 2)
                    effective = posterior["velocity_effective"]
                    if terminal == PRIMARY_TERMINAL:
                        spatial_null = torch.roll(
                            effective, shifts=(FEATURE_SHAPE[0] // 2, FEATURE_SHAPE[1] // 2),
                            dims=(-2, -1)) * posterior["support"].to(effective.dtype)
                        for horizon in PRIMARY_HORIZONS:
                            truth = history[:, terminal + horizon].permute(0, 3, 1, 2)
                            masks = evaluation_masks(source, truth)
                            prediction = warp_field(
                                source, feature_velocity_to_field(float(horizon) * effective))
                            null_prediction = warp_field(
                                source, feature_velocity_to_field(float(horizon) * spatial_null))
                            identity_value = masked_sse(source, truth, masks)
                            native_value = masked_sse(prediction, truth, masks)
                            null_value = masked_sse(null_prediction, truth, masks)
                            for name in MASK_VARIANTS:
                                identity_sse[name][index] += identity_value[name]
                                native_sse[name][index] += native_value[name]
                                spatial_null_sse[name][index] += null_value[name]
                                by_horizon_identity[horizon][name][index] = identity_value[name]
                                by_horizon_native[horizon][name][index] = native_value[name]
                                by_horizon_spatial[horizon][name][index] = null_value[name]
                                count = masks[name].sum((1, 2)).cpu().numpy()
                                support_count[name][index] += count
                                by_horizon_support[horizon][name][index] = count
                    else:
                        truth = history[:, terminal + 1].permute(0, 3, 1, 2)
                        masks = evaluation_masks(source, truth)
                        prediction = warp_field(source, feature_velocity_to_field(effective))
                        identity_value = masked_sse(source, truth, masks)
                        native_value = masked_sse(prediction, truth, masks)
                        for name in MASK_VARIANTS:
                            repeat_identity[terminal][name][index] = identity_value[name]
                            repeat_native[terminal][name][index] = native_value[name]
                    velocities[index, local_terminal] = posterior["velocity_effective"].cpu().numpy()
                    per_lag[index, local_terminal] = posterior["velocity_per_lag"].cpu().numpy()
                    alpha[index, local_terminal] = posterior["alpha"].cpu().numpy()
                    chunk_native_dispersion += posterior["lag_dispersion"].mean((1, 2))
                    chunk_reversed_dispersion += reversed_posterior["lag_dispersion"].mean((1, 2))
                    raw_h20 = (20.0 * posterior["velocity_pre_cap"]).abs().amax(dim=1)
                    raw_h20_values[index, local_terminal] = raw_h20.cpu().numpy()
                    chunk_cap_touch += (raw_h20 >= H20_CAP_FEATURE_PIXELS).float().mean((1, 2))
                    chunk_zero_support += (~posterior["support"]).float().mean((1, 2, 3))
                native_dispersion[index] = (chunk_native_dispersion / count_terminal).cpu().numpy()
                reversed_dispersion[index] = (chunk_reversed_dispersion / count_terminal).cpu().numpy()
                cap_touch[index] = (chunk_cap_touch / count_terminal).cpu().numpy()
                zero_support[index] = (chunk_zero_support / count_terminal).cpu().numpy()
    donor_map, donor_mapping_report = donor_indices(windows)
    with torch.inference_mode():
        for start in range(0, samples, chunk_size):
            stop = min(start + chunk_size, samples)
            history = torch.from_numpy(np.ascontiguousarray(histories[start:stop])).to(device)
            donor_velocity = velocities[donor_map[start:stop]]
            native_velocity = velocities[start:stop]
            native_value = torch.from_numpy(native_velocity[:, 0]).to(device)
            donor_value = torch.from_numpy(donor_velocity[:, 0]).to(device)
            magnitude = native_value.square().sum(dim=1, keepdim=True).sqrt()
            direction = donor_value / donor_value.square().sum(
                dim=1, keepdim=True).sqrt().clamp_min(1.0e-12)
            # Magnitude includes native reliability; only donor direction changes.
            donor_effective = magnitude * direction
            source = history[:, PRIMARY_TERMINAL].permute(0, 3, 1, 2)
            for horizon in PRIMARY_HORIZONS:
                truth = history[:, PRIMARY_TERMINAL + horizon].permute(0, 3, 1, 2)
                masks = evaluation_masks(source, truth)
                prediction = warp_field(
                    source, feature_velocity_to_field(float(horizon) * donor_effective))
                donor_value_sse = masked_sse(prediction, truth, masks)
                for name in MASK_VARIANTS:
                    donor_sse[name][start:stop] += donor_value_sse[name]
                    by_horizon_donor[horizon][name][start:stop] = donor_value_sse[name]
    donor_steps = [_advance(donor_map, index + 1) for index in range(len(LAGS))]
    lag_donor_dispersion = np.zeros(samples, dtype=np.float64)
    for index in range(samples):
        aggregate = 0.0
        for terminal_index in range(count_terminal):
            weights = alpha[index, terminal_index]
            donor_lags = np.stack([per_lag[donor_steps[lag_index][index], terminal_index, lag_index]
                                   for lag_index in range(len(LAGS))], axis=0)
            mean = (weights[:, None] * donor_lags).sum(axis=0)
            aggregate += float((weights[:, None] * np.square(
                donor_lags - mean[None]).sum(axis=1)).sum(axis=0).mean())
        lag_donor_dispersion[index] = aggregate / count_terminal
    dispersion = {"native": native_dispersion, "time_reversed": reversed_dispersion,
                  "lag_donor": lag_donor_dispersion}
    statistics_report = {
        str(fold): {"mean": row["mean"].reshape(-1).cpu().tolist(),
                    "std": row["std"].reshape(-1).cpu().tolist(),
                    "fit_measured_cells": int(row["fit_measured_cells"].item())}
        for fold, row in feature_statistics.items()}
    report: dict[str, object] = {
        "schema": SCHEMA,
        "evidence_scope": "fit-side history-internal preflight; no AoA10 and no frame after index 19",
        "config": {"fit_aoas": list(FIT_AOAS), "forbidden_aoa": FORBIDDEN_AOA,
                   "history_steps": HISTORY_STEPS, "primary_terminal": PRIMARY_TERMINAL,
                   "primary_horizons": list(PRIMARY_HORIZONS),
                   "repeat_one_step_terminals": list(REPEAT_ONE_STEP_TERMINALS),
                   "terminals": list(TERMINALS), "lags": list(LAGS),
                   "radius": RADIUS, "cosine_softmax_beta": BETA,
                   "features": ["u", "v", "dxu", "dyu", "dxv", "dyv"],
                   "feature_standardization": "other-three-Re-fold measured-cell channel mean/std",
                   "matching_descriptor": "fixed standardized 3x3 patches",
                   "feature_mask": "raw nonzero 3x3 forward-difference footprint",
                   "downsample": "fixed 2x2 average to 16x32",
                   "velocity_semantics": "local pattern backtrace displacement; not physical fluid velocity",
                   "velocity_unit": "feature pixels/frame",
                   "reliability": "1-entropy/log81", "alpha": "normalize(reliability+1e-3)",
                   "gate": "mean(q)*exp(-cross-lag-mean-cov/(2*(radius/16)^2))",
                   "spatial_smoothing": "fixed Gaussian 3x3",
                   "warp": "one-shot h*velocity, mask-normalised source, zero-centre fallback",
                   "spatial_null": "same velocity map rolled by (8,16) feature cells",
                   "score_masks": list(MASK_VARIANTS),
                   "one_step_cap": "(h20_cap/20)*tanh(raw/(h20_cap/20))",
                   "prospective_h20_cap_feature_pixels": H20_CAP_FEATURE_PIXELS,
                   "crossfit_feature_statistics": statistics_report},
        "windows": samples,
        "trajectories": len({row.file for row in windows}),
        "data_access": {"aoas_opened": list(FIT_AOAS), "aoa10_field_opened": False,
                        "max_relative_frame_read": HISTORY_STEPS - 1,
                        "model_matcher_accesses_internal_labels": False,
                        "evaluation_uses_history_internal_labels": True,
                        "evaluation_label_relative_frames": [17, 18, 19],
                        "crossfit_normalizer_uses_other_fold_all_20_frames": True,
                        "future_block_or_frame_after_19_read": False},
        "donor_mapping": donor_mapping_report,
    }
    mask_reports: dict[str, object] = {}
    for name in MASK_VARIANTS:
        value = _score_groups(windows, identity_sse[name], native_sse[name],
                              spatial_null_sse[name], donor_sse[name])
        value["scored_pixel_fraction"] = float(
            support_count[name].sum() /
            (samples * len(PRIMARY_HORIZONS) * FIELD_SHAPE[0] * FIELD_SHAPE[1]))
        mask_reports[name] = value
    horizon_reports: dict[str, object] = {}
    for horizon in PRIMARY_HORIZONS:
        horizon_masks: dict[str, object] = {}
        for name in MASK_VARIANTS:
            value = _score_groups(
                windows, by_horizon_identity[horizon][name],
                by_horizon_native[horizon][name], by_horizon_spatial[horizon][name],
                by_horizon_donor[horizon][name])
            value["scored_pixel_fraction"] = float(
                by_horizon_support[horizon][name].sum() /
                (samples * FIELD_SHAPE[0] * FIELD_SHAPE[1]))
            horizon_masks[name] = value
        horizon_reports[str(horizon)] = {"mask_variants": horizon_masks}
    repeat_report: dict[str, object] = {}
    all_rows = np.arange(samples, dtype=np.int64)
    for terminal in REPEAT_ONE_STEP_TERMINALS:
        repeat_report[str(terminal)] = {name: _aggregate_sse(
            repeat_identity[terminal][name], repeat_native[terminal][name], all_rows)
            for name in MASK_VARIANTS}
    quantiles = np.quantile(raw_h20_values.reshape(-1), (.5, .9, .95, .99, 1.0))
    report["primary_multistep"] = {"anchor_relative_frame": PRIMARY_TERMINAL,
                                    "mask_variants": mask_reports,
                                    "by_horizon": horizon_reports}
    report["repeat_one_step"] = repeat_report
    report["lag_consistency"] = _dispersion_groups(windows, dispersion)
    report["velocity_diagnostics"] = {
        "h20_pre_cap_abs_component_quantiles": {
            name: float(value) for name, value in zip(("q50", "q90", "q95", "q99", "max"), quantiles)},
        "h20_cap_touch_fraction": float(cap_touch.mean()),
        "invalid_feature_support_fraction": float(zero_support.mean()),
        "zero_support_policy": "centre velocity exact zero after Gaussian smoothing",
        "cap_is_diagnostic_not_go_gate": True}
    passed, checks = gate(report)
    report["go_checks"] = checks
    report["decision"] = "GO_M8_PAIRED_CROSSFIT" if passed else "STOP_M8_IDENTIFIABILITY"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    args = parser.parse_args()
    if not args.smoke and any(item is None for item in (args.real_dir, args.manifest, args.output_dir)):
        parser.error("--real-dir, --manifest, and --output-dir are required unless --smoke")
    return args


def smoke(device: torch.device) -> dict[str, object]:
    torch.manual_seed(210)
    histories = torch.randn(4, HISTORY_STEPS, *FIELD_SHAPE, 2).numpy().astype(np.float32)
    windows = [Window(f"{10000 + index}_0.h5", index * 20, 0, 10000 + index,
                      index, index, 0) for index in range(4)]
    # A smoke needs valid donor groups, so all synthetic trajectories share one fold.
    windows = [Window(item.file, item.start, item.aoa, item.reynolds, item.trajectory, 0,
                      item.window_ordinal) for item in windows]
    # Report mechanics directly rather than claiming a four-fold statistical result.
    feature = history_features(torch.from_numpy(histories).to(device))
    mask = history_feature_mask(torch.from_numpy(histories).to(device))
    posterior = posterior_from_features(feature, 18, feature_mask=mask)
    check = synthetic_integer_translation_check(device)
    return {"event": "lag_consistent_velocity_posterior_smoke", "synthetic_translation": check,
            "posterior_finite": bool(torch.isfinite(posterior["velocity_effective"]).all()),
            "posterior_shape": list(posterior["velocity_effective"].shape),
            "matcher_target_access": False, "history_internal_label_access": False,
            "aoa10_access": False}


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.execute or args.acknowledge_protocol != "M8_LCPLT_HISTORY_ONLY_V1":
        raise PermissionError("formal run requires --execute and protocol acknowledgement")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output dir")
    git = git_report()
    if not git["clean"]:
        raise RuntimeError(f"formal run requires clean worktree: {git['porcelain']}")
    device = resolve_device(args.device)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    translation = synthetic_integer_translation_check(device)
    if not bool(translation["pass"]):
        raise RuntimeError(f"synthetic grid convention failed: {translation}")
    manifest = load_manifest(args.manifest)
    windows = fit_windows(manifest)
    histories = load_histories(real_dir=args.real_dir, manifest=manifest, windows=windows)
    report = analyze_histories(histories, windows, device, args.chunk_size)
    report["synthetic_integer_translation"] = translation
    report["provenance"] = {
        "git": git,
        "script_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": sha256_file(args.manifest),
        "selected_window_inventory_sha256": canonical_sha256(
            [asdict(window) for window in windows]),
        "selected_h5_inventory_sha256": canonical_sha256([
            {"file": name, "bytes": (args.real_dir / name).stat().st_size}
            for name in sorted({window.file for window in windows})]),
        "command": shlex.join(sys.argv)}
    report["runtime"] = {"device": str(device), "torch": torch.__version__,
                         "cuda": torch.version.cuda,
                         "device_name": (torch.cuda.get_device_name(device)
                                         if device.type == "cuda" else "cpu"),
                         "python": sys.version.replace("\n", " "),
                         "platform": platform.platform(), "pid": os.getpid(),
                         "cublas_workspace_config": os.environ.get(
                             "CUBLAS_WORKSPACE_CONFIG"),
                         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "result.sha256").write_text(
        sha256_file(result_path) + "  result.json\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.smoke:
        print(json.dumps(smoke(device), ensure_ascii=False, sort_keys=True))
        return
    result = run(args)
    print(json.dumps({"event": "lag_consistent_velocity_posterior_complete",
                      "decision": result["decision"], "windows": result["windows"]},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
