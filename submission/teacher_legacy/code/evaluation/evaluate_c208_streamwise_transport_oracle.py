"""Gate D0: target-visible streamwise transport headroom for frozen C208.

This is a train-side oracle audit, not a deployable model.  It caches only the
canonical 2576 point-train windows, searches the preregistered four-block
horizontal-warp family, and never constructs or indexes the 725 temporal tail.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_c208_phase_residual_probe as phase  # noqa: E402
import train_c208_real_sim_rehearsal as c208  # noqa: E402
import train_full_backbone_point_distillation as full  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402


EXPERIMENT = "c208_gate_d0_streamwise_transport_oracle"
FORMAL_C208_SHA256 = phase.FORMAL_C208_SHA256
FORMAL_SPLIT_MANIFEST_SHA256 = phase.FORMAL_SPLIT_MANIFEST_SHA256
FORMAL_REAL_STATS_SHA256 = phase.FORMAL_REAL_STATS_SHA256
FORMAL_COUNTS = dict(phase.FORMAL_COUNTS)
AOA_VALUES = phase.AOA_VALUES

OUTPUT_STEPS = 20
HORIZON_BLOCKS = ((0, 5), (5, 10), (10, 15), (15, 20))
DX_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
DX_COMBINATIONS = np.asarray(
    list(itertools.product(DX_GRID, repeat=len(HORIZON_BLOCKS))),
    dtype=np.float32)
COMBINATION_COUNT = len(DX_GRID) ** len(HORIZON_BLOCKS)
IDENTITY_COMBINATION_INDEX = int(np.flatnonzero(
    np.all(DX_COMBINATIONS == 0.0, axis=1))[0])
SAMPLE_COORDINATE_ROUNDS = 2
COMBINATION_CHUNK_SIZE = 125
FORMAL_EVAL_BATCH_SIZE = 16
FORMAL_NUM_WORKERS = 0
WALL_BUDGET_SECONDS = 20.0 * 60.0
GPU_BUDGET_SECONDS = 12.0 * 60.0
OBJECTIVE_TIE_TOLERANCE = 1e-12
SANITY_ERROR_TOLERANCE = 1e-5
UPPER_BOUND_NUMERICAL_MARGIN = 1e-3
SANITY_SAMPLE_INDICES = (0, 400, 800, 1200, 1600, 2000, 2400, 2575)
SANITY_COMBINATION_INDICES = (0, 44, 62, 156, 312, 468, 562, 580, 624)

SHADOW_WEIGHTS = {
    "rel_l2_score": 0.47201126,
    "tke_rel_l2_score": 0.08586562,
    "mvpe_rel_l2_score": 0.09410214,
}
POOLED_SHADOW_MIN = 0.20
WORST_AOA_SHADOW_MIN = 0.05
MIN_NONZERO_SAMPLE_FRACTION = 0.10
MAX_SATURATED_NONZERO_CHOICE_RATIO = 0.50

# Frozen before any warp enumeration after formal-v1 exposed the incorrect
# assumption that raw PIV zero support was global. These are data-contract
# fingerprints only; they do not select a warp or use a target.
FORMAL_MASK_DIAGNOSTIC = {
    "unique_masks": 2544,
    "mask_pixels_min": 94,
    "mask_pixels_median": 158.0,
    "mask_pixels_max": 240,
    "trajectories_with_varying_masks": 82,
    "samples_different_from_trajectory_first": 2494,
    "global_intersection_pixels": 6,
    "global_consensus_95_pixels": 61,
}
FORMAL_ORDERED_WINDOW_MASK_DIGEST = (
    "8800caabac1f8fe496b255e5504ace01b5a68b41288dbbce3509504d9137d761")

# These fixed x slices are diagnostics only.  The first/last boundaries are
# the first and last official MVPE probe columns at the 32x64 real resolution.
GEOMETRY_REGIONS = {
    "upstream_body_x00_12": (0, 13),
    "measured_wake_x13_37": (13, 38),
    "far_wake_x38_63": (38, 64),
}

EXPERIMENT_CONTRACT = {
    "hypothesis": (
        "The remaining frozen-C208 point error contains streamwise transport "
        "misalignment that a bounded spatial warp can expose without an "
        "additive residual or trainable parameter."),
    "evidence_class": "Real 2576 train-side target-visible oracle only",
    "exact_baseline": {
        "checkpoint_sha256": FORMAL_C208_SHA256,
        "split_manifest_canonical_sha256": FORMAL_SPLIT_MANIFEST_SHA256,
        "real_stats_sha256": FORMAL_REAL_STATS_SHA256,
        "inference": "frozen C208 real iteration 4000; CUDA BF16 autocast",
    },
    "data_boundary": {
        "train_windows": FORMAL_COUNTS["train_windows"],
        "tail_windows": FORMAL_COUNTS["tail_windows"],
        "tail_permission": "zero construction, indexing, target reads, and scoring",
        "legacy_loader": "all raw H5 frames are preloaded before Window indexing",
        "cache": "prediction, history, and truth cached once for all 2576 windows",
        "ordered_window_mask_digest": FORMAL_ORDERED_WINDOW_MASK_DIGEST,
    },
    "warp": {
        "horizon_blocks_1_based": [[a + 1, b] for a, b in HORIZON_BLOCKS],
        "delta_x_grid_cells": list(DX_GRID),
        "dtype": "float32",
        "positive_direction": (
            "right/downstream: out[y,x]=in[y,x-delta_x], "
            "grid_x=2*(x+0.5-delta_x)/W-1"),
        "delta_x_zero": (
            "direct grid_sample bypass per row and bitwise baseline identity"),
        "interpolation": "bilinear",
        "padding_mode": "border",
        "align_corners": False,
        "solid_mask": (
            "per-sample raw-history u and v equal zero in every observed "
            "frame; masks may differ because real PIV zero support is "
            "time-varying"),
        "solid_mask_inputs": ["current sample raw 20-frame history u/v"],
        "solid_mask_forbidden": [
            "truth", "another window", "trajectory consensus",
            "AoA/Re metadata"],
        "solid_source_footprint": (
            "restore baseline at output-solid pixels and whenever any "
            "positive-weight bilinear source pixel is solid"),
        "pressure": "always zero",
        "forbidden": [
            "y displacement", "spatial routing", "wider grid",
            "learnable parameter", "checkpoint", "ZIP", "tail access"],
    },
    "oracle": {
        "target_visible": True,
        "objective_weights": SHADOW_WEIGHTS,
        "combinations_per_sample": COMBINATION_COUNT,
        "combination_order": "lexicographic over [-1,-0.5,0,0.5,1]^4",
        "error_triples": "Rel/TKE/MVPE from block sufficient statistics",
        "direct_official_sanity_max_abs_error": SANITY_ERROR_TOLERANCE,
        "generous_upper": "independent per-sample minimum for each metric",
        "achievable_lower": (
            "identity start; two fixed-order sample-coordinate exact-pooled "
            "ascent rounds; strict improvements only"),
        "tie_break": "smaller sum(abs(delta_x)), then lexicographic index",
        "formal_gate_scorer": "official exact pooled NumPy scorer",
    },
    "gate": {
        "pooled_shadow_delta_min": POOLED_SHADOW_MIN,
        "component_delta_min": 0.0,
        "all_aoa_positive": True,
        "worst_aoa_shadow_min": WORST_AOA_SHADOW_MIN,
        "nonzero_sample_fraction_min": MIN_NONZERO_SAMPLE_FRACTION,
        "saturated_over_nonzero_block_choice_ratio_max": (
            MAX_SATURATED_NONZERO_CHOICE_RATIO),
        "upper_bound_numerical_margin": UPPER_BOUND_NUMERICAL_MARGIN,
    },
    "budget": {
        "wall_seconds": WALL_BUDGET_SECONDS,
        "gpu_seconds": GPU_BUDGET_SECONDS,
    },
    "stop": (
        "GO_WARP_ORACLE_WORTH_LOAO only for an achievable full-gate pass; "
        "STOP_WARP_NO_HEADROOM only when the generous upper cannot pass; "
        "otherwise INCONCLUSIVE_WARP_HEADROOM."),
}


class TailAccessCounters:
    """Mutable counters exist so failure results cannot invent zero access."""

    def __init__(self) -> None:
        self.window_constructions = 0
        self.dataset_constructions = 0
        self.target_read_attempts = 0
        self.target_reads_completed = 0
        self.evaluation_attempts = 0
        self.evaluations_completed = 0

    def report(self) -> dict[str, int | bool]:
        values = {
            "tail_window_constructions": self.window_constructions,
            "tail_dataset_constructions": self.dataset_constructions,
            "tail_target_read_attempts": self.target_read_attempts,
            "tail_target_reads_completed": self.target_reads_completed,
            "tail_evaluation_attempts": self.evaluation_attempts,
            "tail_evaluations_completed": self.evaluations_completed,
        }
        if any(value < 0 for value in values.values()):
            raise ValueError("tail access counters cannot be negative")
        return {
            "tail_accessed": any(values.values()),
            **values,
        }


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, started: float) -> None:
        self.started = started
        self.gpu_started: float | None = None

    def start_gpu(self) -> None:
        if self.gpu_started is not None:
            raise RuntimeError("GPU budget was started twice")
        self.gpu_started = time.perf_counter()

    def elapsed(self) -> dict[str, float]:
        now = time.perf_counter()
        return {
            "wall_seconds": now - self.started,
            "gpu_seconds": (
                0.0 if self.gpu_started is None else now - self.gpu_started),
        }

    def check(self, stage: str) -> None:
        elapsed = self.elapsed()
        if elapsed["wall_seconds"] > WALL_BUDGET_SECONDS:
            raise BudgetExpired(f"wall budget expired during {stage}")
        if elapsed["gpu_seconds"] > GPU_BUDGET_SECONDS:
            raise BudgetExpired(f"GPU budget expired during {stage}")


def canonical_sha256(value: object) -> str:
    return c208.canonical_sha256(value)


def tensor_sha256(value: torch.Tensor) -> str:
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("tensor hashing requires a contiguous CPU tensor")
    array = value.numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def observation_solid_mask(history: torch.Tensor) -> torch.Tensor:
    if (history.ndim != 5 or history.shape[1] != OUTPUT_STEPS
            or history.shape[-1] < 2):
        raise ValueError("history must have shape (B,20,H,W,C>=2)")
    if not bool(torch.isfinite(history).all()):
        raise FloatingPointError("history contains non-finite values")
    return (history[..., :2] == 0).all(dim=1).all(dim=-1)


def per_sample_mask_diagnostics(
        solid_mask: torch.Tensor, windows: Sequence) -> dict[str, object]:
    """Describe raw-history zero support without changing a sample's mask."""
    if (solid_mask.ndim != 3 or solid_mask.dtype != torch.bool
            or solid_mask.device.type != "cpu"):
        raise ValueError("solid mask must be a CPU (N,H,W) bool tensor")
    if len(solid_mask) != len(windows):
        raise ValueError("solid mask/window count mismatch")
    if not len(solid_mask):
        raise ValueError("solid mask diagnostics require at least one sample")

    array = solid_mask.numpy()
    flat = array.reshape(len(array), -1)
    packed = np.packbits(flat, axis=1)
    pixel_counts = flat.sum(axis=1, dtype=np.int64)
    trajectory_ids = np.asarray(
        [int(window.trajectory) for window in windows], dtype=np.int64)
    trajectories = np.unique(trajectory_ids)
    varying = 0
    different_from_first = 0
    for trajectory in trajectories:
        group = flat[trajectory_ids == trajectory]
        equal_first = np.all(group == group[:1], axis=1)
        different_from_first += int(np.count_nonzero(~equal_first))
        varying += int(not bool(np.all(equal_first)))
    consensus_threshold = int(np.ceil(0.95 * len(flat)))
    report: dict[str, object] = {
        "semantics": "per_sample_observation_zero_support",
        "constructor_inputs": ["raw_history_uv"],
        "target_used_for_mask": False,
        "samples": int(len(flat)),
        "trajectories": int(len(trajectories)),
        "unique_masks": int(np.unique(packed, axis=0).shape[0]),
        "mask_pixels_min": int(pixel_counts.min()),
        "mask_pixels_median": float(np.median(pixel_counts)),
        "mask_pixels_mean": float(pixel_counts.mean()),
        "mask_pixels_max": int(pixel_counts.max()),
        "trajectories_with_varying_masks": varying,
        "samples_different_from_trajectory_first": different_from_first,
        "global_intersection_pixels": int(np.all(flat, axis=0).sum()),
        "global_consensus_95_pixels": int(
            (flat.sum(axis=0, dtype=np.int64) >= consensus_threshold).sum()),
        "trajectory_aggregation_used_for_warp": False,
    }
    return report


def ordered_window_mask_digest(
        ordered_window_key_digest: str, solid_mask: torch.Tensor) -> str:
    """Bind ordered split keys to the exact ordered per-sample mask cache."""
    if len(ordered_window_key_digest) != 64:
        raise ValueError("ordered window key digest must be SHA-256 hex")
    try:
        key_bytes = bytes.fromhex(ordered_window_key_digest)
    except ValueError as exc:
        raise ValueError("ordered window key digest must be SHA-256 hex") from exc
    mask_digest = bytes.fromhex(tensor_sha256(solid_mask.contiguous()))
    return hashlib.sha256(key_bytes + mask_digest).hexdigest()


def _per_frame_shifts(block_shifts: torch.Tensor) -> torch.Tensor:
    if block_shifts.ndim != 2 or block_shifts.shape[1] != len(HORIZON_BLOCKS):
        raise ValueError("block shifts must have shape (B,4)")
    if not bool(torch.isfinite(block_shifts).all()):
        raise FloatingPointError("block shifts contain non-finite values")
    allowed = torch.zeros_like(block_shifts, dtype=torch.bool)
    for value in DX_GRID:
        allowed |= block_shifts == value
    if not bool(allowed.all()):
        raise ValueError("block shifts leave the frozen delta-x grid")
    result = block_shifts.new_empty((len(block_shifts), OUTPUT_STEPS))
    for block, (start, stop) in enumerate(HORIZON_BLOCKS):
        result[:, start:stop] = block_shifts[:, block, None]
    return result


def source_footprint_fallback_mask(
        solid_mask: torch.Tensor, delta_x: torch.Tensor) -> torch.Tensor:
    """Mark output pixels whose positive-weight x source touches solid."""
    if solid_mask.ndim != 3 or solid_mask.dtype != torch.bool:
        raise ValueError("solid mask must be a (B,H,W) bool tensor")
    if delta_x.shape != (len(solid_mask),):
        raise ValueError("delta_x must have shape (B,)")
    batch, height, width = solid_mask.shape
    output_x = torch.arange(
        width, device=solid_mask.device, dtype=torch.float32)[None, :]
    source_x = output_x - delta_x.float()[:, None]
    left = torch.floor(source_x).to(torch.long).clamp(0, width - 1)
    right = torch.ceil(source_x).to(torch.long).clamp(0, width - 1)
    gather_shape = (batch, height, width)
    left_solid = torch.gather(
        solid_mask, 2, left[:, None, :].expand(gather_shape))
    right_solid = torch.gather(
        solid_mask, 2, right[:, None, :].expand(gather_shape))
    return solid_mask | left_solid | right_solid


def _streamwise_warp_steps(prediction: torch.Tensor, delta_x: torch.Tensor,
                           solid_mask: torch.Tensor) -> torch.Tensor:
    """FP32 horizontal warp for an arbitrary contiguous horizon block."""
    if prediction.ndim != 5 or prediction.shape[-1] < 3:
        raise ValueError("prediction block must have shape (B,T,H,W,C>=3)")
    if prediction.dtype != torch.float32:
        raise TypeError("formal warp requires FP32 prediction")
    batch, steps, height, width, _channels = prediction.shape
    if solid_mask.shape != (batch, height, width):
        raise ValueError("solid mask does not align with prediction")
    delta_x = delta_x.to(device=prediction.device, dtype=torch.float32)
    if delta_x.shape != (batch,):
        raise ValueError("delta_x must have shape (B,)")
    allowed = torch.zeros_like(delta_x, dtype=torch.bool)
    for value in DX_GRID:
        allowed |= delta_x == value
    if not bool(allowed.all()):
        raise ValueError("delta_x leaves the frozen grid")
    # The identity path must never enter grid_sample.
    nonzero = delta_x != 0
    if not bool(nonzero.any()):
        return prediction.clone()

    active_prediction = prediction[nonzero]
    active_delta = delta_x[nonzero]
    active_solid = solid_mask[nonzero]
    active_batch = len(active_prediction)

    y = (2.0 * (torch.arange(
        height, device=prediction.device, dtype=torch.float32) + 0.5)
         / height - 1.0)
    x = (2.0 * (torch.arange(
        width, device=prediction.device, dtype=torch.float32) + 0.5)
         / width - 1.0)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).view(
        1, 1, height, width, 2).expand(
            active_batch, steps, -1, -1, -1).clone()
    # Positive delta translates right: out[y,x] = in[y,x-delta].
    grid[..., 0] -= 2.0 * active_delta[:, None, None, None] / width
    uv = active_prediction[..., :2]
    image = uv.permute(0, 1, 4, 2, 3).reshape(
        active_batch * steps, 2, height, width)
    sampled = F.grid_sample(
        image,
        grid.reshape(active_batch * steps, height, width, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).reshape(active_batch, steps, 2, height, width).permute(0, 1, 3, 4, 2)
    fallback = source_footprint_fallback_mask(active_solid, active_delta)
    sampled = torch.where(fallback[:, None, :, :, None], uv, sampled)
    result = prediction.clone()
    result[nonzero, ..., :2] = sampled
    return result


def streamwise_warp(prediction: torch.Tensor, block_shifts: torch.Tensor,
                    solid_mask: torch.Tensor) -> torch.Tensor:
    """Apply the four registered shifts with exact zero-block bypasses."""
    if (prediction.ndim != 5 or prediction.shape[1] != OUTPUT_STEPS
            or prediction.shape[-1] < 3):
        raise ValueError("prediction must have shape (B,20,H,W,C>=3)")
    if prediction.dtype != torch.float32:
        raise TypeError("formal warp requires FP32 prediction")
    batch, _steps, height, width, _channels = prediction.shape
    if solid_mask.shape != (batch, height, width):
        raise ValueError("solid mask does not align with prediction")
    shifts = _per_frame_shifts(
        block_shifts.to(device=prediction.device, dtype=torch.float32))
    if bool((shifts == 0).all()):
        return prediction.clone()
    result = prediction.clone()
    for block, (start, stop) in enumerate(HORIZON_BLOCKS):
        if bool((block_shifts[:, block] == 0).all()):
            continue
        result[:, start:stop] = _streamwise_warp_steps(
            prediction[:, start:stop], block_shifts[:, block], solid_mask)
    return result


def _torch_point_errors(prediction: torch.Tensor,
                        target: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Torch equivalent of the official per-sample Rel/TKE/MVPE formulas."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must be aligned 5D tensors")
    pred = prediction[..., :2].float()
    truth = target[..., :2].float()
    batch = len(pred)
    rel = torch.linalg.vector_norm(
        (pred - truth).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        truth.reshape(batch, -1), dim=1).clamp_min(1e-8)

    def kinetic_energy(value: torch.Tensor) -> torch.Tensor:
        centered = value - value.mean(dim=1, keepdim=True)
        return 0.5 * centered.square().mean(dim=1).sum(dim=-1)

    pred_ke = kinetic_energy(pred)
    truth_ke = kinetic_energy(truth)
    tke = torch.linalg.vector_norm(
        (pred_ke - truth_ke).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        truth_ke.reshape(batch, -1), dim=1).clamp_min(1e-8)

    height, width = pred.shape[2:4]
    probe_center_y = 16
    interval_y = min(2, int(height / 10))
    probe_y = [
        probe_center_y + interval_y * index for index in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < height]
    mvpe_parts = []
    for index in range(4):
        if 21 < width:
            probe_x = int(((index + 1) * 16 + 10) / 2)
        else:
            probe_x = int((0.5 * (index + 2) * 16 + 10) / 2)
        if probe_y and 0 <= probe_x < width:
            pp = pred[:, :, probe_y, probe_x, :].mean(dim=1).reshape(batch, -1)
            tt = truth[:, :, probe_y, probe_x, :].mean(dim=1).reshape(batch, -1)
            mvpe_parts.append(
                torch.linalg.vector_norm(pp - tt, dim=1)
                / torch.linalg.vector_norm(tt, dim=1).clamp_min(1e-8))
    mvpe = (
        torch.stack(mvpe_parts).mean(dim=0) if mvpe_parts
        else pred.new_zeros((batch,)))
    return rel, tke, mvpe


def block_sufficient_statistics(
        baseline: torch.Tensor, target: torch.Tensor,
        solid_mask: torch.Tensor, budget: Budget | None = None,
        ) -> dict[str, torch.Tensor | bool]:
    """Generate only 4x5 warped blocks and their registered statistics."""
    if baseline.shape != target.shape or baseline.dtype != torch.float32:
        raise ValueError("baseline/target must be aligned FP32 tensors")
    batch, _steps, height, width, _channels = baseline.shape
    rel_sse = baseline.new_empty((batch, 4, 5))
    temporal_sum = baseline.new_empty((batch, 4, 5, height, width, 2))
    temporal_sum_sq = torch.empty_like(temporal_sum)
    source_fallback_bitwise = True
    for block, (start, stop) in enumerate(HORIZON_BLOCKS):
        prediction_block = baseline[:, start:stop]
        target_uv = target[:, start:stop, ..., :2]
        for shift_index, value in enumerate(DX_GRID):
            if budget is not None:
                budget.check(f"block statistics block {block} dx {value}")
            delta = baseline.new_full((batch,), value)
            warped = _streamwise_warp_steps(
                prediction_block, delta, solid_mask)
            warped_uv = warped[..., :2]
            rel_sse[:, block, shift_index] = (
                warped_uv - target_uv).square().sum(dim=(1, 2, 3, 4))
            temporal_sum[:, block, shift_index] = warped_uv.sum(dim=1)
            temporal_sum_sq[:, block, shift_index] = (
                warped_uv.square().sum(dim=1))
            fallback = source_footprint_fallback_mask(solid_mask, delta)
            source_fallback_bitwise = bool(
                source_fallback_bitwise and torch.equal(
                    warped_uv[
                        fallback[:, None, :, :, None].expand_as(warped_uv)],
                    prediction_block[..., :2][
                        fallback[:, None, :, :, None].expand_as(
                            prediction_block[..., :2])]))
    return {
        "rel_sse": rel_sse,
        "temporal_sum": temporal_sum,
        "temporal_sum_sq": temporal_sum_sq,
        "source_footprint_fallback_bitwise": source_fallback_bitwise,
    }


def exhaustive_error_triples_from_statistics(
        statistics: dict[str, torch.Tensor | bool], target: torch.Tensor,
        budget: Budget | None = None) -> torch.Tensor:
    """Enumerate all 625 combinations as official-equivalent error triples."""
    rel_sse = statistics["rel_sse"]
    temporal_sum = statistics["temporal_sum"]
    temporal_sum_sq = statistics["temporal_sum_sq"]
    if not all(isinstance(value, torch.Tensor) for value in (
            rel_sse, temporal_sum, temporal_sum_sq)):
        raise TypeError("sufficient statistics are missing tensors")
    batch = len(target)
    device = target.device
    combinations = torch.from_numpy(DX_COMBINATIONS).to(device=device)
    grid_indices = torch.empty_like(combinations, dtype=torch.long)
    for index, value in enumerate(DX_GRID):
        grid_indices[combinations == value] = index

    truth = target[..., :2].float()
    rel_denom = torch.linalg.vector_norm(
        truth.reshape(batch, -1), dim=1).clamp_min(1e-8)
    centered_truth = truth - truth.mean(dim=1, keepdim=True)
    truth_ke = 0.5 * centered_truth.square().mean(dim=1).sum(dim=-1)
    tke_denom = torch.linalg.vector_norm(
        truth_ke.reshape(batch, -1), dim=1).clamp_min(1e-8)

    height, width = truth.shape[2:4]
    probe_center_y = 16
    interval_y = min(2, int(height / 10))
    probe_y = [
        probe_center_y + interval_y * index for index in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < height]
    probe_spec = []
    for index in range(4):
        probe_x = (
            int(((index + 1) * 16 + 10) / 2) if 21 < width
            else int((0.5 * (index + 2) * 16 + 10) / 2))
        if probe_y and 0 <= probe_x < width:
            truth_probe = truth[:, :, probe_y, probe_x, :].mean(
                dim=1).reshape(batch, -1)
            denominator = torch.linalg.vector_norm(
                truth_probe, dim=1).clamp_min(1e-8)
            probe_spec.append((probe_x, truth_probe, denominator))

    triples = target.new_empty((batch, COMBINATION_COUNT, 3), dtype=torch.float32)
    for combo_start in range(0, COMBINATION_COUNT, COMBINATION_CHUNK_SIZE):
        if budget is not None:
            budget.check(f"625 enumeration at combination {combo_start}")
        combo_stop = min(
            combo_start + COMBINATION_CHUNK_SIZE, COMBINATION_COUNT)
        choices = grid_indices[combo_start:combo_stop]
        combo_rel_sse = rel_sse.new_zeros((batch, len(choices)))
        combo_sum = temporal_sum.new_zeros(
            (batch, len(choices), height, width, 2))
        combo_sum_sq = torch.zeros_like(combo_sum)
        for block in range(len(HORIZON_BLOCKS)):
            selected = choices[:, block]
            combo_rel_sse += rel_sse[:, block, selected]
            combo_sum += temporal_sum[:, block, selected]
            combo_sum_sq += temporal_sum_sq[:, block, selected]
        rel_error = torch.sqrt(combo_rel_sse.clamp_min(0.0)) / rel_denom[:, None]
        temporal_mean = combo_sum / OUTPUT_STEPS
        variance = (
            combo_sum_sq / OUTPUT_STEPS - temporal_mean.square()).clamp_min(0.0)
        candidate_ke = 0.5 * variance.sum(dim=-1)
        tke_error = torch.linalg.vector_norm(
            (candidate_ke - truth_ke[:, None]).reshape(batch, len(choices), -1),
            dim=2) / tke_denom[:, None]
        if probe_spec:
            mvpe_parts = []
            for probe_x, truth_probe, denominator in probe_spec:
                candidate_probe = temporal_mean[
                    :, :, probe_y, probe_x, :].reshape(batch, len(choices), -1)
                mvpe_parts.append(torch.linalg.vector_norm(
                    candidate_probe - truth_probe[:, None], dim=2
                ) / denominator[:, None])
            mvpe_error = torch.stack(mvpe_parts).mean(dim=0)
        else:
            mvpe_error = rel_error.new_zeros(rel_error.shape)
        triples[:, combo_start:combo_stop] = torch.stack(
            (rel_error, tke_error, mvpe_error), dim=-1)
    if not bool(torch.isfinite(triples).all()):
        raise FloatingPointError("625 error triples contain non-finite values")
    return triples


def direct_official_error_triple(prediction: torch.Tensor,
                                 target: torch.Tensor) -> np.ndarray:
    pred = prediction.detach().cpu().numpy()
    truth = target.detach().cpu().numpy()
    channels = full.OFFICIAL.measured_channels(truth)
    return np.stack((
        full.OFFICIAL.rel_l2_per_sample(pred, truth, channels),
        full.OFFICIAL.tke_rel_l2_per_sample(pred, truth, channels),
        full.OFFICIAL.mvpe_rel_l2_per_sample(pred, truth),
    ), axis=1)


def direct_sanity_max_error(
        baseline: torch.Tensor, target: torch.Tensor,
        solid_mask: torch.Tensor, triples: torch.Tensor,
        combination_indices: Sequence[int]) -> float:
    maximum = 0.0
    for combo_index in combination_indices:
        shifts = torch.from_numpy(
            DX_COMBINATIONS[combo_index]).to(baseline.device)[None].expand(
                len(baseline), -1)
        candidate = streamwise_warp(baseline, shifts, solid_mask)
        direct = direct_official_error_triple(candidate, target)
        sufficient = triples[:, combo_index].detach().cpu().numpy()
        maximum = max(maximum, float(np.max(np.abs(direct - sufficient))))
    return maximum


def metrics_from_error_triples(errors: np.ndarray) -> dict[str, float]:
    if errors.ndim != 2 or errors.shape[1] != 3:
        raise ValueError("errors must have shape (N,3)")
    result = {
        "rel_l2_score": full.OFFICIAL.score_error(float(np.mean(errors[:, 0]))),
        "tke_rel_l2_score": full.OFFICIAL.score_error(float(np.mean(errors[:, 1]))),
        "mvpe_rel_l2_score": full.OFFICIAL.score_error(float(np.mean(errors[:, 2]))),
    }
    result["point_score"] = (
        0.5 * result["rel_l2_score"]
        + 0.3 * result["tke_rel_l2_score"]
        + 0.2 * result["mvpe_rel_l2_score"])
    return result


def error_comparison(baseline_errors: np.ndarray,
                     candidate_errors: np.ndarray) -> dict:
    baseline = metrics_from_error_triples(baseline_errors)
    candidate = metrics_from_error_triples(candidate_errors)
    delta = metric_delta(candidate, baseline)
    return {
        "samples": int(len(baseline_errors)),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "point_shadow_delta": shadow_delta(delta),
    }


def generous_upper_bound(error_triples: np.ndarray,
                         aoas: np.ndarray) -> dict:
    baseline_errors = error_triples[:, IDENTITY_COMBINATION_INDEX]
    independent_minima = np.min(error_triples, axis=1)
    pooled = error_comparison(baseline_errors, independent_minima)
    by_aoa = {}
    for aoa in AOA_VALUES:
        selected = aoas == aoa
        by_aoa[str(aoa)] = error_comparison(
            baseline_errors[selected], independent_minima[selected])
    return {
        "construction": "independent per-sample minimum for each metric",
        "simultaneously_achievable": False,
        "pooled": pooled,
        "by_aoa": by_aoa,
    }


def pooled_shadow_from_error_sums(error_sums: np.ndarray,
                                  samples: int) -> float:
    metrics = metrics_from_error_triples((error_sums / samples)[None])
    return float(sum(
        SHADOW_WEIGHTS[key] * metrics[key] for key in SHADOW_WEIGHTS))


def achievable_sample_coordinate_search(
        error_triples: np.ndarray) -> tuple[np.ndarray, dict]:
    """Two exact-pooled, fixed-order sample-coordinate ascent rounds."""
    if error_triples.shape[1:] != (COMBINATION_COUNT, 3):
        raise ValueError("error triples must have shape (N,625,3)")
    samples = len(error_triples)
    selection = np.full(
        samples, IDENTITY_COMBINATION_INDEX, dtype=np.int64)
    current = error_triples[np.arange(samples), selection].astype(np.float64)
    sums = np.sum(current, axis=0, dtype=np.float64)
    objective = pooled_shadow_from_error_sums(sums, samples)
    initial_objective = objective
    l1 = np.sum(np.abs(DX_COMBINATIONS), axis=1)
    priority = np.lexsort((np.arange(COMBINATION_COUNT), l1))
    priority_rank = np.empty(COMBINATION_COUNT, dtype=np.int64)
    priority_rank[priority] = np.arange(COMBINATION_COUNT)
    accepted_by_round = []

    for _round in range(SAMPLE_COORDINATE_ROUNDS):
        accepted = 0
        for sample in range(samples):
            other = sums - current[sample]
            means = (other[None] + error_triples[sample]) / samples
            scores = 100.0 / (1.0 + 0.5 * np.maximum(means, 0.0))
            objectives = (
                SHADOW_WEIGHTS["rel_l2_score"] * scores[:, 0]
                + SHADOW_WEIGHTS["tke_rel_l2_score"] * scores[:, 1]
                + SHADOW_WEIGHTS["mvpe_rel_l2_score"] * scores[:, 2])
            best_value = float(np.max(objectives))
            if best_value <= objective + OBJECTIVE_TIE_TOLERANCE:
                continue
            tied = np.flatnonzero(
                np.abs(objectives - best_value) <= OBJECTIVE_TIE_TOLERANCE)
            best = int(tied[np.argmin(priority_rank[tied])])
            sums += error_triples[sample, best] - current[sample]
            current[sample] = error_triples[sample, best]
            selection[sample] = best
            objective = pooled_shadow_from_error_sums(sums, samples)
            accepted += 1
        accepted_by_round.append(accepted)
    if objective + 1e-10 < initial_objective:
        raise RuntimeError("achievable search violated identity fallback")
    return selection, {
        "rounds": SAMPLE_COORDINATE_ROUNDS,
        "sample_order": "0..N-1 in both rounds",
        "candidate_count_per_coordinate": COMBINATION_COUNT,
        "strict_improvement_only": True,
        "accepted_updates_by_round": accepted_by_round,
        "initial_exact_pooled_shadow": initial_objective,
        "final_exact_pooled_shadow": objective,
        "objective_gain": objective - initial_objective,
    }


def official_metrics(prediction: np.ndarray,
                     target: np.ndarray) -> dict[str, float]:
    """Call the unmodified official NumPy point scorer."""
    return full.metric_bundle(prediction, target)


def metric_delta(candidate: dict[str, float],
                 baseline: dict[str, float]) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in (
            "rel_l2_score", "tke_rel_l2_score",
            "mvpe_rel_l2_score", "point_score")}


def shadow_delta(delta: dict[str, float]) -> float:
    return float(sum(
        SHADOW_WEIGHTS[key] * delta[key] for key in SHADOW_WEIGHTS))


def metric_comparison(prediction: np.ndarray, candidate: np.ndarray,
                      target: np.ndarray) -> dict:
    baseline_metrics = official_metrics(prediction, target)
    candidate_metrics = official_metrics(candidate, target)
    delta = metric_delta(candidate_metrics, baseline_metrics)
    return {
        "samples": int(len(target)),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "point_shadow_delta": shadow_delta(delta),
    }


def aoa_diagnostics(baseline: np.ndarray, candidate: np.ndarray,
                    target: np.ndarray, aoas: np.ndarray) -> dict[str, dict]:
    if aoas.shape != (len(target),):
        raise ValueError("AoA labels do not align with the cache")
    result = {}
    for aoa in AOA_VALUES:
        selected = aoas == aoa
        if not np.any(selected):
            raise ValueError(f"AoA {aoa} has no train-side sample")
        result[str(aoa)] = metric_comparison(
            baseline[selected], candidate[selected], target[selected])
    return result


def horizon_diagnostics(baseline: np.ndarray, candidate: np.ndarray,
                        target: np.ndarray) -> dict[str, dict]:
    return {
        f"{start + 1:02d}_{stop:02d}": metric_comparison(
            baseline[:, start:stop], candidate[:, start:stop],
            target[:, start:stop])
        for start, stop in HORIZON_BLOCKS}


def geometry_diagnostics(baseline: np.ndarray, candidate: np.ndarray,
                         target: np.ndarray) -> dict[str, dict]:
    result = {}
    for name, (start, stop) in GEOMETRY_REGIONS.items():
        rows = {}
        for branch, value in (("baseline", baseline), ("candidate", candidate)):
            pred = value[:, :, :, start:stop, :]
            truth = target[:, :, :, start:stop, :]
            channels = full.OFFICIAL.measured_channels(truth)
            rel_error = float(np.mean(
                full.OFFICIAL.rel_l2_per_sample(pred, truth, channels)))
            tke_error = float(np.mean(
                full.OFFICIAL.tke_rel_l2_per_sample(pred, truth, channels)))
            rows[branch] = {
                "rel_l2_score": full.OFFICIAL.score_error(rel_error),
                "tke_rel_l2_score": full.OFFICIAL.score_error(tke_error),
            }
        rows["delta"] = {
            key: float(rows["candidate"][key] - rows["baseline"][key])
            for key in rows["baseline"]}
        rows["x_slice_half_open"] = [start, stop]
        rows["gate_use"] = False
        result[name] = rows
    return result


def gate_result(upper: dict, pooled: dict, by_aoa: dict[str, dict],
                shifts: np.ndarray) -> dict:
    nonzero_sample_fraction = float(np.mean(np.any(shifts != 0.0, axis=1)))
    nonzero_choices = int(np.count_nonzero(shifts != 0.0))
    saturated_choices = int(np.count_nonzero(np.abs(shifts) == 1.0))
    saturated_ratio = (
        0.0 if nonzero_choices == 0 else saturated_choices / nonzero_choices)
    aoa_shadows = {
        key: float(value["point_shadow_delta"])
        for key, value in by_aoa.items()}
    component_nonnegative = all(
        pooled["delta"][key] >= 0.0 for key in (
            "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score"))
    checks = {
        "pooled_shadow": pooled["point_shadow_delta"] >= POOLED_SHADOW_MIN,
        "all_components_nonnegative": component_nonnegative,
        "all_aoa_positive": all(value > 0.0 for value in aoa_shadows.values()),
        "worst_aoa": min(aoa_shadows.values()) >= WORST_AOA_SHADOW_MIN,
        "nonzero_sample_fraction": (
            nonzero_sample_fraction >= MIN_NONZERO_SAMPLE_FRACTION),
        "saturated_over_nonzero_block_choices": (
            saturated_ratio <= MAX_SATURATED_NONZERO_CHOICE_RATIO),
    }
    achievable_pass = all(checks.values())
    upper_aoa = {
        key: float(value["point_shadow_delta"])
        for key, value in upper["by_aoa"].items()}
    upper_impossibility = {
        "pooled": (
            upper["pooled"]["point_shadow_delta"]
            + UPPER_BOUND_NUMERICAL_MARGIN < POOLED_SHADOW_MIN),
        "any_aoa_positive": any(
            value + UPPER_BOUND_NUMERICAL_MARGIN <= 0.0
            for value in upper_aoa.values()),
        "worst_aoa": (
            min(upper_aoa.values()) + UPPER_BOUND_NUMERICAL_MARGIN
            < WORST_AOA_SHADOW_MIN),
    }
    definitive_stop = any(upper_impossibility.values())
    if achievable_pass and definitive_stop:
        raise RuntimeError(
            "achievable lower bound exceeds an allegedly impossible upper bound")
    if achievable_pass:
        decision = "GO_WARP_ORACLE_WORTH_LOAO"
    elif definitive_stop:
        decision = "STOP_WARP_NO_HEADROOM"
    else:
        decision = "INCONCLUSIVE_WARP_HEADROOM"
    return {
        "pass": achievable_pass,
        "decision": decision,
        "achievable_checks": checks,
        "upper_bound_impossibility": upper_impossibility,
        "upper_bound_numerical_margin": UPPER_BOUND_NUMERICAL_MARGIN,
        "aoa_point_shadow_delta": aoa_shadows,
        "worst_aoa_point_shadow_delta": min(aoa_shadows.values()),
        "nonzero_sample_fraction": nonzero_sample_fraction,
        "nonzero_block_choices": nonzero_choices,
        "saturated_block_choices": saturated_choices,
        "saturated_over_nonzero_block_choice_ratio": saturated_ratio,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int,
                        default=FORMAL_EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=FORMAL_NUM_WORKERS)
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def validate_formal_inputs(args: argparse.Namespace) -> dict:
    if args.eval_batch_size != FORMAL_EVAL_BATCH_SIZE:
        raise ValueError("formal Gate D0 requires eval-batch-size=16")
    if args.num_workers != FORMAL_NUM_WORKERS:
        raise ValueError("formal Gate D0 requires num-workers=0")
    supplied_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    identities = {
        "point_checkpoint_sha256": c208.sha256(args.point_checkpoint),
        "real_stats_sha256": c208.sha256(args.real_stats),
        "split_manifest_file_sha256": c208.sha256(args.split_manifest),
        "split_manifest_canonical_sha256": canonical_sha256(supplied_manifest),
    }
    expected = {
        "point_checkpoint_sha256": FORMAL_C208_SHA256,
        "real_stats_sha256": FORMAL_REAL_STATS_SHA256,
        "split_manifest_canonical_sha256": FORMAL_SPLIT_MANIFEST_SHA256,
    }
    mismatch = {
        key: {"actual": identities[key], "expected": value}
        for key, value in expected.items() if identities[key] != value}
    if mismatch:
        raise ValueError(f"Gate-D0 provenance mismatch: {mismatch}")
    identities["split_manifest"] = supplied_manifest
    return identities


def _cache_shape(samples: int) -> tuple[int, ...]:
    return samples, OUTPUT_STEPS, 32, 64, 3


@torch.inference_mode()
def build_full_cache(model, dataset, normalizer, device: torch.device,
                     budget: Budget, eval_batch_size: int,
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(dataset) != FORMAL_COUNTS["train_windows"]:
        raise ValueError("Gate-D0 cache accepts exactly 2576 windows")
    shape = _cache_shape(len(dataset))
    baseline_cache = torch.empty(shape, dtype=torch.float32)
    history_cache = torch.empty(shape, dtype=torch.float32)
    target_cache = torch.empty(shape, dtype=torch.float32)
    loader = DataLoader(
        dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=FORMAL_NUM_WORKERS, pin_memory=True)
    offset = 0
    model.eval()
    for history, target in loader:
        budget.check("2576 cache construction")
        batch = len(history)
        history_cache[offset:offset + batch].copy_(history)
        history_device = history.to(device, non_blocking=True)
        target_device = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(
            history_device, target_device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction_norm = model(history_norm)
        prediction = normalizer.postprocess_prediction(
            prediction_norm.float())
        truth = normalizer.postprocess_target(target_norm)
        prediction[..., 2:] = 0.0
        if not bool((truth[..., 2:] == 0).all()):
            raise RuntimeError("real target pressure is not exactly zero")
        baseline_cache[offset:offset + batch].copy_(prediction.cpu())
        target_cache[offset:offset + batch].copy_(truth.cpu())
        offset += batch
    if offset != len(dataset):
        raise RuntimeError("cache did not consume exactly 2576 windows")
    for name, value in (
            ("baseline", baseline_cache), ("history", history_cache),
            ("target", target_cache)):
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"invalid {name} cache")
    return baseline_cache, history_cache, target_cache


def build_exhaustive_error_cache(
        baseline: torch.Tensor, history: torch.Tensor, target: torch.Tensor,
        solid_mask: torch.Tensor, device: torch.device, budget: Budget,
        eval_batch_size: int) -> tuple[torch.Tensor, dict]:
    error_triples = torch.empty(
        (len(baseline), COMBINATION_COUNT, 3), dtype=torch.float32)
    mask_consistent = True
    source_fallback_bitwise = True
    sanity_max_abs_error = 0.0
    for start in range(0, len(baseline), eval_batch_size):
        budget.check("2576 exhaustive error cache")
        stop = min(start + eval_batch_size, len(baseline))
        base_batch = baseline[start:stop].to(device, non_blocking=True)
        history_batch = history[start:stop].to(device, non_blocking=True)
        target_batch = target[start:stop].to(device, non_blocking=True)
        cached_solid = solid_mask[start:stop].to(device, non_blocking=True)
        mask_consistent = bool(
            mask_consistent and torch.equal(
                observation_solid_mask(history_batch), cached_solid))
        statistics = block_sufficient_statistics(
            base_batch, target_batch, cached_solid, budget)
        source_fallback_bitwise = bool(
            source_fallback_bitwise
            and statistics["source_footprint_fallback_bitwise"])
        triples = exhaustive_error_triples_from_statistics(
            statistics, target_batch, budget)
        error_triples[start:stop].copy_(triples.cpu())
        sanity_global = [
            index for index in SANITY_SAMPLE_INDICES if start <= index < stop]
        if sanity_global:
            local = torch.as_tensor(
                [index - start for index in sanity_global],
                device=device, dtype=torch.long)
            sanity_max_abs_error = max(
                sanity_max_abs_error,
                direct_sanity_max_error(
                    base_batch[local], target_batch[local],
                    cached_solid[local], triples[local],
                    SANITY_COMBINATION_INDICES))
    if not bool(torch.isfinite(error_triples).all()):
        raise FloatingPointError("full exhaustive error cache is non-finite")
    return error_triples, {
        "shape": list(error_triples.shape),
        "dtype": "float32",
        "combinations_per_sample": COMBINATION_COUNT,
        "combination_order": "lexicographic",
        "mask_consistent_2576": mask_consistent,
        "source_footprint_fallback_bitwise_all_4x5": source_fallback_bitwise,
        "sanity_sample_indices": list(SANITY_SAMPLE_INDICES),
        "sanity_combination_indices": list(SANITY_COMBINATION_INDICES),
        "sanity_max_abs_error": sanity_max_abs_error,
        "sanity_tolerance": SANITY_ERROR_TOLERANCE,
        "sanity_pass": sanity_max_abs_error <= SANITY_ERROR_TOLERANCE,
    }


def reconstruct_selected_candidate(
        baseline: torch.Tensor, history: torch.Tensor, solid_mask: torch.Tensor,
        selection: np.ndarray, device: torch.device, budget: Budget,
        eval_batch_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor, dict[str, bool]]:
    if selection.shape != (len(baseline),):
        raise ValueError("selection does not align with full cache")
    candidate = torch.empty_like(baseline)
    shifts = torch.from_numpy(DX_COMBINATIONS[selection].copy())
    identity_bitwise = True
    mask_consistent = True
    source_fallback_bitwise = True
    pressure_zero = True
    finite = True
    for start in range(0, len(baseline), eval_batch_size):
        budget.check("selected candidate reconstruction")
        stop = min(start + eval_batch_size, len(baseline))
        base_batch = baseline[start:stop].to(device, non_blocking=True)
        history_batch = history[start:stop].to(device, non_blocking=True)
        solid = solid_mask[start:stop].to(device, non_blocking=True)
        selected_shifts = shifts[start:stop].to(device, non_blocking=True)
        mask_consistent = bool(
            mask_consistent and torch.equal(
                observation_solid_mask(history_batch), solid))
        zero = base_batch.new_zeros((len(base_batch), len(HORIZON_BLOCKS)))
        identity_bitwise = bool(
            identity_bitwise
            and torch.equal(streamwise_warp(base_batch, zero, solid), base_batch))
        selected = streamwise_warp(base_batch, selected_shifts, solid)
        for block, (block_start, block_stop) in enumerate(HORIZON_BLOCKS):
            fallback = source_footprint_fallback_mask(
                solid, selected_shifts[:, block])
            source_fallback_bitwise = bool(
                source_fallback_bitwise and torch.equal(
                    selected[:, block_start:block_stop, ..., :2][
                        fallback[:, None, :, :, None].expand_as(
                            selected[:, block_start:block_stop, ..., :2])],
                    base_batch[:, block_start:block_stop, ..., :2][
                        fallback[:, None, :, :, None].expand_as(
                            base_batch[:, block_start:block_stop, ..., :2])]))
        pressure_zero = bool(
            pressure_zero and (selected[..., 2:] == 0).all())
        finite = bool(finite and torch.isfinite(selected).all())
        candidate[start:stop].copy_(selected.cpu())
    return candidate, shifts, {
        "delta0_bitwise_identity": identity_bitwise,
        "mask_consistent_2576": mask_consistent,
        "source_footprint_fallback_bitwise": source_fallback_bitwise,
        "pressure_zero": pressure_zero,
        "finite": finite,
    }


def write_result(path: Path, result: dict) -> None:
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    budget = Budget(started)
    tail = TailAccessCounters()
    identities: dict = {}
    counts: dict = {}
    raw_frames_preloaded = False
    contract_sha = canonical_sha256(EXPERIMENT_CONTRACT)

    if args.contract_only:
        result = {
            "experiment": EXPERIMENT,
            "decision": "CONTRACT_ONLY_NO_DATA_NO_GPU",
            "contract": EXPERIMENT_CONTRACT,
            "contract_canonical_sha256": contract_sha,
            **tail.report(),
        }
        write_result(result_path, result)
        print(json.dumps({
            "event": "complete", "decision": result["decision"]}))
        return

    try:
        identities = validate_formal_inputs(args)
        budget.check("provenance validation")
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal Gate D0 requires CUDA")
        budget.start_gpu()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")

        trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
        raw_frames_preloaded = True
        train_windows, runtime_manifest, tail_count = phase.make_train_side_split(
            metadata)
        counts = {
            "train_windows": len(train_windows),
            "tail_windows_in_manifest_only": tail_count,
            "trajectories": len(trajectories),
        }
        expected_counts = {
            "train_windows": FORMAL_COUNTS["train_windows"],
            "tail_windows_in_manifest_only": FORMAL_COUNTS["tail_windows"],
            "trajectories": FORMAL_COUNTS["trajectories"],
        }
        if counts != expected_counts:
            raise RuntimeError(f"fixed split counts changed: {counts}")
        runtime_manifest_sha = canonical_sha256(runtime_manifest)
        if runtime_manifest_sha != FORMAL_SPLIT_MANIFEST_SHA256:
            raise RuntimeError("runtime split canonical SHA changed")
        supplied_manifest = identities.pop("split_manifest")
        if runtime_manifest != supplied_manifest:
            raise RuntimeError("supplied split differs from runtime split")
        train_key_digest = phase.ordered_window_key_digest(
            train_windows, metadata)
        run_manifest = {
            "experiment": EXPERIMENT,
            "contract_canonical_sha256": contract_sha,
            "split_manifest_canonical_sha256": runtime_manifest_sha,
            "ordered_train_window_key_digest": train_key_digest,
            "counts": counts,
            "checkpoint_sha256": identities["point_checkpoint_sha256"],
            "real_stats_sha256": identities["real_stats_sha256"],
        }

        train_dataset = common.repro.WindowDataset(trajectories, train_windows)
        aoas = phase.window_aoas(train_windows, metadata)
        normalizer = common.repro.Normalizer(args.real_stats, device)
        point_payload = torch.load(
            args.point_checkpoint, map_location="cpu", weights_only=False)
        if (point_payload.get("phase") != "real"
                or int(point_payload.get("iteration", -1)) != 4000):
            raise ValueError("checkpoint is not frozen C208 real iteration 4000")
        config = point_payload.get("model_config", {})
        config_mismatch = {
            key: {"actual": config.get(key), "expected": value}
            for key, value in c208.EXPECTED_MODEL_CONFIG.items()
            if config.get(key) != value}
        if config_mismatch:
            raise ValueError(f"C208 model config changed: {config_mismatch}")
        point_model = c208.load_model(point_payload, device)
        common.set_model_stats(point_model, normalizer)
        point_model.requires_grad_(False)
        point_model.eval()

        baseline, history, target = build_full_cache(
            point_model, train_dataset, normalizer, device, budget,
            args.eval_batch_size)
        solid_mask = observation_solid_mask(history)
        recomputed_solid_mask = observation_solid_mask(history)
        mask_recompute_mismatch_samples = int(torch.count_nonzero(
            torch.any(
                solid_mask != recomputed_solid_mask,
                dim=tuple(range(1, solid_mask.ndim)))).item())
        mask_diagnostics = per_sample_mask_diagnostics(
            solid_mask, train_windows)
        mask_contract_matches = all(
            mask_diagnostics[key] == expected
            for key, expected in FORMAL_MASK_DIAGNOSTIC.items())
        if not mask_contract_matches:
            raise RuntimeError(
                "per-sample observation mask diagnostic fingerprint changed: "
                f"{mask_diagnostics}")
        window_mask_digest = ordered_window_mask_digest(
            train_key_digest, solid_mask)
        window_mask_digest_matches = (
            window_mask_digest == FORMAL_ORDERED_WINDOW_MASK_DIGEST)
        if not window_mask_digest_matches:
            raise RuntimeError(
                "ordered window/mask digest changed: "
                f"{window_mask_digest}")
        run_manifest["ordered_window_mask_digest"] = window_mask_digest
        cache_hash_before = {
            "baseline": tensor_sha256(baseline),
            "history": tensor_sha256(history),
            "target": tensor_sha256(target),
            "solid_mask": tensor_sha256(solid_mask.contiguous()),
        }
        del point_model, point_payload, train_dataset, train_windows, trajectories
        gc.collect()
        torch.cuda.empty_cache()

        error_triples, exhaustive_report = build_exhaustive_error_cache(
            baseline, history, target, solid_mask, device, budget,
            args.eval_batch_size)
        if not exhaustive_report["sanity_pass"]:
            raise RuntimeError(
                "sufficient-stat/direct-official sanity exceeded tolerance: "
                f"{exhaustive_report['sanity_max_abs_error']:.9g} > "
                f"{SANITY_ERROR_TOLERANCE:.9g}")
        error_array = error_triples.numpy()
        generous_upper = generous_upper_bound(error_array, aoas)
        selection, optimizer = achievable_sample_coordinate_search(error_array)
        selected_errors = error_array[np.arange(len(error_array)), selection]
        identity_errors = error_array[:, IDENTITY_COMBINATION_INDEX]
        achievable_from_triples = {
            "pooled": error_comparison(identity_errors, selected_errors),
            "by_aoa": {
                str(aoa): error_comparison(
                    identity_errors[aoas == aoa], selected_errors[aoas == aoa])
                for aoa in AOA_VALUES},
        }
        candidate, shifts, reconstruction = reconstruct_selected_candidate(
            baseline, history, solid_mask, selection, device, budget,
            args.eval_batch_size)
        torch.cuda.synchronize(device)
        budget.check("post-search synchronization")
        cache_hash_after = {
            "baseline": tensor_sha256(baseline),
            "history": tensor_sha256(history),
            "target": tensor_sha256(target),
            "solid_mask": tensor_sha256(solid_mask.contiguous()),
        }
        input_unchanged = cache_hash_before == cache_hash_after

        baseline_array = baseline.numpy()
        candidate_array = candidate.numpy()
        target_array = target.numpy()
        shift_array = shifts.numpy()
        pooled = metric_comparison(
            baseline_array, candidate_array, target_array)
        by_aoa = aoa_diagnostics(
            baseline_array, candidate_array, target_array, aoas)
        by_horizon = horizon_diagnostics(
            baseline_array, candidate_array, target_array)
        by_geometry = geometry_diagnostics(
            baseline_array, candidate_array, target_array)
        budget.check("official diagnostics")
        elapsed = budget.elapsed()
        guards = {
            "checkpoint_hash": (
                identities["point_checkpoint_sha256"] == FORMAL_C208_SHA256),
            "real_stats_hash": (
                identities["real_stats_sha256"] == FORMAL_REAL_STATS_SHA256),
            "split_manifest_hash": (
                runtime_manifest_sha == FORMAL_SPLIT_MANIFEST_SHA256),
            "tail_access_zero": not tail.report()["tail_accessed"],
            "per_sample_mask_diagnostic_fingerprint": mask_contract_matches,
            "ordered_window_mask_digest": window_mask_digest_matches,
            "mask_recompute_mismatch_samples_zero": (
                mask_recompute_mismatch_samples == 0),
            "mask_cache_consistent": bool(
                exhaustive_report["mask_consistent_2576"]
                and reconstruction["mask_consistent_2576"]),
            "direct_official_sanity": exhaustive_report["sanity_pass"],
            "delta0_bitwise_identity": reconstruction[
                "delta0_bitwise_identity"],
            "source_footprint_fallback": bool(
                exhaustive_report[
                    "source_footprint_fallback_bitwise_all_4x5"]
                and reconstruction["source_footprint_fallback_bitwise"]),
            "pressure_zero": reconstruction["pressure_zero"],
            "finite": reconstruction["finite"],
            "input_unchanged": input_unchanged,
        }
        within_budget = (
            elapsed["wall_seconds"] <= WALL_BUDGET_SECONDS
            and elapsed["gpu_seconds"] <= GPU_BUDGET_SECONDS)
        if not all(guards.values()):
            raise RuntimeError(f"Gate-D0 hard guard failed: {guards}")
        if not within_budget:
            raise BudgetExpired("budget expired before Gate-D0 decision")
        gate = gate_result(generous_upper, pooled, by_aoa, shift_array)
        result = {
            "experiment": EXPERIMENT,
            "decision": gate["decision"],
            "contract": EXPERIMENT_CONTRACT,
            "contract_canonical_sha256": contract_sha,
            "provenance": identities,
            "run_manifest": run_manifest,
            "run_manifest_canonical_sha256": canonical_sha256(run_manifest),
            "counts": counts,
            "raw_real_frames_preloaded_before_window_gate": raw_frames_preloaded,
            "tail_windows_constructed": False,
            "tail_targets_used": False,
            **tail.report(),
            "cache": {
                "cached_once": True,
                "shape": list(baseline.shape),
                "dtype": str(baseline.dtype).removeprefix("torch."),
                "hash_before": cache_hash_before,
                "hash_after": cache_hash_after,
                "solid_mask_semantics": (
                    "per_sample_observation_zero_support"),
                "solid_mask_diagnostics": mask_diagnostics,
                "mask_recompute_mismatch_samples": (
                    mask_recompute_mismatch_samples),
                "ordered_window_mask_digest": window_mask_digest,
            },
            "exhaustive_error_triples": exhaustive_report,
            "generous_upper_bound": generous_upper,
            "achievable_lower_bound_from_error_triples": (
                achievable_from_triples),
            "optimizer": optimizer,
            "selected_shifts": {
                "shape": list(shifts.shape),
                "grid": list(DX_GRID),
                "combination_indices_sha256": hashlib.sha256(
                    selection.tobytes()).hexdigest(),
                "combination_indices": selection.tolist(),
                "counts": {
                    str(value): int(np.count_nonzero(shift_array == value))
                    for value in DX_GRID},
            },
            "official_exact_pooled": pooled,
            "official_exact_by_aoa": by_aoa,
            "official_exact_by_horizon_block": by_horizon,
            "fixed_geometry_diagnostics": by_geometry,
            "guards": guards,
            "gate": gate,
            "elapsed": elapsed,
            "checkpoint_written": False,
            "zip_written": False,
        }
        write_result(result_path, result)
        print(json.dumps({
            "event": "complete", "decision": result["decision"],
            "point_shadow_delta": pooled["point_shadow_delta"]}), flush=True)
    except BudgetExpired as exc:
        result = {
            "experiment": EXPERIMENT,
            "decision": "STOP_WARP_BUDGET",
            "contract": EXPERIMENT_CONTRACT,
            "contract_canonical_sha256": contract_sha,
            "provenance": identities,
            "counts": counts,
            "raw_real_frames_preloaded_before_window_gate": raw_frames_preloaded,
            **tail.report(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed": budget.elapsed(),
            "checkpoint_written": False,
            "zip_written": False,
        }
        write_result(result_path, result)
        print(json.dumps({
            "event": "complete", "decision": result["decision"]}), flush=True)
    except Exception as exc:
        result = {
            "experiment": EXPERIMENT,
            "decision": "STOP_WARP_ERROR",
            "contract": EXPERIMENT_CONTRACT,
            "contract_canonical_sha256": contract_sha,
            "provenance": identities,
            "counts": counts,
            "raw_real_frames_preloaded_before_window_gate": raw_frames_preloaded,
            **tail.report(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed": budget.elapsed(),
            "checkpoint_written": False,
            "zip_written": False,
        }
        write_result(result_path, result)
        raise


if __name__ == "__main__":
    main()
