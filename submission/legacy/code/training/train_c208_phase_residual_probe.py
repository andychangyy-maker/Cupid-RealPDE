#!/usr/bin/env python3
"""Gate C: a frozen-C208, train-truth-only phase residual probe.

This is deliberately a probe, not a release trainer.  It freezes the exact
C208 point checkpoint, derives twelve temporal phase maps from the final eight
normalised input frames, and trains only a small zero-mean u/v residual head on
``features[-1]``.  An otherwise identical head receives zero phase maps as the
parameter-matched control.  Five train-side leave-one-AoA-out fits are the
first gate; the 725 temporal tail may be constructed and evaluated exactly
once only after that gate passes.  No teacher, new UQ training, ZIP, or public
model change is part of this experiment; the already-promoted all-tail UQ
refit is loaded only after LOAO GO to supply fixed terminal bounds.  The legacy
loader preloads all raw H5 frames; the gate controls Window/target indexing and
scientific use, not raw-frame residency.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_c208_real_sim_rehearsal as c208  # noqa: E402
import train_ensemble_uq_oof as uq_train  # noqa: E402
import train_full_backbone_point_distillation as full  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402


EXPERIMENT = "c208_gate_c_phase_residual_probe"
FORMAL_C208_SHA256 = (
    "79b3f68272580f80f1d9e09a54e9f9f5291e74524c8b11cdfd78b77a1566cbc3")
FORMAL_ALL_TAIL_REFIT_SHA256 = (
    "0cb72f227f58b8dbd8bf51cb065c08c216383fa8611d8ca1621c3467e796c892")
FORMAL_SPLIT_MANIFEST_SHA256 = (
    "b83811f908c8d55a9bd255d158f5f2648b3248a4271bca404c3085f36570dbbd")
FORMAL_REAL_STATS_SHA256 = (
    "bae5235f73431ac7d76e6bdcea480b8ca9fda0ac01e57ab26af66bad20dc3b40")
FORMAL_COUNTS = {
    "train_windows": 2576,
    "tail_windows": 725,
    "trajectories": 82,
}
AOA_VALUES = (0, 5, 10, 15, 20)

FEATURE_CHANNELS = 208
PHASE_CHANNELS = 12
PHASE_STEPS = 8
PHASE_MODES = (1, 2)
OUTPUT_STEPS = 20
OUTPUT_CHANNELS = 2
HIDDEN_CHANNELS = 32
HEAD_GROUPS = 8
MAX_HEAD_PARAMETERS = 20_000
HEAD_PARAMETER_COUNT = 8_456
MAX_NORMALIZED_CORRECTION = 0.75

FORMAL_UPDATES = 800
FORMAL_BATCH_SIZE = 16
FORMAL_EVAL_BATCH_SIZE = 16
FORMAL_LR = 3e-4
FORMAL_WEIGHT_DECAY = 0.0
FORMAL_GRAD_CLIP = 1.0
FORMAL_SEED = 0
FORMAL_WINDOW_STRIDE = 20
FORMAL_VAL_FRACTION = 0.2
WALL_BUDGET_SECONDS = 45.0 * 60.0
GPU_BUDGET_SECONDS = 30.0 * 60.0
MAX_LATENCY_RATIO = 1.02
MAX_TIME_SHADOW_RESERVE = 0.01
MVPE_SCORE_DELTA_TOLERANCE = 5e-4
ZERO_MEAN_TOLERANCE = 2e-6

FINAL_COEFFICIENTS = {
    "rel_l2_score": 0.47201126,
    "tke_rel_l2_score": 0.08586562,
    "mvpe_rel_l2_score": 0.09410214,
    "time_score": 0.10102825,
    "native_sps_score": 0.24699273,
}
GROSS_NON_TIME_SHADOW_MIN = 0.08
PHASE_INCREMENT_MIN = 0.02
SHUFFLE_GAP_MIN = 0.02
MIN_POSITIVE_FOLDS = 4
WORST_FOLD_SHADOW_MIN = -0.02
COMPONENT_FLOOR = -0.02
NET_FINAL_SHADOW_MIN = 0.07


EXPERIMENT_CONTRACT = {
    "hypothesis": (
        "C208 features[-1] contains useful amplitude/context while twelve "
        "fixed temporal Fourier phase maps expose phase-conditioned future "
        "fluctuation information not linearly available to a parameter-matched "
        "feature-only head."),
    "exact_baseline": {
        "checkpoint_sha256": FORMAL_C208_SHA256,
        "terminal_uq_refit_sha256": FORMAL_ALL_TAIL_REFIT_SHA256,
        "split_manifest_sha256": FORMAL_SPLIT_MANIFEST_SHA256,
        "real_stats_sha256": FORMAL_REAL_STATS_SHA256,
        "point_path": "unchanged frozen C208 prediction",
    },
    "single_variable": {
        "candidate": "features[-1] plus 12 fixed phase maps",
        "control": "same head, same initialisation/data/order, phase maps zeroed",
        "target": "train truth zero-mean u/v residual only",
        "forbidden": [
            "teacher", "UQ refit training", "public-model edit", "ZIP"],
    },
    "head": {
        "feature_channels": FEATURE_CHANNELS,
        "phase_channels": PHASE_CHANNELS,
        "hidden_channels": HIDDEN_CHANNELS,
        "actual_parameters": HEAD_PARAMETER_COUNT,
        "maximum_parameters": MAX_HEAD_PARAMETERS,
        "zero_initialised_output": True,
        "input_dtype": (
            "feature and phase explicitly quantized to FP16 before the head; "
            "head arithmetic is FP32"),
    },
    "identity_guards": {
        "temporal_mean": (
            "max absolute correction mean over all 20 horizons <= 2e-6"),
        "mvpe": "official MVPE score absolute delta <= 5e-4",
        "bounds": "lower/upper extents returned by object identity, unchanged",
    },
    "validation": {
        "discovery": "five train-side leave-one-AoA-out fits on 2576 windows",
        "discovery_sps": (
            "fixed at delta=0; native model_04000 bounds cannot promote Gate C"),
        "tail": "construct and evaluate the 725 tail exactly once after GO",
        "terminal_bounds": (
            "promoted all_tail_refit UQ extents; point/backbone state must match "
            "the frozen point checkpoint"),
        "shuffle_null": (
            "candidate head with far-stride interleaved held samples, then "
            "within-batch deranged phase maps; no target/AoA metadata"),
        "raw_frame_residency": (
            "legacy load_trajectories preloads all H5 frames before LOAO; no "
            "tail Window/target is indexed, trained on, or scored before GO"),
        "cache": (
            "train-only tensors live in memory for this process; never persisted "
            "or reused; ordered train-window-key digest is recorded"),
    },
    "gate": {
        "gross_non_time_shadow_min": GROSS_NON_TIME_SHADOW_MIN,
        "phase_over_control_min": PHASE_INCREMENT_MIN,
        "aligned_over_shuffle_min": SHUFFLE_GAP_MIN,
        "positive_folds_min": MIN_POSITIVE_FOLDS,
        "worst_fold_shadow_min": WORST_FOLD_SHADOW_MIN,
        "rel_delta_floor": COMPONENT_FLOOR,
        "tke_delta_floor": COMPONENT_FLOOR,
        "net_final_shadow_min": NET_FINAL_SHADOW_MIN,
    },
    "budget": {
        "updates_per_fit": FORMAL_UPDATES,
        "wall_seconds": WALL_BUDGET_SECONDS,
        "gpu_seconds": GPU_BUDGET_SECONDS,
        "latency_ratio_max": MAX_LATENCY_RATIO,
        "time_shadow_reserve_max": MAX_TIME_SHADOW_RESERVE,
    },
    "stop": (
        "STOP_PHASE_PROBE on any contract, LOAO, null, component, MVPE, "
        "bounds, budget, latency, or terminal net-shadow failure."),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: Path) -> str:
    return c208.sha256(path)


def final_head_feature(features: Sequence[torch.Tensor]) -> torch.Tensor:
    """Select and validate the only frozen C208 feature allowed by Gate C."""
    if not features:
        raise ValueError("C208 returned no decoder features")
    feature = features[-1]
    if feature.ndim != 4 or feature.shape[1] != FEATURE_CHANNELS:
        raise ValueError(
            f"features[-1] must be (B,{FEATURE_CHANNELS},H,W), got "
            f"{tuple(feature.shape)}")
    return feature


def formal_head_inputs(feature: torch.Tensor, phase: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze identical FP16 information at fit, eval, and latency boundaries."""
    return feature.to(torch.float16), phase.to(torch.float16)


def _vorticity(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Return dv/dx-du/dy using deterministic first/central differences."""
    dvdx = torch.empty_like(v)
    dudy = torch.empty_like(u)
    dvdx[..., 1:-1] = 0.5 * (v[..., 2:] - v[..., :-2])
    dvdx[..., 0] = v[..., 1] - v[..., 0]
    dvdx[..., -1] = v[..., -1] - v[..., -2]
    dudy[..., 1:-1, :] = 0.5 * (u[..., 2:, :] - u[..., :-2, :])
    dudy[..., 0, :] = u[..., 1, :] - u[..., 0, :]
    dudy[..., -1, :] = u[..., -1, :] - u[..., -2, :]
    return dvdx - dudy


def phase_maps(history_normalized: torch.Tensor) -> torch.Tensor:
    """Build [u,v,vorticity] x modes [1,2] x [cos,sin] = 12 maps."""
    if (history_normalized.ndim != 5
            or history_normalized.shape[1] < PHASE_STEPS
            or history_normalized.shape[-1] < 2):
        raise ValueError("history must have shape (B,T,H,W,C), T>=8 and C>=2")
    velocity = history_normalized[:, -PHASE_STEPS:, ..., :2].float()
    u = velocity[..., 0]
    v = velocity[..., 1]
    fields = torch.stack((u, v, _vorticity(u, v)), dim=2)
    time_index = torch.arange(
        PHASE_STEPS, device=velocity.device, dtype=velocity.dtype)
    maps: list[torch.Tensor] = []
    scale = math.sqrt(2.0 / PHASE_STEPS)
    for field_index in range(3):
        for mode in PHASE_MODES:
            angle = 2.0 * math.pi * mode * time_index / PHASE_STEPS
            for basis in (torch.cos(angle), torch.sin(angle)):
                weighted = fields[:, :, field_index] * basis.view(
                    1, PHASE_STEPS, 1, 1)
                maps.append(scale * weighted.sum(dim=1))
    result = torch.stack(maps, dim=1)
    if result.shape[1] != PHASE_CHANNELS or not bool(torch.isfinite(result).all()):
        raise FloatingPointError("invalid phase-map construction")
    return result


class PhaseResidualHead(nn.Module):
    """The complete <=20k Gate-C trainable surface."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS + PHASE_CHANNELS, HIDDEN_CHANNELS, 1),
            nn.GroupNorm(HEAD_GROUPS, HIDDEN_CHANNELS),
            nn.SiLU(),
            nn.Conv2d(
                HIDDEN_CHANNELS, OUTPUT_STEPS * OUTPUT_CHANNELS, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        parameters = sum(value.numel() for value in self.parameters())
        if parameters != HEAD_PARAMETER_COUNT:
            raise RuntimeError(
                f"phase residual head has {parameters} != {HEAD_PARAMETER_COUNT}")
        if parameters > MAX_HEAD_PARAMETERS:
            raise RuntimeError(
                f"phase residual head has {parameters} > {MAX_HEAD_PARAMETERS}")

    def forward(self, feature: torch.Tensor, phase: torch.Tensor,
                *, phase_enabled: bool = True) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != FEATURE_CHANNELS:
            raise ValueError("invalid frozen feature shape")
        if (phase.ndim != 4 or phase.shape[1] != PHASE_CHANNELS
                or phase.shape[0] != feature.shape[0]
                or phase.shape[-2:] != feature.shape[-2:]):
            raise ValueError("phase maps do not align with the frozen feature")
        if not phase_enabled:
            phase = torch.zeros_like(phase)
        raw = self.body(torch.cat((feature.float(), phase.float()), dim=1))
        batch, _, height, width = raw.shape
        correction = MAX_NORMALIZED_CORRECTION * torch.tanh(
            raw.view(batch, OUTPUT_STEPS, OUTPUT_CHANNELS, height, width)
            .permute(0, 1, 3, 4, 2))
        return correction - correction.mean(dim=1, keepdim=True)


def new_parameter_matched_heads(seed: int = FORMAL_SEED
                                ) -> tuple[PhaseResidualHead, PhaseResidualHead]:
    seed_everything(seed)
    candidate = PhaseResidualHead()
    control = copy.deepcopy(candidate)
    if sum(p.numel() for p in candidate.parameters()) != sum(
            p.numel() for p in control.parameters()):
        raise RuntimeError("candidate/control parameter counts differ")
    return candidate, control


def corrected_distribution(
        baseline: torch.Tensor, lower_extent: torch.Tensor,
        upper_extent: torch.Tensor, correction_uv: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the full-field zero-mean correction without changing extents."""
    if baseline.ndim != 5 or baseline.shape[-1] < 2:
        raise ValueError("baseline must have shape (B,T,H,W,C>=2)")
    if correction_uv.shape != (*baseline.shape[:-1], 2):
        raise ValueError("correction does not align with baseline u/v")
    corrected_uv = baseline[..., :2].float() + correction_uv.float()
    corrected = torch.cat((corrected_uv, baseline[..., 2:].float()), dim=-1)
    return corrected, lower_extent, upper_extent


def assert_identity_guards(
        baseline: torch.Tensor, corrected: torch.Tensor,
        lower_before: torch.Tensor, upper_before: torch.Tensor,
        lower_after: torch.Tensor, upper_after: torch.Tensor,
        ) -> None:
    if lower_after is not lower_before or upper_after is not upper_before:
        raise RuntimeError("prediction-bound object identity guard failed")
    if not torch.equal(lower_after, lower_before) or not torch.equal(
            upper_after, upper_before):
        raise RuntimeError("prediction bounds changed")


def window_aoas(windows: Sequence, metadata: Sequence[dict]) -> np.ndarray:
    return np.asarray([
        int(metadata[window.trajectory]["aoa"]) for window in windows],
        dtype=np.int64)


def make_train_side_split(metadata: Sequence[dict]) -> tuple[list, list[dict], int]:
    """Reproduce make_split without constructing any 725 Window objects."""
    train = []
    manifest = []
    tail_count = 0
    for trajectory, item in enumerate(metadata):
        starts = list(range(
            0, int(item["frames"]) - 40 + 1, FORMAL_WINDOW_STRIDE))
        if len(starts) < 2:
            raise ValueError(f"trajectory {item['file']} is too short")
        n_tail = max(1, math.ceil(len(starts) * FORMAL_VAL_FRACTION))
        tail_starts = starts[-n_tail:]
        boundary = tail_starts[0]
        train_starts = [start for start in starts if start + 40 <= boundary]
        if not train_starts:
            raise ValueError(f"no train window remains for {item['file']}")
        train.extend(
            common.repro.Window(trajectory, start) for start in train_starts)
        manifest.append({
            **item,
            "train_starts": train_starts,
            "val_starts": tail_starts,
            "val_boundary": boundary,
        })
        tail_count += len(tail_starts)
    return train, manifest, tail_count


def tail_windows_from_manifest(manifest: Sequence[dict]) -> list:
    """Construct the 725 Window objects only inside the promoted tail factory."""
    return [
        common.repro.Window(trajectory, start)
        for trajectory, item in enumerate(manifest)
        for start in item["val_starts"]]


def train_side_loao_partitions(aoas: Sequence[int]) -> dict[int, dict[str, list[int]]]:
    labels = np.asarray(aoas, dtype=np.int64)
    if set(labels.tolist()) != set(AOA_VALUES):
        raise ValueError("train-side windows do not contain the five frozen AoAs")
    indices = np.arange(len(labels))
    result = {}
    for held in AOA_VALUES:
        evaluate = indices[labels == held].tolist()
        fit = indices[labels != held].tolist()
        if not fit or not evaluate or set(fit).intersection(evaluate):
            raise RuntimeError("invalid train-side LOAO partition")
        result[held] = {"fit": fit, "evaluate": evaluate}
    if sorted(i for row in result.values() for i in row["evaluate"]) != list(
            range(len(labels))):
        raise RuntimeError("train-side LOAO evaluations do not partition 2576")
    return result


def derange_phase_batch(phase: torch.Tensor) -> torch.Tensor:
    """Deterministic shuffle-null; no target or second dataset read is needed."""
    if phase.ndim != 4 or phase.shape[0] < 2:
        raise ValueError("shuffle-null requires an evaluation batch of at least two")
    return torch.roll(phase, shifts=1, dims=0)


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]
                 ) -> dict[str, float]:
    keys = ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
            "native_sps_score")
    return {key: float(candidate[key]) - float(baseline[key]) for key in keys}


def non_time_shadow(delta: dict[str, float]) -> float:
    return float(sum(
        FINAL_COEFFICIENTS[key] * float(delta[key])
        for key in ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
                    "native_sps_score")))


def _fold_summary(row: dict) -> dict:
    candidate_delta = metric_delta(row["candidate"], row["baseline"])
    control_delta = metric_delta(row["control"], row["baseline"])
    shuffle_delta = metric_delta(row["shuffle"], row["baseline"])
    candidate_shadow = non_time_shadow(candidate_delta)
    return {
        "candidate_delta": candidate_delta,
        "control_delta": control_delta,
        "shuffle_delta": shuffle_delta,
        "candidate_shadow": candidate_shadow,
        "control_shadow": non_time_shadow(control_delta),
        "shuffle_shadow": non_time_shadow(shuffle_delta),
        "phase_increment": candidate_shadow - non_time_shadow(control_delta),
        "shuffle_gap": candidate_shadow - non_time_shadow(shuffle_delta),
    }


def loao_gate(rows: Sequence[dict]) -> dict:
    if len(rows) != len(AOA_VALUES):
        raise ValueError("Gate C requires exactly five train-side LOAO rows")
    summaries = [_fold_summary(row) for row in rows]
    counts = np.asarray([int(row.get("samples", 0)) for row in rows], dtype=np.float64)
    if bool((counts <= 0).any()):
        raise ValueError("each LOAO row needs a positive sample count")
    weights = counts / counts.sum()
    if all(isinstance(row.get("point_error_state"), dict) for row in rows):
        pooled_metrics = {
            branch: metrics_from_point_error_state(merge_point_error_states([
                row["point_error_state"][branch] for row in rows]))
            for branch in ("baseline", "control", "candidate", "shuffle")}
        pooled_summary = _fold_summary(pooled_metrics)
        pooled = {
            key: pooled_summary[key]
            for key in ("candidate_shadow", "control_shadow", "shuffle_shadow",
                        "phase_increment", "shuffle_gap")}
        pooled_delta = pooled_summary["candidate_delta"]
        pooling = "exact_2576_per_sample_error_sums"
    else:
        pooled = {
            key: float(sum(weight * summary[key]
                           for weight, summary in zip(weights, summaries)))
            for key in ("candidate_shadow", "control_shadow", "shuffle_shadow",
                        "phase_increment", "shuffle_gap")}
        pooled_delta = {
            key: float(sum(
                weight * summary["candidate_delta"][key]
                for weight, summary in zip(weights, summaries)))
            for key in ("rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
                        "native_sps_score")}
        pooling = "weighted_score_fallback"
    conditions = {
        "gross_shadow": pooled["candidate_shadow"] >= GROSS_NON_TIME_SHADOW_MIN,
        "phase_over_control": pooled["phase_increment"] >= PHASE_INCREMENT_MIN,
        "aligned_over_shuffle": pooled["shuffle_gap"] >= SHUFFLE_GAP_MIN,
        "positive_folds": sum(
            summary["candidate_shadow"] > 0.0 for summary in summaries)
            >= MIN_POSITIVE_FOLDS,
        "worst_fold": min(summary["candidate_shadow"] for summary in summaries)
            >= WORST_FOLD_SHADOW_MIN,
        "rel_floor": pooled_delta["rel_l2_score"] >= COMPONENT_FLOOR,
        "tke_floor": pooled_delta["tke_rel_l2_score"] >= COMPONENT_FLOOR,
        "mvpe_invariant": all(bool(row.get("mvpe_invariant", False)) for row in rows),
        "zero_mean": all(
            float(row.get("max_abs_correction_temporal_mean", math.inf))
            <= ZERO_MEAN_TOLERANCE for row in rows),
        "no_native_uq": all(
            row.get("sps_source") == "fixed_delta_zero_no_native_uq"
            and all(float(row[branch]["native_sps_score"]) == 0.0
                    for branch in ("baseline", "control", "candidate", "shuffle"))
            for row in rows),
    }
    return {
        "pass": all(conditions.values()),
        "decision": "GO_PHASE_PROBE_TERMINAL_725" if all(conditions.values())
        else "STOP_PHASE_PROBE",
        "conditions": conditions,
        "pooled": pooled,
        "pooled_candidate_delta": pooled_delta,
        "pooling": pooling,
        "folds": summaries,
    }


def terminal_gate(row: dict, latency_ratio: float, *, time_shadow: float | None = None,
                  wall_elapsed: float = 0.0, gpu_elapsed: float = 0.0) -> dict:
    summary = _fold_summary(row)
    gross = summary["candidate_shadow"]
    applied_time_shadow = (
        -MAX_TIME_SHADOW_RESERVE if time_shadow is None else float(time_shadow))
    net_shadow = gross + applied_time_shadow
    conditions = {
        "gross_shadow": gross >= GROSS_NON_TIME_SHADOW_MIN,
        "phase_over_control": summary["phase_increment"] >= PHASE_INCREMENT_MIN,
        "aligned_over_shuffle": summary["shuffle_gap"] >= SHUFFLE_GAP_MIN,
        "rel_floor": summary["candidate_delta"]["rel_l2_score"] >= COMPONENT_FLOOR,
        "tke_floor": summary["candidate_delta"]["tke_rel_l2_score"] >= COMPONENT_FLOOR,
        "mvpe_invariant": bool(row.get("mvpe_invariant", False)),
        "zero_mean": float(row.get(
            "max_abs_correction_temporal_mean", math.inf)) <= ZERO_MEAN_TOLERANCE,
        "bounds_bitwise": bool(row.get("bounds_bitwise", False)),
        "promoted_bounds": row.get("sps_source") == "promoted_all_tail_refit",
        "latency": math.isfinite(latency_ratio) and latency_ratio <= MAX_LATENCY_RATIO,
        "time_shadow": applied_time_shadow >= -MAX_TIME_SHADOW_RESERVE,
        "net_shadow": net_shadow >= NET_FINAL_SHADOW_MIN,
        "wall_budget": wall_elapsed <= WALL_BUDGET_SECONDS,
        "gpu_budget": gpu_elapsed <= GPU_BUDGET_SECONDS,
    }
    return {
        "pass": all(conditions.values()),
        "decision": "GO_PHASE_PROBE_ARTIFACT" if all(conditions.values())
        else "STOP_PHASE_PROBE",
        "conditions": conditions,
        "metrics": summary,
        "latency_ratio": float(latency_ratio),
        "time_shadow": applied_time_shadow,
        "time_shadow_floor": -MAX_TIME_SHADOW_RESERVE,
        "net_shadow": net_shadow,
        "wall_elapsed_seconds": float(wall_elapsed),
        "gpu_elapsed_seconds": float(gpu_elapsed),
    }


def run_terminal_once(loao_result: dict, tail_factory: Callable[[], object],
                      evaluator: Callable[[object], dict]) -> dict | None:
    """The only code path allowed to construct/read the 725 tail."""
    if not bool(loao_result.get("pass", False)):
        return None
    tail = tail_factory()
    return evaluator(tail)


def tail_access_report(constructions: int, attempts: int,
                       completed: int) -> dict:
    if (constructions < 0 or attempts < 0 or completed < 0
            or completed > attempts or attempts > constructions):
        raise ValueError("invalid tail access counters")
    return {
        "tail_dataset_constructed": constructions > 0,
        "tail_constructions": constructions,
        "tail_evaluation_attempts": attempts,
        "tail_evaluations_completed": completed,
    }


class CachedProbeDataset(Dataset):
    def __init__(self, feature: torch.Tensor, phase: torch.Tensor,
                 target_residual: torch.Tensor) -> None:
        if not (len(feature) == len(phase) == len(target_residual)):
            raise ValueError("cache tensor lengths differ")
        self.feature = feature
        self.phase = phase
        self.target_residual = target_residual

    def __len__(self) -> int:
        return len(self.feature)

    def __getitem__(self, index: int):
        return (self.feature[index], self.phase[index],
                self.target_residual[index])


def target_zero_mean_residual(target: torch.Tensor,
                              baseline: torch.Tensor) -> torch.Tensor:
    residual = target[..., :2].float() - baseline[..., :2].float()
    return residual - residual.mean(dim=1, keepdim=True)


def train_head_pair(cache: CachedProbeDataset, fit_indices: Sequence[int],
                    device: torch.device, *, updates: int = FORMAL_UPDATES,
                    seed: int = FORMAL_SEED, deadline: float | None = None,
                    ) -> tuple[PhaseResidualHead, PhaseResidualHead, dict]:
    """Fit candidate/control on the exact same sample stream and target."""
    candidate, control = new_parameter_matched_heads(seed)
    candidate.to(device)
    control.to(device)
    loader = DataLoader(
        Subset(cache, list(fit_indices)), batch_size=FORMAL_BATCH_SIZE,
        shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed))
    if not len(loader):
        raise ValueError("empty Gate-C fit partition")
    candidate_optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=FORMAL_LR,
        weight_decay=FORMAL_WEIGHT_DECAY)
    control_optimizer = torch.optim.AdamW(
        control.parameters(), lr=FORMAL_LR,
        weight_decay=FORMAL_WEIGHT_DECAY)
    iterator = iter(loader)
    final_losses = {}
    for update in range(1, updates + 1):
        if deadline is not None and update % 25 == 1 and time.perf_counter() > deadline:
            raise TimeoutError("Gate-C GPU budget expired during head fit")
        try:
            feature, phase, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            feature, phase, target = next(iterator)
        feature, phase = formal_head_inputs(
            feature.to(device), phase.to(device))
        target = target.to(device, dtype=torch.float32)
        candidate_optimizer.zero_grad(set_to_none=True)
        control_optimizer.zero_grad(set_to_none=True)
        candidate_loss = F.smooth_l1_loss(
            candidate(feature, phase, phase_enabled=True), target)
        control_loss = F.smooth_l1_loss(
            control(feature, phase, phase_enabled=False), target)
        (candidate_loss + control_loss).backward()
        nn.utils.clip_grad_norm_(candidate.parameters(), FORMAL_GRAD_CLIP)
        nn.utils.clip_grad_norm_(control.parameters(), FORMAL_GRAD_CLIP)
        candidate_optimizer.step()
        control_optimizer.step()
        if update == updates:
            final_losses = {
                "candidate": float(candidate_loss.detach()),
                "control": float(control_loss.detach()),
            }
    return candidate.eval(), control.eval(), final_losses


def ordered_window_key_digest(windows: Sequence, metadata: Sequence[dict]) -> str:
    keys = [
        {"file": metadata[window.trajectory]["file"], "start": int(window.start)}
        for window in windows]
    return c208.canonical_sha256(keys)


def validate_cache_tensors(cache: CachedProbeDataset,
                           expected_samples: int) -> dict:
    expected = {
        "feature": (expected_samples, FEATURE_CHANNELS, 32, 64),
        "phase": (expected_samples, PHASE_CHANNELS, 32, 64),
        "target_residual": (
            expected_samples, OUTPUT_STEPS, 32, 64, OUTPUT_CHANNELS),
    }
    tensors = {
        "feature": cache.feature,
        "phase": cache.phase,
        "target_residual": cache.target_residual,
    }
    report = {"persisted": False, "tensors": {}}
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(
                f"cache {name} shape {tuple(tensor.shape)} != {expected[name]}")
        if tensor.dtype != torch.float16:
            raise ValueError(f"cache {name} dtype {tensor.dtype} != torch.float16")
        finite = True
        for start in range(0, len(tensor), FORMAL_EVAL_BATCH_SIZE):
            finite = bool(finite and torch.isfinite(
                tensor[start:start + FORMAL_EVAL_BATCH_SIZE]).all())
        if not finite:
            raise ValueError(f"cache {name} contains non-finite values")
        report["tensors"][name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "finite": True,
        }
    return report


@torch.inference_mode()
def build_probe_cache(model, dataset: Dataset, normalizer, device: torch.device,
                      *, deadline: float | None = None,
                      ) -> CachedProbeDataset:
    """Create the process-local cache using only the 2576 train windows."""
    if len(dataset) != FORMAL_COUNTS["train_windows"]:
        raise ValueError("Gate-C cache builder accepts exactly 2576 windows")
    feature = torch.empty(
        (len(dataset), FEATURE_CHANNELS, 32, 64), dtype=torch.float16)
    phase = torch.empty(
        (len(dataset), PHASE_CHANNELS, 32, 64), dtype=torch.float16)
    residual = torch.empty(
        (len(dataset), OUTPUT_STEPS, 32, 64, OUTPUT_CHANNELS),
        dtype=torch.float16)
    loader = DataLoader(
        dataset, batch_size=FORMAL_EVAL_BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True)
    offset = 0
    model.eval()
    for history, target in loader:
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("Gate-C GPU budget expired during train cache build")
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history, target)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            baseline_norm, features = model.forward_with_features(history_norm)
        batch = len(history)
        feature[offset:offset + batch].copy_(
            final_head_feature(features).detach().cpu().to(torch.float16))
        phase[offset:offset + batch].copy_(
            phase_maps(history_norm).detach().cpu().to(torch.float16))
        residual[offset:offset + batch].copy_(
            target_zero_mean_residual(
                target_norm, baseline_norm).detach().cpu().to(torch.float16))
        offset += batch
    if offset != len(dataset):
        raise RuntimeError("train cache did not consume exactly 2576 windows")
    return CachedProbeDataset(feature, phase, residual)


def far_stride_batches(indices: Sequence[int], batch_size: int
                       ) -> list[list[int]]:
    """Interleave the full held set so roll-null sources are far apart."""
    if batch_size < 2 or len(indices) < 2:
        raise ValueError("shuffle-null evaluation requires >=2 samples/batch")
    stride = math.ceil(len(indices) / batch_size)
    batches = [
        [indices[position] for position in range(offset, len(indices), stride)]
        for offset in range(stride)]
    if any(len(batch) < 2 or len(batch) > batch_size for batch in batches):
        raise RuntimeError("far-stride batching produced an invalid null batch")
    return batches


def _metric_bundle_with_sps(
        prediction: np.ndarray, target: np.ndarray,
        lower: np.ndarray | None, upper: np.ndarray | None,
        ) -> dict[str, float]:
    result = full.metric_bundle(prediction, target)
    if lower is None or upper is None:
        # Discovery deliberately has no permission to use the stale native UQ.
        result["native_sps_score"] = 0.0
        result["native_sps_coverage"] = 0.0
        return result
    channels = full.OFFICIAL.measured_channels(target)
    sps, coverage = full.OFFICIAL.aggregate_sps(
        prediction, target, channels, lower=lower, upper=upper)
    result["native_sps_score"] = full.OFFICIAL.score_sps(sps)
    result["native_sps_coverage"] = float(coverage)
    return result


def point_error_state(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    channels = full.OFFICIAL.measured_channels(target)
    return {
        "samples": int(len(target)),
        "rel_error_sum": float(np.sum(
            full.OFFICIAL.rel_l2_per_sample(prediction, target, channels),
            dtype=np.float64)),
        "tke_error_sum": float(np.sum(
            full.OFFICIAL.tke_rel_l2_per_sample(prediction, target, channels),
            dtype=np.float64)),
        "mvpe_error_sum": float(np.sum(
            full.OFFICIAL.mvpe_rel_l2_per_sample(prediction, target),
            dtype=np.float64)),
    }


def metrics_from_point_error_state(state: dict[str, float]) -> dict[str, float]:
    samples = int(state["samples"])
    if samples <= 0:
        raise ValueError("point error state has no samples")
    result = {
        "rel_l2_score": full.OFFICIAL.score_error(
            float(state["rel_error_sum"]) / samples),
        "tke_rel_l2_score": full.OFFICIAL.score_error(
            float(state["tke_error_sum"]) / samples),
        "mvpe_rel_l2_score": full.OFFICIAL.score_error(
            float(state["mvpe_error_sum"]) / samples),
        "native_sps_score": 0.0,
    }
    result["point_score"] = (
        0.5 * result["rel_l2_score"]
        + 0.3 * result["tke_rel_l2_score"]
        + 0.2 * result["mvpe_rel_l2_score"])
    return result


def merge_point_error_states(states: Sequence[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(float(state[key]) for state in states)
        for key in ("samples", "rel_error_sum", "tke_error_sum",
                    "mvpe_error_sum")}


@torch.inference_mode()
def evaluate_heads(
        model, candidate: PhaseResidualHead, control: PhaseResidualHead,
        dataset: Dataset, indices: Sequence[int], normalizer,
        device: torch.device, *, with_promoted_bounds: bool,
        deadline: float | None = None, capture_latency_input: bool = False,
        ) -> dict:
    """Evaluate baseline/control/aligned/shuffle in one frozen-model pass."""
    batches = far_stride_batches(indices, FORMAL_EVAL_BATCH_SIZE)
    loader = DataLoader(
        dataset, batch_sampler=batches, num_workers=0, pin_memory=True)
    branches = ("baseline", "control", "candidate", "shuffle")
    predictions: dict[str, list[np.ndarray]] = {key: [] for key in branches}
    targets: list[np.ndarray] = []
    lower_extents: list[np.ndarray] = []
    upper_extents: list[np.ndarray] = []
    bounds_bitwise = True
    max_temporal_mean = 0.0
    latency_input = None
    model.eval()
    candidate.eval()
    control.eval()
    target_std = normalizer.std_target[:2].view(1, 1, 1, 1, 2)

    for history, target in loader:
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("Gate-C GPU budget expired during evaluation")
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, target_norm = normalizer.preprocess(history, target)
        if capture_latency_input and latency_input is None:
            latency_input = history_norm.detach().clone()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if with_promoted_bounds:
                baseline_norm, lower_extent, upper_extent, features = (
                    model.forward_distribution_with_features(history_norm))
            else:
                baseline_norm, features = model.forward_with_features(history_norm)
                lower_extent = upper_extent = None
        feature, phase = formal_head_inputs(
            final_head_feature(features).detach(), phase_maps(history_norm))
        corrections = {
            "control": control(feature, phase, phase_enabled=False),
            "candidate": candidate(feature, phase, phase_enabled=True),
            "shuffle": candidate(
                feature, derange_phase_batch(phase), phase_enabled=True),
        }
        corrected_norm = {}
        for key, correction in corrections.items():
            max_temporal_mean = max(
                max_temporal_mean,
                float(correction.mean(dim=1).abs().max()))
            if with_promoted_bounds:
                corrected, lower_after, upper_after = corrected_distribution(
                    baseline_norm, lower_extent, upper_extent, correction)
                assert_identity_guards(
                    baseline_norm.float(), corrected,
                    lower_extent, upper_extent, lower_after, upper_after)
                bounds_bitwise = bool(
                    bounds_bitwise and lower_after is lower_extent
                    and upper_after is upper_extent)
            else:
                dummy = baseline_norm.new_zeros(())
                corrected, _, _ = corrected_distribution(
                    baseline_norm, dummy, dummy, correction)
            corrected_norm[key] = corrected

        physical = {
            "baseline": normalizer.postprocess_prediction(
                baseline_norm.float())}
        physical.update({
            key: normalizer.postprocess_prediction(value)
            for key, value in corrected_norm.items()})
        for key, value in physical.items():
            predictions[key].append(value.cpu().numpy())
        targets.append(
            normalizer.postprocess_target(target_norm).cpu().numpy())
        if with_promoted_bounds:
            lower_extents.append(
                (lower_extent.float() * target_std).cpu().numpy())
            upper_extents.append(
                (upper_extent.float() * target_std).cpu().numpy())

    arrays = {key: np.concatenate(value) for key, value in predictions.items()}
    target_array = np.concatenate(targets)
    lower_array = np.concatenate(lower_extents) if lower_extents else None
    upper_array = np.concatenate(upper_extents) if upper_extents else None
    metrics = {}
    error_states = {}
    for key in branches:
        lower = None if lower_array is None else arrays[key][..., :2] - lower_array
        upper = None if upper_array is None else arrays[key][..., :2] + upper_array
        metrics[key] = _metric_bundle_with_sps(
            arrays[key], target_array, lower, upper)
        error_states[key] = point_error_state(arrays[key], target_array)

    mvpe_score_deltas = {
        key: float(metrics[key]["mvpe_rel_l2_score"]
                   - metrics["baseline"]["mvpe_rel_l2_score"])
        for key in ("control", "candidate", "shuffle")}
    mvpe_invariant = max(
        abs(value) for value in mvpe_score_deltas.values()
    ) <= MVPE_SCORE_DELTA_TOLERANCE
    result = {
        "samples": int(len(target_array)),
        **metrics,
        "point_error_state": error_states,
        "mvpe_invariant": mvpe_invariant,
        "mvpe_score_deltas": mvpe_score_deltas,
        "bounds_bitwise": bounds_bitwise,
        "max_abs_correction_temporal_mean": max_temporal_mean,
        "sps_source": (
            "promoted_all_tail_refit" if with_promoted_bounds
            else "fixed_delta_zero_no_native_uq"),
    }
    if capture_latency_input:
        if latency_input is None:
            raise RuntimeError("terminal evaluation captured no latency input")
        result["_latency_input"] = latency_input
    return result


def assert_refit_point_state_identity(point_payload: dict,
                                      refit_payload: dict) -> None:
    point_state = point_payload.get("model_state_dict")
    refit_state = refit_payload.get("model_state_dict")
    if not isinstance(point_state, dict) or not isinstance(refit_state, dict):
        raise ValueError("point/refit checkpoint lacks model_state_dict")
    differing = []
    for key, value in point_state.items():
        if key.startswith(uq_train.UQ_PREFIXES):
            continue
        other = refit_state.get(key)
        if other is None or not torch.equal(value.cpu(), other.cpu()):
            differing.append(key)
    if differing:
        raise RuntimeError(
            "all-tail refit changed frozen point/backbone state: "
            f"{differing[:5]}")


@torch.inference_mode()
def paired_latency_ratio(model, head: PhaseResidualHead,
                         history_normalized: torch.Tensor,
                         *, warmup: int = 3, repeats: int = 10) -> dict:
    if history_normalized.device.type != "cuda":
        raise ValueError("formal latency measurement requires CUDA")
    model.eval()
    head.eval()

    def baseline_call() -> None:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.forward_distribution(history_normalized)

    def candidate_call() -> None:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base, lower, upper, features = model.forward_distribution_with_features(
                history_normalized)
        feature, phase = formal_head_inputs(
            final_head_feature(features), phase_maps(history_normalized))
        correction = head(feature, phase)
        corrected_distribution(base, lower, upper, correction)

    for _ in range(warmup):
        baseline_call()
        candidate_call()
    torch.cuda.synchronize()
    values = {"baseline": [], "candidate": []}
    for repeat in range(repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else (
            "candidate", "baseline")
        for key in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            (baseline_call if key == "baseline" else candidate_call)()
            end.record()
            end.synchronize()
            values[key].append(float(start.elapsed_time(end)))
    baseline_ms = float(np.median(values["baseline"]))
    candidate_ms = float(np.median(values["candidate"]))
    batch = int(history_normalized.shape[0])
    baseline_time_score = full.OFFICIAL.score_time(
        baseline_ms / (1000.0 * batch))
    candidate_time_score = full.OFFICIAL.score_time(
        candidate_ms / (1000.0 * batch))
    time_score_delta = candidate_time_score - baseline_time_score
    return {
        "baseline_median_ms": baseline_ms,
        "candidate_median_ms": candidate_ms,
        "ratio": candidate_ms / baseline_ms,
        "batch_size": batch,
        "baseline_time_score_local": baseline_time_score,
        "candidate_time_score_local": candidate_time_score,
        "time_score_delta_local": time_score_delta,
        "time_shadow_local": (
            FINAL_COEFFICIENTS["time_score"] * time_score_delta),
        "warmup": warmup,
        "repeats": repeats,
        "order": "alternating_AB_BA",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--all-tail-refit", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--contract-only", action="store_true",
        help="Write the frozen hypothesis/gates without touching data or CUDA.")
    return parser.parse_args()


def validate_formal_inputs(args: argparse.Namespace) -> dict:
    if args.num_workers != 0:
        raise ValueError("formal Gate C requires num-workers=0")
    identities = {
        "point_checkpoint": file_sha256(args.point_checkpoint),
        "real_stats": file_sha256(args.real_stats),
    }
    supplied_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    identities["split_manifest_canonical"] = c208.canonical_sha256(
        supplied_manifest)
    identities["split_manifest_file_sha256"] = file_sha256(args.split_manifest)
    expected = {
        "point_checkpoint": FORMAL_C208_SHA256,
        "split_manifest_canonical": FORMAL_SPLIT_MANIFEST_SHA256,
        "real_stats": FORMAL_REAL_STATS_SHA256,
    }
    mismatch = {
        key: (identities[key], expected[key]) for key in expected
        if identities[key] != expected[key]}
    if mismatch:
        raise ValueError(f"Gate-C provenance mismatch: {mismatch}")
    identities["split_manifest"] = supplied_manifest
    return identities


def validate_refit_input(path: Path) -> str:
    actual = file_sha256(path)
    if actual != FORMAL_ALL_TAIL_REFIT_SHA256:
        raise ValueError(
            f"all-tail refit SHA {actual} != {FORMAL_ALL_TAIL_REFIT_SHA256}")
    return actual


def main() -> None:
    args = parse_args()
    if ((args.output_dir / "summary.json").exists()
            or (args.output_dir / "loao_result.json").exists()):
        raise FileExistsError(f"refusing to overwrite completed {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "contract.json"
    contract_path.write_text(
        json.dumps(EXPERIMENT_CONTRACT, indent=2, sort_keys=True),
        encoding="utf-8")
    if args.contract_only:
        print(json.dumps({
            "experiment": EXPERIMENT,
            "decision": "CONTRACT_ONLY_NO_GPU",
            "contract": str(contract_path.resolve()),
        }, sort_keys=True))
        return
    started = time.perf_counter()
    identities = validate_formal_inputs(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Gate C requires CUDA")
    seed_everything(FORMAL_SEED)
    torch.set_float32_matmul_precision("high")

    trajectories, metadata = common.repro.load_trajectories(args.real_dir, 0)
    train_windows, runtime_manifest, tail_count = make_train_side_split(metadata)
    counts = {
        "train_windows": len(train_windows),
        "tail_windows": tail_count,
        "trajectories": len(trajectories),
    }
    if counts != FORMAL_COUNTS:
        raise RuntimeError(f"fixed split counts changed: {counts}")
    runtime_manifest_sha = c208.canonical_sha256(runtime_manifest)
    if runtime_manifest_sha != FORMAL_SPLIT_MANIFEST_SHA256:
        raise RuntimeError("runtime split canonical SHA differs from Gate C")
    if runtime_manifest != identities.pop("split_manifest"):
        raise RuntimeError("supplied split manifest content differs from make_split")

    # Only the 2576 dataset is constructed before LOAO promotion.  The
    # manifest carries tail start integers, but no 725 Window/Dataset exists.
    train_dataset = common.repro.WindowDataset(trajectories, train_windows)
    aoas = window_aoas(train_windows, metadata)
    partitions = train_side_loao_partitions(aoas)
    normalizer = common.repro.Normalizer(args.real_stats, device)
    point_payload = torch.load(
        args.point_checkpoint, map_location="cpu", weights_only=False)
    if point_payload.get("phase") != "real" or int(
            point_payload.get("iteration", -1)) != 4000:
        raise ValueError("point checkpoint is not the frozen C208 real-4000 endpoint")
    config = point_payload.get("model_config", {})
    mismatch = {
        key: (config.get(key), expected)
        for key, expected in c208.EXPECTED_MODEL_CONFIG.items()
        if config.get(key) != expected}
    if mismatch:
        raise ValueError(f"frozen C208 model config changed: {mismatch}")
    point_model = c208.load_model(point_payload, device)
    common.set_model_stats(point_model, normalizer)
    point_model.requires_grad_(False)
    point_model.eval()
    gpu_started = time.perf_counter()
    deadline = min(
        started + WALL_BUDGET_SECONDS,
        gpu_started + GPU_BUDGET_SECONDS)
    tail_constructions = 0
    tail_evaluation_attempts = 0
    tail_evaluations_completed = 0

    try:
        train_key_digest = ordered_window_key_digest(train_windows, metadata)
        cache = build_probe_cache(
            point_model, train_dataset, normalizer, device,
            deadline=deadline)
        cache_report = validate_cache_tensors(cache, len(train_windows))
        cache_report.update({
            "ordered_train_window_key_digest": train_key_digest,
            "point_checkpoint_sha256": FORMAL_C208_SHA256,
            "split_manifest_sha256": FORMAL_SPLIT_MANIFEST_SHA256,
            "real_stats_sha256": FORMAL_REAL_STATS_SHA256,
            "head_input_dtype": "float16",
        })
        rows = []
        for fold_index, held in enumerate(AOA_VALUES):
            if time.perf_counter() > deadline:
                raise TimeoutError("Gate-C budget expired before LOAO fit")
            partition = partitions[held]
            candidate, control, losses = train_head_pair(
                cache, partition["fit"], device,
                seed=FORMAL_SEED + fold_index, deadline=deadline)
            row = evaluate_heads(
                point_model, candidate, control, train_dataset,
                partition["evaluate"], normalizer, device,
                with_promoted_bounds=False, deadline=deadline)
            row.update({
                "held_aoa": held,
                "fit_samples": len(partition["fit"]),
                "final_train_losses": losses,
            })
            rows.append(row)
            print(json.dumps({
                "event": "loao_fold",
                "held_aoa": held,
                "candidate_shadow": _fold_summary(row)["candidate_shadow"],
                "phase_increment": _fold_summary(row)["phase_increment"],
            }), flush=True)
            del candidate, control
            torch.cuda.empty_cache()

        loao = loao_gate(rows)
        loao_payload = {
            "experiment": EXPERIMENT,
            "contract": EXPERIMENT_CONTRACT,
            "provenance": identities,
            "counts": counts,
            "runtime_manifest_sha256": runtime_manifest_sha,
            "feature_cache": cache_report,
            "raw_real_frames_preloaded_before_gate": True,
            "tail_windows_indexed_before_gate": False,
            "tail_targets_used_before_gate": False,
            "rows": rows,
            "gate": loao,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (args.output_dir / "loao_result.json").write_text(
            json.dumps(loao_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n", encoding="utf-8")
        if not loao["pass"]:
            summary = {
                **loao_payload,
                "decision": "STOP_PHASE_PROBE_LOAO",
                **tail_access_report(
                    tail_constructions, tail_evaluation_attempts,
                    tail_evaluations_completed),
                "release_checkpoint_written": False,
            }
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n", encoding="utf-8")
            print(json.dumps({
                "event": "complete", "decision": summary["decision"]}),
                flush=True)
            return

        # Promotion is now fixed: one full-2576 fit, then one terminal-725 pass.
        candidate, control, full_losses = train_head_pair(
            cache, list(range(len(cache))), device,
            seed=FORMAL_SEED + len(AOA_VALUES), deadline=deadline)
        refit_sha = validate_refit_input(args.all_tail_refit)
        refit_payload = torch.load(
            args.all_tail_refit, map_location="cpu", weights_only=False)
        assert_refit_point_state_identity(point_payload, refit_payload)
        del point_model
        torch.cuda.empty_cache()
        terminal_model = c208.load_model(refit_payload, device)
        common.set_model_stats(terminal_model, normalizer)
        terminal_model.requires_grad_(False)
        terminal_model.eval()

        def tail_factory():
            nonlocal tail_constructions
            tail_constructions += 1
            if tail_constructions != 1:
                raise RuntimeError("725 tail was constructed more than once")
            tail_windows = tail_windows_from_manifest(runtime_manifest)
            if len(tail_windows) != FORMAL_COUNTS["tail_windows"]:
                raise RuntimeError("promoted tail construction is not 725 windows")
            return common.repro.WindowDataset(trajectories, tail_windows)

        def terminal_evaluator(tail_dataset):
            nonlocal tail_evaluation_attempts, tail_evaluations_completed
            tail_evaluation_attempts += 1
            if tail_evaluation_attempts != 1:
                raise RuntimeError("725 tail was evaluated more than once")
            result = evaluate_heads(
                terminal_model, candidate, control, tail_dataset,
                list(range(len(tail_dataset))), normalizer, device,
                with_promoted_bounds=True, deadline=deadline,
                capture_latency_input=True)
            tail_evaluations_completed += 1
            return result

        terminal = run_terminal_once(loao, tail_factory, terminal_evaluator)
        if terminal is None:
            raise RuntimeError("LOAO passed but terminal guard returned no result")
        latency_input = terminal.pop("_latency_input")
        latency = paired_latency_ratio(
            terminal_model, candidate, latency_input)
        wall_elapsed = time.perf_counter() - started
        gpu_elapsed = time.perf_counter() - gpu_started
        final_gate = terminal_gate(
            terminal, latency["ratio"],
            time_shadow=latency["time_shadow_local"],
            wall_elapsed=wall_elapsed, gpu_elapsed=gpu_elapsed)
        summary = {
            **loao_payload,
            "terminal": terminal,
            "terminal_gate": final_gate,
            "latency": latency,
            "full_fit_losses": full_losses,
            "all_tail_refit_sha256": refit_sha,
            **tail_access_report(
                tail_constructions, tail_evaluation_attempts,
                tail_evaluations_completed),
            "release_checkpoint_written": False,
            "decision": final_gate["decision"],
            "elapsed_seconds": wall_elapsed,
            "gpu_elapsed_seconds": gpu_elapsed,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n", encoding="utf-8")
        print(json.dumps({
            "event": "complete", "decision": summary["decision"]}),
            flush=True)
    except TimeoutError as exc:
        summary = {
            "experiment": EXPERIMENT,
            "contract": EXPERIMENT_CONTRACT,
            "provenance": identities,
            "counts": counts,
            "decision": "STOP_PHASE_PROBE_BUDGET",
            "error": str(exc),
            **tail_access_report(
                tail_constructions, tail_evaluation_attempts,
                tail_evaluations_completed),
            "release_checkpoint_written": False,
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_elapsed_seconds": time.perf_counter() - gpu_started,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n", encoding="utf-8")
        print(json.dumps({
            "event": "complete", "decision": summary["decision"]}),
            flush=True)
    except Exception as exc:
        summary = {
            "experiment": EXPERIMENT,
            "contract": EXPERIMENT_CONTRACT,
            "provenance": identities,
            "counts": counts,
            "decision": "STOP_PHASE_PROBE_ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            **tail_access_report(
                tail_constructions, tail_evaluation_attempts,
                tail_evaluations_completed),
            "release_checkpoint_written": False,
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_elapsed_seconds": time.perf_counter() - gpu_started,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
