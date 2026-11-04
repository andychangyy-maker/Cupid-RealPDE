#!/usr/bin/env python3
"""Paired held-Real gate for the parameter-free C208 SMU1 semantics.

Each fold rebuilds a control and candidate from the exact same Sim10k state.
Only other-four-AoA windows and their fold-specific statistics enter training.
The fixed725 temporal tail is never constructed by this program.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import calibrate_sps_bounds as sps_numpy  # noqa: E402
import train_c208_real_sim_rehearsal as c208  # noqa: E402
import train_ensemble_unet2d as ensemble_train  # noqa: E402
import train_full_backbone_point_distillation as full  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


EXPERIMENT = "c208_smu1_shared_member_uv_mean_paired_loao"
FORMAL_START_SHA256 = c208.FORMAL_START_SIM_SHA256
FORMAL_REAL_STATS_SHA256 = c208.FORMAL_REAL_STATS_SHA256
FORMAL_SPLIT_SHA256 = c208.FORMAL_REAL_SPLIT_SHA256
EXPECTED_PARAMETERS = c208.EXPECTED_PARAMETER_COUNT
AOA_VALUES = (0, 5, 10, 15, 20)
FOLD_ORDER = (10, 0, 20, 15, 5)
EXPECTED_AOA_COUNTS = {0: 534, 5: 433, 10: 576, 15: 544, 20: 489}
EXPECTED_FIT_COUNTS = {0: 2042, 5: 2143, 10: 2000, 15: 2032, 20: 2087}

UPDATES = 4000
SCHEDULER_T_MAX = 5000
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SEED = 0
WINDOW_STRIDE = 20
VAL_FRACTION = 0.2
LOG_INTERVAL = 500
WALL_BUDGET_SECONDS = 6.0 * 60.0 * 60.0
GPU_BUDGET_SECONDS = 5.0 * 60.0 * 60.0
IDENTITY_TOLERANCE = 1e-6

FINAL_COEFFICIENTS = {
    "rel_l2_score": 0.47201126,
    "tke_rel_l2_score": 0.08586562,
    "mvpe_rel_l2_score": 0.09410214,
    "native_sps_score": 0.24699273,
}
POINT_DELTA_MIN = 0.15
DIRECT_DELTA_MIN = 0.20
SPS_DELTA_MIN = 0.50
NON_TIME_DELTA_MIN = 0.52
COMPONENT_FLOOR = -0.02
MIN_POSITIVE_FOLDS = 4
WORST_FOLD_NON_TIME = -0.05


CONTRACT = {
    "hypothesis": (
        "Sharing only the four members' u/v temporal mean makes bootstrap "
        "supervision jointly learn the low-frequency field while preserving "
        "member-specific zero-time-mean fluctuation roles."),
    "evidence_class": (
        "paired Real-finetuning leave-one-AoA-out; not full-lifecycle true "
        "LOAO because the Sim P1 prior saw all Real conditions"),
    "single_variable": {
        "control": "shared_member_uv_mean=false",
        "candidate": "shared_member_uv_mean=true",
        "pressure": "bitwise unchanged",
        "aggregate_point": "raw member mean in both arms",
        "parameters": EXPECTED_PARAMETERS,
    },
    "training": {
        "updates": UPDATES,
        "scheduler_T_max": SCHEDULER_T_MAX,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "seed": SEED,
        "fold_order": FOLD_ORDER,
        "paired_stream": "same pre-generated sample and bootstrap manifest",
        "normalizer": "other-four optimizer windows only",
    },
    "gate": {
        "pooled_point_delta_min": POINT_DELTA_MIN,
        "pooled_direct_delta_min": DIRECT_DELTA_MIN,
        "pooled_sps_delta_min": SPS_DELTA_MIN,
        "pooled_non_time_delta_min": NON_TIME_DELTA_MIN,
        "component_floor": COMPONENT_FLOOR,
        "positive_point_and_non_time_folds_min": MIN_POSITIVE_FOLDS,
        "worst_fold_non_time_min": WORST_FOLD_NON_TIME,
    },
    "tail": "fixed725 is never constructed/read/scored by this program",
    "stop": (
        "STOP on contract/identity/stream/mask/leakage/nonfinite/OOM/budget "
        "failure, two non-positive AoA non-Time rows, or any final gate miss"),
}


def git_identity() -> dict[str, str]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", *arguments), cwd=REPO_ROOT, text=True).strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
    }


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)


class FoldNormalizer:
    def __init__(self, statistics: tuple[torch.Tensor, ...],
                 device: torch.device):
        if len(statistics) != 4:
            raise ValueError("fold statistics must contain four tensors")
        values = [value.to(device=device, dtype=torch.float32)
                  for value in statistics]
        if any(value.shape != (3,) for value in values):
            raise ValueError("fold statistics must be three-channel")
        self.mean_input, self.mean_target = values[:2]
        one = torch.ones((), device=device)
        self.std_input = torch.where(values[2] == 0.0, one, values[2])
        self.std_target = torch.where(values[3] == 0.0, one, values[3])

    def preprocess(self, x: torch.Tensor, y: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        return ((x - self.mean_input) / self.std_input,
                (y - self.mean_target) / self.std_target)

    def postprocess_prediction(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std_target + self.mean_target

    def postprocess_target(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std_target + self.mean_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument("--start-sim-sha256", default=FORMAL_START_SHA256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers != 0:
        raise ValueError("formal paired stream requires num-workers=0")
    if args.updates != UPDATES:
        raise ValueError("formal SMU1 requires exactly 4000 updates")
    args.start_sim_sha256 = c208.normalized_sha256(
        args.start_sim_sha256, "start-sim-sha256")
    if args.start_sim_sha256 != FORMAL_START_SHA256:
        raise ValueError("start is not the frozen Sim10k checkpoint")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output {args.output_dir}")


def make_train_side_split(metadata: Sequence[dict]) -> tuple[list, list[dict], int]:
    """Reproduce the frozen split without constructing tail Window objects."""
    train = []
    manifest = []
    tail_count = 0
    for trajectory, item in enumerate(metadata):
        starts = list(range(
            0, int(item["frames"]) - 40 + 1, WINDOW_STRIDE))
        if len(starts) < 2:
            raise ValueError(f"trajectory {item['file']} is too short")
        n_tail = max(1, math.ceil(len(starts) * VAL_FRACTION))
        tail_starts = starts[-n_tail:]
        boundary = tail_starts[0]
        train_starts = [start for start in starts if start + 40 <= boundary]
        if not train_starts:
            raise ValueError(f"trajectory {item['file']} has no train window")
        train.extend(
            ensemble_train.common.repro.Window(trajectory, start)
            for start in train_starts)
        manifest.append({
            **item,
            "train_starts": train_starts,
            "val_starts": tail_starts,
            "val_boundary": boundary,
        })
        tail_count += len(tail_starts)
    return train, manifest, tail_count


def window_aoas(windows: Sequence, metadata: Sequence[dict]) -> np.ndarray:
    return np.asarray([
        int(metadata[window.trajectory]["aoa"]) for window in windows],
        dtype=np.int64)


def loao_partitions(aoas: Sequence[int]) -> dict[int, dict[str, list[int]]]:
    labels = np.asarray(aoas, dtype=np.int64)
    if set(labels.tolist()) != set(AOA_VALUES):
        raise ValueError("fixed2576 does not contain the five expected AoAs")
    all_indices = np.arange(len(labels))
    result = {}
    evaluated = []
    for held in AOA_VALUES:
        fit = all_indices[labels != held].tolist()
        evaluate = all_indices[labels == held].tolist()
        if len(fit) != EXPECTED_FIT_COUNTS[held]:
            raise RuntimeError(f"held {held} fit count changed")
        if len(evaluate) != EXPECTED_AOA_COUNTS[held]:
            raise RuntimeError(f"held {held} evaluate count changed")
        if set(fit).intersection(evaluate):
            raise RuntimeError("held window leaked into optimizer indices")
        result[held] = {"fit": fit, "evaluate": evaluate}
        evaluated.extend(evaluate)
    if sorted(evaluated) != list(range(len(labels))):
        raise RuntimeError("held partitions do not cover fixed2576 exactly once")
    return result


def window_sufficient_statistics(trajectories: Sequence[torch.Tensor],
                                 windows: Sequence) -> dict[str, torch.Tensor]:
    """Compute per-window channel sums from per-trajectory frame prefixes."""
    rows = {
        "input_sum": torch.zeros(len(windows), 3, dtype=torch.float64),
        "input_sum_sq": torch.zeros(len(windows), 3, dtype=torch.float64),
        "target_sum": torch.zeros(len(windows), 3, dtype=torch.float64),
        "target_sum_sq": torch.zeros(len(windows), 3, dtype=torch.float64),
    }
    by_trajectory: dict[int, list[tuple[int, object]]] = {}
    for index, window in enumerate(windows):
        by_trajectory.setdefault(int(window.trajectory), []).append(
            (index, window))
    spatial_count = None
    for trajectory_index, indexed_windows in by_trajectory.items():
        value = trajectories[trajectory_index]
        if value.ndim != 4 or value.shape[-1] != 2:
            raise ValueError("Real trajectories must contain u/v only")
        current_spatial_count = int(value.shape[1] * value.shape[2])
        if spatial_count is None:
            spatial_count = current_spatial_count
        elif spatial_count != current_spatial_count:
            raise ValueError("Real trajectory spatial shapes differ")
        frame_sum_uv = value.to(torch.float64).sum(dim=(1, 2))
        frame_sq_uv = value.to(torch.float64).square().sum(dim=(1, 2))
        zero = torch.zeros(1, 2, dtype=torch.float64)
        prefix = torch.cat((zero, frame_sum_uv.cumsum(dim=0)), dim=0)
        prefix_sq = torch.cat((zero, frame_sq_uv.cumsum(dim=0)), dim=0)
        for index, window in indexed_windows:
            start = int(window.start)
            input_sum = prefix[start + 20] - prefix[start]
            target_sum = prefix[start + 40] - prefix[start + 20]
            input_sq = prefix_sq[start + 20] - prefix_sq[start]
            target_sq = prefix_sq[start + 40] - prefix_sq[start + 20]
            rows["input_sum"][index, :2] = input_sum
            rows["input_sum_sq"][index, :2] = input_sq
            rows["target_sum"][index, :2] = target_sum
            rows["target_sum_sq"][index, :2] = target_sq
    if spatial_count is None:
        raise ValueError("cannot compute statistics for an empty dataset")
    rows["elements_per_partition"] = torch.tensor(
        20 * spatial_count, dtype=torch.int64)
    return rows


def fold_statistics(rows: dict[str, torch.Tensor], indices: Sequence[int]
                    ) -> tuple[torch.Tensor, ...]:
    if not indices:
        raise ValueError("fold normalizer has no optimizer windows")
    selected = torch.as_tensor(indices, dtype=torch.long)
    count = int(rows["elements_per_partition"]) * len(indices)
    means = []
    stds = []
    for prefix in ("input", "target"):
        total = rows[f"{prefix}_sum"][selected].sum(dim=0)
        total_sq = rows[f"{prefix}_sum_sq"][selected].sum(dim=0)
        mean = total / count
        variance = total_sq / count - mean.square()
        means.append(mean.float())
        stds.append(variance.clamp_min(0.0).sqrt().float())
    return means[0], means[1], stds[0], stds[1]


def statistics_digest(statistics: tuple[torch.Tensor, ...]) -> str:
    return c208.canonical_sha256([
        value.detach().cpu().tolist() for value in statistics])


def paired_batch_schedule(
        fit_indices: Sequence[int], windows: Sequence, updates: int = UPDATES,
        seed: int = SEED) -> dict:
    generator = torch.Generator().manual_seed(seed)
    batches: list[list[int]] = []
    while len(batches) < updates:
        order = torch.randperm(len(fit_indices), generator=generator).tolist()
        epoch = [int(fit_indices[index]) for index in order]
        batches.extend([
            epoch[start:start + BATCH_SIZE]
            for start in range(0, len(epoch), BATCH_SIZE)])
    batches = batches[:updates]
    rows = []
    masks = []
    for indices in batches:
        trajectories = [int(windows[index].trajectory) for index in indices]
        mask = ensemble_train.trajectory_member_mask(
            torch.as_tensor(trajectories), 0.75, seed, 4)
        rows.append({
            "sample_indices": indices,
            "trajectory_ids": trajectories,
            "bootstrap_mask": mask.to(torch.uint8).tolist(),
        })
        masks.append(mask)
    return {
        "batches": batches,
        "rows": rows,
        "masks": masks,
        "sha256": c208.canonical_sha256(rows),
    }


def ordered_held_key_digest(indices: Sequence[int], windows: Sequence,
                            metadata: Sequence[dict]) -> str:
    return c208.canonical_sha256([
        {
            "file": metadata[windows[index].trajectory]["file"],
            "start": int(windows[index].start),
        }
        for index in indices
    ])


def load_arm_model(payload: dict, candidate: bool,
                   normalizer: FoldNormalizer,
                   device: torch.device) -> EnsembleUNet2d:
    stored = dict(payload["model_config"])
    if stored.pop("architecture") != "ensemble_unet":
        raise ValueError("Sim10k architecture is not ensemble_unet")
    stored["shared_member_uv_mean"] = bool(candidate)
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**stored))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("SMU1 changed the C208 parameter count")
    ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def _single_tagged_batch(dataset, index: int):
    history, target = dataset[index]
    trajectory = int(dataset.windows[index].trajectory)
    return (
        history.unsqueeze(0), target.unsqueeze(0),
        torch.tensor([trajectory]), torch.tensor([index]))


@torch.inference_mode()
def capture_update0(model: EnsembleUNet2d, dataset, index: int,
                    normalizer: FoldNormalizer,
                    device: torch.device) -> dict[str, torch.Tensor | dict]:
    history, target, _trajectory, _sample = _single_tagged_batch(dataset, index)
    history = history.to(device)
    target = target.to(device)
    history_norm, _target_norm = normalizer.preprocess(history, target)
    prediction, members, _quantiles, lower, upper, _features, _logits = (
        model.forward_uq_training(history_norm))
    prediction_physical = normalizer.postprocess_prediction(prediction.float())
    target_physical = target.float()
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    state = new_metric_state()
    update_metric_state(
        state, prediction_physical, target_physical,
        lower.float() * scale, upper.float() * scale)
    return {
        "prediction": prediction.detach().cpu(),
        "members": members.detach().cpu(),
        "lower": lower.detach().cpu(),
        "upper": upper.detach().cpu(),
        "metric_state": state,
    }


def update0_identity(control: dict, candidate: dict) -> dict:
    control_members = control["members"]
    candidate_members = candidate["members"]
    control_fluctuation = (
        control_members[..., :2]
        - control_members[..., :2].mean(dim=2, keepdim=True))
    candidate_fluctuation = (
        candidate_members[..., :2]
        - candidate_members[..., :2].mean(dim=2, keepdim=True))
    candidate_means = candidate_members[..., :2].mean(dim=2)
    values = {
        "point_max_abs_delta": float(
            (candidate["prediction"] - control["prediction"]).abs().max()),
        "centered_fluctuation_max_abs_delta": float(
            (candidate_fluctuation - control_fluctuation).abs().max()),
        "candidate_member_mean_max_abs_spread": float(
            (candidate_means - candidate_means[:, :1]).abs().max()),
        "pressure_bitwise": bool(torch.equal(
            candidate_members[..., 2:], control_members[..., 2:])),
        "lower_max_abs_delta": float(
            (candidate["lower"] - control["lower"]).abs().max()),
        "upper_max_abs_delta": float(
            (candidate["upper"] - control["upper"]).abs().max()),
    }
    control_metric = finalize_metric_state(control["metric_state"])
    candidate_metric = finalize_metric_state(candidate["metric_state"])
    values["update0_metric_delta"] = metric_delta(candidate_metric, control_metric)
    values["pass"] = bool(
        values["point_max_abs_delta"] <= IDENTITY_TOLERANCE
        and values["centered_fluctuation_max_abs_delta"] <= IDENTITY_TOLERANCE
        and values["candidate_member_mean_max_abs_spread"] <= IDENTITY_TOLERANCE
        and values["pressure_bitwise"])
    return values


def train_arm(model: EnsembleUNet2d, dataset, schedule: dict,
              normalizer: FoldNormalizer, device: torch.device,
              deadline: float, label: str, *,
              bootstrap_mask_by_trajectory: Mapping[
                  int, Sequence[bool]] | None = None) -> dict:
    tagged = ensemble_train.TrajectoryTaggedDataset(dataset)
    loader = DataLoader(
        tagged, batch_sampler=schedule["batches"], num_workers=0,
        pin_memory=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX)
    actual_rows = []
    history = []
    sums: dict[str, float] = {}
    count = 0
    started = time.perf_counter()
    for iteration, batch in enumerate(loader, start=1):
        if iteration > UPDATES:
            break
        if time.perf_counter() > deadline:
            raise TimeoutError("SMU1 wall budget expired during paired training")
        expected = schedule["rows"][iteration - 1]
        trajectory_ids = batch[2].to(dtype=torch.long, device="cpu")
        sample_indices = batch[3].to(dtype=torch.long, device="cpu")
        if bootstrap_mask_by_trajectory is None:
            mask = ensemble_train.trajectory_member_mask(
                trajectory_ids, 0.75, SEED, 4)
        else:
            try:
                mask = torch.as_tensor([
                    bootstrap_mask_by_trajectory[int(trajectory)]
                    for trajectory in trajectory_ids.tolist()
                ], dtype=torch.bool)
            except KeyError as exc:
                raise RuntimeError(
                    f"{label} trajectory {exc.args[0]} has no frozen mask") \
                    from exc
            if tuple(mask.shape) != (len(trajectory_ids), 4):
                raise RuntimeError(
                    f"{label} frozen bootstrap mask is not batch-by-four")
            if not bool(mask.any(dim=1).all()):
                raise RuntimeError(
                    f"{label} frozen bootstrap mask has an empty row")
        actual = {
            "sample_indices": sample_indices.tolist(),
            "trajectory_ids": trajectory_ids.tolist(),
            "bootstrap_mask": mask.to(torch.uint8).tolist(),
        }
        if actual != expected:
            raise RuntimeError(
                f"paired stream mismatch for {label} at update {iteration}")
        actual_rows.append(actual)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_options = {"exact_physical_uq_target": True}
        if bootstrap_mask_by_trajectory is not None:
            loss_options["member_mask"] = mask
        values = c208.backward_real_supervised(
            model, batch, normalizer, device, iteration, **loss_options)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        values["learning_rate"] = optimizer.param_groups[0]["lr"]
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if iteration % LOG_INTERVAL == 0:
            row = {
                "iteration": iteration,
                **{f"train_{key}_mean": value / count
                   for key, value in sums.items()},
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps({
                "event": "smu1_progress", "arm": label, **row,
            }), flush=True)
            sums.clear()
            count = 0
    if len(actual_rows) != UPDATES:
        raise RuntimeError(f"{label} completed {len(actual_rows)} updates")
    actual_sha = c208.canonical_sha256(actual_rows)
    if actual_sha != schedule["sha256"]:
        raise RuntimeError(f"{label} paired stream digest mismatch")
    return {
        "completed_updates": len(actual_rows),
        "planned_stream_sha256": schedule["sha256"],
        "actual_stream_sha256": actual_sha,
        "history": history,
        "wall_seconds": time.perf_counter() - started,
    }


def new_metric_state() -> dict[str, float | int]:
    return {
        "samples": 0,
        "rel_error_sum": 0.0,
        "tke_error_sum": 0.0,
        "mvpe_error_sum": 0.0,
        "sps_numerator": 0.0,
        "sps_scored": 0,
        "sps_covered": 0,
        "sps_width_sum": 0.0,
    }


def update_metric_state(state: dict, prediction: torch.Tensor,
                        target: torch.Tensor, lower: torch.Tensor,
                        upper: torch.Tensor) -> None:
    if (prediction.ndim != 5 or prediction.shape != target.shape
            or prediction.shape[-1] < 2):
        raise ValueError("prediction/target must be aligned 5D fields")
    expected_extent_shape = (*prediction.shape[:-1], 2)
    if (lower.shape != expected_extent_shape
            or upper.shape != expected_extent_shape):
        raise ValueError("interval extents are not aligned with u/v prediction")
    if not all(bool(torch.isfinite(value).all()) for value in (
            prediction, target, lower, upper)):
        raise FloatingPointError("held prediction/target/bounds are non-finite")
    if bool(((lower + upper) < 0.0).any()):
        raise ValueError("held interval has a negative ordered width")
    pred = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
    truth = target.detach().cpu().numpy().astype(np.float32, copy=False)
    lower_np = lower.detach().cpu().numpy().astype(np.float32, copy=False)
    upper_np = upper.detach().cpu().numpy().astype(np.float32, copy=False)
    channels = full.OFFICIAL.measured_channels(truth)
    rel = full.OFFICIAL.rel_l2_per_sample(pred, truth, channels)
    tke = full.OFFICIAL.tke_rel_l2_per_sample(pred, truth, channels)
    mvpe = full.OFFICIAL.mvpe_rel_l2_per_sample(pred, truth)
    factor = sps_numpy.sample_sps_factor(full.OFFICIAL, pred, truth, channels)
    residual = truth[..., :channels] - pred[..., :channels]
    scored = truth[..., :channels] != 0.0
    inside = (residual >= -lower_np[..., :channels]) & (
        residual <= upper_np[..., :channels])
    penalty = np.exp(
        -(lower_np[..., :channels] + upper_np[..., :channels])
        / full.OFFICIAL.SIGMA_GLOBAL)
    shaped_factor = factor.reshape((-1,) + (1,) * (residual.ndim - 1))
    value = shaped_factor * inside * penalty
    if not all(np.all(np.isfinite(array)) for array in (
            rel, tke, mvpe, factor, penalty, value)):
        raise FloatingPointError("held metric sufficient state is non-finite")
    state["samples"] += int(len(truth))
    state["rel_error_sum"] += float(np.sum(rel, dtype=np.float64))
    state["tke_error_sum"] += float(np.sum(tke, dtype=np.float64))
    state["mvpe_error_sum"] += float(np.sum(mvpe, dtype=np.float64))
    state["sps_numerator"] += float(np.sum(value[scored], dtype=np.float64))
    state["sps_scored"] += int(np.count_nonzero(scored))
    state["sps_covered"] += int(np.count_nonzero(inside & scored))
    state["sps_width_sum"] += float(np.sum(
        (lower_np[..., :channels] + upper_np[..., :channels])[scored],
        dtype=np.float64))
    if any(not math.isfinite(float(value)) for value in state.values()):
        raise FloatingPointError("accumulated held metric state is non-finite")


def merge_metric_states(states: Sequence[dict]) -> dict:
    result = new_metric_state()
    for state in states:
        for key in result:
            result[key] += state[key]
    return result


def finalize_metric_state(state: dict) -> dict[str, float]:
    samples = int(state["samples"])
    scored = int(state["sps_scored"])
    if samples <= 0 or scored <= 0:
        raise ValueError("metric state is empty")
    if any(not math.isfinite(float(value)) for value in state.values()):
        raise FloatingPointError("metric state is non-finite")
    result = {
        "rel_l2_score": full.OFFICIAL.score_error(
            float(state["rel_error_sum"]) / samples),
        "tke_rel_l2_score": full.OFFICIAL.score_error(
            float(state["tke_error_sum"]) / samples),
        "mvpe_rel_l2_score": full.OFFICIAL.score_error(
            float(state["mvpe_error_sum"]) / samples),
        "native_sps_score": full.OFFICIAL.score_sps(
            float(state["sps_numerator"]) / scored),
        "native_sps_coverage": float(state["sps_covered"]) / scored,
        "native_sps_mean_width": float(state["sps_width_sum"]) / scored,
    }
    result["point_score"] = (
        0.5 * result["rel_l2_score"]
        + 0.3 * result["tke_rel_l2_score"]
        + 0.2 * result["mvpe_rel_l2_score"])
    return {key: float(value) for key, value in result.items()}


@torch.inference_mode()
def evaluate_model(model: EnsembleUNet2d, dataset, indices: Sequence[int],
                   normalizer: FoldNormalizer, device: torch.device,
                   deadline: float) -> tuple[dict, dict, dict]:
    loader = DataLoader(
        dataset, batch_sampler=[
            list(indices[start:start + EVAL_BATCH_SIZE])
            for start in range(0, len(indices), EVAL_BATCH_SIZE)],
        num_workers=0, pin_memory=True)
    state = new_metric_state()
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    audited_samples = 0
    audited_scored = 0
    model.eval()
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    for history, target in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("SMU1 wall budget expired during held scoring")
        target_cpu = target.contiguous()
        scored_cpu = target_cpu[..., :2].ne(0.0).to(torch.uint8).contiguous()
        target_hasher.update(str(tuple(target_cpu.shape)).encode("ascii"))
        target_hasher.update(target_cpu.numpy().tobytes(order="C"))
        mask_hasher.update(str(tuple(scored_cpu.shape)).encode("ascii"))
        mask_hasher.update(scored_cpu.numpy().tobytes(order="C"))
        audited_samples += len(target_cpu)
        audited_scored += int(scored_cpu.sum())
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        history_norm, _target_norm = normalizer.preprocess(history, target)
        prediction_norm, lower_norm, upper_norm = model.forward_distribution(
            history_norm)
        prediction = normalizer.postprocess_prediction(prediction_norm.float())
        truth = target.float()
        update_metric_state(
            state, prediction, truth,
            lower_norm.float() * scale, upper_norm.float() * scale)
    if int(state["samples"]) != len(indices):
        raise RuntimeError("held scorer did not consume the complete partition")
    if (audited_samples != len(indices)
            or audited_scored != int(state["sps_scored"])):
        raise RuntimeError("held raw-target audit does not match metric state")
    return finalize_metric_state(state), state, {
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
        "samples": audited_samples,
        "scored_elements": audited_scored,
    }


def metric_delta(candidate: dict[str, float], control: dict[str, float]
                 ) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(control[key])
        for key in ("rel_l2_score", "tke_rel_l2_score",
                    "mvpe_rel_l2_score", "point_score", "native_sps_score")}


def direct_shadow(delta: dict[str, float]) -> float:
    return float(sum(
        FINAL_COEFFICIENTS[key] * delta[key]
        for key in ("rel_l2_score", "tke_rel_l2_score",
                    "mvpe_rel_l2_score")))


def non_time_shadow(delta: dict[str, float]) -> float:
    return direct_shadow(delta) + (
        FINAL_COEFFICIENTS["native_sps_score"] * delta["native_sps_score"])


def comparison(candidate: dict[str, float], control: dict[str, float]) -> dict:
    delta = metric_delta(candidate, control)
    return {
        "control": control,
        "candidate": candidate,
        "delta": delta,
        "direct_delta": direct_shadow(delta),
        "non_time_delta": non_time_shadow(delta),
    }


def loao_gate(pooled: dict, rows: Sequence[dict]) -> dict:
    if len(rows) != len(AOA_VALUES):
        raise ValueError("formal gate requires exactly five held-AoA rows")
    delta = pooled["delta"]
    conditions = {
        "pooled_point_delta": delta["point_score"] >= POINT_DELTA_MIN,
        "pooled_direct_delta": pooled["direct_delta"] >= DIRECT_DELTA_MIN,
        "pooled_sps_delta": delta["native_sps_score"] >= SPS_DELTA_MIN,
        "pooled_non_time_delta": pooled["non_time_delta"] >= NON_TIME_DELTA_MIN,
        "rel_floor": delta["rel_l2_score"] >= COMPONENT_FLOOR,
        "tke_floor": delta["tke_rel_l2_score"] >= COMPONENT_FLOOR,
        "mvpe_floor": delta["mvpe_rel_l2_score"] >= COMPONENT_FLOOR,
        "positive_point_folds": sum(
            row["comparison"]["delta"]["point_score"] > 0.0 for row in rows)
            >= MIN_POSITIVE_FOLDS,
        "positive_non_time_folds": sum(
            row["comparison"]["non_time_delta"] > 0.0 for row in rows)
            >= MIN_POSITIVE_FOLDS,
        "worst_fold_non_time": min(
            row["comparison"]["non_time_delta"] for row in rows)
            >= WORST_FOLD_NON_TIME,
        "paired_streams": all(
            row["control_training"]["actual_stream_sha256"]
            == row["candidate_training"]["actual_stream_sha256"]
            == row["stream_sha256"] for row in rows),
        "samples_exact": sum(int(row["samples"]) for row in rows) == 2576,
        "held_aoas_and_counts_exact": (
            {int(row["held_aoa"]) for row in rows} == set(AOA_VALUES)
            and all(int(row["samples"]) == EXPECTED_AOA_COUNTS[
                int(row["held_aoa"])] for row in rows)),
        "held_targets_and_masks_paired": all(
            row["control_held_audit"] == row["candidate_held_audit"]
            and row["control_held_audit"]["samples"] == row["samples"]
            and bool(row["held_ordered_key_sha256"])
            for row in rows),
    }
    passed = all(conditions.values())
    return {
        "pass": passed,
        "decision": "GO_SMU1_FULL2576" if passed else "STOP_SMU1_LOAO",
        "conditions": conditions,
    }


def run(args: argparse.Namespace) -> dict:
    identity = git_identity()
    if identity["status_porcelain"]:
        raise RuntimeError("formal SMU1 requires a clean Git worktree")
    started = time.perf_counter()
    deadline = started + WALL_BUDGET_SECONDS
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal SMU1 paired LOAO requires CUDA")
    if c208.sha256(args.real_stats) != FORMAL_REAL_STATS_SHA256:
        raise RuntimeError("official Real stats SHA changed")
    payload = c208.load_start_payload(args)
    trajectories, metadata = ensemble_train.common.repro.load_trajectories(
        args.real_dir, 0)
    train_windows, manifest, tail_count = make_train_side_split(metadata)
    manifest_sha = c208.canonical_sha256(manifest)
    if manifest_sha != FORMAL_SPLIT_SHA256:
        raise RuntimeError("fixed Real split manifest SHA changed")
    if len(train_windows) != 2576 or tail_count != 725 or len(trajectories) != 82:
        raise RuntimeError("fixed Real split counts changed")
    aoas = window_aoas(train_windows, metadata)
    if {value: int(np.sum(aoas == value)) for value in AOA_VALUES} != (
            EXPECTED_AOA_COUNTS):
        raise RuntimeError("fixed2576 AoA counts changed")
    partitions = loao_partitions(aoas)
    dataset = ensemble_train.common.repro.WindowDataset(
        trajectories, train_windows)
    statistics_rows = window_sufficient_statistics(
        trajectories, train_windows)
    (args.output_dir / "train_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    provenance = {
        "start_sim": str(args.start_sim.resolve()),
        "start_sim_sha256": args.start_sim_sha256,
        "real_stats_sha256": FORMAL_REAL_STATS_SHA256,
        "split_manifest_sha256": manifest_sha,
        "git": identity,
    }
    counts = {
        "train_windows": len(train_windows),
        "tail_windows_not_constructed": tail_count,
        "trajectories": len(trajectories),
    }
    base_report = {
        "experiment": EXPERIMENT,
        "contract": CONTRACT,
        "provenance": provenance,
        "counts": counts,
        "normalizer_contract": {
            "kind": "other-four optimizer-window statistics per fold",
            "held_target_in_normalizer": False,
        },
        "tail_accessed": False,
        "release_checkpoint_written": False,
    }
    gpu_seconds = 0.0
    rows = []
    update0 = None

    for held in FOLD_ORDER:
        partition = partitions[held]
        stats = fold_statistics(statistics_rows, partition["fit"])
        normalizer = FoldNormalizer(stats, device)
        schedule = paired_batch_schedule(partition["fit"], train_windows)
        fold_started = time.perf_counter()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()

        ensemble_train.seed_everything(SEED)
        control_model = load_arm_model(payload, False, normalizer, device)
        control_initial = (
            capture_update0(
                control_model, dataset, partition["fit"][0], normalizer, device)
            if update0 is None else None)
        control_training = train_arm(
            control_model, dataset, schedule, normalizer, device,
            deadline, f"held{held}_control")
        control_metrics, control_state, control_held_audit = evaluate_model(
            control_model, dataset, partition["evaluate"], normalizer,
            device, deadline)
        del control_model
        torch.cuda.empty_cache()

        ensemble_train.seed_everything(SEED)
        candidate_model = load_arm_model(payload, True, normalizer, device)
        if update0 is None:
            candidate_initial = capture_update0(
                candidate_model, dataset, partition["fit"][0],
                normalizer, device)
            update0 = update0_identity(control_initial, candidate_initial)
            if not update0["pass"]:
                raise RuntimeError("SMU1 update-0 identity contract failed")
        candidate_training = train_arm(
            candidate_model, dataset, schedule, normalizer, device,
            deadline, f"held{held}_candidate")
        candidate_metrics, candidate_state, candidate_held_audit = evaluate_model(
            candidate_model, dataset, partition["evaluate"], normalizer,
            device, deadline)
        gpu_end.record()
        gpu_end.synchronize()
        fold_gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000.0
        del candidate_model
        torch.cuda.empty_cache()

        row = {
            "held_aoa": held,
            "fit_samples": len(partition["fit"]),
            "samples": len(partition["evaluate"]),
            "fold_statistics": [value.tolist() for value in stats],
            "fold_statistics_sha256": statistics_digest(stats),
            "stream_sha256": schedule["sha256"],
            "held_ordered_key_sha256": ordered_held_key_digest(
                partition["evaluate"], train_windows, metadata),
            "control_training": control_training,
            "candidate_training": candidate_training,
            "control_held_audit": control_held_audit,
            "candidate_held_audit": candidate_held_audit,
            "control_state": control_state,
            "candidate_state": candidate_state,
            "comparison": comparison(candidate_metrics, control_metrics),
            "wall_seconds": time.perf_counter() - fold_started,
            "cuda_elapsed_seconds": fold_gpu_seconds,
        }
        rows.append(row)
        gpu_seconds += fold_gpu_seconds
        write_json_atomic(args.output_dir / f"fold_{held:02d}.json", row)
        write_json_atomic(args.output_dir / "progress.json", {
            **base_report,
            "decision": "SMU1_LOAO_IN_PROGRESS",
            "completed_folds": len(rows),
            "rows": rows,
            "update0": update0,
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_seconds": gpu_seconds,
        })
        print(json.dumps({
            "event": "smu1_fold_complete",
            "held_aoa": held,
            "point_delta": row["comparison"]["delta"]["point_score"],
            "sps_delta": row["comparison"]["delta"]["native_sps_score"],
            "non_time_delta": row["comparison"]["non_time_delta"],
        }), flush=True)
        if gpu_seconds > GPU_BUDGET_SECONDS:
            raise TimeoutError("SMU1 GPU budget exceeded")
        if time.perf_counter() > deadline:
            raise TimeoutError("SMU1 wall budget exceeded")
        if sum(row["comparison"]["non_time_delta"] <= 0.0
               for row in rows) >= 2:
            return {
                **base_report,
                "decision": "STOP_SMU1_TWO_NONPOSITIVE_AOA",
                "rows": rows,
                "update0": update0,
                "tail_accessed": False,
                "release_checkpoint_written": False,
                "elapsed_seconds": time.perf_counter() - started,
                "gpu_seconds": gpu_seconds,
            }

    control_pooled_state = merge_metric_states([
        row["control_state"] for row in rows])
    candidate_pooled_state = merge_metric_states([
        row["candidate_state"] for row in rows])
    if int(control_pooled_state["samples"]) != 2576 or int(
            candidate_pooled_state["samples"]) != 2576:
        raise RuntimeError("pooled held states do not cover fixed2576")
    pooled = comparison(
        finalize_metric_state(candidate_pooled_state),
        finalize_metric_state(control_pooled_state))
    gate = loao_gate(pooled, rows)
    return {
        **base_report,
        "update0": update0,
        "rows": rows,
        "pooled_control_state": control_pooled_state,
        "pooled_candidate_state": candidate_pooled_state,
        "pooled": pooled,
        "gate": gate,
        "decision": gate["decision"],
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_seconds": gpu_seconds,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True)
    if args.contract_only:
        summary = {
            "experiment": EXPERIMENT,
            "contract": CONTRACT,
            "decision": "CONTRACT_ONLY_NO_DATA_NO_GPU",
            "tail_accessed": False,
        }
    else:
        ensemble_train.seed_everything(SEED)
        torch.set_float32_matmul_precision("high")
        try:
            summary = run(args)
        except Exception as exc:
            summary = {
                "experiment": EXPERIMENT,
                "contract": CONTRACT,
                "attempted_provenance": {
                    "start_sim": str(args.start_sim.resolve()),
                    "start_sim_sha256": args.start_sim_sha256,
                    "git": git_identity(),
                },
                "decision": "STOP_SMU1_CONTRACT_OR_RUNTIME",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "tail_accessed": False,
                "release_checkpoint_written": False,
            }
            write_json_atomic(args.output_dir / "summary.json", summary)
            raise
    write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "event": "smu1_complete", "decision": summary["decision"]}),
        flush=True)


if __name__ == "__main__":
    main()
