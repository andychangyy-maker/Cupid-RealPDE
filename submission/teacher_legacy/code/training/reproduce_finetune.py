#!/usr/bin/env python3
"""Reconstruct RealPDE competition real-data fine-tuning.

This script deliberately starts from the *released final* ``sim_pretrain``
checkpoint, as opposed to the unreleased best-validation snapshot that the
organizer fine-tunes appear to use.  It creates a deterministic, auditable
temporal split from the released real trajectories and compares three models on
exactly the same validation windows:

1. the released simulated-pretraining checkpoint;
2. the locally reconstructed fine-tune;
3. the released organizer fine-tune checkpoint.

The output directory contains JSONL learning curves, a summary JSON, split
manifests, and local best/final checkpoints.  No competition data or private
paths are embedded in those files.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Window:
    trajectory: int
    start: int


class WindowDataset(Dataset):
    def __init__(self, trajectories: list[torch.Tensor], windows: list[Window]):
        self.trajectories = trajectories
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.windows[index]
        uv = self.trajectories[item.trajectory][item.start : item.start + 40]
        # Released baselines always use [u, v, p]; p is unmeasured in real PIV.
        p = torch.zeros((*uv.shape[:-1], 1), dtype=uv.dtype)
        uvp = torch.cat((uv, p), dim=-1)
        return uvp[:20], uvp[20:]


class GroupedWindowDataset(WindowDataset):
    """Return marginal Reynolds/AoA groups for robust point-model training."""

    def __init__(self, trajectories: list[torch.Tensor], windows: list[Window],
                 metadata: list[dict]):
        super().__init__(trajectories, windows)
        self.metadata = metadata

    def __getitem__(self, index: int):
        x, y = super().__getitem__(index)
        item = self.metadata[self.windows[index].trajectory]
        re_group = 0 if item["re"] <= 9000 else (1 if item["re"] < 17000 else 2)
        aoa_group = int(round(float(item["aoa"]) / 5.0))
        return x, y, re_group, aoa_group


class Normalizer:
    def __init__(self, path: Path, device: torch.device):
        mi, mt, si, st = torch.load(path, map_location=device, weights_only=False)
        one = torch.ones((), device=device)
        self.mean_input = mi.float()
        self.mean_target = mt.float()
        self.std_input = torch.where(si == 0, one, si).float()
        self.std_target = torch.where(st == 0, one, st).float()

    def preprocess(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return ((x - self.mean_input) / self.std_input,
                (y - self.mean_target) / self.std_target)

    def postprocess_prediction(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std_target + self.mean_target

    def postprocess_target(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std_target + self.mean_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cno", "fno", "transolver"), default="cno")
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--stats", type=Path,
                        default=Path("kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt"))
    parser.add_argument("--kit-dir", type=Path,
                        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--start-label", default="released_sim_pretrain",
        help="Stable underscore-separated label used in summary metric keys.")
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-mode", choices=("full", "weights"), default="full",
                        help="Save resumable optimizer state or compact model weights only.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-mode", choices=("mse", "official_accuracy"), default="mse",
                        help="Train with normalized MSE or a differentiable proxy for the "
                             "three official accuracy subscores.")
    parser.add_argument("--mse-weight", type=float, default=0.25,
                        help="Extra normalized-MSE weight in official_accuracy mode.")
    parser.add_argument(
        "--spectrum-weight", type=float, default=0.0,
        help="Weight of a temporal FFT-magnitude relative-error loss.")
    parser.add_argument(
        "--group-robust-weight", type=float, default=0.0,
        help=("Mix weight for a smooth worst-group accuracy objective over "
              "marginal Reynolds/AoA groups; 0 preserves legacy training."))
    parser.add_argument(
        "--group-temperature", type=float, default=0.05,
        help="Log-sum-exp temperature for --group-robust-weight.")
    parser.add_argument(
        "--accuracy-weights", type=float, nargs=3, metavar=("REL", "TKE", "MVPE"),
        default=(1.0, 1.0, 1.0),
        help=("Relative weights of the Rel-L2, TKE, and MVPE score losses in "
              "official_accuracy mode. Weights are normalized to sum to one."))
    parser.add_argument("--selection-metric",
                        choices=("rmse", "accuracy_score_mean",
                                 "weighted_accuracy_score", "tke_rel_l2_score"),
                        default="rmse",
                        help=("Validation metric used to save best.pth. "
                              "weighted_accuracy_score uses --accuracy-weights."))
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0,
                        help="Debug only: use the first N trajectories; 0 uses all files.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_baseline_module(kit_dir: Path):
    path = (kit_dir / "load_baseline.py").resolve()
    spec = importlib.util.spec_from_file_location("competition_load_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_trajectories(real_dir: Path, max_files: int) -> tuple[list[torch.Tensor], list[dict]]:
    files = sorted(real_dir.glob("*.h5"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No .h5 trajectories found in {real_dir}")

    trajectories: list[torch.Tensor] = []
    metadata: list[dict] = []
    started = time.time()
    for index, path in enumerate(files, start=1):
        with h5py.File(path, "r") as handle:
            u = np.asarray(handle["u"][:, ::2, ::2], dtype=np.float32)
            v = np.asarray(handle["v"][:, ::2, ::2], dtype=np.float32)
            re_value = int(np.asarray(handle["re"]))
            aoa_value = int(np.asarray(handle["aoa"]))
        uv = np.stack((u, v), axis=-1)
        trajectories.append(torch.from_numpy(np.ascontiguousarray(uv)))
        metadata.append({"file": path.name, "frames": int(uv.shape[0]),
                         "re": re_value, "aoa": aoa_value})
        if index % 10 == 0 or index == len(files):
            print(json.dumps({"event": "preload", "files": index, "total": len(files),
                              "elapsed_s": round(time.time() - started, 2)}), flush=True)
    return trajectories, metadata


def make_split(metadata: list[dict], stride: int, val_fraction: float) -> tuple[list[Window], list[Window], list[dict]]:
    if not (0.0 < val_fraction < 0.5):
        raise ValueError("val_fraction must be between 0 and 0.5")
    train: list[Window] = []
    val: list[Window] = []
    manifest: list[dict] = []
    for trajectory, item in enumerate(metadata):
        starts = list(range(0, item["frames"] - 40 + 1, stride))
        if len(starts) < 2:
            raise ValueError(f"Trajectory {item['file']} is too short")
        n_val = max(1, math.ceil(len(starts) * val_fraction))
        val_starts = starts[-n_val:]
        val_boundary = val_starts[0]
        # A training window must end no later than the first validation input.
        train_starts = [start for start in starts if start + 40 <= val_boundary]
        if not train_starts:
            raise ValueError(f"No training windows remain for {item['file']}")
        train.extend(Window(trajectory, start) for start in train_starts)
        val.extend(Window(trajectory, start) for start in val_starts)
        manifest.append({**item, "train_starts": train_starts, "val_starts": val_starts,
                         "val_boundary": val_boundary})
    return train, val, manifest


def cycling(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred[..., :2].astype(np.float64, copy=False)
    target = target[..., :2].astype(np.float64, copy=False)
    diff = pred - target
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    numerator = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    denominator = np.linalg.norm(target.reshape(target.shape[0], -1), axis=1)
    rel_l2 = float(np.mean(numerator / np.maximum(denominator, 1e-12)))
    target_mean = np.mean(target)
    r2 = float(1.0 - np.sum(diff ** 2) / max(np.sum((target - target_mean) ** 2), 1e-12))

    # Match the official scorer's TKE definition: temporal fluctuation energy.
    u_p = np.mean((pred[..., 0] - np.mean(pred[..., 0], axis=1, keepdims=True)) ** 2, axis=1)
    v_p = np.mean((pred[..., 1] - np.mean(pred[..., 1], axis=1, keepdims=True)) ** 2, axis=1)
    u_t = np.mean((target[..., 0] - np.mean(target[..., 0], axis=1, keepdims=True)) ** 2, axis=1)
    v_t = np.mean((target[..., 1] - np.mean(target[..., 1], axis=1, keepdims=True)) ** 2, axis=1)
    tke_pred = 0.5 * (u_p + v_p)
    tke_target = 0.5 * (u_t + v_t)
    tke_num = np.linalg.norm((tke_pred - tke_target).reshape(pred.shape[0], -1), axis=1)
    tke_den = np.linalg.norm(tke_target.reshape(pred.shape[0], -1), axis=1)
    tke_rel_l2 = float(np.mean(tke_num / np.maximum(tke_den, 1e-12)))
    mvpe_errors = []
    probe_center_y = 32 // 2
    probe_y = [probe_center_y + 2 * offset for offset in range(-4, 5)]
    probe_y = [y for y in probe_y if 0 <= y < pred.shape[2]]
    for index in range(4):
        if (2 * 16 + 10) // 2 < pred.shape[3]:
            probe_x = ((index + 1) * 16 + 10) // 2
        else:
            probe_x = int((0.5 * (index + 2) * 16 + 10) / 2)
        if not 0 <= probe_x < pred.shape[3]:
            continue
        pred_profile = pred[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(pred.shape[0], -1)
        target_profile = target[:, :, probe_y, probe_x, :2].mean(axis=1).reshape(target.shape[0], -1)
        profile_denominator = np.linalg.norm(target_profile, axis=1)
        mvpe_errors.append(np.linalg.norm(pred_profile - target_profile, axis=1) /
                           np.maximum(profile_denominator, 1e-12))
    mvpe_rel_l2 = float(np.mean(np.stack(mvpe_errors))) if mvpe_errors else 0.0
    return {"rmse": rmse, "mae": mae, "rel_l2": rel_l2, "r2": r2,
            "tke_rel_l2": tke_rel_l2, "mvpe_rel_l2": mvpe_rel_l2}


def official_accuracy_objective(pred_norm: torch.Tensor, target_norm: torch.Tensor,
                                normalizer: Normalizer,
                                accuracy_weights: tuple[float, float, float],
                                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable weighted proxy for the Rel-L2/TKE/MVPE scores."""
    pred = normalizer.postprocess_prediction(pred_norm)[..., :2]
    target = normalizer.postprocess_target(target_norm)[..., :2]
    batch = pred.shape[0]

    rel_num = torch.linalg.vector_norm((pred - target).reshape(batch, -1), dim=1)
    rel_den = torch.linalg.vector_norm(target.reshape(batch, -1), dim=1).clamp_min(1e-8)
    rel_error = torch.mean(rel_num / rel_den)

    pred_fluctuation = pred - torch.mean(pred, dim=1, keepdim=True)
    target_fluctuation = target - torch.mean(target, dim=1, keepdim=True)
    pred_tke = 0.5 * torch.sum(torch.mean(pred_fluctuation ** 2, dim=1), dim=-1)
    target_tke = 0.5 * torch.sum(torch.mean(target_fluctuation ** 2, dim=1), dim=-1)
    tke_num = torch.linalg.vector_norm((pred_tke - target_tke).reshape(batch, -1), dim=1)
    tke_den = torch.linalg.vector_norm(target_tke.reshape(batch, -1), dim=1).clamp_min(1e-8)
    tke_error = torch.mean(tke_num / tke_den)

    probe_center_y = pred.shape[2] // 2
    probe_y = [probe_center_y + 2 * offset for offset in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < pred.shape[2]]
    mvpe_errors = []
    for index in range(4):
        probe_x = ((index + 1) * 16 + 10) // 2
        if not 0 <= probe_x < pred.shape[3]:
            continue
        pred_profile = torch.mean(pred[:, :, probe_y, probe_x, :], dim=1).reshape(batch, -1)
        target_profile = torch.mean(target[:, :, probe_y, probe_x, :], dim=1).reshape(batch, -1)
        numerator = torch.linalg.vector_norm(pred_profile - target_profile, dim=1)
        denominator = torch.linalg.vector_norm(target_profile, dim=1).clamp_min(1e-8)
        mvpe_errors.append(numerator / denominator)
    if not mvpe_errors:
        mvpe_error = torch.zeros((), device=pred.device, dtype=pred.dtype)
    else:
        mvpe_error = torch.mean(torch.stack(mvpe_errors, dim=0))

    errors = torch.stack((rel_error, tke_error, mvpe_error))
    normalized_scores = 1.0 / (1.0 + 0.5 * errors)
    weights = normalized_scores.new_tensor(accuracy_weights)
    objective = 1.0 - torch.sum(weights * normalized_scores) / torch.sum(weights)
    return objective, {
        "rel_l2": rel_error.detach(),
        "tke_rel_l2": tke_error.detach(),
        "mvpe_rel_l2": mvpe_error.detach(),
        "accuracy_objective": objective.detach(),
    }


def per_sample_accuracy_objective(
        pred_norm: torch.Tensor, target_norm: torch.Tensor,
        normalizer: Normalizer,
        accuracy_weights: tuple[float, float, float]) -> torch.Tensor:
    """Per-sample counterpart used by the optional group-robust objective."""
    pred = normalizer.postprocess_prediction(pred_norm)[..., :2]
    target = normalizer.postprocess_target(target_norm)[..., :2]
    batch = pred.shape[0]
    rel_error = torch.linalg.vector_norm(
        (pred - target).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        target.reshape(batch, -1), dim=1).clamp_min(1e-8)

    pred_fluctuation = pred - pred.mean(dim=1, keepdim=True)
    target_fluctuation = target - target.mean(dim=1, keepdim=True)
    pred_tke = 0.5 * torch.sum(pred_fluctuation.square().mean(dim=1), dim=-1)
    target_tke = 0.5 * torch.sum(target_fluctuation.square().mean(dim=1), dim=-1)
    tke_error = torch.linalg.vector_norm(
        (pred_tke - target_tke).reshape(batch, -1), dim=1
    ) / torch.linalg.vector_norm(
        target_tke.reshape(batch, -1), dim=1).clamp_min(1e-8)

    probe_center_y = pred.shape[2] // 2
    probe_y = [probe_center_y + 2 * offset for offset in range(-4, 5)]
    probe_y = [value for value in probe_y if 0 <= value < pred.shape[2]]
    mvpe_errors = []
    for index in range(4):
        probe_x = ((index + 1) * 16 + 10) // 2
        if not 0 <= probe_x < pred.shape[3]:
            continue
        pred_profile = pred[:, :, probe_y, probe_x].mean(dim=1).reshape(batch, -1)
        target_profile = target[:, :, probe_y, probe_x].mean(dim=1).reshape(batch, -1)
        mvpe_errors.append(
            torch.linalg.vector_norm(pred_profile - target_profile, dim=1)
            / torch.linalg.vector_norm(target_profile, dim=1).clamp_min(1e-8))
    mvpe_error = torch.stack(mvpe_errors).mean(dim=0)
    errors = torch.stack((rel_error, tke_error, mvpe_error), dim=1)
    scores = 1.0 / (1.0 + 0.5 * errors)
    weights = scores.new_tensor(accuracy_weights)
    return 1.0 - torch.sum(scores * weights, dim=1) / torch.sum(weights)


def marginal_group_robust_objective(
        sample_loss: torch.Tensor, re_group: torch.Tensor,
        aoa_group: torch.Tensor, temperature: float) -> torch.Tensor:
    means = []
    for groups in (re_group, aoa_group):
        for value in torch.unique(groups):
            selected = groups == value
            if torch.any(selected):
                means.append(sample_loss[selected].mean())
    stacked = torch.stack(means)
    return temperature * (
        torch.logsumexp(stacked / temperature, dim=0)
        - math.log(len(means)))


def temporal_spectrum_objective(
        pred_norm: torch.Tensor, target_norm: torch.Tensor,
        normalizer: Normalizer) -> torch.Tensor:
    pred = normalizer.postprocess_prediction(pred_norm)[..., :2]
    target = normalizer.postprocess_target(target_norm)[..., :2]
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    pred_spectrum = torch.abs(torch.fft.rfft(pred, dim=1))
    target_spectrum = torch.abs(torch.fft.rfft(target, dim=1))
    batch = pred.shape[0]
    return torch.mean(
        torch.linalg.vector_norm(
            (pred_spectrum - target_spectrum).reshape(batch, -1), dim=1)
        / torch.linalg.vector_norm(
            target_spectrum.reshape(batch, -1), dim=1).clamp_min(1e-8))


def add_weighted_accuracy_score(metrics: dict[str, float],
                                accuracy_weights: tuple[float, float, float]) -> None:
    scores = np.asarray([
        metrics["rel_l2_score"], metrics["tke_rel_l2_score"],
        metrics["mvpe_rel_l2_score"],
    ], dtype=np.float64)
    weights = np.asarray(accuracy_weights, dtype=np.float64)
    metrics["weighted_accuracy_score"] = float(np.average(scores, weights=weights))


def parameter_vector_metrics(model: torch.nn.Module,
                             reference: dict[str, torch.Tensor]) -> dict[str, object]:
    dot = 0.0
    norm = 0.0
    ref_norm = 0.0
    drift = 0.0
    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"drift_sq": 0.0,
                                                               "ref_sq": 0.0,
                                                               "param_sq": 0.0})
    with torch.no_grad():
        for name, param in model.named_parameters():
            current = real_parameter_components(param)
            base = reference[name]
            delta = current - base
            dot += float(torch.sum(current * base))
            norm += float(torch.sum(current * current))
            ref_norm += float(torch.sum(base * base))
            drift += float(torch.sum(delta * delta))
            group = name.split(".", 1)[0]
            groups[group]["drift_sq"] += float(torch.sum(delta * delta))
            groups[group]["ref_sq"] += float(torch.sum(base * base))
            groups[group]["param_sq"] += float(torch.sum(current * current))
    result_groups = {}
    for group, values in sorted(groups.items()):
        result_groups[group] = {
            "relative_l2": math.sqrt(values["drift_sq"] / max(values["ref_sq"], 1e-30)),
            "drift_l2": math.sqrt(values["drift_sq"]),
            "parameter_l2": math.sqrt(values["param_sq"]),
        }
    return {
        "relative_l2": math.sqrt(drift / max(ref_norm, 1e-30)),
        "drift_l2": math.sqrt(drift),
        "parameter_l2": math.sqrt(norm),
        "reference_l2": math.sqrt(ref_norm),
        "cosine": dot / max(math.sqrt(norm * ref_norm), 1e-30),
        "groups": result_groups,
    }


def real_parameter_components(value: torch.Tensor) -> torch.Tensor:
    """Return real-valued components without dropping complex FNO weights."""
    result = value.detach().cpu()
    if torch.is_complex(result):
        result = torch.view_as_real(result)
    return result.float()


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: real_parameter_components(value).clone()
            for name, value in model.named_parameters()}


def evaluate(model: torch.nn.Module, loader: DataLoader, normalizer: Normalizer,
             device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    normalized_mse_sum = 0.0
    samples = 0
    started = time.time()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_norm, y_norm = normalizer.preprocess(x, y)
            pred_norm = model(x_norm)
            batch = x.shape[0]
            normalized_mse_sum += float(torch.mean((pred_norm[..., :2] - y_norm[..., :2]) ** 2)) * batch
            samples += batch
            predictions.append(normalizer.postprocess_prediction(pred_norm).cpu().numpy())
            targets.append(normalizer.postprocess_target(y_norm).cpu().numpy())
    metrics = regression_metrics(np.concatenate(predictions), np.concatenate(targets))
    accuracy_score_names = ("rel_l2", "tke_rel_l2", "mvpe_rel_l2")
    for name in accuracy_score_names:
        metrics[f"{name}_score"] = 100.0 / (1.0 + 0.5 * metrics[name])
    metrics["accuracy_score_mean"] = float(np.mean(
        [metrics[f"{name}_score"] for name in accuracy_score_names]))
    metrics["normalized_mse"] = normalized_mse_sum / samples
    metrics["eval_seconds"] = time.time() - started
    metrics["samples"] = samples
    return metrics


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int,
                    history: list[dict], args: argparse.Namespace) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "iteration": iteration,
        "history": history,
        "reproduction_config": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()},
    }
    if args.checkpoint_mode == "full":
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if (not args.start_label or not args.start_label.replace("_", "").isalnum()):
        raise ValueError("start label must contain only letters, digits, and underscores")
    if args.lr is None:
        args.lr = {"cno": 3e-4, "fno": 1e-4, "transolver": 7e-4}[args.model]
    if args.updates <= 0 or args.batch_size <= 0:
        raise ValueError("updates and batch size must be positive")
    if (not math.isfinite(args.mse_weight) or args.mse_weight < 0
            or not math.isfinite(args.spectrum_weight)
            or args.spectrum_weight < 0
            or not math.isfinite(args.group_robust_weight)
            or not 0.0 <= args.group_robust_weight <= 1.0
            or not math.isfinite(args.group_temperature)
            or args.group_temperature <= 0):
        raise ValueError("loss weights must be finite and valid")
    accuracy_weights_array = np.asarray(args.accuracy_weights, dtype=np.float64)
    if (not np.all(np.isfinite(accuracy_weights_array)) or
            np.any(accuracy_weights_array < 0) or
            not np.any(accuracy_weights_array > 0)):
        raise ValueError("accuracy weights must be finite, non-negative, and not all zero")
    accuracy_weights = tuple(float(value) for value in accuracy_weights_array)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    trajectories, metadata = load_trajectories(args.real_dir, args.max_files)
    train_windows, val_windows, split_manifest = make_split(
        metadata, args.window_stride, args.val_fraction)
    write_json(args.output_dir / "split_manifest.json", split_manifest)
    print(json.dumps({"event": "split", "trajectories": len(trajectories),
                      "train_windows": len(train_windows), "val_windows": len(val_windows)}), flush=True)

    train_dataset = (
        GroupedWindowDataset(trajectories, train_windows, metadata)
        if args.group_robust_weight > 0 else
        WindowDataset(trajectories, train_windows))
    val_dataset = WindowDataset(trajectories, val_windows)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    train_iterator = cycling(train_loader)
    normalizer = Normalizer(args.stats, device)

    loader_module = load_baseline_module(args.kit_dir)
    model, start_meta = loader_module.load_baseline(
        args.model, str(args.start_checkpoint), device=str(device))
    initial_parameters = trainable_state(model)
    official_model, official_meta = loader_module.load_baseline(
        args.model, str(args.official_checkpoint), device=str(device))
    official_parameters = trainable_state(official_model)

    initial_metrics = evaluate(model, val_loader, normalizer, device)
    official_metrics = evaluate(official_model, val_loader, normalizer, device)
    add_weighted_accuracy_score(initial_metrics, accuracy_weights)
    add_weighted_accuracy_score(official_metrics, accuracy_weights)
    del official_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.updates)
    history: list[dict] = []
    curve_path = args.output_dir / "learning_curve.jsonl"
    best_rmse = float("inf")
    best_metric_value = (float("inf") if args.selection_metric == "rmse"
                         else -float("inf"))
    best_iteration = 0
    train_loss_sum = 0.0
    train_mse_sum = 0.0
    train_component_sums = {"rel_l2": 0.0, "tke_rel_l2": 0.0,
                            "mvpe_rel_l2": 0.0, "accuracy_objective": 0.0,
                            "group_robust_objective": 0.0,
                            "spectrum_objective": 0.0}
    interval_count = 0
    started = time.time()
    last_grad_norm = float("nan")

    with curve_path.open("w", encoding="utf-8") as curve:
        for iteration in range(1, args.updates + 1):
            model.train()
            batch_items = next(train_iterator)
            x, y = batch_items[:2]
            re_group = aoa_group = None
            if len(batch_items) == 4:
                re_group = batch_items[2].to(device, non_blocking=True)
                aoa_group = batch_items[3].to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_norm, y_norm = normalizer.preprocess(x, y)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_norm)
            mse = torch.mean((pred - y_norm) ** 2)
            if args.loss_mode == "mse":
                loss = mse
                components = None
            else:
                if args.group_robust_weight > 0:
                    assert re_group is not None and aoa_group is not None
                    sample_objective = per_sample_accuracy_objective(
                        pred, y_norm, normalizer, accuracy_weights)
                    mean_objective = sample_objective.mean()
                    group_objective = marginal_group_robust_objective(
                        sample_objective, re_group, aoa_group,
                        args.group_temperature)
                    accuracy_objective = (
                        (1.0 - args.group_robust_weight) * mean_objective
                        + args.group_robust_weight * group_objective)
                    components = {
                        "rel_l2": torch.zeros((), device=device),
                        "tke_rel_l2": torch.zeros((), device=device),
                        "mvpe_rel_l2": torch.zeros((), device=device),
                        "accuracy_objective": mean_objective.detach(),
                        "group_robust_objective": group_objective.detach(),
                    }
                else:
                    accuracy_objective, components = official_accuracy_objective(
                        pred, y_norm, normalizer, accuracy_weights)
                loss = accuracy_objective + args.mse_weight * mse
            if args.spectrum_weight > 0:
                spectrum_objective = temporal_spectrum_objective(
                    pred, y_norm, normalizer)
                loss = loss + args.spectrum_weight * spectrum_objective
                if components is not None:
                    components["spectrum_objective"] = spectrum_objective.detach()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at iteration {iteration}: {loss}")
            loss.backward()
            if iteration == 1 or iteration % args.eval_interval == 0:
                grad_sq = torch.zeros((), device=device)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        grad_sq += torch.sum(parameter.grad.detach().abs() ** 2)
                last_grad_norm = float(torch.sqrt(grad_sq))
            optimizer.step()
            scheduler.step()
            train_loss_sum += float(loss)
            train_mse_sum += float(mse)
            if components is not None:
                for key, value in components.items():
                    train_component_sums[key] += float(value)
            interval_count += 1

            if iteration % args.eval_interval == 0 or iteration == args.updates:
                metrics = evaluate(model, val_loader, normalizer, device)
                add_weighted_accuracy_score(metrics, accuracy_weights)
                row = {
                    "iteration": iteration,
                    "train_loss_mean": train_loss_sum / interval_count,
                    "train_mse_mean": train_mse_sum / interval_count,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_l2": last_grad_norm,
                    "elapsed_seconds": time.time() - started,
                    **metrics,
                }
                if args.loss_mode == "official_accuracy":
                    row.update({f"train_{key}_mean": value / interval_count
                                for key, value in train_component_sums.items()})
                history.append(row)
                curve.write(json.dumps(row, ensure_ascii=False) + "\n")
                curve.flush()
                print(json.dumps({"event": "eval", **row}), flush=True)
                train_loss_sum = 0.0
                train_mse_sum = 0.0
                train_component_sums = {key: 0.0 for key in train_component_sums}
                interval_count = 0
                best_rmse = min(best_rmse, metrics["rmse"])
                selected_value = metrics[args.selection_metric]
                improved = (selected_value < best_metric_value
                            if args.selection_metric == "rmse"
                            else selected_value > best_metric_value)
                if improved:
                    best_metric_value = selected_value
                    best_iteration = iteration
                    save_checkpoint(args.output_dir / "best.pth", model, optimizer,
                                    scheduler, iteration, history, args)

            if iteration % args.checkpoint_interval == 0:
                save_checkpoint(args.output_dir / f"model_{iteration:04d}.pth", model,
                                optimizer, scheduler, iteration, history, args)

    save_checkpoint(args.output_dir / "final.pth", model, optimizer, scheduler,
                    args.updates, history, args)
    final_metrics = evaluate(model, val_loader, normalizer, device)
    add_weighted_accuracy_score(final_metrics, accuracy_weights)
    parameter_comparisons = {
        f"local_final_vs_{args.start_label}": parameter_vector_metrics(model, initial_parameters),
        "local_final_vs_released_official_finetune": parameter_vector_metrics(model, official_parameters),
    }
    best_payload = torch.load(args.output_dir / "best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"])
    best_metrics = evaluate(model, val_loader, normalizer, device)
    add_weighted_accuracy_score(best_metrics, accuracy_weights)
    summary = {
        "model": args.model,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "checkpoint_mode": args.checkpoint_mode,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad),
        "learning_rate": args.lr,
        "loss_mode": args.loss_mode,
        "mse_weight": args.mse_weight,
        "spectrum_weight": args.spectrum_weight,
        "group_robust_weight": args.group_robust_weight,
        "group_temperature": args.group_temperature,
        "accuracy_weights_rel_tke_mvpe": list(accuracy_weights),
        "selection_metric": args.selection_metric,
        "best_metric_value": best_metric_value,
        "seed": args.seed,
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "trajectories": len(trajectories),
        "best_iteration": best_iteration,
        "best_rmse_seen": best_rmse,
        "initial_checkpoint_meta": start_meta,
        "official_checkpoint_meta": official_meta,
        "metrics": {
            args.start_label: initial_metrics,
            "local_finetune_best": best_metrics,
            "local_finetune_final": final_metrics,
            "released_official_finetune": official_metrics,
        },
        "parameter_comparisons": parameter_comparisons,
        "wall_seconds": time.time() - started,
    }
    # Include the organizer checkpoint's direct drift from this run's start.
    temp_model, _ = loader_module.load_baseline(
        args.model, str(args.official_checkpoint), device="cpu")
    summary["parameter_comparisons"][
        f"released_official_finetune_vs_{args.start_label}"] = \
        parameter_vector_metrics(temp_model, initial_parameters)
    del temp_model
    write_json(args.output_dir / "summary.json", summary)

    with (args.output_dir / "learning_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({"event": "complete", "summary": str(args.output_dir / "summary.json"),
                      "best_iteration": best_iteration,
                      "selection_metric": args.selection_metric,
                      "best_metric_value": best_metric_value,
                      "best_rmse_seen": best_rmse,
                      "wall_seconds": summary["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
