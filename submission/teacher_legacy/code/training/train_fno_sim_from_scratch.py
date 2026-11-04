#!/usr/bin/env python3
"""Train a Track 1 FNO from random initialization on released simulations.

This is a controlled alternative to the released ``sim_fno.pth`` starting
point.  It uses only competition-released HDF5 files, records the exact window
set and normalization statistics, and keeps the released checkpoint untouched.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
_repro_spec = importlib.util.spec_from_file_location(
    "realpde_reproduce", REPO_ROOT / "code/training/reproduce_finetune.py")
if _repro_spec is None or _repro_spec.loader is None:
    raise ImportError("Cannot import reproduce_finetune.py")
_repro = importlib.util.module_from_spec(_repro_spec)
sys.modules[_repro_spec.name] = _repro
_repro_spec.loader.exec_module(_repro)


@dataclass(frozen=True)
class SimWindow:
    trajectory: int
    start: int


class SimWindowDataset(Dataset):
    def __init__(self, trajectories: list[torch.Tensor], windows: list[SimWindow],
                 mask_probability: float, noise_scale: float):
        self.trajectories = trajectories
        self.windows = windows
        self.mask_probability = float(mask_probability)
        self.noise_scale = float(noise_scale)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.windows[index]
        data = self.trajectories[item.trajectory][item.start:item.start + 40].clone()
        # Match the public RealPDEBench numerical-data augmentation: pressure
        # is hidden for a fraction of samples and multiplicative Gaussian noise
        # is applied independently to both input and target fields.
        if torch.rand(()) < self.mask_probability:
            data[..., 2] = 0.0
        x, y = data[:20], data[20:]
        if self.noise_scale > 0:
            x = x + x * torch.randn_like(x) * self.noise_scale
            y = y + y * torch.randn_like(y) * self.noise_scale
        return x, y


class ChannelNormalizer:
    def __init__(self, mean_input: torch.Tensor, mean_target: torch.Tensor,
                 std_input: torch.Tensor, std_target: torch.Tensor,
                 device: torch.device):
        self.mean_input = mean_input.to(device)
        self.mean_target = mean_target.to(device)
        self.std_input = std_input.clamp_min(1e-8).to(device)
        self.std_target = std_target.clamp_min(1e-8).to(device)

    def preprocess(self, x: torch.Tensor, y: torch.Tensor):
        return ((x - self.mean_input) / self.std_input,
                (y - self.mean_target) / self.std_target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, default=Path("data/sim"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument(
        "--exclude-final-window", action=argparse.BooleanOptionalAction, default=True,
        help="Match RealPDEBench's floor-based numerical window count.")
    parser.add_argument("--mask-probability", type=float, default=0.1)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--kit-dir", type=Path,
        default=Path("kits/track1/realpde_t1_starting_kit_v9"))
    return parser.parse_args()


def load_simulations(sim_dir: Path, max_files: int,
                     stride: int, exclude_final: bool):
    files = sorted(sim_dir.glob("*.h5"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No simulation HDF5 files in {sim_dir}")
    trajectories = []
    windows = []
    manifest = []
    started = time.time()
    for file_index, path in enumerate(files):
        with h5py.File(path, "r") as handle:
            channels = [
                np.asarray(handle[name][:, ::2, ::2], dtype=np.float32)
                for name in ("u", "v", "p")
            ]
            re_value = int(np.asarray(handle["re"]))
            aoa_value = int(np.asarray(handle["aoa"]))
        trajectory = torch.from_numpy(np.ascontiguousarray(np.stack(channels, axis=-1)))
        trajectories.append(trajectory)
        stop = trajectory.shape[0] - 40 + (0 if exclude_final else 1)
        starts = list(range(0, stop, stride))
        windows.extend(SimWindow(file_index, start) for start in starts)
        manifest.append({
            "file": path.name,
            "frames": int(trajectory.shape[0]),
            "re": re_value,
            "aoa": aoa_value,
            "starts": starts,
        })
        count = file_index + 1
        if count % 10 == 0 or count == len(files):
            print(json.dumps({"event": "preload", "files": count,
                              "total": len(files),
                              "elapsed_s": round(time.time() - started, 2)}), flush=True)
    return trajectories, windows, manifest


def window_statistics(trajectories: list[torch.Tensor],
                      windows: list[SimWindow]):
    sums = [torch.zeros(3, dtype=torch.float64) for _ in range(2)]
    sums_sq = [torch.zeros(3, dtype=torch.float64) for _ in range(2)]
    counts = [0, 0]
    for item in windows:
        data = trajectories[item.trajectory][item.start:item.start + 40]
        for index, part in enumerate((data[:20], data[20:])):
            values = part.to(torch.float64)
            sums[index] += values.sum(dim=(0, 1, 2))
            sums_sq[index] += values.square().sum(dim=(0, 1, 2))
            counts[index] += part.numel() // 3
    means = [sums[index] / counts[index] for index in range(2)]
    variances = [
        sums_sq[index] / counts[index] - means[index].square()
        for index in range(2)
    ]
    stds = [variance.clamp_min(0).sqrt() for variance in variances]
    return tuple(value.float() for value in (*means, *stds))


def save_checkpoint(path: Path, model: nn.Module, iteration: int,
                    metric: float, args: argparse.Namespace,
                    stats: tuple[torch.Tensor, ...]) -> None:
    torch.save({
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()},
        "iteration": iteration,
        "best_iteration": iteration,
        "selection_train_interval_median_mse": metric,
        "sim_normalizer": {
            "mean_input": stats[0], "mean_target": stats[1],
            "std_input": stats[2], "std_target": stats[3],
        },
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }, path)


def main() -> None:
    args = parse_args()
    if (args.updates <= 0 or args.batch_size <= 0 or args.eval_interval <= 0
            or args.window_stride <= 0 or args.lr <= 0
            or not 0 <= args.mask_probability <= 1
            or args.noise_scale < 0 or not math.isfinite(args.noise_scale)):
        raise ValueError("Invalid training arguments")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _repro.set_seed(args.seed)
    trajectories, windows, manifest = load_simulations(
        args.sim_dir, args.max_files, args.window_stride,
        args.exclude_final_window)
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats = window_statistics(trajectories, windows)
    torch.save(stats, args.output_dir / "mean_std_sim.pt")
    print(json.dumps({
        "event": "dataset", "trajectories": len(trajectories),
        "windows": len(windows),
        "mean_input": stats[0].tolist(), "mean_target": stats[1].tolist(),
        "std_input": stats[2].tolist(), "std_target": stats[3].tolist(),
    }), flush=True)

    dataset = SimWindowDataset(
        trajectories, windows, args.mask_probability, args.noise_scale)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed))
    iterator = _repro.cycling(loader)
    normalizer = ChannelNormalizer(*stats, device=device)
    baseline = _repro.load_baseline_module(args.kit_dir)
    model = baseline.build_model("fno", device=str(device))
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates)

    started = time.time()
    interval_losses = []
    best_loss = float("inf")
    best_iteration = 0
    history = []
    for iteration in range(1, args.updates + 1):
        x, y = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_norm, y_norm = normalizer.preprocess(x, y)
        prediction = model(x_norm)
        loss = torch.mean((prediction - y_norm) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {iteration}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        interval_losses.append(float(loss))
        if iteration % args.eval_interval == 0 or iteration == args.updates:
            mean_loss = float(np.mean(interval_losses))
            median_loss = float(np.median(interval_losses))
            row = {
                "iteration": iteration,
                "train_mse": mean_loss,
                "train_mse_median": median_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(json.dumps({"event": "eval", **row}), flush=True)
            if median_loss < best_loss:
                best_loss = median_loss
                best_iteration = iteration
                save_checkpoint(
                    args.output_dir / "best.pth", model, iteration,
                    median_loss, args, stats)
            interval_losses = []

    save_checkpoint(
        args.output_dir / "final.pth", model, args.updates,
        history[-1]["train_mse_median"], args, stats)
    summary = {
        "updates": args.updates,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "trajectories": len(trajectories),
        "windows": len(windows),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_iteration": best_iteration,
        "best_interval_train_median_mse": best_loss,
        "wall_seconds": time.time() - started,
        "history": history,
        "normalizer": {
            "mean_input": stats[0].tolist(), "mean_target": stats[1].tolist(),
            "std_input": stats[2].tolist(), "std_target": stats[3].tolist(),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    with (args.output_dir / "learning_curve.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({"event": "complete", "best_iteration": best_iteration,
                      "best_interval_train_median_mse": best_loss,
                      "wall_seconds": summary["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
