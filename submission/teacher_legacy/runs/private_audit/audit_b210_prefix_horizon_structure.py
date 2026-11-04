#!/usr/bin/env python3
"""Audit scale-free fluctuation structure across shifted Real histories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training", PRIVATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_b210_prefix_horizon_teacher as base  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402


SCHEMA = "b210_prefix_horizon_structure_audit_v1"
CONSISTENCY_STEPS = 9


def fluctuation(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 5 or value.shape[-1] != 2:
        raise ValueError("velocity must be [B,T,H,W,2]")
    return value - value.mean(dim=(2, 3), keepdim=True)


def fluctuation_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = fluctuation(left).flatten(start_dim=2).double()
    right_flat = fluctuation(right).flatten(start_dim=2).double()
    numerator = (left_flat * right_flat).sum(dim=-1)
    denominator = torch.sqrt(
        left_flat.square().sum(dim=-1).clamp_min(1e-30)
        * right_flat.square().sum(dim=-1).clamp_min(1e-30))
    return numerator / denominator


def fluctuation_energy(value: torch.Tensor) -> torch.Tensor:
    return fluctuation(value).double().square().mean(dim=(2, 3, 4))


@dataclass
class StructureAccumulator:
    samples: int = 0
    frames: int = 0
    long_cosine_sum: float = 0.0
    short_cosine_sum: float = 0.0
    short_better_cosine: int = 0
    long_energy_log_error_sum: float = 0.0
    short_energy_log_error_sum: float = 0.0
    short_better_energy: int = 0

    def update(
            self, long: torch.Tensor, short: torch.Tensor,
            truth: torch.Tensor) -> None:
        long_cpu = long.detach().cpu()
        short_cpu = short.detach().cpu()
        truth_cpu = truth.detach().cpu()
        long_cosine = fluctuation_cosine(long_cpu, truth_cpu)
        short_cosine = fluctuation_cosine(short_cpu, truth_cpu)
        truth_energy = fluctuation_energy(truth_cpu).clamp_min(1e-30)
        long_energy_error = torch.log(
            fluctuation_energy(long_cpu).clamp_min(1e-30)
            / truth_energy).abs()
        short_energy_error = torch.log(
            fluctuation_energy(short_cpu).clamp_min(1e-30)
            / truth_energy).abs()
        self.samples += int(long.shape[0])
        self.frames += int(long_cosine.numel())
        self.long_cosine_sum += float(long_cosine.sum())
        self.short_cosine_sum += float(short_cosine.sum())
        self.short_better_cosine += int(
            (short_cosine > long_cosine).sum())
        self.long_energy_log_error_sum += float(long_energy_error.sum())
        self.short_energy_log_error_sum += float(short_energy_error.sum())
        self.short_better_energy += int(
            (short_energy_error < long_energy_error).sum())

    def report(self) -> dict:
        if self.frames <= 0:
            raise RuntimeError("empty structure accumulator")
        long_cosine = self.long_cosine_sum / self.frames
        short_cosine = self.short_cosine_sum / self.frames
        long_energy = self.long_energy_log_error_sum / self.frames
        short_energy = self.short_energy_log_error_sum / self.frames
        return {
            "samples": self.samples,
            "frames": self.frames,
            "long_fluctuation_cosine": long_cosine,
            "short_fluctuation_cosine": short_cosine,
            "short_fluctuation_cosine_gain": short_cosine - long_cosine,
            "short_better_fluctuation_frame_fraction": (
                self.short_better_cosine / self.frames),
            "long_energy_absolute_log_error": long_energy,
            "short_energy_absolute_log_error": short_energy,
            "short_energy_log_error_reduction": long_energy - short_energy,
            "short_better_energy_frame_fraction": (
                self.short_better_energy / self.frames),
        }


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, manifest = dr1.load_manifest(
        args.clean_manifest, dr1.FORMAL_MANIFEST_SHA256)
    access = base.PrefixAudit()
    trajectories, metadata, _manifest_windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", access)
    windows = base.build_windows(trajectories, metadata)
    device = torch.device(args.device)
    statistics = torch.load(
        args.normalizer, map_location="cpu", weights_only=False)
    normalizer = smu1.FoldNormalizer(statistics, device)
    model, payload = base.load_model(args.checkpoint, normalizer, device)
    torch.set_float32_matmul_precision("high")

    overall = StructureAccumulator()
    by_aoa = {aoa: StructureAccumulator() for aoa in dr1.AOA_VALUES}
    by_horizon = [StructureAccumulator()
                  for _ in range(CONSISTENCY_STEPS)]
    with torch.inference_mode():
        for offset in range(0, len(windows), args.batch_size):
            batch = windows[offset:offset + args.batch_size]
            values = torch.stack([
                trajectories[row.trajectory][
                    row.start:row.start + base.WINDOW_FRAMES]
                for row in batch])
            history_long = base.uvp(values[:, :base.HISTORY_STEPS]).to(device)
            history_short = base.uvp(values[:, base.SHIFT_STEPS:
                                            base.SHIFT_STEPS
                                            + base.HISTORY_STEPS]).to(device)
            truth = values[:, base.HISTORY_STEPS + base.SHIFT_STEPS:
                           base.HISTORY_STEPS + base.SHIFT_STEPS
                           + CONSISTENCY_STEPS].to(device)
            history_long = (
                history_long - normalizer.mean_input) / normalizer.std_input
            history_short = (
                history_short - normalizer.mean_input) / normalizer.std_input
            long = normalizer.postprocess_prediction(
                model(history_long).float()
            )[:, base.SHIFT_STEPS:
                base.SHIFT_STEPS + CONSISTENCY_STEPS, ..., :2]
            short = normalizer.postprocess_prediction(
                model(history_short).float()
            )[:, :CONSISTENCY_STEPS, ..., :2]
            overall.update(long, short, truth)
            for aoa in dr1.AOA_VALUES:
                indices = [index for index, row in enumerate(batch)
                           if row.aoa == aoa]
                if indices:
                    by_aoa[aoa].update(
                        long[indices], short[indices], truth[indices])
            for horizon in range(CONSISTENCY_STEPS):
                by_horizon[horizon].update(
                    long[:, horizon:horizon + 1],
                    short[:, horizon:horizon + 1],
                    truth[:, horizon:horizon + 1])

    result = {
        "schema": SCHEMA,
        "scope": {
            "source": "released Real prefix only",
            "tail_arrays_read": 0,
            "windows": len(windows),
            "window_stride": base.WINDOW_STRIDE,
            "consistency_steps": CONSISTENCY_STEPS,
            "long_horizons_one_based": [6, 14],
            "short_horizons_one_based": [1, 9],
            "structure": (
                "per-sample/per-frame spatial-mean-centered joint-uv cosine"),
            "energy": (
                "per-sample/per-frame spatial-mean-centered joint-uv mean square"),
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": dr1.sha256(args.checkpoint),
            "model_state_sha256": payload.get("model_state_sha256"),
        },
        "manifest": manifest,
        "access": {
            "opened_prefix_files": access.opened,
            "opened_count": len(access.opened),
            "tail_arrays_read": 0,
        },
        "overall": overall.report(),
        "by_aoa": {str(aoa): by_aoa[aoa].report()
                   for aoa in dr1.AOA_VALUES},
        "by_horizon": [
            {
                "long_horizon_one_based": horizon + base.SHIFT_STEPS + 1,
                "short_horizon_one_based": horizon + 1,
                **accumulator.report(),
            }
            for horizon, accumulator in enumerate(by_horizon)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({
        "event": "b210_prefix_horizon_structure_audit_complete",
        "overall": result["overall"],
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
