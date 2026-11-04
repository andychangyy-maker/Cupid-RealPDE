#!/usr/bin/env python3
"""Audit whether a five-frame-shorter forecast is a useful Real teacher.

Only released Real prefix frames are opened.  For each legal 45-frame window,
the same frozen point model is evaluated from histories ``[s:s+20]`` and
``[s+5:s+25]``.  Their predictions overlap on physical frames ``[s+25:s+40]``:
long horizons 6--20 and short horizons 1--15.  The audit asks whether the
short-context prediction is both more accurate and points from the long
prediction toward truth; it never reads the clean716 tail.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


SCHEMA = "b210_prefix_horizon_teacher_audit_v1"
WINDOW_FRAMES = 45
WINDOW_STRIDE = 5
HISTORY_STEPS = 20
SHIFT_STEPS = 5
OVERLAP_STEPS = 15
BLEND_ALPHAS = (0.25, 0.50, 1.00)


@dataclass(frozen=True)
class AuditWindow:
    trajectory: int
    start: int
    aoa: int


class PrefixAudit:
    def __init__(self) -> None:
        self.opened: list[str] = []

    def before_open(self, row: dict, phase: str) -> None:
        if phase != "fit" or row["file"] == dr1.EXCLUDED_BASENAME:
            raise RuntimeError("horizon audit attempted an invalid prefix open")
        if row["file"] in self.opened:
            raise RuntimeError("horizon audit attempted a duplicate prefix open")

    def record_open(self, row: dict, phase: str) -> None:
        if phase != "fit" or row["file"] in self.opened:
            raise RuntimeError("invalid horizon prefix access record")
        self.opened.append(str(row["file"]))


def build_windows(
        trajectories: Sequence[torch.Tensor], metadata: Sequence[dict],
        stride: int = WINDOW_STRIDE) -> list[AuditWindow]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    if len(trajectories) != len(metadata):
        raise ValueError("trajectory metadata length changed")
    windows = []
    for trajectory, (values, row) in enumerate(zip(trajectories, metadata)):
        windows.extend(
            AuditWindow(trajectory, start, int(row["aoa"]))
            for start in range(
                0, int(values.shape[0]) - WINDOW_FRAMES + 1, stride))
    if not windows:
        raise ValueError("no legal prefix horizon windows")
    return windows


def uvp(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 5 or values.shape[-1] != 2:
        raise ValueError("velocity batch must be [B,T,H,W,2]")
    pressure = torch.zeros((*values.shape[:-1], 1), dtype=values.dtype)
    return torch.cat((values, pressure), dim=-1)


@dataclass
class ErrorAccumulator:
    samples: int = 0
    elements: int = 0
    truth_sq: float = 0.0
    long_sq: float = 0.0
    short_sq: float = 0.0
    correction_sq: float = 0.0
    correction_oracle_dot: float = 0.0
    short_better_samples: int = 0
    correction_positive_samples: int = 0
    sample_cosine_sum: float = 0.0
    blend_sq: dict[float, float] = field(default_factory=lambda: {
        alpha: 0.0 for alpha in BLEND_ALPHAS})

    def update(
            self, long: torch.Tensor, short: torch.Tensor,
            truth: torch.Tensor) -> None:
        if long.shape != short.shape or long.shape != truth.shape:
            raise ValueError("overlap tensors must have identical shapes")
        if long.ndim != 5 or long.shape[-1] != 2:
            raise ValueError("overlap tensors must be [B,T,H,W,2]")
        long64 = long.detach().to(device="cpu", dtype=torch.float64)
        short64 = short.detach().to(device="cpu", dtype=torch.float64)
        truth64 = truth.detach().to(device="cpu", dtype=torch.float64)
        long_error = long64 - truth64
        short_error = short64 - truth64
        correction = short64 - long64
        oracle = truth64 - long64
        reduce = tuple(range(1, long64.ndim))
        long_by_sample = long_error.square().sum(dim=reduce)
        short_by_sample = short_error.square().sum(dim=reduce)
        dot_by_sample = (correction * oracle).sum(dim=reduce)
        correction_by_sample = correction.square().sum(dim=reduce)
        oracle_by_sample = oracle.square().sum(dim=reduce)
        cosine = dot_by_sample / torch.sqrt(
            correction_by_sample.clamp_min(1e-30)
            * oracle_by_sample.clamp_min(1e-30))

        self.samples += int(long.shape[0])
        self.elements += int(long.numel())
        self.truth_sq += float(truth64.square().sum())
        self.long_sq += float(long_by_sample.sum())
        self.short_sq += float(short_by_sample.sum())
        self.correction_sq += float(correction_by_sample.sum())
        self.correction_oracle_dot += float(dot_by_sample.sum())
        self.short_better_samples += int(
            (short_by_sample < long_by_sample).sum())
        self.correction_positive_samples += int((dot_by_sample > 0).sum())
        self.sample_cosine_sum += float(cosine.sum())
        for alpha in BLEND_ALPHAS:
            blended_error = long_error + alpha * correction
            self.blend_sq[alpha] += float(blended_error.square().sum())

    def report(self) -> dict:
        if self.samples <= 0 or self.elements <= 0 or self.long_sq <= 0.0:
            raise RuntimeError("empty horizon error accumulator")
        global_cosine = self.correction_oracle_dot / math.sqrt(
            max(self.correction_sq * self.long_sq, 1e-30))
        optimal_alpha = self.correction_oracle_dot / max(
            self.correction_sq, 1e-30)
        return {
            "samples": self.samples,
            "elements": self.elements,
            "long_mse": self.long_sq / self.elements,
            "short_mse": self.short_sq / self.elements,
            "short_relative_mse_reduction": 1.0 - self.short_sq / self.long_sq,
            "short_better_sample_fraction": (
                self.short_better_samples / self.samples),
            "correction_positive_sample_fraction": (
                self.correction_positive_samples / self.samples),
            "correction_oracle_global_cosine": global_cosine,
            "correction_oracle_mean_sample_cosine": (
                self.sample_cosine_sum / self.samples),
            "least_squares_correction_alpha": optimal_alpha,
            "relative_l2_long": math.sqrt(
                self.long_sq / max(self.truth_sq, 1e-30)),
            "relative_l2_short": math.sqrt(
                self.short_sq / max(self.truth_sq, 1e-30)),
            "blend_relative_mse_reduction": {
                f"{alpha:.2f}": 1.0 - value / self.long_sq
                for alpha, value in self.blend_sq.items()
            },
        }


def load_model(checkpoint: Path, normalizer, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_config = dict(payload["model_config"])
    if raw_config.pop("architecture") != "ensemble_unet":
        raise ValueError("horizon audit requires an ensemble U-Net checkpoint")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**raw_config)).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    return model.to(device).eval(), payload


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, manifest = dr1.load_manifest(
        args.clean_manifest, dr1.FORMAL_MANIFEST_SHA256)
    access = PrefixAudit()
    trajectories, metadata, _manifest_windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", access)
    windows = build_windows(trajectories, metadata)
    device = torch.device(args.device)
    statistics = torch.load(
        args.normalizer, map_location="cpu", weights_only=False)
    normalizer = smu1.FoldNormalizer(statistics, device)
    model, payload = load_model(args.checkpoint, normalizer, device)
    torch.set_float32_matmul_precision("high")

    overall = ErrorAccumulator()
    by_aoa = {aoa: ErrorAccumulator() for aoa in dr1.AOA_VALUES}
    horizon_long_sq = torch.zeros(OVERLAP_STEPS, dtype=torch.float64)
    horizon_short_sq = torch.zeros(OVERLAP_STEPS, dtype=torch.float64)
    horizon_elements = torch.zeros(OVERLAP_STEPS, dtype=torch.int64)

    with torch.inference_mode():
        for offset in range(0, len(windows), args.batch_size):
            batch = windows[offset:offset + args.batch_size]
            chunks = [
                trajectories[row.trajectory][
                    row.start:row.start + WINDOW_FRAMES]
                for row in batch]
            values = torch.stack(chunks)
            history_long = uvp(values[:, :HISTORY_STEPS]).to(device)
            history_short = uvp(values[:, SHIFT_STEPS:
                                       SHIFT_STEPS + HISTORY_STEPS]).to(device)
            truth = values[:, HISTORY_STEPS + SHIFT_STEPS:
                           HISTORY_STEPS * 2].to(device)
            history_long = (
                history_long - normalizer.mean_input) / normalizer.std_input
            history_short = (
                history_short - normalizer.mean_input) / normalizer.std_input
            prediction_long = normalizer.postprocess_prediction(
                model(history_long).float())[:, SHIFT_STEPS:, ..., :2]
            prediction_short = normalizer.postprocess_prediction(
                model(history_short).float())[:, :OVERLAP_STEPS, ..., :2]
            overall.update(prediction_long, prediction_short, truth)
            for aoa in dr1.AOA_VALUES:
                indices = [index for index, row in enumerate(batch)
                           if row.aoa == aoa]
                if indices:
                    by_aoa[aoa].update(
                        prediction_long[indices], prediction_short[indices],
                        truth[indices])
            long_error = (prediction_long - truth).detach().cpu().double()
            short_error = (prediction_short - truth).detach().cpu().double()
            horizon_long_sq += long_error.square().sum(dim=(0, 2, 3, 4))
            horizon_short_sq += short_error.square().sum(dim=(0, 2, 3, 4))
            horizon_elements += torch.tensor(
                [long_error[:, step].numel()
                 for step in range(OVERLAP_STEPS)], dtype=torch.int64)

    result = {
        "schema": SCHEMA,
        "scope": {
            "source": "released Real prefix only",
            "tail_arrays_read": 0,
            "window_frames": WINDOW_FRAMES,
            "window_stride": WINDOW_STRIDE,
            "windows": len(windows),
            "long_history": "frames s:s+20",
            "short_history": "frames s+5:s+25",
            "physical_overlap": "frames s+25:s+40",
            "long_horizons_one_based": [6, 20],
            "short_horizons_one_based": [1, 15],
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
        "by_physical_overlap_step": [
            {
                "long_horizon_one_based": step + SHIFT_STEPS + 1,
                "short_horizon_one_based": step + 1,
                "long_mse": float(horizon_long_sq[step]
                                  / horizon_elements[step]),
                "short_mse": float(horizon_short_sq[step]
                                   / horizon_elements[step]),
                "short_relative_mse_reduction": float(
                    1.0 - horizon_short_sq[step]
                    / horizon_long_sq[step]),
            }
            for step in range(OVERLAP_STEPS)
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
        "event": "b210_prefix_horizon_teacher_audit_complete",
        "overall": result["overall"],
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
