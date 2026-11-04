#!/usr/bin/env python3
"""Paired Real4k test of stop-gradient cross-horizon consistency."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (REPO_ROOT, PRIVATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_standard24k_real_dense_aug as standard24  # noqa: E402


EXTRA_BATCH_SIZE = 4
EXTRA_SEED = 95_004
EXTRA_OUTER_WEIGHT = 0.10
CONSISTENCY_BUDGET_FRACTION = 0.20
SHIFT_STEPS = 5
CONSISTENCY_STEPS = 9


@dataclass(frozen=True)
class CrossHorizonWindow:
    trajectory: int
    start: int


class CrossHorizonDataset(Dataset):
    """Return two histories whose forecasts share physical target frames."""

    def __init__(self, trajectories, windows):
        self.trajectories = trajectories
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        uv = self.trajectories[window.trajectory][
            window.start:window.start + 40]
        if uv.shape[0] != 40 or uv.shape[-1] != 2:
            raise RuntimeError("cross-horizon window is not 40-frame u/v")
        pressure = torch.zeros((*uv.shape[:-1], 1), dtype=uv.dtype)
        uvp = torch.cat((uv, pressure), dim=-1)
        return (
            uvp[:20], uvp[20:40], uvp[SHIFT_STEPS:SHIFT_STEPS + 20],
            window.trajectory, window.start)


def build_cross_horizon_windows(trajectories) -> list[CrossHorizonWindow]:
    real = standard24.self_describing.real
    return [
        CrossHorizonWindow(int(window.trajectory), int(window.start))
        for window in real.dense_windows(trajectories, real.CANDIDATE_STRIDE)
    ]


def extra_batch_schedule(count: int, updates: int) -> list[list[int]]:
    if count < EXTRA_BATCH_SIZE or updates <= 0:
        raise ValueError("cross-horizon inventory is too small")
    generator = torch.Generator().manual_seed(EXTRA_SEED)
    batches = []
    while len(batches) < updates:
        order = torch.randperm(count, generator=generator).tolist()
        batches.extend([
            order[start:start + EXTRA_BATCH_SIZE]
            for start in range(0, count, EXTRA_BATCH_SIZE)
            if len(order[start:start + EXTRA_BATCH_SIZE]) == EXTRA_BATCH_SIZE
        ])
    return batches[:updates]


def budget_matched_loss(
        direct_loss: torch.Tensor, consistency_loss: torch.Tensor,
        fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a fixed direct-loss budget without changing scalar magnitude."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("consistency budget fraction must be in [0, 1]")
    if direct_loss.ndim != 0 or consistency_loss.ndim != 0:
        raise ValueError("budget-matched losses must be scalar")
    scale = direct_loss.detach() / consistency_loss.detach().clamp_min(1e-12)
    normalized_consistency = consistency_loss * scale
    mixed = ((1.0 - fraction) * direct_loss
             + fraction * normalized_consistency)
    return mixed, scale


def normalized_uv_pixel_consistency(
        prediction: torch.Tensor, teacher: torch.Tensor,
        _normalizer) -> torch.Tensor:
    student_overlap = prediction[
        :, SHIFT_STEPS:SHIFT_STEPS + CONSISTENCY_STEPS, ..., :2]
    teacher_overlap = teacher[:, :CONSISTENCY_STEPS, ..., :2]
    return (student_overlap - teacher_overlap.detach()).square().mean()


def whole_direct_budget_mixer(
        direct_loss: torch.Tensor, consistency_loss: torch.Tensor,
        _prediction: torch.Tensor, _target: torch.Tensor, _normalizer,
        _iteration: int, fraction: float) -> tuple[
            torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    mixed, scale = budget_matched_loss(
        direct_loss, consistency_loss, fraction)
    return mixed, scale, {}


class CrossHorizonBackward:
    def __init__(
            self, dataset, normalizer, device, updates: int, *, arm: str,
            consistency_objective=normalized_uv_pixel_consistency,
            consistency_kind: str = "normalized_uv_pixel_mse",
            budget_mixer=whole_direct_budget_mixer,
            budget_mixer_kind: str = "whole_direct_loss_replacement",
            reference_source: str = "short_horizon_teacher"):
        if arm not in {"control", "candidate"}:
            raise ValueError("unknown cross-horizon arm")
        if reference_source not in {"short_horizon_teacher", "target"}:
            raise ValueError("unknown consistency reference source")
        self.dataset = dataset
        self.normalizer = normalizer
        self.device = device
        self.arm = arm
        self.consistency_objective = consistency_objective
        self.consistency_kind = str(consistency_kind)
        self.budget_mixer = budget_mixer
        self.budget_mixer_kind = str(budget_mixer_kind)
        self.reference_source = str(reference_source)
        self.fraction = (
            0.0 if arm == "control" else CONSISTENCY_BUDGET_FRACTION)
        self.batches = extra_batch_schedule(len(dataset), updates)
        self.loader = DataLoader(
            dataset, batch_sampler=self.batches, num_workers=0,
            pin_memory=device.type == "cuda")
        self.iterator = iter(self.loader)
        self.rows = []

    def __call__(self, iteration: int, model):
        batch = next(self.iterator)
        history_long, target_long, history_short, trajectory, start = batch
        actual = {
            "trajectory": trajectory.to(torch.long).tolist(),
            "start": start.to(torch.long).tolist(),
        }
        expected_indices = self.batches[iteration - 1]
        expected = {
            "trajectory": [self.dataset.windows[index].trajectory
                           for index in expected_indices],
            "start": [self.dataset.windows[index].start
                      for index in expected_indices],
        }
        if actual != expected:
            raise RuntimeError("cross-horizon stream changed")
        self.rows.append(actual)
        history_long = history_long.to(self.device, non_blocking=True)
        target_long = target_long.to(self.device, non_blocking=True)
        history_short = history_short.to(self.device, non_blocking=True)

        cuda_devices = (
            [self.device.index or 0] if self.device.type == "cuda" else [])
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(EXTRA_SEED + iteration)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(EXTRA_SEED + iteration)
            long_norm, target_norm = self.normalizer.preprocess(
                history_long, target_long)
            short_norm = (
                history_short - self.normalizer.mean_input
            ) / self.normalizer.std_input

            if self.reference_source == "short_horizon_teacher":
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    reference = model(short_norm)
                model.train(was_training)
            else:
                reference = target_norm.detach()
            prediction = model(long_norm)

            real = standard24.self_describing.real
            direct_loss, components = real.ft12.npt1.ensemble_train.point_loss(
                real.ft12.npt1.joint_real.real_loss_args(),
                prediction, target_norm, self.normalizer,
                real.ft12.npt1.joint_real.REAL_ACCURACY_WEIGHTS, iteration)
            consistency_loss = self.consistency_objective(
                prediction, reference, self.normalizer)
            extra_loss, consistency_scale, mixer_metrics = self.budget_mixer(
                direct_loss, consistency_loss, prediction, target_norm,
                self.normalizer, iteration, self.fraction)
            weighted_loss = EXTRA_OUTER_WEIGHT * extra_loss
            if not torch.isfinite(weighted_loss):
                raise FloatingPointError(
                    "cross-horizon weighted loss is non-finite")
            weighted_loss.backward()

        return {
            "cross_horizon_direct_loss": float(direct_loss.detach()),
            "cross_horizon_consistency_loss": float(
                consistency_loss.detach()),
            "cross_horizon_consistency_scale": float(
                consistency_scale.detach()),
            "cross_horizon_extra_loss": float(extra_loss.detach()),
            "cross_horizon_weighted_loss": float(weighted_loss.detach()),
            "cross_horizon_budget_fraction": self.fraction,
            **{
                f"cross_horizon_mixer_{key}": float(value.detach())
                for key, value in mixer_metrics.items()
            },
            **{
                f"cross_horizon_{key}": float(value)
                for key, value in components.items()
            },
        }

    def report(self) -> dict:
        payload = json.dumps(
            self.rows, sort_keys=True, separators=(",", ":")).encode()
        return {
            "arm": self.arm,
            "updates": len(self.rows),
            "windows": len(self.dataset),
            "window_stride": standard24.self_describing.real.CANDIDATE_STRIDE,
            "batch_size": EXTRA_BATCH_SIZE,
            "outer_weight": EXTRA_OUTER_WEIGHT,
            "consistency_budget_fraction": self.fraction,
            "shift_steps": SHIFT_STEPS,
            "consistency_steps": CONSISTENCY_STEPS,
            "consistency_kind": self.consistency_kind,
            "budget_mixer_kind": self.budget_mixer_kind,
            "reference_source": self.reference_source,
            "teacher": (
                "same online model eval mode stop-gradient"
                if self.reference_source == "short_horizon_teacher"
                else "released target stop-gradient"),
            "student": "same online model train mode",
            "pressure_exact_zero": True,
            "rng_isolated_from_primary_stream": True,
            "stream_sha256": hashlib.sha256(payload).hexdigest(),
        }


def identical_dense_windows(trajectories, _manifest_windows):
    real = standard24.self_describing.real
    windows = real.dense_windows(trajectories, real.CANDIDATE_STRIDE)
    return {"control": list(windows), "candidate": list(windows)}


def identical_augmentation():
    real = standard24.self_describing.real
    return {
        "control": dict(real.AUGMENTATION),
        "candidate": dict(real.AUGMENTATION),
    }


def bind_cross_horizon_trainer(
        trainer, *, consistency_objective=normalized_uv_pixel_consistency,
        consistency_kind: str = "normalized_uv_pixel_mse",
        budget_mixer=whole_direct_budget_mixer,
        budget_mixer_kind: str = "whole_direct_loss_replacement",
        reference_source: str = "short_horizon_teacher"):
    def train(model, dataset, schedule, normalizer, device, deadline,
              output_dir, start_report, access, **kwargs):
        if output_dir.name not in {"control", "candidate"}:
            raise ValueError(f"cannot infer cross-horizon arm from {output_dir}")
        extra_dataset = CrossHorizonDataset(
            dataset.trajectories,
            build_cross_horizon_windows(dataset.trajectories))
        callback = CrossHorizonBackward(
            extra_dataset, normalizer, device,
            standard24.self_describing.real.small4k.UPDATES,
            arm=output_dir.name,
            consistency_objective=consistency_objective,
            consistency_kind=consistency_kind,
            budget_mixer=budget_mixer,
            budget_mixer_kind=budget_mixer_kind,
            reference_source=reference_source)
        kwargs["extra_backward"] = callback
        kwargs["extra_backward_contract"] = {
            "kind": "budget_matched_stop_gradient_cross_horizon_consistency",
            "outer_weight": EXTRA_OUTER_WEIGHT,
            "candidate_consistency_budget_fraction": (
                CONSISTENCY_BUDGET_FRACTION),
            "control_consistency_budget_fraction": 0.0,
            "shift_steps": SHIFT_STEPS,
            "consistency_steps": CONSISTENCY_STEPS,
            "consistency_kind": consistency_kind,
            "budget_mixer_kind": budget_mixer_kind,
            "reference_source": reference_source,
            "teacher_eval_mode": (
                reference_source == "short_horizon_teacher"),
            "reference_stop_gradient": True,
            "same_extra_batches_and_forwards": True,
            "same_extra_scalar_loss_magnitude_by_detached_normalization": True,
            "inference_graph_changed": False,
        }
        result = trainer(
            model, dataset, schedule, normalizer, device, deadline,
            output_dir, start_report, access, **kwargs)
        training, _seal = result
        training["cross_horizon_consistency"] = callback.report()
        return result
    return train


def main() -> None:
    real = standard24.self_describing.real
    real.EXPERIMENT = "b210_standard24k_real_horizon_consistency"
    real.SCHEMA = "b210_standard24k_real_horizon_consistency_v1"
    real.architecture_contract = (
        standard24.self_describing.self_describing_architecture_contract)
    real.ft12.validate_start_checkpoint = standard24.bind_standard24_validator(
        real.ft12.validate_start_checkpoint)
    real.arm_window_contract = identical_dense_windows
    real.arm_augmentation_contract = identical_augmentation
    real.small4k.train_candidate = bind_cross_horizon_trainer(
        real.small4k.train_candidate)
    real.main()


if __name__ == "__main__":
    main()
