#!/usr/bin/env python3
"""Compare standard Real4k with prefix-only rollout exposure augmentation."""

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


ROLLOUT_STRIDE = 20
ROLLOUT_BATCH_SIZE = 4
GENERATED_FRACTION = 0.50
ROLLOUT_OUTER_WEIGHT = 0.10
ROLLOUT_SEED = 94_004


@dataclass(frozen=True)
class RolloutWindow:
    trajectory: int
    start: int


class RolloutDataset(Dataset):
    def __init__(self, trajectories, windows):
        self.trajectories = trajectories
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        uv = self.trajectories[window.trajectory][
            window.start:window.start + 60]
        if uv.shape[0] != 60 or uv.shape[-1] != 2:
            raise RuntimeError("rollout window is not 60-frame u/v")
        pressure = torch.zeros((*uv.shape[:-1], 1), dtype=uv.dtype)
        uvp = torch.cat((uv, pressure), dim=-1)
        return (
            uvp[:20], uvp[20:40], uvp[40:60],
            window.trajectory, window.start)


def build_rollout_windows(trajectories) -> list[RolloutWindow]:
    windows = []
    for trajectory, values in enumerate(trajectories):
        windows.extend(
            RolloutWindow(trajectory, start)
            for start in range(0, int(values.shape[0]) - 60 + 1,
                               ROLLOUT_STRIDE))
    if not windows:
        raise ValueError("no legal 60-frame Real-prefix rollout windows")
    return windows


def rollout_batch_schedule(count: int, updates: int) -> list[list[int]]:
    if count < ROLLOUT_BATCH_SIZE or updates <= 0:
        raise ValueError("rollout schedule inventory is too small")
    generator = torch.Generator().manual_seed(ROLLOUT_SEED)
    batches = []
    while len(batches) < updates:
        order = torch.randperm(count, generator=generator).tolist()
        batches.extend([
            order[start:start + ROLLOUT_BATCH_SIZE]
            for start in range(0, count, ROLLOUT_BATCH_SIZE)
            if len(order[start:start + ROLLOUT_BATCH_SIZE])
            == ROLLOUT_BATCH_SIZE
        ])
    return batches[:updates]


def normalize_generated_history(value, normalizer):
    if value.ndim != 5 or value.shape[-1] != 3:
        raise ValueError("generated rollout history must be [B,T,H,W,3]")
    result = (value - normalizer.mean_input) / normalizer.std_input
    if torch.count_nonzero(value[..., 2]) != 0:
        raise ValueError("generated Real pressure must stay exactly zero")
    return result


class RolloutExposureBackward:
    def __init__(
            self, dataset, normalizer, device, updates: int, *,
            generated_fraction: float = GENERATED_FRACTION,
            outer_weight: float = ROLLOUT_OUTER_WEIGHT):
        if not 0.0 <= generated_fraction <= 1.0:
            raise ValueError("generated_fraction must be in [0, 1]")
        if outer_weight <= 0.0:
            raise ValueError("outer_weight must be positive")
        self.dataset = dataset
        self.normalizer = normalizer
        self.device = device
        self.generated_fraction = float(generated_fraction)
        self.outer_weight = float(outer_weight)
        self.batches = rollout_batch_schedule(len(dataset), updates)
        self.loader = DataLoader(
            dataset, batch_sampler=self.batches, num_workers=0,
            pin_memory=device.type == "cuda")
        self.iterator = iter(self.loader)
        self.rows = []

    def __call__(self, iteration: int, model):
        batch = next(self.iterator)
        history, target1, target2, trajectory, start = batch
        actual = {
            "trajectory": trajectory.to(torch.long).tolist(),
            "start": start.to(torch.long).tolist(),
        }
        expected_indices = self.batches[iteration - 1]
        expected = {
            "trajectory": [
                self.dataset.windows[index].trajectory
                for index in expected_indices],
            "start": [
                self.dataset.windows[index].start
                for index in expected_indices],
        }
        if actual != expected:
            raise RuntimeError("rollout exposure stream changed")
        self.rows.append(actual)
        history = history.to(self.device, non_blocking=True)
        target1 = target1.to(self.device, non_blocking=True)
        target2 = target2.to(self.device, non_blocking=True)

        cuda_devices = (
            [self.device.index or 0] if self.device.type == "cuda" else [])
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(ROLLOUT_SEED + iteration)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(ROLLOUT_SEED + iteration)
            history_norm, _target1_norm = self.normalizer.preprocess(
                history, target1)
            with torch.no_grad():
                prediction1 = model(history_norm)
                prediction1_physical = (
                    self.normalizer.postprocess_prediction(
                        prediction1.float()))
                prediction1_physical[..., 2].zero_()
                generated = (
                    (1.0 - self.generated_fraction) * target1
                    + self.generated_fraction * prediction1_physical)
                generated[..., 2].zero_()
                first_step_mse = (
                    prediction1_physical[..., :2] - target1[..., :2]
                ).square().mean()
            generated_norm = normalize_generated_history(
                generated, self.normalizer)
            target2_norm = (
                target2 - self.normalizer.mean_target
            ) / self.normalizer.std_target
            prediction2 = model(generated_norm)
            real = standard24.self_describing.real
            raw_loss, components = real.ft12.npt1.ensemble_train.point_loss(
                real.ft12.npt1.joint_real.real_loss_args(),
                prediction2, target2_norm, self.normalizer,
                real.ft12.npt1.joint_real.REAL_ACCURACY_WEIGHTS, iteration)
            weighted_loss = self.outer_weight * raw_loss
            if not torch.isfinite(weighted_loss):
                raise FloatingPointError("rollout exposure loss is non-finite")
            weighted_loss.backward()

        return {
            "rollout_exposure_raw_loss": float(raw_loss.detach()),
            "rollout_exposure_weighted_loss": float(weighted_loss.detach()),
            "rollout_exposure_first_step_mse": float(first_step_mse),
            "rollout_exposure_generated_fraction": self.generated_fraction,
            "rollout_exposure_outer_weight": self.outer_weight,
            **{
                f"rollout_exposure_{key}": float(value)
                for key, value in components.items()
            },
        }

    def report(self):
        payload = json.dumps(
            self.rows, sort_keys=True, separators=(",", ":")).encode()
        return {
            "updates": len(self.rows),
            "windows": len(self.dataset),
            "window_stride": ROLLOUT_STRIDE,
            "batch_size": ROLLOUT_BATCH_SIZE,
            "generated_fraction": self.generated_fraction,
            "outer_weight": self.outer_weight,
            "pressure_exact_zero": True,
            "first_prediction_detached": True,
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


def bind_rollout_trainer(trainer):
    def train(model, dataset, schedule, normalizer, device, deadline,
              output_dir, start_report, access, **kwargs):
        callback = None
        if output_dir.name == "candidate":
            rollout_dataset = RolloutDataset(
                dataset.trajectories,
                build_rollout_windows(dataset.trajectories))
            callback = RolloutExposureBackward(
                rollout_dataset, normalizer, device,
                standard24.self_describing.real.small4k.UPDATES)
            kwargs["extra_backward"] = callback
            kwargs["extra_backward_contract"] = {
                "kind": "prefix_only_half_autoregressive_rollout_exposure",
                "generated_fraction": GENERATED_FRACTION,
                "outer_weight": ROLLOUT_OUTER_WEIGHT,
                "first_prediction_detached": True,
                "inference_graph_changed": False,
            }
        elif output_dir.name != "control":
            raise ValueError(f"cannot infer rollout arm from {output_dir}")
        result = trainer(
            model, dataset, schedule, normalizer, device, deadline,
            output_dir, start_report, access, **kwargs)
        training, _seal = result
        training["rollout_exposure"] = (
            None if callback is None else callback.report())
        return result
    return train


def main() -> None:
    real = standard24.self_describing.real
    real.EXPERIMENT = "b210_standard24k_real_rollout_exposure"
    real.SCHEMA = "b210_standard24k_real_rollout_exposure_v1"
    real.architecture_contract = (
        standard24.self_describing.self_describing_architecture_contract)
    real.ft12.validate_start_checkpoint = standard24.bind_standard24_validator(
        real.ft12.validate_start_checkpoint)
    real.arm_window_contract = identical_dense_windows
    real.arm_augmentation_contract = identical_augmentation
    real.small4k.train_candidate = bind_rollout_trainer(
        real.small4k.train_candidate)
    real.main()


if __name__ == "__main__":
    main()
