#!/usr/bin/env python3
"""Compare historical and endpoint-aligned cosine schedules on Real4k.

Both arms use the same standard24 checkpoint, stride-5 clean81 prefix windows,
gain/noise/blur augmentation, sample/bootstrap stream and losses.  The control
retains the historical 12k cosine horizon; the candidate completes one cosine
cycle at the fixed u4000 endpoint.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (REPO_ROOT, PRIVATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_standard24k_real_dense_aug as standard24  # noqa: E402


CONTROL_T_MAX = 12_000
CANDIDATE_T_MAX = 4_000


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


def scheduler_for_output(output_dir: Path) -> int:
    if output_dir.name == "control":
        return CONTROL_T_MAX
    if output_dir.name == "candidate":
        return CANDIDATE_T_MAX
    raise ValueError(f"cannot infer paired arm from {output_dir}")


def bind_schedule_trainer(trainer):
    def train(model, dataset, schedule, normalizer, device, deadline,
              output_dir, start_report, access, **kwargs):
        if "scheduler_t_max" in kwargs:
            raise ValueError("scheduler_t_max is owned by the cosine4k runner")
        horizon = scheduler_for_output(output_dir)
        result = trainer(
            model, dataset, schedule, normalizer, device, deadline,
            output_dir, start_report, access,
            scheduler_t_max=horizon, **kwargs)
        training, _seal = result
        if int(training["scheduler_t_max"]) != horizon:
            raise RuntimeError("recorded cosine horizon differs from the arm")
        return result
    return train


def main() -> None:
    real = standard24.self_describing.real
    real.EXPERIMENT = "b210_standard24k_real_cosine4k"
    real.SCHEMA = "b210_standard24k_real_cosine4k_v1"
    real.architecture_contract = (
        standard24.self_describing.self_describing_architecture_contract)
    real.ft12.validate_start_checkpoint = standard24.bind_standard24_validator(
        real.ft12.validate_start_checkpoint)
    real.arm_window_contract = identical_dense_windows
    real.arm_augmentation_contract = identical_augmentation
    real.small4k.train_candidate = bind_schedule_trainer(
        real.small4k.train_candidate)
    real.main()


if __name__ == "__main__":
    main()
