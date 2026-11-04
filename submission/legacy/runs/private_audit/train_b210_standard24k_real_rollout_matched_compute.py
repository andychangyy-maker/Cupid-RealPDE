#!/usr/bin/env python3
"""Isolate rollout-history semantics from extra Real supervision compute."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (REPO_ROOT, PRIVATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_standard24k_real_rollout_exposure as rollout  # noqa: E402


CONTROL_GENERATED_FRACTION = 0.0
CANDIDATE_GENERATED_FRACTION = rollout.GENERATED_FRACTION


def bind_matched_compute_trainer(trainer):
    def train(model, dataset, schedule, normalizer, device, deadline,
              output_dir, start_report, access, **kwargs):
        fractions = {
            "control": CONTROL_GENERATED_FRACTION,
            "candidate": CANDIDATE_GENERATED_FRACTION,
        }
        if output_dir.name not in fractions:
            raise ValueError(f"cannot infer rollout arm from {output_dir}")
        rollout_dataset = rollout.RolloutDataset(
            dataset.trajectories,
            rollout.build_rollout_windows(dataset.trajectories))
        callback = rollout.RolloutExposureBackward(
            rollout_dataset, normalizer, device,
            rollout.standard24.self_describing.real.small4k.UPDATES,
            generated_fraction=fractions[output_dir.name],
            outer_weight=rollout.ROLLOUT_OUTER_WEIGHT)
        kwargs["extra_backward"] = callback
        kwargs["extra_backward_contract"] = {
            "kind": "matched_compute_prefix_rollout_semantics",
            "generated_fraction": fractions[output_dir.name],
            "outer_weight": rollout.ROLLOUT_OUTER_WEIGHT,
            "first_prediction_detached": True,
            "extra_batch_and_forward_count_matched": True,
            "inference_graph_changed": False,
        }
        result = trainer(
            model, dataset, schedule, normalizer, device, deadline,
            output_dir, start_report, access, **kwargs)
        training, _seal = result
        training["rollout_exposure"] = callback.report()
        return result
    return train


def main() -> None:
    standard24 = rollout.standard24
    real = standard24.self_describing.real
    real.EXPERIMENT = "b210_standard24k_real_rollout_matched_compute"
    real.SCHEMA = "b210_standard24k_real_rollout_matched_compute_v1"
    real.architecture_contract = (
        standard24.self_describing.self_describing_architecture_contract)
    real.ft12.validate_start_checkpoint = standard24.bind_standard24_validator(
        real.ft12.validate_start_checkpoint)
    real.arm_window_contract = rollout.identical_dense_windows
    real.arm_augmentation_contract = rollout.identical_augmentation
    real.small4k.train_candidate = bind_matched_compute_trainer(
        real.small4k.train_candidate)
    real.main()


if __name__ == "__main__":
    main()
