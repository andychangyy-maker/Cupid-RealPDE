#!/usr/bin/env python3
"""Balance released Real optimizer windows between aligned and offset phases."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (REPO_ROOT, PRIVATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_standard24k_real_dense_aug as standard24  # noqa: E402


PHASE_MODULUS = 20
ALIGNED_MULTIPLICITY = 3


def window_key(window) -> tuple[int, int]:
    return int(window.trajectory), int(window.start)


def phase_balanced_windows(trajectories, manifest_windows):
    """Keep every stride-5 window and repeat the released stride-20 lattice.

    The dense inventory has four temporal offset classes modulo 20.  Repeating
    the aligned class three times total makes aligned and off-lattice optimizer
    mass approximately equal while retaining every released prefix window.
    """
    real = standard24.self_describing.real
    dense = real.dense_windows(trajectories, real.CANDIDATE_STRIDE)
    aligned = [
        window for window in dense
        if int(window.start) % PHASE_MODULUS == 0
    ]
    if [window_key(window) for window in aligned] != [
            window_key(window) for window in manifest_windows]:
        raise RuntimeError(
            "aligned dense windows differ from the released train manifest")
    candidate = list(dense)
    for _ in range(ALIGNED_MULTIPLICITY - 1):
        candidate.extend(aligned)
    if len(candidate) != len(dense) + 2 * len(aligned):
        raise RuntimeError("phase-balanced multiplicity changed")
    return {
        "control": list(dense),
        "candidate": candidate,
    }


def identical_augmentation():
    real = standard24.self_describing.real
    return {
        "control": dict(real.AUGMENTATION),
        "candidate": dict(real.AUGMENTATION),
    }


def main() -> None:
    real = standard24.self_describing.real
    real.EXPERIMENT = "b210_standard24k_real_phase_balanced"
    real.SCHEMA = "b210_standard24k_real_phase_balanced_v1"
    real.architecture_contract = (
        standard24.self_describing.self_describing_architecture_contract)
    real.ft12.validate_start_checkpoint = standard24.bind_standard24_validator(
        real.ft12.validate_start_checkpoint)
    real.arm_window_contract = phase_balanced_windows
    real.arm_augmentation_contract = identical_augmentation
    real.main()


if __name__ == "__main__":
    main()
