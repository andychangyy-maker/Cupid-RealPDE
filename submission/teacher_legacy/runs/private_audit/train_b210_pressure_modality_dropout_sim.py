#!/usr/bin/env python3
"""Train B210 with deterministic Sim pressure-modality dropout.

Released Sim histories contain u/v/p, while released Real PIV contains only
u/v and the common loader pads p with exact zeros.  This launcher makes half
of candidate Sim histories match that missing-modality contract.  It changes
training inputs only and leaves the model/deployment graph untouched.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_sim as b210  # noqa: E402


METHOD = "b210_realclosure_pressure_modality_dropout_p050_v1"
PROBABILITY = 0.50
MODULUS = 1_000_003
THRESHOLD = int(round(PROBABILITY * MODULUS))
_ORIGINAL_SIM_UPDATE = b210.sim_update


def pressure_dropout_mask(
        sample_indices: torch.Tensor, iteration: int,
        seed: int = b210.SEED) -> torch.Tensor:
    """Deterministic, sample/update-local selection without RNG consumption."""
    indices = sample_indices.detach().to(dtype=torch.int64, device="cpu")
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("pressure dropout sample indices must be one vector")
    if iteration <= 0:
        raise ValueError("pressure dropout iteration must be positive")
    codes = (
        (indices + 1) * 7919
        + int(iteration) * 104729
        + (int(seed) + 1) * 15485863
    ) % MODULUS
    return codes < THRESHOLD


def pressure_dropout_sim_update(*args, **kwargs):
    if len(args) < 12:
        raise TypeError("B210 sim_update positional contract changed")
    batch = args[1]
    normalizer = args[2]
    iteration = int(args[8])
    arm = str(args[11])
    selected = torch.zeros(batch[0].shape[0], dtype=torch.bool)
    if arm == "candidate":
        history, target, trajectory_ids, sample_indices = batch
        if history.ndim != 5 or history.shape[-1] != 3:
            raise ValueError("pressure dropout requires [B,T,H,W,3] Sim history")
        selected = pressure_dropout_mask(sample_indices, iteration)
        if bool(selected.any()):
            history = history.clone()
            # Base sim_update normalizes after closure augmentation.  Setting
            # physical p to the Sim mean therefore gives exact normalized p=0,
            # matching the zero-filled Real channel after Real normalization.
            pressure_mean = normalizer.mean_input[2].detach().to(
                dtype=history.dtype, device="cpu")
            history[selected, ..., 2] = pressure_mean
            batch = (history, target, trajectory_ids, sample_indices)
            positional = list(args)
            positional[1] = batch
            args = tuple(positional)
    values, audit, key, flags = _ORIGINAL_SIM_UPDATE(*args, **kwargs)
    values["pressure_modality_dropout_probability"] = PROBABILITY
    values["pressure_modality_dropout_fraction"] = float(
        selected.float().mean()) if arm == "candidate" else 0.0
    values["pressure_modality_dropout_normalized_value"] = 0.0
    return values, audit, key, flags


def main() -> None:
    b210.sim_update = pressure_dropout_sim_update
    b210.REAL_CLOSURE_SEED1_METHOD = METHOD
    b210.main()


if __name__ == "__main__":
    main()
