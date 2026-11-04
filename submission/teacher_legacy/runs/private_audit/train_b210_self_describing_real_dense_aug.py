#!/usr/bin/env python3
"""Run frozen paired Real4k for any self-describing ensemble U-Net checkpoint.

Unlike the legacy Real launcher, this wrapper does not reconstruct a B210
configuration from source constants.  It validates the serialized architecture
and exact state counts, then lets the unchanged Real4k runner perform strict
checkpoint loading.  This is needed for preregistered width changes such as
B240 while preserving the same Real data and optimization protocol.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_scratch_real_dense_aug as real  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)


def checkpoint_path(argv: list[str] | None = None) -> Path:
    tokens = sys.argv if argv is None else argv
    try:
        return Path(tokens[tokens.index("--start-sim") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("--start-sim is required before architecture audit") from error


def architecture_from_payload(payload: dict) -> tuple:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("Sim checkpoint has no model_state_dict")
    raw = dict(payload.get("model_config", {}))
    if raw.pop("architecture", None) != "ensemble_unet":
        raise ValueError("Sim checkpoint is not an ensemble U-Net")
    config = EnsembleUNet2dConfig(**raw)
    if config.depth != 4 or config.ensemble_members != 4:
        raise ValueError("self-describing Real wrapper only authorizes depth-4 four-member models")
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise TypeError("Sim checkpoint model_state_dict is empty")
    # A state_dict also contains registered buffers (normalizer statistics,
    # persistence constants and, for some dynamic branches, coordinates).
    # The frozen Real loader contracts model *parameters*, so counting every
    # serialized tensor makes otherwise valid self-describing checkpoints fail.
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    parameters = sum(value.numel() for value in model.parameters())
    controller_parameters = sum(
        value.numel() for value in model.energy_controller.parameters())
    if parameters <= 0 or controller_parameters <= 0:
        raise ValueError("checkpoint parameter contract is incomplete")
    return config, parameters, controller_parameters


def self_describing_architecture_contract(
        horizon_local_energy_controller: bool,
        multiscale_energy_controller: bool = False,
        ) -> tuple:
    if horizon_local_energy_controller or multiscale_energy_controller:
        raise ValueError("self-describing wrapper does not add controllers")
    payload = torch.load(
        checkpoint_path(), map_location="cpu", weights_only=False)
    return architecture_from_payload(payload)


def main() -> None:
    real.EXPERIMENT = "b210_self_describing_real_dense_aug"
    real.SCHEMA = "b210_self_describing_real_dense_aug_v1"
    real.architecture_contract = self_describing_architecture_contract
    real.main()


if __name__ == "__main__":
    main()
