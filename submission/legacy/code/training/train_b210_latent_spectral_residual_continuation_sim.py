#!/usr/bin/env python3
"""Matched-compute Stage 2 for the B210 latent-spectral architecture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

import train_b210_latent_spectral_residual_sim as spectral
import train_b210_np_functional_sim as sim


METHOD = "b210_latent_spectral_residual_r64_m2x4_standard20k_seed1_v1"
STAGE1_METHOD = spectral.METHOD
PRETRAIN_PATH: Path | None = None
PRETRAIN_SHA256: str | None = None
PRETRAIN_PAYLOAD: dict | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pop_pretrain_args(argv: list[str]) -> tuple[list[str], Path, str]:
    remaining = [argv[0]]
    checkpoint = None
    expected_sha = None
    index = 1
    while index < len(argv):
        token = argv[index]
        if token in {"--pretrain-checkpoint", "--pretrain-checkpoint-sha256"}:
            if index + 1 >= len(argv):
                raise ValueError(f"{token} requires one value")
            value = argv[index + 1]
            if token == "--pretrain-checkpoint":
                checkpoint = Path(value)
            else:
                expected_sha = value.lower()
            index += 2
        else:
            remaining.append(token)
            index += 1
    if checkpoint is None or expected_sha is None:
        raise ValueError("both pretrain checkpoint arguments are required")
    if (len(expected_sha) != 64
            or any(value not in "0123456789abcdef" for value in expected_sha)):
        raise ValueError("pretrain checkpoint SHA-256 is invalid")
    return remaining, checkpoint, expected_sha


def validate_pretrain_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("pretrain checkpoint has no model_state_dict")
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("pretrain checkpoint is not a u10000 endpoint")
    training = payload.get("training_config")
    if not isinstance(training, dict):
        raise TypeError("pretrain checkpoint has no training_config")
    if int(training.get("window_stride", -1)) != 20:
        raise RuntimeError("spectral Stage 1 did not use stride 20")
    if str(training.get("method", "")) != STAGE1_METHOD:
        raise RuntimeError("checkpoint is not the frozen spectral Stage 1")
    config = payload.get("model_config")
    expected = {
        "architecture": "ensemble_unet",
        "base": 210,
        "depth": 4,
        "latent_spectral_rank": spectral.RANK,
        "latent_spectral_modes_y": spectral.MODES_Y,
        "latent_spectral_modes_x": spectral.MODES_X,
        "latent_spectral_residual_limit": spectral.RESIDUAL_LIMIT,
    }
    if not isinstance(config, dict) or any(
            config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("checkpoint model_config is not the frozen spectral arm")


def build_pretrained_arm(arm: str, *args, **kwargs):
    if PRETRAIN_PAYLOAD is None:
        raise RuntimeError("spectral pretrain payload was not initialized")
    model = spectral.build_with_contract(arm, *args, **kwargs)
    incompatible = model.load_state_dict(
        PRETRAIN_PAYLOAD["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"spectral pretrain state is incompatible: {incompatible}")
    # Stage-2 update audits must compare with the loaded endpoint, not the
    # throwaway seeded construction checked by build_with_contract.
    spectral.INITIAL_STATES[arm] = spectral.module_state(model)
    return model


def annotate_summary(output_dir: Path) -> None:
    path = output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["method"] = METHOD
    summary["evidence_class"] = (
        "matched-compute latent-spectral test; released-Sim standard u10000 "
        "Stage 1 followed by a restarted standard u10000 Stage 2")
    summary["pretraining"] = {
        "checkpoint": str(PRETRAIN_PATH),
        "checkpoint_sha256": PRETRAIN_SHA256,
        "stage1_method": STAGE1_METHOD,
        "stage1_window_stride": 20,
        "stage1_updates": 10_000,
        "stage2_window_stride": 20,
        "stage2_updates": 10_000,
        "optimizer_and_schedule_restarted_for_stage2": True,
        "external_data_or_model": False,
        "both_paired_arms_loaded_identically": True,
    }
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def install_contract() -> None:
    spectral.METHOD = METHOD
    sim.EXPECTED_PARAMETERS = spectral.EXPECTED_TOTAL_PARAMETERS
    sim.b210_config = spectral.spectral_config
    sim.build_arm_model = build_pretrained_arm
    sim.configure_trainable = spectral.configure_trainable
    sim.sim_update = spectral.sim_update_with_audit
    sim.run_experiment = spectral.run_with_contract
    sim.REAL_CLOSURE_SEED1_METHOD = METHOD


def main() -> None:
    global PRETRAIN_PATH, PRETRAIN_SHA256, PRETRAIN_PAYLOAD
    remaining, checkpoint, expected_sha = pop_pretrain_args(sys.argv)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_sha = sha256(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"pretrain checkpoint SHA mismatch: {actual_sha} != {expected_sha}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_pretrain_payload(payload)
    PRETRAIN_PATH = checkpoint.resolve()
    PRETRAIN_SHA256 = actual_sha
    PRETRAIN_PAYLOAD = payload
    sys.argv[:] = remaining
    install_contract()
    sim.main()

    output_dir = None
    for index, token in enumerate(remaining[:-1]):
        if token == "--output-dir":
            output_dir = Path(remaining[index + 1])
            break
    if output_dir is None:
        raise RuntimeError("base output directory was not found after training")
    annotate_summary(output_dir)


if __name__ == "__main__":
    main()
