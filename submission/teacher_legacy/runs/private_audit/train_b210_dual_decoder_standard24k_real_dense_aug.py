#!/usr/bin/env python3
"""Run the frozen paired Real4k protocol from one dual-decoder Sim24k arm."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE = REPO_ROOT / "runs" / "private_audit"
for path in (REPO_ROOT, PRIVATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_self_describing_real_dense_aug as self_describing  # noqa: E402


EXPECTED_ITERATION = 4_000
CONTRACTS = {
    "shared_fused": {
        "status": "PASS_B210_DUALDECODER_SHAREDFUSED_STANDARD24K_SIM",
        "experiment": "b210_dualdecoder_sharedfused_standard24k_real_dense_aug",
        "schema": "b210_dualdecoder_sharedfused_standard24k_real_dense_aug_v1",
        "parameters": 196_316_169,
    },
    "separate_point_uq_fluctuation_gates": {
        "status": (
            "PASS_B210_DUALDECODER_SEPARATEPOINTUQGATE_STANDARD24K_SIM"),
        "experiment": (
            "b210_dualdecoder_separatepointuqgate_standard24k_real_dense_aug"),
        "schema": (
            "b210_dualdecoder_separatepointuqgate_standard24k_real_dense_aug_v1"),
        "parameters": 196_319_109,
    },
}


def extract_mode(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dual-decoder-mode", choices=tuple(CONTRACTS))
    known, remaining = parser.parse_known_args(argv[1:])
    if known.dual_decoder_mode is None:
        raise ValueError("--dual-decoder-mode is required")
    argv[:] = [argv[0], *remaining]
    return known.dual_decoder_mode


def bind_validator(validator, mode: str):
    return partial(
        validator,
        expected_iteration=EXPECTED_ITERATION,
        expected_status=CONTRACTS[mode]["status"])


def bind_architecture_contract(contract, mode: str):
    def architecture_contract(
            horizon_local_energy_controller: bool,
            multiscale_energy_controller: bool = False,
            ) -> tuple:
        config, parameters, controller_parameters = contract(
            horizon_local_energy_controller,
            multiscale_energy_controller)
        if config.dual_decoder_mode != mode:
            raise RuntimeError("serialized dual-decoder mode differs from CLI")
        expected = int(CONTRACTS[mode]["parameters"])
        if parameters != expected:
            raise RuntimeError(
                f"dual-decoder parameter contract changed: {parameters} != {expected}")
        return config, parameters, controller_parameters
    return architecture_contract


def install(mode: str) -> None:
    contract = CONTRACTS[mode]
    real = self_describing.real
    real.EXPERIMENT = contract["experiment"]
    real.SCHEMA = contract["schema"]
    real.architecture_contract = bind_architecture_contract(
        self_describing.self_describing_architecture_contract, mode)
    real.ft12.validate_start_checkpoint = bind_validator(
        real.ft12.validate_start_checkpoint, mode)


def main() -> None:
    mode = extract_mode(sys.argv)
    install(mode)
    self_describing.real.main()


if __name__ == "__main__":
    main()
