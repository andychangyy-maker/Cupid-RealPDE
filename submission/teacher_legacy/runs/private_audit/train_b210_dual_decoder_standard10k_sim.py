#!/usr/bin/env python3
"""Run one controlled dual-decoder B210 arm from random initialization.

The existing B210 runner still supplies its paired update-0/health control and
the complete Standard10k objective. This wrapper changes only serialized
architecture routing. The three discrete modes instantiate exactly the same
state tensors; continuous modes add 1,470 gates per responsibility while
preserving all common tensors and update-zero outputs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_sim as b210  # noqa: E402


EXPECTED_PARAMETERS = {
    "shared_fused": 196_316_169,
    "mean_fluctuation": 196_316_169,
    "fused_mean_fluctuation": 196_316_169,
    # Three per-channel continuous gates: 840 + 420 + 210.
    "learned_fluctuation_gate": 196_317_639,
    # Independent point and UQ gates over the same three scales.
    "separate_point_uq_fluctuation_gates": 196_319_109,
}
METHODS = {
    "shared_fused": (
        "b210_xcb_member_dualdecoder_sharedfused_standard10k_v1",
        "PASS_B210_DUALDECODER_SHAREDFUSED_STANDARD10K_SIM"),
    "mean_fluctuation": (
        "b210_xcb_member_dualdecoder_meanfluct_standard10k_v1",
        "PASS_B210_DUALDECODER_MEANFLUCT_STANDARD10K_SIM"),
    "fused_mean_fluctuation": (
        "b210_xcb_member_dualdecoder_fusedmean_fluct_standard10k_v2",
        "PASS_B210_DUALDECODER_FUSEDMEAN_FLUCT_STANDARD10K_SIM"),
    "learned_fluctuation_gate": (
        "b210_xcb_member_dualdecoder_learnedfluctgate_standard10k_v3",
        "PASS_B210_DUALDECODER_LEARNEDFLUCTGATE_STANDARD10K_SIM"),
    "separate_point_uq_fluctuation_gates": (
        "b210_xcb_member_dualdecoder_separatepointuqgate_standard10k_v4",
        "PASS_B210_DUALDECODER_SEPARATEPOINTUQGATE_STANDARD10K_SIM"),
}


def extract_mode(argv: list[str]) -> str:
    try:
        index = argv.index("--dual-decoder-mode")
        mode = argv[index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(
            "--dual-decoder-mode shared_fused|mean_fluctuation|"
            "fused_mean_fluctuation|learned_fluctuation_gate|"
            "separate_point_uq_fluctuation_gates is required"
        ) from error
    if mode not in METHODS:
        raise ValueError("unsupported dual decoder mode")
    del argv[index:index + 2]
    return mode


def install(mode: str) -> None:
    method, status = METHODS[mode]
    original_config = b210.b210_config
    original_training_args = b210.training_args

    def dual_config(*args, **kwargs):
        return replace(
            original_config(*args, **kwargs), dual_decoder_mode=mode)

    def dual_training_args(args):
        values = original_training_args(args)
        values.dual_decoder_mode = mode
        values.dual_decoder_equal_capacity = (
            mode not in (
                "learned_fluctuation_gate",
                "separate_point_uq_fluctuation_gates"))
        values.dual_decoder_extra_gate_parameters = (
            2_940 if mode == "separate_point_uq_fluctuation_gates"
            else 1_470 if mode == "learned_fluctuation_gate" else 0)
        values.dual_decoder_role_contract = (
            "shared_feature_capacity_control" if mode == "shared_fused"
            else "separate_learned_point_and_uq_fluctuation_responsibility"
            if mode == "separate_point_uq_fluctuation_gates"
            else "learned_continuous_fluctuation_responsibility"
            if mode == "learned_fluctuation_gate"
            else "fused_mean_vs_exclusive_zero_mean_fluctuation"
            if mode == "fused_mean_fluctuation"
            else "mean_advection_vs_zero_mean_fluctuation")
        return values

    b210.b210_config = dual_config
    b210.training_args = dual_training_args
    b210.EXPECTED_CROSS_CONDITIONED_MEMBER_ADAPTER_PARAMETERS = (
        EXPECTED_PARAMETERS[mode])
    b210.STANDARD_CROSS_CONDITIONED_MEMBER_ADAPTER_METHOD = method
    b210.PASS_STATUS = status


def main() -> None:
    mode = extract_mode(sys.argv)
    install(mode)
    b210.main()


if __name__ == "__main__":
    main()
