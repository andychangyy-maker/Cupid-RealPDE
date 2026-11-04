#!/usr/bin/env python3
"""Run frozen paired Real4k from one exploratory dual-decoder Sim20k arm."""

from __future__ import annotations

from pathlib import Path
import sys

PRIVATE = Path(__file__).resolve().parent
if str(PRIVATE) not in sys.path:
    sys.path.insert(0, str(PRIVATE))

import train_b210_dual_decoder_standard24k_real_dense_aug as base


base.EXPECTED_ITERATION = 10_000
base.CONTRACTS = {
    "shared_fused": {
        "status": "PASS_B210_DUALDECODER_SHAREDFUSED_STANDARD20K_SIM",
        "experiment": "b210_dualdecoder_sharedfused_standard20k_real_dense_aug",
        "schema": "b210_dualdecoder_sharedfused_standard20k_real_dense_aug_v1",
        "parameters": 196_316_169,
    },
    "separate_point_uq_fluctuation_gates": {
        "status": (
            "PASS_B210_DUALDECODER_SEPARATEPOINTUQGATE_STANDARD20K_SIM"),
        "experiment": (
            "b210_dualdecoder_separatepointuqgate_standard20k_real_dense_aug"),
        "schema": (
            "b210_dualdecoder_separatepointuqgate_standard20k_real_dense_aug_v1"),
        "parameters": 196_319_109,
    },
}


if __name__ == "__main__":
    base.main()
