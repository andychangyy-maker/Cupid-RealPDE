#!/usr/bin/env python3
"""Gate the exploratory matched Real4k pair selected at dual-decoder Sim20k."""

from __future__ import annotations

from pathlib import Path
import sys

PRIVATE = Path(__file__).resolve().parent
if str(PRIVATE) not in sys.path:
    sys.path.insert(0, str(PRIVATE))

import select_b210_dual_decoder_standard24k_real4k as base


base.CONTRACTS = {
    "control": {
        "experiment": "b210_dualdecoder_sharedfused_standard20k_real_dense_aug",
        "schema": "b210_dualdecoder_sharedfused_standard20k_real_dense_aug_v1",
        "parameters": 196_316_169,
    },
    "candidate": {
        "experiment": (
            "b210_dualdecoder_separatepointuqgate_standard20k_real_dense_aug"),
        "schema": (
            "b210_dualdecoder_separatepointuqgate_standard20k_real_dense_aug_v1"),
        "parameters": 196_319_109,
    },
}
base.SELECTION_SCHEMA = "b210_dual_decoder_standard20k_real4k_selection_v1"
base.SELECTION_EVENT = "b210_dual_decoder_standard20k_real4k_selection"


if __name__ == "__main__":
    base.main()
