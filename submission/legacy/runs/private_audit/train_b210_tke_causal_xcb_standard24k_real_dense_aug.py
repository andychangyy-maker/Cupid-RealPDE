#!/usr/bin/env python3
"""Run paired Real4k from the TKE-aware causal XCB/member Sim24k endpoint."""

from __future__ import annotations

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
EXPECTED_STATUS = "PASS_B210_TKE_CAUSAL_XCB_MEMBER_SIM24K"


def bind_validator(validator):
    return partial(
        validator,
        expected_iteration=EXPECTED_ITERATION,
        expected_status=EXPECTED_STATUS)


def main() -> None:
    real = self_describing.real
    real.EXPERIMENT = "b210_tke_causal_xcb_member_standard24k_real_dense_aug"
    real.SCHEMA = "b210_tke_causal_xcb_member_standard24k_real_dense_aug_v1"
    real.architecture_contract = (
        self_describing.self_describing_architecture_contract)
    real.ft12.validate_start_checkpoint = bind_validator(
        real.ft12.validate_start_checkpoint)
    real.main()


if __name__ == "__main__":
    main()
