#!/usr/bin/env python3
"""Run the frozen standard stride-20 Stage 1 recipe with base width 240."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
        REPO_ROOT / "code", REPO_ROOT / "code" / "training",
        REPO_ROOT / "runs" / "private_audit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_sim as b210  # noqa: E402
from b210_wide_base_contract import install_wide_base  # noqa: E402


METHOD = "b240_standard_stride20_stage1_v1"


def main() -> None:
    install_wide_base(b210)
    b210.REAL_CLOSURE_SEED1_METHOD = METHOD
    b210.main()


if __name__ == "__main__":
    main()
