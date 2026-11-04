#!/usr/bin/env python3
"""Run standard-to-standard Stage 2 with base width 240."""

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
import train_b210_standard_pretrained_standard_sim as continuation  # noqa: E402
from b210_wide_base_contract import install_wide_base  # noqa: E402


METHOD = "b240_standard_stride20_pretrain_then_standard_stride20_v1"
STAGE1_METHOD = "b240_standard_stride20_stage1_v1"


def validate_b240_standard_pretrain(payload: dict) -> None:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError("pretrain checkpoint has no model_state_dict")
    if int(payload.get("iteration", -1)) != 10_000:
        raise RuntimeError("pretrain checkpoint is not a u10000 endpoint")
    config = payload.get("training_config")
    if not isinstance(config, dict):
        raise TypeError("pretrain checkpoint has no training_config")
    if int(config.get("window_stride", -1)) != 20:
        raise RuntimeError("B240 Stage-1 checkpoint was not trained with stride 20")
    if str(config.get("method", "")) != STAGE1_METHOD:
        raise RuntimeError("checkpoint is not the frozen B240 standard Stage 1")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict) or int(model_config.get("base", -1)) != 240:
        raise RuntimeError("checkpoint model_config is not B240")


def main() -> None:
    install_wide_base(b210)
    continuation.METHOD = METHOD
    continuation.validate_standard_pretrain_payload = (
        validate_b240_standard_pretrain)
    continuation.main()


if __name__ == "__main__":
    main()
