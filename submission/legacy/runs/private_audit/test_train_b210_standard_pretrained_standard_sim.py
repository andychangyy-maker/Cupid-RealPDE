from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_standard_pretrained_standard_sim.py"
SPEC = importlib.util.spec_from_file_location("standard_continuation", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def payload(*, stride: int = 20, method: str | None = None) -> dict:
    return {
        "iteration": 10_000,
        "model_state_dict": {},
        "training_config": {
            "window_stride": stride,
            "method": method or
            "b210_np_functional_sim_realclosure_seed1_soup_member_v1",
        },
    }


def test_accepts_only_frozen_standard_seed1_endpoint():
    MODULE.validate_standard_pretrain_payload(payload())
    with pytest.raises(RuntimeError, match="stride 20"):
        MODULE.validate_standard_pretrain_payload(payload(stride=10))
    with pytest.raises(RuntimeError, match="frozen standard"):
        MODULE.validate_standard_pretrain_payload(payload(method="another_method"))


def test_requires_u10000_and_model_state():
    bad = payload()
    bad["iteration"] = 9_999
    with pytest.raises(RuntimeError, match="u10000"):
        MODULE.validate_standard_pretrain_payload(bad)
    with pytest.raises(TypeError, match="model_state_dict"):
        MODULE.validate_standard_pretrain_payload({})
