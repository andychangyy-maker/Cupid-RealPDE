from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_dual_decoder_standard10k_sim.py"
SPEC = importlib.util.spec_from_file_location("dual_decoder_sim", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_extract_mode_removes_private_argument():
    values = ["runner", "--dual-decoder-mode", "mean_fluctuation", "--device", "cpu"]
    assert MODULE.extract_mode(values) == "mean_fluctuation"
    assert values == ["runner", "--device", "cpu"]


def test_dual_modes_have_identical_parameter_contract_and_distinct_routing():
    original_config = MODULE.b210.b210_config
    try:
        MODULE.install("shared_fused")
        fused = MODULE.b210.b210_config(
            "candidate", cross_conditioned_backbone=True,
            member_feature_adapters=True)
    finally:
        MODULE.b210.b210_config = original_config
    MODULE.install("mean_fluctuation")
    split = MODULE.b210.b210_config(
        "candidate", cross_conditioned_backbone=True,
        member_feature_adapters=True)
    assert fused.dual_decoder_mode == "shared_fused"
    assert split.dual_decoder_mode == "mean_fluctuation"


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        MODULE.extract_mode(["runner", "--dual-decoder-mode", "wide"])


def test_asymmetric_mode_has_fused_mean_and_exclusive_fluctuation():
    original_config = MODULE.b210.b210_config
    try:
        MODULE.install("fused_mean_fluctuation")
        config = MODULE.b210.b210_config(
            "candidate", cross_conditioned_backbone=True,
            member_feature_adapters=True)
        assert config.dual_decoder_mode == "fused_mean_fluctuation"
    finally:
        MODULE.b210.b210_config = original_config
