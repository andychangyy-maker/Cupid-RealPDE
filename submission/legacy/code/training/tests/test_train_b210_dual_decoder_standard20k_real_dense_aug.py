from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "runs" / "private_audit"
    / "train_b210_dual_decoder_standard20k_real_dense_aug.py")
SPEC = importlib.util.spec_from_file_location("dual_decoder_20k_real", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_standard20k_wrapper_rebinds_iteration_status_and_names():
    assert MODULE.base.EXPECTED_ITERATION == 10_000
    for mode, contract in MODULE.base.CONTRACTS.items():
        assert "STANDARD20K_SIM" in contract["status"]
        assert "standard20k_real_dense_aug" in contract["experiment"]
        assert contract["schema"] == contract["experiment"] + "_v1"
        assert contract["parameters"] > 0
        assert mode in (
            "shared_fused", "separate_point_uq_fluctuation_gates")
