from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = (
    ROOT / "runs/private_audit"
    / "select_b210_dual_decoder_standard20k_real4k.py")
SPEC = importlib.util.spec_from_file_location("select_dual_20k_real", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_wrapper_selects_base_module_and_20k_provenance_labels():
    assert MODULE.base.SELECTION_SCHEMA == (
        "b210_dual_decoder_standard20k_real4k_selection_v1")
    assert MODULE.base.SELECTION_EVENT == (
        "b210_dual_decoder_standard20k_real4k_selection")
    assert MODULE.base.CONTRACTS["control"]["experiment"].endswith(
        "standard20k_real_dense_aug")
    assert MODULE.base.CONTRACTS["candidate"]["experiment"].endswith(
        "standard20k_real_dense_aug")
