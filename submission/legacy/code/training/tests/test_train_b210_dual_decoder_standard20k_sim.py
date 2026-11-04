from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "runs" / "private_audit"
    / "train_b210_dual_decoder_standard20k_sim.py")
SPEC = importlib.util.spec_from_file_location("dual_decoder_20k", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def payload(mode: str) -> dict:
    return {
        "iteration": 10_000,
        "model_state_dict": {},
        "model_config": {"dual_decoder_mode": mode},
        "training_config": {
            "method": MODULE.CONTRACTS[mode]["source"],
            "window_stride": 20,
            "cross_conditioned_backbone": True,
            "standard_objective_member_feature_adapters": True,
            "legacy_tke_gradient_floor": False,
        },
    }


@pytest.mark.parametrize("mode", tuple(MODULE.CONTRACTS))
def test_validator_accepts_only_matching_sealed_stage1(mode):
    MODULE.validate_stage1_payload(payload(mode), mode)
    wrong_mode = payload(mode)
    wrong_mode["model_config"]["dual_decoder_mode"] = "none"
    with pytest.raises(RuntimeError, match="mode changed"):
        MODULE.validate_stage1_payload(wrong_mode, mode)
    wrong_method = payload(mode)
    wrong_method["training_config"]["method"] = "other"
    with pytest.raises(RuntimeError, match="sealed stage1"):
        MODULE.validate_stage1_payload(wrong_method, mode)


def test_mode_is_removed_before_base_parser():
    argv = ["runner", "--output-dir", "out", "--dual-decoder-mode",
            "separate_point_uq_fluctuation_gates", "--device", "cuda:0"]
    assert MODULE.extract_mode(argv) == "separate_point_uq_fluctuation_gates"
    assert argv == [
        "runner", "--output-dir", "out", "--device", "cuda:0"]
