from __future__ import annotations

import pytest

import train_b210_tke_causal_xcb_pretrained_standard_sim as stage1
import train_b210_tke_causal_xcb_standard20k_sim as stage2
import train_b210_tke_causal_xcb_standard24k_sim as stage3


def valid_pretext_payload() -> dict:
    return {
        "phase": "sim_real_prefix_causal_pretrain",
        "iteration": 4_000,
        "model_state_dict": {},
        "model_config": {
            "cross_conditioned_backbone": True,
            "member_adapter_channels": 160,
        },
        "training_config": {
            "method": stage1.PRETEXT_METHOD,
            "tke_aware_xcb_member": True,
            "tke_gradient_calibration": {
                "calibrated_gradient_ratio": 0.20,
            },
        },
        "pretraining_audit": {
            "changed_untouched_state_tensors": [],
            "initial_non_backbone_sha256": "a" * 64,
            "final_non_backbone_sha256": "a" * 64,
        },
    }


def standard_payload(method: str) -> dict:
    return {
        "iteration": 10_000,
        "model_state_dict": {},
        "training_config": {
            "method": method,
            "window_stride": 20,
            "cross_conditioned_backbone": True,
            "standard_objective_member_feature_adapters": True,
            "legacy_tke_gradient_floor": False,
        },
    }


def test_pretext_requires_tke_calibrated_xcb_member_identity():
    stage1.validate_pretrain_payload(valid_pretext_payload())
    payload = valid_pretext_payload()
    payload["training_config"]["tke_gradient_calibration"][
        "calibrated_gradient_ratio"] = 0.1
    with pytest.raises(RuntimeError, match="calibration"):
        stage1.validate_pretrain_payload(payload)


def test_stage2_accepts_only_tke_causal_stage1():
    stage2.validate_stage1_payload(standard_payload(stage2.SOURCE_METHOD))
    payload = standard_payload("unrelated")
    with pytest.raises(RuntimeError, match="contract"):
        stage2.validate_stage1_payload(payload)


def test_stage3_accepts_only_tke_causal_stage2():
    stage3.validate_stage2_payload(standard_payload(stage3.SOURCE_METHOD))
    payload = standard_payload(stage3.SOURCE_METHOD)
    payload["training_config"]["window_stride"] = 5
    with pytest.raises(RuntimeError, match="contract"):
        stage3.validate_stage2_payload(payload)
