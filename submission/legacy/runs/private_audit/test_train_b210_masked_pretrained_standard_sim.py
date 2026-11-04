from __future__ import annotations

import pytest

import train_b210_masked_pretrained_standard_sim as stage1
import train_b210_masked_standard_continuation_sim as stage2


def valid_pretext_payload():
    return {
        "phase": "sim_selfsupervised_pretrain",
        "iteration": 4_000,
        "model_state_dict": {},
        "training_config": {"method": stage1.PRETEXT_METHOD},
        "pretraining_audit": {
            "changed_untouched_state_tensors": [],
            "initial_non_backbone_sha256": "a" * 64,
            "final_non_backbone_sha256": "a" * 64,
        },
    }


def test_stage1_accepts_only_identity_preserving_pretext():
    stage1.validate_pretrain_payload(valid_pretext_payload())
    payload = valid_pretext_payload()
    payload["pretraining_audit"]["final_non_backbone_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="non-backbone"):
        stage1.validate_pretrain_payload(payload)


def test_stage2_accepts_only_masked_stage_one():
    payload = {
        "iteration": 10_000,
        "model_state_dict": {},
        "training_config": {"method": stage2.STAGE1_METHOD},
    }
    stage2.validate_stage1(payload)
    payload["training_config"]["method"] = "unrelated"
    with pytest.raises(RuntimeError, match="not the masked"):
        stage2.validate_stage1(payload)
