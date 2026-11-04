from pathlib import Path
import sys

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_msclosure_real_paired4k as subject  # noqa: E402


def test_paired_real_contract_changes_only_support_and_augmentation():
    control = subject.real_contract(
        arm=subject.CONTROL_ARM, windows=2544, stride=20, augmentation=False)
    candidate = subject.real_contract(
        arm=subject.CANDIDATE_ARM, windows=9933, stride=5, augmentation=True)
    for key in ("updates", "batch_size", "seed", "optimizer", "lr", "weight_decay",
                "scheduler", "scheduler_t_max", "grad_clip", "ema_decay",
                "mse_warmup_updates", "accuracy_weights", "mse_weight", "loss"):
        assert control[key] == candidate[key]
    assert control["augmentation"] == {"enabled": False}
    assert candidate["augmentation"] == subject.real.AUGMENTATION


def test_ms_checkpoint_metadata_rejects_wrong_schema_before_state_use():
    payload = {
        "schema": "wrong", "endpoint_kind": "ema999_u12000", "updates": 12000,
        "rollout_mode": "carry_detached", "segment_gradient_mode": "connected",
        "model_config": subject.expected_model_config(), "normalizer": {}, "git": {},
        "model_state": {},
    }
    with pytest.raises(ValueError, match="dense-connected"):
        subject.validate_ms_checkpoint_payload(payload)


def test_ms_checkpoint_metadata_rejects_non_connected_semantics_before_state_use():
    payload = {
        "schema": subject.EXPECTED_SIM_SCHEMA, "endpoint_kind": "ema999_u12000", "updates": 12000,
        "rollout_mode": "carry_detached", "segment_gradient_mode": "truncated",
        "model_config": subject.expected_model_config(), "normalizer": {}, "git": {},
        "model_state": {},
    }
    with pytest.raises(ValueError, match="rollout semantics"):
        subject.validate_ms_checkpoint_payload(payload)


def test_paired_access_audit_forbids_tail_until_both_seals(tmp_path):
    audit = subject.PairedPrefixAccessAudit()
    row = {"file": "10125_0.h5"}
    with pytest.raises(RuntimeError, match="both paired endpoint"):
        audit.before_open(row, "clean716")
    paths = []
    for arm in (subject.CONTROL_ARM, subject.CANDIDATE_ARM):
        path = tmp_path / f"{arm}.pt"
        torch.save({"arm": arm}, path)
        paths.append(path)
        audit.record_endpoint_seal(arm, {
            "path": str(path), "file_sha256": subject.real.file_sha256(path),
        })
    audit.before_open(row, "clean716")
    audit.record_open(row, "clean716", 100)
    assert audit.events[0]["both_endpoints_sealed_before_open"] is True


def test_metric_comparison_uses_public_point_key_only():
    left = {"rel_l2_score": 94.0, "tke_rel_l2_score": 78.0,
            "mvpe_rel_l2_score": 93.0, "point_contribution_46_10_9": 59.0}
    right = {"rel_l2_score": 93.5, "tke_rel_l2_score": 77.0,
             "mvpe_rel_l2_score": 92.0, "point_contribution_46_10_9": 58.5}
    assert subject.compare_metrics(left, right) == {
        "rel_l2_score": .5, "tke_rel_l2_score": 1., "mvpe_rel_l2_score": 1.,
        "point_contribution_46_10_9": .5}


def test_clean716_manifest_identity_is_frozen():
    assert subject.EXPECTED_CLEAN716_MANIFEST_SHA256 == (
        "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958")
