from pathlib import Path
import json
import sys

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_msclosure_combined_start_real4k as subject  # noqa: E402


def test_real_contract_and_schedule_are_exactly_the_frozen_reference_recipe():
    contract = subject.mature_candidate_contract()
    assert contract == subject.paired.real_contract(
        arm=subject.paired.CANDIDATE_ARM,
        windows=subject.paired.real.EXPECTED_OPTIMIZER_WINDOWS,
        stride=subject.paired.real.OPTIMIZER_STRIDE,
        augmentation=True,
    )
    assert contract["updates"] == 4_000
    assert contract["scheduler_t_max"] == 12_000
    assert contract["optimizer_windows"] == 9_933
    assert contract["optimizer_stride"] == 5
    assert contract["augmentation"] == subject.paired.real.AUGMENTATION
    assert subject.expected_schedule_sha256() == subject.REAL_SAMPLE_SCHEDULE_SHA256


def test_combined_start_rejects_a_connected_only_schema_before_state_use():
    payload = {
        "schema": subject.paired.EXPECTED_SIM_SCHEMA,
        "endpoint_kind": "ema999_u12000",
        "updates": 12_000,
        "rollout_mode": "carry_detached",
        "segment_gradient_mode": "connected",
        "model_config": subject.paired.expected_model_config(),
        "normalizer": {},
        "training_window_manifest": {},
        "shifted_teacher": {},
        "git": {},
        "model_state": {},
    }
    with pytest.raises(ValueError, match="not the combined"):
        subject.validate_combined_sim_payload(payload)


def test_reference_rejects_any_real_recipe_change_before_state_use():
    contract = subject.mature_candidate_contract()
    contract["scheduler_t_max"] = 4_000
    payload = {
        "schema": subject.paired.ENDPOINT_SCHEMA,
        "arm": subject.paired.CANDIDATE_ARM,
        "update": 4_000,
        "model_config": subject.paired.expected_model_config(),
        "rollout_mode": "carry_detached",
        "segment_gradient_mode": "connected",
        "model_state": {},
        "model_state_sha256": "",
        "normalizer": {},
        "normalizer_sha256": subject.REAL_NORMALIZER_SHA256,
        "start_ms": {},
        "contract": contract,
        "sample_schedule_sha256": subject.REAL_SAMPLE_SCHEDULE_SHA256,
        "ema": {},
    }
    with pytest.raises(RuntimeError, match="mature Real4k recipe"):
        subject.validate_reference_payload(payload)


def test_access_audit_opens_tail_only_after_both_candidate_seals(tmp_path):
    reference_path = tmp_path / "reference.pt"
    torch.save({"kind": "reference"}, reference_path)
    reference = {
        "path": str(reference_path),
        "file_sha256": subject.paired.real.file_sha256(reference_path),
    }
    audit = subject.CandidateAccessAudit(reference)
    row = {"file": "10125_0.h5"}
    with pytest.raises(RuntimeError, match="raw and EMA"):
        audit.before_open(row, "clean716")

    for name in (subject.CANDIDATE_RAW, subject.CANDIDATE_EMA):
        path = tmp_path / f"{name}.pt"
        torch.save({"name": name}, path)
        audit.record_candidate_seal(name, {
            "path": str(path),
            "file_sha256": subject.paired.real.file_sha256(path),
        })
    audit.before_open(row, "clean716")
    audit.record_open(row, "clean716", 100)
    assert audit.events[0]["candidate_raw_and_ema_sealed_before_open"] is True
    assert audit.events[0]["reference_file_verified_before_open"] is True


def test_causal_comparison_uses_only_the_public_point_components():
    candidate = {
        "rel_l2_score": 94.0,
        "tke_rel_l2_score": 78.0,
        "mvpe_rel_l2_score": 93.0,
        "point_contribution_46_10_9": 59.0,
    }
    reference = {
        "rel_l2_score": 93.5,
        "tke_rel_l2_score": 77.0,
        "mvpe_rel_l2_score": 92.0,
        "point_contribution_46_10_9": 58.5,
    }
    assert subject.paired.compare_metrics(candidate, reference) == {
        "rel_l2_score": .5,
        "tke_rel_l2_score": 1.,
        "mvpe_rel_l2_score": 1.,
        "point_contribution_46_10_9": .5,
    }


def test_all_external_artifact_and_manifest_identities_are_frozen():
    assert subject.COMBINED_SIM_FILE_SHA256 == (
        "741470587e7d9791399673dd7cf194755fc282341443c7c1c5a5f4479afa6d1e")
    assert subject.CONNECTED_REAL_REFERENCE_FILE_SHA256 == (
        "cf1fca2971ec8bfd5d0dca5db326febc331b473bbf4617f6f41a1a5dd71fa4fd")
    assert subject.paired.EXPECTED_CLEAN716_MANIFEST_SHA256 == (
        "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958")
    assert subject.REAL_FIT_MANIFEST_FILE_SHA256 == (
        "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")


def test_fit_manifest_file_contract_rejects_a_different_json_file(tmp_path):
    wrong = tmp_path / "split_manifest.json"
    wrong.write_text("[]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime_train_manifest"):
        subject.validate_fit_manifest_file(wrong)


def test_semantic_fit_manifest_contract_rejects_val_starts(tmp_path):
    wrong = tmp_path / "split_manifest.json"
    wrong.write_text(json.dumps([{
        "file": "10125_0.h5",
        "frames": 607,
        "re": 10142,
        "aoa": 0,
        "train_starts": [0],
        "val_starts": [460],
        "val_boundary": 460,
    }]), encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain val_starts"):
        subject.paired.real.load_clean_manifest(wrong)
