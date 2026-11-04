from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "train_b210_history_coarse_flow_held10.py"
SPEC = importlib.util.spec_from_file_location("history_coarse_flow", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_frozen_structure_features_and_update0_identity():
    history = torch.randn(2, 20, 32, 64, 2)
    base = torch.randn_like(history)
    head = subject.CoarseFlowHead()
    raw = head(subject.make_features(history, base))
    assert subject.UPDATES == 2000 and subject.BATCH_SIZE == 64
    assert raw.shape == (2, 4, 2, 4, 8)
    assert torch.equal(subject.candidate_from_raw(base, raw), base)


def test_held10_split_is_exact_and_disjoint():
    aoa = np.asarray([0] * 150 + [5] * 150 + [10] * 162 + [15] * 127 + [20] * 127)
    fit, held = subject.fit_held_indices(aoa)
    assert len(fit) == 554 and len(held) == 162
    assert not np.intersect1d(fit, held).size
    assert set(aoa[fit]) == {0, 5, 15, 20}


def test_duplicate_manifest_identity_is_rejected_before_training():
    manifest = [{"file": "a.h5", "frames": 44, "re": 100, "aoa": 0,
                 "val_starts": [0, 0]}]
    with pytest.raises(ValueError, match="duplicate sample identity"):
        subject._identity_records(manifest)


def test_target_accessor_rejects_held_and_detects_future_20_frame_misalignment(tmp_path: Path):
    u = np.arange(40 * 64 * 128, dtype=np.float32).reshape(40, 64, 128)
    with h5py.File(tmp_path / "a.h5", "w") as handle:
        handle["u"] = u; handle["v"] = -u
    future = np.stack((u[20:40, ::2, ::2], -u[20:40, ::2, ::2]), axis=-1)
    cache = tmp_path / "target.npy"; np.save(cache, future[None])
    identities = [{"row": 0, "trajectory": 0, "file": "a.h5", "start": 0,
                   "frames": 40, "re": 100, "aoa": 10}]
    accessor = subject.TargetAccessor(target_path=cache, real_dir=tmp_path,
                                     identities=identities, held=np.asarray([0]))
    with pytest.raises(PermissionError, match="forbidden"):
        accessor.cache(np.asarray([0]))
    assert accessor.report()["held_target_access_counter"] == 0
    accessor.unlock_terminal()
    assert subject.verify_target_alignment(accessor, np.asarray([0]), "terminal")["passed"]
    wrong = np.stack((u[:20, ::2, ::2], -u[:20, ::2, ::2]), axis=-1)
    np.save(cache, wrong[None])
    with pytest.raises(ValueError, match="mismatch"):
        subject.verify_target_alignment(accessor, np.asarray([0]), "wrong_20_frames")


def test_raw_teacher_fallback_is_exact_zero_when_target_equals_source():
    source = torch.randn(2, 5, 8, 10, 2)
    raw = subject.optimize_raw_block_oracle(source, source.clone())
    assert torch.equal(raw, torch.zeros_like(raw))


def test_sealed_head_is_reloaded_from_disk_not_live_memory(tmp_path: Path):
    torch.manual_seed(3)
    original = subject.CoarseFlowHead()
    with torch.no_grad():
        original.head.bias.fill_(0.25)
    expected = {key: value.clone() for key, value in original.state_dict().items()}
    seal = subject.seal_head(original, tmp_path)
    with torch.no_grad():
        original.head.bias.fill_(9.0)
    reloaded = subject.reload_sealed_head(seal, torch.device("cpu"))
    for key, value in reloaded.state_dict().items():
        assert torch.equal(value, expected[key])


def test_fold_assignment_mapping_is_checked():
    identities = [
        {"file": "a.h5", "aoa": 0, "re": 100},
        {"file": "b.h5", "aoa": 5, "re": 200},
        {"file": "c.h5", "aoa": 10, "re": 300},
        {"file": "d.h5", "aoa": 15, "re": 400},
        {"file": "e.h5", "aoa": 20, "re": 500},
    ]
    assignments = [
        {"file": item["file"], "aoa": item["aoa"], "re": item["re"],
         "fold": fold, "windows": 1}
        for fold, item in enumerate(identities)
    ]
    report = subject.validate_fold_assignments(
        {"fold_assignments": assignments}, identities,
        np.arange(5, dtype=np.int32))
    assert report["passed"] and report["trajectories"] == 5
    with pytest.raises(ValueError, match="differs"):
        subject.validate_fold_assignments(
            {"fold_assignments": assignments}, identities,
            np.asarray([1, 1, 2, 3, 4]))
