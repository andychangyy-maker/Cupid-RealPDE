from __future__ import annotations

from pathlib import Path
import sys


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_member_integrated_tke_floor_real as subject  # noqa: E402


def test_candidate_contract_aligns_mean_and_member_paths():
    contract = subject.candidate_contract()
    assert contract["point_accuracy_weights"] == (46.0, 10.0, 9.0)
    assert contract["member_accuracy_weights"] == (46.0, 10.0, 9.0)
    assert contract["legacy_point_accuracy_weights"] == (5.0, 3.0, 2.0)
    assert contract["legacy_member_accuracy_weights"] == (5.0, 3.0, 2.0)
    assert contract["legacy_tke_gradient_floor"] is True
    assert contract["legacy_tke_gradient_floor_on_members"] is True


def test_prior_and_stream_identities_are_frozen_sha256():
    for value in (
            subject.PRIOR_RESULT_SHA256,
            subject.PRIOR_CONTROL_FILE_SHA256,
            subject.PRIOR_STREAM_SHA256):
        assert len(value) == 64
        int(value, 16)
