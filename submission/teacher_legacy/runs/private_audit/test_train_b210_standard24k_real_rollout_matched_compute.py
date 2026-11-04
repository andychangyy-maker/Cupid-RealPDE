from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / (
    "runs/private_audit/"
    "train_b210_standard24k_real_rollout_matched_compute.py")
SPEC = importlib.util.spec_from_file_location("rollout_matched_compute", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_only_generated_fraction_differs_between_matched_arms():
    assert MODULE.CONTROL_GENERATED_FRACTION == 0.0
    assert MODULE.CANDIDATE_GENERATED_FRACTION == pytest.approx(0.5)
    assert MODULE.rollout.ROLLOUT_OUTER_WEIGHT == pytest.approx(0.1)


def test_matched_compute_contract_binds_both_arms():
    source = PATH.read_text(encoding="utf-8")
    assert '"control": CONTROL_GENERATED_FRACTION' in source
    assert '"candidate": CANDIDATE_GENERATED_FRACTION' in source
    assert '"extra_batch_and_forward_count_matched": True' in source
    assert source.count('kwargs["extra_backward"] = callback') == 1
    assert source.count('training["rollout_exposure"] = callback.report()') == 1


def test_rollout_callback_rejects_invalid_controls():
    with pytest.raises(ValueError, match="generated_fraction"):
        MODULE.rollout.RolloutExposureBackward(
            None, None, None, 1, generated_fraction=1.1)
    with pytest.raises(ValueError, match="outer_weight"):
        MODULE.rollout.RolloutExposureBackward(
            None, None, None, 1, outer_weight=0.0)
