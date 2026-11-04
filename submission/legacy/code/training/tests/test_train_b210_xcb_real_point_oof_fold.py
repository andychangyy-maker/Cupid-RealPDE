from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "runs" / "private_audit" / (
    "train_b210_xcb_real_point_oof_fold.py")
SPEC = importlib.util.spec_from_file_location("point_oof", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def rows():
    return [
        {"file": f"{re}_{aoa}.h5", "re": re, "aoa": aoa}
        for re in (10, 20, 30, 40, 50)
        for aoa in (0, 5, 10, 15, 20)
    ]


def test_condition_fold_assignments_match_uq_formula():
    assigned = subject.condition_fold_assignments(rows(), 5)
    assert set(assigned.values()) == set(range(5))
    assert assigned["10_0.h5"] == 0
    assert assigned["20_0.h5"] == 1
    assert assigned["10_5.h5"] == 2


def test_condition_fold_assignments_reject_duplicate_file():
    duplicate = rows()
    duplicate.append(dict(duplicate[0]))
    with pytest.raises(ValueError, match="duplicate"):
        subject.condition_fold_assignments(duplicate, 5)


def test_residual_gate_passes_structured_nonidentical_oof_signal():
    rng = np.random.default_rng(4)
    full_error = np.linspace(0.0, 2.0, 2000)
    fold_error = full_error + rng.normal(0.0, 0.3, full_error.shape)
    full_spread = full_error + rng.normal(0.0, 0.4, full_error.shape)
    arrays_full = {
        "spread_z": full_spread,
        "adapter_z": full_spread,
        "error": full_error + 1.0,
    }
    arrays_fold = {
        "spread_z": full_spread,
        "adapter_z": full_spread,
        "error": fold_error + 1.2,
    }
    gate = subject.residual_transfer_gate(arrays_fold, arrays_full)
    assert gate["pass"] is True
    assert all(gate["conditions"].values())


def test_member_integrated_lgf_contract_is_parameter_free_and_complete():
    assert subject.member_integrated_lgf_contract(False) == {}
    contract = subject.member_integrated_lgf_contract(True)
    assert contract["point_accuracy_weights"] == (46.0, 10.0, 9.0)
    assert contract["member_accuracy_weights"] == (46.0, 10.0, 9.0)
    assert contract["legacy_tke_gradient_floor"] is True
    assert contract["legacy_tke_gradient_floor_on_members"] is True
    assert contract["legacy_point_accuracy_weights"] == (5.0, 3.0, 2.0)
    assert contract["legacy_member_accuracy_weights"] == (5.0, 3.0, 2.0)
