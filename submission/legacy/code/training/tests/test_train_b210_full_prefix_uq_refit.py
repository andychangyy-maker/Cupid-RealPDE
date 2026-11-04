import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
SCRIPT = TRAINING / "train_b210_full_prefix_uq_refit.py"
SPEC = importlib.util.spec_from_file_location("b210_full_prefix_uq", SCRIPT)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def test_prefix_audit_allows_each_fit_row_once_and_rejects_tail():
    audit = SUBJECT.PrefixOnlyAudit()
    rows = [{"file": "a.h5"}, {"file": "b.h5"}]
    for row in rows:
        audit.before_open(row, "fit")
        audit.record_open(row, "fit")
    report = audit.report(rows)
    assert report["allowlist_exact"] is True
    assert report["tail_arrays_read"] == 0
    with pytest.raises(RuntimeError):
        audit.before_open(rows[0], "fit")
    with pytest.raises(RuntimeError):
        SUBJECT.PrefixOnlyAudit().before_open(rows[0], "held")


def test_validation_gate_requires_passed_exact_full_loao(tmp_path):
    value = {
        "schema": SUBJECT.VALIDATION_SCHEMA,
        "decision": SUBJECT.VALIDATION_DECISION,
        "held_aoas": list(SUBJECT.dr1.AOA_VALUES),
        "comparison": {"native_sps_delta": 0.42},
        "gate": {
            "validation_gate_pass": True,
            "positive_folds": 5,
            "worst_fold": 0.08,
        },
    }
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    report = SUBJECT.validate_validation_gate(path, SUBJECT.dr1.sha256(path))
    assert report["pooled_sps_delta"] == 0.42
    assert report["positive_folds"] == 5
    value["gate"]["validation_gate_pass"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError):
        SUBJECT.validate_validation_gate(path, SUBJECT.dr1.sha256(path))


def test_point_normalizer_requires_positive_std_and_sha():
    values = {
        "mean_input": [0.1, 0.0, 0.0],
        "mean_target": [0.1, 0.0, 0.0],
        "std_input": [0.2, 0.1, 1.0],
        "std_target": [0.2, 0.1, 1.0],
    }
    payload = {
        "normalizer": values,
        "normalizer_sha256": SUBJECT.dr1.canonical_sha256(values),
    }
    _normalizer, report = SUBJECT.normalizer_from_point(
        payload, torch.device("cpu")
    )
    assert report["sha256"] == payload["normalizer_sha256"]
    bad = dict(payload)
    bad_values = dict(values, std_target=[0.2, 0.0, 1.0])
    bad["normalizer"] = bad_values
    bad["normalizer_sha256"] = SUBJECT.dr1.canonical_sha256(bad_values)
    with pytest.raises(ValueError):
        SUBJECT.normalizer_from_point(bad, torch.device("cpu"))
