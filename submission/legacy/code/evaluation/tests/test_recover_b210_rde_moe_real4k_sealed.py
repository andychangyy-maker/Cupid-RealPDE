from __future__ import annotations

from pathlib import Path
import inspect
import sys

import pytest


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = EVALUATION_ROOT.parent
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import recover_b210_rde_moe_real4k_sealed as subject  # noqa: E402


def test_recovery_is_bound_to_six_exact_endpoints():
    assert tuple(subject.SEALED_ARTIFACTS) == subject.rde.ARM_ORDER
    assert all(tuple(row) == subject.rde.ENDPOINT_ORDER
               for row in subject.SEALED_ARTIFACTS.values())
    identities = [
        endpoint[key]
        for row in subject.SEALED_ARTIFACTS.values()
        for endpoint in row.values()
        for key in ("file_sha256", "model_state_sha256")
    ]
    assert len(identities) == 12
    assert len(set(identities)) == 12
    assert all(len(value) == 64 for value in identities)


def test_recovery_preflight_does_not_read_evaluation_manifest():
    source = inspect.getsource(subject.validate_pre_tail)
    assert "args.evaluation_manifest.is_file()" in source
    assert "sha256(args.evaluation_manifest)" not in source
    assert "load_evaluation_rows" not in source
    recovery = inspect.getsource(subject.recover)
    preflight = recovery.index("context = validate_pre_tail(args)")
    manifest_hash = recovery.index("evaluation_sha = rde.dr1.sha256")
    manifest_load = recovery.index("load_evaluation_rows")
    target_load = recovery.index("load_prefix_rows")
    assert preflight < manifest_hash < manifest_load < target_load


def test_failed_log_requires_both_exact_sha_and_terminal_marker(
        tmp_path, monkeypatch):
    path = tmp_path / "failed.log"
    path.write_text(
        "line 1122, in run\n"
        "RuntimeError: R1/R2 raw/EMA router or standardization changed\n",
        encoding="utf-8")
    monkeypatch.setattr(
        subject, "EXPECTED_FAILED_LOG_SHA256", subject.rde.dr1.sha256(path))
    report = subject.validate_failed_log(path)
    assert report["failure_stage"] == "after six clean716 endpoint evaluations"
    path.write_text("different", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA changed"):
        subject.validate_failed_log(path)


def test_audit_only_precedes_any_device_or_target_evaluation():
    source = inspect.getsource(subject.recover)
    audit_return = source.index("if args.audit_only:")
    device = source.index("device = torch.device")
    evaluation_sha = source.index("evaluation_sha = rde.dr1.sha256")
    assert audit_return < device < evaluation_sha
