from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_point_uq_function_distill_oof as oof  # noqa: E402
import train_b210_point_uq_function_distill_refit as subject  # noqa: E402


def write_summary(path: Path, *, passed: bool = True, weight: float = 1.5) -> str:
    row = {
        "schema": oof.SCHEMA,
        "decision": (
            "PASS_FUNCTION_LEVEL_UQ_DISTILL_OOF"
            if passed
            else "STOP_FUNCTION_LEVEL_UQ_DISTILL_OOF"
        ),
        "selection": {
            "pass": passed,
            "selected": oof.arm_label(weight) if passed else None,
            "selected_weight": weight,
            "selected_comparison": {},
        },
    }
    path.write_text(json.dumps(row), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_oof_summary_accepts_preregistered_passing_weight(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    sha = write_summary(path)
    row, weight = subject.validate_oof_summary(path, sha)
    assert row["selection"]["pass"] is True
    assert weight == 1.5


def test_validate_oof_summary_rejects_failed_gate(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    sha = write_summary(path, passed=False)
    with pytest.raises(RuntimeError, match="did not pass"):
        subject.validate_oof_summary(path, sha)
