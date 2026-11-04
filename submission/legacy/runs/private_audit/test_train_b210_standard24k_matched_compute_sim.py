from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/private_audit/train_b210_standard24k_matched_compute_sim.py"
SPEC = importlib.util.spec_from_file_location("standard24k", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def payload(*, method: str | None = None, stride: int = 20) -> dict:
    return {
        "iteration": 10_000,
        "model_state_dict": {},
        "training_config": {
            "method": method or MODULE.SOURCE_METHOD,
            "window_stride": stride,
        },
    }


def test_accepts_only_sealed_standard20_endpoint():
    MODULE.validate_standard20_payload(payload())
    with pytest.raises(RuntimeError, match="sealed standard-20k"):
        MODULE.validate_standard20_payload(payload(method="unrelated"))
    with pytest.raises(RuntimeError, match="stride 20"):
        MODULE.validate_standard20_payload(payload(stride=10))


def test_short_stage_contract_and_cli(monkeypatch):
    monkeypatch.setattr(MODULE.transfer.b210, "TERMINAL_UPDATE", 10_000)
    MODULE.configure_short_standard_stage()
    assert MODULE.transfer.b210.TERMINAL_UPDATE == 4_000
    assert MODULE.output_dir_from_argv(["runner", "--output-dir", "out"]) == Path("out")
    with pytest.raises(ValueError, match="output-dir"):
        MODULE.output_dir_from_argv(["runner"])


def test_finalize_renames_and_repairs_summary(tmp_path, monkeypatch):
    old = tmp_path / "model_10000.pth"
    old.write_bytes(b"sealed checkpoint")
    (tmp_path / "summary.json").write_text(json.dumps({
        "checkpoint": str(old),
        "checkpoint_sha256": "old",
        "completed_updates": 10_000,
    }))
    monkeypatch.setattr(MODULE.transfer, "sha256", lambda path: "a" * 64)
    new, checksum = MODULE.finalize_short_artifacts(tmp_path)
    assert not old.exists()
    assert new.name == "model_4000.pth"
    assert checksum == "a" * 64
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed_updates"] == 4_000
    assert summary["cumulative_supervised_updates"] == 24_000
    assert summary["checkpoint"] == str(new.resolve())
