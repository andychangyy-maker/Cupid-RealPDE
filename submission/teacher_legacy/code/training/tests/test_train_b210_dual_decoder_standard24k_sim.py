from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "runs" / "private_audit"
    / "train_b210_dual_decoder_standard24k_sim.py")
SPEC = importlib.util.spec_from_file_location("dual_decoder_24k", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def payload(mode: str) -> dict:
    return {
        "iteration": 10_000,
        "model_state_dict": {},
        "model_config": {"dual_decoder_mode": mode},
        "training_config": {
            "method": MODULE.CONTRACTS[mode]["source"],
            "window_stride": 20,
            "cross_conditioned_backbone": True,
            "standard_objective_member_feature_adapters": True,
            "legacy_tke_gradient_floor": False,
        },
    }


@pytest.mark.parametrize("mode", tuple(MODULE.CONTRACTS))
def test_validator_accepts_only_matching_stage2(mode):
    MODULE.validate_stage2_payload(payload(mode), mode)
    wrong = payload(mode)
    wrong["training_config"]["method"] = "other"
    with pytest.raises(RuntimeError, match="sealed stage2"):
        MODULE.validate_stage2_payload(wrong, mode)


def test_short_artifact_finalize(tmp_path, monkeypatch):
    old = tmp_path / "model_10000.pth"
    old.write_bytes(b"checkpoint")
    (tmp_path / "summary.json").write_text(json.dumps({
        "checkpoint": str(old), "checkpoint_sha256": "old",
        "completed_updates": 10_000,
    }))
    monkeypatch.setattr(MODULE.transfer, "sha256", lambda path: "a" * 64)
    new, checksum = MODULE.finalize_short_artifacts(tmp_path)
    assert new.name == "model_4000.pth"
    assert checksum == "a" * 64
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed_updates"] == 4_000
    assert summary["cumulative_supervised_updates"] == 24_000
