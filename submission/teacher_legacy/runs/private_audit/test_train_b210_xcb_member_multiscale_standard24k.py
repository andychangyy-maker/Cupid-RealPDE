from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, filename: str):
    path = ROOT / "runs" / "private_audit" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE2 = load(
    "xcb_member_multiscale_standard20k",
    "train_b210_xcb_member_multiscale_standard20k_sim.py")
STAGE3 = load(
    "xcb_member_multiscale_standard24k",
    "train_b210_xcb_member_multiscale_standard24k_sim.py")
REAL = load(
    "xcb_member_multiscale_standard24k_real",
    "train_b210_xcb_member_multiscale_standard24k_real_dense_aug.py")


def payload(method: str) -> dict:
    return {
        "iteration": 10_000,
        "model_state_dict": {},
        "training_config": {
            "method": method,
            "window_stride": 20,
            "cross_conditioned_backbone": True,
            "standard_objective_member_feature_adapters": True,
            "multiscale_energy_controller": True,
            "legacy_tke_gradient_floor": False,
        },
    }


def test_stage2_requires_complete_composite_stage1_contract():
    STAGE2.validate_stage1_payload(payload(STAGE2.SOURCE_METHOD))
    missing = payload(STAGE2.SOURCE_METHOD)
    missing["training_config"]["multiscale_energy_controller"] = False
    with pytest.raises(RuntimeError, match="contract changed"):
        STAGE2.validate_stage1_payload(missing)
    wrong = payload("another_method")
    with pytest.raises(RuntimeError, match="composite stage1"):
        STAGE2.validate_stage1_payload(wrong)


def test_stage3_requires_sealed_composite_sim20k():
    STAGE3.validate_stage2_payload(payload(STAGE3.SOURCE_METHOD))
    wrong = payload("another_method")
    with pytest.raises(RuntimeError, match="composite Sim20k"):
        STAGE3.validate_stage2_payload(wrong)


def test_stage3_artifact_and_real_identity(tmp_path, monkeypatch):
    old = tmp_path / "model_10000.pth"
    old.write_bytes(b"checkpoint")
    (tmp_path / "summary.json").write_text(json.dumps({
        "checkpoint": str(old),
        "checkpoint_sha256": "old",
        "completed_updates": 10_000,
    }))
    monkeypatch.setattr(STAGE3.transfer, "sha256", lambda path: "b" * 64)
    new, checksum = STAGE3.finalize_short_artifacts(tmp_path)
    assert new.name == "model_4000.pth"
    assert checksum == "b" * 64
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed_updates"] == 4_000
    assert summary["cumulative_supervised_updates"] == 24_000

    calls = []

    def validator(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    assert REAL.bind_validator(validator)("cp", "sha", "norm", "cuda") == "ok"
    assert calls[0][1] == {
        "expected_iteration": REAL.EXPECTED_ITERATION,
        "expected_status": REAL.EXPECTED_STATUS,
    }
