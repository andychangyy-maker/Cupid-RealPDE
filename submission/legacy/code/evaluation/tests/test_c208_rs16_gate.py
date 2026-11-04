from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = EVALUATION_ROOT / "evaluate_c208_rs16_gate.py"
SPEC = importlib.util.spec_from_file_location("c208_rs16_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def _payload(stride: int, output_dir: str, manifest: str | None = None):
    training = {
        "phase": "real",
        "updates": 5000,
        "batch_size": 8,
        "lr": 2e-4,
        "output_dir": output_dir,
        "window_stride": stride,
    }
    if manifest is not None:
        training["baseline_split_manifest"] = manifest
    return {
        "phase": "real",
        "iteration": 4000,
        "model_config": {
            **copy.deepcopy(subject.rehearsal.EXPECTED_MODEL_CONFIG),
            "in_steps": 20,
            "out_steps": 20,
            "quantile_levels": (0.05, 0.95),
        },
        "training_config": training,
    }


def test_checkpoint_contract_allows_only_stride_output_and_manifest(tmp_path):
    manifest = tmp_path / "baseline.json"
    manifest.write_text("[]\n", encoding="utf-8")
    baseline = _payload(20, "/runs/baseline")
    candidate = _payload(16, "/runs/candidate", str(manifest))

    result = subject.validate_checkpoint_contracts(
        baseline, candidate, manifest)

    assert result["iteration"] == 4000
    assert result["architecture"] == "ensemble_unet"
    wrong_frozen_field = copy.deepcopy(candidate)
    wrong_frozen_field["model_config"]["base"] = 160
    with pytest.raises(ValueError, match="model_config\\['base'\\]"):
        subject.validate_checkpoint_contracts(
            baseline, wrong_frozen_field, manifest)

    candidate["training_config"]["lr"] = 3e-4
    with pytest.raises(ValueError, match="outside the RS16 variable"):
        subject.validate_checkpoint_contracts(baseline, candidate, manifest)


def test_checkpoint_contract_rejects_wrong_iteration_architecture_and_stride(
        tmp_path):
    manifest = tmp_path / "baseline.json"
    manifest.write_text("[]\n", encoding="utf-8")
    baseline = _payload(20, "/runs/baseline")
    candidate = _payload(16, "/runs/candidate", str(manifest))

    wrong_iteration = copy.deepcopy(candidate)
    wrong_iteration["iteration"] = 4500
    with pytest.raises(ValueError, match="update 4000"):
        subject.validate_checkpoint_contracts(
            baseline, wrong_iteration, manifest)

    wrong_architecture = copy.deepcopy(candidate)
    wrong_architecture["model_config"]["architecture"] = "other"
    with pytest.raises(ValueError, match="architecture"):
        subject.validate_checkpoint_contracts(
            baseline, wrong_architecture, manifest)

    wrong_stride = copy.deepcopy(candidate)
    wrong_stride["training_config"]["window_stride"] = 18
    with pytest.raises(ValueError, match="not 16"):
        subject.validate_checkpoint_contracts(
            baseline, wrong_stride, manifest)


def _split_rows():
    baseline = []
    candidate = []
    for index, aoa in enumerate(subject.AOA_VALUES):
        common = {
            "file": f"trajectory_{index}.h5",
            "frames": 160,
            "re": 5000 + index,
            "aoa": aoa,
            "val_starts": [100, 120],
            "val_boundary": 100,
        }
        baseline.append({**common, "train_starts": [0, 20, 40, 60]})
        candidate.append({**common, "train_starts": [0, 16, 32, 48]})
    return baseline, candidate


def test_split_contract_requires_exact_fixed_validation_and_zero_overlap(
        monkeypatch):
    baseline, candidate = _split_rows()
    monkeypatch.setattr(subject, "EXPECTED_COUNTS", {
        "baseline_train": 20,
        "candidate_train": 20,
        "validation": 10,
        "trajectories": 5,
    })
    monkeypatch.setattr(
        subject, "EXPECTED_CANDIDATE_TRAIN_AOA_COUNTS",
        {aoa: 4 for aoa in subject.AOA_VALUES})
    monkeypatch.setattr(
        subject, "EXPECTED_VALIDATION_AOA_COUNTS",
        {aoa: 2 for aoa in subject.AOA_VALUES})

    result = subject.validate_split_manifests(baseline, candidate)

    assert all(result["checks"].values())
    assert result["counts"]["interval_overlaps"] == 0
    assert result["counts"]["exact_key_intersections"] == 0

    leaked = copy.deepcopy(candidate)
    leaked[0]["val_starts"] = [96, 120]
    with pytest.raises(ValueError, match="validation keys differ"):
        subject.validate_split_manifests(baseline, leaked)

    wrong_stride = copy.deepcopy(candidate)
    wrong_stride[0]["train_starts"] = [0, 16, 32, 49]
    with pytest.raises(ValueError, match="exact stride16"):
        subject.validate_split_manifests(baseline, wrong_stride)


def test_gate_contract_uses_point_shadow_sps_guard_and_aoa_robustness():
    pooled = {
        "rel_l2_score": 0.12,
        "tke_score": 0.02,
        "mvpe_score": 0.02,
        "point_score": 0.10,
        "sps_score": -0.10,
    }
    rows = [
        {"point_shadow_delta": value}
        for value in (0.04, 0.03, 0.02, 0.01, -0.02)
    ]

    checks = subject.build_gate_checks(pooled, rows, {"hard_guard": True})
    assert all(checks.values())

    stopped = subject.build_gate_checks(
        pooled, rows, {"hard_guard": False})
    assert not stopped["hard_guard"]

    weak_aoa = copy.deepcopy(rows)
    weak_aoa[3]["point_shadow_delta"] = 0.0
    weak_checks = subject.build_gate_checks(
        pooled, weak_aoa, {"hard_guard": True})
    assert not weak_checks["positive_aoa_shadow_ge_4_of_5"]
