from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "evaluate_c208_observation_factor_oracle.py")
SPEC = importlib.util.spec_from_file_location(
    "c208_vf1_observation_factor_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def _pooled(point: float = 0.15, rel: float = 0.01,
            mvpe: float = 0.01) -> dict:
    return {
        "delta": {
            "point_score": point,
            "rel_l2_score": rel,
            "mvpe_score": mvpe,
        },
    }


def _by_aoa(aoa20: float = 0.0) -> dict:
    return {
        str(aoa): {
            "non_time_shadow_delta": aoa20 if aoa == 20 else 0.01,
        }
        for aoa in subject.AOA_VALUES}


def test_contract_freezes_baseline_transform_gate_and_budget():
    contract = subject.EXPERIMENT_CONTRACT
    assert contract["evidence_class"] == (
        "target-visible upper-bound only; not deployable")
    assert contract["exact_baseline"]["checkpoint_sha256"] == (
        "79b3f68272580f80f1d9e09a54e9f9f5291e74524c8b11cdfd78b77a1566cbc3")
    assert contract["data_boundary"]["train_windows"] == 2576
    assert subject.POINT_DELTA_MIN == 0.15
    assert subject.WALL_BUDGET_SECONDS == 180.0
    assert subject.GPU_BUDGET_SECONDS == 180.0


def test_oracle_is_elementwise_exact_and_does_not_mutate_inputs():
    baseline = torch.arange(
        20 * 2 * 3 * 3, dtype=torch.float32).reshape(1, 20, 2, 3, 3)
    baseline[..., 2] = 0.0
    truth = torch.ones_like(baseline)
    truth[..., 2] = 0.0
    truth[0, 0, 0, 0, 0] = 0.0
    truth[0, 0, 0, 0, 1] = 2.0
    truth[0, 1, 0, 1, 1] = -0.0
    truth[0, 2, 1, 2, 0] = 1e-30
    baseline_before = baseline.clone()
    truth_before = truth.clone()

    candidate, mask, report = subject.truth_zero_factor_oracle(
        baseline, truth)

    assert mask[0, 0, 0, 0].tolist() == [True, False]
    assert mask[0, 1, 0, 1, 1]
    assert not mask[0, 2, 1, 2, 0]
    assert candidate[0, 0, 0, 0, 0] == 0.0
    assert candidate[0, 0, 0, 0, 1] == baseline[0, 0, 0, 0, 1]
    assert candidate[0, 1, 0, 1, 1] == 0.0
    assert torch.equal(candidate[..., 2], baseline[..., 2])
    assert torch.equal(baseline, baseline_before)
    assert torch.equal(truth, truth_before)
    assert report["selected_candidate_elements_are_zero"] is True
    assert report["outside_mask_uv_bitwise_equal"] is True
    assert report["pressure_bitwise_equal"] is True


def test_oracle_rejects_nonzero_pressure():
    baseline = torch.zeros(1, 20, 2, 2, 3, dtype=torch.float32)
    truth = torch.zeros_like(baseline)
    baseline[..., 2] = 1.0
    with pytest.raises(ValueError, match="pressure"):
        subject.truth_zero_factor_oracle(baseline, truth)


def test_fixed_bound_scoring_keeps_bound_bytes_coverage_and_width():
    rng = np.random.default_rng(7)
    shape = (2, 20, 32, 64, 3)
    target = rng.normal(0.2, 0.03, size=shape).astype(np.float32)
    target[..., 2] = 0.0
    target[:, :, :2, :3, :2] = 0.0
    baseline = target.copy()
    baseline[:, :, :2, :3, :2] = np.float32(0.5)
    lower = baseline[..., :2] - np.float32(0.1)
    upper = baseline[..., :2] + np.float32(0.1)
    baseline_tensor = torch.from_numpy(np.ascontiguousarray(baseline))
    target_tensor = torch.from_numpy(np.ascontiguousarray(target))
    candidate, _mask, _report = subject.truth_zero_factor_oracle(
        baseline_tensor, target_tensor)
    lower_before = lower.tobytes()
    upper_before = upper.tobytes()

    comparison = subject.metric_comparison_fixed_bounds(
        baseline, candidate.numpy(), target, lower, upper)

    assert lower.tobytes() == lower_before
    assert upper.tobytes() == upper_before
    assert comparison["bounds"]["bitwise_unchanged"] is True
    assert comparison["candidate"]["coverage"] == (
        comparison["baseline"]["coverage"])
    assert comparison["candidate"]["mean_width"] == (
        comparison["baseline"]["mean_width"])
    assert comparison["delta"]["rel_l2_score"] > 0.0


def test_gate_exact_point_and_aoa_boundaries_pass_but_rel_mvpe_are_strict():
    elapsed = {"wall_seconds": 180.0, "gpu_seconds": 180.0}
    passed = subject.gate_result(
        _pooled(), _by_aoa(), {"synthetic_invariant": True}, elapsed)
    assert passed["pass"] is True
    assert passed["decision"] == "GO_VF1_OBSERVATION_FACTOR_HEADROOM"

    cases = [
        (_pooled(point=0.149999), _by_aoa(), {"guard": True}, elapsed),
        (_pooled(rel=0.0), _by_aoa(), {"guard": True}, elapsed),
        (_pooled(mvpe=0.0), _by_aoa(), {"guard": True}, elapsed),
        (_pooled(), _by_aoa(-1e-12), {"guard": True}, elapsed),
        (_pooled(), _by_aoa(), {"guard": False}, elapsed),
        (_pooled(), _by_aoa(), {"guard": True}, {
            "wall_seconds": 180.01, "gpu_seconds": 180.0}),
        (_pooled(), _by_aoa(), {"guard": True}, {
            "wall_seconds": 180.0, "gpu_seconds": 180.01}),
    ]
    for pooled, by_aoa, invariants, timing in cases:
        failed = subject.gate_result(pooled, by_aoa, invariants, timing)
        assert failed["pass"] is False
        assert failed["decision"] == "STOP_VF1_NO_OBSERVATION_HEADROOM"


def test_contract_only_reads_no_inputs_and_starts_no_gpu(tmp_path, monkeypatch):
    output = tmp_path / "contract"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT),
        "--point-checkpoint", str(tmp_path / "missing.pth"),
        "--split-manifest", str(tmp_path / "missing.json"),
        "--output-dir", str(output),
        "--contract-only",
    ])

    subject.main()

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["decision"] == "CONTRACT_ONLY_NO_DATA_NO_GPU"
    assert result["tail_accessed"] is False
    assert result["checkpoint_written"] is False
    assert result["zip_written"] is False
