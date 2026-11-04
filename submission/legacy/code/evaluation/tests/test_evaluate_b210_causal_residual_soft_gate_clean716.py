from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from models.causal_residual_soft_gate import (
    CausalResidualSoftGate,
    CausalResidualSoftGateConfig,
)
import evaluate_b210_causal_residual_soft_gate_clean716 as subject
import train_b210_causal_residual_soft_gate_lifecycle as lifecycle


def _endpoint(mode: str) -> dict:
    model = CausalResidualSoftGate(
        CausalResidualSoftGateConfig(mixer_mode=mode))
    return {
        "sample_schedule_sha256": "real-stream",
        "fusion_config": model.architecture_config(),
        "base_checkpoint": {
            "sha256": "base-file", "model_state_sha256": "base-state",
            "normalizer_sha256": "normalizer",
        },
        "initial_lineage": {
            "sample_schedule_sha256": "sim-stream",
            "initial_fusion_state_sha256": "common-init",
        },
        "normalizer": {
            "mean_input": torch.zeros(3),
            "mean_target": torch.zeros(3),
            "std_input": torch.ones(3),
            "std_target": torch.ones(3),
        },
    }


def test_decorate_uses_only_published_non_time_weights():
    values = subject.decorate({
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 70.0,
        "native_sps_score": 40.0,
    })
    point = .46 * 90.0 + .10 * 80.0 + .09 * 70.0
    assert values["official_point_contribution_46_10_9"] == pytest.approx(point)
    assert values["non_time_contribution_46_10_9_25"] == pytest.approx(
        point + .25 * 40.0)


def test_pair_validation_accepts_only_mixer_difference():
    control = _endpoint("framewise_spatial_control")
    causal = _endpoint("causal_temporal")
    report = subject.validate_pair(control, causal)
    assert report["matched_outside_mixer_mode"] is True
    broken = deepcopy(causal)
    broken["fusion_config"]["point_gate_limit"] = 0.5
    with pytest.raises(RuntimeError, match="outside mixer"):
        subject.validate_pair(control, broken)


def test_pair_validation_rejects_stream_or_base_drift():
    control = _endpoint("framewise_spatial_control")
    causal = _endpoint("causal_temporal")
    causal["sample_schedule_sha256"] = "other-real-stream"
    with pytest.raises(RuntimeError, match="Real4k"):
        subject.validate_pair(control, causal)
    causal = _endpoint("causal_temporal")
    causal["initial_lineage"]["sample_schedule_sha256"] = "other-sim-stream"
    with pytest.raises(RuntimeError, match="Sim4k"):
        subject.validate_pair(control, causal)
    causal = _endpoint("causal_temporal")
    causal["base_checkpoint"]["model_state_sha256"] = "other-state"
    with pytest.raises(RuntimeError, match="base differs"):
        subject.validate_pair(control, causal)


def test_comparison_orientation_is_candidate_minus_control():
    base = {"rel_l2_score": 1.0}
    control = {"rel_l2_score": 2.0}
    candidate = {"rel_l2_score": 5.0}
    values = subject.comparisons({
        "b210": base,
        "spatial_control_raw": control,
        "spatial_control_ema": control,
        "causal_raw": candidate,
        "causal_ema": candidate,
    })
    assert values["causal_minus_spatial_ema"]["rel_l2_score"] == 3.0
    assert values["each_minus_b210"]["causal_ema"]["rel_l2_score"] == 4.0


def test_target_access_requires_explicit_acknowledgement(tmp_path):
    args = subject.parse_args([
        "--base-checkpoint", str(tmp_path / "base.pt"),
        "--base-checkpoint-sha256", "0" * 64,
        "--spatial-control-endpoint", str(tmp_path / "control.pt"),
        "--spatial-control-endpoint-sha256", "1" * 64,
        "--causal-endpoint", str(tmp_path / "causal.pt"),
        "--causal-endpoint-sha256", "2" * 64,
        "--real-dir", str(tmp_path),
        "--clean-manifest", str(tmp_path / "fit.json"),
        "--clean716-manifest", str(tmp_path / "tail.json"),
        "--clean716-manifest-sha256", "3" * 64,
        "--output-json", str(tmp_path / "result.json"),
    ])
    with pytest.raises(PermissionError, match="acknowledgement"):
        subject.run(args)


def test_lifecycle_constants_are_bound_to_formal_replay():
    assert lifecycle.SIM_UPDATES == 4_000
    assert lifecycle.REAL_UPDATES == 4_000
    assert lifecycle.ARMS == (
        "framewise_spatial_control", "causal_temporal")
