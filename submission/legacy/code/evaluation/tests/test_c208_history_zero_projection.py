from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "evaluate_c208_history_zero_projection.py")
SPEC = importlib.util.spec_from_file_location("c208_history_zero_projection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def _pooled(point: float = 0.10, shadow: float = 0.06,
            component: float = 0.0) -> dict:
    return {
        "delta": {
            "rel_l2_score": component,
            "tke_rel_l2_score": component,
            "mvpe_rel_l2_score": component,
            "point_score": point,
        },
        "point_shadow_delta": shadow,
    }


def _by_aoa(values=(0.01, 0.01, 0.01, 0.01, -0.01)) -> dict:
    return {
        str(aoa): {"point_shadow_delta": value}
        for aoa, value in zip(subject.AOA_VALUES, values, strict=True)}


def test_contract_freezes_transform_gate_and_budget():
    contract = subject.EXPERIMENT_CONTRACT
    assert contract["data_boundary"]["train_windows"] == 2576
    assert contract["candidate"]["target_used_for_mask_or_transform"] is False
    assert subject.POINT_DELTA_MIN == 0.10
    assert subject.POINT_SHADOW_DELTA_MIN == 0.06
    assert subject.AOA_POSITIVE_MIN == 4
    assert subject.WORST_AOA_SHADOW_MIN == -0.01
    assert subject.GPU_BUDGET_SECONDS == 300.0
    assert subject.WALL_BUDGET_SECONDS == 360.0


def test_projection_is_per_sample_exact_and_inputs_are_unchanged():
    baseline = torch.arange(
        2 * 20 * 3 * 4 * 3, dtype=torch.float32).reshape(2, 20, 3, 4, 3)
    baseline[..., 2] = 0.0
    history = torch.ones_like(baseline)
    history[..., 2] = 0.0
    history[0, :, 1, 2, :2] = 0.0
    history[1, :, 2, 1, :2] = 0.0
    history[1, 7, 0, 0, 0] = 1e-30  # exact zero, not a threshold
    history[1, :, 0, 0, 1] = 0.0
    baseline_before = baseline.clone()
    history_before = history.clone()

    candidate, mask, report = subject.history_zero_projection(baseline, history)

    assert mask[0, 1, 2]
    assert mask[1, 2, 1]
    assert not mask[1, 0, 0]
    assert torch.count_nonzero(candidate[0, :, 1, 2, :2]) == 0
    assert torch.count_nonzero(candidate[1, :, 2, 1, :2]) == 0
    outside = ~mask
    torch.testing.assert_close(
        candidate[..., :2].permute(0, 2, 3, 1, 4)[outside],
        baseline[..., :2].permute(0, 2, 3, 1, 4)[outside], rtol=0, atol=0)
    assert torch.equal(candidate[..., 2], baseline[..., 2])
    assert torch.equal(baseline, baseline_before)
    assert torch.equal(history, history_before)
    assert report["masked_sample_pixels"] == 2
    assert report["future_uv_elements_projected"] == 80
    assert report["future_uv_zero_on_mask"] is True
    assert report["outside_mask_uv_bitwise_equal"] is True
    assert report["candidate_finite"] is True
    assert report["pressure_zero"] is True


def test_mask_requires_both_velocity_channels_zero_in_all_20_frames():
    baseline = torch.ones(3, 20, 2, 2, 3, dtype=torch.float32)
    baseline[..., 2] = 0.0
    history = torch.ones_like(baseline)
    history[..., 2] = 0.0
    history[0, :, 0, 0, :2] = 0.0
    history[1, :, 0, 0, :2] = 0.0
    history[1, 19, 0, 0, 0] = 1.0
    history[2, :, 0, 0, :2] = 0.0
    history[2, 0, 0, 0, 1] = -1.0

    _, mask, _ = subject.history_zero_projection(baseline, history)

    assert mask[:, 0, 0].tolist() == [True, False, False]


def test_gate_exact_boundaries_pass_and_any_failed_condition_stops():
    elapsed = {"gpu_seconds": 300.0, "wall_seconds": 360.0}
    passed = subject.gate_result(_pooled(), _by_aoa(), elapsed)
    assert passed["pass"] is True
    assert passed["decision"] == "GO_HISTORY_ZERO_PROJECTION_HEADROOM"
    assert passed["positive_aoa_shadow_count"] == 4
    assert passed["worst_aoa_point_shadow_delta"] == -0.01

    cases = [
        (_pooled(point=0.099), _by_aoa(), elapsed),
        (_pooled(shadow=0.059), _by_aoa(), elapsed),
        (_pooled(component=-1e-12), _by_aoa(), elapsed),
        (_pooled(), _by_aoa((0.01, 0.01, 0.01, 0.0, -0.01)), elapsed),
        (_pooled(), _by_aoa((0.01, 0.01, 0.01, 0.01, -0.01001)), elapsed),
        (_pooled(), _by_aoa(), {"gpu_seconds": 300.01, "wall_seconds": 360.0}),
        (_pooled(), _by_aoa(), {"gpu_seconds": 300.0, "wall_seconds": 360.01}),
    ]
    for pooled, by_aoa, timing in cases:
        failed = subject.gate_result(pooled, by_aoa, timing)
        assert failed["pass"] is False
        assert failed["decision"] == "STOP_HISTORY_ZERO_PROJECTION_NO_HEADROOM"


def test_write_result_refuses_overwrite_by_default(tmp_path):
    path = tmp_path / "result.json"
    subject.write_result(path, {"decision": "first"})
    assert json.loads(path.read_text(encoding="utf-8"))["decision"] == "first"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        subject.write_result(path, {"decision": "second"})


def test_contract_only_does_not_read_inputs_or_touch_gpu(tmp_path, monkeypatch):
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
