import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / (
    "evaluate_b210_tke_invariant_mean_teacher.py")
SPEC = importlib.util.spec_from_file_location("b210_mean_teacher", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_temporal_mean_teacher_preserves_seed1_fluctuation():
    torch.manual_seed(7)
    seed0 = torch.randn(2, 20, 3, 4, 2)
    seed1 = torch.randn(2, 20, 3, 4, 2)
    teacher = MODULE.temporal_mean_teacher(seed0, seed1)
    torch.testing.assert_close(
        MODULE.zero_mean_fluctuation(teacher),
        MODULE.zero_mean_fluctuation(seed1), atol=3e-7, rtol=0.0)
    expected_mean = 0.5 * (
        seed0.mean(dim=1, keepdim=True)
        + seed1.mean(dim=1, keepdim=True))
    torch.testing.assert_close(
        teacher.mean(dim=1, keepdim=True), expected_mean)


def test_temporal_mean_teacher_rejects_mismatched_fields():
    with pytest.raises(ValueError, match="shape or dtype"):
        MODULE.temporal_mean_teacher(torch.ones(1, 20, 2, 2, 2),
                                     torch.ones(1, 19, 2, 2, 2))


def test_teacher_gate_requires_signal_and_invariance():
    pooled = {
        "rel_l2_score": 0.021, "tke_rel_l2_score": 0.0,
        "mvpe_rel_l2_score": 0.0, "point_score": 0.016,
        "native_sps_score": 0.0, "native_sps_coverage": 0.0,
        "native_sps_mean_width": 0.0,
    }
    by_aoa = {
        str(value): {"point_score": delta}
        for value, delta in enumerate((0.01, 0.02, 0.01, 0.00, -0.02))}
    assert MODULE.teacher_gate(pooled, by_aoa, 1e-7)["pass"]
    failed = dict(pooled)
    failed["tke_rel_l2_score"] = 2e-5
    assert not MODULE.teacher_gate(failed, by_aoa, 1e-7)["pass"]
