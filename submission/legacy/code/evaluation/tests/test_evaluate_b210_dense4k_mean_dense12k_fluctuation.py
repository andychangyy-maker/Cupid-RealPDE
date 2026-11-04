import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / (
    "evaluate_b210_dense4k_mean_dense12k_fluctuation.py")
SPEC = importlib.util.spec_from_file_location("b210_role_teacher", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_role_teacher_uses_early_mean_and_late_fluctuation():
    torch.manual_seed(9)
    early = torch.randn(2, 20, 3, 4, 2)
    late = torch.randn(2, 20, 3, 4, 2)
    teacher = MODULE.role_teacher(early, late)
    torch.testing.assert_close(
        teacher.mean(dim=1, keepdim=True),
        early.mean(dim=1, keepdim=True), atol=3e-7, rtol=0.0)
    torch.testing.assert_close(
        MODULE.zero_mean_fluctuation(teacher),
        MODULE.zero_mean_fluctuation(late), atol=3e-7, rtol=0.0)


def test_role_teacher_rejects_mismatch():
    with pytest.raises(ValueError, match="shape or dtype"):
        MODULE.role_teacher(torch.ones(1, 20, 2, 2, 2),
                            torch.ones(1, 19, 2, 2, 2))


def test_role_gate_requires_point_and_sps_safety():
    delta = {
        "rel_l2_score": 0.03, "tke_rel_l2_score": 0.0,
        "mvpe_rel_l2_score": -0.01, "point_score": 0.012,
        "native_sps_score": 0.0, "native_sps_coverage": 0.0,
        "native_sps_mean_width": 0.0,
    }
    by_aoa = {
        str(index): {"point_score": value}
        for index, value in enumerate((0.01, 0.01, 0.0, 0.02, -0.02))
    }
    assert MODULE.role_gate(delta, by_aoa, 1e-7)["pass"]
    failed = dict(delta)
    failed["native_sps_score"] = -0.031
    assert not MODULE.role_gate(failed, by_aoa, 1e-7)["pass"]
