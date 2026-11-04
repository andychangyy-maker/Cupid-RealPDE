import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / (
    "evaluate_b210_fixed_function_ensemble.py")
SPEC = importlib.util.spec_from_file_location("b210_function_ensemble", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fixed_blend_is_exact_half_average_in_fp32():
    first = torch.tensor([0.0, 2.0], dtype=torch.float16)
    second = torch.tensor([2.0, 4.0], dtype=torch.float16)
    actual = MODULE.fixed_blend(first, second)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.tensor([1.0, 3.0]))


def test_fixed_blend_rejects_shape_or_dtype_mismatch():
    with pytest.raises(ValueError, match="shape or dtype"):
        MODULE.fixed_blend(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="shape or dtype"):
        MODULE.fixed_blend(torch.ones(2), torch.ones(2, dtype=torch.float16))
