from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "build_track1_b210_xcb_full_point_old_uq_package.py"
)
SPEC = importlib.util.spec_from_file_location("xcb_point_old_uq_builder", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def test_compose_uses_new_dense_point_and_old_compact_uq_exactly() -> None:
    dense = {
        "stem.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "stem.bias": torch.tensor([1.25], dtype=torch.float32),
        "uq_fusion.0.weight": torch.full((3, 4), 999.0),
        "quantile_head.bias": torch.tensor([999.0]),
    }
    old = {
        "stem.weight": {
            "quantized": torch.ones((3, 4), dtype=torch.int8),
            "scale": torch.ones((3, 1), dtype=torch.float32),
        },
        "stem.bias": torch.tensor([-2.0], dtype=torch.float16),
        "uq_fusion.0.weight": {
            "quantized": torch.full((3, 4), 7, dtype=torch.int8),
            "scale": torch.full((3, 1), 0.25, dtype=torch.float32),
        },
        "quantile_head.bias": torch.tensor([-4.0], dtype=torch.float16),
    }
    composed, audit = subject.compose_compact_state(
        dense, old, quantize_min_elements=4
    )
    assert subject.hybrid._tensor_equal(
        composed["uq_fusion.0.weight"], old["uq_fusion.0.weight"]
    )
    assert subject.hybrid._tensor_equal(
        composed["quantile_head.bias"], old["quantile_head.bias"]
    )
    assert not subject.hybrid._tensor_equal(
        composed["stem.weight"], old["stem.weight"]
    )
    assert torch.equal(composed["stem.bias"], dense["stem.bias"].half())
    assert audit["old_uq_source_bitwise_exact"] is True
    assert audit["point_source_exact_after_compaction"] is True
