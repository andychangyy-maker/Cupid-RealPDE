from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "audit_b210_clean716_smooth_flow_oracle.py"
SPEC = importlib.util.spec_from_file_location("smooth_flow_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_zero_flow_is_exact_identity_and_contract_is_explicit():
    source = torch.randn(2, 5, 6, 8, 2)
    raw = torch.zeros(2, 2, 4, 8)
    flow = subject.displacement_from_raw(raw, 6, 8)
    actual = subject.smooth_flow_warp(source, flow)
    assert subject.EVIDENCE_CLASS == "oracle"
    assert subject.CONTROL_GRID == (4, 8)
    assert subject.HORIZON_BLOCKS == ((0, 5), (5, 10), (10, 15), (15, 20))
    assert torch.equal(actual, source)


def test_flow_shape_and_bound_are_frozen():
    raw = torch.full((3, 2, 4, 8), 100.0)
    flow = subject.displacement_from_raw(raw, 32, 64)
    assert flow.shape == (3, 32, 64, 2)
    assert float(flow.abs().max()) <= subject.MAX_DISPLACEMENT_PIXELS


def test_identity_selection_is_monotone_when_target_equals_source():
    source = torch.randn(2, 5, 8, 10, 2)
    selected, chose_flow, flow = subject.optimize_block_oracle(source, source.clone())
    assert not bool(chose_flow.any())
    assert torch.equal(selected, source)
    assert torch.equal(flow, torch.zeros_like(flow))
