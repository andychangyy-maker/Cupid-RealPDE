from collections import OrderedDict
import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / (
    "evaluate_b210_sim_midpoint_connectivity.py")
SPEC = importlib.util.spec_from_file_location("b210_midpoint", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_mixed_state_uses_fixed_midpoint_and_exact_integer_state():
    first = OrderedDict([
        ("weight", torch.tensor([0.0, 2.0])),
        ("count", torch.tensor([3], dtype=torch.int64)),
    ])
    second = OrderedDict([
        ("weight", torch.tensor([2.0, 4.0])),
        ("count", torch.tensor([3], dtype=torch.int64)),
    ])
    mixed = MODULE.mixed_state(first, second)
    torch.testing.assert_close(mixed["weight"], torch.tensor([1.0, 3.0]))
    assert torch.equal(mixed["count"], first["count"])


def test_mixed_state_rejects_different_nonfloating_state():
    with pytest.raises(ValueError, match="non-floating"):
        MODULE.mixed_state(
            OrderedDict([("count", torch.tensor([1]))]),
            OrderedDict([("count", torch.tensor([2]))]))


def test_connectivity_gate_pass_and_stop():
    endpoint_a = {
        "rel_l2_score": 94.0, "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 95.0, "point_score": 91.5}
    endpoint_b = {
        "rel_l2_score": 93.9, "tke_rel_l2_score": 80.2,
        "mvpe_rel_l2_score": 94.9, "point_score": 91.5}
    passing = dict(endpoint_a)
    passing["point_score"] = 91.31
    assert MODULE.connectivity_gate(endpoint_a, endpoint_b, passing)["pass"]
    failing = dict(endpoint_a)
    failing["point_score"] = 91.29
    assert not MODULE.connectivity_gate(endpoint_a, endpoint_b, failing)["pass"]


def test_parse_config_treats_absent_new_fields_as_zero_defaults():
    old = {"architecture": "ensemble_unet", "base": 32}
    new = {
        **old, "acceleration_steps": 0,
        "phase_delta_hidden_channels": 0, "phase_delta_steps": 5,
    }
    assert MODULE.parse_config({"model_config": old}) == MODULE.parse_config(
        {"model_config": new})
