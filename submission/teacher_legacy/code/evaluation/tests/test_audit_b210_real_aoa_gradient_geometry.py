import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "audit_b210_real_aoa_gradient_geometry.py"
SPEC = importlib.util.spec_from_file_location("gradient_audit", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_evenly_spread_indices_and_plan_shape():
    assert subject.evenly_spread_indices(10, 4) == [0, 3, 6, 9]
    windows = [SimpleNamespace(trajectory=aoa_index, start=start)
               for aoa_index in range(5) for start in range(2)]
    metadata = [{"aoa": aoa} for aoa in subject.AOAS]
    plan = subject.make_aoa_index_plan(windows, metadata, rounds=1)
    assert plan == {aoa: [[2 * index, 2 * index + 1]]
                    for index, aoa in enumerate(subject.AOAS)}


def test_scope_routing():
    assert subject.route_scope("decoder.0.weight") == "shared_decoder"
    assert subject.route_scope("energy_controller.bias") == "point_heads"
    assert subject.route_scope("quantile_head.weight") == "uq"
    assert subject.route_scope("encoder.0.weight") is None


def test_cosine_summary_and_decision():
    assert subject.cosine(torch.tensor([1., 0.]), torch.tensor([-1., 0.])) == -1.0
    with pytest.raises(FloatingPointError, match="zero gradient"):
        subject.cosine(torch.zeros(2), torch.ones(2))
    summary = subject.summarize([-0.5, 0.2, 0.4])
    assert summary == {"negative_fraction": 1 / 3, "min": -0.5, "median": 0.2}
    scopes = {"shared_decoder": {**summary, "persistent_min": -0.03},
              "point_heads": {**summary, "persistent_min": 0.0},
              "uq": {**summary, "persistent_min": -1.0}}
    assert subject.decide(scopes) == "PCGRAD_HEADROOM"
    scopes["shared_decoder"] = {"negative_fraction": 0.09, "persistent_min": -1.0}
    assert subject.decide(scopes) == "STOP_GRADIENT_SURGERY_NO_HEADROOM"
