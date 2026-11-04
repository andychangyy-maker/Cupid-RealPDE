import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "audit_b210_mature_point_gradient_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("mature_gradient_audit", SCRIPT)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def test_per_sample_cosine_keeps_sample_boundaries():
    left = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    right = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    observed = SUBJECT.per_sample_cosine(left, right)
    assert observed.tolist() == pytest.approx([1.0, -1.0])
    with pytest.raises(ValueError):
        SUBJECT.per_sample_cosine(left, right.unsqueeze(0))


def test_summary_reports_conflict_fraction_and_quantiles():
    result = SUBJECT.summarize([-1.0, -0.5, 0.5, 1.0])
    assert result["samples"] == 4
    assert result["negative"] == 2
    assert result["negative_fraction"] == pytest.approx(0.5)
    assert result["median"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        SUBJECT.summarize([])


def test_legacy_tke_floor_adds_only_missing_parallel_component():
    official = torch.tensor([[1.0, 1.0], [3.0, 1.0]])
    legacy = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    tke = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    geometry = SUBJECT.legacy_tke_floor_geometry(official, legacy, tke)
    assert geometry["official_parallel"].tolist() == pytest.approx([1.0, 3.0])
    assert geometry["legacy_parallel"].tolist() == pytest.approx([2.0, 2.0])
    assert geometry["missing_parallel"].tolist() == pytest.approx([1.0, 0.0])
    assert geometry["applied"].tolist() == [True, False]
