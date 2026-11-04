from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SUBMISSIONS = Path(__file__).resolve().parents[1]
CODE_ROOT = SUBMISSIONS.parent
for path in (CODE_ROOT, SUBMISSIONS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_track1_b210_st_phase_teacher_probe as builder  # noqa: E402


def _import_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = SUBMISSIONS / "track1_b210_st_phase_teacher_probe.py"
    spec = importlib.util.spec_from_file_location("phase_teacher_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _twin_state(value: float = 1.0) -> dict[str, torch.Tensor]:
    return {
        "stem.weight": torch.tensor([value]),
        "stage0.0.source_bank.axis.weight": torch.tensor([value + 1]),
        "stage0.0.real_bank.axis.weight": torch.tensor([value + 1]),
        "stage0.0.source_bank.channel.weight": torch.tensor([value + 2]),
        "stage0.0.real_bank.channel.weight": torch.tensor([value + 2]),
        "stage0.0.spatial.weight": torch.tensor([value + 3]),
    }


def test_collapse_is_lossless_and_removes_second_bank() -> None:
    collapsed, audit = builder.collapse_identical_temporal_banks(_twin_state())
    assert tuple(collapsed) == (
        "stem.weight",
        "stage0.0.axis.weight",
        "stage0.0.channel.weight",
        "stage0.0.spatial.weight",
    )
    assert torch.equal(collapsed["stage0.0.axis.weight"], torch.tensor([2.0]))
    assert audit["matched_bank_tensors"] == 2
    assert audit["twin_tensors"] == 6
    assert audit["collapsed_tensors"] == 4
    assert audit["bitwise_equal"] is True


def test_collapse_rejects_nonidentical_temporal_banks() -> None:
    state = _twin_state()
    state["stage0.0.real_bank.axis.weight"][0] += 1.0
    with pytest.raises(ValueError, match="not bitwise equal"):
        builder.collapse_identical_temporal_banks(state)


def test_seed_soup_is_fixed_equal_weight_fp32() -> None:
    left = {"weight": torch.tensor([1.0, 3.0], dtype=torch.float16)}
    right = {"weight": torch.tensor([3.0, 7.0], dtype=torch.float16)}
    soup = builder.equal_weight_fp32_soup(left, right)
    assert soup["weight"].dtype == torch.float32
    assert torch.equal(soup["weight"], torch.tensor([2.0, 5.0]))


def test_isoenergetic_projection_restores_mean_and_rms(monkeypatch) -> None:
    runtime = _import_runtime(monkeypatch)
    generator = torch.Generator().manual_seed(210)
    reference = torch.randn(3, 20, 4, 5, 2, generator=generator)
    candidate = torch.randn(3, 20, 4, 5, 2, generator=generator)
    projected = runtime.restore_temporal_mean_rms(candidate, reference)
    assert torch.allclose(
        projected.mean(dim=1), reference.mean(dim=1), atol=2e-7, rtol=1e-6
    )
    projected_rms = (
        projected - projected.mean(dim=1, keepdim=True)
    ).square().mean(dim=1).sqrt()
    reference_rms = (
        reference - reference.mean(dim=1, keepdim=True)
    ).square().mean(dim=1).sqrt()
    assert torch.allclose(projected_rms, reference_rms, atol=2e-7, rtol=1e-6)


def test_isoenergetic_projection_identity(monkeypatch) -> None:
    runtime = _import_runtime(monkeypatch)
    reference = torch.randn(2, 20, 3, 4, 2)
    projected = runtime.restore_temporal_mean_rms(reference, reference)
    assert torch.allclose(projected, reference, atol=3e-7, rtol=1e-6)


class _BaseStub:
    def forward_distribution(self, value):
        extent = torch.full((*value.shape[:-1], 2), 0.25, dtype=value.dtype)
        return value, extent, 2.0 * extent


class _STStub:
    def __call__(self, value):
        return value.flip(dims=(1,))


def test_bounds_are_bitwise_passthrough_and_predict_is_stateless(monkeypatch) -> None:
    runtime = _import_runtime(monkeypatch)
    zeros = torch.zeros(3)
    ones = torch.ones(3)
    runtime._state.update(
        {
            "base_model": _BaseStub(),
            "st_model": _STStub(),
            "device": torch.device("cpu"),
            "dtype": torch.float32,
            "base_mean_input": zeros,
            "base_mean_target": zeros,
            "base_std_input": ones,
            "base_std_target": ones,
            "st_mean_input": zeros,
            "st_mean_target": zeros,
            "st_std_input": ones,
            "st_std_target": ones,
        }
    )
    source = np.random.default_rng(7).normal(size=(2, 20, 8, 8, 3)).astype(
        np.float32
    )
    source[..., 2] = 0.0
    first = runtime.predict(source, metadata={})
    repeated = runtime.predict(source, metadata={"ignored": True})
    changed_input = source.copy()
    changed_input[:, -1, ..., 0] += 0.125
    changed = runtime.predict(changed_input, metadata={})

    expected_lower = source.copy()
    expected_upper = source.copy()
    expected_lower[..., :2] -= 0.25
    expected_upper[..., :2] += 0.5
    assert np.array_equal(first["lower"], expected_lower)
    assert np.array_equal(first["upper"], expected_upper)
    assert np.array_equal(first["lower"], repeated["lower"])
    assert np.array_equal(first["upper"], repeated["upper"])
    assert np.array_equal(first["prediction"], repeated["prediction"])
    assert not np.array_equal(first["prediction"], changed["prediction"])
    assert np.count_nonzero(first["prediction"][..., 2]) == 0


def test_archive_limit_is_strictly_below_256_mib() -> None:
    builder.validate_archive_size_bytes(256 * 1024**2 - 1, 256.0)
    with pytest.raises(RuntimeError, match="not below"):
        builder.validate_archive_size_bytes(256 * 1024**2, 256.0)
