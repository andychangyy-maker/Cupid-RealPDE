from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / (
    "audit_sim_real_disagreement_transfer.py")
SPEC = importlib.util.spec_from_file_location("transfer_audit", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


def test_spatial_zscore_is_per_sample_horizon_channel():
    value = torch.arange(2 * 3 * 4 * 5 * 2, dtype=torch.float32).reshape(
        2, 3, 4, 5, 2)
    normalized = subject.spatial_zscore(value)
    assert normalized.mean(dim=(2, 3)).abs().max() < 1.0e-5
    assert torch.allclose(
        normalized.square().mean(dim=(2, 3)),
        torch.ones(2, 3, 2), atol=1.0e-5, rtol=1.0e-5)


def test_top_fraction_diagnostic_finds_exact_top_decile():
    error = np.arange(100, dtype=np.float64)
    good = subject.top_fraction_diagnostic(error.copy(), error)
    bad = subject.top_fraction_diagnostic(-error, error)
    assert good["top_error_recall"] == pytest.approx(1.0)
    assert good["error_amplification"] > 1.8
    assert bad["top_error_recall"] == pytest.approx(0.0)


def test_cross_map_diagnostic_uses_real_error_as_transfer_target():
    real_error = np.linspace(0.0, 2.0, 100)
    sim = {
        "spread_z": real_error.copy(),
        "adapter_z": -real_error,
        "error": -real_error + 3.0,
    }
    real = {
        "spread_z": -real_error,
        "adapter_z": real_error.copy(),
        "error": real_error,
    }
    result = subject.cross_map_diagnostic(sim, real)
    assert result["sim_total_spread_z_vs_real_error_spearman"] > 0.99
    assert result["real_total_spread_z_vs_real_error_spearman"] < -0.99
    assert result["sim_adapter_spread_z_vs_real_error_spearman"] < -0.99
    assert result["real_adapter_spread_z_vs_real_error_spearman"] > 0.99
    assert result["sim_total_for_real_error_top10"][
        "top_error_recall"] == pytest.approx(1.0)
