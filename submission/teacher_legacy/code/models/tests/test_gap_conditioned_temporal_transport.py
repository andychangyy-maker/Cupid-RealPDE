from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.gap_conditioned_temporal_transport import (  # noqa: E402
    GapConditionedTemporalTransportOperator,
)
from models.temporal_transport import (  # noqa: E402
    TemporalTransportConfig,
    TemporalTransportOperator,
)


def config(**overrides) -> TemporalTransportConfig:
    values = dict(
        in_steps=6, out_steps=5, in_channels=3, out_channels=3,
        hidden_channels=8, spatial_blocks=1, spatial_downsample=2,
        temporal_blocks=2, temporal_kernel=3, groups=4,
        max_displacement=2.0)
    values.update(overrides)
    return TemporalTransportConfig(**values)


def test_zero_gap_is_exactly_the_unconditioned_competition_path():
    torch.manual_seed(7)
    model = GapConditionedTemporalTransportOperator(config(), max_gap_steps=20).eval()
    history = torch.randn(2, 6, 7, 11, 3)
    with torch.inference_mode():
        ordinary = model(history)
        explicit_zero = model(history, torch.zeros(2, dtype=torch.long))
        scalar_zero = model(history, torch.tensor(0))
    torch.testing.assert_close(ordinary, explicit_zero, rtol=0.0, atol=0.0)
    torch.testing.assert_close(ordinary, scalar_zero, rtol=0.0, atol=0.0)


def test_nonzero_gap_changes_query_and_preserves_forecast_shape():
    torch.manual_seed(8)
    model = GapConditionedTemporalTransportOperator(config(), max_gap_steps=20).eval()
    history = torch.randn(2, 6, 7, 11, 3)
    with torch.inference_mode():
        zero = model(history)
        shifted = model(history, torch.tensor([5, 20]))
    assert shifted.shape == zero.shape == (2, 5, 7, 11, 3)
    assert torch.isfinite(shifted).all()
    assert not torch.equal(zero, shifted)

    with torch.inference_mode():
        zero_parts = model.forward_components(history)
        shifted_parts = model.forward_components(
            history, torch.tensor([5, 20]))
    assert not torch.equal(zero_parts.mean_delta, shifted_parts.mean_delta)


def test_conditioning_adds_only_the_declared_small_projection():
    torch.manual_seed(9)
    base = TemporalTransportOperator(config())
    torch.manual_seed(9)
    conditioned = GapConditionedTemporalTransportOperator(config(), max_gap_steps=20)
    difference = (sum(parameter.numel() for parameter in conditioned.parameters())
                  - sum(parameter.numel() for parameter in base.parameters()))
    assert difference == 3 * conditioned.config.hidden_channels
    assert conditioned.horizon_decoder.gap_projection.bias is None


def test_gap_validation_and_architecture_roundtrip_metadata():
    model = GapConditionedTemporalTransportOperator(config(), max_gap_steps=20)
    history = torch.randn(2, 6, 7, 11, 3)
    with pytest.raises(ValueError, match="outside"):
        model(history, torch.tensor([0, 21]))
    with pytest.raises(ValueError, match="shape"):
        model(history, torch.zeros(2, 1))
    stored = model.architecture_config()
    assert stored["architecture"] == "gap_conditioned_temporal_transport"
    assert stored["max_gap_steps"] == 20
