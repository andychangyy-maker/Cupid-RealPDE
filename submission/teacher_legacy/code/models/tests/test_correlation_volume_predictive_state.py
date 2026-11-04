from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.correlation_volume_predictive_state import (  # noqa: E402
    CorrelationVolumePredictiveState, CorrelationVolumePredictiveStateConfig,
    CostBinProjection, LocalCosineCostVolume)


def config() -> CorrelationVolumePredictiveStateConfig:
    return CorrelationVolumePredictiveStateConfig(feature_channels=16, state_channels=32,
                                                   decoder_channels=24, groups=8)


def make(mode: str = "signed_candidate") -> CorrelationVolumePredictiveState:
    torch.manual_seed(210)
    return CorrelationVolumePredictiveState(config(), mode)


def test_known_shift_lands_in_its_signed_local_bin():
    torch.manual_seed(7)
    latest = F.normalize(torch.randn(1, 16, 12, 15), dim=1)
    # lagged[y+1, x-2] is latest[y, x] away from the padded boundary.
    lagged = torch.roll(latest, shifts=(1, -2), dims=(-2, -1))
    volume = LocalCosineCostVolume(CorrelationVolumePredictiveStateConfig(feature_channels=16)).one(latest, lagged)
    radius, y, x = 4, 6, 7
    assert int(volume[0, :, y, x].argmax()) == (1 + radius) * (2 * radius + 1) + (-2 + radius)


def test_radial_operator_is_symmetric_idempotent_and_preserves_groups():
    projection = CostBinProjection(lags=1, radius=4)
    radial = projection.radial_matrix
    torch.testing.assert_close(radial @ radial, radial, rtol=0., atol=0.)
    torch.testing.assert_close(radial.T, radial, rtol=0., atol=0.)
    value = torch.randn(2, 81, 3, 4)
    radial_value = projection.radialize_raw(value).reshape(2, 81, 3, 4)
    for distance in torch.unique(projection.offsets.square().sum(1)):
        indexes = torch.where(projection.offsets.square().sum(1) == distance)[0]
        group = radial_value[:, indexes]
        assert float((group - group[:, :1]).abs().max()) == 0.
        torch.testing.assert_close(group.sum(1), value[:, indexes].sum(1), rtol=0., atol=2e-6)


def test_radial_operator_is_invariant_to_square_rotations_and_reflections():
    projection = CostBinProjection(lags=1, radius=4)
    offsets = [tuple(row.tolist()) for row in projection.offsets]
    index = {offset: position for position, offset in enumerate(offsets)}
    for transform in (lambda y, x: (x, -y), lambda y, x: (y, -x)):
        permutation = torch.tensor([index[transform(y, x)] for y, x in offsets])
        transformed = projection.radial_matrix[permutation][:, permutation]
        torch.testing.assert_close(transformed, projection.radial_matrix, rtol=0., atol=0.)


def test_matched_arms_are_bitwise_identical_and_buffers_are_nonpersistent():
    signed, radial = make("signed_candidate"), make("radial_control")
    assert signed.state_dict().keys() == radial.state_dict().keys()
    assert all(torch.equal(signed.state_dict()[key], radial.state_dict()[key]) for key in signed.state_dict())
    assert not any("matrix" in key or "offsets" in key for key in signed.state_dict())


def test_first_step_gradients_reach_matching_encoder_and_direct_state():
    model = make(); x, target = torch.randn(2, 20, 32, 64, 3), torch.randn(2, 20, 32, 64, 3)
    (model(x) - target).square().mean().backward()
    for parameter in (model.encoder.network[0].weight, model.state_lift.weight, model.horizon.weight):
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad)


def test_only_last_and_fixed_lags_can_change_the_direct_prediction_path():
    model = make().eval(); x = torch.randn(1, 20, 32, 64, 3)
    changed = x.clone(); changed[:, 7].add_(100.)  # frame 7 is not {19,18,15,11,3}
    with torch.inference_mode():
        torch.testing.assert_close(model(x), model(changed), rtol=0., atol=0.)
    assert not any("history" in name or "flatten" in name for name, _ in model.named_parameters())


def test_angular_donor_requires_matched_history_and_changes_only_angular_context():
    model = make().eval(); x, donor = torch.randn(1, 20, 32, 64, 3), torch.randn(1, 20, 32, 64, 3)
    with pytest.raises(ValueError, match="donor"):
        model(x, donor[:, :-1])
    with torch.inference_mode():
        native, shuffled = model.forward_components(x), model.forward_components(x, angular_donor_history=donor)
        native_radial = model.cost_projection.radialize_raw(native["raw_cost"])
        shuffled_radial = model.cost_projection.radialize_raw(shuffled["raw_cost"])
    torch.testing.assert_close(native_radial, shuffled_radial, rtol=0., atol=2e-6)
