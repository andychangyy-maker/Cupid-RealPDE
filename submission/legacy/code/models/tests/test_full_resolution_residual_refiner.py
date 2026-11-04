from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.full_resolution_residual_refiner import (
    FullResolutionResidualRefiner,
    FullResolutionResidualRefinerConfig,
    phase_pack_2,
    phase_unpack_2,
)


def _config(execution_mode: str) -> FullResolutionResidualRefinerConfig:
    return FullResolutionResidualRefinerConfig(
        steps=5, hidden_channels=32, blocks=3, execution_mode=execution_mode)


def _paired_models() -> tuple[FullResolutionResidualRefiner,
                              FullResolutionResidualRefiner]:
    torch.manual_seed(610)
    full = FullResolutionResidualRefiner(_config("full_resolution")).eval()
    torch.manual_seed(610)
    phase = FullResolutionResidualRefiner(_config("phase_pack_2")).eval()
    return full, phase


def test_phase_pack_2_is_a_reversible_permutation():
    future = torch.arange(2 * 5 * 6 * 8 * 3, dtype=torch.float32).reshape(
        2, 5, 6, 8, 3)
    packed = phase_pack_2(future)
    assert packed.shape == (8, 5, 3, 4, 3)
    restored = phase_unpack_2(packed, batch_size=2)
    assert torch.equal(restored, future)
    impulses = torch.zeros(1, 5, 6, 8, 3)
    for phase, (phase_y, phase_x) in enumerate(
            ((0, 0), (0, 1), (1, 0), (1, 1))):
        impulses[0, phase, 2 + phase_y, 2 + phase_x, 0] = phase + 1
    packed_impulses = phase_pack_2(impulses)
    assert torch.equal(
        phase_unpack_2(packed_impulses, batch_size=1), impulses)
    for phase in range(4):
        assert packed_impulses[phase, phase, 1, 1, 0] == phase + 1
    with pytest.raises(ValueError, match="even H and W"):
        phase_pack_2(future[:, :, :-1])


def test_full_and_phase_pack_are_a_strict_update_zero_control():
    full, phase = _paired_models()
    assert full.state_dict().keys() == phase.state_dict().keys()
    assert len(full.state_dict()) == len(phase.state_dict())
    assert all(torch.equal(value, phase.state_dict()[key])
               for key, value in full.state_dict().items())
    assert len(full.state_dict()) == len(phase.state_dict()) == 22
    assert sum(parameter.numel() for parameter in full.parameters()) == sum(
        parameter.numel() for parameter in phase.parameters()) == 8_771
    full_config = full.architecture_config()
    phase_config = phase.architecture_config()
    differing = [key for key in full_config
                 if full_config[key] != phase_config[key]]
    assert differing == ["execution_mode"]
    differing_architecture = {
        key for key in full.architecture_config()
        if full.architecture_config()[key] != phase.architecture_config()[key]
    }
    assert differing_architecture == {"execution_mode"}
    assert full.output.weight.count_nonzero() == 0
    assert full.output.bias.count_nonzero() == 0

    future = torch.randn(2, 5, 6, 8, 3)
    with torch.inference_mode():
        full_correction = full.correction(future)
        phase_correction = phase.correction(future)
        full_prediction = full(future)
        phase_prediction = phase(future)
    assert torch.count_nonzero(full_correction) == 0
    assert torch.count_nonzero(phase_correction) == 0
    assert torch.equal(full_prediction, future)
    assert torch.equal(phase_prediction, future)
    assert torch.equal(full_prediction, phase_prediction)

    full_contract = full.last_execution_contract
    phase_contract = phase.last_execution_contract
    assert full_contract is not None and phase_contract is not None
    assert full_contract["feature_elements"] == phase_contract["feature_elements"]
    assert full_contract["theoretical_macs"] == phase_contract["theoretical_macs"]
    assert full_contract["canonical_shape"] == phase_contract["canonical_shape"]
    assert full_contract["effective_shape"] != phase_contract["effective_shape"]


@pytest.mark.parametrize("mode", ("full_resolution", "phase_pack_2"))
def test_refiner_has_no_cross_batch_path(mode: str):
    model = FullResolutionResidualRefiner(_config(mode)).eval()
    # Exercise the non-identity graph: update-zero alone cannot detect an
    # accidental reduction or normalization across packed batch examples.
    with torch.no_grad():
        model.output.weight.normal_(mean=0.0, std=0.1)
        model.output.bias.uniform_(-0.1, 0.1)
    future = torch.randn(2, 5, 6, 8, 3)
    changed = future.clone()
    changed[0] += 7.0 * torch.randn_like(changed[0])
    with torch.inference_mode():
        baseline = model(future)
        perturbed = model(changed)
    assert torch.equal(baseline[1], perturbed[1])


def test_contract_validation_and_bounded_correction():
    full, phase = _paired_models()
    future = torch.randn(2, 5, 6, 8, 3)
    with torch.no_grad():
        full.output.weight.normal_(mean=0.0, std=0.2)
        phase.load_state_dict(full.state_dict(), strict=True)
    correction = full.correction(future)
    assert float(correction.abs().max()) <= 4.0
    assert phase(future).shape == future.shape
    with pytest.raises(ValueError, match="expected future"):
        full(torch.randn(2, 4, 6, 8, 3))
    with pytest.raises(ValueError, match="execution_mode"):
        FullResolutionResidualRefinerConfig(execution_mode="bad").validate()
