from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

import train_b210_causal_residual_soft_gate_lifecycle as subject


class _Normalizer:
    mean_input = torch.tensor([2.0, -1.0, 0.5])
    mean_target = torch.tensor([3.0, -2.0, 0.25])
    std_input = torch.tensor([4.0, 2.0, 3.0])
    std_target = torch.tensor([5.0, 6.0, 7.0])


def test_contract_has_no_gate_or_clean716_selection():
    contract = subject.experiment_contract()
    assert contract["architecture"]["hard_replacement"] is False
    assert contract["architecture"]["native_interval_width"] == (
        "preserved; only centre may move")
    assert contract["matched_pair"]["same_parameter_shapes"] is True
    assert contract["selection"] is None
    assert contract["clean716_read"] is False
    assert contract["optimization"]["point_loss"].startswith("published 46/10/9")
    assert contract["optimization"]["soft_sps_weight"] == pytest.approx(0.05)


def test_matched_initialization_is_bitwise_and_expected_size():
    report = subject.common_initialization_report()
    assert report["bitwise_equal"] is True
    assert report["parameters"] == subject.EXPECTED_FUSION_PARAMETERS
    assert len(report["tensor_state_sha256"]) == 64


def test_sim_mapping_preserves_structural_zeros_and_maps_zscores():
    sim = _Normalizer()
    real = type("Real", (), {
        "mean_input": torch.tensor([10.0, 20.0, 0.0]),
        "mean_target": torch.tensor([30.0, 40.0, 0.0]),
        "std_input": torch.tensor([8.0, 6.0, 1.0]),
        "std_target": torch.tensor([10.0, 12.0, 1.0]),
    })()
    history = torch.tensor([[[[[0.0, 1.0, 9.0], [6.0, 0.0, 9.0]]]]])
    target = torch.tensor([[[[[0.0, 4.0, 9.0], [8.0, 0.0, 9.0]]]]])
    mapped_history, mapped_target = subject.map_sim_to_real_coordinates(
        history, target, sim, real)
    assert mapped_history[0, 0, 0, 0, 0] == 0.0
    assert mapped_history[0, 0, 0, 1, 1] == 0.0
    assert mapped_target[0, 0, 0, 0, 0] == 0.0
    assert mapped_target[0, 0, 0, 1, 1] == 0.0
    assert mapped_history[..., 2].count_nonzero() == 0
    assert mapped_target[..., 2].count_nonzero() == 0
    assert mapped_history[0, 0, 0, 1, 0] == pytest.approx(18.0)
    assert mapped_target[0, 0, 0, 1, 0] == pytest.approx(40.0)


def test_fusion_config_roundtrip_rejects_metadata_change():
    model = subject.build_fusion("causal_temporal", torch.device("cpu"))
    parsed = subject.parse_fusion_config(model.architecture_config())
    assert asdict(parsed) == asdict(model.config)
    broken = model.architecture_config()
    broken["hard_replacement_available"] = True
    with pytest.raises(ValueError, match="hard_replacement_available"):
        subject.parse_fusion_config(broken)


def test_formal_cli_requires_explicit_acknowledgement(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = subject.parse_args([
        "--mode", "sim", "--base-checkpoint", str(tmp_path / "base.pt"),
        "--base-checkpoint-sha256", "0" * 64,
        "--output-dir", str(tmp_path / "out"), "--device", "cuda:0",
    ])
    with pytest.raises(PermissionError, match="--execute"):
        subject.validate_cli(args)
