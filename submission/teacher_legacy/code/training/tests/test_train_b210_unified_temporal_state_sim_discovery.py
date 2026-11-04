from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_b210_unified_temporal_state_sim_discovery as subject


def test_paired_configs_change_only_the_state_mixer_axis() -> None:
    control = subject.model_config("framewise_spatial_control")
    temporal = subject.model_config("causal_temporal")

    changed = {
        key for key in control.__dataclass_fields__
        if getattr(control, key) != getattr(temporal, key)
    }
    assert changed == {"history_dynamics_state_mode"}
    assert (control.base, control.depth, control.difference_steps) == (210, 4, 5)
    assert control.multi_scale_regime_film and control.cross_conditioned_backbone
    assert (control.cross_conditioned_channels, control.cross_conditioned_steps) == (64, 5)
    assert control.member_adapter_channels == 64
    assert control.backbone_residual_dropout == .20
    assert control.physical_energy_controller and control.physical_energy_member_sync
    assert (control.history_dynamics_state_channels,
            control.history_dynamics_state_blocks,
            control.history_dynamics_state_kernel) == (32, 2, 5)


def test_light_pair_is_bitwise_identical_and_point_forward_only() -> None:
    models, identity = subject.construct_pair(torch.device("cpu"), tiny=True)

    assert identity["bitwise_identical"]
    assert models["spatial_control"].state_dict().keys() == models["causal_temporal"].state_dict().keys()
    assert all(torch.equal(left, models["causal_temporal"].state_dict()[key])
               for key, left in models["spatial_control"].state_dict().items())
    history = torch.randn(1, 20, 16, 16, 3)
    with torch.inference_mode():
        prediction = models["causal_temporal"](history)
    assert prediction.shape == (1, 20, 16, 16, 3)
    assert torch.isfinite(prediction).all()


def test_fit_normalizer_is_installed_identically_in_both_persistence_paths() -> None:
    models, _identity = subject.construct_pair(torch.device("cpu"), tiny=True)
    normalizer = subject.ChannelNormalizer(
        torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.]),
        torch.tensor([2., 3., 4.]), torch.tensor([4., 5., 8.]),
        torch.device("cpu"))

    report = subject.set_pair_normalizer_statistics(models, normalizer)

    expected_scale = torch.tensor([.5, .6, .5])
    expected_bias = torch.tensor([-.75, -.6, -.375])
    for model in models.values():
        assert torch.equal(model.persistence_scale, expected_scale)
        assert torch.equal(model.persistence_bias, expected_bias)
        assert torch.equal(model.target_std, torch.tensor([4., 5., 8.]))
    assert report["persistence_scale"] == expected_scale.tolist()


def test_gate_requires_aggregate_components_and_15_of_20_trajectories() -> None:
    delta = {key: 0.0 for key in (*subject.POINT_WEIGHTS, "point_contribution_46_10_9")}
    passing = {
        "delta_temporal_minus_spatial": {**delta, "point_contribution_46_10_9": .03},
        "positive_trajectories": 15,
        "worst_trajectory_delta_point": -.05,
        "trajectory_count": 20,
        "windows_per_trajectory": 48,
    }
    assert subject.decide_gate(passing)["pass"]
    assert not subject.decide_gate({**passing, "positive_trajectories": 14})["pass"]
    assert not subject.decide_gate({
        **passing,
        "delta_temporal_minus_spatial": {**passing["delta_temporal_minus_spatial"], "tke_rel_l2_score": -.0301},
    })["pass"]
    assert not subject.decide_gate({**passing, "worst_trajectory_delta_point": -.0501})["pass"]
    assert not subject.decide_gate({**passing, "trajectory_count": 19})["pass"]


def test_formal_run_rejects_missing_acknowledgement(tmp_path: Path) -> None:
    args = SimpleNamespace(
        execute=False, acknowledge_protocol="", sim_dir=None, output_dir=tmp_path,
        held_aoa=10, updates=2, batch_size=1)
    try:
        subject.run(args, torch.device("cpu"))
    except PermissionError as error:
        assert "acknowledgement" in str(error)
    else:
        raise AssertionError("formal run accepted without acknowledgement")
