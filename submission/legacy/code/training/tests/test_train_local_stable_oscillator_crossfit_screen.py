from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_local_stable_oscillator_crossfit_screen as screen  # noqa: E402


def inventory():
    return [{"file": f"a{aoa}_re{re:03d}.h5", "aoa": aoa, "re": re,
             "frames": 1, "shape": [1, 1, 1]}
            for aoa in (0, 5, 10, 15, 20) for re in range(100, 120)]


def test_fold_is_reynolds_disjoint_across_all_allowed_aoas_and_never_selects_aoa10():
    train, held, manifest = screen.select_fold(inventory(), 0)
    assert len(train) == 60 and len(held) == 20
    assert {row["aoa"] for row in train + held} == set(screen.ALLOWED_AOAS)
    assert 10 not in {row["aoa"] for row in train + held}
    assert {row["re"] for row in train}.isdisjoint({row["re"] for row in held})
    assert all(len(values) == 5 for values in manifest["held_re_by_aoa"].values())
    assert manifest["held_re_by_aoa"]["0"] == manifest["held_re_by_aoa"]["20"]


def test_every_fold_has_exactly_five_common_held_reynolds_numbers():
    for fold in range(4):
        train, held, _ = screen.select_fold(inventory(), fold)
        assert len(train) == 60 and len(held) == 20
        by_aoa = [{row["re"] for row in held if row["aoa"] == aoa} for aoa in screen.ALLOWED_AOAS]
        assert all(values == by_aoa[0] for values in by_aoa)


def test_arms_are_bitwise_matched_and_only_temporal_family_changes():
    arms, identity = screen.construct_arms(torch.device("cpu"))
    assert tuple(arms) == screen.ARM_NAMES
    assert identity["bitwise_identical"] is True
    assert identity["only_variable"] == "fixed_stable_temporal_family"
    assert len(set(identity["parameters"].values())) == 1


class UnitNormalizer:
    mean_input = torch.zeros(1, 1, 1, 1, 3)
    std_input = torch.ones(1, 1, 1, 1, 3)
    mean_target = torch.zeros(1, 1, 1, 1, 3)
    std_target = torch.ones(1, 1, 1, 1, 3)


def test_loss_has_finite_gradient_to_history_and_modal_rate_heads():
    model = screen.construct_arms(torch.device("cpu"))[0]["complex_candidate"]
    loss, terms = screen.loss_terms(model, torch.randn(1, 20, 32, 64, 3), torch.randn(1, 20, 32, 64, 3), UnitNormalizer())
    loss.backward()
    assert np.isfinite([float(value.detach()) for value in terms.values()]).all()
    assert torch.count_nonzero(model.frame_stem[0].weight.grad)
    assert torch.count_nonzero(model.rate_head[-1].bias.grad)


def test_sealed_endpoint_reload_is_exact_and_zero_frequency_override_is_available(tmp_path: Path):
    model = screen.construct_arms(torch.device("cpu"))[0]["complex_candidate"].eval()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    path = tmp_path / "candidate.pth"
    torch.save({"schema": screen.SCHEMA, "arm": "complex_candidate", "updates": screen.UPDATES,
                "model_config": model.architecture_config(), "model_state_dict": state}, path)
    endpoint = {"path": str(path), "sha256": screen.io.sha256_file(path), "model_state_sha256": screen.io.state_sha(state)}
    loaded = screen.reload_sealed_arm("complex_candidate", endpoint, torch.device("cpu"))
    x = torch.randn(1, 20, 32, 64, 3)
    with torch.inference_mode():
        torch.testing.assert_close(model(x), loaded(x), rtol=0., atol=0.)
        assert loaded.forward_components(x, temporal_family="zero_frequency")["prediction"].shape == (1, 20, 32, 64, 3)


def valid_evaluation():
    return {"complex_candidate_minus_real_control": {"delta": {"point_contribution_46_10_9": .08}, "positive_trajectory_files": 15, "worst_trajectory_delta_point": -.05},
            "horizon_block_deltas": {
                "horizons_5_8": {"point_contribution_46_10_9": 0.},
                "horizons_9_12": {"point_contribution_46_10_9": 0.},
                "horizons_5_12": {"point_contribution_46_10_9": .04}},
            "same_checkpoint_native_minus_zero_frequency": {"point_contribution_46_10_9": .02},
            "same_checkpoint_native_minus_zero_frequency_horizons_5_12": {"point_contribution_46_10_9": .01}}


def test_gate_checks_pair_middle_horizon_coverage_worst_case_and_same_checkpoint_ablation():
    assert screen.gate(valid_evaluation())[0] is True
    for section, key, value in (("complex_candidate_minus_real_control", "positive_trajectory_files", 14),
                                ("complex_candidate_minus_real_control", "worst_trajectory_delta_point", -.051),
                                ("horizon_block_deltas", "horizons_5_12", .039),
                                ("horizon_block_deltas", "horizons_5_8", -.001),
                                ("horizon_block_deltas", "horizons_9_12", -.001),
                                ("same_checkpoint_native_minus_zero_frequency", "point_contribution_46_10_9", .019),
                                ("same_checkpoint_native_minus_zero_frequency_horizons_5_12", "point_contribution_46_10_9", .009)):
        broken = valid_evaluation()
        if section == "horizon_block_deltas": broken[section][key]["point_contribution_46_10_9"] = value
        else: broken[section][key] = value
        assert screen.gate(broken)[0] is False


def test_schedule_is_exactly_shared_stream_shape():
    schedule = screen.make_schedule([object()] * 17)
    assert schedule.shape == (screen.UPDATES, screen.BATCH_SIZE)
    assert int(schedule.min()) >= 0 and int(schedule.max()) < 17


def test_history_order_ablation_reverses_only_preterminal_frames():
    history = torch.arange(2 * 20 * 3).reshape(2, 20, 1, 1, 3)
    reversed_history = screen.reverse_preterminal_history(history)
    assert torch.equal(reversed_history[:, :19], history[:, :19].flip(1))
    assert torch.equal(reversed_history[:, 19], history[:, 19])
    assert torch.equal(reversed_history.sort(dim=1).values, history.sort(dim=1).values)


def test_rejects_missing_or_nonunique_allowed_reynolds():
    bad = inventory()[:-1]
    with pytest.raises(ValueError, match="expected 80 allowed records|20 unique Re"):
        screen.select_fold(bad, 0)
