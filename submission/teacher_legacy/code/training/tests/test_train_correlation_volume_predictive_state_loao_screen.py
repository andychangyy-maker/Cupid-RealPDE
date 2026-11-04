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

import train_correlation_volume_predictive_state_loao_screen as screen  # noqa: E402


def test_arms_are_bitwise_matched_and_projected_geometry_is_the_only_variable():
    arms, identity = screen.construct_arms(torch.device("cpu"))
    assert tuple(arms) == screen.ARM_NAMES
    assert identity["bitwise_identical"] is True
    assert set(identity["parameters"].values()) == {screen.EXPECTED_PARAMETERS}
    assert identity["only_variable"] == "post_projection_bin_geometry"


def test_schedule_is_paired_and_fit_only():
    schedule = screen.make_schedule([object()] * 31)
    assert schedule.shape == (screen.UPDATES, screen.BATCH_SIZE)
    assert int(schedule.min()) >= 0 and int(schedule.max()) < 31


def test_raw_volume_diagnostics_are_reported_separately_for_every_motion_lag():
    raw = torch.randn(2, 3 * 81, 5, 7)
    result = screen.raw_volume_diagnostics(raw)
    assert tuple(result["by_lag"]) == ("4", "8", "16")
    for row in result["by_lag"].values():
        assert len(row["argmax_histogram"]) == 81
        assert sum(row["argmax_histogram"]) == 2 * 5 * 7
        assert 0. <= row["argmax_noncenter_fraction"] <= 1.


def test_angular_donor_mapping_uses_next_trajectory_and_identical_start():
    windows = [screen.backbone.common.scratch.SimWindow(trajectory, start)
               for trajectory in range(3) for start in (0, 20)]
    assert screen.angular_donor_indexes(windows, 3) == [2, 3, 4, 5, 0, 1]


def test_angular_region_diagnostic_reports_padding_free_interior():
    model = screen.construct_arms(torch.device("cpu"))[0]["signed_candidate"]
    raw = torch.randn(2, 3 * 81, 16, 32)
    result = screen.angular_region_rms(raw, model)
    assert result["all"] > 0. and result["interior"] > 0. and result["boundary"] > 0.
    assert result["interior_pixel_fraction"] == pytest.approx(0.375)


class UnitNormalizer:
    mean_input = torch.zeros(1, 1, 1, 1, 3)
    std_input = torch.ones(1, 1, 1, 1, 3)
    mean_target = torch.zeros(1, 1, 1, 1, 3)
    std_target = torch.ones(1, 1, 1, 1, 3)


def test_loss_is_finite_and_matching_gets_first_backward_gradient():
    model = screen.construct_arms(torch.device("cpu"))[0]["signed_candidate"]
    loss, terms = screen.loss_terms(model, torch.randn(2, 20, 32, 64, 3),
                                    torch.randn(2, 20, 32, 64, 3), UnitNormalizer())
    loss.backward()
    assert np.isfinite([float(value.detach()) for value in terms.values()]).all()
    assert torch.count_nonzero(model.encoder.network[0].weight.grad)


def test_sealed_endpoint_is_strictly_reconstructed_before_evaluation(tmp_path: Path):
    model = screen.construct_arms(torch.device("cpu"))[0]["signed_candidate"].eval()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    path = tmp_path / "signed_candidate_endpoint.pth"
    torch.save({"schema": screen.SCHEMA, "arm": "signed_candidate",
                "updates": screen.UPDATES, "model_config": model.architecture_config(),
                "model_state_dict": state}, path)
    endpoint = {"path": str(path), "sha256": screen.sha256_file(path),
                "model_state_sha256": screen.state_sha(state)}
    loaded = screen.reload_sealed_arm("signed_candidate", endpoint, torch.device("cpu"))
    x = torch.randn(1, 20, 32, 64, 3)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(x), model(x), rtol=0., atol=0.)
    assert loaded.training is False


def test_sealed_normalizer_is_strictly_reconstructed(tmp_path: Path):
    state = {"mean_input": torch.randn(1, 1, 1, 1, 3),
             "mean_target": torch.randn(1, 1, 1, 1, 3),
             "std_input": torch.rand(1, 1, 1, 1, 3) + .1,
             "std_target": torch.rand(1, 1, 1, 1, 3) + .1}
    path = tmp_path / "normalizer.pt"
    torch.save({"schema": screen.SCHEMA, **state}, path)
    artifact = {"path": str(path), "sha256": screen.sha256_file(path),
                "state_sha256": screen.state_sha(state)}
    loaded = screen.reload_sealed_normalizer(artifact, torch.device("cpu"))
    for key, expected in state.items():
        torch.testing.assert_close(getattr(loaded, key), expected, rtol=0., atol=0.)


def valid_evaluation() -> dict[str, object]:
    delta = {"rel_l2_score": .09, "tke_score": .01, "mvpe_score": -.02,
             "point_contribution_46_10_9": .081}
    return {"signed_candidate_minus_radial_control": {"delta": delta,
            "positive_trajectory_files": 15, "worst_trajectory_delta_point": -.05},
            "same_checkpoint_signed_minus_radial_diagnostic": {"point_contribution_46_10_9": -.3},
            "same_checkpoint_native_minus_angular_residual_shuffle": {
                "delta": {"point_contribution_46_10_9": .02},
                "positive_trajectory_files": 15},
            "mechanical": {"raw_vs_radial_directional_rms": 1.1e-4,
                            "raw_vs_radial_directional_rms_interior": 1.1e-4,
                            "normalized_projected_rms": {"radial_control": 1., "signed_candidate": 1.}}}


def test_gate_requires_primary_quality_and_angular_shuffle_but_not_radial_override():
    assert screen.gate(valid_evaluation())[0] is True
    broken = valid_evaluation(); broken["same_checkpoint_native_minus_angular_residual_shuffle"]["delta"]["point_contribution_46_10_9"] = .019
    assert screen.gate(broken)[0] is False
    broken = valid_evaluation(); broken["mechanical"]["raw_vs_radial_directional_rms_interior"] = 1e-4
    assert screen.gate(broken)[0] is False
    contract = screen.gate(valid_evaluation())[1]
    assert "signed_minus_radial" not in " ".join(contract)


def test_held20_rejects_forged_go_before_artifact_access(tmp_path: Path):
    path = tmp_path / "fake.json"
    path.write_text('{"schema":"correlation_volume_predictive_state_whole_aoa_sim_screen_v1","decision":"GO_AOA20_ROBUSTNESS"}')
    with pytest.raises(PermissionError, match="verified held10 GO"):
        screen.prior_held10_evidence(path, 20)
