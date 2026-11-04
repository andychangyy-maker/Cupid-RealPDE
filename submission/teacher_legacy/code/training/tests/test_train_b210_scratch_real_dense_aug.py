from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_scratch_real_dense_aug as subject  # noqa: E402


def test_candidate_changes_data_support_and_input_only():
    assert subject.CONTROL_STRIDE == 20
    assert subject.CANDIDATE_STRIDE == 5
    assert subject.AUGMENTATION == {
        "input_gain_std": 0.01,
        "input_noise_std": 0.01,
        "input_blur_probability": 0.15,
        "input_blur_strength": 0.20,
    }


def test_default_arm_augmentation_contract_is_not_aliased():
    value = subject.arm_augmentation_contract()
    assert value == {"control": {}, "candidate": subject.AUGMENTATION}
    value["candidate"]["input_gain_std"] = 999.0
    assert subject.AUGMENTATION["input_gain_std"] == 0.01


def test_horizon_local_architecture_contract_is_explicit():
    base_config, base_total, base_controller = (
        subject.architecture_contract(False))
    horizon_config, horizon_total, horizon_controller = (
        subject.architecture_contract(True))
    assert base_config.physical_energy_horizon_local is False
    assert horizon_config.physical_energy_horizon_local is True
    assert base_total == subject.ft12.EXPECTED_PARAMETERS
    assert base_controller == subject.ft12.EXPECTED_CONTROLLER_PARAMETERS
    assert horizon_total == (
        subject.sim_tke.EXPECTED_HORIZON_LOCAL_ENERGY_PARAMETERS)
    assert horizon_controller == (
        subject.sim_tke.EXPECTED_HORIZON_LOCAL_ENERGY_CONTROLLER_PARAMETERS)
    multiscale_config, multiscale_total, multiscale_controller = (
        subject.architecture_contract(False, True))
    assert multiscale_config.physical_energy_multiscale is True
    assert multiscale_total == (
        subject.sim_tke.EXPECTED_MULTISCALE_ENERGY_PARAMETERS)
    assert multiscale_controller == (
        subject.sim_tke.EXPECTED_MULTISCALE_ENERGY_CONTROLLER_PARAMETERS)
    with pytest.raises(ValueError, match="only one"):
        subject.architecture_contract(True, True)


def test_dense_windows_never_cross_prefix_boundary():
    trajectories = [torch.zeros(100, 2, 2, 2), torch.zeros(83, 2, 2, 2)]
    windows = subject.dense_windows(trajectories, 5)
    assert len(windows) == 13 + 9
    assert all(
        window.start + 40 <= trajectories[window.trajectory].shape[0]
        for window in windows)
    assert {window.start % 5 for window in windows} == {0}


def test_default_arm_window_contract_preserves_manifest_control():
    window_type = subject.ft12.npt1.ensemble_train.common.repro.Window
    trajectories = [torch.zeros(100, 2, 2, 2)]
    manifest = [window_type(0, 0), window_type(0, 20)]
    value = subject.arm_window_contract(trajectories, manifest)
    assert [(row.trajectory, row.start) for row in value["control"]] == [
        (0, 0), (0, 20)]
    assert [(row.trajectory, row.start) for row in value["candidate"]] == [
        (0, start) for start in range(0, 61, 5)]


def test_evaluation_manifest_must_stay_after_prefix(tmp_path):
    fit = [{
        "file": "a.h5", "frames": 100, "re": 5000, "aoa": 0,
        "train_starts": [0, 20], "val_boundary": 60,
    }]
    good = [{**fit[0], "val_starts": [60]}]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    monkey = subject.TAIL_WINDOWS
    subject.TAIL_WINDOWS = 1
    try:
        rows = subject.load_evaluation_rows(path, fit)
        assert rows[0]["train_starts"] == [60]
        assert rows[0]["val_boundary"] == 100
        bad = [{**fit[0], "val_starts": [40]}]
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="valid fixed tail"):
            subject.load_evaluation_rows(path, fit)
    finally:
        subject.TAIL_WINDOWS = monkey


def test_comparison_reports_additive_metric_deltas():
    control = {
        "rel_l2_score": 90.0, "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 91.0, "point_score": 86.8,
        "native_sps_score": 40.0, "native_sps_coverage": 0.8,
        "native_sps_mean_width": 0.01,
    }
    candidate = dict(control)
    candidate["rel_l2_score"] += 1.0
    candidate["point_score"] += 0.5
    row = subject.compare(candidate, control)
    assert row["delta"]["rel_l2_score"] == pytest.approx(1.0)
    assert row["candidate_point_improved"] is True
    assert row["candidate_non_time_proxy_delta"] == pytest.approx(0.5)


def test_group_window_indices_by_aoa_preserves_partition_order():
    window_type = subject.ft12.npt1.ensemble_train.common.repro.Window
    metadata = [{"aoa": 0}, {"aoa": 5}, {"aoa": 0},
                {"aoa": 10}, {"aoa": 15}, {"aoa": 20}]
    windows = [window_type(2, 0), window_type(0, 20), window_type(1, 0),
               window_type(3, 0), window_type(4, 0), window_type(5, 0)]
    grouped = subject.group_window_indices_by_aoa(windows, metadata)
    assert grouped == {0: [0, 1], 5: [2], 10: [3], 15: [4], 20: [5]}


def test_arm_seal_is_single_assignment():
    seal = subject.ArmSeal("candidate")
    seal.seal({"path": "a", "file_sha256": "b"})
    with pytest.raises(RuntimeError, match="sealed twice"):
        seal.seal({"path": "c", "file_sha256": "d"})
