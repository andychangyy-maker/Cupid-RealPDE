from __future__ import annotations

from pathlib import Path
import inspect
import sys
import types

import pytest
import torch

TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_local_lag_covariance_loao as subject  # noqa: E402


def test_contract_has_the_fixed_formal_and_smoke_shapes():
    assert subject.EXPERIMENT.startswith("b210_m13_")
    assert subject.ENDPOINT_SCHEMA.startswith("b210_m13_")
    assert subject.FORMAL_UPDATES == 1_920
    assert subject.FORMAL_BATCH_SIZE == 4
    assert subject.SMOKE_UPDATES == 24
    assert subject.AUX_GRADIENT_FRACTION == pytest.approx(0.10)
    assert subject.AUX_WARMUP_UPDATES == 12
    assert subject.AUX_CALIBRATION_PANEL_UPDATES == 12
    assert subject.AUX_CALIBRATION_QUANTILE == pytest.approx(0.90)
    assert subject.LR == pytest.approx(2e-4)
    assert subject.FORMAL_SCHEDULE_SHA256 == (
        "17ee224e0d169d49273cb307c48ffc21f505586148670cbae1593458c7999082")
    assert subject.BASE_FORMAL_SCHEDULE_SHA256 == (
        "e69cf7cdc6ecaf5f80353b76e3221910bae6c01891c0d339db84c0d480e8dd4c")


def test_m13_gate_uses_the_stricter_component_floor_and_m13_status():
    metrics = {"rel_l2_score": 1.0, "tke_rel_l2_score": 1.0,
               "mvpe_rel_l2_score": 1.0, "native_sps_score": 1.0,
               "official_point_contribution_46_10_9": 1.0}
    names = [f"trajectory_{index}" for index in range(18)]
    candidate = {"pooled": metrics, "target_audits": {"same": True},
                 "per_trajectory": {name: metrics for name in names}}
    control_metrics = dict(metrics); control_metrics["rel_l2_score"] = 1.031
    control = {"pooled": control_metrics, "target_audits": {"same": True},
               "per_trajectory": {name: control_metrics for name in names}}
    result = subject.compare_evaluations(candidate, control)
    assert result["gate"]["conditions"]["rel_floor"] is False
    assert result["gate"]["decision"] == "STOP_M13_LOCAL_LAG_COVARIANCE"


def test_candidate_uses_own_history_and_control_uses_explicit_null(monkeypatch):
    seen = []
    module = types.SimpleNamespace()
    def loss(history, prediction, target, *, anchor, donor_positions=None):
        seen.append((history.clone(), anchor, donor_positions))
        return (prediction - target).square().mean()
    module.local_lag_covariance_loss = loss
    monkeypatch.setitem(sys.modules, "local_lag_covariance", module)
    history = torch.arange(24.0).reshape(3, 2, 2, 2)
    prediction = torch.ones(3, 1, 2, 2, requires_grad=True)
    target = torch.zeros_like(prediction)

    donor_positions = [1, 2, 0]
    candidate = subject.local_lag_loss(
        history, prediction, target, "candidate", donor_positions)
    control = subject.local_lag_loss(
        history, prediction, target, "control", donor_positions)
    assert candidate.ndim == control.ndim == 0
    assert seen[0][1] == "candidate"
    assert torch.equal(seen[0][0], history)
    assert seen[0][2] is None
    assert seen[1][1] == "null"
    assert torch.equal(seen[1][0], history)
    assert seen[1][2] == donor_positions


def test_nonself_donor_permutation_breaks_duplicate_raw_window_identity():
    samples = [
        {"file": "same.h5", "start": 20},
        {"file": "same.h5", "start": 20},
        {"file": "other.h5", "start": 20},
        {"file": "third.h5", "start": 40},
    ]
    positions = subject.nonself_donor_positions(samples)
    assert sorted(positions) == list(range(len(samples)))
    identities = [(sample["file"], sample["start"]) for sample in samples]
    assert all(identities[index] != identities[donor]
               for index, donor in enumerate(positions))


def test_nonself_donor_permutation_fails_when_no_matching_exists():
    samples = [{"file": "same.h5", "start": 20} for _ in range(4)]
    with pytest.raises(RuntimeError, match="no donor permutation"):
        subject.nonself_donor_positions(samples)


def test_1920_schedule_uses_only_the_replayed_a_role(monkeypatch):
    rows = [{"file": f"a{aoa}_{re}", "aoa": aoa, "re": re,
             "train_starts": [0]} for aoa in subject.m11.FIT_AOAS
            for re in (1, 2)]
    monkeypatch.setattr(subject.m11, "FIT_ROWS_SHA256",
                        subject.m11.canonical_sha256(rows))
    schedule = subject.m11.build_cross_condition_schedule(rows, 1_920, 4, 0)
    assert set(schedule["pair_counts"].values()) == {160}
    stream = [{"update": r["update"], "A": r["A"],
               "mask": r["A_bootstrap_mask"], "rng": r["rng"]["A"]}
              for r in schedule["rows"]]
    assert len(stream) == 1_920
    assert all("B" not in row and "A" in row for row in stream)
    replay = subject.m11.build_cross_condition_schedule(rows, 1_920, 4, 0)
    assert schedule["sha256"] == replay["sha256"]


def test_warmup_keeps_aux_off_until_the_calibration_successor_row():
    """A cheap regression tripwire for the update-12/13 off-by-one boundary."""
    calibration = inspect.getsource(subject.calibrate_auxiliary_weight)
    train = inspect.getsource(subject.train_arm)
    assert 'schedule["rows"][:AUX_WARMUP_UPDATES]' in calibration
    assert ('schedule["rows"][AUX_WARMUP_UPDATES:panel_end]'
            in calibration)
    assert 'schedule["rows"][AUX_WARMUP_UPDATES:]' in train
    assert "loss = main + aux_weight * auxiliary" in train


def test_optimizer_branch_state_copy_has_no_shared_tensor_storage():
    original = {"state": {0: {"step": torch.tensor(12.0),
                              "exp_avg": torch.arange(3.0)}},
                "param_groups": [{"params": [0], "lr": 2e-4}]}
    original_hash = subject.nested_state_sha256(original)
    copied = subject.nested_state_to_cpu(original)
    copied["state"][0]["step"].add_(1)
    copied["state"][0]["exp_avg"].zero_()
    assert subject.nested_state_sha256(original) == original_hash
    assert subject.nested_state_sha256(copied) != original_hash
