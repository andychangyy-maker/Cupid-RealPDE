from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "train_c208_smu1_paired_loao.py")
SPEC = importlib.util.spec_from_file_location("c208_smu1_paired_loao", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_contract_freezes_scheduler_strong_gate_and_no_tail_access():
    assert subject.UPDATES == 4000
    assert subject.SCHEDULER_T_MAX == 5000
    assert subject.NON_TIME_DELTA_MIN == 0.52
    assert subject.CONTRACT["tail"].startswith("fixed725 is never")
    assert sum(subject.FINAL_COEFFICIENTS.values()) == pytest.approx(
        0.89897175)


def test_train_side_split_matches_frozen_algorithm_without_tail_dataset():
    metadata = [
        {"file": "a.h5", "frames": 200, "re": 5000, "aoa": 0},
        {"file": "b.h5", "frames": 220, "re": 8000, "aoa": 5},
    ]
    train, manifest, tail_count = subject.make_train_side_split(metadata)
    expected_train, expected_tail, expected_manifest = (
        subject.ensemble_train.common.repro.make_split(metadata, 20, 0.2))
    assert train == expected_train
    assert manifest == expected_manifest
    assert tail_count == len(expected_tail)


def test_window_statistics_match_bruteforce_and_exclude_unselected_rows():
    trajectory = torch.arange(80 * 2 * 3 * 2, dtype=torch.float32).reshape(
        80, 2, 3, 2)
    windows = [
        subject.ensemble_train.common.repro.Window(0, 0),
        subject.ensemble_train.common.repro.Window(0, 20),
    ]
    rows = subject.window_sufficient_statistics([trajectory], windows)
    brute_input = trajectory[:20].double().sum(dim=(0, 1, 2))
    brute_target = trajectory[20:40].double().sum(dim=(0, 1, 2))
    torch.testing.assert_close(rows["input_sum"][0, :2], brute_input)
    torch.testing.assert_close(rows["target_sum"][0, :2], brute_target)
    assert int(rows["elements_per_partition"]) == 20 * 2 * 3

    selected_before = subject.fold_statistics(rows, [0])
    rows["input_sum"][1] = 1e30
    rows["input_sum_sq"][1] = 1e30
    rows["target_sum"][1] = 1e30
    rows["target_sum_sq"][1] = 1e30
    selected_after = subject.fold_statistics(rows, [0])
    for before, after in zip(selected_before, selected_after):
        assert torch.equal(before, after)


def test_paired_schedule_replays_indices_trajectories_and_masks():
    windows = [SimpleNamespace(trajectory=index % 7) for index in range(23)]
    first = subject.paired_batch_schedule(
        list(range(23)), windows, updates=19, seed=0)
    replay = subject.paired_batch_schedule(
        list(range(23)), windows, updates=19, seed=0)
    changed = subject.paired_batch_schedule(
        list(range(23)), windows, updates=19, seed=1)
    assert first["sha256"] == replay["sha256"]
    assert first["rows"] == replay["rows"]
    assert first["sha256"] != changed["sha256"]
    assert all(1 <= len(batch) <= subject.BATCH_SIZE
               for batch in first["batches"])
    expected = subject.ensemble_train.trajectory_member_mask(
        torch.tensor(first["rows"][0]["trajectory_ids"]), 0.75, 0, 4)
    assert first["rows"][0]["bootstrap_mask"] == (
        expected.to(torch.uint8).tolist())


def test_streamed_metric_state_merges_to_single_pooled_finalizer():
    torch.manual_seed(4)
    target = torch.randn(3, 20, 32, 64, 3)
    target[..., 2] = 0.0
    prediction = target + 0.03 * torch.randn_like(target)
    prediction[..., 2] = 0.0
    lower = torch.full((3, 20, 32, 64, 2), 0.04)
    upper = torch.full_like(lower, 0.05)

    whole = subject.new_metric_state()
    subject.update_metric_state(whole, prediction, target, lower, upper)
    parts = []
    for index in range(3):
        state = subject.new_metric_state()
        subject.update_metric_state(
            state, prediction[index:index + 1], target[index:index + 1],
            lower[index:index + 1], upper[index:index + 1])
        parts.append(state)
    merged = subject.merge_metric_states(parts)

    assert merged == pytest.approx(whole)
    assert subject.finalize_metric_state(merged) == pytest.approx(
        subject.finalize_metric_state(whole), abs=1e-12)


def test_exact_physical_target_controls_training_masks_and_sps_factor():
    statistics = (
        torch.tensor([0.2, -0.1, 0.0]),
        torch.tensor([0.1234567, -0.2345678, 0.0]),
        torch.tensor([0.7, 1.1, 1.0]),
        torch.tensor([0.37, 0.83, 1.0]),
    )
    normalizer = subject.FoldNormalizer(statistics, torch.device("cpu"))
    raw_target = torch.zeros(1, 20, 32, 64, 3)
    raw_target[:, :, 12:20, 10:22, 0] = 0.4
    raw_target[:, :, 12:20, 10:22, 1] = -0.2
    target_norm = (
        raw_target - normalizer.mean_target) / normalizer.std_target
    prediction = target_norm.detach().clone().requires_grad_(True)
    quantiles = torch.zeros(1, 20, 32, 64, 2, 7, requires_grad=True)
    model = SimpleNamespace(config=SimpleNamespace(
        quantile_levels=(0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95)))

    quantile_loss = subject.ensemble_train.residual_quantile_loss(
        model, prediction, quantiles, target_norm, normalizer,
        physical_target=raw_target)
    assert torch.isfinite(quantile_loss)
    assert float(quantile_loss) == pytest.approx(0.0, abs=1e-12)

    prediction = (target_norm + 0.02).detach().requires_grad_(True)
    lower = torch.full((1, 20, 32, 64, 2), 0.08, requires_grad=True)
    upper = torch.full_like(lower, 0.09, requires_grad=True)
    loss, score = subject.ensemble_train.soft_native_sps_objective(
        prediction, target_norm, lower, upper, normalizer,
        tau_fraction=0.05, physical_target=raw_target)

    prediction_physical = (
        prediction.detach() * normalizer.std_target
        + normalizer.mean_target)[..., :2]
    target_physical = raw_target[..., :2]
    scale = normalizer.std_target[:2]
    lower_physical = lower * scale
    upper_physical = upper * scale
    residual = target_physical - prediction_physical
    scored = target_physical != 0.0
    reference = lower.new_tensor([0.01075, 0.00550]).view(
        1, 1, 1, 1, 2)
    tau = 0.05 * reference
    efficiency = (
        torch.sigmoid((residual + lower_physical) / tau)
        * torch.sigmoid((upper_physical - residual) / tau)
        * torch.exp(
            -(lower_physical + upper_physical)
            / subject.ensemble_train.common.SPS_SIGMA))
    count = scored.sum(dim=(1, 2, 3, 4)).clamp_min(1)
    per_sample = (efficiency * scored).sum(dim=(1, 2, 3, 4)) / count
    factor = subject.ensemble_train.common.sample_sps_factor(
        prediction_physical, target_physical)
    expected = (factor * per_sample).mean()
    torch.testing.assert_close(score, expected)
    torch.testing.assert_close(loss, -expected)


def test_held_evaluator_uses_raw_target_without_roundtrip_mask_corruption():
    statistics = (
        torch.tensor([0.2, -0.1, 0.0]),
        torch.tensor([0.1234567, -0.2345678, 0.0]),
        torch.tensor([0.7, 1.1, 1.0]),
        torch.tensor([0.37, 0.83, 1.0]),
    )

    class GuardedNormalizer(subject.FoldNormalizer):
        def postprocess_target(self, value):
            raise AssertionError("held evaluator must use raw physical target")

    normalizer = GuardedNormalizer(statistics, torch.device("cpu"))
    history = torch.zeros(20, 32, 64, 3)
    target = torch.zeros(20, 32, 64, 3)
    target[:, 15:17, 12:14, :2] = torch.tensor([0.4, -0.2])

    class FakeModel:
        def eval(self):
            return self

        def forward_distribution(self, history_norm):
            batch, steps, height, width, channels = history_norm.shape
            prediction = torch.zeros(
                batch, steps, height, width, channels)
            lower = torch.full((batch, steps, height, width, 2), 0.1)
            upper = torch.full_like(lower, 0.1)
            return prediction, lower, upper

    _metrics, state, audit = subject.evaluate_model(
        FakeModel(), [(history, target)], [0], normalizer,
        torch.device("cpu"), deadline=float("inf"))
    assert state["sps_scored"] == int(torch.count_nonzero(target[..., :2]))
    assert audit["scored_elements"] == state["sps_scored"]


def _metrics(rel=90.0, tke=80.0, mvpe=90.0, sps=50.0):
    return {
        "rel_l2_score": rel,
        "tke_rel_l2_score": tke,
        "mvpe_rel_l2_score": mvpe,
        "point_score": 0.5 * rel + 0.3 * tke + 0.2 * mvpe,
        "native_sps_score": sps,
    }


def test_gate_requires_five_rows_and_every_strong_condition():
    control = _metrics()
    candidate = _metrics(rel=90.45, tke=80.15, mvpe=90.35, sps=51.25)
    pooled = subject.comparison(candidate, control)
    stream = "a" * 64
    rows = [{
        "held_aoa": held,
        "samples": 100,
        "stream_sha256": stream,
        "held_ordered_key_sha256": stream,
        "control_held_audit": {
            "ordered_raw_target_sha256": stream,
            "ordered_scored_mask_sha256": stream,
            "samples": 100,
            "scored_elements": 1000,
        },
        "candidate_held_audit": {
            "ordered_raw_target_sha256": stream,
            "ordered_scored_mask_sha256": stream,
            "samples": 100,
            "scored_elements": 1000,
        },
        "control_training": {"actual_stream_sha256": stream},
        "candidate_training": {"actual_stream_sha256": stream},
        "comparison": pooled,
    } for held in subject.AOA_VALUES]
    for row in rows:
        row["samples"] = subject.EXPECTED_AOA_COUNTS[row["held_aoa"]]
        row["control_held_audit"]["samples"] = row["samples"]
        row["candidate_held_audit"]["samples"] = row["samples"]

    gate = subject.loao_gate(pooled, rows)
    assert gate["pass"]
    assert gate["decision"] == "GO_SMU1_FULL2576"

    failed = dict(pooled)
    failed["non_time_delta"] = subject.NON_TIME_DELTA_MIN - 1e-9
    assert not subject.loao_gate(failed, rows)["pass"]
    with pytest.raises(ValueError, match="exactly five"):
        subject.loao_gate(pooled, rows[:4])


def test_shared_mean_config_rejects_non_mean_fluctuation_model():
    with pytest.raises(ValueError, match="requires mean_fluctuation"):
        subject.EnsembleUNet2dConfig(
            base=8, depth=3, groups=4,
            mean_fluctuation=False, predict_uncertainty=True,
            shared_member_uv_mean=True).validate()
