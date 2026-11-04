from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "train_c208_bb3_balanced_bootstrap_loao.py")
SPEC = importlib.util.spec_from_file_location(
    "c208_bb3_balanced_bootstrap_loao", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_contract_keeps_the_exact_smu1_strong_gate():
    assert subject.CONTRACT["gate"] == subject.smu1.CONTRACT["gate"]
    assert subject.smu1.POINT_DELTA_MIN == 0.15
    assert subject.smu1.DIRECT_DELTA_MIN == 0.20
    assert subject.smu1.SPS_DELTA_MIN == 0.50
    assert subject.smu1.NON_TIME_DELTA_MIN == 0.52


def _synthetic_split():
    counts = {0: 5, 5: 6, 10: 7, 15: 8, 20: 9}
    metadata = []
    windows = []
    trajectories_by_aoa = {}
    for aoa, count in counts.items():
        trajectories_by_aoa[aoa] = []
        for local_index in range(count):
            trajectory = len(metadata)
            trajectories_by_aoa[aoa].append(trajectory)
            metadata.append({
                "file": f"aoa{aoa:02d}_{local_index:02d}.h5",
                "aoa": aoa,
                "frames": 100 + local_index,
            })
            windows.extend([
                SimpleNamespace(trajectory=trajectory, start=0),
                SimpleNamespace(trajectory=trajectory, start=20),
            ])
    return metadata, windows, trajectories_by_aoa


def _fit_indices(windows, metadata, held):
    return [
        index for index, window in enumerate(windows)
        if metadata[window.trajectory]["aoa"] != held]


def test_balanced_mask_is_exact3_per_aoa_and_has_exact_totals():
    metadata, windows, trajectories_by_aoa = _synthetic_split()
    held = 10
    fit = _fit_indices(windows, metadata, held)
    mapping, audit = subject.balanced_exact3_masks(
        fit, windows, metadata, held)

    expected_trajectories = {
        trajectory
        for aoa, trajectories in trajectories_by_aoa.items()
        if aoa != held
        for trajectory in trajectories
    }
    assert set(mapping) == expected_trajectories
    assert all(sum(mask) == 3 for mask in mapping.values())
    assert audit["trajectory_assignments"] == 3 * len(mapping)
    assert audit["window_assignments"] == 3 * len(fit)
    assert all(row["included_range"] <= 1
               for row in audit["per_aoa"].values())


def test_balanced_mask_is_deterministic_and_ignores_noncontract_metadata():
    metadata, windows, _ = _synthetic_split()
    held = 5
    fit = _fit_indices(windows, metadata, held)
    first, first_audit = subject.balanced_exact3_masks(
        fit, windows, metadata, held)
    replay, replay_audit = subject.balanced_exact3_masks(
        list(reversed(fit)), windows, metadata, held)
    changed_irrelevant = [dict(item, frames=item["frames"] + 999)
                          for item in metadata]
    metadata_replay, metadata_audit = subject.balanced_exact3_masks(
        fit, windows, changed_irrelevant, held)

    assert first == replay == metadata_replay
    assert first_audit["mapping_sha256"] == replay_audit["mapping_sha256"]
    assert first_audit["mapping_sha256"] == metadata_audit["mapping_sha256"]


def test_balanced_mask_is_other_four_only_and_rejects_held_leakage():
    metadata, windows, trajectories_by_aoa = _synthetic_split()
    held = 20
    fit = _fit_indices(windows, metadata, held)
    mapping, audit = subject.balanced_exact3_masks(
        fit, windows, metadata, held)
    assert set(audit["fit_aoas"]) == set(subject.AOA_VALUES) - {held}
    assert not set(mapping).intersection(trajectories_by_aoa[held])

    leaked = fit + [next(
        index for index, window in enumerate(windows)
        if window.trajectory in trajectories_by_aoa[held])]
    with pytest.raises(RuntimeError, match="held-AoA trajectory"):
        subject.balanced_exact3_masks(leaked, windows, metadata, held)
    with pytest.raises(RuntimeError, match="exactly the other-four windows"):
        subject.balanced_exact3_masks(fit[:-1], windows, metadata, held)


def test_candidate_schedule_preserves_samples_and_exact3_assignments():
    metadata, windows, _ = _synthetic_split()
    held = 0
    fit = _fit_indices(windows, metadata, held)
    control = subject.smu1.paired_batch_schedule(
        fit, windows, updates=17, seed=0)
    candidate, _mapping = subject.balanced_candidate_schedule(
        control, fit, windows, metadata, held)

    assert subject.sample_stream_sha256(control) == (
        candidate["sample_stream_sha256"])
    assert candidate["scheduled_assignments"] == (
        3 * candidate["scheduled_samples"])
    assert all(sum(member_row) == 3 for row in candidate["rows"]
               for member_row in row["bootstrap_mask"])
    assert control["sha256"] != candidate["sha256"]


def test_raw_held_target_audit_is_deterministic_and_content_bound():
    history = torch.zeros(2, 2, 2, 2)
    targets = [torch.full((2, 2, 2, 2), float(index + 1))
               for index in range(3)]
    dataset = [(history, target) for target in targets]
    first = subject.raw_held_target_audit(dataset, [0, 1, 2])
    replay = subject.raw_held_target_audit(dataset, [0, 1, 2])
    changed_targets = [target.clone() for target in targets]
    changed_targets[1][0, 0, 0, 0] += 1.0
    changed = subject.raw_held_target_audit(
        [(history, target) for target in changed_targets], [0, 1, 2])

    assert first == replay
    assert first["ordered_raw_target_sha256"] != (
        changed["ordered_raw_target_sha256"])
    assert first["samples"] == 3


def _metric_state(samples: int) -> dict:
    return {
        "samples": samples,
        "rel_error_sum": 0.1 * samples,
        "tke_error_sum": 0.2 * samples,
        "mvpe_error_sum": 0.15 * samples,
        "sps_numerator": 500.0,
        "sps_scored": 1000,
        "sps_covered": 700,
        "sps_width_sum": 12.0,
    }


def _write_reusable_control(directory: Path, held: int,
                            statistics: tuple[torch.Tensor, ...],
                            stream: str, held_key: str) -> None:
    state = _metric_state(subject.EXPECTED_AOA_COUNTS[held])
    audit = {
        "ordered_raw_target_sha256": "a" * 64,
        "ordered_scored_mask_sha256": "b" * 64,
        "samples": subject.EXPECTED_AOA_COUNTS[held],
        "scored_elements": state["sps_scored"],
    }
    training = {
        "completed_updates": subject.UPDATES,
        "planned_stream_sha256": stream,
        "actual_stream_sha256": stream,
        "history": [],
        "wall_seconds": 1.0,
    }
    fold = {
        "held_aoa": held,
        "fit_samples": subject.EXPECTED_FIT_COUNTS[held],
        "samples": subject.EXPECTED_AOA_COUNTS[held],
        "fold_statistics": [value.tolist() for value in statistics],
        "fold_statistics_sha256": subject.smu1.statistics_digest(statistics),
        "stream_sha256": stream,
        "held_ordered_key_sha256": held_key,
        "control_training": training,
        "control_held_audit": audit,
        "control_state": state,
    }
    report = {
        "experiment": subject.smu1.EXPERIMENT,
        "contract": subject.smu1.CONTRACT,
        "provenance": {
            "start_sim_sha256": subject.smu1.FORMAL_START_SHA256,
            "real_stats_sha256": subject.smu1.FORMAL_REAL_STATS_SHA256,
            "split_manifest_sha256": subject.smu1.FORMAL_SPLIT_SHA256,
            "git": {
                "status_porcelain": "",
                "commit": next(iter(subject.SMU1_CONTROL_COMPATIBLE_COMMITS)),
            },
        },
        "counts": {
            "train_windows": 2576,
            "tail_windows_not_constructed": 725,
            "trajectories": 82,
        },
        "normalizer_contract": {
            "kind": "other-four optimizer-window statistics per fold",
            "held_target_in_normalizer": False,
        },
        "tail_accessed": False,
        "release_checkpoint_written": False,
        "rows": [fold],
    }
    (directory / f"fold_{held:02d}.json").write_text(
        json.dumps(fold), encoding="utf-8")
    (directory / "summary.json").write_text(
        json.dumps(report), encoding="utf-8")


def test_control_reuse_is_strict_and_only_missing_fold_retrains(tmp_path):
    held = 10
    statistics = tuple(torch.tensor([value, value + 1, value + 2.0])
                       for value in range(4))
    stream = "c" * 64
    held_key = "d" * 64
    _write_reusable_control(tmp_path, held, statistics, stream, held_key)

    reused = subject.load_reusable_control_fold(
        tmp_path, held, expected_statistics=statistics,
        expected_stream_sha256=stream,
        expected_held_key_sha256=held_key,
        expected_held_audit={
            "ordered_raw_target_sha256": "a" * 64,
            "ordered_scored_mask_sha256": "b" * 64,
            "samples": subject.EXPECTED_AOA_COUNTS[held],
            "scored_elements": 1000,
        })
    assert reused is not None
    assert reused["training"]["actual_stream_sha256"] == stream
    assert reused["state"]["samples"] == subject.EXPECTED_AOA_COUNTS[held]
    assert reused["source"]["report_file"] == "summary.json"

    mismatched_audit = {
        "ordered_raw_target_sha256": "f" * 64,
        "ordered_scored_mask_sha256": "b" * 64,
        "samples": subject.EXPECTED_AOA_COUNTS[held],
        "scored_elements": 1000,
    }
    with pytest.raises(ValueError, match="fresh held data"):
        subject.load_reusable_control_fold(
            tmp_path, held, expected_statistics=statistics,
            expected_stream_sha256=stream,
            expected_held_key_sha256=held_key,
            expected_held_audit=mismatched_audit)

    report_path = tmp_path / "summary.json"
    incompatible_report = json.loads(report_path.read_text(encoding="utf-8"))
    incompatible_report["provenance"]["git"]["commit"] = "deadbeef"
    report_path.write_text(json.dumps(incompatible_report), encoding="utf-8")
    with pytest.raises(ValueError, match="not reusable"):
        subject.load_reusable_control_fold(
            tmp_path, held, expected_statistics=statistics,
            expected_stream_sha256=stream,
            expected_held_key_sha256=held_key,
            expected_held_audit={
                "ordered_raw_target_sha256": "a" * 64,
                "ordered_scored_mask_sha256": "b" * 64,
                "samples": subject.EXPECTED_AOA_COUNTS[held],
                "scored_elements": 1000,
            })

    assert subject.load_reusable_control_fold(
        tmp_path, 0, expected_statistics=statistics,
        expected_stream_sha256=stream,
        expected_held_key_sha256=held_key,
        expected_held_audit={}) is None

    fold_path = tmp_path / f"fold_{held:02d}.json"
    incompatible = json.loads(fold_path.read_text(encoding="utf-8"))
    incompatible["stream_sha256"] = "e" * 64
    fold_path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        subject.load_reusable_control_fold(
            tmp_path, held, expected_statistics=statistics,
            expected_stream_sha256=stream,
            expected_held_key_sha256=held_key,
            expected_held_audit={})


def test_legacy_bootstrap_loss_path_matches_explicit_legacy_mask_bitwise():
    torch.manual_seed(11)
    args = SimpleNamespace(
        point_loss_mode="official_accuracy", mse_weight=0.25,
        mse_warmup_updates=1500, bootstrap_probability=0.75, seed=0)
    members = torch.randn(5, 4, 2, 3, 4, 3)
    target = torch.randn(5, 2, 3, 4, 3)
    trajectory_ids = torch.tensor([0, 1, 2, 3, 4])
    statistics = (
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3))
    normalizer = subject.smu1.FoldNormalizer(
        statistics, torch.device("cpu"))
    legacy_mask = subject.smu1.ensemble_train.trajectory_member_mask(
        trajectory_ids, 0.75, 0, 4)

    legacy_loss, legacy_fraction = (
        subject.smu1.ensemble_train.bootstrap_member_point_loss(
            args, members, target, trajectory_ids, normalizer,
            (0.5, 0.3, 0.2), iteration=10))
    explicit_loss, explicit_fraction = (
        subject.smu1.ensemble_train.bootstrap_member_point_loss(
            args, members, target, trajectory_ids, normalizer,
            (0.5, 0.3, 0.2), iteration=10,
            member_mask=legacy_mask))

    assert torch.equal(legacy_loss, explicit_loss)
    assert torch.equal(legacy_fraction, explicit_fraction)


def test_formal_gpu_path_requires_double_opt_in(tmp_path):
    base = dict(
        start_sim=tmp_path / "start.pth",
        start_sim_sha256=subject.smu1.FORMAL_START_SHA256,
        output_dir=tmp_path / "new-output",
        real_dir=tmp_path / "real",
        real_stats=tmp_path / "stats.pt",
        reuse_smu1_control_dir=None,
        device="cuda:0", num_workers=0, updates=subject.UPDATES,
        contract_only=False, execute_formal_gpu=True,
        acknowledge_fixed2576_only=None,
    )
    with pytest.raises(ValueError, match="acknowledgement"):
        subject.validate_args(SimpleNamespace(**base))
    base["acknowledge_fixed2576_only"] = subject.FORMAL_ACK
    subject.validate_args(SimpleNamespace(**base))


def test_run_itself_cannot_bypass_formal_gpu_opt_in():
    with pytest.raises(ValueError, match="execute-formal-gpu"):
        subject.run(SimpleNamespace(
            execute_formal_gpu=False,
            acknowledge_fixed2576_only=None))
