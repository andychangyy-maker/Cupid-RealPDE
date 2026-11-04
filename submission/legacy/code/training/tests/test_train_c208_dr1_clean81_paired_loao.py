from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "train_c208_dr1_clean81_paired_loao.py")
SPEC = importlib.util.spec_from_file_location("c208_dr1_clean81", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def _spread(total: int, rows: int) -> list[int]:
    quotient, remainder = divmod(total, rows)
    return [quotient + (index < remainder) for index in range(rows)]


def formal_shape_manifest() -> list[dict]:
    trajectory_counts = {0: 16, 5: 17, 10: 16, 15: 16, 20: 16}
    rows = []
    serial = 0
    for aoa in subject.AOA_VALUES:
        per_file = _spread(
            subject.EXPECTED_AOA_COUNTS[aoa], trajectory_counts[aoa])
        for windows in per_file:
            starts = [20 * index for index in range(windows)]
            boundary = starts[-1] + 40
            rows.append({
                "file": f"synthetic_{serial:03d}_{aoa}.h5",
                "frames": boundary + 60,
                "re": 6300 + serial,
                "aoa": aoa,
                "train_starts": starts,
                "val_boundary": boundary,
            })
            serial += 1
    return rows


def write_trajectory(path: Path, *, aoa: int, frames: int = 100,
                     boundary: int = 60) -> None:
    u = np.full((frames, 4, 6), 2.0, dtype=np.float32)
    v = np.full((frames, 4, 6), -3.0, dtype=np.float32)
    u[boundary:] = 1000.0
    v[boundary:] = -1000.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=u)
        handle.create_dataset("v", data=v)
        handle.create_dataset("re", data=6300)
        handle.create_dataset("aoa", data=aoa)


def one_row(filename: str, aoa: int) -> dict:
    return {
        "file": filename,
        "frames": 100,
        "re": 6300,
        "aoa": aoa,
        "train_starts": [0, 20],
        "val_boundary": 60,
    }


def fake_sim_audit(variant: str) -> dict:
    training = {
        "phase": "sim",
        "output_dir": f"/runs/{variant}",
        "updates": 10000,
        "batch_size": 8,
        "seed": 4,
        "num_workers": 0,
        "window_stride": 10,
        "teacher_cache": None,
        "distill_weight": 0.0,
        "sim_audit_variant": variant,
        **(subject.CONTROL_DR if variant == "control"
           else subject.CANDIDATE_DR),
    }
    model_config = {"architecture": "ensemble_unet", "base": 208}
    return {
        "schema": subject.SIM_AUDIT_SCHEMA,
        "variant": variant,
        "initial_model_state_sha256": "1" * 64,
        "base_sample_stream_sha256": "2" * 64,
        "jitter_shift_stream_sha256": ("3" if variant == "control"
                                        else "4") * 64,
        "model_config": model_config,
        "model_config_sha256": subject.canonical_sha256(model_config),
        "training_config": training,
        "training_config_sha256": subject.canonical_sha256(training),
        "endpoint_model_state_sha256": ("5" if variant == "control"
                                         else "6") * 64,
        "endpoint_checkpoint_sha256": ("7" if variant == "control"
                                        else "8") * 64,
        "run_ordinal": 2 if variant == "control" else 1,
        "started_at_utc": (
            "2026-08-26T01:00:00+00:00" if variant == "control"
            else "2026-08-26T00:00:00+00:00"),
        "completed_at_utc": (
            "2026-08-26T02:00:00+00:00" if variant == "control"
            else "2026-08-26T00:59:00+00:00"),
        "hardware_fingerprint": {
            "gpu_uuid": "GPU-1",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "torch_version": "2.0",
            "cuda_runtime_version": "12.1",
            "cudnn_version": 8900,
        },
    }


def test_runtime_manifest_accepts_only_exact_clean2544_contract():
    manifest = formal_shape_manifest()
    digest = subject.canonical_sha256(manifest)

    rows, report = subject.validate_runtime_manifest(manifest, digest)

    assert len(rows) == 81
    assert report["windows"] == 2544
    assert report["aoa_windows"] == {
        str(key): value for key, value in subject.EXPECTED_AOA_COUNTS.items()}
    assert report["val_starts_present"] is False


def test_runtime_manifest_rejects_val_starts_and_count_drift():
    manifest = formal_shape_manifest()
    manifest[0]["val_starts"] = [manifest[0]["val_boundary"]]
    with pytest.raises(ValueError, match="must not contain val_starts"):
        subject.validate_runtime_manifest(
            manifest, subject.canonical_sha256(manifest))

    manifest = formal_shape_manifest()
    manifest[0]["train_starts"].pop()
    with pytest.raises(RuntimeError, match="contract failed"):
        subject.validate_runtime_manifest(
            manifest, subject.canonical_sha256(manifest))


def test_prefix_loader_never_reads_tail_sentinel(tmp_path):
    path = tmp_path / "6300_10.h5"
    write_trajectory(path, aoa=10)
    audit = subject.PrefixAccessAudit(held_aoa=0)

    trajectories, metadata, windows = subject.load_prefix_rows(
        tmp_path, [one_row(path.name, 10)], "fit", audit)

    assert len(trajectories) == 1
    assert len(windows) == 2
    assert metadata[0]["frames"] == 60
    assert float(trajectories[0][..., 0].max()) == 2.0
    assert float(trajectories[0][..., 1].min()) == -3.0
    report = audit.report([path.name])
    assert report["max_read_frame_exclusive_by_file"] == {path.name: 60}
    assert report["tail_arrays_read"] == 0


def test_held_prefix_is_blocked_before_both_seals_without_opening_h5(
        tmp_path, monkeypatch):
    path = tmp_path / "6300_10.h5"
    write_trajectory(path, aoa=10)
    audit = subject.PrefixAccessAudit(held_aoa=10)
    opened = []
    original = subject.h5py.File

    def spy(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(subject.h5py, "File", spy)
    with pytest.raises(RuntimeError, match="before both seals"):
        subject.load_prefix_rows(
            tmp_path, [one_row(path.name, 10)], "held", audit)
    assert opened == []


def test_excluded_file_is_rejected_before_h5_open(tmp_path, monkeypatch):
    excluded = tmp_path / subject.EXCLUDED_BASENAME
    write_trajectory(excluded, aoa=10)
    audit = subject.PrefixAccessAudit(held_aoa=0)
    opened = []

    def forbidden(*args, **kwargs):
        opened.append(args[0])
        raise AssertionError("excluded H5 must never be opened")

    monkeypatch.setattr(subject.h5py, "File", forbidden)
    with pytest.raises(RuntimeError, match="excluded H5"):
        subject.load_prefix_rows(
            tmp_path, [one_row(excluded.name, 10)], "fit", audit)
    assert opened == []


def test_held_prefix_opens_only_after_control_and_candidate_seals(tmp_path):
    path = tmp_path / "6300_10.h5"
    write_trajectory(path, aoa=10)
    audit = subject.PrefixAccessAudit(held_aoa=10)
    audit.seal("control", tmp_path / "control.pth", "a" * 64, "b" * 64)
    audit.seal("candidate", tmp_path / "candidate.pth", "c" * 64, "d" * 64)

    trajectories, _metadata, windows = subject.load_prefix_rows(
        tmp_path, [one_row(path.name, 10)], "held", audit)

    assert len(trajectories) == 1
    assert len(windows) == 2
    report = audit.report([path.name])
    assert report["held_first_open_after_both_seals"] is True
    assert report["events"][0]["both_arms_sealed_before_open"] is True


def test_sim_pair_requires_only_five_registered_scientific_differences():
    control = fake_sim_audit("control")
    candidate = fake_sim_audit("candidate")

    report = subject.validate_sim_audit_pair(control, candidate)

    assert set(report["config_diff"]) == set(subject.DR_FIELDS)
    assert report["initial_model_state_sha256"] == "1" * 64
    assert report["base_sample_stream_sha256"] == "2" * 64
    assert report["teacher_cache"] is None


def test_sim_pair_rejects_unregistered_config_or_stream_difference():
    control = fake_sim_audit("control")
    candidate = fake_sim_audit("candidate")
    candidate["training_config"]["lr"] = 4e-4
    candidate["training_config_sha256"] = subject.canonical_sha256(
        candidate["training_config"])
    with pytest.raises(RuntimeError, match="unregistered Sim config"):
        subject.validate_sim_audit_pair(control, candidate)

    candidate = fake_sim_audit("candidate")
    candidate["base_sample_stream_sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="base sample streams"):
        subject.validate_sim_audit_pair(control, candidate)


def test_sim_pair_rejects_different_gpu_or_wrong_run_order():
    control = fake_sim_audit("control")
    candidate = fake_sim_audit("candidate")
    control["hardware_fingerprint"]["gpu_uuid"] = "GPU-2"
    with pytest.raises(RuntimeError, match="same physical GPU"):
        subject.validate_sim_audit_pair(control, candidate)

    control = fake_sim_audit("control")
    candidate = fake_sim_audit("candidate")
    candidate["completed_at_utc"] = "2026-08-26T01:30:00+00:00"
    with pytest.raises(RuntimeError, match="sequential candidate then control"):
        subject.validate_sim_audit_pair(control, candidate)


def test_legacy_sim_audits_require_external_sequential_manifest():
    control = fake_sim_audit("control")
    candidate = fake_sim_audit("candidate")
    control.pop("completed_at_utc")
    candidate.pop("completed_at_utc")
    with pytest.raises(ValueError, match="external sequential run manifest"):
        subject.validate_sim_audit_pair(control, candidate)

    report = subject.validate_sim_audit_pair(
        control, candidate, {
            "same_shell_sequential": True,
        })
    assert report["sequential_order"]["audit_intervals_verified"] is False
    assert report["sequential_order"]["external_run_manifest_required"] is True


def test_run_manifest_binds_source_commit_and_both_artifacts():
    run_manifest = {
        "schema": subject.RUN_MANIFEST_SCHEMA,
        "source_git_commit": subject.FORMAL_SIM_SOURCE_COMMIT,
        "sequence": ["candidate", "control"],
        "same_shell_sequential": True,
        "candidate_completed_before_control_started": True,
        "control": {"audit_sha256": "a" * 64,
                    "checkpoint_sha256": "b" * 64},
        "candidate": {"audit_sha256": "c" * 64,
                      "checkpoint_sha256": "d" * 64},
    }
    report = subject.validate_sim_run_manifest(
        run_manifest,
        control_audit_sha256="a" * 64,
        candidate_audit_sha256="c" * 64,
        control_checkpoint_sha256="b" * 64,
        candidate_checkpoint_sha256="d" * 64)
    assert report["source_git_commit"] == subject.FORMAL_SIM_SOURCE_COMMIT

    run_manifest["source_git_commit"] = "e" * 40
    with pytest.raises(ValueError, match="source_git_commit"):
        subject.validate_sim_run_manifest(
            run_manifest,
            control_audit_sha256="a" * 64,
            candidate_audit_sha256="c" * 64,
            control_checkpoint_sha256="b" * 64,
            candidate_checkpoint_sha256="d" * 64)


def test_exact_generalized_final_keeps_time_fixed():
    control = {
        "rel_l2_score": 94.0,
        "tke_rel_l2_score": 78.0,
        "mvpe_rel_l2_score": 93.0,
        "native_sps_score": 42.0,
        "point_score": 90.0,
    }
    candidate = {**control, "rel_l2_score": 94.1, "point_score": 90.05}

    report = subject.compare_metrics(candidate, control)

    assert report["exact_non_time_final_delta"] > 0.0
    assert report["delta"]["rel_l2_score"] == pytest.approx(0.1)
