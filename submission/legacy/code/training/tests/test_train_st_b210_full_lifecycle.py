from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_st_b210_full_lifecycle as subject


def _normalizer(device: torch.device):
    return subject.base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), device)


def _tiny_model(arm: str):
    config = subject.st_screen.model_config(tiny=True)
    return subject.SpatiotemporalB210(config, subject.ARM_MODE[arm]).float()


def test_contract_is_full_paired_sim_to_real_without_selection() -> None:
    contract = subject.lifecycle_contract()
    assert contract["only_model_variable"] == "interaction_mode"
    assert contract["effective_grid"] == [32, 64]
    assert "no_cross_frame_interaction" in contract["control_semantics"]
    assert "20x20_observed_to_future_projection" in contract["control_semantics"]
    assert contract["sim"]["updates"] == 10_000
    assert contract["sim"]["stride"] == 20
    assert contract["real"]["updates"] == 4_000
    assert contract["real"]["optimizer_stride"] == 5
    assert contract["real"]["scheduler_t_max"] == 12_000
    assert contract["real"]["endpoints_each"] == ["raw", "ema"]
    assert contract["real"]["each_arm_starts_from_its_own_sim_terminal"]
    assert contract["real"]["unmeasured_pressure"] == (
        "exact_zero_in_dataset_and_after_input_augmentation")
    assert contract["evaluation"][
        "clean716_first_read_after_all_four_real_endpoint_seals"]
    assert contract["score_gate"] is None
    assert not contract["early_stopping"]
    assert not contract["checkpoint_selection"]
    assert not contract["uq"]
    assert not contract["packaging"]


def test_reused_sim_and_real_streams_are_the_frozen_mature_streams() -> None:
    assert subject.sim_long.validate_formal_schedule(
        subject.sim_long.make_schedule(4_800)) == subject.SIM_SCHEDULE_SHA256
    schedule = subject.real_long.make_sample_schedule(9_933, 4_000, 8)
    assert subject.real_long.schedule_sha(schedule) == subject.REAL_SCHEDULE_SHA256
    assert len(schedule) == 4_000


def test_real_pressure_stays_exact_zero_through_mature_augmentation() -> None:
    history = torch.randn(2, 20, 32, 64, 3)
    history[..., 2] = 0.0
    torch.manual_seed(123)
    augmented = subject.real_long.augment_real_history(
        history, _normalizer(torch.device("cpu")))
    assert torch.equal(
        augmented[..., 2], torch.zeros_like(augmented[..., 2]))


def test_each_real_arm_is_bound_to_its_own_sim_terminal(tmp_path: Path) -> None:
    normalizer = _normalizer(torch.device("cpu"))
    manifest = {"semantic_sha256": "manifest"}
    clean_manifest = tmp_path / "clean.json"
    clean_manifest.write_text("{}")
    git = {"commit": "same", "branch": "same", "clean": True,
           "porcelain": []}
    control_seal = {"arm": subject.ARMS[0], "model_state_sha256": "control"}
    candidate_seal = {"arm": subject.ARMS[1], "model_state_sha256": "candidate"}
    control = subject._real_source(
        subject.ARMS[0], control_seal, manifest, clean_manifest,
        normalizer, git)
    candidate = subject._real_source(
        subject.ARMS[1], candidate_seal, manifest, clean_manifest,
        normalizer, git)
    assert control["sim_endpoint"] == control_seal
    assert candidate["sim_endpoint"] == candidate_seal
    assert control["sim_endpoint"] != candidate["sim_endpoint"]


def test_compact_endpoint_roundtrip_is_exact_for_tiny_real_model(
        tmp_path: Path, monkeypatch) -> None:
    arm = subject.ARMS[1]
    model = _tiny_model(arm)
    monkeypatch.setattr(
        subject, "EXPECTED_PARAMETERS",
        sum(parameter.numel() for parameter in model.parameters()))
    monkeypatch.setattr(subject, "EXPECTED_STATE_TENSORS", len(model.state_dict()))
    normalizer = _normalizer(torch.device("cpu"))
    model.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    seal = subject.seal_endpoint(
        tmp_path / "endpoint.pt", model=model, state=None,
        normalizer=normalizer, arm=arm, phase="sim",
        endpoint="raw_u10000", update=10_000,
        source={"sample_stream": subject.SIM_SCHEDULE_SHA256})
    loaded, loaded_normalizer, payload = subject.load_endpoint(
        seal, torch.device("cpu"))
    assert subject.base.state_sha(subject._state_snapshot(loaded)) == (
        seal["model_state_sha256"])
    assert subject.base.state_sha(
        subject._normalizer_payload(loaded_normalizer)) == (
            seal["normalizer_sha256"])
    assert loaded.interaction_mode == subject.ARM_MODE[arm]
    assert payload["checkpoint_selected"] is False
    assert payload["uq"] is False
    with (tmp_path / "endpoint.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="file seal changed"):
        subject.load_endpoint(seal, torch.device("cpu"))


def _fake_seal(path: Path, key: str) -> dict[str, object]:
    path.write_bytes(key.encode("ascii"))
    return {"path": str(path), "file_sha256": subject.sim_long.file_sha256(path)}


def test_tail_barrier_needs_two_sim_and_all_four_real_seals(
        tmp_path: Path) -> None:
    audit = subject.LifecycleAccessAudit()
    row = {"file": "one.h5"}
    with pytest.raises(RuntimeError, match="before all lifecycle endpoints"):
        audit.before_open(row, "clean716")
    keys = {
        *(f"sim/{arm}" for arm in subject.ARMS),
        *(f"real/{arm}/{endpoint}" for arm in subject.ARMS
          for endpoint in subject.ENDPOINTS),
    }
    seals = {
        key: _fake_seal(tmp_path / f"{index}.pt", key)
        for index, key in enumerate(sorted(keys))}
    with pytest.raises(RuntimeError, match="all four Real seals"):
        audit.record_endpoint_seals(
            {key: value for index, (key, value) in enumerate(seals.items())
             if index})
    audit.record_endpoint_seals(seals)
    audit.before_open(row, "clean716")
    audit.record_open({"file": "one.h5"}, "clean716", 100)
    assert audit.events[-1]["all_lifecycle_endpoints_sealed_before_open"]


def test_candidate_control_comparison_and_retention_have_no_gate() -> None:
    control = {
        "pooled": {"rel_l2_score": 90., "tke_rel_l2_score": 80.,
                   "mvpe_rel_l2_score": 70.,
                   "public_point_contribution_46_10_9": 55.7},
        "by_aoa": {},
    }
    candidate = {
        "pooled": {"rel_l2_score": 91., "tke_rel_l2_score": 82.,
                   "mvpe_rel_l2_score": 73.,
                   "public_point_contribution_46_10_9": 56.63},
        "by_aoa": {},
    }
    for aoa in ("0", "5", "10", "15", "20"):
        control["by_aoa"][aoa] = dict(control["pooled"])
        candidate["by_aoa"][aoa] = dict(candidate["pooled"])
    panel = {"control": control, "candidate": candidate}
    comparison = subject._compare_panel(panel, "control", "candidate")
    assert comparison["pooled_delta_temporal_minus_spatial"] == {
        "rel_l2_score": 1.0,
        "tke_rel_l2_score": 2.0,
        "mvpe_rel_l2_score": 3.0,
        "public_point_contribution_46_10_9": pytest.approx(.93),
    }
    assert subject._retention(.25, .5) == .5
    assert subject._retention(.25, 0.) is None
    assert subject.lifecycle_contract()["score_gate"] is None


def test_formal_rejects_dirty_git_before_cuda_or_data(
        monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subject.base, "git_report", lambda: {
        "clean": False, "porcelain": [" M tracked.py"],
        "commit": "bad", "branch": "bad"})
    args = SimpleNamespace(
        execute=True, acknowledge_protocol=subject.ACKNOWLEDGEMENT,
        sim_dir=tmp_path / "sim", real_dir=tmp_path / "real",
        clean_manifest=tmp_path / "clean.json",
        evaluation_manifest=tmp_path / "tail.json",
        output_dir=tmp_path / "out", device="cuda:0",
        sim_checkpoint_every=2_000, real_checkpoint_every=250,
        resume=False)
    with pytest.raises(RuntimeError, match="requires clean Git"):
        subject.run(args)


def test_formal_keeps_default_fp32_matmul_precision(
        monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subject.base, "git_report", lambda: {
        "clean": True, "porcelain": [], "commit": "clean",
        "branch": "test"})
    monkeypatch.setattr(subject.torch.cuda, "is_available", lambda: True)
    precision_calls = []
    monkeypatch.setattr(
        subject.torch, "set_float32_matmul_precision",
        lambda value: precision_calls.append(value))
    monkeypatch.setattr(
        subject.base, "inventory",
        lambda _path: (_ for _ in ()).throw(RuntimeError("inventory sentinel")))
    evaluation_manifest = tmp_path / "tail.json"
    evaluation_manifest.write_text("[]")
    args = SimpleNamespace(
        execute=True, acknowledge_protocol=subject.ACKNOWLEDGEMENT,
        sim_dir=tmp_path / "sim", real_dir=tmp_path / "real",
        clean_manifest=tmp_path / "clean.json",
        evaluation_manifest=evaluation_manifest,
        output_dir=tmp_path / "out", device="cuda:0",
        sim_checkpoint_every=2_000, real_checkpoint_every=250,
        resume=False)
    with pytest.raises(RuntimeError, match="inventory sentinel"):
        subject.run(args)
    assert precision_calls == []
