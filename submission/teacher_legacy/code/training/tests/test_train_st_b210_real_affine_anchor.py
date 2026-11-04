from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_st_b210_full_function_anchor_real as subject


POLICY = subject.REAL_AFFINE_SOURCE_NORMALIZER_POLICY


def _config():
    return subject.SpatiotemporalB210Config(
        e0_channels=4, e1_channels=8, e2_channels=12, e3_channels=16,
        e0_blocks=1, e1_blocks=1, e2_blocks=1, e3_blocks=1,
        groups=2, channel_mlp_ratio=1.0, decoder_blocks=1)


def _normalizer(offset: float):
    return subject.base.ChannelNormalizer(
        torch.tensor([offset + .1, offset - .2, offset + .3]),
        torch.tensor([offset - .4, offset + .5, offset - .6]),
        torch.tensor([1.1, 1.2, 1.3]),
        torch.tensor([.9, .8, .7]),
        device=torch.device("cpu"))


def _single_and_normalizers():
    torch.manual_seed(711)
    single = subject.SpatiotemporalB210(
        _config(), "causal_temporal_candidate")
    sim = _normalizer(1.0)
    real = _normalizer(-1.0)
    single.set_normalizer_statistics(
        sim.mean_input, sim.mean_target, sim.std_input, sim.std_target)
    return single, sim, real


def _aligned_model(arm: str):
    single, sim, real = _single_and_normalizers()
    model = subject.FullFunctionAnchoredSTB210.from_single_model(
        single, subject.ARM_MODE[arm],
        subject._normalizer_payload(real), subject._normalizer_payload(real),
        source_normalizer_policy=POLICY)
    return model, single, sim, real


def _patch_counts(monkeypatch, model) -> None:
    monkeypatch.setattr(
        subject, "EXPECTED_PARAMETERS",
        sum(value.numel() for value in model.parameters()))
    monkeypatch.setattr(
        subject, "EXPECTED_STATE_TENSORS", len(model.state_dict()))
    monkeypatch.setattr(
        subject, "EXPECTED_SOURCE_PARAMETERS",
        sum(value.numel() for value in model.source_model.parameters()))
    monkeypatch.setitem(
        subject.EXPECTED_TRAINABLE_PARAMETERS, subject.ARMS[0],
        sum(value.numel() for value in model.source_model.parameters()))
    monkeypatch.setitem(
        subject.EXPECTED_TRAINABLE_PARAMETERS, subject.ARMS[1],
        sum(value.numel()
            for value in model.real_residual_model.parameters())
        + model.residual_gain.numel())


def _patch_tiny_start(monkeypatch, path: Path, single, sim) -> dict[str, object]:
    state = subject._state_snapshot(single)
    sim_values = subject._normalizer_payload(sim)
    file_sha = subject.sim_long.file_sha256(path)
    state_sha = subject.base.state_sha(state)
    normalizer_sha = subject.base.state_sha(sim_values)
    function_sha = subject._sealed_source_function_sha(state, sim_values)
    monkeypatch.setattr(subject, "EXPECTED_START_FILE_SHA256", file_sha)
    monkeypatch.setattr(subject, "EXPECTED_START_MODEL_STATE_SHA256", state_sha)
    monkeypatch.setattr(
        subject, "EXPECTED_START_NORMALIZER_SHA256", normalizer_sha)
    monkeypatch.setattr(
        subject, "EXPECTED_START_PHYSICAL_FUNCTION_SHA256", function_sha)
    monkeypatch.setattr(
        subject, "EXPECTED_SINGLE_PARAMETERS",
        sum(value.numel() for value in single.parameters()))
    monkeypatch.setattr(
        subject, "EXPECTED_SINGLE_STATE_TENSORS", len(single.state_dict()))
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha,
        "model_state_sha256": state_sha,
        "normalizer_sha256": normalizer_sha,
        "physical_function_sha256": function_sha,
    }


def _write_tiny_start(path: Path):
    single, sim, real = _single_and_normalizers()
    state = subject._state_snapshot(single)
    normalizer = subject._normalizer_payload(sim)
    torch.save({
        "schema": subject.st_lifecycle.ENDPOINT_SCHEMA,
        "lifecycle_schema": subject.st_lifecycle.SCHEMA,
        "arm": "temporal_candidate",
        "interaction_mode": "causal_temporal_candidate",
        "phase": "sim",
        "endpoint": "raw_u10000",
        "update": 10_000,
        "uq": False,
        "checkpoint_selected": False,
        "model_config": single.architecture_config(),
        "model_state_dict": state,
        "model_state_sha256": subject.base.state_sha(state),
        "normalizer": normalizer,
        "normalizer_sha256": subject.base.state_sha(normalizer),
        "source": {
            "sample_stream_sha256": subject.st_lifecycle.SIM_SCHEDULE_SHA256},
    }, path)
    return single, sim, real


def test_real_affine_contract_is_explicit_and_ffar_default_is_unchanged() -> None:
    legacy = subject.contract_for_args(4_000, 4)
    aligned = subject.contract_for_args(4_000, 4, POLICY)
    assert legacy["schema"] == subject.SCHEMA
    assert legacy["start"] == (
        "one_common_sealed_causal_ST_B210_Sim_raw_u10000")
    assert "source_normalizer_policy" not in legacy["normalizer_semantics"]
    assert aligned["schema"] == subject.REAL_AFFINE_SCHEMA
    assert aligned["normalizer_semantics"]["source_normalizer_policy"] == POLICY
    assert aligned["normalizer_semantics"][
        "source_and_residual_affine_bitwise_equal_at_update0"]
    for key in ("arms", "only_training_variable", "real", "ema_semantics",
                "precision", "score_gate", "early_stopping",
                "checkpoint_selection"):
        assert aligned[key] == legacy[key]
    assert subject.acknowledgement(POLICY) == (
        subject.REAL_AFFINE_ACKNOWLEDGEMENT)
    assert subject.acknowledgement("sealed_sim") == subject.ACKNOWLEDGEMENT


def test_real_affine_installs_one_coordinate_system_without_changing_weights(
        monkeypatch) -> None:
    candidate, single, sim, real = _aligned_model(subject.ARMS[1])
    _patch_counts(monkeypatch, candidate)
    models, identity = subject.construct_pair(single, real, real, POLICY)
    assert identity["bitwise_identical_initial_states"]
    assert identity["source_parameters_bitwise_equal_sealed_sim"]
    assert identity["source_and_residual_normalizers_bitwise_equal"]
    assert identity["initial_source_parameter_state_sha256"] == (
        subject.base.state_sha(subject._single_parameter_state(single)))
    assert identity["initial_source_model_state_sha256"] != (
        subject.base.state_sha(subject._state_snapshot(single)))
    for model in models.values():
        torch.testing.assert_close(
            model.source_model.persistence_scale,
            real.std_input / real.std_target, rtol=0.0, atol=0.0)
        assert subject.base.state_sha(model.normalizer_payload("sim")) == (
            subject.base.state_sha(model.normalizer_payload("real")))
        with pytest.raises(RuntimeError, match="fixed at pair construction"):
            model.set_normalizer_statistics()
    physical = torch.randn(1, 20, 8, 8, 3)
    with torch.inference_mode():
        source = models[subject.ARMS[0]].eval().source_physical(physical)
        for model in models.values():
            torch.testing.assert_close(
                model.eval().forward_physical(physical)["prediction_physical"],
                source, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="same Real normalizer"):
        subject.FullFunctionAnchoredSTB210.from_single_model(
            single, subject.ARM_MODE[subject.ARMS[1]],
            subject._normalizer_payload(sim),
            subject._normalizer_payload(real),
            source_normalizer_policy=POLICY)
    with pytest.raises(RuntimeError, match="bitwise common clean81"):
        subject.construct_pair(single, sim, real, POLICY)


def test_real_affine_endpoint_roundtrip_binds_weights_and_normalizer(
        tmp_path: Path, monkeypatch) -> None:
    start_path = tmp_path / "start.pt"
    single, sim, real = _write_tiny_start(start_path)
    start = _patch_tiny_start(monkeypatch, start_path, single, sim)
    probe, _single, _sim, _real = _aligned_model(subject.ARMS[1])
    _patch_counts(monkeypatch, probe)
    models, identity = subject.construct_pair(single, real, real, POLICY)
    source = {
        "common_sim_endpoint": start,
        "initial_complete_source_function_sha256":
            identity["initial_complete_source_function_sha256"],
        "initial_real_residual_branch_state_sha256":
            identity["initial_real_residual_branch_state_sha256"],
        "source_normalizer_policy": POLICY,
        "source_normalizer_sha256": identity["source_normalizer_sha256"],
        "sealed_sim_parameter_state_sha256":
            identity["sealed_sim_parameter_state_sha256"],
        "initial_source_parameter_state_sha256":
            identity["initial_source_parameter_state_sha256"],
    }
    seal = subject.seal_endpoint(
        tmp_path / "aligned.pt", model=models[subject.ARMS[1]],
        state=models[subject.ARMS[1]].state_dict(), real_normalizer=real,
        arm=subject.ARMS[1], endpoint="raw", update=4_000,
        source=source)
    loaded, _normalizer, payload = subject.load_endpoint(
        seal, torch.device("cpu"))
    assert seal["schema"] == subject.REAL_AFFINE_ENDPOINT_SCHEMA
    assert "source_normalizer" in payload and "sim_normalizer" not in payload
    assert payload["source_normalizer_sha256"] == (
        payload["real_normalizer_sha256"])
    assert subject.base.state_sha(subject._source_parameter_state(loaded)) == (
        identity["sealed_sim_parameter_state_sha256"])
    assert subject._source_function_sha(loaded) == (
        identity["initial_complete_source_function_sha256"])


def test_real_affine_checkpoint_smoke_reads_fit_only_and_checks_mechanics(
        tmp_path: Path, monkeypatch) -> None:
    start_path = tmp_path / "start.pt"
    single, sim, real = _write_tiny_start(start_path)
    _patch_tiny_start(monkeypatch, start_path, single, sim)
    probe, _single, _sim, _real = _aligned_model(subject.ARMS[1])
    _patch_counts(monkeypatch, probe)
    torch.manual_seed(991)
    fake_dataset = [
        (torch.randn(20, 8, 8, 3), torch.randn(20, 8, 8, 3))
        for _ in range(11)]
    first_batch = subject.make_sample_schedule(11, 4_000, 8, 4)[0]
    history, target = zip(*(fake_dataset[index] for index in first_batch))
    measured = subject.mechanical_alignment_metrics(
        single=single, sim_normalizer=sim, aligned_source=probe,
        real_normalizer=real, history_physical=torch.stack(history),
        target_physical=torch.stack(target))
    monkeypatch.setattr(subject, "MECHANICAL_EXPECTED", {
        key: round(measured[key], subject.MECHANICAL_REPORTED_DECIMALS[key])
        for key in subject.MECHANICAL_EXPECTED})
    monkeypatch.setattr(subject.real_long, "EXPECTED_OPTIMIZER_WINDOWS", 11)
    monkeypatch.setattr(
        subject.real_long, "load_clean_manifest",
        lambda _path: ([{"file": "tiny.h5"}], {"semantic_sha256": "tiny"}))

    def fake_load_real(_root, rows, *, phase, audit):
        assert phase == "fit"
        audit.before_open(rows[0], phase)
        audit.record_open(rows[0], phase, 40)
        return [torch.zeros(40, 8, 8, 2)], []

    monkeypatch.setattr(subject.real_long, "load_real_rows", fake_load_real)
    monkeypatch.setattr(
        subject.real_long, "normalizer_from_sparse_windows",
        lambda _trajectories, _windows, _device: (
            real, {"kind": "tiny-clean81"}))
    monkeypatch.setattr(
        subject.real_long, "dense_windows", lambda _values: list(range(11)))
    monkeypatch.setattr(
        subject.real_long, "RealWindowDataset",
        lambda _trajectories, _windows: fake_dataset)
    result = subject.real_affine_checkpoint_smoke(
        start_path, tmp_path / "smoke", torch.device("cpu"),
        real_dir=tmp_path, clean_manifest=tmp_path / "clean.json")
    assert result["schema"] == subject.REAL_AFFINE_SCHEMA
    assert result["smoke"] == "PASS"
    assert result["clean81_files_opened"] == 1
    assert result["clean716_files_opened"] == 0
    assert result["mechanical_acceptance"]["acceptance"] == "PASS"
