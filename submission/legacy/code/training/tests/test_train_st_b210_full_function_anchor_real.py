from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_st_b210_full_function_anchor_real as subject


def _config():
    return subject.SpatiotemporalB210Config(
        e0_channels=4, e1_channels=8, e2_channels=12, e3_channels=16,
        e0_blocks=1, e1_blocks=1, e2_blocks=1, e3_blocks=1,
        groups=2, channel_mlp_ratio=1.0, decoder_blocks=1)


def _normalizer(offset: float = 0.0):
    return subject.base.ChannelNormalizer(
        torch.tensor([offset + .1, offset - .2, offset + .3]),
        torch.tensor([offset - .4, offset + .5, offset - .6]),
        torch.tensor([1.1, 1.2, 1.3]),
        torch.tensor([.9, .8, .7]),
        device=torch.device("cpu"))


def _single_and_normalizers():
    torch.manual_seed(611)
    single = subject.SpatiotemporalB210(
        _config(), "causal_temporal_candidate")
    sim = _normalizer(1.0)
    single.set_normalizer_statistics(
        sim.mean_input, sim.mean_target, sim.std_input, sim.std_target)
    return single, sim, _normalizer(-1.0)


def _tiny_model(arm: str):
    single, sim, real = _single_and_normalizers()
    model = subject.FullFunctionAnchoredSTB210.from_single_model(
        single, subject.ARM_MODE[arm],
        subject._normalizer_payload(sim), subject._normalizer_payload(real))
    return model, sim, real


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
        sum(value.numel() for value in model.real_residual_model.parameters())
        + model.residual_gain.numel())


def test_contract_preregisters_lengths_seeds_roles_and_no_gate() -> None:
    contract = subject.contract_for_args(12_000, 17)
    assert contract["start"] == (
        "one_common_sealed_causal_ST_B210_Sim_raw_u10000")
    assert contract["only_training_variable"] == (
        "Real_gradient_destination_complete_source_or_additive_residual")
    assert contract["arms"][subject.ARMS[0]] == {
        "adaptation_mode": "trainable_source_control",
        "complete_source_trainable": True,
        "real_residual_trainable": False,
    }
    assert contract["arms"][subject.ARMS[1]][
        "complete_source_trainable"] is False
    assert contract["real"]["fixed_endpoint_updates"] == [4_000, 12_000]
    assert contract["real"]["seed"] == 17
    assert contract["real"]["scheduler_t_max"] == 12_000
    assert contract["normalizer_semantics"]["fusion_space"] == "physical_units"
    assert contract["ema_semantics"][
        "candidate_frozen_source_overlaid_bitwise_from_raw_at_endpoint"]
    assert contract["ema_semantics"][
        "control_frozen_residual_overlaid_bitwise_from_raw_at_endpoint"]
    assert contract["evaluation"][
        "sealed_Sim_start_rehashed_at_clean716_barrier"]
    assert contract["score_gate"] is None
    assert not contract["early_stopping"]
    assert not contract["checkpoint_selection"]
    assert not contract["uq"]


def test_stream_is_seeded_continuous_and_u4k_is_formal_in_12k() -> None:
    first = subject.make_sample_schedule(37, 12_000, 8, 4)
    second = subject.make_sample_schedule(37, 12_000, 8, 17)
    assert first != second
    assert first[:4_000] == subject.make_sample_schedule(37, 4_000, 8, 4)
    assert subject.milestone_updates(4_000) == (4_000,)
    assert subject.milestone_updates(12_000) == (4_000, 12_000)
    assert {
        key: subject.real_long.schedule_sha(subject.make_sample_schedule(
            subject.real_long.EXPECTED_OPTIMIZER_WINDOWS, key[0],
            subject.real_long.BATCH_SIZE, key[1]))
        for key in subject.EXPECTED_REAL_STREAM_SHA256
    } == subject.EXPECTED_REAL_STREAM_SHA256


def test_pair_is_bitwise_matched_but_trainable_roles_are_disjoint(monkeypatch):
    single, sim, real = _single_and_normalizers()
    probe = subject.FullFunctionAnchoredSTB210.from_single_model(
        single, subject.ARM_MODE[subject.ARMS[0]],
        subject._normalizer_payload(sim), subject._normalizer_payload(real))
    _patch_counts(monkeypatch, probe)
    models, identity = subject.construct_pair(single, sim, real)
    assert identity["bitwise_identical_initial_states"]
    assert identity["control_real_residual_frozen"]
    assert identity["candidate_real_residual_trainable"]
    assert identity["initial_source_model_state_sha256"] == (
        subject.base.state_sha(subject._state_snapshot(single)))


def test_function_hashes_cover_domain_normalizer_buffers() -> None:
    candidate, _sim, _real = _tiny_model(subject.ARMS[1])
    source_before = subject._source_function_sha(candidate)
    residual_before = subject._residual_branch_sha(candidate)
    with torch.no_grad():
        candidate._sim_mean_input.add_(.125)
    assert subject._source_function_sha(candidate) != source_before
    assert subject._residual_branch_sha(candidate) == residual_before
    with torch.no_grad():
        candidate._real_std_input.add_(.125)
    assert subject._residual_branch_sha(candidate) != residual_before


def test_ema_canonicalizer_restores_only_each_frozen_function() -> None:
    for arm in subject.ARMS:
        model, _sim, _real = _tiny_model(arm)
        raw = subject._state_snapshot(model)
        drifted = subject._state_snapshot(model)
        source_parameter = next(
            name for name in drifted if name.startswith("source_model.")
            and drifted[name].is_floating_point())
        residual_parameter = next(
            name for name in drifted
            if name.startswith("real_residual_model.")
            and drifted[name].is_floating_point())
        drifted[source_parameter].add_(1.0)
        drifted[residual_parameter].add_(2.0)
        drifted["residual_gain"].add_(2.5)
        drifted["_sim_mean_input"].add_(3.0)
        drifted["_real_mean_input"].add_(4.0)
        materialized = subject._canonicalize_ema_frozen_role(model, drifted)
        if arm == subject.ARMS[1]:
            torch.testing.assert_close(
                materialized[source_parameter], raw[source_parameter],
                rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                materialized["_sim_mean_input"], raw["_sim_mean_input"],
                rtol=0.0, atol=0.0)
            assert torch.equal(
                materialized[residual_parameter], drifted[residual_parameter])
            assert torch.equal(
                materialized["residual_gain"], drifted["residual_gain"])
        else:
            torch.testing.assert_close(
                materialized[residual_parameter], raw[residual_parameter],
                rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                materialized["_real_mean_input"], raw["_real_mean_input"],
                rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                materialized["residual_gain"], raw["residual_gain"],
                rtol=0.0, atol=0.0)
            assert torch.equal(
                materialized[source_parameter], drifted[source_parameter])


def test_optimizer_inventory_is_exactly_each_trainable_role() -> None:
    for arm in subject.ARMS:
        model, _sim, _real = _tiny_model(arm)
        optimizer = torch.optim.AdamW(
            (value for value in model.parameters() if value.requires_grad),
            lr=1.0e-3)
        subject._assert_optimizer_inventory(model, optimizer)
        frozen = next(value for value in model.parameters()
                      if not value.requires_grad)
        optimizer.param_groups[0]["params"].append(frozen)
        with pytest.raises(RuntimeError, match="optimizer inventory"):
            subject._assert_optimizer_inventory(model, optimizer)


def test_gradient_audit_enforces_zero_gate_transition_and_no_partial_leak() -> None:
    real = _normalizer(-1.0)
    torch.manual_seed(612)
    physical = torch.randn(1, 20, 8, 8, 3)
    history, _ = real.preprocess(physical, physical)
    target = torch.randn_like(history) + 1.5
    for arm in subject.ARMS:
        model, _sim, _real = _tiny_model(arm)
        model.train()
        optimizer = torch.optim.SGD(
            (value for value in model.parameters() if value.requires_grad),
            lr=.1)
        (model(history) - target).square().mean().backward()
        first = subject._gradient_audit(model, arm, 1)
        if arm == subject.ARMS[1]:
            assert not first["real_residual_has_nonzero_gradient"]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        (model(history) - target).square().mean().backward()
        second = subject._gradient_audit(model, arm, 2)
        if arm == subject.ARMS[1]:
            assert second["real_residual_has_nonzero_gradient"]
            leaked = next(model.source_model.parameters())
        else:
            leaked = next(model.real_residual_model.parameters())
        leaked.grad = torch.zeros_like(leaked)
        with pytest.raises(RuntimeError, match="gradient boundary"):
            subject._gradient_audit(model, arm, 2)


def _write_tiny_start(path: Path, schedule_sha: str) -> tuple[object, object]:
    single, sim, _real = _single_and_normalizers()
    state = subject._state_snapshot(single)
    normalizer = subject._normalizer_payload(sim)
    payload = {
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
        "source": {"sample_stream_sha256": schedule_sha},
    }
    torch.save(payload, path)
    return single, sim


def test_sim_loader_requires_file_model_and_schedule_seals(
        tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "start.pt"
    single, sim = _write_tiny_start(
        path, subject.st_lifecycle.SIM_SCHEDULE_SHA256)
    with pytest.raises(RuntimeError, match="file SHA changed"):
        subject.load_causal_sim_endpoint(path, torch.device("cpu"))
    monkeypatch.setattr(
        subject, "EXPECTED_START_FILE_SHA256", subject.sim_long.file_sha256(path))
    monkeypatch.setattr(
        subject, "EXPECTED_START_MODEL_STATE_SHA256",
        subject.base.state_sha(subject._state_snapshot(single)))
    monkeypatch.setattr(
        subject, "EXPECTED_START_NORMALIZER_SHA256",
        subject.base.state_sha(subject._normalizer_payload(sim)))
    monkeypatch.setattr(
        subject, "EXPECTED_START_PHYSICAL_FUNCTION_SHA256",
        subject._sealed_source_function_sha(
            subject._state_snapshot(single),
            subject._normalizer_payload(sim)))
    monkeypatch.setattr(
        subject, "EXPECTED_SINGLE_PARAMETERS",
        sum(value.numel() for value in single.parameters()))
    monkeypatch.setattr(
        subject, "EXPECTED_SINGLE_STATE_TENSORS", len(single.state_dict()))
    loaded, _normalizer, report = subject.load_causal_sim_endpoint(
        path, torch.device("cpu"))
    assert report["model_state_sha256"] == subject.base.state_sha(
        subject._state_snapshot(loaded))

    bad = tmp_path / "wrong_schedule.pt"
    _write_tiny_start(bad, "wrong")
    monkeypatch.setattr(
        subject, "EXPECTED_START_FILE_SHA256", subject.sim_long.file_sha256(bad))
    with pytest.raises(RuntimeError, match="tensor seal or topology"):
        subject.load_causal_sim_endpoint(bad, torch.device("cpu"))


def test_tiny_checkpoint_smoke_is_no_data_full_function_roundtrip(
        tmp_path: Path, monkeypatch) -> None:
    start = tmp_path / "start.pt"
    single, sim = _write_tiny_start(
        start, subject.st_lifecycle.SIM_SCHEDULE_SHA256)
    real = _normalizer(-1.0)
    probe = subject.FullFunctionAnchoredSTB210.from_single_model(
        single, subject.ARM_MODE[subject.ARMS[1]],
        subject._normalizer_payload(sim), subject._normalizer_payload(real))
    _patch_counts(monkeypatch, probe)
    monkeypatch.setattr(
        subject, "EXPECTED_START_FILE_SHA256",
        subject.sim_long.file_sha256(start))
    monkeypatch.setattr(
        subject, "EXPECTED_START_MODEL_STATE_SHA256",
        subject.base.state_sha(subject._state_snapshot(single)))
    monkeypatch.setattr(
        subject, "EXPECTED_START_NORMALIZER_SHA256",
        subject.base.state_sha(subject._normalizer_payload(sim)))
    monkeypatch.setattr(
        subject, "EXPECTED_START_PHYSICAL_FUNCTION_SHA256",
        subject._sealed_source_function_sha(
            subject._state_snapshot(single),
            subject._normalizer_payload(sim)))
    monkeypatch.setattr(
        subject, "EXPECTED_SINGLE_PARAMETERS",
        sum(value.numel() for value in single.parameters()))
    monkeypatch.setattr(
        subject, "EXPECTED_SINGLE_STATE_TENSORS", len(single.state_dict()))
    result = subject.checkpoint_smoke(
        start, tmp_path / "smoke", torch.device("cpu"))
    assert result["smoke"] == "PASS"
    assert result["data_files_opened"] == 0
    assert result["update0_output_bitwise_equals_complete_source"]
    assert result["causal_axis_calls_each_function"] == 4


def _assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for one, two in zip(left, right, strict=True):
            _assert_nested_equal(one, two)
    else:
        assert left == right


@pytest.mark.parametrize("arm", subject.ARMS)
def test_resume_after_step1_matches_uninterrupted_step2_exactly(
        tmp_path: Path, arm: str) -> None:
    model, sim, real = _tiny_model(arm)
    model.train()
    optimizer = torch.optim.AdamW(
        (value for value in model.parameters() if value.requires_grad),
        lr=2.0e-4, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=12_000)
    ema = subject.ema_lib.FP32EMAShadow(model, .999)
    schedule = [[index % 11] for index in range(4_000)]
    source = {
        "common_sim_endpoint": {
            "normalizer_sha256": subject.base.state_sha(
                subject._normalizer_payload(sim))},
        "initial_complete_source_function_sha256":
            subject._source_function_sha(model),
        "initial_real_residual_branch_state_sha256":
            subject._residual_branch_sha(model),
    }
    contract = {"schema": "tiny-resume"}
    torch.manual_seed(613)
    physical = torch.randn(1, 20, 8, 8, 3)
    history, target = real.preprocess(physical, physical.roll(1, dims=1))

    def step(local_model, local_optimizer, local_scheduler, local_ema, update):
        local_optimizer.zero_grad(set_to_none=True)
        noisy = history + .01 * torch.randn_like(history)
        loss = (local_model(noisy) - target).square().mean()
        loss.backward()
        local_optimizer.step()
        local_ema.update_after_optimizer_step(
            local_model, optimizer_step=update)
        local_scheduler.step()

    step(model, optimizer, scheduler, ema, 1)
    payload = subject._resume_payload(
        model=model, optimizer=optimizer, scheduler=scheduler, ema=ema,
        update=1, schedule=schedule, source=source,
        real_normalizer=real, device=torch.device("cpu"), trace=[],
        endpoint_seals={}, contract=contract)
    resume_path = tmp_path / f"{arm}.pt"
    torch.save(payload, resume_path)
    step(model, optimizer, scheduler, ema, 2)
    expected_model = subject._state_snapshot(model)
    expected_optimizer = optimizer.state_dict()
    expected_scheduler = scheduler.state_dict()
    expected_ema = ema.materialize_model_state(model)
    expected_rng = torch.get_rng_state().clone()

    resumed, _sim2, real2 = _tiny_model(arm)
    resumed.train()
    resumed_optimizer = torch.optim.AdamW(
        (value for value in resumed.parameters() if value.requires_grad),
        lr=2.0e-4, weight_decay=1.0e-5)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer, T_max=12_000)
    cursor, resumed_ema, trace, seals = subject._restore_resume(
        resume_path, model=resumed, optimizer=resumed_optimizer,
        scheduler=resumed_scheduler, expected_source=source,
        real_normalizer=real2, device=torch.device("cpu"),
        schedule=schedule, contract=contract)
    assert cursor == 1 and trace == [] and seals == {}
    step(resumed, resumed_optimizer, resumed_scheduler, resumed_ema, 2)
    _assert_nested_equal(subject._state_snapshot(resumed), expected_model)
    _assert_nested_equal(resumed_optimizer.state_dict(), expected_optimizer)
    _assert_nested_equal(resumed_scheduler.state_dict(), expected_scheduler)
    _assert_nested_equal(
        resumed_ema.materialize_model_state(resumed), expected_ema)
    torch.testing.assert_close(
        torch.get_rng_state(), expected_rng, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("arm", subject.ARMS)
def test_endpoint_roundtrip_and_cross_normalizer_rejection(
        tmp_path: Path, monkeypatch, arm: str) -> None:
    model, sim, real = _tiny_model(arm)
    _patch_counts(monkeypatch, model)
    monkeypatch.setattr(subject, "EXPECTED_START_FILE_SHA256", "tiny-file")
    monkeypatch.setattr(
        subject, "EXPECTED_START_MODEL_STATE_SHA256",
        subject._source_model_sha(model))
    monkeypatch.setattr(
        subject, "EXPECTED_START_NORMALIZER_SHA256",
        subject.base.state_sha(subject._normalizer_payload(sim)))
    monkeypatch.setattr(
        subject, "EXPECTED_START_PHYSICAL_FUNCTION_SHA256",
        subject._source_function_sha(model))
    start = {
        "path": str(tmp_path / "start.pt"),
        "file_sha256": "tiny-file",
        "model_state_sha256": subject._source_model_sha(model),
        "normalizer_sha256": subject.base.state_sha(
            subject._normalizer_payload(sim)),
        "physical_function_sha256": subject._source_function_sha(model),
    }
    source = {
        "common_sim_endpoint": start,
        "initial_complete_source_function_sha256":
            subject._source_function_sha(model),
        "initial_real_residual_branch_state_sha256":
            subject._residual_branch_sha(model),
    }
    seal = subject.seal_endpoint(
        tmp_path / "endpoint.pt", model=model, state=model.state_dict(),
        real_normalizer=real, arm=arm, endpoint="raw", update=4_000,
        source=source)
    loaded, loaded_real, payload = subject.load_endpoint(
        seal, torch.device("cpu"))
    assert subject.base.state_sha(subject._state_snapshot(loaded)) == (
        seal["model_state_sha256"])
    assert subject._source_function_sha(loaded) == (
        source["initial_complete_source_function_sha256"])
    assert subject.base.state_sha(subject._normalizer_payload(loaded_real)) == (
        payload["real_normalizer_sha256"])
    with pytest.raises(RuntimeError, match="domain normalizers differ"):
        subject.seal_endpoint(
            tmp_path / "bad.pt", model=model, state=model.state_dict(),
            real_normalizer=_normalizer(9.0), arm=arm, endpoint="raw",
            update=4_000, source=source)

    bad_protocol_payload = torch.load(
        seal["path"], map_location="cpu", weights_only=False)
    bad_protocol_payload["uq"] = True
    bad_protocol_path = tmp_path / "self_resealed_bad_protocol.pt"
    torch.save(bad_protocol_payload, bad_protocol_path)
    bad_protocol_seal = {
        **seal, "path": str(bad_protocol_path),
        "file_sha256": subject.sim_long.file_sha256(bad_protocol_path)}
    with pytest.raises(RuntimeError, match="protocol identity"):
        subject.load_endpoint(bad_protocol_seal, torch.device("cpu"))

    drifted_payload = torch.load(
        seal["path"], map_location="cpu", weights_only=False)
    state = subject._state_snapshot(drifted_payload["model_state_dict"])
    if arm == subject.ARMS[1]:
        name = next(key for key in state if key.startswith("source_model.")
                    and state[key].is_floating_point())
    else:
        name = next(
            key for key in state if key.startswith("real_residual_model.")
            and state[key].is_floating_point())
    state[name].reshape(-1)[0].add_(.25)
    drifted_payload["model_state_dict"] = state
    drifted_payload["model_state_sha256"] = subject.base.state_sha(state)
    drifted_payload["complete_source_function_sha256"] = (
        subject._source_function_sha(state))
    drifted_payload["real_residual_branch_state_sha256"] = (
        subject._residual_branch_sha(state))
    drifted_path = tmp_path / "self_resealed_frozen_role.pt"
    torch.save(drifted_payload, drifted_path)
    drifted_seal = {
        **seal, "path": str(drifted_path),
        "file_sha256": subject.sim_long.file_sha256(drifted_path),
        "model_state_sha256": drifted_payload["model_state_sha256"],
    }
    expected_message = (
        "candidate frozen source" if arm == subject.ARMS[1]
        else "control frozen residual")
    with pytest.raises(RuntimeError, match=expected_message):
        subject.load_endpoint(drifted_seal, torch.device("cpu"))


def _fake_seal(path: Path, value: str) -> dict[str, object]:
    path.write_text(value, encoding="ascii")
    return {
        "path": str(path),
        "file_sha256": subject.sim_long.file_sha256(path),
    }


def test_tail_barrier_rehashes_start_and_every_endpoint(tmp_path: Path) -> None:
    audit = subject.EndpointAccessAudit()
    row = {"file": "one.h5"}
    with pytest.raises(RuntimeError, match="before every FFAR endpoint"):
        audit.before_open(row, "clean716")
    seals = {"start/sim_source": _fake_seal(
        tmp_path / "start.pt", "start")}
    for index, key in enumerate(sorted(
            subject.prescribed_barrier_keys((4_000,)) - {"start/sim_source"})):
        _real, arm, raw_update, endpoint = key.split("/")
        seal = _fake_seal(tmp_path / f"{index}.pt", key)
        seal.update({
            "arm": arm, "update": int(raw_update[1:]),
            "endpoint": endpoint})
        seals[key] = seal
    expected = subject.prescribed_barrier_keys((4_000,))
    audit.record_endpoint_seals(seals, expected)
    audit.before_open(row, "clean716")
    Path(seals["start/sim_source"]["path"]).write_text(
        "changed", encoding="ascii")
    with pytest.raises(RuntimeError, match="changed after barrier"):
        audit.before_open({"file": "two.h5"}, "clean716")
    with pytest.raises(RuntimeError, match="fit prefix opened after"):
        audit.before_open({"file": "three.h5"}, "fit")
    with pytest.raises(ValueError, match="unsupported"):
        audit.before_open({"file": "four.h5"}, "unknown")


def test_tail_barrier_rejects_missing_extra_duplicate_and_wrong_identity(
        tmp_path: Path) -> None:
    expected = subject.prescribed_barrier_keys((4_000,))
    seals = {"start/sim_source": _fake_seal(
        tmp_path / "start.pt", "start")}
    for index, key in enumerate(sorted(expected - {"start/sim_source"})):
        _real, arm, raw_update, endpoint = key.split("/")
        seal = _fake_seal(tmp_path / f"e{index}.pt", key)
        seal.update({"arm": arm, "update": int(raw_update[1:]),
                     "endpoint": endpoint})
        seals[key] = seal
    missing = dict(seals)
    missing.pop(next(iter(expected - {"start/sim_source"})))
    with pytest.raises(RuntimeError, match="inventory"):
        subject.EndpointAccessAudit().record_endpoint_seals(missing, expected)
    extra = {**seals, "unexpected": dict(seals["start/sim_source"])}
    with pytest.raises(RuntimeError, match="inventory"):
        subject.EndpointAccessAudit().record_endpoint_seals(extra, expected)
    duplicate = {key: dict(value) for key, value in seals.items()}
    real_keys = sorted(expected - {"start/sim_source"})
    duplicate[real_keys[1]]["path"] = duplicate[real_keys[0]]["path"]
    duplicate[real_keys[1]]["file_sha256"] = duplicate[real_keys[0]][
        "file_sha256"]
    with pytest.raises(RuntimeError, match="not unique"):
        subject.EndpointAccessAudit().record_endpoint_seals(duplicate, expected)
    wrong = {key: dict(value) for key, value in seals.items()}
    wrong[real_keys[0]]["endpoint"] = "wrong"
    with pytest.raises(RuntimeError, match="identity differs"):
        subject.EndpointAccessAudit().record_endpoint_seals(wrong, expected)


def test_clean716_manifest_hash_read_and_tail_open_follow_barrier(
        monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    manifest = tmp_path / "tail.json"
    manifest.write_text("[]", encoding="utf-8")

    class SpyAccess:
        def assert_barrier_intact(self):
            events.append("barrier_rehash")

    def file_sha(path):
        assert path == manifest
        events.append("manifest_hash")
        return "manifest-sha"

    def load_rows(path, fit_rows):
        assert path == manifest and fit_rows == [{"file": "fit.h5"}]
        events.append("manifest_read")
        return [{"file": "fit.h5", "aoa": 10}]

    window = SimpleNamespace(trajectory=0)

    def load_real(_real_dir, rows, *, phase, audit):
        assert rows[0]["aoa"] == 10 and phase == "clean716"
        assert isinstance(audit, SpyAccess)
        events.append("tail_h5_open")
        return [torch.zeros(40, 8, 8, 2)], [window]

    monkeypatch.setattr(subject.sim_long, "file_sha256", file_sha)
    monkeypatch.setattr(subject.real_long, "load_clean716_rows", load_rows)
    monkeypatch.setattr(subject.real_long, "load_real_rows", load_real)
    monkeypatch.setattr(
        subject.real_long, "RealWindowDataset",
        lambda trajectories, windows: (trajectories, windows))
    digest, dataset, aoas = subject.load_clean716_after_barrier(
        evaluation_manifest=manifest,
        fit_rows=[{"file": "fit.h5"}], real_dir=tmp_path,
        access=SpyAccess())
    assert events == [
        "barrier_rehash", "manifest_hash", "manifest_read", "tail_h5_open"]
    assert digest == "manifest-sha" and len(dataset) == 2 and aoas == [10]


def test_start_artifact_is_hard_pinned() -> None:
    assert subject.EXPECTED_START_FILE_SHA256 == (
        "debb322723ea0f7b895c798c04825ec2899349ae9acbabd3018cd89d80e2f74c")
    assert subject.EXPECTED_START_MODEL_STATE_SHA256 == (
        "a474c1b535eed28349be3374ba6cc7ab66672cd2c366fd95cae176211008ccf3")
    assert subject.EXPECTED_START_NORMALIZER_SHA256 == (
        "90ea373cc0f38d3f576215c24d57491dd2dcb18058fea0965618f4781a36a69c")
    assert subject.EXPECTED_START_PHYSICAL_FUNCTION_SHA256 == (
        "4cc64fb26357e18fc9d8a0b35416961300f586fbb3259d4555f324ed3e835030")


def test_registered_precision_is_highest_with_tf32_disabled(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        subject.torch, "set_float32_matmul_precision", calls.append)
    monkeypatch.setattr(subject.torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(subject.torch.backends.cudnn, "allow_tf32", True)
    subject.configure_strict_fp32(torch.device("cuda:0"))
    assert calls == ["highest"]
    assert subject.torch.backends.cuda.matmul.allow_tf32 is False
    assert subject.torch.backends.cudnn.allow_tf32 is False


def test_formal_rejects_dirty_git_before_cuda_or_data(
        monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subject.base, "git_report", lambda: {
        "clean": False, "porcelain": [" M tracked.py"],
        "commit": "bad", "branch": "bad"})
    args = SimpleNamespace(
        execute=True, acknowledge_protocol=subject.ACKNOWLEDGEMENT,
        start_sim=tmp_path / "start.pt", real_dir=tmp_path / "real",
        clean_manifest=tmp_path / "clean.json",
        evaluation_manifest=tmp_path / "tail.json",
        output_dir=tmp_path / "out", device="cuda:0",
        real_updates=4_000, real_seed=4, checkpoint_every=500,
        resume=False)
    with pytest.raises(RuntimeError, match="requires clean Git"):
        subject.run(args)
