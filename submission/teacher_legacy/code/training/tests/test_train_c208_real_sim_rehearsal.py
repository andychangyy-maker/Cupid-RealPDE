from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "train_c208_real_sim_rehearsal.py")
SPEC = importlib.util.spec_from_file_location("c208_sim_rehearsal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def args(weight=0.0, cache=None, cache_sha=None):
    return argparse.Namespace(
        start_sim=Path("start.pth"),
        start_sim_sha256=subject.FORMAL_START_SIM_SHA256,
        output_dir=Path("unused-output"),
        updates=5000,
        real_dir=Path("real"),
        sim_dir=Path("sim"),
        real_stats=Path("stats.pt"),
        rehearsal_weight=weight,
        p1_cache=cache,
        p1_cache_sha256=cache_sha,
        identity_summary=(
            Path("identity-summary.json") if weight else None),
        identity_summary_sha256=(
            subject.FORMAL_IDENTITY_SUMMARY_SHA256 if weight else None),
        device="cpu",
        num_workers=0,
    )


def test_only_identity_and_single_preregistered_weight_are_accepted(tmp_path):
    identity = args()
    identity.output_dir = tmp_path / "identity"
    subject.validate_args(identity)
    assert identity.rehearsal_weight == 0.0

    formal = args(
        0.05, Path("cache.pt"), subject.FORMAL_P1_CACHE_SHA256)
    formal.output_dir = tmp_path / "formal"
    subject.validate_args(formal)
    assert formal.rehearsal_weight == subject.FORMAL_REHEARSAL_WEIGHT

    swept = args(
        0.01, Path("cache.pt"), subject.FORMAL_P1_CACHE_SHA256)
    swept.output_dir = tmp_path / "swept"
    with pytest.raises(ValueError, match="exactly 0.*0.05"):
        subject.validate_args(swept)


def test_start_and_cache_hashes_are_the_preregistered_artifacts(tmp_path):
    wrong_start = args()
    wrong_start.start_sim_sha256 = "a" * 64
    wrong_start.output_dir = tmp_path / "wrong-start"
    with pytest.raises(ValueError, match="pre-registered Sim start"):
        subject.validate_args(wrong_start)

    wrong_cache = args(0.05, Path("cache.pt"), "b" * 64)
    wrong_cache.output_dir = tmp_path / "wrong-cache"
    with pytest.raises(ValueError, match="pre-registered P1 cache"):
        subject.validate_args(wrong_cache)


def test_identity_control_rejects_cache_and_consumes_no_rng(tmp_path):
    invalid = args(
        0.0, Path("cache.pt"), subject.FORMAL_P1_CACHE_SHA256)
    invalid.output_dir = tmp_path / "invalid"
    with pytest.raises(ValueError, match="identity control"):
        subject.validate_args(invalid)

    identity = args()
    identity.output_dir = tmp_path / "identity"
    subject.validate_args(identity)
    torch.manual_seed(1234)
    state = torch.get_rng_state().clone()
    assert subject.prepare_sim_rehearsal(
        identity, torch.device("cpu")) is None
    assert torch.equal(torch.get_rng_state(), state)


def test_cache_sha_is_checked_before_cache_deserialization(monkeypatch, tmp_path):
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"not a torch payload")
    formal = args(0.05, cache, subject.FORMAL_P1_CACHE_SHA256)
    formal.output_dir = tmp_path / "formal"
    subject.validate_args(formal)

    called = False

    def fail_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("cache deserialization must not happen")

    monkeypatch.setattr(subject, "load_and_validate_cache", fail_load)
    with pytest.raises(RuntimeError, match="P1 cache SHA-256"):
        subject.prepare_sim_rehearsal(
            formal, torch.device("cpu"), sim_normalizer=object())
    assert called is False


def test_frozen_sim_normalizer_uses_checkpoint_scales_not_real_buffers():
    payload = {
        "normalizer": {
            "mean_input": [1.0, 2.0, 3.0],
            "mean_target": [10.0, 20.0, 30.0],
            "std_input": [2.0, 4.0, 5.0],
            "std_target": [5.0, 10.0, 15.0],
        },
    }
    normalizer, values, digest = subject.frozen_sim_normalizer(
        payload, torch.device("cpu"))
    history = torch.tensor([[[[[3.0, 6.0, 8.0]]]]])
    target = torch.tensor([[[[[15.0, 30.0, 45.0]]]]])

    history_norm, target_norm = normalizer.preprocess(history, target)

    assert torch.equal(history_norm, torch.ones_like(history_norm))
    assert torch.equal(target_norm, torch.ones_like(target_norm))
    assert values == payload["normalizer"]
    assert digest == subject.canonical_sha256(payload["normalizer"])


def test_sequential_update_orders_graphs_and_takes_one_step():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    events = []

    def real_backward():
        events.append("real")
        loss = 0.5 * (model.weight - 0.2).square().sum()
        loss.backward()
        return {"real": float(loss.detach())}

    def rehearsal_backward():
        events.append("sim")
        assert model.weight.item() == 0.0
        loss = 0.5 * (model.weight - 0.3).square().sum()
        loss.backward()
        return {"sim": float(loss.detach())}

    values = subject.sequential_backward_update(
        optimizer, model, real_backward, rehearsal_backward)

    assert events == ["real", "sim"]
    assert values.keys() == {"real", "sim"}
    assert model.weight.item() == pytest.approx(0.05)


def test_zero_rehearsal_update_is_rng_and_gradient_identity():
    first = torch.nn.Linear(1, 1, bias=False)
    second = torch.nn.Linear(1, 1, bias=False)
    second.load_state_dict(first.state_dict())
    first_optimizer = torch.optim.SGD(first.parameters(), lr=0.1)
    second_optimizer = torch.optim.SGD(second.parameters(), lr=0.1)

    def callback(model):
        def backward():
            loss = model.weight.square().sum()
            loss.backward()
            return {"loss": float(loss.detach())}
        return backward

    torch.manual_seed(99)
    state = torch.get_rng_state().clone()
    subject.sequential_backward_update(
        first_optimizer, first, callback(first), None)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    subject.sequential_backward_update(
        second_optimizer, second, callback(second), None)

    assert torch.equal(first.weight, second.weight)
    assert torch.equal(after, torch.get_rng_state())


def test_frozen_loss_scale_and_endpoint_gate_are_explicit():
    assert subject.FORMAL_REHEARSAL_WEIGHT == 0.05
    assert subject.FORMAL_P1_DISTILL_WEIGHT == 0.005
    assert subject.FORMAL_REQUESTED_UPDATES == 5000
    assert subject.FORMAL_GATE_UPDATE == 2000
    assert subject.FORMAL_TERMINAL_UPDATE == 4000
    assert subject.FORMAL_SCHEDULER_T_MAX == 5000
    assert subject.EXPECTED_REHEARSAL_TO_REAL_LOSS_RATIO == pytest.approx(
        0.06068, rel=2e-3)
    assert subject.NON_TIME_COEFFICIENTS == {
        "rel_l2_score": 0.47201126,
        "tke_rel_l2_score": 0.08586562,
        "mvpe_rel_l2_score": 0.09410214,
        "native_sps_score": 0.24699273,
    }

    gate = subject.gate_contract()
    assert gate["point_score_delta_min"] == 0.03
    assert gate["native_sps_delta_floor"] == -0.10
    assert gate["on_failure"] == "STOP_SIM_REHEARSAL_NO_WEIGHT_SWEEP"


def test_identity_summary_contract_and_exact_step2000_metrics(
        monkeypatch, tmp_path):
    summary_path = tmp_path / "identity.json"
    summary_path.write_text(
        json.dumps({
            "training_config": dict(subject.IDENTITY_TRAINING_CONTRACT),
            "history": [{
                "iteration": 2000,
                **subject.FORMAL_IDENTITY_STEP2000_METRICS,
            }],
        }),
        encoding="utf-8")
    formal = args(
        0.05, Path("cache.pt"), subject.FORMAL_P1_CACHE_SHA256)
    formal.identity_summary = summary_path
    formal.identity_summary_sha256 = subject.FORMAL_IDENTITY_SUMMARY_SHA256
    monkeypatch.setattr(
        subject, "sha256", lambda _path: subject.FORMAL_IDENTITY_SUMMARY_SHA256)

    identity = subject.load_identity_step2000(formal)

    assert identity["metrics"] == subject.FORMAL_IDENTITY_STEP2000_METRICS
    assert identity["sha256"] == subject.FORMAL_IDENTITY_SUMMARY_SHA256


def test_step2000_gate_passes_once_then_forbids_second_fixed725_access(
        monkeypatch):
    candidate = dict(subject.FORMAL_IDENTITY_STEP2000_METRICS)
    candidate.update({
        "rel_l2_score": candidate["rel_l2_score"] + 0.04,
        "tke_rel_l2_score": candidate["tke_rel_l2_score"] + 0.04,
        "mvpe_rel_l2_score": candidate["mvpe_rel_l2_score"] + 0.04,
        "point_score": candidate["point_score"] + 0.04,
        "native_sps_score": candidate["native_sps_score"] + 0.01,
    })
    monkeypatch.setattr(
        subject.ensemble_train, "evaluate",
        lambda *_args, **_kwargs: candidate)
    real = {
        "validation_access_updates": [],
        "validation_loader": object(),
        "normalizer": object(),
    }
    identity = {
        "metrics": dict(subject.FORMAL_IDENTITY_STEP2000_METRICS),
        "sha256": subject.FORMAL_IDENTITY_SUMMARY_SHA256,
    }

    gate = subject.evaluate_step2000_gate(
        object(), real, torch.device("cpu"), identity)

    assert gate["pass"] is True
    assert gate["status"] == "GO_CONTINUE_TO_UPDATE4000"
    assert real["validation_access_updates"] == [2000]
    with pytest.raises(RuntimeError, match="already accessed"):
        subject.evaluate_step2000_gate(
            object(), real, torch.device("cpu"), identity)


@pytest.mark.parametrize("gate_pass, expected_updates, expected_saves", [
    (False, 2, 0),
    (True, 4, 1),
])
def test_gate_stops_or_continues_same_optimizer_process(
        monkeypatch, tmp_path, gate_pass, expected_updates, expected_saves):
    monkeypatch.setattr(subject, "FORMAL_GATE_UPDATE", 2)
    monkeypatch.setattr(subject, "FORMAL_TERMINAL_UPDATE", 4)
    monkeypatch.setattr(subject, "FORMAL_LOG_INTERVAL", 2)
    model = torch.nn.Linear(1, 1, bias=False)
    saves = []

    def backward(model, *_args, **_kwargs):
        loss = model.weight.square().sum()
        loss.backward()
        return {"real_loss": float(loss.detach())}

    def gate(*_args, **_kwargs):
        return {
            "status": (
                "GO_CONTINUE_TO_UPDATE4000" if gate_pass
                else "STOP_SIM_REHEARSAL_UPDATE2000_GATE"),
            "pass": gate_pass,
        }

    monkeypatch.setattr(subject, "backward_real_supervised", backward)
    monkeypatch.setattr(subject, "evaluate_step2000_gate", gate)
    monkeypatch.setattr(
        subject.ensemble_train.common, "normalizer_dict", lambda _value: {})
    monkeypatch.setattr(
        subject.ensemble_train.common, "save_checkpoint",
        lambda path, *_args, **_kwargs: saves.append(path))
    run_args = argparse.Namespace(
        updates=5, output_dir=tmp_path, rehearsal_weight=0.0)
    real = {"loader": [None], "normalizer": object()}

    result = subject.train_endpoint(
        run_args, model, real, None, None, torch.device("cpu"))

    assert result["completed_updates"] == expected_updates
    assert len(saves) == expected_saves


def test_sim_rehearsal_uses_point_forward_and_frozen_sim_normalizer(
        monkeypatch):
    events = []

    class PointOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, value):
            assert self.current_stats == "sim"
            events.append("point_forward")
            return value * self.scale

        def forward_uq_training(self, _value):
            raise AssertionError("Sim rehearsal must not enter the UQ path")

    class FrozenNormalizer:
        label = "sim"
        mean_input = torch.zeros(1)
        mean_target = torch.zeros(1)
        std_input = torch.ones(1)
        std_target = torch.ones(1)

        def preprocess(self, history, target):
            events.append("frozen_sim_normalizer")
            return history, target

    class RealNormalizer:
        label = "real"
        mean_input = torch.zeros(1)
        mean_target = torch.zeros(1)
        std_input = torch.ones(1)
        std_target = torch.ones(1)

    def point_loss(_args, prediction, target, normalizer, _weights, iteration):
        assert isinstance(normalizer, FrozenNormalizer)
        assert iteration == 1
        return (prediction - target).square().mean(), {}

    def distill(prediction, teacher, normalizer):
        assert isinstance(normalizer, FrozenNormalizer)
        return (prediction - teacher).square().mean(), {}

    monkeypatch.setattr(subject.ensemble_train, "point_loss", point_loss)
    monkeypatch.setattr(
        subject.ensemble_train, "component_point_distillation_loss", distill)
    def set_model_stats(model, normalizer):
        model.current_stats = normalizer.label
        events.append(f"model_stats_{normalizer.label}")

    monkeypatch.setattr(
        subject.ensemble_train.common, "set_model_stats", set_model_stats)
    model = PointOnly()
    history = torch.ones(1, 1)
    target = torch.zeros(1, 1)
    batch = (history, target, torch.tensor([0]), torch.tensor([0]))
    cache = torch.zeros(1, 1)

    values = subject.backward_sim_rehearsal(
        model, batch, FrozenNormalizer(), RealNormalizer(), cache,
        torch.device("cpu"), 0.05)

    assert events == [
        "model_stats_sim", "frozen_sim_normalizer", "point_forward",
        "model_stats_real"]
    assert "rehearsal_weighted_loss" in values
    assert model.scale.grad is not None


def test_trainer_has_one_fixed725_gate_call_and_no_optimizer_access():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("ensemble_train.evaluate(") == 1
    assert '"validation_optimizer_access": False' in source
