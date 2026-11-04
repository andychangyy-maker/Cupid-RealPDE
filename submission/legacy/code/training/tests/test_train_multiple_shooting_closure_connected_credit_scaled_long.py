from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_multiple_shooting_closure_connected_credit_scaled_long as subject  # noqa: E402
from models.multiple_shooting_closure import (  # noqa: E402
    MultipleShootingClosure,
)


def test_recipe_is_exactly_the_scaled_long_control_protocol():
    names = (
        "FIT_AOAS", "HELD_AOA", "SEED", "SCHEDULE_SEED", "UPDATES",
        "BATCH_SIZE", "LR", "WEIGHT_DECAY", "GRAD_CLIP", "EMA_DECAY",
        "TRACE_INTERVAL", "LOSS_WEIGHTS", "MSE_WEIGHT", "MODEL_CONFIG",
    )
    for name in names:
        assert getattr(subject, name) == getattr(subject.control, name)
    assert subject.UPDATES == 12_000
    assert subject.BATCH_SIZE == 8
    assert subject.LR == 2.0e-4
    assert subject.WEIGHT_DECAY == 1.0e-5
    assert subject.GRAD_CLIP == 1.0
    assert subject.EMA_DECAY == 0.999
    assert subject.LOSS_WEIGHTS == (1.0, 4.0, 1.0)
    assert subject.MSE_WEIGHT == 0.25
    formal_schedule = subject.control.make_schedule(
        3840, subject.UPDATES, subject.BATCH_SIZE)
    assert hashlib.sha256(formal_schedule.tobytes()).hexdigest() == (
        "8fd25c5b184b3764c29bc85a7b3e18c"
        "ccedcf742e067277938a505fea5afe277")


def test_matched_candidate_has_exact_control_state_and_only_graph_difference():
    model, identity = subject.construct_matched_candidate(
        torch.device("cpu"), tiny=True)
    assert isinstance(model, MultipleShootingClosure)
    assert model.rollout_mode == "carry_detached"
    assert model.segment_gradient_mode == "connected"
    assert identity["state_key_sets_equal"]
    assert identity["common_state_bitwise_identical"]
    assert identity["truncated_state_sha256"] == (
        identity["connected_state_sha256"])
    assert identity["parameter_counts"]["truncated"] == (
        identity["parameter_counts"]["connected"])
    assert identity["only_architecture_config_difference"] == [
        "segment_gradient_mode"]
    assert identity["paired_seed"] == subject.SEED


def test_train_delegates_unchanged_recipe_and_rejects_truncated(monkeypatch):
    model = subject.construct_model(torch.device("cpu"), tiny=True)
    sentinel = ({"delegated": True}, object())
    observed = {}

    def fake_train(received_model, dataset, normalizer, device, **kwargs):
        observed["gradient_mode"] = received_model.segment_gradient_mode
        observed["dataset"] = dataset
        observed["normalizer"] = normalizer
        observed["device"] = device
        observed["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(subject.control, "train", fake_train)
    result = subject.train(
        model, "dataset", "normalizer", torch.device("cpu"),
        updates=17, batch_size=3, trace_interval=5)
    assert result is sentinel
    assert observed == {
        "gradient_mode": "connected",
        "dataset": "dataset",
        "normalizer": "normalizer",
        "device": torch.device("cpu"),
        "kwargs": {"updates": 17, "batch_size": 3, "trace_interval": 5},
    }

    truncated = subject.control.construct_model(
        torch.device("cpu"), tiny=True)
    with pytest.raises(ValueError, match="requires connected graph"):
        subject.train(
            truncated, None, None, torch.device("cpu"),
            updates=1, batch_size=1, trace_interval=1)


def test_saved_endpoint_strictly_rebuilds_connected_model(
        tmp_path, monkeypatch):
    config = subject.model_config(tiny=True)
    monkeypatch.setattr(subject, "MODEL_CONFIG", config)
    model = subject.construct_model(torch.device("cpu"), tiny=True)
    normalizer = subject.base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3),
        torch.device("cpu"))
    state = subject.control.snapshot(model)
    path = tmp_path / "endpoint.pt"
    record = subject.save_endpoint(
        path, kind="test", state=state, git={}, normalizer=normalizer,
        updates=2)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert record["model_state_sha256"] == subject.base.state_sha(state)
    assert payload["schema"] == subject.SCHEMA
    assert payload["rollout_mode"] == "carry_detached"
    assert payload["segment_gradient_mode"] == "connected"
    rebuilt = MultipleShootingClosure(
        subject.MultipleShootingClosureConfig(**payload["model_config"]),
        rollout_mode=payload["rollout_mode"],
        segment_gradient_mode=payload["segment_gradient_mode"])
    rebuilt.load_state_dict(payload["model_state"], strict=True)
    history = torch.randn(1, 20, 16, 16, 3)
    assert torch.equal(rebuilt.eval()(history), model.eval()(history))


def test_endpoint_evaluation_is_single_mode_single_pass(monkeypatch):
    model = subject.construct_model(torch.device("cpu"), tiny=True)
    state = subject.control.snapshot(model)
    calls = []

    def fake_evaluate(received_model, *args):
        calls.append((received_model.segment_gradient_mode, args))
        return {"aggregate": {"samples": 1}}

    monkeypatch.setattr(subject.loao, "_evaluate_same_model", fake_evaluate)
    result = subject.evaluate_endpoint(
        model, state, "trajectories", "windows", "files", "normalizer",
        torch.device("cpu"))
    assert result == {"aggregate": {"samples": 1}}
    assert len(calls) == 1
    assert calls[0][0] == "connected"


def test_formal_requires_explicit_acknowledgement_before_any_cuda_work():
    args = subject.argparse.Namespace(
        execute=False, acknowledge="", sim_dir=None, output_dir=None)
    with pytest.raises(PermissionError, match="formal run requires"):
        subject.formal(args, torch.device("cpu"))
