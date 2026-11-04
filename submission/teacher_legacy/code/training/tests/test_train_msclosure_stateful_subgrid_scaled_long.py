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

import train_msclosure_stateful_subgrid_scaled_long as subject  # noqa: E402
from models.msclosure_stateful_subgrid import (  # noqa: E402
    StatefulSubgridMultipleShootingClosure,
)


def test_recipe_is_exactly_the_scaled_long_control_protocol():
    names = (
        "FIT_AOAS", "HELD_AOA", "SEED", "SCHEDULE_SEED", "UPDATES",
        "BATCH_SIZE", "LR", "WEIGHT_DECAY", "GRAD_CLIP", "EMA_DECAY",
        "TRACE_INTERVAL", "LOSS_WEIGHTS", "MSE_WEIGHT",
    )
    for name in names:
        assert getattr(subject, name) == getattr(subject.control, name)
    assert subject.UPDATES == 12_000
    assert subject.BATCH_SIZE == 8
    assert subject.MODEL_CONFIG.e0_channels == 128
    assert subject.MODEL_CONFIG.e1_channels == 256
    assert subject.MODEL_CONFIG.e2_channels == 512
    assert subject.MODEL_CONFIG.context_channels == 128
    assert subject.MODEL_CONFIG.cell_channels == 192
    assert subject.MODEL_CONFIG.subgrid_channels == 12
    formal_schedule = subject.control.make_schedule(
        3840, subject.UPDATES, subject.BATCH_SIZE)
    assert hashlib.sha256(formal_schedule.tobytes()).hexdigest() == (
        "8fd25c5b184b3764c29bc85a7b3e18c"
        "ccedcf742e067277938a505fea5afe277")


def test_constructed_model_is_recurrent_carry_detached_without_extra_loss():
    model = subject.construct_model(torch.device("cpu"), tiny=True)
    control_model = subject.control.construct_model(
        torch.device("cpu"), tiny=True)
    assert isinstance(model, StatefulSubgridMultipleShootingClosure)
    assert model.rollout_mode == "carry_detached"
    assert model._subgrid_override == "recurrent"
    assert subject.LOSS_WEIGHTS == (1.0, 4.0, 1.0)
    assert subject.MSE_WEIGHT == 0.25
    assert "AUX" not in vars(subject)
    assert "GATE" not in vars(subject)
    control_state = control_model.state_dict()
    stateful_state = model.state_dict()
    assert all(torch.equal(value, stateful_state[name])
               for name, value in control_state.items())


def test_train_delegates_to_control_after_forcing_recurrent_mode(monkeypatch):
    model = subject.construct_model(torch.device("cpu"), tiny=True)
    model.set_subgrid_override("reset")
    sentinel = ({"delegated": True}, object())
    observed = {}

    def fake_train(received_model, dataset, normalizer, device, **kwargs):
        observed["mode"] = received_model._subgrid_override
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
        "mode": "recurrent",
        "dataset": "dataset",
        "normalizer": "normalizer",
        "device": torch.device("cpu"),
        "kwargs": {"updates": 17, "batch_size": 3, "trace_interval": 5},
    }


def test_ema_mechanism_uses_one_state_and_never_resets_physical_carry(
        monkeypatch):
    model = subject.construct_model(torch.device("cpu"), tiny=True)
    state = subject.control.snapshot(model)
    expected_sha = subject.base.state_sha(state)
    calls = []
    values = {"recurrent": 80.0, "static": 79.5, "reset": 80.25}

    def fake_evaluate(received, *_args):
        mode = received._subgrid_override
        calls.append({
            "mode": mode,
            "sha": subject.base.state_sha(subject.control.snapshot(received)),
            "rollout": received.rollout_mode,
            "physical_reset": received._reset_override,
        })
        score = values[mode]
        return {"aggregate": {
            "samples": 1,
            "rel_l2_score": score,
            "tke_rel_l2_score": score,
            "mvpe_rel_l2_score": score,
            "point_contribution_46_10_9": score,
        }}

    monkeypatch.setattr(subject.loao, "_evaluate_same_model", fake_evaluate)
    result = subject.evaluate_ema_mechanisms(
        model, state, None, None, None, None, torch.device("cpu"))
    assert [row["mode"] for row in calls] == [
        "recurrent", "static", "reset"]
    assert {row["sha"] for row in calls} == {expected_sha}
    assert all(row["rollout"] == "carry_detached" for row in calls)
    assert not any(row["physical_reset"] for row in calls)
    assert result["same_checkpoint_all_arms"]
    assert result["checkpoint_model_state_sha256"] == expected_sha
    assert result["comparisons"]["static_minus_recurrent"][
        "point_contribution_46_10_9"] == -0.5
    assert result["comparisons"]["reset_minus_recurrent"][
        "point_contribution_46_10_9"] == 0.25
    assert model._subgrid_override == "recurrent"


def test_saved_endpoint_strictly_rebuilds_stateful_model(tmp_path, monkeypatch):
    config = subject.model_config(tiny=True)
    monkeypatch.setattr(subject, "MODEL_CONFIG", config)
    model = subject.construct_model(torch.device("cpu"), tiny=True)
    normalizer = subject.base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3),
        torch.device("cpu"))
    path = tmp_path / "endpoint.pt"
    state = subject.control.snapshot(model)
    record = subject.save_endpoint(
        path, kind="test", state=state, git={}, normalizer=normalizer,
        updates=2)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert record["model_state_sha256"] == subject.base.state_sha(state)
    assert payload["schema"] == subject.SCHEMA
    assert payload["rollout_mode"] == "carry_detached"
    assert payload["subgrid_default_mode"] == "recurrent"
    assert payload["normalizer"] == subject.control.normalizer_record(normalizer)
    rebuilt = StatefulSubgridMultipleShootingClosure(
        subject.StatefulSubgridMultipleShootingConfig(
            **payload["model_config"]))
    rebuilt.load_state_dict(payload["model_state"], strict=True)
    history = torch.randn(1, 20, 16, 16, 3)
    assert torch.equal(rebuilt.eval()(history), model.eval()(history))


def test_formal_requires_explicit_acknowledgement_before_any_cuda_work():
    args = subject.argparse.Namespace(
        execute=False, acknowledge="", sim_dir=None, output_dir=None)
    with pytest.raises(PermissionError, match="formal run requires"):
        subject.formal(args, torch.device("cpu"))
