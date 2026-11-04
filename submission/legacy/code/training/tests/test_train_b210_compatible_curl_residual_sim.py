from __future__ import annotations

from collections import OrderedDict
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

import train_b210_compatible_curl_residual_sim as subject
from models.ensemble_unet2d import EnsembleUNet2d


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Linear(3, 5)
        self.energy_controller = torch.nn.Linear(5, 2)
        self.scalar_potential_residual = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.Linear(4, 5))


def test_private_residual_kind_is_removed_without_reordering():
    remaining, kind = subject.pop_residual_kind([
        "runner.py", "--output-dir", "run", "--residual-kind", "gradient",
        "--num-workers", "0"])
    assert kind == "gradient"
    assert remaining == ["runner.py", "--output-dir", "run", "--num-workers", "0"]
    with pytest.raises(ValueError, match="gradient or curl"):
        subject.pop_residual_kind(["runner.py", "--output-dir", "run"])
    with pytest.raises(ValueError, match="exactly once"):
        subject.pop_residual_kind([
            "runner.py", "--residual-kind", "curl",
            "--residual-kind", "gradient"])


def test_exact_parameter_contract_on_meta_device():
    original = subject.RESIDUAL_KIND
    subject.RESIDUAL_KIND = "curl"
    try:
        with torch.device("meta"):
            model = EnsembleUNet2d(subject.compatible_config("candidate"))
        assert sum(p.numel() for p in model.parameters()) == subject.EXPECTED_PARAMETERS
        assert sum(
            p.numel()
            for p in model.scalar_potential_residual.parameters()
        ) == subject.EXPECTED_MODULE_PARAMETERS
    finally:
        subject.RESIDUAL_KIND = original


def test_internal_control_and_candidate_keep_branch_trainable(monkeypatch):
    model = TinyModel()
    monkeypatch.setattr(
        subject, "ORIGINAL_CONFIGURE",
        lambda model, arm: tuple(model.stem.parameters()))
    trainable = subject.configure_trainable(model, "control")
    assert set(model.scalar_potential_residual.parameters()).issubset(trainable)
    assert all(p.requires_grad for p in model.scalar_potential_residual.parameters())


def test_state_groups_and_mechanical_failure_are_explicit():
    assert subject.state_group("output.weight") == "projection"
    assert subject.state_group("body.4.conv1.weight") == "body"
    subject.AUDIT.clear()
    subject.AUDIT["control_projection_gradient_seen"] = False
    try:
        subject.validate_mechanical_audit()
    except RuntimeError as error:
        assert "control_projection_gradient_seen" in str(error)
    else:
        raise AssertionError("mechanical failure was not enforced")


def test_summary_labels_are_frozen(monkeypatch):
    summary = {"completed_updates": 0, "training_contract": {}, "paired_candidate": {}}
    monkeypatch.setattr(subject, "ORIGINAL_RUN", lambda *a, **k: summary)
    args = [None] * 9
    args[1] = type("Args", (), {})()
    subject.RESIDUAL_KIND = "gradient"
    output = subject.run_with_contract(*args)
    assert output["schema"] == subject.SCHEMA
    assert output["method"] == subject.method_for("gradient")
    assert output["residual_kind"] == "gradient"
    assert output["training_contract"]["scalar_potential_residual"][
        "anchor_128_256"] == "train_set_fixed_diagnostic_not_heldout"


def test_terminal_reload_uses_frozen_scale_contract_and_device(monkeypatch):
    class Dataset:
        def __getitem__(self, _index):
            return torch.zeros(20, 4, 4, 3), None, None, None

    class Model:
        def load_state_dict(self, _state, strict):
            assert strict is True

    summary = {
        "completed_updates": subject.sim.TERMINAL_UPDATE,
        "checkpoint": "endpoint.pth",
        "training_contract": {},
        "paired_candidate": {},
        "paired_control": {},
    }
    monkeypatch.setattr(subject, "ORIGINAL_RUN", lambda *a, **k: summary)
    monkeypatch.setattr(subject, "INITIAL_STATES", {
        "candidate": OrderedDict(weight=torch.zeros(1)),
    })
    monkeypatch.setattr(subject, "AUDIT", {"all_checks": True})
    monkeypatch.setattr(subject, "GEOMETRY_METRICS", {})
    monkeypatch.setattr(subject, "KEY_AUDIT", {
        "candidate": {
            "hasher": hashlib.sha256(),
            "count": subject.sim.TERMINAL_UPDATE,
        },
        "control": {
            "hasher": hashlib.sha256(),
            "count": subject.sim.CONTROL_UPDATES,
        },
    })
    built = {}
    monkeypatch.setattr(
        subject, "build_with_contract",
        lambda arm, device: built.update(arm=arm, device=device) or Model())
    monkeypatch.setattr(
        subject.torch, "load",
        lambda *a, **k: {"model_state_dict": {}})
    monkeypatch.setattr(
        subject.sim.rehearsal.ensemble_train.common, "set_model_stats",
        lambda *a, **k: None)
    observed = {}

    def evaluate(
            _model, _dataset, indices, _normalizer, _soft_sps,
            teacher_cache, scale_contract, sim_args, device):
        observed.update(
            indices=indices, teacher_cache=teacher_cache,
            scale_contract=scale_contract, sim_args=sim_args, device=device)
        return {"point_score": 1.0}

    monkeypatch.setattr(subject.sim.sim_hgrad, "evaluate_anchor", evaluate)
    monkeypatch.setattr(
        subject, "geometry_audit",
        lambda *a, **k: ({"all_checks": True}, {"rms": 0.1}))
    monkeypatch.setattr(
        subject, "module_state",
        lambda _model: OrderedDict(weight=torch.ones(1)))

    sim_args = type("Args", (), {})()
    raw_loader = type("Loader", (), {"dataset": Dataset()})()
    teacher_cache = object()
    scale_contract = object()
    device = torch.device("cpu")
    args = [
        object(), sim_args, raw_loader, object(), object(), teacher_cache,
        object(), scale_contract, device,
    ]
    output = subject.run_with_contract(*args)
    assert built == {"arm": "candidate", "device": device}
    assert observed["indices"] == list(range(128, 256))
    assert observed["teacher_cache"] is teacher_cache
    assert observed["scale_contract"] is scale_contract
    assert observed["sim_args"] is sim_args
    assert observed["device"] == device
    assert output["paired_candidate"]["terminal_anchor_128_256"] == {
        "point_score": 1.0}
