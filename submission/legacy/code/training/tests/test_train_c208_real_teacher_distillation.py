from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
SCRIPT = TRAINING_ROOT / "train_c208_real_teacher_distillation.py"
SPEC = importlib.util.spec_from_file_location("c208_real_teacher_distillation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def formal_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        start_sim=Path("sim10k.pth"),
        start_sim_sha256=subject.rehearsal.FORMAL_START_SIM_SHA256,
        teacher_cache=Path("teacher.pt"),
        teacher_cache_sha256="a" * 64,
        identity_summary=Path("identity.json"),
        identity_summary_sha256=(
            subject.rehearsal.FORMAL_IDENTITY_SUMMARY_SHA256),
        output_dir=tmp_path / "run",
        updates=5000,
        real_dir=Path("real"),
        real_stats=Path("stats.pt"),
        device="cpu",
        num_workers=0,
    )


def test_single_configuration_and_artifact_contract_are_frozen(tmp_path):
    args = formal_args(tmp_path)
    subject.validate_args(args)
    assert subject.POINT_DISTILL_WEIGHT == 4e-5
    assert subject.ENDPOINT_DISTILL_WEIGHT == 0.007
    assert subject.DISTILL_RAMP_UPDATES == 50
    assert subject.POINT_WEIGHTS == {"b": 0.5, "c": 0.5}
    assert subject.BOUNDS_WEIGHTS == {"b": 0.625, "c": 0.25}
    args.updates = 4000
    with pytest.raises(ValueError, match="updates 5000"):
        subject.validate_args(args)
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("rehearsal.ensemble_train.evaluate(") == 1
    assert source.count('args.output_dir / "model_04000.pth"') == 1


class TinyDataset:
    def __init__(self):
        self.windows = [
            SimpleNamespace(trajectory=0, start=0),
            SimpleNamespace(trajectory=1, start=20),
        ]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        history = torch.zeros(2, 3, 4, 3)
        target = torch.zeros(2, 3, 4, 3)
        return history, target


def cache_fixture(monkeypatch):
    monkeypatch.setattr(subject, "EXPECTED_WINDOWS", 2)
    dataset = TinyDataset()
    manifest = [
        {"file": "r0.h5", "train_starts": [0]},
        {"file": "r1.h5", "train_starts": [20]},
    ]
    point = torch.zeros(2, 2, 3, 4, 2, dtype=torch.float16)
    lower = torch.full((2, 2, 3, 4, 2), -0.25, dtype=torch.float32)
    upper = torch.full((2, 2, 3, 4, 2), 0.25, dtype=torch.float32)
    tensors = {
        "point_uv_physical": point,
        "lower_uv_physical": lower,
        "upper_uv_physical": upper,
    }
    fold_manifest = {"assignments": ["tiny"]}
    source_contracts = {
        name: {"sha256": digest}
        for name, digest in subject.quantile_teacher.FORMAL_SOURCE_SHA256.items()
    }
    oof_contracts = {}
    for source, expected in subject.quantile_teacher.FORMAL_OOF_SHA256.items():
        oof_contracts[source] = {
            "summary_file": {"sha256": expected["summary.json"]},
            "folds": [{
                "fold": fold,
                "checkpoint": {
                    "sha256": expected[f"fold_{fold:02d}_best.pth"]},
            } for fold in range(5)],
        }
    metadata = {
        "split_manifest_sha256": subject.rehearsal.canonical_sha256(manifest),
        "real_stats_sha256": "b" * 64,
        "train_keys": subject.ordered_train_keys(dataset.windows, manifest),
        "point_weights": dict(subject.POINT_WEIGHTS),
        "bounds_weights": dict(subject.BOUNDS_WEIGHTS),
        "evidence_class": subject.EVIDENCE_CLASS,
        "units": dict(subject.PHYSICAL_UNITS),
        "source_contracts": source_contracts,
        "oof_contracts": oof_contracts,
        "fold_manifest": fold_manifest,
        "reconstruction": {"folds": [
            {"fold": 0, "samples": 1},
            {"fold": 1, "samples": 1},
        ]},
        "tensor_sha256": {
            name: subject.tensor_sha256(value)
            for name, value in tensors.items()
        },
    }
    payload = {
        "schema_version": 1,
        "method": subject.CACHE_METHOD,
        **tensors,
        "fold": torch.tensor([0, 1], dtype=torch.int8),
        "metadata": metadata,
    }
    return payload, manifest, dataset, [0, 1], fold_manifest


def test_cache_validates_physical_units_order_weights_and_tensor_hash(monkeypatch):
    payload, manifest, dataset, folds, fold_manifest = cache_fixture(monkeypatch)
    tensors, metadata = subject.validate_teacher_cache_payload(
        payload, manifest, dataset, folds, fold_manifest, "b" * 64)
    assert tensors["point_uv_physical"].dtype == torch.float16
    assert tensors["lower_uv_physical"].dtype == torch.float32
    assert metadata["point_weights"] != metadata["bounds_weights"]

    normalized = copy.deepcopy(payload)
    normalized["prediction_uv_normalized"] = normalized.pop(
        "point_uv_physical")
    with pytest.raises(ValueError, match="normalized"):
        subject.validate_teacher_cache_payload(
            normalized, manifest, dataset, folds, fold_manifest, "b" * 64)

    wrong_weights = copy.deepcopy(payload)
    wrong_weights["metadata"]["point_weights"] = {"b": 0.25, "c": 0.5}
    with pytest.raises(ValueError, match="current P1"):
        subject.validate_teacher_cache_payload(
            wrong_weights, manifest, dataset, folds, fold_manifest, "b" * 64)

    wrong_units = copy.deepcopy(payload)
    wrong_units["metadata"]["units"]["point_uv_physical"] = "normalized"
    with pytest.raises(ValueError, match="units"):
        subject.validate_teacher_cache_payload(
            wrong_units, manifest, dataset, folds, fold_manifest, "b" * 64)

    wrong_order = copy.deepcopy(payload)
    wrong_order["metadata"]["train_keys"] = list(reversed(
        wrong_order["metadata"]["train_keys"]))
    with pytest.raises(ValueError, match="keys/order"):
        subject.validate_teacher_cache_payload(
            wrong_order, manifest, dataset, folds, fold_manifest, "b" * 64)

    corrupt = copy.deepcopy(payload)
    corrupt["upper_uv_physical"][0, 0, 0, 0, 0] += 0.125
    with pytest.raises(ValueError, match="tensor SHA"):
        subject.validate_teacher_cache_payload(
            corrupt, manifest, dataset, folds, fold_manifest, "b" * 64)


def test_cache_rejects_fold_or_stats_mismatch(monkeypatch):
    payload, manifest, dataset, folds, fold_manifest = cache_fixture(monkeypatch)
    with pytest.raises(ValueError, match="stats/unit"):
        subject.validate_teacher_cache_payload(
            payload, manifest, dataset, folds, fold_manifest, "c" * 64)
    with pytest.raises(ValueError, match="fold labels"):
        subject.validate_teacher_cache_payload(
            payload, manifest, dataset, [1, 0], fold_manifest, "b" * 64)


def test_endpoint_objective_detaches_point_and_keeps_extent_gradients():
    prediction = torch.ones(1, 2, 2, 2, 3, requires_grad=True)
    lower = torch.full((1, 2, 2, 2, 2), 0.2, requires_grad=True)
    upper = torch.full((1, 2, 2, 2, 2), 0.3, requires_grad=True)
    normalizer = argparse.Namespace(
        mean_target=torch.zeros(3), std_target=torch.ones(3))
    teacher_lower = torch.zeros_like(lower)
    teacher_upper = torch.zeros_like(upper)
    loss = subject.endpoint_distillation_objective(
        prediction, lower, upper, teacher_lower, teacher_upper, normalizer)
    prediction_gradient, lower_gradient, upper_gradient = torch.autograd.grad(
        loss, (prediction, lower, upper), allow_unused=True)
    assert prediction_gradient is None
    assert lower_gradient is not None and torch.isfinite(lower_gradient).all()
    assert upper_gradient is not None and torch.isfinite(upper_gradient).all()


def test_endpoint_auxiliary_gradient_cannot_change_non_uq_gradient():
    non_uq = torch.nn.Parameter(torch.tensor(2.0))
    uq = torch.nn.Parameter(torch.tensor(3.0))
    standard = non_uq.square() + uq.square()
    endpoint = (non_uq.detach() + uq).square()
    standard.backward(retain_graph=True)
    non_uq_before = non_uq.grad.clone()
    uq_before = uq.grad.clone()
    endpoint_gradients = torch.autograd.grad(endpoint, [uq])
    subject._add_gradients([uq], endpoint_gradients)
    assert torch.equal(non_uq.grad, non_uq_before)
    assert torch.equal(uq.grad, uq_before + endpoint_gradients[0])


def test_separate_point_and_endpoint_training_signal_audit():
    prediction = torch.ones(4, requires_grad=True)
    quantiles = torch.ones(8, requires_grad=True)
    supervised = 0.5 * prediction.square().sum()
    native_uq = 0.25 * quantiles.square().sum()
    weighted_point = 0.05 * supervised
    weighted_endpoint = 0.05 * native_uq
    audit = subject.training_signal_audit(
        supervised, native_uq, weighted_point, weighted_endpoint,
        prediction, quantiles, 50)
    assert audit["point"]["gradient_ratio"] == pytest.approx(0.05)
    assert audit["point"]["loss_ratio"] == pytest.approx(0.05)
    assert audit["point"]["cosine"] == pytest.approx(1.0)
    assert audit["endpoint"]["gradient_ratio"] == pytest.approx(0.05)
    assert audit["endpoint"]["loss_ratio"] == pytest.approx(0.05)
    assert prediction.grad is None and quantiles.grad is None


def passing_audits():
    return [{
        "iteration": iteration,
        "point": {
            "finite_nonzero": True, "loss_ratio": 0.02,
            "gradient_ratio": 0.05, "cosine": 0.1,
        },
        "endpoint": {
            "finite_nonzero": True, "loss_ratio": 0.02,
            "gradient_ratio": 0.05, "cosine": 0.1,
        },
    } for iteration in subject.PREFLIGHT_AUDIT_UPDATES]


def test_preflight_requires_four_separate_bounded_branch_audits():
    assert subject.evaluate_preflight(passing_audits())["pass"] is True
    assert subject.evaluate_preflight(passing_audits()[:-1])["pass"] is False
    failed = passing_audits()
    failed[2]["endpoint"]["gradient_ratio"] = 0.31
    result = subject.evaluate_preflight(failed)
    assert result["pass"] is False
    assert result["checks"]["point"]["gradient_ratio_range"] is True
    assert result["checks"]["endpoint"]["gradient_ratio_range"] is False


def passing_candidate(baseline):
    result = dict(baseline)
    result["rel_l2_score"] += 0.20
    result["tke_rel_l2_score"] += 0.0
    result["mvpe_rel_l2_score"] += 0.0
    result["native_sps_score"] += 0.10
    result["point_score"] += 0.051
    return result


def test_step2000_gate_is_single_access_and_provenance_labeled(monkeypatch):
    baseline = dict(subject.rehearsal.FORMAL_IDENTITY_STEP2000_METRICS)
    candidate = passing_candidate(baseline)
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train, "evaluate",
        lambda *_args, **_kwargs: candidate)
    real = {
        "validation_access_updates": [],
        "validation_loader": object(), "normalizer": object(),
    }
    gate = subject.evaluate_step2000_gate(
        object(), real, torch.device("cpu"), {"metrics": baseline})
    assert gate["pass"] is True
    assert gate["evidence_class"] == subject.EVIDENCE_CLASS
    assert real["validation_access_updates"] == [2000]
    with pytest.raises(RuntimeError, match="already accessed"):
        subject.evaluate_step2000_gate(
            object(), real, torch.device("cpu"), {"metrics": baseline})


@pytest.mark.parametrize("preflight_pass,gate_pass,updates,saves,gate_calls", [
    (False, False, 2, 0, 0),
    (True, False, 3, 0, 1),
    (True, True, 4, 1, 1),
])
def test_failure_never_saves_and_success_saves_only_update4000(
        monkeypatch, tmp_path, preflight_pass, gate_pass,
        updates, saves, gate_calls):
    monkeypatch.setattr(subject, "PREFLIGHT_UPDATE", 2)
    monkeypatch.setattr(subject, "PREFLIGHT_AUDIT_UPDATES", (1, 2))
    monkeypatch.setattr(subject, "GATE_UPDATE", 3)
    monkeypatch.setattr(subject, "TERMINAL_UPDATE", 4)
    monkeypatch.setattr(subject.rehearsal, "FORMAL_LOG_INTERVAL", 2)
    model = torch.nn.Linear(1, 1, bias=False)
    calls = {"iteration": 0, "save": [], "gate": 0}

    def backward(model, *_args, **_kwargs):
        calls["iteration"] += 1
        model.weight.square().sum().backward()
        audit = {
            "iteration": calls["iteration"], "point": {}, "endpoint": {},
        } if calls["iteration"] <= 2 else None
        return {"real_loss": 0.0}, audit

    monkeypatch.setattr(subject, "backward_real_teacher_distillation", backward)
    monkeypatch.setattr(
        subject, "evaluate_preflight",
        lambda audits: {
            "status": "GO" if preflight_pass else "STOP_RTD1_PREFLIGHT",
            "pass": preflight_pass, "audits": audits,
        })

    def gate(*_args, **_kwargs):
        calls["gate"] += 1
        return {
            "status": "GO" if gate_pass else "STOP_RTD1_UPDATE2000_GATE",
            "pass": gate_pass,
        }

    monkeypatch.setattr(subject, "evaluate_step2000_gate", gate)
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common, "normalizer_dict",
        lambda _normalizer: {})
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common, "save_checkpoint",
        lambda path, *_args, **_kwargs: calls["save"].append(path))
    args = argparse.Namespace(updates=5, output_dir=tmp_path)
    real = {"loader": [None], "normalizer": object(), "teacher": {}}
    result = subject.train_endpoint(
        args, model, real, {}, torch.device("cpu"))
    assert result["completed_updates"] == updates
    assert len(calls["save"]) == saves
    assert calls["gate"] == gate_calls
    assert (result["checkpoint"] is None) == (saves == 0)
