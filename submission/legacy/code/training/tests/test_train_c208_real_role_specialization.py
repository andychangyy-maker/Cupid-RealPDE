from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from torch.utils.data import TensorDataset


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
SCRIPT = TRAINING_ROOT / "train_c208_real_role_specialization.py"
SPEC = importlib.util.spec_from_file_location("c208_real_rsm1", SCRIPT)
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
        real_dir=Path("real"), real_stats=Path("stats.pt"),
        device="cpu", num_workers=0,
    )


def test_single_configuration_native_loss_and_single_access_are_frozen(tmp_path):
    args = formal_args(tmp_path)
    subject.validate_args(args)
    assert subject.ROLE_DISTILL_WEIGHT == 0.00125
    assert subject.MEAN_DISTILL_WEIGHT == 0.00125
    assert subject.DISTILL_RAMP_UPDATES == 50
    assert subject.PREFLIGHT_AUDIT_UPDATES == (50, 100, 150, 200)
    assert subject.ANCHOR_SIZE == 128
    assert subject.TERMINAL_UPDATE == 4000
    args.updates = 4000
    with pytest.raises(ValueError, match="updates 5000"):
        subject.validate_args(args)
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("rehearsal.ensemble_train.evaluate(") == 1
    assert source.count('args.output_dir / "model_04000.pth"') == 1
    assert "endpoint_distill" not in source
    assert "supervised_loss.backward(retain_graph=True)" in source
    assert "soft_sps_weight * soft_sps_loss, uq_parameters" in source
    assert '"event": "c208_rsm1b_contract"' in source
    assert '"method": "c208_rsm1b_foldaware_role_specialization_calibrated"' \
        in source


def test_rsm1b_matches_v1_effective_weight_through_update50_then_caps():
    for iteration in (1, 25, 50):
        old_effective = 0.005 * min(1.0, iteration / 200)
        assert subject.ramp_weight(
            subject.ROLE_DISTILL_WEIGHT, iteration) == pytest.approx(
                old_effective)
        assert subject.ramp_weight(
            subject.MEAN_DISTILL_WEIGHT, iteration) == pytest.approx(
                old_effective)
    assert subject.ramp_weight(subject.ROLE_DISTILL_WEIGHT, 100) == 0.00125
    assert subject.ramp_weight(subject.MEAN_DISTILL_WEIGHT, 200) == 0.00125


def test_identity_summary_requires_unique_exact_update4000_row(
        monkeypatch, tmp_path):
    path = tmp_path / "identity.json"
    row = {"iteration": 4000, **subject.NATIVE_STEP4000_METRICS}
    path.write_text(json.dumps({"history": [row]}),
                    encoding="utf-8")
    args = argparse.Namespace(
        identity_summary=path,
        identity_summary_sha256=subject.rehearsal.FORMAL_IDENTITY_SUMMARY_SHA256)
    monkeypatch.setattr(
        subject.rehearsal, "load_identity_step2000",
        lambda _args: {"path": str(path), "sha256": "frozen", "metrics": {}})
    identity = subject.load_identity_step4000(args)
    assert identity["metrics_update4000"] == subject.NATIVE_STEP4000_METRICS
    bad = dict(row)
    bad["native_sps_score"] += 1e-8
    path.write_text(json.dumps({"history": [bad]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="update-4000 native_sps_score"):
        subject.load_identity_step4000(args)


class TinyDataset:
    def __init__(self):
        self.windows = [
            SimpleNamespace(trajectory=0, start=0),
            SimpleNamespace(trajectory=1, start=20),
        ]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.zeros(2, 3, 4, 3), torch.zeros(2, 3, 4, 3)


def cache_fixture(monkeypatch):
    monkeypatch.setattr(subject, "EXPECTED_WINDOWS", 2)
    dataset = TinyDataset()
    manifest = [
        {"file": "r0.h5", "train_starts": [0]},
        {"file": "r1.h5", "train_starts": [20]},
    ]
    a = torch.zeros(2, 2, 3, 4, 2, dtype=torch.float16)
    b = torch.zeros_like(a)
    mean = torch.ones(2, 1, 3, 4, 2, dtype=torch.float16)
    tensors = {
        "a_fluctuation_uv_physical": a,
        "b_fluctuation_uv_physical": b,
        "p1_temporal_mean_uv_physical": mean,
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
        "train_keys": subject.rtd1.ordered_train_keys(dataset.windows, manifest),
        "fluctuation_weights": dict(subject.FLUCTUATION_WEIGHTS),
        "mean_weights": dict(subject.MEAN_WEIGHTS),
        "evidence_class": subject.EVIDENCE_CLASS,
        "units": dict(subject.PHYSICAL_UNITS),
        "composition_precision": "float32_before_final_float16_storage",
        "source_contracts": source_contracts,
        "oof_contracts": oof_contracts,
        "fold_manifest": fold_manifest,
        "reconstruction": {"folds": [
            {"fold": 0, "samples": 1},
            {"fold": 1, "samples": 1},
        ]},
        "tensor_sha256": {
            name: subject.rtd1.tensor_sha256(value)
            for name, value in tensors.items()
        },
    }
    payload = {
        "schema_version": 1, "method": subject.CACHE_METHOD,
        "metadata": metadata, **tensors,
        "fold": torch.tensor([0, 1], dtype=torch.int8),
    }
    return payload, manifest, dataset, [0, 1], fold_manifest


def validate_cache(subject_payload, manifest, dataset, folds, fold_manifest):
    return subject.validate_teacher_cache_payload(
        subject_payload, manifest, dataset, folds, fold_manifest, "b" * 64)


def test_cache_locks_units_natural_indices_formulas_and_tensor_identity(monkeypatch):
    payload, manifest, dataset, folds, fold_manifest = cache_fixture(monkeypatch)
    tensors, _metadata = validate_cache(
        payload, manifest, dataset, folds, fold_manifest)
    assert all(value.dtype == torch.float16 for value in tensors.values())

    normalized = copy.deepcopy(payload)
    normalized["prediction_uv_normalized"] = torch.zeros(1)
    with pytest.raises(ValueError, match="normalized"):
        validate_cache(normalized, manifest, dataset, folds, fold_manifest)

    endpoints = copy.deepcopy(payload)
    endpoints["lower_uv_physical"] = torch.zeros(1)
    with pytest.raises(ValueError, match="endpoint"):
        validate_cache(endpoints, manifest, dataset, folds, fold_manifest)

    wrong_order = copy.deepcopy(payload)
    wrong_order["metadata"]["train_keys"].reverse()
    with pytest.raises(ValueError, match="natural index order"):
        validate_cache(wrong_order, manifest, dataset, folds, fold_manifest)

    wrong_formula = copy.deepcopy(payload)
    wrong_formula["metadata"]["mean_weights"]["c"] = 0.25
    with pytest.raises(ValueError, match="P1 mean formula"):
        validate_cache(wrong_formula, manifest, dataset, folds, fold_manifest)

    wrong_units = copy.deepcopy(payload)
    wrong_units["metadata"]["units"]["a_fluctuation_uv_physical"] = "normalized"
    with pytest.raises(ValueError, match="units"):
        validate_cache(wrong_units, manifest, dataset, folds, fold_manifest)

    corrupt = copy.deepcopy(payload)
    corrupt["p1_temporal_mean_uv_physical"][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="tensor SHA"):
        validate_cache(corrupt, manifest, dataset, folds, fold_manifest)


def test_cache_rejects_nonzero_fluctuation_and_fold_mismatch(monkeypatch):
    payload, manifest, dataset, folds, fold_manifest = cache_fixture(monkeypatch)
    nonzero = copy.deepcopy(payload)
    nonzero["a_fluctuation_uv_physical"][:] = 0.1
    nonzero["metadata"]["tensor_sha256"]["a_fluctuation_uv_physical"] = (
        subject.rtd1.tensor_sha256(nonzero["a_fluctuation_uv_physical"]))
    with pytest.raises(ValueError, match="zero-time-mean"):
        validate_cache(nonzero, manifest, dataset, folds, fold_manifest)
    with pytest.raises(ValueError, match="fold labels"):
        validate_cache(payload, manifest, dataset, [1, 0], fold_manifest)


def test_role_assignment_and_physical_normalization_are_exact():
    normalizer = argparse.Namespace(
        mean_target=torch.tensor([10.0, 20.0, 0.0]),
        std_target=torch.tensor([2.0, 4.0, 1.0]))
    a_norm = torch.tensor([[[[[-0.02, -0.04]]], [[[0.02, 0.04]]]]])
    b_norm = torch.tensor([[[[[-0.01, -0.03]]], [[[0.01, 0.03]]]]])
    teacher_a = a_norm * normalizer.std_target[:2]
    teacher_b = b_norm * normalizer.std_target[:2]
    teacher_mean = torch.tensor([[[[[11.0, 18.0]]]]])
    mean_norm = (
        teacher_mean - normalizer.mean_target[:2]) / normalizer.std_target[:2]
    members = torch.zeros(1, 4, 2, 1, 1, 3)
    members[:, 0, ..., :2] = a_norm
    members[:, 1, ..., :2] = a_norm
    members[:, 2, ..., :2] = b_norm
    members[:, 3, ..., :2] = b_norm
    prediction = torch.zeros(1, 2, 1, 1, 3)
    prediction[..., :2] = mean_norm + 0.5 * (a_norm + b_norm)
    objectives = subject.role_distillation_objectives(
        prediction, members, teacher_a, teacher_b, teacher_mean, normalizer)
    for name in ("a_fluctuation", "b_fluctuation", "role", "p1_mean",
                 "p1_aggregate"):
        assert objectives[name] == pytest.approx(0.0, abs=1e-8)
    swapped = members.clone()
    swapped[:, :2], swapped[:, 2:] = members[:, 2:].clone(), members[:, :2].clone()
    wrong = subject.role_distillation_objectives(
        prediction, swapped, teacher_a, teacher_b, teacher_mean, normalizer)
    assert float(wrong["role"]) > 0.0


class TinyRoleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.member_point_heads = torch.nn.ModuleList([
            torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False)])
            for _ in range(4)])
        self.mean_profile_refiner = torch.nn.Sequential(
            torch.nn.Linear(1, 1, bias=False))
        self.other = torch.nn.Linear(1, 1, bias=False)


class TinyLoader:
    dataset = object()

    def __iter__(self):
        return iter([None])


def test_parameter_sets_and_auxiliary_gradient_writes_are_isolated():
    model = TinyRoleModel()
    member_parameters, mean_parameters = subject.role_parameters(model)
    assert {id(p) for p in member_parameters}.isdisjoint(
        id(p) for p in mean_parameters)
    other = next(model.other.parameters())
    supervised = sum(p.square().sum() for p in model.parameters())
    weighted_role = 0.05 * sum(p.square().sum() for p in member_parameters)
    weighted_mean = 0.05 * sum(p.square().sum() for p in mean_parameters)
    supervised.backward(retain_graph=True)
    before = {id(p): p.grad.clone() for p in model.parameters()}
    subject._add_gradients(member_parameters, torch.autograd.grad(
        weighted_role, member_parameters, retain_graph=True))
    subject._add_gradients(mean_parameters, torch.autograd.grad(
        weighted_mean, mean_parameters))
    assert torch.equal(other.grad, before[id(other)])
    for parameter in member_parameters + mean_parameters:
        assert not torch.equal(parameter.grad, before[id(parameter)])


def test_parameter_audit_does_not_accumulate_gradients():
    member = torch.nn.Parameter(torch.tensor(1.0))
    mean = torch.nn.Parameter(torch.tensor(1.0))
    supervised = member.square() + mean.square()
    weighted_role = 0.1 * member.square()
    weighted_mean = 0.1 * mean.square()
    audit = subject.parameter_gradient_audit(
        supervised, weighted_role, weighted_mean, [member], [mean], 50)
    assert audit["role_heads"]["gradient_ratio"] == pytest.approx(0.1)
    assert audit["mean_profile_refiner"]["gradient_ratio"] == pytest.approx(0.1)
    assert audit["role_heads"]["cosine"] == pytest.approx(1.0)
    assert member.grad is None and mean.grad is None


def passing_audits():
    return [{
        "iteration": iteration,
        "role_heads": {
            "finite_nonzero": True, "loss_ratio": 0.02,
            "gradient_ratio": 0.05, "cosine": 0.1,
        },
        "mean_profile_refiner": {
            "finite_nonzero": True, "loss_ratio": 0.02,
            "gradient_ratio": 0.05, "cosine": 0.1,
        },
    } for iteration in subject.PREFLIGHT_AUDIT_UPDATES]


def test_preflight_requires_four_audits_and_four_anchor_declines():
    start = {
        "a_fluctuation": 1.0, "b_fluctuation": 1.0,
        "p1_mean": 1.0, "p1_aggregate": 1.0,
    }
    end = {name: 0.9 * value for name, value in start.items()}
    assert subject.evaluate_preflight(passing_audits(), start, end)["pass"]
    assert not subject.evaluate_preflight(
        passing_audits()[:-1], start, end)["pass"]
    failed = dict(end)
    failed["p1_mean"] = 0.901
    assert not subject.evaluate_preflight(
        passing_audits(), start, failed)["pass"]
    zero_start = {name: 0.0 for name in start}
    zero_end = {name: 1e-12 for name in start}
    assert subject.evaluate_preflight(
        passing_audits(), zero_start, zero_end)["pass"]


def test_anchor_is_exact_natural_train_0_to_127_without_global_rng_change():
    assert subject.anchor_indices(2576) == list(range(128))
    dataset = TensorDataset(torch.arange(2576))
    torch.manual_seed(123)
    expected = torch.rand(1)
    torch.manual_seed(123)
    loader = subject.make_anchor_loader(dataset, 0)
    observed_rows = torch.cat([batch[0] for batch in loader]).tolist()
    observed = torch.rand(1)
    assert observed_rows == list(range(128))
    assert torch.equal(observed, expected)


def passing_terminal_metrics():
    metrics = dict(subject.NATIVE_STEP4000_METRICS)
    metrics["point_score"] += 0.10
    metrics["rel_l2_score"] += 0.30
    metrics["native_sps_score"] += 0.70
    return metrics


def test_update4000_gate_uses_exact_baseline_and_single_fixed725_access(monkeypatch):
    assert subject.NATIVE_STEP4000_METRICS["point_score"] == pytest.approx(
        91.52029057134612)
    metrics = passing_terminal_metrics()
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train, "evaluate",
        lambda *_args, **_kwargs: metrics)
    real = {
        "validation_access_updates": [], "validation_loader": object(),
        "normalizer": object(),
    }
    identity = {
        "metrics_update4000": dict(subject.NATIVE_STEP4000_METRICS)}
    gate = subject.evaluate_terminal_gate(
        object(), real, torch.device("cpu"), identity)
    assert gate["pass"]
    assert real["validation_access_updates"] == [4000]
    with pytest.raises(RuntimeError, match="before its unique"):
        subject.evaluate_terminal_gate(
            object(), real, torch.device("cpu"), identity)
    failed = passing_terminal_metrics()
    failed["tke_rel_l2_score"] -= 0.020001
    assert not subject.terminal_metric_gate(failed)["pass"]


@pytest.mark.parametrize("preflight_pass,gate_pass,updates,saves,gate_calls", [
    (False, False, 2, 0, 0),
    (True, False, 4, 0, 1),
    (True, True, 4, 1, 1),
])
def test_failure_never_saves_and_success_saves_only_model_04000(
        monkeypatch, tmp_path, preflight_pass, gate_pass,
        updates, saves, gate_calls):
    monkeypatch.setattr(subject, "PREFLIGHT_UPDATE", 2)
    monkeypatch.setattr(subject, "PREFLIGHT_AUDIT_UPDATES", (1, 2))
    monkeypatch.setattr(subject, "TERMINAL_UPDATE", 4)
    monkeypatch.setattr(subject.rehearsal, "FORMAL_LOG_INTERVAL", 2)
    model = TinyRoleModel()
    calls = {"iteration": 0, "anchor": 0, "save": [], "gate": 0}

    monkeypatch.setattr(subject, "make_anchor_loader", lambda *_args: object())

    def anchor(*_args):
        calls["anchor"] += 1
        value = 1.0 if calls["anchor"] == 1 else 0.8
        return {name: value for name in (
            "a_fluctuation", "b_fluctuation", "p1_mean", "p1_aggregate")}

    monkeypatch.setattr(subject, "evaluate_train_anchor", anchor)

    def backward(model, *_args, **_kwargs):
        calls["iteration"] += 1
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        audit = {
            "iteration": calls["iteration"], "role_heads": {},
            "mean_profile_refiner": {},
        } if calls["iteration"] <= 2 else None
        return {"real_loss": 0.0}, audit

    monkeypatch.setattr(subject, "backward_real_role_specialization", backward)
    monkeypatch.setattr(
        subject, "evaluate_preflight",
        lambda *_args: {
            "status": "GO" if preflight_pass else "STOP_RSM1B_PREFLIGHT",
            "pass": preflight_pass,
        })

    def gate(*_args):
        calls["gate"] += 1
        return {
            "status": "GO" if gate_pass else "STOP_RSM1B_UPDATE4000_GATE",
            "pass": gate_pass,
        }

    monkeypatch.setattr(subject, "evaluate_terminal_gate", gate)
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common, "normalizer_dict",
        lambda _normalizer: {})
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common, "save_checkpoint",
        lambda path, *_args, **_kwargs: calls["save"].append(path))
    args = argparse.Namespace(updates=5, output_dir=tmp_path, num_workers=0)
    real = {"loader": TinyLoader(), "normalizer": object(), "teacher": {}}
    result = subject.train_endpoint(
        args, model, real, {}, torch.device("cpu"))
    assert result["completed_updates"] == updates
    assert len(calls["save"]) == saves
    assert calls["gate"] == gate_calls
    assert (result["checkpoint"] is None) == (saves == 0)
    if saves:
        assert calls["save"] == [tmp_path / "model_04000.pth"]
