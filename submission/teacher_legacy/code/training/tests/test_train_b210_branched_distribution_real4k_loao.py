from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for value in (CODE_ROOT, TRAINING_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import train_b210_branched_distribution_real4k_loao as subject  # noqa: E402
from models.ensemble_unet2d import EnsembleUNet2d  # noqa: E402
from test_train_b210_distribution_preserving_real4k_loao import (  # noqa: E402
    tiny_config,
)


class IdentityNormalizer:
    def preprocess(self, history, target):
        return history, target


class FixedFuture:
    def augment_future(self, _history, target, _trajectory_ids, **_kwargs):
        future = target.clone()
        future[..., :2] += 0.1
        selected = torch.ones(
            target.shape[0], dtype=torch.bool, device=target.device)
        return future, selected


def test_branched_backward_keeps_native_adapter_gradient_isolated(monkeypatch):
    model = EnsembleUNet2d(tiny_config()).train()
    with torch.no_grad():
        for heads in model.member_point_heads:
            for head in heads:
                head.weight.normal_(mean=0.0, std=0.1)
    history = torch.randn(1, 20, 16, 16, 3)
    target = torch.randn_like(history)
    batch = (
        history, target, torch.tensor([0]), torch.tensor([0]))

    def native_backward(model, batch, _normalizer, _device, _iteration,
                        **_kwargs):
        prediction, members, *_rest = model.forward_uq_training(batch[0])
        # The context substitutes base members, so this native graph has no
        # stochastic-adapter dependency.
        (prediction.square().mean() + members.square().mean()).backward(
            retain_graph=True)
        return {"native": 1.0}

    monkeypatch.setattr(subject, "_FUTURE_AUGMENTOR", FixedFuture())
    monkeypatch.setattr(
        subject.same_target, "ORIGINAL_REAL_BACKWARD", native_backward)
    values = subject.backward_branched_distribution(
        model, batch, IdentityNormalizer(), torch.device("cpu"), 1)
    assert values["native_adapter_gradient_norm"] == 0.0
    assert values["future_crps_adapter_gradient_norm"] > 0.0
    assert values["future_selected_fraction"] == 1.0
    assert values["future_target_delta_uv_rms"] > 0.0
    assert values["member_mean_prediction_max_abs_gap"] <= 2.0e-6
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.member_feature_adapters.parameters())


def _bank_payload(records):
    return {
        "format": "realpde_clean_real_closure_bank_v2",
        "records": records,
        "source_contract": {
            "manifest_sha256": subject.b210_sim.CLEAN_REAL_MANIFEST_SHA256,
            "records": 81,
            "known_invalid_trajectory_excluded": True,
            "validation_or_test_labels_read": False,
            "prefix_rule": "frames [0,val_boundary) only",
        },
    }


def test_fit_prefix_augmentor_excludes_held_records(tmp_path):
    records = [
        {"file": "a.h5", "aoa": 0, "re": 4000,
         "residual": torch.zeros(25, 4, 4, 2)},
        {"file": "b.h5", "aoa": 5, "re": 5000,
         "residual": torch.zeros(25, 4, 4, 2)},
        {"file": "held.h5", "aoa": 10, "re": 6000,
         "residual": torch.zeros(25, 4, 4, 2)},
    ]
    path = tmp_path / "bank.pt"
    torch.save(_bank_payload(records), path)
    metadata = [
        {"file": "a.h5", "aoa": 0, "re": 4000},
        {"file": "b.h5", "aoa": 5, "re": 5000},
    ]
    augmentor, audit = subject.prepare_fit_prefix_augmentor(
        path, subject.dr1.sha256(path), metadata, 10,
        torch.device("cpu"))
    assert len(augmentor.records) == 2
    assert audit["held_records"] == 0
    assert audit["fit_records"] == 2


def test_fit_prefix_augmentor_rejects_held_metadata(tmp_path):
    records = [
        {"file": "held.h5", "aoa": 10, "re": 6000,
         "residual": torch.zeros(25, 4, 4, 2)},
    ]
    path = tmp_path / "bank.pt"
    torch.save(_bank_payload(records), path)
    with pytest.raises(RuntimeError, match="held-condition"):
        subject.prepare_fit_prefix_augmentor(
            path, subject.dr1.sha256(path),
            [{"file": "held.h5", "aoa": 10, "re": 6000}],
            10, torch.device("cpu"))


def test_decision_gate_requires_identity_and_material_sps(monkeypatch):
    metrics = {
        "rel_l2_score": 96.0,
        "tke_rel_l2_score": 81.0,
        "mvpe_rel_l2_score": 96.5,
        "point_score": 91.0,
        "native_sps_score": 52.0,
    }
    result = {
        "arm": {"metrics": dict(metrics), "target_audit": {"x": 1}},
        "training_contract": {"sample_stream_sha256": "same"},
        "normalizer": {"statistics_sha256": "same"},
        "comparison": {
            "point_final_contribution_delta": 0.0,
            "non_time_final_contribution_delta": 0.05,
        },
        "disagreement_diagnostic": {
            "native_minus_collapsed_sps": 0.40,
            "member_error_pairwise_correlation_mean": 0.50,
            "element_spread_log_spearman": 0.60,
        },
    }
    same = {
        "arm": {"metrics": dict(metrics), "target_audit": {"x": 1}},
        "training_contract": {"sample_stream_sha256": "same"},
        "normalizer": {"statistics_sha256": "same"},
    }
    mechanism = {"delta": {"native_sps_score": 0.20}}
    monkeypatch.setattr(subject, "_FUTURE_BANK_AUDIT", {"held_records": 0})
    assert subject.decision_gate(result, same, mechanism)["pass"] is True
    mechanism["delta"]["native_sps_score"] = 0.02
    gate = subject.decision_gate(result, same, mechanism)
    assert gate["pass"] is False
    assert gate["conditions"]["sps_vs_same_target"] is False
