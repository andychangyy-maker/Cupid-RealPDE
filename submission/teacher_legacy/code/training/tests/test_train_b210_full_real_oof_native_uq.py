from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "code/training"
sys.path.insert(0, str(TRAINING))
SCRIPT = TRAINING / "train_b210_full_real_oof_native_uq.py"
SPEC = importlib.util.spec_from_file_location("b210_full_oof_uq", SCRIPT)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


class Normalizer:
    mean_input = torch.zeros(3)
    mean_target = torch.zeros(3)
    std_input = torch.ones(3)
    std_target = torch.ones(3)


class ObjectiveModel:
    config = SimpleNamespace(quantile_levels=(0.1, 0.5, 0.9))

    @staticmethod
    def _utility_interval(quantiles):
        return (
            F.softplus(-quantiles[..., 0]) + 0.01,
            F.softplus(quantiles[..., -1]) + 0.01,
        )


def objective_args():
    return SimpleNamespace(
        pinball_weight=0.05,
        soft_sps_weight=2.0,
        cvar_weight=0.35,
        cvar_fraction=0.25,
        soft_tau_fraction=0.035,
        coverage_target=0.85,
        robust_sps_weight=0.0,
        robust_residual_jitter_fraction=0.0,
        robust_residual_tail_scale=1.0,
        peer_sps_weight=0.0,
        peer_residual_mix_fraction=0.0,
        interval_score_weight=0.0025,
        extent_distill_weight=0.0,
        coverage_bce_weight=0.0,
    )


def _objective(target_value: float):
    prediction = torch.zeros((2, 2, 2, 2, 3))
    quantiles = torch.zeros((2, 2, 2, 2, 2, 3), requires_grad=True)
    target = torch.full_like(prediction, target_value)
    residual = torch.full((2, 2, 2, 2, 2), 0.125)
    factor = torch.tensor([0.8, 0.9])
    loss, components = SUBJECT.uq_train.uq_objective(
        objective_args(), ObjectiveModel(), prediction, quantiles,
        target, Normalizer(), residual_physical_override=residual,
        point_factor_override=factor,
        scored_override=torch.ones_like(residual, dtype=torch.bool))
    return loss, components, quantiles


def test_oof_objective_uses_cached_residual_not_in_sample_target():
    loss_a, components_a, quantiles = _objective(1.0)
    loss_b, components_b, _ = _objective(7.0)
    assert torch.equal(loss_a, loss_b)
    assert torch.equal(components_a["pinball"], components_b["pinball"])
    assert torch.equal(
        components_a["soft_official"], components_b["soft_official"])
    loss_a.backward()
    assert quantiles.grad is not None
    assert bool(torch.isfinite(quantiles.grad).all())
    assert bool(torch.count_nonzero(quantiles.grad))


def test_oof_objective_requires_residual_and_factor_together():
    prediction = torch.zeros((1, 1, 1, 1, 3))
    quantiles = torch.zeros((1, 1, 1, 1, 2, 3))
    target = torch.ones_like(prediction)
    residual = torch.zeros((1, 1, 1, 1, 2))
    with pytest.raises(ValueError, match="supplied together"):
        SUBJECT.uq_train.uq_objective(
            objective_args(), ObjectiveModel(), prediction, quantiles,
            target, Normalizer(), residual_physical_override=residual)


def test_cache_loader_binds_tensor_and_identity_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(SUBJECT, "EXPECTED_SAMPLES", 2)
    identities = [
        {"file": "a.h5", "start": 0, "re": 4_000, "aoa": 0, "fold": 0},
        {"file": "b.h5", "start": 0, "re": 4_001, "aoa": 5, "fold": 1},
    ]
    residuals = (torch.arange(
        2 * 20 * 32 * 64 * 2, dtype=torch.float32) % 101).to(
            torch.float16).reshape(2, 20, 32, 64, 2)
    factors = torch.tensor([0.5, 0.75])
    cache = {
        "schema": SUBJECT.cache_builder.SCHEMA,
        "folds": SUBJECT.point_fold.FOLDS,
        "aoa_values": list(SUBJECT.point_fold.AOA_VALUES),
        "window_stride": 20,
        "samples": 2,
        "manifest": {
            "semantic_sha256": SUBJECT.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256},
        "manifest_file_sha256": SUBJECT.ft12.FORMAL_MANIFEST_FILE_SHA256,
        "clean716_manifest_read": False,
        "sample_identities": identities,
        "sample_identity_sha256": SUBJECT.dr1.canonical_sha256(identities),
        "signed_residual_physical_uv": residuals,
        "signed_residual_physical_uv_sha256": SUBJECT._sha_tensor_bytes(residuals),
        "oof_point_factor": factors,
        "oof_point_factor_sha256": SUBJECT._sha_tensor_bytes(factors),
        "ordered_history_stream_sha256": "1" * 64,
        "ordered_target_stream_sha256": "2" * 64,
        "fold_results": [],
    }
    path = tmp_path / "cache.pth"
    torch.save(cache, path)
    payload, observed_residuals, observed_factors, report = SUBJECT.load_oof_cache(
        path, SUBJECT.dr1.sha256(path))
    assert payload["sample_identities"] == identities
    assert torch.equal(observed_residuals, residuals)
    assert torch.equal(observed_factors, factors)
    assert report["samples"] == 2
    assert report["clean716_manifest_read"] is False


def test_final_point_consumer_requires_full13202_schema(tmp_path, monkeypatch):
    normalizer = {
        "mean_input": [0.0, 0.0, 0.0],
        "mean_target": [0.0, 0.0, 0.0],
        "std_input": [1.0, 1.0, 1.0],
        "std_target": [1.0, 1.0, 1.0],
    }
    state = {"weight": torch.tensor([1.0])}
    state_sha = SUBJECT.dr1.model_state_sha256(state)
    checkpoint = tmp_path / "ema_u4000.pth"
    torch.save({
        "schema": SUBJECT.POINT_ENDPOINT_SCHEMA,
        "training_scope": "all_clean81_full13202",
        "optimizer_windows": 13_202,
        "normalizer_windows": 2_544,
        "optimizer_stride": 5,
        "normalizer_stride": 20,
        "phase": "real",
        "iteration": 4_000,
        "endpoint": "ema_u4000",
        "primary": True,
        "held_aoa": None,
        "start": {"checkpoint_sha256": SUBJECT.point_fold.START_SHA256},
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "model_config": {"architecture": "ensemble_unet"},
        "normalizer": normalizer,
        "normalizer_sha256": SUBJECT.dr1.canonical_sha256(normalizer),
        "fresh_native_uq_crossfit_refit_required": True,
    }, checkpoint)
    seal = {
        "path": str(checkpoint),
        "file_sha256": SUBJECT.dr1.sha256(checkpoint),
        "model_state_sha256": state_sha,
    }
    result = {
        "schema": SUBJECT.POINT_RESULT_SCHEMA,
        "inventory": {
            "full_optimizer_windows": 13_202,
            "normalizer_windows": 2_544,
        },
        "optimizer_support": {"windows": 13_202, "stride": 5},
        "normalizer": {"windows": 2_544, "full_tail_in_normalizer": False},
        "primary_endpoint": "ema_u4000",
        "fresh_native_uq_crossfit_refit_required": True,
        "access_audit": {
            "clean716_manifest_read": False,
            "allowlist_exact_and_ordered": True,
        },
        "start": {"checkpoint_sha256": SUBJECT.point_fold.START_SHA256},
        "endpoints": {"ema_u4000": {"seal": seal}},
    }
    result_path = tmp_path / "point_stage_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    fake_model = SimpleNamespace()
    monkeypatch.setattr(
        SUBJECT.uq_train, "load_model",
        lambda *_args, **_kwargs: fake_model)
    monkeypatch.setattr(
        SUBJECT.uq_release, "partition_state",
        lambda _model, uq: {"weight": torch.tensor([1.0])} if not uq else {})
    _payload, observed_model, _normalizer, report = SUBJECT.load_final_point(
        result_path, SUBJECT.dr1.sha256(result_path), torch.device("cpu"))
    assert observed_model is fake_model
    assert report["training_scope"] == "all_clean81_full13202"
    assert report["model_state_sha256"] == state_sha


def test_fresh_uq_reset_uses_sim_only_and_keeps_point_bitwise(
        tmp_path, monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.point = torch.nn.Linear(1, 1)
            self.uq_fusion = torch.nn.Linear(1, 1)

    model = TinyModel()
    with torch.no_grad():
        model.point.weight.fill_(7.0)
        model.point.bias.fill_(8.0)
        model.uq_fusion.weight.fill_(9.0)
        model.uq_fusion.bias.fill_(10.0)
    sim_model = TinyModel()
    with torch.no_grad():
        sim_model.point.weight.fill_(-1.0)
        sim_model.point.bias.fill_(-2.0)
        sim_model.uq_fusion.weight.fill_(3.0)
        sim_model.uq_fusion.bias.fill_(4.0)
    source = tmp_path / "sim_u10000.pth"
    torch.save({
        "phase": "sim",
        "iteration": 10_000,
        "model_config": {"architecture": "tiny"},
        "model_state_dict": sim_model.state_dict(),
    }, source)
    source_sha = SUBJECT.dr1.sha256(source)
    monkeypatch.setattr(SUBJECT.point_fold, "START_SHA256", source_sha)
    point_before = SUBJECT.dr1.model_state_sha256(
        SUBJECT.uq_release.partition_state(model, uq=False))
    report = SUBJECT.reset_uq_from_pure_sim(
        model, source, source_sha, {"architecture": "tiny"})
    assert SUBJECT.dr1.model_state_sha256(
        SUBJECT.uq_release.partition_state(model, uq=False)) == point_before
    assert torch.equal(model.point.weight, torch.full_like(model.point.weight, 7.0))
    assert torch.equal(
        model.uq_fusion.weight,
        torch.full_like(model.uq_fusion.weight, 3.0))
    assert report["full_real_uq_state_inherited"] is False
    assert report["point_partition_bitwise"] is True
    assert report["uq_sha256_after_reset"] == report["pure_sim_uq_sha256"]
    assert report["final_point_uq_sha256_before_reset"] != report[
        "uq_sha256_after_reset"]


def test_fixed_four_cycle_contract_and_no_clean716_cli():
    assert (SUBJECT.CYCLES, SUBJECT.UPDATES_PER_CYCLE, SUBJECT.TOTAL_UPDATES) == (
        4, 800, 3_200)
    assert SUBJECT.ENDPOINT == "fixed_full_oof_u3200"
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'add_argument("--evaluation-manifest' not in source
    assert 'add_argument("--clean716' not in source
    assert 'add_argument("--uq-init-sim"' in source
    assert 'add_argument("--loao-uq-aggregate"' in source
