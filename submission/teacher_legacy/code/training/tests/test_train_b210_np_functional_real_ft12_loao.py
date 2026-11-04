from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_real_ft12_loao as subject  # noqa: E402


def test_contract_freezes_single_candidate_real_ft12_and_primary_endpoint():
    assert subject.FOLD_ORDER == (10, 0)
    assert subject.UPDATES == 12_000
    assert subject.DIAGNOSTIC_UPDATE == 4_000
    assert subject.BATCH_SIZE == 8
    assert subject.SEED == 4
    assert subject.LR == pytest.approx(2e-4)
    assert subject.WEIGHT_DECAY == pytest.approx(1e-5)
    assert subject.SCHEDULER_T_MAX == 12_000
    assert subject.WALL_BUDGET_SECONDS == 120 * 60
    assert subject.ENDPOINT_ORDER == (
        "ema_u4000", "raw_u12000", "ema_u12000")
    assert subject.PRIMARY_ENDPOINT == "ema_u12000"
    assert subject.CONTRACT["endpoints"] == {
        "seal_before_any_held_open": list(subject.ENDPOINT_ORDER),
        "primary": "ema_u12000",
        "diagnostic_only": ["ema_u4000", "raw_u12000"],
        "held_endpoint_selection": False,
    }
    training = subject.CONTRACT["training"]
    assert training["native_losses"] == (
        "Real joint point/member/quantile/soft-SPS")
    assert training["hgrad"] is False
    assert training["teacher"] is False
    assert training["sim_rehearsal"] is False


def test_real_tke_aux_uses_npt1_calibration_and_early_cosine():
    assert subject.TKE_AUX_WEIGHT == pytest.approx(0.01)
    assert subject.REAL_AUX_CALIBRATION_MEDIAN == pytest.approx(0.20735451)
    assert subject.tke_aux_weight(1) == pytest.approx(0.01)
    assert 0.0 < subject.tke_aux_weight(750) < 0.01
    assert subject.tke_aux_weight(1500) == pytest.approx(0.0, abs=1e-15)
    assert subject.tke_aux_weight(1501) == 0.0
    with pytest.raises(ValueError, match="positive"):
        subject.tke_aux_weight(0)


def test_exact_b210_config_rejects_architecture_or_bundle_drift():
    exact = {
        "architecture": "ensemble_unet",
        **vars(subject.expected_b210_config()),
    }
    assert subject.validate_b210_model_config(exact) == (
        subject.expected_b210_config())
    wrong_base = {**exact, "base": 208}
    with pytest.raises(ValueError, match="frozen candidate"):
        subject.validate_b210_model_config(wrong_base)
    wrong_dropout = {**exact, "backbone_residual_dropout": 0.0}
    with pytest.raises(ValueError, match="frozen candidate"):
        subject.validate_b210_model_config(wrong_dropout)
    wrong_architecture = {**exact, "architecture": "other"}
    with pytest.raises(ValueError, match="not an ensemble"):
        subject.validate_b210_model_config(wrong_architecture)


def test_start_checkpoint_uses_strict_state_load_and_exact_counts(
        tmp_path, monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.legacy = torch.nn.Parameter(torch.zeros(3))
            self.energy_controller = torch.nn.Linear(1, 1, bias=True)

        def set_normalizer_statistics(self, *_values):
            pass

    tiny = TinyModel(object())
    payload = {
        "phase": "sim",
        "iteration": 10_000,
        "metrics": {"status": subject.b210_sim.PASS_STATUS},
        "model_config": {"architecture": "ensemble_unet"},
        "model_state_dict": tiny.state_dict(),
    }
    path = tmp_path / "start.pth"
    torch.save(payload, path)
    monkeypatch.setattr(subject, "EnsembleUNet2d", TinyModel)
    monkeypatch.setattr(
        subject, "validate_b210_model_config", lambda _config: object())
    monkeypatch.setattr(subject, "EXPECTED_PARAMETERS", 5)
    monkeypatch.setattr(subject, "EXPECTED_CONTROLLER_PARAMETERS", 2)

    class Normalizer:
        mean_input = torch.zeros(3)
        mean_target = torch.zeros(3)
        std_input = torch.ones(3)
        std_target = torch.ones(3)

    loaded, report = subject.validate_start_checkpoint(
        path, subject.dr1.sha256(path), Normalizer(), torch.device("cpu"))
    assert report["strict_load"] is True
    assert report["parameters"] == 5
    assert report["controller_parameters"] == 2
    del loaded
    gc.collect()

    broken = dict(payload)
    broken["model_state_dict"] = {
        key: value for key, value in payload["model_state_dict"].items()
        if key != "legacy"
    }
    broken_path = tmp_path / "broken.pth"
    torch.save(broken, broken_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        subject.validate_start_checkpoint(
            broken_path, subject.dr1.sha256(broken_path),
            Normalizer(), torch.device("cpu"))


def test_trainability_accepts_only_explicit_architecture_count():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model.energy_controller = torch.nn.Linear(1, 1)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = subject.configure_trainable(
        model, expected_parameters=total)
    assert sum(parameter.numel() for parameter in trainable) == total
    with pytest.raises(RuntimeError, match="trainability changed"):
        subject.configure_trainable(
            model, expected_parameters=total + 1)


def test_endpoint_uses_flat_normalizer_schema_and_strict_load(
        tmp_path, monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self, _config=None):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def architecture_config(self):
            return {"architecture": "ensemble_unet", "tiny": True}

        def set_normalizer_statistics(self, *_values):
            pass

    class Normalizer:
        mean_input = torch.zeros(3)
        mean_target = torch.ones(3)
        std_input = torch.full((3,), 2.0)
        std_target = torch.full((3,), 3.0)

    model = TinyModel()
    normalizer = Normalizer()
    normalizer_report = subject.normalizer_payload(normalizer)
    path = tmp_path / "endpoint.pth"
    seal = subject.save_endpoint(
        path, model, model.state_dict(), endpoint="ema_u12000",
        iteration=subject.UPDATES, held_aoa=10,
        start_report={"checkpoint_sha256": "a" * 64},
        normalizer_report=normalizer_report,
        stream_sha256="b" * 64, ema_metadata={"updates": subject.UPDATES})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["normalizer"] == normalizer_report["values"]
    assert set(payload["normalizer"]) == {
        "mean_input", "mean_target", "std_input", "std_target"}
    assert payload["normalizer_sha256"] == normalizer_report["sha256"]
    assert "values" not in payload["normalizer"]

    monkeypatch.setattr(subject, "EnsembleUNet2d", TinyModel)
    monkeypatch.setattr(
        subject, "validate_b210_model_config", lambda _config: object())
    loaded = subject.load_endpoint_model(
        seal, normalizer, torch.device("cpu"))
    assert torch.equal(loaded.state_dict()["weight"], torch.ones(1))


def test_u4000_stream_digest_is_exact_schedule_prefix():
    schedule = {"rows": [{"iteration": value} for value in range(5)]}
    prefix = subject.schedule_prefix_sha256(schedule, 3)
    assert prefix == subject.dr1.canonical_sha256(schedule["rows"][:3])
    assert prefix != subject.dr1.canonical_sha256(schedule["rows"])
    with pytest.raises(ValueError, match="invalid"):
        subject.schedule_prefix_sha256(schedule, 6)


def _seal(tmp_path: Path, endpoint: str, index: int) -> dict:
    return {
        "path": str(tmp_path / f"{endpoint}.pth"),
        "file_sha256": f"{index + 1:x}" * 64,
        "model_state_sha256": f"{index + 4:x}" * 64,
    }


def test_held_open_waits_for_three_seals_and_excludes_bad_file(tmp_path):
    ledger = subject.ThreeEndpointAccessAudit(10)
    held = {"file": "held.h5", "aoa": 10, "val_boundary": 80}
    with pytest.raises(RuntimeError, match="before three seals"):
        ledger.before_open(held, "held")
    with pytest.raises(RuntimeError, match="excluded H5"):
        ledger.before_open({
            "file": subject.dr1.EXCLUDED_BASENAME,
            "aoa": 0,
            "val_boundary": 80,
        }, "fit")
    for index, endpoint in enumerate(subject.ENDPOINT_ORDER):
        ledger.seal(endpoint, _seal(tmp_path, endpoint, index))
    ledger.before_open(held, "held")
    ledger.record_open(held, "held")
    for endpoint in subject.ENDPOINT_ORDER:
        ledger.record_evaluation(endpoint)
    report = ledger.report(["held.h5"])
    assert report["held_first_open_after_three_seals"] is True
    assert report["held_evaluation_order"] == list(subject.ENDPOINT_ORDER)
    assert report["held_complete_passes"] == 3


def _state(samples: int, rel_sum: float, tke_sum: float,
           mvpe_sum: float, sps_numerator: float) -> dict:
    state = subject.smu1.new_metric_state()
    state.update({
        "samples": samples,
        "rel_error_sum": rel_sum,
        "tke_error_sum": tke_sum,
        "mvpe_error_sum": mvpe_sum,
        "sps_numerator": sps_numerator,
        "sps_scored": 100 * samples,
        "sps_covered": 80 * samples,
        "sps_width_sum": 5 * samples,
    })
    return state


def test_two_fold_aggregate_merges_metric_state_not_fold_scores(monkeypatch):
    monkeypatch.setattr(subject, "validate_fold_rows", lambda _rows: None)
    primary_states = [
        _state(1, 0.10, 0.20, 0.30, 50.0),
        _state(3, 1.20, 1.50, 1.80, 90.0),
    ]
    rows = []
    for held, samples, primary in zip((10, 0), (1, 3), primary_states):
        endpoints = {
            "ema_u4000": {
                "metric_state": _state(
                    samples, 0.5 * samples, 0.6 * samples,
                    0.7 * samples, 20.0 * samples)},
            "raw_u12000": {
                "metric_state": _state(
                    samples, 0.4 * samples, 0.5 * samples,
                    0.6 * samples, 25.0 * samples)},
            "ema_u12000": {"metric_state": primary},
        }
        rows.append({"held_aoa": held, "samples": samples,
                     "endpoints": endpoints})
    result = subject.aggregate_fold_rows(rows)
    merged = subject.smu1.merge_metric_states(primary_states)
    expected = subject.smu1.finalize_metric_state(merged)
    assert result["pooled_primary_metric_state"] == merged
    assert result["pooled_primary_metrics"] == expected
    assert result["samples"] == 4
    fold_scores = [
        subject.smu1.finalize_metric_state(state)["rel_l2_score"]
        for state in primary_states
    ]
    assert expected["rel_l2_score"] != pytest.approx(
        sum(fold_scores) / len(fold_scores))
    assert result["decision"] == "AWAIT_FROZEN_C208_ANCHOR_COMPARISON"


def _metrics(**overrides) -> dict[str, float]:
    values = {
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 90.0,
        "point_score": 87.0,
        "native_sps_score": 45.0,
        "native_sps_coverage": 0.8,
        "native_sps_mean_width": 0.1,
    }
    values.update(overrides)
    return values


def test_postrun_promotion_gate_uses_frozen_c208_not_diagnostics():
    anchor = _metrics()
    candidate = _metrics(
        rel_l2_score=89.90,
        tke_rel_l2_score=81.00,
        mvpe_rel_l2_score=90.00,
        native_sps_score=45.50)
    passing = subject.two_fold_promotion_gate(
        candidate, anchor,
        [{"non_time_delta": 0.01}, {"non_time_delta": 0.02}])
    assert passing["promotion_pass"] is True
    assert passing["primary_endpoint"] == "ema_u12000"
    failed = subject.two_fold_promotion_gate(
        {**candidate, "native_sps_score": 45.299}, anchor,
        [{"non_time_delta": 0.01}, {"non_time_delta": 0.02}])
    assert failed["promotion_pass"] is False


def test_held0_requires_sha_sealed_passing_held10(tmp_path):
    increment = {"non_time_delta": 0.0}
    row = {
        "schema": subject.SCHEMA,
        "experiment": subject.EXPERIMENT,
        "contract_sha256": subject.CONTRACT_SHA256,
        "held_aoa": 10,
        "start": {"checkpoint_sha256": "c" * 64},
        "ft12_increment_diagnostic": increment,
        "fold10_disaster_gate": subject.fold10_disaster_gate(increment),
    }
    path = tmp_path / "held10.json"
    path.write_text(json.dumps(row), encoding="utf-8")
    report = subject.validate_prior_fold10(
        0, [str(path), subject.dr1.sha256(path)])
    assert report["kind"] == "sealed_held10_pass"
    assert report["start_checkpoint_sha256"] == "c" * 64
    with pytest.raises(RuntimeError, match="SHA changed"):
        subject.validate_prior_fold10(0, [str(path), "a" * 64])
    with pytest.raises(ValueError, match="cannot receive"):
        subject.validate_prior_fold10(
            10, [str(path), subject.dr1.sha256(path)])


def _fold_args(held: int) -> argparse.Namespace:
    return argparse.Namespace(
        fold_worker=True,
        aggregate_only=False,
        contract_only=False,
        held_aoa=held,
        output_dir=Path("out"),
        real_dir=Path("real"),
        clean_manifest=Path("manifest.json"),
        start_sim=Path("start.pth"),
        start_sim_sha256="a" * 64,
        device="cuda:0",
        hardware_class="rtx4080_super",
        execute_formal_gpu=True,
        acknowledge_protocol=subject.FORMAL_ACKNOWLEDGEMENT,
        prior_fold10=(None if held == 10 else ["held10.json", "b" * 64]),
        fold_result=None,
        aggregate_output=None,
    )


def test_cli_freezes_held10_then_sha_authorized_held0():
    subject.validate_args(_fold_args(10))
    subject.validate_args(_fold_args(0))
    missing = _fold_args(0)
    missing.prior_fold10 = None
    with pytest.raises(ValueError, match="requires prior"):
        subject.validate_args(missing)
    wrong = _fold_args(20)
    with pytest.raises(ValueError, match="outside"):
        subject.validate_args(wrong)
