from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
SCRIPT = TRAINING_ROOT / "train_c208_standard_real_unique_gate.py"
SPEC = importlib.util.spec_from_file_location("c208_shg1_real", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def formal_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        start_shg=tmp_path / "model_10000.pth",
        start_shg_sha256="a" * 64,
        shg_summary=tmp_path / "shg_summary.json",
        shg_summary_sha256="b" * 64,
        identity_summary=tmp_path / "identity.json",
        identity_summary_sha256=(
            subject.rehearsal.FORMAL_IDENTITY_SUMMARY_SHA256),
        output_dir=tmp_path / "real_run",
        updates=subject.rehearsal.FORMAL_REQUESTED_UPDATES,
        real_dir=Path("real"), real_stats=Path("stats.pt"),
        device="cpu", num_workers=0,
    )


def shg_summary_payload(checkpoint_sha: str = "a" * 64) -> dict:
    return {
        "method": subject.SHG_METHOD,
        "status": subject.SHG_PROMOTE_STATUS,
        "completed_updates": subject.SHG_TERMINAL_UPDATE,
        "parameters": subject.rehearsal.EXPECTED_PARAMETER_COUNT,
        "model_config": dict(subject.rehearsal.EXPECTED_MODEL_CONFIG),
        "preflight": {
            "status": subject.SHG_PREFLIGHT_STATUS, "pass": True},
        "sim_gate": {
            "status": subject.SHG_PROMOTE_STATUS, "pass": True},
        "checkpoint": "/run/model_10000.pth",
        "checkpoint_sha256": checkpoint_sha,
        "baseline_sim": {
            "path": "/baseline/model_10000.pth",
            "sha256": subject.rehearsal.FORMAL_START_SIM_SHA256,
            "model_config_sha256": "c" * 64,
            "training_config_sha256": "d" * 64,
            "anchor_metrics": {"point": 0.2, "hgrad": 0.1},
            "source_summary": {
                "path": "/baseline/summary.json",
                "sha256": subject.SHG_BASELINE_SUMMARY_SHA256,
                "model_config_sha256": "c" * 64,
                "training_config_sha256": "d" * 64,
                "history_sha256": "e" * 64,
                "teacher_cache_sha256": subject.SHG_TEACHER_CACHE_SHA256,
                "teacher_cache_method": "sim_current_p1_point_teacher",
                "distill_weight": subject.rehearsal.FORMAL_P1_DISTILL_WEIGHT,
            },
        },
        "teacher_cache": {"sha256": subject.SHG_TEACHER_CACHE_SHA256},
        "training_contract": {
            "seed": 0,
            "control_updates": 200,
            "terminal_update": 10_000,
            "only_scientific_delta": subject.SHG_ONLY_SCIENTIFIC_DELTA,
            "soft_sps_gradient_scope": "full_graph_uq_only_false",
            "checkpoint_policy": "only_after_terminal_sim_gate_pass",
            "calibration_source_summary_sha256": (
                subject.SHG_CALIBRATION_SOURCE_SUMMARY_SHA256),
            "hgrad_weight": subject.SHG_HGRAD_WEIGHT,
            "hgrad_ramp_updates": subject.SHG_HGRAD_RAMP_UPDATES,
            "frozen_sim_loss": {
                "batch_size": 8,
                "lr": 2e-4,
                "weight_decay": 1e-5,
                "grad_clip": 1.0,
                "seed": 0,
                "max_files": 0,
                "window_stride": 20,
                "input_noise_std": 0.0,
                "scheduler_t_max": 10_000,
                "point_loss_mode": "official_accuracy",
                "accuracy_weights": [1.0, 4.0, 1.0],
                "mse_weight": 0.25,
                "mse_warmup_updates": 0,
                "mean_loss_weight": 0.25,
                "member_loss_weight": 1.0,
                "member_loss_mode": "point",
                "bootstrap_probability": 0.75,
                "quantile_loss_weight": 0.05,
                "soft_sps_loss_weight": 0.05,
                "soft_sps_tau_fraction": 0.05,
                "soft_sps_warmup_updates": 1500,
                "soft_sps_uq_only": False,
                "soft_sps_proxy_real_scale": True,
                "coverage_bce_weight": 0.0,
                "profile_loss_weight": 0.0,
                "distill_weight": 0.005,
            },
        },
    }


def write_summary(args: argparse.Namespace, payload: dict) -> None:
    args.shg_summary.write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8")
    args.shg_summary_sha256 = subject.rehearsal.sha256(args.shg_summary)


def test_args_freeze_updates_identity_and_fresh_output(tmp_path):
    args = formal_args(tmp_path)
    subject.validate_args(args)
    args.updates = 4000
    with pytest.raises(ValueError, match="updates 5000"):
        subject.validate_args(args)
    args = formal_args(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        subject.validate_args(args)


def test_shg_summary_requires_both_promotions_and_binds_checkpoint_sha(tmp_path):
    args = formal_args(tmp_path)
    payload = shg_summary_payload()
    write_summary(args, payload)
    loaded = subject.load_shg_summary(args)
    assert loaded["method"] == subject.SHG_METHOD
    assert loaded["checkpoint_sha256"] == args.start_shg_sha256

    failed = shg_summary_payload()
    failed["sim_gate"]["pass"] = False
    write_summary(args, failed)
    with pytest.raises(ValueError, match="SHG1 Sim gate"):
        subject.load_shg_summary(args)

    unbound = shg_summary_payload("d" * 64)
    write_summary(args, unbound)
    with pytest.raises(ValueError, match="does not bind"):
        subject.load_shg_summary(args)

    legacy = shg_summary_payload()
    legacy["method"] = "c208_shg1_sim_hgrad_pretrain"
    write_summary(args, legacy)
    with pytest.raises(ValueError, match="method"):
        subject.load_shg_summary(args)

    legacy = shg_summary_payload()
    legacy["training_contract"]["hgrad_weight"] = 0.004
    write_summary(args, legacy)
    with pytest.raises(ValueError, match="hgrad_weight"):
        subject.load_shg_summary(args)


@pytest.mark.parametrize("field,value,match", [
    ("sha256", "f" * 64, "frozen summary"),
    ("teacher_cache_sha256", "f" * 64, "teacher_cache_sha256"),
    ("distill_weight", 0.004, "distill_weight"),
])
def test_shg_summary_historical_provenance_tamper_stops(
        tmp_path, field, value, match):
    args = formal_args(tmp_path)
    payload = shg_summary_payload()
    payload["baseline_sim"]["source_summary"][field] = value
    write_summary(args, payload)
    with pytest.raises(ValueError, match=match):
        subject.load_shg_summary(args)


def test_shg_checkpoint_requires_sim10000_config_and_dual_gate_provenance(
        monkeypatch, tmp_path):
    args = formal_args(tmp_path)
    summary = shg_summary_payload()
    training = {
        **{key: value for key, value in
           summary["training_contract"]["frozen_sim_loss"].items()
           if key != "scheduler_t_max"},
        "method": subject.SHG_METHOD,
        "baseline_sim_sha256": summary["baseline_sim"]["sha256"],
        "baseline_summary": summary["baseline_sim"]["source_summary"]["path"],
        "baseline_summary_sha256": subject.SHG_BASELINE_SUMMARY_SHA256,
        "teacher_cache_sha256": subject.SHG_TEACHER_CACHE_SHA256,
        "hgrad_weight": subject.SHG_HGRAD_WEIGHT,
        "hgrad_ramp_updates": subject.SHG_HGRAD_RAMP_UPDATES,
        "calibration_source_summary_sha256": (
            subject.SHG_CALIBRATION_SOURCE_SUMMARY_SHA256),
        "preflight_pass": True,
        "sim_gate_pass": True,
    }
    payload = {
        "phase": "sim", "iteration": 10_000,
        "model_config": dict(subject.rehearsal.EXPECTED_MODEL_CONFIG),
        "training_config": training,
        "metrics": {"pass": True, "status": subject.SHG_PROMOTE_STATUS},
    }
    monkeypatch.setattr(subject.rehearsal, "sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(subject.torch, "load", lambda *_args, **_kwargs: payload)
    assert subject.load_shg_start_payload(args, summary) is payload
    payload["training_config"]["sim_gate_pass"] = False
    with pytest.raises(ValueError, match="sim_gate_pass"):
        subject.load_shg_start_payload(args, summary)
    payload["training_config"]["sim_gate_pass"] = True
    payload["training_config"]["hgrad_weight"] = 0.004
    with pytest.raises(ValueError, match="hgrad_weight"):
        subject.load_shg_start_payload(args, summary)


def test_shg_checkpoint_baseline_summary_sha_tamper_stops(
        monkeypatch, tmp_path):
    args = formal_args(tmp_path)
    summary = shg_summary_payload()
    frozen = summary["training_contract"]["frozen_sim_loss"]
    training = {
        **{key: value for key, value in frozen.items()
           if key != "scheduler_t_max"},
        "method": subject.SHG_METHOD,
        "baseline_sim_sha256": summary["baseline_sim"]["sha256"],
        "baseline_summary": summary["baseline_sim"]["source_summary"]["path"],
        "baseline_summary_sha256": "f" * 64,
        "teacher_cache_sha256": subject.SHG_TEACHER_CACHE_SHA256,
        "hgrad_weight": subject.SHG_HGRAD_WEIGHT,
        "hgrad_ramp_updates": subject.SHG_HGRAD_RAMP_UPDATES,
        "preflight_pass": True,
        "sim_gate_pass": True,
    }
    payload = {
        "phase": "sim", "iteration": 10_000,
        "model_config": dict(subject.rehearsal.EXPECTED_MODEL_CONFIG),
        "training_config": training,
        "metrics": {"pass": True, "status": subject.SHG_PROMOTE_STATUS},
    }
    monkeypatch.setattr(subject.rehearsal, "sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(subject.torch, "load", lambda *_args, **_kwargs: payload)
    with pytest.raises(ValueError, match="baseline_summary_sha256"):
        subject.load_shg_start_payload(args, summary)


def test_standard_real_path_is_exact_identity_and_has_no_auxiliary_loss():
    subject.validate_standard_real_contract()
    assert subject.STANDARD_IDENTITY_CONTRACT == (
        subject.rehearsal.IDENTITY_TRAINING_CONTRACT)
    assert subject.STANDARD_IDENTITY_CONTRACT["accuracy_weights"] == [5.0, 3.0, 2.0]
    assert subject.STANDARD_IDENTITY_CONTRACT["mse_warmup_updates"] == 1500
    assert subject.STANDARD_IDENTITY_CONTRACT["soft_sps_uq_only"] is True
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("rehearsal.backward_real_supervised(") == 1
    assert source.count("rehearsal.ensemble_train.evaluate(") == 1
    assert source.count('args.output_dir / "model_04000.pth"') == 1
    assert "role_distillation_objectives" not in source
    assert "teacher_distillation" not in source
    assert "hgrad_loss" not in source
    assert '"shg_or_other_auxiliary_loss": None' in source


def test_identity_summary_requires_unique_precise_update4000_row(
        monkeypatch, tmp_path):
    args = formal_args(tmp_path)
    row = {"iteration": 4000, **subject.NATIVE_STEP4000_METRICS}
    args.identity_summary.write_text(
        json.dumps({"history": [row]}), encoding="utf-8")
    monkeypatch.setattr(
        subject.rehearsal, "load_identity_step2000",
        lambda _args: {"path": str(args.identity_summary),
                       "sha256": args.identity_summary_sha256,
                       "metrics": {}})
    identity = subject.load_identity_step4000(args)
    assert identity["metrics_update4000"] == subject.NATIVE_STEP4000_METRICS
    bad = {**row, "native_sps_score": row["native_sps_score"] + 1e-8}
    args.identity_summary.write_text(
        json.dumps({"history": [bad]}), encoding="utf-8")
    with pytest.raises(ValueError, match="update-4000 native_sps_score"):
        subject.load_identity_step4000(args)


def passing_terminal_metrics() -> dict:
    metrics = dict(subject.NATIVE_STEP4000_METRICS)
    metrics["point_score"] += 0.10
    metrics["rel_l2_score"] += 0.30
    metrics["native_sps_score"] += 0.70
    return metrics


def test_gate_thresholds_and_unique_fixed725_access(monkeypatch):
    metrics = passing_terminal_metrics()
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train, "evaluate",
        lambda *_args, **_kwargs: metrics)
    real = {
        "validation_access_updates": [], "validation_loader": object(),
        "normalizer": object(),
    }
    identity = {"metrics_update4000": dict(subject.NATIVE_STEP4000_METRICS)}
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


class TinyLoader:
    def __iter__(self):
        return iter([None])


@pytest.mark.parametrize("gate_pass,saves", [(False, 0), (True, 1)])
def test_gate_failure_saves_nothing_success_saves_only_model04000(
        monkeypatch, tmp_path, gate_pass, saves):
    monkeypatch.setattr(subject, "TERMINAL_UPDATE", 2)
    monkeypatch.setattr(subject.rehearsal, "FORMAL_LOG_INTERVAL", 1)
    model = torch.nn.Linear(1, 1, bias=False)
    calls = {"updates": 0, "gate": 0, "save": []}

    def backward(model, *_args, **_kwargs):
        calls["updates"] += 1
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        return {"real_loss": 0.1, "real_supervised_loss": 0.1}

    def gate(*_args):
        calls["gate"] += 1
        return {
        "status": "PROMOTE_SHG1B_REAL_MODEL_04000" if gate_pass
        else "STOP_SHG1B_REAL_UPDATE4000_GATE",
            "pass": gate_pass, "metrics": {}, "delta": {},
        }

    monkeypatch.setattr(subject.rehearsal, "backward_real_supervised", backward)
    monkeypatch.setattr(subject, "evaluate_terminal_gate", gate)
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common, "normalizer_dict",
        lambda _normalizer: {})
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common, "save_checkpoint",
        lambda path, *_args, **_kwargs: calls["save"].append(path))
    args = argparse.Namespace(updates=5, output_dir=tmp_path)
    real = {"loader": TinyLoader(), "normalizer": object()}
    result = subject.train_endpoint(
        args, model, real, {}, torch.device("cpu"))
    assert result["completed_updates"] == 2
    assert calls["updates"] == 2
    assert calls["gate"] == 1
    assert len(calls["save"]) == saves
    if saves:
        assert calls["save"] == [tmp_path / "model_04000.pth"]
        assert result["checkpoint"] == str(tmp_path / "model_04000.pth")
    else:
        assert result["checkpoint"] is None
