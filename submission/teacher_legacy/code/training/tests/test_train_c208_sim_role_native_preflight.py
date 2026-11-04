from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
SCRIPT = TRAINING_ROOT / "train_c208_sim_role_native_preflight.py"
SPEC = importlib.util.spec_from_file_location("c208_sim_rn1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def formal_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        baseline_sim=Path("baseline.pth"),
        baseline_sim_sha256=subject.BASELINE_SIM_SHA256,
        baseline_summary=Path("summary.json"),
        baseline_summary_sha256=subject.BASELINE_SUMMARY_SHA256,
        baseline_split_manifest=Path("split_manifest.json"),
        baseline_split_manifest_sha256=(
            subject.BASELINE_SPLIT_MANIFEST_SHA256),
        control_checkpoint=Path("control.pth"),
        control_checkpoint_sha256=subject.CONTROL_CHECKPOINT_SHA256,
        aggregate_teacher_cache=Path("aggregate.pt"),
        aggregate_teacher_cache_sha256=(
            subject.AGGREGATE_TEACHER_CACHE_SHA256),
        role_teacher_cache=Path("roles.pt"),
        role_teacher_cache_sha256="f" * 64,
        real_stats=Path("stats.pt"),
        real_stats_sha256=subject.REAL_STATS_SHA256,
        sim_dir=Path("sim"), output_dir=tmp_path / "run",
        device="cpu", num_workers=0)


def frozen_training_config() -> dict:
    return {
        "phase": "sim", "updates": 10_000, "start_checkpoint": None,
        "trainable_scope": "all", "batch_size": 8, "lr": 3e-4,
        "weight_decay": 1e-5, "grad_clip": 1.0, "seed": 0,
        "num_workers": 0, "max_files": 0, "window_stride": 20,
        "input_noise_std": 0.0, "point_loss_mode": "official_accuracy",
        "mse_weight": 0.25, "mse_warmup_updates": 0,
        "mean_loss_weight": 0.25, "member_loss_weight": 1.0,
        "member_loss_mode": "point", "bootstrap_probability": 0.75,
        "quantile_loss_weight": 0.05, "soft_sps_loss_weight": 0.05,
        "soft_sps_tau_fraction": 0.05, "soft_sps_warmup_updates": 1500,
        "soft_sps_uq_only": False, "soft_sps_proxy_real_scale": True,
        "coverage_bce_weight": 0.0, "profile_loss_weight": 0.0,
        "distill_weight": 0.005, "accuracy_weights": [1.0, 4.0, 1.0],
    }


def test_cli_freezes_all_existing_artifacts_and_empty_output(tmp_path):
    args = formal_args(tmp_path)
    subject.validate_args(args)
    assert args.role_teacher_cache_sha256 == "f" * 64

    args.control_checkpoint_sha256 = "0" * 64
    with pytest.raises(ValueError, match="control_checkpoint_sha256"):
        subject.validate_args(args)

    occupied = formal_args(tmp_path)
    occupied.output_dir.mkdir()
    (occupied.output_dir / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be empty"):
        subject.validate_args(occupied)


def test_frozen_training_config_rejects_any_scientific_drift():
    training = frozen_training_config()
    subject.validate_frozen_training_config(training)
    training["mean_loss_weight"] = 1.0
    with pytest.raises(ValueError, match="mean_loss_weight"):
        subject.validate_frozen_training_config(training)


def test_control_validator_checks_iteration_and_config(monkeypatch):
    model_config = {"architecture": "ensemble_unet", "base": 208}
    normalizer = {
        "mean_input": [0.8, -0.1, 1.2],
        "mean_target": [0.7, -0.2, 1.0],
        "std_input": [0.4, 0.2, 12.0],
        "std_target": [0.5, 0.25, 10.0],
    }
    training = frozen_training_config()
    buffers = subject.expected_normalizer_buffers(normalizer)
    payload = {
        "phase": "sim", "iteration": 200,
        "model_config": model_config, "training_config": training,
        "normalizer": normalizer,
        "model_state_dict": {"weight": torch.ones(1), **buffers},
        "history": [{"iteration": 200}],
    }
    baseline = {
        "model_config": model_config, "normalizer": normalizer,
        "training_config": training,
    }
    monkeypatch.setattr(
        subject, "MODEL_CONFIG_SHA256",
        subject.rehearsal.canonical_sha256(model_config))
    monkeypatch.setattr(
        subject, "CONTROL_TRAINING_CONFIG_SHA256",
        subject.rehearsal.canonical_sha256(training))
    monkeypatch.setattr(
        subject, "NORMALIZER_SHA256",
        subject.rehearsal.canonical_sha256(normalizer))
    subject.validate_control_payload(payload, baseline)

    payload["iteration"] = 199
    with pytest.raises(ValueError, match="update-200"):
        subject.validate_control_payload(payload, baseline)


class TinyCandidate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.register_buffer("persistence_scale", torch.ones(3))
        self.register_buffer("persistence_bias", torch.zeros(3))
        self.register_buffer("target_std", torch.ones(3))

    def set_normalizer_statistics(
            self, mean_input, mean_target, std_input, std_target):
        safe = torch.as_tensor(std_target)[:3].clamp_min(1e-8)
        self.persistence_scale.copy_(torch.as_tensor(std_input)[:3] / safe)
        self.persistence_bias.copy_((
            torch.as_tensor(mean_input)[:3]
            - torch.as_tensor(mean_target)[:3]) / safe)
        self.target_std.copy_(safe)


def test_initial_identity_distinguishes_constructor_and_training_ready(
        monkeypatch):
    normalizer = SimpleNamespace(
        mean_input=torch.tensor([0.8, -0.1, 1.2]),
        mean_target=torch.tensor([0.7, -0.2, 1.0]),
        std_input=torch.tensor([0.4, 0.2, 12.0]),
        std_target=torch.tensor([0.5, 0.25, 10.0]))
    raw = TinyCandidate()
    raw_sha = subject.shg.model_state_sha256(raw)
    raw.set_normalizer_statistics(
        normalizer.mean_input, normalizer.mean_target,
        normalizer.std_input, normalizer.std_target)
    ready_sha = subject.shg.model_state_sha256(raw)
    monkeypatch.setattr(subject, "INITIAL_MODEL_STATE_SHA256", raw_sha)
    monkeypatch.setattr(
        subject, "TRAINING_READY_MODEL_STATE_SHA256", ready_sha)
    monkeypatch.setattr(
        subject.shg, "build_model",
        lambda _payload, _device, load_weights: TinyCandidate())
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train, "seed_everything", lambda _seed: None)

    model, metadata = subject.build_training_ready_candidate(
        {}, normalizer, torch.device("cpu"))
    assert metadata["raw_constructor_state_sha256"] == raw_sha
    assert metadata["training_ready_state_sha256"] == ready_sha
    assert raw_sha != ready_sha
    assert metadata["trainable_parameters_unchanged"] is True
    assert metadata["changed_state_keys"] == [
        "persistence_scale", "persistence_bias", "target_std"]
    subject.validate_normalizer_buffers(model.state_dict(), normalizer)


def test_formal_initial_hashes_lock_both_real_boundaries():
    assert subject.INITIAL_MODEL_STATE_SHA256 == (
        "16f3341b5352ec42ea0df326e1e797bcc088c551a3f76834bd34bba4ca8a70b1")
    assert subject.TRAINING_READY_MODEL_STATE_SHA256 == (
        "1f5292c972669fbd5ab40b53dcd0932f3c3c38cefe1bf9387e0f1391d9f2d113")


def test_stored_role_cache_consistency_is_bounded_and_fail_closed():
    torch.manual_seed(0)
    a = torch.randn(3, 4, 2, 3, 2)
    b = torch.randn(3, 4, 2, 3, 2)
    a = (a - a.mean(dim=1, keepdim=True)).half()
    b = (b - b.mean(dim=1, keepdim=True)).half()
    mean = torch.randn(3, 1, 2, 3, 2).half()
    aggregate = (0.5 * a.float() + 0.5 * b.float() + mean.float()).half()
    roles = {
        "a_fluctuation_uv_physical": a,
        "b_fluctuation_uv_physical": b,
        "p1_temporal_mean_uv_physical": mean,
    }
    audit = subject.role_cache_consistency_audit(
        aggregate, roles, chunk_rows=2)
    assert audit["pass"] is True

    broken = dict(roles)
    broken["p1_temporal_mean_uv_physical"] = mean + 1.0
    with pytest.raises(RuntimeError, match="consistency audit"):
        subject.role_cache_consistency_audit(
            aggregate, broken, chunk_rows=2)


def test_preflight_source_cannot_save_a_checkpoint():
    source = inspect.getsource(subject)
    assert "save_checkpoint(" not in source
    assert "torch.save(" not in source
    assert '"checkpoints_saved": []' in source
