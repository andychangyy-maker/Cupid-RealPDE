from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
SCRIPT = TRAINING_ROOT / "train_c208_sim_hgrad_pretrain.py"
SPEC = importlib.util.spec_from_file_location("c208_sim_shg1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def training_config() -> dict:
    return {
        "phase": "sim",
        "updates": 10_000,
        "start_checkpoint": None,
        "trainable_scope": "all",
        "batch_size": 8,
        "eval_interval": 500,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "seed": 0,
        "num_workers": 0,
        "max_files": 0,
        "window_stride": 20,
        "input_noise_std": 0.0,
        "point_loss_mode": "official_accuracy",
        "mse_weight": 0.25,
        "mse_warmup_updates": 0,
        "mean_loss_weight": 1.0,
        "member_loss_weight": 0.25,
        "member_loss_mode": "mse",
        "bootstrap_probability": 0.75,
        "quantile_loss_weight": 0.05,
        "soft_sps_loss_weight": 0.05,
        "soft_sps_tau_fraction": 0.05,
        "soft_sps_warmup_updates": 1500,
        "soft_sps_uq_only": False,
        "soft_sps_proxy_real_scale": True,
        "coverage_bce_weight": 0.0,
        "profile_loss_weight": 0.0,
        "accuracy_weights": [1.0, 4.0, 1.0],
        "distill_weight": 0.005,
        "teacher_cache": "p1.pt",
    }


def baseline_payload() -> dict:
    return {
        "phase": "sim",
        "iteration": 10_000,
        "model_config": dict(subject.rehearsal.EXPECTED_MODEL_CONFIG),
        "training_config": training_config(),
        "history": [{"iteration": 10_000, "train_loss": 1.0}],
    }


def baseline_summary(payload: dict | None = None) -> dict:
    payload = baseline_payload() if payload is None else payload
    return {
        "phase": "sim",
        "updates": 10_000,
        "parameters": subject.rehearsal.EXPECTED_PARAMETER_COUNT,
        "model_config": copy.deepcopy(payload["model_config"]),
        "training_config": copy.deepcopy(payload["training_config"]),
        "history": copy.deepcopy(payload["history"]),
        "point_distillation": {
            "distill_weight": subject.rehearsal.FORMAL_P1_DISTILL_WEIGHT,
            "teacher_cache": {
                "sha256": subject.TEACHER_CACHE_SHA256,
                "method": "sim_current_p1_point_teacher",
            },
        },
    }


def formal_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        baseline_sim=Path("baseline.pth"),
        baseline_sim_sha256=subject.BASELINE_SIM_SHA256,
        baseline_summary=Path("baseline-summary.json"),
        baseline_summary_sha256=subject.BASELINE_SUMMARY_SHA256,
        teacher_cache=Path("teacher.pt"),
        teacher_cache_sha256=subject.TEACHER_CACHE_SHA256,
        output_dir=tmp_path / "run",
        sim_dir=Path("sim"),
        real_stats=Path("stats.pt"),
        device="cpu",
        num_workers=0,
    )


def test_only_configuration_and_artifact_hashes_are_frozen(tmp_path):
    args = formal_args(tmp_path)
    subject.validate_args(args)
    assert subject.METHOD == "c208_shg1b_sim_hgrad_pretrain"
    assert subject.HGRAD_WEIGHT == pytest.approx(0.020)
    assert subject.HGRAD_RAMP_UPDATES == 200
    assert subject.calibrated_hgrad_weight(50) == pytest.approx(0.005)
    assert subject.calibrated_hgrad_weight(100) == pytest.approx(0.010)
    assert subject.calibrated_hgrad_weight(150) == pytest.approx(0.015)
    assert subject.calibrated_hgrad_weight(200) == pytest.approx(0.020)
    assert subject.calibrated_hgrad_weight(500) == pytest.approx(0.020)
    assert subject.hgrad.HGRAD_WEIGHT == pytest.approx(0.004)
    assert "hgrad.hgrad_weight(" not in inspect.getsource(subject.sim_update)
    assert subject.CONTROL_UPDATES == 200
    assert subject.TERMINAL_UPDATE == 10_000
    assert subject.AUDIT_UPDATES == (50, 100, 150, 200)
    args.baseline_summary_sha256 = "b" * 64
    with pytest.raises(ValueError, match="baseline summary SHA"):
        subject.validate_args(args)
    args.baseline_summary_sha256 = subject.BASELINE_SUMMARY_SHA256
    args.teacher_cache_sha256 = "a" * 64
    with pytest.raises(ValueError, match="teacher cache SHA"):
        subject.validate_args(args)


def test_baseline_contract_requires_current_p1_and_full_graph_soft_sps():
    metadata = subject.validate_baseline_payload(baseline_payload())
    assert len(metadata["model_config_sha256"]) == 64
    assert len(metadata["training_config_sha256"]) == 64

    wrong = baseline_payload()
    wrong["training_config"]["soft_sps_uq_only"] = True
    with pytest.raises(ValueError, match="soft_sps_uq_only"):
        subject.validate_baseline_payload(wrong)

    wrong = baseline_payload()
    wrong["training_config"]["distill_weight"] = 0.004
    with pytest.raises(ValueError, match="distill_weight"):
        subject.validate_baseline_payload(wrong)

    wrong = baseline_payload()
    wrong["training_config"]["teacher_cache"] = None
    with pytest.raises(ValueError, match="teacher_cache"):
        subject.validate_baseline_payload(wrong)

    wrong = baseline_payload()
    wrong["training_config"]["trainable_scope"] = "mean_profile_refiner"
    with pytest.raises(ValueError, match="trainable_scope"):
        subject.validate_baseline_payload(wrong)


def test_frozen_summary_binds_checkpoint_model_training_history_and_teacher():
    payload = baseline_payload()
    checkpoint_metadata = subject.validate_baseline_payload(payload)
    summary = baseline_summary(payload)
    metadata = subject.validate_baseline_summary_payload(
        summary, payload, checkpoint_metadata)
    assert metadata["teacher_cache_sha256"] == subject.TEACHER_CACHE_SHA256
    assert metadata["teacher_cache_method"] == (
        "sim_current_p1_point_teacher")

    mutations = (
        ("model_config", lambda value: value.__setitem__("channels", 160)),
        ("training_config", lambda value: value.__setitem__("lr", 1e-4)),
        ("history", lambda value: value[0].__setitem__("train_loss", 2.0)),
    )
    for expected, mutate in mutations:
        wrong = baseline_summary(payload)
        mutate(wrong[expected])
        with pytest.raises(ValueError, match=expected):
            subject.validate_baseline_summary_payload(
                wrong, payload, checkpoint_metadata)

    wrong = baseline_summary(payload)
    wrong["point_distillation"]["teacher_cache"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="teacher cache SHA"):
        subject.validate_baseline_summary_payload(
            wrong, payload, checkpoint_metadata)

    wrong = baseline_summary(payload)
    wrong["point_distillation"]["teacher_cache"]["method"] = "old_p1"
    with pytest.raises(ValueError, match="teacher cache method"):
        subject.validate_baseline_summary_payload(
            wrong, payload, checkpoint_metadata)

    wrong = baseline_summary(payload)
    wrong["point_distillation"]["distill_weight"] = 0.004
    with pytest.raises(ValueError, match="distill weight"):
        subject.validate_baseline_summary_payload(
            wrong, payload, checkpoint_metadata)

    wrong = baseline_summary(payload)
    wrong["history"] = []
    with pytest.raises(ValueError, match="history is absent"):
        subject.validate_baseline_summary_payload(
            wrong, payload, checkpoint_metadata)


def test_runtime_teacher_is_bound_to_frozen_baseline_summary():
    summary_metadata = {
        "teacher_cache_sha256": subject.TEACHER_CACHE_SHA256,
        "teacher_cache_method": "sim_current_p1_point_teacher",
    }
    teacher_metadata = {
        "sha256": subject.TEACHER_CACHE_SHA256,
        "method": "sim_current_p1_point_teacher",
    }
    subject.validate_teacher_summary_binding(
        teacher_metadata, summary_metadata)

    wrong = dict(teacher_metadata, sha256="d" * 64)
    with pytest.raises(RuntimeError, match="teacher cache differs"):
        subject.validate_teacher_summary_binding(wrong, summary_metadata)
    wrong = dict(teacher_metadata, method="old_p1")
    with pytest.raises(RuntimeError, match="teacher method differs"):
        subject.validate_teacher_summary_binding(wrong, summary_metadata)


class GradientDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        y = torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 4)
        x = torch.arange(4, dtype=torch.float32).view(1, 4).expand(3, 4)
        target = torch.zeros(2, 3, 4, 3)
        target[..., 0] = x
        target[..., 1] = 2.0 * y
        return target.clone(), target


def test_sim_scale_uses_only_sim_targets_and_preserves_rng():
    torch.manual_seed(1234)
    state = torch.get_rng_state().clone()
    contract = subject.compute_sim_hgrad_scale_contract(
        GradientDataset(), "split", "normalizer")
    assert torch.equal(torch.get_rng_state(), state)
    assert contract["method"] == (
        "sim_train_physical_uv_forward_difference_rms")
    assert contract["real_windows_used"] == 0
    assert contract["fixed725_windows_used"] == 0
    assert contract["scales"]["dx"] == pytest.approx([1.0, 1e-6])
    assert contract["scales"]["dy"] == pytest.approx([1e-6, 2.0])
    payload = dict(contract)
    digest = payload.pop("canonical_sha256")
    assert digest == subject.rehearsal.canonical_sha256(payload)


def test_sim_evaluation_adapter_supplies_physical_output_api():
    class SimNormalizer:
        mean_input = torch.tensor([1.0, 2.0, 3.0])
        mean_target = torch.tensor([4.0, 5.0, 6.0])
        std_input = torch.tensor([2.0, 3.0, 4.0])
        std_target = torch.tensor([5.0, 6.0, 7.0])

        def preprocess(self, history, target):
            return (
                (history - self.mean_input) / self.std_input,
                (target - self.mean_target) / self.std_target,
            )

    source = SimNormalizer()
    assert not hasattr(source, "postprocess_prediction")
    adapter = subject.SimEvaluationNormalizer(source)
    history = torch.tensor([[[[[3.0, 5.0, 7.0]]]]])
    target = torch.tensor([[[[[9.0, 11.0, 13.0]]]]])
    history_norm, target_norm = adapter.preprocess(history, target)
    assert torch.equal(adapter.postprocess_prediction(target_norm), target)
    assert torch.equal(adapter.postprocess_target(target_norm), target)
    assert torch.equal(
        history_norm,
        torch.tensor([[[[[1.0, 1.0, 1.0]]]]]),
    )


class Window:
    def __init__(self, trajectory):
        self.trajectory = trajectory


class TaggedToyDataset(Dataset):
    def __init__(self):
        self.windows = [Window(index % 3) for index in range(20)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        value = torch.tensor(float(index))
        return value, value


def test_seeded_training_loaders_have_identical_batch_keys():
    args = argparse.Namespace(batch_size=4, num_workers=0, seed=0)
    base = DataLoader(TaggedToyDataset(), batch_size=4, shuffle=False)
    left = subject.rehearsal.ensemble_train.tag_training_loader(base, args)
    right = subject.rehearsal.ensemble_train.tag_training_loader(base, args)
    for iteration, (left_batch, right_batch) in enumerate(
            zip(left, right), start=1):
        left_key = subject.batch_key(left_batch[2], left_batch[3])
        right_key = subject.batch_key(right_batch[2], right_batch[3])
        subject.require_paired_batch(left_key, right_key, iteration)
    with pytest.raises(RuntimeError, match="batch differs"):
        subject.require_paired_batch("left", "right", 7)


def test_hgrad_reaches_shared_point_graph_and_identity_guards():
    shared = torch.nn.Parameter(
        torch.arange(1 * 2 * 3 * 4 * 2, dtype=torch.float32).reshape(
            1, 2, 3, 4, 2))
    prediction = torch.cat(
        (shared, torch.zeros(1, 2, 3, 4, 1)), dim=-1)
    target = torch.zeros_like(prediction)
    normalizer = argparse.Namespace(
        mean_target=torch.zeros(3), std_target=torch.ones(3))
    contract = {"scales": {"dx": [1.0, 1.0], "dy": [1.0, 1.0]}}
    loss = subject.hgrad.hgrad_loss(
        prediction, target, normalizer, contract)
    loss.backward()
    assert shared.grad is not None
    assert torch.linalg.vector_norm(shared.grad) > 0

    subject.require_paired_identity("state", "state", "rng", "rng")
    with pytest.raises(RuntimeError, match="model states differ"):
        subject.require_paired_identity("left", "right", "rng", "rng")
    with pytest.raises(RuntimeError, match="RNG states differ"):
        subject.require_paired_identity("state", "state", "left", "right")


def anchor(point=10.0, rel=20.0, tke=30.0, mvpe=40.0,
           hgrad_value=2.0, p1=3.0, sps=0.5):
    return {
        "point_score": point,
        "rel_l2_score": rel,
        "tke_rel_l2_score": tke,
        "mvpe_rel_l2_score": mvpe,
        "hgrad_loss": hgrad_value,
        "p1_distill_loss": p1,
        "native_soft_sps_proxy": sps,
    }


def audits():
    return [
        {
            "iteration": update,
            "loss_ratio": 0.02,
            "gradient_ratio": 0.05,
            "cosine": 0.1,
            "point_gradient_norm": 1.0,
            "hgrad_gradient_norm": 0.05,
            "finite_nonzero": True,
        }
        for update in subject.AUDIT_UPDATES]


def test_paired_preflight_and_terminal_sim_gates_are_exact():
    initial = anchor(hgrad_value=2.0)
    control = anchor(hgrad_value=1.8)
    candidate = anchor(
        point=9.95, rel=19.9, tke=29.9, mvpe=39.9,
        hgrad_value=1.7, p1=3.03, sps=0.495)
    gate = subject.paired_preflight_gate(
        audits(), initial, control, candidate)
    assert gate["pass"]
    assert gate["status"] == subject.PREFLIGHT_PASS_STATUS
    candidate["native_soft_sps_proxy"] = 0.494
    assert not subject.paired_preflight_gate(
        audits(), initial, control, candidate)["pass"]

    baseline = anchor()
    promoted = anchor(
        point=10.03, rel=19.95, tke=29.95, mvpe=39.95,
        hgrad_value=1.9, p1=3.03, sps=0.495)
    terminal = subject.terminal_sim_gate(baseline, promoted)
    assert terminal["pass"]
    assert terminal["status"] == subject.PASS_STATUS
    promoted["native_soft_sps_proxy"] = 0.494
    assert not subject.terminal_sim_gate(baseline, promoted)["pass"]


def test_failure_never_calls_checkpoint_writer(monkeypatch, tmp_path):
    calls = []

    def save(path, *_args):
        calls.append(path)
        path.write_bytes(b"checkpoint")

    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common, "save_checkpoint", save)
    args = argparse.Namespace(phase="sim")
    normalizer = argparse.Namespace(
        mean_input=torch.zeros(3), mean_target=torch.zeros(3),
        std_input=torch.ones(3), std_target=torch.ones(3))
    failed = {"pass": False}
    assert subject.save_terminal_checkpoint(
        failed, tmp_path, object(), args, [], normalizer) == (None, None)
    assert calls == []

    passed = {"pass": True, "status": subject.PASS_STATUS}
    path, digest = subject.save_terminal_checkpoint(
        passed, tmp_path, object(), args, [], normalizer)
    assert Path(path).name == "model_10000.pth"
    assert len(digest) == 64
    assert calls == [tmp_path / "model_10000.pth"]
    assert args.preflight_pass is True
    assert args.sim_gate_pass is True


def test_source_has_one_terminal_checkpoint_call_and_no_generic_train_call():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("common.save_checkpoint(") == 1
    assert "ensemble_train.train(" not in source
    assert '"model_10000.pth"' in source
