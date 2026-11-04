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
SCRIPT = TRAINING_ROOT / "train_c208_sim_low_lr_refine.py"
SPEC = importlib.util.spec_from_file_location("c208_sim_s4k1", SCRIPT)
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
        teacher_cache=Path("teacher.pt"),
        teacher_cache_sha256=subject.TEACHER_CACHE_SHA256,
        real_stats=Path("stats.pt"),
        real_stats_sha256=subject.REAL_STATS_SHA256,
        sim_dir=Path("sim"), output_dir=tmp_path / "run",
        device="cpu", num_workers=0)


def training_config() -> dict:
    return {
        "phase": "sim", "updates": 10_000, "start_checkpoint": None,
        "trainable_scope": "all", "batch_size": 8, "eval_interval": 500,
        "lr": 3e-4, "weight_decay": 1e-5, "grad_clip": 1.0,
        "seed": 0, "num_workers": 0, "max_files": 0,
        "window_stride": 20, "input_noise_std": 0.0,
        "point_loss_mode": "official_accuracy", "mse_weight": 0.25,
        "mse_warmup_updates": 0, "mean_loss_weight": 0.25,
        "member_loss_weight": 1.0, "member_loss_mode": "point",
        "bootstrap_probability": 0.75, "quantile_loss_weight": 0.05,
        "soft_sps_loss_weight": 0.05, "soft_sps_tau_fraction": 0.05,
        "soft_sps_warmup_updates": 1500, "soft_sps_uq_only": False,
        "soft_sps_proxy_real_scale": True, "coverage_bce_weight": 0.0,
        "profile_loss_weight": 0.0, "accuracy_weights": [1.0, 4.0, 1.0],
        "distill_weight": 0.005, "teacher_cache": "p1.pt",
    }


def baseline_payload() -> dict:
    normalizer = {
        "mean_input": [0.0, 0.0, 0.0],
        "mean_target": [0.0, 0.0, 0.0],
        "std_input": [1.0, 1.0, 1.0],
        "std_target": [1.0, 1.0, 1.0],
    }
    return {
        "phase": "sim", "iteration": 10_000,
        "model_config": copy.deepcopy(subject.rehearsal.EXPECTED_MODEL_CONFIG),
        "training_config": training_config(),
        "history": [{"iteration": 10_000}],
        "normalizer": normalizer,
        "model_state_dict": {"target_std": torch.ones(3)},
    }


def test_frozen_configuration_and_cli_hashes(tmp_path):
    args = formal_args(tmp_path)
    subject.validate_args(args)
    assert subject.METHOD == "c208_s4k1_sim_low_lr_refine"
    assert subject.REFINEMENT_LR == pytest.approx(3e-5)
    assert subject.SCHEDULER_T_MAX == 4_000
    assert subject.START_GLOBAL_UPDATE == 10_000
    assert subject.TERMINAL_GLOBAL_UPDATE == 14_000
    args.teacher_cache_sha256 = "a" * 64
    with pytest.raises(ValueError, match="teacher_cache_sha256"):
        subject.validate_args(args)


def test_baseline_contract_binds_model_training_and_normalizer(monkeypatch):
    payload = baseline_payload()
    monkeypatch.setattr(
        subject, "BASELINE_MODEL_CONFIG_SHA256",
        subject.rehearsal.canonical_sha256(payload["model_config"]))
    monkeypatch.setattr(
        subject, "BASELINE_TRAINING_CONFIG_SHA256",
        subject.rehearsal.canonical_sha256(payload["training_config"]))
    monkeypatch.setattr(
        subject, "BASELINE_NORMALIZER_SHA256",
        subject.rehearsal.canonical_sha256(payload["normalizer"]))
    metadata = subject.validate_baseline_contract(payload)
    assert metadata["normalizer_sha256"] == subject.BASELINE_NORMALIZER_SHA256

    wrong = copy.deepcopy(payload)
    wrong["training_config"]["soft_sps_uq_only"] = True
    with pytest.raises(ValueError, match="soft_sps_uq_only"):
        subject.validate_baseline_contract(wrong)
    wrong = copy.deepcopy(payload)
    wrong["normalizer"]["std_target"][0] = 2.0
    with pytest.raises(ValueError, match="normalizer hash"):
        subject.validate_baseline_contract(wrong)
    wrong = copy.deepcopy(payload)
    wrong["model_state_dict"]["target_std"][0] = 2.0
    with pytest.raises(ValueError, match="target_std"):
        subject.validate_baseline_contract(wrong)


def test_teacher_hash_is_checked_before_deserialization(monkeypatch):
    calls = []
    args = argparse.Namespace(teacher_cache=Path("teacher.pt"))
    monkeypatch.setattr(subject.shg, "sha256", lambda _path: "bad")
    monkeypatch.setattr(
        subject.rehearsal, "load_and_validate_cache",
        lambda *_args: calls.append("deserialize"))
    with pytest.raises(RuntimeError, match="before deserialization"):
        subject.load_teacher_cache(args, [], object())
    assert calls == []


class Window:
    def __init__(self, trajectory: int):
        self.trajectory = trajectory


class ToyDataset(Dataset):
    def __init__(self, length=20, conditions=4):
        self.windows = [Window(index % conditions) for index in range(length)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        value = torch.tensor(float(index))
        return value, value


def test_prefix_contract_matches_actual_seeded_loader(monkeypatch):
    monkeypatch.setattr(subject, "EXPECTED_SIM_WINDOWS", 20)
    monkeypatch.setattr(subject, "PREFIX_LOCAL_UPDATE", 3)
    monkeypatch.setattr(subject, "ANCHOR_SIZE", 4)
    args = argparse.Namespace(batch_size=4, num_workers=0, seed=0)
    dataset = ToyDataset()
    contract = subject.derive_prefix_contract(dataset, args)
    base = DataLoader(dataset, batch_size=4, shuffle=False)
    actual = subject.rehearsal.ensemble_train.tag_training_loader(base, args)
    for update, batch in enumerate(actual, start=1):
        if update > subject.PREFIX_LOCAL_UPDATE:
            break
        assert batch[3].tolist() == contract["batch_sample_indices"][update - 1]
        assert subject.shg.batch_key(batch[2], batch[3]) == (
            contract["batch_keys"][update - 1])
    prefix = {
        value for batch in contract["batch_sample_indices"] for value in batch}
    assert not prefix.intersection(contract["unseen_indices"])
    assert len(contract["seen_indices"]) == len(contract["unseen_indices"]) == 4
    payload = dict(contract)
    digest = payload.pop("canonical_sha256")
    assert digest == subject.rehearsal.canonical_sha256(payload)


def test_condition_balanced_contract_is_all_natural_windows(monkeypatch):
    monkeypatch.setattr(subject, "EXPECTED_SIM_WINDOWS", 20)
    monkeypatch.setattr(subject, "EXPECTED_SIM_CONDITIONS", 4)
    monkeypatch.setattr(subject, "EXPECTED_WINDOWS_PER_CONDITION", 5)
    contract = subject.condition_balanced_contract(ToyDataset())
    assert contract["indices"] == list(range(20))
    assert set(contract["condition_counts"].values()) == {5}
    wrong = ToyDataset(20, conditions=3)
    with pytest.raises(RuntimeError, match="not 100 conditions"):
        subject.condition_balanced_contract(wrong)


def anchor(point=10.0, rel=20.0, tke=30.0, mvpe=40.0,
           p1=2.0, sps=0.5):
    return {
        "point_score": point, "rel_l2_score": rel,
        "tke_rel_l2_score": tke, "mvpe_rel_l2_score": mvpe,
        "p1_distill_loss": p1, "native_soft_sps_proxy": sps,
    }


def test_prefix_and_terminal_gates_are_exact():
    initial = anchor()
    seen = anchor(point=10.01, rel=19.98, tke=29.98, mvpe=39.98,
                  p1=1.99, sps=0.498)
    unseen = anchor(point=10.0, rel=19.98, tke=29.98, mvpe=39.98,
                    p1=1.995, sps=0.498)
    gate = subject.prefix_gate(initial, seen, initial, unseen)
    assert gate["pass"]
    assert gate["status"] == subject.PREFIX_PASS_STATUS
    bad = dict(unseen, p1_distill_loss=1.9950001)
    assert not subject.prefix_gate(initial, seen, initial, bad)["pass"]

    terminal = anchor(point=10.03, rel=19.95, tke=29.95, mvpe=39.95,
                      p1=1.98, sps=0.495)
    gate = subject.terminal_gate(initial, terminal)
    assert gate["pass"]
    assert gate["status"] == subject.PASS_STATUS
    terminal["native_soft_sps_proxy"] = 0.4949
    assert not subject.terminal_gate(initial, terminal)["pass"]


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.1))

    def forward_uq_training(self, history):
        prediction = self.value.expand_as(history)
        members = prediction[:, None].expand(-1, 4, *prediction.shape[1:])
        quantiles = prediction[:, None]
        return (prediction, members, quantiles, prediction, prediction,
                None, None)


def test_native_step_uses_global_loss_iteration(monkeypatch):
    calls = []

    def point_loss(_args, prediction, *_rest):
        calls.append(_rest[-1])
        return prediction.square().mean(), {"truth": prediction.mean()}

    def member_loss(_args, members, _target, _ids, _normalizer,
                    _weights, iteration):
        calls.append(iteration)
        return members.square().mean(), members.new_tensor(1.0)

    monkeypatch.setattr(subject.rehearsal.ensemble_train, "point_loss", point_loss)
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train,
        "bootstrap_member_point_loss", member_loss)
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train, "residual_quantile_loss",
        lambda _m, p, *_a: p.square().mean())
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train, "soft_native_sps_objective",
        lambda p, *_a, **_k: (p.square().mean(), p.new_tensor(0.5)))
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train,
        "component_point_distillation_loss",
        lambda p, *_a: (p.square().mean(), {"distill": p.mean()}))
    model = FakeModel()
    args = argparse.Namespace(
        accuracy_weights=(1.0, 4.0, 1.0), member_loss_mode="point",
        bootstrap_probability=0.75, seed=0, mean_loss_weight=0.25,
        member_loss_weight=1.0, quantile_loss_weight=0.05,
        coverage_bce_weight=0.0, profile_loss_weight=0.0,
        distill_weight=0.005, soft_sps_loss_weight=0.05,
        soft_sps_warmup_updates=1500, soft_sps_tau_fraction=0.05,
        grad_clip=1.0)
    normalizer = argparse.Namespace(
        preprocess=lambda x, y: (x, y))
    history = torch.ones(2, 1, 1, 1, 3)
    batch = (history, torch.zeros_like(history), torch.tensor([0, 1]),
             torch.tensor([0, 1]))
    teacher = torch.zeros(2, 1, 1, 1, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=subject.REFINEMENT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4000)
    values, _key, _indices = subject.sim_update(
        model, batch, normalizer, normalizer, teacher, args,
        torch.device("cpu"), 1, optimizer, scheduler)
    assert calls == [10_001, 10_001]
    assert values["global_update"] == 10_001
    assert values["soft_sps_weight"] == pytest.approx(0.05)


def test_failure_never_writes_checkpoint(monkeypatch, tmp_path):
    calls = []

    def save(path, *_args):
        calls.append(path)
        path.write_bytes(b"checkpoint")

    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common, "save_checkpoint", save)
    normalizer = argparse.Namespace(
        mean_input=torch.zeros(3), mean_target=torch.zeros(3),
        std_input=torch.ones(3), std_target=torch.ones(3))
    args = argparse.Namespace(phase="sim")
    assert subject.save_terminal_checkpoint(
        {"pass": False}, tmp_path, object(), args, [], normalizer) == (None, None)
    assert calls == []
    path, digest = subject.save_terminal_checkpoint(
        {"pass": True}, tmp_path, object(), args, [], normalizer)
    assert Path(path).name == "model_14000.pth"
    assert len(digest) == 64
    assert calls == [tmp_path / "model_14000.pth"]


def test_prefix_pass_continues_same_optimizer_scheduler_and_iterator(
        monkeypatch, tmp_path):
    model = object()
    optimizer = object()
    scheduler = object()
    updates = []
    creations = {"optimizer": 0, "loader": 0, "save": 0}
    expected_indices = [[update] for update in range(200)]
    expected_keys = [f"batch-{update}" for update in range(200)]
    prefix = {
        "canonical_sha256": "prefix", "batch_keys_sha256": "keys",
        "seen_indices_sha256": "seen", "unseen_indices_sha256": "unseen",
        "seen_indices": [0], "unseen_indices": [1],
        "batch_sample_indices": expected_indices, "batch_keys": expected_keys,
    }
    balanced = {"indices": [0], "canonical_sha256": "balanced"}
    monkeypatch.setattr(subject, "derive_prefix_contract", lambda *_a: prefix)
    monkeypatch.setattr(subject, "condition_balanced_contract", lambda *_a: balanced)
    monkeypatch.setattr(subject.shg, "build_model", lambda *_a, **_k: model)
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common, "set_model_stats",
        lambda *_a: None)
    monkeypatch.setattr(subject, "evaluate_anchor", lambda *_a: anchor())
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train, "seed_everything", lambda *_a: None)

    def tag_loader(*_args):
        creations["loader"] += 1
        return object()

    monkeypatch.setattr(
        subject.rehearsal.ensemble_train, "tag_training_loader", tag_loader)
    monkeypatch.setattr(
        subject.rehearsal.ensemble_train.common.repro, "cycling",
        lambda _loader: iter(range(4000)))

    def make_optimizer(*_args):
        creations["optimizer"] += 1
        return optimizer, scheduler

    monkeypatch.setattr(subject, "make_optimizer", make_optimizer)

    def update(_model, _batch, *_args):
        local_update = _args[-3]
        seen_optimizer, seen_scheduler = _args[-2:]
        updates.append((local_update, seen_optimizer, seen_scheduler))
        index = local_update - 1
        key = expected_keys[index] if index < 200 else f"batch-{index}"
        natural = expected_indices[index] if index < 200 else [index]
        return ({"local_update": local_update}, key, natural)

    monkeypatch.setattr(subject, "sim_update", update)
    monkeypatch.setattr(
        subject, "prefix_gate",
        lambda *_a: {"pass": True, "status": subject.PREFIX_PASS_STATUS})
    monkeypatch.setattr(
        subject, "terminal_gate",
        lambda *_a: {"pass": True, "status": subject.PASS_STATUS})

    def save(*_args):
        creations["save"] += 1
        return "model_14000.pth", "f" * 64

    monkeypatch.setattr(subject, "save_terminal_checkpoint", save)
    raw_loader = argparse.Namespace(dataset=ToyDataset())
    sim_args = argparse.Namespace(seed=0, eval_interval=1000, weight_decay=1e-5,
                                  grad_clip=1.0)
    frozen = {key: 0 for key in subject.shg.FROZEN_SIM_LOSS_KEYS}
    baseline_metadata = {
        "path": "baseline", "sha256": "a", "model_config_sha256": "b",
        "training_config_sha256": "c", "normalizer_sha256": "d",
        "training_config": frozen,
    }
    summary = subject.run_experiment(
        argparse.Namespace(output_dir=tmp_path), {"model_config": {}},
        baseline_metadata, {}, sim_args, raw_loader, object(), object(),
        torch.empty(0), {}, {}, torch.device("cpu"))
    assert summary["status"] == subject.PASS_STATUS
    assert summary["completed_local_updates"] == 4000
    assert creations == {"optimizer": 1, "loader": 1, "save": 1}
    assert [item[0] for item in updates] == list(range(1, 4001))
    assert {id(item[1]) for item in updates} == {id(optimizer)}
    assert {id(item[2]) for item in updates} == {id(scheduler)}


def test_source_has_one_save_and_same_process_continuation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("common.save_checkpoint(") == 1
    assert "ensemble_train.train(" not in source
    run_source = inspect.getsource(subject.run_experiment)
    assert run_source.count("make_optimizer(") == 1
    assert run_source.count("tag_training_loader(") == 1
    assert '"model_14000.pth"' in inspect.getsource(
        subject.save_terminal_checkpoint)
