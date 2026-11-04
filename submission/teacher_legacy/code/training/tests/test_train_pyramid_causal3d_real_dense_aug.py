from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_pyramid_causal3d_real_dense_aug as subject


def _normalizer():
    return subject.smu1.FoldNormalizer((
        torch.tensor([1.0, -2.0, 0.0]), torch.tensor([3.0, 4.0, 0.0]),
        torch.tensor([2.0, 5.0, 0.0]), torch.tensor([6.0, 7.0, 0.0]),
    ), torch.device("cpu"))


def test_frozen_real_dense_contract_uses_mature_weights_and_audited_windows() -> None:
    assert (subject.NORMALIZER_STRIDE, subject.OPTIMIZER_STRIDE) == (20, 5)
    assert (subject.EXPECTED_NORMALIZER_WINDOWS, subject.EXPECTED_OPTIMIZER_WINDOWS) == (2544, 9933)
    assert (subject.UPDATES, subject.BATCH_SIZE, subject.LR, subject.EMA_DECAY) == (4000, 8, 2e-4, .999)
    assert subject.SCHEDULER_T_MAX == 12_000
    assert subject.ACCURACY_WEIGHTS == (5.0, 3.0, 2.0)
    assert subject.MSE_WEIGHT == .25
    assert subject.AUGMENTATION == {
        "gain_std": .01, "normalized_noise_std": .01,
        "mask_aware_blur_probability": .15, "mask_aware_blur_strength": .20,
    }


def test_real_pressure_is_zero_preserved_and_all3_mse_is_mature_recipe() -> None:
    history = torch.randn(2, 20, 8, 8, 3)
    history[..., 2] = 0.0
    torch.manual_seed(12)
    augmented = subject.augment_real_history(history, _normalizer())
    assert torch.equal(augmented[..., 2], torch.zeros_like(augmented[..., 2]))
    target = torch.zeros_like(history)
    prediction = target.clone()
    prediction[..., 2] = 999.0
    assert abs(float(subject.real_mse(prediction, target)) - 999.0**2 / 3.0) < .1


def test_normalizer_rebinds_model_and_reports_zero_pressure_source() -> None:
    config = subject.PyramidCausal3DOperatorConfig(
        e0_channels=8, e1_channels=8, e2_channels=8,
        e0_blocks=1, e1_blocks=1, e2_blocks=1)
    model = subject.PyramidCausal3DOperator(config, "causal_temporal_candidate")
    normalizer = _normalizer()
    subject.install_normalizer(model, normalizer)
    assert torch.equal(model.persistence_scale, normalizer.std_input / normalizer.std_target)
    assert torch.equal(model.persistence_bias, (normalizer.mean_input - normalizer.mean_target) / normalizer.std_target)
    assert float(normalizer.std_input[2]) == float(normalizer.std_target[2]) == 1.0
    assert float(model.target_std[2]) == float(normalizer.std_target[2])
    # The official evaluator classifies an all-zero Real pressure target as
    # unmeasured, not as a third supervised scalar.
    target = torch.zeros(1, 20, 8, 8, 3)
    target[..., :2] = 1.0
    target = target.numpy()
    assert subject.base.SCORING.measured_channels(target) == 2


def test_sim_loader_accepts_only_self_described_raw_u10000(tmp_path: Path,
                                                           monkeypatch) -> None:
    config = subject.PyramidCausal3DOperatorConfig(
        e0_channels=8, e1_channels=8, e2_channels=8,
        e0_blocks=1, e1_blocks=1, e2_blocks=1)
    model = subject.PyramidCausal3DOperator(config, "causal_temporal_candidate")
    monkeypatch.setattr(subject.sim_long, "EXPECTED_PARAMETERS",
                        sum(parameter.numel() for parameter in model.parameters()))
    model_config = model.architecture_config()
    contract = {**subject.sim_long.training_contract(),
                "architecture": model_config, "device_type": "cuda"}
    normalizer = {name: torch.zeros(3) if "mean" in name else torch.ones(3)
                  for name in ("mean_input", "mean_target", "std_input", "std_target")}
    payload = {
        "checkpoint_version": subject.sim_long.CHECKPOINT_VERSION,
        "schema": subject.sim_long.SCHEMA,
        "endpoint_kind": subject.sim_long.ENDPOINT_KIND,
        "ema_state": None, "model_config": model_config,
        "interaction_mode": "causal_temporal_candidate",
        "model_state": model.state_dict(),
        "model_state_sha256": subject.tensor_state_sha(model.state_dict()),
        "normalizer_tensors": normalizer,
        "normalizer_sha256": subject.tensor_state_sha(normalizer),
        "next_update": subject.sim_long.UPDATES,
        "sample_cursor": subject.sim_long.UPDATES,
        "schedule_sha256": subject.sim_long.SAMPLE_INDEX_SHA256,
        "config_seal": subject.sim_long.canonical_sha(contract),
        "data_seal": {"sim": "sealed"}, "git_seal": {"git": "sealed"},
        "device_type": "cuda",
    }
    path = tmp_path / "stage_s.pt"
    torch.save(payload, path)
    loaded, report = subject.load_pc3d_sim_checkpoint(path, torch.device("cpu"))
    assert subject.tensor_state_sha(loaded.state_dict()) == payload["model_state_sha256"]
    assert report["stage_s_update"] == 10_000
    payload["sample_cursor"] = 9_999
    torch.save(payload, path)
    try:
        subject.load_pc3d_sim_checkpoint(path, torch.device("cpu"))
    except RuntimeError as error:
        assert "raw-u10000" in str(error)
    else:
        raise AssertionError("intermediate Stage-S checkpoint bypassed Real start gate")


def test_exact_resume_restores_optimizer_scheduler_ema_rng_and_cursor(tmp_path: Path) -> None:
    model = nn.Linear(3, 3)
    normalizer = _normalizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=subject.LR, weight_decay=subject.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    ema = subject.ema_lib.FP32EMAShadow(model, subject.EMA_DECAY)
    generator = torch.Generator().manual_seed(17)
    schedule = subject.make_sample_schedule(5, 3, 1)
    model(torch.ones(1, 3)).sum().backward(); optimizer.step()
    ema.update_after_optimizer_step(model, optimizer_step=1); scheduler.step()
    source = {"sim": "sealed", "manifest": "sealed"}
    payload = subject.build_resume_payload(
        model=model, optimizer=optimizer, scheduler=scheduler, ema=ema, update=1,
        schedule=schedule, source=source, normalizer=normalizer,
        device=torch.device("cpu"), trace=[{"update": 1}])
    path = tmp_path / "resume.pt"
    subject.atomic_torch_save(path, payload)
    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_ema = ema.seal(model)["model_state"]
    for value in model.parameters():
        value.data.zero_()
    optimizer.param_groups[0]["lr"] = 99.0
    scheduler.last_epoch = 99
    cursor, restored_ema, trace = subject.restore_resume(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        expected_source=source, normalizer=normalizer,
        device=torch.device("cpu"), schedule=schedule)
    assert cursor == 1
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected_model.items())
    assert all(torch.equal(restored_ema.seal(model)["model_state"][name], value)
               for name, value in expected_ema.items())
    assert trace == [{"update": 1}]
    assert optimizer.param_groups[0]["lr"] != 99.0 and scheduler.last_epoch != 99
    # A fresh process restoring the same cursor consumes the same next batch
    # and produces the same raw and materialized EMA endpoint state.
    resumed_model = nn.Linear(3, 3)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=subject.LR,
                                          weight_decay=subject.WEIGHT_DECAY)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=3)
    resumed_cursor, resumed_ema, resumed_trace = subject.restore_resume(
        path, model=resumed_model, optimizer=resumed_optimizer, scheduler=resumed_scheduler,
        expected_source=source, normalizer=normalizer, device=torch.device("cpu"), schedule=schedule)
    assert (resumed_cursor, resumed_trace, schedule[resumed_cursor]) == (1, [{"update": 1}], schedule[1])
    for current_model, current_optimizer, current_scheduler, current_ema in (
            (model, optimizer, scheduler, restored_ema),
            (resumed_model, resumed_optimizer, resumed_scheduler, resumed_ema)):
        current_optimizer.zero_grad(set_to_none=True)
        current_model(torch.tensor([[float(schedule[1][0]), 1.0, -1.0]])).sum().backward()
        current_optimizer.step()
        current_ema.update_after_optimizer_step(current_model, optimizer_step=2)
        current_scheduler.step()
    assert all(torch.equal(model.state_dict()[name], value)
               for name, value in resumed_model.state_dict().items())
    assert subject.ema_lib.tensor_state_sha256(restored_ema.seal(model)["model_state"]) == (
        subject.ema_lib.tensor_state_sha256(resumed_ema.seal(resumed_model)["model_state"]))


def test_clean716_open_requires_immutable_endpoint_seal(tmp_path: Path) -> None:
    audit = subject.PrefixAccessAudit()
    row = {"file": "x.h5", "val_boundary": 60}
    try:
        audit.before_open(row, "clean716")
    except RuntimeError as error:
        assert "before endpoint seal" in str(error)
    else:
        raise AssertionError("clean716 opener bypassed endpoint guard")
    endpoint = tmp_path / "ema_endpoint.pt"
    endpoint.write_bytes(b"sealed")
    audit.record_endpoint_seal({"path": str(endpoint), "file_sha256": subject.file_sha256(endpoint)})
    audit.before_open(row, "clean716")
    endpoint.write_bytes(b"changed")
    try:
        audit.before_open(row, "clean716")
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("mutated endpoint bypassed clean716 guard")
