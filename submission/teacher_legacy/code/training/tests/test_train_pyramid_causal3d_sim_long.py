from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_pyramid_causal3d_sim_long as subject


@dataclass(frozen=True)
class ToyConfig:
    e0_channels: int = 8


class ToyPC3D(nn.Module):
    def __init__(self, config: ToyConfig, mode: str):
        super().__init__()
        assert mode == "causal_temporal_candidate"
        self.config, self.mode = config, mode
        self.linear = nn.Linear(3, 3)
        self.dropout = nn.Dropout(.25)
        self.register_buffer("persistence_scale", torch.ones(3))
        self.register_buffer("persistence_bias", torch.zeros(3))
        self.register_buffer("target_std", torch.ones(3))

    def architecture_config(self):
        return self.config

    def set_normalizer_statistics(self, mean_input, mean_target, std_input, std_target):
        self.persistence_scale.copy_(std_input / std_target)
        self.persistence_bias.copy_((mean_input - mean_target) / std_target)
        self.target_std.copy_(std_target)

    def forward(self, history):
        # Global RNG is intentionally consumed so the resume assertion covers it.
        values = self.dropout(history)
        return self.linear(values)


@pytest.fixture(autouse=True)
def toy_model(monkeypatch):
    monkeypatch.setattr(subject, "_model_classes", lambda: (ToyConfig, ToyPC3D))


def _dataset():
    trajectory = torch.linspace(-1., 1., 40 * 40 * 40 * 3).reshape(40, 40, 40, 3)
    return subject.base.SimWindowDataset([trajectory], [subject.base.SimWindow(0, 0)])


def _normalizer():
    return subject.base.ChannelNormalizer(torch.zeros(3), torch.zeros(3),
                                           torch.ones(3), torch.ones(3), torch.device("cpu"))


def _setup_model():
    model = subject.construct_model(torch.device("cpu"), tiny=True)
    normalizer = _normalizer()
    model.set_normalizer_statistics(normalizer.mean_input, normalizer.mean_target,
                                    normalizer.std_input, normalizer.std_target)
    return model, normalizer


def test_stage_s_contract_is_fixed_single_arm_and_raw_endpoint() -> None:
    contract = subject.training_contract()
    assert (subject.SEED, subject.SCHEDULE_SEED) == (210, 220)
    assert (subject.UPDATES, subject.BATCH_SIZE) == (10_000, 8)
    assert (subject.LR, subject.WEIGHT_DECAY, subject.GRAD_CLIP) == (3e-4, 1e-5, 1.0)
    assert subject.DIAGNOSTIC_UPDATES == (2_000, 5_000, 10_000)
    assert (subject.CHECKPOINT_INTERVAL, subject.TRACE_INTERVAL) == (1_000, 250)
    assert subject.EXPECTED_PARAMETERS == 8_929_552
    assert contract["arm"] == "temporal_candidate"
    assert contract["interaction_mode"] == "causal_temporal_candidate"
    assert contract["endpoint_kind"] == "raw_model_u10000"
    assert not contract["diagnostics_select_checkpoint"] and not subject.EMA_ENABLED


def test_all100_window_contract_and_schedule_are_sealed() -> None:
    trajectories = [torch.empty(1000, 2, 2, 3) for _ in range(100)]
    windows = [subject.base.SimWindow(index, start)
               for index in range(100) for start in range(0, 960, 20)]
    assert subject.validate_all100_dataset(trajectories, windows)["windows"] == 4800
    schedule = subject.make_schedule(4800)
    assert schedule.shape == (10_000, 8) and schedule.dtype == np.int64
    assert subject.validate_formal_schedule(schedule) == subject.SAMPLE_INDEX_SHA256
    assert subject.schedule_seal(schedule) == subject.schedule_seal(subject.make_schedule(4800))
    with pytest.raises(RuntimeError, match="4800 windows"):
        subject.validate_all100_dataset(trajectories, windows[:-1])
    seal = subject.data_seal(Path("/tmp/sim"), [], trajectories, windows)
    assert seal["seal_kind"] == "semantic_inventory_and_window_contract_not_h5_content_sha256"


def test_atomic_checkpoint_and_resume_are_bitwise_exact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subject, "TRACE_INTERVAL", 2)
    dataset = _dataset()
    full_model, normalizer = _setup_model()
    full = subject.train(model=full_model, dataset=dataset, normalizer=normalizer,
                         device=torch.device("cpu"), updates=6, batch_size=1,
                         checkpoint_interval=2, checkpoint_dir=tmp_path / "full",
                         config_seal="cfg", data={"data": "seal"}, git={"git": "seal"})

    interrupted_model, normalizer = _setup_model()
    subject.train(model=interrupted_model, dataset=dataset, normalizer=normalizer,
                  device=torch.device("cpu"), updates=6, batch_size=1,
                  checkpoint_interval=3, checkpoint_dir=tmp_path / "resume",
                  config_seal="cfg", data={"data": "seal"}, git={"git": "seal"})
    checkpoint = tmp_path / "resume" / "checkpoint_u00000003.pt"
    resumed_model, normalizer = _setup_model()
    resumed = subject.train(model=resumed_model, dataset=dataset, normalizer=normalizer,
                            device=torch.device("cpu"), updates=6, batch_size=1,
                            checkpoint_interval=2, checkpoint_dir=tmp_path / "resume",
                            resume=checkpoint, config_seal="cfg",
                            data={"data": "seal"}, git={"git": "seal"})
    assert full["endpoint_state_sha256"] == resumed["endpoint_state_sha256"]
    assert [row["update"] for row in resumed["trace"]] == [2, 4, 6]
    assert full["trace"] == resumed["trace"]
    assert resumed["sample_cursor"] == 6
    payload = torch.load(tmp_path / "resume" / "checkpoint_u00000006.pt",
                         map_location="cpu", weights_only=False)
    assert payload["ema_state"] is None
    assert payload["model_config"] == {"e0_channels": 8}
    assert payload["interaction_mode"] == "causal_temporal_candidate"
    assert set(payload["normalizer_tensors"]) == {
        "mean_input", "mean_target", "std_input", "std_target"}
    assert payload["normalizer_sha256"] == subject.normalizer_sha256(normalizer)
    assert payload["sample_cursor"] == payload["next_update"] == 6
    assert {"python", "numpy", "torch_cpu", "cuda"} == set(payload["rng_state"])
    assert payload["trace"] == resumed["trace"]
    assert json.loads((tmp_path / "resume" / "trace.json").read_text()) == resumed["trace"]
    final = resumed["final_checkpoint"]
    assert final["path"].endswith("checkpoint_u00000006.pt")
    assert final["sha256"] == subject.file_sha256(Path(final["path"]))


def test_resume_rejects_changed_seals(tmp_path: Path) -> None:
    dataset = _dataset()
    model, normalizer = _setup_model()
    subject.train(model=model, dataset=dataset, normalizer=normalizer,
                  device=torch.device("cpu"), updates=2, batch_size=1,
                  checkpoint_interval=2, checkpoint_dir=tmp_path,
                  config_seal="cfg", data={"data": "seal"}, git={"git": "seal"})
    restored, normalizer = _setup_model()
    with pytest.raises(RuntimeError, match="data seal mismatch"):
        subject.train(model=restored, dataset=dataset, normalizer=normalizer,
                      device=torch.device("cpu"), updates=2, batch_size=1,
                      resume=tmp_path / "checkpoint_u00000002.pt",
                      config_seal="cfg", data={"data": "other"}, git={"git": "seal"})
    restored, changed_normalizer = _setup_model()
    changed_normalizer.mean_input.add_(1.0)
    with pytest.raises(RuntimeError, match="normalizer state seal mismatch"):
        subject.train(model=restored, dataset=dataset, normalizer=changed_normalizer,
                      device=torch.device("cpu"), updates=2, batch_size=1,
                      resume=tmp_path / "checkpoint_u00000002.pt",
                      config_seal="cfg", data={"data": "seal"}, git={"git": "seal"})


def test_diagnostics_are_only_the_frozen_updates_and_not_selection(monkeypatch) -> None:
    model, normalizer = _setup_model()
    dataset = _dataset()
    monkeypatch.setattr(subject, "DIAGNOSTIC_UPDATES", (2, 4))
    result = subject.train(model=model, dataset=dataset, normalizer=normalizer,
                           device=torch.device("cpu"), updates=4, batch_size=1,
                           checkpoint_interval=4)
    assert [row["update"] for row in result["diagnostics"]] == [2, 4]
    assert all(row["role"] == "fixed_fit_diagnostic_not_checkpoint_selection"
               for row in result["diagnostics"])
    assert result["checkpoint_selection"] == "none_raw_terminal_endpoint"


def test_formal_run_rejects_dirty_git_before_cuda(monkeypatch) -> None:
    monkeypatch.setattr(subject.base, "git_report", lambda: {
        "clean": False, "porcelain": [" M tracked.py"]})
    args = SimpleNamespace(execute=True, acknowledge=subject.ACKNOWLEDGEMENT)
    with pytest.raises(RuntimeError, match="requires clean Git"):
        subject.run(args, torch.device("cpu"))
