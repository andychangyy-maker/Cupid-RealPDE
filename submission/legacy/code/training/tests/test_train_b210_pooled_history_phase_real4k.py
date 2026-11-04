from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fp32_ema_shadow as ema_lib  # noqa: E402
from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
    POOLED_HISTORY_MODE,
)
import train_b210_pooled_history_phase_real4k as subject  # noqa: E402


def test_contract_is_one_fixed_real4k_arm_without_target_surface() -> None:
    contract = subject.experiment_contract()
    assert contract["single_arm"] == POOLED_HISTORY_MODE
    assert contract["architecture"]["learned_history_mixer"] is None
    assert contract["architecture"]["parameters"] == 34_827
    assert contract["comparison"]["updates"] == 4_000
    assert contract["comparison"]["scheduler_t_max"] == 12_000
    assert contract["comparison"]["sample_schedule_sha256"] == (
        "9381a97fc114e4234a889d686158533d0088701676b683d268d12d4203f1be43")
    assert contract["evaluation"]["clean716_argument"] is False
    assert contract["evaluation"]["score_gate"] is None
    with pytest.raises(SystemExit):
        subject.parse_args(("--clean716-manifest", "forbidden"))


def test_pooled_default_has_exact_parameter_and_common_init_contract() -> None:
    pooled = subject.build_adapter(torch.device("cpu"))
    assert sum(parameter.numel() for parameter in pooled.parameters()) == 34_827
    assert len(pooled.state_dict()) == 35
    assert not any(name.startswith("temporal_mixer.")
                   for name in pooled.state_dict())
    report = subject.common_initialization_report()
    assert report["bitwise_equal_to_causal_initialization"] is True
    assert report["common_tensors"] == 35
    assert report["removed_mixer_state_tensors"] == 25
    assert report["removed_mixer_parameters"] == 6_840


def test_compute_profile_matches_exact_mac_and_module_delta() -> None:
    report = subject.compute_report()
    assert report["reference_causal"]["conv_mac_per_sample"] == 411_648_000
    assert report["pooled_history_no_mixer"]["conv_mac_per_sample"] == 346_521_600
    assert report["removed"]["conv_mac_per_sample"] == 65_126_400
    assert report["removed"]["conv3d_module_invocations"] == 10
    assert report["removed"]["groupnorm_module_invocations"] == 5


def test_pooled_update_zero_projection_and_common_path_gradients() -> None:
    torch.manual_seed(29)
    adapter = CausalPhaseDirectionAdapter(CausalPhaseDirectionConfig(
        in_steps=5, out_steps=5, hidden_channels=8,
        spatial_blocks=1, spatial_downsample=2,
        temporal_blocks=2, temporal_kernel=3, groups=4,
        mixer_mode=POOLED_HISTORY_MODE)).train()
    history = torch.randn(2, 5, 8, 10, 3)
    base = torch.randn_like(history)
    target = torch.randn_like(base[..., :2])
    initial = adapter(history, base)
    assert torch.equal(initial, base)
    loss = (initial[..., :2] - target).square().mean()
    loss.backward()
    assert adapter.direction_head.weight.grad is not None
    assert torch.count_nonzero(adapter.direction_head.weight.grad)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    output = adapter(history, base)
    loss = (output[..., :2] - target).square().mean()
    loss.backward()
    for parameter in (
            adapter.frame_encoder.stem.weight,
            adapter.forecast_encoder.stem.weight,
            adapter.fusion[0].weight,
            adapter.gate_head.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad)
    base_mean = base[..., :2].mean(dim=1, keepdim=True)
    output_mean = output[..., :2].mean(dim=1, keepdim=True)
    base_rms = (base[..., :2] - base_mean).square().mean(
        dim=1, keepdim=True).sqrt()
    output_rms = (output[..., :2] - output_mean).square().mean(
        dim=1, keepdim=True).sqrt()
    torch.testing.assert_close(output_mean, base_mean, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(output_rms, base_rms, atol=2e-6, rtol=2e-6)
    assert torch.equal(output[..., 2:], base[..., 2:])


def test_target_free_panel_reads_only_the_selected_observed_slice(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject.real4k, "EXPECTED_OPTIMIZER_WINDOWS", 1)
    path = tmp_path / "case.h5"
    values = np.arange(40 * 8 * 10, dtype=np.float32).reshape(40, 8, 10)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("u", data=values)
        handle.create_dataset("v", data=-values)
        handle.create_dataset("p", data=np.full_like(values, 99.0))
    rows = [{"file": path.name, "val_boundary": 40}]
    history, access = subject._target_free_history_panel(
        tmp_path, rows, (0,))
    assert history.shape == (1, 20, 4, 5, 3)
    np.testing.assert_array_equal(history[0, ..., 0].numpy(), values[:20, ::2, ::2])
    np.testing.assert_array_equal(history[0, ..., 1].numpy(), -values[:20, ::2, ::2])
    assert torch.count_nonzero(history[..., 2]).item() == 0
    assert access == [{
        "dense_index": 0, "file": path.name,
        "frame_slice": [0, 20], "channels": ["u", "v"],
        "pressure_opened": False, "target_returned": False,
    }]


def test_endpoint_is_adapter_only_and_strictly_sealed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "UPDATES", 1)
    adapter = subject.build_adapter(torch.device("cpu"))
    ema = ema_lib.FP32EMAShadow(adapter, subject.EMA_DECAY)
    ema.update_after_optimizer_step(adapter, optimizer_step=1)
    raw = subject._state_snapshot(adapter)
    ema_state = ema.materialize_model_state(adapter)
    schedule_sha = "b" * 64
    payload = {
        "schema": subject.ENDPOINT_SCHEMA,
        "experiment_schema": subject.SCHEMA,
        "arm": subject.ARM,
        "update": 1,
        "adapter_parameters": subject.POOLED_ADAPTER_PARAMETERS,
        "adapter_config": adapter.architecture_config(),
        "raw_adapter_state": raw,
        "raw_adapter_state_sha256": ema_lib.tensor_state_sha256(raw),
        "ema_adapter_state": ema_state,
        "ema_adapter_state_sha256": ema_lib.tensor_state_sha256(ema_state),
        "ema": {key: value for key, value in ema.seal(adapter).items()
                if key != "model_state"},
        "base_checkpoint": {
            "sha256": "a" * 64, "model_state_sha256": "c" * 64},
        "normalizer": {
            "mean_input": torch.zeros(3),
            "mean_target": torch.zeros(3),
            "std_input": torch.ones(3),
            "std_target": torch.ones(3),
        },
        "sample_schedule_sha256": schedule_sha,
        "mechanics": {
            "initial_common_state_sha256": subject.common_initialization_report()[
                "common_state_sha256"],
            "update0_bitwise_identity": True,
            "direction_head_gradient_u1": True,
            "frame_history_gradient_u2": True,
            "forecast_gradient_u2": True,
            "direction_gate_gradient_u2": True,
            "temporal_mixer_absent": True,
            "temporal_mixer_state_keys": [],
        },
    }
    path = tmp_path / "pooled.pt"
    seal = subject._save_endpoint(
        path, payload, "a" * 64, schedule_sha)
    assert seal["strict_load"] is True
    assert seal["temporal_mixer_absent"] is True
    assert Path(seal["seal_path"]).is_file()
    stored = torch.load(path, map_location="cpu", weights_only=False)
    assert "base_state" not in stored
    assert "model_state_dict" not in stored
    assert not any(name.startswith("temporal_mixer.")
                   for name in stored["raw_adapter_state"])


def test_result_json_contract_is_serializable() -> None:
    json.dumps(subject.experiment_contract(), allow_nan=False)
