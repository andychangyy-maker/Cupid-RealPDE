from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
for path in (CODE_ROOT, CODE_ROOT / "training", CODE_ROOT / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_b210_causal_phase_direction_clean716 as subject  # noqa: E402
import fp32_ema_shadow as ema_lib  # noqa: E402
from models.causal_phase_direction import (  # noqa: E402
    CausalPhaseDirectionAdapter,
    CausalPhaseDirectionConfig,
)
import train_b210_causal_phase_direction_real4k as phase_train  # noqa: E402
import train_b210_pooled_history_phase_real4k as pooled_train  # noqa: E402


class IdentityNormalizer:
    def __init__(self) -> None:
        self.mean_input = torch.zeros(3)
        self.mean_target = torch.zeros(3)
        self.std_input = torch.ones(3)
        self.std_target = torch.ones(3)


def formal_endpoint(path: Path, arm: str, base_sha: str) -> dict:
    adapter = CausalPhaseDirectionAdapter(
        CausalPhaseDirectionConfig(mixer_mode=arm))
    ema = ema_lib.FP32EMAShadow(adapter, phase_train.EMA_DECAY)
    # Produce a structurally valid formal seal without an expensive optimizer
    # loop; tensor values remain the update-zero state for this parser test.
    ema._updates = phase_train.UPDATES
    payload = phase_train.endpoint_payload(
        adapter=adapter, ema=ema, arm=arm,
        source={"sha256": base_sha},
        normalizer=IdentityNormalizer(),
        schedule_sha=pooled_train.EXPECTED_SAMPLE_SCHEDULE_SHA256,
        trace=[], elapsed_seconds=1.0,
        updates=phase_train.UPDATES)
    payload["mechanics"] = {
        "update0_bitwise_identity": True,
        "direction_head_gradient_u1": True,
        "history_encoder_gradient_u2": True,
        "direction_gate_gradient_u2": True,
        "initial_adapter_state_sha256": "c" * 64,
    }
    torch.save(payload, path)
    return payload


def formal_pooled_endpoint(path: Path, base_sha: str) -> dict:
    adapter = pooled_train.build_adapter(torch.device("cpu"))
    ema = ema_lib.FP32EMAShadow(adapter, pooled_train.EMA_DECAY)
    ema._updates = pooled_train.UPDATES
    raw = pooled_train._state_snapshot(adapter)
    ema_state = ema.materialize_model_state(adapter)
    common_sha = pooled_train.common_initialization_report()[
        "common_state_sha256"]
    payload = {
        "schema": pooled_train.ENDPOINT_SCHEMA,
        "experiment_schema": pooled_train.SCHEMA,
        "arm": pooled_train.ARM,
        "update": pooled_train.UPDATES,
        "adapter_parameters": pooled_train.POOLED_ADAPTER_PARAMETERS,
        "adapter_config": adapter.architecture_config(),
        "raw_adapter_state": raw,
        "raw_adapter_state_sha256": ema_lib.tensor_state_sha256(raw),
        "ema_adapter_state": ema_state,
        "ema_adapter_state_sha256": ema_lib.tensor_state_sha256(ema_state),
        "ema": {key: value for key, value in ema.seal(adapter).items()
                if key != "model_state"},
        "base_checkpoint": {
            "sha256": base_sha, "model_state_sha256": "e" * 64},
        "normalizer": {
            "mean_input": torch.zeros(3),
            "mean_target": torch.zeros(3),
            "std_input": torch.ones(3),
            "std_target": torch.ones(3),
        },
        "sample_schedule_sha256": (
            pooled_train.EXPECTED_SAMPLE_SCHEDULE_SHA256),
        "mechanics": {
            "initial_common_state_sha256": common_sha,
            "update0_bitwise_identity": True,
            "direction_head_gradient_u1": True,
            "frame_history_gradient_u2": True,
            "forecast_gradient_u2": True,
            "direction_gate_gradient_u2": True,
            "temporal_mixer_absent": True,
            "temporal_mixer_state_keys": [],
        },
    }
    pooled_train._save_endpoint(
        path, payload, base_sha,
        pooled_train.EXPECTED_SAMPLE_SCHEDULE_SHA256)
    return torch.load(path, map_location="cpu", weights_only=False)


def test_endpoint_barrier_validates_raw_and_ema_strict_load(tmp_path: Path) -> None:
    base_sha = "a" * 64
    path = tmp_path / "causal.pt"
    payload = formal_endpoint(path, "causal_temporal", base_sha)
    loaded, report = subject.validate_endpoint(
        path, phase_train.sha256(path), "causal_temporal", base_sha)
    assert loaded["raw_adapter_state_sha256"] == (
        payload["raw_adapter_state_sha256"])
    assert report["raw_state_sha256"] == payload["raw_adapter_state_sha256"]
    assert report["ema_state_sha256"] == payload["ema_adapter_state_sha256"]

    changed = torch.load(path, map_location="cpu", weights_only=False)
    name = next(iter(changed["raw_adapter_state"]))
    changed["raw_adapter_state"][name] = (
        changed["raw_adapter_state"][name].clone() + 1)
    changed["raw_adapter_state_sha256"] = ema_lib.tensor_state_sha256(
        changed["raw_adapter_state"])
    torch.save(changed, path)
    with pytest.raises(RuntimeError, match="does not bind"):
        subject.validate_endpoint(
            path, phase_train.sha256(path), "causal_temporal", base_sha)

    formal_endpoint(path, "causal_temporal", base_sha)
    changed = torch.load(path, map_location="cpu", weights_only=False)
    changed["raw_adapter_state"]["unexpected"] = torch.ones(1)
    changed["raw_adapter_state_sha256"] = ema_lib.tensor_state_sha256(
        changed["raw_adapter_state"])
    torch.save(changed, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        subject.validate_endpoint(
            path, phase_train.sha256(path), "causal_temporal", base_sha)


def test_pair_requires_same_stream_normalizer_and_initialization(tmp_path: Path) -> None:
    base_sha = "a" * 64
    control_path = tmp_path / "control.pt"
    causal_path = tmp_path / "causal.pt"
    control = formal_endpoint(
        control_path, "framewise_spatial_control", base_sha)
    causal = formal_endpoint(causal_path, "causal_temporal", base_sha)
    subject.validate_pair(control, causal)
    causal["mechanics"]["initial_adapter_state_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="did not start"):
        subject.validate_pair(control, causal)
    causal["mechanics"]["initial_adapter_state_sha256"] = "c" * 64
    causal["adapter_config"]["direction_limit"] = 0.5
    with pytest.raises(RuntimeError, match="outside mixer_mode"):
        subject.validate_pair(control, causal)


def test_pooled_endpoint_validates_topology_states_and_file_seal(
        tmp_path: Path) -> None:
    base_sha = "a" * 64
    path = tmp_path / "pooled.pt"
    payload = formal_pooled_endpoint(path, base_sha)
    loaded, report = subject.validate_pooled_endpoint(
        path, phase_train.sha256(path), base_sha)
    assert loaded["adapter_parameters"] == 34_827
    assert report["temporal_mixer_absent"] is True
    assert report["raw_state_sha256"] == payload["raw_adapter_state_sha256"]
    assert report["ema_state_sha256"] == payload["ema_adapter_state_sha256"]
    assert report["file_seal"]["path"].endswith("pooled.pt.seal.json")
    assert not any(name.startswith("temporal_mixer.")
                   for name in loaded["raw_adapter_state"])

    seal_path = path.with_suffix(path.suffix + ".seal.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["temporal_mixer_absent"] = False
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="seal changed"):
        subject.validate_pooled_endpoint(
            path, phase_train.sha256(path), base_sha)


def test_three_arm_fairness_uses_only_common_initialization_sha(
        tmp_path: Path) -> None:
    base_sha = "a" * 64
    control = formal_endpoint(
        tmp_path / "control.pt", "framewise_spatial_control", base_sha)
    causal = formal_endpoint(
        tmp_path / "causal.pt", "causal_temporal", base_sha)
    pooled = formal_pooled_endpoint(tmp_path / "pooled.pt", base_sha)
    report = subject.validate_three_arm_fairness(control, causal, pooled)
    assert report["comparison_basis"] == "shared_non_mixer_tensors_only"
    assert report["common_tensors"] == 35
    assert report["full_state_compared_across_topologies"] is False

    pooled["mechanics"]["initial_common_state_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="sealed common state"):
        subject.validate_three_arm_fairness(control, causal, pooled)


def test_invalid_pooled_endpoint_stops_before_any_manifest_or_target_access(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject, "validate_endpoint", lambda *_args: ({}, {}))

    def reject_pooled(*_args):
        raise ValueError("invalid pooled endpoint")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("manifest or target access crossed endpoint barrier")

    monkeypatch.setattr(subject, "validate_pooled_endpoint", reject_pooled)
    monkeypatch.setattr(subject.real4k, "load_clean_manifest", forbidden)
    monkeypatch.setattr(subject.real4k, "load_clean716_rows", forbidden)
    monkeypatch.setattr(subject.real4k, "load_real_rows", forbidden)
    args = SimpleNamespace(
        acknowledge_target_access=subject.ACKNOWLEDGEMENT,
        output_json=tmp_path / "result.json",
        base_checkpoint_sha256="a" * 64,
        spatial_control_endpoint=tmp_path / "control.pt",
        spatial_control_endpoint_sha256="b" * 64,
        causal_endpoint=tmp_path / "causal.pt",
        causal_endpoint_sha256="c" * 64,
        pooled_endpoint=tmp_path / "pooled.pt",
        pooled_endpoint_sha256="d" * 64,
        base_checkpoint=tmp_path / "base.pt",
        real_dir=tmp_path,
        clean_manifest=tmp_path / "clean.json",
        clean716_manifest=tmp_path / "clean716.json",
        device="cpu",
    )
    with pytest.raises(ValueError, match="invalid pooled endpoint"):
        subject.run(args)


def test_one_replay_reports_all_three_arms_raw_and_ema() -> None:
    class PassthroughNormalizer:
        std_target = torch.ones(3)

        @staticmethod
        def preprocess(history, target):
            return history, target

        @staticmethod
        def postprocess_prediction(prediction):
            return prediction.clone()

    class Base:
        @staticmethod
        def forward_distribution(history):
            extent = torch.full_like(history[..., :2], 0.1)
            return history.clone(), extent, extent

    class Adapter:
        @staticmethod
        def __call__(_history, base_prediction):
            return base_prediction.clone()

    torch.manual_seed(41)
    history = torch.randn(20, 8, 8, 3)
    history[..., 2] = 0.0
    adapters = {
        name: Adapter() for name in subject.MODEL_NAMES if name != "b210"}
    result = subject.evaluate(
        Base(), adapters, [(history, history.clone())], [0], ["case"],
        PassthroughNormalizer(), torch.device("cpu"))
    assert tuple(result["pooled"]) == subject.MODEL_NAMES
    assert tuple(result["prediction_sha256"]) == subject.MODEL_NAMES
    assert result["prediction_sha256"]["pooled_ema"] == (
        result["prediction_sha256"]["causal_ema"])
    assert "pooled_minus_control_ema" in result["comparisons"]
    assert "pooled_minus_causal_ema" in result["comparisons"]


def test_score_decoration_uses_published_weights() -> None:
    metrics = {
        "rel_l2_score": 94.0,
        "tke_rel_l2_score": 79.0,
        "mvpe_rel_l2_score": 93.0,
        "native_sps_score": 43.0,
        "point_score": 0.0,
    }
    result = subject._decorate(metrics)
    expected_point = 0.46 * 94.0 + 0.10 * 79.0 + 0.09 * 93.0
    assert result["official_point_contribution_46_10_9"] == pytest.approx(
        expected_point)
    assert result["official_normalized_point_46_10_9"] == pytest.approx(
        expected_point / 0.65)
    assert result["non_time_contribution_46_10_9_25"] == pytest.approx(
        expected_point + 0.25 * 43.0)
