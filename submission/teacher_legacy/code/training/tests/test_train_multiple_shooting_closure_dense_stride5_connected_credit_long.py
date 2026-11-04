from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_multiple_shooting_closure_dense_stride5_connected_credit_long as subject  # noqa: E402


def test_recipe_is_the_dense_stride5_control_with_connected_credit_only():
    assert subject.TRAIN_STRIDE == subject.dense.TRAIN_STRIDE == 5
    assert subject.HELD_STRIDE == subject.dense.HELD_STRIDE == 20
    assert subject.control.FIT_AOAS == (0, 5, 15, 20)
    assert subject.control.HELD_AOA == 10
    assert subject.control.UPDATES == 12_000
    assert subject.control.BATCH_SIZE == 8
    assert subject.control.LR == 2.0e-4
    assert subject.control.WEIGHT_DECAY == 1.0e-5
    assert subject.control.GRAD_CLIP == 1.0
    assert subject.control.EMA_DECAY == 0.999
    assert subject.control.LOSS_WEIGHTS == (1.0, 4.0, 1.0)
    assert subject.control.MSE_WEIGHT == 0.25
    assert subject.dense.dense_windows is not None
    assert subject.dense.window_manifest is not None


def test_dense_control_reference_is_the_completed_run():
    reference = subject.DENSE_CONTROL_REFERENCE
    assert reference["sample_stream_sha256"] == (
        "118d296c3421ff75ec71ffd867aea35bf"
        "218622bf258477aadba9db820658cde")
    assert reference["raw"]["point_contribution_46_10_9"] == (
        60.40376082689495)
    assert reference["ema999"]["point_contribution_46_10_9"] == (
        60.40952461274668)


def test_dense_manifest_normalizer_and_sample_stream_match_completed_control():
    manifest = {
        "windows": 15_360,
        "stride": 5,
        "trajectory_start_sha256": (
            "f84a435bc9d49b39eb350eb99f590fd9"
            "9c119843efd7f199d478d577f3744be1"),
        "first": [0, 0],
        "last": [79, 955],
        "fit_inventory_sha256": (
            "c8dbbc06b61701d0b259a32c0d6a8451"
            "d4b5a0ae7c12966af27e097ccba4aab8"),
    }
    normalizer = subject.base.ChannelNormalizer(
        torch.tensor((0.8572806119918823, -0.006138242781162262,
                      1.4708149433135986)),
        torch.tensor((0.8571790456771851, -0.006042769644409418,
                      1.4449578523635864)),
        torch.tensor((0.44620218873023987, 0.21658000349998474,
                      14.079401969909668)),
        torch.tensor((0.4462870657444, 0.2166111022233963,
                      13.80954360961914)),
        torch.device("cpu"))
    identity = subject.dense_control_identity(
        manifest, normalizer, training_window_count=15_360)
    assert identity["all_matched"]
    assert identity["observed"] == identity["expected"]


def test_matched_model_is_bitwise_identical_except_connected_graph():
    model, identity = subject.construct_matched_candidate(
        torch.device("cpu"), tiny=True)
    assert model.rollout_mode == "carry_detached"
    assert model.segment_gradient_mode == "connected"
    assert identity["common_state_bitwise_identical"]
    assert identity["truncated_state_sha256"] == (
        identity["connected_state_sha256"])
    assert identity["only_architecture_config_difference"] == [
        "segment_gradient_mode"]


def test_training_delegates_to_connected_control_recipe(monkeypatch):
    model = subject.construct_model(torch.device("cpu"), tiny=True)
    sentinel = ({"sample_stream_sha256": "matched"}, object())
    observed = {}

    def fake_train(received_model, dataset, normalizer, device, **kwargs):
        observed.update({
            "gradient_mode": received_model.segment_gradient_mode,
            "dataset": dataset,
            "normalizer": normalizer,
            "device": device,
            "kwargs": kwargs,
        })
        return sentinel

    monkeypatch.setattr(subject.connected, "train", fake_train)
    result = subject.train(
        model, "dataset", "normalizer", torch.device("cpu"),
        updates=17, batch_size=3, trace_interval=5)
    assert result is sentinel
    assert observed == {
        "gradient_mode": "connected",
        "dataset": "dataset",
        "normalizer": "normalizer",
        "device": torch.device("cpu"),
        "kwargs": {"updates": 17, "batch_size": 3, "trace_interval": 5},
    }


def test_saved_endpoint_records_dense_manifest_and_connected_graph(
        tmp_path, monkeypatch):
    config = subject.control.model_config(tiny=True)
    monkeypatch.setattr(subject.control, "MODEL_CONFIG", config)
    model = subject.construct_model(torch.device("cpu"), tiny=True)
    normalizer = subject.base.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3),
        torch.device("cpu"))
    state = subject.control.snapshot(model)
    manifest = {"windows": 4, "stride": 5}
    path = tmp_path / "endpoint.pt"
    record = subject.save_endpoint(
        path, kind="test", state=state, git={}, normalizer=normalizer,
        updates=2, manifest=manifest)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert record["model_state_sha256"] == subject.base.state_sha(state)
    assert payload["schema"] == subject.SCHEMA
    assert payload["rollout_mode"] == "carry_detached"
    assert payload["segment_gradient_mode"] == "connected"
    assert payload["training_window_manifest"] == manifest
    rebuilt = subject.connected.MultipleShootingClosure(
        subject.connected.MultipleShootingClosureConfig(
            **payload["model_config"]),
        rollout_mode=payload["rollout_mode"],
        segment_gradient_mode=payload["segment_gradient_mode"])
    rebuilt.load_state_dict(payload["model_state"], strict=True)


def test_formal_requires_explicit_acknowledgement_before_any_cuda_work():
    args = subject.argparse.Namespace(
        execute=False, acknowledge="", sim_dir=None, output_dir=None)
    with pytest.raises(PermissionError, match="formal run requires"):
        subject.formal(args, torch.device("cpu"))
