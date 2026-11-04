from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


CODE_ROOT = Path(__file__).resolve().parents[2]
for path in (CODE_ROOT, CODE_ROOT / "training", CODE_ROOT / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_b210_causal_phase_direction_two_stage_kd_clean716 as subject  # noqa: E402
import fp32_ema_shadow as ema_lib  # noqa: E402
import train_b210_causal_phase_direction_real4k as direct_phase  # noqa: E402
import train_b210_causal_phase_direction_two_stage_kd as stage_train  # noqa: E402
import train_pyramid_causal3d_real_dense_aug as real4k  # noqa: E402


class IdentityNormalizer:
    def __init__(self) -> None:
        self.mean_input = torch.zeros(3)
        self.mean_target = torch.zeros(3)
        self.std_input = torch.ones(3)
        self.std_target = torch.ones(3)

    def preprocess(self, history, target):
        return history, target

    def postprocess_prediction(self, prediction):
        return prediction


class TinyBase(nn.Module):
    def forward_distribution(self, history: torch.Tensor):
        prediction = history.mul(0.25)
        extent = torch.full_like(prediction[..., :2], 0.5)
        return prediction, extent, extent


class TinyAdapter(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, history: torch.Tensor, base: torch.Tensor):
        wave = torch.linspace(
            -1.0, 1.0, base.shape[1], device=base.device,
            dtype=base.dtype).view(1, -1, 1, 1)
        result = base.clone()
        result[..., 0] = result[..., 0] + self.scale * wave
        return result


class TinyDataset:
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(71)
        self.samples = []
        for _ in range(2):
            history = torch.randn(20, 8, 8, 3, generator=generator)
            target = torch.randn(20, 8, 8, 3, generator=generator)
            history[..., 2] = 0.0
            target[..., 2] = 0.0
            self.samples.append((history, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _common_contract(
        base_sha: str, manifest_sha: str, kd_initial_sha: str,
        normalizer: IdentityNormalizer) -> tuple[dict, dict]:
    schedule = real4k.make_sample_schedule(
        stage_train.REAL_HISTORIES, stage_train.STAGE2_UPDATES,
        stage_train.BATCH_SIZE)
    schedule_sha = real4k.schedule_sha(schedule)
    probes = {
        str(update): {
            "indices_sha256": stage_train._canonical_sha(
                [int(index) for index in schedule[update - 1]]),
            "augmented_history_sha256": "a" * 64,
            "rng_after_augmentation_sha256": "b" * 64,
        }
        for update in stage_train.AUGMENTATION_PROBE_UPDATES
    }
    normalizer_sha = stage_train._normalizer_sha(normalizer)
    source = {
        "base": {
            "sha256": base_sha,
            "model_state_sha256": "c" * 64,
            "normalizer": {"sha256": normalizer_sha},
        },
        "stage1": {
            "path": "/sealed/stage1.pt",
            "file_sha256": "8" * 64,
            "endpoint": f"raw_u{stage_train.STAGE1_UPDATES}",
            "update": stage_train.STAGE1_UPDATES,
            "raw_adapter_state_sha256": kd_initial_sha,
        },
        "clean_manifest": {"file_sha256": manifest_sha},
        "optimizer_windows": stage_train.REAL_HISTORIES,
        "optimizer_stride": real4k.OPTIMIZER_STRIDE,
        "sample_schedule_sha256": schedule_sha,
    }
    training = {
        "updates": stage_train.STAGE2_UPDATES,
        "batch_size": stage_train.BATCH_SIZE,
        "seed": stage_train.SEED,
        "sample_schedule_sha256": schedule_sha,
        "augmentation": dict(real4k.AUGMENTATION),
        "objective": "physical_uv_official_Rel-L2_mapping_only",
        "optimizer": "AdamW",
        "lr": stage_train.LR,
        "weight_decay": stage_train.WEIGHT_DECAY,
        "grad_clip": stage_train.GRAD_CLIP,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": stage_train.SCHEDULER_T_MAX,
        "ema_decay": stage_train.EMA_DECAY,
        "ema_initialized_from_real_u0": True,
        "trace": [],
        "elapsed_seconds": 1.0,
    }
    mechanics = {
        "rng_u0_sha256": "d" * 64,
        "augmentation_probes": probes,
        "direction_head_gradient_u1": True,
        "history_encoder_gradient_u2": True,
    }
    return source, {"training": training, "mechanics": mechanics}


def _write_endpoint(
        path: Path, arm: str, source: dict, common: dict,
        normalizer: IdentityNormalizer, initial_sha: str,
        ) -> tuple[dict, dict]:
    adapter = direct_phase.build_adapter("causal_temporal", torch.device("cpu"))
    if arm == stage_train.STAGE2_ARMS[1]:
        with torch.no_grad():
            adapter.direction_head.bias.add_(0.05)
    ema = ema_lib.FP32EMAShadow(adapter, stage_train.EMA_DECAY)
    ema._updates = stage_train.STAGE2_UPDATES
    raw = stage_train._state_snapshot(adapter)
    ema_state = ema.materialize_model_state(adapter)
    ema_seal = ema.seal(adapter)
    mechanics = {
        **common["mechanics"],
        "initial_adapter_state_sha256": initial_sha,
        "ema_u0_state_sha256": initial_sha,
        "prediction_bitwise_B210_at_real_u0": (
            arm == stage_train.STAGE2_ARMS[0]),
    }
    payload = {
        "schema": stage_train.STAGE2_ENDPOINT_SCHEMA,
        "experiment_schema": stage_train.SCHEMA,
        "stage": "stage2_matched_real4k",
        "arm": arm,
        "endpoint": "fixed_raw_and_ema_u4000",
        "update": stage_train.STAGE2_UPDATES,
        "adapter_config": adapter.architecture_config(),
        "adapter_parameters": stage_train.EXPECTED_ADAPTER_PARAMETERS,
        "raw_adapter_state": raw,
        "raw_adapter_state_sha256": ema_lib.tensor_state_sha256(raw),
        "ema_adapter_state": ema_state,
        "ema_adapter_state_sha256": ema_lib.tensor_state_sha256(ema_state),
        "ema": {key: value for key, value in ema_seal.items()
                if key != "model_state"},
        "normalizer": stage_train._normalizer_tensors(normalizer),
        "normalizer_sha256": stage_train._normalizer_sha(normalizer),
        "source": source,
        "training": common["training"],
        "mechanics": mechanics,
        "reported_states": ["raw_u4000", "ema_u4000"],
        "checkpoint_selected": False,
        "score_gate": None,
        "clean716_opened": False,
        "submission_authorized": False,
    }
    seal = stage_train.write_endpoint_and_seal(
        path, payload, ("raw_adapter_state", "ema_adapter_state"))
    return payload, seal


def _write_pair(tmp_path: Path, manifest_sha: str = "e" * 64):
    normalizer = IdentityNormalizer()
    base_sha = "f" * 64
    zero = direct_phase.build_adapter("causal_temporal", torch.device("cpu"))
    zero_sha = ema_lib.tensor_state_sha256(zero.state_dict())
    kd_initial_sha = "9" * 64
    source, common = _common_contract(
        base_sha, manifest_sha, kd_initial_sha, normalizer)
    direct, direct_seal = _write_endpoint(
        tmp_path / "direct.pt", stage_train.STAGE2_ARMS[0],
        source, common, normalizer, zero_sha)
    kd, kd_seal = _write_endpoint(
        tmp_path / "kd.pt", stage_train.STAGE2_ARMS[1],
        source, common, normalizer, kd_initial_sha)
    return {
        "normalizer": normalizer,
        "base_sha": base_sha,
        "manifest_sha": manifest_sha,
        "direct": direct,
        "direct_seal": direct_seal,
        "kd": kd,
        "kd_seal": kd_seal,
    }


def test_stage2_pair_raw_ema_and_only_initial_difference_validate(
        tmp_path: Path) -> None:
    pair = _write_pair(tmp_path)
    direct, direct_report = subject.validate_endpoint(
        tmp_path / "direct.pt", pair["direct_seal"]["file_sha256"],
        stage_train.STAGE2_ARMS[0])
    kd, kd_report = subject.validate_endpoint(
        tmp_path / "kd.pt", pair["kd_seal"]["file_sha256"],
        stage_train.STAGE2_ARMS[1])
    report = subject.validate_pair(
        direct, kd, expected_base_sha256=pair["base_sha"],
        expected_clean_manifest_sha256=pair["manifest_sha"])
    assert direct_report["strict_raw_ema_load"] is True
    assert kd_report["strict_raw_ema_load"] is True
    assert report["only_initial_adapter_state_differs"] is True
    assert report["augmentation_probes_bitwise_equal"] is True


def test_endpoint_companion_tamper_fails_closed(tmp_path: Path) -> None:
    pair = _write_pair(tmp_path)
    seal_path = Path(pair["direct_seal"]["seal_path"])
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["arm"] = "tampered"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="companion seal is invalid"):
        subject.validate_endpoint(
            tmp_path / "direct.pt", pair["direct_seal"]["file_sha256"],
            stage_train.STAGE2_ARMS[0])


def test_pair_contract_tamper_fails_before_clean716(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clean_manifest = tmp_path / "clean81.json"
    clean_manifest.write_text("[]\n", encoding="utf-8")
    manifest_sha = direct_phase.sha256(clean_manifest)
    pair = _write_pair(tmp_path, manifest_sha)
    kd = pair["kd"]
    kd["training"] = {
        **kd["training"],
        "augmentation": {**kd["training"]["augmentation"], "gain_std": 99.0},
    }
    pair["kd_seal"] = stage_train.write_endpoint_and_seal(
        tmp_path / "kd.pt", kd,
        ("raw_adapter_state", "ema_adapter_state"))
    clean716_called = False
    base_called = False

    def forbidden_clean716(*_args, **_kwargs):
        nonlocal clean716_called
        clean716_called = True
        raise AssertionError("clean716 was touched")

    def forbidden_base(*_args, **_kwargs):
        nonlocal base_called
        base_called = True
        raise AssertionError("base load should follow pair validation")

    monkeypatch.setattr(real4k, "load_clean716_rows", forbidden_clean716)
    monkeypatch.setattr(stage_train, "load_b210_u3200", forbidden_base)
    args = argparse.Namespace(
        acknowledge_target_access=subject.ACKNOWLEDGEMENT,
        output_json=tmp_path / "result.json",
        base_checkpoint=tmp_path / "base.pt",
        base_checkpoint_sha256=pair["base_sha"],
        direct_endpoint=tmp_path / "direct.pt",
        direct_endpoint_sha256=pair["direct_seal"]["file_sha256"],
        kd_endpoint=tmp_path / "kd.pt",
        kd_endpoint_sha256=pair["kd_seal"]["file_sha256"],
        real_dir=tmp_path,
        clean_manifest=clean_manifest,
        clean716_manifest=tmp_path / "clean716.json",
        clean716_manifest_sha256="1" * 64,
    )
    with pytest.raises(RuntimeError, match="training contract changed"):
        subject.run(args)
    assert clean716_called is False
    assert base_called is False


def test_channel_normalizer_is_converted_for_existing_phase_evaluator() -> None:
    values = {
        "mean_input": torch.tensor((0.1, -0.2, 0.0)),
        "mean_target": torch.tensor((0.3, -0.4, 0.0)),
        "std_input": torch.tensor((2.0, 3.0, 1.0)),
        "std_target": torch.tensor((4.0, 5.0, 1.0)),
    }
    channel_normalizer = stage_train._normalizer_from_payload(
        values, torch.device("cpu"))
    assert not hasattr(channel_normalizer, "postprocess_prediction")
    evaluator_normalizer = subject.evaluation_normalizer(
        {"normalizer": values}, channel_normalizer, torch.device("cpu"))
    restored = evaluator_normalizer.postprocess_prediction(
        torch.zeros(1, 20, 2, 2, 3))
    torch.testing.assert_close(
        restored, values["mean_target"].view(1, 1, 1, 1, 3).expand_as(restored))


def test_base_manifest_check_reloads_full_stage1_lineage(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clean_manifest = tmp_path / "clean81.json"
    clean_manifest.write_text("fixed clean81", encoding="utf-8")
    manifest_sha = direct_phase.sha256(clean_manifest)
    pair = _write_pair(tmp_path, manifest_sha)
    calls = []

    def load_stage1(path, sha, normalizer, base_report, clean_sha):
        calls.append((path, sha, normalizer, base_report, clean_sha))
        return {}, {"raw_adapter_state_sha256": "9" * 64}

    monkeypatch.setattr(stage_train, "load_stage1_endpoint", load_stage1)
    monkeypatch.setattr(
        real4k, "load_clean_manifest", lambda _path: ([], {"rows": 81}))
    rows, _manifest, stage1 = subject.validate_base_and_fit_manifest(
        pair["direct"], pair["kd"],
        base_report={"model_state_sha256": "c" * 64},
        normalizer=pair["normalizer"], clean_manifest_path=clean_manifest)
    assert rows == []
    assert stage1["raw_adapter_state_sha256"] == "9" * 64
    assert calls[0][0] == Path("/sealed/stage1.pt")
    assert calls[0][1] == "8" * 64
    assert calls[0][4] == manifest_sha


def test_cpu_metric_replay_reports_all_fixed_states_without_selection() -> None:
    adapters = {
        "direct_raw": TinyAdapter(0.00),
        "direct_ema": TinyAdapter(0.01),
        "kd_raw": TinyAdapter(0.02),
        "kd_ema": TinyAdapter(0.03),
    }
    report = subject.evaluate(
        TinyBase(), adapters, TinyDataset(), (0, 1), ("a", "b"),
        IdentityNormalizer(), torch.device("cpu"))
    assert set(report["pooled"]) == set(subject.PUBLIC_MODEL_NAMES)
    assert "kd_minus_direct_raw" in report["comparisons"]
    assert "kd_minus_direct_ema" in report["comparisons"]
    for metrics in report["pooled"].values():
        for name in (
                "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
                "native_sps_score", "non_time_contribution_46_10_9_25"):
            assert name in metrics
            assert torch.isfinite(torch.tensor(metrics[name]))
    assert report["metric_implementation"] == subject.phase_eval.SCHEMA
