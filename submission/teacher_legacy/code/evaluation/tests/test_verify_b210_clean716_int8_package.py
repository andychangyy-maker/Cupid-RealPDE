from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import pytest
import torch


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = EVALUATION_ROOT.parent
TRAINING_ROOT = CODE_ROOT / "training"
SUBMISSIONS_ROOT = CODE_ROOT / "submissions"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT, SUBMISSIONS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import verify_b210_clean716_int8_package as subject  # noqa: E402
import track1_ensemble_unet2d as release_wrapper  # noqa: E402


def _normalizer() -> dict[str, list[float]]:
    return {
        "mean_input": [0.0, 0.0, 0.0],
        "mean_target": [0.0, 0.0, 0.0],
        "std_input": [1.0, 1.0, 1.0],
        "std_target": [1.0, 1.0, 1.0],
    }


def _metric_state(samples: int = 2, error: float = 0.2, sps_ratio: float = 1.0) -> dict:
    return {
        "samples": samples,
        "rel_error_sum": error,
        "tke_error_sum": error,
        "mvpe_error_sum": error,
        "sps_numerator": float(samples) * sps_ratio,
        "sps_scored": samples,
        "sps_covered": samples,
        "sps_width_sum": 0.0,
    }


def _valid_gate_and_folds() -> tuple[dict, list[dict]]:
    counts = [144, 143, 143, 143, 143]
    baseline_folds = [
        _metric_state(samples=count, error=0.1 * count, sps_ratio=0.500)
        for count in counts
    ]
    candidate_folds = [
        _metric_state(samples=count, error=0.1 * count, sps_ratio=0.502)
        for count in counts
    ]
    baseline_aoas = {
        aoa: _metric_state(samples=count, error=0.1 * count, sps_ratio=0.500)
        for aoa, count in zip(subject.dr1.AOA_VALUES, counts, strict=True)
    }
    candidate_aoas = {
        aoa: _metric_state(samples=count, error=0.1 * count, sps_ratio=0.502)
        for aoa, count in zip(subject.dr1.AOA_VALUES, counts, strict=True)
    }
    gate = subject.uq_release.promotion_gate(
        baseline_folds, candidate_folds, baseline_aoas, candidate_aoas
    )
    folds = []
    for index, (baseline, candidate) in enumerate(
        zip(baseline_folds, candidate_folds, strict=True)
    ):
        folds.append(
            {
                "fold": index,
                "baseline": {
                    "metrics": subject.smu1.finalize_metric_state(baseline),
                    "metric_state": baseline,
                },
                "candidate": {
                    "metrics": subject.smu1.finalize_metric_state(candidate),
                    "metric_state": candidate,
                },
                "point_metric_state_bitwise": True,
            }
        )
    assert gate["pass"] is True
    return gate, folds


def _uq_summary(checkpoint: Path, checkpoint_sha: str, state_sha: str) -> dict:
    frozen_sha = "7" * 64
    gate, folds = _valid_gate_and_folds()
    return {
        "schema": subject.uq_release.SCHEMA,
        "experiment": subject.uq_release.EXPERIMENT,
        "contract_sha256": subject.uq_release.CONTRACT_SHA256,
        "manifest": {
            "manifest_file_sha256": subject.uq_release.FORMAL_MANIFEST_FILE_SHA256,
            "clean2544": {
                "semantic_sha256": subject.uq_release.FORMAL_MANIFEST_SEMANTIC_SHA256,
                "trajectories": subject.uq_release.EXPECTED_TRAJECTORIES,
            },
            "clean716": {
                "semantic_sha256": "8" * 64,
                "tail_windows": subject.uq_release.EXPECTED_TAIL_WINDOWS,
                "excluded": subject.dr1.EXCLUDED_BASENAME,
            },
        },
        "tail_access": {
            "allowlist_exact": True,
            "prefix_frames_read": 0,
            "excluded_opened": False,
            "opened_file_allowlist": [f"row-{index}.h5" for index in range(81)],
        },
        "point": {
            "checkpoint_sha256": "a" * 64,
            "model_state_sha256": "b" * 64,
            "frozen_state_sha256": frozen_sha,
            "normalizer": {
                "values": _normalizer(),
                "sha256": subject.dr1.canonical_sha256(_normalizer()),
                "flat": True,
            },
        },
        "folds": folds,
        "promotion_gate": gate,
        "refit_training": {
            "updates": subject.uq_release.REFIT_UPDATES,
            "frozen_state_bitwise": True,
            "frozen_state_sha256_before": frozen_sha,
            "frozen_state_sha256_after": frozen_sha,
        },
        "refit_endpoint": {
            "path": str(checkpoint),
            "file_sha256": checkpoint_sha,
            "model_state_sha256": state_sha,
            "schema": subject.uq_release.REFIT_ENDPOINT_SCHEMA,
            "stage": subject.uq_release.REFIT_STAGE,
            "iteration": subject.uq_release.REFIT_UPDATES,
            "exact_verifier_required": True,
            "zip_written": False,
            "submission_authorized": False,
        },
        "decision": "PASS_B210_SMALL4K_CLEAN716_UQ_OOF_PROMOTION",
        "point_gain_attribution": False,
        "uq_stage_checkpoint_written": True,
        "exact_verifier_required": True,
        "zip_written": False,
        "submission_authorized": False,
    }


def _refit_payload(state: dict[str, torch.Tensor]) -> dict:
    normalizer = _normalizer()
    state_sha = subject.dr1.model_state_sha256(state)
    gate, _folds = _valid_gate_and_folds()
    frozen_sha = subject.dr1.model_state_sha256(
        {
            key: value
            for key, value in state.items()
            if not key.startswith(subject.uq_release.UQ_PREFIXES)
        }
    )
    training = {
        "updates": subject.uq_release.REFIT_UPDATES,
        "frozen_state_bitwise": True,
        "frozen_state_sha256_before": frozen_sha,
        "frozen_state_sha256_after": frozen_sha,
    }
    return {
        "schema": subject.uq_release.REFIT_ENDPOINT_SCHEMA,
        "experiment": subject.uq_release.EXPERIMENT,
        "contract_sha256": subject.uq_release.CONTRACT_SHA256,
        "stage": subject.uq_release.REFIT_STAGE,
        "phase": "real",
        "iteration": subject.uq_release.REFIT_UPDATES,
        "endpoint": "fixed_terminal_all716_u800",
        "primary": True,
        "model_config": {"architecture": "ensemble_unet"},
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "point_checkpoint_sha256": "a" * 64,
        "point_model_state_sha256": "b" * 64,
        "normalizer": normalizer,
        "normalizer_sha256": subject.dr1.canonical_sha256(normalizer),
        "training": training,
        "oof_promotion_gate": gate,
        "optimizer_state_saved": False,
        "point_gain_attribution": False,
        "exact_verifier_required": True,
        "zip_written": False,
        "submission_authorized": False,
    }


def _quantize(value: torch.Tensor) -> dict[str, torch.Tensor]:
    value = value.detach().cpu().float()
    dimensions = tuple(range(1, value.ndim))
    scale = (value.abs().amax(dim=dimensions, keepdim=True) / 127.0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    quantized = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    return {"quantized": quantized, "scale": scale}


def _int8_payload(direct: dict) -> dict:
    stored = {}
    for key, value in direct["model_state_dict"].items():
        if (
            value.is_floating_point()
            and value.ndim >= 2
            and value.numel() >= subject.QUANTIZE_MIN_ELEMENTS
        ):
            stored[key] = _quantize(value)
        elif value.is_floating_point():
            stored[key] = value.to(torch.float16)
        else:
            stored[key] = value
    return {
        "phase": "real",
        "model_config": direct["model_config"],
        "quantized_state_dict": stored,
        "normalizer": direct["normalizer"],
        "inference_dtype": "float16",
        "storage_format": subject.STORAGE_FORMAT,
    }


def _write_archive(
    path: Path, payload: dict, changed_member: str | None = None
) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    with zipfile.ZipFile(path, "w") as archive:
        for name in subject.EXPECTED_MEMBERS:
            if name == "ensemble_unet2d.pth":
                value = buffer.getvalue()
            else:
                value = subject.TRUSTED_MEMBER_SOURCES[name].read_bytes()
            if name == changed_member:
                value += b"\n# changed\n"
            archive.writestr(name, value)


def _worker(kind: str, state: dict, audit_overrides: dict | None = None) -> dict:
    audit = {
        "ordered_raw_input_sha256": "1" * 64,
        "ordered_raw_input_after_sha256": "1" * 64,
        "input_bitwise_unchanged": True,
        "ordered_raw_target_sha256": "2" * 64,
        "ordered_scored_mask_sha256": "3" * 64,
        "samples": 716,
        "scored_elements": 99,
        "input_pressure_nonzero": 0,
        "output_pressure_nonzero": 0,
        "batches": 45,
    }
    audit.update(audit_overrides or {})
    return {
        "schema": subject.WORKER_SCHEMA,
        "experiment": subject.EXPERIMENT,
        "kind": kind,
        "fresh_process": True,
        "data": {
            "manifest_file_sha256": subject.uq_release.FORMAL_MANIFEST_FILE_SHA256,
            "manifest_semantic_sha256": (
                subject.uq_release.FORMAL_MANIFEST_SEMANTIC_SHA256
            ),
            "clean716_semantic_sha256": "4" * 64,
            "ordered_key_sha256": "5" * 64,
        },
        "metric_state": state,
        "replay_audit": audit,
    }


def test_contract_is_clean716_packaging_only_and_forbids_submission():
    assert subject.CONTRACT["data"]["trajectories"] == 81
    assert subject.CONTRACT["data"]["tail_windows"] == 716
    assert subject.CONTRACT["data"]["excluded"] == "7575_0.h5"
    assert subject.CONTRACT["data"]["legacy_725_evaluator_forbidden"] is True
    assert subject.CONTRACT["submission_authorized"] is False
    assert subject.CONTRACT["package"]["quantize_min_elements"] == 1024
    assert "diagnostic only" in subject.CONTRACT["sources"]["cross_run_uq_point_sums"]
    assert "not generalization" in subject.CONTRACT["evidence_class"]
    assert subject.PACKAGING_FLOORS == {
        "rel_l2_score": -0.03,
        "tke_rel_l2_score": -0.05,
        "mvpe_rel_l2_score": -0.03,
        "native_sps_score": -0.10,
        "non_time_delta": -0.05,
    }
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "evaluate_track1_submission_archive" not in source


def test_uq_summary_binds_release_sha_and_refit_seal(tmp_path):
    checkpoint = tmp_path / "refit.pth"
    checkpoint.write_bytes(b"refit")
    summary = _uq_summary(checkpoint, subject.dr1.sha256(checkpoint), "6" * 64)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    loaded, source = subject.validate_uq_result(path, subject.dr1.sha256(path))
    assert loaded["manifest"]["clean716"]["tail_windows"] == 716
    assert source["refit_endpoint"]["file_sha256"] == subject.dr1.sha256(checkpoint)

    summary["manifest"]["clean716"]["tail_windows"] = 725
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest evidence"):
        subject.validate_uq_result(path, subject.dr1.sha256(path))


def test_refit_checkpoint_is_strict_and_flat_normalizer(tmp_path, monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.energy_controller = torch.nn.Linear(1, 1)

        def set_normalizer_statistics(self, *_values):
            pass

    model = TinyModel(None)
    payload = _refit_payload(model.state_dict())
    path = tmp_path / "refit.pth"
    torch.save(payload, path)
    seal = {
        "file_sha256": subject.dr1.sha256(path),
        "model_state_sha256": payload["model_state_sha256"],
        "point_checkpoint_sha256": payload["point_checkpoint_sha256"],
        "point_model_state_sha256": payload["point_model_state_sha256"],
        "point_frozen_state_sha256": payload["training"]["frozen_state_sha256_before"],
        "point_normalizer_sha256": payload["normalizer_sha256"],
        "refit_training_sha256": subject.dr1.canonical_sha256(payload["training"]),
        "promotion_gate_sha256": subject.dr1.canonical_sha256(
            payload["oof_promotion_gate"]
        ),
    }
    monkeypatch.setattr(subject, "EnsembleUNet2d", TinyModel)
    monkeypatch.setattr(subject.ft12, "validate_b210_model_config", lambda _x: None)
    monkeypatch.setattr(subject.uq_release, "EXPECTED_PARAMETERS", 6)
    monkeypatch.setattr(subject.uq_release, "EXPECTED_CONTROLLER_PARAMETERS", 2)
    _payload, report, _normalizer_value, loaded = subject.validate_refit_checkpoint(
        path, seal, torch.device("cpu")
    )
    assert report["strict_load"] is True
    assert loaded is not None
    assert report["normalizer"]["flat"] is True

    tampered_frozen = dict(payload)
    tampered_frozen["training"] = {
        **payload["training"],
        "frozen_state_sha256_before": "f" * 64,
        "frozen_state_sha256_after": "f" * 64,
    }
    tampered_frozen_path = tmp_path / "tampered_frozen.pth"
    torch.save(tampered_frozen, tampered_frozen_path)
    with pytest.raises(RuntimeError, match="non-UQ state"):
        subject.validate_refit_checkpoint(
            tampered_frozen_path,
            {
                **seal,
                "file_sha256": subject.dr1.sha256(tampered_frozen_path),
                "point_frozen_state_sha256": "f" * 64,
                "refit_training_sha256": subject.dr1.canonical_sha256(
                    tampered_frozen["training"]
                ),
            },
        )

    with pytest.raises(RuntimeError, match="normalizer is not bound"):
        subject.validate_refit_checkpoint(
            path,
            {**seal, "point_normalizer_sha256": "f" * 64},
        )

    nested = _refit_payload(model.state_dict())
    nested["normalizer"] = {"values": _normalizer()}
    broken = tmp_path / "nested.pth"
    torch.save(nested, broken)
    with pytest.raises(ValueError, match="flat schema"):
        subject.validate_refit_checkpoint(
            broken,
            {
                "file_sha256": subject.dr1.sha256(broken),
                "model_state_sha256": nested["model_state_sha256"],
                "point_checkpoint_sha256": nested["point_checkpoint_sha256"],
                "point_model_state_sha256": nested["point_model_state_sha256"],
                "point_frozen_state_sha256": nested["training"][
                    "frozen_state_sha256_before"
                ],
                "point_normalizer_sha256": nested["normalizer_sha256"],
                "refit_training_sha256": subject.dr1.canonical_sha256(
                    nested["training"]
                ),
                "promotion_gate_sha256": subject.dr1.canonical_sha256(
                    nested["oof_promotion_gate"]
                ),
            },
        )


def test_int8_payload_is_exactly_bound_to_raw_state(monkeypatch):
    direct = _refit_payload(
        {
            "weight": torch.linspace(-2.0, 2.0, 1024).reshape(32, 32),
            "persistence_scale": torch.ones(3),
            "persistence_bias": torch.zeros(3),
            "target_std": torch.ones(3),
        }
    )
    payload = _int8_payload(direct)
    monkeypatch.setattr(subject.ft12, "validate_b210_model_config", lambda _x: None)
    report = subject.validate_int8_payload(payload, direct)
    assert report["quantized_float_fraction"] == pytest.approx(1024 / 1033)
    assert report["per_dim0_error_bound"] is True

    extra = dict(payload)
    extra["unbound"] = True
    with pytest.raises(RuntimeError, match="payload contract"):
        subject.validate_int8_payload(extra, direct)

    tampered = _int8_payload(direct)
    tampered["quantized_state_dict"]["weight"]["quantized"][0, 0] += 1
    with pytest.raises(RuntimeError, match="not bound"):
        subject.validate_int8_payload(tampered, direct)

    inconsistent = _refit_payload(
        {
            **direct["model_state_dict"],
            "persistence_bias": torch.ones(3),
        }
    )
    with pytest.raises(RuntimeError, match="inconsistent with flat normalizer"):
        subject.validate_int8_payload(_int8_payload(inconsistent), inconsistent)


def test_archive_has_exact_layout_size_sha_and_trusted_code(tmp_path):
    direct = _refit_payload({"weight": torch.ones(2, 2)})
    archive = tmp_path / "candidate.zip"
    _write_archive(archive, _int8_payload(direct))
    report, payload = subject.archive_identity(archive, subject.dr1.sha256(archive))
    assert report["members"] == list(subject.EXPECTED_MEMBERS)
    assert report["within_256_mib"] is True
    assert set(report["trusted_code_sources"]) == set(subject.TRUSTED_MEMBER_SOURCES)
    assert payload["storage_format"] == subject.STORAGE_FORMAT

    changed = tmp_path / "changed.zip"
    _write_archive(changed, _int8_payload(direct), "submission.py")
    with pytest.raises(RuntimeError, match="trusted source"):
        subject.archive_identity(changed, subject.dr1.sha256(changed))

    with pytest.raises(RuntimeError, match="SHA changed"):
        subject.archive_identity(archive, "0" * 64)


def test_fresh_extract_refuses_dirty_destination(tmp_path):
    archive = tmp_path / "candidate.zip"
    _write_archive(archive, _int8_payload(_refit_payload({"w": torch.ones(2, 2)})))
    destination = tmp_path / "extract"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not empty"):
        subject.safe_extract_exact(archive, destination)


def test_module_source_report_rejects_import_outside_fresh_extract(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    local = package / "submission.py"
    local.write_text("", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")
    fake = SimpleNamespace(__file__=str(local))
    old = {
        name: sys.modules.get(name)
        for name in ("models", "models.unet2d", "models.ensemble_unet2d")
    }
    try:
        sys.modules["models"] = SimpleNamespace(__file__=str(outside))
        sys.modules["models.unet2d"] = SimpleNamespace(__file__=str(outside))
        sys.modules["models.ensemble_unet2d"] = SimpleNamespace(__file__=str(outside))
        with pytest.raises(ImportError, match="escaped fresh extraction"):
            subject._module_source_report(package, fake)
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_release_wrapper_does_not_recompute_strict_forecast_buffers(
    tmp_path, monkeypatch
):
    class TinyConfig:
        def __init__(self, **values):
            self.values = values

    class TinyModel(torch.nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.register_buffer("persistence_scale", torch.ones(3))
            self.register_buffer("persistence_bias", torch.zeros(3))
            self.register_buffer("target_std", torch.ones(3))

        @torch.no_grad()
        def set_normalizer_statistics(
            self, mean_input, mean_target, std_input, std_target
        ):
            safe = std_target.clamp_min(1e-8)
            self.persistence_scale.copy_(std_input / safe)
            self.persistence_bias.copy_((mean_input - mean_target) / safe)
            self.target_std.copy_(safe)

    model = TinyModel(None)
    with torch.no_grad():
        model.persistence_bias.copy_(torch.tensor([0.25, -0.5, 0.75]))
    payload = {
        "model_config": {"architecture": "ensemble_unet"},
        "model_state_dict": model.state_dict(),
        "normalizer": _normalizer(),
        "inference_dtype": "float16",
    }
    checkpoint = tmp_path / "compact.pth"
    torch.save(payload, checkpoint)
    monkeypatch.setattr(release_wrapper, "_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(release_wrapper, "EnsembleUNet2dConfig", TinyConfig)
    monkeypatch.setattr(release_wrapper, "EnsembleUNet2d", TinyModel)
    monkeypatch.setattr(release_wrapper.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        release_wrapper,
        "_state",
        {
            "model": None,
            "device": None,
            "dtype": None,
            "mean_input": None,
            "mean_target": None,
            "std_input": None,
            "std_target": None,
        },
    )
    runtime = release_wrapper._get_model()
    restored = release_wrapper._restore_state_dict(payload, torch.float32)
    loaded = runtime["model"].state_dict()
    assert set(loaded) == set(restored)
    assert all(torch.equal(loaded[key], restored[key]) for key in loaded)
    assert torch.equal(loaded["persistence_bias"], torch.tensor([0.25, -0.5, 0.75]))


def test_runtime_verifier_still_rejects_any_post_strict_buffer_change(monkeypatch):
    class TinyRuntimeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.energy_controller = torch.nn.Linear(1, 1)
            self.register_buffer("persistence_scale", torch.ones(3))
            self.register_buffer("persistence_bias", torch.ones(3))
            self.register_buffer("target_std", torch.ones(3))

        @staticmethod
        def architecture_config():
            return {"architecture": "ensemble_unet"}

    model = TinyRuntimeModel().eval()
    restored = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    restored["persistence_bias"] = torch.zeros(3)

    class ChangedRuntimeSubmission:
        _BATCH_SIZE = subject.EVAL_BATCH_SIZE

        @staticmethod
        def _get_model():
            statistics = {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in _normalizer().items()
            }
            return {
                "model": model,
                "device": torch.device("cpu"),
                "dtype": torch.float32,
                **statistics,
            }

        @staticmethod
        def _restore_state_dict(_payload, _dtype):
            return restored

    monkeypatch.setattr(subject.ft12, "validate_b210_model_config", lambda _x: None)
    monkeypatch.setattr(subject.uq_release, "EXPECTED_PARAMETERS", 3)
    monkeypatch.setattr(subject.uq_release, "EXPECTED_CONTROLLER_PARAMETERS", 2)
    with pytest.raises(RuntimeError, match="persistence_bias"):
        subject.validate_runtime_restore(
            ChangedRuntimeSubmission(), {"normalizer": _normalizer()}
        )


def test_worker_alignment_binds_order_inputs_targets_masks_and_pressure():
    direct = _worker("direct_refit_clean716", _metric_state())
    package = _worker("int8_package_clean716", _metric_state())
    aligned = subject.validate_worker_alignment(direct, package)
    assert all(aligned.values())

    package["replay_audit"]["ordered_raw_target_sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="not aligned"):
        subject.validate_worker_alignment(direct, package)


def test_cross_run_point_sum_difference_is_diagnostic_only():
    summary = _uq_summary(Path("refit.pth"), "1" * 64, "2" * 64)
    reference = dict(summary["promotion_gate"]["candidate_oof"]["metric_state"])
    direct = _worker("direct_refit_clean716", reference)
    direct["metric_state"]["rel_error_sum"] += 1e-3
    report = subject.direct_uq_point_diagnostic(direct, summary)
    assert report["hard_gate"] is False
    assert report["exact_equivalence_claimed"] is False
    assert report["decision_effect"] == "diagnostic_only"
    assert report["observed_raw_deltas_all_zero"] is False
    assert report["raw_deltas_direct_minus_uq_oof"]["rel_error_sum"] == pytest.approx(
        1e-3
    )
    assert report["score_deltas_direct_minus_uq_oof"]["rel_l2_score"] != 0.0


def test_package_replay_preserves_signed_extent_semantics(monkeypatch):
    history = torch.zeros(20, 2, 2, 3)
    target = torch.ones(20, 2, 2, 3)
    target[..., 2] = 0.0

    class SignedExtentSubmission:
        @staticmethod
        def predict(raw, metadata=None):
            del metadata
            prediction = torch.full(tuple(raw.shape), 2.0).numpy().astype("float32")
            prediction[..., 2] = 0.0
            lower = torch.zeros(tuple(raw.shape)).numpy().astype("float32")
            upper = torch.ones(tuple(raw.shape)).numpy().astype("float32")
            lower[..., 2] = 0.0
            upper[..., 2] = 0.0
            return {"prediction": prediction, "lower": lower, "upper": upper}

    monkeypatch.setattr(subject.uq_release, "EXPECTED_TAIL_WINDOWS", 1)
    state, audit = subject.replay_package(
        SignedExtentSubmission(), [(history, target)], time.perf_counter() + 10.0
    )
    assert audit["input_bitwise_unchanged"] is True
    assert audit["output_pressure_nonzero"] == 0
    assert state["samples"] == 1
    # Ordered absolute bounds are valid even though the point lies above upper.
    assert state["sps_width_sum"] > 0.0


def test_metric_gate_uses_sufficient_states_and_exact_component_floors(monkeypatch):
    direct = {
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 85.0,
        "native_sps_score": 50.0,
    }
    package = {
        "rel_l2_score": 89.97,
        "tke_rel_l2_score": 79.95,
        "mvpe_rel_l2_score": 84.97,
        "native_sps_score": 49.90,
    }
    monkeypatch.setattr(subject.smu1, "finalize_metric_state", lambda state: state)
    result = subject.metric_comparison(package, direct)
    assert result["pass"] is True
    assert result["decision"] == "PASS_B210_CLEAN716_EXACT_INT8_PACKAGE_VERIFY"

    package["tke_rel_l2_score"] = 79.949
    result = subject.metric_comparison(package, direct)
    assert result["pass"] is False
    assert result["conditions"]["tke_ge_minus_0p05"] is False


def test_additive_metric_state_is_not_fold_score_averaging():
    first = _metric_state(samples=1, error=0.01)
    second = _metric_state(samples=3, error=0.90)
    merged = subject.smu1.merge_metric_states([first, second])
    pooled = subject.smu1.finalize_metric_state(merged)
    fold_mean = (
        sum(
            subject.smu1.finalize_metric_state(state)["rel_l2_score"]
            for state in (first, second)
        )
        / 2.0
    )
    assert merged["samples"] == 4
    assert pooled["rel_l2_score"] != pytest.approx(fold_mean)


def test_worker_command_is_sequential_subprocess_contract(tmp_path):
    args = argparse.Namespace(
        refit_checkpoint=tmp_path / "refit.pth",
        refit_checkpoint_sha256="1" * 64,
        refit_model_state_sha256="2" * 64,
        refit_point_checkpoint_sha256="4" * 64,
        refit_point_model_state_sha256="5" * 64,
        refit_point_frozen_state_sha256="8" * 64,
        refit_point_normalizer_sha256="9" * 64,
        refit_training_sha256="6" * 64,
        promotion_gate_sha256="7" * 64,
        clean_manifest=tmp_path / "clean.json",
        real_dir=tmp_path / "real",
        device="cuda:0",
        hardware_class="rtx4090",
        archive=tmp_path / "candidate.zip",
        archive_sha256="3" * 64,
    )
    direct = subject.worker_command(args, "direct", tmp_path / "direct.json")
    package = subject.worker_command(args, "package", tmp_path / "package.json")
    assert direct[0] == sys.executable
    assert "--worker-direct" in direct and "--archive" not in direct
    assert "--worker-package" in package and "--archive" in package
    assert subject.WORKER_ACKNOWLEDGEMENT in direct


def test_contract_only_is_cpu_safe_and_emits_frozen_contract():
    completed = subprocess.run(
        [sys.executable, str(Path(subject.__file__)), "--contract-only"],
        cwd=subject.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["contract_sha256"] == subject.CONTRACT_SHA256
    assert result["contract"]["data"]["tail_windows"] == 716
    assert result["contract"]["submission_authorized"] is False
