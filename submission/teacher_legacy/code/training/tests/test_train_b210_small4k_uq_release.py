from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_small4k_uq_release as subject  # noqa: E402


def _clean81_rows() -> list[dict]:
    rows = []
    for index in range(81):
        total_starts = 45 if index < 68 else 40
        tail_count = 9 if index < 68 else 8
        frames = 40 + 20 * (total_starts - 1)
        boundary = 20 * (total_starts - tail_count)
        rows.append(
            {
                "file": f"{6000 + index}_{(index % 5) * 5}.h5",
                "frames": frames,
                "re": 6000 + index,
                "aoa": (index % 5) * 5,
                "train_starts": [0],
                "val_boundary": boundary,
            }
        )
    return rows


def _metric_state(samples: int, scored: int, numerator: float) -> dict:
    return {
        "samples": samples,
        "rel_error_sum": 0.0,
        "tke_error_sum": 0.0,
        "mvpe_error_sum": 0.0,
        "sps_numerator": numerator,
        "sps_scored": scored,
        "sps_covered": 0,
        "sps_width_sum": 0.0,
    }


def _normalizer_values() -> dict:
    return {
        "mean_input": [0.0, 0.0, 0.0],
        "mean_target": [0.0, 0.0, 0.0],
        "std_input": [1.0, 1.0, 1.0],
        "std_target": [1.0, 1.0, 1.0],
    }


def _point_payload(state: dict[str, torch.Tensor]) -> dict:
    state_sha = subject.dr1.model_state_sha256(state)
    normalizer = _normalizer_values()
    bindings = dict(subject.EXPECTED_EVIDENCE_BINDINGS)
    return {
        "schema": subject.INPUT_ENDPOINT_SCHEMA,
        "experiment": subject.INPUT_EXPERIMENT,
        "phase": "real",
        "iteration": 4000,
        "endpoint": "ema_u4000",
        "primary": True,
        "training_scope": "full_clean2544",
        "fit_samples": subject.dr1.EXPECTED_WINDOWS,
        "status": subject.INPUT_STATUS,
        "manifest_file_sha256": subject.FORMAL_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": subject.FORMAL_MANIFEST_SEMANTIC_SHA256,
        "manifest_windows": subject.dr1.EXPECTED_WINDOWS,
        "manifest_excluded": subject.dr1.EXCLUDED_BASENAME,
        "optimizer_state_saved": False,
        "held_or_tail_labels_read": False,
        "tail_arrays_read": 0,
        "endpoint_selection_allowed": False,
        "candidate_specific_oof_uq_refit_required": True,
        "release_checkpoint_written": False,
        "submission_authorized": False,
        "sample_stream_sha256": "5" * 64,
        "authorization_sha256": "6" * 64,
        "evidence_bindings": bindings,
        "start": {"checkpoint_sha256": bindings["sim10k_sha256"]},
        "ema": {
            "schema": subject.ema_lib.EMA_SEAL_SCHEMA,
            "update_order": subject.ema_lib.UPDATE_ORDER_CONTRACT,
            "updates": 4000,
            "decay": 0.999,
            "shadow_dtype": "torch.float32",
            "materialized_model_state_sha256": state_sha,
        },
        "model_config": {"architecture": "ensemble_unet"},
        "model_state_dict": state,
        "model_state_sha256": state_sha,
        "normalizer": normalizer,
        "normalizer_sha256": subject.dr1.canonical_sha256(normalizer),
    }


def test_contract_reuses_fixed_condition_oof800_refit800_recipe():
    assert subject.FOLDS == 5
    assert subject.FOLD_UPDATES == 800
    assert subject.REFIT_UPDATES == 800
    assert subject.BATCH_SIZE == 8
    assert subject.SEED == 12
    assert subject.LR == pytest.approx(1e-4)
    assert subject.WEIGHT_DECAY == pytest.approx(1e-5)
    assert subject.SCHEDULER_T_MAX == 800
    assert subject.EXPECTED_TAIL_WINDOWS == 716
    assert "not AoA-LOAO" in subject.CONTRACT["oof"]["fold_definition"]
    assert subject.CONTRACT["oof"]["held_checkpoint_selection"] is False
    assert subject.CONTRACT["training"]["point_gain_attribution"] is False
    assert subject.CONTRACT["refit"]["stage"] == "UQ_REFIT_RELEASE_STAGE"
    assert subject.CONTRACT["zip_written"] is False
    assert subject.CONTRACT["submission_authorized"] is False


def test_clean716_derivation_excludes_bad_file_and_has_exact_count():
    rows, report = subject.derive_clean716_manifest(_clean81_rows())
    assert report["tail_windows"] == 716
    assert len(rows) == 81
    assert sum(len(row["val_starts"]) for row in rows) == 716
    assert all(row["val_starts"][0] == row["val_boundary"] for row in rows)
    bad = _clean81_rows()
    bad[0]["file"] = subject.dr1.EXCLUDED_BASENAME
    with pytest.raises(RuntimeError, match="excluded trajectory"):
        subject.derive_clean716_manifest(bad)


def test_condition_fold_formula_is_whole_re_aoa_not_loao():
    rows, _report = subject.derive_clean716_manifest(_clean81_rows())
    labels, manifest = subject.trajectory_folds(rows, subject.FOLDS)
    assert manifest["formula"] == "(rank(re)+2*rank(aoa)) % folds"
    assert len(labels) == 716
    assignments = manifest["assignments"]
    assert all(
        row["windows"] == len(rows[index]["val_starts"])
        for index, row in enumerate(assignments)
    )


def test_flat_normalizer_rejects_nested_schema():
    values = _normalizer_values()
    statistics, report = subject.validate_flat_normalizer(
        {
            "normalizer": values,
            "normalizer_sha256": subject.dr1.canonical_sha256(values),
        }
    )
    assert all(value.shape == (3,) for value in statistics)
    assert report["flat"] is True
    with pytest.raises(ValueError, match="flat schema"):
        subject.validate_flat_normalizer(
            {
                "normalizer": {"values": values},
                "normalizer_sha256": "0" * 64,
            }
        )


def test_point_input_is_full_clean_strict_b210_and_sha_bound(tmp_path, monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.legacy = torch.nn.Parameter(torch.zeros(3))
            self.energy_controller = torch.nn.Linear(1, 1)

        def set_normalizer_statistics(self, *_values):
            pass

    tiny = TinyModel(object())
    payload = _point_payload(tiny.state_dict())
    path = tmp_path / "point.pth"
    torch.save(payload, path)
    monkeypatch.setattr(subject, "EnsembleUNet2d", TinyModel)
    monkeypatch.setattr(
        subject.ft12, "validate_b210_model_config", lambda _value: object()
    )
    monkeypatch.setattr(subject, "EXPECTED_PARAMETERS", 5)
    monkeypatch.setattr(subject, "EXPECTED_CONTROLLER_PARAMETERS", 2)
    loaded, _normalizer, report = subject.load_point_artifact(
        path, subject.dr1.sha256(path), torch.device("cpu")
    )
    assert report["strict_load"] is True
    assert report["training_scope"] == "full_clean2544"
    assert report["fit_samples"] == 2544
    assert loaded["normalizer"] == _normalizer_values()

    held_fold = dict(payload)
    held_fold["training_scope"] = "other_four_held0"
    held_path = tmp_path / "held.pth"
    torch.save(held_fold, held_path)
    with pytest.raises(ValueError, match="contract changed"):
        subject.load_point_artifact(
            held_path, subject.dr1.sha256(held_path), torch.device("cpu")
        )

    broken = _point_payload(
        {key: value for key, value in tiny.state_dict().items() if key != "legacy"}
    )
    broken_path = tmp_path / "broken.pth"
    torch.save(broken, broken_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        subject.load_point_artifact(
            broken_path, subject.dr1.sha256(broken_path), torch.device("cpu")
        )


def test_uq_only_mode_keeps_point_dropout_controller_and_members_eval():
    class TinyBundle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.Dropout(0.2)
            )
            self.member_point_heads = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.energy_controller = torch.nn.Linear(2, 2)
            self.uq_projections = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.uq_fusion = torch.nn.Linear(2, 2)
            self.quantile_head = torch.nn.Linear(2, 2)
            self.utility_refinement_head = torch.nn.Linear(2, 2)
            self.temporal_uq_refiner = torch.nn.Linear(2, 2)
            self.coverage_calibration_head = None

    model = TinyBundle()
    parameters = subject.configure_uq_only(model)
    subject.set_uq_training_mode(model)
    assert parameters
    assert model.training is False
    assert model.backbone.training is False
    assert model.backbone[1].training is False
    assert model.member_point_heads.training is False
    assert model.energy_controller.training is False
    assert model.uq_fusion.training is True
    assert all(not value.requires_grad for value in model.backbone.parameters())
    assert all(
        not value.requires_grad for value in model.energy_controller.parameters()
    )
    assert all(value.requires_grad for value in model.uq_fusion.parameters())


def test_held_scoring_requires_fixed_terminal_seal(tmp_path):
    audit = subject.FoldEndpointAudit()
    with pytest.raises(RuntimeError, match="precede terminal seal"):
        audit.before_evaluate(0, "native_baseline")
    seal = {
        "path": str(tmp_path / "fold.pth"),
        "file_sha256": "1" * 64,
        "uq_state_sha256": "2" * 64,
        "fold": 0,
        "iteration": 800,
        "endpoint": "fixed_terminal_u800",
    }
    audit.seal(0, seal)
    audit.before_evaluate(0, "native_baseline")
    audit.before_evaluate(0, "fixed_terminal_candidate")
    assert len(audit.evaluations) == 2
    assert all(row["terminal_sealed_before_held_score"] for row in audit.evaluations)


def test_pooled_oof_uses_sufficient_state_not_unweighted_fold_mean():
    baseline = [_metric_state(1, 1, 0.0)] + [_metric_state(9, 9, 0.0) for _ in range(4)]
    candidate = [_metric_state(1, 1, 1.0)] + [
        _metric_state(9, 9, 0.0) for _ in range(4)
    ]
    baseline_aoa = {
        aoa: _metric_state(1 if index == 0 else 9, 1 if index == 0 else 9, 0.0)
        for index, aoa in enumerate(subject.dr1.AOA_VALUES)
    }
    candidate_aoa = {
        aoa: _metric_state(
            1 if index == 0 else 9, 1 if index == 0 else 9, 1.0 if index == 0 else 0.0
        )
        for index, aoa in enumerate(subject.dr1.AOA_VALUES)
    }
    gate = subject.promotion_gate(baseline, candidate, baseline_aoa, candidate_aoa)
    pooled = gate["candidate_oof"]["metrics"]["native_sps_score"]
    fold_mean = sum(row["candidate"]["native_sps_score"] for row in gate["folds"]) / 5
    assert gate["aggregation"] == "additive_sufficient_metric_state"
    assert pooled == pytest.approx(100.0 / 37.0)
    assert fold_mean == pytest.approx(20.0)
    assert pooled != pytest.approx(fold_mean)


def test_refit_endpoint_is_flat_uq_stage_and_never_zip(tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def architecture_config(self):
            return {"architecture": "ensemble_unet", "tiny": True}

    path = tmp_path / "refit.pth"
    gate = {"pass": True, "decision": "PASS"}
    seal = subject.save_refit_endpoint(
        path,
        TinyModel(),
        {"checkpoint_sha256": "1" * 64, "model_state_sha256": "2" * 64},
        {"values": _normalizer_values(), "sha256": "3" * 64},
        {"updates": 800, "frozen_state_bitwise": True},
        gate,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert seal["stage"] == "UQ_REFIT_RELEASE_STAGE"
    assert payload["normalizer"] == _normalizer_values()
    assert "values" not in payload["normalizer"]
    assert payload["point_gain_attribution"] is False
    assert payload["exact_verifier_required"] is True
    assert payload["zip_written"] is False
    assert payload["submission_authorized"] is False


def test_formal_run_requires_explicit_acknowledgement(tmp_path):
    args = argparse.Namespace(
        run=True,
        contract_only=False,
        point_checkpoint=tmp_path / "point.pth",
        point_checkpoint_sha256="1" * 64,
        clean_manifest=tmp_path / "manifest.json",
        output_dir=tmp_path / "out",
        execute_formal_gpu=False,
        acknowledge_protocol="",
    )
    with pytest.raises(ValueError, match="requires acknowledgement"):
        subject.validate_args(args)
    args.execute_formal_gpu = True
    args.acknowledge_protocol = subject.FORMAL_ACKNOWLEDGEMENT
    subject.validate_args(args)
