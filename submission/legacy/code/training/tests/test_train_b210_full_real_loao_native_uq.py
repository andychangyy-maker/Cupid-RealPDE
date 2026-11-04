from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import TensorDataset


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "code/training"
sys.path.insert(0, str(TRAINING))
SCRIPT = TRAINING / "train_b210_full_real_loao_native_uq.py"
SPEC = importlib.util.spec_from_file_location("b210_full_loao_uq", SCRIPT)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def identities():
    result = []
    for aoa in SUBJECT.point_fold.AOA_VALUES:
        count = SUBJECT.point_fold.EXPECTED_FOLDS[aoa]["held_oof_windows"]
        result.extend({"aoa": aoa, "sample": index} for index in range(count))
    return result


def metric_state(sps_numerator: float):
    return {
        "samples": 1,
        "rel_error_sum": 0.1,
        "tke_error_sum": 0.2,
        "mvpe_error_sum": 0.3,
        "sps_numerator": sps_numerator,
        "sps_scored": 10,
        "sps_covered": 8,
        "sps_width_sum": 0.1,
    }


def fold_result(aoa: int, before: float, after: float):
    fold = SUBJECT.point_fold.aoa_fold(aoa)
    before_state = metric_state(before)
    after_state = metric_state(after)
    before_metrics = SUBJECT.smu1.finalize_metric_state(before_state)
    after_metrics = SUBJECT.smu1.finalize_metric_state(after_state)
    reference_state = metric_state(before - 0.5)
    reference_metrics = SUBJECT.smu1.finalize_metric_state(reference_state)
    return SUBJECT.point_fold.seal_json_payload({
        "schema": SUBJECT.FOLD_SCHEMA,
        "experiment": SUBJECT.EXPERIMENT,
        "held_aoa": aoa,
        "held_fold": fold,
        "partition": {
            "held_samples": SUBJECT.point_fold.EXPECTED_FOLDS[aoa][
                "held_oof_windows"]},
        "training": {"updates": SUBJECT.UPDATES},
        "endpoint": {"endpoint": "fixed_oof_u800"},
        "held_evaluation": {
            "endpoint_was_sealed_before_held_scoring": True,
            "arms": {
                "A_final_point_original_uq_reference": {
                    "metrics": reference_metrics,
                    "metric_state": reference_state},
                "B_pure_sim_reset_native": {
                    "metrics": before_metrics, "metric_state": before_state},
                "C_loao_u800": {
                    "metrics": after_metrics, "metric_state": after_state},
            },
            "native_before": {
                "metrics": before_metrics, "metric_state": before_state},
            "native_after": {
                "metrics": after_metrics, "metric_state": after_state},
        },
        "comparison": SUBJECT.sps_comparison(before_metrics, after_metrics),
        "oof_residual_cache": {"file_sha256": "c" * 64},
        "uq_initialization": {
            "source_file_sha256": "d" * 64,
            "full_real_uq_state_inherited": False},
        "reference_final_uq": {
            "result_sha256": "e" * 64,
            "uq_state_sha256": "f" * 64,
            "real_in_sample_initialization": True,
            "reference_only": True},
        "held_optimizer_samples": 0,
        "point_state_bitwise": True,
        "held_checkpoint_selection": False,
        "clean716_manifest_read": False,
    })


def test_partition_is_exact_whole_aoa_and_held_never_optimizes():
    values = identities()
    fit, held, report = SUBJECT.partition_indices(values, 10)
    assert len(values) == 3_341
    assert len(held) == 756
    assert len(fit) == 3_341 - 756
    assert all(values[index]["aoa"] != 10 for index in fit)
    assert all(values[index]["aoa"] == 10 for index in held)
    assert report["held_optimizer_samples"] == 0


def test_aggregate_reports_each_fold_and_pooled_before_after_without_gate(
        tmp_path):
    pairs = []
    for fold, aoa in enumerate(SUBJECT.point_fold.AOA_VALUES):
        path = tmp_path / f"held{aoa}.json"
        path.write_text(json.dumps(
            fold_result(aoa, 4.0 + fold, 5.0 + fold)), encoding="utf-8")
        pairs.append((str(path), SUBJECT.dr1.sha256(path)))
    result = SUBJECT.aggregate_results(pairs)
    assert result["held_aoas"] == [0, 5, 10, 15, 20]
    assert result["updates_per_fold"] == 800
    assert result["total_crossfit_updates"] == 4_000
    assert len(result["per_fold_native_sps"]) == 5
    assert set(result["pooled"]["arms"]) == {
        "A_final_point_original_uq_reference",
        "B_pure_sim_reset_native", "C_loao_u800"}
    assert result["pooled"]["native_sps_after"] > result["pooled"][
        "native_sps_before"]
    assert result["pooled"]["comparisons"]["primary_c_minus_b"][
        "native_sps_delta"] > 0
    assert result["pooled"]["comparisons"]["reference_c_minus_a"][
        "native_sps_delta"] > 0
    assert result["automatic_promotion_threshold"] is None
    assert result["decision"] == "COMPLETE_LOAO_EVIDENCE_NO_AUTOMATIC_GATE"


def test_aggregate_rejects_missing_fold_and_tampered_roundtrip_seal(tmp_path):
    with pytest.raises(ValueError, match="exactly five"):
        SUBJECT.aggregate_results([])
    pairs = []
    for fold, aoa in enumerate(SUBJECT.point_fold.AOA_VALUES):
        path = tmp_path / f"held{aoa}.json"
        payload = fold_result(aoa, 4.0, 5.0)
        if aoa == 15:
            payload["held_optimizer_samples"] = 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        pairs.append((str(path), SUBJECT.dr1.sha256(path)))
    with pytest.raises(RuntimeError, match="self-seal changed"):
        SUBJECT.aggregate_results(pairs)


def test_contract_has_no_clean716_or_external_evaluation_manifest():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'add_argument("--evaluation-manifest' not in source
    assert 'add_argument("--clean716' not in source
    assert SUBJECT.AGGREGATE_SCHEMA == SUBJECT.full_uq.LOAO_AGGREGATE_SCHEMA


def test_held_scorer_uses_cached_residual_and_factor_override():
    class Normalizer:
        std_target = torch.ones(3)

        @staticmethod
        def preprocess(history, target):
            return history, target

        @staticmethod
        def postprocess_prediction(prediction):
            return prediction

    class Model(torch.nn.Module):
        @staticmethod
        def forward_distribution(history):
            batch = history.shape[0]
            prediction = torch.zeros((batch, 20, 32, 64, 3))
            extent = torch.full((batch, 20, 32, 64, 2), 0.1)
            return prediction, extent, extent

    history = torch.zeros((1, 20, 32, 64, 3))
    # Raw residual is +1 and lies outside +/-0.1, while the bound cache
    # residual is zero and lies inside. Coverage=1 proves override use.
    target = torch.ones((1, 20, 32, 64, 3))
    target[..., 2] = 0.0
    residual = torch.zeros((1, 20, 32, 64, 2), dtype=torch.float16)
    factor = torch.ones(1)
    dataset = TensorDataset(history, target, residual, factor)
    metrics, state, _target_audit, override = SUBJECT.evaluate_oof_native(
        Model(), dataset, [0], Normalizer(), torch.device("cpu"), float("inf"))
    assert metrics["native_sps_coverage"] == 1.0
    assert state["sps_covered"] == state["sps_scored"]
    assert override["cached_residual_used_for_inside_test"] is True
    assert override["cached_point_factor_used_for_sps"] is True
    assert override["raw_target_used_for_sps_residual_or_factor"] is False
