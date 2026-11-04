from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = EVALUATION_ROOT.parent
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_b210_np_functional_held as subject  # noqa: E402


def _state(offset: float = 0.0) -> dict:
    state = subject.smu1.new_metric_state()
    state.update({
        "samples": 2,
        "rel_error_sum": 0.20 + offset,
        "tke_error_sum": 0.30 + offset,
        "mvpe_error_sum": 0.40 + offset,
        "sps_numerator": 80.0 + offset,
        "sps_scored": 200,
        "sps_covered": 150,
        "sps_width_sum": 20.0,
    })
    return state


def _fold_result() -> dict:
    audit = {
        "ordered_raw_target_sha256": "1" * 64,
        "ordered_scored_mask_sha256": "2" * 64,
        "samples": 2,
        "scored_elements": 200,
    }
    endpoints = {}
    for index, endpoint in enumerate(subject.real_ft.ENDPOINT_ORDER):
        state = _state(0.01 * index)
        endpoints[endpoint] = {
            "seal": {
                "endpoint": endpoint,
                "path": f"{endpoint}.pth",
                "file_sha256": f"{index + 3:x}" * 64,
                "model_state_sha256": f"{index + 6:x}" * 64,
                "primary": endpoint == subject.real_ft.PRIMARY_ENDPOINT,
            },
            "metrics": subject.smu1.finalize_metric_state(state),
            "metric_state": state,
            "held_target_audit": audit,
        }
    primary = endpoints[subject.real_ft.PRIMARY_ENDPOINT]
    row = {
        "schema": subject.real_ft.SCHEMA,
        "experiment": subject.real_ft.EXPERIMENT,
        "contract_sha256": subject.real_ft.CONTRACT_SHA256,
        "held_aoa": 10,
        "endpoint_order": list(subject.real_ft.ENDPOINT_ORDER),
        "primary_endpoint": subject.real_ft.PRIMARY_ENDPOINT,
        "diagnostics_cannot_select_endpoint": True,
        "endpoints": endpoints,
        "primary": {
            "endpoint": subject.real_ft.PRIMARY_ENDPOINT,
            "metrics": primary["metrics"],
            "metric_state": primary["metric_state"],
        },
    }
    row["ft12_increment_diagnostic"] = subject.real_ft.metric_comparison(
        primary["metrics"], endpoints["ema_u4000"]["metrics"])
    row["raw_vs_ema_u12000_diagnostic"] = (
        subject.real_ft.metric_comparison(
            primary["metrics"], endpoints["raw_u12000"]["metrics"]))
    return row


def test_existing_endpoint_deltas_are_recomputed_and_cannot_select():
    row = _fold_result()
    report = subject.validate_existing_diagnostics(row)
    assert report["ema_u4000_to_ema_u12000"] == (
        row["ft12_increment_diagnostic"])
    assert report["raw_u12000_to_ema_u12000"] == (
        row["raw_vs_ema_u12000_diagnostic"])
    assert report["recomputed_exactly_from_sealed_metric_states"] is True
    assert report["cannot_select_endpoint"] is True

    row["ft12_increment_diagnostic"]["non_time_delta"] += 1e-3
    with pytest.raises(RuntimeError, match="diagnostics changed"):
        subject.validate_existing_diagnostics(row)


def test_existing_diagnostics_reject_target_mask_audit_drift():
    row = _fold_result()
    row["endpoints"]["raw_u12000"]["held_target_audit"] = {
        **row["endpoints"]["raw_u12000"]["held_target_audit"],
        "ordered_scored_mask_sha256": "f" * 64,
    }
    with pytest.raises(RuntimeError, match="target/mask audits differ"):
        subject.validate_existing_diagnostics(row)


def _scale_contract() -> dict:
    value = {
        "method": subject.HGRAD_METHOD,
        "train_windows": 4800,
        "split_manifest_sha256": "a" * 64,
        "sim_normalizer_sha256": "b" * 64,
        "scale_floor": 1e-6,
        "counts_per_channel": {"dx": 10, "dy": 8},
        "sum_squares": {"dx": [1.0, 2.0], "dy": [3.0, 4.0]},
        "scales": {"dx": [0.1, 0.2], "dy": [0.3, 0.4]},
        "real_windows_used": 0,
        "fixed725_windows_used": 0,
    }
    return {**value, "canonical_sha256": subject.dr1.canonical_sha256(value)}


def test_hgrad_scale_must_be_canonical_sim_train_only():
    contract = _scale_contract()
    assert subject.validate_hgrad_scale_contract(contract) == contract
    leaked = {**contract, "fixed725_windows_used": 1}
    payload = dict(leaked)
    payload.pop("canonical_sha256")
    leaked["canonical_sha256"] = subject.dr1.canonical_sha256(payload)
    with pytest.raises(ValueError, match="provenance changed"):
        subject.validate_hgrad_scale_contract(leaked)
    changed = json.loads(json.dumps(contract))
    changed["scales"]["dx"][0] = 9.0
    with pytest.raises(RuntimeError, match="canonical SHA-256 changed"):
        subject.validate_hgrad_scale_contract(changed)


class _TinyEnergyController(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(2, 2, 1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(2, 1, 1),
        )


class _TinyControllerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Conv2d(2, 2, 1)
        self.energy_controller = _TinyEnergyController()
        with torch.no_grad():
            self.energy_controller.body[-1].weight.fill_(0.5)
            self.energy_controller.body[-1].bias.fill_(0.25)


def test_controller_projection_ablation_restores_full_model_on_success():
    model = _TinyControllerModel()
    initial = subject.dr1.model_state_sha256(model.state_dict())
    with subject.zero_controller_projection(model) as report:
        projection = model.energy_controller.body[-1]
        assert torch.count_nonzero(projection.weight) == 0
        assert torch.count_nonzero(projection.bias) == 0
    assert report["bitwise_restored"] is True
    assert report["model_state_sha256_before"] == initial
    assert report["model_state_sha256_after_restore"] == initial


def test_controller_projection_ablation_restores_after_evaluator_error():
    model = _TinyControllerModel()
    initial = subject.dr1.model_state_sha256(model.state_dict())
    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        with subject.zero_controller_projection(model):
            raise RuntimeError("synthetic evaluator failure")
    assert subject.dr1.model_state_sha256(model.state_dict()) == initial


class _IdentityNormalizer:
    mean_input = torch.zeros(3)
    mean_target = torch.zeros(3)
    std_input = torch.ones(3)
    std_target = torch.ones(3)

    def preprocess(self, history, target):
        return history, target


class _IdentityForecast(torch.nn.Module):
    def forward_distribution(self, history):
        prediction = history.clone()
        widths = torch.ones((*history.shape[:-1], 2))
        return prediction, widths, widths


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        base = torch.arange(2 * 3 * 4 * 3, dtype=torch.float32).reshape(
            2, 3, 4, 3)
        self.rows = [base, base + 1.0]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        value = self.rows[index]
        return value, value.clone()


def test_physical_hgrad_replays_existing_loss_and_binds_target_mask():
    dataset = _TinyDataset()
    contract = _scale_contract()
    value, audit = subject.evaluate_physical_hgrad(
        _IdentityForecast(), dataset, [0, 1], _IdentityNormalizer(),
        contract, torch.device("cpu"), float("inf"))
    assert value == pytest.approx(subject.real_hgrad.HGRAD_RMS_EPSILON ** 0.5)
    assert audit["samples"] == 2
    expected_scored = sum(
        int(row[..., :2].ne(0.0).sum()) for row in dataset.rows)
    assert audit["scored_elements"] == expected_scored
    assert len(audit["ordered_raw_target_sha256"]) == 64
    assert len(audit["ordered_scored_mask_sha256"]) == 64


def test_hgrad_retention_uses_existing_lower_is_better_advantage():
    assert subject.relative_hgrad_advantage(0.8, 1.0) == pytest.approx(0.2)
    assert subject.relative_hgrad_advantage(1.2, 1.0) == pytest.approx(-0.2)


def test_artifact_hash_validation_binds_every_input():
    row = _fold_result()
    row["manifest_file_sha256"] = "a" * 64
    row["start"] = {"checkpoint_sha256": "b" * 64}
    expected = {
        "fold_result": "1" * 64,
        "clean_manifest": "a" * 64,
        "sim_summary": "2" * 64,
        "hgrad_scale_contract": "3" * 64,
        "sim10k": "b" * 64,
        **{
            endpoint: row["endpoints"][endpoint]["seal"]["file_sha256"]
            for endpoint in subject.real_ft.ENDPOINT_ORDER
        },
    }
    subject.validate_artifact_hashes(
        row, expected, "1" * 64, "2" * 64, "3" * 64)
    changed = {**expected, "ema_u4000": "f" * 64}
    with pytest.raises(RuntimeError, match="ema_u4000"):
        subject.validate_artifact_hashes(
            row, changed, "1" * 64, "2" * 64, "3" * 64)
