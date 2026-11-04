from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_c208_mef2_paired_joint_loao as subject  # noqa: E402
from models.ensemble_unet2d import PhysicalEnergyController  # noqa: E402


def test_controller_zero_contract_only_requires_zero_output_projection():
    state = {
        "body.0.weight": torch.randn(4, 4, 1, 1),
        "body.0.bias": torch.randn(4),
        "body.5.weight": torch.zeros(1, 4, 1, 1),
        "body.5.bias": torch.zeros(1),
    }
    assert subject.controller_output_projection_is_zero(state)
    state["body.5.bias"][0] = 1.0
    assert not subject.controller_output_projection_is_zero(state)


def test_frozen_controller_ema_canonicalizes_only_roundoff_drift():
    class MiniModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.energy_controller = PhysicalEnergyController(4, 4, 2)

    torch.manual_seed(4)
    model = MiniModel()
    raw = subject.controller_state(model)
    shadow = {
        f"energy_controller.{name}": value.clone()
        for name, value in raw.items()
    }
    shadow["energy_controller.body.0.weight"].add_(1e-6)
    ema = SimpleNamespace(_shadow=shadow)
    expected = subject.ema_lib.tensor_state_sha256(raw)
    report = subject.canonicalize_frozen_controller_ema(
        ema, model, expected)
    assert report["changed_elements_before"] > 0
    assert report["max_abs_drift_before"] > 0
    assert report["exact_after"] is True
    after = {
        name.removeprefix("energy_controller."): value
        for name, value in ema._shadow.items()
    }
    assert subject.tensor_state_equal(after, raw)


def test_four_endpoint_ledger_blocks_held_until_all_seals(tmp_path):
    ledger = subject.FourEndpointAccessAudit(10)
    held = {"file": "held.h5", "aoa": 10, "val_boundary": 80}
    with pytest.raises(RuntimeError, match="before four seals"):
        ledger.before_open(held, "held")
    for index, endpoint in enumerate(subject.ENDPOINT_ORDER):
        ledger.seal(endpoint, {
            "path": str(tmp_path / f"{endpoint}.pth"),
            "file_sha256": f"{index + 1}" * 64,
            "model_state_sha256": f"{index + 5}" * 64,
        })
    ledger.before_open(held, "held")
    ledger.record_open(held, "held")
    for endpoint in subject.ENDPOINT_ORDER:
        ledger.record_evaluation(endpoint)
    report = ledger.report(["held.h5"])
    assert report["held_first_open_after_four_seals"] is True
    assert report["held_evaluation_order"] == list(subject.ENDPOINT_ORDER)


def test_wave_aggregation_stops_and_rejects_post_stop_rows(monkeypatch):
    monkeypatch.setattr(subject, "validate_fold_rows", lambda rows: None)
    first_two = [
        {"held_aoa": 10, "primary": {"non_time_delta": 0.10}},
        {"held_aoa": 0, "primary": {"non_time_delta": 0.20}},
    ]
    assert subject.aggregate_fold_rows(first_two) == {
        "decision": "CONTINUE_MEF2_LOAO",
        "completed_folds": 2,
        "next_wave": [20, 15],
    }
    stopped = [
        {"held_aoa": 10, "primary": {"non_time_delta": -0.01}},
        {"held_aoa": 0, "primary": {"non_time_delta": 0.0}},
    ]
    assert subject.aggregate_fold_rows(stopped)["decision"] == (
        "STOP_MEF2_FIRST_TWO_NONPOSITIVE")
    with pytest.raises(RuntimeError, match="after wave1 STOP"):
        subject.aggregate_fold_rows(stopped + [
            {"held_aoa": 20, "primary": {"non_time_delta": 0.2}},
            {"held_aoa": 15, "primary": {"non_time_delta": 0.2}},
        ])


def _args(held_aoa: int, prior_aggregate=None) -> argparse.Namespace:
    return argparse.Namespace(
        start_sim_sha256=subject.FORMAL_START_SHA256,
        fold_worker=True,
        aggregate_only=False,
        contract_only=False,
        held_aoa=held_aoa,
        output_dir=Path("out"),
        clean_manifest=Path("manifest.json"),
        start_sim=Path("start.pth"),
        hardware_class="rtx4090",
        execute_formal_gpu=True,
        acknowledge_protocol=subject.FORMAL_ACKNOWLEDGEMENT,
        prior_aggregate=prior_aggregate,
        fold_result=None,
        aggregate_output=None,
    )


def test_fold_cli_enforces_wave_authorization_before_execution():
    subject.validate_args(_args(10))
    subject.validate_args(_args(0))
    with pytest.raises(ValueError, match="cannot receive"):
        subject.validate_args(_args(10, ["prior.json", "a" * 64]))
    with pytest.raises(ValueError, match="require prior"):
        subject.validate_args(_args(20))
    subject.validate_args(_args(20, ["prior.json", "a" * 64]))
    source = inspect.getsource(subject.run_fold)
    assert source.index("validate_prior_authorization") < source.index(
        "load_prefix_rows")


def test_primary_fold_gate_has_no_raw_endpoint_selection():
    assert subject.fold_gate({"non_time_delta": -0.051})["pass"] is False
    assert subject.fold_gate({"non_time_delta": -0.05})["pass"] is True
