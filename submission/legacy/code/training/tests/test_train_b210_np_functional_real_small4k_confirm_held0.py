from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_np_functional_real_small4k_confirm_held0 as subject  # noqa: E402


def test_frozen_anchor_sha_is_a_valid_sha256():
    assert len(subject.ANCHOR_SHA256) == 64
    assert subject.ANCHOR_SHA256 == subject.ANCHOR_SHA256.lower()
    int(subject.ANCHOR_SHA256, 16)


def test_train_candidate_exposes_one_optional_extra_backward_hook():
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert source.count("extra_backward(iteration, model)") == 1
    assert "extra backward metric collision" in source
    assert '"extra_backward": extra_backward is not None' in source


def test_train_candidate_exposes_explicit_scheduler_horizon():
    import inspect

    parameter = inspect.signature(subject.train_candidate).parameters[
        "scheduler_t_max"]
    assert parameter.default == subject.SCHEDULER_T_MAX == 12_000
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "T_max=scheduler_t_max" in source
    assert '"scheduler_t_max": scheduler_t_max' in source


def test_optional_ema_snapshots_are_intermediate_and_paired_with_sink():
    sink = lambda _update, _state: None
    assert subject.validate_ema_snapshot_contract(
        (500, 1500, 3000), sink, 4000) == (500, 1500, 3000)
    assert subject.validate_ema_snapshot_contract((), None, 4000) == ()
    for updates in ((0,), (4000,), (1500, 500), (500, 500)):
        with pytest.raises(ValueError):
            subject.validate_ema_snapshot_contract(updates, sink, 4000)
    with pytest.raises(ValueError, match="supplied together"):
        subject.validate_ema_snapshot_contract((500,), None, 4000)
    with pytest.raises(ValueError, match="supplied together"):
        subject.validate_ema_snapshot_contract((), sink, 4000)


def test_train_candidate_forwards_real_legacy_tke_floor_contract():
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "point_accuracy_weights=point_accuracy_weights" in source
    assert "member_accuracy_weights=member_accuracy_weights" in source
    assert "legacy_tke_gradient_floor=legacy_tke_gradient_floor" in source
    assert ("legacy_tke_gradient_floor_on_members=(" in source)
    assert "legacy_point_accuracy_weights=legacy_point_accuracy_weights" in source
    assert "legacy_member_accuracy_weights=(" in source
    assert "STOP_B210_REAL_LEGACY_TKE_FLOOR_PREFLIGHT" in source
    assert subject.math.isfinite(1.0)


def _state(samples: int, error: float, sps: float) -> dict:
    state = subject.smu1.new_metric_state()
    state.update({
        "samples": samples,
        "rel_error_sum": error * samples,
        "tke_error_sum": error * samples,
        "mvpe_error_sum": error * samples,
        "sps_numerator": sps * 100 * samples,
        "sps_scored": 100 * samples,
        "sps_covered": 80 * samples,
        "sps_width_sum": 5 * samples,
    })
    return state


def _audit(samples: int) -> dict:
    return {
        "ordered_raw_target_sha256": "a" * 64,
        "ordered_scored_mask_sha256": "b" * 64,
        "samples": samples,
        "scored_elements": 100 * samples,
    }


def _endpoint(state: dict, endpoint: str, stream: str) -> dict:
    return {
        "seal": {
            "endpoint": endpoint,
            "sample_stream_sha256": stream,
            "file_sha256": "c" * 64,
            "model_state_sha256": "d" * 64,
        },
        "metrics": subject.smu1.finalize_metric_state(state),
        "metric_state": state,
        "held_target_audit": _audit(int(state["samples"])),
    }


def _discovery_row() -> dict:
    samples = subject.dr1.EXPECTED_AOA_COUNTS[10]
    u4 = _endpoint(_state(samples, 0.10, 0.60), "ema_u4000", "e" * 64)
    u12 = _endpoint(_state(samples, 0.20, 0.20), "ema_u12000", "f" * 64)
    increment = subject.ft12.metric_comparison(
        u12["metrics"], u4["metrics"])
    gate = subject.ft12.fold10_disaster_gate(increment)
    assert gate["pass"] is False
    return {
        "schema": subject.ft12.SCHEMA,
        "experiment": subject.ft12.EXPERIMENT,
        "contract_sha256": subject.ft12.CONTRACT_SHA256,
        "held_aoa": 10,
        "samples": samples,
        "primary_endpoint": "ema_u12000",
        "diagnostics_cannot_select_endpoint": True,
        "decision": subject.DISCOVERY_DECISION,
        "start": {"checkpoint_sha256": "1" * 64},
        "training": {
            "updates": 12_000,
            "batch_size": subject.BATCH_SIZE,
            "seed": subject.SEED,
            "lr": subject.LR,
            "weight_decay": subject.WEIGHT_DECAY,
            "grad_clip": subject.GRAD_CLIP,
            "scheduler_t_max": subject.SCHEDULER_T_MAX,
            "ema_decay": subject.EMA_DECAY,
            "ema_updates": 12_000,
            "u4000_stream_sha256": "e" * 64,
        },
        "endpoints": {"ema_u4000": u4, "ema_u12000": u12},
        "primary": {
            "endpoint": "ema_u12000",
            "metrics": u12["metrics"],
            "metric_state": u12["metric_state"],
        },
        "ft12_increment_diagnostic": increment,
        "fold10_disaster_gate": gate,
    }


def test_contract_freezes_small4k_as_held0_only_confirmation():
    assert subject.HELD_AOA == 0
    assert subject.UPDATES == subject.ft12.DIAGNOSTIC_UPDATE == 4_000
    assert subject.SCHEDULER_T_MAX == subject.ft12.SCHEDULER_T_MAX == 12_000
    assert subject.BATCH_SIZE == subject.ft12.BATCH_SIZE == 8
    assert subject.SEED == subject.ft12.SEED == 4
    assert subject.LR == subject.ft12.LR == pytest.approx(2e-4)
    assert subject.WEIGHT_DECAY == subject.ft12.WEIGHT_DECAY == pytest.approx(1e-5)
    assert subject.EMA_DECAY == subject.ft12.EMA_DECAY == pytest.approx(0.999)
    assert subject.WALL_BUDGET_SECONDS == 60 * 60
    assert subject.PRIMARY_ENDPOINT == "ema_u4000"
    assert subject.CONTRACT["evidence_roles"] == {
        "held10": "discovery_only_post_hoc_not_confirmation",
        "held0": "single_independent_confirmation",
        "pooled": False,
        "official": False,
        "submission": False,
    }
    assert subject.CONTRACT["confirmation"]["endpoint_selection_allowed"] is False


def test_first_4000_stream_is_exact_ft12_prefix():
    windows = [SimpleNamespace(trajectory=index % 7) for index in range(41)]
    fit = list(range(len(windows)))
    short = subject.smu1.paired_batch_schedule(
        fit, windows, updates=subject.UPDATES, seed=subject.SEED)
    full = subject.smu1.paired_batch_schedule(
        fit, windows, updates=subject.ft12.UPDATES, seed=subject.SEED)
    assert short["rows"] == full["rows"][:subject.UPDATES]
    assert short["sha256"] == subject.dr1.canonical_sha256(
        full["rows"][:subject.UPDATES])


def test_tke_aux_schedule_is_exact_ft12_schedule():
    for iteration in (1, 200, 750, 1499, 1500, 1501, 4000):
        assert subject.ft12.tke_aux_weight(iteration) == pytest.approx(
            subject.npt1.tke_aux_weight(iteration))
    assert subject.ft12.tke_aux_weight(1) == pytest.approx(0.01)
    assert subject.ft12.tke_aux_weight(1500) == pytest.approx(0.0, abs=1e-15)


def test_discovery_validator_requires_full_u4_state_and_stopped_u12():
    row = _discovery_row()
    report = subject.validate_discovery_row(row)
    assert report["confirmation_role"] is False
    assert report["decision"] == subject.DISCOVERY_DECISION
    assert report["start_checkpoint_sha256"] == "1" * 64
    assert report["u4000_metrics"] == row["endpoints"]["ema_u4000"]["metrics"]

    changed = _discovery_row()
    changed["decision"] = "GO_B210_REAL_FT12_RUN_HELD0"
    with pytest.raises(RuntimeError, match="discovery/STOP"):
        subject.validate_discovery_row(changed)
    changed = _discovery_row()
    changed["endpoints"]["ema_u4000"]["metric_state"].pop("rel_error_sum")
    with pytest.raises(RuntimeError, match="metric_state schema"):
        subject.validate_discovery_row(changed)


def test_access_keeps_held0_behind_primary_seal(tmp_path):
    access = subject.EndpointAccessAudit()
    held = {"file": "held0.h5", "aoa": 0, "val_boundary": 80}
    with pytest.raises(RuntimeError, match="before endpoint seal"):
        access.before_open(held, "held")
    with pytest.raises(RuntimeError, match="excluded H5"):
        access.before_open({
            "file": subject.dr1.EXCLUDED_BASENAME,
            "aoa": 5,
            "val_boundary": 80,
        }, "fit")
    access.seal({
        "endpoint": "ema_u4000",
        "path": str(tmp_path / "endpoint.pth"),
        "file_sha256": "1" * 64,
        "model_state_sha256": "2" * 64,
    })
    access.before_open(held, "held")
    access.record_open(held, "held")
    access.record_evaluation()
    report = access.report(["held0.h5"])
    assert report["held_first_open_after_endpoint_seal"] is True
    assert report["held_complete_passes"] == 1


def test_primary_checkpoint_is_ema_u4000_and_flat_normalizer(tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def architecture_config(self):
            return {"architecture": "ensemble_unet", "tiny": True}

    class FakeEMA:
        def seal(self, model):
            state = OrderedDict(
                (name, value.detach().clone())
                for name, value in model.state_dict().items())
            return {
                "model_state": state,
                "updates": subject.UPDATES,
                "decay": subject.EMA_DECAY,
                "materialized_model_state_sha256": (
                    subject.ema_lib.tensor_state_sha256(state)),
            }

    model = TinyModel()
    normalizer = {
        "values": {
            "mean_input": [0.0, 0.0, 0.0],
            "mean_target": [0.0, 0.0, 0.0],
            "std_input": [1.0, 1.0, 1.0],
            "std_target": [1.0, 1.0, 1.0],
        },
        "sha256": "3" * 64,
    }
    path = tmp_path / "endpoint.pth"
    seal, _report = subject.save_primary_endpoint(
        path, model, FakeEMA(),
        start_report={"checkpoint_sha256": "4" * 64},
        normalizer_report=normalizer, stream_sha256="5" * 64)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert seal["endpoint"] == subject.PRIMARY_ENDPOINT
    assert seal["primary"] is True
    assert payload["primary"] is True
    assert payload["iteration"] == 4_000
    assert payload["normalizer"] == normalizer["values"]
    assert "values" not in payload["normalizer"]


def _aligned_candidate_anchor() -> tuple[dict, dict]:
    samples = subject.dr1.EXPECTED_AOA_COUNTS[0]
    audit = _audit(samples)
    state = _state(samples, 0.10, 0.50)
    metrics = subject.smu1.finalize_metric_state(state)
    manifest = {
        "semantic_sha256": "6" * 64,
        "windows": subject.dr1.EXPECTED_WINDOWS,
        "excluded": subject.dr1.EXCLUDED_BASENAME,
    }
    candidate = {
        "samples": samples,
        "manifest_file_sha256": "7" * 64,
        "manifest": manifest,
        "normalizer": {"sha256": "8" * 64},
        "held_ordered_key_sha256": "9" * 64,
        "endpoint": {
            "metrics": metrics,
            "metric_state": state,
            "held_target_audit": audit,
        },
    }
    anchor = {
        "schema": subject.mef2.SCHEMA,
        "experiment": subject.mef2.EXPERIMENT,
        "held_aoa": 0,
        "endpoint_order": list(subject.mef2.ENDPOINT_ORDER),
        "samples": samples,
        "manifest_file_sha256": "7" * 64,
        "manifest": dict(manifest),
        "normalizer": {"sha256": "8" * 64},
        "held_ordered_key_sha256": "9" * 64,
        "endpoints": {
            "control_ema": {
                "metrics": metrics,
                "metric_state": dict(state),
                "held_target_audit": dict(audit),
            },
        },
    }
    return candidate, anchor


def test_anchor_alignment_is_strict_and_uses_control_ema():
    candidate, anchor = _aligned_candidate_anchor()
    candidate_report, anchor_report = subject.validate_anchor_row(
        candidate, anchor)
    assert candidate_report["metrics"] == anchor_report["metrics"]
    changed = {**anchor, "held_ordered_key_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="ordered keys"):
        subject.validate_anchor_row(candidate, changed)


def _comparison(rel=0.0, tke=0.20, mvpe=0.0, sps=0.30, non_time=0.10):
    return {
        "delta": {
            "rel_l2_score": rel,
            "tke_rel_l2_score": tke,
            "mvpe_rel_l2_score": mvpe,
            "native_sps_score": sps,
        },
        "non_time_delta": non_time,
    }


def test_confirmation_gate_requires_every_frozen_threshold():
    passed = subject.confirmation_gate(_comparison())
    assert passed["pass"] is True
    assert passed["decision"] == (
        "PASS_B210_SMALL4K_HELD0_INDEPENDENT_CONFIRMATION")
    for field, value in (
            ("rel", -1e-8), ("tke", 0.199), ("mvpe", -1e-8),
            ("sps", 0.299), ("non_time", 0.099)):
        values = {field: value}
        failed = subject.confirmation_gate(_comparison(**values))
        assert failed["pass"] is False
        assert failed["decision"].startswith("STOP_")


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        run_held0=True,
        contract_only=False,
        output_dir=Path("out"),
        real_dir=Path("real"),
        clean_manifest=Path("manifest.json"),
        start_sim=Path("sim.pth"),
        start_sim_sha256="a" * 64,
        discovery_fold10=Path("discovery.json"),
        anchor_held0=Path("anchor.json"),
        device="cuda:0",
        hardware_class="rtx4080_super",
        execute_formal_gpu=True,
        acknowledge_protocol=subject.FORMAL_ACKNOWLEDGEMENT,
    )


def test_cli_has_no_held_fold_choice_and_requires_acknowledgement():
    subject.validate_args(_args())
    wrong = _args()
    wrong.acknowledge_protocol = ""
    with pytest.raises(ValueError, match="requires acknowledgement"):
        subject.validate_args(wrong)
    missing = _args()
    missing.discovery_fold10 = None
    with pytest.raises(ValueError, match="missing required"):
        subject.validate_args(missing)


def test_source_has_no_forbidden_development_split_or_aggregate_mode():
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    assert "fixed725" not in source
    assert "clean716" not in source
    assert "aggregate-only" not in source
