from __future__ import annotations

from pathlib import Path
import sys

import json


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_realclosure_dense12k as subject  # noqa: E402


def _metrics(**updates):
    value = {
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 91.0,
        "point_score": 86.8,
        "native_sps_score": 40.0,
        "native_sps_coverage": 0.8,
        "native_sps_mean_width": 0.01,
    }
    value.update(updates)
    return value


def _comparison(point=0.05, tke=0.0, sps=0.0, aoa_proxy=0.02,
                aoa_point=0.01):
    control = _metrics()
    candidate = _metrics(
        point_score=control["point_score"] + point,
        tke_rel_l2_score=control["tke_rel_l2_score"] + tke,
        native_sps_score=control["native_sps_score"] + sps)
    row = subject.compare(candidate, control)
    row["by_aoa"] = {}
    for aoa in (0, 5, 10, 15, 20):
        arm = subject.compare(
            _metrics(point_score=86.8 + aoa_point), _metrics())
        arm["candidate_non_time_proxy_delta"] = aoa_proxy
        row["by_aoa"][str(aoa)] = arm
    return row


def test_protocol_is_terminal_dense12k_not_continuation():
    assert subject.UPDATES == 12_000
    assert subject.PRIMARY_ENDPOINT == "ema_u12000"
    assert subject.dense4k.CANDIDATE_STRIDE == 5
    assert subject.small4k.SCHEDULER_T_MAX == 12_000


def test_scoped_contract_restores_generic_trainer_globals():
    names = ("EXPERIMENT", "ENDPOINT_SCHEMA", "PRIMARY_ENDPOINT", "UPDATES")
    before = {name: getattr(subject.small4k, name) for name in names}
    with subject.dense12k_training_contract():
        assert subject.small4k.EXPERIMENT == subject.EXPERIMENT
        assert subject.small4k.ENDPOINT_SCHEMA == subject.ENDPOINT_SCHEMA
        assert subject.small4k.PRIMARY_ENDPOINT == subject.PRIMARY_ENDPOINT
        assert subject.small4k.UPDATES == subject.UPDATES
    assert {name: getattr(subject.small4k, name) for name in names} == before


def test_scoped_multiscale_contract_changes_only_training_identity():
    names = ("EXPERIMENT", "ENDPOINT_SCHEMA", "PRIMARY_ENDPOINT", "UPDATES")
    before = {name: getattr(subject.small4k, name) for name in names}
    with subject.dense12k_training_contract(multiscale_tke=True):
        assert subject.small4k.EXPERIMENT == subject.MULTISCALE_EXPERIMENT
        assert subject.small4k.ENDPOINT_SCHEMA == (
            subject.MULTISCALE_ENDPOINT_SCHEMA)
        assert subject.small4k.PRIMARY_ENDPOINT == subject.PRIMARY_ENDPOINT
        assert subject.small4k.UPDATES == subject.UPDATES
    assert {name: getattr(subject.small4k, name) for name in names} == before


def test_scoped_multiscale_energy_contract_changes_only_training_identity():
    names = ("EXPERIMENT", "ENDPOINT_SCHEMA", "PRIMARY_ENDPOINT", "UPDATES")
    before = {name: getattr(subject.small4k, name) for name in names}
    with subject.dense12k_training_contract(multiscale_energy=True):
        assert subject.small4k.EXPERIMENT == (
            subject.MULTISCALE_ENERGY_EXPERIMENT)
        assert subject.small4k.ENDPOINT_SCHEMA == (
            subject.MULTISCALE_ENERGY_ENDPOINT_SCHEMA)
        assert subject.small4k.PRIMARY_ENDPOINT == subject.PRIMARY_ENDPOINT
        assert subject.small4k.UPDATES == subject.UPDATES
    assert {name: getattr(subject.small4k, name) for name in names} == before


def test_scoped_contract_rejects_two_multiscale_modes():
    try:
        with subject.dense12k_training_contract(
                multiscale_tke=True, multiscale_energy=True):
            pass
    except ValueError as error:
        assert "only one" in str(error)
    else:
        raise AssertionError("two multiscale modes were accepted")


def test_dense12k_reference_reads_frozen_candidate_not_old_baseline(
        tmp_path, monkeypatch):
    manifest = tmp_path / "evaluation.json"
    manifest.write_text(json.dumps([]), encoding="utf-8")
    monkeypatch.setattr(subject.dr1, "sha256", lambda _path: "manifest-sha")
    metrics = _metrics()
    payload = {
        "schema": subject.SCHEMA,
        "experiment": subject.EXPERIMENT,
        "evaluation_manifest": {
            "sha256": "manifest-sha",
            "tail_windows": subject.dense4k.TAIL_WINDOWS,
        },
        "normalizer": {"statistics_sha256": "stats"},
        "start": {"checkpoint_sha256": "old-sim"},
        "seal": {"file_sha256": "old-real"},
        "candidate": {
            "metrics": metrics,
            "metrics_by_aoa": {
                str(aoa): metrics for aoa in subject.dr1.AOA_VALUES},
            "target_audit": {"samples": 716},
        },
    }
    reference = subject.validate_dense12k_reference(payload, manifest)
    assert reference["metrics"] == metrics
    assert reference["start"]["checkpoint_sha256"] == "old-sim"
    assert reference["seal"]["file_sha256"] == "old-real"


def test_gate_accepts_direct_point_signal():
    gate = subject.release_gate(_comparison())
    assert gate["pass"] is True
    assert gate["decision"] == "PASS_B210_DENSE12K_TO_NATIVE_UQ"


def test_gate_accepts_tke_alternative_with_component_floors():
    row = _comparison(point=0.0, tke=0.13)
    row["delta"]["rel_l2_score"] = -0.01
    row["delta"]["mvpe_rel_l2_score"] = -0.01
    assert subject.release_gate(row)["pass"] is True


def test_gate_rejects_bad_aoa_even_when_pooled_improves():
    row = _comparison()
    row["by_aoa"]["20"]["delta"]["point_score"] = -0.081
    row["by_aoa"]["20"]["candidate_non_time_proxy_delta"] = -0.02
    gate = subject.release_gate(row)
    assert gate["pass"] is False
    assert gate["checks"]["aoa_floor"] is False
