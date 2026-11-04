from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "train_b210_domain_joint_loao.py")
SPEC = importlib.util.spec_from_file_location(
    "b210_domain_joint_loao_quick_tests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


SPLIT_SOURCE_ORDER = (
    (10, 2), (0, 3), (5, 1), (20, 4), (15, 0),
    (0, 0), (10, 4), (5, 4), (20, 1), (15, 3),
    (0, 4), (10, 1), (5, 2), (20, 3), (15, 1),
    (0, 2), (10, 3), (5, 0), (20, 0), (15, 4),
    (0, 1), (10, 0), (5, 3), (20, 2), (15, 2),
)

EXPECTED_INNER_TRAIN_FILES = [
    "aoa0_traj3.h5", "aoa5_traj1.h5", "aoa15_traj0.h5",
    "aoa20_traj1.h5", "aoa0_traj4.h5", "aoa5_traj2.h5",
    "aoa20_traj3.h5", "aoa15_traj1.h5", "aoa0_traj2.h5",
    "aoa5_traj0.h5", "aoa20_traj0.h5", "aoa15_traj4.h5",
    "aoa0_traj1.h5", "aoa5_traj3.h5", "aoa20_traj2.h5",
    "aoa15_traj2.h5",
]
EXPECTED_INNER_VAL_FILES = [
    "aoa20_traj4.h5", "aoa0_traj0.h5", "aoa5_traj4.h5",
    "aoa15_traj3.h5",
]
EXPECTED_HELD_FILES = [
    "aoa10_traj2.h5", "aoa10_traj4.h5", "aoa10_traj1.h5",
    "aoa10_traj3.h5", "aoa10_traj0.h5",
]


def split_rows() -> list[dict[str, object]]:
    return [{
        "file": f"aoa{aoa}_traj{index}.h5",
        "aoa": aoa,
        "re": 10_000 + 100 * aoa + index,
        "train_starts": [0, 20],
        "val_boundary": 40,
    } for aoa, index in SPLIT_SOURCE_ORDER]


def tiny_model() -> subject.EnsembleUNet2d:
    config = subject.EnsembleUNet2dConfig(
        base=8, depth=3, groups=4, difference_steps=2,
        ensemble_members=4, uq_hidden_channels=8,
        mean_fluctuation=True, mean_profile_channels=4,
        uq_flow_context=True, residual_quantile_intervals=True,
        utility_refinement=True, utility_refinement_kernel=3,
        temporal_uq_refinement=True, temporal_uq_channels=4,
        temporal_uq_blocks=1, temporal_uq_kernel=3,
        physical_energy_controller=True, physical_energy_member_sync=True,
        physical_energy_hidden_channels=4)
    return subject.EnsembleUNet2d(config)


def real_normalizer() -> subject.smu1.FoldNormalizer:
    statistics = (
        torch.tensor([0.1, -0.2, 0.0]),
        torch.tensor([0.2, -0.1, 0.0]),
        torch.tensor([0.8, 1.1, 1.0]),
        torch.tensor([0.9, 1.2, 1.0]),
    )
    return subject.smu1.FoldNormalizer(statistics, torch.device("cpu"))


def _files(rows) -> list[str]:
    return [str(row["file"]) for row in rows]


def _seal_all(gate: subject.QuickAccessGate, directory: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for run in subject.QUICK_RUNS:
        for kind in subject.ENDPOINT_KINDS:
            path = directory / f"{run}_{kind}.pth"
            path.write_bytes(f"{run}:{kind}".encode("ascii"))
            gate.seal(run, kind, {
                "arm": run,
                "endpoint_kind": kind,
                "path": str(path),
                "file_sha256": subject.dr1.sha256(path),
            })
            paths[f"{run}:{kind}"] = path
    return paths


def test_frozen_sha_split_has_exact_membership_hashes_and_manifest_order():
    rows = split_rows()
    before = deepcopy(rows)
    split = subject.deterministic_stratified_trajectory_split(rows, held_aoa=10)

    assert rows == before
    assert split["selection_key"] == "TTDO-P2-stageB-inner-v1"
    assert _files(split["inner_train_rows"]) == EXPECTED_INNER_TRAIN_FILES
    assert _files(split["inner_val_rows"]) == EXPECTED_INNER_VAL_FILES
    assert _files(split["held_rows"]) == EXPECTED_HELD_FILES
    assert split["source_rows_sha256"] == (
        "73927c2937931aabe68686b7f91f47cb305c0cf1e12a00a40bf12cc90bfe469a")
    assert split["inner_train_rows_sha256"] == (
        "37e962ea695b0b302499d9f0453128d750c9a6fb7b6731403fcd9600d27f04db")
    assert split["inner_val_rows_sha256"] == (
        "0cffc6d83d34748ec82174698a8f3b419fb8f89606b3be043b39ffb4f71481f1")
    assert split["held_rows_sha256"] == (
        "2070736505252a5cd8ebc18d828c0ebaacab329a37d586cb86a81cc204efeb01")
    assert split["split_sha256"] == (
        "31fb4071f3c7be19f67372ee3db44ffd95e3c8866d1543e989be9b6d7980da3d")

    source_files = _files(rows)
    for partition in (EXPECTED_INNER_TRAIN_FILES, EXPECTED_INNER_VAL_FILES,
                      EXPECTED_HELD_FILES):
        assert partition == [name for name in source_files if name in partition]
    train = set(EXPECTED_INNER_TRAIN_FILES)
    val = set(EXPECTED_INNER_VAL_FILES)
    held = set(EXPECTED_HELD_FILES)
    assert not train & val
    assert not train & held
    assert not val & held
    assert train | val | held == set(source_files)
    assert {int(row["aoa"]) for row in split["inner_val_rows"]} == {0, 5, 15, 20}


def test_real_only_events_reuse_exact_real_stream_and_rng_and_have_no_sim():
    updates = 7
    orders = subject.build_quick_event_orders(updates)
    reference = orders[subject.REFERENCE_ARM]

    assert reference["domain_exposures"] == {"sim": 0, "real": updates}
    assert reference["optimizer_updates"] == updates
    assert reference["reference_only"] is True
    assert reference["equal_compute_causal_arm"] is False
    assert all(row["domain"] == "real" for row in reference["rows"])

    def domain_identity(rows):
        return [{key: value for key, value in row.items()
                 if key != "optimizer_step"}
                for row in rows if row["domain"] == "real"]

    expected = domain_identity(reference["rows"])
    for arm in subject.ARMS:
        assert domain_identity(orders[arm]["rows"]) == expected

    rng_draws = {}
    for run in subject.QUICK_RUNS:
        rows = [row for row in orders[run]["rows"] if row["domain"] == "real"]
        run_draws = []
        for row in rows:
            with subject.domain_event_rng(
                    "real", int(row["domain_step"]),
                    torch.device("cpu")) as rng:
                run_draws.append((
                    rng["seed"], rng["contract_fingerprint"],
                    torch.rand(3).tolist(), float(np.random.rand())))
        rng_draws[run] = run_draws
    assert rng_draws[subject.ARMS[0]] == rng_draws[subject.ARMS[1]]
    assert rng_draws[subject.ARMS[0]] == rng_draws[subject.REFERENCE_ARM]


def test_real_only_reference_audit_requires_shared_real_stream_rng_and_no_sim():
    updates = 3
    schedules = {
        "sim": {"sha256": "sim-stream"},
        "real": {"sha256": "real-stream"},
    }
    reports = {}
    for arm in subject.ARMS:
        reports[arm] = {
            "start_state_sha256": "a" * 64,
            "training": {
                "optimizer_updates": 2 * updates,
                "ema_updates": 2 * updates,
                "domain_exposures": {"sim": updates, "real": updates},
                "actual_stream_sha256": {
                    "sim": "sim-stream", "real": "real-stream"},
                "paired_trace_sha256": {
                    "sim": "sim-rng-lr", "real": "real-rng-lr"},
            },
        }
    reports[subject.REFERENCE_ARM] = {
        "start_state_sha256": "a" * 64,
        "training": {
            "optimizer_updates": updates,
            "ema_updates": updates,
            "domain_exposures": {"sim": 0, "real": updates},
            "actual_stream_sha256": {
                "sim": subject.canonical_sha256([]), "real": "real-stream"},
            "paired_trace_sha256": {
                "sim": subject.canonical_sha256([]), "real": "real-rng-lr"},
        },
    }
    assert subject.real_only_reference_audit(reports, schedules, updates)["pass"]
    reports[subject.REFERENCE_ARM]["training"]["paired_trace_sha256"][
        "real"] = "changed-rng"
    assert not subject.real_only_reference_audit(reports, schedules, updates)["pass"]


def test_quick_access_waits_for_six_seals_and_never_allows_held(tmp_path):
    inner_train = [{"file": "train.h5", "aoa": 0, "val_boundary": 40}]
    inner_val = [{"file": "val.h5", "aoa": 5, "val_boundary": 40}]
    held = {"file": "held.h5", "aoa": 10, "val_boundary": 40}
    gate = subject.QuickAccessGate(10, inner_train, inner_val)

    gate.before_open(inner_train[0], "inner_train")
    gate.record_open(inner_train[0], "inner_train")
    with pytest.raises(RuntimeError, match="held target is forbidden"):
        gate.before_open(held, "held")

    seal_index = 0
    for run in subject.QUICK_RUNS:
        for kind in subject.ENDPOINT_KINDS:
            path = tmp_path / f"endpoint_{seal_index}.pth"
            path.write_bytes(f"endpoint-{seal_index}".encode("ascii"))
            seal_index += 1
            gate.seal(run, kind, {
                "arm": run,
                "endpoint_kind": kind,
                "path": str(path),
                "file_sha256": subject.dr1.sha256(path),
            })
            if seal_index < 6:
                with pytest.raises(RuntimeError, match="every raw/EMA endpoint"):
                    gate.before_open(inner_val[0], "inner_val")

    assert seal_index == 6
    assert gate.all_sealed
    with pytest.raises(RuntimeError, match="held target is forbidden"):
        gate.before_open(held, "held")
    gate.before_open(inner_val[0], "inner_val")
    gate.record_open(inner_val[0], "inner_val")
    for run in subject.QUICK_RUNS:
        for kind in subject.ENDPOINT_KINDS:
            gate.record_evaluation(run, kind)
    report = gate.report()
    assert report["all_endpoints_sealed"] is True
    assert len(report["endpoint_seals"]) == 6
    assert report["opened_exact"] == {"inner_train": True, "inner_val": True}
    assert report["held_target_opened"] is False


def test_quick_access_rejects_tampered_seal_before_inner_validation(tmp_path):
    train = [{"file": "train.h5", "aoa": 0, "val_boundary": 40}]
    val = [{"file": "val.h5", "aoa": 5, "val_boundary": 40}]
    gate = subject.QuickAccessGate(10, train, val)
    paths = _seal_all(gate, tmp_path)
    paths[f"{subject.ARMS[0]}:raw"].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="sealed endpoint changed"):
        gate.before_open(val[0], "inner_val")


def test_point_score_uses_public_46_10_9_weights_not_legacy_50_30_20(
        monkeypatch):
    monkeypatch.setattr(
        subject, "_OFFICIAL", SimpleNamespace(score_error=lambda value: value))
    errors = np.asarray([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
    result = subject.point_metrics_from_official_errors(errors)

    public_value = 0.46 * 10.0 + 0.10 * 20.0 + 0.09 * 30.0
    legacy_value = 0.50 * 10.0 + 0.30 * 20.0 + 0.20 * 30.0
    assert subject.POINT_WEIGHTS == {
        "rel_l2_score": 0.46,
        "tke_score": 0.10,
        "mvpe_score": 0.09,
    }
    assert result["point_final_contribution_46_10_9"] == pytest.approx(
        public_value)
    assert result["point_final_contribution_46_10_9"] != pytest.approx(
        legacy_value)
    assert result["point_formula"] == (
        "0.46*Rel-L2 + 0.10*TKE + 0.09*MVPE")


def test_quick_gate_uses_raw_thresholds_and_fails_each_boundary():
    d1_minus_d0 = {
        "rel_l2_score": -0.20,
        "tke_score": 0.0,
        "mvpe_score": 0.0,
        "point_final_contribution_46_10_9": 0.03,
    }
    d1_minus_reference = {
        **d1_minus_d0,
        "point_final_contribution_46_10_9": 0.0,
    }
    passed = subject.stage_b_quick_gate(
        d1_minus_d0, d1_minus_reference)
    assert passed["decision"] == "GO"
    assert passed["pass"] is True
    assert passed["primary_endpoint"] == "raw"
    assert passed["ema_role"] == "secondary_report_only"
    assert passed["thresholds"] == {
        "raw_d1_minus_d0_point_min": 0.03,
        "raw_d1_minus_d0_component_floor": -0.20,
        "raw_d1_minus_dminus1_point_min": 0.0,
    }
    assert passed["stage_c_auto_promotion"] is False
    assert passed["stage_c_authorized"] is False

    variants = []
    below_point = dict(d1_minus_d0)
    below_point["point_final_contribution_46_10_9"] = 0.03 - 1e-12
    variants.append((below_point, d1_minus_reference))
    below_component = dict(d1_minus_d0)
    below_component["tke_score"] = -0.20 - 1e-12
    variants.append((below_component, d1_minus_reference))
    below_reference = dict(d1_minus_reference)
    below_reference["point_final_contribution_46_10_9"] = -1e-12
    variants.append((d1_minus_d0, below_reference))
    for causal, reference in variants:
        failed = subject.stage_b_quick_gate(causal, reference)
        assert failed["decision"] == "NO_GO"
        assert failed["pass"] is False


def test_raw_and_ema_endpoint_metadata_are_strict_loadable_for_tiny_model(
        tmp_path):
    torch.manual_seed(17)
    model = tiny_model().float()
    normalizer = real_normalizer()
    subject.ensemble_train.common.set_model_stats(model, normalizer)
    subject.configure_point_only_trainable(model)
    uq_start_sha256 = subject.dr1.model_state_sha256(
        subject.uq_parameter_state(model))
    ema = subject.ema_lib.FP32EMAShadow(model, subject.EMA_DECAY)
    updates = 3
    for optimizer_step in range(1, updates + 1):
        ema.update_after_optimizer_step(model, optimizer_step=optimizer_step)
    order = {"optimizer_updates": updates, "sha256": "b" * 64}
    training = {"optimizer_updates": updates, "ema_updates": updates}
    start = {"checkpoint_sha256": "a" * 64}

    raw_seal = subject.save_raw_endpoint(
        tmp_path / "raw.pth", model, arm=subject.ARMS[0], held_aoa=10,
        start=start, real_normalizer=normalizer, order=order,
        training=training, uq_start_sha256=uq_start_sha256)
    ema_seal = subject.save_endpoint(
        tmp_path / "ema.pth", model, ema, arm=subject.ARMS[0], held_aoa=10,
        start=start, real_normalizer=normalizer, order=order,
        training=training, uq_start_sha256=uq_start_sha256)

    expected_parameters = sum(parameter.numel() for parameter in model.parameters())
    for kind, seal in (("raw", raw_seal), ("ema", ema_seal)):
        payload = torch.load(seal["path"], map_location="cpu", weights_only=False)
        assert payload["schema"] == subject.ENDPOINT_SCHEMA
        assert payload["arm"] == subject.ARMS[0]
        assert payload["endpoint_kind"] == kind
        assert payload["held_aoa"] == 10
        assert payload["optimizer_updates"] == updates
        assert payload["event_order_sha256"] == "b" * 64
        assert payload["held_target_opened_before_seal"] is False
        assert payload["optimizer_state_saved"] is False
        assert payload["uq_parameter_endpoint_sha256"] == uq_start_sha256
        loaded, report = subject.load_endpoint_model(
            seal, normalizer, torch.device("cpu"),
            expected_parameters=expected_parameters)
        assert report["strict_load"] is True
        assert report["endpoint_kind"] == kind
        assert report["model_state_sha256"] == seal["model_state_sha256"]
        torch.testing.assert_close(loaded.target_std, normalizer.std_target)

    raw_payload = torch.load(
        raw_seal["path"], map_location="cpu", weights_only=False)
    ema_payload = torch.load(
        ema_seal["path"], map_location="cpu", weights_only=False)
    assert raw_payload["endpoint_materialization"]["kind"] == (
        "exact_raw_terminal_model_state")
    assert ema_payload["endpoint_materialization"]["kind"] == (
        "trainable_parameter_ema_plus_exact_raw_frozen_uq")


def test_cli_keeps_legacy_pilot_and_freezes_quick_to_held10_n1000(tmp_path):
    def arguments(stage_arguments, *, held=10, updates=200, suffix="run"):
        return [
            *stage_arguments,
            "--held-aoa", str(held),
            "--domain-updates", str(updates),
            "--start-sim", "start.pth",
            "--start-sim-sha256", "a" * 64,
            "--output-dir", str(tmp_path / suffix),
        ]

    legacy = subject.parse_args(arguments(["--pilot"], suffix="legacy"))
    explicit = subject.parse_args(arguments(
        ["--stage", "pilot"], suffix="explicit"))
    quick = subject.parse_args(arguments(
        ["--stage", "quick"], updates=1_000, suffix="quick"))
    assert legacy.stage == explicit.stage == "pilot"
    assert legacy.pilot is explicit.pilot is True
    assert legacy.held_aoa == explicit.held_aoa == 10
    assert legacy.domain_updates == explicit.domain_updates == 200
    assert quick.stage == "quick"
    assert quick.pilot is False
    assert quick.held_aoa == 10
    assert quick.domain_updates == 1_000

    with pytest.raises(ValueError, match="held AoA 10"):
        subject.parse_args(arguments(
            ["--stage", "quick"], held=5, updates=1_000,
            suffix="bad-held"))
    with pytest.raises(ValueError, match="N=1000/domain"):
        subject.parse_args(arguments(
            ["--stage", "quick"], updates=999, suffix="bad-updates"))
    with pytest.raises(ValueError, match="N=200/domain"):
        subject.parse_args(arguments(
            ["--pilot"], updates=1_000, suffix="bad-pilot"))
