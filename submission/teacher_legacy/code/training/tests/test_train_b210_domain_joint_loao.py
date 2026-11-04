from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "train_b210_domain_joint_loao.py")
SPEC = importlib.util.spec_from_file_location("b210_domain_joint_loao", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


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


def normalizer() -> subject.smu1.FoldNormalizer:
    statistics = (
        torch.tensor([0.1, -0.2, 0.0]),
        torch.tensor([0.2, -0.1, 0.0]),
        torch.tensor([0.8, 1.1, 1.0]),
        torch.tensor([0.9, 1.2, 1.0]),
    )
    return subject.smu1.FoldNormalizer(statistics, torch.device("cpu"))


def test_domain_orders_are_same_events_with_only_cross_domain_order_changed():
    orders = subject.build_domain_event_orders(4)
    d0 = orders[subject.domain_events.ARM_D0]["rows"]
    d1 = orders[subject.domain_events.ARM_D1]["rows"]

    assert [(row["domain"], row["domain_step"]) for row in d0] == [
        ("sim", 1), ("sim", 2), ("sim", 3), ("sim", 4),
        ("real", 1), ("real", 2), ("real", 3), ("real", 4),
    ]
    assert [(row["domain"], row["domain_step"]) for row in d1] == [
        ("sim", 1), ("real", 1), ("sim", 2), ("real", 2),
        ("sim", 3), ("real", 3), ("sim", 4), ("real", 4),
    ]
    for domain in subject.DOMAINS:
        assert [row["seed"] for row in d0 if row["domain"] == domain] == [
            row["seed"] for row in d1 if row["domain"] == domain]
    assert orders[subject.ARMS[0]]["optimizer_updates"] == 8
    assert orders[subject.ARMS[1]]["domain_exposures"] == {
        "sim": 4, "real": 4}
    assert (orders[subject.ARMS[0]]["event_contract_sha256"]
            == orders[subject.ARMS[1]]["event_contract_sha256"])


def test_quick_orders_add_same_real_stream_reference_without_sim_exposure():
    orders = subject.build_quick_event_orders(4)
    reference = orders[subject.REFERENCE_ARM]
    d0_real = [row for row in orders[subject.ARMS[0]]["rows"]
               if row["domain"] == "real"]
    d1_real = [row for row in orders[subject.ARMS[1]]["rows"]
               if row["domain"] == "real"]

    assert [(row["domain"], row["domain_step"], row["seed"])
            for row in reference["rows"]] == [
        (row["domain"], row["domain_step"], row["seed"])
        for row in d0_real] == [
        (row["domain"], row["domain_step"], row["seed"])
        for row in d1_real]
    assert reference["domain_exposures"] == {"sim": 0, "real": 4}
    assert reference["optimizer_updates"] == 4
    assert reference["equal_compute_causal_arm"] is False


def test_domain_local_lr_and_rng_are_arm_order_independent_and_restore_state():
    assert subject.domain_learning_rate("sim", 1, 200) == pytest.approx(2e-4)
    assert subject.domain_learning_rate("real", 200, 200) == pytest.approx(2e-4)
    assert subject.domain_event_seed("real", 7) == (
        subject.domain_events.derive_event_seed(subject.SEED, "real", 6))

    torch.manual_seed(123)
    random.seed(456)
    np.random.seed(789)
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    draws = []
    for _ in range(2):
        with subject.domain_event_rng(
                "real", 7, torch.device("cpu")) as record:
            draws.append((torch.rand(3), random.random(), np.random.rand(), record))
        assert torch.equal(torch.get_rng_state(), torch_state)
        assert random.getstate() == python_state
        restored_numpy = np.random.get_state()
        assert restored_numpy[0] == numpy_state[0]
        assert np.array_equal(restored_numpy[1], numpy_state[1])
        assert restored_numpy[2:] == numpy_state[2:]
    torch.testing.assert_close(draws[0][0], draws[1][0], rtol=0, atol=0)
    assert draws[0][1:] == draws[1][1:]
    assert draws[0][3]["contract_fingerprint"] == (
        subject.domain_events.event_rng_fingerprint(0, "real", 6))


def test_stage_b_split_hashes_membership_per_aoa_but_preserves_source_order():
    rows = []
    for index in range(6):
        for aoa in subject.AOA_VALUES:
            rows.append({
                "file": f"a{aoa}_{index}.h5",
                "aoa": aoa,
                "re": 1000 + 100 * aoa + index,
                "train_starts": list(range(index + 1)),
                "val_boundary": 100,
            })
    split = subject.deterministic_stratified_trajectory_split(rows, held_aoa=10)
    train = split["inner_train_rows"]
    val = split["inner_val_rows"]
    held = split["held_rows"]
    train_files = {row["file"] for row in train}
    val_files = {row["file"] for row in val}
    held_files = {row["file"] for row in held}

    assert not train_files & val_files
    assert not (train_files | val_files) & held_files
    assert [row["file"] for row in train] == [
        row["file"] for row in rows if row["file"] in train_files]
    assert [row["file"] for row in val] == [
        row["file"] for row in rows if row["file"] in val_files]
    for aoa in (0, 5, 15, 20):
        ranked = sorted(
            (subject._split_rank(aoa, f"a{aoa}_{index}.h5"),
             f"a{aoa}_{index}.h5") for index in range(6))
        assert {filename for _digest, filename in ranked[:2]} == {
            row["file"] for row in val if row["aoa"] == aoa}
        summary = split["by_aoa"][str(aoa)]
        assert summary["inner_val_trajectories"] == 2
        assert summary["inner_train_trajectories"] == 4
        assert summary["inner_val_summary"]["sparse_windows"] == sum(
            len(row["train_starts"]) for row in val if row["aoa"] == aoa)
        assert summary["inner_train_summary"]["re_min"] <= (
            summary["inner_train_summary"]["re_mean"])

    reverse = subject.deterministic_stratified_trajectory_split(
        list(reversed(rows)), held_aoa=10)
    assert {row["file"] for row in reverse["inner_train_rows"]} == train_files
    assert {row["file"] for row in reverse["inner_val_rows"]} == val_files
    assert split["split_sha256"] == subject.canonical_sha256({
        key: value for key, value in split.items() if key != "split_sha256"})


def test_fit_only_normalizer_rejects_held_metadata_and_uses_sparse_windows():
    values = torch.arange(40 * 2 * 2 * 2, dtype=torch.float32).reshape(
        40, 2, 2, 2)
    window = subject.ensemble_train.common.repro.Window(0, 0)
    statistics = subject.fit_only_real_statistics(
        [values], [{"aoa": 0, "file": "train.h5"}], [window], held_aoa=10,
        allowed_files=["train.h5"])
    expected_input = values[:20].mean(dim=(0, 1, 2))
    expected_target = values[20:].mean(dim=(0, 1, 2))
    torch.testing.assert_close(statistics[0][:2], expected_input)
    torch.testing.assert_close(statistics[1][:2], expected_target)
    assert statistics[0][2].item() == 0.0
    with pytest.raises(RuntimeError, match="held condition"):
        subject.fit_only_real_statistics(
            [values], [{"aoa": 10, "file": "held.h5"}], [window],
            held_aoa=10, allowed_files=["held.h5"])
    with pytest.raises(RuntimeError, match="inner-train allowlist"):
        subject.fit_only_real_statistics(
            [values], [{"aoa": 0, "file": "inner_val.h5"}], [window],
            held_aoa=10, allowed_files=["train.h5"])


def test_held_target_gate_fails_closed_until_both_verified_seals(tmp_path):
    gate = subject.HeldTargetGate(10)
    fit = {"file": "fit.h5", "aoa": 0}
    held = {"file": "held.h5", "aoa": 10}
    gate.before_open(fit, "fit")
    gate.record_open(fit, "fit")
    with pytest.raises(RuntimeError, match="before both endpoints"):
        gate.before_open(held, "held")

    for arm in subject.ARMS:
        path = tmp_path / f"{arm}.pth"
        path.write_bytes(arm.encode("ascii"))
        gate.seal(arm, {
            "path": str(path), "file_sha256": subject.dr1.sha256(path)})
    gate.before_open(held, "held")
    gate.record_open(held, "held")
    report = gate.report()
    assert report["all_endpoints_sealed"] is True
    assert report["held_target_opened_before_all_seals"] is False
    assert report["held_opened_after_all_seals"] == ["held.h5"]


def test_quick_access_gate_requires_all_raw_ema_seals_and_never_opens_held(
        tmp_path):
    train = {"file": "train.h5", "aoa": 0, "val_boundary": 100}
    val = {"file": "val.h5", "aoa": 5, "val_boundary": 120}
    held = {"file": "held.h5", "aoa": 10, "val_boundary": 140}
    gate = subject.QuickAccessGate(10, [train], [val])
    gate.before_open(train, "inner_train")
    gate.record_open(train, "inner_train")
    with pytest.raises(RuntimeError, match="before every raw/EMA"):
        gate.before_open(val, "inner_val")
    with pytest.raises(RuntimeError, match="held target is forbidden"):
        gate.before_open(held, "held")

    for run in subject.QUICK_RUNS:
        for kind in subject.ENDPOINT_KINDS:
            path = tmp_path / f"{run}_{kind}.pth"
            path.write_bytes(f"{run}:{kind}".encode("ascii"))
            gate.seal(run, kind, {
                "arm": run,
                "endpoint_kind": kind,
                "path": str(path),
                "file_sha256": subject.dr1.sha256(path),
            })
    gate.before_open(val, "inner_val")
    gate.record_open(val, "inner_val")
    for run in subject.QUICK_RUNS:
        for kind in subject.ENDPOINT_KINDS:
            gate.record_evaluation(run, kind)
    with pytest.raises(RuntimeError, match="held target is forbidden"):
        gate.before_open(held, "held")
    report = gate.report()
    assert report["all_endpoints_sealed"] is True
    assert report["held_target_opened"] is False
    assert report["opened_exact"] == {"inner_train": True, "inner_val": True}
    assert len(report["endpoint_seals"]) == 6
    assert len(report["access_events_sha256"]) == 64


def test_point_forward_matches_full_uq_prefix_without_calling_uq_head(monkeypatch):
    torch.manual_seed(5)
    model = tiny_model().eval()
    x = torch.randn(1, 20, 8, 8, 3)
    with torch.no_grad():
        expected_prediction, expected_members = model.forward_uq_training(x)[:2]
        prediction, members = subject.forward_point_training(model, x)
    torch.testing.assert_close(prediction, expected_prediction, rtol=0, atol=0)
    torch.testing.assert_close(members, expected_members, rtol=0, atol=0)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("point-only path must not execute quantile head")

    monkeypatch.setattr(model.quantile_head, "forward", forbidden)
    with torch.no_grad():
        subject.forward_point_training(model, x)


def test_point_only_partition_freezes_uq_and_frozen_params_receive_no_gradients():
    model = tiny_model().eval()
    trainable, report = subject.configure_point_only_trainable(model)
    frozen = subject.frozen_named_parameters(model)
    assert report["trainable_parameters"] == sum(p.numel() for p in trainable)
    assert report["frozen_uq_parameters"] == sum(p.numel() for _, p in frozen)
    assert any(name.startswith("quantile_head.") for name, _ in frozen)
    assert all(parameter.requires_grad for parameter in trainable)
    assert all(not parameter.requires_grad for _, parameter in frozen)

    x = torch.randn(1, 20, 8, 8, 3)
    prediction, members = subject.forward_point_training(model, x)
    (prediction.square().mean() + members.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in trainable)
    assert all(parameter.grad is None for _, parameter in frozen)


def test_domain_loss_contract_uses_historical_mixtures_without_real_warmup():
    assert subject.DOMAIN_LOSS == {
        "sim": {
            "point_weight": 1.0, "member_weight": 0.25,
            "member_mode": "mse", "accuracy_weights": (1.0, 4.0, 1.0),
        },
        "real": {
            "point_weight": 0.25, "member_weight": 1.0,
            "member_mode": "official_accuracy",
            "accuracy_weights": (5.0, 3.0, 2.0),
        },
    }
    args = subject._loss_args()
    assert args.point_loss_mode == "official_accuracy"
    assert args.mse_weight == 0.25
    assert args.mse_warmup_updates == 0


def test_endpoint_seal_has_real_buffers_and_unchanged_uq_parameters(tmp_path):
    model = tiny_model()
    subject.ensemble_train.common.set_model_stats(model, normalizer())
    _trainable, _report = subject.configure_point_only_trainable(model)
    uq_sha = subject.dr1.model_state_sha256(subject.uq_parameter_state(model))
    ema = subject.ema_lib.FP32EMAShadow(model, subject.EMA_DECAY)
    # Exercise the real failure mode: an unchanged FP32 value can drift under
    # hundreds of nominally idempotent full-shadow EMA updates.
    for update in range(1, 401):
        ema.update_after_optimizer_step(model, optimizer_step=update)
    raw_path = tmp_path / "raw.pth"
    raw_seal = subject.save_raw_endpoint(
        raw_path, model, arm=subject.ARMS[0], held_aoa=10,
        start={"checkpoint_sha256": "a" * 64},
        real_normalizer=normalizer(),
        order={"optimizer_updates": 400, "sha256": "b" * 64},
        training={"optimizer_updates": 400},
        uq_start_sha256=uq_sha)
    ema_path = tmp_path / "ema.pth"
    ema_seal = subject.save_endpoint(
        ema_path, model, ema, arm=subject.ARMS[0], held_aoa=10,
        start={"checkpoint_sha256": "a" * 64},
        real_normalizer=normalizer(),
        order={"optimizer_updates": 400, "sha256": "b" * 64},
        training={"optimizer_updates": 400},
        uq_start_sha256=uq_sha)
    raw_payload = torch.load(
        raw_path, map_location="cpu", weights_only=False)
    payload = torch.load(ema_path, map_location="cpu", weights_only=False)
    assert raw_seal["file_sha256"] == subject.dr1.sha256(raw_path)
    assert ema_seal["file_sha256"] == subject.dr1.sha256(ema_path)
    assert raw_seal["endpoint_kind"] == "raw"
    assert ema_seal["endpoint_kind"] == "ema"
    assert raw_payload["endpoint_materialization"]["kind"] == (
        "exact_raw_terminal_model_state")
    assert payload["held_target_opened_before_seal"] is False
    assert payload["uq_parameter_start_sha256"] == uq_sha
    assert payload["uq_parameter_end_sha256"] == uq_sha
    assert payload["uq_parameter_endpoint_sha256"] == uq_sha
    assert payload["endpoint_materialization"]["kind"] == (
        "trainable_parameter_ema_plus_exact_raw_frozen_uq")
    frozen_names = payload["endpoint_materialization"][
        "frozen_parameter_names"]
    frozen_endpoint = {
        name: payload["model_state_dict"][name] for name in frozen_names}
    assert subject.dr1.model_state_sha256(frozen_endpoint) == uq_sha
    torch.testing.assert_close(
        payload["model_state_dict"]["target_std"],
        normalizer().std_target)
    expected_parameters = sum(value.numel() for value in model.parameters())
    for seal in (raw_seal, ema_seal):
        loaded, report = subject.load_endpoint_model(
            seal, normalizer(), torch.device("cpu"),
            expected_parameters=expected_parameters)
        assert report["strict_load"] is True
        assert report["optimizer_updates"] == 400
        assert report["event_order_sha256"] == "b" * 64
        del loaded
    assert subject.EXPECTED_B210_PARAMETERS == 178_705_575
    with pytest.raises(RuntimeError, match="parameter count"):
        subject.load_endpoint_model(
            raw_seal, normalizer(), torch.device("cpu"),
            expected_parameters=expected_parameters + 1)


def test_paired_arm_audit_requires_same_stream_start_rng_and_exact_counts():
    schedules = {domain: {"sha256": domain * 16} for domain in subject.DOMAINS}
    reports = {}
    for arm in subject.ARMS:
        reports[arm] = {
            "start_state_sha256": "a" * 64,
            "training": {
                "optimizer_updates": 8,
                "ema_updates": 8,
                "domain_exposures": {"sim": 4, "real": 4},
                "actual_stream_sha256": {
                    domain: schedules[domain]["sha256"]
                    for domain in subject.DOMAINS},
                "paired_trace_sha256": {"sim": "s", "real": "r"},
            },
        }
    assert subject.paired_arm_audit(reports, schedules, 4)["pass"]
    reports[subject.ARMS[1]]["training"]["paired_trace_sha256"]["real"] = "x"
    assert not subject.paired_arm_audit(reports, schedules, 4)["pass"]


def test_real_only_reference_audit_locks_real_stream_and_zero_sim_exposure():
    schedules = {domain: {"sha256": domain * 16}
                 for domain in subject.DOMAINS}
    reports = {}
    for arm in subject.ARMS:
        reports[arm] = {
            "start_state_sha256": "a" * 64,
            "training": {
                "optimizer_updates": 8,
                "ema_updates": 8,
                "domain_exposures": {"sim": 4, "real": 4},
                "actual_stream_sha256": {
                    domain: schedules[domain]["sha256"]
                    for domain in subject.DOMAINS},
                "paired_trace_sha256": {"sim": "s", "real": "r"},
            },
        }
    reports[subject.REFERENCE_ARM] = {
        "start_state_sha256": "a" * 64,
        "training": {
            "optimizer_updates": 4,
            "ema_updates": 4,
            "domain_exposures": {"sim": 0, "real": 4},
            "actual_stream_sha256": {
                "sim": subject.canonical_sha256([]),
                "real": schedules["real"]["sha256"],
            },
            "paired_trace_sha256": {
                "sim": subject.canonical_sha256([]), "real": "r"},
        },
    }
    assert subject.real_only_reference_audit(reports, schedules, 4)["pass"]
    reports[subject.REFERENCE_ARM]["training"]["domain_exposures"]["sim"] = 1
    assert not subject.real_only_reference_audit(reports, schedules, 4)["pass"]


def test_official_metric_mapping_uses_tke_mvpe_names_and_46_10_9_gate():
    errors = np.asarray([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]])
    metrics = subject.point_metrics_from_official_errors(errors)
    assert "tke_score" in metrics and "tke_rel_l2_score" not in metrics
    assert "mvpe_score" in metrics and "mvpe_rel_l2_score" not in metrics
    expected = (
        0.46 * metrics["rel_l2_score"]
        + 0.10 * metrics["tke_score"]
        + 0.09 * metrics["mvpe_score"])
    assert metrics["point_final_contribution_46_10_9"] == pytest.approx(expected)

    causal_delta = {
        "rel_l2_score": 0.01,
        "tke_score": -0.20,
        "mvpe_score": 0.51,
        "point_final_contribution_46_10_9": 0.03,
    }
    reference_delta = {**causal_delta,
                       "point_final_contribution_46_10_9": 0.0}
    gate = subject.stage_b_quick_gate(causal_delta, reference_delta)
    assert gate["decision"] == "GO"
    assert gate["primary_endpoint"] == "raw"
    assert gate["ema_role"] == "secondary_report_only"
    assert gate["stage_c_auto_promotion"] is False
    reference_delta["point_final_contribution_46_10_9"] = -1e-9
    assert not subject.stage_b_quick_gate(causal_delta, reference_delta)["pass"]


def test_point_evaluator_reports_pooled_and_per_aoa_official_metrics(monkeypatch):
    histories = torch.randn(3, 20, 32, 64, 3)
    targets = torch.randn(3, 20, 32, 64, 3)
    targets[..., 2] = 0.0
    dataset = torch.utils.data.TensorDataset(histories, targets)

    class IdentityPoint(torch.nn.Module):
        pass

    def fake_point(_model, value):
        return torch.zeros_like(value), torch.empty(0, device=value.device)

    monkeypatch.setattr(subject, "forward_point_training", fake_point)
    report = subject.evaluate_point_endpoint(
        IdentityPoint(), dataset, [0, 5, 0], normalizer(),
        torch.device("cpu"), deadline=float("inf"))
    assert report["metrics"]["samples"] == 3
    assert report["per_aoa_metrics"]["0"]["samples"] == 2
    assert report["per_aoa_metrics"]["5"]["samples"] == 1
    assert set(report["metrics"]).issuperset({
        "rel_l2_score", "tke_score", "mvpe_score",
        "point_final_contribution_46_10_9"})
    assert len(report["ordered_raw_input_sha256"]) == 64
    assert len(report["ordered_raw_target_sha256"]) == 64


def test_cli_supports_legacy_pilot_and_explicit_frozen_stages(tmp_path):
    base = [
        "--held-aoa", "10", "--domain-updates", "200",
        "--start-sim", "start.pth", "--start-sim-sha256", "a" * 64,
        "--output-dir", str(tmp_path / "output"),
    ]
    legacy = subject.parse_args(["--pilot", *base])
    assert legacy.stage == "pilot" and legacy.pilot is True
    explicit = subject.parse_args(["--stage", "pilot", *base])
    assert explicit.stage == "pilot" and explicit.domain_updates == 200
    with pytest.raises(ValueError, match="N=200"):
        subject.parse_args([
            "--stage", "pilot", "--held-aoa", "10",
            "--domain-updates", "199", "--start-sim", "start.pth",
            "--start-sim-sha256", "a" * 64,
            "--output-dir", str(tmp_path / "bad-pilot")])
    quick = subject.parse_args([
        "--stage", "quick", "--held-aoa", "10",
        "--domain-updates", "1000", "--start-sim", "start.pth",
        "--start-sim-sha256", "a" * 64,
        "--output-dir", str(tmp_path / "quick")])
    assert quick.stage == "quick" and quick.pilot is False
    assert quick.domain_updates == 1000
    with pytest.raises(ValueError, match="N=1000"):
        subject.parse_args([
            "--stage", "quick", "--held-aoa", "10",
            "--domain-updates", "200", "--start-sim", "start.pth",
            "--start-sim-sha256", "a" * 64,
            "--output-dir", str(tmp_path / "bad-quick")])
