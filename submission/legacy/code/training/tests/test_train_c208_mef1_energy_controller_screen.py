from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import time

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_c208_mef1_energy_controller_screen as subject  # noqa: E402
from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
    PhysicalEnergyController,
)


def row(name: str, re_value: int, aoa: int, starts=(0,)) -> dict:
    return {
        "file": name,
        "frames": 80,
        "re": re_value,
        "aoa": aoa,
        "train_starts": list(starts),
        "val_boundary": 80,
    }


def metrics(rel=90.0, tke=90.0, mvpe=90.0) -> dict[str, float]:
    return {
        "rel_l2_score": rel,
        "tke_rel_l2_score": tke,
        "mvpe_rel_l2_score": mvpe,
        "point_532_score": 0.5 * rel + 0.3 * tke + 0.2 * mvpe,
    }


def open_row(name: str, aoa: int) -> dict:
    return {"file": name, "aoa": aoa, "val_boundary": 40}


def seal_all(ledger: subject.MEF1AccessLedger, path: Path) -> None:
    for index, endpoint in enumerate(subject.ENDPOINT_ORDER):
        ledger.seal_endpoint(endpoint, path, "a" * 64, f"{index + 1}" * 64)


def seal_internal(
        ledger: subject.MEF1AccessLedger, path: Path, passed: bool) -> None:
    subject.dr1.write_json_atomic(path, {"gate_pass": passed})
    ledger.seal_internal_result(path, subject.dr1.sha256(path), passed)


def test_contract_freezes_point_only_screen_and_no_release():
    assert subject.FORMAL_START_SHA256 == (
        "6674fcc6a1d777e6407b99bff1a14ee0e106c706a4b1cb67a9108e850ae809e3")
    assert subject.CONTRACT["training"] == {
        "seed": 4,
        "batch_size": 8,
        "updates": 1000,
        "optimizer": "AdamW",
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": 5000,
        "scheduler_eta_min": 0.0,
        "ema_decay": 0.999,
        "update_order": "after_optimizer_step_before_scheduler_step",
        "accuracy_weights": [5.0, 3.0, 2.0],
        "dtype": "float32",
        "amp": False,
        "compile": False,
        "losses": ["accuracy_objective_532"],
    }
    assert subject.CONTRACT["evaluation"] == {
        "point_only": True,
        "endpoint_order": ["identity", "raw", "ema"],
        "internal_complete_passes": 3,
        "held_complete_passes_after_internal_go": 3,
        "sps": "not_computed/undefined",
        "non_time": "not_computed/undefined",
        "final": "not_computed/undefined",
    }
    assert subject.CONTRACT["budgets"] == {
        "gpu_seconds": 1800.0,
        "wall_seconds": 2700.0,
        "automatic_retries": 0,
        "maximum_recorded_infra_repair_attempt": 1,
    }
    assert subject.CONTRACT["release_checkpoint_written"] is False
    assert subject.CONTRACT["submission_authorized"] is False


def test_fold_assignment_ranks_full_manifest_before_filtering_held5():
    rows = [
        row("a.h5", 100, 0),
        row("b.h5", 200, 5),
        row("c.h5", 300, 10),
        row("d.h5", 400, 15),
        row("e.h5", 500, 20),
    ]
    frozen = subject.freeze_partitions(rows, verify_frozen=False)
    records = {
        item["file"]: item for item in frozen["report"]["assignments"]}
    assert records["c.h5"]["fold"] == (2 + 2 * 2) % 5
    assert frozen["report"]["full_manifest_ranked_before_held_filter"] is True
    assert all(item["aoa"] != 5 for item in frozen["train_rows"])
    assert all(item["aoa"] != 5 for item in frozen["internal_rows"])
    assert [item["file"] for item in frozen["held_rows"]] == ["b.h5"]


def test_schedule_is_deterministic_global_id_stream_without_bootstrap():
    rows = [
        row("a.h5", 100, 0, (0, 20)),
        row("b.h5", 200, 5, (0,)),
        row("c.h5", 300, 10, (0, 20)),
    ]
    train = [rows[0], rows[2]]
    one = subject.build_batch_schedule(
        rows, train, updates=3, batch_size=2, seed=4,
        verify_frozen=False)
    two = subject.build_batch_schedule(
        rows, train, updates=3, batch_size=2, seed=4,
        verify_frozen=False)
    assert one["rows"] == two["rows"]
    assert one["sha256"] == subject.dr1.canonical_sha256(one["rows"])
    assert all(set(item) == {"sample_indices", "trajectory_ids"}
               for item in one["rows"])
    assert set(index for item in one["rows"]
               for index in item["sample_indices"]).issubset({0, 1, 3, 4})
    assert set(index for item in one["rows"]
               for index in item["trajectory_ids"]) == {0, 2}


def test_access_ledger_seals_then_scores_once_and_keeps_failed_held_closed(
        tmp_path):
    stopped = subject.MEF1AccessLedger(["t.h5"], ["i.h5"], ["h.h5"])
    stopped.before_open(open_row("t.h5", 0), "train")
    stopped.record_open(open_row("t.h5", 0), "train")
    with pytest.raises(RuntimeError, match="three seals"):
        stopped.before_open(open_row("i.h5", 10), "internal")
    seal_all(stopped, tmp_path / "bundle.pth")
    stopped.before_open(open_row("i.h5", 10), "internal")
    stopped.record_open(open_row("i.h5", 10), "internal")
    with pytest.raises(RuntimeError, match="expected internal identity"):
        stopped.record_evaluation("internal", "raw")
    for endpoint in subject.ENDPOINT_ORDER:
        stopped.record_evaluation("internal", endpoint)
    stopped.record_internal_gate(False)
    seal_internal(stopped, tmp_path / "failed_internal.json", False)
    with pytest.raises(RuntimeError, match="before internal GO"):
        stopped.before_open(open_row("h.h5", 5), "held")
    report = stopped.report()
    assert report["held5_open_count"] == 0
    assert report["internal_result_seal"]["gate_pass"] is False
    assert report["evaluation_order"]["internal"] == [
        "identity", "raw", "ema"]

    promoted = subject.MEF1AccessLedger(["t.h5"], ["i.h5"], ["h.h5"])
    promoted.before_open(open_row("t.h5", 0), "train")
    promoted.record_open(open_row("t.h5", 0), "train")
    seal_all(promoted, tmp_path / "bundle.pth")
    promoted.before_open(open_row("i.h5", 10), "internal")
    promoted.record_open(open_row("i.h5", 10), "internal")
    for endpoint in subject.ENDPOINT_ORDER:
        promoted.record_evaluation("internal", endpoint)
    promoted.record_internal_gate(True)
    with pytest.raises(RuntimeError, match="before internal GO"):
        promoted.before_open(open_row("h.h5", 5), "held")
    seal_internal(promoted, tmp_path / "passed_internal.json", True)
    promoted.before_open(open_row("h.h5", 5), "held")
    promoted.record_open(open_row("h.h5", 5), "held")
    for endpoint in subject.ENDPOINT_ORDER:
        promoted.record_evaluation("held", endpoint)
    assert promoted.report()["evaluation_order"]["held"] == [
        "identity", "raw", "ema"]

    tampered = subject.MEF1AccessLedger(["t.h5"], ["i.h5"], ["h.h5"])
    tampered.before_open(open_row("t.h5", 0), "train")
    tampered.record_open(open_row("t.h5", 0), "train")
    seal_all(tampered, tmp_path / "bundle.pth")
    tampered.before_open(open_row("i.h5", 10), "internal")
    tampered.record_open(open_row("i.h5", 10), "internal")
    for endpoint in subject.ENDPOINT_ORDER:
        tampered.record_evaluation("internal", endpoint)
    evidence = tmp_path / "tampered_internal.json"
    seal_internal(tampered, evidence, True)
    evidence.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="before internal GO"):
        tampered.before_open(open_row("h.h5", 5), "held")
    assert tampered.report()["held_authorized"] is False


def test_primary_gate_uses_official_score_deltas_and_strict_positive_aoas():
    identity = metrics()
    candidate = metrics(tke=90.2)
    per_identity = {str(aoa): metrics() for aoa in (0, 10, 15, 20)}
    per_candidate = {str(aoa): metrics(tke=90.2)
                     for aoa in (0, 10, 15, 20)}
    passed = subject.primary_gate(
        candidate, identity,
        per_aoa_candidate=per_candidate,
        per_aoa_identity=per_identity,
        require_per_aoa=True)
    assert passed["pass"] is True
    assert passed["delta"]["tke_rel_l2_score"] == pytest.approx(0.2)
    assert passed["delta"]["point_532_score"] == pytest.approx(0.06)

    per_candidate["15"] = metrics()
    per_candidate["20"] = metrics()
    failed = subject.primary_gate(
        candidate, identity,
        per_aoa_candidate=per_candidate,
        per_aoa_identity=per_identity,
        require_per_aoa=True)
    assert failed["conditions"]["positive_aoas"] is False
    assert failed["pass"] is False


def test_controller_endpoint_bundle_contains_only_three_small_states(tmp_path):
    controller = PhysicalEnergyController(208, 32, 8)
    identity = subject._clone_tensor_state(  # noqa: SLF001
        controller.state_dict())
    raw = subject._clone_tensor_state(identity)  # noqa: SLF001
    ema = subject._clone_tensor_state(identity)  # noqa: SLF001
    raw["body.5.bias"].add_(0.1)
    ema["body.5.bias"].add_(0.05)
    path = tmp_path / "controllers.pth"
    seal = subject.save_endpoint_bundle(
        path,
        identity_state=identity,
        raw_state=raw,
        ema_state=ema,
        ema_metadata={"updates": 1000, "decay": 0.999},
        start={"checkpoint_sha256": subject.FORMAL_START_SHA256},
        normalizer={"sha256": "n" * 64},
        stream_sha256=subject.STREAM_SHA256,
        legacy_state_sha256_value="b" * 64)
    payload = subject.load_endpoint_bundle(path, seal["file_sha256"])
    assert payload["legacy_model_state_saved"] is False
    assert "model_state_dict" not in payload
    assert tuple(payload["controller_states"]) == subject.ENDPOINT_ORDER
    assert sum(value.numel() for state in payload["controller_states"].values()
               for value in state.values()) == 3 * 7105


def test_controller_only_cpu_smoke_updates_ema_and_preserves_legacy(monkeypatch):
    torch.manual_seed(4)
    model = EnsembleUNet2d(EnsembleUNet2dConfig(
        base=4, depth=2, groups=2, difference_steps=1,
        ensemble_members=2, uq_hidden_channels=4,
        mean_fluctuation=True, physical_energy_controller=True,
        physical_energy_hidden_channels=4))
    model.set_normalizer_statistics(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3))
    with torch.no_grad():
        for heads in model.member_point_heads:
            for head in heads:
                head.weight.normal_(std=0.02)
                head.bias.normal_(std=0.01)
    controller_parameters = sum(
        value.numel() for value in model.energy_controller.parameters())
    monkeypatch.setattr(subject, "CONTROLLER_PARAMETERS", controller_parameters)
    trajectories = [torch.randn(60, 8, 40, 2)]
    windows = [
        subject.ensemble_train.common.repro.Window(0, 0),
        subject.ensemble_train.common.repro.Window(0, 20),
    ]
    dataset = subject.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    rows = [
        {"sample_indices": [10], "trajectory_ids": [3]},
        {"sample_indices": [11], "trajectory_ids": [3]},
    ]
    schedule = {
        "batches": [[0], [1]],
        "rows": rows,
        "local_sample_to_global": [10, 11],
        "local_trajectory_to_global": [3],
        "sha256": subject.dr1.canonical_sha256(rows),
        "updates": 2,
        "batch_size": 1,
        "seed": 4,
    }
    normalizer = subject.smu1.FoldNormalizer((
        torch.zeros(3), torch.zeros(3),
        torch.ones(3), torch.ones(3)), torch.device("cpu"))
    identity = subject.controller_state_sha256(subject.controller_state(model))
    deadline = time.perf_counter() + 60.0
    report, ema = subject.train_controller(
        model, dataset, schedule, normalizer, torch.device("cpu"),
        deadline, deadline)
    assert report["updates"] == 2
    assert report["initial_legacy_state_sha256"] == (
        report["final_legacy_state_sha256"])
    assert subject.controller_state_sha256(
        subject.controller_state(model)) != identity
    assert ema.updates == 2


def test_point_evaluator_uses_model_forward_and_never_distribution(tmp_path):
    class PointOnly(torch.nn.Module):
        def forward(self, value):
            return value

        def forward_distribution(self, _value):
            raise AssertionError("distribution path must remain unopened")

    histories = torch.randn(2, 20, 8, 40, 3)
    targets = histories + 0.05
    dataset = torch.utils.data.TensorDataset(histories, targets)
    normalizer = subject.smu1.FoldNormalizer((
        torch.zeros(3), torch.zeros(3),
        torch.ones(3), torch.ones(3)), torch.device("cpu"))
    ledger = subject.MEF1AccessLedger(["t.h5"], ["i.h5"], ["h.h5"])
    ledger.before_open(open_row("t.h5", 0), "train")
    ledger.record_open(open_row("t.h5", 0), "train")
    seal_all(ledger, tmp_path / "bundle.pth")
    ledger.before_open(open_row("i.h5", 10), "internal")
    ledger.record_open(open_row("i.h5", 10), "internal")
    deadline = time.perf_counter() + 60.0
    report = subject.evaluate_point_endpoint(
        PointOnly(), dataset, [0, 10], normalizer, torch.device("cpu"),
        deadline, deadline, ledger, phase="internal", endpoint="identity")
    assert report["forward_api"] == "model(x)"
    assert set(report["metrics"]) == {
        "rel_l2_score", "tke_rel_l2_score", "mvpe_rel_l2_score",
        "point_532_score"}
    assert "native_sps" not in json.dumps(report)


def test_source_has_no_distribution_call_or_submission_path():
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    distribution_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "forward_distribution"
    ]
    assert distribution_calls == []
    lowered = source.lower()
    assert "codabench" not in lowered
    assert ".research" not in lowered
    assert "val_starts" not in lowered


def test_contract_only_needs_no_paths_or_gpu(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(subject.__file__), "--contract-only"])
    subject.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "CONTRACT_ONLY_NO_DATA_NO_GPU"


def test_formal_cli_requires_explicit_execution_and_exact_acknowledgement(
        tmp_path):
    args = argparse.Namespace(
        output_dir=tmp_path / "new",
        clean_manifest=tmp_path / "manifest.json",
        clean_manifest_sha256=subject.FORMAL_MANIFEST_SHA256,
        start_sim=tmp_path / "sim.pth",
        start_sim_sha256=subject.FORMAL_START_SHA256,
        execute_formal_gpu=False,
        acknowledge_protocol="",
    )
    with pytest.raises(ValueError, match="execute-formal-gpu"):
        subject.validate_formal_args(args)
    args.execute_formal_gpu = True
    with pytest.raises(ValueError, match="acknowledgement"):
        subject.validate_formal_args(args)
    args.acknowledge_protocol = subject.FORMAL_ACKNOWLEDGEMENT
    subject.validate_formal_args(args)
