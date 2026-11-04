from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_c208_ema999_tke_gain_clean81_held15 as subject  # noqa: E402


def comparison(*, tke=0.1, point=0.1, rel=0.0, mvpe=0.0,
               sps=0.0, exact=0.1):
    return {
        "delta": {
            "rel_l2_score": rel,
            "tke_rel_l2_score": tke,
            "mvpe_rel_l2_score": mvpe,
            "point_score": point,
            "native_sps_score": sps,
        },
        "exact_non_time_final_delta": exact,
    }


def passing_transport():
    return {"pass": True}


def test_contract_freezes_held15_same_run_and_one_unclipped_gain():
    assert subject.FORMAL_START_SHA256 == (
        "6674fcc6a1d777e6407b99bff1a14ee0e106c706a4b1cb67a9108e850ae809e3")
    assert subject.HELD_AOA == 15
    assert subject.CONTRACT["data"]["fit_windows"] == 2000
    assert subject.CONTRACT["data"]["held_windows"] == 544
    assert subject.CONTRACT["training"] == {
        "updates": 4000,
        "batch_size": 8,
        "seed": 4,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "scheduler_t_max": 5000,
        "grad_clip": 1.0,
    }
    assert subject.CONTRACT["ema"]["decay"] == 0.999
    assert subject.CONTRACT["ema"]["decay_sweep"] is False
    assert subject.CONTRACT["gain"]["grid"] is False
    assert subject.CONTRACT["gain"]["range_clip"] is False
    assert subject.CONTRACT["gain"]["objective_improvement_atol"] == 1e-9
    assert subject.CONTRACT["gate"]["primary_vs_raw_mvpe_floor"] == -0.05
    assert subject.CONTRACT["wall_budget_seconds"] == 3600.0


def test_official_float32_tke_field_matches_frozen_scorer():
    rng = np.random.default_rng(208)
    value = rng.normal(size=(3, 5, 4, 6, 3)).astype(np.float32)
    expected = subject.smu1.full.OFFICIAL.kinetic_energy(value[..., :2])
    actual = subject.official_float32_tke_field(value)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_target_and_mask_hash_matches_inherited_held_scorer_contract():
    target = torch.arange(2 * 3 * 2 * 2 * 3, dtype=torch.float32).reshape(
        2, 3, 2, 2, 3)
    mask = target[..., :2].ne(0.0).to(torch.uint8).contiguous()
    actual_target = hashlib.sha256()
    actual_mask = hashlib.sha256()
    subject._update_raw_tensor_hash(actual_target, target)  # noqa: SLF001
    subject._update_raw_tensor_hash(actual_mask, mask)  # noqa: SLF001

    expected_target = hashlib.sha256()
    expected_target.update(str(tuple(target.shape)).encode("ascii"))
    expected_target.update(target.numpy().tobytes(order="C"))
    expected_mask = hashlib.sha256()
    expected_mask.update(str(tuple(mask.shape)).encode("ascii"))
    expected_mask.update(mask.numpy().tobytes(order="C"))
    assert actual_target.hexdigest() == expected_target.hexdigest()
    assert actual_mask.hexdigest() == expected_mask.hexdigest()


def test_complete_held_target_audit_matches_scorer_batch_contract():
    samples = subject.smu1.EVAL_BATCH_SIZE + 2
    histories = torch.zeros(samples, 2, dtype=torch.float32)
    targets = torch.arange(
        samples * 3 * 2 * 2 * 3, dtype=torch.float32).reshape(
            samples, 3, 2, 2, 3)
    dataset = torch.utils.data.TensorDataset(histories, targets)
    actual = subject.audit_ordered_targets(dataset, list(range(samples)))

    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    scored = 0
    for start in range(0, samples, subject.smu1.EVAL_BATCH_SIZE):
        target = targets[start:start + subject.smu1.EVAL_BATCH_SIZE].contiguous()
        mask = target[..., :2].ne(0.0).to(torch.uint8).contiguous()
        subject._update_raw_tensor_hash(target_hasher, target)  # noqa: SLF001
        subject._update_raw_tensor_hash(mask_hasher, mask)  # noqa: SLF001
        scored += int(mask.sum())
    assert actual == {
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
        "samples": samples,
        "scored_elements": scored,
    }


def test_wall_deadline_audit_accepts_exact_boundary_and_flags_overrun(
        monkeypatch):
    checked = iter((3610.0, 3610.000001))
    monkeypatch.setattr(subject.time, "perf_counter", lambda: next(checked))
    boundary = subject.wall_deadline_audit(10.0, 3610.0)
    overrun = subject.wall_deadline_audit(10.0, 3610.0)
    assert boundary["within_wall_budget"] is True
    assert boundary["checked_elapsed_seconds"] == 3600.0
    assert overrun["within_wall_budget"] is False
    assert overrun["checked_elapsed_seconds"] > 3600.0


def test_wls_uses_per_sample_inverse_squared_target_norm():
    predicted = np.asarray([
        [[1.0, 2.0]],
        [[3.0, 4.0]],
    ], dtype=np.float32)
    target = np.asarray([
        [[2.0, 1.0]],
        [[1.0, 2.0]],
    ], dtype=np.float32)
    row = subject.tke_wls_sufficient_statistics(predicted, target)
    x = predicted.reshape(2, -1).astype(np.float64)
    y = target.reshape(2, -1).astype(np.float64)
    weight = 1.0 / np.square(np.linalg.norm(y, axis=1))
    expected_a = np.sum(weight * np.sum(x * x, axis=1))
    expected_b = np.sum(weight * np.sum(x * y, axis=1))
    expected_c = np.sum(weight * np.sum(y * y, axis=1))
    assert row["samples"] == 2
    assert row["A"] == pytest.approx(expected_a, rel=0.0, abs=1e-14)
    assert row["B"] == pytest.approx(expected_b, rel=0.0, abs=1e-14)
    assert row["C"] == pytest.approx(expected_c, rel=0.0, abs=1e-14)
    solution = subject.solve_positive_tke_gain(
        row["A"], row["B"], row["C"])
    assert solution["s"] == pytest.approx(expected_b / expected_a)
    assert solution["g"] == pytest.approx(np.sqrt(expected_b / expected_a))
    assert solution["normal_equation_residual"] == pytest.approx(
        0.0, abs=1e-14)


def test_gain_solver_has_only_nonnegative_feasibility_boundary():
    with pytest.raises(ValueError, match="A must be positive"):
        subject.solve_positive_tke_gain(0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="B must be non-negative"):
        subject.solve_positive_tke_gain(1.0, -1.0, 1.0)
    with pytest.raises(ValueError, match="gain must be positive"):
        subject.solve_positive_tke_gain(1.0, 0.0, 1.0)
    result = subject.solve_positive_tke_gain(1.0, 100.0, 1.0)
    assert result["s"] == 100.0
    assert result["g"] == 10.0
    assert np.isfinite(result["objective_at_s_1"])
    assert np.isfinite(result["objective_at_s"])
    assert result["objective_at_s"] <= (
        result["objective_at_s_1"]
        + result["objective_improvement_atol"])


def test_gain_solver_rejects_nonfinite_objective():
    with pytest.raises(FloatingPointError, match="objective is non-finite"):
        subject.solve_positive_tke_gain(1e308, 1e308, 1e308)


def test_gain_preserves_mean_pressure_and_signed_asymmetric_extents():
    rng = np.random.default_rng(15)
    prediction = rng.normal(size=(2, 20, 3, 4, 3)).astype(np.float32)
    gain = 1.125
    s = gain ** 2
    gained = subject.apply_global_fluctuation_gain(prediction, gain)
    assert np.array_equal(gained[..., 2], prediction[..., 2])

    lower = rng.normal(size=(2, 20, 3, 4, 2)).astype(np.float32)
    width = rng.uniform(0.01, 0.2, size=lower.shape).astype(np.float32)
    upper = -lower + width
    assert np.any(lower < 0.0)
    assert np.any(upper < 0.0)
    assert np.all(lower + upper >= 0.0)

    state = subject.new_transport_audit()
    subject.update_transport_audit(
        state, prediction, gained, lower, upper, s)
    audit = subject.finalize_transport_audit(state, s)
    assert audit["pass"] is True
    assert audit["pressure_bitwise"] is True
    assert audit["lower_extent_bitwise"] is True
    assert audit["upper_extent_bitwise"] is True
    assert audit["width_max_abs_delta"] == 0.0
    assert audit["asymmetry_max_abs_delta"] == 0.0
    assert audit["temporal_mean_max_abs_delta"] <= (
        subject.MEAN_INVARIANCE_ATOL)
    assert audit["observed_s_l2"] == pytest.approx(s, rel=3e-5)


def test_primary_gate_is_gained_vs_raw_with_extra_tke_vs_ema_guard():
    passed = subject.primary_gate(
        comparison(), comparison(tke=0.01, mvpe=1e-6),
        passing_transport())
    assert passed["pass"] is True
    assert passed["decision"] == "GO_EMA999_TKE_GAIN_FULL_CLEAN81_LOAO"

    assert subject.primary_gate(
        comparison(), comparison(tke=0.0), passing_transport())["pass"] is False
    assert subject.primary_gate(
        comparison(tke=0.0), comparison(), passing_transport())["pass"] is False
    assert subject.primary_gate(
        comparison(rel=-0.050001), comparison(),
        passing_transport())["pass"] is False
    assert subject.primary_gate(
        comparison(mvpe=-0.060000), comparison(mvpe=0.0),
        passing_transport())["pass"] is False
    assert subject.primary_gate(
        comparison(), comparison(mvpe=1.0001e-5),
        passing_transport())["pass"] is False
    assert subject.primary_gate(
        comparison(sps=-0.100001), comparison(),
        passing_transport())["pass"] is False


def test_partial_result_preserves_staged_provenance(tmp_path):
    path = tmp_path / "partial_result.json"
    payload = subject.write_partial_result(
        path, "RAW_SCORED",
        raw_endpoint_seal={"file_sha256": "raw"},
        gain_metadata_seal={"file_sha256": "gain"},
        raw_target_audit={"ordered_raw_target_sha256": "target"},
        prefix_access={"held_opened": True})
    staged = subject.staged_file_evidence(path)
    assert staged is not None
    assert staged["file_sha256"] == subject.dr1.sha256(path)
    assert staged["payload"] == payload
    assert json.loads(path.read_text(encoding="utf-8"))["stage"] == "RAW_SCORED"


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--clean-manifest-sha256", "0" * 64], "frozen clean81 manifest"),
        (["--wall-budget-seconds", "3600.001"], "exactly 3600 seconds"),
    ],
)
def test_main_rejects_formal_cli_contract_overrides(
        monkeypatch, tmp_path, extra_args, message):
    output = tmp_path / "new-output"
    monkeypatch.setattr(sys, "argv", [
        str(subject.__file__),
        "--output-dir", str(output),
        "--clean-manifest", str(tmp_path / "manifest.json"),
        "--start-sim", str(tmp_path / "start.pth"),
        *extra_args,
    ])
    with pytest.raises(ValueError, match=message):
        subject.main()
    assert not output.exists()


def test_main_refuses_existing_output_without_overwriting_result(
        monkeypatch, tmp_path):
    output = tmp_path / "existing-output"
    output.mkdir()
    result_path = output / "result.json"
    sentinel = b'{"owner":"pre-existing"}\n'
    result_path.write_bytes(sentinel)
    monkeypatch.setattr(sys, "argv", [
        str(subject.__file__),
        "--output-dir", str(output),
        "--clean-manifest", str(tmp_path / "manifest.json"),
        "--start-sim", str(tmp_path / "start.pth"),
    ])
    with pytest.raises(FileExistsError, match="refusing to reuse output"):
        subject.main()
    assert result_path.read_bytes() == sentinel


def test_main_stop_embeds_staged_post_held_provenance(monkeypatch, tmp_path):
    output = tmp_path / "failed-run"

    def fail_after_held(args):
        args.output_dir.mkdir(parents=True)
        subject.dr1.write_json_atomic(args.output_dir / "run_owner.json", {
            "schema": subject.RUN_OWNER_SCHEMA,
            "experiment": subject.EXPERIMENT,
        })
        subject.dr1.write_json_atomic(
            args.output_dir / "pre_scoring_seal.json",
            {"schema": subject.PRE_SCORING_SCHEMA, "held_prefix_opened": False})
        subject.dr1.write_json_atomic(
            args.output_dir / "global_tke_gain_sealed.json",
            {"schema": subject.GAIN_METADATA_SCHEMA, "g": 1.1})
        subject.write_partial_result(
            args.output_dir / "partial_result.json", "HELD_TARGETS_SEALED",
            raw_endpoint_seal={"file_sha256": "raw"},
            ema_endpoint_seal={"file_sha256": "ema"},
            gain_metadata_seal={"file_sha256": "gain"},
            held_target_audit={
                "ordered_raw_target_sha256": "target",
                "ordered_scored_mask_sha256": "mask",
            },
            prefix_access={"held_opened": True})
        raise RuntimeError("synthetic raw-scorer failure")

    monkeypatch.setattr(subject, "run", fail_after_held)
    monkeypatch.setattr(sys, "argv", [
        str(subject.__file__),
        "--output-dir", str(output),
        "--clean-manifest", str(tmp_path / "manifest.json"),
        "--start-sim", str(tmp_path / "start.pth"),
    ])
    with pytest.raises(RuntimeError, match="synthetic raw-scorer failure"):
        subject.main()
    stop = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert set(stop["staged_provenance"]) == {
        "run_owner", "pre_scoring_seal", "partial_result", "gain_metadata"}
    partial = stop["staged_provenance"]["partial_result"]["payload"]
    assert partial["stage"] == "HELD_TARGETS_SEALED"
    assert partial["held_target_audit"] == {
        "ordered_raw_target_sha256": "target",
        "ordered_scored_mask_sha256": "mask",
    }
    assert partial["prefix_access"]["held_opened"] is True


def test_source_seals_gain_before_held_open_and_never_submits():
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run")
    text = ast.get_source_segment(source, function)
    raw_seal = text.index('access.seal(\n        "control"')
    ema_seal = text.index('access.seal(\n        "candidate"')
    gain_seal = text.index("gain_seal = save_gain_metadata")
    pre_scoring_seal = text.index(
        "dr1.write_json_atomic(pre_scoring_path, pre_scoring_payload)")
    held_open = text.index(
        "held_trajectories, held_metadata, held_windows = dr1.load_prefix_rows")
    held_open_staged = text.index('partial_path, "HELD_OPENED"')
    held_dataset = text.index(
        "held_dataset = ensemble_train.common.repro.WindowDataset")
    held_target_audit = text.index(
        "held_target_audit = audit_ordered_targets")
    held_target_staged = text.index('partial_path, "HELD_TARGETS_SEALED"')
    raw_endpoint_load = text.index("raw_model = load_endpoint")
    raw_scored_staged = text.index('partial_path, "RAW_SCORED"')
    raw_cleanup = text.index("del raw_model")
    all_arms_staged = text.index(
        'partial_path, "ALL_ARMS_SCORED_UNVERIFIED"')
    ema_cleanup = text.index("del held_ema")
    pre_gate_deadline = text.index(
        "pre_gate_deadline = wall_deadline_audit")
    gate = text.index("gate = primary_gate")
    pre_return_deadline = text.index(
        "pre_return_deadline = wall_deadline_audit")
    result_return = text.rindex("return {")
    assert raw_seal < ema_seal < gain_seal < pre_scoring_seal < held_open
    assert held_open < held_open_staged < held_dataset
    assert (held_dataset < held_target_audit < held_target_staged
            < raw_endpoint_load)
    assert raw_scored_staged < raw_cleanup
    assert all_arms_staged < ema_cleanup
    assert all_arms_staged < pre_gate_deadline < gate
    assert gate < pre_return_deadline < result_return
    assert '"HELD_LOAD_FAILED"' in text
    assert '"RAW_SCORING_FAILED"' in text
    assert '"EMA_GAIN_SCORING_FAILED"' in text
    assert '"ALL_ARMS_SCORED_UNVERIFIED"' in text
    assert '"TARGET_IDENTITY_FAILED"' in text
    lowered = source.lower()
    assert "codabench" not in lowered
    assert "fixed725" not in lowered
    assert "val_starts" not in lowered
    assert subject.CONTRACT["release_checkpoint_written"] is False
    assert subject.CONTRACT["submission_authorized"] is False
