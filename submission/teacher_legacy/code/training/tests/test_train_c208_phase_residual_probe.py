from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "train_c208_phase_residual_probe.py")
SPEC = importlib.util.spec_from_file_location("c208_phase_residual_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_contract_freezes_full_official_coefficient_mapping_and_budget():
    assert subject.FINAL_COEFFICIENTS == {
        "rel_l2_score": 0.47201126,
        "tke_rel_l2_score": 0.08586562,
        "mvpe_rel_l2_score": 0.09410214,
        "time_score": 0.10102825,
        "native_sps_score": 0.24699273,
    }
    assert sum(subject.FINAL_COEFFICIENTS.values()) == pytest.approx(1.0)
    assert subject.MAX_HEAD_PARAMETERS == 20_000
    assert subject.EXPERIMENT_CONTRACT["identity_guards"]["mvpe"] == (
        "official MVPE score absolute delta <= 5e-4")
    assert subject.EXPERIMENT_CONTRACT["identity_guards"]["bounds"].endswith(
        "unchanged")


def test_phase_maps_are_twelve_maps_and_resolve_temporal_phase():
    history = torch.zeros(2, 20, 8, 10, 3)
    step = torch.arange(8, dtype=torch.float32)
    wave = torch.sin(2.0 * torch.pi * step / 8.0)
    history[0, -8:, ..., 0] = wave[:, None, None]
    history[1, -8:, ..., 0] = torch.roll(wave, shifts=2)[:, None, None]

    maps = subject.phase_maps(history)

    assert maps.shape == (2, 12, 8, 10)
    assert torch.isfinite(maps).all()
    # u/mode-1 sine is channel 1 in the frozen field/mode/basis order.
    assert maps[0, 1].abs().mean() > 1.0
    assert not torch.equal(maps[0], maps[1])


def test_parameter_matched_control_zero_identity_and_head_under_budget():
    candidate, control = subject.new_parameter_matched_heads(seed=7)
    candidate_parameters = sum(p.numel() for p in candidate.parameters())
    control_parameters = sum(p.numel() for p in control.parameters())
    assert candidate_parameters == control_parameters
    assert candidate_parameters == subject.HEAD_PARAMETER_COUNT == 8_456
    assert candidate_parameters <= subject.MAX_HEAD_PARAMETERS
    assert all(torch.equal(a, b) for a, b in zip(
        candidate.state_dict().values(), control.state_dict().values()))

    feature = torch.randn(1, subject.FEATURE_CHANNELS, 32, 64)
    phase = torch.randn(1, subject.PHASE_CHANNELS, 32, 64)
    assert torch.equal(
        candidate(feature, phase),
        torch.zeros(1, subject.OUTPUT_STEPS, 32, 64, 2))


def test_formal_head_inputs_quantize_feature_and_phase_identically_to_fp16():
    feature = torch.tensor([0.1234567], dtype=torch.float32)
    phase = torch.tensor([0.7654321], dtype=torch.bfloat16)

    formal_feature, formal_phase = subject.formal_head_inputs(feature, phase)

    assert formal_feature.dtype == torch.float16
    assert formal_phase.dtype == torch.float16
    assert torch.equal(formal_feature, feature.to(torch.float16))
    assert torch.equal(formal_phase, phase.to(torch.float16))
    assert subject.EXPERIMENT_CONTRACT["head"]["input_dtype"].startswith(
        "feature and phase explicitly quantized to FP16")


def test_process_local_cache_is_fp16_finite_and_order_bound():
    samples = 2
    cache = subject.CachedProbeDataset(
        torch.zeros(
            samples, subject.FEATURE_CHANNELS, 32, 64, dtype=torch.float16),
        torch.zeros(
            samples, subject.PHASE_CHANNELS, 32, 64, dtype=torch.float16),
        torch.zeros(
            samples, subject.OUTPUT_STEPS, 32, 64, 2,
            dtype=torch.float16))
    report = subject.validate_cache_tensors(cache, samples)
    assert report["persisted"] is False
    assert all(row == {
        "shape": row["shape"], "dtype": "float16", "finite": True}
        for row in report["tensors"].values())

    cache.phase[0, 0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="non-finite"):
        subject.validate_cache_tensors(cache, samples)

    metadata = [{"file": "a.h5"}, {"file": "b.h5"}]
    windows = [
        SimpleNamespace(trajectory=0, start=0),
        SimpleNamespace(trajectory=1, start=20)]
    forward = subject.ordered_window_key_digest(windows, metadata)
    reverse = subject.ordered_window_key_digest(list(reversed(windows)), metadata)
    assert forward != reverse
    assert "never persisted" in subject.EXPERIMENT_CONTRACT["validation"]["cache"]


def test_zero_mean_full_field_and_bound_identity_are_hard_guards():
    head = subject.PhaseResidualHead()
    with torch.no_grad():
        head.body[-1].bias.copy_(torch.linspace(-0.4, 0.4, 40))
    feature = torch.randn(2, subject.FEATURE_CHANNELS, 32, 64)
    phase = torch.randn(2, subject.PHASE_CHANNELS, 32, 64)
    correction = head(feature, phase)
    assert correction.mean(dim=1).abs().max() < 2e-7
    # No metric-specific spatial holes: former MVPE probe locations are editable.
    assert correction[:, :, 16, 13].abs().max() > 0.0

    baseline = torch.randn(2, 20, 32, 64, 3)
    lower = torch.rand(2, 20, 32, 64, 2)
    upper = torch.rand(2, 20, 32, 64, 2)
    corrected, lower_after, upper_after = subject.corrected_distribution(
        baseline, lower, upper, correction)
    subject.assert_identity_guards(
        baseline, corrected, lower, upper, lower_after, upper_after)
    assert lower_after is lower
    assert upper_after is upper
    assert torch.equal(corrected[..., 2], baseline[..., 2])
    assert not torch.equal(corrected[..., :2], baseline[..., :2])


def test_train_side_loao_partitions_exclude_held_and_cover_2576_once():
    labels = [aoa for aoa in subject.AOA_VALUES for _ in range(3)]
    partitions = subject.train_side_loao_partitions(labels)
    evaluated = []
    for held, partition in partitions.items():
        assert set(partition["fit"]).isdisjoint(partition["evaluate"])
        assert all(labels[index] != held for index in partition["fit"])
        assert all(labels[index] == held for index in partition["evaluate"])
        evaluated.extend(partition["evaluate"])
    assert sorted(evaluated) == list(range(len(labels)))


def test_train_split_matches_common_manifest_without_prebuilding_tail_windows():
    metadata = [
        {"file": "a.h5", "frames": 200, "re": 5000, "aoa": 0},
        {"file": "b.h5", "frames": 220, "re": 8000, "aoa": 5},
    ]
    train, manifest, tail_count = subject.make_train_side_split(metadata)
    expected_train, expected_tail, expected_manifest = (
        subject.common.repro.make_split(metadata, 20, 0.2))

    assert train == expected_train
    assert manifest == expected_manifest
    assert tail_count == len(expected_tail)
    assert subject.tail_windows_from_manifest(manifest) == expected_tail


def _metrics(rel_delta: float) -> dict[str, float]:
    return {
        "rel_l2_score": 90.0 + rel_delta,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 95.0,
        "native_sps_score": 0.0,
    }


def _passing_row() -> dict:
    return {
        "samples": 100,
        "baseline": _metrics(0.0),
        "candidate": _metrics(0.18),
        "control": _metrics(0.12),
        "shuffle": _metrics(0.12),
        "mvpe_invariant": True,
        "bounds_bitwise": True,
        "sps_source": "fixed_delta_zero_no_native_uq",
        "max_abs_correction_temporal_mean": 0.0,
    }


def test_loao_and_terminal_gates_require_phase_increment_null_and_guards():
    rows = [_passing_row() for _ in subject.AOA_VALUES]
    gate = subject.loao_gate(rows)
    assert gate["pass"] is True
    assert gate["decision"] == "GO_PHASE_PROBE_TERMINAL_725"

    null_failure = [_passing_row() for _ in subject.AOA_VALUES]
    for row in null_failure:
        row["shuffle"] = _metrics(0.18)
    assert subject.loao_gate(null_failure)["pass"] is False

    mvpe_failure = [_passing_row() for _ in subject.AOA_VALUES]
    mvpe_failure[2]["mvpe_invariant"] = False
    assert subject.loao_gate(mvpe_failure)["pass"] is False

    mean_failure = [_passing_row() for _ in subject.AOA_VALUES]
    mean_failure[1]["max_abs_correction_temporal_mean"] = 3e-6
    assert subject.loao_gate(mean_failure)["pass"] is False

    terminal_row = _passing_row()
    terminal_row["sps_source"] = "promoted_all_tail_refit"
    assert subject.terminal_gate(terminal_row, latency_ratio=1.02)["pass"]
    assert not subject.terminal_gate(
        terminal_row, latency_ratio=1.0200001)["pass"]


def test_tail_is_never_constructed_before_loao_go_and_runs_once_after():
    calls = {"factory": 0, "evaluate": 0}

    def factory():
        calls["factory"] += 1
        return SimpleNamespace(name="tail725")

    def evaluate(tail):
        calls["evaluate"] += 1
        return {"name": tail.name}

    assert subject.run_terminal_once(
        {"pass": False}, factory, evaluate) is None
    assert calls == {"factory": 0, "evaluate": 0}
    assert subject.run_terminal_once(
        {"pass": True}, factory, evaluate) == {"name": "tail725"}
    assert calls == {"factory": 1, "evaluate": 1}

    attempted = subject.tail_access_report(1, 1, 0)
    assert attempted == {
        "tail_dataset_constructed": True,
        "tail_constructions": 1,
        "tail_evaluation_attempts": 1,
        "tail_evaluations_completed": 0,
    }
    assert subject.EXPERIMENT_CONTRACT["validation"][
        "raw_frame_residency"].startswith("legacy load_trajectories preloads")


def test_shuffle_null_is_a_derangement_and_rejects_singleton_batch():
    phase = torch.arange(3 * 12, dtype=torch.float32).view(3, 12, 1, 1)
    shuffled = subject.derange_phase_batch(phase)
    assert torch.equal(shuffled[0], phase[-1])
    assert all(not torch.equal(shuffled[index], phase[index]) for index in range(3))
    with pytest.raises(ValueError, match="at least two"):
        subject.derange_phase_batch(phase[:1])


def test_far_stride_null_sources_are_not_adjacent_held_positions():
    indices = list(range(101))
    batches = subject.far_stride_batches(indices, batch_size=16)

    assert sorted(index for batch in batches for index in batch) == indices
    assert all(2 <= len(batch) <= 16 for batch in batches)
    for batch in batches:
        sources = [batch[-1], *batch[:-1]]  # torch.roll(..., shifts=1)
        assert all(abs(target - source) > 1
                   for target, source in zip(batch, sources))


def test_split_manifest_validation_uses_canonical_json_not_raw_file_sha(
        tmp_path, monkeypatch):
    point = tmp_path / "point.pth"
    stats = tmp_path / "stats.pt"
    manifest = tmp_path / "split.json"
    point.write_bytes(b"point")
    stats.write_bytes(b"stats")
    manifest.write_text('[ { "b": 2, "a": 1 } ]\n', encoding="utf-8")

    def fake_file_sha(path):
        if path == point:
            return subject.FORMAL_C208_SHA256
        if path == stats:
            return subject.FORMAL_REAL_STATS_SHA256
        return "f" * 64  # Deliberately not the canonical manifest SHA.

    monkeypatch.setattr(subject, "file_sha256", fake_file_sha)
    monkeypatch.setattr(
        subject.c208, "canonical_sha256",
        lambda value: subject.FORMAL_SPLIT_MANIFEST_SHA256)
    args = SimpleNamespace(
        num_workers=0, point_checkpoint=point,
        all_tail_refit=tmp_path / "not-read-before-gate.pth",
        split_manifest=manifest, real_stats=stats)

    result = subject.validate_formal_inputs(args)

    assert result["split_manifest_canonical"] == (
        subject.FORMAL_SPLIT_MANIFEST_SHA256)
    assert result["split_manifest_file_sha256"] == "f" * 64
    assert result["split_manifest"] == [{"a": 1, "b": 2}]


def test_refit_may_change_only_registered_uq_state():
    point = {
        "model_state_dict": {
            "decoder.weight": torch.tensor([1.0]),
            "uq_fusion.weight": torch.tensor([2.0]),
        }}
    valid_refit = {
        "model_state_dict": {
            "decoder.weight": torch.tensor([1.0]),
            "uq_fusion.weight": torch.tensor([9.0]),
        }}
    subject.assert_refit_point_state_identity(point, valid_refit)

    invalid_refit = {
        "model_state_dict": {
            "decoder.weight": torch.tensor([1.1]),
            "uq_fusion.weight": torch.tensor([9.0]),
        }}
    with pytest.raises(RuntimeError, match="changed frozen point"):
        subject.assert_refit_point_state_identity(point, invalid_refit)
