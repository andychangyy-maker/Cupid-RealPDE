from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = EVALUATION_ROOT / "evaluate_c208_q50_mean_point_shift.py"
SPEC = importlib.util.spec_from_file_location("c208_e0_q50", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_q50_unit_sign_and_time_mean_are_exactly_registered():
    q50 = torch.tensor(
        [[[[[1.0, 2.0]]], [[[3.0, 4.0]]]]], dtype=torch.float32)
    std = torch.tensor([0.1, 0.2], dtype=torch.float32)

    bias_tensor = subject.temporal_mean_bias(q50, std)
    bias = bias_tensor.numpy()
    base = np.zeros((1, 2, 1, 1, 2), dtype=np.float32)
    candidate = subject.candidate_point(base, bias, 0.5)

    assert bias.shape == (1, 1, 1, 1, 2)
    assert np.allclose(bias[0, 0, 0, 0], [0.2, 0.6])
    assert np.allclose(candidate[0, :, 0, 0], [[0.1, 0.3], [0.1, 0.3]])
    assert np.array_equal(subject.candidate_point(base, bias, 0.0), base)


def test_alpha_grid_rejects_unregistered_rescue_values():
    base = np.zeros((1, 2, 1, 1, 2), dtype=np.float32)
    bias = np.ones((1, 1, 1, 1, 2), dtype=np.float32)
    for alpha in (-0.25, 0.1, 1.25):
        try:
            subject.candidate_point(base, bias, alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unregistered alpha {alpha} was accepted")


def test_pressure_is_explicitly_constructed_as_zero_without_changing_uv():
    uv = np.arange(80, dtype=np.float32).reshape(1, 2, 4, 5, 2)

    full = subject.pad_zero_pressure(uv)

    assert full.shape == (1, 2, 4, 5, 3)
    assert np.array_equal(full[..., :2], uv)
    assert np.count_nonzero(full[..., 2]) == 0


def _synthetic_bundle(seed: int = 4):
    rng = np.random.default_rng(seed)
    shape = (10, 20, 25, 38, 2)
    target = rng.normal(0.4, 0.08, size=shape).astype(np.float32)
    base = target + rng.normal(0.0, 0.025, size=shape).astype(np.float32)
    bias = rng.normal(
        0.0, 0.006, size=(shape[0], 1, shape[2], shape[3], 2)
    ).astype(np.float32)
    candidates = {
        alpha: subject.candidate_point(base, bias, alpha)
        for alpha in subject.ALPHAS
    }
    lower = base - np.float32(0.06)
    upper = base + np.float32(0.06)
    aoas = np.repeat(np.asarray(subject.AOA_VALUES), 2)
    return base, target, bias, candidates, lower, upper, aoas


def test_absolute_bounds_fix_coverage_width_and_tke_for_time_mean_shift():
    base, target, _bias, candidates, lower, upper, _aoas = _synthetic_bundle()
    baseline = subject.score_bundle(base, target, lower, upper)
    shifted = subject.score_bundle(candidates[1.0], target, lower, upper)

    assert shifted["coverage"] == baseline["coverage"]
    assert shifted["covered_elements"] == baseline["covered_elements"]
    assert shifted["scored_elements"] == baseline["scored_elements"]
    assert shifted["mean_width"] == baseline["mean_width"]
    assert abs(shifted["tke_score"] - baseline["tke_score"]) <= 1e-5
    assert subject.centered_fluctuation_max(base, candidates[1.0]) <= 1e-6


def test_loao_held_target_mutation_cannot_change_its_alpha():
    (_base, target, _bias, candidates,
     lower, upper, aoas) = _synthetic_bundle()
    first = subject.select_loao_alphas(
        candidates, target, lower, upper, aoas)
    held = subject.AOA_VALUES[0]
    mutated = target.copy()
    mutated[aoas == held] *= np.float32(17.0)
    second = subject.select_loao_alphas(
        candidates, mutated, lower, upper, aoas)

    assert first[held]["selected_alpha"] == second[held]["selected_alpha"]
    assert first[held]["rows"] == second[held]["rows"]


def test_unrounded_ties_choose_the_smallest_alpha():
    base, target, _bias, _candidates, lower, upper, _aoas = _synthetic_bundle()
    candidates = {alpha: base.copy() for alpha in subject.ALPHAS}
    selected = subject.select_alpha_on_mask(
        candidates, target, lower, upper,
        np.ones(len(target), dtype=bool))

    assert selected["selected_alpha"] == 0.0
    assert all(row["non_time_shadow_delta"] == 0.0
               for row in selected["rows"])


def test_gate_uses_only_nested_pooled_rows_and_hard_guards():
    pooled_delta = {
        "rel_l2_score": 0.08,
        "tke_score": 0.0,
        "mvpe_score": 0.04,
        "sps_score": 0.04,
        "point_score": 0.048,
        "coverage": 0.0,
        "mean_width": 0.0,
    }
    rows = [
        {"non_time_shadow_delta": value}
        for value in (0.08, 0.07, 0.06, 0.05, -0.005)
    ]
    guards = {"synthetic_guard": True}

    checks = subject.build_gate_checks(
        pooled_delta, rows, guards, elapsed=10.0, gpu_seconds=5.0)
    assert all(checks.values())

    stopped = subject.build_gate_checks(
        pooled_delta, rows, {"synthetic_guard": False},
        elapsed=10.0, gpu_seconds=5.0)
    assert not stopped["synthetic_guard"]
