from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = EVALUATION_ROOT / "evaluate_c208_refinement_aware_selection.py"
SPEC = importlib.util.spec_from_file_location("c208_rus1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def stack(values: tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, 1, -1)


def test_refined_selection_equals_raw_selection_without_refinement():
    lower = stack((0.004, 0.015, 0.020))
    upper = stack((0.006, 0.015, 0.020))
    valid = torch.ones_like(lower, dtype=torch.bool)
    masses = torch.tensor((0.30, 0.55, 0.80), dtype=torch.float32)
    raw_width = lower + upper

    baseline = subject.uc1.identity_select(raw_width, valid, masses)
    selected, utility = subject.refined_utility_select(
        lower, upper, valid, masses)

    assert torch.equal(selected, baseline)
    chosen = subject.uc1.gather_candidate(utility, selected)
    assert torch.equal(chosen, utility.max(dim=-1).values)


def test_refinement_aware_argmax_can_switch_and_is_proxy_nondecreasing():
    masses = torch.tensor((0.50, 0.80), dtype=torch.float32)
    valid = torch.ones((1, 1, 1, 1, 1, 2), dtype=torch.bool)
    raw_width = stack((0.01, 0.04))
    baseline = subject.uc1.identity_select(raw_width, valid, masses)
    assert int(baseline.item()) == 0

    refined_lower = stack((0.039, 0.002))
    refined_upper = stack((0.001, 0.003))
    selected, utility = subject.refined_utility_select(
        refined_lower, refined_upper, valid, masses)
    assert int(selected.item()) == 1
    baseline_proxy = subject.uc1.gather_candidate(utility, baseline)
    candidate_proxy = subject.uc1.gather_candidate(utility, selected)
    assert torch.all(candidate_proxy > baseline_proxy)


def test_refined_selection_rejects_unordered_or_nonfinite_candidates():
    masses = torch.tensor((0.5, 0.8), dtype=torch.float32)
    valid = torch.ones((1, 1, 1, 1, 1, 2), dtype=torch.bool)
    lower = stack((0.01, -0.04))
    upper = stack((0.01, 0.01))
    try:
        subject.refined_utility_select(lower, upper, valid, masses)
    except ValueError as error:
        assert "not ordered" in str(error)
    else:
        raise AssertionError("unordered refined candidates were accepted")

    lower = stack((0.01, float("nan")))
    upper = stack((0.01, 0.01))
    try:
        subject.refined_utility_select(lower, upper, valid, masses)
    except FloatingPointError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite refined candidates were accepted")


def test_held_selection_identity_requires_single_terminal_contract():
    folds = [
        {
            "fold": fold,
            "best_iteration": subject.FOLD_UPDATES,
            "selected": {"sps_score": 53.0},
            "final": {"sps_score": 53.0},
            "history": [{"iteration": subject.FOLD_UPDATES}],
        }
        for fold in range(5)
    ]
    summary = {
        "training_config": {
            "fold_updates": subject.FOLD_UPDATES,
            "eval_interval": subject.FOLD_UPDATES,
        },
        "folds": folds,
    }
    assert subject.held_selection_identity(summary)
    summary["folds"][2]["history"].append({"iteration": 400})
    assert not subject.held_selection_identity(summary)


def test_gate_requires_exact_groups_and_all_thresholds():
    folds = [
        {"group": group, "sps_delta": value}
        for group, value in enumerate((0.12, 0.11, 0.10, 0.09, -0.02))]
    aoas = [
        {"group": group, "sps_delta": value}
        for group, value in zip(
            subject.uc1.AOA_VALUES, (0.13, 0.12, 0.11, 0.10, -0.04))]
    checks = subject.gate_checks(
        0.10, folds, aoas, {"synthetic_guard": True},
        gpu_seconds=10.0, elapsed=20.0)
    assert all(checks.values())

    folds[-1]["sps_delta"] = -0.04
    checks = subject.gate_checks(
        0.10, folds, aoas, {"synthetic_guard": True},
        gpu_seconds=10.0, elapsed=20.0)
    assert not checks["worst_fold_ge_minus_0_03"]

    try:
        subject.gate_checks(
            0.10, folds[:-1], aoas, {"synthetic_guard": True},
            gpu_seconds=10.0, elapsed=20.0)
    except ValueError as error:
        assert "five frozen folds" in str(error)
    else:
        raise AssertionError("gate accepted fewer than five folds")
