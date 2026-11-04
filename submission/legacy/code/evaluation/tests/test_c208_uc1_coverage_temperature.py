from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import torch
from types import SimpleNamespace


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = EVALUATION_ROOT / "evaluate_c208_uc1_coverage_temperature.py"
SPEC = importlib.util.spec_from_file_location("c208_uc1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


class _ConstantRefiner(torch.nn.Module):
    def __init__(self, raw: torch.Tensor):
        super().__init__()
        self.register_buffer("raw", raw)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.raw.expand(len(fused), -1, -1, -1)


class _FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(utility_refinement_log_limit=1.0)
        self.coverage_calibration_head = None
        self.fused_calls = 0
        self._lower = torch.tensor(
            [[[[[[0.005, 0.010], [0.005, 0.010]]]]]])
        self._upper = self._lower.clone()
        self._valid = torch.ones_like(self._lower, dtype=torch.bool)
        self._masses = torch.tensor([0.4, 0.8])
        # Flattened as (T=1, C=2, lower/upper=2).
        self._raw_refinement = torch.tensor(
            [[[[0.4]], [[-0.2]], [[0.1]], [[-0.3]]]])
        self.utility_refinement_head = _ConstantRefiner(self._raw_refinement)

    def _interval_candidates(self, _quantiles):
        return self._lower, self._upper, self._valid, self._masses

    def _fused_uq_features(self, _features, _members, history):
        self.fused_calls += 1
        return torch.zeros((len(history), 1, 1, 1))

    def _refined_candidates(self):
        raw = self._raw_refinement.reshape(1, 1, 2, 2, 1, 1)
        raw = raw.permute(0, 1, 4, 5, 2, 3)
        log_scale = torch.tanh(raw)
        return (
            self._lower * torch.exp(log_scale[..., 0:1]),
            self._upper * torch.exp(log_scale[..., 1:2]),
        )

    def forward_uq_training(self, history):
        lower, upper = self._refined_candidates()
        std = torch.tensor([2.0, 3.0]).view(1, 1, 1, 1, 2, 1)
        selected = subject.identity_select(
            (self._lower + self._upper) * std,
            self._valid, self._masses)
        baseline_lower = subject.gather_candidate(lower, selected)
        baseline_upper = subject.gather_candidate(upper, selected)
        prediction = torch.zeros((len(history), 1, 1, 1, 2))
        members = torch.zeros((len(history), 1, 1, 1, 1, 2))
        quantiles = torch.zeros((len(history), 1, 1, 1, 2, 3))
        return (
            prediction, members, quantiles, baseline_lower, baseline_upper,
            [torch.zeros((len(history), 1, 1, 1))], None)


def test_identity_select_gathers_baseline_candidates_bitwise():
    masses = torch.tensor([0.3, 0.6, 0.9], dtype=torch.float32)
    widths = torch.tensor(
        [[[[[[0.01, 0.04, 0.10]]]]]], dtype=torch.float32)
    valid = torch.ones_like(widths, dtype=torch.bool)
    candidates = torch.tensor(
        [[[[[[11.0, 22.0, 33.0]]]]]], dtype=torch.float32)

    selected = subject.identity_select(widths, valid, masses)
    gathered = subject.gather_candidate(candidates, selected)
    expected = candidates[..., int(selected.item())]

    assert torch.equal(gathered, expected)
    assert gathered.item() in (11.0, 22.0, 33.0)


def test_replay_preamble_matches_frozen_runner_and_precedes_model_load(
        monkeypatch):
    calls = []
    monkeypatch.setattr(
        subject.uq_train, "seed_everything",
        lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(
        subject.torch, "set_float32_matmul_precision",
        lambda value: calls.append(("matmul", value)))

    subject.configure_replay(12)

    assert calls == [("seed", 12), ("matmul", "high")]
    source = inspect.getsource(subject.main)
    assert source.index("configure_replay(") < source.index("torch.load(")


def test_candidate_replay_refines_endpoints_but_selects_on_raw_width():
    model = _FakeModel()
    normalizer = SimpleNamespace(std_target=torch.tensor([2.0, 3.0, 1.0]))
    values = subject.candidate_tensors(
        model, torch.zeros((1, 1, 1, 1, 1)), normalizer)
    _prediction, lower, upper, valid, raw_width, masses, base_lower, _base_upper = values
    refined_lower, _ = model._refined_candidates()
    std = normalizer.std_target[:2].view(1, 1, 1, 1, 2, 1)

    assert model.fused_calls == 1
    assert torch.equal(raw_width, (model._lower + model._upper) * std)
    assert torch.equal(lower, refined_lower * std)
    selected = subject.identity_select(raw_width, valid, masses)
    assert torch.equal(base_lower, subject.gather_candidate(lower, selected))

    # A residual just beyond the raw lower endpoint is covered only because
    # hit labels use the already-frozen post-refinement candidate endpoint.
    residual = -0.011
    assert residual < -float((model._lower * std)[0, 0, 0, 0, 0, 0])
    hits = np.zeros((1, 2, 2), dtype=np.float64)
    totals = np.zeros_like(hits)
    subject.update_counts(
        hits, totals,
        torch.full((1, 1, 1, 1, 2), residual),
        torch.ones((1, 1, 1, 1, 2), dtype=torch.bool),
        lower, upper, valid)
    assert hits[0, 0, 0] == 1.0


def test_temperature_fit_is_bounded_and_raises_probability_for_undercoverage():
    masses = np.asarray([0.2, 0.5, 0.8], dtype=np.float64)
    totals = np.full((2, 2, 3), 10_000.0)
    observed = np.asarray([0.35, 0.68, 0.93])
    hits = totals * observed.reshape(1, 1, 3)

    parameters = subject.fit_temperature(hits, totals, masses)
    probabilities = subject.calibrated_probabilities(parameters, masses)

    assert parameters["bias"].shape == (2, 2)
    assert np.all(np.abs(parameters["bias"]) <= subject.BIAS_LIMIT)
    assert np.all(parameters["slope"] >= np.exp(-subject.LOG_SLOPE_LIMIT))
    assert np.all(parameters["slope"] <= np.exp(subject.LOG_SLOPE_LIMIT))
    assert np.all(probabilities[..., 1] > masses[1])


def test_leave_one_group_out_never_uses_held_statistics():
    masses = np.asarray([0.25, 0.50, 0.75], dtype=np.float64)
    groups = (0, 1, 2)
    totals = {group: np.full((1, 1, 3), 1000.0) for group in groups}
    hits = {
        0: totals[0] * np.asarray([0.10, 0.20, 0.30]),
        1: totals[1] * np.asarray([0.30, 0.55, 0.80]),
        2: totals[2] * np.asarray([0.45, 0.70, 0.92]),
    }

    fitted = subject.leave_one_group_out(hits, totals, groups, masses)
    expected = subject.fit_temperature(
        hits[1] + hits[2], totals[1] + totals[2], masses)

    assert np.allclose(fitted[0]["bias"], expected["bias"], atol=1e-10)
    assert np.allclose(fitted[0]["slope"], expected["slope"], atol=1e-10)


def test_gate_requires_oof_loao_identity_and_point_contracts():
    pooled = {"sps_delta": 0.11}
    condition = [{"sps_delta": value}
                 for value in (0.12, 0.11, 0.10, 0.09, -0.02)]
    loao = [{"sps_delta": value}
            for value in (0.11, 0.10, 0.08, 0.05, -0.04)]

    checks = subject.gate(
        pooled, condition, loao, True, True, True, elapsed=10.0)
    assert all(checks.values())

    stopped = subject.gate(
        pooled, condition, loao, True, False, True, elapsed=10.0)
    assert not stopped["point_stream_bitwise_unchanged"]


def _baseline_metrics(
        *, sps: float = 53.4471644789, covered: int = 891_244,
        width: float = 0.0158635909, scored: int = 1_000_000) -> dict:
    return {
        "sps_score": sps,
        "coverage": covered / scored,
        "covered_elements": covered,
        "mean_width": width,
        "scored_elements": scored,
    }


def test_baseline_replay_accepts_observed_small_numeric_drift():
    expected = _baseline_metrics()
    actual = _baseline_metrics(
        sps=expected["sps_score"] + 3.41953e-5,
        covered=expected["covered_elements"] + 100,
        width=expected["mean_width"] + 2.5e-7)
    expected_folds = [
        _baseline_metrics(covered=178_244, scored=200_000)
        for _ in range(5)]
    actual_folds = [
        _baseline_metrics(
            sps=row["sps_score"] + 3.41953e-5,
            covered=row["covered_elements"] + 20,
            width=row["mean_width"] + 2.5e-7,
            scored=row["scored_elements"])
        for row in expected_folds]

    result = subject.baseline_replay_gate(
        expected, actual, expected_folds, actual_folds)

    assert result["pass"]
    assert result["metric_checks"] == {
        "sps": True, "coverage": True, "mean_width": True}


def test_baseline_replay_rejects_each_pooled_axis_independently():
    expected = _baseline_metrics()
    folds = [
        _baseline_metrics(covered=178_244, scored=200_000)
        for _ in range(5)]
    mutations = (
        ("sps", _baseline_metrics(
            sps=expected["sps_score"] + 1.01e-4)),
        ("coverage", _baseline_metrics(
            covered=expected["covered_elements"] + 129)),
        ("mean_width", _baseline_metrics(
            width=expected["mean_width"] + 5.01e-7)),
    )
    for failed_metric, actual in mutations:
        result = subject.baseline_replay_gate(
            expected, actual, folds, folds)
        assert not result["pass"]
        assert not result["metric_checks"][failed_metric]


def test_baseline_replay_rejects_fold_cancellation_and_count_drift():
    pooled = _baseline_metrics()
    expected_folds = [
        _baseline_metrics(covered=178_244, scored=200_000)
        for _ in range(5)]
    cancelling_folds = [dict(row) for row in expected_folds]
    cancelling_folds[0]["sps_score"] += 1.01e-4
    cancelling_folds[1]["sps_score"] -= 1.01e-4

    cancelled = subject.baseline_replay_gate(
        pooled, pooled, expected_folds, cancelling_folds)
    assert not cancelled["metric_checks"]["sps"]

    wrong_count_folds = [dict(row) for row in expected_folds]
    wrong_count_folds[2]["scored_elements"] += 1
    wrong_count = subject.baseline_replay_gate(
        pooled, pooled, expected_folds, wrong_count_folds)
    assert not wrong_count["metric_checks"]["coverage"]
