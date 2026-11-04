from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "code/training"
sys.path.insert(0, str(TRAINING))
PATH = TRAINING / "train_b210_continuous_rate_jitter_real_gate.py"
SPEC = importlib.util.spec_from_file_location("continuous_rate_jitter", PATH)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)


def trajectory(frames: int) -> torch.Tensor:
    values = torch.arange(frames, dtype=torch.float32).view(frames, 1, 1, 1)
    return torch.cat((values, -values), dim=-1)


def test_rate_inventory_reserves_raw_offset_43():
    windows = SUBJECT.rate_capable_windows([trajectory(49)], stride=5)
    assert [window.start for window in windows] == [0, 5]
    fast = SUBJECT.INTERPOLATION_PLANS[2]
    assert int(fast[0][-1]) == 42
    assert int(fast[1][-1]) == 43
    assert float(fast[2][-1]) == torch.tensor(0.9).item()


def test_identity_is_bitwise_historical_slice_and_pressure_is_zero():
    values = trajectory(44)
    windows = SUBJECT.rate_capable_windows([values])
    dataset = SUBJECT.ScheduledContinuousRateDataset(
        [values], windows, [0], [0])
    history, target = dataset[0]
    combined = torch.cat((history, target))
    assert torch.equal(combined[..., :2], values[:40])
    assert torch.count_nonzero(combined[..., 2]) == 0


def test_slow_and_fast_transform_history_and_target_together():
    values = trajectory(44)
    windows = SUBJECT.rate_capable_windows([values])
    dataset = SUBJECT.ScheduledContinuousRateDataset(
        [values], windows, [0, 0], [1, 2])
    slow = torch.cat(dataset[0])[..., 0].flatten()
    fast = torch.cat(dataset[1])[..., 0].flatten()
    expected_slow = torch.arange(40, dtype=torch.float32) * 0.9
    expected_fast = torch.arange(40, dtype=torch.float32) * 1.1
    # Two valid float32 evaluation orders differ by at most one small ULP.
    assert torch.allclose(slow, expected_slow, atol=5e-6, rtol=0.0)
    assert torch.allclose(fast, expected_fast, atol=5e-6, rtol=0.0)
    assert torch.isclose(fast[-1], torch.tensor(42.9))


def test_rate_schedule_is_exact_deterministic_and_rng_isolated():
    before = torch.random.get_rng_state().clone()
    first = SUBJECT.candidate_rate_codes(12_000)
    second = SUBJECT.candidate_rate_codes(12_000)
    after = torch.random.get_rng_state()
    assert first == second
    assert torch.equal(before, after)
    assert first.count(0) == 6_000
    assert first.count(1) == 3_000
    assert first.count(2) == 3_000


def test_occurrence_expansion_preserves_pairing_and_source_identity():
    base = {
        "sha256": "a" * 64,
        "rows": [
            {"sample_indices": [3, 1], "trajectory_ids": [2, 0],
             "bootstrap_mask": [[1, 0, 1, 1], [1, 1, 0, 1]]}
            for _ in range(SUBJECT.UPDATES)
        ],
    }
    result = SUBJECT.expand_occurrence_schedule(base)
    assert result["batches"][0] == [0, 1]
    assert result["batches"][-1] == [2_998, 2_999]
    assert result["source_indices"][:4] == [3, 1, 3, 1]
    assert result["rows"][0]["trajectory_ids"] == [2, 0]
    assert result["base_stream_sha256"] == "a" * 64


def test_endpoint_contract_restores_shared_trainer_globals():
    names = (
        "EXPERIMENT", "ENDPOINT_SCHEMA", "PRIMARY_ENDPOINT", "UPDATES",
        "SCHEDULER_T_MAX", "HELD_AOA")
    before = {name: getattr(SUBJECT.small4k, name) for name in names}
    with SUBJECT.endpoint_contract("rate_jitter", 10):
        assert SUBJECT.small4k.UPDATES == 1_500
        assert SUBJECT.small4k.PRIMARY_ENDPOINT == "ema_u1500_rate_jitter"
        assert SUBJECT.small4k.HELD_AOA == 10
    assert {name: getattr(SUBJECT.small4k, name) for name in names} == before


def test_standard24k_identity_cannot_be_replaced(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        SUBJECT.dr1, "sha256",
        lambda _path: SUBJECT.EXPECTED_STANDARD24K_SHA256)
    assert SUBJECT.validate_standard24k_identity(checkpoint) == (
        SUBJECT.EXPECTED_STANDARD24K_SHA256)
    try:
        SUBJECT.validate_standard24k_identity(checkpoint, "f" * 64)
    except RuntimeError as error:
        assert "differs from preregistration" in str(error)
    else:
        raise AssertionError("mismatched claimed SHA was accepted")


def test_two_fold_aggregate_applies_pooled_point_gate(tmp_path, monkeypatch):
    metrics = {
        "rel_l2_score": 90.0,
        "tke_rel_l2_score": 80.0,
        "mvpe_rel_l2_score": 70.0,
        "native_sps_score": 40.0,
    }
    pairs = []
    for aoa in (10, 20):
        path = tmp_path / f"held{aoa}.json"
        row = {
            "schema": SUBJECT.SCHEMA,
            "experiment": SUBJECT.EXPERIMENT,
            "held_aoa": aoa,
            "start": {
                "sha256": SUBJECT.EXPECTED_STANDARD24K_SHA256,
                "expected_sha256": SUBJECT.EXPECTED_STANDARD24K_SHA256,
                "identity_validated": True,
            },
            "arms": {
                "control": {"metric_state": {"arm": "control"}},
                "rate_jitter": {"metric_state": {"arm": "candidate"}},
            },
            "comparison": {"point_final_contribution_delta": 0.05},
        }
        path.write_text(__import__("json").dumps(row), encoding="utf-8")
        pairs.append((str(path), SUBJECT.dr1.sha256(path)))
    candidate = dict(metrics)
    candidate["rel_l2_score"] += 0.30
    monkeypatch.setattr(
        SUBJECT.smu1, "merge_metric_states", lambda states: states[0])
    monkeypatch.setattr(
        SUBJECT.smu1, "finalize_metric_state",
        lambda state: metrics if state["arm"] == "control" else candidate)
    result = SUBJECT.aggregate_results(pairs)
    assert result["gate"]["folds_pass"]
    assert result["comparison"]["point_final_contribution_delta"] > 0.12
    assert result["gate"]["pass"]
