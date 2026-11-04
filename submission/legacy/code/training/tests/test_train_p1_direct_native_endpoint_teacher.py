from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))


import train_p1_direct_native_endpoint_teacher as endpoint_teacher  # noqa: E402
import train_p1_direct_native_quantile_teacher as quantile_teacher  # noqa: E402


class _Normalizer:
    mean_input = torch.zeros(4)
    std_input = torch.ones(4)
    mean_target = torch.tensor([1.0, -2.0, 0.0, 0.0])
    std_target = torch.tensor([2.0, 4.0, 1.0, 1.0])


def _formal_args(**overrides) -> argparse.Namespace:
    values = {
        "updates": 4000,
        "anchor_only_updates": 1000,
        "batch_size": 8,
        "eval_batch_size": 16,
        "reconstruction_batch_size": 16,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "truth_pinball_weight": 0.25,
        "soft_sps_weight": 1.0,
        "coverage_weight": 0.25,
        "coverage_target": quantile_teacher.FOLD04_REFERENCE["coverage"],
        "soft_tau_fraction": 0.10,
        "time_budget_seconds": 1800.0,
        "seed": 0,
        "num_workers": 0,
        "device": "cuda:0",
        "expected_point_sha256": endpoint_teacher.FORMAL_POINT_SHA256,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_compose_p0_absolute_endpoints_preserves_exact_formula() -> None:
    shape = (2, 3, 1, 1, 2)
    prediction_a = torch.full(shape, 2.0)
    prediction_b = torch.full(shape, 6.0)
    prediction_c = torch.full(shape, 7.0)
    lower_a = prediction_a - 0.4
    upper_a = prediction_a + 0.8
    lower_b = prediction_b - 1.2
    upper_b = prediction_b + 1.6

    lower, upper = endpoint_teacher.compose_p0_absolute_endpoints(
        prediction_a, lower_a, upper_a,
        prediction_b, lower_b, upper_b, prediction_c)

    base = prediction_a.lerp(prediction_b, 0.625)
    shift = 0.25 * (
        prediction_c.mean(dim=1, keepdim=True)
        - base.mean(dim=1, keepdim=True))
    torch.testing.assert_close(
        lower, lower_a.lerp(lower_b, 0.625) + shift)
    torch.testing.assert_close(
        upper, upper_a.lerp(upper_b, 0.625) + shift)
    torch.testing.assert_close(
        upper - lower,
        upper_a.lerp(upper_b, 0.625) - lower_a.lerp(lower_b, 0.625))


def test_absolute_endpoints_are_physical_and_allow_signed_extents() -> None:
    prediction_norm = torch.zeros(1, 1, 1, 1, 4)
    lower_extent = torch.tensor([[[[[-0.25, 0.5]]]]])
    upper_extent = torch.tensor([[[[[0.75, -0.25]]]]])

    lower, upper = endpoint_teacher.absolute_endpoints_from_native(
        prediction_norm, lower_extent, upper_extent, _Normalizer())

    torch.testing.assert_close(
        lower, torch.tensor([[[[[1.5, -4.0]]]]]))
    torch.testing.assert_close(
        upper, torch.tensor([[[[[2.5, -3.0]]]]]))
    assert torch.all(lower <= upper)


def test_collect_uses_postselection_forward_distribution() -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(0.0))
            self.calls = 0

        def forward_distribution(self, history):
            self.calls += 1
            prediction = torch.zeros(*history.shape[:-1], 4)
            lower = torch.full((*history.shape[:-1], 2), 0.5)
            upper = torch.full((*history.shape[:-1], 2), 1.0)
            return prediction, lower, upper

    model = FakeModel()
    history = torch.zeros(2, 3, 1, 1, 4)
    loader = [(history, torch.empty(0))]

    prediction, lower, upper = endpoint_teacher.collect_absolute_endpoints(
        model, loader, _Normalizer(), torch.device("cpu"))

    assert model.calls == 1
    assert prediction.dtype == lower.dtype == upper.dtype == torch.float16
    expected_lower_delta = torch.tensor([1.0, 2.0]).view(1, 1, 1, 1, 2)
    torch.testing.assert_close(
        lower.float(), prediction.float() - expected_lower_delta)
    expected_upper_delta = torch.tensor([2.0, 4.0]).view(1, 1, 1, 1, 2)
    torch.testing.assert_close(
        upper.float(), prediction.float() + expected_upper_delta)


def test_endpoint_distillation_is_zero_only_at_exact_endpoints() -> None:
    normalizer = _Normalizer()
    prediction_norm = torch.zeros(1, 1, 1, 1, 4)
    lower_extent = torch.tensor([[[[[0.5, 0.25]]]]])
    upper_extent = torch.tensor([[[[[1.0, 0.5]]]]])
    teacher_lower, teacher_upper = (
        endpoint_teacher.absolute_endpoints_from_native(
            prediction_norm, lower_extent, upper_extent, normalizer))

    exact, _ = endpoint_teacher.endpoint_distillation_objective(
        prediction_norm, lower_extent, upper_extent,
        teacher_lower, teacher_upper, normalizer)
    shifted, _ = endpoint_teacher.endpoint_distillation_objective(
        prediction_norm, lower_extent, upper_extent,
        teacher_lower + 0.1, teacher_upper, normalizer)

    assert exact.item() == 0.0
    assert shifted.item() > 0.0


def test_formal_partitions_exclude_fold04_from_both_training_loaders() -> None:
    point_folds = [0] * 508 + [1] * 508 + [2] * 508 + [3] * 508 + [4] * 544
    tail_folds = [0] * 143 + [1] * 143 + [2] * 143 + [3] * 143 + [4] * 153

    point_train, point_held, tail_train, tail_held = (
        endpoint_teacher.formal_training_partitions(
            point_folds, tail_folds))

    assert (len(point_train), len(point_held)) == (2032, 544)
    assert (len(tail_train), len(tail_held)) == (572, 153)
    assert {point_folds[index] for index in point_train} == {0, 1, 2, 3}
    assert {tail_folds[index] for index in tail_train} == {0, 1, 2, 3}
    assert {point_folds[index] for index in point_held} == {4}
    assert {tail_folds[index] for index in tail_held} == {4}


def test_formal_contract_locks_point_and_all_p0_source_hashes() -> None:
    assert endpoint_teacher.ARCHITECTURE == (
        "p1_direct_native_exact_endpoint_teacher")
    assert endpoint_teacher.FORMAL_POINT_SHA256 == (
        "758b44b2ce124d9c1bf72657d1a39247481c4794a6f2837bac8ffc072836ce07")
    assert endpoint_teacher.FORMAL_SOURCE_SHA256 == (
        quantile_teacher.FORMAL_SOURCE_SHA256)
    assert set(quantile_teacher.FORMAL_OOF_SHA256) == {"a", "b"}
    assert all(len(files) == 6
               for files in quantile_teacher.FORMAL_OOF_SHA256.values())
    assert all(len(value) == 64
               for files in quantile_teacher.FORMAL_OOF_SHA256.values()
               for value in files.values())


def test_validate_args_rejects_any_change_to_frozen_run() -> None:
    endpoint_teacher.validate_args(_formal_args())
    with pytest.raises(ValueError, match="SHA758b"):
        endpoint_teacher.validate_args(_formal_args(
            expected_point_sha256="0" * 64))
    with pytest.raises(ValueError, match="hyperparameters are frozen"):
        endpoint_teacher.validate_args(_formal_args(updates=3999))


def test_anchor_storage_is_one_fp16_endpoint_bank_under_one_gibibyte() -> None:
    elements = endpoint_teacher.EXPECTED_TRAIN_POINT * 20 * 32 * 64 * 2 * 2
    stored_bytes = elements * torch.empty((), dtype=torch.float16).element_size()
    assert stored_bytes == 665_845_760
    assert stored_bytes < 1024 ** 3


def test_oof_contract_calls_are_source_specific(monkeypatch) -> None:
    calls = []

    def fake_contract(point, oof, manifest, source):
        calls.append((point, oof, manifest, source))
        return source

    monkeypatch.setattr(
        quantile_teacher, "validate_oof_quantile_contract", fake_contract)
    args = argparse.Namespace(
        point_checkpoint_a=Path("a.pth"), oof_dir_a=Path("a"),
        point_checkpoint_b=Path("b.pth"), oof_dir_b=Path("b"))
    manifest = {"folds": 5}

    result = endpoint_teacher.validate_formal_oof_contracts(args, manifest)

    assert result == {"a": "a", "b": "b"}
    assert [call[-1] for call in calls] == ["a", "b"]
    assert all(call[-2] is manifest for call in calls)
