from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "train_full_backbone_point_distillation_loao.py")
SPEC = importlib.util.spec_from_file_location("point_distillation_loao", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


class TinyDataset:
    def __len__(self):
        return 6

    def __getitem__(self, index):
        value = torch.tensor([float(index)])
        return value, value + 10.0


def test_original_index_subset_preserves_teacher_cache_indices():
    subset = subject.OriginalIndexSubset(TinyDataset(), [4, 1, 5])
    original, history, target = subset[1]
    assert original == 1
    assert history.item() == 1.0
    assert target.item() == 11.0


def test_loao_partition_excludes_held_training_and_selects_only_held_tail():
    metadata = [
        {"aoa": aoa, "file": f"{aoa}.h5"}
        for aoa in subject.HELD_AOAS]
    train_windows = [
        SimpleNamespace(trajectory=trajectory, start=start)
        for trajectory in range(5) for start in (0, 20)]
    validation_windows = [
        SimpleNamespace(trajectory=trajectory, start=40)
        for trajectory in range(5)]

    partition = subject.build_loao_partition(
        train_windows, validation_windows, metadata, held_aoa=10.0)

    assert partition["train_windows"] == 8
    assert partition["excluded_held_train_windows"] == 2
    assert partition["held_validation_windows"] == 1
    assert partition["train_aoas"] == [0.0, 5.0, 15.0, 20.0]
    assert partition["validation_aoas"] == [10.0]
    assert partition["train_indices"] == [0, 1, 2, 3, 6, 7, 8, 9]
    assert partition["held_validation_indices"] == [2]


def _fold_row(
        held_aoa: float, point_delta: float,
        tke_delta: float = 0.0) -> dict:
    return {
        "held_aoa": held_aoa,
        "completed_updates": 800,
        "point_state_finite": True,
        "delta": {
            "point_score": point_delta,
            "tke_rel_l2_score": tke_delta,
        },
    }


def test_loao_gate_allows_one_bounded_negative_fold():
    rows = [
        _fold_row(aoa, delta) for aoa, delta in zip(
            subject.HELD_AOAS, (0.1, 0.2, 0.3, 0.4, 0.0))]
    assert subject.loao_gate(rows)["pass"] is True

    rows[-1] = _fold_row(20.0, -0.01, -0.05)
    gate = subject.loao_gate(rows)
    assert gate["point_nonnegative_folds"] == 4
    assert gate["worst_point_delta_safe"] is True
    assert gate["worst_tke_delta_safe"] is True
    assert gate["pass"] is True

    rows[-1] = _fold_row(20.0, -0.021, -0.05)
    assert subject.loao_gate(rows)["pass"] is False

    rows[-1] = _fold_row(20.0, -0.01, -0.101)
    assert subject.loao_gate(rows)["pass"] is False
