from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import train_b210_standard24k_real_phase_balanced as phase


def windows(trajectories, stride):
    result = []
    for trajectory, values in enumerate(trajectories):
        result.extend(
            SimpleNamespace(trajectory=trajectory, start=start)
            for start in range(0, int(values.shape[0]) - 40 + 1, stride)
        )
    return result


def test_phase_balanced_windows_preserve_dense_support(monkeypatch):
    trajectories = [torch.zeros(100, 2), torch.zeros(80, 2)]
    dense = windows(trajectories, 5)
    manifest = windows(trajectories, 20)
    real = phase.standard24.self_describing.real
    monkeypatch.setattr(real, "dense_windows", lambda values, stride: dense)
    monkeypatch.setattr(real, "CANDIDATE_STRIDE", 5)

    result = phase.phase_balanced_windows(trajectories, manifest)
    assert result["control"] == dense
    assert result["candidate"][:len(dense)] == dense
    aligned = [window for window in dense if window.start % 20 == 0]
    assert result["candidate"][len(dense):] == aligned + aligned
    assert len(result["candidate"]) == len(dense) + 2 * len(manifest)

    counts = {}
    for window in result["candidate"]:
        counts[phase.window_key(window)] = counts.get(phase.window_key(window), 0) + 1
    assert all(counts[phase.window_key(window)] == 3 for window in manifest)
    assert all(
        counts[phase.window_key(window)] == 1
        for window in dense if window.start % 20 != 0)


def test_phase_balanced_windows_reject_manifest_drift(monkeypatch):
    trajectories = [torch.zeros(100, 2)]
    dense = windows(trajectories, 5)
    real = phase.standard24.self_describing.real
    monkeypatch.setattr(real, "dense_windows", lambda values, stride: dense)
    monkeypatch.setattr(real, "CANDIDATE_STRIDE", 5)
    wrong_manifest = windows(trajectories, 20)[1:]
    with pytest.raises(RuntimeError, match="released train manifest"):
        phase.phase_balanced_windows(trajectories, wrong_manifest)


def test_identical_augmentation_returns_independent_copies(monkeypatch):
    real = phase.standard24.self_describing.real
    monkeypatch.setattr(real, "AUGMENTATION", {"input_gain_std": 0.01})
    result = phase.identical_augmentation()
    assert result["control"] == result["candidate"]
    assert result["control"] is not result["candidate"]
