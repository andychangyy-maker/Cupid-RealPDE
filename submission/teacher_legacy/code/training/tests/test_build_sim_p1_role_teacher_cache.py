from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
SCRIPT = TRAINING_ROOT / "build_sim_p1_role_teacher_cache.py"
SPEC = importlib.util.spec_from_file_location("sim_p1_role_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


class DummyDataset:
    def __init__(self, starts=(0, 20)):
        self.windows = [
            SimpleNamespace(trajectory=0, start=start) for start in starts]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, _index):
        history = torch.zeros(1, 2, 3, 3)
        target = torch.zeros(4, 2, 3, 3)
        return history, target


def teacher_shas():
    return {"a": "a" * 64, "b": "b" * 64, "c": "c" * 64}


def make_payload(dataset, manifest):
    generator = torch.Generator().manual_seed(7)
    shape = subject.expected_tensor_shapes(dataset)[
        subject.TENSOR_KEYS[0]]
    source = [torch.randn(shape, generator=generator) for _ in range(3)]
    roles = subject.compose_p1_roles_float32(*source)
    a_fluctuation, b_fluctuation, p1_mean, validation = roles
    tensors = {
        subject.TENSOR_KEYS[0]: a_fluctuation.half().contiguous(),
        subject.TENSOR_KEYS[1]: b_fluctuation.half().contiguous(),
        subject.TENSOR_KEYS[2]: p1_mean.half().contiguous(),
    }
    shas = teacher_shas()
    keys = subject.point_cache.ordered_window_keys(dataset.windows, manifest)
    return {
        "schema_version": subject.CACHE_SCHEMA_VERSION,
        "method": subject.CACHE_METHOD,
        "composition": subject.COMPOSITION,
        "teacher_checkpoints": {
            name: {
                "path": f"/teachers/{name}.pth",
                "size_bytes": 123,
                "actual_sha256": digest,
                "expected_sha256": digest,
                "sha256_match": True,
                "phase": "sim",
            }
            for name, digest in shas.items()
        },
        "split_manifest": manifest,
        "split_manifest_sha256": subject.point_cache.canonical_sha256(
            manifest),
        "window_keys": keys,
        "window_order_sha256": subject.point_cache.canonical_sha256(keys),
        "windows": len(keys),
        "tensor_shapes": subject.expected_tensor_shapes(dataset),
        "tensor_storage": subject.TENSOR_STORAGE,
        "tensor_sha256": {
            name: subject.tensor_sha256(value)
            for name, value in tensors.items()
        },
        "units": subject.TENSOR_UNITS,
        "decomposition_validation": {
            **validation,
            "tolerance": subject.DECOMPOSITION_TOLERANCE,
        },
        **tensors,
    }


def test_float32_roles_reconstruct_formal_p1_and_have_zero_temporal_mean():
    prediction_a = torch.tensor(
        [[[[[1.1, 2.2]]], [[[3.3, 4.4]]]]], dtype=torch.float32)
    prediction_b = torch.tensor(
        [[[[[5.5, 6.6]]], [[[9.9, 10.1]]]]], dtype=torch.float32)
    prediction_c = torch.tensor(
        [[[[[7.7, 8.8]]], [[[11.2, 12.3]]]]], dtype=torch.float32)
    a_fluct, b_fluct, p1_mean, metrics = (
        subject.compose_p1_roles_float32(
            prediction_a, prediction_b, prediction_c))
    reconstructed = 0.5 * a_fluct + 0.5 * b_fluct + p1_mean
    base = prediction_a.lerp(prediction_b, 0.5)
    formal = base + 0.5 * (
        prediction_c.mean(dim=1, keepdim=True)
        - base.mean(dim=1, keepdim=True))
    assert (reconstructed - formal).abs().max() <= 1e-6
    assert a_fluct.mean(dim=1).abs().max() <= 1e-6
    assert b_fluct.mean(dim=1).abs().max() <= 1e-6
    assert all(value <= 1e-6 for value in metrics.values())
    expected_mean = (
        0.25 * prediction_a.mean(dim=1, keepdim=True)
        + 0.25 * prediction_b.mean(dim=1, keepdim=True)
        + 0.5 * prediction_c.mean(dim=1, keepdim=True))
    assert torch.equal(p1_mean, expected_mean)
    with pytest.raises(TypeError, match="float32"):
        subject.compose_p1_roles_float32(
            prediction_a.half(), prediction_b.half(), prediction_c.half())


def test_validator_binds_manifest_order_shape_dtype_finite_and_hash():
    dataset = DummyDataset()
    manifest = [{"file": "sim0.h5", "starts": [0, 20]}]
    payload = make_payload(dataset, manifest)
    tensors, metadata = subject.validate_cache_payload(
        payload, manifest, dataset, teacher_shas())
    assert set(tensors) == set(subject.TENSOR_KEYS)
    assert metadata["windows"] == 2

    wrong_manifest = [{"file": "different.h5", "starts": [0, 20]}]
    with pytest.raises(ValueError, match="manifest hash"):
        subject.validate_cache_payload(
            payload, wrong_manifest, dataset, teacher_shas())
    with pytest.raises(ValueError, match="window order"):
        subject.validate_cache_payload(
            payload, manifest, DummyDataset((20, 0)), teacher_shas())

    name = subject.TENSOR_KEYS[0]
    wrong_shape = copy.copy(payload)
    wrong_shape[name] = payload[name][:-1].contiguous()
    with pytest.raises(ValueError, match="shape mismatch"):
        subject.validate_cache_payload(
            wrong_shape, manifest, dataset, teacher_shas())
    wrong_dtype = copy.copy(payload)
    wrong_dtype[name] = payload[name].float().contiguous()
    with pytest.raises(TypeError, match="float16"):
        subject.validate_cache_payload(
            wrong_dtype, manifest, dataset, teacher_shas())
    non_finite = copy.copy(payload)
    non_finite[name] = payload[name].clone()
    non_finite[name][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        subject.validate_cache_payload(
            non_finite, manifest, dataset, teacher_shas())
    wrong_hash = copy.deepcopy(payload)
    wrong_hash["tensor_sha256"][name] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        subject.validate_cache_payload(
            wrong_hash, manifest, dataset, teacher_shas())


def test_atomic_save_revalidates_and_refuses_overwrite(tmp_path):
    dataset = DummyDataset()
    manifest = [{"file": "sim0.h5", "starts": [0, 20]}]
    payload = make_payload(dataset, manifest)
    output = tmp_path / "roles.pt"
    metadata = subject.save_cache_atomically(
        payload, output, manifest, dataset, teacher_shas())
    assert output.is_file()
    assert not subject.temporary_path(output).exists()
    assert metadata["sha256"] == subject.point_cache.sha256(output)
    loaded, _metadata = subject.load_and_validate_cache(
        output, manifest, dataset, teacher_shas())
    assert set(loaded) == set(subject.TENSOR_KEYS)
    original_sha = subject.point_cache.sha256(output)
    with pytest.raises(FileExistsError, match="overwrite"):
        subject.save_cache_atomically(
            payload, output, manifest, dataset, teacher_shas())
    assert subject.point_cache.sha256(output) == original_sha


def test_validate_args_rejects_stale_temporary_file(tmp_path):
    checkpoints = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.pth"
        path.write_bytes(name.encode())
        checkpoints.append(path)
    output = tmp_path / "roles.pt"
    subject.temporary_path(output).write_bytes(b"stale")
    args = argparse.Namespace(
        checkpoints=checkpoints,
        expected_checkpoint_sha256=["a" * 64, "b" * 64, "c" * 64],
        output=output,
        batch_size=1,
        window_stride=20,
        max_files=0,
        num_workers=0,
    )
    with pytest.raises(FileExistsError, match="stale temporary"):
        subject.validate_args(args)
