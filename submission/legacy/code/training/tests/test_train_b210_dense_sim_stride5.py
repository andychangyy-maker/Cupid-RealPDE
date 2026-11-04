from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_dense_sim_stride5 as subject  # noqa: E402


def base_args(tmp_path: Path) -> argparse.Namespace:
    cache_sha = "b" * 64
    return argparse.Namespace(
        teacher_cache=tmp_path / "teacher.pt",
        teacher_cache_sha256=cache_sha,
        research_equivalent_cache_sha256=cache_sha,
        acknowledge_research_cache=(
            subject.b210.RESEARCH_CACHE_ACKNOWLEDGEMENT),
        output_dir=tmp_path / "run",
        sim_dir=tmp_path / "sim",
        real_stats=tmp_path / "real_stats.pt",
        device="cuda:0",
        num_workers=0,
        real_closure_bank=None,
        real_closure_bank_sha256=None,
    )


def tiny_inventory():
    trajectories = []
    manifest = []
    for trajectory in range(2):
        time = torch.arange(50, dtype=torch.float32).view(50, 1, 1, 1)
        field = time.repeat(1, 2, 3, 3)
        field = field + torch.tensor(
            [trajectory * 3.0, trajectory * 5.0, trajectory * 7.0])
        trajectories.append(field)
        manifest.append({
            "file": f"sim_{trajectory}.h5",
            "frames": 50,
            "re": 100 + trajectory,
            "aoa": trajectory * 5,
            "starts": [0, 5],
        })
    windows, expected = subject.manifest_for_stride(
        manifest, subject.OPTIMIZER_STRIDE)
    assert expected == manifest
    return trajectories, windows, manifest


def prepare_tiny(monkeypatch, tmp_path: Path):
    trajectories, windows, manifest = tiny_inventory()
    scratch = subject.b210.rehearsal.ensemble_train.common.scratch
    statistics_windows, statistics_manifest = subject.manifest_for_stride(
        manifest, subject.STATISTICS_STRIDE)
    statistics = scratch.window_statistics(
        trajectories, statistics_windows)
    expected_normalizer = scratch.ChannelNormalizer(
        *statistics, device=torch.device("cpu"))
    expected_normalizer_sha = subject.b210.rehearsal.canonical_sha256(
        subject.b210.rehearsal.ensemble_train.common.normalizer_dict(
            expected_normalizer))
    statistics_manifest_sha = subject.b210.rehearsal.canonical_sha256(
        statistics_manifest)
    optimizer_manifest_sha = subject.b210.rehearsal.canonical_sha256(manifest)
    optimizer_order_sha = subject.b210.rehearsal.canonical_sha256(
        subject.point_cache.ordered_window_keys(windows, manifest))

    def fake_load(sim_dir, max_files, stride, exclude_final):
        assert max_files == 0
        assert stride == subject.OPTIMIZER_STRIDE
        assert exclude_final is True
        return trajectories, windows, manifest

    monkeypatch.setattr(scratch, "load_simulations", fake_load)
    monkeypatch.setattr(subject, "FORMAL_TRAJECTORIES", 2)
    monkeypatch.setattr(subject, "FORMAL_OPTIMIZER_WINDOWS", 4)
    monkeypatch.setattr(subject, "FORMAL_STATISTICS_WINDOWS", 2)
    monkeypatch.setattr(
        subject, "REFERENCE_STATISTICS_MANIFEST_SHA256",
        statistics_manifest_sha)
    monkeypatch.setattr(
        subject, "REFERENCE_OPTIMIZER_MANIFEST_SHA256",
        optimizer_manifest_sha)
    monkeypatch.setattr(
        subject, "REFERENCE_OPTIMIZER_WINDOW_ORDER_SHA256",
        optimizer_order_sha)
    monkeypatch.setattr(
        subject, "REFERENCE_NORMALIZER_SHA256", expected_normalizer_sha)
    args = argparse.Namespace(
        seed=subject.SEED,
        window_stride=subject.OPTIMIZER_STRIDE,
        max_files=0,
        temporal_jitter_max=0,
        sim_dir=tmp_path / "sim",
        output_dir=tmp_path / "run",
        batch_size=2,
        num_workers=0,
    )
    args.output_dir.mkdir()
    loader, normalizer, count, trajectories_count = subject.prepare_dense_sim(
        args, torch.device("cpu"))
    return (
        args, loader, normalizer, count, trajectories_count,
        statistics_manifest)


def teacher_payload(manifest, dataset, *, stride=5):
    identities = {
        name: {
            "path": f"/teacher/{name}.pth",
            "size_bytes": 123,
            "actual_sha256": digest,
            "expected_sha256": digest,
            "sha256_match": True,
            "phase": "sim",
        }
        for name, digest in subject.REFERENCE_TEACHER_SHA256.items()
    }
    keys = subject.point_cache.ordered_window_keys(dataset.windows, manifest)
    shape = subject.point_cache.expected_prediction_shape(dataset)
    return {
        "schema_version": subject.point_cache.CACHE_SCHEMA_VERSION,
        "method": subject.point_cache.CACHE_METHOD,
        "composition": subject.point_cache.COMPOSITION,
        "teacher_checkpoints": identities,
        "sim_dir": "/released/sim",
        "window_stride": stride,
        "max_files": 0,
        "exclude_final_window": True,
        "inference_dtype": "float32",
        "split_manifest": manifest,
        "split_manifest_sha256": (
            subject.point_cache.canonical_sha256(manifest)),
        "window_keys": keys,
        "window_order_sha256": subject.point_cache.canonical_sha256(keys),
        "prediction_shape": list(shape),
        "prediction_dtype": "float16",
        "prediction_uv_physical": torch.zeros(shape, dtype=torch.float16),
    }


def test_import_does_not_change_default_trainer_and_wrapper_changes_only_support(
        tmp_path):
    args = base_args(tmp_path)
    original = subject._ORIGINAL_TRAINING_ARGS(args)
    dense = subject.dense_training_args(args)
    assert subject.b210.training_args is subject._ORIGINAL_TRAINING_ARGS
    assert subject.b210.parse_args is subject._ORIGINAL_PARSE_ARGS
    assert (original.window_stride, dense.window_stride) == (20, 5)
    assert dense.seed == 0
    assert dense.method == subject.METHOD
    assert dense.updates == original.updates == 10_000
    assert dense.batch_size == original.batch_size == 8
    assert dense.lr == original.lr == pytest.approx(3.0e-4)
    assert dense.weight_decay == original.weight_decay == pytest.approx(1.0e-5)
    assert dense.grad_clip == original.grad_clip == pytest.approx(1.0)
    assert dense.accuracy_weights == original.accuracy_weights == (1.0, 4.0, 1.0)
    assert dense.distill_weight == original.distill_weight
    assert dense.input_noise_std == dense.input_gain_std == 0.0
    assert dense.input_blur_probability == dense.input_blur_strength == 0.0
    assert dense.start_checkpoint is None
    for name, expected in subject.REFERENCE_TRAINING_FIELDS.items():
        assert getattr(original, name) == expected


def test_dense_entrypoint_requires_new_cache_and_forbids_closure(tmp_path):
    args = base_args(tmp_path)
    subject.validate_dense_only_args(args)

    missing = copy.copy(args)
    missing.research_equivalent_cache_sha256 = None
    with pytest.raises(ValueError, match="explicit new research cache SHA"):
        subject.validate_dense_only_args(missing)

    closure = copy.copy(args)
    closure.real_closure_bank = Path("closure.pt")
    with pytest.raises(ValueError, match="separate from Real-closure"):
        subject.validate_dense_only_args(closure)

    architecture = copy.copy(args)
    architecture.temporal_point_decoder = True
    with pytest.raises(ValueError, match="changes only optimizer support"):
        subject.validate_dense_only_args(architecture)


def test_stride_manifests_share_inventory_and_exclude_the_final_window():
    source = [{
        "file": "sim.h5", "frames": 1000, "re": 100, "aoa": 0,
        "starts": [999],
    }]
    dense, dense_manifest = subject.manifest_for_stride(source, 5)
    sparse, sparse_manifest = subject.manifest_for_stride(source, 20)
    assert len(dense) == 192
    assert len(sparse) == 48
    assert dense_manifest[0]["starts"][:3] == [0, 5, 10]
    assert dense_manifest[0]["starts"][-1] == 955
    assert sparse_manifest[0]["starts"][-1] == 940
    for key in ("file", "frames", "re", "aoa"):
        assert dense_manifest[0][key] == sparse_manifest[0][key]


def test_prepare_uses_dense_optimizer_windows_but_sparse_normalizer_and_hgrad(
        monkeypatch, tmp_path):
    (args, loader, normalizer, windows, trajectories,
     statistics_manifest) = prepare_tiny(monkeypatch, tmp_path)
    context = subject.dense_support_context(loader.dataset)
    assert windows == 4
    assert trajectories == 2
    assert subject.window_keys(loader.dataset.windows) == [
        (0, 0), (0, 5), (1, 0), (1, 5)]
    assert subject.window_keys(context.statistics_dataset.windows) == [
        (0, 0), (1, 0)]

    scratch = subject.b210.rehearsal.ensemble_train.common.scratch
    dense_statistics = scratch.window_statistics(
        loader.dataset.trajectories, loader.dataset.windows)
    dense_normalizer = scratch.ChannelNormalizer(
        *dense_statistics, device=torch.device("cpu"))
    dense_sha = subject.b210.rehearsal.canonical_sha256(
        subject.b210.rehearsal.ensemble_train.common.normalizer_dict(
            dense_normalizer))
    assert dense_sha != context.normalizer_sha256
    assert context.normalizer_sha256 == subject.b210.rehearsal.canonical_sha256(
        subject.b210.rehearsal.ensemble_train.common.normalizer_dict(normalizer))

    support = json.loads(
        (args.output_dir / subject.SUPPORT_CONTRACT_NAME).read_text())
    assert support["optimizer_window_stride"] == 5
    assert support["optimizer_windows"] == 4
    assert support["statistics_window_stride"] == 20
    assert support["statistics_windows"] == 2
    assert support["normalizer_source"] == "released_sim_stride20_inventory"
    assert support["hgrad_scale_source"] == "released_sim_stride20_inventory"
    assert support["teacher_cache_source"] == (
        "released_sim_stride5_optimizer_inventory")
    assert support["real_closure_enabled"] is False
    assert json.loads(
        (args.output_dir / subject.STATISTICS_MANIFEST_NAME).read_text()
    ) == statistics_manifest

    expected_hgrad = subject._ORIGINAL_HGRAD_CONTRACT(
        context.statistics_dataset,
        context.statistics_manifest_sha256,
        context.normalizer_sha256)
    monkeypatch.setattr(
        subject, "REFERENCE_HGRAD_CONTRACT_SHA256",
        expected_hgrad["canonical_sha256"])
    observed_hgrad = subject.compute_stride20_hgrad_contract(
        loader.dataset, context.optimizer_manifest_sha256,
        context.normalizer_sha256)
    assert observed_hgrad == expected_hgrad
    assert observed_hgrad["train_windows"] == 2
    assert observed_hgrad["split_manifest_sha256"] == (
        context.statistics_manifest_sha256)


def test_teacher_cache_requires_stride5_header_and_exact_dense_manifest(
        monkeypatch, tmp_path):
    args, loader, _normalizer, _windows, _trajectories, _manifest = (
        prepare_tiny(monkeypatch, tmp_path))
    context = subject.dense_support_context(loader.dataset)
    payload = teacher_payload(context.optimizer_manifest, loader.dataset)
    cache = tmp_path / "teacher_stride5.pt"
    torch.save(payload, cache)
    prediction, metadata = subject.load_dense_teacher_cache(
        cache, context.optimizer_manifest, loader.dataset)
    assert tuple(prediction.shape) == tuple(payload["prediction_shape"])
    assert metadata["window_stride"] == 5
    assert metadata["windows"] == 4
    assert metadata["split_manifest_sha256"] == (
        context.optimizer_manifest_sha256)
    cache_contract = json.loads(
        (args.output_dir / subject.CACHE_CONTRACT_NAME).read_text())
    assert cache_contract["window_stride"] == 5
    assert cache_contract["split_manifest_sha256"] == (
        context.optimizer_manifest_sha256)
    assert cache_contract["window_order_sha256"] == (
        context.optimizer_window_order_sha256)

    wrong_stride = dict(payload, window_stride=20)
    wrong_stride_path = tmp_path / "teacher_stride20_header.pt"
    torch.save(wrong_stride, wrong_stride_path)
    with pytest.raises(ValueError, match="header is not stride 5"):
        subject.load_dense_teacher_cache(
            wrong_stride_path, context.optimizer_manifest, loader.dataset)

    wrong_manifest = copy.deepcopy(payload)
    wrong_manifest["split_manifest"][0]["starts"] = [0]
    wrong_manifest_path = tmp_path / "teacher_wrong_manifest.pt"
    torch.save(wrong_manifest, wrong_manifest_path)
    with pytest.raises(ValueError, match="split manifest mismatch"):
        subject.load_dense_teacher_cache(
            wrong_manifest_path, context.optimizer_manifest, loader.dataset)

    wrong_teacher = copy.deepcopy(payload)
    wrong_teacher["teacher_checkpoints"]["a"].update({
        "actual_sha256": "d" * 64,
        "expected_sha256": "d" * 64,
    })
    wrong_teacher_path = tmp_path / "teacher_wrong_source.pt"
    torch.save(wrong_teacher, wrong_teacher_path)
    with pytest.raises(RuntimeError, match="source a differs"):
        subject.load_dense_teacher_cache(
            wrong_teacher_path, context.optimizer_manifest, loader.dataset)
