"""Real loader, normalization, Native-UQ and scoring APIs on tiny CPU data."""

from dataclasses import replace
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_b210_held_residual_uq as subject


@pytest.fixture(scope="module")
def bound():
    path = os.environ.get("REALPDE_CLEAN81_MANIFEST")
    if not path:
        pytest.skip("set REALPDE_CLEAN81_MANIFEST to the bound prefix manifest")
    return subject.load_split(Path(path))


@pytest.fixture(scope="module")
def runtime():
    torch.set_num_threads(1)
    return subject.runtime()


def test_real_window_objects_and_paired_loader_stream_cross_epoch(bound, runtime):
    rows, split = bound
    _, _, native, _ = runtime
    repro = native.ensemble_train.common.repro
    streams, inventories = [], []
    for role in ("control_uq", "uq_calibration"):
        metadata = subject.filter_rows(rows, split, role)
        windows = [repro.Window(i, start) for i, row in enumerate(metadata)
                   for start in row["train_starts"]]
        selected = subject.matched_windows(windows, metadata, split, role)
        assert len(selected) == len(set(selected))
        offset = 0
        for pair in split["control_matches"]:
            source = {r["file"]: r for r in rows}
            count = min(len(source[pair[k]]["train_starts"])
                        for k in ("control_uq", "uq_calibration"))
            part = selected[offset:offset + count]
            assert {metadata[w.trajectory]["file"] for w in part} == {pair[role]}
            assert [part[0].start, part[-1].start] == [
                source[pair[role]]["train_starts"][0],
                source[pair[role]]["train_starts"][-1]]
            offset += count
        trajectories = [torch.zeros(row["val_boundary"], 1, 1, 2) for row in metadata]
        tagged = native.ensemble_train.TrajectoryTaggedDataset(
            repro.WindowDataset(trajectories, selected))
        loader = DataLoader(tagged, batch_size=8, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(18))
        batches = repro.cycling(loader)
        stream = [next(batches)[3].tolist() for _ in range(2 * len(loader) + 1)]
        assert set(i for batch in stream[:len(loader)] for i in batch) == set(range(len(selected)))
        inventories.append(len(selected)); streams.append(stream)
    assert inventories[0] == inventories[1]
    assert streams[0] == streams[1]


def test_point49_loader_and_statistics_exclude_sentinel_tail_and_absent_outer(bound, runtime, tmp_path):
    rows, split = bound
    _, dense, _, prefix = runtime
    fit = subject.filter_rows(rows, split, "point_fit")
    for index, row in enumerate(fit):
        with h5py.File(tmp_path / row["file"], "w") as handle:
            for channel in ("u", "v"):
                values = np.full((row["frames"], 2, 2), index + 1, np.float32)
                values[row["val_boundary"]:] = 1.e6
                handle[channel] = values
            handle["re"], handle["aoa"] = row["re"], row["aoa"]
    audit = subject.RoleAudit(fit)
    trajectories, metadata, windows = dense.dr1.load_prefix_rows(tmp_path, fit, "fit", audit)
    assert len(trajectories) == 49 and len(windows) == 1568
    assert audit.report()["opened"] == split["roles"]["point_fit"]["files"]
    assert max(float(value.max()) for value in trajectories) == 49
    with pytest.raises(RuntimeError, match="outside this training role"):
        dense.dr1.load_prefix_rows(tmp_path, split["roles"]["outer_test"]["prefix_rows"], "fit", audit)
    statistics = dense.smu1.fold_statistics(
        dense.smu1.window_sufficient_statistics(trajectories, windows), range(len(windows)))
    expected_mean = sum((i + 1) * len(r["train_starts"]) for i, r in enumerate(fit)) / 1568
    assert statistics[0][:2].tolist() == pytest.approx([expected_mean] * 2)
    assert torch.equal(statistics[0], statistics[1])
    normalizer = dense.smu1.FoldNormalizer(statistics, torch.device("cpu"))
    report = dense.ft12.normalizer_payload(normalizer)
    restored, actual = prefix.normalizer_from_point(
        {"normalizer": report["values"], "normalizer_sha256": report["sha256"]}, torch.device("cpu"))
    assert actual == report and torch.equal(restored.std_target, normalizer.std_target)


def test_real_native_uq_initialization_and_objective_preserve_same_point(runtime):
    _, dense, native, _ = runtime
    from train_b210_transfer_uq import initialize_uq_state
    config = replace(dense.sim_tke.b210_config("candidate"), base=4, groups=2,
                     uq_hidden_channels=4, temporal_uq_channels=4,
                     physical_energy_hidden_channels=4)
    models = [dense.sim_tke.EnsembleUNet2d(config).float() for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict(), strict=True)
    normalizer = native.smu1.FoldNormalizer(
        (torch.zeros(3), torch.zeros(3), torch.ones(3), torch.tensor([.1, .2, 1.])),
        torch.device("cpu"))
    initial, point_hashes = [], []
    for model in models:
        native.ensemble_train.common.set_model_stats(model, normalizer)
        point_hashes.append(native.dr1.model_state_sha256(native.partition_state(model, uq=False)))
        initialize_uq_state(model, native.UQ_PREFIXES, "fresh")
        initial.append(native.dr1.model_state_sha256(native.partition_state(model, uq=True)))
        assert native.dr1.model_state_sha256(native.partition_state(model, uq=False)) == point_hashes[-1]
    assert initial[0] == initial[1] and point_hashes[0] == point_hashes[1]
    model = models[0]
    parameters = native.configure_uq_only(model)
    native.set_uq_training_mode(model)
    x, y = torch.randn(2, 20, 16, 16, 3), torch.randn(2, 20, 16, 16, 3)
    x[..., 2] = y[..., 2] = 0
    prediction, quantiles, extents, logits = native.uq_train.quantile_forward(model, x)
    loss, _ = native.uq_train.uq_objective(native.sharp2_args(), model, prediction,
        quantiles, y, normalizer, interval_extents=extents, coverage_logits=logits)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-5)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    assert torch.isfinite(norm) and norm > 0
    optimizer.step()
    assert native.dr1.model_state_sha256(native.partition_state(model, uq=False)) == point_hashes[0]


def test_signed_extents_absolute_bounds_and_official_sps(runtime):
    _, dense, _, _ = runtime
    import evaluate_b210_transfer as evaluation
    official = evaluation.import_file(evaluation.SCORER, "held_residual_test_scoring")
    normalizer = dense.smu1.FoldNormalizer(
        (torch.zeros(3), torch.tensor([.2, -.1, 0.]), torch.ones(3), torch.tensor([.1, .2, 1.])),
        torch.device("cpu"))
    raw = torch.randn(2, 20, 2, 2, 3) * .1
    lo, hi = torch.full_like(raw[..., :2], -.02), torch.full_like(raw[..., :2], .1)
    prediction, lower, upper = subject.absolute_distribution(raw, lo, hi, normalizer)
    assert torch.all(lower[..., :2] > prediction[..., :2])
    expected_lower = normalizer.postprocess_prediction(raw - torch.cat((lo, torch.zeros_like(raw[..., :1])), -1))
    assert torch.allclose(lower[..., :2], expected_lower[..., :2], atol=3.e-8, rtol=0)
    assert all(torch.count_nonzero(value[..., 2]) == 0 for value in (prediction, lower, upper))
    truth = ((lower + upper) * .5).numpy()
    prediction, lower, upper = (value.numpy() for value in (prediction, lower, upper))
    state = evaluation.metric_state(official, prediction, truth, lower, upper)
    scores = evaluation.finalize(official, state)
    sps, coverage = official.aggregate_sps(prediction, truth, 2, lower, upper)
    assert scores["sps_score"] == pytest.approx(official.score_sps(sps))
    assert scores["coverage"] == coverage == 1.
