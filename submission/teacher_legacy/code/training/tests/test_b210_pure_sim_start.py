from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_b210_pure_sim_start as pure
import train_b210_np_functional_sim as sim
import fp32_ema_shadow as ema_lib


def test_plan_does_not_import_torch_or_touch_paths(tmp_path):
    command = ("import runpy,sys;sys.argv=[sys.argv[1],'--sim-dir',sys.argv[2],"
               "'--output-dir',sys.argv[3]];runpy.run_path(sys.argv[0],run_name='__main__');"
               "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-B", "-c", command, pure.__file__,
                    str(tmp_path / "absent_data"), str(tmp_path / "absent_output")], check=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("budget", [None, 0, -1, 4201, float("nan")])
def test_execution_rejects_invalid_budget_before_data(tmp_path, budget):
    args = pure.parse_args(["--sim-dir", str(tmp_path / "absent"), "--output-dir",
                            str(tmp_path / "new"), "--source-commit", "a" * 40])
    args.wall_budget_seconds = budget
    with pytest.raises(ValueError, match="wall budget"):
        pure.validate_execution(args)
    assert not args.output_dir.exists()


def test_fixed_configuration_has_no_real_or_teacher_input():
    args = pure.parse_args([])
    configured = pure.training_args(sim, args)
    assert configured.seed == 0
    assert configured.teacher_cache is None and configured.real_stats is None
    assert configured.start_checkpoint is None and configured.distill_weight == 0
    assert configured.soft_sps_proxy_real_scale is False
    assert configured.batch_size == 8 and configured.updates == 10000
    assert configured.accuracy_weights == (1., 4., 1.)


def test_missing_teacher_cannot_silently_disable_requested_distillation():
    with pytest.raises(ValueError, match="zero distillation"):
        sim.sim_update(None, None, None, None, None, {},
                       SimpleNamespace(distill_weight=.005), torch.device("cpu"),
                       1, None, None, "candidate")


@pytest.fixture
def tiny_step():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = replace(sim.b210_config("candidate"), base=4, depth=2, groups=1,
                     uq_hidden_channels=4, mean_profile_channels=4, temporal_uq_channels=4,
                     physical_energy_hidden_channels=4)
    model = sim.EnsembleUNet2d(config)
    normalizer = sim.rehearsal.ensemble_train.common.scratch.ChannelNormalizer(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3), torch.device("cpu"))
    sim.rehearsal.ensemble_train.common.set_model_stats(model, normalizer)
    args = pure.training_args(sim, pure.parse_args([]))
    optimizer, scheduler = sim.make_optimizer(model, args, "candidate")
    batch = (torch.randn(2, 20, 16, 32, 3), torch.randn(2, 20, 16, 32, 3),
             torch.tensor([0, 1]), torch.tensor([0, 1]))
    scales = {"scales": {"dx": [1., 1.], "dy": [1., 1.]}}
    yield model, normalizer, args, optimizer, scheduler, batch, scales
    torch.set_num_threads(previous_threads)


def test_real_update_without_teacher_keeps_native_losses_and_ema_order(tiny_step, monkeypatch):
    model, norm, args, optimizer, scheduler, batch, scales = tiny_step
    def forbidden(*_args, **_kwargs):
        raise AssertionError("teacher distillation must not run")
    monkeypatch.setattr(sim.rehearsal.ensemble_train, "component_point_distillation_loss", forbidden)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    ema = ema_lib.FP32EMAShadow(model, .999)
    optimizer.register_step_post_hook(lambda *_: ema.update_after_optimizer_step(model, optimizer_step=1))
    original_step = scheduler.step
    def scheduled():
        assert ema.updates == 1
        original_step()
    monkeypatch.setattr(scheduler, "step", scheduled)
    values, _audit, _key, _flags = sim.sim_update(
        model, batch, norm, norm, None, scales, args, torch.device("cpu"),
        1, optimizer, scheduler, "candidate")
    assert values["p1_distill_loss"] == 0
    assert values["point_loss"] > 0 and values["member_loss"] > 0 and values["quantile_loss"] > 0
    assert values["real_closure_selected_fraction"] == 0
    assert ema.updates == 1
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())


@pytest.mark.parametrize("iteration", [1, 10000])
def test_existing_teacher_path_still_uses_real_cache_and_distillation(tiny_step, monkeypatch, iteration):
    model, norm, args, optimizer, scheduler, batch, scales = tiny_step
    args = pure.training_args(sim, pure.parse_args([
        "--teacher-cache", "cache.pt", "--teacher-cache-sha256", pure.TEACHER_CACHE_SHA256]))
    assert args.distill_weight == .005 and args.real_stats is None
    assert args.soft_sps_proxy_real_scale is False and args.start_checkpoint is None
    cache = torch.randn(2, 20, 16, 32, 2)
    observed = []
    original = sim.rehearsal.ensemble_train.component_point_distillation_loss
    def recording(prediction, teacher, normalizer):
        observed.append(teacher.detach().clone())
        return original(prediction, teacher, normalizer)
    monkeypatch.setattr(sim.rehearsal.ensemble_train, "component_point_distillation_loss", recording)
    values, _audit, _key, _flags = sim.sim_update(
        model, batch, norm, norm, cache, scales, args, torch.device("cpu"),
        iteration, optimizer, scheduler, "candidate")
    assert len(observed) == 1 and torch.equal(observed[0], cache[batch[-1]])
    assert values["p1_distill_loss"] > 0


@pytest.mark.parametrize("arguments", [
    ["--teacher-cache", "missing.pt"],
    ["--teacher-cache-sha256", pure.TEACHER_CACHE_SHA256],
    ["--teacher-cache", "missing.pt", "--teacher-cache-sha256", "0" * 64],
])
def test_teacher_arguments_must_be_paired_and_fixed(arguments):
    with pytest.raises(ValueError, match="teacher"):
        pure.plan(pure.parse_args(arguments))


def test_teacher_endpoint_cannot_claim_pure_or_unmatched_identity():
    off = pure.completed_identity(pure.parse_args([]), "unused", "unused", None)
    assert off["schema"] == pure.SCHEMA and off["decision"] == pure.STATUS
    assert off["training_visibility"] == "pure_sim_only" and off["teacher_reads"] == 0
    args = pure.parse_args(["--teacher-cache", "missing.pt",
                           "--teacher-cache-sha256", pure.TEACHER_CACHE_SHA256])
    planned = pure.plan(args)
    assert planned["teacher_reads"] == 0 and planned["decision"] == "PLAN_ONLY_NO_DATA_NO_GPU"
    identity = pure.completed_identity(args, pure.OFF_INITIAL_SHA256, pure.OFF_STREAM_SHA256,
                                       {"sha256": pure.TEACHER_CACHE_SHA256})
    assert identity["schema"] == pure.TEACHER_SCHEMA != pure.SCHEMA
    assert identity["decision"] == pure.TEACHER_STATUS != pure.STATUS
    assert identity["training_visibility"] == "sim_with_p1_teacher_no_direct_real"
    assert identity["teacher_reads"] == 1 and identity["p1_distill_weight"] == .005
    assert (identity["teacher_start_update"], identity["teacher_end_update"]) == (1, 10000)
    for initial, stream, metadata in (
        ("wrong", pure.OFF_STREAM_SHA256, {"sha256": pure.TEACHER_CACHE_SHA256}),
        (pure.OFF_INITIAL_SHA256, "wrong", {"sha256": pure.TEACHER_CACHE_SHA256}),
        (pure.OFF_INITIAL_SHA256, pure.OFF_STREAM_SHA256, None),
    ):
        with pytest.raises(ValueError):
            pure.completed_identity(args, initial, stream, metadata)


def test_teacher_cache_uses_real_validator_and_rejects_window_or_file_drift(tmp_path, monkeypatch):
    import build_sim_p1_point_teacher_cache as cache_api
    repro = sim.rehearsal.ensemble_train.common.repro
    manifest = [{"file": "synthetic_sim.h5", "starts": [0, 20]}]
    dataset = repro.WindowDataset([torch.zeros(60, 4, 4, 2)],
                                 [repro.Window(0, 0), repro.Window(0, 20)])
    keys = cache_api.ordered_window_keys(dataset.windows, manifest)
    tensor = torch.ones(2, 20, 4, 4, 2, dtype=torch.float16)
    payload = {"schema_version": cache_api.CACHE_SCHEMA_VERSION,
               "method": cache_api.CACHE_METHOD, "composition": cache_api.COMPOSITION,
               "split_manifest": manifest, "split_manifest_sha256": cache_api.canonical_sha256(manifest),
               "window_keys": keys, "window_order_sha256": cache_api.canonical_sha256(keys),
               "teacher_checkpoints": {k: {"actual_sha256": "1" * 64, "expected_sha256": "1" * 64,
                                            "sha256_match": True, "phase": "sim"} for k in "abc"},
               "prediction_shape": list(tensor.shape), "prediction_dtype": "float16",
               "prediction_uv_physical": tensor}
    path = tmp_path / "tiny_cache.pt"
    torch.save(payload, path)
    # Substitute only the pinned artifact identity for this tiny serialized fixture.
    # The actual cache loader, window-order checks, tensor validation and hashing run.
    monkeypatch.setattr(pure, "TEACHER_CACHE_SHA256", pure.sha256(path))
    args = pure.parse_args(["--teacher-cache", str(path),
                           "--teacher-cache-sha256", pure.TEACHER_CACHE_SHA256])
    before_rng = torch.get_rng_state().clone()
    actual, metadata = pure.load_teacher_cache(sim, args, manifest, dataset)
    assert torch.equal(actual, tensor) and metadata["path"] == str(path.resolve())
    assert torch.equal(before_rng, torch.get_rng_state())
    reversed_data = repro.WindowDataset(dataset.trajectories, list(reversed(dataset.windows)))
    with pytest.raises(ValueError, match="window order"):
        pure.load_teacher_cache(sim, args, manifest, reversed_data)
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="file SHA"):
        pure.load_teacher_cache(sim, args, manifest, dataset)
