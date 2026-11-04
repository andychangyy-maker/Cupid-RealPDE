#!/usr/bin/env python3
"""Seal B210 EMA10k with optional fixed P1 teaching; default is data-free planning."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

SCHEMA = "b210_pure_sim_start_v1"
STATUS = "PURE_SIM_10000_SEALED"
TEACHER_SCHEMA = "b210_teacher_from_start_sim_v1"
TEACHER_STATUS = "TEACHER_FROM_START_SIM_10000_SEALED"
TEACHER_CACHE_SHA256 = "8f73ecbfb792101ce7b7f900947326f8946263be4ce30b5aab0fb8c1fd063301"
OFF_CHECKPOINT_SHA256 = "9eabf0523a54381d0bbc4bdf1d9aa6a61f21f040cbc23177bd88fb90db7f24fa"
OFF_INITIAL_SHA256 = "86842e831d674b75b403cbbed47bd91a529b4f82d19760a28cf915d0290fd834"
OFF_STREAM_SHA256 = "bd52d8197e98364285405ba700d0e5acf286f143d14405757b751ecaaf8e4130"
UPDATES, SEED, EMA_DECAY = 10_000, 0, .999


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--teacher-cache-sha256")
    parser.add_argument("--wall-budget-seconds", type=float)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def teacher_enabled(args):
    if (args.teacher_cache is None) != (args.teacher_cache_sha256 is None):
        raise ValueError("teacher cache path and SHA must be supplied together")
    if args.teacher_cache is not None and args.teacher_cache_sha256 != TEACHER_CACHE_SHA256:
        raise ValueError("teacher-on requires the fixed original P1 cache SHA")
    return args.teacher_cache is not None


def plan(args):
    enabled = teacher_enabled(args)
    result = {"schema": SCHEMA, "decision": "PLAN_ONLY_NO_DATA_NO_GPU",
            "training_visibility": "pure_sim_only", "start": "random_initialization",
            "seed": SEED, "updates": UPDATES, "batch_size": 8, "ema_decay": EMA_DECAY,
            "sim_trajectories": 100, "sim_windows": 4800, "window_stride": 20,
            "normalization": "Sim training windows only, including soft-SPS scale",
            "real_arrays_read": 0, "real_statistics_read": 0, "teacher_reads": 0,
            "real_closure": False, "pretext_checkpoint": None, "p1_distill_weight": 0.,
            "checkpoint_selection": False, "wall_budget_seconds": args.wall_budget_seconds}
    if enabled:
        result.update(schema=TEACHER_SCHEMA,
                      training_visibility="sim_with_p1_teacher_no_direct_real",
                      p1_distill_weight=.005, teacher_start_update=1, teacher_end_update=UPDATES,
                      teacher_cache_path=str(args.teacher_cache), teacher_cache_sha256=TEACHER_CACHE_SHA256,
                      paired_off_checkpoint_sha256=OFF_CHECKPOINT_SHA256,
                      expected_initial_model_state_sha256=OFF_INITIAL_SHA256,
                      expected_sample_stream_sha256=OFF_STREAM_SHA256)
    return result


def load_teacher_cache(sim, args, manifest, dataset):
    if not teacher_enabled(args):
        return None, None
    if sha256(args.teacher_cache) != TEACHER_CACHE_SHA256:
        raise ValueError("teacher cache file SHA changed")
    cache, metadata = sim.rehearsal.load_and_validate_cache(args.teacher_cache, manifest, dataset)
    if metadata["sha256"] != TEACHER_CACHE_SHA256:
        raise ValueError("teacher cache SHA changed during loading")
    return cache, metadata


def completed_identity(args, initial_sha, stream_sha, teacher_metadata):
    result = plan(args)
    result["decision"] = TEACHER_STATUS if teacher_enabled(args) else STATUS
    if teacher_enabled(args):
        if initial_sha != OFF_INITIAL_SHA256 or stream_sha != OFF_STREAM_SHA256:
            raise ValueError("teacher-on initialization or full sample stream differs from sealed off")
        if not teacher_metadata or teacher_metadata.get("sha256") != TEACHER_CACHE_SHA256:
            raise ValueError("teacher-on endpoint lacks validated cache identity")
        result.update(teacher_reads=1, teacher_cache=teacher_metadata,
                      initial_state_matches_off=True, sample_stream_matches_off=True)
    return result


def validate_execution(args):
    teacher_enabled(args)
    if args.sim_dir is None or args.output_dir is None or args.source_commit is None:
        raise ValueError("execution requires Sim directory, new output directory, and source commit")
    if (len(args.source_commit) != 40
            or any(c not in "0123456789abcdef" for c in args.source_commit)):
        raise ValueError("source commit must be a full lowercase Git SHA")
    budget = args.wall_budget_seconds
    if budget is None or not math.isfinite(budget) or not 0 < budget <= 4200:
        raise ValueError("explicit wall budget must be positive and at most 4200 seconds")
    if args.output_dir.exists():
        raise FileExistsError("use a new output directory")


def training_args(sim, args):
    enabled = teacher_enabled(args)
    base = argparse.Namespace(**{**vars(args), "real_stats": None, "num_workers": 0})
    configured = sim.training_args(base)
    configured.seed = SEED
    configured.distill_weight = .005 if enabled else 0.
    configured.soft_sps_proxy_real_scale = False
    configured.method = TEACHER_SCHEMA if enabled else SCHEMA
    return configured


def execute(args):
    validate_execution(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for path in (Path(__file__).parent, Path(__file__).resolve().parents[1]):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    import fp32_ema_shadow as ema_lib
    import train_b210_np_functional_sim as sim
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Sim10k requires CUDA")
    started = time.monotonic()
    deadline = started + args.wall_budget_seconds
    args.output_dir.mkdir(parents=True)
    configured = training_args(sim, args)
    common = sim.rehearsal.ensemble_train.common
    sim.rehearsal.ensemble_train.seed_everything(SEED)
    torch.set_float32_matmul_precision("high")
    loader, normalizer, windows, trajectories = common.prepare_sim(configured, device)
    if (trajectories, windows) != (100, 4800):
        raise ValueError("pure Sim requires released Sim100 / stride20 / 4800 windows")
    manifest_path = args.output_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    teacher_cache, teacher_metadata = load_teacher_cache(sim, args, manifest, loader.dataset)
    statistics = common.normalizer_dict(normalizer)
    statistics_sha = sim.rehearsal.canonical_sha256(statistics)
    scales = sim.sim_hgrad.compute_sim_hgrad_scale_contract(
        loader.dataset, sim.rehearsal.canonical_sha256(manifest), statistics_sha)
    previous_seed = sim.SEED
    try:
        sim.SEED = SEED
        model = sim.build_arm_model("candidate", device)
    finally:
        sim.SEED = previous_seed
    common.set_model_stats(model, normalizer)
    initial_state_sha = sim.state_sha256(model.state_dict())
    if teacher_enabled(args) and initial_state_sha != OFF_INITIAL_SHA256:
        raise ValueError("teacher-on initialization differs from sealed off")
    optimizer, scheduler = sim.make_optimizer(model, configured, "candidate")
    ema = ema_lib.FP32EMAShadow(model, EMA_DECAY)
    hook = optimizer.register_step_post_hook(lambda *_: ema.update_after_optimizer_step(
        model, optimizer_step=ema.updates + 1))
    stream = common.repro.cycling(sim.rehearsal.ensemble_train.tag_training_loader(loader, configured))
    sample_sha, history = hashlib.sha256(), []
    for iteration in range(1, UPDATES + 1):
        sim._deadline_check(deadline, iteration)
        values, _audit, key, _flags = sim.sim_update(
            model, next(stream), normalizer, normalizer, teacher_cache, scales,
            configured, device, iteration, optimizer, scheduler, "candidate")
        if (ema.updates != iteration
                or (teacher_cache is None and values["p1_distill_loss"] != 0)
                or (teacher_cache is not None and iteration == 1 and values["p1_distill_loss"] <= 0)):
            raise RuntimeError("Sim EMA or teacher-loss contract changed")
        sample_sha.update((key + "\n").encode())
        if iteration == 1 or iteration % 200 == 0:
            row = {"iteration": iteration, "elapsed_seconds": time.monotonic() - started,
                   "loss": values["loss"], "point_loss": values["point_loss"],
                   "member_loss": values["member_loss"], "quantile_loss": values["quantile_loss"],
                   "soft_sps_loss": values["soft_sps_loss"], "hgrad_loss": values["hgrad_loss"],
                   "tke_aux_loss": values["tke_aux_loss"], "p1_distill_loss": values["p1_distill_loss"]}
            history.append(row)
            print(json.dumps(row), flush=True)
    hook.remove()
    sim._deadline_check(deadline, UPDATES)
    state = ema.materialize_model_state(model)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all()
           for value in state.values()):
        raise FloatingPointError("pure Sim endpoint must be finite FP32")
    config = model.architecture_config()
    receipt = {**completed_identity(args, initial_state_sha, sample_sha.hexdigest(), teacher_metadata),
               "phase": "sim", "iteration": UPDATES,
               "source_commit": args.source_commit,
               "source_files": {Path(path).name: sha256(path) for path in (__file__, sim.__file__)},
               "sim_dir": str(args.sim_dir.resolve()), "split_manifest_sha256": sha256(manifest_path),
               "normalizer": statistics, "normalizer_sha256": statistics_sha,
               "hgrad_scale_contract": scales, "initial_model_state_sha256": initial_state_sha,
               "model_config_sha256": sim.rehearsal.canonical_sha256(config),
               "model_state_sha256": sim.state_sha256(state),
               "sample_stream_sha256": sample_sha.hexdigest(), "ema_updates": ema.updates,
               "ema_update_order": ema_lib.UPDATE_ORDER_CONTRACT,
               "optimizer": {"name": "AdamW", "lr": configured.lr,
                             "weight_decay": configured.weight_decay,
                             "scheduler": "cosine", "scheduler_t_max": UPDATES,
                             "grad_clip": configured.grad_clip},
               "native_loss_config": {key: getattr(configured, key) for key in (
                   "accuracy_weights", "mean_loss_weight", "member_loss_weight",
                   "quantile_loss_weight", "soft_sps_loss_weight", "soft_sps_warmup_updates")},
               "precision": "FP32 with torch float32_matmul_precision=high",
               "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(device),
               "history": history, "held_arrays_read": 0, "tail_arrays_read": 0}
    checkpoint = args.output_dir / "ema999_u10000_primary.pth"
    with checkpoint.open("xb") as handle:
        torch.save({"schema": receipt["schema"], "phase": "sim", "iteration": UPDATES,
                    "model_state_dict": state, "model_config": config,
                    "normalizer": statistics, "training_config": receipt,
                    "metrics": {"status": receipt["decision"]}}, handle)
    receipt.update(checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                   wall_seconds=time.monotonic() - started, sealed=True)
    sim._deadline_check(deadline, UPDATES)
    with (args.output_dir / "sim_endpoint.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
    return receipt


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(execute(arguments) if arguments.execute else plan(arguments), indent=2))
