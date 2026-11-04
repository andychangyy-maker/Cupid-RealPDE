#!/usr/bin/env python3
"""Fixed B256 P1-Sim10k -> legacy Real4k offline teacher; default is data-free planning."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import train_b210_pure_sim_start as pure
import train_b210_teacher_real as real
import train_b210_transfer_real4k as transfer

SCHEMA = "b256_offline_teacher_v1"
SIM_SCHEMA = SCHEMA + "_sim"
SIM_STATUS = "B256_P1_SIM_10000_SEALED"
PARAMETERS, CONTROLLER_PARAMETERS = 263574885, 8641
REAL_STREAM_SHA256 = "14331e0ff2d61c7f62a384061b47f629f6a71999fad6ba4e97f37d414f17a7fb"
CAPS = {"preflight": 600., "sim": 4800., "real": 2400.}
TOTAL_CAP = 7800.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(CAPS), required=True)
    for name in ("sim-dir", "real-dir", "clean-manifest", "teacher-cache", "output-dir",
                 "preflight-receipt", "sim-receipt", "sim-checkpoint"):
        parser.add_argument("--" + name, type=Path)
    for name in ("source-commit", "preflight-receipt-sha256", "sim-receipt-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--teacher-cache-sha256", default=pure.TEACHER_CACHE_SHA256)
    parser.add_argument("--wall-budget-seconds", type=float)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def plan(args):
    return {"schema": SCHEMA, "phase": args.phase, "decision": "PLAN_ONLY_NO_DATA_NO_GPU",
            "model": {"base": 256, "depth": 4, "parameters": PARAMETERS,
                      "controller_parameters": CONTROLLER_PARAMETERS,
                      "other_config": "unchanged b210_config(candidate)",
                      "group_norm_change": "5 modules change num_groups from 7 to 8"},
            "sim": {"seed": 0, "updates": 10000, "batch_size": 8, "ema_decay": .999,
                    "lr": 3e-4, "weight_decay": 1e-5, "scheduler_t_max": 10000,
                    "p1_distill_weight": .005, "teacher_start_update": 1,
                    "teacher_end_update": 10000, "teacher_cache_sha256": pure.TEACHER_CACHE_SHA256,
                    "native_loss_and_augmentation": "unchanged previous B210 teacher-on"},
            "real": real.real_recipe("A"), "manifest_sha256": real.MANIFEST_SHA,
            "precision": "FP32 with torch float32_matmul_precision=high",
            "preflight_updates": {"sim": 2, "real": 3}, "preflight_weights_saved": False,
            "formal_sim_start": "fresh seed0 initialization; never resume preflight",
            "phase_caps_seconds": CAPS, "total_cap_seconds": TOTAL_CAP,
            "time_budget_scope": "sum of measured preflight, Sim and Real phase wall times",
            "teacher_ancestry": "unknown", "tail_arrays_read": 0, "uq_calibration": False,
            "checkpoint_selection": False, "submission_authorized": False}


def model_config(sim):
    config = replace(sim.b210_config("candidate"), base=256)
    if config.depth != 4:
        raise ValueError("fixed B256 requires depth4")
    return config


def validate_execution(args):
    required = ["output_dir", "source_commit"]
    required += (["real_dir", "clean_manifest", "sim_receipt", "sim_receipt_sha256"]
                 if args.phase == "real" else ["sim_dir", "teacher_cache"])
    required += (["real_dir", "clean_manifest"] if args.phase == "preflight" else [])
    required += (["preflight_receipt", "preflight_receipt_sha256"] if args.phase == "sim" else [])
    if any(getattr(args, name) is None for name in required):
        raise ValueError("missing execution arguments: " + ", ".join(required))
    if len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    budget = CAPS[args.phase] if args.wall_budget_seconds is None else args.wall_budget_seconds
    if not math.isfinite(budget) or not 0 < budget <= CAPS[args.phase]:
        raise ValueError("wall budget exceeds the fixed phase cap")
    if args.teacher_cache_sha256 != pure.TEACHER_CACHE_SHA256:
        raise ValueError("only the original P1 cache is allowed")
    if args.output_dir.exists():
        raise FileExistsError("use a new output directory; preflight cannot be resumed")
    return budget


def validate_previous(value, args):
    expected = {"parameters": PARAMETERS, "source_commit": args.source_commit,
                "runner_sha256": pure.sha256(__file__)}
    if args.phase == "sim":
        expected.update(schema=SCHEMA + "_preflight", decision="PASS_HARDWARE_PREFLIGHT",
                        sealed=False, preflight_weights_saved=False,
                        completed_updates={"sim": 2, "real": 3})
    else:
        expected.update(schema=SIM_SCHEMA, decision=SIM_STATUS, phase="sim", sealed=True,
                        seed=0, batch_size=8, updates=10000, ema_updates=10000, ema_decay=.999,
                        p1_distill_weight=.005, teacher_start_update=1, teacher_end_update=10000,
                        teacher_cache_sha256=pure.TEACHER_CACHE_SHA256,
                        training_visibility="sim_with_p1_teacher_no_direct_real",
                        real_arrays_read=0, real_statistics_read=0, tail_arrays_read=0,
                        sample_stream_sha256=pure.OFF_STREAM_SHA256)
    if any(key not in value or value[key] != item for key, item in expected.items()):
        raise ValueError("prior receipt does not bind this B256 phase and fixed recipe")
    consumed = float(value.get("cumulative_wall_seconds", float("nan")))
    if not math.isfinite(consumed) or not 0 < consumed < TOTAL_CAP:
        raise ValueError("invalid or exhausted cumulative wall budget")
    return consumed


@contextmanager
def endpoint_contract(small4k):
    # Reuse the fixed loop with an honest B256 identity, never a B210 receipt.
    updates = {"EXPERIMENT": SCHEMA, "ENDPOINT_SCHEMA": SCHEMA + "_point", "HELD_AOA": -1}
    previous = {key: getattr(small4k, key) for key in updates}
    try:
        for key, value in updates.items():
            setattr(small4k, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(small4k, key, value)


def prepare_real(args, dense, device):
    dr1, ft12, smu1 = dense.dr1, dense.ft12, dense.smu1
    transfer.require_sha(args.clean_manifest, real.MANIFEST_SHA)
    rows, manifest = dr1.load_manifest(args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    audit = dense.PrefixAudit()
    trajectories, metadata, prefix = dr1.load_prefix_rows(args.real_dir, rows, "fit", audit)
    statistics = smu1.fold_statistics(smu1.window_sufficient_statistics(trajectories, prefix),
                                     range(len(prefix)))
    if smu1.statistics_digest(statistics) != transfer.RAW_STATISTICS_SHA256:
        raise ValueError("clean2544 normalizer statistics changed")
    windows = dense.dense_windows(trajectories, 5)
    if len(prefix) != 2544 or len(windows) != 9933 or len(audit.opened) != 81:
        raise ValueError("clean81 prefix/dense inventory changed")
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    schedule = smu1.paired_batch_schedule(range(len(windows)), windows, updates=4000, seed=4)
    schedule["updates"] = 4000
    if schedule["sha256"] != REAL_STREAM_SHA256:
        raise ValueError("Real full sample/bootstrap stream differs from the sealed B210 ON/A")
    proof = {"manifest": manifest, "manifest_sha256": real.MANIFEST_SHA,
             "prefix_files_opened": audit.opened, "normalizer_windows": 2544,
             "optimizer_windows": 9933, "raw_statistics_sha256": smu1.statistics_digest(statistics),
             "max_read_frame_exclusive_by_file": {r["file"]: int(r["val_boundary"]) for r in rows},
             "ordered_prefix_window_sha256": smu1.ordered_held_key_digest(range(len(windows)), windows, metadata),
             "tail_arrays_read": 0}
    return dataset, schedule, smu1.FoldNormalizer(statistics, device), statistics, proof


def run_sim(args, dense, device, deadline, previous, preflight=False):
    import fp32_ema_shadow as ema_lib
    import train_b210_np_functional_sim as sim
    torch, common = dense.torch, sim.rehearsal.ensemble_train.common
    started = time.perf_counter()
    configured = pure.training_args(sim, args)
    configured.method = SIM_SCHEMA
    sim.rehearsal.ensemble_train.seed_everything(0)
    loader, normalizer, windows, trajectories = common.prepare_sim(configured, device)
    if (trajectories, windows) != (100, 4800):
        raise ValueError("Sim must use released Sim100 / stride20 / 4800 windows")
    manifest_path = args.output_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache, cache_metadata = pure.load_teacher_cache(sim, args, manifest, loader.dataset)
    statistics = common.normalizer_dict(normalizer)
    statistics_sha = sim.rehearsal.canonical_sha256(statistics)
    scales = sim.sim_hgrad.compute_sim_hgrad_scale_contract(
        loader.dataset, sim.rehearsal.canonical_sha256(manifest), statistics_sha)
    sim.rehearsal.ensemble_train.seed_everything(0)
    model = sim.EnsembleUNet2d(model_config(sim)).float().to(device)
    if (sum(p.numel() for p in model.parameters()) != PARAMETERS
            or sum(p.numel() for p in model.energy_controller.parameters()) != CONTROLLER_PARAMETERS):
        raise ValueError("B256 parameter count changed")
    common.set_model_stats(model, normalizer)
    initial_sha = sim.state_sha256(model.state_dict())
    if previous and initial_sha != previous["initial_model_state_sha256"]:
        raise ValueError("formal Sim did not recreate the fresh preflight initialization")
    optimizer, scheduler = sim.make_optimizer(model, configured, "candidate")
    ema = ema_lib.FP32EMAShadow(model, .999)
    hook = optimizer.register_step_post_hook(lambda *_: ema.update_after_optimizer_step(
        model, optimizer_step=ema.updates + 1))
    stream = common.repro.cycling(sim.rehearsal.ensemble_train.tag_training_loader(loader, configured))
    sample_sha, history, timings = hashlib.sha256(), [], []
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for iteration in range(1, (2 if preflight else 10000) + 1):
            if time.perf_counter() > deadline:
                raise TimeoutError("B256 Sim wall budget exhausted")
            tick = time.perf_counter()
            values, _, key, _ = sim.sim_update(model, next(stream), normalizer, normalizer,
                cache, scales, configured, device, iteration, optimizer, scheduler, "candidate")
            if ema.updates != iteration or (iteration == 1 and values["p1_distill_loss"] <= 0):
                raise RuntimeError("Sim EMA or original P1 loss contract changed")
            sample_sha.update((key + "\n").encode())
            if preflight:
                torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - tick)
            if iteration == 1 or iteration % 200 == 0:
                row = {"iteration": iteration, "elapsed_seconds": time.perf_counter() - started,
                       **{key: values[key] for key in ("loss", "point_loss", "p1_distill_loss",
                          "member_loss", "quantile_loss", "soft_sps_loss", "hgrad_loss", "tke_aux_loss")}}
                history.append(row)
                print(json.dumps(row), flush=True)
    finally:
        hook.remove()
    config = model.architecture_config()
    receipt = {"schema": SIM_SCHEMA, "phase": "sim", "decision": SIM_STATUS,
               "parameters": PARAMETERS, "controller_parameters": CONTROLLER_PARAMETERS,
               "seed": 0, "batch_size": 8, "updates": 10000, "iteration": 10000,
               "ema_decay": .999, "ema_updates": ema.updates,
               "ema_update_order": ema_lib.UPDATE_ORDER_CONTRACT,
               "model_config": config, "model_config_sha256": sim.rehearsal.canonical_sha256(config),
               "group_norm_groups": {name: module.num_groups for name, module in model.named_modules()
                                     if isinstance(module, torch.nn.GroupNorm)},
               "initial_model_state_sha256": initial_sha, "sample_stream_sha256": sample_sha.hexdigest(),
               "normalizer": statistics, "normalizer_sha256": statistics_sha,
               "split_manifest_sha256": pure.sha256(manifest_path), "hgrad_scale_contract": scales,
               "teacher_cache": cache_metadata, "teacher_cache_sha256": pure.TEACHER_CACHE_SHA256,
               "p1_distill_weight": .005, "teacher_start_update": 1, "teacher_end_update": 10000,
               "training_visibility": "sim_with_p1_teacher_no_direct_real", "teacher_ancestry": "unknown",
               "real_arrays_read": 0, "real_statistics_read": 0, "tail_arrays_read": 0,
               "teacher_reads": 1, "start": "random_initialization", "real_closure": False,
               "pretext_checkpoint": None, "checkpoint_selection": False,
               "history": history,
               "setup_seconds": setup_seconds, "preflight_step_seconds": timings,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    receipt["native_training_config"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(configured).items()}
    if preflight:
        return model, receipt
    if sample_sha.hexdigest() != pure.OFF_STREAM_SHA256:
        raise ValueError("B256 Sim full sample stream differs from B210 ON/A")
    state = ema.materialize_model_state(model)
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise FloatingPointError("B256 Sim endpoint must be finite FP32")
    checkpoint = args.output_dir / "ema999_u10000_primary.pth"
    with checkpoint.open("xb") as handle:
        torch.save({"schema": SIM_SCHEMA, "phase": "sim", "iteration": 10000,
                    "model_state_dict": state, "model_config": config, "normalizer": statistics,
                    "training_config": receipt, "metrics": {"status": SIM_STATUS}}, handle)
    receipt.update(checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=pure.sha256(checkpoint),
                   model_state_sha256=sim.state_sha256(state), sealed=True)
    return receipt


class PreflightComplete(Exception):
    pass


class RealPreflight:
    """Stop before the fourth backward, after three unmodified native updates."""
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.started = None
        self.tick = None
        self.timings = []
        self.seconds = None
        self.peak_bytes = None

    def __call__(self, iteration, *_):
        self.torch.cuda.synchronize(self.device)
        if iteration == 1 and self.started is None:
            self.torch.cuda.reset_peak_memory_stats(self.device)
            self.started = self.tick = time.perf_counter()
            return nullcontext()
        if self.started is None or iteration not in (2, 3, 4) or len(self.timings) != iteration - 2:
            raise RuntimeError("preflight changed the three Real updates")
        now = time.perf_counter()
        self.timings.append(now - self.tick)
        if iteration == 4:
            self.seconds = self.timings[-1]
            self.peak_bytes = self.torch.cuda.max_memory_allocated(self.device)
            raise PreflightComplete
        self.tick = now
        return nullcontext()


def budget_projection(sim_seconds, real_seconds, setup_sim, setup_real, preflight_wall):
    estimates = {"sim": setup_sim + 10000 * sim_seconds, "real": setup_real + 4000 * real_seconds}
    total = sum(estimates.values()) + preflight_wall
    fits = all(math.isfinite(value) and 0 < value <= CAPS[phase] for phase, value in estimates.items())
    return {"decision": "PASS_HARDWARE_PREFLIGHT" if fits and total <= TOTAL_CAP else "NEEDS_FASTER_GPU",
            "estimated_phase_seconds": estimates, "estimated_total_seconds": total,
            "timing_caveat": "Sim update2 and Real update3; short timing estimate, warmup excess in setup"}


def run_preflight(args, dense, device, deadline):
    started = time.perf_counter()
    model, sim_report = run_sim(args, dense, device, deadline, None, preflight=True)
    gc.collect()
    dense.torch.cuda.empty_cache()
    real_started = time.perf_counter()
    dataset, schedule, normalizer, _, proof = prepare_real(args, dense, device)
    dense.ft12.npt1.ensemble_train.seed_everything(4)
    dense.ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    probe = RealPreflight(dense.torch, device)
    try:
        dense.small4k.train_candidate(model, dataset, {**schedule, "batches": schedule["batches"][:4]},
            normalizer, device, deadline, args.output_dir, {"preflight_only": True}, dense.ArmSeal("preflight"),
            expected_parameters=PARAMETERS, input_augmentation=dense.AUGMENTATION,
            gradient_ratio_diagnostics_only=True, scheduler_t_max=12000, forward_context=probe)
    except PreflightComplete:
        pass
    if probe.seconds is None:
        raise RuntimeError("preflight did not complete exactly three Real updates")
    warmup_excess = max(0., sum(probe.timings[:2]) - 2 * probe.seconds)
    projection = budget_projection(sim_report["preflight_step_seconds"][1], probe.seconds,
        sim_report["setup_seconds"], probe.started - real_started + warmup_excess,
        time.perf_counter() - started)
    return {**projection, "schema": SCHEMA + "_preflight", "sealed": False,
            "parameters": PARAMETERS, "completed_updates": {"sim": 2, "real": 3},
            "preflight_weights_saved": False, "initial_model_state_sha256": sim_report["initial_model_state_sha256"],
            "real_probe_start": "throwaway Sim-u2 raw state; memory/time check only",
            "model_config_sha256": sim_report["model_config_sha256"], "group_norm_groups": sim_report["group_norm_groups"],
            "sim_step_seconds": sim_report["preflight_step_seconds"], "real_step_seconds": probe.timings,
            "real_warmup_excess_seconds": warmup_excess,
            "sim_peak_cuda_allocated_bytes": sim_report["peak_cuda_allocated_bytes"],
            "real_peak_cuda_allocated_bytes": probe.peak_bytes, "real_data_access": proof}


def run_real(args, dense, device, deadline, binding):
    import train_b210_np_functional_sim as sim
    dataset, schedule, normalizer, statistics, proof = prepare_real(args, dense, device)
    dense.ft12.npt1.ensemble_train.seed_everything(4)
    model, start = dense.ft12.validate_start_checkpoint(
        args.sim_checkpoint or Path(binding["checkpoint_path"]), binding["checkpoint_sha256"], normalizer, device,
        expected_config=model_config(sim), expected_parameters=PARAMETERS,
        expected_controller_parameters=CONTROLLER_PARAMETERS, expected_status=SIM_STATUS)
    if start["checkpoint_model_config_sha256"] != binding["model_config_sha256"]:
        raise ValueError("Sim receipt and actual B256 checkpoint config differ")
    start.update(sim_receipt_sha256=args.sim_receipt_sha256, real_recipe=real.real_recipe("A"))
    normalizer_path = args.output_dir / "point_normalizer.pt"
    dense.torch.save(tuple(v.detach().cpu() for v in statistics), normalizer_path)
    access = dense.ArmSeal("B256")
    with endpoint_contract(dense.small4k):
        training, seal = dense.small4k.train_candidate(model, dataset, schedule, normalizer, device,
            deadline, args.output_dir, start, access, expected_parameters=PARAMETERS,
            input_augmentation=dense.AUGMENTATION, gradient_ratio_diagnostics_only=True, scheduler_t_max=12000)
    if access.value != seal:
        raise RuntimeError("B256 Real endpoint seal changed")
    return {**proof, "schema": SCHEMA + "_point_receipt", "phase": "real", "sealed": True,
            "decision": "B256_REAL4000_SEALED_POINT_EVALUATION_PENDING", "parameters": PARAMETERS,
            "updates": 4000, "ema_updates": training["ema_updates"], "ema_decay": .999, "seed": 4, "batch_size": 8,
            "start": start, "training": training, "seal": seal, "real_recipe": real.real_recipe("A"),
            "checkpoint_path": seal["path"], "checkpoint_sha256": seal["file_sha256"],
            "model_config_sha256": seal["model_config_sha256"], "start_sim_sha256": binding["checkpoint_sha256"],
            "start_metadata_sha256": args.sim_receipt_sha256, "sample_stream_sha256": training["actual_stream_sha256"],
            "normalizer": {"path": str(normalizer_path.resolve()), "file_sha256": pure.sha256(normalizer_path),
                           **dense.ft12.normalizer_payload(normalizer)}, "uq_calibration": False}


def run(args):
    if not args.execute:
        return plan(args)
    cap = validate_execution(args)
    started, previous, consumed = time.perf_counter(), None, 0.
    if args.phase != "preflight":
        path = args.preflight_receipt if args.phase == "sim" else args.sim_receipt
        digest = args.preflight_receipt_sha256 if args.phase == "sim" else args.sim_receipt_sha256
        transfer.require_sha(path, digest)
        previous = json.loads(path.read_text(encoding="utf-8"))
        consumed = validate_previous(previous, args)
    deadline = started + min(cap, TOTAL_CAP - consumed)
    dense = transfer.load_runtime()
    torch, device = dense.torch, dense.torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("execution requires CUDA; plan-only opens no data")
    torch.set_float32_matmul_precision("high")
    hardware = {"gpu": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
                "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory}
    if previous and previous["hardware"] != hardware:
        raise ValueError("hardware changed; repeat a fresh preflight on the intended GPU")
    args.output_dir.mkdir(parents=True)
    try:
        result = (run_preflight(args, dense, device, deadline) if args.phase == "preflight" else
                  run_sim(args, dense, device, deadline, previous) if args.phase == "sim" else
                  run_real(args, dense, device, deadline, previous))
    except torch.cuda.OutOfMemoryError:
        dense.dr1.write_json_atomic(args.output_dir / "stop.json", {
            "schema": SCHEMA, "phase": args.phase, "decision": "NEEDS_FASTER_GPU",
            "reason": "CUDA_OOM_fixed_batch8", "hardware": hardware, "sealed": False})
        raise
    elapsed = time.perf_counter() - started
    if elapsed > cap or consumed + elapsed > TOTAL_CAP:
        raise TimeoutError("B256 phase or cumulative wall budget exhausted; endpoint not admitted")
    result.update(source_commit=args.source_commit, runner_sha256=pure.sha256(__file__), hardware=hardware,
                  wall_seconds=elapsed, cumulative_wall_seconds=consumed + elapsed,
                  precision=plan(args)["precision"], teacher_ancestry="unknown")
    destination = "preflight.json" if args.phase == "preflight" else "sim_endpoint.json" if args.phase == "sim" else "result.json"
    dense.dr1.write_json_atomic(args.output_dir / destination, result)
    if args.phase == "real":
        dense.dr1.write_json_atomic(args.output_dir / "point_endpoint.json", {
            **{key: value for key, value in result.items() if key not in {"training", "start", "seal"}},
            "result_sha256": pure.sha256(args.output_dir / "result.json")})
    return result


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["decision"] == "NEEDS_FASTER_GPU":
        raise SystemExit(2)
