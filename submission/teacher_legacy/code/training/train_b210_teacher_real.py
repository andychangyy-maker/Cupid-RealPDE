#!/usr/bin/env python3
"""Fixed teacher-from-start x Real-recipe comparison; default is plan-only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import train_b210_transfer_real4k as transfer
import train_b210_pure_sim_start as sim_start

SCHEMA = "b210_teacher_real_v1"
ARMS = {"A": (True, False), "B": (True, True),
        "C": (False, False), "D": (False, True)}
MANIFEST_SHA = "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e"
SLOW_PREFIXES = ("stem.", "encoder.", "bottleneck.", "regime_film.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--plan-only", action="store_true")
    for name in ("sim-receipt", "sim-checkpoint", "clean-manifest", "real-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--sim-receipt-sha256")
    parser.add_argument("--wall-budget-seconds", type=float)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def real_recipe(arm):
    layered = ARMS[arm][1]
    return {"updates": 4000, "seed": 4, "batch_size": 8, "optimizer": "AdamW",
            "lr": 2e-4, "weight_decay": 1e-5, "ema_decay": .999,
            "scheduler_t_max": 4000 if layered else 12000,
            "slow_lr": 2e-5 if layered else 2e-4,
            "slow_prefixes": list(SLOW_PREFIXES) if layered else [],
            "frozen_parameters": 0, "normalizer_windows": 2544,
            "optimizer_windows": 9933, "optimizer_stride": 5,
            "loss_and_augmentation": "unchanged legacy Real joint loss and dense augmentation"}


def plan(args):
    teacher, layered = ARMS[args.arm]
    return {"schema": SCHEMA, "arm": args.arm, "decision": "PLAN_ONLY_NO_DATA_NO_GPU",
            "sim_family": "teacher_on" if teacher else "teacher_off",
            "real_recipe": "layered" if layered else "legacy", "real": real_recipe(args.arm),
            "manifest_sha256": MANIFEST_SHA, "training_visibility": "clean81_prefix_only",
            "teacher_ancestry": "unknown" if teacher else "no_teacher",
            "checkpoint_selection": False, "wall_budget_seconds": args.wall_budget_seconds,
            "tail_arrays_read": 0, "submission_authorized": False}


def validate_sim_receipt(receipt, arm):
    teacher = ARMS[arm][0]
    expected = {
        "schema": sim_start.TEACHER_SCHEMA if teacher else sim_start.SCHEMA,
        "decision": sim_start.TEACHER_STATUS if teacher else sim_start.STATUS,
        "training_visibility": "sim_with_p1_teacher_no_direct_real" if teacher else "pure_sim_only",
        "phase": "sim", "start": "random_initialization", "updates": 10000,
        "iteration": 10000, "seed": 0, "ema_decay": .999, "ema_updates": 10000,
        "batch_size": 8, "real_arrays_read": 0, "real_statistics_read": 0,
        "held_arrays_read": 0, "tail_arrays_read": 0, "real_closure": False,
        "pretext_checkpoint": None, "checkpoint_selection": False, "sealed": True,
        "teacher_reads": 1 if teacher else 0, "p1_distill_weight": .005 if teacher else 0.,
        "initial_model_state_sha256": sim_start.OFF_INITIAL_SHA256,
        "sample_stream_sha256": sim_start.OFF_STREAM_SHA256,
    }
    if teacher:
        expected.update(teacher_cache_sha256=sim_start.TEACHER_CACHE_SHA256,
                        teacher_start_update=1, teacher_end_update=10000)
    else:
        expected["checkpoint_sha256"] = sim_start.OFF_CHECKPOINT_SHA256
    if any(key not in receipt or receipt[key] != value for key, value in expected.items()):
        raise ValueError("Sim receipt does not bind the fixed teacher arm, initialization and stream")
    if teacher and receipt.get("teacher_cache", {}).get("sha256") != sim_start.TEACHER_CACHE_SHA256:
        raise ValueError("teacher-on receipt lacks the validated original P1 cache")


def parameter_groups(model, arm):
    named = list(model.named_parameters())
    if not ARMS[arm][1]:
        return [{"name": "all", "lr": 2e-4, "params": [p for _, p in named],
                 "param_names": [name for name, _ in named]}]
    groups = []
    for slow, label, lr in ((True, "encoder_slow", 2e-5), (False, "decoder_readout_uq", 2e-4)):
        selected = [(name, parameter) for name, parameter in named
                    if name.startswith(SLOW_PREFIXES) == slow]
        groups.append({"name": label, "lr": lr, "params": [p for _, p in selected],
                       "param_names": [name for name, _ in selected]})
    return groups


def run(args):
    if not args.execute:
        return plan(args)
    required = ("sim_receipt", "sim_receipt_sha256", "clean_manifest", "real_dir", "output_dir")
    if any(getattr(args, key) is None for key in required):
        raise ValueError("execution requires the Sim receipt SHA, clean manifest, Real data and output directory")
    if (args.wall_budget_seconds is None or not math.isfinite(args.wall_budget_seconds)
            or not 0 < args.wall_budget_seconds <= 1800):
        raise ValueError("explicit Real wall budget must be positive and at most 1800 seconds")
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    transfer.require_sha(args.sim_receipt, args.sim_receipt_sha256)
    binding = json.loads(args.sim_receipt.read_text(encoding="utf-8"))
    validate_sim_receipt(binding, args.arm)
    checkpoint = args.sim_checkpoint or Path(binding["checkpoint_path"])
    transfer.require_sha(checkpoint, binding["checkpoint_sha256"])
    transfer.require_sha(args.clean_manifest, MANIFEST_SHA)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    dense = transfer.load_runtime()
    torch, dr1, ft12, smu1 = dense.torch, dense.dr1, dense.ft12, dense.smu1
    rows, manifest = dr1.load_manifest(args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("execution requires CUDA; default plan-only opens no data")
    audit = dense.PrefixAudit()
    trajectories, metadata, prefix_windows = dr1.load_prefix_rows(args.real_dir, rows, "fit", audit)
    statistics = smu1.fold_statistics(smu1.window_sufficient_statistics(trajectories, prefix_windows),
                                     range(len(prefix_windows)))
    if smu1.statistics_digest(statistics) != transfer.RAW_STATISTICS_SHA256:
        raise ValueError("clean2544 normalizer statistics changed")
    normalizer = smu1.FoldNormalizer(statistics, device)
    windows = dense.dense_windows(trajectories, 5)
    if len(prefix_windows) != 2544 or len(windows) != 9933 or len(audit.opened) != 81:
        raise ValueError("clean81 prefix/dense inventory changed")
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    schedule = smu1.paired_batch_schedule(range(len(windows)), windows, updates=4000, seed=4)
    schedule["updates"] = 4000
    ft12.npt1.ensemble_train.seed_everything(4)
    torch.set_float32_matmul_precision("high")
    model, start = ft12.validate_start_checkpoint(checkpoint, binding["checkpoint_sha256"],
        normalizer, device, expected_iteration=10000, expected_status=binding["decision"])
    recipe = real_recipe(args.arm)
    start.update(transfer_arm=args.arm, transfer_binding=binding,
                 transfer_binding_sha256=args.sim_receipt_sha256.lower(), real_recipe=recipe)
    args.output_dir.mkdir(parents=True)
    normalizer_path = args.output_dir / "point_normalizer.pt"
    torch.save(tuple(value.detach().cpu() for value in statistics), normalizer_path)
    dr1.write_json_atomic(args.output_dir / "execution_contract.json", {
        **plan(args), "decision": "EXECUTION_STARTED", "start": start})
    access = dense.ArmSeal(args.arm)
    with transfer.endpoint_contract(dense.small4k):
        training, seal = dense.small4k.train_candidate(model, dataset, schedule, normalizer, device,
            deadline, args.output_dir, start, access, input_augmentation=dense.AUGMENTATION,
            gradient_ratio_diagnostics_only=True, scheduler_t_max=recipe["scheduler_t_max"],
            parameter_groups=parameter_groups(model, args.arm))
    if access.value != seal or time.perf_counter() > deadline:
        raise RuntimeError("Real endpoint seal or 1800-second budget failed")
    result = {**plan(args), "decision": "REAL4000_SEALED_UQ_AND_COMPARISON_PENDING",
        "start": start, "manifest": manifest, "training": training, "seal": seal,
        "normalizer": {"path": str(normalizer_path.resolve()), "sha256": transfer.sha256(normalizer_path),
                       "raw_statistics_sha256": smu1.statistics_digest(statistics),
                       "fit_files": audit.opened, "prefix_windows": len(prefix_windows)},
        "sample_stream_sha256": schedule["sha256"], "prefix_files_opened": audit.opened,
        "max_read_frame_exclusive_by_file": {row["file"]: int(row["val_boundary"]) for row in rows},
        "ordered_prefix_window_sha256": smu1.ordered_held_key_digest(range(len(windows)), windows, metadata),
        "elapsed_seconds": time.perf_counter() - started}
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    dr1.write_json_atomic(args.output_dir / "point_endpoint.json", {
        "schema": transfer.SCHEMA + "_point_receipt", "arm": args.arm,
        "checkpoint_path": seal["path"], "checkpoint_sha256": seal["file_sha256"],
        "updates": 4000, "ema_decay": .999, "seed": 4, "sample_stream_sha256": schedule["sha256"],
        "start_sim_sha256": binding["checkpoint_sha256"],
        "start_metadata_sha256": args.sim_receipt_sha256.lower(),
        "sim_family": plan(args)["sim_family"], "real_recipe": recipe,
        "auxiliary_enabled": False, "tail_read": False, "uq_pending": True,
        "result_sha256": transfer.sha256(args.output_dir / "result.json")})
    return result


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
