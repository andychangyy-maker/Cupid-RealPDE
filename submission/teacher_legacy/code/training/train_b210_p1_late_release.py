#!/usr/bin/env python3
"""Prepare or explicitly run a paired P1 late-training intervention.

Each arm reruns the same ordinary B210 Sim10k recipe from its fixed seed.
No missing intermediate optimizer state is assumed. The first 5000 updates
remain identical by construction; Real4k and matched prefix UQ follow separately.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys

SOURCE_COMMIT = "dd9fda1b765ac701354d8662b5bab67628d2562e"
ARMS = ("keep", "off", "anneal")
BRANCH_UPDATE = 5000
TERMINAL_UPDATE = 10000
BASE_WEIGHT = 0.005


def p1_weight(arm: str, iteration: int) -> float:
    if arm not in ARMS or not 1 <= iteration <= TERMINAL_UPDATE:
        raise ValueError("unknown arm or iteration outside Sim10k")
    if arm == "keep" or iteration <= BRANCH_UPDATE:
        return BASE_WEIGHT
    if arm == "off":
        return 0.0
    return BASE_WEIGHT * (TERMINAL_UPDATE - iteration) / (
        TERMINAL_UPDATE - BRANCH_UPDATE)


def schedule_contract(arm: str, seed: int) -> dict:
    return {
        "schema": "b210_p1_late_release_v1",
        "arm": arm,
        "seed": seed,
        "source_commit": SOURCE_COMMIT,
        "phase": "sim",
        "updates": TERMINAL_UPDATE,
        "branch_after_update": BRANCH_UPDATE,
        "base_distill_weight": BASE_WEIGHT,
        "weight_at_updates": {
            str(i): p1_weight(arm, i) for i in (1, 5000, 5001, 7500, 10000)
        },
        "start": "random_initialization_same_seed_for_all_three_arms",
        "real_closure": False,
        "real_teacher": False,
        "requires_followup": ["matched_Real4k_EMA999", "matched_prefix_UQ"],
        "scientific_gate_at_Sim_endpoint": None,
        "official_score_prediction": None,
    }


def wrap_update(original, arm: str):
    """Change only the effective P1 coefficient during the original update."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def update(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        config = bound.arguments["args"]
        iteration = bound.arguments["iteration"]
        coefficient = p1_weight(arm, iteration)
        previous = config.distill_weight
        config.distill_weight = coefficient
        try:
            result = original(*args, **kwargs)
            result[0]["effective_p1_distill_weight"] = coefficient
            return result
        finally:
            config.distill_weight = previous

    return update


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sim", "real"), default="sim")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--teacher-cache-sha256")
    parser.add_argument("--real-stats", type=Path)
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    for name in ("start-sim", "start-metadata", "clean-manifest", "real-dir"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--start-sim-sha256")
    parser.add_argument("--start-metadata-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=int, default=7200)
    parser.add_argument("--allow-research-equivalent-cache", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def execute(args, contract):
    required = ("teacher_cache", "teacher_cache_sha256", "real_stats",
                "sim_dir", "output_dir")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError("execute requires: " + ", ".join(missing))
    if args.wall_budget_seconds <= 0:
        raise ValueError("wall budget must be positive")
    if args.output_dir.exists():
        raise FileExistsError("use a fresh output directory")
    if len(args.teacher_cache_sha256) != 64:
        raise ValueError("teacher cache SHA-256 must be explicit")
    for path in (args.teacher_cache, args.real_stats, args.sim_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    # Importing the training stack is deliberately below the execute boundary.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import train_b210_np_functional_sim as sim

    if args.teacher_cache_sha256 != sim.TEACHER_CACHE_SHA256:
        if not args.allow_research_equivalent_cache:
            raise ValueError("non-formal cache requires explicit acknowledgement")

    original_update = sim.sim_update
    original_training_args = sim.training_args
    original_run = sim.run_experiment
    original_seed = sim.SEED
    original_budget = sim.WALL_BUDGET_SECONDS
    original_argv = sys.argv

    def configured(base_args):
        config = original_training_args(base_args)
        config.p1_distillation_schedule = dict(contract)
        return config

    def recorded_run(*run_args, **kwargs):
        summary = original_run(*run_args, **kwargs)
        summary["p1_distillation_schedule"] = dict(contract)
        summary["evidence_class"] = "Sim intervention; Real/UQ not evaluated"
        summary["submission_authorized"] = False
        if summary.get("checkpoint"):
            sim.write_json(args.output_dir / "sim_binding.json", {
                "schema": "b210_transfer_sim_binding_v1",
                "family": "p1_late_" + args.arm, "phase": "sim",
                "iteration": TERMINAL_UPDATE, "status": sim.PASS_STATUS,
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "prepared_source_commit": SOURCE_COMMIT,
                "training_visibility": "clean81_prefix_only",
                "lineage": [contract],
                "visibility_qualification": (
                    "No direct Real HDF5 or closure augmentation in Sim; "
                    "shared teacher and Real statistics provenance is inherited. "
                    "Not a full-pipeline unseen-condition claim."),
            })
        return summary

    sys.argv = [
        "train_b210_np_functional_sim.py",
        "--teacher-cache", str(args.teacher_cache),
        "--teacher-cache-sha256", args.teacher_cache_sha256,
        "--real-stats", str(args.real_stats),
        "--sim-dir", str(args.sim_dir),
        "--output-dir", str(args.output_dir),
        "--device", args.device, "--num-workers", "0",
    ]
    if (args.allow_research_equivalent_cache
            and args.teacher_cache_sha256 != sim.TEACHER_CACHE_SHA256):
        sys.argv += [
            "--research-equivalent-cache-sha256", args.teacher_cache_sha256,
            "--acknowledge-research-cache",
            sim.RESEARCH_CACHE_ACKNOWLEDGEMENT,
        ]
    try:
        sim.SEED = args.seed
        sim.WALL_BUDGET_SECONDS = args.wall_budget_seconds
        sim.training_args = configured
        sim.sim_update = wrap_update(original_update, args.arm)
        sim.run_experiment = recorded_run
        sim.main()
    finally:
        sim.sim_update = original_update
        sim.training_args = original_training_args
        sim.run_experiment = original_run
        sim.SEED = original_seed
        sim.WALL_BUDGET_SECONDS = original_budget
        sys.argv = original_argv


def main(argv=None):
    args = parse_args(argv)
    if args.phase == "real":
        import train_b210_transfer_real4k as transfer
        original_arms = transfer.ARMS
        try:
            transfer.ARMS = {arm: ("p1_late_" + arm, False) for arm in ARMS}
            print(json.dumps(transfer.run(args), ensure_ascii=False, indent=2))
        finally:
            transfer.ARMS = original_arms
        return
    contract = schedule_contract(args.arm, args.seed)
    contract["runner_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    if not args.execute:
        print(json.dumps({
            "mode": "plan_only", "training_started": False,
            "contract": contract,
        }, ensure_ascii=False, indent=2))
        return
    execute(args, contract)


if __name__ == "__main__":
    main()
