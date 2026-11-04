#!/usr/bin/env python3
"""Prepare or execute one fixed Sim x Real-aux transfer arm.

Default: print a plan without importing torch, opening HDF5, or touching CUDA.
Execution trains one Real4k endpoint only; matched UQ and evaluation follow
after all four endpoints are sealed. No score selects an intermediate model.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Sequence


SCHEMA = "b210_transfer_real4k_v1"
METADATA_SCHEMA = "b210_transfer_sim_binding_v1"
ARMS = {
    "A": ("standard24", False),
    "B": ("standard24", True),
    "C": ("causal4k_standard20", False),
    "D": ("causal4k_standard20", True),
}
UPDATES = 4_000
AUX_SEED = 21_001
RAW_STATISTICS_SHA256 = (
    "3f2605d84ad8da57fedddfaeff292050f1173ce37f75defa9dd0de6e4b4e9492")
C_START_SIM_SHA256 = "628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b"
C_START_METADATA_SHA256 = "282307574088299de967a7d64597a417150dd81cb18ea7b8d2897ee29b00e96f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> str:
    expected = str(expected).lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("every asset SHA must contain 64 hexadecimal digits")
    if sha256(path) != expected:
        raise ValueError(f"asset SHA mismatch: {path}")
    return expected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--held-aoa", type=int, choices=(0, 5, 10, 15, 20))
    parser.add_argument("--inner-held-aoa", type=int, choices=(0, 5, 10, 15, 20))
    parser.add_argument("--constant-controller-tke", action="store_true")
    parser.add_argument("--full-real-trajectories", action="store_true")
    parser.add_argument("--vertical-reflection", action="store_true")
    parser.add_argument('--real-time-jitter', action='store_true')
    parser.add_argument('--no-input-corruption', action='store_true')
    parser.add_argument('--sam-real', action='store_true')
    parser.add_argument('--real-seed', type=int, choices=(4, 5), default=4)
    parser.add_argument('--batch-mean-score', action='store_true')
    parser.add_argument('--raw-grid-phase', action='store_true')
    parser.add_argument("--full-real-final", action="store_true")
    parser.add_argument("--source-commit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    for name in ("start-sim", "start-metadata", "clean-manifest", "real-dir",
                 "output-dir"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--start-sim-sha256")
    parser.add_argument("--start-metadata-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float)
    args = parser.parse_args(argv)
    if args.raw_grid_phase and (args.arm != 'C' or not args.full_real_trajectories
            or args.held_aoa not in (15, 20) or args.real_seed != 4 or args.inner_held_aoa is not None
            or args.vertical_reflection or args.real_time_jitter or args.no_input_corruption
            or args.sam_real or args.constant_controller_tke or args.batch_mean_score):
        parser.error('raw grid phase requires otherwise unchanged full C Real4000 held15/20')
    if args.batch_mean_score and (args.arm != 'C' or not args.full_real_trajectories
            or args.held_aoa not in (15, 20) or args.real_seed != 4 or args.inner_held_aoa is not None
            or args.vertical_reflection or args.real_time_jitter or args.no_input_corruption
            or args.sam_real or args.constant_controller_tke):
        parser.error('batch-mean score requires otherwise unchanged full C Real4000 held15/20')
    if args.real_seed == 5 and (args.arm != 'C' or not args.full_real_trajectories
            or args.held_aoa != 15 or args.inner_held_aoa is not None or args.vertical_reflection
            or args.real_time_jitter or args.no_input_corruption or args.sam_real or args.constant_controller_tke):
        parser.error('seed5 is the fixed held15 shared-Sim Real4000 midpoint experiment')
    if args.inner_held_aoa is not None and (args.held_aoa is None or args.inner_held_aoa == args.held_aoa
            or args.arm != 'C' or not args.full_real_trajectories or args.vertical_reflection
            or args.real_time_jitter or args.no_input_corruption or args.sam_real or args.constant_controller_tke):
        parser.error('nested point calibration requires unmodified full C Real4000 with two distinct held angles')
    if args.vertical_reflection and not args.full_real_trajectories:
        parser.error('reflection requires matched full Real training support')
    if args.real_time_jitter and (not args.full_real_trajectories or args.vertical_reflection):
        parser.error('Real time jitter requires full Real support with no reflection')
    if args.no_input_corruption and (not args.full_real_trajectories or args.held_aoa != 15
                                    or args.real_time_jitter or args.vertical_reflection or args.constant_controller_tke):
        parser.error('clean-input screen is fixed to held15 full C Real4000 without other interventions')
    if args.sam_real and (not args.full_real_trajectories or args.held_aoa != 15 or args.arm != 'C'
                         or args.no_input_corruption or args.real_time_jitter
                         or args.vertical_reflection or args.constant_controller_tke):
        parser.error('SAM screen is fixed to original full C Real4000 held15')
    if args.constant_controller_tke and args.arm != "C":
        parser.error("constant controller TKE requires --arm C")
    if args.held_aoa is not None and (args.arm != 'C' or args.constant_controller_tke):
        parser.error('stage LOAO is fixed to unmodified C Real recipe')
    if args.full_real_final and (not args.full_real_trajectories or args.held_aoa is not None):
        parser.error('final fit requires full Real support with no held angle')
    if args.full_real_trajectories and (args.arm != 'C' or (args.held_aoa is None and not args.full_real_final) or args.constant_controller_tke):
        parser.error('full Real support requires held-AoA old-C recipe')
    return args


def intervention_fields(args: argparse.Namespace) -> dict:
    reflection = ({'vertical_reflection': {'probability': .25, 'isolated_cpu_seed': 29009,
                                         'space': 'physical_uvp_before_normalization'}}
                  if getattr(args, 'vertical_reflection', False) else {})
    if getattr(args, 'real_time_jitter', False):
        reflection.update(real_time_jitter={'max_shift': 4, 'isolated_cpu_seed': 29010,
                          'space': 'physical_uvp_before_normalization'}, optimizer_stride=None, base_anchor_stride=5)
    if getattr(args, 'no_input_corruption', False):
        reflection['real_input_corruption'] = 'off_after_matching_rng_draws'
    if getattr(args, 'sam_real', False):
        reflection['real_optimizer'] = {'name': 'SAM_AdamW', 'rho': .05, 'norm': 'l2',
                                        'matched_torch_rng': True, 'passes_per_update': 2}
    if getattr(args, 'batch_mean_score', False):
        reflection['point_score_aggregation'] = 'score_of_batch_mean_errors_point_and_members'
    if getattr(args, 'raw_grid_phase', False):
        reflection['raw_grid_phase'] = {'source_hw': [64, 128], 'model_hw': [32, 64],
                                       'nonzero_probability': .25, 'cpu_seed': 29011}
    if getattr(args, 'full_real_trajectories', False):
        if getattr(args, 'inner_held_aoa', None) is not None:
            return {'optimizer_support': 'full_other_three_trajectories',
                    'normalizer_support': 'other_three_prefixes', 'held_data_read': False,
                    'inner_held_aoa': args.inner_held_aoa, 'source_commit': args.source_commit}
        if getattr(args, 'full_real_final', False):
            return {'optimizer_support': 'full_clean81_trajectories',
                    **reflection,
                    'normalizer_support': 'clean81_prefix2544', 'held_data_read': None,
                    'source_commit': args.source_commit, 'final_fit': True,
                    'clean716_overlaps_point_training': True}
        return {'optimizer_support': 'full_other_four_trajectories',
                **reflection,
                'normalizer_support': 'other_four_prefixes', 'held_data_read': False,
                'source_commit': args.source_commit}
    return ({"controller_tke_schedule": "constant_0p01_4000", "source_commit": args.source_commit}
            if args.constant_controller_tke else {})


def plan(args: argparse.Namespace) -> dict:
    family, enabled = ARMS[args.arm]
    required = ("start_sim", "start_sim_sha256", "start_metadata",
                "start_metadata_sha256", "clean_manifest", "real_dir",
                "output_dir", "wall_budget_seconds")
    if args.constant_controller_tke or getattr(args, 'full_real_trajectories', False):
        required += ("source_commit",)
    return {
        "schema": SCHEMA, "arm": args.arm, "sim_family": family,
        "held_aoa": getattr(args, 'held_aoa', None),
        "holdout_scope": "Real/UQ stages only; shared Sim ancestors are not held out",
        **intervention_fields(args),
        "decision": "PLAN_ONLY_NO_DATA_NO_GPU",
        "missing_execution_arguments": [
            "--" + key.replace("_", "-") for key in required
            if getattr(args, key) is None],
        "start_sim": str(args.start_sim) if args.start_sim else None,
        "start_sim_sha256": args.start_sim_sha256,
        "start_metadata_schema": METADATA_SCHEMA,
        "real": {
            "updates": UPDATES, "seed": getattr(args, 'real_seed', 4), "batch_size": 8,
            "optimizer": "AdamW", "lr": 2e-4, "weight_decay": 1e-5,
            "scheduler_t_max": 12_000, "ema_decay": 0.999,
            "normalizer_windows": ('other-three prefixes' if getattr(args, 'inner_held_aoa', None) is not None
                                   else 2544 if getattr(args, 'held_aoa', None) is None else 'other-four prefixes'),
            "optimizer_windows": ('clean81 complete trajectories stride5' if getattr(args, 'full_real_final', False)
                                  else 'other-three complete trajectories stride5' if getattr(args, 'inner_held_aoa', None) is not None
                                  else 'other-four complete trajectories stride5' if getattr(args, 'full_real_trajectories', False)
                                  else 9933 if getattr(args, 'held_aoa', None) is None else 'other-four prefixes stride5'),
            "optimizer_stride": None if getattr(args, 'real_time_jitter', False) else 5,
            "base_anchor_stride": 5, "clean81_prefix_only": not getattr(args, 'full_real_trajectories', False),
            "augmentation": {"input_gain_std": .01, "input_noise_std": .01,
                             "input_blur_probability": .15,
                             "input_blur_strength": .20,
                             **({'applied': False, 'draw_only_to_preserve_rng': True}
                                if getattr(args, 'no_input_corruption', False) else {})},
            "same_sample_bootstrap_and_rng_contract_all_arms": True,
            "extra_backward": False,
        },
        "auxiliary": {
            "enabled": enabled, "weight": .005 if enabled else 0.,
            "target": "0.5 * sum_uv(mean_time((future_uv-mean_time(future_uv))^2))",
            "future_frames": 20, "transform": "log1p(K / .010)",
            "level_weights": [.25, .50, 1.], "loss": "weighted mean smooth_l1",
            "fresh_real_heads_all_arms": True, "isolated_cpu_seed": AUX_SEED,
            "gradients_reach_shared_backbone": enabled,
            "optimizer_includes_heads": enabled,
            "heads_in_ema_or_deploy_state": False,
            "weight_persists_for_all_4000_updates": enabled,
        },
        "gradient_ratio_policy": "diagnostic only; finite/nonzero safety check",
        "comparison": "Real4k plus matched point-frozen prefix-only UQ",
        "tail_read_by_this_runner": getattr(args, 'full_real_trajectories', False),
        "wall_budget_seconds": args.wall_budget_seconds,
        "submission_authorized": False,
    }


def validate_binding(args: argparse.Namespace) -> dict:
    missing = plan(args)["missing_execution_arguments"]
    if missing:
        raise ValueError("missing execution bindings: " + ", ".join(missing))
    if (not math.isfinite(args.wall_budget_seconds)
            or args.wall_budget_seconds <= 0):
        raise ValueError("set an explicit positive finite per-arm wall budget")
    if (args.constant_controller_tke or getattr(args, 'full_real_trajectories', False)) and (
            args.arm != "C" or args.start_sim_sha256.lower() != C_START_SIM_SHA256
            or args.start_metadata_sha256.lower() != C_START_METADATA_SHA256
            or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit or "")
            or args.wall_budget_seconds > (2700 if UPDATES == 6000 else 2200 if getattr(args, 'sam_real', False)
                                          else 3600 if UPDATES == 5334 and args.vertical_reflection
                                          else 2000 if getattr(args, 'real_member_teacher', False)
                                          else 2000 if UPDATES == 8000 and args.vertical_reflection else 1500)):
        raise ValueError("Real intervention requires exact old-C Sim/metadata, full source commit and its bounded wall cap")
    require_sha(args.start_metadata, args.start_metadata_sha256)
    binding = json.loads(args.start_metadata.read_text(encoding="utf-8"))
    if (binding.get("schema") != METADATA_SCHEMA
            or binding.get("family") != ARMS[args.arm][0]
            or binding.get("phase") != "sim"
            or binding.get("training_visibility") != "clean81_prefix_only"
            or binding.get("checkpoint_sha256") != args.start_sim_sha256.lower()
            or not isinstance(binding.get("iteration"), int)
            or binding["iteration"] <= 0
            or not isinstance(binding.get("status"), str)
            or not binding["status"]
            or not binding.get("prepared_source_commit")
            or not isinstance(binding.get("lineage"), list)
            or not binding["lineage"]):
        raise ValueError("Sim metadata is incomplete or does not bind this arm")
    require_sha(args.start_sim, args.start_sim_sha256)
    return binding


def load_runtime():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for path in (Path(__file__).resolve().parents[1], Path(__file__).parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import train_b210_scratch_real_dense_aug as dense
    return dense


def fresh_supervisor(model, device, sim):
    """Construct the same CPU heads in every arm without changing CUDA RNG."""
    import torch
    channels = tuple(model.config.base * 2 ** level
                     for level in range(model.config.depth - 2, -1, -1))
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also seeds CUDA, whose state devices=[] does not save.
        torch.random.default_generator.manual_seed(AUX_SEED)
        supervisor = sim.MultiScaleTKEDeepSupervisor(channels)
    return supervisor.to(device)


def auxiliary_arguments(arm: str, supervisor, sim) -> dict:
    enabled = ARMS[arm][1]
    return {
        "full_backbone_auxiliary": supervisor if enabled else None,
        "full_backbone_auxiliary_weight": sim.MULTISCALE_TKE_WEIGHT if enabled else 0.,
        "full_backbone_gradient_audit_update": 2 if enabled else 0,
        "gradient_ratio_diagnostics_only": True,
    }


@contextmanager
def endpoint_contract(small4k, held_aoa=None, seed=None):
    # Match the existing scoped Dense12k adapter without a second train loop.
    replacements = {"EXPERIMENT": SCHEMA, "ENDPOINT_SCHEMA": SCHEMA + "_endpoint",
                    "HELD_AOA": -1 if held_aoa is None else held_aoa}
    if seed is not None:
        replacements['SEED'] = seed
    previous = {key: getattr(small4k, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(small4k, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(small4k, key, value)


@contextmanager
def point_score_aggregation(dense, enabled):
    if not enabled:
        yield
        return
    from functools import partial
    common = dense.ft12.npt1.ensemble_train.common
    original = common.accuracy_objective
    common.accuracy_objective = partial(original, batch_mean_errors=True)
    try:
        yield
    finally:
        common.accuracy_objective = original


def run(args: argparse.Namespace) -> dict:
    if not args.execute:
        return plan(args)
    entry_started = time.perf_counter()
    binding = validate_binding(args)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    dense = load_runtime()
    torch, ft12, small4k = dense.torch, dense.ft12, dense.small4k
    seed = getattr(args, 'real_seed', 4)
    dr1, smu1, sim = dense.dr1, dense.smu1, dense.sim_tke
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise ValueError("clean81 prefix manifest byte identity changed")
    rows, manifest = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    from b210_stage_fold import select_rows
    held_aoa = getattr(args, 'held_aoa', None)
    rows = select_rows(rows, held_aoa, inner_held_aoa=getattr(args, 'inner_held_aoa', None))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("execution requires CUDA; use plan-only for preparation")
    started = entry_started if (args.constant_controller_tke or args.full_real_trajectories) else time.perf_counter()
    deadline = started + args.wall_budget_seconds
    audit = dense.PrefixAudit()
    trajectories, metadata, manifest_windows = dr1.load_prefix_rows(
        args.real_dir, rows, "fit", audit)
    statistics = smu1.fold_statistics(
        smu1.window_sufficient_statistics(trajectories, manifest_windows),
        range(len(manifest_windows)))
    if held_aoa is None and smu1.statistics_digest(statistics) != RAW_STATISTICS_SHA256:
        raise ValueError("clean2544 normalizer statistics changed")
    normalizer = smu1.FoldNormalizer(statistics, device)
    full_audit = None
    if getattr(args, 'full_real_trajectories', False):
        from train_b210_full_released_real_support import load_full_rows
        del trajectories
        full_audit = dense.PrefixAudit()
        trajectories, metadata = load_full_rows(args.real_dir, rows, 'fit', full_audit,
                                                raw_grid=getattr(args, 'raw_grid_phase', False))
        if set(full_audit.opened) != {r['file'] for r in rows}:
            raise ValueError('full Real training file allowlist changed')
    windows = dense.dense_windows(trajectories, 5)
    if ((held_aoa is None and (len(manifest_windows) != 2544 or (not args.full_real_trajectories and len(windows) != 9933)))
            or len(manifest_windows) != sum(len(r['train_starts']) for r in rows)
            or set(audit.opened) != {r['file'] for r in rows}):
        raise ValueError("clean81 prefix window inventory changed")
    if args.full_real_trajectories and len(windows) != sum((r['frames'] - 40) // 5 + 1 for r in rows):
        raise ValueError('full trajectory window inventory changed')
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    if getattr(args, 'raw_grid_phase', False):
        from raw_grid_phase_dataset import RawGridPhaseDataset
        dataset = RawGridPhaseDataset(dataset)
    if getattr(args, 'vertical_reflection', False):
        from vertical_reflection_dataset import VerticalReflectionDataset
        dataset = VerticalReflectionDataset(dataset)
    if getattr(args, 'real_time_jitter', False):
        from prefix_time_jitter_dataset import TimeJitterDataset
        dataset = TimeJitterDataset(dataset, full_time=True, max_shift=4)
    schedule = smu1.paired_batch_schedule(
        range(len(windows)), windows, updates=UPDATES, seed=seed)
    schedule["updates"] = UPDATES
    ft12.npt1.ensemble_train.seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    model, start = ft12.validate_start_checkpoint(
        args.start_sim, args.start_sim_sha256, normalizer, device,
        expected_iteration=binding["iteration"], expected_status=binding["status"])
    supervisor = fresh_supervisor(model, device, sim)
    start["transfer_binding"] = binding
    start["transfer_binding_sha256"] = args.start_metadata_sha256.lower()
    start["transfer_arm"] = args.arm
    start['stage_held_aoa'] = held_aoa
    start.update(intervention_fields(args))
    start["supervisor_initial_state_sha256"] = sim.state_sha256(supervisor.state_dict())
    args.output_dir.mkdir(parents=True)
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "point_normalizer.pt")
    dr1.write_json_atomic(args.output_dir / "execution_contract.json", {
        **plan(args), "decision": "EXECUTION_STARTED"})
    seal_access = dense.ArmSeal(args.arm)
    with endpoint_contract(small4k, held_aoa, seed), point_score_aggregation(dense, getattr(args, 'batch_mean_score', False)):
        training, seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            args.output_dir, start, seal_access,
            input_augmentation=dense.AUGMENTATION,
            **({'discard_input_corruption': True} if args.no_input_corruption else {}),
            **({'sam_rho': .05} if getattr(args, 'sam_real', False) else {}),
            **({"constant_controller_tke": True} if args.constant_controller_tke else {}),
            **auxiliary_arguments(args.arm, supervisor, sim))
    if seal_access.value != seal:
        raise RuntimeError("endpoint seal was not bound")
    sampling = {}
    if getattr(args, 'raw_grid_phase', False):
        sampling['raw_grid_phase_sampling'] = dataset.report()
        assert min(dataset.phase_counts) > 0
    if getattr(args, 'real_time_jitter', False):
        sampling = {'real_time_jitter_sampling': dataset.report()}
        if dataset.report()['max_support_overrun'] or min(dataset.report()['shift_histogram']) == 0:
            raise RuntimeError('Real time jitter bounds or support coverage failed')
    if args.constant_controller_tke and time.perf_counter() > deadline:
        raise TimeoutError("constant controller TKE budget expired; endpoint not admitted")
    result = {
        **plan(args), "decision": "REAL4000_SEALED_UQ_AND_COMPARISON_PENDING",
        "start": start, "manifest": manifest,
        "normalizer": {"path": str((args.output_dir / "point_normalizer.pt").resolve()),
                       "raw_statistics_sha256": smu1.statistics_digest(statistics)},
        "sample_stream_sha256": schedule["sha256"],
        "ordered_prefix_window_sha256": smu1.ordered_held_key_digest(
            range(len(windows)), windows, metadata),
        "training": training, "seal": seal,
        **sampling,
        **({'reflection_draws': dataset.seen, 'reflected_windows': dataset.reflected}
           if getattr(args, 'vertical_reflection', False) else {}),
        "prefix_files_opened": audit.opened,
        **({'full_training_files_opened': full_audit.opened} if full_audit else {}),
        "max_read_frame_exclusive_by_file": {
            row["file"]: int(row['frames'] if full_audit else row["val_boundary"]) for row in rows},
        "elapsed_seconds": time.perf_counter() - started,
    }
    if ARMS[args.arm][1]:
        sidecar = args.output_dir / "training_only_tke_heads_u4000.pth"
        torch.save({"schema": SCHEMA + "_supervisor", "deployable": False,
                    "model_endpoint_sha256": seal["file_sha256"],
                    "state_dict": {k: v.detach().cpu()
                                   for k, v in supervisor.state_dict().items()}}, sidecar)
        result["training_only_supervisor"] = {
            "path": str(sidecar.resolve()), "sha256": sha256(sidecar),
            "deployable": False}
    dr1.write_json_atomic(args.output_dir / "result.json", result)
    dr1.write_json_atomic(args.output_dir / "point_endpoint.json", {
        "schema": SCHEMA + "_point_receipt", "arm": args.arm,
        "held_aoa": held_aoa, "fit_files": [r['file'] for r in rows],
        "fit_windows": len(manifest_windows),
        "optimizer_windows": len(windows),
        "holdout_scope": "Real/UQ stages only; shared Sim ancestors are not held out",
        **intervention_fields(args),
        **sampling,
        "checkpoint_path": seal["path"],
        "checkpoint_sha256": seal["file_sha256"],
        "updates": UPDATES, "ema_decay": small4k.EMA_DECAY,
        "seed": seed, "sample_stream_sha256": schedule["sha256"],
        "start_sim_sha256": args.start_sim_sha256.lower(),
        "start_metadata_sha256": args.start_metadata_sha256.lower(),
        "sim_family": ARMS[args.arm][0],
        "auxiliary_enabled": ARMS[args.arm][1],
        "tail_read": getattr(args, 'full_real_trajectories', False), "uq_pending": True,
    })
    return result


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
