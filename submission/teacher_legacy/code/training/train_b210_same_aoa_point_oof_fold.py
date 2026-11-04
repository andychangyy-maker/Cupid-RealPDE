#!/usr/bin/env python3
"""Train one trajectory-disjoint, all-AoA B210 point cross-fit endpoint.

Each of five folds restarts from the formal pure-Sim10k checkpoint.  Complete
held trajectories are absent from the Real4k optimizer and from normalizer
statistics, while both sides retain all five AoAs.  Training uses every legal
stride-5 window of the fit trajectories and the historical manifest-prefix
windows for fit-only normalization.  Held H5 files are never opened here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import time
from typing import Iterator

import torch

import b210_same_aoa_trajectory_folds as fold_plan
import train_b210_full_real_point_oof_fold as whole_aoa


support = whole_aoa.support
dr1 = whole_aoa.dr1
ft12 = whole_aoa.ft12
smu1 = whole_aoa.smu1
small4k = whole_aoa.small4k
sim_tke = whole_aoa.sim_tke

EXPERIMENT = "b210_same_aoa_trajectory_point_oof"
SCHEMA = "b210_same_aoa_trajectory_point_oof_fold_v1"
ENDPOINT_SCHEMA = "b210_same_aoa_trajectory_point_oof_endpoint_v1"
ACKNOWLEDGEMENT = "RUN_B210_SAME_AOA_TRAJECTORY_POINT_OOF_V1"
COMPLETION_EVENT = "b210_same_aoa_point_oof_fold_complete"
FOLDS = fold_plan.FOLDS
ENDPOINT = whole_aoa.ENDPOINT
UPDATES = whole_aoa.UPDATES
OPTIMIZER_STRIDE = whole_aoa.OPTIMIZER_STRIDE
EVALUATION_STRIDE = whole_aoa.EVALUATION_STRIDE
SCHEDULER_T_MAX = whole_aoa.SCHEDULER_T_MAX
START_SHA256 = whole_aoa.START_SHA256
START_EXPECTED_ITERATION = 10_000
START_EXPECTED_STATUS = sim_tke.PASS_STATUS
MODEL_LINEAGE = "b210_pure_sim10k"
LINEAGE_FIELD_REQUIRED = False
UQ_SOURCE_LABEL = "pure_sim10k"
FINAL_POINT_RESULT_SCHEMA = "b210_full_released_real_final_point_v1"
FINAL_POINT_ENDPOINT_SCHEMA = "b210_full_released_real_final_endpoint_v1"
FORMAL_FINAL_POINT_RESULT_SHA256 = None
MIXED_AOA_SENTINEL = -1


def architecture_contract() -> tuple:
    """Return the exact topology and parameter counts for this lineage."""
    return (
        sim_tke.b210_config("candidate"),
        ft12.EXPECTED_PARAMETERS,
        ft12.EXPECTED_CONTROLLER_PARAMETERS,
    )


class FitOnlyAccessAudit:
    """Open exactly the planned fit allowlist before one immutable endpoint."""

    def __init__(self, held_fold: int, fit_files: Sequence[str],
                 held_files: Sequence[str]) -> None:
        if held_fold not in fold_plan.FOLD_VALUES:
            raise ValueError(f"unsupported held fold {held_fold}")
        self.held_fold = int(held_fold)
        self.fit_files = sorted(str(value) for value in fit_files)
        self.held_files = sorted(str(value) for value in held_files)
        if set(self.fit_files).intersection(self.held_files):
            raise ValueError("fit and held trajectory allowlists overlap")
        self.opened: list[str] = []
        self.endpoint: dict | None = None

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        if (phase != "fit" or filename not in self.fit_files
                or filename in self.held_files or filename in self.opened
                or filename == dr1.EXCLUDED_BASENAME
                or self.endpoint is not None):
            raise RuntimeError("same-AoA point fold rejected a non-fit H5 open")

    def record_open(self, row: Mapping[str, object], phase: str) -> None:
        self.before_open(row, phase)
        self.opened.append(str(row["file"]))

    def seal(self, value: Mapping[str, object]) -> None:
        if self.endpoint is not None:
            raise RuntimeError("same-AoA point endpoint sealed twice")
        if (value.get("endpoint") != ENDPOINT
                or value.get("primary") is not True
                or int(value.get("iteration", -1)) != UPDATES):
            raise RuntimeError("same-AoA point endpoint identity changed")
        path = Path(str(value["path"]))
        if not path.is_file() or dr1.sha256(path) != value.get("file_sha256"):
            raise RuntimeError("same-AoA point endpoint file is not sealed")
        self.endpoint = dict(value)

    def report(self) -> dict:
        return {
            "held_fold": self.held_fold,
            "opened_file_allowlist": sorted(self.opened),
            "expected_fit_allowlist": self.fit_files,
            "held_file_denylist": self.held_files,
            "allowlist_exact": sorted(self.opened) == self.fit_files,
            "held_trajectories_opened": False,
            "clean716_manifest_read": False,
            "evaluation_manifest_read": False,
            "excluded_opened": dr1.EXCLUDED_BASENAME in self.opened,
            "endpoint_seal": self.endpoint,
        }


@contextmanager
def endpoint_contract() -> Iterator[None]:
    """Scope the reused Real4k trainer to a mixed-AoA held-fold endpoint."""
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "PRIMARY_ENDPOINT": ENDPOINT,
        "UPDATES": UPDATES,
        "SCHEDULER_T_MAX": SCHEDULER_T_MAX,
        # There is deliberately no single held AoA in this protocol.
        "HELD_AOA": MIXED_AOA_SENTINEL,
    }
    original = {name: getattr(small4k, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(small4k, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(small4k, name, value)


def validate_endpoint(
        path: Path, seal: Mapping[str, object], *, held_fold: int) -> dict:
    if held_fold not in fold_plan.FOLD_VALUES:
        raise ValueError(f"unsupported held fold {held_fold}")
    if not path.is_file() or dr1.sha256(path) != seal.get("file_sha256"):
        raise RuntimeError("same-AoA point checkpoint file SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != ENDPOINT_SCHEMA
            or payload.get("experiment") != EXPERIMENT
            or payload.get("phase") != "real"
            or int(payload.get("iteration", -1)) != UPDATES
            or payload.get("endpoint") != ENDPOINT
            or payload.get("primary") is not True
            or int(payload.get("held_aoa", 0)) != MIXED_AOA_SENTINEL
            or not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal.get("model_state_sha256")
            or payload.get("start", {}).get("checkpoint_sha256")
            != START_SHA256
            or int(payload.get("start", {}).get("iteration", -1))
            != START_EXPECTED_ITERATION
            or payload.get("start", {}).get("status")
            != START_EXPECTED_STATUS):
        raise RuntimeError("same-AoA point checkpoint payload identity changed")
    expected_config, expected_parameters, expected_controller = (
        architecture_contract())
    config = ft12.validate_b210_model_config(
        payload["model_config"], expected_config)
    model = small4k.EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    if (sum(value.numel() for value in model.parameters())
            != expected_parameters
            or sum(value.numel() for value in model.energy_controller.parameters())
            != expected_controller):
        raise RuntimeError("same-AoA point parameter count changed")
    normalizer = payload.get("normalizer")
    if (not isinstance(normalizer, Mapping)
            or dr1.canonical_sha256(normalizer)
            != payload.get("normalizer_sha256")):
        raise RuntimeError("same-AoA point normalizer changed")
    return {
        "path": str(path.resolve()),
        "file_sha256": dr1.sha256(path),
        "model_state_sha256": dr1.model_state_sha256(state),
        "model_config_sha256": dr1.canonical_sha256(payload["model_config"]),
        "normalizer_sha256": payload["normalizer_sha256"],
        "sample_stream_sha256": payload["sample_stream_sha256"],
        "endpoint": ENDPOINT,
        "iteration": UPDATES,
        "primary": True,
        "held_fold": held_fold,
        "model_lineage": MODEL_LINEAGE,
        "checkpoint_held_aoa_sentinel": MIXED_AOA_SENTINEL,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal same-AoA point cross-fit requires CUDA")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 runtime manifest file identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    plan = fold_plan.build_plan(rows)
    assignments = {row["file"]: int(row["fold"])
                   for row in plan["assignments"]}
    expected = fold_plan.EXPECTED_FOLDS[args.held_fold]
    fit_rows = [row for row in rows
                if assignments[str(row["file"])] != args.held_fold]
    held_rows = [row for row in rows
                 if assignments[str(row["file"])] == args.held_fold]
    observed = fold_plan.fold_counts(rows, assignments, args.held_fold)
    if observed != expected:
        raise RuntimeError("same-AoA point fold counts changed")
    start_sha = dr1.sha256(args.start_sim)
    if start_sha != START_SHA256 or start_sha != args.start_sim_sha256:
        raise RuntimeError("same-AoA point Sim-lineage start changed")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    audit = FitOnlyAccessAudit(
        args.held_fold,
        [str(row["file"]) for row in fit_rows],
        [str(row["file"]) for row in held_rows])
    trajectories, metadata = support.load_full_rows(
        args.real_dir, fit_rows, "fit", audit)
    normalizer_windows = support.manifest_prefix_windows(fit_rows)
    optimizer_windows = support.support_windows(
        trajectories, metadata, support="full", stride=OPTIMIZER_STRIDE)
    if (len(normalizer_windows) != expected["normalizer_prefix_windows"]
            or len(optimizer_windows) != expected["full_optimizer_windows"]):
        raise RuntimeError("same-AoA point materialized window count changed")
    statistics = smu1.fold_statistics(
        smu1.window_sufficient_statistics(trajectories, normalizer_windows),
        range(len(normalizer_windows)))
    normalizer = smu1.FoldNormalizer(statistics, device)
    torch.save(tuple(value.detach().cpu() for value in statistics),
               args.output_dir / "fit_normalizer.pt")
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, optimizer_windows)
    schedule = smu1.paired_batch_schedule(
        range(len(optimizer_windows)), optimizer_windows,
        updates=UPDATES, seed=support.SEED)
    schedule["updates"] = UPDATES
    ft12.npt1.ensemble_train.seed_everything(support.SEED)
    expected_config, expected_parameters, expected_controller = (
        architecture_contract())
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha, normalizer, device,
        expected_config=expected_config,
        expected_parameters=expected_parameters,
        expected_controller_parameters=expected_controller,
        expected_iteration=START_EXPECTED_ITERATION,
        expected_status=START_EXPECTED_STATUS)
    initial_state_sha = dr1.model_state_sha256(model.state_dict())
    with endpoint_contract():
        training, seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            args.output_dir, start_report, audit,
            input_augmentation=support.AUGMENTATION,
            endpoint_filename="ema_u4000.pth",
            expected_parameters=expected_parameters,
            scheduler_t_max=SCHEDULER_T_MAX)
    if audit.endpoint != seal:
        raise RuntimeError("same-AoA point endpoint was not audit-bound")
    verified = validate_endpoint(
        Path(str(seal["path"])), seal, held_fold=args.held_fold)
    access = audit.report()
    if not (access["allowlist_exact"]
            and not access["held_trajectories_opened"]
            and not access["clean716_manifest_read"]
            and not access["excluded_opened"]):
        raise RuntimeError("same-AoA point fit-only access contract failed")
    held_assignments = [row for row in plan["assignments"]
                        if int(row["fold"]) == args.held_fold]
    result = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "held_fold": args.held_fold,
        "folds": FOLDS,
        "fold_definition": fold_plan.FORMULA,
        "fold_plan_schema": fold_plan.SCHEMA,
        "fold_plan_assignment_sha256": plan["assignment_sha256"],
        "held_assignment_sha256": dr1.canonical_sha256(held_assignments),
        "held_assignments": held_assignments,
        "held_aoas": list(fold_plan.AOA_VALUES),
        "held_files": [row["file"] for row in held_assignments],
        "manifest": manifest_report,
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "inventory": plan["inventory"],
        "counts": expected,
        "model_lineage": MODEL_LINEAGE,
        "start": {
            "checkpoint_sha256": START_SHA256,
            "iteration": START_EXPECTED_ITERATION,
            "status": START_EXPECTED_STATUS,
            "model_lineage": MODEL_LINEAGE,
        },
        "training_scope": "other-four-fold complete trajectories stride5",
        "normalizer_scope": "other-four-fold manifest prefix windows",
        "residual_scope": "held-fold complete trajectories stride20",
        "fit": {
            "files": len(fit_rows),
            "normalizer_prefix_windows": len(normalizer_windows),
            "full_optimizer_windows": len(optimizer_windows),
            "normalizer_sha256": smu1.statistics_digest(statistics),
            "sample_stream_sha256": schedule["sha256"],
            "initial_model_state_sha256": initial_state_sha,
            "training": training,
            "seal": verified,
        },
        "access_audit": access,
        "held_data_opened": False,
        "clean716_manifest_read": False,
        "evaluation_manifest_read": False,
        "endpoint_selection": False,
        "elapsed_seconds": time.perf_counter() - started,
        "submission_authorized": False,
        "decision": "COMPLETE_SAME_AOA_TRAJECTORY_POINT_OOF_FOLD",
    }
    del model, dataset, trajectories
    gc.collect()
    torch.cuda.empty_cache()
    return whole_aoa.seal_json_payload(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-fold", action="store_true")
    mode.add_argument("--contract-only", action="store_true")
    parser.add_argument("--held-fold", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 60 * 60)
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    values = (args.held_fold, args.output_dir, args.real_dir,
              args.clean_manifest, args.start_sim, args.start_sim_sha256)
    if args.contract_only:
        if any(value is not None for value in values):
            raise ValueError("contract-only mode received execution inputs")
        return
    if any(value is None for value in values):
        raise ValueError("same-AoA point fold is missing required input")
    if args.held_fold not in fold_plan.FOLD_VALUES:
        raise ValueError(f"unsupported held fold {args.held_fold}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (not args.execute_formal_gpu
            or args.acknowledge_protocol != ACKNOWLEDGEMENT):
        raise PermissionError("formal same-AoA point fold needs acknowledgement")
    args.start_sim_sha256 = dr1.normalized_sha256(
        args.start_sim_sha256, "same-AoA point Sim-lineage SHA")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        print(json.dumps({
            "schema": SCHEMA,
            "model_lineage": MODEL_LINEAGE,
            "start_checkpoint_sha256": START_SHA256,
            "start_iteration": START_EXPECTED_ITERATION,
            "start_status": START_EXPECTED_STATUS,
            "architecture_parameters": architecture_contract()[1],
            "controller_parameters": architecture_contract()[2],
            "uq_source_label": UQ_SOURCE_LABEL,
            "formal_final_point_result_sha256": (
                FORMAL_FINAL_POINT_RESULT_SHA256),
            "fold_plan_schema": fold_plan.SCHEMA,
            "fold_formula": fold_plan.FORMULA,
            "assignment_sha256": fold_plan.EXPECTED_ASSIGNMENT_SHA256,
            "expected_total": fold_plan.EXPECTED_TOTAL,
            "expected_folds": fold_plan.EXPECTED_FOLDS,
            "every_fold_covers_all_aoas": True,
            "trajectory_disjoint": True,
            "clean716_manifest_accepted": False,
            "evaluation_manifest_accepted": False,
        }, indent=2, sort_keys=True))
        return
    result = run(args)
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    whole_aoa.verify_json_payload(json.loads(
        (args.output_dir / "fold_result.json").read_text(encoding="utf-8")))
    print(json.dumps({
        "event": COMPLETION_EVENT,
        "held_fold": args.held_fold,
        "result": str((args.output_dir / "fold_result.json").resolve()),
    }), flush=True)


if __name__ == "__main__":
    main()
