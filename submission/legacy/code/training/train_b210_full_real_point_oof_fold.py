#!/usr/bin/env python3
"""Train or adopt one whole-AoA B210 point-OOF endpoint.

Every trained fold restarts from the SHA-bound pure-Sim10k checkpoint.  The
held AoA is absent from both the prefix-derived normalizer and the complete
released-Real stride-5 optimizer support.  This program never accepts an
evaluation/clean716 manifest and never opens held trajectories.

The already sealed full-support held-AoA 10/20 endpoints can be adopted
without retraining.  Adoption verifies the source JSON, checkpoint file,
model state, normalizer, and start identities before emitting the same fold
result interface used by newly trained AoA 0/5/15 endpoints.
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

import train_b210_full_released_real_support as support


dr1 = support.dr1
ft12 = support.ft12
smu1 = support.smu1
small4k = support.small4k
sim_tke = support.sim_tke

EXPERIMENT = "b210_full_real_point_oof"
SCHEMA = "b210_full_real_point_oof_fold_v1"
ENDPOINT_SCHEMA = "b210_full_real_point_oof_endpoint_v1"
ACKNOWLEDGEMENT = "RUN_B210_FULL_REAL_POINT_OOF_V1"
AOA_VALUES = (0, 5, 10, 15, 20)
FOLDS = len(AOA_VALUES)
ENDPOINT = "ema_u4000"
UPDATES = 4_000
OPTIMIZER_STRIDE = 5
EVALUATION_STRIDE = 20
SCHEDULER_T_MAX = 12_000
START_SHA256 = support.FORMAL_START_SHA256
EXPECTED_TOTAL = {
    "trajectories": 81,
    "normalizer_prefix_windows": 2_544,
    "full_optimizer_windows": 13_202,
    "full_oof_windows": 3_341,
}
EXPECTED_FOLDS = {
    0: {"fit_trajectories": 65, "held_trajectories": 16,
        "normalizer_prefix_windows": 2_042, "full_optimizer_windows": 10_598,
        "held_oof_windows": 659},
    5: {"fit_trajectories": 67, "held_trajectories": 14,
        "normalizer_prefix_windows": 2_111, "full_optimizer_windows": 10_953,
        "held_oof_windows": 569},
    10: {"fit_trajectories": 63, "held_trajectories": 18,
         "normalizer_prefix_windows": 1_968, "full_optimizer_windows": 10_214,
         "held_oof_windows": 756},
    15: {"fit_trajectories": 64, "held_trajectories": 17,
         "normalizer_prefix_windows": 2_000, "full_optimizer_windows": 10_380,
         "held_oof_windows": 714},
    20: {"fit_trajectories": 65, "held_trajectories": 16,
         "normalizer_prefix_windows": 2_055, "full_optimizer_windows": 10_663,
         "held_oof_windows": 643},
}
JSON_SELF_SEAL_KEY = "json_self_seal"


def seal_json_payload(payload: Mapping[str, object]) -> dict:
    """Return a JSON-native payload sealed after dump/load normalization."""
    if JSON_SELF_SEAL_KEY in payload:
        raise ValueError("JSON payload is already self-sealed")
    normalized = json.loads(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    digest = dr1.canonical_sha256(normalized)
    normalized[JSON_SELF_SEAL_KEY] = {
        "algorithm": "sha256",
        "payload_without_self_seal_sha256": digest,
        "computed_after_json_roundtrip": True,
    }
    verify_json_payload(normalized)
    return normalized


def verify_json_payload(payload: Mapping[str, object]) -> str:
    normalized = json.loads(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    seal = normalized.pop(JSON_SELF_SEAL_KEY, None)
    if (not isinstance(seal, Mapping)
            or seal.get("algorithm") != "sha256"
            or seal.get("computed_after_json_roundtrip") is not True):
        raise RuntimeError("JSON round-trip self-seal is missing")
    digest = dr1.canonical_sha256(normalized)
    if digest != seal.get("payload_without_self_seal_sha256"):
        raise RuntimeError("JSON round-trip self-seal changed")
    return digest


def aoa_fold(aoa: int) -> int:
    try:
        return AOA_VALUES.index(int(aoa))
    except ValueError as error:
        raise ValueError(f"unsupported held AoA {aoa}") from error


def full_window_count(rows: Sequence[Mapping[str, object]], stride: int) -> int:
    if stride <= 0:
        raise ValueError("window stride must be positive")
    total = 0
    for row in rows:
        frames = int(row["frames"])
        if frames < 40:
            raise ValueError("released trajectory cannot form 20-to-20 window")
        total += len(range(0, frames - 40 + 1, stride))
    return total


def validate_inventory(rows: Sequence[Mapping[str, object]]) -> dict:
    if (len(rows) != EXPECTED_TOTAL["trajectories"]
            or {int(row["aoa"]) for row in rows} != set(AOA_VALUES)
            or len({str(row["file"]) for row in rows}) != len(rows)
            or any(str(row["file"]) == dr1.EXCLUDED_BASENAME for row in rows)):
        raise RuntimeError("clean81 trajectory inventory changed")
    observed = {
        "trajectories": len(rows),
        "normalizer_prefix_windows": sum(
            len(row["train_starts"]) for row in rows),
        "full_optimizer_windows": full_window_count(rows, OPTIMIZER_STRIDE),
        "full_oof_windows": full_window_count(rows, EVALUATION_STRIDE),
    }
    if observed != EXPECTED_TOTAL:
        raise RuntimeError(f"full Real inventory changed: {observed}")
    return observed


class FitOnlyAccessAudit:
    """Allow exactly other-AoA complete trajectories before one EMA seal."""

    def __init__(self, held_aoa: int, expected_files: Sequence[str]) -> None:
        aoa_fold(held_aoa)
        self.held_aoa = int(held_aoa)
        self.expected_files = sorted(str(value) for value in expected_files)
        self.opened: list[str] = []
        self.endpoint: dict | None = None

    def before_open(self, row: Mapping[str, object], phase: str) -> None:
        filename = str(row["file"])
        if (phase != "fit" or int(row["aoa"]) == self.held_aoa
                or filename == dr1.EXCLUDED_BASENAME
                or filename not in self.expected_files
                or filename in self.opened or self.endpoint is not None):
            raise RuntimeError("point-OOF opener rejected non-fit/held/late data")

    def record_open(self, row: Mapping[str, object], phase: str) -> None:
        self.before_open(row, phase)
        self.opened.append(str(row["file"]))

    def seal(self, value: Mapping[str, object]) -> None:
        if self.endpoint is not None:
            raise RuntimeError("point-OOF endpoint sealed twice")
        if (value.get("endpoint") != ENDPOINT
                or value.get("primary") is not True
                or int(value.get("iteration", -1)) != UPDATES):
            raise RuntimeError("point-OOF endpoint identity changed")
        path = Path(str(value["path"]))
        if not path.is_file() or dr1.sha256(path) != value.get("file_sha256"):
            raise RuntimeError("point-OOF endpoint file is not sealed")
        self.endpoint = dict(value)

    def report(self) -> dict:
        return {
            "held_aoa": self.held_aoa,
            "opened_file_allowlist": sorted(self.opened),
            "expected_file_allowlist": self.expected_files,
            "allowlist_exact": sorted(self.opened) == self.expected_files,
            "held_trajectories_opened": False,
            "clean716_manifest_read": False,
            "excluded_opened": dr1.EXCLUDED_BASENAME in self.opened,
            "endpoint_seal": self.endpoint,
        }


@contextmanager
def endpoint_contract(held_aoa: int) -> Iterator[None]:
    aoa_fold(held_aoa)
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "ENDPOINT_SCHEMA": ENDPOINT_SCHEMA,
        "PRIMARY_ENDPOINT": ENDPOINT,
        "UPDATES": UPDATES,
        "SCHEDULER_T_MAX": SCHEDULER_T_MAX,
        "HELD_AOA": int(held_aoa),
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
        path: Path, seal: Mapping[str, object], *, held_aoa: int,
        accepted_schemas: Sequence[str], accepted_experiments: Sequence[str],
        ) -> dict:
    if not path.is_file() or dr1.sha256(path) != seal.get("file_sha256"):
        raise RuntimeError("point-OOF checkpoint file SHA changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if (payload.get("schema") not in set(accepted_schemas)
            or payload.get("experiment") not in set(accepted_experiments)
            or payload.get("phase") != "real"
            or int(payload.get("iteration", -1)) != UPDATES
            or payload.get("endpoint") != ENDPOINT
            or payload.get("primary") is not True
            or int(payload.get("held_aoa", -1)) != int(held_aoa)
            or not isinstance(state, Mapping)
            or dr1.model_state_sha256(state) != seal.get("model_state_sha256")
            or payload.get("start", {}).get("checkpoint_sha256") != START_SHA256):
        raise RuntimeError("point-OOF checkpoint payload identity changed")
    config = ft12.validate_b210_model_config(
        payload["model_config"], sim_tke.b210_config("candidate"))
    model = small4k.EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    if sum(value.numel() for value in model.parameters()) != ft12.EXPECTED_PARAMETERS:
        raise RuntimeError("point-OOF checkpoint parameter count changed")
    normalizer = payload.get("normalizer")
    if (not isinstance(normalizer, Mapping)
            or dr1.canonical_sha256(normalizer)
            != payload.get("normalizer_sha256")):
        raise RuntimeError("point-OOF checkpoint normalizer changed")
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
    }


def base_result(
        args: argparse.Namespace, manifest_report: Mapping[str, object],
        inventory: Mapping[str, int], expected: Mapping[str, int],
        ) -> dict:
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "held_aoa": args.held_aoa,
        "held_fold": aoa_fold(args.held_aoa),
        "folds": FOLDS,
        "fold_definition": "one complete released AoA per fold",
        "aoa_values": list(AOA_VALUES),
        "manifest": dict(manifest_report),
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "inventory": dict(inventory),
        "counts": dict(expected),
        "start": {"checkpoint_sha256": START_SHA256},
        "training_scope": "other-four-AoA full trajectories stride5",
        "normalizer_scope": "other-four-AoA manifest prefix windows",
        "residual_scope": "held-AoA every legal full-trajectory stride20 window",
        "held_data_opened": False,
        "clean716_manifest_read": False,
        "endpoint_selection": False,
        "submission_authorized": False,
    }


def train_fold(
        args: argparse.Namespace, rows: Sequence[Mapping[str, object]],
        manifest_report: Mapping[str, object], inventory: Mapping[str, int],
        ) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal point-OOF training requires CUDA")
    start_sha = dr1.sha256(args.start_sim)
    if start_sha != START_SHA256 or start_sha != args.start_sim_sha256:
        raise RuntimeError("point-OOF pure-Sim10k start changed")
    expected = EXPECTED_FOLDS[args.held_aoa]
    fit_rows = [row for row in rows if int(row["aoa"]) != args.held_aoa]
    held_rows = [row for row in rows if int(row["aoa"]) == args.held_aoa]
    if (len(fit_rows) != expected["fit_trajectories"]
            or len(held_rows) != expected["held_trajectories"]):
        raise RuntimeError("whole-AoA fold membership changed")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + args.wall_budget_seconds
    audit = FitOnlyAccessAudit(
        args.held_aoa, [str(row["file"]) for row in fit_rows])
    trajectories, metadata = support.load_full_rows(
        args.real_dir, fit_rows, "fit", audit)
    normalizer_windows = support.manifest_prefix_windows(fit_rows)
    optimizer_windows = support.support_windows(
        trajectories, metadata, support="full", stride=OPTIMIZER_STRIDE)
    if (len(normalizer_windows) != expected["normalizer_prefix_windows"]
            or len(optimizer_windows) != expected["full_optimizer_windows"]
            or full_window_count(held_rows, EVALUATION_STRIDE)
            != expected["held_oof_windows"]):
        raise RuntimeError("whole-AoA fold window count changed")
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
    model, start_report = ft12.validate_start_checkpoint(
        args.start_sim, start_sha, normalizer, device,
        expected_config=sim_tke.b210_config("candidate"),
        expected_parameters=ft12.EXPECTED_PARAMETERS,
        expected_controller_parameters=ft12.EXPECTED_CONTROLLER_PARAMETERS)
    initial_state_sha = dr1.model_state_sha256(model.state_dict())
    with endpoint_contract(args.held_aoa):
        training, seal = small4k.train_candidate(
            model, dataset, schedule, normalizer, device, deadline,
            args.output_dir, start_report, audit,
            input_augmentation=support.AUGMENTATION,
            endpoint_filename="ema_u4000.pth",
            expected_parameters=ft12.EXPECTED_PARAMETERS,
            scheduler_t_max=SCHEDULER_T_MAX)
    if audit.endpoint != seal:
        raise RuntimeError("trained point-OOF endpoint was not audit-bound")
    verified = validate_endpoint(
        Path(str(seal["path"])), seal, held_aoa=args.held_aoa,
        accepted_schemas=(ENDPOINT_SCHEMA,),
        accepted_experiments=(EXPERIMENT,))
    access = audit.report()
    if not (access["allowlist_exact"]
            and not access["held_trajectories_opened"]
            and not access["clean716_manifest_read"]
            and not access["excluded_opened"]):
        raise RuntimeError("point-OOF fit-only access contract failed")
    result = base_result(args, manifest_report, inventory, expected)
    result.update({
        "source_kind": "trained_from_pure_sim10k",
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
        "elapsed_seconds": time.perf_counter() - started,
        "decision": "COMPLETE_FULL_REAL_POINT_OOF_FOLD",
    })
    del model, dataset, trajectories
    gc.collect()
    torch.cuda.empty_cache()
    return result


def adopt_fold(
        args: argparse.Namespace, manifest_report: Mapping[str, object],
        inventory: Mapping[str, int]) -> dict:
    if args.held_aoa not in support.HELD_AOAS:
        raise ValueError("only held-AoA10/20 full-support endpoints are reusable")
    actual = dr1.sha256(args.full_support_result)
    if actual != args.full_support_result_sha256:
        raise RuntimeError("full-support result SHA changed")
    source = json.loads(args.full_support_result.read_text(encoding="utf-8"))
    if (source.get("schema") != support.SCHEMA
            or source.get("contract_sha256") != support.CONTRACT_SHA256
            or int(source.get("held_aoa", -1)) != args.held_aoa
            or source.get("primary_endpoint") != support.PRIMARY_ENDPOINT
            or source.get("decision") != "COMPLETE_MATCHED_EVIDENCE_NO_SCORE_GATE"
            or source.get("start", {}).get("checkpoint_sha256") != START_SHA256
            or source.get("access_audit", {}).get(
                "all_seals_verified_before_held") is not True
            or source.get("access_audit", {}).get("clean716_manifest_read") is not False):
        raise RuntimeError("full-support endpoint is not adoptable")
    expected = EXPECTED_FOLDS[args.held_aoa]
    full = source.get("arms", {}).get("full", {})
    if (int(full.get("fit_windows", -1)) != expected["full_optimizer_windows"]
            or int(source.get("counts", {}).get("observed_normalizer_prefix_windows", -1))
            != expected["normalizer_prefix_windows"]):
        raise RuntimeError("adopted full-support fold counts changed")
    seal = full.get("endpoints", {}).get(ENDPOINT, {}).get("seal", {})
    verified = validate_endpoint(
        Path(str(seal.get("path", ""))), seal, held_aoa=args.held_aoa,
        accepted_schemas=(support.ENDPOINT_SCHEMA,),
        accepted_experiments=(support.EXPERIMENT,))
    result = base_result(args, manifest_report, inventory, expected)
    result.update({
        "source_kind": "adopted_full_support_full_arm",
        "source": {
            "path": str(args.full_support_result.resolve()),
            "sha256": actual,
            "schema": support.SCHEMA,
            "contract_sha256": support.CONTRACT_SHA256,
        },
        "fit": {
            "files": expected["fit_trajectories"],
            "normalizer_prefix_windows": expected["normalizer_prefix_windows"],
            "full_optimizer_windows": expected["full_optimizer_windows"],
            "normalizer_sha256": verified["normalizer_sha256"],
            "sample_stream_sha256": verified["sample_stream_sha256"],
            "seal": verified,
        },
        "access_audit": {
            "source_access_audit_sha256": dr1.canonical_sha256(
                source["access_audit"]),
            "held_target_was_opened_only_after_all_source_endpoints": True,
            "clean716_manifest_read": False,
        },
        "elapsed_seconds": 0.0,
        "decision": "COMPLETE_FULL_REAL_POINT_OOF_FOLD",
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-fold", action="store_true")
    mode.add_argument("--adopt-full-support", action="store_true")
    mode.add_argument("--contract-only", action="store_true")
    parser.add_argument("--held-aoa", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--start-sim", type=Path)
    parser.add_argument("--start-sim-sha256")
    parser.add_argument("--full-support-result", type=Path)
    parser.add_argument("--full-support-result-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3 * 60 * 60)
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-protocol", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.contract_only:
        supplied = (
            args.held_aoa, args.output_dir, args.real_dir, args.clean_manifest,
            args.start_sim, args.start_sim_sha256, args.full_support_result,
            args.full_support_result_sha256)
        if any(value is not None for value in supplied):
            raise ValueError("contract-only mode received execution inputs")
        return
    required = (args.held_aoa, args.output_dir, args.clean_manifest)
    if any(value is None for value in required):
        raise ValueError("point-OOF fold is missing required inputs")
    aoa_fold(args.held_aoa)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.train_fold:
        values = (args.real_dir, args.start_sim, args.start_sim_sha256)
        if any(value is None for value in values):
            raise ValueError("trained point-OOF fold is missing data/start inputs")
        if (args.full_support_result is not None
                or args.full_support_result_sha256 is not None):
            raise ValueError("trained fold received adoption inputs")
        if (not args.execute_formal_gpu
                or args.acknowledge_protocol != ACKNOWLEDGEMENT):
            raise PermissionError("formal point-OOF training needs acknowledgement")
        args.start_sim_sha256 = dr1.normalized_sha256(
            args.start_sim_sha256, "point-OOF Sim10k SHA")
    else:
        if (args.full_support_result is None
                or args.full_support_result_sha256 is None):
            raise ValueError("adoption needs source result and SHA")
        args.full_support_result_sha256 = dr1.normalized_sha256(
            args.full_support_result_sha256, "full-support result SHA")
        if (args.real_dir is not None or args.start_sim is not None
                or args.start_sim_sha256 is not None
                or args.execute_formal_gpu or args.acknowledge_protocol):
            raise ValueError("adoption received training-only inputs")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.contract_only:
        print(json.dumps({
            "schema": SCHEMA,
            "aoa_values": list(AOA_VALUES),
            "expected_total": EXPECTED_TOTAL,
            "expected_folds": EXPECTED_FOLDS,
            "clean716_manifest_accepted": False,
        }, indent=2, sort_keys=True))
        return
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 runtime manifest file identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    inventory = validate_inventory(rows)
    result = (train_fold(args, rows, manifest_report, inventory)
              if args.train_fold
              else adopt_fold(args, manifest_report, inventory))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = seal_json_payload(result)
    dr1.write_json_atomic(args.output_dir / "fold_result.json", result)
    reloaded = json.loads(
        (args.output_dir / "fold_result.json").read_text(encoding="utf-8"))
    verify_json_payload(reloaded)
    print(json.dumps({
        "event": "b210_full_real_point_oof_fold_complete",
        "held_aoa": args.held_aoa,
        "source_kind": result["source_kind"],
        "result": str((args.output_dir / "fold_result.json").resolve()),
    }), flush=True)


if __name__ == "__main__":
    main()
