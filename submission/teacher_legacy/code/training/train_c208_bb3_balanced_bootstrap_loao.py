#!/usr/bin/env python3
"""BB3 balanced-bootstrap paired held-Real gate for the unchanged C208.

The control keeps the legacy deterministic 0.75 trajectory bootstrap.  BB3
changes only the candidate trajectory/member assignment to an AoA-stratified,
exact-three-of-four mask.  The fixed725 temporal tail is never constructed.
Formal CUDA execution requires two explicit opt-ins; importing this module and
the default contract-only path are CPU-only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "code" / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_c208_smu1_paired_loao as smu1  # noqa: E402


EXPERIMENT = "c208_bb3_balanced_bootstrap_paired_loao"
FORMAL_ACK = "BB3_FIXED2576_ONLY"
MEMBER_COUNT = 4
MEMBERS_PER_TRAJECTORY = 3
SMU1_CONTROL_COMPATIBLE_COMMITS = frozenset({
    "17bc5f6de6e58881916e766ed7cbce11943f14c0",
})

AOA_VALUES = smu1.AOA_VALUES
FOLD_ORDER = smu1.FOLD_ORDER
EXPECTED_AOA_COUNTS = smu1.EXPECTED_AOA_COUNTS
EXPECTED_FIT_COUNTS = smu1.EXPECTED_FIT_COUNTS
EXPECTED_TRAJECTORY_COUNTS = {0: 17, 5: 14, 10: 18, 15: 17, 20: 16}
UPDATES = smu1.UPDATES
SEED = smu1.SEED


CONTRACT = {
    "hypothesis": (
        "The legacy pseudo-Bernoulli trajectory bootstrap is severely "
        "imbalanced within AoA.  Exact-three-of-four, AoA-balanced masks may "
        "improve member role coverage and hence Direct plus native SPS without "
        "changing the model or inference graph."),
    "evidence_class": (
        "paired Real-finetuning leave-one-AoA-out; not full-lifecycle true "
        "LOAO because the Sim P1 prior saw all Real conditions"),
    "single_variable": {
        "control": "legacy deterministic trajectory bootstrap p=0.75",
        "candidate": (
            "target-free AoA-stratified exact-3-of-4 trajectory mask; each "
            "fit AoA member trajectory count range <=1"),
        "model": "shared_member_uv_mean=false in both arms",
        "parameters": smu1.EXPECTED_PARAMETERS,
        "inference_graph": "bitwise-identical architecture and state schema",
    },
    "training": {
        **smu1.CONTRACT["training"],
        "paired_stream": (
            "same pre-generated sample-index and trajectory stream; only the "
            "candidate bootstrap mask differs"),
        "control_reuse": (
            "strictly validated SMU1 fold control_state may be reused; a "
            "missing fold is retrained, an incompatible existing fold STOPs"),
        "control_compatible_commits": sorted(
            SMU1_CONTROL_COMPATIBLE_COMMITS),
    },
    "gate": dict(smu1.CONTRACT["gate"]),
    "tail": (
        "fixed725 windows/targets are never constructed, sampled, or scored; "
        "the legacy loader still preloads complete raw H5 frame arrays"),
    "execution_guard": (
        "formal CUDA requires --execute-formal-gpu and "
        f"--acknowledge-fixed2576-only={FORMAL_ACK}"),
    "stop": (
        "STOP on provenance/control-reuse/stream/mask/leakage/nonfinite/OOM/"
        "budget failure, two non-positive AoA non-Time rows, or any strong "
        "gate miss; do not construct/score fixed725 windows or package a "
        "checkpoint"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-sim", type=Path, required=True)
    parser.add_argument(
        "--start-sim-sha256", default=smu1.FORMAL_START_SHA256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--reuse-smu1-control-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--execute-formal-gpu", action="store_true")
    parser.add_argument("--acknowledge-fixed2576-only")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    smu1.validate_args(args)
    if bool(args.contract_only) == bool(args.execute_formal_gpu):
        raise ValueError(
            "choose exactly one of --contract-only or --execute-formal-gpu")
    if args.execute_formal_gpu:
        require_formal_gpu_opt_in(args)
    if (args.reuse_smu1_control_dir is not None
            and not args.reuse_smu1_control_dir.is_dir()):
        raise FileNotFoundError(
            f"SMU1 control directory not found: "
            f"{args.reuse_smu1_control_dir}")


def require_formal_gpu_opt_in(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "execute_formal_gpu", False)) or (
            getattr(args, "acknowledge_fixed2576_only", None) != FORMAL_ACK):
        raise ValueError(
            "formal BB3 requires --execute-formal-gpu and the exact "
            "fixed2576-only acknowledgement")


def _stable_exclusion_key(seed: int, aoa: int, trajectory: int,
                          file_name: str) -> str:
    payload = (
        f"bb3-exact3-v1\0{seed}\0{aoa}\0{trajectory}\0{file_name}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def balanced_exact3_masks(
        fit_indices: Sequence[int], windows: Sequence,
        metadata: Sequence[dict], held_aoa: int, seed: int = SEED,
        ) -> tuple[dict[int, tuple[bool, ...]], dict]:
    """Create target-free exact-3/4 masks for only the four fit AoAs."""
    indices = [int(index) for index in fit_indices]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("fit indices must be a non-empty unique sequence")
    if held_aoa not in AOA_VALUES:
        raise ValueError(f"unexpected held AoA {held_aoa}")
    if any(int(metadata[int(windows[index].trajectory)]["aoa"]) == held_aoa
           for index in indices):
        raise RuntimeError("held-AoA trajectory entered the BB3 mask")
    expected_indices = {
        index for index, window in enumerate(windows)
        if int(metadata[int(window.trajectory)]["aoa"]) != held_aoa}
    if set(indices) != expected_indices:
        raise RuntimeError(
            "BB3 fit indices are not exactly the other-four windows")

    fit_trajectories = sorted({
        int(windows[index].trajectory) for index in indices})
    grouped: dict[int, list[int]] = {}
    for trajectory in fit_trajectories:
        item = metadata[trajectory]
        aoa = int(item["aoa"])
        if not str(item["file"]):
            raise ValueError("trajectory metadata has an empty file key")
        grouped.setdefault(aoa, []).append(trajectory)
    expected_fit_aoas = set(AOA_VALUES) - {held_aoa}
    if set(grouped) != expected_fit_aoas:
        raise RuntimeError(
            "BB3 mask does not contain exactly the other four AoAs")

    mapping: dict[int, tuple[bool, ...]] = {}
    per_aoa = {}
    for aoa in sorted(grouped):
        ordered = sorted(
            grouped[aoa],
            key=lambda trajectory: (
                _stable_exclusion_key(
                    seed, aoa, trajectory,
                    str(metadata[trajectory]["file"])),
                str(metadata[trajectory]["file"]), trajectory))
        offset = int(hashlib.sha256(
            f"bb3-offset-v1\0{seed}\0{aoa}".encode("ascii")
        ).hexdigest(), 16) % MEMBER_COUNT
        excluded_counts = [0] * MEMBER_COUNT
        for rank, trajectory in enumerate(ordered):
            excluded = (offset + rank) % MEMBER_COUNT
            excluded_counts[excluded] += 1
            mapping[trajectory] = tuple(
                member != excluded for member in range(MEMBER_COUNT))
        included_counts = [
            len(ordered) - excluded for excluded in excluded_counts]
        if max(included_counts) - min(included_counts) > 1:
            raise RuntimeError("BB3 per-AoA member balance contract failed")
        per_aoa[str(aoa)] = {
            "trajectories": len(ordered),
            "included_trajectories_per_member": included_counts,
            "excluded_trajectories_per_member": excluded_counts,
            "included_range": max(included_counts) - min(included_counts),
        }

    if set(mapping) != set(fit_trajectories):
        raise RuntimeError("BB3 mask mapping does not cover fit trajectories")
    if any(sum(row) != MEMBERS_PER_TRAJECTORY for row in mapping.values()):
        raise RuntimeError("BB3 mask is not exact-three-of-four")
    trajectory_assignments = sum(sum(row) for row in mapping.values())
    window_assignments = sum(
        sum(mapping[int(windows[index].trajectory)]) for index in indices)
    if trajectory_assignments != MEMBERS_PER_TRAJECTORY * len(mapping):
        raise RuntimeError("BB3 trajectory assignment total changed")
    if window_assignments != MEMBERS_PER_TRAJECTORY * len(indices):
        raise RuntimeError("BB3 optimizer-window assignment total changed")
    audit = {
        "scheme": "bb3_target_free_aoa_stratified_exact3_v1",
        "seed": seed,
        "held_aoa": held_aoa,
        "fit_aoas": sorted(grouped),
        "fit_trajectories": len(mapping),
        "fit_windows": len(indices),
        "members_per_trajectory": MEMBERS_PER_TRAJECTORY,
        "trajectory_assignments": trajectory_assignments,
        "window_assignments": window_assignments,
        "per_aoa": per_aoa,
        "mapping_sha256": smu1.c208.canonical_sha256({
            str(key): [int(value) for value in mapping[key]]
            for key in sorted(mapping)}),
    }
    return mapping, audit


def sample_stream_sha256(schedule: Mapping) -> str:
    return smu1.c208.canonical_sha256([{
        "sample_indices": row["sample_indices"],
        "trajectory_ids": row["trajectory_ids"],
    } for row in schedule["rows"]])


def raw_held_target_audit(dataset, indices: Sequence[int], *,
                          deadline: float = float("inf")) -> dict:
    """Hash held raw targets without exposing them to mask or optimization."""
    loader = DataLoader(
        dataset, batch_sampler=[
            list(indices[start:start + smu1.EVAL_BATCH_SIZE])
            for start in range(0, len(indices), smu1.EVAL_BATCH_SIZE)],
        num_workers=0, pin_memory=True)
    target_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    samples = 0
    scored_elements = 0
    for _history, target in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("BB3 wall budget expired during target audit")
        target = target.contiguous()
        scored = target[..., :2].ne(0.0).to(torch.uint8).contiguous()
        target_hasher.update(str(tuple(target.shape)).encode("ascii"))
        target_hasher.update(target.numpy().tobytes(order="C"))
        mask_hasher.update(str(tuple(scored.shape)).encode("ascii"))
        mask_hasher.update(scored.numpy().tobytes(order="C"))
        samples += len(target)
        scored_elements += int(scored.sum())
    if samples != len(indices) or scored_elements <= 0:
        raise RuntimeError("BB3 fresh held target audit is incomplete")
    return {
        "ordered_raw_target_sha256": target_hasher.hexdigest(),
        "ordered_scored_mask_sha256": mask_hasher.hexdigest(),
        "samples": samples,
        "scored_elements": scored_elements,
    }


def balanced_candidate_schedule(
        control_schedule: Mapping, fit_indices: Sequence[int],
        windows: Sequence, metadata: Sequence[dict], held_aoa: int,
        seed: int = SEED) -> tuple[dict, dict[int, tuple[bool, ...]]]:
    mapping, audit = balanced_exact3_masks(
        fit_indices, windows, metadata, held_aoa, seed)
    rows = []
    masks = []
    for control_row in control_schedule["rows"]:
        mask = torch.as_tensor([
            mapping[int(trajectory)]
            for trajectory in control_row["trajectory_ids"]
        ], dtype=torch.bool)
        if not bool((mask.sum(dim=1) == MEMBERS_PER_TRAJECTORY).all()):
            raise RuntimeError("candidate schedule contains a non-exact3 row")
        rows.append({
            "sample_indices": list(control_row["sample_indices"]),
            "trajectory_ids": list(control_row["trajectory_ids"]),
            "bootstrap_mask": mask.to(torch.uint8).tolist(),
        })
        masks.append(mask)
    schedule = {
        "batches": [list(batch) for batch in control_schedule["batches"]],
        "rows": rows,
        "masks": masks,
        "sha256": smu1.c208.canonical_sha256(rows),
        "sample_stream_sha256": smu1.c208.canonical_sha256([{
            "sample_indices": row["sample_indices"],
            "trajectory_ids": row["trajectory_ids"],
        } for row in rows]),
        "mask_audit": audit,
        "scheduled_assignments": sum(
            sum(sum(member_row) for member_row in row["bootstrap_mask"])
            for row in rows),
        "scheduled_samples": sum(len(row["sample_indices"]) for row in rows),
    }
    if schedule["sample_stream_sha256"] != sample_stream_sha256(
            control_schedule):
        raise RuntimeError("BB3 changed the paired sample/trajectory stream")
    if schedule["scheduled_assignments"] != (
            MEMBERS_PER_TRAJECTORY * schedule["scheduled_samples"]):
        raise RuntimeError("BB3 scheduled assignment total is not exact3")
    return schedule, mapping


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _validate_source_report(report: dict, fold_row: dict) -> None:
    if report.get("experiment") != smu1.EXPERIMENT:
        raise ValueError("control source is not a SMU1 report")
    if smu1.c208.canonical_sha256(report.get("contract")) != (
            smu1.c208.canonical_sha256(smu1.CONTRACT)):
        raise ValueError("control source SMU1 contract changed")
    provenance = report.get("provenance", {})
    expected_hashes = {
        "start_sim_sha256": smu1.FORMAL_START_SHA256,
        "real_stats_sha256": smu1.FORMAL_REAL_STATS_SHA256,
        "split_manifest_sha256": smu1.FORMAL_SPLIT_SHA256,
    }
    if any(provenance.get(key) != value
           for key, value in expected_hashes.items()):
        raise ValueError("control source provenance hashes changed")
    if provenance.get("git", {}).get("status_porcelain") != "":
        raise ValueError("control source was not run from a clean worktree")
    if provenance.get("git", {}).get("commit") not in (
            SMU1_CONTROL_COMPATIBLE_COMMITS):
        raise ValueError("control source commit is not BB3-compatible")
    if report.get("counts") != {
            "train_windows": 2576,
            "tail_windows_not_constructed": 725,
            "trajectories": 82,
            }:
        raise ValueError("control source split counts changed")
    if report.get("normalizer_contract") != {
            "kind": "other-four optimizer-window statistics per fold",
            "held_target_in_normalizer": False,
            }:
        raise ValueError("control source normalizer contract changed")
    if report.get("tail_accessed") is not False:
        raise ValueError("control source accessed the fixed725 tail")
    if report.get("release_checkpoint_written") is not False:
        raise ValueError("control source wrote a release checkpoint")
    if not any(row == fold_row for row in report.get("rows", [])):
        raise ValueError("control fold is not embedded in its source report")


def _validate_control_state(state: object, expected_samples: int) -> dict:
    if not isinstance(state, dict) or set(state) != set(
            smu1.new_metric_state()):
        raise ValueError("reused control_state schema changed")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(float(value)) for value in state.values()):
        raise ValueError("reused control_state is non-numeric or non-finite")
    for key in ("samples", "sps_scored", "sps_covered"):
        if int(state[key]) != state[key]:
            raise ValueError(f"reused control_state {key} is not an integer")
    if int(state["samples"]) != expected_samples:
        raise ValueError("reused control_state sample count changed")
    if int(state["sps_scored"]) <= 0:
        raise ValueError("reused control_state has no SPS-scored elements")
    if not (0 <= int(state["sps_covered"]) <= int(state["sps_scored"])):
        raise ValueError("reused control_state SPS coverage is invalid")
    if float(state["sps_width_sum"]) < 0.0:
        raise ValueError("reused control_state SPS width is negative")
    if any(float(state[key]) < 0.0 for key in (
            "rel_error_sum", "tke_error_sum", "mvpe_error_sum",
            "sps_numerator")):
        raise ValueError("reused control_state contains a negative sum")
    smu1.finalize_metric_state(state)
    return state


def load_reusable_control_fold(
        control_dir: Path, held_aoa: int, *,
        expected_statistics: tuple[torch.Tensor, ...],
        expected_stream_sha256: str, expected_held_key_sha256: str,
        expected_held_audit: Mapping,
        ) -> dict | None:
    """Return a validated SMU1 control; only an absent fold may retrain."""
    fold_path = control_dir / f"fold_{held_aoa:02d}.json"
    if not fold_path.exists():
        return None
    fold_row = _load_json(fold_path)
    expected_samples = EXPECTED_AOA_COUNTS[held_aoa]
    expected_fit = EXPECTED_FIT_COUNTS[held_aoa]
    expected_statistics_list = [
        value.detach().cpu().tolist() for value in expected_statistics]
    expected_statistics_sha = smu1.statistics_digest(expected_statistics)
    exact_fields = {
        "held_aoa": held_aoa,
        "fit_samples": expected_fit,
        "samples": expected_samples,
        "fold_statistics": expected_statistics_list,
        "fold_statistics_sha256": expected_statistics_sha,
        "stream_sha256": expected_stream_sha256,
        "held_ordered_key_sha256": expected_held_key_sha256,
    }
    if any(fold_row.get(key) != value for key, value in exact_fields.items()):
        raise ValueError("existing SMU1 control fold is incompatible with BB3")

    training = fold_row.get("control_training")
    if not isinstance(training, dict) or any((
            training.get("completed_updates") != UPDATES,
            training.get("planned_stream_sha256") != expected_stream_sha256,
            training.get("actual_stream_sha256") != expected_stream_sha256,
            )):
        raise ValueError("existing SMU1 control training stream is invalid")
    state = _validate_control_state(
        fold_row.get("control_state"), expected_samples)
    audit = fold_row.get("control_held_audit")
    if not isinstance(audit, dict) or any((
            audit.get("samples") != expected_samples,
            audit.get("scored_elements") != int(state["sps_scored"]),
            not _is_sha256(audit.get("ordered_raw_target_sha256")),
            not _is_sha256(audit.get("ordered_scored_mask_sha256")),
            )):
        raise ValueError("existing SMU1 control held audit is invalid")
    if audit != dict(expected_held_audit):
        raise ValueError(
            "existing SMU1 control target audit differs from fresh held data")
    if not _is_sha256(expected_held_key_sha256):
        raise ValueError("fresh held ordered-key digest is invalid")

    report_paths = [
        path for path in (
            control_dir / "summary.json", control_dir / "progress.json")
        if path.exists()]
    compatible_report = None
    errors = []
    for report_path in report_paths:
        try:
            _validate_source_report(_load_json(report_path), fold_row)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{report_path.name}: {exc}")
        else:
            compatible_report = report_path
            break
    if compatible_report is None:
        detail = "; ".join(errors) if errors else "no summary/progress report"
        raise ValueError(f"SMU1 control provenance is not reusable: {detail}")
    return {
        "training": training,
        "state": state,
        "metrics": smu1.finalize_metric_state(state),
        "held_audit": audit,
        "source": {
            "directory": str(control_dir.resolve()),
            "fold_file": fold_path.name,
            "fold_file_sha256": smu1.c208.sha256(fold_path),
            "report_file": compatible_report.name,
            "report_file_sha256": smu1.c208.sha256(compatible_report),
        },
    }


def update0_identity(control: dict, candidate: dict) -> dict:
    fields = ("prediction", "members", "lower", "upper")
    equal = {f"{field}_bitwise": bool(torch.equal(
        control[field], candidate[field])) for field in fields}
    equal["metric_state_exact"] = (
        control["metric_state"] == candidate["metric_state"])
    equal["pass"] = all(equal.values())
    return equal


def loao_gate(pooled: dict, rows: Sequence[dict]) -> dict:
    if len(rows) != len(AOA_VALUES):
        raise ValueError("formal gate requires exactly five held-AoA rows")
    delta = pooled["delta"]
    conditions = {
        "pooled_point_delta": delta["point_score"] >= smu1.POINT_DELTA_MIN,
        "pooled_direct_delta": pooled["direct_delta"] >= smu1.DIRECT_DELTA_MIN,
        "pooled_sps_delta": (
            delta["native_sps_score"] >= smu1.SPS_DELTA_MIN),
        "pooled_non_time_delta": (
            pooled["non_time_delta"] >= smu1.NON_TIME_DELTA_MIN),
        "rel_floor": delta["rel_l2_score"] >= smu1.COMPONENT_FLOOR,
        "tke_floor": delta["tke_rel_l2_score"] >= smu1.COMPONENT_FLOOR,
        "mvpe_floor": delta["mvpe_rel_l2_score"] >= smu1.COMPONENT_FLOOR,
        "positive_point_folds": sum(
            row["comparison"]["delta"]["point_score"] > 0.0 for row in rows)
            >= smu1.MIN_POSITIVE_FOLDS,
        "positive_non_time_folds": sum(
            row["comparison"]["non_time_delta"] > 0.0 for row in rows)
            >= smu1.MIN_POSITIVE_FOLDS,
        "worst_fold_non_time": min(
            row["comparison"]["non_time_delta"] for row in rows)
            >= smu1.WORST_FOLD_NON_TIME,
        "paired_sample_streams": all(
            row["shared_sample_stream_sha256"]
            == row["control_sample_stream_sha256"]
            == row["candidate_sample_stream_sha256"] for row in rows),
        "arm_streams_verified": all(
            row["control_training"]["actual_stream_sha256"]
            == row["control_stream_sha256"]
            and row["candidate_training"]["actual_stream_sha256"]
            == row["candidate_stream_sha256"] for row in rows),
        "candidate_masks_exact3_balanced": all(
            row["candidate_mask_audit"]["scheme"]
            == "bb3_target_free_aoa_stratified_exact3_v1"
            and row["candidate_mask_audit"]["held_aoa"] == row["held_aoa"]
            and set(row["candidate_mask_audit"]["fit_aoas"])
            == set(AOA_VALUES) - {row["held_aoa"]}
            and row["candidate_mask_audit"]["members_per_trajectory"]
            == MEMBERS_PER_TRAJECTORY
            and row["candidate_mask_audit"]["fit_trajectories"]
            == 82 - EXPECTED_TRAJECTORY_COUNTS[row["held_aoa"]]
            and row["candidate_mask_audit"]["fit_windows"]
            == row["fit_samples"]
            and row["candidate_mask_audit"]["trajectory_assignments"]
            == MEMBERS_PER_TRAJECTORY * (
                82 - EXPECTED_TRAJECTORY_COUNTS[row["held_aoa"]])
            and row["candidate_mask_audit"]["window_assignments"]
            == MEMBERS_PER_TRAJECTORY * row["fit_samples"]
            and _is_sha256(row["candidate_mask_audit"]["mapping_sha256"])
            and set(row["candidate_mask_audit"]["per_aoa"])
            == {str(value) for value in set(AOA_VALUES) - {row["held_aoa"]}}
            and all(value["included_range"] <= 1 for value in
                    row["candidate_mask_audit"]["per_aoa"].values())
            for row in rows),
        "samples_exact": sum(int(row["samples"]) for row in rows) == 2576,
        "held_aoas_and_counts_exact": (
            {int(row["held_aoa"]) for row in rows} == set(AOA_VALUES)
            and all(int(row["samples"]) == EXPECTED_AOA_COUNTS[
                int(row["held_aoa"])] for row in rows)),
        "held_targets_and_masks_paired": all(
            row["control_held_audit"] == row["candidate_held_audit"]
            and row["control_held_audit"]["samples"] == row["samples"]
            and bool(row["held_ordered_key_sha256"])
            for row in rows),
    }
    passed = all(conditions.values())
    return {
        "pass": passed,
        "decision": "GO_BB3_FULL2576" if passed else "STOP_BB3_LOAO",
        "conditions": conditions,
    }


def run(args: argparse.Namespace) -> dict:
    require_formal_gpu_opt_in(args)
    identity = smu1.git_identity()
    if identity["status_porcelain"]:
        raise RuntimeError("formal BB3 requires a clean Git worktree")
    started = time.perf_counter()
    deadline = started + smu1.WALL_BUDGET_SECONDS
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal BB3 paired LOAO requires CUDA")
    if smu1.c208.sha256(args.real_stats) != smu1.FORMAL_REAL_STATS_SHA256:
        raise RuntimeError("official Real stats SHA changed")
    payload = smu1.c208.load_start_payload(args)
    trajectories, metadata = smu1.ensemble_train.common.repro.load_trajectories(
        args.real_dir, 0)
    train_windows, manifest, tail_count = smu1.make_train_side_split(metadata)
    manifest_sha = smu1.c208.canonical_sha256(manifest)
    if manifest_sha != smu1.FORMAL_SPLIT_SHA256:
        raise RuntimeError("fixed Real split manifest SHA changed")
    if len(train_windows) != 2576 or tail_count != 725 or len(trajectories) != 82:
        raise RuntimeError("fixed Real split counts changed")
    aoas = smu1.window_aoas(train_windows, metadata)
    if {value: int((aoas == value).sum()) for value in AOA_VALUES} != (
            EXPECTED_AOA_COUNTS):
        raise RuntimeError("fixed2576 AoA counts changed")
    partitions = smu1.loao_partitions(aoas)
    dataset = smu1.ensemble_train.common.repro.WindowDataset(
        trajectories, train_windows)
    statistics_rows = smu1.window_sufficient_statistics(
        trajectories, train_windows)
    smu1.write_json_atomic(
        args.output_dir / "train_split_manifest.json", manifest)
    base_report = {
        "experiment": EXPERIMENT,
        "contract": CONTRACT,
        "provenance": {
            "start_sim": str(args.start_sim.resolve()),
            "start_sim_sha256": args.start_sim_sha256,
            "real_stats_sha256": smu1.FORMAL_REAL_STATS_SHA256,
            "split_manifest_sha256": manifest_sha,
            "git": identity,
            "reuse_smu1_control_dir": (
                None if args.reuse_smu1_control_dir is None
                else str(args.reuse_smu1_control_dir.resolve())),
        },
        "counts": {
            "train_windows": len(train_windows),
            "tail_windows_not_constructed": tail_count,
            "trajectories": len(trajectories),
        },
        "normalizer_contract": {
            "kind": "other-four optimizer-window statistics per fold",
            "held_target_in_normalizer": False,
        },
        "raw_h5_frame_residency": True,
        "tail_accessed": False,
        "release_checkpoint_written": False,
    }
    gpu_seconds = 0.0
    rows = []
    update0 = None

    for held in FOLD_ORDER:
        partition = partitions[held]
        stats = smu1.fold_statistics(statistics_rows, partition["fit"])
        normalizer = smu1.FoldNormalizer(stats, device)
        control_schedule = smu1.paired_batch_schedule(
            partition["fit"], train_windows)
        candidate_schedule, candidate_mapping = balanced_candidate_schedule(
            control_schedule, partition["fit"], train_windows, metadata, held)
        shared_sample_stream = sample_stream_sha256(control_schedule)
        held_key = smu1.ordered_held_key_digest(
            partition["evaluate"], train_windows, metadata)
        fresh_held_audit = raw_held_target_audit(
            dataset, partition["evaluate"], deadline=deadline)
        reusable = (
            None if args.reuse_smu1_control_dir is None else
            load_reusable_control_fold(
                args.reuse_smu1_control_dir, held,
                expected_statistics=stats,
                expected_stream_sha256=control_schedule["sha256"],
                expected_held_key_sha256=held_key,
                expected_held_audit=fresh_held_audit))

        fold_started = time.perf_counter()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
        control_initial = None
        if reusable is None:
            smu1.ensemble_train.seed_everything(SEED)
            control_model = smu1.load_arm_model(
                payload, False, normalizer, device)
            if update0 is None:
                control_initial = smu1.capture_update0(
                    control_model, dataset, partition["fit"][0],
                    normalizer, device)
            control_training = smu1.train_arm(
                control_model, dataset, control_schedule, normalizer, device,
                deadline, f"held{held}_control")
            control_metrics, control_state, control_audit = (
                smu1.evaluate_model(
                    control_model, dataset, partition["evaluate"], normalizer,
                    device, deadline))
            if control_audit != fresh_held_audit:
                raise RuntimeError(
                    "retrained control target audit differs from fresh held data")
            control_source = {"kind": "retrained_missing_control_fold"}
            del control_model
            torch.cuda.empty_cache()
        else:
            control_training = reusable["training"]
            control_metrics = reusable["metrics"]
            control_state = reusable["state"]
            control_audit = reusable["held_audit"]
            control_source = {
                "kind": "strictly_reused_smu1_control_state",
                **reusable["source"],
            }

        if update0 is None and control_initial is None:
            smu1.ensemble_train.seed_everything(SEED)
            reference_model = smu1.load_arm_model(
                payload, False, normalizer, device)
            control_initial = smu1.capture_update0(
                reference_model, dataset, partition["fit"][0],
                normalizer, device)
            del reference_model
            torch.cuda.empty_cache()
        smu1.ensemble_train.seed_everything(SEED)
        candidate_model = smu1.load_arm_model(
            payload, False, normalizer, device)
        if update0 is None:
            candidate_initial = smu1.capture_update0(
                candidate_model, dataset, partition["fit"][0], normalizer,
                device)
            update0 = update0_identity(control_initial, candidate_initial)
            if not update0["pass"]:
                raise RuntimeError("BB3 update-0 model identity failed")
        candidate_training = smu1.train_arm(
            candidate_model, dataset, candidate_schedule, normalizer, device,
            deadline, f"held{held}_candidate",
            bootstrap_mask_by_trajectory=candidate_mapping)
        candidate_metrics, candidate_state, candidate_audit = (
            smu1.evaluate_model(
                candidate_model, dataset, partition["evaluate"], normalizer,
                device, deadline))
        if candidate_audit != fresh_held_audit:
            raise RuntimeError(
                "candidate target audit differs from fresh held data")
        gpu_end.record()
        gpu_end.synchronize()
        fold_gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000.0
        del candidate_model
        torch.cuda.empty_cache()

        row = {
            "held_aoa": held,
            "fit_samples": len(partition["fit"]),
            "samples": len(partition["evaluate"]),
            "fold_statistics": [value.tolist() for value in stats],
            "fold_statistics_sha256": smu1.statistics_digest(stats),
            "control_stream_sha256": control_schedule["sha256"],
            "candidate_stream_sha256": candidate_schedule["sha256"],
            "shared_sample_stream_sha256": shared_sample_stream,
            "control_sample_stream_sha256": sample_stream_sha256(
                control_schedule),
            "candidate_sample_stream_sha256": candidate_schedule[
                "sample_stream_sha256"],
            "candidate_mask_audit": candidate_schedule["mask_audit"],
            "held_ordered_key_sha256": held_key,
            "control_source": control_source,
            "control_training": control_training,
            "candidate_training": candidate_training,
            "control_held_audit": control_audit,
            "candidate_held_audit": candidate_audit,
            "control_state": control_state,
            "candidate_state": candidate_state,
            "comparison": smu1.comparison(
                candidate_metrics, control_metrics),
            "wall_seconds": time.perf_counter() - fold_started,
            "cuda_elapsed_seconds": fold_gpu_seconds,
        }
        rows.append(row)
        gpu_seconds += fold_gpu_seconds
        smu1.write_json_atomic(
            args.output_dir / f"fold_{held:02d}.json", row)
        smu1.write_json_atomic(args.output_dir / "progress.json", {
            **base_report,
            "decision": "BB3_LOAO_IN_PROGRESS",
            "completed_folds": len(rows),
            "rows": rows,
            "update0": update0,
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_seconds": gpu_seconds,
        })
        print(json.dumps({
            "event": "bb3_fold_complete",
            "held_aoa": held,
            "control_source": control_source["kind"],
            "point_delta": row["comparison"]["delta"]["point_score"],
            "sps_delta": row["comparison"]["delta"]["native_sps_score"],
            "non_time_delta": row["comparison"]["non_time_delta"],
        }), flush=True)
        if gpu_seconds > smu1.GPU_BUDGET_SECONDS:
            raise TimeoutError("BB3 GPU budget exceeded")
        if time.perf_counter() > deadline:
            raise TimeoutError("BB3 wall budget exceeded")
        if sum(row["comparison"]["non_time_delta"] <= 0.0
               for row in rows) >= 2:
            return {
                **base_report,
                "decision": "STOP_BB3_TWO_NONPOSITIVE_AOA",
                "rows": rows,
                "update0": update0,
                "tail_accessed": False,
                "release_checkpoint_written": False,
                "elapsed_seconds": time.perf_counter() - started,
                "gpu_seconds": gpu_seconds,
            }

    control_pooled_state = smu1.merge_metric_states([
        row["control_state"] for row in rows])
    candidate_pooled_state = smu1.merge_metric_states([
        row["candidate_state"] for row in rows])
    if int(control_pooled_state["samples"]) != 2576 or int(
            candidate_pooled_state["samples"]) != 2576:
        raise RuntimeError("pooled held states do not cover fixed2576")
    pooled = smu1.comparison(
        smu1.finalize_metric_state(candidate_pooled_state),
        smu1.finalize_metric_state(control_pooled_state))
    gate = loao_gate(pooled, rows)
    return {
        **base_report,
        "update0": update0,
        "rows": rows,
        "pooled_control_state": control_pooled_state,
        "pooled_candidate_state": candidate_pooled_state,
        "pooled": pooled,
        "gate": gate,
        "decision": gate["decision"],
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_seconds": gpu_seconds,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True)
    if args.contract_only:
        summary = {
            "experiment": EXPERIMENT,
            "contract": CONTRACT,
            "decision": "CONTRACT_ONLY_NO_DATA_NO_GPU",
            "tail_accessed": False,
            "release_checkpoint_written": False,
        }
    else:
        smu1.ensemble_train.seed_everything(SEED)
        torch.set_float32_matmul_precision("high")
        try:
            summary = run(args)
        except Exception as exc:
            summary = {
                "experiment": EXPERIMENT,
                "contract": CONTRACT,
                "attempted_provenance": {
                    "start_sim": str(args.start_sim.resolve()),
                    "start_sim_sha256": args.start_sim_sha256,
                    "git": smu1.git_identity(),
                },
                "decision": "STOP_BB3_CONTRACT_OR_RUNTIME",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "tail_accessed": False,
                "release_checkpoint_written": False,
            }
            smu1.write_json_atomic(args.output_dir / "summary.json", summary)
            raise
    smu1.write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "event": "bb3_complete", "decision": summary["decision"]}),
        flush=True)


if __name__ == "__main__":
    main()
