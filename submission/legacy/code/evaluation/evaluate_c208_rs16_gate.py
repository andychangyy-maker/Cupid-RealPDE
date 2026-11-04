#!/usr/bin/env python3
"""Evaluate the preregistered C208 Real stride-16 update-4000 gate.

Both checkpoints are scored on the original frozen stride-20 fixed725 tail.
This script does not train, save a checkpoint, build a ZIP, or select a model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for path in (CODE_ROOT, TRAINING_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_flow_backbone_blend as blend  # noqa: E402
import train_c208_real_sim_rehearsal as rehearsal  # noqa: E402


EXPERIMENT = "c208_rs16_real_phase_coverage_gate"
BASELINE_CHECKPOINT_SHA256 = (
    "79b3f68272580f80f1d9e09a54e9f9f5291e74524c8b11cdfd78b77a1566cbc3")
BASELINE_SPLIT_MANIFEST_SHA256 = (
    "403278adf2012baeb1678dc2859702a6ac14b90422cb0ca0f878b705ef27bf04")
REAL_STATS_SHA256 = rehearsal.FORMAL_REAL_STATS_SHA256
TERMINAL_UPDATE = 4_000
BASELINE_STRIDE = 20
CANDIDATE_STRIDE = 16
WINDOW_LENGTH = 40
EXPECTED_COUNTS = {
    "baseline_train": 2576,
    "candidate_train": 3140,
    "validation": 725,
    "trajectories": 82,
}
AOA_VALUES = (0, 5, 10, 15, 20)
EXPECTED_CANDIDATE_TRAIN_AOA_COUNTS = {
    0: 651, 5: 528, 10: 702, 15: 663, 20: 596}
EXPECTED_VALIDATION_AOA_COUNTS = {
    0: 150, 5: 122, 10: 162, 15: 153, 20: 138}
BASELINE_METRICS = {
    "rel_l2_score": 95.86352389426571,
    "tke_score": 80.95221179698406,
    "mvpe_score": 96.51432542559023,
    "point_score": 91.52029057134612,
    "sps_score": 53.07979215521265,
}
REPLAY_TOLERANCE = 1e-5
POINT_SHADOW_WEIGHTS = {
    "rel_l2_score": 0.47201126,
    "tke_score": 0.08586562,
    "mvpe_score": 0.09410214,
}
ALLOWED_TRAINING_CONFIG_DIFFERENCES = {
    "output_dir", "window_stride", "baseline_split_manifest"}
METRIC_KEYS = (
    "rel_l2_score", "tke_score", "mvpe_score", "point_score", "sps_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-split-manifest", type=Path, required=True)
    parser.add_argument("--candidate-split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--real-stats", type=Path,
        default=Path(
            "kits/track2/realpde_t2_starting_kit_v6/"
            "example_data/mean_std_real.pt"))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_identity() -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "porcelain": porcelain}


def _require_mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_one_checkpoint(payload: dict, name: str) -> tuple[dict, dict]:
    payload = _require_mapping(payload, f"{name} checkpoint")
    if payload.get("phase") != "real":
        raise ValueError(f"{name} checkpoint phase is not real")
    if payload.get("iteration") != TERMINAL_UPDATE:
        raise ValueError(f"{name} checkpoint is not update 4000")
    model_config = _require_mapping(
        payload.get("model_config"), f"{name} model_config")
    if model_config.get("architecture") != "ensemble_unet":
        raise ValueError(f"{name} architecture is not ensemble_unet")
    for key, expected in rehearsal.EXPECTED_MODEL_CONFIG.items():
        if model_config.get(key) != expected:
            raise ValueError(
                f"{name} model_config[{key!r}]={model_config.get(key)!r}, "
                f"expected {expected!r}")
    training_config = _require_mapping(
        payload.get("training_config"), f"{name} training_config")
    return model_config, training_config


def validate_checkpoint_contracts(
        baseline: dict, candidate: dict, baseline_manifest_path: Path) -> dict:
    """Require one-variable stride20-to-stride16 checkpoint provenance."""
    baseline_model, baseline_training = _validate_one_checkpoint(
        baseline, "baseline")
    candidate_model, candidate_training = _validate_one_checkpoint(
        candidate, "candidate")
    if candidate_model != baseline_model:
        raise ValueError("candidate model_config differs from baseline")
    if baseline_training.get("window_stride") != BASELINE_STRIDE:
        raise ValueError("baseline training_config window_stride is not 20")
    if candidate_training.get("window_stride") != CANDIDATE_STRIDE:
        raise ValueError("candidate training_config window_stride is not 16")
    if not isinstance(baseline_training.get("output_dir"), str):
        raise ValueError("baseline output_dir provenance is missing")
    if not isinstance(candidate_training.get("output_dir"), str):
        raise ValueError("candidate output_dir provenance is missing")
    recorded_manifest = candidate_training.get("baseline_split_manifest")
    if not isinstance(recorded_manifest, str) or not recorded_manifest:
        raise ValueError("candidate does not record baseline_split_manifest")
    if Path(recorded_manifest).resolve() != baseline_manifest_path.resolve():
        raise ValueError("candidate baseline_split_manifest path differs")
    if baseline_training.get("baseline_split_manifest") not in (None, ""):
        raise ValueError("baseline unexpectedly records the RS16 manifest option")

    def frozen(config: dict) -> dict:
        return {
            key: value for key, value in config.items()
            if key not in ALLOWED_TRAINING_CONFIG_DIFFERENCES}

    baseline_frozen = frozen(baseline_training)
    candidate_frozen = frozen(candidate_training)
    if baseline_frozen != candidate_frozen:
        baseline_keys = set(baseline_frozen)
        candidate_keys = set(candidate_frozen)
        changed = sorted(
            key for key in baseline_keys & candidate_keys
            if baseline_frozen[key] != candidate_frozen[key])
        raise ValueError(
            "training_config differs outside the RS16 variable: "
            f"missing={sorted(baseline_keys-candidate_keys)}, "
            f"extra={sorted(candidate_keys-baseline_keys)}, changed={changed}")
    return {
        "iteration": TERMINAL_UPDATE,
        "architecture": baseline_model["architecture"],
        "model_config_sha256": canonical_sha256(baseline_model),
        "frozen_training_config_sha256": canonical_sha256(baseline_frozen),
        "allowed_differences": {
            "baseline": {
                key: baseline_training.get(key)
                for key in sorted(ALLOWED_TRAINING_CONFIG_DIFFERENCES)},
            "candidate": {
                key: candidate_training.get(key)
                for key in sorted(ALLOWED_TRAINING_CONFIG_DIFFERENCES)},
        },
    }


def _integer_starts(row: dict, key: str) -> list[int]:
    starts = row.get(key)
    if (not isinstance(starts, list)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in starts)):
        raise ValueError(f"manifest {key} must be a list of integers")
    if starts != sorted(set(starts)):
        raise ValueError(f"manifest {key} must be sorted and unique")
    return starts


def _key_sha(rows: list[tuple[str, int]]) -> str:
    return canonical_sha256([[name, start] for name, start in rows])


def validate_split_manifests(baseline: list, candidate: list) -> dict:
    """Validate exact fixed725 reuse and the isolated stride-16 train set."""
    if (not isinstance(baseline, list) or not isinstance(candidate, list)
            or len(baseline) != EXPECTED_COUNTS["trajectories"]
            or len(candidate) != len(baseline)):
        raise ValueError("split manifests must contain the same 82 trajectories")
    baseline_train_keys = []
    candidate_train_keys = []
    baseline_val_keys = []
    candidate_val_keys = []
    candidate_train_aoa = {value: 0 for value in AOA_VALUES}
    validation_aoa = {value: 0 for value in AOA_VALUES}
    interval_overlaps = 0
    metadata_keys = ("file", "frames", "re", "aoa")

    for trajectory, (base_row, candidate_row) in enumerate(
            zip(baseline, candidate)):
        base_row = _require_mapping(base_row, f"baseline manifest row {trajectory}")
        candidate_row = _require_mapping(
            candidate_row, f"candidate manifest row {trajectory}")
        if set(candidate_row) != set(base_row):
            raise ValueError(f"candidate row {trajectory} schema differs")
        for key in metadata_keys:
            if candidate_row.get(key) != base_row.get(key):
                raise ValueError(
                    f"candidate row {trajectory} metadata {key} differs")
        name = base_row.get("file")
        frames = base_row.get("frames")
        aoa = base_row.get("aoa")
        boundary = base_row.get("val_boundary")
        if (not isinstance(name, str) or not name
                or isinstance(frames, bool) or not isinstance(frames, int)
                or isinstance(boundary, bool) or not isinstance(boundary, int)
                or aoa not in AOA_VALUES):
            raise ValueError(f"baseline row {trajectory} metadata is invalid")
        if candidate_row.get("val_boundary") != boundary:
            raise ValueError(f"candidate row {trajectory} val_boundary differs")
        base_train = _integer_starts(base_row, "train_starts")
        base_val = _integer_starts(base_row, "val_starts")
        candidate_train = _integer_starts(candidate_row, "train_starts")
        candidate_val = _integer_starts(candidate_row, "val_starts")
        if candidate_val != base_val:
            raise ValueError(f"candidate row {trajectory} validation keys differ")
        expected_train = [
            start for start in range(0, frames - WINDOW_LENGTH + 1,
                                     CANDIDATE_STRIDE)
            if start + WINDOW_LENGTH <= boundary]
        if candidate_train != expected_train:
            raise ValueError(
                f"candidate row {trajectory} is not the exact stride16 train set")
        if any(start < 0 or start + WINDOW_LENGTH > frames
               for start in (*base_train, *base_val, *candidate_train)):
            raise ValueError(f"manifest row {trajectory} has an invalid window")
        baseline_train_keys.extend((name, start) for start in base_train)
        candidate_train_keys.extend((name, start) for start in candidate_train)
        baseline_val_keys.extend((name, start) for start in base_val)
        candidate_val_keys.extend((name, start) for start in candidate_val)
        candidate_train_aoa[aoa] += len(candidate_train)
        validation_aoa[aoa] += len(base_val)
        interval_overlaps += sum(
            train_start < val_start + WINDOW_LENGTH
            and val_start < train_start + WINDOW_LENGTH
            for train_start in candidate_train for val_start in base_val)

    exact_intersection = len(
        set(candidate_train_keys).intersection(baseline_val_keys))
    checks = {
        "baseline_train_windows_2576": (
            len(baseline_train_keys) == EXPECTED_COUNTS["baseline_train"]),
        "candidate_train_windows_3140": (
            len(candidate_train_keys) == EXPECTED_COUNTS["candidate_train"]),
        "validation_windows_725": (
            len(baseline_val_keys) == EXPECTED_COUNTS["validation"]),
        "candidate_validation_exact_baseline": (
            candidate_val_keys == baseline_val_keys),
        "candidate_train_interval_overlap_zero": interval_overlaps == 0,
        "candidate_train_validation_key_intersection_zero": (
            exact_intersection == 0),
        "candidate_train_aoa_counts_frozen": (
            candidate_train_aoa == EXPECTED_CANDIDATE_TRAIN_AOA_COUNTS),
        "validation_aoa_counts_frozen": (
            validation_aoa == EXPECTED_VALIDATION_AOA_COUNTS),
    }
    if not all(checks.values()):
        raise ValueError(f"RS16 split isolation failed: {checks}")
    return {
        "checks": checks,
        "counts": {
            "baseline_train": len(baseline_train_keys),
            "candidate_train": len(candidate_train_keys),
            "validation": len(baseline_val_keys),
            "trajectories": len(baseline),
            "interval_overlaps": interval_overlaps,
            "exact_key_intersections": exact_intersection,
        },
        "candidate_train_aoa_counts": candidate_train_aoa,
        "validation_aoa_counts": validation_aoa,
        "ordered_key_sha256": {
            "candidate_train": _key_sha(candidate_train_keys),
            "validation": _key_sha(baseline_val_keys),
        },
    }


def metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in METRIC_KEYS}


def point_shadow(delta: dict[str, float]) -> float:
    return float(sum(
        POINT_SHADOW_WEIGHTS[key] * delta[key]
        for key in POINT_SHADOW_WEIGHTS))


def build_gate_checks(
        pooled_delta: dict[str, float], aoa_rows: list[dict],
        hard_guards: dict[str, bool]) -> dict[str, bool]:
    shadows = [float(row["point_shadow_delta"]) for row in aoa_rows]
    if len(shadows) != len(AOA_VALUES):
        raise ValueError("RS16 gate requires exactly five AoA rows")
    checks = dict(hard_guards)
    checks.update({
        "pooled_point_delta_ge_0_10": pooled_delta["point_score"] >= 0.10,
        "pooled_point_shadow_ge_0_06": point_shadow(pooled_delta) >= 0.06,
        "pooled_rel_delta_ge_minus_0_02": (
            pooled_delta["rel_l2_score"] >= -0.02),
        "pooled_tke_delta_ge_minus_0_02": (
            pooled_delta["tke_score"] >= -0.02),
        "pooled_mvpe_delta_ge_minus_0_02": (
            pooled_delta["mvpe_score"] >= -0.02),
        "pooled_native_sps_delta_ge_minus_0_10": (
            pooled_delta["sps_score"] >= -0.10),
        "positive_aoa_shadow_ge_4_of_5": (
            sum(value > 0.0 for value in shadows) >= 4),
        "worst_aoa_shadow_ge_minus_0_02": min(shadows) >= -0.02,
    })
    return checks


def baseline_replay(metrics: dict) -> dict:
    rows = {}
    for key, expected in BASELINE_METRICS.items():
        actual = float(metrics[key])
        rows[key] = {
            "actual": actual,
            "expected": expected,
            "absolute_error": abs(actual - expected),
            "pass": abs(actual - expected) <= REPLAY_TOLERANCE,
        }
    return {"tolerance": REPLAY_TOLERANCE, "rows": rows,
            "pass": all(row["pass"] for row in rows.values())}


def _load_json_list(path: Path, name: str) -> list:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{name} must contain a JSON list")
    return value


def _normalizer_matches(payload: dict, normalizer) -> bool:
    stored = payload.get("normalizer")
    if not isinstance(stored, dict):
        return False
    for key in rehearsal.NORMALIZER_KEYS:
        if key not in stored:
            return False
        expected = getattr(normalizer, key).detach().cpu().numpy()
        actual = np.asarray(stored[key], dtype=np.float32)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            return False
    return True


def _arrays_valid(prediction, lower, upper, target) -> bool:
    arrays = (prediction, lower, upper, target)
    return bool(
        all(array.dtype == np.float32 and np.all(np.isfinite(array))
            for array in arrays)
        and np.all(lower <= upper)
        and np.count_nonzero(prediction[..., 2]) == 0
        and np.count_nonzero(lower[..., 2]) == 0
        and np.count_nonzero(upper[..., 2]) == 0
        and np.count_nonzero(target[..., 2]) == 0)


def _load_model(payload: dict, normalizer, device: torch.device):
    model = rehearsal.load_model(payload, device)
    rehearsal.ensemble_train.common.set_model_stats(model, normalizer)
    return model.eval()


def _collect_timed(model, loader, normalizer, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    arrays = blend.collect(model, loader, normalizer, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return arrays, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.eval_batch_size != 16 or args.num_workers != 0:
        raise ValueError("formal RS16 evaluation requires batch16/workers0")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal RS16 evaluation requires CUDA")

    identities = {
        "baseline_checkpoint": file_identity(args.baseline_checkpoint),
        "candidate_checkpoint": file_identity(args.candidate_checkpoint),
        "baseline_split_manifest": file_identity(args.baseline_split_manifest),
        "candidate_split_manifest": file_identity(args.candidate_split_manifest),
        "real_stats": file_identity(args.real_stats),
    }
    if identities["baseline_checkpoint"]["sha256"] != BASELINE_CHECKPOINT_SHA256:
        raise ValueError("baseline checkpoint SHA differs from preregistration")
    if (identities["baseline_split_manifest"]["sha256"]
            != BASELINE_SPLIT_MANIFEST_SHA256):
        raise ValueError("baseline split manifest SHA differs from preregistration")
    if identities["real_stats"]["sha256"] != REAL_STATS_SHA256:
        raise ValueError("Real stats SHA differs from preregistration")

    baseline_manifest = _load_json_list(
        args.baseline_split_manifest, "baseline split manifest")
    candidate_manifest = _load_json_list(
        args.candidate_split_manifest, "candidate split manifest")
    trajectories, metadata = rehearsal.ensemble_train.common.repro.load_trajectories(
        args.real_dir, 0)
    _runtime_train, _runtime_val, runtime_manifest = (
        rehearsal.ensemble_train.common.repro.make_split(
            metadata, BASELINE_STRIDE, 0.2))
    if runtime_manifest != baseline_manifest:
        raise RuntimeError("runtime Real data does not reproduce frozen manifest")
    split_audit = validate_split_manifests(
        baseline_manifest, candidate_manifest)

    baseline_payload = torch.load(
        args.baseline_checkpoint, map_location="cpu", weights_only=False)
    candidate_payload = torch.load(
        args.candidate_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_contract = validate_checkpoint_contracts(
        baseline_payload, candidate_payload, args.baseline_split_manifest)
    normalizer = rehearsal.ensemble_train.common.repro.Normalizer(
        args.real_stats, device)
    normalizer_guard = bool(
        _normalizer_matches(baseline_payload, normalizer)
        and _normalizer_matches(candidate_payload, normalizer))
    if not normalizer_guard:
        raise RuntimeError("checkpoint normalizer differs from frozen Real stats")

    validation_windows = [
        rehearsal.ensemble_train.common.repro.Window(trajectory, start)
        for trajectory, row in enumerate(baseline_manifest)
        for start in row["val_starts"]]
    aoas = np.asarray([
        int(metadata[window.trajectory]["aoa"])
        for window in validation_windows], dtype=np.int64)
    dataset = rehearsal.ensemble_train.common.repro.WindowDataset(
        trajectories, validation_windows)
    loader = DataLoader(
        dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    rehearsal.ensemble_train.seed_everything(0)
    torch.set_float32_matmul_precision("high")

    baseline_model = _load_model(baseline_payload, normalizer, device)
    baseline_arrays, baseline_seconds = _collect_timed(
        baseline_model, loader, normalizer, device)
    del baseline_model, baseline_payload
    gc.collect()
    torch.cuda.empty_cache()
    baseline_metrics = blend.score(*baseline_arrays)
    replay = baseline_replay(baseline_metrics)
    if not replay["pass"]:
        raise RuntimeError("baseline fixed725 metrics failed replay tolerance")

    candidate_model = _load_model(candidate_payload, normalizer, device)
    candidate_arrays, candidate_seconds = _collect_timed(
        candidate_model, loader, normalizer, device)
    del candidate_model, candidate_payload
    gc.collect()
    torch.cuda.empty_cache()
    if not np.array_equal(baseline_arrays[3], candidate_arrays[3]):
        raise RuntimeError("target order changed between checkpoint evaluations")
    arrays_guard = bool(
        _arrays_valid(*baseline_arrays) and _arrays_valid(*candidate_arrays))
    if not arrays_guard:
        raise RuntimeError("prediction/bounds/target invariants failed")

    candidate_metrics = blend.score(*candidate_arrays)
    pooled_delta = metric_delta(candidate_metrics, baseline_metrics)
    aoa_rows = []
    for aoa in AOA_VALUES:
        mask = aoas == aoa
        before = blend.score(*(array[mask] for array in baseline_arrays))
        after = blend.score(*(array[mask] for array in candidate_arrays))
        delta = metric_delta(after, before)
        aoa_rows.append({
            "aoa": aoa,
            "samples": int(mask.sum()),
            "baseline": before,
            "candidate": after,
            "delta": delta,
            "point_shadow_delta": point_shadow(delta),
        })
    hard_guards = {
        "baseline_replay_pass": bool(replay["pass"]),
        "checkpoint_contract_pass": True,
        "split_isolation_pass": all(split_audit["checks"].values()),
        "normalizer_matches_real_stats": normalizer_guard,
        "target_order_exact": True,
        "finite_pressure_zero_bounds_ordered": arrays_guard,
        "all_five_aoas_present": (
            set(aoas.tolist()) == set(AOA_VALUES)),
    }
    checks = build_gate_checks(pooled_delta, aoa_rows, hard_guards)
    decision = (
        "GO_RS16_TO_OOF_UQ" if all(checks.values())
        else "STOP_RS16_NO_GENERALIZABLE_POINT_SIGNAL")
    elapsed = time.perf_counter() - started
    report = {
        "experiment": EXPERIMENT,
        "evidence_class": "fixed725 local diagnostic; not official evidence",
        "git": git_identity(),
        "identities": identities,
        "checkpoint_contract": checkpoint_contract,
        "split_audit": split_audit,
        "validation_order_sha256": canonical_sha256([
            [baseline_manifest[window.trajectory]["file"], window.start]
            for window in validation_windows]),
        "target_sha256": hashlib.sha256(
            memoryview(np.ascontiguousarray(baseline_arrays[3])).cast("B")
        ).hexdigest(),
        "baseline": baseline_metrics,
        "baseline_replay": replay,
        "candidate": candidate_metrics,
        "pooled_delta": pooled_delta,
        "pooled_point_shadow_delta": point_shadow(pooled_delta),
        "aoa_rows": aoa_rows,
        "gate_contract": {
            "point_delta_min": 0.10,
            "point_shadow_min": 0.06,
            "component_delta_floor": -0.02,
            "native_sps_delta_floor": -0.10,
            "positive_aoas_min": 4,
            "worst_aoa_shadow_floor": -0.02,
            "point_shadow_weights": POINT_SHADOW_WEIGHTS,
        },
        "gate_checks": checks,
        "decision": decision,
        "elapsed_seconds": elapsed,
        "gpu_inference_seconds": {
            "baseline": baseline_seconds,
            "candidate": candidate_seconds,
            "total": baseline_seconds + candidate_seconds,
        },
        "checkpoints_saved": [],
        "submission_created": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "event": "rs16_gate_complete",
        "decision": decision,
        "point_delta": pooled_delta["point_score"],
        "point_shadow_delta": point_shadow(pooled_delta),
        "native_sps_delta": pooled_delta["sps_score"],
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        try:
            error_args = parse_args()
            if not error_args.output.exists():
                error_args.output.parent.mkdir(parents=True, exist_ok=True)
                error_args.output.write_text(
                    json.dumps({
                        "experiment": EXPERIMENT,
                        "decision": "STOP_RS16_ERROR",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "git": git_identity(),
                        "checkpoints_saved": [],
                        "submission_created": False,
                    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
        except Exception:
            pass
        raise
