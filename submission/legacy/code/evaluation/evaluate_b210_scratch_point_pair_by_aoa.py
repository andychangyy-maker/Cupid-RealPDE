#!/usr/bin/env python3
"""Re-score sealed scratch-B210 Real endpoints by angle of attack."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
TRAINING_ROOT = CODE_ROOT / "training"
for source in (CODE_ROOT, TRAINING_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import train_b210_np_functional_real_ft12_loao as ft12  # noqa: E402
import train_b210_np_functional_real_small4k_confirm_held0 as small4k  # noqa: E402
import train_b210_scratch_real_dense_aug as scratch  # noqa: E402
import train_c208_dr1_clean81_paired_loao as dr1  # noqa: E402
import train_c208_smu1_paired_loao as smu1  # noqa: E402


SCHEMA = "b210_scratch_point_pair_by_aoa_v1"


def close_metrics(observed: dict, expected: dict,
                  tolerance: float = 1e-3) -> None:
    if set(observed) != set(expected):
        raise RuntimeError("pooled metric keys changed")
    for key in observed:
        if not math.isclose(float(observed[key]), float(expected[key]),
                            rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError(
                f"pooled metric replay changed for {key}: "
                f"{observed[key]} != {expected[key]}")


def load_primary_endpoint_mmap(seal: dict,
                               normalizer: smu1.FoldNormalizer,
                               device: torch.device):
    """Validate the sealed endpoint while avoiding ZIP read amplification."""
    path = Path(str(seal["path"]))
    if dr1.sha256(path) != seal["file_sha256"]:
        raise RuntimeError("small4k endpoint file changed")
    payload = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("model_state_dict")
    if (payload.get("schema") != small4k.ENDPOINT_SCHEMA
            or payload.get("experiment") != small4k.EXPERIMENT
            or payload.get("endpoint") != small4k.PRIMARY_ENDPOINT
            or payload.get("primary") is not True
            or int(payload.get("iteration", -1)) != small4k.UPDATES
            or int(payload.get("held_aoa", -1)) != small4k.HELD_AOA
            or not isinstance(state, dict)
            or dr1.model_state_sha256(state) != seal["model_state_sha256"]):
        raise RuntimeError("small4k endpoint payload changed")
    config = ft12.validate_b210_model_config(payload["model_config"])
    model = small4k.EnsembleUNet2d(config).float()
    model.load_state_dict(state, strict=True)
    ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    return model.to(device)


def load_tail_rows(real_dir: Path, rows: list[dict], audit: scratch.TailAudit):
    """Decode only the frame span needed by each fixed-tail partition."""
    trajectories = []
    metadata = []
    windows = []
    root = real_dir.resolve()
    window_type = ft12.npt1.ensemble_train.common.repro.Window
    for row in rows:
        audit.before_open(row, "held")
        filename = str(row["file"])
        path = (root / filename).resolve()
        if path.parent != root or path.name != filename or not path.is_file():
            raise ValueError(f"unsafe or missing H5 path {filename}")
        starts = list(row["train_starts"])
        first = min(starts)
        stop = max(starts) + 40
        with h5py.File(path, "r") as handle:
            if handle["u"].shape != handle["v"].shape:
                raise ValueError(f"u/v shapes differ in {filename}")
            if int(handle["u"].shape[0]) != int(row["frames"]):
                raise ValueError(f"frame count changed for {filename}")
            if int(np.asarray(handle["re"])) != int(row["re"]):
                raise ValueError(f"Re changed for {filename}")
            if int(np.asarray(handle["aoa"])) != int(row["aoa"]):
                raise ValueError(f"AoA changed for {filename}")
            u = np.asarray(handle["u"][first:stop, ::2, ::2], dtype=np.float32)
            v = np.asarray(handle["v"][first:stop, ::2, ::2], dtype=np.float32)
        audit.record_open(row, "held")
        trajectory = len(trajectories)
        values = np.stack((u, v), axis=-1)
        trajectories.append(torch.from_numpy(np.ascontiguousarray(values)))
        metadata.append({
            "file": filename, "re": int(row["re"]), "aoa": int(row["aoa"]),
            "decoded_first_frame": first, "decoded_stop_frame": stop,
        })
        windows.extend(window_type(trajectory, int(start) - first)
                       for start in starts)
    return trajectories, metadata, windows


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.pair_result.read_text(encoding="utf-8"))
    if source.get("schema") != scratch.SCHEMA:
        raise ValueError("input is not a scratch Real pair result")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean prefix manifest byte identity changed")
    rows, _manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    evaluation_path = Path(source["evaluation_manifest"]["path"])
    if dr1.sha256(evaluation_path) != source["evaluation_manifest"]["sha256"]:
        raise RuntimeError("evaluation manifest changed after paired run")
    evaluation_rows = scratch.load_evaluation_rows(evaluation_path, rows)
    seals = {arm: source["arms"][arm]["seal"]
             for arm in ("control", "candidate")}
    audit = scratch.TailAudit(seals)
    trajectories, metadata, windows = load_tail_rows(
        args.real_dir, evaluation_rows, audit)
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(
        trajectories, windows)
    indices_by_aoa = scratch.group_window_indices_by_aoa(windows, metadata)
    statistics_path = Path(source["normalizer"]["path"])
    statistics = torch.load(
        statistics_path, map_location="cpu", weights_only=False)
    if smu1.statistics_digest(statistics) != source["normalizer"][
            "statistics_sha256"]:
        raise RuntimeError("point normalizer changed after paired run")
    device = torch.device(args.device)
    normalizer = smu1.FoldNormalizer(statistics, device)
    deadline = time.perf_counter() + args.wall_budget_seconds
    reports = {}
    for arm in ("control", "candidate"):
        model = load_primary_endpoint_mmap(seals[arm], normalizer, device)
        by_aoa = {}
        states = []
        for aoa, indices in indices_by_aoa.items():
            metrics, state, target_audit = smu1.evaluate_model(
                model, dataset, indices, normalizer, device, deadline)
            by_aoa[str(aoa)] = {
                "windows": len(indices),
                "metrics": metrics,
                "target_audit": target_audit,
            }
            states.append(state)
        pooled_state = smu1.merge_metric_states(states)
        pooled = smu1.finalize_metric_state(pooled_state)
        close_metrics(pooled, source["arms"][arm]["metrics"])
        reports[arm] = {"pooled": pooled, "by_aoa": by_aoa}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    comparisons = {
        str(aoa): scratch.compare(
            reports["candidate"]["by_aoa"][str(aoa)]["metrics"],
            reports["control"]["by_aoa"][str(aoa)]["metrics"])
        for aoa in dr1.AOA_VALUES
    }
    result = {
        "schema": SCHEMA,
        "pair_result": {
            "path": str(args.pair_result.resolve()),
            "sha256": dr1.sha256(args.pair_result),
        },
        "reports": reports,
        "comparison_by_aoa": comparisons,
        "candidate_positive_aoas": sum(
            row["candidate_non_time_proxy_delta"] > 0.0
            for row in comparisons.values()),
        "all_pooled_metrics_replayed": True,
        "tail_opened_after_both_seals": True,
        "submission_authorized": False,
    }
    dr1.write_json_atomic(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-result", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float, default=3600)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "event": "b210_scratch_point_pair_by_aoa_complete",
        "candidate_positive_aoas": result["candidate_positive_aoas"],
    }), flush=True)


if __name__ == "__main__":
    main()
