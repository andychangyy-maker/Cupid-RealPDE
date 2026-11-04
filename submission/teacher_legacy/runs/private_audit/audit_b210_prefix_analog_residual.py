#!/usr/bin/env python3
"""Target-safe audit of analog residual retrieval from released Real prefixes.

The audit has a hard two-phase boundary.  It first reads only the released
training prefix of every clean Real trajectory, predicts those windows with a
frozen submission archive, and selects K/shrinkage using cross-trajectory
prefix residuals.  Only after the selection record is hashed does it open the
clean temporal tail.  A tail query is forbidden from retrieving any prefix
from its own trajectory.

This is a mechanism audit, not a submission builder.  Raw fields, retrieved
neighbour identities and predictions are never written to the result JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


PRIVATE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PRIVATE_ROOT.parents[1]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "training",
             REPO_ROOT / "code" / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_track1_submission_archive as archive_eval  # noqa: E402
import train_hybrid_fno2d as common  # noqa: E402


OFFICIAL = archive_eval.OFFICIAL
SCHEMA = "b210_prefix_analog_residual_audit_v1"
FORMAL_PREFIX_MANIFEST_SHA256 = (
    "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e")
CLEAN716_EVALUATION_MANIFEST_SHA256 = (
    "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958")
EXPECTED_TRAJECTORIES = 81
EXPECTED_PREFIX_WINDOWS = 2544
EXPECTED_TAIL_WINDOWS = 716
K_CANDIDATES = (4, 8, 16, 32)
MAX_SHRINKAGE = 0.25
SHRINKAGE_QUANTUM = 0.025
FEATURE_RANK = 64
COARSE_HEIGHT = 8
COARSE_WIDTH = 16
QUERY_CHUNK = 32
QUALITY_WEIGHTS = {
    "rel_l2_score": 0.46,
    "tke_score": 0.10,
    "mvpe_score": 0.09,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temp-root", type=Path, default=Path("runs/tmp"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _normalized_rows(raw: object, *, require_tail: bool) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest must be a non-empty list")
    rows = []
    seen = set()
    for value in raw:
        if not isinstance(value, dict):
            raise TypeError("manifest row must be a mapping")
        filename = str(value["file"])
        if Path(filename).name != filename or filename in seen:
            raise ValueError(f"unsafe or duplicate filename: {filename}")
        seen.add(filename)
        train_starts = [int(item) for item in value["train_starts"]]
        if train_starts != sorted(set(train_starts)) or not train_starts:
            raise ValueError(f"invalid training starts: {filename}")
        row = {
            "file": filename,
            "frames": int(value["frames"]),
            "re": int(value["re"]),
            "aoa": int(value["aoa"]),
            "train_starts": train_starts,
            "val_boundary": int(value["val_boundary"]),
        }
        if require_tail:
            tail = [int(item) for item in value["val_starts"]]
            if tail != sorted(set(tail)) or not tail:
                raise ValueError(f"invalid evaluation starts: {filename}")
            row["val_starts"] = tail
        rows.append(row)
    return rows


def load_and_match_manifests(prefix_path: Path, evaluation_path: Path
                             ) -> tuple[list[dict], list[dict]]:
    if sha256(prefix_path) != FORMAL_PREFIX_MANIFEST_SHA256:
        raise ValueError("prefix manifest is not the frozen clean81 artifact")
    if sha256(evaluation_path) != CLEAN716_EVALUATION_MANIFEST_SHA256:
        raise ValueError("evaluation manifest is not the frozen clean716 artifact")
    prefix = _normalized_rows(
        json.loads(prefix_path.read_text(encoding="utf-8")),
        require_tail=False)
    evaluation = _normalized_rows(
        json.loads(evaluation_path.read_text(encoding="utf-8")),
        require_tail=True)
    by_name = {row["file"]: row for row in evaluation}
    if len(prefix) != EXPECTED_TRAJECTORIES or set(by_name) != {
            row["file"] for row in prefix}:
        raise RuntimeError("clean manifests do not cover the same 81 trajectories")
    for row in prefix:
        other = by_name[row["file"]]
        for key in ("frames", "re", "aoa", "train_starts", "val_boundary"):
            if row[key] != other[key]:
                raise RuntimeError(f"manifest mismatch for {row['file']}: {key}")
        if any(start + 40 > row["val_boundary"]
               for start in row["train_starts"]):
            raise RuntimeError("prefix target crosses the clean tail boundary")
        if min(other["val_starts"]) < row["val_boundary"]:
            raise RuntimeError("tail input starts before the frozen boundary")
    if sum(len(row["train_starts"]) for row in prefix) != EXPECTED_PREFIX_WINDOWS:
        raise RuntimeError("unexpected clean prefix window count")
    if sum(len(row["val_starts"]) for row in evaluation) != EXPECTED_TAIL_WINDOWS:
        raise RuntimeError("unexpected clean tail window count")
    return prefix, [by_name[row["file"]] for row in prefix]


def load_phase_dataset(real_dir: Path, rows: list[dict], phase: str):
    if phase not in ("prefix", "tail"):
        raise ValueError("phase must be prefix or tail")
    root = real_dir.resolve()
    trajectories = []
    windows = []
    trajectory_ids = []
    aoa = []
    opened = []
    for trajectory, row in enumerate(rows):
        path = (root / row["file"]).resolve()
        if path.parent != root or path.name != row["file"]:
            raise ValueError(f"unsafe Real path: {row['file']}")
        stop = row["val_boundary"] if phase == "prefix" else row["frames"]
        starts = row["train_starts"] if phase == "prefix" else row["val_starts"]
        with h5py.File(path, "r") as handle:
            if int(handle["u"].shape[0]) != row["frames"]:
                raise RuntimeError(f"frame count changed: {row['file']}")
            u = np.asarray(handle["u"][:stop, ::2, ::2], dtype=np.float32)
            v = np.asarray(handle["v"][:stop, ::2, ::2], dtype=np.float32)
        values = np.stack((u, v), axis=-1)
        trajectories.append(torch.from_numpy(np.ascontiguousarray(values)))
        opened.append({
            "file": row["file"],
            "max_frame_exclusive": stop,
            "phase": phase,
        })
        for start in starts:
            windows.append(common.repro.Window(trajectory, start))
            trajectory_ids.append(trajectory)
            aoa.append(row["aoa"])
    dataset = common.repro.WindowDataset(trajectories, windows)
    return dataset, torch.tensor(trajectory_ids), torch.tensor(aoa), opened


def _pool(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Pool [B,T,H,W,C] to [B,T,C,height,width]."""
    batch, steps, source_h, source_w, channels = value.shape
    frames = value.permute(0, 1, 4, 2, 3).reshape(
        batch * steps, channels, source_h, source_w)
    pooled = F.adaptive_avg_pool2d(frames, (height, width))
    return pooled.reshape(batch, steps, channels, height, width)


def target_free_features(history: torch.Tensor,
                         prediction: torch.Tensor) -> torch.Tensor:
    """Fixed descriptor using only the scored input and frozen forecast."""
    history = history[..., :2].float()
    prediction = prediction[..., :2].float()
    history_views = torch.stack((
        history[:, -1],
        history[:, -5:].mean(dim=1),
        history[:, -1] - history[:, -5],
    ), dim=1)
    forecast_views = torch.stack((
        prediction[:, :5].mean(dim=1),
        prediction.mean(dim=1),
        prediction[:, -5:].mean(dim=1),
    ), dim=1)
    spatial = torch.cat((
        _pool(history_views, 4, 8).flatten(1),
        _pool(forecast_views, 4, 8).flatten(1),
    ), dim=1)

    def statistics(value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(dim=(2, 3))
        std = value.std(dim=(2, 3), unbiased=False)
        return torch.cat((mean, std), dim=-1).flatten(1)

    return torch.cat((spatial, statistics(history), statistics(prediction)), dim=1)


def masked_coarse_residual(prediction: torch.Tensor, target: torch.Tensor,
                           height: int = COARSE_HEIGHT,
                           width: int = COARSE_WIDTH) -> torch.Tensor:
    prediction = prediction[..., :2].float()
    target = target[..., :2].float()
    scored = target != 0.0
    residual = (target - prediction) * scored
    numerator = _pool(residual, height, width)
    denominator = _pool(scored.float(), height, width)
    return torch.where(
        denominator > 0.0, numerator / denominator.clamp_min(1.0e-6),
        torch.zeros_like(numerator))


@torch.inference_mode()
def collect_phase(submission, dataset, batch_size: int,
                  retain_fields: bool) -> dict[str, torch.Tensor]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=True)
    cache: dict[str, list[torch.Tensor]] = {
        "feature": [], "coarse_residual": [], "valid": []}
    if retain_fields:
        cache.update({"prediction": [], "target": []})
    for history, target in loader:
        output = submission.predict(history.numpy(), metadata={})
        prediction = torch.from_numpy(np.asarray(
            output["prediction"], dtype=np.float32))
        if prediction.shape != target.shape:
            raise RuntimeError("submission prediction shape changed")
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("submission returned non-finite prediction")
        cache["feature"].append(
            target_free_features(history, prediction).to(torch.float16))
        cache["coarse_residual"].append(
            masked_coarse_residual(prediction, target).to(torch.float16))
        valid = history[..., :2].abs().amax(dim=(1, 4)) > 1.0e-8
        cache["valid"].append(valid)
        if retain_fields:
            cache["prediction"].append(prediction)
            cache["target"].append(target.float())
    return {key: torch.cat(value, dim=0) for key, value in cache.items()}


def fit_feature_projection(feature: torch.Tensor, rank: int = FEATURE_RANK
                           ) -> tuple[torch.Tensor, dict]:
    feature = feature.float()
    mean = feature.mean(dim=0)
    std = feature.std(dim=0, unbiased=False)
    active = std > 1.0e-6
    standardized = (feature[:, active] - mean[active]) / std[active]
    q = min(rank, standardized.shape[0] - 1, standardized.shape[1])
    _u, singular, vectors = torch.pca_lowrank(
        standardized, q=q, center=False, niter=6)
    embedding = standardized @ vectors[:, :q]
    scale = embedding.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    state = {
        "mean": mean,
        "std": std,
        "active": active,
        "vectors": vectors[:, :q],
        "scale": scale,
    }
    return embedding / scale, {
        "source_features": int(feature.shape[1]),
        "active_features": int(active.sum()),
        "rank": q,
        "singular_values": singular[:q].tolist(),
        "state": state,
    }


def project_features(feature: torch.Tensor, state: dict) -> torch.Tensor:
    feature = feature.float()
    standardized = (
        feature[:, state["active"]] - state["mean"][state["active"]]
    ) / state["std"][state["active"]]
    return (standardized @ state["vectors"]) / state["scale"]


def retrieve_coarse_residual(
        query_embedding: torch.Tensor, memory_embedding: torch.Tensor,
        memory_residual: torch.Tensor, query_trajectory: torch.Tensor,
        memory_trajectory: torch.Tensor, k: int,
        query_chunk: int = QUERY_CHUNK) -> tuple[torch.Tensor, dict]:
    if k <= 0:
        raise ValueError("k must be positive")
    if memory_embedding.shape[0] != memory_residual.shape[0]:
        raise ValueError("memory embeddings and residuals are misaligned")
    if memory_embedding.shape[0] != memory_trajectory.numel():
        raise ValueError("memory trajectory ids are misaligned")
    results = []
    distances = []
    for start in range(0, len(query_embedding), query_chunk):
        stop = min(start + query_chunk, len(query_embedding))
        current = query_embedding[start:stop]
        squared = torch.cdist(current, memory_embedding).square()
        forbidden = (
            query_trajectory[start:stop, None] == memory_trajectory[None, :])
        squared = squared.masked_fill(forbidden, float("inf"))
        nearest_distance, nearest = torch.topk(
            squared, k=k, dim=1, largest=False, sorted=True)
        if not bool(torch.isfinite(nearest_distance).all()):
            raise RuntimeError("trajectory exclusion left too few analogs")
        local_scale = nearest_distance[:, -1:].clamp_min(1.0e-6)
        weights = torch.softmax(
            -(nearest_distance - nearest_distance[:, :1]) / local_scale,
            dim=1)
        selected = memory_residual[nearest]
        shape = (len(current), k) + (1,) * (selected.ndim - 2)
        results.append((selected * weights.view(shape)).sum(dim=1))
        distances.append(nearest_distance.sqrt().mean(dim=1))
    distance = torch.cat(distances)
    return torch.cat(results), {
        "mean_neighbor_distance": float(distance.mean()),
        "p90_neighbor_distance": float(torch.quantile(distance, 0.90)),
    }


def residual_alignment(retrieved: torch.Tensor, truth: torch.Tensor) -> dict:
    retrieved = retrieved.float().flatten(1)
    truth = truth.float().flatten(1)
    numerator = (retrieved * truth).sum(dim=1)
    cosine = numerator / (
        retrieved.square().sum(dim=1).sqrt()
        * truth.square().sum(dim=1).sqrt()).clamp_min(1.0e-12)
    least_squares = float(
        (retrieved * truth).sum(dtype=torch.float64)
        / retrieved.square().sum(dtype=torch.float64).clamp_min(1.0e-12))
    return {
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(cosine.median()),
        "positive_cosine_fraction": float((cosine > 0.0).float().mean()),
        "unclipped_least_squares_shrinkage": least_squares,
    }


def quantized_shrinkage(value: float) -> float:
    clipped = min(MAX_SHRINKAGE, max(0.0, value))
    return round(clipped / SHRINKAGE_QUANTUM) * SHRINKAGE_QUANTUM


def select_prefix_rule(
        embedding: torch.Tensor, residual: torch.Tensor,
        trajectory: torch.Tensor) -> tuple[dict, dict[int, dict]]:
    candidates = {}
    truth = residual.float()
    denominator = truth.square().sum(dtype=torch.float64).clamp_min(1.0e-12)
    for k in K_CANDIDATES:
        retrieved, distance = retrieve_coarse_residual(
            embedding, embedding, truth, trajectory, trajectory, k)
        alignment = residual_alignment(retrieved, truth)
        shrinkage = quantized_shrinkage(
            alignment["unclipped_least_squares_shrinkage"])
        corrected = truth - shrinkage * retrieved
        normalized_mse = float(
            corrected.square().sum(dtype=torch.float64) / denominator)
        candidates[k] = {
            "k": k,
            "shrinkage": shrinkage,
            "normalized_coarse_residual_mse": normalized_mse,
            "coarse_mse_improvement": 1.0 - normalized_mse,
            "alignment": alignment,
            "distance": distance,
        }
    selected = min(
        candidates.values(),
        key=lambda row: (row["normalized_coarse_residual_mse"], row["k"]))
    return selected, candidates


def upsample_correction(coarse: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    batch, steps, channels, height, width = coarse.shape
    frames = coarse.float().reshape(batch * steps, channels, height, width)
    full = F.interpolate(
        frames, size=(32, 64), mode="bilinear", align_corners=False)
    full = full.reshape(batch, steps, channels, 32, 64).permute(0, 1, 3, 4, 2)
    return full * valid[:, None, :, :, None]


def point_scores(prediction: torch.Tensor, target: torch.Tensor,
                 indices: torch.Tensor | None = None) -> dict[str, float]:
    if indices is not None:
        prediction = prediction.index_select(0, indices)
        target = target.index_select(0, indices)
    pred = prediction.numpy().astype(np.float32, copy=False)
    truth = target.numpy().astype(np.float32, copy=False)
    channels = OFFICIAL.measured_channels(truth)
    rel = OFFICIAL.score_error(float(np.mean(
        OFFICIAL.rel_l2_per_sample(pred, truth, channels))))
    tke = OFFICIAL.score_error(float(np.mean(
        OFFICIAL.tke_rel_l2_per_sample(pred, truth, channels))))
    mvpe = OFFICIAL.score_error(float(np.mean(
        OFFICIAL.mvpe_rel_l2_per_sample(pred, truth))))
    quality = (
        QUALITY_WEIGHTS["rel_l2_score"] * rel
        + QUALITY_WEIGHTS["tke_score"] * tke
        + QUALITY_WEIGHTS["mvpe_score"] * mvpe)
    return {
        "rel_l2_score": float(rel),
        "tke_score": float(tke),
        "mvpe_score": float(mvpe),
        "weighted_quality_contribution": float(quality),
        "samples": int(len(pred)),
    }


def metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in ("rel_l2_score", "tke_score", "mvpe_score",
                    "weighted_quality_contribution")
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(31)
    started = time.perf_counter()
    prefix_rows, evaluation_rows = load_and_match_manifests(
        args.prefix_manifest, args.evaluation_manifest)

    args.temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="analog-audit-", dir=args.temp_root) as temporary:
        package_dir = Path(temporary)
        archive_eval.safe_extract(args.archive, package_dir)
        submission = archive_eval.load_submission(package_dir)

        prefix_dataset, prefix_trajectory, prefix_aoa, prefix_open = (
            load_phase_dataset(args.real_dir, prefix_rows, "prefix"))
        prefix_cache = collect_phase(
            submission, prefix_dataset, args.batch_size, retain_fields=False)
        memory_embedding, projection_report = fit_feature_projection(
            prefix_cache["feature"])
        projection_state = projection_report.pop("state")
        selected, prefix_candidates = select_prefix_rule(
            memory_embedding, prefix_cache["coarse_residual"],
            prefix_trajectory)
        selected_record = {
            "selection_source": "clean81 prefix only; cross-trajectory analogs",
            "selected": selected,
            "candidates": {
                str(key): value for key, value in prefix_candidates.items()},
            "max_shrinkage": MAX_SHRINKAGE,
            "shrinkage_quantum": SHRINKAGE_QUANTUM,
            "prefix_windows": len(prefix_dataset),
            "tail_arrays_read": 0,
        }
        selection_sha = canonical_sha256(selected_record)
        print(json.dumps({
            "event": "analog_prefix_rule_sealed",
            "selection_sha256": selection_sha,
            "selected": selected,
        }), flush=True)

        # The clean tail is not constructed or opened until the immutable
        # prefix-only rule above has been serialized and hashed.
        tail_dataset, tail_trajectory, tail_aoa, tail_open = load_phase_dataset(
            args.real_dir, evaluation_rows, "tail")
        tail_cache = collect_phase(
            submission, tail_dataset, args.batch_size, retain_fields=True)
        tail_embedding = project_features(
            tail_cache["feature"], projection_state)
        retrieved, distance = retrieve_coarse_residual(
            tail_embedding, memory_embedding,
            prefix_cache["coarse_residual"].float(), tail_trajectory,
            prefix_trajectory, int(selected["k"]))
        correction = upsample_correction(retrieved, tail_cache["valid"])
        prediction = tail_cache["prediction"].clone()
        prediction[..., :2] += float(selected["shrinkage"]) * correction
        baseline = point_scores(tail_cache["prediction"], tail_cache["target"])
        candidate = point_scores(prediction, tail_cache["target"])
        delta = metric_delta(candidate, baseline)
        alignment = residual_alignment(
            retrieved, tail_cache["coarse_residual"])

        by_aoa = {}
        for value in sorted(set(tail_aoa.tolist())):
            indices = torch.nonzero(tail_aoa == value).flatten()
            base_row = point_scores(
                tail_cache["prediction"], tail_cache["target"], indices)
            candidate_row = point_scores(prediction, tail_cache["target"], indices)
            by_aoa[str(value)] = {
                "baseline": base_row,
                "candidate": candidate_row,
                "delta": metric_delta(candidate_row, base_row),
            }

        # Tail-label strength sweep is an explicitly non-deployable mechanism
        # ceiling.  It is never used to replace the sealed prefix-only rule.
        tail_oracle = []
        for shrinkage in (
                0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25):
            current = tail_cache["prediction"].clone()
            current[..., :2] += shrinkage * correction
            scores = point_scores(current, tail_cache["target"])
            tail_oracle.append({
                "shrinkage": shrinkage,
                "scores": scores,
                "delta": metric_delta(scores, baseline),
            })
        oracle_best = max(
            tail_oracle,
            key=lambda row: row["scores"]["weighted_quality_contribution"])

        result = {
            "schema": SCHEMA,
            "submission_authorized": False,
            "archive": {
                "path": str(args.archive.resolve()),
                "sha256": sha256(args.archive),
            },
            "manifests": {
                "prefix_sha256": sha256(args.prefix_manifest),
                "evaluation_sha256": sha256(args.evaluation_manifest),
                "trajectories": len(prefix_rows),
                "prefix_windows": len(prefix_dataset),
                "tail_windows": len(tail_dataset),
            },
            "protocol": {
                "tail_open_after_prefix_rule_seal": True,
                "selection_sha256": selection_sha,
                "same_trajectory_retrieval_forbidden": True,
                "metadata_used_by_predictor": False,
                "raw_fields_written": False,
                "prefix_access": prefix_open,
                "tail_access": tail_open,
            },
            "feature_projection": projection_report,
            "prefix_selection": selected_record,
            "tail_primary": {
                "baseline": baseline,
                "candidate": candidate,
                "delta": delta,
                "alignment": alignment,
                "distance": distance,
                "by_aoa": by_aoa,
            },
            "tail_oracle_strength_diagnostic_not_deployable": {
                "rows": tail_oracle,
                "best": oracle_best,
            },
            "promotion_gate": {
                "weighted_quality_delta_min": 0.02,
                "rel_l2_delta_min": -0.02,
                "tail_mean_cosine_min": 0.03,
                "positive_aoa_weighted_quality_min": 4,
            },
            "gate_result": {
                "weighted_quality": (
                    delta["weighted_quality_contribution"] >= 0.02),
                "rel_l2": delta["rel_l2_score"] >= -0.02,
                "tail_alignment": alignment["mean_cosine"] >= 0.03,
                "positive_aoa": sum(
                    row["delta"]["weighted_quality_contribution"] >= 0.0
                    for row in by_aoa.values()) >= 4,
            },
            "interpretation_boundary": (
                "PASS only authorizes distilling the analog correction into a "
                "fixed-latency model. This audit does not authorize a runtime "
                "retrieval submission or tail-selected shrinkage."),
            "elapsed_seconds": time.perf_counter() - started,
        }
        result["gate_result"]["pass"] = all(
            result["gate_result"][key]
            for key in ("weighted_quality", "rel_l2", "tail_alignment",
                        "positive_aoa"))
        write_json_atomic(args.output, result)
        print(json.dumps({
            "event": "analog_residual_audit_complete",
            "output": str(args.output.resolve()),
            "selected": selected,
            "tail_delta": delta,
            "tail_alignment": alignment,
            "gate": result["gate_result"],
            "oracle_best": oracle_best,
        }), flush=True)


if __name__ == "__main__":
    main()
