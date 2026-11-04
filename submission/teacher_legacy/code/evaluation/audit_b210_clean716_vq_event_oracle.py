#!/usr/bin/env python3
"""Cross-AoA target-visible VQ residual-event representation oracle."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/evaluation"))
import audit_b210_clean716_integer_transport_oracle as integer  # noqa: E402


SCHEMA = "b210_clean716_vq_residual_event_oracle_v1"
AOAS = (0, 5, 10, 15, 20)
DEVELOPMENT_AOA = 10
CONFIRMATION_AOAS = (0, 5, 15, 20)
PATCH_SHAPE = (5, 8, 8, 2)
CODEBOOK_SIZE = 64
LLOYD_STEPS = 20
PATCHES_PER_STRATUM = 64
SHUFFLE_SEEDS = (210, 211, 212, 213)
EXPECTED_PREDICTION_SHA256 = "4e373860669ebaa4368833cbf705bf07452fee7fae0d5682d2370d9ceab5c075"
EXPECTED_TARGET_SHA256 = "533d913297b969613db8ebb3e41a4cb519c207c4270b11fb6bc96be4ff8a622b"
EXPECTED_CONDITION_SHA256 = {
    "aoa": "3610334507b7b876f59efbebb99fc84639b66d02af47f5ffc694e84688cdfe4e",
    "re": "2e59a0efae1c1bc9dc5f91e2b75cf11d0f6a19846e169b2081de60f76ead915b",
    "trajectory": "3d114ce92922744e583dc024ec52cab1117af4cd3ab477a9f2b811e086d80c8f",
    "split_manifest": "bd6f8375e89bbc71571b35a25a307a99abb68bb170d8e5ceba462ec2cec26958",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def patchify(values: np.ndarray) -> np.ndarray:
    if values.ndim != 5 or values.shape[1:] != (20, 32, 64, 2):
        raise ValueError("values must have shape (N,20,32,64,2)")
    batch = values.shape[0]
    return values.reshape(batch, 4, 5, 4, 8, 8, 8, 2).transpose(
        0, 1, 3, 5, 2, 4, 6, 7)


def unpatchify(values: np.ndarray) -> np.ndarray:
    if values.ndim != 8 or values.shape[1:] != (4, 4, 8, 5, 8, 8, 2):
        raise ValueError("patches must have shape (N,4,4,8,5,8,8,2)")
    batch = values.shape[0]
    return values.transpose(0, 1, 4, 2, 5, 3, 6, 7).reshape(
        batch, 20, 32, 64, 2)


def zero_temporal_mean(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float32, copy=True)
    result -= result.mean(axis=1, keepdims=True, dtype=np.float32)
    return result


def fit_scale(patches: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(
        patches.astype(np.float64) ** 2,
        axis=(0, 2, 3, 4, 5, 6))).astype(np.float32)
    return np.maximum(scale, np.float32(1e-6))


def balanced_patch_sample(standardized: np.ndarray, labels: np.ndarray,
                          fit_aoas: tuple[int, ...], rng: np.random.Generator,
                          per_stratum: int = PATCHES_PER_STRATUM) -> tuple[np.ndarray, str]:
    if standardized.shape[1:] != (4, 4, 8, 5, 8, 8, 2):
        raise ValueError("unexpected standardized patch shape")
    rows = []
    digest = hashlib.sha256()
    for aoa in fit_aoas:
        local = np.flatnonzero(labels == aoa)
        if len(local) < per_stratum:
            raise ValueError(f"AoA {aoa} has fewer than {per_stratum} fit samples")
        for block in range(4):
            for tile in range(32):
                chosen = np.sort(rng.choice(local, size=per_stratum, replace=False))
                digest.update(np.asarray((aoa, block, tile), dtype=np.int16).tobytes())
                digest.update(chosen.astype(np.int64, copy=False).tobytes())
                tile_y, tile_x = divmod(tile, 8)
                rows.append(standardized[chosen, block, tile_y, tile_x].reshape(
                    per_stratum, -1))
    result = np.concatenate(rows).astype(np.float32, copy=False)
    expected = len(fit_aoas) * 4 * 32 * per_stratum
    if result.shape != (expected, int(np.prod(PATCH_SHAPE))):
        raise RuntimeError("balanced patch sample has wrong shape")
    return result, digest.hexdigest()


def squared_distances(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (values.square().sum(1, keepdim=True)
            + centers.square().sum(1)[None]
            - 2.0 * values @ centers.T)


def deterministic_kmeans(values: np.ndarray, device: torch.device,
                         seed: int) -> tuple[np.ndarray, list[float], np.ndarray]:
    x = torch.from_numpy(np.ascontiguousarray(values)).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    initial = torch.randperm(len(x), generator=generator, device=device)[:CODEBOOK_SIZE]
    centers = x[initial].clone()
    trace: list[float] = []
    counts = torch.zeros(CODEBOOK_SIZE, dtype=torch.int64, device=device)
    for _ in range(LLOYD_STEPS):
        distance = squared_distances(x, centers)
        assignment = distance.argmin(1)
        membership = F.one_hot(assignment, CODEBOOK_SIZE).to(dtype=x.dtype)
        counts = membership.sum(0).to(torch.int64)
        sums = membership.T @ x
        nonempty = counts > 0
        updated = centers.clone()
        updated[nonempty] = sums[nonempty] / counts[nonempty, None]
        centers = updated
        trace.append(float(distance.gather(1, assignment[:, None]).mean().cpu()))
    return (centers.cpu().numpy().astype(np.float32, copy=False), trace,
            counts.cpu().numpy())


def nearest_codes(values: np.ndarray, centers: np.ndarray,
                  device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(values.reshape(-1, 640))).to(device)
    c = torch.from_numpy(np.ascontiguousarray(centers)).to(device)
    return squared_distances(x, c).argmin(1).to(torch.uint8).cpu().numpy()


def decode(codes: np.ndarray, centers: np.ndarray,
           shape: tuple[int, ...], scale: np.ndarray) -> np.ndarray:
    values = centers[codes.astype(np.int64)].reshape(shape)
    return values * scale[None, :, None, None, None, None, None, :]


def correction_from_patches(values: np.ndarray) -> np.ndarray:
    return zero_temporal_mean(unpatchify(values))


def shuffle_spatial_patches(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = values.copy()
    for sample in range(len(shuffled)):
        for block in range(4):
            local = shuffled[sample, block].reshape(32, 5, 8, 8, 2)
            shuffled[sample, block] = local[rng.permutation(32)].reshape(
                4, 8, 5, 8, 8, 2)
    return shuffled


def delta(after: dict, before: dict) -> dict[str, float]:
    keys = ("rel_l2_score", "tke_score", "mvpe_score",
            "point_final_contribution_46_10_9")
    return {key: float(after[key] - before[key]) for key in keys}


def environment(device: torch.device) -> str:
    return "".join(f"{key}={value}\n" for key, value in {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(), "numpy": np.__version__,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "device": str(device), "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }.items())


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True)
    paths, metadata, _manifest = integer.validate_inputs(args.cache_dir)
    extras = {name: args.cache_dir / f"{name}.npy" for name in ("re", "trajectory")}
    if any(not path.is_file() for path in extras.values()):
        raise FileNotFoundError(extras)
    prediction_sha = sha256_file(paths["prediction"])
    target_sha = sha256_file(paths["target"])
    if prediction_sha != EXPECTED_PREDICTION_SHA256 or target_sha != EXPECTED_TARGET_SHA256:
        raise RuntimeError("frozen clean716 point/target SHA changed")
    condition_paths = {
        "aoa": paths["aoa"], "re": extras["re"],
        "trajectory": extras["trajectory"], "split_manifest": paths["split_manifest"],
    }
    condition_sha = {name: sha256_file(path) for name, path in condition_paths.items()}
    if condition_sha != EXPECTED_CONDITION_SHA256:
        raise RuntimeError("frozen clean716 condition-label/manifest SHA changed")
    script_relative = Path(__file__).resolve().relative_to(REPO_ROOT)
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(script_relative)], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True)
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True).stdout
    if tracked.strip():
        raise RuntimeError("Git worktree must be clean for formal oracle")

    prediction = np.load(paths["prediction"], mmap_mode="r")
    target = np.load(paths["target"], mmap_mode="r")
    aoa = np.load(paths["aoa"], mmap_mode="r")
    reynolds = np.load(extras["re"], mmap_mode="r")
    trajectory = np.load(extras["trajectory"], mmap_mode="r")
    if any(tuple(value.shape) != integer.EXPECTED_SHAPE or value.dtype != np.float32
           for value in (prediction, target)):
        raise ValueError("point/target array contract changed")
    if any(tuple(value.shape) != (integer.EXPECTED_SHAPE[0],)
           for value in (aoa, reynolds, trajectory)):
        raise ValueError("condition labels do not align")

    config = {
        "schema": SCHEMA, "evidence_class": "oracle", "target_visible": True,
        "deployable": False, "base_seen_all_aoa": True,
        "development_aoa": DEVELOPMENT_AOA,
        "confirmation_aoas": list(CONFIRMATION_AOAS), "all_aoas": list(AOAS),
        "patch_shape": list(PATCH_SHAPE), "codebook_size": CODEBOOK_SIZE,
        "lloyd_steps": LLOYD_STEPS, "patches_per_fit_aoa": 8192,
        "sampling": "64 windows without replacement per fit-AoA/time-block/spatial-tile",
        "scale": "fit-only RMS per time-block and channel", "zero_temporal_mean": True,
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "public_point_weights": integer.POINT_WEIGHTS,
        "go_gate": {
            "confirmation_vq_minus_base": {"point": .20, "rel": .10, "tke": .80,
                                              "positive_aoa": 3, "worst_point": -.03},
            "confirmation_vq_minus_each_shuffle": {"point": .10, "tke": .40},
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    (args.output_dir / "environment.txt").write_text(environment(device), encoding="utf-8")

    arms = ("baseline", "k1_mean_code", "vq64_target_assignment",
            *(f"spatial_shuffle_{seed}" for seed in SHUFFLE_SEEDS),
            "continuous_zero_mean_oracle")
    errors = {arm: np.empty((716, 3), dtype=np.float64) for arm in arms}
    sse = {arm: np.empty(716, dtype=np.float64) for arm in arms}
    fold_results = {}
    max_mean_drift = 0.0
    max_continuous_identity_error = 0.0
    started = time.perf_counter()
    for held_aoa in AOAS:
        fit_rows = np.flatnonzero(aoa != held_aoa)
        held_rows = np.flatnonzero(aoa == held_aoa)
        fit_aoas = tuple(value for value in AOAS if value != held_aoa)
        # Only fit-side target values enter scale, sample, and codebook.
        fit_base = np.asarray(prediction[fit_rows], dtype=np.float32)
        fit_target = np.asarray(target[fit_rows], dtype=np.float32)
        fit_residual = zero_temporal_mean(fit_target - fit_base)
        fit_patches = patchify(fit_residual)
        scale = fit_scale(fit_patches)
        standardized = fit_patches / scale[None, :, None, None, None, None, None, :]
        rng = np.random.default_rng(210 + held_aoa)
        train, stream_sha = balanced_patch_sample(
            standardized, np.asarray(aoa[fit_rows]), fit_aoas, rng)
        centers, inertia, train_counts = deterministic_kmeans(
            train, device, 210 + held_aoa)
        if not np.isfinite(centers).all():
            raise FloatingPointError("non-finite codebook")
        codebook_path = args.output_dir / f"codebook_held{held_aoa}.npy"
        np.save(codebook_path, centers)
        codebook_sha = sha256_file(codebook_path)

        # Held target is first used after the fit-only codebook is materialized and hashed.
        held_base = np.asarray(prediction[held_rows], dtype=np.float32)
        held_target = np.asarray(target[held_rows], dtype=np.float32)
        held_residual_raw = held_target - held_base
        held_residual = zero_temporal_mean(held_residual_raw)
        held_patches = patchify(held_residual)
        if not np.array_equal(unpatchify(held_patches), held_residual):
            raise RuntimeError("actual held patch round-trip is not exact")
        held_standard = held_patches / scale[None, :, None, None, None, None, None, :]
        codes = nearest_codes(held_standard, centers, device)
        assignments_path = args.output_dir / f"assignments_held{held_aoa}.npy"
        np.save(assignments_path, codes)
        decoded = decode(codes, centers, held_patches.shape, scale)
        mean_code = train.mean(axis=0, dtype=np.float64).astype(np.float32)
        mean_patches = np.broadcast_to(
            mean_code.reshape(1, 1, 1, 1, 5, 8, 8, 2), held_patches.shape).copy()
        mean_patches *= scale[None, :, None, None, None, None, None, :]
        correction = correction_from_patches(decoded)
        candidates = {
            "baseline": held_base,
            "k1_mean_code": held_base + correction_from_patches(mean_patches),
            "vq64_target_assignment": held_base + correction,
            "continuous_zero_mean_oracle": held_base + held_residual,
        }
        for seed in SHUFFLE_SEEDS:
            candidates[f"spatial_shuffle_{seed}"] = (
                held_base + correction_from_patches(shuffle_spatial_patches(decoded, seed)))

        expected_continuous_residual = -held_residual_raw.mean(
            axis=1, keepdims=True, dtype=np.float32)
        actual_continuous_residual = candidates["continuous_zero_mean_oracle"] - held_target
        max_continuous_identity_error = max(max_continuous_identity_error, float(np.max(
            np.abs(actual_continuous_residual - expected_continuous_residual))))
        local_metrics = {}
        for arm, candidate in candidates.items():
            if not np.isfinite(candidate).all():
                raise FloatingPointError(f"non-finite candidate {arm}")
            errors[arm][held_rows] = integer.official_error_triples(candidate, held_target)
            sse[arm][held_rows] = integer.per_sample_sse(candidate, held_target)
            drift = float(np.max(np.abs(
                candidate.mean(axis=1, dtype=np.float64)
                - held_base.mean(axis=1, dtype=np.float64))))
            max_mean_drift = max(max_mean_drift, drift)
            local_metrics[arm] = {
                **integer.metrics_from_errors(errors[arm][held_rows]),
                "uv_sse": float(sse[arm][held_rows].sum()),
                "temporal_mean_max_abs_delta_vs_base": drift,
            }
        fold_results[str(held_aoa)] = {
            "samples": int(len(held_rows)), "fit_samples": int(len(fit_rows)),
            "fit_aoas": list(fit_aoas), "held_unique_re": int(np.unique(reynolds[held_rows]).size),
            "held_re_all_present_in_fit": bool(set(np.asarray(reynolds[held_rows]).tolist())
                                               <= set(np.asarray(reynolds[fit_rows]).tolist())),
            "scale_by_block_channel": scale.tolist(),
            "sample_stream_sha256": stream_sha,
            "codebook": {"path": str(codebook_path), "sha256": codebook_sha,
                         "train_nonempty_codes": int(np.count_nonzero(train_counts)),
                         "inertia_first_last": [inertia[0], inertia[-1]]},
            "assignment": {"path": str(assignments_path),
                           "sha256": sha256_file(assignments_path),
                           "held_codes_used": int(np.unique(codes).size)},
            "metrics": local_metrics,
            "vq_minus_baseline": delta(local_metrics["vq64_target_assignment"],
                                        local_metrics["baseline"]),
        }
        print(json.dumps({"event": "vq_event_fold_complete", "held_aoa": held_aoa,
                          "vq_minus_baseline": fold_results[str(held_aoa)]["vq_minus_baseline"],
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)

    def summarize(rows: np.ndarray) -> dict:
        metrics = {arm: integer.metrics_from_errors(errors[arm][rows]) for arm in arms}
        metrics["uv_sse"] = {arm: float(sse[arm][rows].sum()) for arm in arms}
        return metrics

    all_rows = np.arange(716)
    confirmation_rows = np.flatnonzero(np.isin(aoa, CONFIRMATION_AOAS))
    pooled = summarize(all_rows)
    confirmation = summarize(confirmation_rows)
    vq_delta = delta(confirmation["vq64_target_assignment"], confirmation["baseline"])
    k1_delta = delta(confirmation["k1_mean_code"], confirmation["baseline"])
    shuffle_deltas = {
        str(seed): delta(confirmation["vq64_target_assignment"],
                         confirmation[f"spatial_shuffle_{seed}"])
        for seed in SHUFFLE_SEEDS}
    aoa_point_deltas = {
        str(value): float(fold_results[str(value)]["vq_minus_baseline"][
            "point_final_contribution_46_10_9"]) for value in AOAS}
    confirmation_values = [aoa_point_deltas[str(value)] for value in CONFIRMATION_AOAS]
    min_codes = min(int(fold_results[str(value)]["assignment"]["held_codes_used"])
                    for value in AOAS)
    checks = {
        "k1_abs_confirmation_point_le_0p01": abs(k1_delta[
            "point_final_contribution_46_10_9"]) <= .01,
        "k1_abs_confirmation_mvpe_le_1e_5": abs(k1_delta["mvpe_score"]) <= 1e-5,
        "vq_abs_confirmation_mvpe_le_1e_5": abs(vq_delta["mvpe_score"]) <= 1e-5,
        "held_codes_used_ge_32_each_fold": min_codes >= 32,
        "temporal_mean_drift_le_2e_6": max_mean_drift <= 2e-6,
        "continuous_identity_error_le_2e_6": max_continuous_identity_error <= 2e-6,
        "confirmation_vq_point_ge_0p20": vq_delta[
            "point_final_contribution_46_10_9"] >= .20,
        "confirmation_vq_rel_ge_0p10": vq_delta["rel_l2_score"] >= .10,
        "confirmation_vq_tke_ge_0p80": vq_delta["tke_score"] >= .80,
        "confirmation_positive_aoa_ge_3": sum(value > 0 for value in confirmation_values) >= 3,
        "confirmation_worst_aoa_point_ge_minus_0p03": min(confirmation_values) >= -.03,
        "vq_beats_every_shuffle_point_ge_0p10": min(
            value["point_final_contribution_46_10_9"] for value in shuffle_deltas.values()) >= .10,
        "vq_beats_every_shuffle_tke_ge_0p40": min(
            value["tke_score"] for value in shuffle_deltas.values()) >= .40,
    }
    result = {
        "schema": SCHEMA, "evidence_class": "oracle", "target_visible": True,
        "deployable": False, "config": config,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path),
                           "bytes": path.stat().st_size} for name, path in {
            "prediction": paths["prediction"], "target": paths["target"],
            "aoa": paths["aoa"], "re": extras["re"],
            "trajectory": extras["trajectory"], "manifest": paths["split_manifest"],
            "official_scorer": integer.SCORING_PATH, "script": Path(__file__).resolve(),
        }.items()},
        "source_archive": metadata.get("archive"), "folds": fold_results,
        "all_aoa_pooled": pooled, "confirmation_without_development_aoa10": {
            "metrics": confirmation, "vq_minus_baseline": vq_delta,
            "k1_minus_baseline": k1_delta, "vq_minus_each_shuffle": shuffle_deltas,
            "vq_minus_baseline_point_by_held_aoa": {
                str(value): aoa_point_deltas[str(value)] for value in CONFIRMATION_AOAS},
        },
        "development_aoa10_vq_point_delta": aoa_point_deltas[str(DEVELOPMENT_AOA)],
        "mechanics": {"max_temporal_mean_drift": max_mean_drift,
                      "max_continuous_identity_error": max_continuous_identity_error,
                      "minimum_held_codes_used": min_codes},
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "decision": ("GO_HISTORY_TO_VQ_TOKEN_SHORT_SCREEN" if all(checks.values())
                     else "STOP_VQ_EVENT_REPRESENTATION"),
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = sorted(path for path in args.output_dir.iterdir() if path.is_file()
                     and path.name != "sha256.txt")
    (args.output_dir / "sha256.txt").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in outputs) + "\n",
        encoding="utf-8")
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"event": "vq_event_oracle_complete",
                      "decision": result["decision"],
                      "checks": result["checks"]}), flush=True)


if __name__ == "__main__":
    main()
