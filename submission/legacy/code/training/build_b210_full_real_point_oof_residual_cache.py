#!/usr/bin/env python3
"""Build 3,341 whole-AoA point-OOF residual labels for fresh Native-UQ.

The sample inventory is derived only from the SHA-bound clean81 training
manifest: every legal 20->20 window on every complete released trajectory at
stride 20.  Each sample is predicted by the B210 Real4k endpoint whose fit
excluded the sample's entire AoA.  No clean716/evaluation manifest is accepted.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import train_b210_full_real_point_oof_fold as point_fold


dr1 = point_fold.dr1
ft12 = point_fold.ft12
SCHEMA = "b210_full_real_point_oof_residual_cache_v1"
EXPECTED_SAMPLES = point_fold.EXPECTED_TOTAL["full_oof_windows"]


def read_trajectory(
        real_dir: Path, row: Mapping[str, object]) -> torch.Tensor:
    root = real_dir.resolve()
    filename = str(row["file"])
    if Path(filename).name != filename or filename == dr1.EXCLUDED_BASENAME:
        raise ValueError(f"unsafe/excluded H5 filename {filename}")
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if handle["u"].shape != handle["v"].shape:
            raise ValueError(f"u/v shape mismatch in {filename}")
        if int(handle["u"].shape[0]) != int(row["frames"]):
            raise ValueError(f"frame count changed in {filename}")
        if int(np.asarray(handle["re"])) != int(row["re"]):
            raise ValueError(f"Re changed in {filename}")
        if int(np.asarray(handle["aoa"])) != int(row["aoa"]):
            raise ValueError(f"AoA changed in {filename}")
        u = np.asarray(handle["u"][:, ::2, ::2], dtype=np.float32)
        v = np.asarray(handle["v"][:, ::2, ::2], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(np.stack((u, v), axis=-1)))


def uvp_windows(
        trajectory: torch.Tensor, starts: Sequence[int],
        ) -> tuple[torch.Tensor, torch.Tensor]:
    windows = torch.stack([trajectory[start:start + 40] for start in starts])
    pressure = torch.zeros((*windows.shape[:-1], 1), dtype=windows.dtype)
    values = torch.cat((windows, pressure), dim=-1)
    return values[:, :20], values[:, 20:]


def checkpoint_normalizer(
        payload: Mapping[str, object], device: torch.device,
        ) -> dict[str, torch.Tensor]:
    raw = payload.get("normalizer")
    if not isinstance(raw, Mapping):
        raise RuntimeError("point-OOF endpoint has no normalizer")
    result = {}
    for key in ("mean_input", "std_input", "mean_target", "std_target"):
        value = torch.as_tensor(raw.get(key), device=device, dtype=torch.float32)
        if value.shape != (3,) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"invalid point-OOF normalizer field {key}")
        result[key] = value
    return result


def predict_fold(
        checkpoint: Path, rows: Sequence[Mapping[str, object]],
        positions: Mapping[tuple[str, int], int], real_dir: Path,
        residuals: torch.Tensor, factors: torch.Tensor,
        history_hashes: list[str | None], target_hashes: list[str | None],
        device: torch.device, batch_size: int) -> int:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    normalizer = checkpoint_normalizer(payload, device)
    config = ft12.validate_b210_model_config(
        payload["model_config"], point_fold.sim_tke.b210_config("candidate"))
    model = point_fold.small4k.EnsembleUNet2d(config).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()
    written = 0
    with torch.inference_mode():
        for row in rows:
            starts = [int(value) for value in row["val_starts"]]
            history, target = uvp_windows(read_trajectory(real_dir, row), starts)
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset:offset + batch_size]
                batch_history = history[offset:offset + batch_size].to(device)
                batch_target = target[offset:offset + batch_size].to(device)
                mean_input = normalizer["mean_input"].view(1, 1, 1, 1, 3)
                std_input = normalizer["std_input"].view(1, 1, 1, 1, 3)
                prediction_norm = model((batch_history - mean_input) / std_input)
                prediction = (
                    prediction_norm
                    * normalizer["std_target"].view(1, 1, 1, 1, 3)
                    + normalizer["mean_target"].view(1, 1, 1, 1, 3)
                )[..., :2]
                truth = batch_target[..., :2]
                signed = (truth - prediction).to(dtype=torch.float16).cpu()
                factor = ft12.npt1.ensemble_train.common.sample_sps_factor(
                    prediction, truth).float().cpu()
                for local, start in enumerate(batch_starts):
                    position = positions[(str(row["file"]), start)]
                    if torch.isfinite(factors[position]):
                        raise RuntimeError("point-OOF cache position written twice")
                    residuals[position].copy_(signed[local])
                    factors[position] = factor[local]
                    history_hashes[position] = hashlib.sha256(
                        history[offset + local].numpy().tobytes(order="C")
                    ).hexdigest()
                    target_hashes[position] = hashlib.sha256(
                        target[offset + local].numpy().tobytes(order="C")
                    ).hexdigest()
                    written += 1
    del model, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return written


def full_evaluation_rows(
        rows: Sequence[Mapping[str, object]], stride: int = 20) -> list[dict]:
    if stride != point_fold.EVALUATION_STRIDE:
        raise ValueError("formal OOF residual cache fixes stride 20")
    result = []
    identities: set[tuple[str, int]] = set()
    for row in rows:
        filename = str(row["file"])
        if (Path(filename).name != filename
                or filename == dr1.EXCLUDED_BASENAME):
            raise ValueError("unsafe/excluded released-Real trajectory")
        frames = int(row["frames"])
        starts = list(range(0, frames - 40 + 1, stride))
        if not starts:
            raise ValueError("released trajectory has no legal OOF window")
        for start in starts:
            identity = (filename, start)
            if identity in identities:
                raise RuntimeError("duplicate full OOF window identity")
            identities.add(identity)
        result.append({
            "file": filename,
            "frames": frames,
            "re": int(row["re"]),
            "aoa": int(row["aoa"]),
            "val_starts": starts,
        })
    if len(identities) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"full stride20 OOF inventory changed: {len(identities)}")
    return result


def fold_assignments(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    assignments = {}
    for row in rows:
        filename = str(row["file"])
        if filename in assignments:
            raise ValueError("OOF manifest contains duplicate trajectory")
        assignments[filename] = point_fold.aoa_fold(int(row["aoa"]))
    if set(assignments.values()) != set(range(point_fold.FOLDS)):
        raise RuntimeError("whole-AoA OOF assignment has an empty fold")
    return assignments


def validate_fold_results(
        rows: Sequence[Mapping[str, object]], assignments: Mapping[str, int],
        result_pairs: Sequence[Sequence[str]],
        ) -> tuple[dict[int, Path], list[dict]]:
    if len(result_pairs) != point_fold.FOLDS:
        raise ValueError("one SHA-bound point result is required per held AoA")
    expected_files = {
        fold: sorted(name for name, value in assignments.items() if value == fold)
        for fold in range(point_fold.FOLDS)}
    checkpoints: dict[int, Path] = {}
    reports = []
    for raw_path, raw_sha in result_pairs:
        result_path = Path(raw_path)
        expected_sha = dr1.normalized_sha256(raw_sha, "point-OOF result SHA")
        if not result_path.is_file() or dr1.sha256(result_path) != expected_sha:
            raise RuntimeError("point-OOF result file SHA changed")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        point_fold.verify_json_payload(payload)
        held_aoa = int(payload.get("held_aoa", -1))
        fold = point_fold.aoa_fold(held_aoa)
        if (fold in checkpoints
                or payload.get("schema") != point_fold.SCHEMA
                or int(payload.get("held_fold", -1)) != fold
                or int(payload.get("folds", -1)) != point_fold.FOLDS
                or payload.get("aoa_values") != list(point_fold.AOA_VALUES)
                or payload.get("decision") != "COMPLETE_FULL_REAL_POINT_OOF_FOLD"
                or payload.get("held_data_opened") is not False
                or payload.get("clean716_manifest_read") is not False
                or payload.get("start", {}).get("checkpoint_sha256")
                != point_fold.START_SHA256
                or payload.get("manifest_file_sha256")
                != ft12.FORMAL_MANIFEST_FILE_SHA256
                or payload.get("manifest", {}).get("semantic_sha256")
                != ft12.FORMAL_MANIFEST_SEMANTIC_SHA256):
            raise RuntimeError("point-OOF fold result identity changed")
        held_names = sorted(
            str(row["file"]) for row in rows if int(row["aoa"]) == held_aoa)
        if held_names != expected_files[fold]:
            raise RuntimeError("point-OOF held-AoA membership changed")
        seal = payload.get("fit", {}).get("seal", {})
        checkpoint = Path(str(seal.get("path", "")))
        if (not checkpoint.is_file()
                or dr1.sha256(checkpoint) != seal.get("file_sha256")):
            raise RuntimeError("point-OOF endpoint file SHA changed")
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint_payload.get("model_state_dict")
        accepted_identity = {
            (point_fold.ENDPOINT_SCHEMA, point_fold.EXPERIMENT),
            (point_fold.support.ENDPOINT_SCHEMA, point_fold.support.EXPERIMENT),
        }
        if ((checkpoint_payload.get("schema"),
             checkpoint_payload.get("experiment")) not in accepted_identity
                or checkpoint_payload.get("phase") != "real"
                or int(checkpoint_payload.get("iteration", -1)) != point_fold.UPDATES
                or checkpoint_payload.get("endpoint") != point_fold.ENDPOINT
                or int(checkpoint_payload.get("held_aoa", -1)) != held_aoa
                or checkpoint_payload.get("start", {}).get("checkpoint_sha256")
                != point_fold.START_SHA256
                or not isinstance(state, Mapping)
                or dr1.model_state_sha256(state)
                != seal.get("model_state_sha256")):
            raise RuntimeError("point-OOF endpoint payload identity changed")
        checkpoints[fold] = checkpoint.resolve()
        reports.append({
            "fold": fold,
            "held_aoa": held_aoa,
            "held_files": held_names,
            "result": str(result_path.resolve()),
            "result_sha256": expected_sha,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": seal["file_sha256"],
            "model_state_sha256": seal["model_state_sha256"],
            "source_kind": payload["source_kind"],
        })
    if set(checkpoints) != set(range(point_fold.FOLDS)):
        raise RuntimeError("point-OOF checkpoint coverage is incomplete")
    return checkpoints, sorted(reports, key=lambda row: int(row["fold"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument(
        "--fold-result", nargs=2, action="append", required=True,
        metavar=("PATH", "SHA256"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.output.with_suffix(
            args.output.suffix + ".tmp").exists():
        raise FileExistsError(args.output)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if dr1.sha256(args.clean_manifest) != ft12.FORMAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("clean81 runtime manifest file identity changed")
    rows, manifest_report = dr1.load_manifest(
        args.clean_manifest, ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    point_fold.validate_inventory(rows)
    evaluation_rows = full_evaluation_rows(rows)
    assignments = fold_assignments(evaluation_rows)
    checkpoints, fold_reports = validate_fold_results(
        evaluation_rows, assignments, args.fold_result)
    identities = [
        {"file": str(row["file"]), "start": int(start),
         "re": int(row["re"]), "aoa": int(row["aoa"]),
         "fold": assignments[str(row["file"])]}
        for row in evaluation_rows for start in row["val_starts"]]
    positions = {
        (str(row["file"]), int(row["start"])): index
        for index, row in enumerate(identities)}
    if len(positions) != EXPECTED_SAMPLES:
        raise RuntimeError("OOF cache sample identities are not exact")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    residuals = torch.empty(
        (EXPECTED_SAMPLES, 20, 32, 64, 2), dtype=torch.float16)
    factors = torch.full(
        (EXPECTED_SAMPLES,), float("nan"), dtype=torch.float32)
    history_hashes: list[str | None] = [None] * EXPECTED_SAMPLES
    target_hashes: list[str | None] = [None] * EXPECTED_SAMPLES
    written = 0
    for fold in range(point_fold.FOLDS):
        fold_rows = [
            row for row in evaluation_rows
            if assignments[str(row["file"])] == fold]
        written += predict_fold(
            checkpoints[fold], fold_rows, positions, args.real_dir,
            residuals, factors, history_hashes, target_hashes,
            device, args.batch_size)
    if (written != EXPECTED_SAMPLES or not torch.isfinite(factors).all()
            or not torch.isfinite(residuals).all()
            or any(value is None for value in history_hashes + target_hashes)):
        raise RuntimeError("OOF residual cache is incomplete/non-finite")
    residual_sha = hashlib.sha256(
        residuals.numpy().tobytes(order="C")).hexdigest()
    factor_sha = hashlib.sha256(
        factors.numpy().tobytes(order="C")).hexdigest()
    payload = {
        "schema": SCHEMA,
        "folds": point_fold.FOLDS,
        "fold_definition": "one complete released AoA per fold",
        "aoa_values": list(point_fold.AOA_VALUES),
        "window_source": "all legal complete-trajectory 20-to-20 windows",
        "window_stride": point_fold.EVALUATION_STRIDE,
        "manifest": manifest_report,
        "manifest_file_sha256": ft12.FORMAL_MANIFEST_FILE_SHA256,
        "sample_identities": identities,
        "sample_identity_sha256": dr1.canonical_sha256(identities),
        "ordered_history_sample_sha256": history_hashes,
        "ordered_target_sample_sha256": target_hashes,
        "ordered_history_stream_sha256": dr1.canonical_sha256(history_hashes),
        "ordered_target_stream_sha256": dr1.canonical_sha256(target_hashes),
        "fold_results": fold_reports,
        "signed_residual_physical_uv": residuals,
        "signed_residual_physical_uv_sha256": residual_sha,
        "oof_point_factor": factors,
        "oof_point_factor_sha256": factor_sha,
        "mean_absolute_residual": float(residuals.float().abs().mean()),
        "samples": EXPECTED_SAMPLES,
        "clean716_manifest_read": False,
        "submission_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    report = {
        key: value for key, value in payload.items()
        if not isinstance(value, torch.Tensor)}
    report.update({
        "output": str(args.output.resolve()),
        "output_sha256": dr1.sha256(args.output),
        "cache_bytes": args.output.stat().st_size,
    })
    report = point_fold.seal_json_payload(report)
    report_path = args.output.with_suffix(".json")
    dr1.write_json_atomic(report_path, report)
    point_fold.verify_json_payload(json.loads(
        report_path.read_text(encoding="utf-8")))
    return report


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps({
        "event": "b210_full_real_point_oof_residual_cache_complete",
        "samples": report["samples"],
        "mean_absolute_residual": report["mean_absolute_residual"],
        "output": report["output"],
    }), flush=True)


if __name__ == "__main__":
    main()
